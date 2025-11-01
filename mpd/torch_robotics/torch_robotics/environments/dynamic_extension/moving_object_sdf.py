"""
Transformation-based time-varying SDF for moving obstacles.

This module implements efficient SDF queries for moving obstacles by:
1. Precomputing reference SDF grid once per object (at t=0)
2. Applying coordinate transformations at query time
3. Using autograd to compute correct world-frame gradients automatically

Key insight:
    x_ref = (x_world - p(t)) @ R(t).T
    sdf = sdf_ref(x_ref)

Autograd automatically computes: ∇_x_world sdf = (∇_x_ref sdf_ref) @ R(t).T
This is exactly the correct world-frame gradient!
"""

import torch
import torch.nn.functional as F
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.torch_utils.torch_timer import TimerCUDA


class MovingObjectSDF:
    """
    Precomputed SDF grid for a moving object with transformation-based queries.

    This class stores a single precomputed SDF grid in the object's reference frame,
    then applies rigid transformations at query time to handle object motion.
    Gradients are computed automatically via autograd.
    """

    def __init__(
        self,
        obj_field,
        trajectory_fn,
        limits,
        cell_size,
        tensor_args=DEFAULT_TENSOR_ARGS
    ):
        """
        Args:
            obj_field: ObjectField at reference configuration (t=0)
            trajectory_fn: Function(t) -> (position, rotation)
                          Returns: pos (dim,), rot (dim, dim) or None
            limits: Spatial limits [[x_min, y_min, (z_min)], [x_max, y_max, (z_max)]]
            cell_size: Spatial grid resolution
            tensor_args: Tensor device and dtype
        """
        self.obj_field = obj_field
        self.trajectory_fn = trajectory_fn
        self.limits = limits
        self.cell_size = cell_size
        self.tensor_args = tensor_args
        self.dim = limits.shape[-1]

        # Grid parameters
        self.map_dim = torch.abs(limits[1] - limits[0])
        self.cmap_dim = torch.ceil(self.map_dim / cell_size).type(torch.LongTensor).to(tensor_args["device"])

        # Precompute reference SDF grid
        self.sdf_grid = None
        self.grid_points = None
        self._precompute_reference_sdf()

    def _precompute_reference_sdf(self):
        """Precompute SDF grid at reference (t=0) configuration."""
        print(f"  Precomputing reference SDF for {self.obj_field.name}...")
        print(f"    Grid dimensions: {self.cmap_dim.tolist()}")
        print(f"    Total grid points: {self.cmap_dim.prod().item():,}")

        with TimerCUDA() as timer:
            # Create spatial grid
            basis_ranges = [
                torch.linspace(self.limits[0][0], self.limits[1][0], self.cmap_dim[0], **self.tensor_args),
                torch.linspace(self.limits[0][1], self.limits[1][1], self.cmap_dim[1], **self.tensor_args),
            ]
            if self.dim == 3:
                basis_ranges.append(
                    torch.linspace(self.limits[0][2], self.limits[1][2], self.cmap_dim[2], **self.tensor_args)
                )

            # Create meshgrid
            grid_meshgrid = torch.meshgrid(*basis_ranges, indexing="ij")
            self.grid_points = torch.stack(grid_meshgrid, dim=-1)  # (nx, ny, [nz], dim)

            # Flatten for SDF computation
            points_flat = self.grid_points.reshape(-1, self.dim)

            # Compute SDF at reference configuration
            sdf_flat = self.obj_field.compute_signed_distance(points_flat, get_gradient=False)

            # Reshape to grid
            if self.dim == 2:
                nx, ny = self.cmap_dim
                self.sdf_grid = sdf_flat.reshape(nx, ny)
            else:
                nx, ny, nz = self.cmap_dim
                self.sdf_grid = sdf_flat.reshape(nx, ny, nz)

        print(f"    Completed in {timer.elapsed:.2f} seconds")

    def _transform_to_reference_frame(self, x_world, pos, rot):
        """
        Transform world-frame points to object reference frame.

        Args:
            x_world: (..., dim) points in world frame
            pos: (dim,) object position at time t
            rot: (dim, dim) or None - object rotation at time t

        Returns:
            x_ref: (..., dim) points in reference frame
        """
        # Translate: x_centered = x - p(t)
        x_centered = x_world - pos

        if rot is not None:
            # Rotate: x_ref = (x - p) @ R.T = R.T @ (x - p)
            # Using @ for matrix multiplication
            # x_centered: (..., dim), rot.T: (dim, dim)
            x_ref = x_centered @ rot.T
        else:
            x_ref = x_centered

        return x_ref

    def _transform_to_reference_frame_batch(self, X, positions, rotations):
        """
        Batched transformation to reference frame for multiple timesteps.

        Args:
            X: (B, H, N, dim) points in world frame
            positions: (H, dim) object positions over time
            rotations: (H, dim, dim) or None - object rotations over time

        Returns:
            X_ref: (B, H, N, dim) points in reference frame
        """
        # Translate: broadcast positions to (1, H, 1, dim)
        pos_broadcast = positions.unsqueeze(0).unsqueeze(2)
        X_centered = X - pos_broadcast  # (B, H, N, dim)

        if rotations is not None:
            # Rotate: X_ref = X_centered @ R.T
            # X_centered: (B, H, N, dim)
            # rotations.T: (H, dim, dim)
            # Use einsum for batched matrix multiplication
            rot_T = rotations.transpose(-2, -1)  # (H, dim, dim)
            X_ref = torch.einsum('bhnd,hde->bhne', X_centered, rot_T)
        else:
            X_ref = X_centered

        return X_ref

    def _grid_interpolate(self, grid, x_ref):
        """
        Differentiable grid interpolation using bilinear/trilinear interpolation.

        Args:
            grid: (nx, ny, [nz]) SDF grid
            x_ref: (..., dim) query points in reference frame

        Returns:
            sdf_vals: (...) interpolated SDF values
        """
        # Project to grid coordinates (continuous)
        # grid_coords in [0, grid_size-1]
        grid_coords = (x_ref - self.limits[0]) / self.cell_size

        # Clamp to valid range
        max_coords = self.cmap_dim.to(x_ref.device).float() - 1
        grid_coords = torch.clamp(grid_coords, torch.zeros_like(max_coords), max_coords)

        # Use bilinear/trilinear interpolation
        # This is differentiable! Autograd will track gradients.
        if self.dim == 2:
            sdf_vals = self._bilinear_interpolate(grid, grid_coords)
        else:
            sdf_vals = self._trilinear_interpolate(grid, grid_coords)

        return sdf_vals

    def _bilinear_interpolate(self, grid, coords):
        """
        Bilinear interpolation (differentiable).

        Args:
            grid: (nx, ny) grid values
            coords: (..., 2) grid coordinates [ix, iy] in [0, grid_size-1]

        Returns:
            vals: (...) interpolated values
        """
        # Get integer and fractional parts
        coords_floor = torch.floor(coords).long()
        coords_frac = coords - coords_floor.float()

        # Get corner indices
        ix0 = coords_floor[..., 0]
        iy0 = coords_floor[..., 1]
        ix1 = torch.clamp(ix0 + 1, 0, grid.shape[0] - 1)
        iy1 = torch.clamp(iy0 + 1, 0, grid.shape[1] - 1)

        # Get fractional parts
        tx = coords_frac[..., 0]
        ty = coords_frac[..., 1]

        # Gather corner values
        v00 = grid[ix0, iy0]
        v01 = grid[ix0, iy1]
        v10 = grid[ix1, iy0]
        v11 = grid[ix1, iy1]

        # Bilinear interpolation
        v0 = v00 * (1 - tx) + v10 * tx
        v1 = v01 * (1 - tx) + v11 * tx
        vals = v0 * (1 - ty) + v1 * ty

        return vals

    def _trilinear_interpolate(self, grid, coords):
        """
        Trilinear interpolation (differentiable).

        Args:
            grid: (nx, ny, nz) grid values
            coords: (..., 3) grid coordinates [ix, iy, iz] in [0, grid_size-1]

        Returns:
            vals: (...) interpolated values
        """
        # Get integer and fractional parts
        coords_floor = torch.floor(coords).long()
        coords_frac = coords - coords_floor.float()

        # Get corner indices
        ix0 = coords_floor[..., 0]
        iy0 = coords_floor[..., 1]
        iz0 = coords_floor[..., 2]
        ix1 = torch.clamp(ix0 + 1, 0, grid.shape[0] - 1)
        iy1 = torch.clamp(iy0 + 1, 0, grid.shape[1] - 1)
        iz1 = torch.clamp(iz0 + 1, 0, grid.shape[2] - 1)

        # Get fractional parts
        tx = coords_frac[..., 0]
        ty = coords_frac[..., 1]
        tz = coords_frac[..., 2]

        # Gather corner values (8 corners of cube)
        v000 = grid[ix0, iy0, iz0]
        v001 = grid[ix0, iy0, iz1]
        v010 = grid[ix0, iy1, iz0]
        v011 = grid[ix0, iy1, iz1]
        v100 = grid[ix1, iy0, iz0]
        v101 = grid[ix1, iy0, iz1]
        v110 = grid[ix1, iy1, iz0]
        v111 = grid[ix1, iy1, iz1]

        # Trilinear interpolation
        v00 = v000 * (1 - tx) + v100 * tx
        v01 = v001 * (1 - tx) + v101 * tx
        v10 = v010 * (1 - tx) + v110 * tx
        v11 = v011 * (1 - tx) + v111 * tx

        v0 = v00 * (1 - ty) + v10 * ty
        v1 = v01 * (1 - ty) + v11 * ty

        vals = v0 * (1 - tz) + v1 * tz

        return vals

    def query_sdf(self, x_world, t, get_gradient=False):
        """
        Query SDF at world-frame points at time t.

        Args:
            x_world: (..., dim) query points in world frame
            t: scalar time value
            get_gradient: whether to compute gradient

        Returns:
            sdf_vals: (...) SDF values
            grad_vals: (..., dim) gradients if get_gradient=True
        """
        # Enable gradients if requested
        if get_gradient:
            x_world = x_world.requires_grad_(True)

        # Get transformation at time t
        pos_t, rot_t = self.trajectory_fn(t)

        # Transform to reference frame (autograd tracks this!)
        x_ref = self._transform_to_reference_frame(x_world, pos_t, rot_t)

        # Interpolate from precomputed grid (autograd tracks this!)
        sdf_vals = self._grid_interpolate(self.sdf_grid, x_ref)

        if get_gradient:
            # Autograd computes correct world-frame gradient via chain rule!
            grad_vals = torch.autograd.grad(
                sdf_vals.sum(),
                x_world,
                create_graph=False
            )[0]
            return sdf_vals, grad_vals
        else:
            return sdf_vals

    def query_sdf_batched(self, X, timesteps, get_gradient=False):
        """
        Query SDF at multiple timesteps simultaneously (VECTORIZED!).

        This is the key method for efficient time-varying SDF queries.
        Single autograd call handles all timesteps and transformations.

        Args:
            X: (B, H, N, dim) query points at each timestep
            timesteps: (H,) time values
            get_gradient: whether to compute gradients

        Returns:
            sdf_vals: (B, H, N) SDF values
            grad_vals: (B, H, N, dim) gradients if get_gradient=True
        """
        B, H, N, dim = X.shape

        # Enable gradients if requested
        if get_gradient:
            X = X.requires_grad_(True)

        # Get transformations for all timesteps (vectorized!)
        positions = []
        rotations = []

        for t in timesteps:
            pos_t, rot_t = self.trajectory_fn(t.item())
            positions.append(pos_t)
            if rot_t is not None:
                rotations.append(rot_t)

        positions = torch.stack(positions, dim=0)  # (H, dim)

        if len(rotations) > 0:
            rotations = torch.stack(rotations, dim=0)  # (H, dim, dim)
        else:
            rotations = None

        # Transform all points to reference frame (batched!)
        X_ref = self._transform_to_reference_frame_batch(X, positions, rotations)

        # Interpolate from grid (vectorized over all points)
        # Flatten batch and horizon for interpolation
        X_ref_flat = X_ref.reshape(-1, dim)  # (B*H*N, dim)
        sdf_flat = self._grid_interpolate(self.sdf_grid, X_ref_flat)

        # Reshape back
        sdf_vals = sdf_flat.reshape(B, H, N)

        if get_gradient:
            # Autograd computes gradient through entire batched transformation!
            grad_vals = torch.autograd.grad(
                sdf_vals.sum(),
                X,
                create_graph=False
            )[0]
            return sdf_vals, grad_vals
        else:
            return sdf_vals


class TimeVaryingSDFComposer:
    """
    Composes multiple moving object SDFs and static SDFs.

    Handles smooth union of multiple moving objects at query time.
    """

    def __init__(
        self,
        moving_object_sdfs,
        static_sdf_grid=None,
        k_smooth=20.0,
        smoothing_method="Quadratic",
        tensor_args=DEFAULT_TENSOR_ARGS
    ):
        """
        Args:
            moving_object_sdfs: List[MovingObjectSDF] - moving obstacles
            static_sdf_grid: GridMapSDF or None - static obstacles
            k_smooth: smoothness parameter
            smoothing_method: "LSE", "Quadratic", or "Hard"
            tensor_args: tensor device and dtype
        """
        self.moving_object_sdfs = moving_object_sdfs
        self.static_sdf_grid = static_sdf_grid
        self.k_smooth = k_smooth
        self.smoothing_method = smoothing_method
        self.tensor_args = tensor_args

    def _smooth_union(self, sdfs):
        """
        Compute smooth union of multiple SDFs.

        Args:
            sdfs: List of SDF tensors with same shape

        Returns:
            Combined SDF
        """
        if len(sdfs) == 0:
            return None
        if len(sdfs) == 1:
            return sdfs[0]

        sdfs_stacked = torch.stack(sdfs, dim=-1)

        if self.smoothing_method == "LSE":
            # Log-sum-exp smooth minimum
            return -self.k_smooth * torch.logsumexp(-sdfs_stacked / self.k_smooth, dim=-1)

        elif self.smoothing_method == "Quadratic":
            # Quadratic smooth minimum (pairwise)
            k = self.k_smooth * 4.0
            result = sdfs[0]
            for i in range(1, len(sdfs)):
                a = result
                b = sdfs[i]
                h = torch.maximum(k - torch.abs(a - b), torch.zeros_like(a)) / k
                result = torch.minimum(a, b) - k * 0.25 * h * h
            return result

        else:  # Hard minimum
            return sdfs_stacked.min(dim=-1)[0]

    def query_sdf_batched(self, X, timesteps, get_gradient=False):
        """
        Query combined SDF from all moving and static objects.

        Args:
            X: (B, H, N, dim) query points
            timesteps: (H,) time values
            get_gradient: whether to compute gradients

        Returns:
            sdf_combined: (B, H, N) combined SDF
            grad_combined: (B, H, N, dim) if get_gradient=True
        """
        B, H, N, dim = X.shape

        if get_gradient:
            X = X.requires_grad_(True)

        all_sdfs = []

        # Query each moving object
        for moving_sdf in self.moving_object_sdfs:
            sdf = moving_sdf.query_sdf_batched(X, timesteps, get_gradient=False)
            all_sdfs.append(sdf)

        # Add static SDF if present
        if self.static_sdf_grid is not None:
            # Static SDF doesn't change with time - query once and tile
            X_flat = X.reshape(-1, dim)
            sdf_static_flat = self.static_sdf_grid.compute_signed_distance(X_flat, get_gradient=False)
            sdf_static = sdf_static_flat.reshape(B, H, N)
            all_sdfs.append(sdf_static)

        # Smooth union
        sdf_combined = self._smooth_union(all_sdfs)

        if get_gradient:
            # Autograd computes gradient through smooth union
            grad_combined = torch.autograd.grad(
                sdf_combined.sum(),
                X,
                create_graph=False
            )[0]
            return sdf_combined, grad_combined
        else:
            return sdf_combined


if __name__ == "__main__":
    print("MovingObjectSDF: Transformation-based time-varying SDF")
    print("Uses autograd for correct gradient computation")
