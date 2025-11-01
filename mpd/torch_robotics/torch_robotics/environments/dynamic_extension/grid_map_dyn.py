"""
Time-varying Grid Map SDF for moving obstacles.

This module extends GridMapSDF to handle time-varying obstacles by precomputing
a 4D SDF grid (x, y, z, t) or 3D for 2D environments (x, y, t).
"""

import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy
from torch_robotics.environments.grid_map_sdf import GridMapSDF


class GridMapDynSDF(GridMapSDF):
    """
    Generates a time-varying SDF grid for moving obstacles.

    The grid has dimensions (x, y, [z], t) and is precomputed for all time steps.
    At runtime, querying the SDF at a specific time involves interpolation between
    the nearest time grid points.
    """

    def __init__(
        self,
        limits,
        cell_size,
        moving_obj_list_fn,
        time_range=(0.0, 1.0),
        num_time_steps=100,
        k_smooth=20.0,
        smoothing_method="Quadratic",
        batch_size=64,
        tensor_args=DEFAULT_TENSOR_ARGS
    ):
        """
        Args:
            limits: Spatial limits [[x_min, y_min, (z_min)], [x_max, y_max, (z_max)]]
            cell_size: Spatial grid resolution
            moving_obj_list_fn: Function that takes time t and returns list of ObjectField instances
            time_range: (t_min, t_max) time range for precomputation
            num_time_steps: Number of time steps to discretize
            k_smooth: Smoothness parameter for log-sum-exp union (higher = sharper)
            batch_size: Batch size for computation (not used currently, kept for compatibility)
            tensor_args: Tensor device and dtype configuration
        """
        self.limits = limits
        self.dim = limits.shape[-1]  # Spatial dimension (2 or 3)
        self.tensor_args = tensor_args

        # Moving objects function
        self.moving_obj_list_fn = moving_obj_list_fn

        # Time parameters
        self.time_range = time_range
        self.num_time_steps = num_time_steps
        self.time_steps = torch.linspace(time_range[0], time_range[1], num_time_steps, **tensor_args)
        self.dt = (time_range[1] - time_range[0]) / (num_time_steps - 1) if num_time_steps > 1 else 1.0

        # Smoothness parameters
        self.k_smooth = k_smooth
        self.smoothing_method = smoothing_method

        # Spatial grid parameters
        map_dim = torch.abs(limits[1] - limits[0])
        self.map_dim = map_dim
        self.cell_size = cell_size
        self.cmap_dim = torch.ceil(map_dim / cell_size).type(torch.LongTensor).to(self.tensor_args["device"])

        # Precomputed SDF tensors
        self.points_for_sdf = None
        self.sdf_tensor = None  # Shape: (nx, ny, [nz], nt)
        self.grad_sdf_tensor = None  # Shape: (nx, ny, [nz], nt, dim)

        self.batch_size = batch_size
        self.precompute_sdf()

    def precompute_sdf(self):
        """
        Precomputes the signed distance field and its gradient for all spatial and temporal grid points.
        """
        print(f"Precomputing time-varying SDF grid...")
        print(f"  Spatial dimensions: {self.cmap_dim.tolist()}")
        print(f"  Time steps: {self.num_time_steps}")
        print(f"  Total grid points: {self.cmap_dim.prod().item() * self.num_time_steps:,}")

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

            points_for_sdf_meshgrid = torch.meshgrid(*basis_ranges, indexing="ij")
            self.points_for_sdf = torch.stack(points_for_sdf_meshgrid, dim=-1)
            # points_for_sdf shape: (nx, ny, [nz], dim)

            # Precompute SDF for each time step
            if self.dim == 2:
                nx, ny = self.cmap_dim
                self.sdf_tensor = torch.zeros((nx, ny, self.num_time_steps), **self.tensor_args)
                self.grad_sdf_tensor = torch.zeros((nx, ny, self.num_time_steps, self.dim), **self.tensor_args)
            else:  # dim == 3
                nx, ny, nz = self.cmap_dim
                self.sdf_tensor = torch.zeros((nx, ny, nz, self.num_time_steps), **self.tensor_args)
                self.grad_sdf_tensor = torch.zeros((nx, ny, nz, self.num_time_steps, self.dim), **self.tensor_args)

            # Compute SDF at each time step
            for t_idx, t in enumerate(self.time_steps):
                if t_idx % max(1, self.num_time_steps // 10) == 0:
                    print(f"  Processing time step {t_idx+1}/{self.num_time_steps} (t={t.item():.3f})")

                # Get object configuration at this time
                obj_list = self.moving_obj_list_fn(t.item())

                # Flatten spatial points for vectorized computation
                points_flat = self.points_for_sdf.reshape(-1, self.dim)

                # Compute SDF based on smoothing method
                if self.smoothing_method == 'LSE':
                    # Compute SDF from each object
                    all_sdfs = []
                    for obj in obj_list:
                        sdf_obj = obj.compute_signed_distance(points_flat)
                        all_sdfs.append(sdf_obj)

                    # Apply smooth union to all SDFs at once
                    # φ_union = -k * logsumexp(-φ_i/k)
                    if len(all_sdfs) == 1:
                        sdf_flat = all_sdfs[0]
                    else:
                        sdfs_stacked = torch.stack(all_sdfs, dim=-1)
                        sdf_flat = -self.k_smooth * torch.logsumexp(-sdfs_stacked / self.k_smooth, dim=-1)

                    # Reshape back to grid and store
                    if self.dim == 2:
                        self.sdf_tensor[:, :, t_idx] = sdf_flat.reshape(nx, ny)
                    else:
                        self.sdf_tensor[:, :, :, t_idx] = sdf_flat.reshape(nx, ny, nz)

                    # Compute gradient using automatic differentiation
                    points_for_grad = self.points_for_sdf.clone().requires_grad_(True)
                    points_for_grad_flat = points_for_grad.reshape(-1, self.dim)

                    # Compute SDF from each object for gradient computation
                    all_sdfs_grad = []
                    for obj in obj_list:
                        sdf_obj = obj.compute_signed_distance(points_for_grad_flat)
                        all_sdfs_grad.append(sdf_obj)

                    # Apply smooth union
                    if len(all_sdfs_grad) == 1:
                        sdf_for_grad = all_sdfs_grad[0]
                    else:
                        sdfs_stacked_grad = torch.stack(all_sdfs_grad, dim=-1)
                        sdf_for_grad = -self.k_smooth * torch.logsumexp(-sdfs_stacked_grad / self.k_smooth, dim=-1)

                    # Compute gradients
                    grad_sdf_flat = torch.autograd.grad(
                        sdf_for_grad.sum(),
                        points_for_grad,
                        retain_graph=False,
                        create_graph=False
                    )[0]

                    # Reshape and store gradients
                    if self.dim == 2:
                        self.grad_sdf_tensor[:, :, t_idx, :] = grad_sdf_flat.reshape(nx, ny, self.dim)
                    else:
                        self.grad_sdf_tensor[:, :, :, t_idx, :] = grad_sdf_flat.reshape(nx, ny, nz, self.dim)

                elif self.smoothing_method == 'Quadratic':
                    # Helper function to compute SDF at a single point with Quadratic smoothing
                    def compute_sdf_single_point(x):
                        """Compute SDF at single point using quadratic smooth minimum."""
                        sdf = None
                        for obj in obj_list:
                            sdf_obj = obj.compute_signed_distance(x.unsqueeze(0)).squeeze(0)
                            if sdf is None:
                                sdf = sdf_obj
                            else:
                                # Quadratic smooth minimum
                                a = sdf
                                b = sdf_obj
                                k = self.k_smooth * 4.0
                                h = torch.maximum(k - torch.abs(a - b), torch.zeros_like(a)) / k
                                sdf = torch.minimum(a, b) - k * 0.25 * h * h
                        return sdf

                    # Vectorize over grid points
                    compute_sdf_vmap = torch.vmap(compute_sdf_single_point)
                    sdf_flat = compute_sdf_vmap(points_flat)

                    # Reshape and store SDF
                    if self.dim == 2:
                        self.sdf_tensor[:, :, t_idx] = sdf_flat.reshape(nx, ny)
                    else:
                        self.sdf_tensor[:, :, :, t_idx] = sdf_flat.reshape(nx, ny, nz)

                    # Compute gradient using autograd
                    points_for_grad = self.points_for_sdf.clone().requires_grad_(True)
                    points_for_grad_flat = points_for_grad.reshape(-1, self.dim)

                    sdf_for_grad = compute_sdf_vmap(points_for_grad_flat)

                    grad_sdf_flat = torch.autograd.grad(
                        sdf_for_grad.sum(),
                        points_for_grad,
                        retain_graph=False,
                        create_graph=False
                    )[0]

                    # Reshape and store gradients
                    if self.dim == 2:
                        self.grad_sdf_tensor[:, :, t_idx, :] = grad_sdf_flat.reshape(nx, ny, self.dim)
                    else:
                        self.grad_sdf_tensor[:, :, :, t_idx, :] = grad_sdf_flat.reshape(nx, ny, nz, self.dim)

                else:
                    raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

        print(f"Precomputation completed in {timer.elapsed:.2f} seconds")

    def compute_signed_distance_raw(self, x):
        sdf = None
        for obj in self.obj_list:
            sdf_obj = obj.compute_signed_distance(x)
            if sdf is None:
                sdf = sdf_obj
            else:
                sdf = torch.minimum(sdf, sdf_obj)
        return sdf

    

    def project_x_to_grid_points(self, X, **kwargs):
        """
        Project spatial coordinates X to grid indices.

        Args:
            X: Spatial coordinates, shape (..., dim)

        Returns:
            Grid indices, shape (..., dim), values in [0, cmap_dim-1]
        """
        X_in_map = ((X - self.limits[0]) / self.map_dim * self.cmap_dim).round().type(torch.LongTensor)

        # Clamp to valid grid range
        max_idx = torch.tensor(self.points_for_sdf.shape[:-1]) - 1
        X_in_map = X_in_map.clamp(torch.zeros_like(max_idx), max_idx)

        return X_in_map

    def project_t_to_time_indices(self, t):
        """
        Project time values to grid indices with fractional part for interpolation.

        Args:
            t: Time values, shape (...)

        Returns:
            t_idx_low: Lower time index, shape (...), integer
            t_idx_high: Upper time index, shape (...), integer
            alpha: Interpolation weight [0, 1], shape (...)
        """
        # Normalize time to [0, num_time_steps-1]
        t_normalized = (t - self.time_range[0]) / (self.time_range[1] - self.time_range[0])
        t_grid = t_normalized * (self.num_time_steps - 1)

        # Get integer indices
        t_idx_low = torch.floor(t_grid).long().clamp(0, self.num_time_steps - 1)
        t_idx_high = torch.ceil(t_grid).long().clamp(0, self.num_time_steps - 1)

        # Interpolation weight
        alpha = (t_grid - t_idx_low.float()).clamp(0, 1)

        return t_idx_low, t_idx_high, alpha

    def __call__(self, X, t, **kwargs):
        """
        Query SDF at spatial positions X and time t.

        Args:
            X: Spatial coordinates, shape (..., dim)
            t: Time values, shape (...) or scalar

        Returns:
            SDF values, shape (...)
        """
        return self.get_sdf(X, t, **kwargs)

    def compute_cost(self, X, t, **kwargs):
        """Alias for get_sdf for compatibility."""
        return self.get_sdf(X, t, **kwargs)

    def compute_signed_distance(self, X, t, get_gradient=False, **kwargs):
        """
        Compute signed distance (and optionally gradient) at positions X and time t.

        Args:
            X: Spatial coordinates, shape (..., dim)
            t: Time values, shape (...) or scalar
            get_gradient: If True, also return gradient

        Returns:
            sdf_vals: SDF values, shape (...)
            grad_sdf: (if get_gradient=True) Gradient, shape (..., dim)
        """
        X_in_map = self.project_x_to_grid_points(X, **kwargs)

        if get_gradient:
            return self.get_sdf_and_gradient(X, X_in_map, t, **kwargs)
        else:
            return self.get_sdf(X, t, X_in_map, **kwargs)

    def get_sdf(self, X, t, X_in_map=None, **kwargs):
        """
        Get SDF values with temporal interpolation and spatial surrogate gradients.

        Args:
            X: Spatial coordinates, shape (..., dim)
            t: Time values, shape (...) or scalar
            X_in_map: Precomputed spatial grid indices (optional)

        Returns:
            SDF values, shape (...)
        """
        if X_in_map is None:
            X_in_map = self.project_x_to_grid_points(X, **kwargs)

        # Handle scalar time
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, **self.tensor_args)
        if t.ndim == 0:
            t = t.expand(X.shape[:-1])

        # Get time indices and interpolation weight
        t_idx_low, t_idx_high, alpha = self.project_t_to_time_indices(t)

        # Query spatial grid at both time steps
        X_in_map_detached = X_in_map.detach()

        if self.dim == 2:
            X_query = X_in_map_detached[..., 0], X_in_map_detached[..., 1]
            sdf_vals_low = self.sdf_tensor[X_query + (t_idx_low,)]
            sdf_vals_high = self.sdf_tensor[X_query + (t_idx_high,)]

            grad_sdf_low = self.grad_sdf_tensor[X_query + (t_idx_low,)]
            grad_sdf_high = self.grad_sdf_tensor[X_query + (t_idx_high,)]
        else:  # dim == 3
            X_query = X_in_map_detached[..., 0], X_in_map_detached[..., 1], X_in_map_detached[..., 2]
            sdf_vals_low = self.sdf_tensor[X_query + (t_idx_low,)]
            sdf_vals_high = self.sdf_tensor[X_query + (t_idx_high,)]

            grad_sdf_low = self.grad_sdf_tensor[X_query + (t_idx_low,)]
            grad_sdf_high = self.grad_sdf_tensor[X_query + (t_idx_high,)]

        # Linear interpolation in time
        sdf_vals = (1 - alpha) * sdf_vals_low + alpha * sdf_vals_high
        grad_sdf = (1 - alpha.unsqueeze(-1)) * grad_sdf_low + alpha.unsqueeze(-1) * grad_sdf_high

        # Apply surrogate gradient trick for spatial differentiability
        # surrogate_sdf(x) = sdf(x_detached) + x @ grad_sdf(x_detached) - x_detached @ grad_sdf(x_detached)
        X_detached = X.detach()
        sdf_vals = sdf_vals + (X * grad_sdf).sum(-1) - (X_detached * grad_sdf).sum(-1)

        return sdf_vals

    def get_sdf_and_gradient(self, X, X_in_map, t, **kwargs):
        """
        Get both SDF values and gradients.

        Args:
            X: Spatial coordinates, shape (..., dim)
            X_in_map: Spatial grid indices
            t: Time values

        Returns:
            sdf_vals: SDF values, shape (...)
            grad_sdf: Gradients, shape (..., dim)
        """
        # Handle scalar time
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, **self.tensor_args)
        if t.ndim == 0:
            t = t.expand(X.shape[:-1])

        # Get time indices and interpolation weight
        t_idx_low, t_idx_high, alpha = self.project_t_to_time_indices(t)

        # Query spatial grid
        if self.dim == 2:
            X_query = X_in_map[..., 0], X_in_map[..., 1]
            sdf_vals_low = self.sdf_tensor[X_query + (t_idx_low,)]
            sdf_vals_high = self.sdf_tensor[X_query + (t_idx_high,)]

            grad_sdf_low = self.grad_sdf_tensor[X_query + (t_idx_low,)]
            grad_sdf_high = self.grad_sdf_tensor[X_query + (t_idx_high,)]
        else:  # dim == 3
            X_query = X_in_map[..., 0], X_in_map[..., 1], X_in_map[..., 2]
            sdf_vals_low = self.sdf_tensor[X_query + (t_idx_low,)]
            sdf_vals_high = self.sdf_tensor[X_query + (t_idx_high,)]

            grad_sdf_low = self.grad_sdf_tensor[X_query + (t_idx_low,)]
            grad_sdf_high = self.grad_sdf_tensor[X_query + (t_idx_high,)]

        # Linear interpolation in time
        sdf_vals = (1 - alpha) * sdf_vals_low + alpha * sdf_vals_high
        grad_sdf = (1 - alpha.unsqueeze(-1)) * grad_sdf_low + alpha.unsqueeze(-1) * grad_sdf_high

        return sdf_vals, grad_sdf

    def zero_grad(self):
        """Compatibility method."""
        pass

    def render_sdf_at_time(self, t, ax=None, fig=None, num_points=200):
        """
        Render the SDF at a specific time slice.

        Args:
            t: Time value to visualize
            ax: Matplotlib axis (created if None)
            fig: Matplotlib figure (created if None)
            num_points: Resolution for visualization

        Returns:
            fig, ax
        """
        if self.dim != 2:
            raise NotImplementedError("Rendering only implemented for 2D environments")

        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Create query grid
        xs = torch.linspace(self.limits[0][0], self.limits[1][0], num_points, **self.tensor_args)
        ys = torch.linspace(self.limits[0][1], self.limits[1][1], num_points, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        # Query SDF
        t_tensor = torch.tensor([t], **self.tensor_args).expand(points.shape[0])
        sdf_vals = self.get_sdf(points, t_tensor)

        # Reshape and plot
        sdf_grid = to_numpy(sdf_vals.reshape(num_points, num_points))
        X_np, Y_np = to_numpy(X), to_numpy(Y)

        cs = ax.contourf(X_np, Y_np, sdf_grid, levels=20, cmap='RdBu')
        ax.contour(X_np, Y_np, sdf_grid, levels=[0], colors='black', linewidths=2)
        ax.set_title(f'SDF at t={t:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        if fig is not None:
            fig.colorbar(cs, ax=ax)

        return fig, ax


if __name__ == "__main__":
    # Example: Two spheres moving towards each other and overlapping
    from torch_robotics.environments.primitives import MultiSphereField, ObjectField

    tensor_args = DEFAULT_TENSOR_ARGS

    def moving_objects_fn(t):
        """
        Create two spheres that move towards each other.
        At t=0, they are far apart. At t=1, they overlap.
        """
        # Sphere 1 moves from left to center
        center1_x = -0.8 + 0.8 * t
        sphere1 = MultiSphereField(
            centers=np.array([[center1_x, 0.0]]),
            radii=np.array([0.3]),
            tensor_args=tensor_args
        )

        # Sphere 2 moves from right to center
        center2_x = 0.8 - 0.8 * t
        sphere2 = MultiSphereField(
            centers=np.array([[center2_x, 0.0]]),
            radii=np.array([0.3]),
            tensor_args=tensor_args
        )

        obj1 = ObjectField([sphere1], name="sphere1")
        obj2 = ObjectField([sphere2], name="sphere2")

        return [obj1, obj2]

    # Create time-varying SDF grid
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

    print("Creating GridMapSDFTimeVarying...")
    grid_sdf = GridMapSDFTimeVarying(
        limits=limits,
        cell_size=0.02,
        moving_obj_list_fn=moving_objects_fn,
        time_range=(0.0, 1.0),
        num_time_steps=20,
        k_smooth=20.0,
        overlap_margin=0.1,
        tensor_args=tensor_args
    )

    # Visualize SDF at different times
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for ax, t in zip(axes, times):
        grid_sdf.render_sdf_at_time(t, ax=ax, fig=fig, num_points=100)

    plt.tight_layout()
    plt.savefig('/tmp/time_varying_sdf.png', dpi=150)
    print("\nSaved time-varying SDF visualization to /tmp/time_varying_sdf.png")

    # Test querying at specific points and times
    print("\n" + "="*60)
    print("Testing SDF queries...")

    test_points = torch.tensor([
        [0.0, 0.0],  # Center - should have collision at t=1
        [-0.5, 0.0],  # Left - near sphere 1
        [0.5, 0.0],  # Right - near sphere 2
    ], **tensor_args)

    test_times = torch.tensor([0.0, 0.5, 1.0], **tensor_args)

    for point in test_points:
        print(f"\nPoint: {to_numpy(point)}")
        for t in test_times:
            sdf_val = grid_sdf(point.unsqueeze(0), t)
            print(f"  t={t.item():.1f}: SDF={sdf_val.item():.4f} {'[COLLISION]' if sdf_val < 0 else ''}")
