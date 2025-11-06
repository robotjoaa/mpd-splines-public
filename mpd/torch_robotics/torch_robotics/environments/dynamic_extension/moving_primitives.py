"""
Moving primitives and objects with time-parameterized trajectories.

This module provides classes for defining obstacles that move over time,
including trajectory specifications and interpolation methods.
"""

import torch
import numpy as np
from typing import List, Callable, Optional
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from .trajectory import TrajectoryInterpolator, LinearTrajectory

class MovingObjectField(ObjectField):
    """
    An ObjectField that moves over time according to a trajectory.

    This class extends ObjectField with time-dependent position and orientation.
    """

    def __init__(
        self,
        primitive_fields,
        trajectory: TrajectoryInterpolator,
        name="moving_object",
        reference_frame="world",
        **kwargs
    ):
        """
        Args:
            primitive_fields: List of primitive shapes that make up this object
            trajectory: TrajectoryInterpolator defining motion over time
            name: Object name
            reference_frame: Reference frame
        """
        # Initialize with identity position/orientation (will be updated by trajectory)
        super().__init__(
            primitive_fields=primitive_fields,
            name=name,
            pos=None,
            ori=None,
            reference_frame=reference_frame,
            **kwargs
        )

        self.trajectory = trajectory
        self.current_time = 0.0

    def update_pose_at_time(self, t):
        """
        Update object pose according to trajectory at time t.

        Args:
            t: Time value
        """
        self.current_time = t
        pos, ori = self.trajectory(t)
        self.set_position_orientation(pos=pos, ori=ori)

    def compute_signed_distance(self, x, get_gradient=False, timesteps=None, **kwargs):
        """
        Override parent's compute_signed_distance to handle time-varying queries.

        Args:
            x: Query positions, shape (..., dim)
            get_gradient: Whether to compute gradient
            timesteps: Optional time values for each query point
                      - If None: uses current_time
                      - If scalar: updates pose to that time
                      - If tensor matching x: computes per-point time-varying SDF

        Returns:
            SDF values (and gradients if requested)
        """
        # Handle timesteps if provided
        if timesteps is not None:
            if isinstance(timesteps, (int, float)):
                # Single timestep for all points
                self.update_pose_at_time(timesteps)
            elif isinstance(timesteps, torch.Tensor):
                # Per-point timesteps
                return self._compute_sdf_varying_time(x, timesteps, get_gradient)
            else : 
                raise NotImplementedError

        # Default behavior - use parent's compute_signed_distance
        return super().compute_signed_distance(x, get_gradient=get_gradient)

    def compute_signed_distance_impl(self, x, get_gradient=False):
        """
        Override parent's compute_signed_distance_impl to handle primitives
        that don't support gradient computation (like MultiBoxField).

        Args:
            x: Query positions, shape (..., dim)
            get_gradient: Whether to compute gradient

        Returns:
            SDF values (and gradients if requested)
        """
        # Try parent implementation first
        try:
            return super().compute_signed_distance_impl(x, get_gradient=get_gradient)
        except NotImplementedError:
            # If gradients not supported by child primitives, compute without gradients
            if get_gradient:
                # Fall back to computing without gradients from primitives
                # but compute gradient via autograd on the SDF function
                x_with_grad = x.detach().clone().requires_grad_(True)
                sdf = super().compute_signed_distance_impl(x_with_grad, get_gradient=False)

                # Compute gradient using autograd
                if sdf.requires_grad:
                    grad_sdf = torch.autograd.grad(
                        sdf, x_with_grad,
                        grad_outputs=torch.ones_like(sdf),
                        create_graph=False
                    )[0]
                    return sdf.detach(), grad_sdf
                else:
                    # If sdf doesn't require grad, return zero gradients
                    return sdf, torch.zeros_like(x)
            else:
                # Re-raise if get_gradient is False (shouldn't happen)
                raise

    def _compute_sdf_varying_time(self, x, timesteps, get_gradient, use_naive = False) :
        """
        Compute SDF when different query points have different timesteps.

        Attempts vectorized computation for simple cases (single sphere with linear trajectory),
        falls back to per-timestep iteration for complex cases.

        Args:
            x: Query positions, shape (N, dim)
            timesteps: Time for each query point, shape (N,)
            get_gradient: Whether to compute gradients

        Returns:
            SDF values and optionally gradients
        """
        # Try vectorized implementation for simple analytical cases
        if not use_naive :
            vectorized_result = self._try_vectorized_time_varying_sdf(x, timesteps, get_gradient)
            if vectorized_result is not None:
                return vectorized_result

        # Fall back to per-timestep computation
        else : 
            raise NotImplementedError # never use iterative method
        # return self._compute_sdf_varying_time_iterative(x, timesteps, get_gradient)

    def _try_vectorized_time_varying_sdf(self, x, timesteps, get_gradient):
        """
        Attempt vectorized SDF computation for simple analytical cases.

        Tries to vectorize per-field and combines results. If all fields can be
        vectorized, returns combined result. Otherwise returns None for fallback.

        Returns None if vectorization not possible, otherwise returns (sdf, grad) or sdf.
        """
        # Try to vectorize each field
        field_results = []
        can_vectorize_all = True

        for primitive in self.fields:
            # Check for sphere with linear trajectory
            if isinstance(primitive, MultiSphereField) and isinstance(self.trajectory, LinearTrajectory):
                result = self._vectorized_sphere_linear(x, timesteps, primitive, get_gradient)
                if result is not None:
                    field_results.append(result)
                else:
                    can_vectorize_all = False
                    break
            # Check for box with linear trajectory
            elif isinstance(primitive, MultiBoxField) and isinstance(self.trajectory, LinearTrajectory):
                result = self._vectorized_box_linear(x, timesteps, primitive, get_gradient)
                if result is not None:
                    field_results.append(result)
                else:
                    can_vectorize_all = False
                    break
            else:
                # This primitive type can't be vectorized
                can_vectorize_all = False
                break

        if not can_vectorize_all or len(field_results) == 0:
            return None

        # Combine results from all fields (take minimum SDF)
        return self._combine_field_sdfs(field_results, get_gradient)

    def _combine_field_sdfs(self, field_results, get_gradient):
        """
        Combine SDF results from multiple fields by taking minimum.

        Args:
            field_results: List of (sdf, grad) tuples or sdf tensors
            get_gradient: Whether gradients are included

        Returns:
            Combined (sdf, grad) or sdf
        """
        if get_gradient:
            # field_results is list of (sdf, grad) tuples
            sdfs = [result[0] for result in field_results]
            grads = [result[1] for result in field_results]

            # Stack along new dimension: (num_fields, ...)
            sdfs_stacked = torch.stack(sdfs, dim=0)
            grads_stacked = torch.stack(grads, dim=0)

            # Find minimum SDF and corresponding index
            sdf_min, min_idx = torch.min(sdfs_stacked, dim=0)

            # Select gradient corresponding to minimum SDF
            # min_idx: (...), grads_stacked: (num_fields, ..., dim)
            # Need to gather along dim=0
            min_idx_expanded = min_idx
            for _ in range(grads_stacked.ndim - min_idx.ndim - 1):
                min_idx_expanded = min_idx_expanded.unsqueeze(-1)
            min_idx_expanded = min_idx_expanded.expand(*min_idx.shape, grads_stacked.shape[-1])

            grad_min = torch.gather(grads_stacked, 0, min_idx_expanded.unsqueeze(0)).squeeze(0)

            return sdf_min, grad_min
        else:
            # field_results is list of sdf tensors
            sdfs_stacked = torch.stack(field_results, dim=0)
            sdf_min, _ = torch.min(sdfs_stacked, dim=0)
            return sdf_min

    def _vectorized_sphere_linear(self, x, timesteps, sphere_field, get_gradient):
        """
        Vectorized SDF for sphere(s) with linear trajectory.

        For linear motion: pos(t) = pos_0 + velocity * t
        SDF(x, t) = min_i(||x - (center_i + pos(t))|| - radius_i)
        """
        # Flatten inputs
        timesteps_flat = timesteps.flatten()
        x_flat = x.reshape(-1, x.shape[-1])

        # Get trajectory velocity (pos(t) = start + velocity * t)
        if not hasattr(self.trajectory, 'get_velocity'):
            return None  # Can't vectorize without velocity info

        velocity = self.trajectory.get_velocity()  # Shape: (3,)
        start_pos = self.trajectory.get_start_position()  # Shape: (3,)

        # Compute time-varying translation: (N, 3)
        translation_t = start_pos.unsqueeze(0) + velocity.unsqueeze(0) * timesteps_flat.unsqueeze(-1)

        # Sphere centers in world frame at each timestep: (N, num_spheres, 3)
        # Account for 2D/3D mismatch
        centers = sphere_field.centers  # Shape: (num_spheres, 2 or 3)
        if centers.shape[-1] == 2:
            # Pad to 3D
            centers_3d = torch.cat([centers, torch.zeros(centers.shape[0], 1, **self.tensor_args)], dim=-1)
        else:
            centers_3d = centers

        # Transform centers by trajectory
        centers_at_t = centers_3d.unsqueeze(0) + translation_t.unsqueeze(1)  # (N, num_spheres, 3)

        # Compute distances from query points to each sphere center
        # x_flat: (N, 3), centers_at_t: (N, num_spheres, 3)
        if x_flat.shape[-1] == 2:
            x_flat_3d = torch.cat([x_flat, torch.zeros(x_flat.shape[0], 1, **self.tensor_args)], dim=-1)
        else:
            x_flat_3d = x_flat

        # Distance vectors: (N, num_spheres, 3)
        diff = x_flat_3d.unsqueeze(1) - centers_at_t
        distances = torch.norm(diff, dim=-1)  # (N, num_spheres)

        # SDF for each sphere: (N, num_spheres)
        radii = sphere_field.radii.unsqueeze(0)  # (1, num_spheres)
        sdfs = distances - radii

        # Take minimum over spheres: (N,)
        sdf_min, min_idx = torch.min(sdfs, dim=-1)

        # Reshape to original shape
        sdf_result = sdf_min.reshape(x.shape[:-1])

        if get_gradient:
            # Gradient is in direction of closest sphere
            # Select the diff vector for minimum sphere: (N, 3)
            min_idx_expanded = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
            closest_diff = torch.gather(diff, 1, min_idx_expanded).squeeze(1)  # (N, 3)

            # Normalize to get gradient direction
            closest_distance = torch.gather(distances, 1, min_idx.unsqueeze(-1)).squeeze(-1)  # (N,)
            grad = closest_diff / (closest_distance.unsqueeze(-1) + 1e-8)  # (N, 3)

            # Match dimension with x
            if x_flat.shape[-1] == 2:
                grad = grad[..., :2]

            grad_result = grad.reshape(x.shape)
            return sdf_result, grad_result
        else:
            return sdf_result

    def _vectorized_box_linear(self, x, timesteps, box_field, get_gradient):
        """
        Vectorized SDF for box(es) with linear trajectory.

        For linear motion: pos(t) = pos_0 + velocity * t
        For axis-aligned boxes: SDF(x, t) = max_i(|x_i - center_i(t)| - half_size_i)

        Args:
            x: Query positions, shape (..., dim)
            timesteps: Time for each query point, shape matching x
            box_field: MultiBoxField instance
            get_gradient: Whether to compute gradients

        Returns:
            SDF values and optionally gradients
        """
        # Flatten inputs
        timesteps_flat = timesteps.flatten()
        x_flat = x.reshape(-1, x.shape[-1])

        # Get trajectory velocity
        if not hasattr(self.trajectory, 'get_velocity'):
            return None  # Can't vectorize without velocity info

        velocity = self.trajectory.get_velocity()  # Shape: (3,)
        start_pos = self.trajectory.get_start_position()  # Shape: (3,)

        # Compute time-varying translation: (N, 3)
        translation_t = start_pos.unsqueeze(0) + velocity.unsqueeze(0) * timesteps_flat.unsqueeze(-1)

        # Box centers in world frame at each timestep: (N, num_boxes, 3)
        centers = box_field.centers  # Shape: (num_boxes, 2 or 3)
        if centers.shape[-1] == 2:
            # Pad to 3D
            centers_3d = torch.cat([centers, torch.zeros(centers.shape[0], 1, **self.tensor_args)], dim=-1)
        else:
            centers_3d = centers

        # Transform centers by trajectory: (N, num_boxes, 3)
        centers_at_t = centers_3d.unsqueeze(0) + translation_t.unsqueeze(1)

        # Prepare query points
        if x_flat.shape[-1] == 2:
            x_flat_3d = torch.cat([x_flat, torch.zeros(x_flat.shape[0], 1, **self.tensor_args)], dim=-1)
        else:
            x_flat_3d = x_flat

        # Compute SDF for axis-aligned boxes
        # Distance to centers: (N, num_boxes, 3)
        diff = x_flat_3d.unsqueeze(1) - centers_at_t

        # Half sizes: (num_boxes, 2 or 3)
        half_sizes = box_field.half_sizes
        if half_sizes.shape[-1] == 2:
            # Pad to 3D with zero (infinite extent in z)
            half_sizes_3d = torch.cat([half_sizes, torch.full((half_sizes.shape[0], 1), 1e10, **self.tensor_args)], dim=-1)
        else:
            half_sizes_3d = half_sizes

        # Distance to box faces: (N, num_boxes, 3)
        distance_to_faces = torch.abs(diff) - half_sizes_3d.unsqueeze(0)

        # SDF for each box: max over dimensions
        # For axis-aligned box: SDF = max(distance_to_faces, dim=-1)
        sdf_per_box, max_dim_idx = torch.max(distance_to_faces, dim=-1)  # (N, num_boxes)

        # Take minimum over boxes
        sdf_min, min_box_idx = torch.min(sdf_per_box, dim=-1)  # (N,)

        # Reshape to original shape
        sdf_result = sdf_min.reshape(x.shape[:-1])

        if get_gradient:
            # Gradient computation for boxes
            # The gradient points along the dimension with maximum distance

            # For each query point, we need:
            # 1. Which box is closest (min_box_idx)
            # 2. Which dimension is maximal for that box (max_dim_idx)
            # 3. Sign of the difference along that dimension

            # Select the box that gives minimum SDF: (N, 3)
            min_box_idx_expanded = min_box_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
            closest_diff = torch.gather(diff, 1, min_box_idx_expanded).squeeze(1)  # (N, 3)

            # Select which dimension was maximal for the closest box: (N,)
            closest_max_dim = torch.gather(max_dim_idx, 1, min_box_idx.unsqueeze(-1)).squeeze(-1)

            # Create gradient: zero except in the maximal dimension
            grad = torch.zeros_like(x_flat_3d)  # (N, 3)

            # For each point, set gradient along the maximal dimension
            batch_idx = torch.arange(x_flat_3d.shape[0], device=x_flat_3d.device)
            grad[batch_idx, closest_max_dim] = torch.sign(closest_diff[batch_idx, closest_max_dim])

            # Match dimension with x
            if x_flat.shape[-1] == 2:
                grad = grad[..., :2]

            grad_result = grad.reshape(x.shape)
            return sdf_result, grad_result
        else:
            return sdf_result

    def _compute_sdf_varying_time_iterative(self, x, timesteps, get_gradient):
        """
        Iterative per-timestep SDF computation (fallback for complex cases).

        This is the original implementation - used when vectorization not possible.
        Memory-optimized version that uses torch.no_grad() when gradients not needed.
        """
        # Flatten timesteps if needed
        timesteps_flat = timesteps.flatten()
        x_flat = x.reshape(-1, x.shape[-1])

        # Group queries by timestep for efficiency
        unique_times, inverse_indices = torch.unique(timesteps_flat, return_inverse=True)

        # Preallocate result tensors (more memory efficient than lists)
        sdf_result = torch.zeros(x_flat.shape[0], **self.tensor_args)
        if get_gradient:
            grad_result = torch.zeros(x_flat.shape[0], x.shape[-1], **self.tensor_args)

        # Process each unique timestep with appropriate gradient context
        context = torch.no_grad() if not get_gradient else torch.enable_grad()

        with context:
            for t_idx, t in enumerate(unique_times):
                # Find points at this timestep
                mask = (inverse_indices == t_idx)
                x_at_t = x_flat[mask]

                if x_at_t.shape[0] == 0:
                    continue

                # Update pose to this time
                self.update_pose_at_time(t.item())

                # Compute SDF using parent class method
                try:
                    if get_gradient:
                        sdf_t, grad_t = self.compute_signed_distance_impl(x_at_t, get_gradient=True)
                        sdf_result[mask] = sdf_t
                        grad_result[mask] = grad_t
                    else:
                        sdf_t = self.compute_signed_distance_impl(x_at_t, get_gradient=False)
                        sdf_result[mask] = sdf_t
                except NotImplementedError:
                    # This shouldn't happen since compute_signed_distance_impl handles it
                    raise

        # Reshape to original shape
        sdf_result = sdf_result.reshape(x.shape[:-1])

        if get_gradient:
            grad_result = grad_result.reshape(x.shape)
            return sdf_result, grad_result
        else:
            return sdf_result

    def compute_signed_distance_at_time(self, x, t, get_gradient=False):
        """
        Compute SDF at positions x and time t.

        Args:
            x: Query positions, shape (..., dim)
            t: Time value (scalar or tensor matching x batch dimension)
            get_gradient: Whether to compute gradient

        Returns:
            SDF values (and gradients if requested)
        """
        # Update pose to time t
        self.update_pose_at_time(t)

        # Compute SDF using parent class method
        return self.compute_signed_distance(x, get_gradient=get_gradient)


def create_moving_objects_from_trajectories(
    primitives_list: List[List],
    trajectories: List[TrajectoryInterpolator],
    names: Optional[List[str]] = None,
    tensor_args=DEFAULT_TENSOR_ARGS
) -> List[MovingObjectField]:
    """
    Create a list of MovingObjectField instances from primitives and trajectories.

    Args:
        primitives_list: List of primitive lists, one per object
        trajectories: List of trajectory interpolators, one per object
        names: Optional list of object names
        tensor_args: Tensor device and dtype

    Returns:
        List of MovingObjectField instances
    """
    assert len(primitives_list) == len(trajectories)

    if names is None:
        names = [f"moving_object_{i}" for i in range(len(primitives_list))]

    moving_objects = []
    for prims, traj, name in zip(primitives_list, trajectories, names):
        obj = MovingObjectField(
            primitive_fields=prims,
            trajectory=traj,
            name=name,
            tensor_args=tensor_args
        )
        moving_objects.append(obj)

    return moving_objects


if __name__ == "__main__":
    # Example: Create moving spheres
    from torch_robotics.torch_utils.torch_utils import to_numpy
    import matplotlib.pyplot as plt

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create a sphere that moves in a straight line
    sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    linear_traj = LinearTrajectory(
        keyframe_times=[0.0, 1.0, 2.0],
        keyframe_positions=[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ],
        tensor_args=tensor_args
    )

    moving_sphere_linear = MovingObjectField(
        primitive_fields=[sphere_prim],
        trajectory=linear_traj,
        name="linear_sphere"
    )

    # Create a sphere that moves in a circle
    sphere_prim2 = MultiSphereField(
        centers=np.array([[0.0, 0.0, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    circular_traj = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.5,
        angular_velocity=np.pi,  # Half rotation per second
        initial_phase=0.0,
        axis='z',
        tensor_args=tensor_args
    )

    moving_sphere_circular = MovingObjectField(
        primitive_fields=[sphere_prim2],
        trajectory=circular_traj,
        name="circular_sphere"
    )

    # Test trajectory evaluation
    print("Testing trajectories...")
    times = torch.linspace(0, 2, 21, **tensor_args)

    print("\nLinear trajectory:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pos, ori = linear_traj(t)
        print(f"  t={t:.1f}: pos={to_numpy(pos)}")

    print("\nCircular trajectory:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pos, ori = circular_traj(t)
        print(f"  t={t:.1f}: pos={to_numpy(pos)}")

    # Visualize trajectories
    fig = plt.figure(figsize=(12, 5))

    # Linear trajectory
    ax1 = fig.add_subplot(121)
    positions_linear = []
    for t in times:
        pos, _ = linear_traj(t.item())
        positions_linear.append(to_numpy(pos[:2]))
    positions_linear = np.array(positions_linear)

    ax1.plot(positions_linear[:, 0], positions_linear[:, 1], 'b-', label='Linear trajectory')
    ax1.scatter(positions_linear[0, 0], positions_linear[0, 1], c='g', s=100, marker='o', label='Start')
    ax1.scatter(positions_linear[-1, 0], positions_linear[-1, 1], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Trajectory')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True)

    # Circular trajectory
    ax2 = fig.add_subplot(122)
    positions_circular = []
    for t in times:
        pos, _ = circular_traj(t.item())
        positions_circular.append(to_numpy(pos[:2]))
    positions_circular = np.array(positions_circular)

    ax2.plot(positions_circular[:, 0], positions_circular[:, 1], 'b-', label='Circular trajectory')
    ax2.scatter(positions_circular[0, 0], positions_circular[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(positions_circular[-1, 0], positions_circular[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Circular Trajectory')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/moving_object_trajectories.png', dpi=150)
    print("\nSaved trajectory visualization to /tmp/moving_object_trajectories.png")

    # Test SDF computation at different times
    print("\n" + "="*60)
    print("Testing SDF computation...")

    test_point = torch.tensor([[0.0, 0.0, 0.0]], **tensor_args)
    for t in [0.0, 0.5, 1.0]:
        sdf = moving_sphere_linear.compute_signed_distance_at_time(test_point, t)
        print(f"Linear sphere at t={t:.1f}, point {to_numpy(test_point[0])}: SDF={sdf.item():.4f}")

    for t in [0.0, 0.5, 1.0]:
        sdf = moving_sphere_circular.compute_signed_distance_at_time(test_point, t)
        print(f"Circular sphere at t={t:.1f}, point {to_numpy(test_point[0])}: SDF={sdf.item():.4f}")
