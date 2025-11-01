"""
Obstacle group utilities for pb_diff_envs migration.

This module provides MPD-compatible replacements for pb_diff_envs obstacle groups,
using SDF-based collision checking instead of PyBullet.
"""

import numpy as np
import torch
from typing import List, Optional, Union

from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class RectangleWallGroupSDF:
    """
    MPD-compatible replacement for pb_diff_envs RectangleWallGroup.

    Uses MultiBoxField for efficient SDF-based collision checking instead of
    PyBullet's discrete collision detection.

    Key Differences from pb_diff_envs:
    - Uses signed distance fields (continuous) instead of binary AABB checks
    - Supports batch processing for multiple points simultaneously
    - Differentiable (useful for optimization)
    - GPU-accelerated

    Example:
        >>> centers = np.array([[0.0, 0.5], [0.5, 0.0]])
        >>> half_extents = np.array([[0.1, 0.2], [0.15, 0.1]])
        >>> wall_group = RectangleWallGroupSDF(centers, half_extents)
        >>>
        >>> pose = np.array([0.05, 0.55])
        >>> collision = wall_group.is_point_inside(pose, min_to_wall_dist=0.05)
        >>> print(f"Collision: {collision}")  # True or False
    """

    def __init__(
        self,
        centers: Union[np.ndarray, torch.Tensor],
        half_extents: Union[np.ndarray, torch.Tensor],
        tensor_args: dict = None
    ):
        """
        Initialize rectangle wall group from centers and half-extents.

        Args:
            centers: (n_walls, 2) - Wall centers in world coordinates
            half_extents: (n_walls, 2) - Half-extents for each wall
                         (NOT full sizes - this matches pb_diff_envs convention)
            tensor_args: Device and dtype configuration (default: CPU, float32)

        Note:
            MultiBoxField expects full sizes, not half-extents.
            We convert internally: sizes = 2 * half_extents
        """
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS

        self.tensor_args = tensor_args

        # Store original parameters
        if isinstance(centers, torch.Tensor):
            self.centers = centers.cpu().numpy()
        else:
            self.centers = np.array(centers)

        if isinstance(half_extents, torch.Tensor):
            self.half_extents = half_extents.cpu().numpy()
        else:
            self.half_extents = np.array(half_extents)

        # Validate shapes
        assert self.centers.ndim == 2, f"centers must be (n_walls, dim), got shape {self.centers.shape}"
        assert self.half_extents.ndim == 2, f"half_extents must be (n_walls, dim), got shape {self.half_extents.shape}"
        assert self.centers.shape[0] == self.half_extents.shape[0], \
            f"centers and half_extents must have same number of walls"

        self.n_walls = self.centers.shape[0]
        self.dim = self.centers.shape[1]

        # Convert to MultiBoxField
        # IMPORTANT: MultiBoxField expects FULL sizes, not half-extents
        sizes = 2 * self.half_extents
        self.multi_box = MultiBoxField(self.centers, sizes, tensor_args=tensor_args)

    def is_point_inside(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        min_to_wall_dist: float = 0.0
    ) -> bool:
        """
        Check if point is in collision with any wall.

        Equivalent to pb_diff_envs RectangleWallGroup.is_point_inside_wg()

        Args:
            pose: (dim,) - Point position in world coordinates
            min_to_wall_dist: Safety margin around walls (default 0.0)
                             If > 0, point is considered in collision if within
                             this distance of any wall

        Returns:
            True if point is in collision (inside any wall or within margin)

        Implementation:
            Collision condition: SDF(pose) <= min_to_wall_dist
            - SDF < 0: Inside wall (always collision)
            - SDF = 0: On wall boundary
            - SDF < margin: Within safety margin (collision with margin > 0)
        """
        # Convert numpy to torch
        if isinstance(pose, np.ndarray):
            pose_torch = torch.from_numpy(pose).float().to(**self.tensor_args)
        else:
            pose_torch = pose.to(**self.tensor_args)

        # Add batch dimension if needed
        if pose_torch.ndim == 1:
            pose_torch = pose_torch.unsqueeze(0)  # (1, dim)

        # Compute signed distance
        sdf = self.multi_box.compute_signed_distance(pose_torch)

        # Collision if SDF <= margin
        collision = sdf <= min_to_wall_dist

        return collision.item()

    def is_point_inside_wg(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        min_to_wall_dist: float = 0.0
    ) -> bool:
        """
        Alias for is_point_inside() for backward compatibility with pb_diff_envs.

        Note: In pb_diff_envs, RectangleWallGroup has this method. Here, since
        MultiBoxField already computes the minimum SDF across all boxes, we don't
        need a separate "group" method - is_point_inside() already handles all walls.
        """
        return self.is_point_inside(pose, min_to_wall_dist)

    def is_point_inside_batch(
        self,
        poses: Union[np.ndarray, torch.Tensor],
        min_to_wall_dist: float = 0.0
    ) -> torch.Tensor:
        """
        Batch version: check collision for multiple points simultaneously.

        Args:
            poses: (N, dim) - Multiple points to check
            min_to_wall_dist: Safety margin

        Returns:
            torch.Tensor (N,) of bool - Collision mask for each point

        Example:
            >>> poses = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> collisions = wall_group.is_point_inside_batch(poses, 0.05)
            >>> print(collisions)  # tensor([False, True, False])
        """
        if isinstance(poses, np.ndarray):
            poses_torch = torch.from_numpy(poses).float().to(**self.tensor_args)
        else:
            poses_torch = poses.to(**self.tensor_args)

        # Compute SDF for all points
        sdfs = self.multi_box.compute_signed_distance(poses_torch)

        # Collision mask
        collisions = sdfs <= min_to_wall_dist

        return collisions

    def compute_sdf(
        self,
        poses: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute signed distance field values for points.

        Useful for debugging, visualization, or optimization.

        Args:
            poses: (N, dim) or (dim,) - Query points

        Returns:
            torch.Tensor (N,) or scalar - SDF values
                - Negative: inside wall
                - Zero: on wall boundary
                - Positive: outside wall
        """
        if isinstance(poses, np.ndarray):
            poses_torch = torch.from_numpy(poses).float().to(**self.tensor_args)
        else:
            poses_torch = poses.to(**self.tensor_args)

        if poses_torch.ndim == 1:
            poses_torch = poses_torch.unsqueeze(0)

        return self.multi_box.compute_signed_distance(poses_torch)

    def to_object_field(self) -> ObjectField:
        """
        Convert to ObjectField for use with EnvBase.

        Returns:
            ObjectField wrapping the MultiBoxField

        Example:
            >>> wall_group = RectangleWallGroupSDF(centers, half_extents)
            >>> env = EnvBase(
            ...     limits=limits,
            ...     obj_fixed_list=[wall_group.to_object_field()],
            ...     ...
            ... )
        """
        return ObjectField([self.multi_box], name="rectangle_walls")

    def update_centers(
        self,
        new_centers: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update wall centers (for dynamic obstacles).

        Args:
            new_centers: (n_walls, dim) - New center positions

        Note:
            This recreates the internal MultiBoxField. For frequent updates,
            consider using DynamicRectangleWallGroup instead.
        """
        if isinstance(new_centers, torch.Tensor):
            self.centers = new_centers.cpu().numpy()
        else:
            self.centers = np.array(new_centers)

        assert self.centers.shape == (self.n_walls, self.dim), \
            f"new_centers shape mismatch: expected {(self.n_walls, self.dim)}, got {self.centers.shape}"

        # Recreate MultiBoxField
        sizes = 2 * self.half_extents
        self.multi_box = MultiBoxField(self.centers, sizes, tensor_args=self.tensor_args)

    def __repr__(self):
        return (f"RectangleWallGroupSDF(n_walls={self.n_walls}, dim={self.dim}, "
                f"device={self.tensor_args['device']})")


class DynamicRectangleWallGroup:
    """
    Dynamic rectangle wall group with time-parameterized trajectories.

    This is for time-varying obstacles where walls move along predefined trajectories.
    Uses the same SDF-based collision checking as RectangleWallGroupSDF.

    Example:
        >>> from torch_robotics.environments.pb_diff_envs.utils.trajectories import WaypointLinearTrajectory
        >>>
        >>> # Create trajectories for each wall
        >>> trajectories = []
        >>> for i in range(n_walls):
        >>>     waypoints = np.linspace(start_pos[i], end_pos[i], num=100)
        >>>     trajectories.append(WaypointLinearTrajectory(waypoints))
        >>>
        >>> # Create dynamic wall group
        >>> wall_group = DynamicRectangleWallGroup(
        >>>     trajectories=trajectories,
        >>>     half_extents=half_extents
        >>> )
        >>>
        >>> # Check collision at time t
        >>> pose = np.array([0.5, 0.5])
        >>> collision = wall_group.is_point_inside(pose, t=10, min_to_wall_dist=0.05)
    """

    def __init__(
        self,
        trajectories: List,  # List of trajectory objects with get_spec(t) method
        half_extents: Union[np.ndarray, torch.Tensor],
        tensor_args: dict = None
    ):
        """
        Initialize dynamic wall group with trajectories.

        Args:
            trajectories: List of trajectory objects (one per wall)
                         Each must have get_spec(t) method that returns position at time t
            half_extents: (n_walls, dim) - Half-extents for each wall (constant)
            tensor_args: Device and dtype configuration
        """
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS

        self.tensor_args = tensor_args
        self.trajectories = trajectories

        if isinstance(half_extents, torch.Tensor):
            self.half_extents = half_extents.cpu().numpy()
        else:
            self.half_extents = np.array(half_extents)

        self.n_walls = len(trajectories)
        assert self.half_extents.shape[0] == self.n_walls, \
            f"half_extents must match number of trajectories"

        self.dim = self.half_extents.shape[1]

        # Current time and wall positions
        self.current_time = 0
        self.current_centers = None
        self.current_wall_group = None

    def set_time(self, t: float):
        """
        Update wall positions to time t.

        Args:
            t: Time parameter (integer or float)
        """
        self.current_time = t

        # Get wall positions at time t from trajectories
        centers = []
        for traj in self.trajectories:
            pos = traj.get_spec(t)
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            centers.append(pos)

        self.current_centers = np.array(centers)  # (n_walls, dim)

        # Create static wall group for this time
        self.current_wall_group = RectangleWallGroupSDF(
            self.current_centers,
            self.half_extents,
            tensor_args=self.tensor_args
        )

    def is_point_inside(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        t: float,
        min_to_wall_dist: float = 0.0
    ) -> bool:
        """
        Check collision at specific time.

        Args:
            pose: (dim,) - Point position
            t: Time parameter
            min_to_wall_dist: Safety margin

        Returns:
            True if collision at time t
        """
        # Update to time t
        self.set_time(t)

        # Check collision
        return self.current_wall_group.is_point_inside(pose, min_to_wall_dist)

    def is_point_inside_wg(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        t: float,
        min_to_wall_dist: float = 0.0
    ) -> bool:
        """Alias for is_point_inside (pb_diff_envs compatibility)"""
        return self.is_point_inside(pose, t, min_to_wall_dist)

    def to_moving_obj_list_fn(self):
        """
        Create function that returns ObjectField at time t.

        Returns:
            Function t -> List[ObjectField]

        Usage with TimeVaryingEnvBase:
            >>> wall_group = DynamicRectangleWallGroup(trajectories, half_extents)
            >>> env = TimeVaryingEnvBase(
            ...     moving_obj_list_fn=wall_group.to_moving_obj_list_fn(),
            ...     time_range=(0, 100),
            ...     ...
            ... )
        """
        def get_obstacles_at_time(t):
            self.set_time(t)
            return [self.current_wall_group.to_object_field()]

        return get_obstacles_at_time

    def __repr__(self):
        return (f"DynamicRectangleWallGroup(n_walls={self.n_walls}, dim={self.dim}, "
                f"current_time={self.current_time})")


# ============================================================================
# Helper Functions
# ============================================================================

def create_rectangle_wall_group_from_pb_diff(
    recWall_list: List,
    tensor_args: dict = None
) -> RectangleWallGroupSDF:
    """
    Create RectangleWallGroupSDF from pb_diff_envs RectangleWall list.

    Args:
        recWall_list: List of pb_diff_envs RectangleWall objects
        tensor_args: Device and dtype configuration

    Returns:
        RectangleWallGroupSDF instance

    Example:
        >>> # pb_diff_envs code
        >>> from pb_diff_envs.environment.rand_rec_group import RectangleWall
        >>> walls = [
        >>>     RectangleWall(center=np.array([0.0, 0.5]), hExt=np.array([0.1, 0.2])),
        >>>     RectangleWall(center=np.array([0.5, 0.0]), hExt=np.array([0.15, 0.1])),
        >>> ]
        >>>
        >>> # Convert to MPD
        >>> wall_group = create_rectangle_wall_group_from_pb_diff(walls)
    """
    centers = np.array([wall.center for wall in recWall_list])
    half_extents = np.array([wall.hExt for wall in recWall_list])

    return RectangleWallGroupSDF(centers, half_extents, tensor_args=tensor_args)


def visualize_wall_group_sdf(
    wall_group: RectangleWallGroupSDF,
    xlim: tuple = (-1, 1),
    ylim: tuple = (-1, 1),
    resolution: int = 100,
    min_to_wall_dist: float = 0.0
):
    """
    Visualize SDF and collision region for debugging.

    Args:
        wall_group: RectangleWallGroupSDF to visualize
        xlim: X-axis limits
        ylim: Y-axis limits
        resolution: Grid resolution
        min_to_wall_dist: Safety margin to visualize

    Example:
        >>> wall_group = RectangleWallGroupSDF(centers, half_extents)
        >>> visualize_wall_group_sdf(wall_group, xlim=(-2, 2), ylim=(-2, 2))
    """
    import matplotlib.pyplot as plt

    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Flatten grid points
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (N, 2)

    # Compute SDF
    sdfs = wall_group.compute_sdf(points).cpu().numpy().reshape(X.shape)

    # Compute collision mask
    collisions = wall_group.is_point_inside_batch(points, min_to_wall_dist)
    collisions = collisions.cpu().numpy().reshape(X.shape)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot SDF
    im1 = ax1.contourf(X, Y, sdfs, levels=20, cmap='RdYlGn')
    ax1.contour(X, Y, sdfs, levels=[0], colors='black', linewidths=2)
    ax1.contour(X, Y, sdfs, levels=[min_to_wall_dist], colors='red', linewidths=2, linestyles='--')
    ax1.set_title('Signed Distance Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='SDF value')

    # Plot collision region
    ax2.imshow(collisions, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
               origin='lower', cmap='RdYlGn_r', alpha=0.7)
    ax2.set_title(f'Collision Region (margin={min_to_wall_dist})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Plot wall centers
    for center, hext in zip(wall_group.centers, wall_group.half_extents):
        rect = plt.Rectangle(center - hext, 2*hext[0], 2*hext[1],
                           fill=False, edgecolor='blue', linewidth=2)
        ax1.add_patch(rect)
        rect2 = plt.Rectangle(center - hext, 2*hext[0], 2*hext[1],
                            fill=False, edgecolor='blue', linewidth=2)
        ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import torch

    # Create some walls
    centers = np.array([
        [0.0, 0.5],
        [0.5, 0.0],
        [-0.3, -0.3]
    ])
    half_extents = np.array([
        [0.1, 0.2],
        [0.15, 0.1],
        [0.1, 0.1]
    ])

    # Create wall group
    wall_group = RectangleWallGroupSDF(centers, half_extents)
    print(wall_group)

    # Test single point
    pose = np.array([0.05, 0.55])
    collision = wall_group.is_point_inside(pose, min_to_wall_dist=0.05)
    sdf = wall_group.compute_sdf(pose)
    print(f"\nPose: {pose}")
    print(f"SDF: {sdf.item():.4f}")
    print(f"Collision (margin=0.05): {collision}")

    # Test batch
    poses = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    collisions = wall_group.is_point_inside_batch(poses, min_to_wall_dist=0.05)
    sdfs = wall_group.compute_sdf(poses)
    print(f"\nBatch test:")
    print(f"Poses:\n{poses}")
    print(f"SDFs: {sdfs}")
    print(f"Collisions: {collisions}")

    # Visualize (requires matplotlib)
    try:
        visualize_wall_group_sdf(wall_group, xlim=(-1, 1), ylim=(-1, 1), min_to_wall_dist=0.05)
    except ImportError:
        print("\nMatplotlib not available for visualization")
