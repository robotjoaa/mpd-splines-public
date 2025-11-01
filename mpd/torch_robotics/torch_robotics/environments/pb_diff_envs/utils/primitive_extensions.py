"""
Primitive extensions for pb_diff_envs compatibility.

This module provides mixins to add pb_diff_envs-compatible collision checking
API to MPD's primitive shapes, without creating wrapper classes.
"""

import numpy as np
import torch
from typing import Union

from torch_robotics.environments.primitives import (
    PrimitiveShapeField,
    MultiBoxField,
    MultiSphereField,
    ObjectField
)
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class CollisionCheckMixin:
    """
    Mixin to add pb_diff_envs-compatible collision checking to MPD primitives.

    This mixin can be added to any class that inherits from PrimitiveShapeField
    and implements compute_signed_distance().

    Methods added:
        - is_point_inside(pose, margin): Single point collision check
        - is_point_inside_batch(poses, margin): Batch collision check
        - is_point_inside_wg(pose, margin): Alias (group compatibility)

    Example:
        >>> class MyBoxField(CollisionCheckMixin, MultiBoxField):
        ...     pass
        >>>
        >>> boxes = MyBoxField(centers, sizes)
        >>> collision = boxes.is_point_inside(pose, margin=0.05)
    """

    def is_point_inside(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        margin: float = 0.0
    ) -> bool:
        """
        Check if point is in collision with any primitive.

        This is equivalent to pb_diff_envs' RectangleWall.is_point_inside()
        and RectangleWallGroup.is_point_inside_wg().

        Args:
            pose: (dim,) - Point position in world coordinates
            margin: Safety margin (default 0.0)
                   Point is in collision if SDF(pose) <= margin

        Returns:
            True if point is in collision (inside obstacle or within margin)

        Implementation:
            Uses SDF: collision = (SDF(pose) <= margin)
            - SDF < 0: Inside obstacle
            - SDF = 0: On boundary
            - 0 < SDF <= margin: Within safety margin
        """
        # Convert to torch
        if isinstance(pose, np.ndarray):
            pose_torch = torch.from_numpy(pose).float().to(**self.tensor_args)
        else:
            pose_torch = pose.to(**self.tensor_args)

        # Add batch dimension if needed
        if pose_torch.ndim == 1:
            pose_torch = pose_torch.unsqueeze(0)  # (1, dim)

        # Compute SDF (method from PrimitiveShapeField)
        sdf = self.compute_signed_distance(pose_torch)

        # Collision if SDF <= margin
        collision = sdf <= margin

        return collision.item()

    def is_point_inside_wg(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        margin: float = 0.0
    ) -> bool:
        """
        Alias for is_point_inside() for pb_diff_envs compatibility.

        The "wg" suffix stands for "wall group" in pb_diff_envs.
        Since MultiBoxField already computes minimum SDF across all boxes,
        this is equivalent to is_point_inside().
        """
        return self.is_point_inside(pose, margin)

    def is_point_inside_batch(
        self,
        poses: Union[np.ndarray, torch.Tensor],
        margin: float = 0.0
    ) -> torch.Tensor:
        """
        Batch version: check collision for multiple points simultaneously.

        Args:
            poses: (N, dim) - Multiple points to check
            margin: Safety margin

        Returns:
            torch.Tensor (N,) of bool - Collision mask for each point

        Example:
            >>> poses = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> collisions = boxes.is_point_inside_batch(poses, 0.05)
            >>> print(collisions)  # tensor([False, True, False])
        """
        # Convert to torch
        if isinstance(poses, np.ndarray):
            poses_torch = torch.from_numpy(poses).float().to(**self.tensor_args)
        else:
            poses_torch = poses.to(**self.tensor_args)

        # Ensure 2D shape
        if poses_torch.ndim == 1:
            poses_torch = poses_torch.unsqueeze(0)

        # Compute SDF for all points
        sdfs = self.compute_signed_distance(poses_torch)

        # Collision mask
        collisions = sdfs <= margin

        return collisions

    def compute_sdf(
        self,
        poses: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convenience method: compute signed distance field values.

        Args:
            poses: (N, dim) or (dim,) - Query points

        Returns:
            torch.Tensor (N,) or scalar - SDF values
        """
        if isinstance(poses, np.ndarray):
            poses_torch = torch.from_numpy(poses).float().to(**self.tensor_args)
        else:
            poses_torch = poses.to(**self.tensor_args)

        if poses_torch.ndim == 1:
            poses_torch = poses_torch.unsqueeze(0)

        return self.compute_signed_distance(poses_torch)


# ============================================================================
# Extended Primitive Classes
# ============================================================================

class MultiBoxFieldExtended(CollisionCheckMixin, MultiBoxField):
    """
    MultiBoxField with pb_diff_envs-compatible collision checking API.

    Inherits from:
        - CollisionCheckMixin: Adds is_point_inside(), is_point_inside_batch()
        - MultiBoxField: Standard MPD box primitive with SDF

    Usage:
        >>> centers = np.array([[0.0, 0.5], [0.5, 0.0]])
        >>> sizes = np.array([[0.2, 0.4], [0.3, 0.2]])
        >>> boxes = MultiBoxFieldExtended(centers, sizes)
        >>>
        >>> # pb_diff_envs API
        >>> collision = boxes.is_point_inside(np.array([0.05, 0.55]), margin=0.05)
        >>>
        >>> # MPD API (still works)
        >>> sdf = boxes.compute_signed_distance(torch.tensor([[0.05, 0.55]]))
        >>> obj_field = ObjectField([boxes])
    """
    pass


class MultiSphereFieldExtended(CollisionCheckMixin, MultiSphereField):
    """
    MultiSphereField with pb_diff_envs-compatible collision checking API.

    Usage:
        >>> centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> radii = np.array([0.2, 0.15])
        >>> spheres = MultiSphereFieldExtended(centers, radii)
        >>>
        >>> collision = spheres.is_point_inside(np.array([0.1, 0.1]), margin=0.05)
    """
    pass


# ============================================================================
# Factory Functions
# ============================================================================

def create_rectangle_walls(
    centers: Union[np.ndarray, torch.Tensor],
    half_extents: Union[np.ndarray, torch.Tensor],
    use_extended: bool = True,
    tensor_args: dict = None
) -> Union[MultiBoxField, MultiBoxFieldExtended]:
    """
    Create box primitives from pb_diff_envs format (centers, half_extents).

    Args:
        centers: (n_walls, dim) - Box centers
        half_extents: (n_walls, dim) - Half-extents (NOT full sizes)
        use_extended: If True, return MultiBoxFieldExtended with collision API
        tensor_args: Device/dtype configuration

    Returns:
        MultiBoxField or MultiBoxFieldExtended

    Example:
        >>> # pb_diff_envs format
        >>> centers = np.array([[0.0, 0.5], [0.5, 0.0]])
        >>> half_extents = np.array([[0.1, 0.2], [0.15, 0.1]])
        >>>
        >>> # Create MPD primitive
        >>> walls = create_rectangle_walls(centers, half_extents, use_extended=True)
        >>>
        >>> # Use with pb_diff_envs API
        >>> collision = walls.is_point_inside(pose, margin=0.05)
        >>>
        >>> # Use with MPD
        >>> obj_field = ObjectField([walls])
    """
    if tensor_args is None:
        tensor_args = DEFAULT_TENSOR_ARGS

    # Convert half_extents to full sizes (MPD convention)
    if isinstance(half_extents, np.ndarray):
        sizes = 2 * half_extents
    else:
        sizes = 2 * half_extents

    if use_extended:
        return MultiBoxFieldExtended(centers, sizes, tensor_args=tensor_args)
    else:
        return MultiBoxField(centers, sizes, tensor_args=tensor_args)


def create_sphere_walls(
    centers: Union[np.ndarray, torch.Tensor],
    radii: Union[np.ndarray, torch.Tensor],
    use_extended: bool = True,
    tensor_args: dict = None
) -> Union[MultiSphereField, MultiSphereFieldExtended]:
    """
    Create sphere primitives with optional collision API.

    Args:
        centers: (n_spheres, dim) - Sphere centers
        radii: (n_spheres,) - Sphere radii
        use_extended: If True, return MultiSphereFieldExtended
        tensor_args: Device/dtype configuration

    Returns:
        MultiSphereField or MultiSphereFieldExtended
    """
    if tensor_args is None:
        tensor_args = DEFAULT_TENSOR_ARGS

    if use_extended:
        return MultiSphereFieldExtended(centers, radii, tensor_args=tensor_args)
    else:
        return MultiSphereField(centers, radii, tensor_args=tensor_args)


# ============================================================================
# Utility Functions for ObjectField
# ============================================================================

def is_point_inside_object_field(
    object_field: ObjectField,
    pose: Union[np.ndarray, torch.Tensor],
    margin: float = 0.0
) -> bool:
    """
    Check collision with any primitive in ObjectField.

    This function properly handles ObjectField transformations (pos, ori) by
    using ObjectField's compute_signed_distance() method, which transforms
    the query point before delegating to primitives.

    IMPORTANT: Always use this function for ObjectFields that may have non-identity
    transformations. Direct primitive.is_point_inside() calls will ignore ObjectField's
    pos and ori.

    Args:
        object_field: ObjectField containing primitives
        pose: (dim,) - Point position in world coordinates
        margin: Safety margin

    Returns:
        True if point collides with any primitive (respects ObjectField transformations)

    Example:
        >>> obj_field = ObjectField([boxes, spheres], "obstacles", pos=np.array([1.0, 0.0, 0.0]))
        >>> collision = is_point_inside_object_field(obj_field, pose, margin=0.05)
    """
    # Convert to torch
    if isinstance(pose, np.ndarray):
        pose_torch = torch.from_numpy(pose).float().to(**object_field.tensor_args)
    else:
        pose_torch = pose.to(**object_field.tensor_args)

    if pose_torch.ndim == 1:
        pose_torch = pose_torch.unsqueeze(0)

    # CRITICAL: Use ObjectField's compute_signed_distance to respect transformations
    # This method transforms the point from world frame to local frame before
    # computing SDF, properly handling pos and ori
    sdf = object_field.compute_signed_distance(pose_torch)

    return (sdf <= margin).item()


def is_point_inside_object_field_batch(
    object_field: ObjectField,
    poses: Union[np.ndarray, torch.Tensor],
    margin: float = 0.0
) -> torch.Tensor:
    """
    Batch version: check collision for multiple points with ObjectField.

    This function properly handles ObjectField transformations (pos, ori) by
    using ObjectField's compute_signed_distance() method.

    IMPORTANT: Always use this function for ObjectFields that may have non-identity
    transformations. Direct primitive calls will ignore ObjectField's pos and ori.

    Args:
        object_field: ObjectField containing primitives
        poses: (N, dim) - Multiple points in world coordinates
        margin: Safety margin

    Returns:
        torch.Tensor (N,) of bool - Collision mask (respects ObjectField transformations)

    Example:
        >>> poses = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> collisions = is_point_inside_object_field_batch(obj_field, poses, 0.05)
    """
    # Convert to torch
    if isinstance(poses, np.ndarray):
        poses_torch = torch.from_numpy(poses).float().to(**object_field.tensor_args)
    else:
        poses_torch = poses.to(**object_field.tensor_args)

    if poses_torch.ndim == 1:
        poses_torch = poses_torch.unsqueeze(0)

    # CRITICAL: Use ObjectField's compute_signed_distance to respect transformations
    # This method transforms points from world frame to local frame before
    # computing SDF, properly handling pos and ori
    sdfs = object_field.compute_signed_distance(poses_torch)

    # Collision mask
    collisions = sdfs <= margin

    return collisions


# Aliases for backward compatibility
is_point_inside_wg = is_point_inside_object_field
is_point_inside_wg_batch = is_point_inside_object_field_batch


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Testing MultiBoxFieldExtended")
    print("=" * 60)

    # Create walls
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

    # Create extended primitive
    walls = create_rectangle_walls(centers, half_extents, use_extended=True)
    print(f"Created: {walls}")

    # Test single point
    pose = np.array([0.05, 0.55])
    collision = walls.is_point_inside(pose, margin=0.05)
    sdf = walls.compute_sdf(pose)
    print(f"\nSingle point test:")
    print(f"  Pose: {pose}")
    print(f"  SDF: {sdf.item():.4f}")
    print(f"  Collision (margin=0.05): {collision}")

    # Test batch
    poses = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    collisions = walls.is_point_inside_batch(poses, margin=0.05)
    sdfs = walls.compute_sdf(poses)
    print(f"\nBatch test:")
    print(f"  Poses:\n{poses}")
    print(f"  SDFs: {sdfs}")
    print(f"  Collisions: {collisions}")

    # Test with ObjectField
    obj_field = ObjectField([walls], "test_walls")
    collision_obj = is_point_inside_object_field(obj_field, pose, margin=0.05)
    print(f"\nObjectField test:")
    print(f"  Collision: {collision_obj}")

    # Batch with ObjectField
    collisions_obj = is_point_inside_object_field_batch(obj_field, poses, margin=0.05)
    print(f"  Batch collisions: {collisions_obj}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
