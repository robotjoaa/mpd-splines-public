"""
Dynamic extension module for torch_robotics environments.

This module provides support for:
- Wrapper-based dynamic environments (EnvDynBase wraps EnvBase)
- MovingObjectField for trajectory-based obstacles
- Smooth SDF composition with overlap handling
- Differentiable SDF operations
- Automatic time-aware rendering and animation
"""

from .sdf_utils import (
    smooth_union_sdf,
    check_sphere_sphere_overlap,
    check_sphere_box_overlap,
    check_box_box_overlap,
    detect_primitive_overlaps,
    compute_smooth_sdf_with_overlap_handling
)

from .env_dyn_base import EnvDynBase

from .moving_primitives import (
    MovingObjectField,
    create_moving_objects_from_trajectories
)

from .trajectory import (
    TrajectoryInterpolator,
    LinearTrajectory,
    CircularTrajectory
)

__all__ = [
    # SDF utilities
    'smooth_union_sdf',
    'check_sphere_sphere_overlap',
    'check_sphere_box_overlap',
    'check_box_box_overlap',
    'detect_primitive_overlaps',
    'compute_smooth_sdf_with_overlap_handling',

    # Environment (wrapper-based)
    'EnvDynBase',

    # Trajectory classes
    'TrajectoryInterpolator',
    'LinearTrajectory',
    'CircularTrajectory',

    # Moving primitives
    'MovingObjectField',
    'create_moving_objects_from_trajectories',
]
