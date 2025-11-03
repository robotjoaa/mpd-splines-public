"""
Moving primitives and objects with time-parameterized trajectories.

This module provides classes for defining obstacles that move over time,
including trajectory specifications and interpolation methods.
"""

import torch
import numpy as np
from typing import List, Callable, Optional
from mpd.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension.trajectory import TrajectoryInterpolator

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
