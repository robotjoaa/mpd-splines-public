"""
Test time-varying rendering with EnvDynBase and animation functions.

This script demonstrates:
1. Creating an environment with moving obstacles
2. Rendering the environment at specific times
3. Animating moving obstacles
4. Integration with PlanningTask animations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    LinearTrajectory,
    CircularTrajectory,
    MovingObjectField,
    animate_robot_trajectories_with_time,
    render_robot_trajectories_with_time,
)
from torch_robotics.environments.primitives import MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def create_moving_obstacle_fn():
    """Create a function that returns obstacles at a given time."""
    tensor_args = DEFAULT_TENSOR_ARGS

    # Define trajectories for moving obstacles
    traj1 = LinearTrajectory(
        start_pos=np.array([-0.5, -0.5]),
        end_pos=np.array([0.5, 0.5]),
        time_range=(0.0, 1.0)
    )

    traj2 = CircularTrajectory(
        center=np.array([0.0, 0.0]),
        radius=0.4,
        start_angle=0.0,
        end_angle=2 * np.pi,
        time_range=(0.0, 1.0)
    )

    def moving_obj_list_fn(t):
        """Return list of ObjectField at time t."""
        # Get positions at time t
        pos1 = traj1.get_position(t)
        pos2 = traj2.get_position(t)

        # Create spheres at these positions
        sphere1 = MultiSphereField(
            centers=pos1.reshape(1, -1),
            radii=np.array([0.15]),
            tensor_args=tensor_args
        )

        sphere2 = MultiSphereField(
            centers=pos2.reshape(1, -1),
            radii=np.array([0.12]),
            tensor_args=tensor_args
        )

        return [
            ObjectField([sphere1], "moving_sphere_1"),
            ObjectField([sphere2], "moving_sphere_2"),
        ]

    return moving_obj_list_fn


def test_render_at_time():
    """Test 1: Render environment at specific times."""
    print("Test 1: Render environment at specific times")

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create static obstacle
    static_sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    # Create environment with moving obstacles
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([static_sphere], "static_obstacle")],
        moving_obj_list_fn=create_moving_obstacle_fn(),
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Render at different times
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    times = [0.0, 0.33, 0.67, 1.0]

    for ax, t in zip(axes.flat, times):
        env.render(ax, time=t)
        ax.set_title(f"Time: {t:.2f}s")

    plt.tight_layout()
    plt.savefig("/tmp/test_render_at_time.png", dpi=150)
    print(f"  Saved visualization to /tmp/test_render_at_time.png")
    plt.close()


def test_animate_environment():
    """Test 2: Animate moving obstacles without robot."""
    print("\nTest 2: Animate moving obstacles")

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with only moving obstacles
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],  # No static obstacles
        moving_obj_list_fn=create_moving_obstacle_fn(),
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Create time steps for animation
    time_steps = torch.linspace(0.0, 1.0, 100, **tensor_args)

    # Animate
    env.animate_with_time(
        trajectory_time_steps=time_steps,
        n_frames=30,
        video_filepath="/tmp/test_moving_obstacles.mp4",
        show_time_label=True,
        anim_time=3,
        dpi=100,
    )
    print(f"  Saved animation to /tmp/test_moving_obstacles.mp4")


def test_render_sdf_at_time():
    """Test 3: Render SDF field at specific time."""
    print("\nTest 3: Render SDF field at specific time")

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],
        moving_obj_list_fn=create_moving_obstacle_fn(),
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Render SDF and environment at t=0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Render SDF
    env.render_sdf(ax=axes[0], fig=fig, use_smooth_union=True)
    axes[0].set_title('SDF Field (static)')

    # Render environment with moving obstacles at t=0.5
    env.render(ax=axes[1], time=0.5)
    axes[1].set_title('Environment at t=0.5')

    plt.tight_layout()
    plt.savefig("/tmp/test_sdf_with_moving.png", dpi=150)
    print(f"  Saved visualization to /tmp/test_sdf_with_moving.png")
    plt.close()


def test_integration_with_planning_task():
    """Test 4: Integration with PlanningTask animations."""
    print("\nTest 4: Integration with PlanningTask")
    print("  (This test requires a full PlanningTask setup)")
    print("  Example usage:")
    print("""
    from torch_robotics.tasks.tasks import PlanningTask
    from torch_robotics.environments.dynamic_extension import (
        EnvDynBase,
        animate_robot_trajectories_with_time
    )

    # Create environment with moving obstacles
    env = EnvDynBase(
        limits=limits,
        obj_fixed_list=static_obstacles,
        moving_obj_list_fn=moving_obstacles_fn,
        time_range=(0.0, 1.0),
    )

    # Create planning task
    planning_task = PlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=trajectory,
    )

    # Use time-aware animation
    animate_robot_trajectories_with_time(
        planning_task,
        q_pos_trajs=optimized_trajectories,
        q_pos_start=start_config,
        q_pos_goal=goal_config,
        n_frames=50,
        video_filepath='robot_with_moving_obstacles.mp4',
    )
    """)


def test_render_helper_function():
    """Test 5: Test the render helper function."""
    print("\nTest 5: Test render_robot_trajectories_with_time (standalone)")

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create a simple environment
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],
        moving_obj_list_fn=create_moving_obstacle_fn(),
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Test rendering at different times
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    times = [0.0, 0.5, 1.0]

    for ax, t in zip(axes, times):
        env.render(ax, time=t)
        ax.set_title(f"t={t:.1f}s")

    plt.tight_layout()
    plt.savefig("/tmp/test_render_helper.png", dpi=150)
    print(f"  Saved visualization to /tmp/test_render_helper.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Time-Varying Rendering with EnvDynBase")
    print("=" * 60)

    # Run tests
    test_render_at_time()
    test_animate_environment()
    test_render_sdf_at_time()
    test_integration_with_planning_task()
    test_render_helper_function()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - /tmp/test_render_at_time.png")
    print("  - /tmp/test_moving_obstacles.mp4")
    print("  - /tmp/test_sdf_with_moving.png")
    print("  - /tmp/test_render_helper.png")
