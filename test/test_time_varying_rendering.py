"""
Test time-varying rendering with EnvDynBase wrapper and automatic MovingObjectField handling.

This script demonstrates:
1. Creating an environment with MovingObjectField instances
2. Automatic time-aware rendering (no need for moving_obj_list_fn)
3. Animating moving obstacles with the wrapper pattern
4. Integration with time-dependent SDF computation
"""

import torch
import numpy as np
from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()
import matplotlib.pyplot as plt

from mpd.torch_robotics.torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    LinearTrajectory,
    CircularTrajectory,
    MovingObjectField,
)
from mpd.torch_robotics.torch_robotics.environments.primitives import MultiSphereField, ObjectField
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from mpd.torch_robotics.torch_robotics.visualizers.plot_utils import create_fig_and_axes


def create_moving_obstacles():
    """Create MovingObjectField instances with different trajectories."""
    tensor_args = DEFAULT_TENSOR_ARGS

    # Moving sphere 1: Linear trajectory (2D primitives, 3D trajectory)
    sphere1_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    linear_traj = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
        tensor_args=tensor_args
    )

    moving_sphere1 = MovingObjectField(
        primitive_fields=[sphere1_prim],
        trajectory=linear_traj,
        name="moving_sphere_1"
    )

    # Moving sphere 2: Circular trajectory
    sphere2_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.12]),
        tensor_args=tensor_args
    )

    circular_traj = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.4,
        angular_velocity=2 * np.pi,
        initial_phase=0.0,
        axis='z',
        tensor_args=tensor_args
    )

    moving_sphere2 = MovingObjectField(
        primitive_fields=[sphere2_prim],
        trajectory=circular_traj,
        name="moving_sphere_2"
    )

    return [moving_sphere1, moving_sphere2]


def test_render_at_time():
    """Test 1: Render environment at specific times with automatic MovingObjectField handling."""
    print("\n" + "="*70)
    print("Test 1: Render at Specific Times (Automatic MovingObjectField)")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create static obstacle
    static_sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    # Create environment with MovingObjectField instances
    # No need for moving_obj_list_fn - just add MovingObjectField to obj_extra_list!
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([static_sphere], "static_obstacle")],
        obj_extra_list=create_moving_obstacles(),  # MovingObjectField automatically handled!
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Render at different times - MovingObjectField poses are updated automatically
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    times = [0.0, 0.33, 0.67, 1.0]

    for ax, t in zip(axes.flat, times):
        env.render(ax, time=t)
        ax.set_title(f"Time: {t:.2f}s")

    plt.tight_layout()
    plt.savefig("/tmp/test_render_at_time.png", dpi=150)
    print(f"✓ Saved visualization to /tmp/test_render_at_time.png")
    print("✓ MovingObjectField instances automatically updated at each time")
    plt.close()


def test_animate_environment():
    """Test 2: Animate moving obstacles using built-in animation method."""
    print("\n" + "="*70)
    print("Test 2: Animate Moving Obstacles")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with only moving obstacles
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],  # No static obstacles
        obj_extra_list=create_moving_obstacles(),
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Use the built-in animation method
    env.animate_with_time(
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath="/tmp/test_moving_obstacles.mp4",
        show_time_label=True,
        anim_time=3,
        dpi=100,
    )
    print(f"✓ Saved animation to /tmp/test_moving_obstacles.mp4")
    print("✓ Animation automatically handles MovingObjectField updates")


def test_render_sdf_at_time():
    """Test 3: Render time-dependent SDF field."""
    print("\n" + "="*70)
    print("Test 3: Time-Dependent SDF Field")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with moving obstacles
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],
        obj_extra_list=create_moving_obstacles(),
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        tensor_args=tensor_args
    )

    # Render SDF at different times
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    times = [0.0, 0.33, 0.67, 1.0]

    for ax, t in zip(axes.flat, times):
        env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
        ax.set_title(f'SDF at t={t:.2f}s')

    plt.tight_layout()
    plt.savefig("/tmp/test_sdf_with_moving.png", dpi=150)
    print(f"✓ Saved SDF visualization to /tmp/test_sdf_with_moving.png")
    print("✓ SDF automatically computed with MovingObjectField at correct time")
    plt.close()


def test_animate_sdf():
    """Test 4: Animate SDF field as obstacles move."""
    print("\n" + "="*70)
    print("Test 4: Animate SDF Field")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[],
        obj_extra_list=create_moving_obstacles(),
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Use built-in SDF animation method
    env.animate_sdf_with_extra_objects(
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath="/tmp/test_sdf_animation.mp4",
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=3,
        dpi=100,
    )
    print(f"✓ Saved SDF animation to /tmp/test_sdf_animation.mp4")


def test_comparison_smooth_methods():
    """Test 5: Compare smooth union methods."""
    print("\n" + "="*70)
    print("Test 5: Compare Smoothing Methods")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with overlapping obstacles
    moving_objs = create_moving_obstacles()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    methods = ["Quadratic", "LSE"]
    times = [0.0, 0.5]

    for i, method in enumerate(methods):
        for j, t in enumerate(times):
            env = EnvDynBase(
                limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
                obj_fixed_list=[],
                obj_extra_list=create_moving_obstacles(),  # Create fresh instances
                precompute_sdf_obj_fixed=False,
                precompute_sdf_obj_extra=False,
                time_range=(0.0, 1.0),
                k_smooth=30.0,
                smoothing_method=method,
                tensor_args=tensor_args
            )

            ax = axes[i, j]
            env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
            ax.set_title(f'{method} at t={t:.1f}s')

    plt.tight_layout()
    plt.savefig("/tmp/test_smoothing_comparison.png", dpi=150)
    print(f"✓ Saved smoothing comparison to /tmp/test_smoothing_comparison.png")
    plt.close()


def test_integration_notes():
    """Test 6: Print integration notes."""
    print("\n" + "="*70)
    print("Test 6: Integration with Planning Tasks")
    print("="*70)
    print("Integration example with the new wrapper pattern:")
    print("""
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    MovingObjectField,
    LinearTrajectory
)

# Create moving obstacles as MovingObjectField instances
moving_sphere_prim = MultiSphereField(...)
trajectory = LinearTrajectory(...)
moving_sphere = MovingObjectField(
    primitive_fields=[moving_sphere_prim],
    trajectory=trajectory,
    name="moving_obstacle"
)

# Create environment - just add MovingObjectField to obj_extra_list!
env = EnvDynBase(
    limits=limits,
    obj_fixed_list=static_obstacles,
    obj_extra_list=[moving_sphere],  # Automatic handling!
    time_range=(0.0, 1.0),
)

# Rendering and SDF computation automatically handle time
env.render(ax, time=t)  # Automatically updates moving objects
sdf = env.compute_sdf(points, time=t)  # Time-aware SDF

# Animation methods automatically detect and animate moving objects
env.animate_with_time(...)
env.animate_sdf_with_extra_objects(...)
""")


if __name__ == "__main__":
    print("="*70)
    print("Testing Time-Varying Rendering with Wrapper Pattern")
    print("="*70)
    print("\nKey features:")
    print("- EnvDynBase wraps EnvBase (composition, not inheritance)")
    print("- MovingObjectField automatically detected in object lists")
    print("- No need for moving_obj_list_fn parameter")
    print("- Automatic time-aware rendering and SDF computation")

    # Run tests
    test_render_at_time()
    test_animate_environment()
    test_render_sdf_at_time()
    test_animate_sdf()
    test_comparison_smooth_methods()
    test_integration_notes()

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - /tmp/test_render_at_time.png")
    print("  - /tmp/test_moving_obstacles.mp4")
    print("  - /tmp/test_sdf_with_moving.png")
    print("  - /tmp/test_sdf_animation.mp4")
    print("  - /tmp/test_smoothing_comparison.png")
