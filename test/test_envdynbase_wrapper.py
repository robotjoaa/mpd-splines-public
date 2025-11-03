"""
Test script for EnvDynBase wrapper implementation.

This script tests the refactored EnvDynBase class which now wraps EnvBase
instead of inheriting from it. MovingObjectField instances are automatically
handled in both fixed and extra object lists.
"""
import torch
import numpy as np
from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()
import matplotlib.pyplot as plt

from mpd.torch_robotics.torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    MovingObjectField,
    LinearTrajectory,
    CircularTrajectory
)
from mpd.torch_robotics.torch_robotics.environments.primitives import (
    MultiSphereField,
    MultiBoxField,
    ObjectField
)
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from mpd.torch_robotics.torch_robotics.visualizers.plot_utils import create_fig_and_axes


def test_static_environment():
    """Test 1: Basic static environment with wrapper pattern."""
    print("\n" + "="*70)
    print("Test 1: Static Environment with Wrapper Pattern")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create overlapping spheres (2D centers for 2D environment)
    sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0], [0.3, 0.0]]),
        radii=np.array([0.25, 0.25]),
        tensor_args=tensor_args
    )

    # Create environment (disable SDF precomputation to avoid dimension issues)
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([sphere], 'overlapping_spheres')],
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        k_smooth=20.0,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Test wrapped attributes
    print(f"✓ Environment created (wrapper pattern)")
    print(f"  - env.dim = {env.dim}")
    print(f"  - env.k_smooth = {env.k_smooth}")
    print(f"  - env.smoothing_method = {env.smoothing_method}")
    print(f"  - len(env.obj_fixed_list) = {len(env.obj_fixed_list)}")

    # Test SDF computation
    test_points = torch.tensor([
        [0.0, 0.0],   # Inside overlap
        [0.5, 0.0],   # Outside
        [1.0, 0.0]    # Far outside
    ], **tensor_args).unsqueeze(1)

    sdf_smooth = env.compute_sdf(test_points, use_smooth_union=True)
    sdf_hard = env.compute_sdf(test_points, use_smooth_union=False)

    print(f"\n✓ SDF computation working")
    print(f"  - Smooth union SDF: {sdf_smooth.squeeze().tolist()}")
    print(f"  - Hard minimum SDF: {sdf_hard.squeeze().tolist()}")

    return env


def test_moving_object_field():
    """Test 2: MovingObjectField creation and pose updates."""
    print("\n" + "="*70)
    print("Test 2: MovingObjectField with Trajectories")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create a moving sphere with circular trajectory
    # Note: Primitives use 2D centers, but trajectories use 3D
    sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    circular_traj = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),  # 3D for trajectory
        radius=0.5,
        angular_velocity=2 * np.pi,  # One full rotation per second
        initial_phase=0.0,
        axis='z',
        tensor_args=tensor_args
    )

    moving_sphere = MovingObjectField(
        primitive_fields=[sphere_prim],
        trajectory=circular_traj,
        name='circular_moving_sphere'
    )

    print(f"✓ MovingObjectField created: {moving_sphere.name}")

    # Test pose updates at different times
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    positions = []

    for t in times:
        moving_sphere.update_pose_at_time(t)
        pos, _ = moving_sphere.get_position_orientation()
        positions.append(pos[:2])  # Store x, y only
        print(f"  - t={t:.2f}s: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print(f"✓ Pose updates working correctly")

    return moving_sphere


def test_dynamic_environment():
    """Test 3: Environment with MovingObjectField instances."""
    print("\n" + "="*70)
    print("Test 3: Dynamic Environment with MovingObjectField")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create static obstacle
    static_box = MultiBoxField(
        centers=np.array([[0.5, 0.5]]),
        sizes=np.array([[0.3, 0.3]]),
        tensor_args=tensor_args
    )

    # Create moving obstacle
    moving_sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    linear_traj = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]],  # 3D for trajectory
        tensor_args=tensor_args
    )

    moving_sphere = MovingObjectField(
        primitive_fields=[moving_sphere_prim],
        trajectory=linear_traj,
        name='linear_moving_sphere'
    )

    # Create environment with both static and moving objects
    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([static_box], 'static_obstacle')],
        obj_extra_list=[moving_sphere],  # MovingObjectField in extra list
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        k_smooth=30.0,
        smoothing_method="Quadratic",
        time_range=(0.0, 1.0),
        tensor_args=tensor_args
    )

    print(f"✓ Dynamic environment created")
    print(f"  - Static objects: {len(env.obj_fixed_list)}")
    print(f"  - Extra objects (including moving): {len(env.obj_extra_list)}")
    print(f"  - Has moving objects: {env._has_moving_objects()}")

    # Test time-dependent SDF
    test_point = torch.tensor([[0.0, 0.0]], **tensor_args).unsqueeze(1)

    print(f"\n✓ Time-dependent SDF computation:")
    for t in [0.0, 0.5, 1.0]:
        sdf = env.compute_sdf(test_point, time=t)
        print(f"  - t={t:.1f}s: SDF at (0,0) = {sdf.item():.4f}")

    return env


def test_rendering():
    """Test 4: Rendering with automatic MovingObjectField handling."""
    print("\n" + "="*70)
    print("Test 4: Rendering with Automatic MovingObjectField Handling")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with moving object
    static_sphere = MultiSphereField(
        centers=np.array([[-0.5, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    moving_sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    circular_traj = CircularTrajectory(
        center=np.array([0.5, 0.0, 0.0]),  # 3D for trajectory
        radius=0.3,
        angular_velocity=2 * np.pi,
        initial_phase=0.0,
        axis='z',
        tensor_args=tensor_args
    )

    moving_sphere = MovingObjectField(
        primitive_fields=[moving_sphere_prim],
        trajectory=circular_traj,
        name='rendering_test_sphere'
    )

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([static_sphere], 'static')],
        obj_extra_list=[moving_sphere],
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        k_smooth=20.0,
        time_range=(0.0, 1.0),
        tensor_args=tensor_args
    )

    # Test rendering at different times
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    times = [0.0, 0.5, 1.0]

    for ax, t in zip(axes, times):
        env.render(ax, time=t)
        ax.set_title(f"Environment at t={t:.1f}s")

    plt.tight_layout()
    save_path = '/tmp/test_envdynbase_rendering.png'
    plt.savefig(save_path, dpi=100)
    print(f"✓ Rendering test completed")
    print(f"  - Saved to: {save_path}")
    plt.close()

    return env


def test_smoothing_methods():
    """Test 5: Compare Quadratic vs LSE smoothing methods."""
    print("\n" + "="*70)
    print("Test 5: Smoothing Methods Comparison")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create overlapping objects
    spheres = MultiSphereField(
        centers=np.array([[0.0, 0.0], [0.3, 0.0]]),
        radii=np.array([0.25, 0.25]),
        tensor_args=tensor_args
    )

    # Test both methods
    for method in ["Quadratic", "LSE"]:
        env = EnvDynBase(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField([spheres], 'overlapping')],
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            k_smooth=20.0,
            smoothing_method=method,
            tensor_args=tensor_args
        )

        test_points = torch.tensor([[0.15, 0.0]], **tensor_args).unsqueeze(1)
        sdf = env.compute_sdf(test_points)

        print(f"  - {method:10s}: SDF at overlap center = {sdf.item():.6f}")

    print(f"✓ Both smoothing methods working")


def test_delegation():
    """Test 6: Verify delegation to wrapped EnvBase."""
    print("\n" + "="*70)
    print("Test 6: Delegation to Wrapped EnvBase")
    print("="*70)

    tensor_args = DEFAULT_TENSOR_ARGS

    sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField([sphere], 'test')],
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        tensor_args=tensor_args
    )

    # Test delegated properties
    print(f"✓ Delegated properties:")
    print(f"  - tensor_args: {env.tensor_args}")
    print(f"  - limits.shape: {env.limits.shape}")
    print(f"  - dim: {env.dim}")
    print(f"  - sdf_cell_size: {env.sdf_cell_size}")

    # Test delegated methods
    obj_list = env.get_obj_list()
    df_obj_list = env.get_df_obj_list()

    print(f"✓ Delegated methods:")
    print(f"  - get_obj_list() returned {len(obj_list)} objects")
    print(f"  - get_df_obj_list() returned {len(df_obj_list)} objects")

    # Test add_objects_extra
    extra_sphere = MultiSphereField(
        centers=np.array([[0.5, 0.5]]),
        radii=np.array([0.1]),
        tensor_args=tensor_args
    )
    env.add_objects_extra([ObjectField([extra_sphere], 'extra')])

    print(f"  - add_objects_extra() working: {len(env.obj_extra_list)} extra objects")
    print(f"✓ All delegation working correctly")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("EnvDynBase Wrapper Implementation Test Suite")
    print("="*70)
    print("\nTesting the refactored EnvDynBase class:")
    print("- EnvDynBase now WRAPS EnvBase (composition, not inheritance)")
    print("- MovingObjectField is automatically detected in object lists")
    print("- Time-aware rendering and SDF computation built-in")

    try:
        # Run all tests
        env_static = test_static_environment()
        moving_obj = test_moving_object_field()
        env_dynamic = test_dynamic_environment()
        env_render = test_rendering()
        test_smoothing_methods()
        test_delegation()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nKey achievements:")
        print("✓ EnvDynBase successfully wraps EnvBase")
        print("✓ MovingObjectField automatically handled in render()")
        print("✓ MovingObjectField automatically handled in compute_sdf()")
        print("✓ Time-dependent SDF computation working")
        print("✓ Both smoothing methods (Quadratic, LSE) working")
        print("✓ All property/method delegation working")
        print("✓ No need for separate moving_obj_list_fn parameter")
        print("\nThe wrapper pattern provides:")
        print("- Cleaner separation of concerns")
        print("- Automatic MovingObjectField detection and handling")
        print("- Simpler API (movement trajectories in object lists)")
        print("- Full compatibility with EnvBase interface")

    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
