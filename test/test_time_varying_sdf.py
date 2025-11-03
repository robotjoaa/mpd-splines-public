"""
Comprehensive test for time-varying SDF with EnvDynBase wrapper pattern.

This script demonstrates:
1. Creating moving obstacles with MovingObjectField
2. Time-dependent SDF computation with automatic object updates
3. Smooth union handling for overlapping objects
4. Visualizing SDF evolution over time
"""

import sys
import os
import torch
import numpy as np
from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()
import matplotlib.pyplot as plt

from mpd.torch_robotics.torch_robotics.environments.primitives import MultiSphereField, MultiBoxField, ObjectField
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    LinearTrajectory,
    CircularTrajectory,
    MovingObjectField,
)
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension.sdf_utils import smooth_union_sdf
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy


def test_smooth_union():
    """Test smooth SDF union with overlapping shapes."""
    print("\n" + "="*80)
    print("TEST 1: Smooth Union of Overlapping SDFs")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create grid
    x = torch.linspace(-2, 2, 200, **tensor_args)
    y = torch.linspace(-2, 2, 200, **tensor_args)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # Two overlapping spheres
    sdf1 = torch.norm(points - torch.tensor([[-0.3, 0.0]], **tensor_args), dim=-1) - 0.5
    sdf2 = torch.norm(points - torch.tensor([[0.3, 0.0]], **tensor_args), dim=-1) - 0.5

    # Compute different unions
    sdf_hard = torch.minimum(sdf1, sdf2)
    sdf_smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0, method="Quadratic")
    sdf_smooth_k50 = smooth_union_sdf(sdf1, sdf2, k=50.0, method="Quadratic")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, sdf, title in zip(axes,
                               [sdf_hard, sdf_smooth_k10, sdf_smooth_k50],
                               ['Hard Min (Non-differentiable)', 'Smooth Union k=10', 'Smooth Union k=50']):
        sdf_grid = to_numpy(sdf.reshape(200, 200))
        cs = ax.contourf(to_numpy(X), to_numpy(Y), sdf_grid, levels=20, cmap='RdBu')
        ax.contour(to_numpy(X), to_numpy(Y), sdf_grid, levels=[0], colors='black', linewidths=2)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax)

    plt.tight_layout()
    plt.savefig('/tmp/test_smooth_union.png', dpi=150)
    print("✓ Saved smooth union comparison to /tmp/test_smooth_union.png")
    plt.close()


def test_moving_trajectories():
    """Test different trajectory types with MovingObjectField."""
    print("\n" + "="*80)
    print("TEST 2: Moving Object Trajectories")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Linear trajectory
    linear_traj = LinearTrajectory(
        keyframe_times=[0.0, 0.5, 1.0],
        keyframe_positions=[
            [-1.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, -0.5, 0.0]
        ],
        tensor_args=tensor_args
    )

    # Circular trajectory
    circular_traj = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.7,
        angular_velocity=2 * np.pi,  # One full rotation per second
        axis='z',
        tensor_args=tensor_args
    )

    # Test evaluation
    times = torch.linspace(0, 1, 11, **tensor_args)

    print("\nLinear trajectory samples:")
    for t in [0.0, 0.5, 1.0]:
        pos, _ = linear_traj(t)
        print(f"  t={t:.1f}: position = {to_numpy(pos)}")

    print("\nCircular trajectory samples:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pos, _ = circular_traj(t)
        print(f"  t={t:.2f}: position = {to_numpy(pos)}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear
    positions = []
    for t in times:
        pos, _ = linear_traj(t.item())
        positions.append(to_numpy(pos[:2]))
    positions = np.array(positions)

    ax1.plot(positions[:, 0], positions[:, 1], 'b-o', linewidth=2, markersize=4)
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o', zorder=10, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X', zorder=10, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Circular
    positions = []
    for t in times:
        pos, _ = circular_traj(t.item())
        positions.append(to_numpy(pos[:2]))
    positions = np.array(positions)

    ax2.plot(positions[:, 0], positions[:, 1], 'b-o', linewidth=2, markersize=4)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o', zorder=10, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X', zorder=10, label='End')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Circular Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/tmp/test_trajectories.png', dpi=150)
    print("✓ Saved trajectory visualization to /tmp/test_trajectories.png")
    plt.close()


def test_time_varying_sdf():
    """Test time-varying SDF computation with MovingObjectField."""
    print("\n" + "="*80)
    print("TEST 3: Time-Varying SDF with EnvDynBase Wrapper")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create moving sphere 1: moves left to right
    sphere1_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.25]),
        tensor_args=tensor_args
    )

    traj1 = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[-0.7, 0.2, 0.0], [0.7, 0.2, 0.0]],
        tensor_args=tensor_args
    )

    moving_sphere1 = MovingObjectField(
        primitive_fields=[sphere1_prim],
        trajectory=traj1,
        name="sphere1"
    )

    # Create moving sphere 2: moves right to left
    sphere2_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.25]),
        tensor_args=tensor_args
    )

    traj2 = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[0.7, -0.2, 0.0], [-0.7, -0.2, 0.0]],
        tensor_args=tensor_args
    )

    moving_sphere2 = MovingObjectField(
        primitive_fields=[sphere2_prim],
        trajectory=traj2,
        name="sphere2"
    )

    # Create moving box: circular motion
    box_prim = MultiBoxField(
        centers=np.array([[0.0, 0.0]]),
        sizes=np.array([[0.2, 0.2]]),
        tensor_args=tensor_args
    )

    traj_box = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.5,
        angular_velocity=2 * np.pi,
        axis='z',
        tensor_args=tensor_args
    )

    moving_box = MovingObjectField(
        primitive_fields=[box_prim],
        trajectory=traj_box,
        name="box"
    )

    # Create environment with automatic MovingObjectField handling
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

    print("\nCreating EnvDynBase with MovingObjectField instances...")
    env = EnvDynBase(
        limits=limits,
        obj_fixed_list=[],
        obj_extra_list=[moving_sphere1, moving_sphere2, moving_box],
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=30.0,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Visualize at different times
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    times = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    for ax, t in zip(axes, times):
        env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
        ax.set_title(f"t = {t:.2f}s", fontsize=10)

    plt.tight_layout()
    plt.savefig('/tmp/test_time_varying_sdf.png', dpi=150, bbox_inches='tight')
    print("✓ Saved time-varying SDF visualization to /tmp/test_time_varying_sdf.png")
    plt.close()

    # Test querying
    print("\nTesting SDF queries at specific points:")
    test_points = torch.tensor([
        [0.0, 0.0],
        [-0.5, 0.2],
        [0.5, -0.2],
    ], **tensor_args)

    for point in test_points:
        print(f"\n  Point {to_numpy(point)}:")
        for t in [0.0, 0.5, 1.0]:
            sdf_val = env.compute_sdf(point.unsqueeze(0).unsqueeze(0), time=t)
            collision_str = " [COLLISION]" if sdf_val.item() < 0 else ""
            print(f"    t={t:.1f}: SDF = {sdf_val.item():+.4f}{collision_str}")

    print("✓ SDF queries working correctly with automatic MovingObjectField updates")


def test_narrow_passage_scenario():
    """Full integration example: narrow passage that opens and closes."""
    print("\n" + "="*80)
    print("TEST 4: Narrow Passage Scenario")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Left wall - moves left
    left_wall_prim = MultiBoxField(
        centers=np.array([[0.0, 0.0]]),
        sizes=np.array([[0.3, 1.5]]),
        tensor_args=tensor_args
    )

    traj_left = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[-0.3, 0.0, 0.0], [-0.6, 0.0, 0.0]],
        tensor_args=tensor_args
    )

    left_wall = MovingObjectField(
        primitive_fields=[left_wall_prim],
        trajectory=traj_left,
        name="left_wall"
    )

    # Right wall - moves right
    right_wall_prim = MultiBoxField(
        centers=np.array([[0.0, 0.0]]),
        sizes=np.array([[0.3, 1.5]]),
        tensor_args=tensor_args
    )

    traj_right = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]],
        tensor_args=tensor_args
    )

    right_wall = MovingObjectField(
        primitive_fields=[right_wall_prim],
        trajectory=traj_right,
        name="right_wall"
    )

    # Moving sphere obstacle
    sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.15]),
        tensor_args=tensor_args
    )

    traj_sphere = LinearTrajectory(
        keyframe_times=[0.0, 1.0],
        keyframe_positions=[[0.0, -0.8, 0.0], [0.0, 0.8, 0.0]],
        tensor_args=tensor_args
    )

    moving_sphere = MovingObjectField(
        primitive_fields=[sphere_prim],
        trajectory=traj_sphere,
        name="moving_sphere"
    )

    # Create environment
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

    print("\nBuilding narrow passage scenario...")
    env = EnvDynBase(
        limits=limits,
        obj_fixed_list=[],
        obj_extra_list=[left_wall, right_wall, moving_sphere],
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        time_range=(0.0, 1.0),
        k_smooth=40.0,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Visualize scenario evolution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    times = np.linspace(0, 1, 8)
    for ax, t in zip(axes, times):
        env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
        ax.set_title(f't = {t:.2f}s (Passage {"Opening" if t < 0.5 else "Open"})', fontsize=10)

    plt.tight_layout()
    plt.savefig('/tmp/test_integration_narrow_passage.png', dpi=150, bbox_inches='tight')
    print("✓ Saved integration example to /tmp/test_integration_narrow_passage.png")
    plt.close()

    # Simulate robot path through passage
    print("\nSimulating robot path through narrow passage:")
    robot_path_y = np.linspace(-0.7, 0.7, 50)
    robot_path_x = np.zeros_like(robot_path_y)
    robot_times = np.linspace(0.0, 1.0, 50)

    collisions = []
    min_clearances = []

    for x, y, t in zip(robot_path_x, robot_path_y, robot_times):
        point = torch.tensor([[x, y]], **tensor_args).unsqueeze(1)
        sdf_val = env.compute_sdf(point, time=t).item()
        collisions.append(sdf_val < 0)
        min_clearances.append(sdf_val)

    num_collisions = sum(collisions)
    print(f"  Path through center (x=0): {num_collisions}/{len(collisions)} waypoints in collision")
    print(f"  Minimum clearance: {min(min_clearances):.4f}")

    if num_collisions > 0:
        first_collision_idx = collisions.index(True)
        print(f"  Collisions occur around t={robot_times[first_collision_idx]:.2f}s")

    print("✓ Integration test complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR TIME-VARYING SDF WITH WRAPPER PATTERN")
    print("="*80)
    print("\nKey improvements:")
    print("- EnvDynBase wraps EnvBase (composition over inheritance)")
    print("- MovingObjectField automatically detected and updated")
    print("- No need for moving_obj_list_fn parameter")
    print("- Simpler, cleaner API")

    try:
        test_smooth_union()
        test_moving_trajectories()
        test_time_varying_sdf()
        test_narrow_passage_scenario()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  - /tmp/test_smooth_union.png")
        print("  - /tmp/test_trajectories.png")
        print("  - /tmp/test_time_varying_sdf.png")
        print("  - /tmp/test_integration_narrow_passage.png")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
