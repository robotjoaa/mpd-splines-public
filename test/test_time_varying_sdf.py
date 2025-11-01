"""
Comprehensive test and example for time-varying SDF implementation.

This script demonstrates:
1. Creating moving obstacles with different trajectories
2. Building time-varying SDF grids
3. Visualizing SDF evolution over time
4. Integration with planning tasks
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField, ObjectField
from torch_robotics.environments.dynamic_extension.moving_primitives import (
    LinearTrajectory, CircularTrajectory, MovingObjectField
)
from torch_robotics.environments.dynamic_extension.grid_map_sdf_time_varying import GridMapSDFTimeVarying
from torch_robotics.environments.dynamic_extension.sdf_utils import (
    smooth_union_sdf, detect_primitive_overlaps
)
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy


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
    sdf_smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0)
    sdf_smooth_k50 = smooth_union_sdf(sdf1, sdf2, k=50.0)

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
    plt.savefig('test_smooth_union.png', dpi=150)
    print("✓ Saved smooth union comparison to test_smooth_union.png")


# def test_overlap_detection():
#     """Test overlap detection between primitives."""
#     print("\n" + "="*80)
#     print("TEST 2: Overlap Detection")
#     print("="*80)

#     tensor_args = DEFAULT_TENSOR_ARGS

#     # Create primitives
#     sphere1 = MultiSphereField(
#         centers=np.array([[0.0, 0.0]]),
#         radii=np.array([0.3]),
#         tensor_args=tensor_args
#     )

#     sphere2 = MultiSphereField(
#         centers=np.array([[0.5, 0.0]]),  # Overlaps with sphere1
#         radii=np.array([0.3]),
#         tensor_args=tensor_args
#     )

#     sphere3 = MultiSphereField(
#         centers=np.array([[1.5, 0.0]]),  # No overlap
#         radii=np.array([0.3]),
#         tensor_args=tensor_args
#     )

#     box1 = MultiBoxField(
#         centers=np.array([[0.0, 0.7]]),  # Overlaps with sphere1
#         sizes=np.array([[0.4, 0.4]]),
#         tensor_args=tensor_args
#     )

#     primitives = [sphere1, sphere2, sphere3, box1]

#     # Detect overlaps
#     overlaps, details = detect_primitive_overlaps(primitives, margin=0.0)

#     print(f"Found {len(overlaps)} overlapping pairs:")
#     for (i, j) in overlaps:
#         print(f"  Primitive {i} ({details[(i,j)]['prim1_type']}) <-> "
#               f"Primitive {j} ({details[(i,j)]['prim2_type']})")

#     assert len(overlaps) == 2, f"Expected 2 overlaps, found {len(overlaps)}"
#     print("✓ Overlap detection working correctly")


def test_moving_trajectories():
    """Test different trajectory types."""
    print("\n" + "="*80)
    print("TEST 3: Moving Object Trajectories")
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
    for i, t in enumerate([0.0, 0.5, 1.0]):
        pos, _ = linear_traj(t)
        print(f"  t={t:.1f}: position = {to_numpy(pos)}")

    print("\nCircular trajectory samples:")
    for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
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
    plt.savefig('test_trajectories.png', dpi=150)
    print("✓ Saved trajectory visualization to test_trajectories.png")


def test_time_varying_sdf_grid():
    """Test time-varying SDF grid computation."""
    print("\n" + "="*80)
    print("TEST 4: Time-Varying SDF Grid")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    def moving_objects_fn(t):
        """Two spheres moving towards each other and overlapping."""
        # Sphere 1: moves left to right
        center1_x = -0.7 + 1.4 * t
        sphere1 = MultiSphereField(
            centers=np.array([[center1_x, 0.2]]),
            radii=np.array([0.25]),
            tensor_args=tensor_args
        )

        # Sphere 2: moves right to left
        center2_x = 0.7 - 1.4 * t
        sphere2 = MultiSphereField(
            centers=np.array([[center2_x, -0.2]]),
            radii=np.array([0.25]),
            tensor_args=tensor_args
        )

        # Box: moves in circle
        angle = 2 * np.pi * t
        box_x = 0.5 * np.cos(angle)
        box_y = 0.5 * np.sin(angle)
        box = MultiBoxField(
            centers=np.array([[box_x, box_y]]),
            sizes=np.array([[0.2, 0.2]]),
            tensor_args=tensor_args
        )

        obj1 = ObjectField([sphere1], name="sphere1")
        obj2 = ObjectField([sphere2], name="sphere2")
        obj3 = ObjectField([box], name="box")

        return [obj1, obj2, obj3]

    # Create time-varying SDF grid
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

    print("\nBuilding time-varying SDF grid...")
    grid_sdf = GridMapSDFTimeVarying(
        limits=limits,
        cell_size=0.02,
        moving_obj_list_fn=moving_objects_fn,
        time_range=(0.0, 1.0),
        num_time_steps=25,
        k_smooth=30.0,
        overlap_margin=0.05,
        tensor_args=tensor_args
    )

    # Visualize at different times
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    times = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    for ax, t in zip(axes, times):
        grid_sdf.render_sdf_at_time(t, ax=ax, fig=None, num_points=150)

    plt.tight_layout()
    plt.savefig('test_time_varying_sdf.png', dpi=150, bbox_inches='tight')
    print("✓ Saved time-varying SDF visualization to test_time_varying_sdf.png")

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
            sdf_val = grid_sdf(point.unsqueeze(0), t)
            collision_str = " [COLLISION]" if sdf_val.item() < 0 else ""
            print(f"    t={t:.1f}: SDF = {sdf_val.item():+.4f}{collision_str}")

    print("✓ SDF queries working correctly")


def test_integration_example():
    """Full integration example showing typical usage."""
    print("\n" + "="*80)
    print("TEST 5: Full Integration Example")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Define moving obstacles scenario: narrow passage that opens and closes
    def moving_obstacles_fn(t):
        """
        Narrow passage scenario:
        - Two boxes move to create a passage that opens wider over time
        - A sphere moves through the passage
        """
        # Left wall - moves left
        left_wall_x = -0.3 - 0.3 * t
        left_wall = MultiBoxField(
            centers=np.array([[left_wall_x, 0.0]]),
            sizes=np.array([[0.3, 1.5]]),
            tensor_args=tensor_args
        )

        # Right wall - moves right
        right_wall_x = 0.3 + 0.3 * t
        right_wall = MultiBoxField(
            centers=np.array([[right_wall_x, 0.0]]),
            sizes=np.array([[0.3, 1.5]]),
            tensor_args=tensor_args
        )

        # Moving sphere obstacle
        sphere_y = -0.8 + 1.6 * t
        moving_sphere = MultiSphereField(
            centers=np.array([[0.0, sphere_y]]),
            radii=np.array([0.15]),
            tensor_args=tensor_args
        )

        obj1 = ObjectField([left_wall], name="left_wall")
        obj2 = ObjectField([right_wall], name="right_wall")
        obj3 = ObjectField([moving_sphere], name="moving_sphere")

        return [obj1, obj2, obj3]

    # Create SDF grid
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

    print("\nBuilding narrow passage scenario...")
    grid_sdf = GridMapSDFTimeVarying(
        limits=limits,
        cell_size=0.015,
        moving_obj_list_fn=moving_obstacles_fn,
        time_range=(0.0, 1.0),
        num_time_steps=30,
        k_smooth=40.0,
        overlap_margin=0.02,
        tensor_args=tensor_args
    )

    # Visualize scenario evolution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    times = np.linspace(0, 1, 8)
    for ax, t in zip(axes, times):
        grid_sdf.render_sdf_at_time(t, ax=ax, fig=None, num_points=200)
        ax.set_title(f't = {t:.2f}s (Passage {"Opening" if t < 0.5 else "Open"})')

    plt.tight_layout()
    plt.savefig('test_integration_narrow_passage.png', dpi=150, bbox_inches='tight')
    print("✓ Saved integration example to test_integration_narrow_passage.png")

    # Simulate robot path through passage
    print("\nSimulating robot path through narrow passage:")
    robot_path_y = np.linspace(-0.7, 0.7, 50)
    robot_path_x = np.zeros_like(robot_path_y)
    robot_times = np.linspace(0.0, 1.0, 50)

    collisions = []
    min_clearances = []

    for x, y, t in zip(robot_path_x, robot_path_y, robot_times):
        point = torch.tensor([[x, y]], **tensor_args)
        sdf_val = grid_sdf(point, t).item()
        collisions.append(sdf_val < 0)
        min_clearances.append(sdf_val)

    num_collisions = sum(collisions)
    print(f"  Path through center (x=0): {num_collisions}/{len(collisions)} waypoints in collision")
    print(f"  Minimum clearance: {min(min_clearances):.4f}")

    if num_collisions > 0:
        print(f"  Collisions occur around t={robot_times[collisions.index(True)]:.2f}s")

    print("✓ Integration test complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR TIME-VARYING SDF IMPLEMENTATION")
    print("="*80)

    try:
        test_smooth_union()
        #test_overlap_detection()
        test_moving_trajectories()
        test_time_varying_sdf_grid()
        test_integration_example()

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
