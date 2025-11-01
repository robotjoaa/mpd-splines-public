"""
Test composed SDF with Jacobian computation and gradient visualization.

This test verifies that:
1. SDFs can be composed with smooth union
2. Jacobians can be computed through the composed SDF
3. Gradients are continuous and well-behaved
4. Visualization matches expectations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

from torch_robotics.environments.primitives import (
    MultiSphereField, MultiBoxField, ObjectField
)
from torch_robotics.environments.dynamic_extension.env_dyn_base import EnvDynBase
from torch_robotics.environments.dynamic_extension.sdf_utils import smooth_union_sdf
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy


def test_smooth_union_differentiability():
    """Test that smooth union maintains differentiability."""
    print("\n" + "="*80)
    print("TEST 1: Smooth Union Differentiability")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create test point that requires gradient
    x = torch.tensor([[0.0, 0.0]], **tensor_args, requires_grad=True)

    # Two SDFs
    center1 = torch.tensor([[-0.3, 0.0]], **tensor_args)
    center2 = torch.tensor([[0.3, 0.0]], **tensor_args)

    sdf1 = torch.norm(x - center1, dim=-1) - 0.5
    sdf2 = torch.norm(x - center2, dim=-1) - 0.5

    # Smooth union
    sdf_smooth = smooth_union_sdf(sdf1, sdf2, k=20.0)

    # Compute gradient
    sdf_smooth.backward()

    print(f"Point: {to_numpy(x[0])}")
    print(f"SDF1: {sdf1.item():.4f}")
    print(f"SDF2: {sdf2.item():.4f}")
    print(f"Smooth Union: {sdf_smooth.item():.4f}")
    print(f"Gradient: {to_numpy(x.grad[0])}")

    assert x.grad is not None, "Gradient should be computed"
    assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Gradient should not contain Inf"

    print("✓ Smooth union is differentiable")


def test_jacobian_computation():
    """Test Jacobian computation for composed SDF."""
    print("\n" + "="*80)
    print("TEST 2: Jacobian Computation")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with overlapping objects
    obj_list = [
        MultiSphereField(
            centers=np.array([[0.0, 0.0], [0.4, 0.0]]),  # Overlapping
            radii=np.array([0.3, 0.3]),
            tensor_args=tensor_args
        ),
    ]

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "overlapping_spheres")],
        k_smooth=1,
        tensor_args=tensor_args
    )

    # Test points
    test_points = torch.tensor([
        [0.0, 0.0],    # Center (inside overlap)
        [0.2, 0.0],    # In overlap region
        [-0.5, 0.0],   # Near first sphere
        [0.9, 0.0],    # Near second sphere
        [0.0, 0.5],    # Away from both
    ], **tensor_args)

    print(f"\nTesting {len(test_points)} points:")
    print(f"{'Point':<20} {'SDF':<15} {'Gradient':<30} {'Grad Norm':<15}")
    print("-" * 80)

    for i, point in enumerate(test_points):
        point_input = point.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 2)
        point_input.requires_grad_(True)

        # Compute SDF
        sdf = env.compute_sdf(point_input, use_smooth_union=True)

        # Compute Jacobian
        f = lambda x: env.compute_sdf(x, use_smooth_union=True).sum()
        jac = jacobian(f, point_input)

        sdf_val = sdf.item()
        grad = to_numpy(jac.squeeze())
        grad_norm = np.linalg.norm(grad)

        print(f"{str(to_numpy(point)):<20} {sdf_val:+.4f}{'':>9} "
              f"[{grad[0]:+.4f}, {grad[1]:+.4f}]{'':>10} {grad_norm:.4f}")

        # Verify gradient properties
        assert not np.isnan(grad).any(), f"Gradient at point {i} contains NaN"
        assert not np.isinf(grad).any(), f"Gradient at point {i} contains Inf"

    print("✓ Jacobian computation successful for all points")


def test_gradient_continuity():
    """Test that gradients are continuous across the field."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Continuity")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment
    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.2, 0.0], [0.2, 0.0]]),
            radii=np.array([0.25, 0.25]),
            tensor_args=tensor_args
        ),
    ]

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.1,
        tensor_args=tensor_args
    )

    # Sample points along a line through the overlap region
    num_points = 50
    x_coords = torch.linspace(-0.5, 0.5, num_points, **tensor_args)
    y_coord = torch.tensor([0.0], **tensor_args)

    sdf_values = []
    grad_x_values = []

    for x in x_coords:
        point = torch.stack([x, y_coord.squeeze()]).unsqueeze(0).unsqueeze(0)
        point.requires_grad_(True)

        sdf = env.compute_sdf(point, use_smooth_union=True)
        f = lambda p: env.compute_sdf(p, use_smooth_union=True).sum()
        jac = jacobian(f, point)

        sdf_values.append(sdf.item())
        grad_x_values.append(jac[0, 0, 0].item())

    sdf_values = np.array(sdf_values)
    grad_x_values = np.array(grad_x_values)

    # Check for continuity (no sudden jumps)
    sdf_diff = np.abs(np.diff(sdf_values))
    grad_diff = np.abs(np.diff(grad_x_values))

    max_sdf_jump = np.max(sdf_diff)
    max_grad_jump = np.max(grad_diff)

    print(f"Maximum SDF jump: {max_sdf_jump:.6f}")
    print(f"Maximum gradient jump: {max_grad_jump:.6f}")

    # Plot continuity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(to_numpy(x_coords), sdf_values, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x position')
    ax1.set_ylabel('SDF value')
    ax1.set_title('SDF along x-axis (y=0)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(to_numpy(x_coords), grad_x_values, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('x position')
    ax2.set_ylabel('∂SDF/∂x')
    ax2.set_title('SDF Gradient along x-axis')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_gradient_continuity.png', dpi=150)
    print("✓ Saved gradient continuity plot to test_gradient_continuity.png")

    # Verify smoothness (should be small jumps with smooth union)
    assert max_sdf_jump < 0.1, f"SDF has large discontinuity: {max_sdf_jump}"
    assert max_grad_jump < 2.0, f"Gradient has large discontinuity: {max_grad_jump}"

    print("✓ Gradients are continuous")


def test_comparison_hard_vs_smooth():
    """Compare hard minimum vs smooth union with full visualization."""
    print("\n" + "="*80)
    print("TEST 4: Hard Minimum vs Smooth Union Comparison")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with clear overlap
    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.15, 0.0], [0.15, 0.0]]),
            radii=np.array([0.3, 0.3]),
            tensor_args=tensor_args
        ),
    ]

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "overlapping")],
        k_smooth=0.1,
        tensor_args=tensor_args
    )

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: SDF visualizations
    env.render_sdf(ax=axes[0, 0], fig=fig, use_smooth_union=False)
    axes[0, 0].set_title('Hard Minimum SDF\n(Non-differentiable at overlap)', fontsize=12)

    env.render_sdf(ax=axes[0, 1], fig=fig, use_smooth_union=True)
    axes[0, 1].set_title(f'Smooth Union SDF (k={env.k_smooth})\n(Fully differentiable)', fontsize=12)

    # Difference plot
    xs = torch.linspace(env.limits_np[0][0], env.limits_np[1][0], steps=200, **tensor_args)
    ys = torch.linspace(env.limits_np[0][1], env.limits_np[1][1], steps=200, **tensor_args)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    X_flat = torch.flatten(X)
    Y_flat = torch.flatten(Y)
    points = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, 2)

    sdf_hard = env.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=False)
    sdf_smooth = env.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=True)

    print(sdf_hard.shape, sdf_smooth.shape)
    sdf_diff = torch.abs(sdf_hard - sdf_smooth)

    ctf = axes[0, 2].contourf(to_numpy(X), to_numpy(Y), to_numpy(sdf_diff), levels=20, cmap='hot')
    axes[0, 2].contour(to_numpy(X), to_numpy(Y), to_numpy(sdf_hard), levels=[0], colors='blue', linewidths=2, linestyles='--')
    axes[0, 2].contour(to_numpy(X), to_numpy(Y), to_numpy(sdf_smooth), levels=[0], colors='green', linewidths=2)
    fig.colorbar(ctf, ax=axes[0, 2])
    axes[0, 2].set_title('Absolute Difference\n(Blue=Hard, Green=Smooth)', fontsize=12)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')

    # Row 2: Gradient visualizations
    env.render_grad_sdf(ax=axes[1, 0], fig=fig, use_smooth_union=False)
    axes[1, 0].set_title('Hard Minimum Gradients\n(May have discontinuities)', fontsize=12)

    env.render_grad_sdf(ax=axes[1, 1], fig=fig, use_smooth_union=True)
    axes[1, 1].set_title('Smooth Union Gradients\n(Continuous everywhere)', fontsize=12)

    # Gradient magnitude comparison along a line
    x_line = torch.linspace(-0.6, 0.6, 100, **tensor_args)
    y_line = torch.zeros_like(x_line)

    grad_norms_hard = []
    grad_norms_smooth = []

    for x_val in x_line:
        point = torch.stack([x_val, y_line[0]]).unsqueeze(0).unsqueeze(0)
        point.requires_grad_(True)

        # Hard minimum gradient
        f_hard = lambda p: env.compute_sdf(p, use_smooth_union=False).sum()
        jac_hard = jacobian(f_hard, point)
        grad_norms_hard.append(torch.norm(jac_hard).item())

        # Smooth union gradient
        f_smooth = lambda p: env.compute_sdf(p, use_smooth_union=True).sum()
        jac_smooth = jacobian(f_smooth, point)
        grad_norms_smooth.append(torch.norm(jac_smooth).item())

    axes[1, 2].plot(to_numpy(x_line), grad_norms_hard, 'b-', linewidth=2, label='Hard Minimum', alpha=0.7)
    axes[1, 2].plot(to_numpy(x_line), grad_norms_smooth, 'g-', linewidth=2, label='Smooth Union')
    axes[1, 2].set_xlabel('x position (y=0)')
    axes[1, 2].set_ylabel('Gradient Magnitude ||∇SDF||')
    axes[1, 2].set_title('Gradient Magnitude Comparison\nalong y=0', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_hard_vs_smooth_comprehensive.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comprehensive comparison to test_hard_vs_smooth_comprehensive.png")


def test_multiple_overlaps():
    """Test environment with multiple overlapping regions."""
    print("\n" + "="*80)
    print("TEST 5: Multiple Overlapping Objects")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create complex environment
    obj_list = [
        MultiSphereField(
            centers=np.array([
                [0.0, 0.0],
                [0.3, 0.0],
                [0.15, 0.26],
            ]),
            radii=np.array([0.2, 0.2, 0.2]),
            tensor_args=tensor_args
        ),
        MultiBoxField(
            centers=np.array([[-0.4, 0.4]]),
            sizes=np.array([[0.3, 0.3]]),
            tensor_args=tensor_args
        ),
    ]

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "complex")],
        k_smooth=0.05,
        tensor_args=tensor_args
    )

    # # Detect overlaps
    # overlaps, details = env.detect_overlaps()
    # print(f"Detected {len(overlaps)} overlapping primitive pairs:")
    # for (i, j) in overlaps:
    #     print(f"  Primitives {i} <-> {j}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    env.render_sdf(ax=axes[0], fig=fig, use_smooth_union=True)
    #env.render(axes[0])
    axes[0].set_title(f'SDF with Overlapping Pairs')

    env.render_grad_sdf(ax=axes[1], fig=fig, use_smooth_union=True)
    #env.render(axes[1])
    axes[1].set_title('Gradient Field')

    plt.tight_layout()
    plt.savefig('test_multiple_overlaps.png', dpi=150)
    print("✓ Saved multiple overlaps visualization to test_multiple_overlaps.png")

    # Test Jacobian at several points
    test_points = torch.tensor([
        [0.15, 0.13],  # Center of triangle
        [-0.4, 0.4],   # Center of box
        [0.5, 0.5],    # Outside all objects
    ], **tensor_args)

    print("\nJacobian test at key points:")
    for point in test_points:
        point_input = point.unsqueeze(0).unsqueeze(0)
        point_input.requires_grad_(True)

        f = lambda x: env.compute_sdf(x, use_smooth_union=True).sum()
        jac = jacobian(f, point_input)

        assert not torch.isnan(jac).any(), "Jacobian contains NaN"
        assert not torch.isinf(jac).any(), "Jacobian contains Inf"

        print(f"  Point {to_numpy(point)}: ✓ Valid Jacobian")

    print("✓ All Jacobians computed successfully")


def run_all_tests():
    """Run all Jacobian and gradient tests."""
    print("\n" + "="*80)
    print("RUNNING COMPOSED SDF JACOBIAN TESTS")
    print("="*80)

    try:
        test_smooth_union_differentiability()
        test_jacobian_computation()
        test_gradient_continuity()
        test_comparison_hard_vs_smooth()
        test_multiple_overlaps()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  - test_gradient_continuity.png")
        print("  - test_hard_vs_smooth_comprehensive.png")
        print("  - test_multiple_overlaps.png")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
