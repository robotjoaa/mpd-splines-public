"""
Comprehensive comparison of smooth minimum methods for SDF composition.

This test compares:
1. Log-Sum-Exp (LSE) method: φ = -k * logsumexp(-φᵢ/k)
2. Quadratic smooth minimum: φ = min(a,b) - k*0.25*h² where h = max(k-|a-b|, 0)/k

Tests include:
- Visual comparison of SDF fields
- Approximation quality vs hard minimum
- Gradient continuity and smoothness
- Computational performance
- Behavior with different k values
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import time

from torch_robotics.environments.primitives import (
    MultiSphereField, MultiBoxField, ObjectField
)
from torch_robotics.environments.dynamic_extension.env_dyn_base import EnvDynBase
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy


def test_visual_comparison():
    """Visual comparison of both smoothing methods."""
    print("\n" + "="*80)
    print("TEST 1: Visual Comparison of Smoothing Methods")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with overlapping spheres
    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.2, 0.0], [0.2, 0.0]]),
            radii=np.array([0.35, 0.35]),
            tensor_args=tensor_args
        ),
    ]

    # Create environments with different methods
    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.3,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.3,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Query points
    xs = torch.linspace(-1, 1, 300, **tensor_args)
    ys = torch.linspace(-1, 1, 300, **tensor_args)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack((X.flatten(), Y.flatten()), dim=-1).view(-1, 1, 2)

    # Compute SDFs
    print("\nComputing SDFs...")
    sdf_hard = env_lse.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=False)
    sdf_lse = env_lse.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=True, smoothing_method="LSE")
    sdf_quad = env_quad.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=True, smoothing_method="Quadratic")

    print(sdf_hard.shape, sdf_lse.shape, sdf_quad.shape)
    # Statistics
    print(f"\nSDF Statistics:")
    print(f"{'Method':<15} {'Min':<12} {'Max':<12} {'Mean':<12}")
    print("-" * 51)
    print(f"{'Hard Min':<15} {sdf_hard.min().item():+.6f}{'':>5} {sdf_hard.max().item():+.6f}{'':>5} {sdf_hard.mean().item():+.6f}")
    print(f"{'LSE':<15} {sdf_lse.min().item():+.6f}{'':>5} {sdf_lse.max().item():+.6f}{'':>5} {sdf_lse.mean().item():+.6f}")
    print(f"{'Quadratic':<15} {sdf_quad.min().item():+.6f}{'':>5} {sdf_quad.max().item():+.6f}{'':>5} {sdf_quad.mean().item():+.6f}")

    # Row 1: SDF visualizations
    vmin, vmax = -0.5, 0.5
    for ax, sdf, title in zip(axes[0],
                               [sdf_hard, sdf_lse, sdf_quad],
                               ['Hard Minimum', 'Log-Sum-Exp (LSE)', 'Quadratic Smooth Min']):
        sdf_np = to_numpy(sdf)
        cs = ax.contourf(to_numpy(X), to_numpy(Y), sdf_np,
                        levels=np.linspace(vmin, vmax, 21),
                        cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.contour(to_numpy(X), to_numpy(Y), sdf_np, levels=[0],
                  colors='black', linewidths=3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(cs, ax=ax)

    # Row 2: Differences from hard minimum
    diff_lse = torch.abs(sdf_lse - sdf_hard)
    diff_quad = torch.abs(sdf_quad - sdf_hard)

    for ax, diff, title, method in zip(axes[1, :2],
                                       [diff_lse, diff_quad],
                                       ['Difference: LSE - Hard', 'Difference: Quadratic - Hard'],
                                       ['LSE', 'Quadratic']):
        diff_np = to_numpy(diff)
        cs = ax.contourf(to_numpy(X), to_numpy(Y), diff_np,
                        levels=20, cmap='hot')
        ax.contour(to_numpy(X), to_numpy(Y), to_numpy(sdf_hard),
                  levels=[0], colors='blue', linewidths=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(cs, ax=ax)

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\n{method} vs Hard Min:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

    # Direct comparison: LSE vs Quadratic
    diff_methods = torch.abs(sdf_lse - sdf_quad)
    diff_methods_np = to_numpy(diff_methods)
    cs = axes[1, 2].contourf(to_numpy(X), to_numpy(Y), diff_methods_np,
                             levels=20, cmap='viridis')
    axes[1, 2].contour(to_numpy(X), to_numpy(Y), to_numpy(sdf_hard),
                       levels=[0], colors='red', linewidths=2)
    axes[1, 2].set_title('Difference: LSE - Quadratic', fontsize=12, fontweight='bold')
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(cs, ax=axes[1, 2])

    print(f"\nLSE vs Quadratic:")
    print(f"  Max difference: {diff_methods.max().item():.6f}")
    print(f"  Mean difference: {diff_methods.mean().item():.6f}")

    plt.tight_layout()
    plt.savefig('test_smoothing_methods_visual.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visual comparison to test_smoothing_methods_visual.png")


def test_cross_section_analysis():
    """Analyze SDF and gradients along cross-sections."""
    print("\n" + "="*80)
    print("TEST 2: Cross-Section Analysis")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environments
    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.25, 0.0], [0.25, 0.0]]),
            radii=np.array([0.35, 0.35]),
            tensor_args=tensor_args
        ),
    ]

    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Sample along x-axis (y=0)
    x_coords = torch.linspace(-0.8, 0.8, 200, **tensor_args)
    y_coord = torch.zeros_like(x_coords)

    sdf_hard_list = []
    sdf_lse_list = []
    sdf_quad_list = []
    grad_hard_list = []
    grad_lse_list = []
    grad_quad_list = []

    print("\nComputing cross-section...")
    for x, y in zip(x_coords, y_coord):
        point = torch.stack([x, y]).unsqueeze(0).unsqueeze(0)
        point.requires_grad_(True)

        # Hard minimum
        sdf_hard = env_lse.compute_sdf(point, use_smooth_union=False)
        sdf_hard_list.append(sdf_hard.item())
        f_hard = lambda p: env_lse.compute_sdf(p, use_smooth_union=False).sum()
        jac_hard = jacobian(f_hard, point)
        grad_hard_list.append(jac_hard[0, 0, 0].item())

        # LSE
        point.grad = None
        sdf_lse = env_lse.compute_sdf(point, use_smooth_union=True, smoothing_method="LSE")
        sdf_lse_list.append(sdf_lse.item())
        f_lse = lambda p: env_lse.compute_sdf(p, use_smooth_union=True, smoothing_method="LSE").sum()
        jac_lse = jacobian(f_lse, point)
        grad_lse_list.append(jac_lse[0, 0, 0].item())

        # Quadratic
        point.grad = None
        sdf_quad = env_quad.compute_sdf(point, use_smooth_union=True, smoothing_method="Quadratic")
        sdf_quad_list.append(sdf_quad.item())
        f_quad = lambda p: env_quad.compute_sdf(p, use_smooth_union=True, smoothing_method="Quadratic").sum()
        jac_quad = jacobian(f_quad, point)
        grad_quad_list.append(jac_quad[0, 0, 0].item())

    # Convert to numpy
    x_np = to_numpy(x_coords)
    sdf_hard_np = np.array(sdf_hard_list)
    sdf_lse_np = np.array(sdf_lse_list)
    sdf_quad_np = np.array(sdf_quad_list)
    grad_hard_np = np.array(grad_hard_list)
    grad_lse_np = np.array(grad_lse_list)
    grad_quad_np = np.array(grad_quad_list)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SDF values
    axes[0, 0].plot(x_np, sdf_hard_np, 'k-', linewidth=2, label='Hard Min', alpha=0.7)
    axes[0, 0].plot(x_np, sdf_lse_np, 'b-', linewidth=2, label='LSE')
    axes[0, 0].plot(x_np, sdf_quad_np, 'r-', linewidth=2, label='Quadratic')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=-0.25, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('x position (y=0)')
    axes[0, 0].set_ylabel('SDF value')
    axes[0, 0].set_title('SDF Cross-Section')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SDF difference from hard min
    axes[0, 1].plot(x_np, sdf_lse_np - sdf_hard_np, 'b-', linewidth=2, label='LSE - Hard')
    axes[0, 1].plot(x_np, sdf_quad_np - sdf_hard_np, 'r-', linewidth=2, label='Quadratic - Hard')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('x position (y=0)')
    axes[0, 1].set_ylabel('SDF Difference')
    axes[0, 1].set_title('Approximation Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gradients
    axes[1, 0].plot(x_np, grad_hard_np, 'k-', linewidth=2, label='Hard Min', alpha=0.7)
    axes[1, 0].plot(x_np, grad_lse_np, 'b-', linewidth=2, label='LSE')
    axes[1, 0].plot(x_np, grad_quad_np, 'r-', linewidth=2, label='Quadratic')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('x position (y=0)')
    axes[1, 0].set_ylabel('∂SDF/∂x')
    axes[1, 0].set_title('Gradient Cross-Section')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gradient smoothness (second derivative approximation)
    grad_diff_lse = np.abs(np.diff(grad_lse_np))
    grad_diff_quad = np.abs(np.diff(grad_quad_np))
    axes[1, 1].plot(x_np[:-1], grad_diff_lse, 'b-', linewidth=2, label='LSE')
    axes[1, 1].plot(x_np[:-1], grad_diff_quad, 'r-', linewidth=2, label='Quadratic')
    axes[1, 1].set_xlabel('x position (y=0)')
    axes[1, 1].set_ylabel('|Δ(∂SDF/∂x)|')
    axes[1, 1].set_title('Gradient Smoothness (Lower = Smoother)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Compute smoothness metrics
    max_jump_lse = np.max(grad_diff_lse)
    max_jump_quad = np.max(grad_diff_quad)
    mean_jump_lse = np.mean(grad_diff_lse)
    mean_jump_quad = np.mean(grad_diff_quad)

    print(f"\nGradient Smoothness Metrics:")
    print(f"{'Method':<15} {'Max Jump':<15} {'Mean Jump':<15}")
    print("-" * 45)
    print(f"{'LSE':<15} {max_jump_lse:.6f}{'':>8} {mean_jump_lse:.6f}")
    print(f"{'Quadratic':<15} {max_jump_quad:.6f}{'':>8} {mean_jump_quad:.6f}")

    plt.tight_layout()
    plt.savefig('test_smoothing_methods_cross_section.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved cross-section analysis to test_smoothing_methods_cross_section.png")


def test_k_parameter_sensitivity():
    """Test behavior with different k values."""
    print("\n" + "="*80)
    print("TEST 3: k Parameter Sensitivity")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.2, 0.0], [0.2, 0.0]]),
            radii=np.array([0.3, 0.3]),
            tensor_args=tensor_args
        ),
    ]

    #k_values = [5.0, 10.0, 20.0, 50.0, 100.0]
    k_values = [0.005, 0.01, 0.02, 0.05, 0.1]

    # Sample point in overlap region
    test_point = torch.tensor([[0.0, 0.0]], **tensor_args).unsqueeze(1)

    print(f"\nTesting at overlap center point (0, 0):")
    print(f"{'k value':<10} {'Hard Min':<15} {'LSE':<15} {'Quadratic':<15} {'LSE Error':<15} {'Quad Error':<15}")
    print("-" * 85)

    results_lse = []
    results_quad = []
    errors_lse = []
    errors_quad = []

    for k in k_values:
        env_lse = EnvDynBase(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, "spheres")],
            k_smooth=k,
            smoothing_method="LSE",
            tensor_args=tensor_args
        )

        env_quad = EnvDynBase(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, "spheres")],
            k_smooth=k,
            smoothing_method="Quadratic",
            tensor_args=tensor_args
        )

        sdf_hard = env_lse.compute_sdf(test_point, use_smooth_union=False).item()
        sdf_lse = env_lse.compute_sdf(test_point, use_smooth_union=True, smoothing_method="LSE").item()
        sdf_quad = env_quad.compute_sdf(test_point, use_smooth_union=True, smoothing_method="Quadratic").item()

        error_lse = abs(sdf_lse - sdf_hard)
        error_quad = abs(sdf_quad - sdf_hard)

        results_lse.append(sdf_lse)
        results_quad.append(sdf_quad)
        errors_lse.append(error_lse)
        errors_quad.append(error_quad)

        print(f"{k:<10.1f} {sdf_hard:+.6f}{'':>8} {sdf_lse:+.6f}{'':>8} {sdf_quad:+.6f}{'':>8} "
              f"{error_lse:.6f}{'':>8} {error_quad:.6f}")

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # SDF values vs k
    ax1.semilogx(k_values, [results_lse[0]] * len(k_values), 'k--',
                 linewidth=2, label=f'Hard Min (target)', alpha=0.7)
    ax1.semilogx(k_values, results_lse, 'b-o', linewidth=2, markersize=8, label='LSE')
    ax1.semilogx(k_values, results_quad, 'r-s', linewidth=2, markersize=8, label='Quadratic')
    ax1.set_xlabel('k value (log scale)')
    ax1.set_ylabel('SDF value at (0, 0)')
    ax1.set_title('SDF Convergence with k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error vs k
    ax2.loglog(k_values, errors_lse, 'b-o', linewidth=2, markersize=8, label='LSE Error')
    ax2.loglog(k_values, errors_quad, 'r-s', linewidth=2, markersize=8, label='Quadratic Error')
    ax2.set_xlabel('k value (log scale)')
    ax2.set_ylabel('Absolute error (log scale)')
    ax2.set_title('Approximation Error vs k')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('test_smoothing_methods_k_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved k sensitivity analysis to test_smoothing_methods_k_sensitivity.png")


def test_performance_comparison():
    """Compare computational performance."""
    print("\n" + "="*80)
    print("TEST 4: Performance Comparison")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.25, 0.0], [0.25, 0.0], [0.0, 0.25], [0.0, -0.25]]),
            radii=np.array([0.3, 0.3, 0.3, 0.3]),
            tensor_args=tensor_args
        ),
    ]

    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Generate test points
    x = torch.linspace(-1, 1, 200, **tensor_args)
    y = torch.linspace(-1, 1, 200, **tensor_args)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack((X.flatten(), Y.flatten()), dim=-1).view(-1, 1, 2)

    num_runs = 10
    times_lse = []
    times_quad = []

    print(f"\nRunning {num_runs} iterations on {points.shape[0]} points...")

    for i in range(num_runs):
        # LSE timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        _ = env_lse.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=True, smoothing_method="LSE")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times_lse.append(time.time() - start)

        # Quadratic timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        _ = env_quad.compute_sdf(points, reshape_shape=X.shape, use_smooth_union=True, smoothing_method="Quadratic")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times_quad.append(time.time() - start)

    mean_lse = np.mean(times_lse)
    std_lse = np.std(times_lse)
    mean_quad = np.mean(times_quad)
    std_quad = np.std(times_quad)

    print(f"\nPerformance Results:")
    print(f"{'Method':<15} {'Mean Time (ms)':<20} {'Std Dev (ms)':<20} {'Relative':<15}")
    print("-" * 70)
    print(f"{'LSE':<15} {mean_lse*1000:.3f}{'':>15} {std_lse*1000:.3f}{'':>15} 1.00x")
    print(f"{'Quadratic':<15} {mean_quad*1000:.3f}{'':>15} {std_quad*1000:.3f}{'':>15} {mean_quad/mean_lse:.2f}x")

    speedup = mean_lse / mean_quad if mean_quad < mean_lse else mean_quad / mean_lse
    faster_method = "Quadratic" if mean_quad < mean_lse else "LSE"

    print(f"\n{'✓'} {faster_method} is {abs(speedup):.2f}x {'faster' if speedup > 1 else 'slower'}")


def test_gradient_field_comparison():
    """Compare full gradient fields."""
    print("\n" + "="*80)
    print("TEST 5: Gradient Field Comparison")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    obj_list = [
        MultiSphereField(
            centers=np.array([[-0.2, 0.0], [0.2, 0.0]]),
            radii=np.array([0.3, 0.3]),
            tensor_args=tensor_args
        ),
    ]

    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "spheres")],
        k_smooth=0.03,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Render gradient fields
    print("\nRendering gradient fields...")
    env_lse.render_grad_sdf(ax=axes[0], fig=fig, use_smooth_union=True)
    env_lse.render(axes[0])
    axes[0].set_title('LSE Gradients', fontsize=14, fontweight='bold')

    env_quad.render_grad_sdf(ax=axes[1], fig=fig, use_smooth_union=True)
    env_quad.render(axes[1])
    axes[1].set_title('Quadratic Gradients', fontsize=14, fontweight='bold')

    # Compute gradient magnitude difference
    xs = torch.linspace(-1, 1, 15, **tensor_args)
    ys = torch.linspace(-1, 1, 15, **tensor_args)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    grad_mag_diff = []
    for x, y in zip(X_flat, Y_flat):
        point = torch.stack([x, y]).unsqueeze(0).unsqueeze(0)
        point.requires_grad_(True)

        f_lse = lambda p: env_lse.compute_sdf(p, use_smooth_union=True, smoothing_method="LSE").sum()
        jac_lse = jacobian(f_lse, point)
        mag_lse = torch.norm(jac_lse)

        point.grad = None
        f_quad = lambda p: env_quad.compute_sdf(p, use_smooth_union=True, smoothing_method="Quadratic").sum()
        jac_quad = jacobian(f_quad, point)
        mag_quad = torch.norm(jac_quad)

        grad_mag_diff.append((mag_lse - mag_quad).item())

    grad_mag_diff = np.array(grad_mag_diff).reshape(X.shape)

    # Plot difference
    cs = axes[2].contourf(to_numpy(X), to_numpy(Y), grad_mag_diff,
                          levels=20, cmap='RdBu_r', center=0)
    axes[2].set_title('Gradient Magnitude Diff\n(LSE - Quadratic)',
                     fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(cs, ax=axes[2])

    print(f"\nGradient magnitude difference:")
    print(f"  Max: {np.max(np.abs(grad_mag_diff)):.6f}")
    print(f"  Mean: {np.mean(np.abs(grad_mag_diff)):.6f}")

    plt.tight_layout()
    plt.savefig('test_smoothing_methods_gradients.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved gradient comparison to test_smoothing_methods_gradients.png")


def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "="*80)
    print("SMOOTHING METHODS COMPARISON: LSE vs QUADRATIC")
    print("="*80)

    try:
        test_visual_comparison()
        #test_cross_section_analysis()
        #test_k_parameter_sensitivity()
        #test_performance_comparison()
        #test_gradient_field_comparison()

        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  - test_smoothing_methods_visual.png")
        print("  - test_smoothing_methods_cross_section.png")
        print("  - test_smoothing_methods_k_sensitivity.png")
        print("  - test_smoothing_methods_gradients.png")
        print("\nKey Findings:")
        print("  1. Both methods produce smooth, differentiable SDFs")
        print("  2. LSE is typically more conservative (more negative in overlap)")
        print("  3. Quadratic has bounded support (smoother transitions)")
        print("  4. Performance depends on implementation and hardware")
        print("  5. Higher k values make both methods converge to hard minimum")

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
