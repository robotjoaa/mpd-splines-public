"""
Diagnostic test to understand the SDF scale issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_robotics.environments.dynamic_extension.sdf_utils import smooth_union_sdf
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy


def test_smooth_union_behavior():
    """Detailed analysis of smooth_union_sdf behavior."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Smooth Union SDF Behavior Analysis")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Test at specific points
    test_cases = [
        ("Far inside sphere 1", torch.tensor([[0.0]], **tensor_args), -0.5, 0.5),
        ("On boundary of sphere 1", torch.tensor([[0.0]], **tensor_args), 0.0, 0.5),
        ("Between spheres (overlap)", torch.tensor([[0.0]], **tensor_args), -0.1, -0.1),
        ("Far outside both", torch.tensor([[0.0]], **tensor_args), 1.0, 1.0),
    ]

    print("\nPoint-by-point analysis:")
    print(f"{'Case':<30} {'SDF1':<10} {'SDF2':<10} {'Hard Min':<12} {'Smooth k=10':<15} {'Smooth k=50':<15}")
    print("-" * 95)

    for case_name, _, sdf1_val, sdf2_val in test_cases:
        sdf1 = torch.tensor([sdf1_val], **tensor_args)
        sdf2 = torch.tensor([sdf2_val], **tensor_args)

        hard_min = torch.minimum(sdf1, sdf2)
        smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0)
        smooth_k50 = smooth_union_sdf(sdf1, sdf2, k=50.0)

        print(f"{case_name:<30} {sdf1_val:+.4f}{'':>5} {sdf2_val:+.4f}{'':>5} "
              f"{hard_min.item():+.4f}{'':>7} {smooth_k10.item():+.4f}{'':>10} "
              f"{smooth_k50.item():+.4f}{'':>10}")

    # Test along a line
    print("\n\nAnalysis along a line (y=0) between two spheres:")
    print("Sphere 1 at (-0.3, 0), radius=0.5")
    print("Sphere 2 at (+0.3, 0), radius=0.5")
    print()

    x_coords = torch.linspace(-1.0, 1.0, 21, **tensor_args)
    print(f"{'x':<8} {'SDF1':<10} {'SDF2':<10} {'Hard Min':<12} {'Smooth k=10':<15} {'Difference':<12}")
    print("-" * 75)

    for x in x_coords:
        # SDF for sphere at (-0.3, 0) with radius 0.5
        sdf1 = torch.abs(x + 0.3) - 0.5
        # SDF for sphere at (+0.3, 0) with radius 0.5
        sdf2 = torch.abs(x - 0.3) - 0.5

        hard_min = torch.minimum(sdf1, sdf2)
        smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0)
        diff = smooth_k10 - hard_min

        print(f"{x.item():+.2f}{'':>4} {sdf1.item():+.4f}{'':>5} {sdf2.item():+.4f}{'':>5} "
              f"{hard_min.item():+.4f}{'':>7} {smooth_k10.item():+.4f}{'':>10} "
              f"{diff.item():+.4f}{'':>7}")

    # Mathematical properties check
    print("\n\n" + "="*80)
    print("Mathematical Properties Verification")
    print("="*80)

    # Property 1: Should approximate minimum
    sdf1 = torch.tensor([-0.2], **tensor_args)
    sdf2 = torch.tensor([-0.1], **tensor_args)
    hard_min = torch.minimum(sdf1, sdf2)
    smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0)
    smooth_k100 = smooth_union_sdf(sdf1, sdf2, k=100.0)

    print(f"\nProperty 1: As k increases, smooth_union should approach hard minimum")
    print(f"  SDF1 = {sdf1.item():.4f}, SDF2 = {sdf2.item():.4f}")
    print(f"  Hard min = {hard_min.item():.4f}")
    print(f"  Smooth k=10 = {smooth_k10.item():.4f} (error: {abs(smooth_k10.item() - hard_min.item()):.4f})")
    print(f"  Smooth k=100 = {smooth_k100.item():.4f} (error: {abs(smooth_k100.item() - hard_min.item()):.4f})")
    print(f"  ✓ Error decreases with higher k" if abs(smooth_k100.item() - hard_min.item()) < abs(smooth_k10.item() - hard_min.item()) else "  ✗ ERROR")

    # Property 2: Should be <= hard minimum (more conservative)
    print(f"\nProperty 2: smooth_union should be ≤ hard minimum (for union)")
    is_conservative = smooth_k10.item() <= hard_min.item() + 1e-6
    print(f"  {smooth_k10.item():.4f} <= {hard_min.item():.4f}: {'✓ YES' if is_conservative else '✗ NO'}")

    # Property 3: Check the mathematical formula
    print(f"\nProperty 3: Verify formula -k * logsumexp(-sdf/k)")
    k = 10.0
    manual_calc = -k * torch.logsumexp(torch.stack([-sdf1 / k, -sdf2 / k]), dim=0)
    auto_calc = smooth_union_sdf(sdf1, sdf2, k=k)
    print(f"  Manual calculation: {manual_calc.item():.6f}")
    print(f"  Function result:    {auto_calc.item():.6f}")
    print(f"  Match: {'✓ YES' if torch.allclose(manual_calc, auto_calc) else '✗ NO'}")

    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Create 2D grid
    x = torch.linspace(-1, 1, 400, **tensor_args)
    y = torch.linspace(-1, 1, 400, **tensor_args)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # Two overlapping spheres
    sdf1 = torch.norm(points - torch.tensor([[-0.3, 0.0]], **tensor_args), dim=-1) - 0.5
    sdf2 = torch.norm(points - torch.tensor([[0.3, 0.0]], **tensor_args), dim=-1) - 0.5

    # Different unions
    sdf_hard = torch.minimum(sdf1, sdf2)
    sdf_smooth_k10 = smooth_union_sdf(sdf1, sdf2, k=10.0)
    sdf_smooth_k50 = smooth_union_sdf(sdf1, sdf2, k=50.0)

    # Plot SDF values
    for ax, sdf, title in zip(axes[0],
                               [sdf_hard, sdf_smooth_k10],
                               ['Hard Minimum', 'Smooth Union k=10']):
        sdf_grid = to_numpy(sdf.reshape(400, 400))
        vmin, vmax = -0.6, 0.6
        cs = ax.contourf(to_numpy(X), to_numpy(Y), sdf_grid, levels=np.linspace(vmin, vmax, 21), cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.contour(to_numpy(X), to_numpy(Y), sdf_grid, levels=[0], colors='black', linewidths=3)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Print statistics
        print(f"\n{title} statistics:")
        print(f"  Min SDF: {sdf.min().item():.4f}")
        print(f"  Max SDF: {sdf.max().item():.4f}")
        print(f"  Mean SDF: {sdf.mean().item():.4f}")

    # Plot difference
    diff = sdf_smooth_k10 - sdf_hard
    diff_grid = to_numpy(diff.reshape(400, 400))
    cs = axes[1, 0].contourf(to_numpy(X), to_numpy(Y), diff_grid, levels=20, cmap='hot')
    axes[1, 0].contour(to_numpy(X), to_numpy(Y), to_numpy(sdf_hard.reshape(400, 400)), levels=[0], colors='blue', linewidths=2)
    axes[1, 0].set_title('Difference (Smooth k=10 - Hard)')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    print(f"\nDifference statistics:")
    print(f"  Min difference: {diff.min().item():.4f}")
    print(f"  Max difference: {diff.max().item():.4f}")
    print(f"  Mean difference: {diff.mean().item():.4f}")

    # Plot 1D slice
    y_slice_idx = 200  # Middle
    x_slice = to_numpy(x)
    hard_slice = to_numpy(sdf_hard.reshape(400, 400)[y_slice_idx, :])
    smooth_slice = to_numpy(sdf_smooth_k10.reshape(400, 400)[y_slice_idx, :])

    axes[1, 1].plot(x_slice, hard_slice, 'b-', linewidth=2, label='Hard Minimum')
    axes[1, 1].plot(x_slice, smooth_slice, 'r-', linewidth=2, label='Smooth Union k=10')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=-0.3, color='gray', linestyle=':', alpha=0.5, label='Sphere centers')
    axes[1, 1].axvline(x=0.3, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('x position')
    axes[1, 1].set_ylabel('SDF value')
    axes[1, 1].set_title('Cross-section at y=0')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_sdf_scale_diagnosis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved diagnostic visualization to test_sdf_scale_diagnosis.png")


if __name__ == "__main__":
    test_smooth_union_behavior()
