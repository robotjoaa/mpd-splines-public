"""
Utilities for smooth and differentiable SDF operations.

This module provides functions for:
- Smooth union/intersection of SDFs using log-sum-exp
- Overlap detection between primitives
- Differentiable SDF operations for moving obstacles
"""

import torch
import numpy as np
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField, MultiRoundedBoxField

def smooth_union_sdf(sdf1, sdf2, k=10.0):
    """
    Smooth union of two SDFs using log-sum-exp approximation.

    The smooth union approximates min(sdf1, sdf2) while maintaining differentiability.
    Formula: φ_union = -k * logsumexp(-sdf/k)

    As k → ∞, this approaches min(sdf1, sdf2).
    Typical values: k = 5-20 for smooth blending, k = 50-100 for sharper transitions.

    Args:
        sdf1: First SDF tensor of shape (...)
        sdf2: Second SDF tensor of shape (...)
        k: Smoothness parameter (higher = sharper, lower = smoother)

    Returns:
        Smooth union SDF of shape (...)
    """
    # Stack SDFs and use torch.logsumexp for numerical stability
    # φ_union = -k * logsumexp(-sdfs/k)
    sdfs_stacked = torch.stack([-sdf1 / k, -sdf2 / k], dim=-1)
    smooth_union = -k * torch.logsumexp(sdfs_stacked, dim=-1)

    return smooth_union




def check_sphere_sphere_overlap(center1, radius1, center2, radius2, margin=0.0):
    """
    Check if two spheres overlap with an optional margin.

    Args:
        center1: Center of first sphere, shape (..., dim)
        radius1: Radius of first sphere, scalar or shape (...)
        center2: Center of second sphere, shape (..., dim)
        radius2: Radius of second sphere, scalar or shape (...)
        margin: Additional margin for conservative overlap detection

    Returns:
        Boolean tensor indicating overlap, shape (...)
    """
    distance = torch.linalg.norm(center1 - center2, dim=-1)
    overlap = distance <= (radius1 + radius2 + margin)
    return overlap


def check_sphere_box_overlap(sphere_center, sphere_radius, box_center, box_half_sizes, margin=0.0):
    """
    Check if a sphere and an axis-aligned box overlap.

    Uses the closest point on the box to the sphere center.

    Args:
        sphere_center: Center of sphere, shape (..., dim)
        sphere_radius: Radius of sphere, scalar or shape (...)
        box_center: Center of box, shape (..., dim)
        box_half_sizes: Half sizes of box (half width, half height, ...), shape (..., dim)
        margin: Additional margin for conservative overlap detection

    Returns:
        Boolean tensor indicating overlap, shape (...)
    """
    # Find closest point on box to sphere center
    relative_pos = sphere_center - box_center
    closest_point = torch.clamp(relative_pos, -box_half_sizes, box_half_sizes)

    # Distance from sphere center to closest point on box
    distance = torch.linalg.norm(relative_pos - closest_point, dim=-1)

    overlap = distance <= (sphere_radius + margin)
    return overlap


def check_box_box_overlap(center1, half_sizes1, center2, half_sizes2, margin=0.0):
    """
    Check if two axis-aligned boxes overlap (using separating axis theorem).

    Args:
        center1: Center of first box, shape (..., dim)
        half_sizes1: Half sizes of first box, shape (..., dim)
        center2: Center of second box, shape (..., dim)
        half_sizes2: Half sizes of second box, shape (..., dim)
        margin: Additional margin for conservative overlap detection

    Returns:
        Boolean tensor indicating overlap, shape (...)
    """
    # Distance between centers along each axis
    distance = torch.abs(center1 - center2)

    # Sum of half sizes along each axis (with margin)
    sum_half_sizes = half_sizes1 + half_sizes2 + margin

    # Overlap if distance is less than sum of half sizes along ALL axes
    overlap = torch.all(distance <= sum_half_sizes, dim=-1)
    return overlap


def detect_primitive_overlaps(primitives, margin=0.0):
    """
    Detect all pairwise overlaps between primitives.

    Args:
        primitives: List of primitive objects (MultiSphereField, MultiBoxField, etc.)
        margin: Additional margin for conservative overlap detection

    Returns:
        List of tuples (idx1, idx2) indicating overlapping primitive pairs
        Dict mapping (idx1, idx2) to overlap details
    """
    

    overlaps = []
    overlap_details = {}

    for i, prim1 in enumerate(primitives):
        for j, prim2 in enumerate(primitives):
            if i >= j:  # Skip self-comparison and duplicates
                continue

            # Get all single primitives from multi-primitive fields
            single_prims1 = prim1.get_all_single_primitives()
            single_prims2 = prim2.get_all_single_primitives()

            has_overlap = False

            for sp1 in single_prims1:
                for sp2 in single_prims2:
                    # Check based on primitive types
                    if isinstance(sp1, MultiSphereField) and isinstance(sp2, MultiSphereField):
                        overlap = check_sphere_sphere_overlap(
                            sp1.centers[0], sp1.radii[0],
                            sp2.centers[0], sp2.radii[0],
                            margin=margin
                        )
                    elif isinstance(sp1, MultiSphereField) and isinstance(sp2, (MultiBoxField, MultiRoundedBoxField)):
                        overlap = check_sphere_box_overlap(
                            sp1.centers[0], sp1.radii[0],
                            sp2.centers[0], sp2.half_sizes[0],
                            margin=margin
                        )
                    elif isinstance(sp1, (MultiBoxField, MultiRoundedBoxField)) and isinstance(sp2, MultiSphereField):
                        overlap = check_sphere_box_overlap(
                            sp2.centers[0], sp2.radii[0],
                            sp1.centers[0], sp1.half_sizes[0],
                            margin=margin
                        )
                    elif isinstance(sp1, (MultiBoxField, MultiRoundedBoxField)) and isinstance(sp2, (MultiBoxField, MultiRoundedBoxField)):
                        overlap = check_box_box_overlap(
                            sp1.centers[0], sp1.half_sizes[0],
                            sp2.centers[0], sp2.half_sizes[0],
                            margin=margin
                        )
                    else:
                        # Unknown primitive type, skip
                        continue

                    if overlap:
                        has_overlap = True
                        break

                if has_overlap:
                    break

            if has_overlap:
                overlaps.append((i, j))
                overlap_details[(i, j)] = {
                    'prim1_type': type(prim1).__name__,
                    'prim2_type': type(prim2).__name__,
                }

    return overlaps, overlap_details


def compute_smooth_sdf_with_overlap_handling(primitives, x, k_smooth=10.0, overlap_margin=0.1,
                                              tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Compute smooth SDF handling potential overlaps between primitives.

    This function:
    1. Detects overlapping primitives
    2. Groups overlapping primitives
    3. Applies smooth union within each group
    4. Takes hard minimum across groups (since they don't overlap)

    Args:
        primitives: List of primitive objects
        x: Query points, shape (..., dim)
        k_smooth: Smoothness parameter for log-sum-exp union
        overlap_margin: Margin for overlap detection
        tensor_args: Tensor arguments for device/dtype

    Returns:
        SDF values at query points, shape (...)
    """
    if len(primitives) == 0:
        return torch.ones(x.shape[:-1], **tensor_args) * float('inf')

    # Detect overlaps
    overlaps, _ = detect_primitive_overlaps(primitives, margin=overlap_margin)

    # Build connected components (groups of overlapping primitives)
    groups = []
    assigned = set()

    for i in range(len(primitives)):
        if i in assigned:
            continue

        # Start a new group
        group = {i}
        to_check = {i}

        while to_check:
            current = to_check.pop()
            # Find all primitives that overlap with current
            for idx1, idx2 in overlaps:
                if idx1 == current and idx2 not in group:
                    group.add(idx2)
                    to_check.add(idx2)
                elif idx2 == current and idx1 not in group:
                    group.add(idx1)
                    to_check.add(idx1)

        groups.append(list(group))
        assigned.update(group)

    # Compute SDF for each group
    group_sdfs = []

    for group in groups:
        if len(group) == 1:
            # Single primitive, no smoothing needed
            prim_idx = group[0]
            sdf = primitives[prim_idx].compute_signed_distance(x)
            group_sdfs.append(sdf)
        else:
            # Multiple overlapping primitives, use smooth union
            prim_sdfs = []
            for prim_idx in group:
                sdf = primitives[prim_idx].compute_signed_distance(x)
                prim_sdfs.append(sdf)

            # Apply smooth union using torch.logsumexp
            # φ_union = -k * logsumexp(-sdfs/k)
            sdfs_stacked = torch.stack(prim_sdfs, dim=-1)
            smooth_sdf = -k_smooth * torch.logsumexp(-sdfs_stacked / k_smooth, dim=-1)
            group_sdfs.append(smooth_sdf)

    # Take minimum across groups (they don't overlap, so hard min is fine)
    if len(group_sdfs) == 1:
        return group_sdfs[0]
    else:
        # Use torch.minimum recursively for multiple groups
        result = group_sdfs[0]
        for sdf in group_sdfs[1:]:
            result = torch.minimum(result, sdf)
        return result


if __name__ == "__main__":
    # Test smooth union
    tensor_args = DEFAULT_TENSOR_ARGS

    # Create test SDFs (two overlapping spheres)
    x = torch.linspace(-2, 2, 100, **tensor_args)
    y = torch.linspace(-2, 2, 100, **tensor_args)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # SDF for sphere at (-0.5, 0) with radius 0.5
    sdf1 = torch.norm(points - torch.tensor([[-0.5, 0.0]], **tensor_args), dim=-1) - 0.5

    # SDF for sphere at (0.5, 0) with radius 0.5
    sdf2 = torch.norm(points - torch.tensor([[0.5, 0.0]], **tensor_args), dim=-1) - 0.5

    # Hard minimum (non-differentiable at overlap boundary)
    sdf_hard = torch.minimum(sdf1, sdf2)

    # Smooth union (differentiable everywhere)
    sdf_smooth_k5 = smooth_union_sdf(sdf1, sdf2, k=5.0)
    sdf_smooth_k20 = smooth_union_sdf(sdf1, sdf2, k=20.0)
    sdf_smooth_k100 = smooth_union_sdf(sdf1, sdf2, k=100.0)

    # Visualize
    import matplotlib.pyplot as plt
    from torch_robotics.torch_utils.torch_utils import to_numpy

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, sdf, title in zip(axes,
                               [sdf_hard, sdf_smooth_k5, sdf_smooth_k20, sdf_smooth_k100],
                               ['Hard Min', 'Smooth k=5', 'Smooth k=20', 'Smooth k=100']):
        sdf_grid = to_numpy(sdf.reshape(100, 100))
        cs = ax.contourf(to_numpy(X), to_numpy(Y), sdf_grid, levels=20, cmap='RdBu')
        ax.contour(to_numpy(X), to_numpy(Y), sdf_grid, levels=[0], colors='black', linewidths=2)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax)

    plt.tight_layout()
    plt.savefig('smooth_sdf_comparison.png', dpi=150)
    print("Saved comparison plot to /tmp/smooth_sdf_comparison.png")

    # Test overlap detection
    print("\n" + "="*60)
    print("Testing overlap detection...")

    # Test sphere-sphere overlap
    overlap = check_sphere_sphere_overlap(
        torch.tensor([0.0, 0.0]), 0.5,
        torch.tensor([0.8, 0.0]), 0.5,
        margin=0.0
    )
    print(f"Spheres at (0,0) r=0.5 and (0.8,0) r=0.5 overlap: {overlap.item()}")

    overlap = check_sphere_sphere_overlap(
        torch.tensor([0.0, 0.0]), 0.5,
        torch.tensor([1.2, 0.0]), 0.5,
        margin=0.0
    )
    print(f"Spheres at (0,0) r=0.5 and (1.2,0) r=0.5 overlap: {overlap.item()}")

    # Test sphere-box overlap
    overlap = check_sphere_box_overlap(
        torch.tensor([0.0, 0.0]), 0.5,
        torch.tensor([0.6, 0.0]), torch.tensor([0.3, 0.3]),
        margin=0.0
    )
    print(f"Sphere at (0,0) r=0.5 and box at (0.6,0) size=0.6x0.6 overlap: {overlap.item()}")
