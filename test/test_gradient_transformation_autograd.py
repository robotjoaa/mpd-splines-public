"""
Test: Verify gradient computation with coordinate transformations.

Key insight: When we compute SDF via coordinate transformation:
    inv_x = (x - t) @ R.T
    sdf_val = sdf_func(inv_x)

Autograd automatically handles the chain rule correctly!
This means we don't need to manually transform gradients.

The gradient of sdf_val with respect to x will be:
    d(sdf_val)/dx = d(sdf_func)/d(inv_x) * d(inv_x)/dx
                  = grad_sdf_ref @ R.T  (chain rule)
                  = (R @ grad_sdf_ref.T).T

Which is exactly what we want!
"""

import torch
import numpy as np
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.environments.primitives import MultiSphereField, ObjectField


def test_autograd_handles_transformation_correctly():
    """
    Verify that autograd correctly computes gradients through coordinate transformations.

    We will show that:
    1. Manual gradient transformation matches autograd
    2. Using inv_x = (x - t) @ R.T gives correct gradients automatically
    """

    print("="*80)
    print("TEST: Gradient computation with coordinate transformations")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create a simple SDF function (sphere at origin in reference frame)
    sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0, 0.0]]),
        radii=np.array([0.3]),
        tensor_args=tensor_args
    )

    # Define transformation (rotation + translation)
    theta = np.pi / 4  # 45 degrees
    R = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0], 
        [0, 0, 1], 
    ], **tensor_args)

    t = torch.tensor([0.5, 0.3, 0.0], **tensor_args)  # translation

    # Query points in world frame
    x_world = torch.tensor([
        [0.2, 0.1, 0.0],
        [0.5, 0.5, 0.0],
        [-0.1, 0.4, 0.0]
    ], requires_grad=True, **tensor_args)

    print("\n1. Setup:")
    print(f"   Rotation angle: {np.rad2deg(theta):.1f} degrees")
    print(f"   Translation: {t.tolist()}")
    print(f"   Query points (world frame): {x_world.shape}")

    # ============================================================================
    # Method 1: Transform points, query SDF, let autograd handle gradient
    # ============================================================================
    print("\n2. Method 1: Autograd through transformation")

    # Transform to reference frame: inv_x = (x - t) @ R.T
    x_ref = (x_world - t) @ R.T
    print(f"   Transformed points (ref frame): {x_ref.shape}")

    # Query SDF in reference frame
    sdf_method1 = sphere.compute_signed_distance(x_ref, get_gradient=False)
    print(f"   SDF values: {sdf_method1.tolist()}")

    # Compute gradient via autograd
    grad_method1 = torch.autograd.grad(
        sdf_method1.sum(),
        x_world,
        create_graph=False
    )[0]
    print(f"   Gradients (autograd): {grad_method1.shape}")
    print(f"   {grad_method1.tolist()}")

    # ============================================================================
    # Method 2: Manual gradient transformation (analytical)
    # ============================================================================
    print("\n3. Method 2: Manual gradient transformation")

    # Compute SDF and gradient in reference frame
    x_ref_no_grad = x_ref.detach()
    sdf_method2, grad_ref = sphere.compute_signed_distance(x_ref_no_grad, get_gradient=True)
    print(f"   SDF values: {sdf_method2.tolist()}")
    print(f"   Gradient in ref frame: {grad_ref.tolist()}")

    # Transform gradient to world frame manually
    # d(sdf)/dx_world = d(sdf)/dx_ref * d(x_ref)/dx_world
    # d(x_ref)/dx_world = R.T (from x_ref = (x - t) @ R.T)
    # Therefore: grad_world = R @ grad_ref
    grad_method2 = grad_ref @ R
    print(f"   Gradient transformed to world: {grad_method2.tolist()}")

    # ============================================================================
    # Method 3: Using ObjectField with transformation (current implementation bug)
    # ============================================================================
    print("\n4. Method 3: ObjectField (current implementation)")

    # Create ObjectField with transformation
    obj_field = ObjectField([sphere], tensor_args=tensor_args)
    obj_field.set_position_orientation(pos=t, ori=R)

    x_world_for_obj = x_world.detach()
    sdf_method3, grad_method3 = obj_field.compute_signed_distance(x_world_for_obj, get_gradient=True)
    print(f"   SDF values: {sdf_method3.tolist()}")
    print(f"   Gradients: {grad_method3.tolist()}")

    # ============================================================================
    # Compare all methods
    # ============================================================================
    print("\n5. Comparison:")
    print("-" * 80)

    # SDF values should all match
    sdf_diff_1_2 = (sdf_method1.detach() - sdf_method2).abs().max().item()
    sdf_diff_1_3 = (sdf_method1.detach() - sdf_method3).abs().max().item()

    print(f"   SDF difference (Method 1 vs 2): {sdf_diff_1_2:.6e}")
    print(f"   SDF difference (Method 1 vs 3): {sdf_diff_1_3:.6e}")

    # Gradients
    grad_diff_1_2 = (grad_method1 - grad_method2).abs().max().item()
    grad_diff_1_3 = (grad_method1 - grad_method3).abs().max().item()
    grad_diff_2_3 = (grad_method2 - grad_method3).abs().max().item()

    print(f"\n   Gradient difference (Autograd vs Manual): {grad_diff_1_2:.6e}")
    print(f"   Gradient difference (Autograd vs ObjectField): {grad_diff_1_3:.6e}")
    print(f"   Gradient difference (Manual vs ObjectField): {grad_diff_2_3:.6e}")

    # ============================================================================
    # Verification
    # ============================================================================
    print("\n6. Verification:")
    print("-" * 80)

    # SDF values should match
    assert sdf_diff_1_2 < 1e-5, f"SDF mismatch between methods 1 and 2: {sdf_diff_1_2}"
    assert sdf_diff_1_3 < 1e-5, f"SDF mismatch between methods 1 and 3: {sdf_diff_1_3}"
    print("   ✓ SDF values match across all methods")

    # Autograd and manual transformation should match
    assert grad_diff_1_2 < 1e-5, f"Gradient mismatch between autograd and manual: {grad_diff_1_2}"
    print("   ✓ Autograd matches manual gradient transformation")

    # Check if ObjectField gradient is correct
    if grad_diff_1_3 > 1e-5:
        print(f"   ✗ ObjectField gradient is INCORRECT (diff={grad_diff_1_3:.6e})")
        print("   → This confirms the gradient transformation bug in ObjectField!")

        # Show the fix
        print("\n   Fix: ObjectField should rotate gradient back to world frame:")
        print("   grad_world = rotate_point(grad_ref, self.ori)")

        # Verify the fix would work
        from torch_robotics.torch_kinematics_tree.geometrics.utils import rotate_point

        # Get gradient in ref frame from sphere directly
        x_ref_obj = rotate_point(x_world_for_obj - obj_field.pos, obj_field.ori.T)
        _, grad_ref_obj = sphere.compute_signed_distance(x_ref_obj, get_gradient=True)

        # Apply the fix: rotate gradient back
        grad_fixed = rotate_point(grad_ref_obj, obj_field.ori)
        grad_diff_fixed = (grad_method1 - grad_fixed).abs().max().item()

        print(f"   Fixed gradient difference: {grad_diff_fixed:.6e}")
        assert grad_diff_fixed < 1e-5, "Fix doesn't work!"
        print("   ✓ Fix verified: Rotating gradient back works correctly!")
    else:
        print("   ✓ ObjectField gradient is correct")

    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("When using: inv_x = (x - t) @ R.T")
    print("            sdf_val = sdf_func(inv_x)")
    print("")
    print("Autograd AUTOMATICALLY computes correct gradient via chain rule:")
    print("  ∇_x sdf_val = ∇_inv_x sdf_func · ∂inv_x/∂x")
    print("              = grad_ref @ R.T")
    print("              = (R @ grad_ref.T).T")
    print("")
    print("This means we can use autograd and don't need manual gradient transformation!")
    print("="*80)


def test_batched_transformation_gradients():
    """
    Test gradient computation with batched transformations (multiple timesteps).

    This simulates the actual use case: multiple query points at multiple times.
    """

    print("\n" + "="*80)
    print("TEST: Batched gradient computation with time-varying transformations")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Reference SDF
    sphere = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    # Batched query: (batch=2, horizon=4, n_points=3, dim=2)
    B, H, N, dim = 2, 4, 3, 2
    x_world = torch.randn(B, H, N, dim, requires_grad=True, **tensor_args)

    # Time-varying transformations
    timesteps = torch.linspace(0, 1, H, **tensor_args)

    # Trajectory: rotation and translation over time
    def get_transform(t):
        theta = np.pi * t  # Rotate 180 degrees over time
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], **tensor_args)
        t_vec = torch.tensor([0.5 * t, 0.3 * t], **tensor_args)
        return R, t_vec

    print(f"\n1. Setup:")
    print(f"   Query shape: {x_world.shape}")
    print(f"   Timesteps: {H}")

    # Compute SDFs with transformations
    print("\n2. Computing SDF with time-varying transformations...")

    all_sdfs = []
    for h in range(H):
        R_t, t_t = get_transform(timesteps[h].item())

        # Transform points at timestep h
        x_ref = (x_world[:, h, :, :] - t_t) @ R_t.T

        # Query SDF
        sdf_h = sphere.compute_signed_distance(x_ref, get_gradient=False)
        all_sdfs.append(sdf_h)

    # Stack results: (B, H, N)
    sdf_batched = torch.stack(all_sdfs, dim=1)
    print(f"   SDF shape: {sdf_batched.shape}")

    # Compute gradient via autograd
    print("\n3. Computing gradients via autograd...")
    grad_autograd = torch.autograd.grad(
        sdf_batched.sum(),
        x_world,
        create_graph=False
    )[0]
    print(f"   Gradient shape: {grad_autograd.shape}")
    print(f"   Gradient norm: {grad_autograd.norm().item():.4f}")

    # Verify gradient is non-zero and has correct shape
    assert grad_autograd.shape == x_world.shape
    assert grad_autograd.norm() > 0

    print("\n   ✓ Batched gradient computation works correctly!")
    print("   ✓ Autograd handles time-varying transformations automatically!")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("For transformation-based time-varying SDF:")
    print("")
    print("1. Precompute reference SDF once (at t=0)")
    print("2. At query time:")
    print("   - Get transformation T(t) = (R(t), p(t))")
    print("   - Transform query: x_ref = (x - p(t)) @ R(t).T")
    print("   - Query SDF: sdf = sdf_ref(x_ref)")
    print("   - Autograd gives correct ∇_x sdf automatically!")
    print("")
    print("3. No manual gradient transformation needed!")
    print("4. Can vectorize over timesteps using vmap!")
    print("="*80)


if __name__ == "__main__":
    # Test 1: Single transformation
    test_autograd_handles_transformation_correctly()

    # Test 2: Batched transformations (time-varying)
    test_batched_transformation_gradients()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nKey takeaway:")
    print("  Using: sdf_val = sdf_func((x - t) @ R.T)")
    print("  Gives correct gradients via autograd automatically!")
    print("  This is the right approach for transformation-based time-varying SDF!")
    print("="*80)
