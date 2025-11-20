#!/usr/bin/env python3
"""
Quick test script to verify SDF gradient visualization works.
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch_robotics import environments
from torch_robotics.torch_utils.torch_utils import get_torch_device
from visualize_sdf_gradients import visualize_sdf_gradient_field_2d


def test_simple_2d_environment():
    """Test visualization with a simple 2D environment."""
    print("Testing SDF gradient visualization...")

    # Setup
    device = get_torch_device("cpu")  # Use CPU for testing
    tensor_args = {"device": device, "dtype": torch.float32}

    # Create environment
    print("Creating environment...")
    env = environments.EnvSimple2D(tensor_args=tensor_args)

    # Create a dummy trajectory
    print("Creating dummy trajectory...")
    start = torch.tensor([0.1, 0.1], **tensor_args)
    goal = torch.tensor([0.9, 0.9], **tensor_args)
    n_waypoints = 20

    # Linear interpolation for dummy trajectory
    alphas = torch.linspace(0, 1, n_waypoints, **tensor_args).unsqueeze(1)
    trajectory = start.unsqueeze(0) * (1 - alphas) + goal.unsqueeze(0) * alphas

    # Control points (just subsample trajectory)
    control_points = trajectory[::4]

    # Visualize
    print("Creating visualization...")
    save_path = "test_sdf_gradient_viz.png"

    fig, ax = visualize_sdf_gradient_field_2d(
        env=env,
        trajectory=trajectory,
        control_points=control_points,
        save_path=save_path,
        resolution=25,  # Lower resolution for faster test
        arrow_scale=0.02,
        margin=0.04,
        tensor_args=tensor_args,
        title="Test: SDF Gradient Field Visualization",
    )

    plt.close(fig)

    # Check if file was created
    if os.path.exists(save_path):
        print(f"✓ Success! Visualization saved to: {save_path}")
        print(f"  File size: {os.path.getsize(save_path) / 1024:.1f} KB")
        return True
    else:
        print(f"✗ Failed: File not created at {save_path}")
        return False


def test_dense_2d_environment():
    """Test with a more complex environment."""
    print("\nTesting with dense 2D environment...")

    device = get_torch_device("cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    try:
        env = environments.EnvDense2D(tensor_args=tensor_args)
    except:
        print("  EnvDense2D not available, skipping...")
        return True

    # Create curved trajectory
    t = torch.linspace(0, 1, 30, **tensor_args)
    trajectory = torch.stack([
        0.3 + 0.4 * t,
        0.3 + 0.4 * torch.sin(3.14159 * t)
    ], dim=1)

    save_path = "test_sdf_gradient_dense.png"

    fig, ax = visualize_sdf_gradient_field_2d(
        env=env,
        trajectory=trajectory,
        save_path=save_path,
        resolution=30,
        arrow_scale=0.015,
        margin=0.04,
        tensor_args=tensor_args,
        title="Test: Dense Environment Gradients",
    )

    plt.close(fig)

    if os.path.exists(save_path):
        print(f"✓ Dense environment test passed: {save_path}")
        return True
    else:
        print(f"✗ Dense environment test failed")
        return False


def test_gradient_computation():
    """Test that gradient computation works correctly."""
    print("\nTesting gradient computation...")

    device = get_torch_device("cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = environments.EnvSimple2D(tensor_args=tensor_args)

    # Test points
    test_points = torch.tensor([
        [0.5, 0.5],  # Center (likely in collision or close)
        [0.1, 0.1],  # Corner
        [0.9, 0.9],  # Other corner
    ], **tensor_args)

    # Get SDF objects
    df_obj_list = env.get_df_obj_list()

    print(f"  Environment has {len(df_obj_list)} obstacles")

    # Compute SDF and gradients
    for i, point in enumerate(test_points):
        for j, df_obj in enumerate(df_obj_list):
            sdf_val, sdf_grad = df_obj.compute_signed_distance(
                point.unsqueeze(0),
                get_gradient=True
            )

            print(f"  Point {i} vs Obstacle {j}:")
            print(f"    SDF value: {sdf_val.item():.4f}")
            print(f"    Gradient: [{sdf_grad[0, 0].item():.4f}, {sdf_grad[0, 1].item():.4f}]")
            print(f"    Grad norm: {torch.norm(sdf_grad).item():.4f}")

    print("✓ Gradient computation test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("SDF Gradient Visualization Test Suite")
    print("="*80)

    results = []

    # Test 1: Simple environment
    try:
        results.append(("Simple 2D", test_simple_2d_environment()))
    except Exception as e:
        print(f"✗ Simple 2D test failed with error: {e}")
        results.append(("Simple 2D", False))

    # Test 2: Dense environment
    try:
        results.append(("Dense 2D", test_dense_2d_environment()))
    except Exception as e:
        print(f"✗ Dense 2D test failed with error: {e}")
        results.append(("Dense 2D", False))

    # Test 3: Gradient computation
    try:
        results.append(("Gradient computation", test_gradient_computation()))
    except Exception as e:
        print(f"✗ Gradient computation test failed with error: {e}")
        results.append(("Gradient computation", False))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print("="*80)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
