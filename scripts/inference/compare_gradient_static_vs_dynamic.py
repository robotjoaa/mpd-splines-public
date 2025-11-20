#!/usr/bin/env python3
"""
Compare collision cost gradients between static and dynamic environments.

This script loads a trajectory and compares the SDF gradients (which drive the collision cost gradients)
for both static (EnvSimple2DExtraObjectsV00) and dynamic (EnvDynSimple2DExtraObjects)
environments at different timesteps.
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics import environments


def load_trajectory_and_control_points(result_path):
    """Load trajectory and control points from result file."""
    print(f"Loading result from: {result_path}")
    result = torch.load(result_path, weights_only=False)

    # Extract best trajectory
    if 'q_trajs_pos_best' in result and result.q_trajs_pos_best is not None:
        trajectory = result.q_trajs_pos_best
        print(f"  Loaded best trajectory: {trajectory.shape}")
    elif 'q_trajs_pos_valid' in result and result.q_trajs_pos_valid is not None:
        trajectory = result.q_trajs_pos_valid[0]
        print(f"  Loaded valid trajectory (first): {trajectory.shape}")
    else:
        raise ValueError("No trajectory found in result!")

    # Extract control points
    control_points = None
    if 'q_control_points_best' in result and result.q_control_points_best is not None:
        control_points = result.q_control_points_best
        print(f"  Loaded control points: {control_points.shape}")
    else:
        # Use trajectory as control points if not available
        control_points = trajectory
        print(f"  No control points found, using trajectory as control points")

    return trajectory, control_points, result


def check_if_dynamic(env):
    """
    Check if environment is dynamic (has time-varying obstacles).

    Args:
        env: Environment object

    Returns:
        bool: True if environment is dynamic
    """
    # Check for EnvDynBase wrapper - has _has_moving_objects method
    if hasattr(env, '_has_moving_objects'):
        return env._has_moving_objects()

    # Check for is_static attribute
    if hasattr(env, 'is_static'):
        return not env.is_static

    # Check if env is EnvDynBase by checking for time_range attribute
    if hasattr(env, 'time_range'):
        return True

    # Default: assume static
    return False


def compute_sdf_gradients_for_environment(env, trajectory, timesteps, tensor_args, time_range=(0.0, 1.0)):
    """
    Compute SDF gradients for a given environment at multiple timesteps.

    For dynamic environments, the SDF varies with time. For static environments,
    the SDF is constant.

    Args:
        env: Environment object (static or dynamic)
        trajectory: Trajectory tensor (H, D) where H is horizon, D is state dim
        timesteps: List of timestep indices to compute gradients for
        tensor_args: PyTorch device/dtype
        time_range: Time range for dynamic environments

    Returns:
        gradients_dict: Dictionary mapping timestep to SDF values and gradients
    """
    gradients_dict = {}
    H = trajectory.shape[0]
    is_dynamic = check_if_dynamic(env)

    for t in timesteps:
        # Get position at this timestep
        position = trajectory[t]  # Shape: (D,)

        # Compute time for dynamic environment
        alpha = t / (H - 1) if H > 1 else 0.0
        time = time_range[0] + alpha * (time_range[1] - time_range[0])

        # Compute SDF and gradient
        with torch.enable_grad():
            pos_grad = position.clone().detach().requires_grad_(True)

            # Compute SDF at this position
            if is_dynamic:
                # Dynamic environment - use time parameter
                sdf_val = env.compute_sdf(pos_grad.unsqueeze(0), time=time)
            else:
                # Static environment - no time parameter
                sdf_val = env.compute_sdf(pos_grad.unsqueeze(0))

            # Compute gradient via autograd
            sdf_grad = torch.autograd.grad(
                outputs=sdf_val,
                inputs=pos_grad,
                create_graph=False
            )[0]

        gradients_dict[t] = {
            'sdf': to_numpy(sdf_val.squeeze()),
            'gradient': to_numpy(sdf_grad),
            'position': to_numpy(position),
            'time': time,
            'timestep': t,
            'is_dynamic': is_dynamic
        }

    return gradients_dict


def visualize_gradient_comparison(
    trajectory,
    gradients_static,
    gradients_dynamic,
    timesteps,
    save_dir,
    env_static=None,
    env_dynamic=None,
    time_range=(0.0, 1.0)
):
    """
    Visualize the comparison of SDF gradients between static and dynamic environments.

    Args:
        trajectory: Trajectory numpy array (H, D)
        gradients_static: Dictionary of SDF gradients for static environment
        gradients_dynamic: Dictionary of SDF gradients for dynamic environment
        timesteps: List of timesteps to visualize
        save_dir: Directory to save figures
        env_static: Static environment (optional, for rendering obstacles)
        env_dynamic: Dynamic environment (optional, for rendering obstacles)
        time_range: Time range for dynamic obstacles
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert trajectory to numpy if needed
    trajectory = to_numpy(trajectory) if torch.is_tensor(trajectory) else trajectory
    H = len(trajectory)

    for t in timesteps:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        time = gradients_dynamic[t]['time']

        grad_static = gradients_static[t]['gradient']
        grad_dynamic = gradients_dynamic[t]['gradient']
        position = gradients_static[t]['position']
        sdf_static = gradients_static[t]['sdf']
        sdf_dynamic = gradients_dynamic[t]['sdf']

        # Gradient difference
        grad_diff = grad_dynamic - grad_static
        sdf_diff = sdf_dynamic - sdf_static

        # Gradient scaling for visualization
        grad_scale = 0.1

        # Plot 1: Static environment SDF gradient
        ax = axes[0, 0]
        if env_static is not None:
            env_static.render(ax)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, alpha=0.6, label='Trajectory')
        ax.scatter(position[0], position[1], c='orange', s=300, marker='D',
                  label=f'Position (t={t})', zorder=18, edgecolor='black', linewidth=2)

        # Plot SDF gradient vector at current position
        if np.linalg.norm(grad_static) > 1e-6:
            ax.arrow(position[0], position[1],
                    -grad_static[0] * grad_scale, -grad_static[1] * grad_scale,
                    head_width=0.03, head_length=0.04, fc='blue', ec='blue', alpha=0.8,
                    linewidth=3, label=f'SDF Gradient (mag={np.linalg.norm(grad_static):.3f})')

        ax.set_title(f'Static Environment\nTimestep: {t}/{H-1}\nSDF={sdf_static:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot 2: Dynamic environment SDF gradient
        ax = axes[0, 1]
        if env_dynamic is not None:
            env_dynamic.render(ax, time=time)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, alpha=0.6, label='Trajectory')
        ax.scatter(position[0], position[1], c='orange', s=300, marker='D',
                  label=f'Position (t={t})', zorder=18, edgecolor='black', linewidth=2)

        # Plot SDF gradient vector at current position
        if np.linalg.norm(grad_dynamic) > 1e-6:
            ax.arrow(position[0], position[1],
                    -grad_dynamic[0] * grad_scale, -grad_dynamic[1] * grad_scale,
                    head_width=0.03, head_length=0.04, fc='red', ec='red', alpha=0.8,
                    linewidth=3, label=f'SDF Gradient (mag={np.linalg.norm(grad_dynamic):.3f})')

        ax.set_title(f'Dynamic Environment\nTimestep: {t}/{H-1}, time={time:.3f}\nSDF={sdf_dynamic:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot 3: Gradient difference (dynamic - static)
        ax = axes[1, 0]
        if env_dynamic is not None:
            env_dynamic.render(ax, time=time)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, alpha=0.6, label='Trajectory')
        ax.scatter(position[0], position[1], c='orange', s=300, marker='D',
                  label=f'Position (t={t})', zorder=18, edgecolor='black', linewidth=2)

        # Plot gradient difference vector
        if np.linalg.norm(grad_diff) > 1e-6:
            ax.arrow(position[0], position[1],
                    -grad_diff[0] * grad_scale, -grad_diff[1] * grad_scale,
                    head_width=0.03, head_length=0.04, fc='purple', ec='purple', alpha=0.8,
                    linewidth=3, label=f'Gradient Diff (mag={np.linalg.norm(grad_diff):.3f})')

        ax.set_title(f'SDF Gradient Difference (Dynamic - Static)\nTimestep: {t}/{H-1}\nSDF Diff={sdf_diff:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot 4: Gradient and SDF values over timesteps (if we have data for all timesteps)
        ax = axes[1, 1]

        # Collect all timestep data
        timesteps_all = sorted(gradients_static.keys())
        grad_static_mags = [np.linalg.norm(gradients_static[ts]['gradient']) for ts in timesteps_all]
        grad_dynamic_mags = [np.linalg.norm(gradients_dynamic[ts]['gradient']) for ts in timesteps_all]
        grad_diff_mags = [np.linalg.norm(gradients_dynamic[ts]['gradient'] - gradients_static[ts]['gradient'])
                         for ts in timesteps_all]
        sdf_static_vals = [gradients_static[ts]['sdf'] for ts in timesteps_all]
        sdf_dynamic_vals = [gradients_dynamic[ts]['sdf'] for ts in timesteps_all]

        # Create twin axes for SDF values
        ax2 = ax.twinx()

        # Plot gradient magnitudes on left axis
        l1 = ax.plot(timesteps_all, grad_static_mags, 'b-o', label='Static Grad', linewidth=2, markersize=6)
        l2 = ax.plot(timesteps_all, grad_dynamic_mags, 'r-s', label='Dynamic Grad', linewidth=2, markersize=6)
        l3 = ax.plot(timesteps_all, grad_diff_mags, 'purple', linestyle='--',
               marker='^', label='Grad Diff', linewidth=2, markersize=6)

        # Mark current timestep
        ax.axvline(x=t, color='orange', linestyle=':', linewidth=2, alpha=0.5, label=f'Current t={t}')

        # Plot SDF values on right axis
        l4 = ax2.plot(timesteps_all, sdf_static_vals, 'b:', label='Static SDF', linewidth=2, alpha=0.5)
        l5 = ax2.plot(timesteps_all, sdf_dynamic_vals, 'r:', label='Dynamic SDF', linewidth=2, alpha=0.5)

        ax.set_title(f'SDF Gradient & Value Evolution\nCurrent Timestep: {t}/{H-1}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Gradient Magnitude', color='black')
        ax2.set_ylabel('SDF Value', color='gray')
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Combine legends
        lns = l1 + l2 + l3 + l4 + l5
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'gradient_comparison_t{t:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)


def main():
    """Main function to run the gradient comparison analysis."""
    print("="*80)
    print("Comparing Collision Cost Gradients: Static vs Dynamic Environments")
    print("="*80)

    # ========================================================================
    # Configuration
    # ========================================================================

    # Path to result file
    result_path = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-test_2025-11-17_14-44-54/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/selection_start_goal___validation/extra_objects___True/planner_alg___mpd/model_selection___bspline/phase_time_class___PhaseTimeLinear/diffusion_sampling_method___ddim/n_diffusion_steps_without_noise___0/project_gradient_hierarchy___False/trajectory_duration___10.0/0/results_single_plan-023.pt"

    # Timesteps to analyze (evenly spaced)
    timesteps_to_analyze = 'auto'  # or list like [0, 32, 64, 96, 127]

    # Time range for dynamic environment
    time_range = (0.0, 10.0)

    # ========================================================================

    if not os.path.exists(result_path):
        print(f"ERROR: Result file not found: {result_path}")
        return False

    # Setup
    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Load trajectory and control points
    print("\n1. Loading trajectory and control points...")
    trajectory, control_points, result = load_trajectory_and_control_points(result_path)

    # Convert to tensors if needed
    if not torch.is_tensor(trajectory):
        trajectory = torch.tensor(trajectory, **tensor_args)
    if not torch.is_tensor(control_points):
        control_points = torch.tensor(control_points, **tensor_args)

    H = trajectory.shape[0]

    # Determine timesteps to analyze
    if timesteps_to_analyze == 'auto':
        timesteps = [0, H // 4, H // 2, 3 * H // 4, H - 1]
    elif timesteps_to_analyze == 'all':
        timesteps = list(range(H))
    else:
        timesteps = timesteps_to_analyze

    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Control points shape: {control_points.shape}")
    print(f"  Analyzing {len(timesteps)} timesteps: {timesteps}")

    # Create environments
    print("\n2. Creating environments...")
    try:
        # Static environment
        env_static = environments.EnvSimple2DExtraObjectsV00(tensor_args=tensor_args)
        print(f"  Created static environment: EnvSimple2DExtraObjectsV00")

        # Dynamic environment
        env_dynamic = environments.EnvDynSimple2DExtraObjects(
            tensor_args=tensor_args,
            time_range=time_range
        )
        print(f"  Created dynamic environment: EnvDynSimple2DExtraObjects")
    except Exception as e:
        print(f"  ERROR creating environments: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute SDF gradients for both environments
    print("\n3. Computing SDF gradients for static environment...")
    gradients_static = compute_sdf_gradients_for_environment(
        env=env_static,
        trajectory=trajectory,
        timesteps=timesteps,
        tensor_args=tensor_args,
        time_range=time_range
    )
    print(f"  Computed gradients for {len(gradients_static)} timesteps")

    print("\n4. Computing SDF gradients for dynamic environment...")
    gradients_dynamic = compute_sdf_gradients_for_environment(
        env=env_dynamic,
        trajectory=trajectory,
        timesteps=timesteps,
        tensor_args=tensor_args,
        time_range=time_range
    )
    print(f"  Computed gradients for {len(gradients_dynamic)} timesteps")

    # Visualize comparison
    print("\n5. Visualizing gradient comparison...")
    save_dir = os.path.join(os.path.dirname(result_path), "gradient_comparison")
    visualize_gradient_comparison(
        trajectory=trajectory,
        gradients_static=gradients_static,
        gradients_dynamic=gradients_dynamic,
        timesteps=timesteps,
        save_dir=save_dir,
        env_static=env_static,
        env_dynamic=env_dynamic,
        time_range=time_range
    )
    print(f"  Saved visualizations to: {save_dir}")

    # Print summary statistics
    print("\n6. Summary Statistics:")
    print("="*80)
    for t in timesteps:
        grad_static_mag = np.linalg.norm(gradients_static[t]['gradient'])
        grad_dynamic_mag = np.linalg.norm(gradients_dynamic[t]['gradient'])
        grad_diff_mag = np.linalg.norm(gradients_dynamic[t]['gradient'] - gradients_static[t]['gradient'])
        sdf_static = gradients_static[t]['sdf']
        sdf_dynamic = gradients_dynamic[t]['sdf']
        sdf_diff = sdf_dynamic - sdf_static

        print(f"\nTimestep {t}/{H-1} (time={gradients_dynamic[t]['time']:.3f}):")
        print(f"  Position: {gradients_static[t]['position']}")
        print(f"  Static  - SDF: {sdf_static:8.4f}, Grad mag: {grad_static_mag:8.4f}")
        print(f"  Dynamic - SDF: {sdf_dynamic:8.4f}, Grad mag: {grad_dynamic_mag:8.4f}")
        print(f"  Difference    - SDF: {sdf_diff:8.4f}, Grad mag: {grad_diff_mag:8.4f}")
        print(f"  Relative change: {(grad_diff_mag / (grad_static_mag + 1e-8) * 100):.2f}%")

    print("\n" + "="*80)
    print("✓ Success! Gradient comparison completed.")
    print("="*80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
