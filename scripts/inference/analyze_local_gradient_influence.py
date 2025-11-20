#!/usr/bin/env python3
"""
Analyze how collision costs from dynamic obstacles at specific timesteps
affect local control points.

This script visualizes:
1. Which timesteps have collisions with dynamic obstacles
2. How the SDF gradients propagate to control points
3. The spatial locality of gradient influence (which control points are affected)
4. The magnitude of gradient contributions from collision timesteps
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy
from torch_robotics import environments


def check_if_dynamic(env):
    """Check if environment is dynamic (has time-varying obstacles)."""
    if hasattr(env, '_has_moving_objects'):
        return env._has_moving_objects()
    if hasattr(env, 'is_static'):
        return not env.is_static
    if hasattr(env, 'time_range'):
        return True
    return False


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
        control_points = trajectory
        print(f"  No control points found, using trajectory as control points")

    return trajectory, control_points, result


def compute_trajectory_sdf_profile(env, trajectory, time_range=(0.0, 1.0), tensor_args=None):
    """
    Compute SDF values and gradients along the entire trajectory.

    Returns:
        sdf_profile: Dict with timestep -> (sdf, gradient, time) mapping
    """
    is_dynamic = check_if_dynamic(env)
    H = trajectory.shape[0]

    sdf_profile = {}

    for t in range(H):
        position = trajectory[t]

        # Compute time for this timestep
        alpha = t / (H - 1) if H > 1 else 0.0
        time = time_range[0] + alpha * (time_range[1] - time_range[0])

        with torch.enable_grad():
            pos_grad = position.clone().detach().requires_grad_(True)

            if is_dynamic:
                sdf_val = env.compute_sdf(pos_grad.unsqueeze(0), time=time)
            else:
                sdf_val = env.compute_sdf(pos_grad.unsqueeze(0))

            sdf_grad = torch.autograd.grad(
                outputs=sdf_val,
                inputs=pos_grad,
                create_graph=False
            )[0]

        sdf_profile[t] = {
            'sdf': to_numpy(sdf_val.squeeze()),
            'gradient': to_numpy(sdf_grad),
            'position': to_numpy(position),
            'time': time,
            'in_collision': float(to_numpy(sdf_val.squeeze())) < 0.0
        }

    return sdf_profile


def compute_control_point_influence_matrix(trajectory, control_points, basis_matrix=None):
    """
    Compute influence matrix showing which control points affect which trajectory points.

    For B-splines, this is the basis matrix.
    For waypoints, trajectory points are interpolated from control points.

    Returns:
        influence_matrix: (H, K) matrix where H is trajectory length, K is num control points
                         influence_matrix[t, k] = how much control point k affects trajectory point t
    """
    H = trajectory.shape[0]  # trajectory length
    K = control_points.shape[0]  # number of control points

    if basis_matrix is not None:
        # Use provided basis matrix (for B-splines)
        return to_numpy(basis_matrix) if torch.is_tensor(basis_matrix) else basis_matrix

    # For waypoints or if basis matrix not available, estimate influence
    # Use simple nearest-neighbor influence with Gaussian falloff
    influence_matrix = np.zeros((H, K))

    for t in range(H):
        # Compute distances to all control points
        distances = np.linalg.norm(
            to_numpy(trajectory[t:t+1]) - to_numpy(control_points),
            axis=-1
        )

        # Gaussian falloff (sigma = average distance between control points / 2)
        if K > 1:
            cp_spacing = np.mean(np.linalg.norm(
                to_numpy(control_points[1:]) - to_numpy(control_points[:-1]),
                axis=-1
            ))
            sigma = cp_spacing / 2
        else:
            sigma = 1.0

        influence = np.exp(-distances**2 / (2 * sigma**2))
        influence = influence / (influence.sum() + 1e-8)  # normalize

        influence_matrix[t, :] = influence

    return influence_matrix


def analyze_gradient_locality(sdf_profile, influence_matrix, collision_threshold=0.0):
    """
    Analyze how collision gradients at specific timesteps affect control points.

    Args:
        sdf_profile: Dict mapping timestep -> {sdf, gradient, ...}
        influence_matrix: (H, K) matrix of control point influences
        collision_threshold: SDF threshold for collision (default: 0.0)

    Returns:
        analysis: Dict with locality analysis results
    """
    H = len(sdf_profile)  # trajectory length
    K = influence_matrix.shape[1]  # number of control points

    # Identify collision timesteps
    collision_timesteps = [
        t for t in range(H)
        if sdf_profile[t]['sdf'] < collision_threshold
    ]

    # Compute gradient magnitude at each timestep
    gradient_magnitudes = np.array([
        np.linalg.norm(sdf_profile[t]['gradient'])
        for t in range(H)
    ])

    # Compute control point gradient contributions
    # For each control point, sum the weighted gradients from all timesteps
    cp_gradient_contributions = np.zeros((K, 2))  # (K, state_dim)
    cp_collision_contributions = np.zeros((K, 2))  # contributions from collision timesteps only
    cp_influence_weights = np.zeros(K)  # total influence from collision timesteps

    for t in range(H):
        grad = sdf_profile[t]['gradient']
        is_collision = t in collision_timesteps

        for k in range(K):
            weight = influence_matrix[t, k]
            cp_gradient_contributions[k] += weight * grad

            if is_collision:
                cp_collision_contributions[k] += weight * grad
                cp_influence_weights[k] += weight

    # Identify which control points are most affected by collisions
    collision_cp_magnitudes = np.linalg.norm(cp_collision_contributions, axis=-1)
    total_cp_magnitudes = np.linalg.norm(cp_gradient_contributions, axis=-1)

    # Compute locality metrics
    analysis = {
        'collision_timesteps': collision_timesteps,
        'num_collision_timesteps': len(collision_timesteps),
        'total_timesteps': H,
        'collision_ratio': len(collision_timesteps) / H,
        'gradient_magnitudes': gradient_magnitudes,
        'cp_gradient_contributions': cp_gradient_contributions,
        'cp_collision_contributions': cp_collision_contributions,
        'cp_influence_weights': cp_influence_weights,
        'collision_cp_magnitudes': collision_cp_magnitudes,
        'total_cp_magnitudes': total_cp_magnitudes,
        'num_affected_cps': np.sum(collision_cp_magnitudes > 1e-6),
        'influence_matrix': influence_matrix,
    }

    return analysis


def visualize_gradient_locality_analysis(
    trajectory,
    control_points,
    sdf_profile,
    analysis,
    env=None,
    save_path=None,
    time_range=(0.0, 1.0)
):
    """
    Create comprehensive visualization of gradient locality.
    """
    trajectory_np = to_numpy(trajectory) if torch.is_tensor(trajectory) else trajectory
    control_points_np = to_numpy(control_points) if torch.is_tensor(control_points) else control_points

    H = len(sdf_profile)
    K = control_points_np.shape[0]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Trajectory with collision regions highlighted
    ax1 = fig.add_subplot(gs[0, 0])
    if env is not None:
        # Show environment at middle time
        mid_time = (time_range[0] + time_range[1]) / 2
        env.render(ax1, time=mid_time)

    ax1.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'g-', linewidth=2, alpha=0.6, label='Trajectory')
    ax1.scatter(control_points_np[:, 0], control_points_np[:, 1], c='purple', s=100,
               marker='x', linewidth=2, label='Control Points', zorder=16)

    # Highlight collision regions
    for t in analysis['collision_timesteps']:
        pos = sdf_profile[t]['position']
        ax1.scatter(pos[0], pos[1], c='red', s=50, alpha=0.7, zorder=15)

    # Add control point indices
    for k in range(K):
        ax1.text(control_points_np[k, 0], control_points_np[k, 1] + 0.05,
                f'{k}', fontsize=9, ha='center', va='bottom')

    ax1.set_title(f'Trajectory with {analysis["num_collision_timesteps"]} Collision Points',
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: SDF profile over time
    ax2 = fig.add_subplot(gs[0, 1])
    timesteps = np.arange(H)
    sdf_values = [sdf_profile[t]['sdf'] for t in range(H)]

    ax2.plot(timesteps, sdf_values, 'b-', linewidth=2, label='SDF')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Collision threshold')
    ax2.fill_between(timesteps, sdf_values, 0,
                     where=np.array(sdf_values) < 0,
                     alpha=0.3, color='red', label='Collision region')

    ax2.set_title('SDF Profile Along Trajectory', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('SDF Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient magnitude over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(timesteps, analysis['gradient_magnitudes'], 'g-', linewidth=2, label='Gradient Magnitude')

    # Highlight collision timesteps
    collision_mask = np.array([t in analysis['collision_timesteps'] for t in range(H)])
    ax3.scatter(timesteps[collision_mask], analysis['gradient_magnitudes'][collision_mask],
               c='red', s=50, zorder=10, label='Collision timesteps')

    ax3.set_title('Gradient Magnitude Profile', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('||∇SDF||')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Control point influence heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    im = ax4.imshow(analysis['influence_matrix'].T, aspect='auto', cmap='viridis',
                    interpolation='nearest')
    ax4.set_title('Control Point Influence Matrix\n(rows=control points, cols=timesteps)',
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trajectory Timestep')
    ax4.set_ylabel('Control Point Index')

    # Mark collision timesteps
    for t in analysis['collision_timesteps']:
        ax4.axvline(x=t, color='red', alpha=0.3, linewidth=1)

    plt.colorbar(im, ax=ax4, label='Influence Weight')

    # Plot 5: Control point gradient contributions
    ax5 = fig.add_subplot(gs[1, 2])
    cp_indices = np.arange(K)

    # Bar plot showing gradient contributions
    width = 0.35
    ax5.bar(cp_indices - width/2, analysis['total_cp_magnitudes'], width,
           label='Total Gradient', alpha=0.7, color='blue')
    ax5.bar(cp_indices + width/2, analysis['collision_cp_magnitudes'], width,
           label='Collision Gradient', alpha=0.7, color='red')

    ax5.set_title('Control Point Gradient Magnitudes', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Control Point Index')
    ax5.set_ylabel('Gradient Magnitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Affected control points visualization
    ax6 = fig.add_subplot(gs[2, 0])
    if env is not None:
        env.render(ax6, time=time_range[0])

    ax6.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'g-', linewidth=1, alpha=0.4)

    # Color control points by collision gradient magnitude
    scatter = ax6.scatter(control_points_np[:, 0], control_points_np[:, 1],
                         c=analysis['collision_cp_magnitudes'],
                         s=200, cmap='hot', vmin=0,
                         vmax=np.max(analysis['collision_cp_magnitudes']) + 1e-8,
                         edgecolors='black', linewidth=2, zorder=20)

    # Add control point indices
    for k in range(K):
        ax6.text(control_points_np[k, 0], control_points_np[k, 1] + 0.05,
                f'{k}', fontsize=9, ha='center', va='bottom', fontweight='bold')

    plt.colorbar(scatter, ax=ax6, label='Collision Gradient Magnitude')
    ax6.set_title('Control Points Colored by Collision Gradient', fontsize=12, fontweight='bold')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Locality statistics text
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')

    # Compute statistics
    affected_cp_indices = np.where(analysis['collision_cp_magnitudes'] > 1e-6)[0]
    total_collision_grad = np.sum(analysis['collision_cp_magnitudes'])

    stats_text = f"""
GRADIENT LOCALITY ANALYSIS

Collision Statistics:
  • Collision timesteps: {analysis['num_collision_timesteps']} / {H} ({analysis['collision_ratio']*100:.1f}%)
  • Collision timestep indices: {analysis['collision_timesteps'][:10]}{'...' if len(analysis['collision_timesteps']) > 10 else ''}

Control Point Impact:
  • Total control points: {K}
  • Control points affected by collisions: {analysis['num_affected_cps']}
  • Affected CP indices: {affected_cp_indices.tolist()}
  • Total collision gradient magnitude: {total_collision_grad:.4f}

Gradient Distribution:
  • Min collision CP gradient: {np.min(analysis['collision_cp_magnitudes']):.6f}
  • Max collision CP gradient: {np.max(analysis['collision_cp_magnitudes']):.6f}
  • Mean collision CP gradient: {np.mean(analysis['collision_cp_magnitudes']):.6f}
  • Std collision CP gradient: {np.std(analysis['collision_cp_magnitudes']):.6f}

Top 5 Most Affected Control Points:
"""

    # Add top 5 control points
    sorted_indices = np.argsort(analysis['collision_cp_magnitudes'])[::-1]
    for i, k in enumerate(sorted_indices[:5]):
        grad_mag = analysis['collision_cp_magnitudes'][k]
        influence_weight = analysis['cp_influence_weights'][k]
        stats_text += f"  {i+1}. CP {k}: gradient={grad_mag:.6f}, influence={influence_weight:.3f}\n"

    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Gradient Locality Analysis: {analysis["num_collision_timesteps"]} Collision Timesteps',
                fontsize=14, fontweight='bold')

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def main():
    """Main function."""
    print("="*80)
    print("Analyzing Local Gradient Influence from Dynamic Obstacle Collisions")
    print("="*80)

    # Configuration
    result_path = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-test_2025-11-17_14-44-54/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/selection_start_goal___validation/extra_objects___True/planner_alg___mpd/model_selection___bspline/phase_time_class___PhaseTimeLinear/diffusion_sampling_method___ddim/n_diffusion_steps_without_noise___0/project_gradient_hierarchy___False/trajectory_duration___10.0/0/results_single_plan-023.pt"

    time_range = (0.0, 10.0)

    if not os.path.exists(result_path):
        print(f"ERROR: Result file not found: {result_path}")
        return False

    # Setup
    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Load data
    print("\n1. Loading trajectory and control points...")
    trajectory, control_points, result = load_trajectory_and_control_points(result_path)

    if not torch.is_tensor(trajectory):
        trajectory = torch.tensor(trajectory, **tensor_args)
    if not torch.is_tensor(control_points):
        control_points = torch.tensor(control_points, **tensor_args)

    print(f"  Trajectory: {trajectory.shape}")
    print(f"  Control points: {control_points.shape}")

    # Create dynamic environment
    print("\n2. Creating dynamic environment...")
    try:
        env = environments.EnvDynSimple2DExtraObjects(
            tensor_args=tensor_args,
            time_range=time_range
        )
        print(f"  Created: EnvDynSimple2DExtraObjects")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute SDF profile
    print("\n3. Computing SDF profile along trajectory...")
    sdf_profile = compute_trajectory_sdf_profile(
        env, trajectory, time_range=time_range, tensor_args=tensor_args
    )

    collision_count = sum(1 for t in sdf_profile if sdf_profile[t]['in_collision'])
    print(f"  Found {collision_count} collision timesteps out of {len(sdf_profile)}")

    # Compute influence matrix
    print("\n4. Computing control point influence matrix...")
    influence_matrix = compute_control_point_influence_matrix(trajectory, control_points)
    print(f"  Influence matrix: {influence_matrix.shape}")

    # Analyze gradient locality
    print("\n5. Analyzing gradient locality...")
    analysis = analyze_gradient_locality(sdf_profile, influence_matrix)

    print(f"  Collision timesteps: {analysis['num_collision_timesteps']} / {analysis['total_timesteps']}")
    print(f"  Affected control points: {analysis['num_affected_cps']} / {control_points.shape[0]}")

    # Visualize
    print("\n6. Creating visualization...")
    save_dir = os.path.join(os.path.dirname(result_path), "gradient_locality_analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gradient_locality_analysis.png")

    fig = visualize_gradient_locality_analysis(
        trajectory, control_points, sdf_profile, analysis,
        env=env, save_path=save_path, time_range=time_range
    )
    plt.close(fig)

    print("\n" + "="*80)
    print("✓ Success! Gradient locality analysis completed.")
    print(f"  Results saved to: {save_dir}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
