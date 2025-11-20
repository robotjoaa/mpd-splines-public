#!/usr/bin/env python3
"""
Visualize SDF gradients for trajectories loaded from result_plan_single.pt files.
Shows dynamic obstacles at specific timesteps.
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy
from visualize_sdf_gradients import (
    visualize_sdf_gradient_field_2d,
    create_workspace_grid,
    compute_sdf_and_gradients,
)
from dotmap import DotMap


def load_single_result(result_path):
    """
    Load a single result_plan_single.pt file.

    Args:
        result_path: Path to result_plan_single-XXX.pt file

    Returns:
        result: DotMap containing result data
    """
    print(f"Loading result from: {result_path}")
    result = torch.load(result_path, weights_only=False)

    print(f"  Keys: {list(result.keys())}")
    if 'dyn_obj_config' in result and result.dyn_obj_config is not None:
        print(f"  Dynamic objects: {list(result.dyn_obj_config.keys())}")

    return result


def compute_time_from_timestep(timestep, total_timesteps, time_range=(0.0, 1.0)):
    """
    Compute the time value for a given timestep index.

    Args:
        timestep: Current timestep index
        total_timesteps: Total number of timesteps
        time_range: (t_min, t_max) for time-varying objects

    Returns:
        Time value in [t_min, t_max]
    """
    if total_timesteps <= 1:
        return time_range[0]

    alpha = timestep / (total_timesteps - 1)
    return time_range[0] + alpha * (time_range[1] - time_range[0])


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


def visualize_trajectory_with_dynamic_obstacles(
    env,
    trajectory,
    timestep,
    control_points=None,
    save_path=None,
    resolution=30,
    margin=0.01,
    tensor_args=None,
    time_range=(0.0, 1.0),
):
    """
    Visualize SDF gradients with dynamic obstacles at a specific timestep.

    Args:
        env: Environment object (supports render(ax, time) for dynamic environments)
        trajectory: Trajectory positions (num_T_pts, 2)
        timestep: Timestep index for dynamic obstacle positions
        control_points: Control points (n_control_points, 2) - optional
        save_path: Path to save figure
        resolution: Grid resolution
        margin: Collision margin
        tensor_args: PyTorch device/dtype
        time_range: (t_min, t_max) for time-varying objects
    """
    if tensor_args is None:
        tensor_args = {"device": "cpu", "dtype": torch.float32}

    trajectory = to_numpy(trajectory) if torch.is_tensor(trajectory) else trajectory
    if control_points is not None:
        control_points = to_numpy(control_points) if torch.is_tensor(control_points) else control_points

    total_timesteps = len(trajectory)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Compute time for this timestep
    time = compute_time_from_timestep(timestep, total_timesteps, time_range)

    # Get workspace grid
    X, Y, grid_points = create_workspace_grid(env, resolution=resolution, tensor_args=tensor_args)

    # Check if environment is dynamic
    is_dynamic = check_if_dynamic(env)

    # Compute SDF and gradients at the current time (includes both static and dynamic obstacles)
    # For dynamic environments, use the time-aware compute_sdf method
    if is_dynamic:
        # Dynamic environment - compute SDF at specific time
        grid_points_tensor = torch.tensor(grid_points, **tensor_args)
        grid_points_tensor.requires_grad_(True)

        sdf_vals_tensor = env.compute_sdf(grid_points_tensor, time=time)

        # Compute gradients
        sdf_grads_list = []
        for i in range(len(sdf_vals_tensor)):
            grad = torch.autograd.grad(
                sdf_vals_tensor[i], grid_points_tensor,
                retain_graph=True, create_graph=False
            )[0][i]
            sdf_grads_list.append(grad)
        sdf_grads_tensor = torch.stack(sdf_grads_list)

        sdf_vals = to_numpy(sdf_vals_tensor)
        sdf_grads = to_numpy(sdf_grads_tensor)
    else:
        # Static environment - use helper function
        sdf_vals, sdf_grads = compute_sdf_and_gradients(env, grid_points, tensor_args=tensor_args)

    # Reshape for plotting
    sdf_vals_grid = sdf_vals.reshape(X.shape)
    sdf_grads_x = sdf_grads[:, 0].reshape(X.shape)
    sdf_grads_y = sdf_grads[:, 1].reshape(X.shape)

    # Plot SDF contours
    contour_levels = np.linspace(-0.5, 0.5, 20)
    contourf = ax.contourf(X, Y, sdf_vals_grid, levels=contour_levels, cmap='RdYlBu', alpha=0.3)
    contour = ax.contour(X, Y, sdf_vals_grid, levels=[0, margin], colors=['black', 'blue'],
                         linewidths=[2, 1.5], linestyles=['solid', 'dashed'])
    ax.clabel(contour, inline=True, fontsize=8, fmt={0: 'obstacle surface', margin: f'margin={margin}'})

    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='SDF value')

    # Plot gradient vector field
    mask = np.abs(sdf_vals_grid) < margin * 5
    skip = max(1, resolution // 20)

    ax.quiver(
        X[::skip, ::skip][mask[::skip, ::skip]],
        Y[::skip, ::skip][mask[::skip, ::skip]],
        -sdf_grads_x[::skip, ::skip][mask[::skip, ::skip]],
        -sdf_grads_y[::skip, ::skip][mask[::skip, ::skip]],
        color='blue',
        alpha=0.4,
        scale=50.0,
        width=0.003,
        headwidth=3,
        headlength=4,
        label='Obstacle gradients',
        zorder=5
    )

    # Use environment's render method to plot all obstacles (static and dynamic)
    # This automatically handles dynamic obstacles at the specified time
    env.render(ax, time=time)

    # Annotate timestep
    alpha_val = timestep / (total_timesteps - 1) if total_timesteps > 1 else 0.0
    time_text = f"Timestep: {timestep}/{total_timesteps-1} (α={alpha_val:.2f}, time={time:.3f})"
    ax.text(0.5, 0.98, time_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=20)
    
    # Plot full trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, alpha=0.6, label='Full trajectory', zorder=12)

    # Highlight current position on trajectory
    current_pos = trajectory[timestep]
    ax.scatter(current_pos[0], current_pos[1], c='orange', s=200, marker='D',
              label=f'Current position (t={timestep})', zorder=18, edgecolor='black', linewidth=2)

    # Plot start and goal
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=150, marker='o',
              label='Start', zorder=17, edgecolor='black', linewidth=2)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='s',
              label='Goal', zorder=17, edgecolor='black', linewidth=2)

    # Plot control points if provided
    if control_points is not None:
        ax.scatter(control_points[:, 0], control_points[:, 1], c='purple', s=80, marker='x',
                  linewidth=2, label='Control Points', zorder=16)

    # Set labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(f'SDF Gradients with Dynamic Obstacles\nTimestep: {timestep}/{total_timesteps-1}',
                fontsize=16, fontweight='bold')

    # Set aspect and limits
    ax.set_aspect('equal')
    limits = env.limits
    ax.set_xlim(limits[0, 0], limits[1, 0])
    ax.set_ylim(limits[0, 1], limits[1, 1])

    # Legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig, ax


def visualize_result_at_multiple_timesteps(
    result_path,
    env,
    timesteps='auto',
    save_dir=None,
    resolution=30,
    margin=0.01,
    tensor_args=None,
    time_range=(0.0, 1.0),
):
    """
    Load a result file and visualize at multiple timesteps.

    Args:
        result_path: Path to result_plan_single-XXX.pt
        env: Environment object
        timesteps: List of timestep indices, 'auto' for evenly spaced, or 'all'
        save_dir: Directory to save figures
        resolution: Grid resolution
        margin: Collision margin
        tensor_args: PyTorch device/dtype
        time_range: (t_min, t_max) for time-varying objects
    """
    # Load result
    result = load_single_result(result_path)

    # Extract best trajectory
    if 'q_trajs_pos_best' in result and result.q_trajs_pos_best is not None:
        trajectory = to_numpy(result.q_trajs_pos_best)  # Take first batch
        print(f"  Loaded best trajectory: {trajectory.shape}")
    elif 'q_trajs_pos_valid' in result and result.q_trajs_pos_valid is not None:
        trajectory = to_numpy(result.q_trajs_pos_valid[0])
        print(f"  Loaded valid trajectory (first): {trajectory.shape}")
    else:
        print("  ERROR: No trajectory found in result!")
        return

    # Extract control points if available - following pattern from dynamic_tasks.py lines 495-498
    # If control_points is None, use trajectory as control_points (matching dynamic_tasks.py pattern)
    control_points = None
    if 'q_control_points_best' in result and result.q_control_points_best is not None:
        control_points = to_numpy(result.q_control_points_best)
        print(f"  Loaded control points: {control_points.shape}")
    else:
        control_points = trajectory
        print(f"  No control points found, using trajectory as control points")

    # Determine timesteps to visualize
    total_timesteps = len(trajectory)
    if timesteps == 'auto':
        # Evenly spaced: start, middle, end, plus 2 more
        timesteps = [0, total_timesteps // 4, total_timesteps // 2,
                    3 * total_timesteps // 4, total_timesteps - 1]
    elif timesteps == 'all':
        timesteps = list(range(total_timesteps))
    elif isinstance(timesteps, int):
        timesteps = [timesteps]

    print(f"  Visualizing {len(timesteps)} timesteps: {timesteps}")

    # Create save directory
    if save_dir is None:
        save_dir = os.path.dirname(result_path)
    os.makedirs(save_dir, exist_ok=True)

    # Visualize each timestep
    figures = []
    for t in timesteps:
        result_name = os.path.basename(result_path).replace('.pt', '')
        save_path = os.path.join(save_dir, f'{result_name}_timestep_{t:03d}.png')

        fig, ax = visualize_trajectory_with_dynamic_obstacles(
            env=env,
            trajectory=trajectory,
            timestep=t,
            control_points=control_points,
            save_path=save_path,
            resolution=resolution,
            margin=margin,
            tensor_args=tensor_args,
            time_range=time_range,
        )

        figures.append(fig)
        plt.close(fig)

    print(f"\n  Created {len(figures)} visualizations in {save_dir}")
    return figures


def test_with_actual_result():
    """
    Test with an actual result_plan_single.pt file.
    Modify the paths below to point to your actual files.
    """
    print("="*80)
    print("Testing SDF Gradient Visualization with Actual Results")
    print("="*80)

    # ========================================================================
    # MODIFY THESE PATHS
    # ========================================================================

    # Path to a result file
    #result_path = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-test_2025-11-07_17-35-52/waypoints/results_single_plan-000.pt"
    # bspline
    result_path = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-test_2025-11-17_14-44-54/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/selection_start_goal___validation/extra_objects___True/planner_alg___mpd/model_selection___bspline/phase_time_class___PhaseTimeLinear/diffusion_sampling_method___ddim/n_diffusion_steps_without_noise___0/project_gradient_hierarchy___False/trajectory_duration___10.0/0/results_single_plan-023.pt"
    # waypoint
    # result_path = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-test_2025-11-17_14-44-54/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/selection_start_goal___validation/extra_objects___True/planner_alg___mpd/model_selection___waypoints/phase_time_class___PhaseTimeLinear/diffusion_sampling_method___ddim/n_diffusion_steps_without_noise___0/project_gradient_hierarchy___False/trajectory_duration___10.0/0/results_single_plan-023.pt"
    # OR: Automatically find first result in a log directory
    # LOG_DIR = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs"
    # exp_name = "launch_inference-experiments-test_2025-11-07_17-35-52"
    # import glob
    # pattern = os.path.join(LOG_DIR, exp_name, "**/results_single_plan-*.pt")
    # result_files = glob.glob(pattern, recursive=True)
    # if result_files:
    #     result_path = result_files[0]
    #     print(f"Found result file: {result_path}")
    # else:
    #     print(f"No result files found in {os.path.join(LOG_DIR, exp_name)}")
    #     return False

    # ========================================================================

    if not os.path.exists(result_path):
        print(f"ERROR: Result file not found: {result_path}")
        print("\nPlease modify the result_path variable in test_with_actual_result()")
        print("to point to an actual result_plan_single-XXX.pt file.")
        return False

    # Setup
    device = get_torch_device("cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Create environment (must match the one used during inference)
    # Modify this to match your environment
    try:
        from torch_robotics import environments
        #env = environments.EnvSimple2D(tensor_args=tensor_args)
        env = environments.EnvDynSimple2DExtraObjects(tensor_args=tensor_args)
        print(f"Created environment: EnvSimple2D")
    except Exception as e:
        print(f"ERROR creating environment: {e}")
        print("Modify the environment creation code to match your setup")
        return False

    # Visualize at multiple timesteps
    try:
        visualize_result_at_multiple_timesteps(
            result_path=result_path,
            env=env,
            timesteps='auto',  # or [0, 30, 60, 90, 127] or 'all'
            save_dir=os.path.join(os.path.dirname(result_path), "gradient_viz"),
            resolution=30,
            margin=0.01,
            tensor_args=tensor_args,
            time_range=(0,10)
        )

        print("\n" + "="*80)
        print("✓ Success! Visualizations created.")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n✗ ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_actual_result()
    sys.exit(0 if success else 1)
