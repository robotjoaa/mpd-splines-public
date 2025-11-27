"""
Integration module to render COBL trajectories in original scale.

This module provides a wrapper around the existing animate_opt_iters_robots
to automatically convert trajectories to original scale and overlay reference trajectories.
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def load_eval_dataset_info(eval_dataset_path):
    """Load evaluation dataset and extract scale info."""
    with open(eval_dataset_path, 'rb') as f:
        eval_data = pickle.load(f)
    return eval_data


def render_trajectory_with_reference_original_scale(
    task,
    trajs_pos,
    traj_pos_best,
    start_state,
    goal_state,
    control_points,
    eval_dataset_path,
    scenario_idx,
    n_frames=10,
    video_filepath=None,
    anim_time=None,
    render_time=None,
    **kwargs
):
    """
    Enhanced version of animate_opt_iters_robots that:
    1. Converts trajectories to original scale
    2. Overlays reference ego trajectory from eval dataset
    3. Properly scales the rendering

    Args:
        task: Planning task object
        trajs_pos: Optimization iterations (S, B, H, D) in normalized scale
        traj_pos_best: Best trajectory (H, D) in normalized scale
        start_state: Start position in normalized scale
        goal_state: Goal position in normalized scale
        control_points: Control points for trajectories
        eval_dataset_path: Path to eval_dataset.pkl
        scenario_idx: Index of scenario in eval dataset
        n_frames: Number of animation frames
        video_filepath: Output video path
        anim_time: Animation duration in seconds
        render_time: Specific time for rendering obstacles
        **kwargs: Additional arguments
    """
    # Load eval dataset
    eval_data = load_eval_dataset_info(eval_dataset_path)
    scale_factor = eval_data['scale_factor']
    reference_traj = eval_data['ego_trajectories'][scenario_idx]  # Original scale
    obstacle_trajs = eval_data['obstacle_trajectories'][scenario_idx]
    obstacle_radius_orig = eval_data['obstacle_radius'] * scale_factor  # Convert to meters

    print(f"\nRendering with original scale:")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Reference trajectory shape: {reference_traj.shape}")
    print(f"  Obstacle radius (meters): {obstacle_radius_orig:.3f}")

    # Convert trajectories to original scale
    trajs_pos_orig = trajs_pos * scale_factor
    traj_pos_best_orig = traj_pos_best * scale_factor if traj_pos_best is not None else None
    start_state_orig = start_state * scale_factor if start_state is not None else None
    goal_state_orig = goal_state * scale_factor if goal_state is not None else None
    control_points_orig = control_points * scale_factor if control_points is not None else None

    # Convert reference trajectory to torch tensor in normalized scale (for compatibility)
    reference_traj_norm = torch.tensor(reference_traj / scale_factor,
                                      dtype=trajs_pos.dtype,
                                      device=trajs_pos.device)

    # Create custom rendering function
    from mpd.torch_robotics.torch_robotics.utils.plotting_utils import (
        create_fig_and_axes, create_animation_video, remove_axes_labels_ticks
    )

    S, B, H, D = trajs_pos.shape
    idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
    trajs_pos_selection = trajs_pos_orig[idxs]
    if control_points_orig is None:
        control_points_selection = trajs_pos_selection
    else:
        control_points_selection = control_points_orig[idxs]

    # Get render time
    if render_time is None:
        mid_idx = H // 2
        render_range = np.arange(max(0, mid_idx - 5), min(H, mid_idx + 5))
        time_steps = task._get_timesteps_for_horizon(H)
        if time_steps is not None:
            render_time = time_steps[mid_idx]
            if isinstance(render_time, torch.Tensor):
                render_time = float(render_time.cpu())
    else:
        render_range = np.arange(max(0, render_time - 5), min(H, render_time + 5))
        time_steps = task._get_timesteps_for_horizon(H)
        render_time = time_steps[render_time]

    fig, ax = create_fig_and_axes(dim=task.env.dim)

    def animate_fn(i, ax):
        ax.clear()
        ax.set_title(f"iter: {idxs[i]}/{S-1} (Original Scale - Meters)")

        # Render environment (obstacles at render_time)
        if hasattr(task.env, 'render'):
            import inspect
            sig = inspect.signature(task.env.render)
            if 'time' in sig.parameters and render_time is not None:
                task.env.render(ax, time=render_time)
            else:
                task.env.render(ax)

        # Scale obstacle rendering to original scale
        # Adjust axis limits to meters
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] * scale_factor, xlim[1] * scale_factor)
        ax.set_ylim(ylim[0] * scale_factor, ylim[1] * scale_factor)

        # Render reference trajectory (already in original scale)
        if render_range is not None:
            ax.plot(reference_traj[render_range, 0], reference_traj[render_range, 1],
                   'g-', linewidth=3, label='Reference (COBL)', alpha=0.7, zorder=1)
        else:
            ax.plot(reference_traj[:, 0], reference_traj[:, 1],
                   'g-', linewidth=3, label='Reference (COBL)', alpha=0.7, zorder=1)

        # Render predicted trajectories (converted to original scale)
        q_pos_trajs = trajs_pos_selection[i]
        colors = []
        for j in range(len(q_pos_trajs)):
            colors.append('red' if j < len(q_pos_trajs) // 2 else 'orange')

        if render_range is not None:
            task.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs[:, render_range, :],
                                         colors=colors, **kwargs)
        else:
            task.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs,
                                         colors=colors, **kwargs)

        # Render best trajectory
        if i == n_frames - 1 and traj_pos_best_orig is not None:
            if render_range is not None:
                task.robot.render_trajectories(ax,
                    q_pos_trajs=traj_pos_best_orig[render_range, :].unsqueeze(0),
                    colors=["blue"], linewidths=[3], **kwargs)
            else:
                task.robot.render_trajectories(ax,
                    q_pos_trajs=traj_pos_best_orig.unsqueeze(0),
                    colors=["blue"], linewidths=[3], **kwargs)

        # Render start and goal
        if start_state_orig is not None:
            task.robot.render(ax, start_state_orig, color="green", cmap="Greens")
        if goal_state_orig is not None:
            task.robot.render(ax, goal_state_orig, color="purple", cmap="Purples")

        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,),
                          video_filepath=video_filepath, anim_time=anim_time, **kwargs)

    print(f"Saved video to {video_filepath}")


def create_comparison_plot_original_scale(
    predicted_traj,
    eval_dataset_path,
    scenario_idx,
    save_path,
    time_idx=None,
    title=None,
):
    """
    Create a static comparison plot in original scale.

    Args:
        predicted_traj: Predicted trajectory in normalized scale, shape (H, 2)
        eval_dataset_path: Path to eval_dataset.pkl
        scenario_idx: Index of scenario
        save_path: Output image path
        time_idx: Time index for obstacle rendering (None for mid-point)
        title: Plot title
    """
    # Load eval dataset
    eval_data = load_eval_dataset_info(eval_dataset_path)
    scale_factor = eval_data['scale_factor']
    reference_traj = eval_data['ego_trajectories'][scenario_idx] * scale_factor 
    obstacle_trajs = eval_data['obstacle_trajectories'][scenario_idx] * scale_factor 
    obstacle_radius_orig = eval_data['obstacle_radius'] * scale_factor

    # Convert to original scale
    if isinstance(predicted_traj, torch.Tensor):
        predicted_traj = predicted_traj.cpu().numpy()
    pred_orig = predicted_traj * scale_factor

    H = len(predicted_traj)
    if time_idx is None:
        time_idx = H // 2

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot trajectories
    ax.plot(reference_traj[:, 0], reference_traj[:, 1],
           'g-', linewidth=3, label='Reference (COBL)', alpha=0.7)
    ax.plot(pred_orig[:, 0], pred_orig[:, 1],
           'b-', linewidth=3, label='Predicted (MPD)', alpha=0.7)

    # Plot start and goal
    ax.plot(reference_traj[0, 0], reference_traj[0, 1],
           'go', markersize=15, label='Start', zorder=10)
    ax.plot(reference_traj[-1, 0], reference_traj[-1, 1],
           'ro', markersize=15, label='Goal', zorder=10)

    # Plot obstacles at time_idx
    if obstacle_trajs is not None:
        for obs_traj in obstacle_trajs:
            if obs_traj is None:
                continue

            # Convert to original scale
            obs_orig = obs_traj

            # Check for NaN
            if np.isnan(obs_orig[time_idx, 0]):
                continue

            # Plot obstacle
            obs_x, obs_y = obs_orig[time_idx, 0], obs_orig[time_idx, 1]
            circle = Circle((obs_x, obs_y), obstacle_radius_orig,
                          color='red', alpha=0.3, zorder=5)
            ax.add_patch(circle)

            # Plot obstacle trajectory
            valid_mask = ~np.isnan(obs_orig[:, 0])
            ax.plot(obs_orig[valid_mask, 0], obs_orig[valid_mask, 1],
                   'r--', alpha=0.3, linewidth=1)

    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    # ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f'Trajectory Comparison - Scenario {scenario_idx} (Original Scale)', fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    print("This module provides utilities for rendering COBL trajectories in original scale.")
    print("\nUsage example:")
    print("""
    from render_trajectory_original_scale_integration import render_trajectory_with_reference_original_scale

    # In your inference script, replace:
    # planning_task.animate_opt_iters_robots(...)

    # With:
    render_trajectory_with_reference_original_scale(
        task=planning_task,
        trajs_pos=results.q_trajs_pos_iters,
        traj_pos_best=results.q_trajs_pos_best,
        start_state=q_pos_start,
        goal_state=q_pos_goal,
        control_points=results.control_points_iters,
        eval_dataset_path='/path/to/eval_dataset.pkl',
        scenario_idx=idx,
        n_frames=10,
        video_filepath='output.mp4',
        anim_time=8.0,
        render_time=40,
    )
    """)