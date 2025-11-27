"""
Plot COBL trajectory results in original dataset scale (meters) for comparison with reference trajectories.

This script provides utilities to:
1. Load trajectories from inference results
2. Convert from normalized scale back to original scale (meters)
3. Overlay reference ego_trajectory from evaluation dataset
4. Create comparison visualizations
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import torch


def load_eval_dataset(eval_dataset_path):
    """
    Load evaluation dataset containing reference trajectories.

    Args:
        eval_dataset_path: Path to eval_dataset.pkl

    Returns:
        dict with keys: ego_trajectories, obstacle_trajectories, scale_factor, etc.
    """
    with open(eval_dataset_path, 'rb') as f:
        eval_data = pickle.load(f)
    return eval_data


def convert_to_original_scale(trajectories, scale_factor=10.0):
    """
    Convert trajectories from normalized scale back to original scale (meters).

    Args:
        trajectories: Array of trajectories in normalized scale
                     Shape: (..., 2) for positions or (..., H, 2) for full trajectories
        scale_factor: Scale factor used for normalization (default: 10.0)

    Returns:
        Trajectories in original scale (meters)
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()

    return trajectories * scale_factor


def plot_trajectory_comparison(
    predicted_traj,
    reference_traj,
    start_pos,
    goal_pos,
    obstacles=None,
    time=None,
    scale_factor=10.0,
    title=None,
    save_path=None,
    show_plot=True,
    figsize=(10, 10),
    obstacle_radius=0.35,  # meters (original scale)
):
    """
    Plot predicted trajectory vs reference trajectory in original scale.

    Args:
        predicted_traj: Predicted trajectory in normalized scale, shape (H, 2)
        reference_traj: Reference trajectory in original scale, shape (H, 2)
        start_pos: Start position in normalized scale, shape (2,)
        goal_pos: Goal position in normalized scale, shape (2,)
        obstacles: List of obstacle trajectories (each [H, 6]) in normalized scale
        time: Time index for rendering obstacles (if None, uses mid-point)
        scale_factor: Scale factor for conversion (default: 10.0)
        title: Plot title
        save_path: Path to save figure (if None, doesn't save)
        show_plot: Whether to display plot
        figsize: Figure size
        obstacle_radius: Obstacle radius in meters (original scale)
    """
    # Convert to original scale
    pred_orig = convert_to_original_scale(predicted_traj, scale_factor)
    # start_orig = convert_to_original_scale(start_pos, scale_factor)
    # goal_orig = convert_to_original_scale(goal_pos, scale_factor)

    # Reference trajectory should already be in original scale
    ref_orig = convert_to_original_scale(reference_traj, scale_factor)
    start_orig = ref_orig[0]
    goal_orig = ref_orig[-1]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot trajectories
    ax.plot(ref_orig[:, 0], ref_orig[:, 1], 'g-', linewidth=2, label='Reference (COBL)', alpha=0.7)
    ax.plot(pred_orig[:, 0], pred_orig[:, 1], 'b-', linewidth=2, label='Predicted (MPD)', alpha=0.7)

    # Plot start and goal
    ax.plot(start_orig[0], start_orig[1], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(goal_orig[0], goal_orig[1], 'ro', markersize=15, label='Goal', zorder=10)

    # Plot obstacles if provided
    if obstacles is not None and len(obstacles) > 0:
        H = predicted_traj.shape[0]
        if time is None:
            time = H // 2  # Mid-point

        for obs_traj in obstacles:
            if obs_traj is None:
                continue

            # Convert obstacle trajectory to original scale
            obs_orig = convert_to_original_scale(obs_traj, scale_factor)

            # Check if trajectory contains NaN (obstacle not present at this time)
            if np.isnan(obs_orig[time, 0]):
                continue

            # Plot obstacle position at specified time
            obs_x, obs_y = obs_orig[time, 0], obs_orig[time, 1]
            circle = plt.Circle((obs_x, obs_y), obstacle_radius,
                              color='red', alpha=0.3, zorder=5)
            ax.add_patch(circle)

            # Plot obstacle trajectory (positions only)
            valid_mask = ~np.isnan(obs_orig[:, 0])
            ax.plot(obs_orig[valid_mask, 0], obs_orig[valid_mask, 1],
                   'r--', alpha=0.3, linewidth=1)

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Trajectory Comparison (Original Scale, t={time})', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def animate_trajectory_comparison(
    predicted_traj,
    reference_traj,
    start_pos,
    goal_pos,
    obstacles=None,
    scale_factor=10.0,
    dt=0.1,
    save_path=None,
    figsize=(10, 10),
    obstacle_radius=0.35,
    fps=10,
):
    """
    Create animation comparing predicted and reference trajectories in original scale.

    Args:
        predicted_traj: Predicted trajectory in normalized scale, shape (H, 2)
        reference_traj: Reference trajectory in original scale, shape (H, 2)
        start_pos: Start position in normalized scale, shape (2,)
        goal_pos: Goal position in normalized scale, shape (2,)
        obstacles: List of obstacle trajectories (each [H, 6]) in normalized scale
        scale_factor: Scale factor for conversion (default: 10.0)
        dt: Time step in seconds (default: 0.1)
        save_path: Path to save video (required)
        figsize: Figure size
        obstacle_radius: Obstacle radius in meters
        fps: Frames per second for video
    """
    # Convert to original scale
    pred_orig = convert_to_original_scale(predicted_traj, scale_factor)
    start_orig = convert_to_original_scale(start_pos, scale_factor)
    goal_orig = convert_to_original_scale(goal_pos, scale_factor)
    ref_orig =  convert_to_original_scale(reference_traj, scale_factor)

    H = predicted_traj.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Determine plot limits
    all_x = np.concatenate([pred_orig[:, 0], ref_orig[:, 0], [start_orig[0], goal_orig[0]]])
    all_y = np.concatenate([pred_orig[:, 1], ref_orig[:, 1], [start_orig[1], goal_orig[1]]])
    margin = 2.0  # meters
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Initialize plot elements
    ref_line, = ax.plot([], [], 'g-', linewidth=2, label='Reference (COBL)', alpha=0.7)
    pred_line, = ax.plot([], [], 'b-', linewidth=2, label='Predicted (MPD)', alpha=0.7)
    ref_point, = ax.plot([], [], 'go', markersize=10)
    pred_point, = ax.plot([], [], 'bo', markersize=10)

    # Start and goal (static)
    ax.plot(start_orig[0], start_orig[1], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(goal_orig[0], goal_orig[1], 'ro', markersize=15, label='Goal', zorder=10)

    # Prepare obstacle data
    obs_circles = []
    obs_lines = []
    if obstacles is not None:
        for obs_traj in obstacles:
            if obs_traj is None:
                obs_circles.append(None)
                obs_lines.append(None)
                continue

            obs_orig = convert_to_original_scale(obs_traj, scale_factor)

            circle = plt.Circle((0, 0), obstacle_radius, color='red', alpha=0.3, zorder=5)
            ax.add_patch(circle)
            obs_circles.append(circle)

            line, = ax.plot([], [], 'r--', alpha=0.3, linewidth=1)
            obs_lines.append((line, obs_orig))

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top')

    def init():
        ref_line.set_data([], [])
        pred_line.set_data([], [])
        ref_point.set_data([], [])
        pred_point.set_data([], [])
        time_text.set_text('')
        return [ref_line, pred_line, ref_point, pred_point, time_text]

    def animate(frame):
        # Update trajectories (show up to current frame)
        ref_line.set_data(ref_orig[:frame+1, 0], ref_orig[:frame+1, 1])
        pred_line.set_data(pred_orig[:frame+1, 0], pred_orig[:frame+1, 1])

        # Update current positions
        ref_point.set_data([ref_orig[frame, 0]], [ref_orig[frame, 1]])
        pred_point.set_data([pred_orig[frame, 0]], [pred_orig[frame, 1]])

        # Update obstacles
        for i, circle in enumerate(obs_circles):
            if circle is None or obs_lines[i] is None:
                continue

            line, obs_orig = obs_lines[i]

            if not np.isnan(obs_orig[frame, 0]):
                circle.center = (obs_orig[frame, 0], obs_orig[frame, 1])
                circle.set_visible(True)

                # Show obstacle trajectory up to current frame
                valid_mask = ~np.isnan(obs_orig[:frame+1, 0])
                if valid_mask.any():
                    line.set_data(obs_orig[:frame+1][valid_mask, 0],
                                obs_orig[:frame+1][valid_mask, 1])
            else:
                circle.set_visible(False)
                line.set_data([], [])

        time_text.set_text(f'Time: {frame * dt:.2f}s / {(H-1) * dt:.2f}s')

        artists = [ref_line, pred_line, ref_point, pred_point, time_text]
        artists.extend(obs_circles)
        artists.extend([line for line, _ in obs_lines if line is not None])
        return artists

    anim = FuncAnimation(fig, animate, init_func=init, frames=H,
                        interval=1000*dt, blit=True, repeat=True)

    if save_path:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")

    plt.close()


def main():
    """
    Example usage showing how to plot trajectories in original scale.
    """
    # Paths (adjust these to your setup)
    eval_dataset_path = '/home/sisrel/pjw/mpd-splines-public/data_trajectories/EnvCoblEmpty2D-RobotPointMass2DBig/eval_dataset.pkl'

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_data = load_eval_dataset(eval_dataset_path)

    scale_factor = eval_data['scale_factor']
    print(f"Scale factor: {scale_factor}")
    print(f"Number of scenarios: {eval_data['n_scenarios']}")
    print(f"Horizon: {eval_data['horizon']}")
    print(f"dt: {eval_data['dt']}")

    # Example: Plot first scenario
    idx = 0
    reference_traj = eval_data['ego_trajectories'][idx]  # Already in original scale
    obstacle_trajs = eval_data['obstacle_trajectories'][idx] if eval_data['obstacle_trajectories'] else None

    # For demonstration, create a dummy predicted trajectory
    # In practice, you would load this from your inference results
    print("\nCreating example plot...")
    print("NOTE: In actual use, load predicted_traj from your inference results!")

    # Dummy prediction: reference trajectory with some noise (in normalized scale)
    predicted_traj = reference_traj / scale_factor + np.random.randn(*reference_traj.shape) * 0.05

    start_pos = predicted_traj[0]
    goal_pos = predicted_traj[-1]

    # Plot comparison
    plot_trajectory_comparison(
        predicted_traj=predicted_traj,
        reference_traj=reference_traj,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacle_trajs,
        time=40,  # Mid-point
        scale_factor=scale_factor,
        title=f"Trajectory Comparison - Scenario {idx}",
        save_path=f"/home/sisrel/pjw/mpd-splines-public/data_trajectories/EnvCoblEmpty2D-RobotPointMass2DBig/figures/trajectory_comparison_{idx}.png",
        show_plot=False,
        obstacle_radius=eval_data['obstacle_radius'] * scale_factor,  # Convert back to meters
    )

    print("\nExample completed!")
    print("To use with real inference results:")
    print("1. Load your predicted trajectory from results")
    print("2. Pass it to plot_trajectory_comparison() or animate_trajectory_comparison()")
    print("3. The functions will handle scale conversion automatically")


if __name__ == "__main__":
    main()