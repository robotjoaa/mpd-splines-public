"""
Visualize SDF gradients across the workspace for sampled trajectories during denoising steps.
This script creates vector field visualizations showing how obstacles' gradients influence trajectory optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, DEFAULT_TENSOR_ARGS


def create_workspace_grid(env, resolution=50, tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Create a meshgrid covering the workspace for SDF evaluation.

    Args:
        env: Environment object with limits
        resolution: Number of grid points per dimension
        tensor_args: Torch device/dtype arguments

    Returns:
        X, Y: Meshgrid coordinates
        grid_points: Flattened grid points (N, 2) for 2D or (N, 3) for 3D
    """
    limits = env.limits
    dim = limits.shape[1]

    if dim == 2:
        x = np.linspace(limits[0, 0], limits[1, 0], resolution)
        y = np.linspace(limits[0, 1], limits[1, 1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        return X, Y, to_torch(grid_points, **tensor_args)
    elif dim == 3:
        x = np.linspace(limits[0, 0], limits[1, 0], resolution)
        y = np.linspace(limits[0, 1], limits[1, 1], resolution)
        z = np.linspace(limits[0, 2], limits[1, 2], resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        return X, Y, Z, to_torch(grid_points, **tensor_args)
    else:
        raise NotImplementedError(f"Dimension {dim} not supported")


def compute_sdf_and_gradients(env, grid_points, tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Compute SDF values and gradients across the workspace grid.

    Args:
        env: Environment with distance field objects
        grid_points: Grid coordinates (N, dim)
        tensor_args: Torch device/dtype arguments

    Returns:
        sdf_vals: SDF values at grid points (N,)
        sdf_grads: SDF gradients at grid points (N, dim)
    """
    df_obj_list = env.get_df_obj_list()

    sdf_vals_list = []
    sdf_grads_list = []

    # Compute SDF for each obstacle
    for df_obj in df_obj_list:
        sdf_val, sdf_grad = df_obj.compute_signed_distance(grid_points, get_gradient=True)
        sdf_vals_list.append(sdf_val)
        sdf_grads_list.append(sdf_grad)

    # Take minimum SDF (closest obstacle)
    sdf_vals_stacked = torch.stack(sdf_vals_list, dim=0)  # (num_obstacles, N)
    sdf_vals_min, min_idxs = torch.min(sdf_vals_stacked, dim=0)  # (N,)

    # Get gradient from closest obstacle
    sdf_grads_stacked = torch.stack(sdf_grads_list, dim=0)  # (num_obstacles, N, dim)
    sdf_grads_min = sdf_grads_stacked[min_idxs, torch.arange(len(min_idxs))]  # (N, dim)

    return to_numpy(sdf_vals_min), to_numpy(sdf_grads_min)


def visualize_sdf_gradient_field_2d(
    env,
    trajectory=None,
    control_points=None,
    denoising_step=None,
    save_path=None,
    resolution=30,
    arrow_scale=0.02,
    margin=0.01,
    tensor_args=DEFAULT_TENSOR_ARGS,
    show_obstacles=True,
    title=None,
):
    """
    Visualize SDF gradient vector field in 2D workspace.

    Args:
        env: Environment object
        trajectory: Current trajectory positions (num_T_pts, 2) - optional
        control_points: Control points (n_control_points, 2) - optional
        denoising_step: Current denoising iteration number - optional
        save_path: Path to save figure - optional
        resolution: Grid resolution for vector field
        arrow_scale: Scale factor for gradient arrows
        margin: Collision margin to visualize
        tensor_args: Torch device/dtype arguments
        show_obstacles: Whether to draw obstacle shapes
        title: Custom title - optional
    """
    # Create grid
    X, Y, grid_points = create_workspace_grid(env, resolution=resolution, tensor_args=tensor_args)

    # Compute SDF and gradients
    sdf_vals, sdf_grads = compute_sdf_and_gradients(env, grid_points, tensor_args=tensor_args)

    # Reshape for plotting
    sdf_vals_grid = sdf_vals.reshape(X.shape)
    sdf_grads_x = sdf_grads[:, 0].reshape(X.shape)
    sdf_grads_y = sdf_grads[:, 1].reshape(X.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot SDF contours (distance field)
    contour_levels = np.linspace(-0.5, 0.5, 20)
    contourf = ax.contourf(X, Y, sdf_vals_grid, levels=contour_levels, cmap='RdYlBu', alpha=0.3)
    contour = ax.contour(X, Y, sdf_vals_grid, levels=[0, margin], colors=['black', 'red'],
                         linewidths=[2, 1.5], linestyles=['solid', 'dashed'])
    ax.clabel(contour, inline=True, fontsize=8, fmt={0: 'surface', margin: f'margin={margin}'})

    # Colorbar for SDF values
    cbar = plt.colorbar(contourf, ax=ax, label='SDF value')

    # Plot gradient vector field
    # Only show gradients in the "danger zone" (near obstacles)
    mask = np.abs(sdf_vals_grid) < margin * 5  # Show gradients within 5x margin

    # Subsample for better visualization
    skip = max(1, resolution // 20)
    quiver = ax.quiver(
        X[::skip, ::skip][mask[::skip, ::skip]],
        Y[::skip, ::skip][mask[::skip, ::skip]],
        -sdf_grads_x[::skip, ::skip][mask[::skip, ::skip]],  # Negative: cost gradient direction
        -sdf_grads_y[::skip, ::skip][mask[::skip, ::skip]],
        color='blue',
        alpha=0.6,
        scale=1.0/arrow_scale,
        width=0.003,
        headwidth=3,
        headlength=4,
        label='Cost gradient (pushes away)'
    )

    # Draw obstacles if enabled
    if show_obstacles:
        df_obj_list = env.get_df_obj_list()
        for df_obj in df_obj_list:
            # Handle sphere obstacles
            if hasattr(df_obj, 'centers') and hasattr(df_obj, 'radii'):
                centers = to_numpy(df_obj.centers)
                radii = to_numpy(df_obj.radii)
                for center, radius in zip(centers, radii):
                    circle = Circle(center, radius, color='gray', alpha=0.5, linewidth=2,
                                   edgecolor='black', label='Obstacle')
                    ax.add_patch(circle)
                    # Show margin
                    circle_margin = Circle(center, radius + margin, color='red', alpha=0.2,
                                          linewidth=1, edgecolor='red', linestyle='--')
                    ax.add_patch(circle_margin)
            # Handle box obstacles
            elif hasattr(df_obj, 'centers') and hasattr(df_obj, 'sizes'):
                centers = to_numpy(df_obj.centers)
                sizes = to_numpy(df_obj.sizes)
                for center, size in zip(centers, sizes):
                    half_size = size / 2
                    rect = Rectangle(center - half_size, size[0], size[1],
                                    color='gray', alpha=0.5, linewidth=2, edgecolor='black')
                    ax.add_patch(rect)

    # Plot trajectory if provided
    if trajectory is not None:
        traj_np = to_numpy(trajectory) if torch.is_tensor(trajectory) else trajectory
        ax.plot(traj_np[:, 0], traj_np[:, 1], 'g-', linewidth=2, alpha=0.7, label='Trajectory')
        ax.scatter(traj_np[0, 0], traj_np[0, 1], c='green', s=100, marker='o',
                  label='Start', zorder=10, edgecolor='black')
        ax.scatter(traj_np[-1, 0], traj_np[-1, 1], c='red', s=100, marker='s',
                  label='Goal', zorder=10, edgecolor='black')

    # Plot control points if provided
    if control_points is not None:
        cp_np = to_numpy(control_points) if torch.is_tensor(control_points) else control_points
        ax.scatter(cp_np[:, 0], cp_np[:, 1], c='orange', s=80, marker='x',
                  linewidth=2, label='Control Points', zorder=9)

    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    if title is None:
        if denoising_step is not None:
            title = f'SDF Gradient Field - Denoising Step {denoising_step}'
        else:
            title = 'SDF Gradient Field'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set aspect and limits
    ax.set_aspect('equal')
    limits = env.limits
    ax.set_xlim(limits[0, 0], limits[1, 0])
    ax.set_ylim(limits[0, 1], limits[1, 1])

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SDF gradient visualization to {save_path}")

    return fig, ax


def visualize_denoising_trajectory_with_gradients(
    env,
    trajectories_per_step,
    control_points_per_step=None,
    save_dir="./figures/sdf_gradients",
    resolution=30,
    arrow_scale=0.02,
    margin=0.01,
    tensor_args=DEFAULT_TENSOR_ARGS,
):
    """
    Create a sequence of visualizations showing SDF gradients at each denoising step.

    Args:
        env: Environment object
        trajectories_per_step: List of trajectories, one per denoising step [(num_T_pts, dim), ...]
        control_points_per_step: List of control points, one per step [(n_cp, dim), ...] - optional
        save_dir: Directory to save figures
        resolution: Grid resolution
        arrow_scale: Arrow scale factor
        margin: Collision margin
        tensor_args: Torch device/dtype arguments

    Returns:
        List of figure handles
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    figures = []
    n_steps = len(trajectories_per_step)

    for step_idx, trajectory in enumerate(trajectories_per_step):
        cp = control_points_per_step[step_idx] if control_points_per_step is not None else None

        save_path = os.path.join(save_dir, f"sdf_gradient_step_{step_idx:03d}.png")

        fig, ax = visualize_sdf_gradient_field_2d(
            env=env,
            trajectory=trajectory,
            control_points=cp,
            denoising_step=step_idx,
            save_path=save_path,
            resolution=resolution,
            arrow_scale=arrow_scale,
            margin=margin,
            tensor_args=tensor_args,
        )

        figures.append(fig)
        plt.close(fig)  # Close to save memory

    print(f"Created {n_steps} SDF gradient visualizations in {save_dir}")

    return figures


def create_animation_from_gradients(save_dir, output_path="sdf_gradient_animation.mp4", fps=5):
    """
    Create an animation from saved gradient visualization images.

    Args:
        save_dir: Directory containing saved images
        output_path: Output video path
        fps: Frames per second
    """
    import os
    import glob
    from matplotlib.animation import FFMpegWriter

    image_files = sorted(glob.glob(os.path.join(save_dir, "sdf_gradient_step_*.png")))

    if len(image_files) == 0:
        print(f"No images found in {save_dir}")
        return

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    from PIL import Image
    writer = FFMpegWriter(fps=fps, bitrate=1800)

    with writer.saving(fig, output_path, dpi=150):
        for img_path in image_files:
            img = Image.open(img_path)
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            writer.grab_frame()

    plt.close(fig)
    print(f"Created animation: {output_path}")


# Example usage function
def example_usage_with_planning_task(planning_task, trajectories, control_points=None):
    """
    Example showing how to use this with a planning task during inference.

    Args:
        planning_task: PlanningTask object
        trajectories: List of trajectories from denoising steps
        control_points: List of control points from denoising steps (optional)
    """
    from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS

    # Visualize final trajectory with gradients
    fig, ax = visualize_sdf_gradient_field_2d(
        env=planning_task.env,
        trajectory=trajectories[-1],  # Final denoised trajectory
        control_points=control_points[-1] if control_points else None,
        denoising_step=len(trajectories) - 1,
        save_path="./sdf_gradient_final.png",
        resolution=40,
        arrow_scale=0.015,
        margin=planning_task.min_distance_robot_env,
        tensor_args=planning_task.tensor_args,
    )
    plt.show()

    # Create full animation
    visualize_denoising_trajectory_with_gradients(
        env=planning_task.env,
        trajectories_per_step=trajectories,
        control_points_per_step=control_points,
        save_dir="./figures/sdf_gradients_animation",
        resolution=30,
        margin=planning_task.min_distance_robot_env,
        tensor_args=planning_task.tensor_args,
    )

    # Create video
    create_animation_from_gradients(
        save_dir="./figures/sdf_gradients_animation",
        output_path="./sdf_gradient_denoising.mp4",
        fps=5
    )


if __name__ == "__main__":
    # Example: Load environment and visualize static gradient field
    from torch_robotics import environments
    from torch_robotics.torch_utils.torch_utils import get_torch_device

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Create a simple 2D environment
    env = environments.EnvSimple2D(tensor_args=tensor_args)

    # Visualize the gradient field
    fig, ax = visualize_sdf_gradient_field_2d(
        env=env,
        save_path="./sdf_gradient_example.png",
        resolution=40,
        arrow_scale=0.02,
        margin=0.01,
        tensor_args=tensor_args,
    )
    plt.show()
