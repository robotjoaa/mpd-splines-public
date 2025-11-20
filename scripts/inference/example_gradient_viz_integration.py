"""
Simple example showing how to integrate SDF gradient visualization into existing inference code.

Usage:
    1. Import the visualization function
    2. Call it during/after your denoising loop
    3. Collect trajectories and visualize
"""

import torch
import matplotlib.pyplot as plt
from visualize_sdf_gradients import (
    visualize_sdf_gradient_field_2d,
    visualize_denoising_trajectory_with_gradients,
)


def example_integration_with_diffusion_sampling():
    """
    Example showing where to add gradient visualization in your denoising loop.
    """
    # ... your existing code to setup planning_task, model, etc. ...

    # Example: During the denoising loop
    trajectories_history = []
    control_points_history = []

    # Pseudo-code for your denoising loop:
    # for t in reversed(range(n_diffusion_steps)):
    #     # Your denoising step
    #     x_t = diffusion_model.p_sample(x_t, t, ...)
    #
    #     # Get trajectory from control points
    #     q_control_points = extract_control_points(x_t)
    #     q_trajectory = planning_task.parametric_trajectory.get_position(
    #         q_control_points, q_pos_start, q_pos_goal
    #     )
    #
    #     # Store for visualization
    #     if t % visualization_interval == 0:  # Save every N steps
    #         trajectories_history.append(q_trajectory[0].detach().cpu())
    #         control_points_history.append(q_control_points[0].detach().cpu())

    # After denoising completes, visualize the progression
    # visualize_denoising_trajectory_with_gradients(
    #     env=planning_task.env,
    #     trajectories_per_step=trajectories_history,
    #     control_points_per_step=control_points_history,
    #     save_dir="./gradient_viz_output",
    #     margin=planning_task.min_distance_robot_env,
    #     tensor_args=planning_task.tensor_args,
    # )

    pass


def example_simple_visualization(planning_task, q_control_points, q_pos_start, q_pos_goal):
    """
    Minimal example: Visualize gradients for a single trajectory.

    Args:
        planning_task: Your PlanningTask instance
        q_control_points: Control points tensor (batch, n_control_points, state_dim)
        q_pos_start: Start configuration
        q_pos_goal: Goal configuration
    """
    # Get trajectory from control points
    q_trajectory = planning_task.parametric_trajectory.get_position(
        q_control_points,
        q_pos_start.unsqueeze(0) if q_pos_start.ndim == 1 else q_pos_start,
        q_pos_goal.unsqueeze(0) if q_pos_goal.ndim == 1 else q_pos_goal,
    )

    # Visualize
    fig, ax = visualize_sdf_gradient_field_2d(
        env=planning_task.env,
        trajectory=q_trajectory[0],  # Take first batch element
        control_points=q_control_points[0],
        save_path="./sdf_gradient_viz.png",
        resolution=40,  # Adjust for speed vs quality
        arrow_scale=0.02,  # Adjust arrow size
        margin=planning_task.min_distance_robot_env,
        tensor_args=planning_task.tensor_args,
    )

    plt.show()
    return fig, ax


def example_hook_into_guide_gradient_steps():
    """
    Example showing how to hook into the guide_gradient_steps function
    to visualize gradients during optimization.
    """
    # You can modify mpd/models/diffusion_models/sample_functions.py
    # to add a callback in the guide_gradient_steps function:

    # Original code in guide_gradient_steps:
    # for _ in range(n_guide_steps):
    #     x.requires_grad_(True)
    #     loss = guide(x, t, ...)
    #     grad = torch.autograd.grad(loss.sum(), x)[0]
    #     x = x - scale * grad
    #     x.requires_grad_(False)

    # Modified with callback:
    # for step_idx in range(n_guide_steps):
    #     x.requires_grad_(True)
    #     loss = guide(x, t, ...)
    #     grad = torch.autograd.grad(loss.sum(), x)[0]
    #     x = x - scale * grad
    #     x.requires_grad_(False)
    #
    #     # Call visualization callback
    #     if callback is not None:
    #         callback(x, grad, step_idx, t)

    pass


# ============================================================================
# PRACTICAL INTEGRATION EXAMPLE
# ============================================================================

def add_to_existing_inference_script():
    """
    Here's how to modify your existing inference.py script.
    """

    modification_instructions = """
    # 1. At the top of your inference.py, add:
    from visualize_sdf_gradients import visualize_sdf_gradient_field_2d

    # 2. After getting your final optimized trajectory, add:

    # Inside your inference loop, after optimization:
    q_control_points_best = result["q_control_points_best"]
    q_trajectory_best = planning_task.parametric_trajectory.get_position(
        q_control_points_best,
        q_pos_start.unsqueeze(0),
        q_pos_goal.unsqueeze(0),
    )

    # Visualize SDF gradients for the final trajectory
    fig, ax = visualize_sdf_gradient_field_2d(
        env=planning_task.env,
        trajectory=q_trajectory_best[0],
        control_points=q_control_points_best[0],
        save_path=os.path.join(results_dir, f"sdf_gradient_task_{task_idx}.png"),
        resolution=30,
        margin=planning_task.min_distance_robot_env,
        tensor_args=tensor_args,
    )
    plt.close(fig)

    # 3. If you want to visualize during denoising, modify the p_sample_loop method:

    # In your model's p_sample_loop (or wherever you iterate through denoising):
    trajectories_history = []

    for i in tqdm(reversed(range(0, self.n_diffusion_steps)), desc="sampling"):
        # Your existing denoising code
        x = self.p_sample(x, i, ...)

        # Every K steps, save trajectory for visualization
        if i % 10 == 0 or i == 0:  # Save every 10 steps + final
            q_cp = extract_control_points_from_x(x)  # Your extraction logic
            q_traj = planning_task.parametric_trajectory.get_position(q_cp, ...)
            trajectories_history.append(q_traj[0].detach().cpu())

    # After loop, create animation
    from visualize_sdf_gradients import visualize_denoising_trajectory_with_gradients
    visualize_denoising_trajectory_with_gradients(
        env=planning_task.env,
        trajectories_per_step=trajectories_history,
        save_dir=os.path.join(results_dir, "gradient_animation"),
        margin=planning_task.min_distance_robot_env,
        tensor_args=tensor_args,
    )
    """

    print(modification_instructions)


if __name__ == "__main__":
    print("="*80)
    print("SDF Gradient Visualization Integration Guide")
    print("="*80)
    add_to_existing_inference_script()
    print("\nFor complete examples, see:")
    print("  - visualize_sdf_gradients.py (visualization functions)")
    print("  - inference_with_gradient_viz.py (full integration)")
    print("="*80)
