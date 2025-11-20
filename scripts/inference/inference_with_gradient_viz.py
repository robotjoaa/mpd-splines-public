"""
Inference script with SDF gradient visualization at each denoising step.

This script extends the standard inference to capture and visualize how
SDF gradients guide trajectory optimization during the denoising process.
"""

from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()

import time
from functools import partial
import os
import torch
from einops._torch_specific import allow_ops_in_compiled_graph

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner, render_results
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy
from dotmap import DotMap

# Import our visualization module
from visualize_sdf_gradients import (
    visualize_sdf_gradient_field_2d,
    visualize_denoising_trajectory_with_gradients,
    create_animation_from_gradients,
)

allow_ops_in_compiled_graph()


class GenerativeOptimizerWithGradientViz(GenerativeOptimizationPlanner):
    """
    Extended planner that captures trajectories and gradients at each denoising step.
    """

    def __init__(self, *args, enable_gradient_viz=True, gradient_viz_resolution=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_gradient_viz = enable_gradient_viz
        self.gradient_viz_resolution = gradient_viz_resolution

        # Storage for visualization
        self.trajectories_per_step = []
        self.control_points_per_step = []
        self.costs_per_step = []
        self.gradient_norms_per_step = []

    def run_sampling_and_optim_one_start_goal_pair(self, *args, **kwargs):
        """
        Override to capture intermediate results during denoising.
        """
        # Clear previous run data
        self.trajectories_per_step = []
        self.control_points_per_step = []
        self.costs_per_step = []
        self.gradient_norms_per_step = []

        # Monkey-patch the diffusion model's sample function to capture intermediate states
        if self.enable_gradient_viz:
            self._setup_gradient_capture_hooks()

        # Run the original sampling
        result = super().run_sampling_and_optim_one_start_goal_pair(*args, **kwargs)

        return result

    def _setup_gradient_capture_hooks(self):
        """
        Setup hooks to capture intermediate denoising states.

        Note: This requires modifying the sampling loop.
        For a cleaner implementation, you may want to modify the base class's
        p_sample_loop method to accept a callback function.
        """
        # This is a placeholder - the actual implementation depends on
        # how your diffusion model is structured
        # You would need to add callbacks in the denoising loop
        pass

    def visualize_gradients_for_trajectory(
        self,
        q_control_points,
        q_pos_start,
        q_pos_goal,
        step_idx=None,
        save_dir=None,
    ):
        """
        Visualize SDF gradients for a given trajectory configuration.

        Args:
            q_control_points: Control points (batch, n_control_points, state_dim)
            q_pos_start: Start position
            q_pos_goal: Goal position
            step_idx: Denoising step index
            save_dir: Directory to save visualization
        """
        # Get trajectory from control points
        q_pos = self.planning_task.parametric_trajectory.get_position(
            q_control_points,
            q_pos_start.unsqueeze(0),
            q_pos_goal.unsqueeze(0),
        )

        # Convert to numpy for visualization
        trajectory_np = to_numpy(q_pos[0])  # Take first batch
        control_points_np = to_numpy(q_control_points[0])

        # Create save path
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"gradient_step_{step_idx:03d}.png")
        else:
            save_path = None

        # Visualize
        fig, ax = visualize_sdf_gradient_field_2d(
            env=self.planning_task.env,
            trajectory=trajectory_np,
            control_points=control_points_np,
            denoising_step=step_idx,
            save_path=save_path,
            resolution=self.gradient_viz_resolution,
            margin=self.planning_task.min_distance_robot_env,
            tensor_args=self.tensor_args,
        )

        return fig, ax


@single_experiment_yaml
def experiment(
    ########################################################################
    # Configuration
    cfg_inference_path: str = './cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml',
    ########################################################################
    selection_start_goal: str = "validation",
    n_start_goal_states: int = 1,
    ########################################################################
    # Gradient visualization options
    enable_gradient_viz: bool = True,
    gradient_viz_resolution: int = 30,
    gradient_viz_save_dir: str = "figures/sdf_gradients",
    create_gradient_animation: bool = True,
    ########################################################################
    # Standard visualization options
    render_joint_space_time_iters: bool = False,
    render_joint_space_env_iters: bool = False,
    render_env_robot_opt_iters: bool = False,
    render_env_robot_trajectories: bool = True,
    render_pybullet: bool = False,
    ########################################################################
    device: str = "cuda:0",
    debug: bool = False,
    seed: int = 2,
    results_dir: str = "logs",
    ########################################################################
    **kwargs,
):
    """
    Run inference with SDF gradient visualization.
    """
    # Set random seed
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # Load configuration
    args_inference = DotMap(load_params_from_yaml(cfg_inference_path))

    # Setup model directory
    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_cvae_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_cvae_waypoints
    else:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_ddpm_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_ddpm_waypoints

    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)
    save_to_yaml(args_inference.toDict(), os.path.join(results_dir, "args_inference.yaml"))

    print(f"\n{'='*100}")
    print(f"Running inference with SDF gradient visualization")
    print(f"Model: {args_inference.model_dir}")
    print(f"Gradient viz enabled: {enable_gradient_viz}")
    print(f"{'='*100}\n")

    ################################################################################################################
    # Load dataset and planning task
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=results_dir,
        load_indices=True,
        tensor_args=tensor_args,
    )
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    ################################################################################################################
    # Create evaluation samples generator
    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
        render_pybullet=render_pybullet,
        **args_inference,
    )

    ################################################################################################################
    # Create gradient-visualizing planner
    generative_planner = GenerativeOptimizerWithGradientViz(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        sampling_based_planner_fn=partial(
            evaluation_samples_generator.generate_data_ompl_worker.run,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        ),
        enable_gradient_viz=enable_gradient_viz,
        gradient_viz_resolution=gradient_viz_resolution,
    )

    ################################################################################################################
    # Run inference on selected start-goal pairs
    gradient_viz_dir = os.path.join(results_dir, gradient_viz_save_dir)

    for idx in range(n_start_goal_states):
        print(f"\n{'='*100}")
        print(f"Processing start-goal pair {idx + 1}/{n_start_goal_states}")
        print(f"{'='*100}\n")

        # Get start and goal
        start_goal_batch = evaluation_samples_generator.generate_samples(
            n_samples=1,
            idx_start_goal=idx,
        )

        q_pos_start = start_goal_batch["q_pos_start"][0]
        q_pos_goal = start_goal_batch["q_pos_goal"][0]

        # Run sampling and optimization
        result = generative_planner.run_sampling_and_optim_one_start_goal_pair(
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            **args_inference,
        )

        # Extract results
        q_control_points_best = result["q_control_points_best"]
        trajectories_opt = result.get("trajectories_opt", [])

        ################################################################################################################
        # Visualize gradients for optimization iterations
        if enable_gradient_viz and len(trajectories_opt) > 0:
            task_viz_dir = os.path.join(gradient_viz_dir, f"task_{idx:03d}")
            os.makedirs(task_viz_dir, exist_ok=True)

            print(f"\nCreating SDF gradient visualizations...")

            # Visualize each optimization iteration
            for iter_idx, traj_dict in enumerate(trajectories_opt):
                q_cp = traj_dict.get("q_control_points", q_control_points_best)

                generative_planner.visualize_gradients_for_trajectory(
                    q_control_points=q_cp,
                    q_pos_start=q_pos_start,
                    q_pos_goal=q_pos_goal,
                    step_idx=iter_idx,
                    save_dir=task_viz_dir,
                )

                print(f"  Created visualization for iteration {iter_idx}")

            # Create animation
            if create_gradient_animation:
                print(f"\nCreating animation...")
                animation_path = os.path.join(task_viz_dir, "gradient_animation.mp4")
                create_animation_from_gradients(
                    save_dir=task_viz_dir,
                    output_path=animation_path,
                    fps=5,
                )

        ################################################################################################################
        # Standard rendering
        if any([render_joint_space_time_iters, render_joint_space_env_iters,
                render_env_robot_opt_iters, render_env_robot_trajectories]):
            print(f"\nRendering standard visualizations...")
            render_results(
                planning_task=planning_task,
                result=result,
                q_pos_start=q_pos_start,
                q_pos_goal=q_pos_goal,
                results_dir=os.path.join(results_dir, f"task_{idx:03d}"),
                render_joint_space_time_iters=render_joint_space_time_iters,
                render_joint_space_env_iters=render_joint_space_env_iters,
                render_env_robot_opt_iters=render_env_robot_opt_iters,
                render_env_robot_trajectories=render_env_robot_trajectories,
            )

    print(f"\n{'='*100}")
    print(f"Inference complete! Results saved to: {results_dir}")
    if enable_gradient_viz:
        print(f"SDF gradient visualizations saved to: {gradient_viz_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    run_experiment(experiment)
