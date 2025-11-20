from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()

import time
from functools import partial

import isaacgym


from dotmap import DotMap

import gc
import os
from pprint import pprint

import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner, render_results
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml, get_problem_config
from torch_robotics.isaac_gym_envs.motion_planning_envs import (
    MotionPlanningIsaacGymEnv,
    MotionPlanningControllerIsaacGym,
)
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


@single_experiment_yaml
def experiment(
    ########################################################################
    # Configuration path defining the model and the inference parameters
    # cfg_inference_path: str = './cfgs/config_EnvNarrowPassageDense2D-RobotPointMass2D_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar4Link-RobotPlanar4Link_00.yaml',
    cfg_inference_path: str = './cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvSpheres3D-RobotPanda_00.yaml',
    # cfg_inference_path: str = "./cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml",
    ########################################################################
    # Select the start and goal from the training or validation/test set.
    selection_start_goal: str = "validation",  # training, validation/test
    ########################################################################
    # number of start and goal states to evaluate
    n_start_goal_states: int = 1,
    ########################################################################
    save_results_single_plan_low_mem: bool = False,
    ########################################################################
    # Visualization options
    render_joint_space_time_iters: bool = False,
    render_joint_space_env_iters: bool = False,
    render_env_robot_opt_iters: bool = False,
    render_env_robot_trajectories: bool = True,
    render_pybullet: bool = False,
    draw_collision_spheres: bool = False,
    run_evaluation_issac_gym: bool = False,
    render_isaacgym_viewer: bool = False,
    render_isaacgym_movie: bool = False,
    ########################################################################
    device: str = "cuda:0",  # cpu, cuda
    debug: bool = False,
    ########################################################################
    # MANDATORY
    # seed: int = int(time.time()),
    seed: int = 2,
    results_dir: str = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/test_logs",
    ########################################################################
    # # dummy variables for subfolders
    # dataset_subdir__=dataset_subdir,
    # extra_objects__=extra_objects,
    # planner_alg__=planner_alg,
    # model_selection__=model_selection,
    # phase_time_class__=phase_time_class,
    # diffusion_sampling_method__=diffusion_sampling_method,
    # n_diffusion_steps_without_noise__=n_diffusion_steps_without_noise,
    # project_gradient_hierarchy__=project_gradient_hierarchy,
    # trajectory_duration__=trajectory_duration,
    **kwargs,
):
    # Set random seed for reproducibility
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # Save and load the inference configuration
    args_inference = DotMap(load_params_from_yaml(cfg_inference_path))

    extra_objects = True 
    extraobjects_env = "EnvDynSimple2DExtraObjects" # "EnvSimple2DExtraObjectsV00"
    model_selection = "bspline" # "waypoints"
    trajectory_duration = 4.0
    num_T_pts = 50
    n_trajectory_samples = 100 #100
    moving_object_margin_scale = 2.0
    moving_object_gradient_scale = 10.0

    if extra_objects:
        args_inference.env_id_replace = extraobjects_env
    else:
        args_inference.env_id_replace = None

    args_inference.model_selection = model_selection
    args_inference.phase_time_class = "PhaseTimeLinear"
    args_inference.planner_alg = "mpd"
    args_inference.diffusion_sampling_method = "ddim"
    args_inference.n_diffusion_steps_without_noise = 0
    args_inference.project_gradient_hierarchy = False
    args_inference.trajectory_duration = trajectory_duration
    args_inference.n_trajectory_samples = n_trajectory_samples
    args_inference.num_T_pts = num_T_pts
    args_inference.moving_object_margin_scale = moving_object_margin_scale
    args_inference.moving_object_gradient_scale = moving_object_gradient_scale

    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_cvae_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_cvae_waypoints
        else:
            raise NotImplementedError
    else:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_ddpm_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
        else:
            raise NotImplementedError

    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    save_to_yaml(args_inference.toDict(), os.path.join(results_dir, "args_inference.yaml"))

    print(f"\n-------------------------------------------------------------------------------------------------")
    print(f"cfg_inference_path:\n{cfg_inference_path}")
    print(f"Model:\n{args_inference.model_dir}")
    print(f"--------------------------------------------------------------------------------------------------")

    ################################################################################################################
    # Load dataset, environment, robot and planning task.
    # Override training parameters.
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

    loaded_map = get_problem_config(**args_train)
    ################################################################################################################
    # Generator of evaluation samples
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
    # Load the generative model planner
    generative_optimization_planner = GenerativeOptimizationPlanner(
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
        debug=debug,
    )

    ################################################################################################################
    # IsaacGym environment and motion planning controller
    motion_planning_isaac_env = None
    if run_evaluation_issac_gym:
        robot_asset_file = planning_task.robot.robot_urdf_file
        if draw_collision_spheres:
            robot_asset_file = planning_task.robot.robot_urdf_collision_spheres_file
        motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
            planning_task.env,
            planning_task.robot,
            asset_root=get_robot_path().as_posix(),
            robot_asset_file=robot_asset_file.replace(get_robot_path().as_posix() + "/", ""),
            num_envs=args_inference.n_trajectory_samples,
            # all_robots_in_one_env=True if n_start_goal_states == 1 else False,
            all_robots_in_one_env=True,
            render_isaacgym_viewer=render_isaacgym_viewer,
            render_camera_global=render_isaacgym_movie,
            render_camera_global_append_to_recorder=render_isaacgym_movie,
            sync_viewer_with_real_time=False,
            show_viewer=render_isaacgym_viewer,
            camera_global_from_top=True if planning_task.env.dim == 2 else False,
            add_ground_plane=False,
            viewer_time_between_steps=torch.diff(planning_task.parametric_trajectory.get_timesteps()[:2]).item(),
            draw_goal_configuration=True if not train_subset.dataset.context_ee_goal_pose else False,
            draw_ee_pose_goal=True if train_subset.dataset.context_ee_goal_pose else False,
            color_robots=False,
            draw_contact_forces=False,
            draw_end_effector_frame=False,
            draw_end_effector_path=True,
        )

        motion_planning_controller_isaac_gym = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)

    ################################################################################################################
    # Metrics calculator
    planning_metrics_calculator = PlanningMetricsCalculator(planning_task)

    ################################################################################################################
    # Plan for several start and goal states sequentially
    if selection_start_goal == "training":
        idx_sample_l = np.random.choice(np.arange(len(train_subset)), n_start_goal_states)
    else:
        idx_sample_l = np.random.choice(np.arange(len(val_subset)), n_start_goal_states)

    # reproduce failed index
    # failed_dict = {
    #     "bspline" : [18, 20, 29, 42, 57, 83, 90],
    #     "waypoints" : [18, 20, 24, 29, 42, 68, 87, 90],
    # }
    # failed_dict = {
    #     "bspline" : [18, 20, 29, 42],
    #     "waypoints" : [18, 20, 29, 42],
    # }
    failed_dict = {
        "bspline" : [23],
        "waypoints" : [23],
    }
    failed_obj_dict = {
        23 : {
            'dyn-simple2d-box': {
                'keyframe_positions': np.array([[ 0.9033574, -0.23202  ,  0.       ],
                    [-0.9033574,  0.23202  ,  0.       ]],dtype=np.float32),
                'keyframe_times': np.array([ 0., 10.],dtype=np.float32)},
            'dyn-simple2d-sphere': {
                'keyframe_positions': np.array([[ 0.90552557, -0.22340837,  0.        ],
                    [-0.90552557,  0.22340837,  0.        ]], dtype=np.float32),
                'keyframe_times': np.array([ 0., 10.], dtype=np.float32)}
        }
    }
    # should be index
    failed_time_dict = {
        23 : #np.array([35/128 , 64/128]) * args_inference.trajectory_duration
            np.array([35, 64])
    }
    failed_idx = failed_dict[args_inference.model_selection]
    debug_failed = True
    for idx_sg, sample_id in enumerate(failed_idx):

        print(f"\n-------------------------------------------------------------------------------------------------")
        print(f"----------------PLANNING {idx_sg+1}/{n_start_goal_states}------------------")
        print(f"--------------------------------------------------------------------------------------------------")

        results_single_plan = DotMap(t_generator=0.0, t_guide=0.0)

        q_pos_start = to_torch([-0.7684, -0.9165], **tensor_args)
        q_pos_goal = to_torch([0.4,  -0.25],  **tensor_args) # torch.Tensor([-0.8922,  0.9447])
        ee_pose_goal = to_torch([[ 1.0000,  0.0000,  0.0000, 0.4],
                                [ 0.0000,  1.0000,  0.0000,  -0.25],
                                [ 0.0000,  0.0000,  1.0000,  0.0000]],  **tensor_args) # torch.Tensor([-0.8922,  0.9447])
        # q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx_sg)

        print("\n----------------START AND GOAL states----------------")
        print(f"q_pos_start: {q_pos_start}")
        print(f"q_pos_goal: {q_pos_goal}")
        print(f"ee_pose_goal: {ee_pose_goal}")

        if debug:
            evaluation_samples_generator.add_start_goal_marker(q_pos_start, q_pos_goal)

        ############################################################################################################
        # Run motion planning inference
        print(f"\n----------------PLAN TRAJECTORIES----------------")
        print(f"Starting inference...")

        if loaded_map is not None : 
            map_config = loaded_map[sample_id]

        if debug_failed and sample_id in failed_idx : 
            map_config = failed_obj_dict[sample_id]
        
        results_single_plan = generative_optimization_planner.plan_trajectory(
            q_pos_start, q_pos_goal, ee_pose_goal, results_ns=results_single_plan, debug=debug, 
            debug_failed=debug_failed, map_config = map_config
        )
    
        print(f"...inference finished.")
        # if 'dyn_obj_config' in results_single_plan : 
        #     print(f"dyn_obj_config: {results_single_plan.dyn_obj_config}")

        # reproduce failed index, delayed for obj sampling in plan_trajectory 
        
        ############################################################################################################
        # Show in pybullet the best trajectory
        if render_pybullet and results_single_plan.q_trajs_pos_best is not None:
            time.sleep(3)
            ########################
            # Visualize in Pybullet
            q_pos_path = to_numpy(results_single_plan.q_trajs_pos_best)
            # add panda grippers to the path
            if (
                isinstance(planning_task.robot, RobotPanda)
                and q_pos_path.shape[1] == 7
                and evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.robot.num_dim == 9
            ):
                q_pos_path = np.concatenate((q_pos_path, np.zeros((q_pos_path.shape[0], 2))), axis=-1)
            evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.execute(
                q_pos_path, sleep_time=planning_task.parametric_trajectory.dt
            )

        ############################################################################################################
        # Evaluate and show in IsaacGym
        isaacgym_statistics = None
        if run_evaluation_issac_gym and results_single_plan.q_trajs_pos_valid is not None:
            ########################
            motion_planning_isaac_env.ee_pose_goal = planning_task.robot.get_EE_pose(
                to_torch(q_pos_goal.unsqueeze(0), device), flatten_pos_quat=True, quat_xyzw=True
            ).squeeze(0)

            # Execute all valid trajectories
            if results_single_plan.q_trajs_pos_valid.shape[0] > 0:
                q_trajs_pos = results_single_plan.q_trajs_pos_valid.movedim(1, 0)  # horizon, batch, D
                isaacgym_statistics = motion_planning_controller_isaac_gym.execute_trajectories(
                    q_trajs_pos,
                    q_pos_starts=q_trajs_pos[0],
                    q_pos_goal=q_trajs_pos[-1][0],  # add steps for better visualization
                    n_pre_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
                    n_post_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
                    stop_robot_if_in_contact=False, #Only valid trajectories are given
                    make_video=render_isaacgym_movie,
                    video_duration=args_inference.trajectory_duration,
                    video_path=os.path.join(results_dir, f"isaacgym-{sample_id:03d}.mp4"),
                    make_gif=False,
                )
            results_single_plan.isaacgym_statistics = isaacgym_statistics

        ############################################################################################################
        # Compute motion planning metrics
        print(f"\n----------------METRICS----------------")
        results_single_plan.metrics = planning_metrics_calculator.compute_metrics(results_single_plan)

        print(f"t_inference_total: {results_single_plan.t_inference_total:.3f} sec")
        print(f"t_generator: {results_single_plan.t_generator:.3f} sec")
        print(f"t_guide: {results_single_plan.t_guide:.3f} sec")

        print(f"isaacgym_statistics:")
        pprint(results_single_plan.isaacgym_statistics)

        print(f"metrics:")
        pprint(results_single_plan.metrics)

        # Save data
        results_single_plan_to_save = results_single_plan
        if save_results_single_plan_low_mem:
            results_single_plan_to_save = DotMap(
                t_generator=results_single_plan.t_generator,
                t_guide=results_single_plan.t_guide,
                t_inference_total=results_single_plan.t_inference_total,
                q_pos_start=q_pos_start,
                q_pos_goal=q_pos_goal,
                ee_pose_goal=ee_pose_goal,
                control_points_iters=results_single_plan.control_points_iters,
                metrics=results_single_plan.metrics,
                isaacgym_statistics=results_single_plan.isaacgym_statistics,
            )
        torch.save(
            results_single_plan_to_save,
            os.path.join(results_dir, f"results_single_plan-{sample_id:03d}.pt"),
            _use_new_zipfile_serialization=True,
        )

        ############################################################################################################
        # Render sampling results
        render_time = None 
        if failed_dict is not None : 
            render_time = failed_time_dict[sample_id] if sample_id in failed_idx else None

        render_results(
            args_inference,
            planning_task,
            q_pos_start,
            q_pos_goal,
            results_single_plan,
            sample_id,
            results_dir,
            render_joint_space_time_iters=render_joint_space_time_iters,
            render_joint_space_env_iters=render_joint_space_env_iters,
            render_planning_env_robot_opt_iters=render_env_robot_opt_iters,
            render_planning_env_robot_trajectories=render_env_robot_trajectories, 
            debug=debug,
            render_time = render_time
        )

        ############################################################################################################
        # empty memory
        del results_single_plan
        gc.collect()
        torch.cuda.empty_cache()

    ################################################################################################################
    # clean up
    evaluation_samples_generator.generate_data_ompl_worker.terminate()
    if motion_planning_isaac_env is not None:
        motion_planning_isaac_env.clean_up()
        del motion_planning_isaac_env
        del motion_planning_controller_isaac_gym
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_experiment(experiment)
