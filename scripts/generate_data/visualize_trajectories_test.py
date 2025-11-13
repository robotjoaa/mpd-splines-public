from pathlib import Path

import click

import isaacgym

from pprint import pprint


from dotmap import DotMap
import h5py
import seaborn

from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.paths import DATASET_BASE_DIR
from pb_ompl.pb_ompl import fit_bspline_to_path
from torch_robotics.isaac_gym_envs.motion_planning_envs import (
    MotionPlanningIsaacGymEnv,
    MotionPlanningControllerIsaacGym,
)
from scripts.generate_data.generate_trajectories import GenerateDataOMPL

import matplotlib.pyplot as plt

import os.path

import numpy as np
import torch
import yaml
from scipy import interpolate
import itertools
#from mpd.utils.loaders import load_params_from_yaml
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy, get_torch_device
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, plot_multiline

import torch.nn.functional as F

# @click.argument(
#     "data_dir",
#     type=click.Path(exists=True),
#     default=Path(DATASET_BASE_DIR) / "EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect",
# )
# @click.argument(
#     "data_dir",
#     type=click.Path(exists=True),
#     default=Path(DATASET_BASE_DIR) / "EnvSpheres3D-RobotPanda-joint_joint-one-RRTConnect",
# )


def get_new_control_point(idx_d, pos, vel, acc, h, T) : 
    #print(pos.shape)
    idx_l = idx_d.get("left")
    idx_r = idx_d.get("right")
    # left
    res_dict = {}
    if idx_l : 
        P_0 = pos[:, idx_l, :]
        P_1 = P_0 + h / 5 *  vel[:, idx_l, :]
        P_2 = 2 * P_1 - P_0 + h * h / 20 * acc[:, idx_l, :]    
        res_dict["left"] = torch.stack((P_0, P_1, P_2), axis = 1)
    
    if idx_r : 
        # right 
        P_N = pos[:, idx_r, :]
        P_N_1 = P_N - h / 5 * vel[:, idx_r, :]
        P_N_2 = 2 * P_N_1 - P_N + h * h / 20 * acc[:, idx_r, :]    
        res_dict["right"] = torch.stack((P_N_2, P_N_1, P_N), axis = 1)

    return res_dict

def _h_from_m_p(m, p):
    # number of interior spans = m - 2p, each of size h
    return 1.0 / (m - 2*p)

def _span_uniform(u, p, m, tol=1e-14):
    """
    Return span k such that U[k] <= u < U[k+1] for a uniform clamped knot vector.
    For u == 1, return the last span.
    """
    n = m - p - 1
    if u <= 0.0 + tol:
        return p
    if u >= 1.0 - tol:
        return n
    h = _h_from_m_p(m, p)
    j = int(np.floor(u / h))              # interior span index (0..(m-2p-1))
    j = min(max(j, 0), (m - 2*p - 1))
    return p + j                          # convert to global span index

def _knot_value_uniform(i, p, m):
    """
    U[i] for clamped uniform:
      0 for i <= p,
      1 for i >= m - p,
      (i - p) * h otherwise
    """
    if i <= p:          return 0.0
    if i >= m - p:      return 1.0
    h = _h_from_m_p(m, p)
    return (i - p) * h

def _multiplicity_uniform(u, p, m, tol=1e-14):
    """
    For uniform clamped:
      mult(0) = mult(1) = p+1
      interior knots each have multiplicity 1
      non-knot parameters have multiplicity 0
    """
    if abs(u - 0.0) < tol or abs(u - 1.0) < tol:
        return p + 1
    h = _h_from_m_p(m, p)
    t = u / h
    if abs(t - round(t)) < 1e-12:  # u is exactly on an interior grid line
        # Make sure it's not the trivial 0 or 1 we already caught
        if 0 < round(t) < (m - 2*p):
            return 1
    return 0

def insert_knot_once_uniform(U, P, p, u):
    """
    Insert `u` once into a clamped, uniform B-spline (degree p).
    U must be clamped-uniform-compatible, but we don't rely on U's values except
    for building the new vector at the end. Geometry is preserved exactly.
    """
    P = np.asarray(P)
    U = np.asarray(U, dtype=float)
    m_old = len(U) - 1
    n_old = P.shape[0] - 1
    assert m_old == p + n_old + 1, "U/P size mismatch with degree p"

    s = _multiplicity_uniform(u, p, m_old)
    if s > p:
        # already fully clamped at u; no-op
        return U.copy(), P.copy()

    # Determine affected span (global)
    k = _span_uniform(u, p, m_old)
    # New sizes
    m_new = m_old + 1
    n_new = n_old + 1

    # Allocate outputs
    U_new = np.empty(m_new + 1, dtype=float)
    if P.ndim == 1:
        Q_new = np.empty(n_new + 1, dtype=P.dtype)
    else:
        Q_new = np.empty((n_new + 1,) + P.shape[1:], dtype=P.dtype)

    # Build U_new by inserting u in sorted order (just copy & insert at k+1).
    U_new[:k+1] = U[:k+1]
    U_new[k+1]  = u
    U_new[k+2:] = U[k+1:]

    # Copy unaffected control points
    left_end  = k - p                # indices 0..left_end unchanged
    right_beg = k - s + 1            # right block (shifted by +1)
    Q_new[:left_end+1] = P[:left_end+1]
    Q_new[right_beg+1:] = P[right_beg:]

    # Recompute affected band: i = k - p + 1 .. k - s
    # For uniform clamped, U[i+p] - U[i] = p*h (except at the extreme ends,
    # but this formula still holds because the ends are clamped to 0/1).
    h = _h_from_m_p(m_old, p)
    denom = p * h if p > 0 else 1.0

    for i in range(left_end + 1, k - s + 1):
        Ui   = _knot_value_uniform(i,   p, m_old)
        # Handle the (rare) zero-denominator case defensively
        alpha = 0.0 if abs(denom) < 1e-16 else (u - Ui) / denom
        alpha = np.clip(alpha, 0.0, 1.0)    # numerically safe
        Q_new[i] = alpha * P[i] + (1.0 - alpha) * P[i - 1]

    return U_new, Q_new

def insert_knot_uniform(U, P, p, u, r=1):
    """Insert knot `u` r times for clamped-uniform, preserving the curve exactly."""
    Uo = np.asarray(U, dtype=float)
    Po = np.asarray(P).copy()
    for _ in range(r):
        Uo, Po = insert_knot_once_uniform(Uo, Po, p, u)
    return Uo, Po

# p : degree, [a,b] : range to extract
def extract_subcurve(P, U, p, a, b):
    # 1) clamp at a and b
    U1, P1 = insert_knot_uniform(U, P, p, a)
    U2, P2 = insert_knot_uniform(U1, P1, p, b)
    # U2, P2 = refine_to_clamp(U1, P1, p, b)

    # 2) locate fully clamped blocks
    # ia: last index with U2[ia] == a
    ia = max(i for i, Ui in enumerate(U2) if abs(Ui - a) < 1e-14)
    # ib: first index with U2[ib] == b
    ib = min(i for i, Ui in enumerate(U2) if abs(Ui - b) < 1e-14)

    # 3) slice control points and knots
    P_sub = P2[ia - p : ib]           # note: stop index is exclusive, gives (ib - (ia - p)) points
    U_sub = U2[ia - p : ib + p + 1]   # knot vector for the sub-curve

    # 4) re-normalize knots to [0,1]
    U0, U1v = a, b
    U_sub_norm = np.array([(Ui - U0) / (U1v - U0) for Ui in U_sub])
    return P_sub, U_sub_norm

def find_closest(find_t, value) : 
    assert find_t.ndim == 2 
    tmp = torch.square(find_t - value)
    ind = torch.where(torch.all(tmp < 1e-10, dim = 0))
    return ind

@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True),
    default=Path(DATASET_BASE_DIR) / "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
)
def visualize(data_dir):
    os.makedirs(os.path.join(data_dir, "figures"), exist_ok=True)

    fix_random_seed(2)

    isaac_gym_render_all_trajectories = False

    # tensor_args = DEFAULT_TENSOR_ARGS
    # tensor_args["device"] = "cpu"
    device = "cuda:0"
    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # -------------------------------- Load trajectories -------------------------
    n_tasks_display = 4 # plot this number of tasks
    n_trajs_display = 10  # plot this number of trajectories per task

    # get the args
    args = load_params_from_yaml(os.path.join(data_dir, "args.yaml"))
    print(f"\n-------------- ARGS --------------")
    print(yaml.dump(args))

    # get the merged dataset file
    dataset_h5 = h5py.File(os.path.join(data_dir, "dataset_merged_doubled.hdf5"), "r")

    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    env_class = getattr(environments, args["env_id"])
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, args["robot_id"])
    robot = robot_class(tensor_args=tensor_args, gripper=True)

    # # Task
    parametric_trajectory = ParametricTrajectoryBspline(
        n_control_points=30,
        degree=5,
        num_T_pts=128,
        zero_vel_at_start_and_goal=True,
        zero_acc_at_start_and_goal=True,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        trajectory_duration=5.0,
        tensor_args=tensor_args,
        phase_time_class="PhaseTimeLinear",
    )

    planning_task = PlanningTask(
        parametric_trajectory=parametric_trajectory,
        env=env,
        robot=robot,
        obstacle_cutoff_margin=args["min_distance_robot_env"],
        tensor_args=tensor_args,
    )

    # -------------------------------- Fit the b-spline ---------------------------------
    q_pos_trajs = []
    q_control_points = []

    
    # results_dir: str = "logs"
    # cfg_inference_path: str = '/home/sisrel/pjw/mpd-splines-public/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml'
    # args_inference = DotMap(load_params_from_yaml(cfg_inference_path))
    # if args_inference.model_selection == "bspline":
    #     args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    # elif args_inference.model_selection == "waypoints":
    #     args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
    # args_inference.model_dir = os.path.expandvars(args_inference.model_dir)
    
    # args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))

    # args_train.update(
    #     **args_inference,
    #     gripper=True,
    #     reload_data=False,
    #     results_dir=results_dir,
    #     load_indices=True,
    #     tensor_args=tensor_args,
    # )
    # planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    # # if selection_start_goal == "training":
    # #    idx_sample_l = np.random.choice(np.arange(len(train_subset)), n_start_goal_states)
    # # else:
    # idx_sample_l = np.random.choice(np.arange(len(val_subset)), 100)

    # for idx_sg, idx_sample in enumerate(idx_sample_l):
    #     task_ids_selected = np.array([idx_sample_l])
    
    # task_ids_selected = idx_sample_l[:2]
    
    # print("task ids selected ",task_ids_selected)
    # raise NotImplementedError
    # task_ids_selected = np.random.choice(np.unique(dataset_h5["task_id"]), n_tasks_display, replace=False)

    task_ids_selected = np.array([836])

    for task_id_selected in task_ids_selected:
        idxs_with_task_id = np.argwhere(dataset_h5["task_id"] == task_id_selected).squeeze(1)
        print(idxs_with_task_id)
        for i in np.random.choice(idxs_with_task_id, min(n_trajs_display, len(idxs_with_task_id))):
            if "bspline_params_cc" in dataset_h5:
                bspline_params = (
                    dataset_h5["bspline_params_tt"][i],
                    dataset_h5["bspline_params_cc"][i],
                    dataset_h5["bspline_params_k"][i],
                )
            else:
                # fit a spline to the path
                bspline_params = fit_bspline_to_path(
                    dataset_h5["sol_path"][i],
                    bspline_degree=planning_task.parametric_trajectory.bspline.d,
                    bspline_num_control_points=planning_task.parametric_trajectory.bspline.n_pts,
                    bspline_zero_vel_at_start_and_goal=planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                    bspline_zero_acc_at_start_and_goal=planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                    debug=True,
                )

            tt, cc, k = bspline_params
            cc_np = np.array(cc)

            bspl = interpolate.BSpline(tt, cc_np.T, k)  # note the transpose
            interpolate_num = 128
            u_interpolation = np.linspace(0, 1, interpolate_num)
            bspline_path_interpolated = bspl(u_interpolation)
            current_traj = to_torch(bspline_path_interpolated, **tensor_args)
            q_pos_trajs.append(current_traj)
            current_cps = to_torch(cc_np.T, **tensor_args)
            q_control_points.append(current_cps)

        tmp_traj = to_numpy(current_traj.unsqueeze(0))
        tmp_cps = to_numpy(current_cps.unsqueeze(0))
        fig_cps, axs_cps = create_fig_and_axes(2, figsize=(6, 6))
        axs_cps.scatter(tmp_cps[0,:,0], tmp_cps[0,:,1], c="blue", marker="o", s=10**2, zorder=100)
        axs_cps.plot(tmp_traj[0,:, 0], tmp_traj[0,:, 1], color="orange", linestyle="solid", linewidth=3, marker="x")
        fig_cps.savefig(os.path.join(data_dir, f"figures/bspline-control-points-{task_id_selected:03d}.png"), bbox_inches="tight")
        # planning_task.animate_robot_trajectories(
        #     q_pos_trajs=tmp_traj,
        #     q_pos_start=tmp_traj[:,0,:],
        #     q_pos_goal=tmp_traj[:,-1,:],
        #     plot_x_trajs=True,
        #     video_filepath=os.path.join(data_dir, f"figures/pointmass2d-robot-env-{task_id_selected:03d}.mp4"),
        #     n_frames=128,
        #     anim_time=parametric_trajectory.trajectory_duration,
        #     filter_joint_limits_vel_acc=True,
        # )


    q_control_points = torch.stack(q_control_points)
    q_pos_trajs = torch.stack(q_pos_trajs)

    # Get the position, velocity and acceleration trajectories from the B-spline, based on the control points
    q_start = q_pos_trajs[:, 0, :]
    q_goal = q_pos_trajs[:, -1, :]
    q_trajs_d = planning_task.parametric_trajectory.get_q_trajectory(
        q_control_points, q_start, q_goal, get_type=("pos", "vel", "acc")
    )
    q_trajs_pos_ref = q_trajs_d["pos"]
    q_trajs_vel_ref = q_trajs_d["vel"]
    q_trajs_acc_ref = q_trajs_d["acc"]

    # # -------------------------------- Visualize ---------------------------------
    # fig, axs = planning_task.plot_joint_space_trajectories(
    #     q_pos_trajs=q_trajs_pos,
    #     q_vel_trajs=q_trajs_vel,
    #     q_acc_trajs=q_trajs_acc,
    #     # control_points=control_points_concat,
    #     set_q_pos_limits=False,
    #     set_q_vel_limits=False,
    #     set_q_acc_limits=False,
    # )
    # fig.savefig(os.path.join(data_dir, f"figures/joint_space_trajectories.png"), bbox_inches="tight")
    n_hrz = 3 
    num_T_pts = 40
    # shorter bspline
    q_trajs_pos_all = []
    q_trajs_vel_all = []
    q_trajs_acc_all = []
    q_cps_all = []
    #subgoal_idx = [0, 39, 78, 128] # 50 , ovlp 11
    subgoal_idx = [0, 43, 85]
    for i_h in range(n_hrz) : 
        u_tmp = parametric_trajectory.bspline.u.copy()
        p_tmp = to_numpy(q_control_points[0])
        print("knot : {}, control point : {}",u_tmp.shape,p_tmp.shape)
        new_q_cps, new_u = extract_subcurve(p_tmp, u_tmp, 5, 1/3 * i_h, 1/3 * i_h + 1/3 )
        print(new_q_cps) # 14

        print(len(new_u))
        new_q_cps = to_torch(new_q_cps,**tensor_args)
        q_start = new_q_cps[0,:]
        q_end = new_q_cps[-1,:]
        # shorter bspline
        partial_traj = ParametricTrajectoryBspline(
            n_control_points=len(new_q_cps),
            degree=5,
            num_T_pts=num_T_pts,
            zero_vel_at_start_and_goal=False,
            zero_acc_at_start_and_goal=False,
            remove_outer_control_points=False,
            keep_last_control_point=False,
            trajectory_duration=5.0 / 3,
            tensor_args=tensor_args,
            phase_time_class="PhaseTimeLinear",
        )

        idx_d = {}
        if i_h == 0 : 
            idx_d['right'] = subgoal_idx[i_h+1] 
        elif i_h == n_hrz -1 : 
            idx_d['left'] = subgoal_idx[i_h]
        else : 
            idx_d['left'] = subgoal_idx[i_h] 
            idx_d['right'] = subgoal_idx[i_h+1]

        spl_dict = {
            'h' : 1.0 / (partial_traj.bspline.m - 2 * partial_traj.bspline.d),
            'T' : partial_traj.bspline.num_T_pts
        }

        replace_pts = get_new_control_point(idx_d, q_trajs_pos_ref, q_trajs_vel_ref, q_trajs_acc_ref,
                                **spl_dict
                            )
        # print("q_start :",q_start,"q_end :", q_end)
        # idx_start = find_closest(q_pos_trajs[0,:,:], q_start)
        # idx_end = find_closest(q_pos_trajs[0,:,:], q_end)
        # print("idx_start :",idx_start, "idx_end : ",idx_end)
        # q_trajs_d = partial_traj.get_q_trajectory(
        #     new_q_cps, q_start, q_end, get_type = ("pos","vel","acc"),
        #     get_time_representation=False
        # )
        if replace_pts.get('left') is not None : 
            new_q_cps[:3] = replace_pts['left']
        
        if replace_pts.get('right') is not None: 
            new_q_cps[-3:] = replace_pts['right']
        
        q_trajs_d = partial_traj.get_q_trajectory_in_phase(new_q_cps, get_type=("pos","vel","acc"))
        # print(partial_traj.phase_time.rs[..., None])
        scale = 0.5850 # 3/5 * 39/40
        q_trajs_pos = q_trajs_d["pos"] 
        q_trajs_vel = q_trajs_d["vel"] * scale
        q_trajs_acc = q_trajs_d["acc"] * (scale * scale)
        q_trajs_pos_all.append(q_trajs_pos.unsqueeze(0))
        q_trajs_vel_all.append(q_trajs_vel.unsqueeze(0))
        q_trajs_acc_all.append(q_trajs_acc.unsqueeze(0))
        q_cps_all.append(new_q_cps.unsqueeze(0))
    
    q_trajs_pos_all = to_numpy(torch.cat(q_trajs_pos_all, axis = 0))
    q_trajs_vel_all = to_numpy(torch.cat(q_trajs_vel_all, axis = 0))
    q_trajs_acc_all = to_numpy(torch.cat(q_trajs_acc_all, axis = 0))
    q_cps_all = to_numpy(torch.cat(q_cps_all, axis = 0))
    #print(q_cps_all.shape) #(3, 14, 2)

    # -------------------------------- Visualize ---------------------------------
    fig, axs = create_fig_and_axes(2, figsize=(6, 6))
    colors = ['red', 'green', 'blue', 'orange']
    for i in range(n_hrz):
        axs.scatter(q_cps_all[i,:,0], q_cps_all[i,:,1], c =colors[i] , marker="o", s=10**2, zorder=100)
        axs.plot(q_trajs_pos_all[i,:, 0], q_trajs_pos_all[i,:, 1], c = colors[i], linestyle="solid", linewidth=3, marker="x")
        
    axs.set_aspect('equal')
    fig.savefig(os.path.join(data_dir, f"figures/bspline-combined-{task_id_selected:03d}.png"), bbox_inches="tight")
    
    fig, axs = planning_task.plot_joint_space_trajectories(
        q_pos_trajs=q_trajs_pos_ref,
        q_vel_trajs=q_trajs_vel_ref,
        q_acc_trajs=q_trajs_acc_ref,
        # control_points=control_points_concat,
        set_q_pos_limits=False,
        set_q_vel_limits=False,
        set_q_acc_limits=False,
    )
    #fig, axs = plt.subplots(planning_task.robot.q_dim, 3, squeeze=False, figsize=(18, 2.5 * planning_task.robot.q_dim))

    #axs[0, 0].set_title("Position")
    #axs[0, 1].set_title("Velocity")
    #axs[0, 2].set_title("Acceleration")
    #axs[-1, 1].set_xlabel("Time [s]")

    set_q_pos_limits = True
    set_q_vel_limits = True
    set_q_acc_limits = True
    subgoal_idx = [0, 42, 85, 128]
    timesteps = to_numpy(parametric_trajectory.get_timesteps().reshape(1, -1))
    t_start, t_goal = timesteps[0, 0], timesteps[0, -1]
    dt = (t_goal - t_start) / 3 / (num_T_pts-1)
    # Positions, velocities, accelerations
    q_trajs_filtered = (q_trajs_pos_all, q_trajs_vel_all, q_trajs_acc_all)

    for i, ax in enumerate(axs):
        for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
            if q_trajs_filtered_item is not None:
                for i_hz in range(n_hrz) : 
                    tmp_timesteps = 5/3 * i_hz + dt*np.arange(num_T_pts)
                    ax[j].plot(tmp_timesteps, q_trajs_filtered_item[i_hz,:,i], c = colors[i_hz], linestyle="solid")
                    print(tmp_timesteps.shape, q_trajs_filtered_item[i_hz,:].shape)

        # if q_pos_traj_best is not None:
        #     q_pos_traj_best_np = to_numpy(q_pos_traj_best)
        #     plot_multiline(ax[0], timesteps, q_pos_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
        # if q_vel_traj_best is not None:
        #     q_vel_traj_best_np = to_numpy(q_vel_traj_best)
        #     plot_multiline(ax[1], timesteps, q_vel_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
        # if q_acc_traj_best is not None:
        #     q_acc_traj_best_np = to_numpy(q_acc_traj_best)
        #     plot_multiline(ax[2], timesteps, q_acc_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)

        # # Start and goal
        # for j, x in enumerate([q_pos_start, q_vel_start, q_acc_start]):
        #     if x is not None:
        #         ax[j].scatter(t_start, x[i], color="green")

        # for j, x in enumerate([q_pos_goal, q_vel_goal, q_acc_goal]):
        #     if x is not None:
        #         ax[j].scatter(t_goal, x[i], color="purple")

        ax[0].set_ylabel(f"$q_{i}$")

        # if set_q_pos_limits:
        #     q_pos_min, q_pos_max = planning_task.robot.q_pos_min_np[i], planning_task.robot.q_pos_max_np[i]
        #     padding = 0.1 * np.abs(q_pos_max - q_pos_min)
        #     ax[0].set_ylim(q_pos_min - padding, q_pos_max + padding)
        #     ax[0].plot([t_start, t_goal], [q_pos_max, q_pos_max], color="k", linestyle="--")
        #     ax[0].plot([t_start, t_goal], [q_pos_min, q_pos_min], color="k", linestyle="--")
        # if set_q_vel_limits and planning_task.robot.dq_max_np is not None:
        #     ax[1].plot(
        #         [t_start, t_goal], [planning_task.robot.dq_max_np[i], planning_task.robot.dq_max_np[i]], color="k", linestyle="--"
        #     )
        #     ax[1].plot(
        #         [t_start, t_goal], [-planning_task.robot.dq_max_np[i], -planning_task.robot.dq_max_np[i]], color="k", linestyle="--"
        #     )
        # if set_q_acc_limits and planning_task.robot.ddq_max_np is not None:
        #     ax[2].plot(
        #         [t_start, t_goal], [planning_task.robot.ddq_max_np[i], planning_task.robot.ddq_max_np[i]], color="k", linestyle="--"
        #     )
        #     ax[2].plot(
        #         [t_start, t_goal], [-planning_task.robot.ddq_max_np[i], -planning_task.robot.ddq_max_np[i]], color="k", linestyle="--"
        #     )

    # time limits
    t_eps = 0.1
    for ax in list(itertools.chain(*axs)):
        ax.set_xlim(t_start - t_eps, t_goal + t_eps)

    # plot control points
    # if control_points is not None:
    #     control_points_np = to_numpy(control_points)
    #     control_points_timesteps = to_numpy(planning_task.parametric_trajectory.get_phase_steps())
    #     for control_points_np_one in control_points_np:
    #         for i, ax in enumerate(axs):
    #             ax[0].scatter(control_points_timesteps, control_points_np_one[:, i], color="red", s=2**2, zorder=10)
    
    fig.savefig(os.path.join(data_dir, f"figures/joint_space_trajectories_combined.png"), bbox_inches="tight")


    # plot histogram of costs
    # costs_position = q_trajs_pos.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_position), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_position), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs position histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_position.png'), bbox_inches='tight')
    #
    # costs_velocity = q_trajs_vel.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_velocity), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_velocity), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs velocity histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_velocity.png'), bbox_inches='tight')
    #
    # costs_acceleration = q_trajs_acc.pow(2).sum(dim=-1).sum(dim=-1)
    # fig, axs = plt.subplots(1, 1, squeeze=False)
    # axs[0, 0].hist(to_numpy(costs_acceleration), bins=10)
    # seaborn.kdeplot(data=to_numpy(costs_acceleration), ax=axs[0, 0].twinx())
    # axs[0, 0].set_title('Costs acceleration histogram')
    # fig.savefig(os.path.join(DATA_DIR, f'figures/costs_acceleration.png'), bbox_inches='tight')

    ########################
    # Visualize in Pybullet
    # print("========= visualize in pybullet =========")
    # generate_data = GenerateDataOMPL(
    #     args["env_id"],
    #     args["robot_id"],
    #     min_distance_robot_env=args["min_distance_robot_env"],
    #     pybullet_mode="GUI", #"DIRECT"
    #     tensor_args=tensor_args,
    #     debug=True,
    # )

    # path = to_numpy(q_trajs_pos[0])  # select only the first trajectory (pybullet does not allow parallelization)
    # generate_data.pbompl_interface.execute(path, sleep_time=5.0 / len(path))
    # print("========= visualize in pybullet =========")
    ########################
    # Visualize in Isaac Gym
    # POSITION CONTROL
    # add initial positions for better visualization
    # n_pre_steps = 10
    # n_post_steps = 10

    # if isaac_gym_render_all_trajectories:
    #     assert q_trajs_pos.shape[1] == 1
    #     q_pos_trajs_isaac = q_trajs_pos.squeeze()
    # else:
    #     q_pos_trajs_isaac = q_trajs_pos

    # q_pos_trajs_isaac = q_pos_trajs_isaac.movedim(1, 0)

    # motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
    #     env,
    #     robot,
    #     asset_root=get_robot_path().as_posix(),
    #     robot_asset_file=robot.robot_urdf_file.replace(get_robot_path().as_posix() + "/", ""),
    #     num_envs=q_pos_trajs_isaac.shape[1],
    #     all_robots_in_one_env=True,
    #     show_viewer=True,
    #     sync_viewer_with_real_time=False,
    #     viewer_time_between_steps=parametric_trajectory.phase_time.trajectory_duration / q_pos_trajs_isaac.shape[0],
    #     render_camera_global=True,
    #     render_camera_global_append_to_recorder=True,
    #     color_robots=False,
    #     # draw_goal_configuration=True if not args['sample_joint_position_goals_with_same_ee_pose'] else False,
    #     draw_goal_configuration=False,
    #     draw_collision_spheres=False,
    #     draw_contact_forces=False,
    #     draw_end_effector_frame=False,
    #     draw_end_effector_path=True,
    #     draw_ee_pose_goal=None,
    #     camera_global_from_top=True if env.dim == 2 else False,
    #     # add_ground_plane=False if env.dim == 2 else True,
    #     add_ground_plane=False,
    # )

    # motion_planning_controller = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)
    # isaac_statistics = motion_planning_controller.execute_trajectories(
    #     q_pos_trajs_isaac,
    #     q_pos_starts=q_pos_trajs_isaac[0],
    #     q_pos_goal=q_pos_trajs_isaac[-1][0],
    #     n_pre_steps=n_pre_steps,
    #     n_post_steps=n_post_steps,
    #     make_video=True,
    #     video_path=os.path.join(data_dir, f"figures/isaac-planning.mp4"),
    #     make_gif=False,
    #     save_step_images=True,
    # )

    # print("-----------------")
    # print(f"isaac_statistics:")
    # pprint(isaac_statistics)
    # print("-----------------")


if __name__ == "__main__":
    visualize()
