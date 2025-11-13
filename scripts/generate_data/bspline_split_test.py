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

def _find_span(U, p, u):
    """Return k such that U[k] <= u < U[k+1]; for u==U[-p-1] return last span."""
    m = len(U) - 1
    n = m - p - 1
    if u <= U[p]:
        return p
    if u >= U[n+1]:
        return n
    # binary search on U[p..n+1]
    low, high = p, n + 1
    while True:
        mid = (low + high) // 2
        if U[mid] <= u < U[mid+1]:
            return mid
        if u < U[mid]:
            high = mid
        else:
            low = mid    # convert to global span index

def _multiplicity_from_vector(U, u, tol=1e-12):
    U = np.asarray(U)
    return int(np.count_nonzero(np.isclose(U, u, atol=tol, rtol=0.0)))

def insert_knot_once(U, P, p, u, tol=1e-14):
    """
    General Boehm/de Boor insertion (r=1). Works for non-uniform vectors and
    repeated insertions at the same u. Preserves the curve exactly.
    """
    U = np.asarray(U, dtype=float)
    P = np.asarray(P)
    m_old = len(U) - 1
    n_old = P.shape[0] - 1
    assert len(U) == len(P) + p + 1, "len(U) must be len(P) + p + 1"

    s = _multiplicity_from_vector(U, u)
    if s > p:
        # already clamped to p+1 at u
        return U.copy(), P.copy()

    k = _find_span(U, p, u)

    # allocate
    U_new = np.empty(m_old + 2, dtype=U.dtype)
    if P.ndim == 1:
        Q_new = np.empty(n_old + 2, dtype=P.dtype)
    else:
        Q_new = np.empty((n_old + 2,) + P.shape[1:], dtype=P.dtype)

    # insert u after U[k]
    U_new[:k+1] = U[:k+1]
    U_new[k+1] = u
    U_new[k+2:] = U[k+1:]

    left_end  = k - p          # unchanged left CPs: 0..left_end
    right_beg = k - s + 1      # unchanged right CPs (shifted by +1): right_beg..n_old

    # copy unchanged blocks
    Q_new[:left_end+1] = P[:left_end+1]
    Q_new[right_beg+1:] = P[right_beg:]

    # recompute affected band: i = k-p+1 .. k-s
    for i in range(left_end + 1, k - s + 1):
        denom = U[i + p] - U[i]
        if abs(denom) < tol:
            alpha = 0.0  # degenerate span (repeated knots)
        else:
            alpha = (u - U[i]) / denom
            # tiny numerical safety
            if alpha < 0.0 and alpha > -1e-12: alpha = 0.0
            if alpha > 1.0 and alpha <  1e-12: alpha = 1.0
        Q_new[i] = alpha * P[i] + (1.0 - alpha) * P[i - 1]

    return U_new, Q_new


def insert_knot(U, P, p, u, r=1):
    """Insert knot u exactly r times using the general routine above."""
    Uo = np.asarray(U, dtype=float)
    Po = np.asarray(P).copy()
    for _ in range(int(r)):
        Uo, Po = insert_knot_once(Uo, Po, p, u)
    return Uo, Po

def refine_to_clamp(U, P, p, u):
    """Clamp parameter u to multiplicity p+1 by repeated insertion."""
    s = _multiplicity_from_vector(U, u)
    r = max(0, (p + 1) - s)
    if r:
        U, P = insert_knot(U, P, p, u, r=r)
    return U, P

# p : degree, [a,b] : range to extract
def extract_subcurve(P, U, p, a, b):
    U1, P1 = refine_to_clamp(U, P, p, a)
    U2, P2 = refine_to_clamp(U1, P1, p, b)
    ia = max(i for i, Ui in enumerate(U2) if abs(Ui - a) < 1e-14)
    ib = min(i for i, Ui in enumerate(U2) if abs(Ui - b) < 1e-14)
    print(P2)
    print(U2)
    P_sub = P2[ia - p : ib]
    U_sub = U2[ia - p : ib + p + 1]
    U_sub = (U_sub - a) / (b - a)
    return P_sub, U_sub

def find_closest(find_t, value) : 
    assert find_t.ndim == 2 
    tmp = torch.square(find_t - value)
    #print("find_closest",tmp)
    ind = torch.where(torch.all(tmp < 1e-10, dim = -1))
    return ind

def get_multiplicity(u, a) :
    # u : knot array 
    s = np.count_nonzero(np.isclose(u, a))        # current multiplicity at a
    return s


def split_curve_scipy(spl, a, b) : 
    # spl : scipy.interpolate.BSpline 
    # a : (0, 1), assume a in grid 
    # len : 16, 8 
    # Insert an interior cut 'a' to multiplicity (k+1)
    # choose split spline to have control point length len 
    #assert b == a + (13 - spl.k)*h
   
    if a > 0 : 
        r_a = max(0, (spl.k + 1) - get_multiplicity(spl.t, a)) # current multiplicity at a
        print(len(spl.t), len(spl.c)) # 36 30
        spl_a = interpolate.insert(a, spl ,m=r_a) 
    else : 
        spl_a = spl
    if b < 1 : 
        r_b = max(0, (spl.k + 1) - get_multiplicity(spl.t, b))
        spl_b = interpolate.insert(b, spl_a, m=r_b)
    else : 
        spl_b = spl_a

    #print(spl_b.t) 
    #print(spl_b.c)
    t2, c2, k = spl_b.t, spl_b.c, spl_b.k

    ia = np.where(np.isclose(t2, a))[0].max() 
    ib = np.where(np.isclose(t2, b))[0].min()
    print(f"knot size : {len(t2)}, control point size : {len(c2)}") # 42 42
    print("knot range :", ia, ib) # len = ib - ia + k

    t_split = t2[ia-k:ib+k+1]
    # assume multiplicity > 1 only occurs from a and b
    c_split = c2[ia-k:ib]
    print("original c: ",spl.c)
    print("modified c: ",c_split)
    spl_split = interpolate.BSpline(t_split, c_split, k)
    return spl_split

    # # Left piece [0,a]
    # i0 = k
    # t_left = t2[i0 - k : ia]      # knots
    # c_left = c2[i0 - k : ia-(k+1)]              # control points
    # t_left = t2[:24]
    # c_left= c2[:18]
    # print(len(t_left), len(c_left)) # 29 23
    # spl_left = interpolate.BSpline(t_left, c_left, k)

    # # Right piece [a,1]
    # # (m = len(t2)-1; first index of trailing 1s is m-k)
    # m = len(t2) - 1
    # i1 = m - k
    # #t_right = t2[ia - k : i1 + k + 1]
    # #c_right = c2[ia - k : i1]
    # t_right = t2[18:]
    # c_right = c2[18:len(c2) - 6]
    # print(len(t_right), len(c_right)) # 24 18
    # spl_right = interpolate.BSpline(t_right, c_right, k)

    # no normalization
    #return [spl_left, spl_right]



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

    task_ids_selected = np.array([836])
    bspl_l = []
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
            bspl_l.append(bspl)
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
    n_hrz = 2 
    num_T_pts = 64
    # shorter bspline
    q_trajs_pos_all = []
    q_trajs_vel_all = []
    q_trajs_acc_all = []
    q_cps_all = []
    full_N = parametric_trajectory.bspline.N[0]
    full_dN = parametric_trajectory.bspline.dN[0]
    full_ddN = parametric_trajectory.bspline.ddN[0]
    full_rs = parametric_trajectory.phase_time.rs[0]
    print(full_N[num_T_pts,:])
    print(full_dN[num_T_pts,:])
    print(full_ddN[num_T_pts,:])
    #print(full_rs[0])
    idx = find_closest(q_trajs_vel_ref[0],full_dN[num_T_pts,:] @ q_control_points[0] * full_rs)
    print(idx)
    #print("original : ", q_control_points[0])
    print(full_N[num_T_pts,:] @ q_control_points[0])
    print(full_dN[num_T_pts,:] @ q_control_points[0] * full_rs)
    print(full_ddN[num_T_pts,:] @ q_control_points[0] * full_rs * full_rs)

    print(q_trajs_pos_ref[0, num_T_pts - 2 : num_T_pts + 2])
    print(q_trajs_vel_ref[0, num_T_pts - 2 : num_T_pts + 2])
    print(q_trajs_acc_ref[0, num_T_pts-2 : num_T_pts + 2])
    #print()
    # spline_l = split_curve_scipy(bspl_l[0], 0.5) # do not exist in knot
    a = 0
    b = 0.44
    spline_l = split_curve_scipy(bspl_l[0], a, b) 

    #subgoal_idx = [0, 43, 85]
    subgoal_idx = [0, 64]

    new_q_cps = spline_l.c
    print("control point length : ",len(new_q_cps))
    new_q_cps = to_torch(new_q_cps,**tensor_args)
    partial_traj = ParametricTrajectoryBspline(
            n_control_points=len(new_q_cps),
            degree=5,
            num_T_pts=num_T_pts,
            zero_vel_at_start_and_goal=False,
            zero_acc_at_start_and_goal=False,
            remove_outer_control_points=False,
            keep_last_control_point=False,
            trajectory_duration=5.0 *(b-a), 
            tensor_args=tensor_args,
            phase_time_class="PhaseTimeLinear",
        )
    q_trajs_d = partial_traj.get_q_trajectory_in_phase(new_q_cps, get_type=("pos","vel","acc"))
    # print(partial_traj.phase_time.rs[..., None])
    scale = partial_traj.phase_time.rs[0]
    #scale = 0.5850 # 3/5 * 39/40
    #scale = 1.0 / partial_traj.trajectory_duration
    q_trajs_pos = q_trajs_d["pos"] 
    q_trajs_vel = q_trajs_d["vel"] * scale
    q_trajs_acc = q_trajs_d["acc"] * (scale * scale)

    q_trajs_pos_all.append(q_trajs_pos.unsqueeze(0))
    q_trajs_vel_all.append(q_trajs_vel.unsqueeze(0))
    q_trajs_acc_all.append(q_trajs_acc.unsqueeze(0))
    q_cps_all.append(to_numpy(new_q_cps)) # new_q_cps can have different length

    # for i_h in range(n_hrz) : 
    #     u_tmp = parametric_trajectory.bspline.u.copy()
    #     p_tmp = to_numpy(q_control_points[0])
    #     print("knot : {}, control point : {}",u_tmp.shape,p_tmp.shape)
    #     #print(p_tmp)
    #     #new_q_cps, new_u = extract_subcurve(p_tmp, u_tmp, 5, 1/3 * i_h, 1/3 * i_h + 1/3 )
    #     # new_q_cps, new_u = extract_subcurve(p_tmp, u_tmp, 5, 1/2 * i_h, 1/2 * i_h + 1/2)
    #     new_q_cps = spline_l[i_h].c
    #     print(new_q_cps)
    #     #continue 
    #     #raise NotImplementedError
    #     # if i_h == 0 :
    #     #     new_q_cps = p_tmp[:15]
    #     # else : 
    #     #     new_q_cps = p_tmp[15:]
    #     # print(new_q_cps) # 14

    #     #print(len(new_u))
    #     new_q_cps = to_torch(new_q_cps,**tensor_args)
    #     # q_start = new_q_cps[0,:]
    #     # q_end = new_q_cps[-1,:]
    #     # shorter bspline
    #     partial_traj = ParametricTrajectoryBspline(
    #         n_control_points=len(new_q_cps),
    #         degree=5,
    #         num_T_pts=num_T_pts,
    #         zero_vel_at_start_and_goal=False,
    #         zero_acc_at_start_and_goal=False,
    #         remove_outer_control_points=False,
    #         keep_last_control_point=False,
    #         trajectory_duration=5.0 / 2, 
    #         tensor_args=tensor_args,
    #         phase_time_class="PhaseTimeLinear",
    #     )
        

    #     idx_d = {}
    #     if i_h == 0 : 
    #         idx_d['right'] = subgoal_idx[i_h+1] 
    #     elif i_h == n_hrz -1 : 
    #         idx_d['left'] = subgoal_idx[i_h]
    #     else : 
    #         idx_d['left'] = subgoal_idx[i_h]
    #         idx_d['right'] = subgoal_idx[i_h+1]

    #     spl_dict = {
    #         'h' : 1.0 / (partial_traj.bspline.m - 2 * partial_traj.bspline.d),
    #         'T' : partial_traj.bspline.num_T_pts
    #     }

    #     # replace_pts = get_new_control_point(idx_d, q_trajs_pos_ref, q_trajs_vel_ref, q_trajs_acc_ref,
    #     #                         **spl_dict
    #     #                     )
    #     # # print("q_start :",q_start,"q_end :", q_end)
    #     # # idx_start = find_closest(q_pos_trajs[0,:,:], q_start)
    #     # # idx_end = find_closest(q_pos_trajs[0,:,:], q_end)
    #     # # print("idx_start :",idx_start, "idx_end : ",idx_end)
    #     # # q_trajs_d = partial_traj.get_q_trajectory(
    #     # #     new_q_cps, q_start, q_end, get_type = ("pos","vel","acc"),
    #     # #     get_time_representation=False
    #     # # )

    #     # if replace_pts.get('left') is not None : 
    #     #     #new_q_cps[:3] = replace_pts['left']
    #     #     print(replace_pts['left'])
    #     #     new_q_cps = torch.concat([replace_pts['left'].squeeze(0), new_q_cps])
        
    #     # if replace_pts.get('right') is not None: 
    #     #     #new_q_cps[-3:] = replace_pts['right']
    #     #     print(replace_pts['right'])
    #     #     new_q_cps = torch.concat([new_q_cps, replace_pts['right'].squeeze(0)])
    #     print(f"--------------segment{i_h}-----------")
    #     part_N = partial_traj.bspline.N[0]
    #     part_dN = partial_traj.bspline.dN[0]
    #     part_ddN = partial_traj.bspline.ddN[0]
    #     part_rs = partial_traj.phase_time.rs[0]
    #     if i_h  ==  0 : 
    #         print(part_N[-1,:])
    #         print(part_dN[-1,:])
    #         print(part_ddN[-1,:])
    #         print(part_rs)
    #         #idx = find_closest(q_trajs_vel_ref[0],full_dN[num_T_pts,:] @ q_control_points[0] * full_rs)
    #         #print(idx)
    #         #print("original : ", q_control_points[0])
    #         print(part_N[-1,:] @ new_q_cps)
    #         print(part_dN[-1,:] @ new_q_cps * part_rs)
    #         print(part_ddN[-1,:] @ new_q_cps * part_rs * part_rs)

    #         # print(q_trajs_pos_ref[0, num_T_pts - 2 : num_T_pts + 2])
    #         # print(q_trajs_vel_ref[0, num_T_pts - 2 : num_T_pts + 2])
    #         # print(q_trajs_acc_ref[0, num_T_pts-2 : num_T_pts + 2])
    #     elif i_h == n_hrz - 1 : 
    #         print(part_N[0,:])
    #         print(part_dN[0,:])
    #         print(part_ddN[0,:])
    #         print(part_rs)
    #         #idx = find_closest(q_trajs_vel_ref[0],full_dN[num_T_pts,:] @ q_control_points[0] * full_rs)
    #         #print(idx)
    #         #print("original : ", q_control_points[0])
    #         print(part_N[0,:] @ new_q_cps)
    #         print(part_dN[0,:] @ new_q_cps * part_rs)
    #         print(part_ddN[0,:] @ new_q_cps * part_rs * part_rs)
        
    #     q_trajs_d = partial_traj.get_q_trajectory_in_phase(new_q_cps, get_type=("pos","vel","acc"))
    #     # print(partial_traj.phase_time.rs[..., None])
    #     scale = partial_traj.phase_time.rs[0]
    #     #scale = 0.5850 # 3/5 * 39/40
    #     #scale = 1.0 / partial_traj.trajectory_duration
    #     q_trajs_pos = q_trajs_d["pos"] 
    #     q_trajs_vel = q_trajs_d["vel"] * scale
    #     q_trajs_acc = q_trajs_d["acc"] * (scale * scale)

    #     print("True pos, vel, acc :")
    #     if i_h == 0 :
    #         print(q_trajs_pos[-1,:])
    #         print(q_trajs_vel[-1,:])
    #         print(q_trajs_acc[-1,:])
    #     elif i_h == n_hrz - 1 :
    #         print(q_trajs_pos[0,:])
    #         print(q_trajs_vel[0,:]) 
    #         print(q_trajs_acc[0,:])

    #     q_trajs_pos_all.append(q_trajs_pos.unsqueeze(0))
    #     q_trajs_vel_all.append(q_trajs_vel.unsqueeze(0))
    #     q_trajs_acc_all.append(q_trajs_acc.unsqueeze(0))
    #     q_cps_all.append(to_numpy(new_q_cps)) # new_q_cps can have different length
    
    q_trajs_pos_all = to_numpy(torch.cat(q_trajs_pos_all, axis = 0))
    q_trajs_vel_all = to_numpy(torch.cat(q_trajs_vel_all, axis = 0))
    q_trajs_acc_all = to_numpy(torch.cat(q_trajs_acc_all, axis = 0))
    #q_cps_all = to_numpy(torch.cat(q_cps_all, axis = 0))
    #print(q_cps_all.shape) #(3, 14, 2)
    # -------------------------------- Visualize ---------------------------------
    # fig, axs = create_fig_and_axes(2, figsize=(6, 6))
    colors = ['red', 'green', 'blue', 'orange']
    # print(q_cps_all)
    # for i in range(n_hrz):
    #     axs.scatter(q_cps_all[i][:,0], q_cps_all[i][:,1], c =colors[i] , marker="o", s=10**2, zorder=100)
    #     axs.plot(q_trajs_pos_all[i,:, 0], q_trajs_pos_all[i,:, 1], c = colors[i], linestyle="solid", linewidth=3, marker="x")
        
    # axs.set_aspect('equal')
    # fig.savefig(os.path.join(data_dir, f"figures/bspline-combined-{task_id_selected:03d}.png"), bbox_inches="tight")
    
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
    #dt = (t_goal - t_start) / 3 / (num_T_pts-1)
    dt = (t_goal - t_start) * (b-a) / (num_T_pts-1)
    # Positions, velocities, accelerations
    q_trajs_filtered = (q_trajs_pos_all, q_trajs_vel_all, q_trajs_acc_all)

    for i, ax in enumerate(axs):
        for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
            if q_trajs_filtered_item is not None:
                # for i_hz in range(n_hrz) : 
                #     #tmp_timesteps = 5/3 * i_hz + dt*np.arange(num_T_pts)
                #     tmp_timesteps = 5/2 * i_hz + dt*(np.arange(num_T_pts) + 1) # adhoc offset
                #     # offset = 5
                #     # if i_hz == 0 :
                #     #     ax[j].plot(tmp_timesteps[:-offset], q_trajs_filtered_item[i_hz,:-offset,i], c = colors[i_hz], linestyle="solid")
                #     # elif i_hz == n_hrz-1 :
                #     #     ax[j].plot(tmp_timesteps[offset:], q_trajs_filtered_item[i_hz,offset:,i], c = colors[i_hz], linestyle="solid")
                #     ax[j].plot(tmp_timesteps, q_trajs_filtered_item[i_hz,:,i], c = colors[i_hz], linestyle="solid")
                #     print(tmp_timesteps.shape, q_trajs_filtered_item[i_hz,:].shape)
                tmp_timesteps = 5*a + dt*(np.arange(num_T_pts) + 1) # adhoc offset
                ax[j].plot(tmp_timesteps, q_trajs_filtered_item[0,:,i], c = colors[0], linestyle="solid")
                print(tmp_timesteps.shape, q_trajs_filtered_item[0,:].shape)


        ax[0].set_ylabel(f"$q_{i}$")

    # time limits
    t_eps = 0.1
    for ax in list(itertools.chain(*axs)):
        ax.set_xlim(t_start - t_eps, t_goal + t_eps)
    fig.savefig(os.path.join(data_dir, f"figures/joint_space_trajectories_combined.png"), bbox_inches="tight")


if __name__ == "__main__":
    visualize()
