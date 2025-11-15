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

from bspline_split_test import split_curve_scipy

def find_equal(find_t, value) : 
    assert find_t.ndim == 2 
    tmp = torch.abs(find_t - value)
    #print("find_closest",tmp)
    ind = torch.where(torch.all(tmp < 1e-10, dim = -1))
    return ind

def find_closest(find_t, value) :
    assert find_t.ndim == 2 
    tmp = torch.norm(find_t - value, dim = -1)
    #print("find_closest",tmp)
    _, ind = torch.min(tmp, dim = -1)
    return ind

def get_new_control_point(pos, vel, acc, h, T) : 
    # left
    res_dict = {}
    P_0 = pos
    P_1 = P_0 + h / 5 *  vel
    P_2 = 3 * P_1 - 2 * P_0 + 2*h * h / 20 * acc    
    res_dict["left"] = torch.stack((P_0, P_1, P_2), axis = 1)
    
    # if idx_r : 
    #     # right 
    #     P_N = pos[:, idx_r, :]
    #     P_N_1 = P_N - h / 5 * vel[:, idx_r, :]
    #     P_N_2 = 2 * P_N_1 - P_N + h * h / 20 * acc[:, idx_r, :]    
    #     res_dict["right"] = torch.stack((P_N_2, P_N_1, P_N), axis = 1)

    return res_dict

# def get_new_control_point(idx_d, pos, vel, acc, h, T) : 
#     #print(pos.shape)
#     idx_l = idx_d.get("left")
#     idx_r = idx_d.get("right")
#     # left
#     res_dict = {}
#     if idx_l : 
#         P_0 = pos[:, idx_l, :]
#         P_1 = P_0 + h / 5 *  vel[:, idx_l, :]
#         P_2 = 2 * P_1 - P_0 + h * h / 20 * acc[:, idx_l, :]    
#         res_dict["left"] = torch.stack((P_0, P_1, P_2), axis = 1)
    
#     if idx_r : 
#         # right 
#         P_N = pos[:, idx_r, :]
#         P_N_1 = P_N - h / 5 * vel[:, idx_r, :]
#         P_N_2 = 2 * P_N_1 - P_N + h * h / 20 * acc[:, idx_r, :]    
#         res_dict["right"] = torch.stack((P_N_2, P_N_1, P_N), axis = 1)

    # return res_dict
def remove_knot(t, c, k, u, tol=1e-10):
    """
    Remove knot value `u` exactly once from a (clamped) B-spline curve (t,c,k).
    Returns (t_new, c_new). Preserves geometry exactly when removal is feasible.
    
    Parameters
    ----------
    t : (m+1,) array_like
        Knot vector (nondecreasing).
    c : (n+1, d) or (n+1,) array_like
        Control points (coefficients). First axis is the control-point index.
    k : int
        Degree.
    u : float
        Knot value to remove (must exist in `t` with multiplicity >= 1).
    tol : float
        Tolerance for accepting the removal (left/right candidates agreement).

    Notes
    -----
    - Assumes an open (clamped) curve; works for non-uniform interior spacing.
    - For your split/merge use case (you split by *fully clamping* an interior
      cut and now want to undo it), each removal here will succeed *exactly*.
    - After one successful removal:
          len(t_new) = len(t) - 1
          c_new.shape[0] = c.shape[0] - 1
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c)
    m = len(t) - k #final removed knot
    n = c.shape[0] - k #final removed control point
    print(len(t), len(c)) 
    print(t)
    print(c)
    # m = n + k + 1
    assert m == n + k + 1, "knot/coeff size mismatch: len(t) must equal len(c)+k+1"

    # Locate an occurrence of u and its multiplicity s (count exact with a small tol)
    idx = np.where(np.isclose(t, u, atol=1e-12, rtol=0.0))[0]
    if idx.size == 0:
        raise ValueError("u is not a knot in t")
    r_first = idx[0]
    r_last  = idx[-1]
    s = idx.size # multiplicity 
    print(f"r_first : {r_first}, {u} multiplicity : {s}")

    # Choose which copy to remove. For interior removals it's customary to
    # remove the *last* interior occurrence (any copy gives the same result).
    r = r_last

    # Make sure u is removable (must be interior: k <= r <= m-k-1)
    if r < k or r > m - k - 1:
        raise ValueError("cannot remove at boundary; try interior u")

    # Affected band
    first = r - k
    last  = r - s      # inclusive
    # Work arrays (candidates from left and right)
    L = c.copy()
    R  = c.copy()

    # March from the left: overwrite L[first..last] in-place
    print("number of inner layer", k-s+1)

    i_start = first + 1 # 16
    i_last  = last - 1  # 14
    for layer in range(1, k+1) : # remove k times
        i_start -= 1 
        i_last += 1
        i = i_start
        j = i_last 
        print(layer,"th ")
        while (j-i > layer-1) : 
            denom = t[i + k + layer] - t[i]
            a_i = 0.0 if denom == 0.0 else (u - t[i]) / denom
            denom = t[j + k + 1] - t[j - layer + 1]
            a_j = 0.0 if denom == 0.0 else (u - t[j - layer + 1]) / denom
            print("i :", i, a_i, ", j :",j, a_j)
            L[i] = L[i-1] if a_i == 0.0 else (L[i] - (1-a_i) * L[i - 1]) / a_i
            R[j] = R[j+1] if a_j == 1.0 else (R[j] - a_j * R[j + 1]) / (1.0 - a_j)
            i += 1
            j -= 1

        print(L[i-1], i-1)
        print(R[j+1], j+1)
    # Consistency check: the two constructions must agree on the overlap.
    # It suffices to compare any one index inside [first .. last]; use mid.
    # mid = (first + last) // 2
    # if not np.allclose(L[mid], R[mid], atol=tol, rtol=0.0):
    #     raise ValueError("knot removal would change the curve beyond tolerance")
    # print (last, first) # 15, 16
    # print(c)
    # print(L)
    # print(R)
    # if last+1 > first : 
    # print("original part :")
    # print(c[11:21])
    # print("modified part : ")
    # # 0 ~ 16 -> 0 ~ 14
    # print(L[11:21]) # 12, 13, 14, 15
    # # 16 ~ 32 -> 19 ~ 32
    # print(R[11:21]) # 15, 16, 17, 18, 19
    # print(r_first, r_last) # r_first 16 / r_last 21
    # print(L[r_first + 1 - (k-1)], R[r_last + 1 - (k-1)]) # r_first + 1 - (k-1)
    
    if not np.allclose(L[r_first + 1 - (k-1)], R[r_last + 1 - (k-1)], atol=tol):
        raise ValueError("knot removal changes the curve")
    if c.ndim == 1:
        c_new = np.empty(n, dtype=c.dtype)
    else:
        c_new = np.empty((n,) + c.shape[1:], dtype=c.dtype)
    c_new[:r_first + 1 - (k-1)] = L[:r_first + 1 - (k-1)]
    c_new[r_first + 1 - (k-1):] = R[r_last + 1 - (k-1):]
    
    t_new = np.empty(m, dtype=t.dtype)
    t_new[:r_first + 1] = t[:r_first + 1]
    t_new[r_first + 1:] = t[r_last + 1:]   

    return t_new, c_new

def merge_splines(left, right) :
    # ParametricTrajectoryBspline
    k = left.k
    # map back to global parameter
    # assume left, right has same time duration
    a = 0.5
    tL = a * left.t
    tR = a + (1 - a) * right.t
    cL, cR = left.c, right.c
    # concatenate (keep one block of 'a' knots and avoid duplicating end CPs)
    t_glued = np.hstack([tL[:-(k+1)], tR])
    #c_glued = np.vstack([ cL[:-k], cR ])
    c_glued = np.vstack([ cL, cR ])
    print("t_glued : ",len(t_glued), "c_glued : ", len(c_glued))
    #print(t_glued)
    #print(c_glued)
    # now remove knot 'a' exactly k times (inverse of insertion)
    t, c = t_glued, c_glued
    # remove knot for k times
    t,c = remove_knot(t,c,k,a)
    return interpolate.BSpline(t, c, k)

def load_spline(data_dir) : 
    fix_random_seed(2)
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

    print("long scale :",parametric_trajectory.phase_time.rs[0])

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

    task_ids_selected = np.array([836, 900, 1000, 1100])
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
    #q_trajs_pos_ref = q_trajs_d["pos"]
    #q_trajs_vel_ref = q_trajs_d["vel"]
    #q_trajs_acc_ref = q_trajs_d["acc"]

    return bspl_l, q_trajs_d, planning_task

@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True),
    default=Path(DATASET_BASE_DIR) / "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
)
def main(data_dir) : 
    device = "cuda:0"
    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}
    
    bspl_l, q_trajs_d, planning_task = load_spline(data_dir)
    q_trajs_pos_ref = q_trajs_d["pos"]
    q_trajs_vel_ref = q_trajs_d["vel"]
    q_trajs_acc_ref = q_trajs_d["acc"]

    #prob = [[0.12, 0.56], [0.56, 1]]
    #prob = [[0.04, 0.48], [0.48, 0.92]]
    prob = [[0.0, 0.44], [0.44, 0.88]]
    partial_dur = 5.0 * 0.44 # len 16
    num_T_pts = 60
    scipy_l = []
    for task_id in range(len(bspl_l)) :
        spline_l = []
        scipy_l = []
        new_q_cps = []
        q_cps_all = []
        q_trajs_pos_all = []
        q_trajs_vel_all = []
        q_trajs_acc_all = []
        print("answer bspline before split ")
        ans_spline = split_curve_scipy(bspl_l[task_id],prob[0][0], prob[1][1])
        print(ans_spline.t)
        print(ans_spline.c)
        for i,p in enumerate(prob) : 
            tmp_spline = split_curve_scipy(bspl_l[task_id], p[0], p[1])
            #scipy_l.append(tmp_spline)
            new_q_cps = to_torch(tmp_spline.c, **tensor_args)
            tmp_traj = ParametricTrajectoryBspline(
                n_control_points=len(new_q_cps),
                degree=5,
                num_T_pts=num_T_pts,
                zero_vel_at_start_and_goal=False,
                zero_acc_at_start_and_goal=False,
                remove_outer_control_points=False,
                keep_last_control_point=False,
                trajectory_duration=partial_dur,
                tensor_args=tensor_args,
                phase_time_class="PhaseTimeLinear",
            )
            spline_l.append(tmp_traj)
            # create interpolate.BSpline from bspline (normalized knots)
            scipy_l.append(interpolate.BSpline( t = tmp_traj.bspline.u,
                                               c = tmp_spline.c, 
                                               k = tmp_traj.bspline.d))
            
            scale = tmp_traj.phase_time.rs[0] # same for right
            if i == 1 : 
                h = 1.0 / (tmp_traj.bspline.m - 2 * tmp_traj.bspline.d)
                T = tmp_traj.bspline.num_T_pts
                #print("before : ",new_q_cps)
                # previous end value 
                #scale = 1
                new_right_cps = get_new_control_point(q_trajs_pos_all[-1][0,-1,:], 
                                                    q_trajs_vel_all[-1][0,-1,:]/scale, 
                                                    q_trajs_acc_all[-1][0,-1,:]/(scale*scale),
                                                        h, T)
                
                # new_right_cps_2 = get_new_control_point(q_trajs_pos_all[-1][0,-1,:], 
                #                                       q_trajs_vel_all[-1][0,-2,:]/scale, 
                #                                       q_trajs_acc_all[-1][0,-3,:]/(scale*scale),
                #                                         h, T)
                
                # print(new_right_cps["left"] - new_right_cps_2["left"])

                new_q_cps[:3] = new_right_cps["left"].T

                # original value
                # new_right_cps = get_new_control_point(q_trajs_pos_ref[:,71,:], 
                #                                       q_trajs_vel_ref[:,71,:]*scale, 
                #                                       q_trajs_acc_ref[:,71,:]*(scale*scale),
                #                                         h, T)
                # new_q_cps[:3] = new_right_cps["left"][0]
                
                #print("after : ",new_q_cps)



            q_cps_all.append(to_numpy(new_q_cps)) 
            q_trajs_d = tmp_traj.get_q_trajectory_in_phase(new_q_cps, get_type=("pos","vel","acc"))
            #scale = 0.5850 # 3/5 * 39/40
            #scale = 1.0 / tmp_traj.trajectory_duration
            q_trajs_pos = q_trajs_d["pos"] 
            q_trajs_vel = q_trajs_d["vel"] * scale
            q_trajs_acc = q_trajs_d["acc"] * (scale * scale)

            q_trajs_pos_all.append(q_trajs_pos.unsqueeze(0))
            q_trajs_vel_all.append(q_trajs_vel.unsqueeze(0))
            q_trajs_acc_all.append(q_trajs_acc.unsqueeze(0))

        q_trajs_pos_all = to_numpy(torch.cat(q_trajs_pos_all, axis = 0))
        q_trajs_vel_all = to_numpy(torch.cat(q_trajs_vel_all, axis = 0))
        q_trajs_acc_all = to_numpy(torch.cat(q_trajs_acc_all, axis = 0))

        colors = ['red', 'green', 'blue', 'orange']

        fig, axs = planning_task.plot_joint_space_trajectories(
            q_pos_trajs=q_trajs_pos_ref[task_id].unsqueeze(0),
            q_vel_trajs=q_trajs_vel_ref[task_id].unsqueeze(0),
            q_acc_trajs=q_trajs_acc_ref[task_id].unsqueeze(0),
            # control_points=control_points_concat,
            set_q_pos_limits=False,
            set_q_vel_limits=False,
            set_q_acc_limits=False,
        )

        # print("find matching index")
        # # print(q_trajs_vel_ref[0])
        # print("position -----------")
        # print(q_trajs_pos_all[0][-1],q_trajs_pos_all[1][0])
        # l_pos_idx = find_closest(q_trajs_pos_ref[task_id], to_torch(q_trajs_pos_all[0][-1],**tensor_args))
        # r_pos_idx = find_closest(q_trajs_pos_ref[task_id], to_torch(q_trajs_pos_all[1][0],**tensor_args))
        # print(q_trajs_pos_ref[task_id, l_pos_idx])
        # print(l_pos_idx, r_pos_idx)
        # print("velocity -----------")
        # print(q_trajs_vel_all[0][-1],q_trajs_vel_all[1][0])
        # l_vel_idx = find_closest(q_trajs_vel_ref[task_id], to_torch(q_trajs_vel_all[0][-1],**tensor_args))
        # r_vel_idx = find_closest(q_trajs_vel_ref[task_id], to_torch(q_trajs_vel_all[1][0],**tensor_args))
        # print(q_trajs_vel_ref[task_id, l_vel_idx], q_trajs_vel_ref[task_id, 71])
        # print(l_vel_idx, r_vel_idx)
        # print("acceleration -----------")
        # print(q_trajs_acc_all[0][-1],q_trajs_acc_all[1][0])
        # l_acc_idx = find_closest(q_trajs_acc_ref[task_id], to_torch(q_trajs_acc_all[0][-1],**tensor_args))
        # r_acc_idx = find_closest(q_trajs_acc_ref[task_id], to_torch(q_trajs_acc_all[1][0],**tensor_args))
        # print(q_trajs_acc_ref[task_id, l_acc_idx], q_trajs_acc_ref[task_id, l_vel_idx])
        # print(l_acc_idx, r_acc_idx)

        #print("new control point")
        h = 1.0 / (spline_l[-1].bspline.m - 2 * spline_l[-1].bspline.d)
        T = spline_l[-1].bspline.num_T_pts
        scale = spline_l[-1].phase_time.rs[0]
        #scale = 1
        match_idx = 71
        #new_right_cps = get_new_control_point({'left' : 71}, q_trajs_pos_ref, q_trajs_vel_ref, q_trajs_acc_ref, h, T)
        # new_right_cps = get_new_control_point(
        #     to_torch(q_trajs_pos_all[0][-1],**tensor_args), 
        #     to_torch(q_trajs_vel_all[0][-1], **tensor_args)/scale,
        #     to_torch(q_trajs_acc_all[0][-1],**tensor_args)/(scale*scale),
        #              h, T)
        #print(q_cps_all[1][:3])
        #print(new_right_cps)

        t_start = 0
        t_goal = 5 
        #partial_dur * len(prob)
        dt = partial_dur / (num_T_pts-1)
        dt_merged = partial_dur * 2 / (2* num_T_pts-1)
        # Positions, velocities, accelerations
        q_trajs_filtered = (q_trajs_pos_all, q_trajs_vel_all, q_trajs_acc_all)

        # merge bspline
        merged_spline = merge_splines(scipy_l[0], scipy_l[1])
        merged_q_cps = to_torch(merged_spline.c, **tensor_args)
        
        merged_traj = ParametricTrajectoryBspline(
                n_control_points=len(merged_q_cps),
                degree=5,
                num_T_pts=num_T_pts*2,
                zero_vel_at_start_and_goal=False,
                zero_acc_at_start_and_goal=False,
                remove_outer_control_points=False,
                keep_last_control_point=False,
                trajectory_duration=partial_dur * 2,
                tensor_args=tensor_args,
                phase_time_class="PhaseTimeLinear",
            )
        q_trajs_merged_d = merged_traj.get_q_trajectory_in_phase(merged_q_cps, get_type=("pos","vel","acc"))
        scale = merged_traj.phase_time.rs[0]
        q_trajs_merged_pos = to_numpy(q_trajs_merged_d["pos"])
        q_trajs_merged_vel = to_numpy(q_trajs_merged_d["vel"] * scale)
        q_trajs_merged_acc = to_numpy(q_trajs_merged_d["acc"] * (scale * scale))
        q_trajs_merged = (q_trajs_merged_pos, q_trajs_merged_vel, q_trajs_merged_acc)
        for i, ax in enumerate(axs):
            for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
                # if q_trajs_filtered_item is not None:
                #     # for i_hz in range(n_hrz) : 
                #     #     #tmp_timesteps = 5/3 * i_hz + dt*np.arange(num_T_pts)
                #     #     tmp_timesteps = 5/2 * i_hz + dt*(np.arange(num_T_pts) + 1) # adhoc offset
                #     #     # offset = 5
                #     #     # if i_hz == 0 :
                #     #     #     ax[j].plot(tmp_timesteps[:-offset], q_trajs_filtered_item[i_hz,:-offset,i], c = colors[i_hz], linestyle="solid")
                #     #     # elif i_hz == n_hrz-1 :
                #     #     #     ax[j].plot(tmp_timesteps[offset:], q_trajs_filtered_item[i_hz,offset:,i], c = colors[i_hz], linestyle="solid")
                #     #     ax[j].plot(tmp_timesteps, q_trajs_filtered_item[i_hz,:,i], c = colors[i_hz], linestyle="solid")
                #     #     print(tmp_timesteps.shape, q_trajs_filtered_item[i_hz,:].shape)
                #     for i_hz in range(len(prob)) : 
                #         tmp_timesteps = 5 * prob[i_hz][0] + dt*(np.arange(num_T_pts)) # adhoc offset
                #         ax[j].plot(tmp_timesteps, q_trajs_filtered_item[i_hz,:,i], c = colors[i_hz], linestyle="solid")
                        # print(tmp_timesteps.shape, q_trajs_filtered_item[0,:].shape)
                # plot merged spline
                tmp_timesteps = 5 * prob[0][0] + dt_merged*(np.arange(num_T_pts*2))
                ax[j].plot(tmp_timesteps, q_trajs_merged[j][:,i], c = colors[0], linestyle="solid")
            ax[0].set_ylabel(f"$q_{i}$")

        # time limits
        t_eps = 0.1
        for ax in list(itertools.chain(*axs)):
            ax.set_xlim(t_start - t_eps, t_goal + t_eps)
        fig.savefig(os.path.join(data_dir, f"figures/b_spline_merged-merged-{task_id:03d}.png"), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__" : 
    main()
