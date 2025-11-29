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
from bspline_merge_test import load_spline, get_new_control_point

def merge_splines(
    l_cps,
    r_cps,
    left_traj,
    right_traj,
    overlap=6,
    K=3,
    tensor_args=None,
):
    """
    Merge two quintic B-splines with C² continuity using an overlap band
    and multi-sample least-squares for stability.

    Args:
        left_traj:  ParametricTrajectoryBspline (sml_traj for left)
        right_traj: ParametricTrajectoryBspline (sml_traj for right)
        overlap:    number of unknown merged control points to blend (default 6)
        K:          number of continuity samples per side (default 3)
        tensor_args: torch device/dtype dict

    Returns:
        merged_traj: ParametricTrajectoryBspline using merged control points
        merged_cps:  control points of merged spline (Tensor [N, dim])
    """

    assert overlap == 6, "This function assumed degree=5 (support width=6)."
    
    device = tensor_args["device"]
    dtype  = tensor_args["dtype"]

    # -- Extract original control points
    #l_cps = left_traj.control_points.to(device=device, dtype=dtype)
    #r_cps = right_traj.control_points.to(device=device, dtype=dtype)

    dim = l_cps.shape[-1]
    n_sml = l_cps.shape[0]          # e.g., 16
    n_merged = n_sml + n_sml - overlap

    # ----------------------------------------------------------------------
    # 1. Construct merged trajectory with correct duration scaling
    # ----------------------------------------------------------------------
    dur_left  = left_traj.trajectory_duration
    dur_right = right_traj.trajectory_duration

    # assert dur_left == dur_right

    # scale merged duration proportional to span count ratio
    deg = left_traj.bspline.d
    spans_left = n_sml - deg
    spans_merge = n_merged - deg

    merged_dur = dur_left * spans_merge / spans_left

    merged_traj = ParametricTrajectoryBspline(
        n_control_points=n_merged,
        degree=deg,
        num_T_pts=left_traj.num_T_pts + right_traj.num_T_pts - 1,
        trajectory_duration=merged_dur,
        zero_vel_at_start_and_goal=False,
        zero_acc_at_start_and_goal=False,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        tensor_args=tensor_args,
        phase_time_class="PhaseTimeLinear",
    )

    final_spl = merged_traj.bspline
    sml_spl   = left_traj.bspline

    # ----------------------------------------------------------------------
    # 2. Build initial merged control points (left + right minus overlap)
    # ----------------------------------------------------------------------
    merged_cps = torch.zeros((n_merged, dim), **tensor_args)
    merged_cps[:n_sml] = l_cps
    merged_cps[n_sml:] = r_cps[overlap:]    # right 6 overlap CPs replaced later

    # overlap indices:
    j0 = n_sml - overlap
    unknown_idx = torch.arange(j0, j0 + overlap, device=device)

    # ----------------------------------------------------------------------
    # 3. Build reference trajectories (pos/vel/acc) in PHASE DOMAIN
    # scaling for phase→time conversion
    # ----------------------------------------------------------------------
    sml_scale = left_traj.phase_time.rs[0]
    merge_scale = merged_traj.phase_time.rs[0]

    qL = left_traj.get_q_trajectory_in_phase(l_cps, get_type=("pos","vel","acc"))
    qR = right_traj.get_q_trajectory_in_phase(r_cps, get_type=("pos","vel","acc"))

    # ----------------------------------------------------------------------
    # 4. Choose K sample locations in overlap support window
    # ----------------------------------------------------------------------
    # Overlap support = u[j0] .. u[j0+6]
    u = final_spl.u

    # Take midpoints of spans inside overlap region
    span_mid = []
    for j in range(j0 + 3, j0+overlap + 3 ):
        span_mid.append(0.5*(u[j] + u[j+1]))
    span_mid = torch.tensor(span_mid, **tensor_args)
    print(span_mid)
    # tensor([0.4048, 0.4524, 0.5000, 0.5476, 0.5952, 0.6429]
    # Select K interior sample points
    ts = span_mid[1:-1]            # remove extremes
    ts = ts[:K]                    # pick first K
    #ts= torch.tensor([ 0.3095, 0.3571, 0.4048, 0.4524, 0.5000, 0.5476, 0.5952], **tensor_args)
    #ts = torch.tensor([0.4048, 0.5000, 0.5952], **tensor_args)
    #ts = torch.tensor([0.4048, 0.5952], **tensor_args)
    print(f'{ts=}')
    # time grid of merged traj
    Tm = np.linspace(0.0, 1.0, merged_traj.num_T_pts)

    # time grid of small traj
    Ts = np.linspace(0.0, 1.0, left_traj.num_T_pts)

    # basis support is same for all samples (because join block fixed)
    k_join = j0 + overlap - 1   # last CP in overlap band
    idx_block = torch.arange(k_join - deg, k_join + 1, device=device)

    # mask for unknown CPs
    mask_block = torch.isin(idx_block, unknown_idx)

    # ----------------------------------------------------------------------
    # 5. Build tall A, b
    # ----------------------------------------------------------------------
    A_list = []
    b_list = []
    glob_target_l = []
    def add_constraint_row(t_sample, target):
        """Add pos/vel/acc continuity rows for a single sample."""
        # 1) Find knot span k such that u[k] <= t < u[k+1]
        t_val = float(t_sample)
        k = np.searchsorted(u, t_val, side='right') - 1
        k = int(np.clip(k, deg, len(u)-deg-2))   # clamp for safety

        # 2) Active control points for degree p=5:
        #    [k-5, k-4, k-3, k-2, k-1, k]
        idx_block = torch.arange(k-deg, k+1, device=device)

        # 3) Unknown vs fixed CPs in this block
        mask_block = torch.isin(idx_block, unknown_idx)

        # 4) Find closest time sample index in merged grid
        t_idx = int(np.searchsorted(Tm, t_val))

        for deriv, key in zip([0,1,2], ["pos","vel","acc"]):
            # basis coefficients
            if deriv == 0:
                coeffs = final_spl.N[0, t_idx, idx_block]
            elif deriv == 1:
                coeffs = final_spl.dN[0, t_idx, idx_block]
            else:
                coeffs = final_spl.ddN[0, t_idx, idx_block]
            # unknown/fixed split
            idx_unknown = idx_block[mask_block]
            idx_fixed   = idx_block[~mask_block]
            print(idx_unknown, idx_fixed)
            # fixed contrib 
            if idx_fixed.numel() > 0:
                fc = (coeffs[~mask_block].unsqueeze(1) *
                      merged_cps[idx_fixed]).sum(dim=0)
            else:
                fc = torch.zeros(dim, **tensor_args)

            # A row
            Arow = torch.zeros(overlap, **tensor_args)
            if idx_unknown.numel() > 0:
                cols = (idx_unknown - j0).long()
                Arow[cols] = coeffs[mask_block]

            # b row
            brow = target[key] - fc
            A_list.append(Arow)
            b_list.append(brow)
        
        glob_target_l.append(target['pos'].unsqueeze(0))

    # left samples → sample at end of left trajectory
    for t_merged in ts:
        # map into LEFT phase domain:
        # t_s_left = (support phase in merged) scaled back to left
        # t_s_left = t_merged.item() * spans_merge / spans_left
        k = int(np.searchsorted(u, float(t_merged), side='right') - 1)
        # map merged span → left local span near the END of left spline
        t_s_left = 1.0 - (spans_merge - 1 - k) / spans_left
        t_s_left = float(np.clip(t_s_left, 0.0, 1.0))
        # nearest index in left samples
        sl_idx = int(np.abs(Ts - t_s_left).argmin())
        print(Ts[sl_idx], t_s_left )
        #assert np.abs(Ts[sl_idx] - t_s_left) < 1e-4 
        # reference jets scaled to merged‐time domain
        target_l = {
            "pos": qL["pos"][sl_idx],
            "vel": qL["vel"][sl_idx] * sml_scale / merge_scale,
            "acc": qL["acc"][sl_idx] * (sml_scale/merge_scale)**2,
        }
        add_constraint_row(t_merged, target_l)

    # right samples → sample at start of right trajectory
    for t_merged in ts:
        # t_s_right = (t_merged.item() * spans_merge -j0) / spans_left # TODO use spans right
        # merged span index
        k = int(np.searchsorted(u, float(t_merged), side='right') - 1)
        # map merged span → right local span near the START of right spline
        t_s_right = (k - j0) / spans_left
        t_s_right = float(np.clip(t_s_right, 0.0, 1.0))
        sr_idx = int(np.abs(Ts - t_s_right).argmin())
        print(Ts[sr_idx], t_s_right )
        #assert np.abs(Ts[sr_idx] - t_s_right) < 1e-4
        target_r = {
            "pos": qR["pos"][sr_idx],
            "vel": qR["vel"][sr_idx] * sml_scale / merge_scale,
            "acc": qR["acc"][sr_idx] * (sml_scale/merge_scale)**2,
        }
        add_constraint_row(t_merged, target_r)

    # stack
    A_big = torch.stack(A_list, dim=0)   # (6K*2, 6)
    b_big = torch.stack(b_list, dim=0)   # (6K*2, dim)
    
    # ----------------------------------------------------------------------
    # 6. Least-squares solve
    # ----------------------------------------------------------------------
    lsq = torch.linalg.lstsq(A_big, b_big)
    x_overlap = lsq.solution   # [6, dim]

    # write into merged cps
    merged_cps[unknown_idx] = x_overlap
    print(x_overlap)
    print(glob_target_l)
    return merged_cps, merged_traj, glob_target_l

def main() : 
    data_dir = Path(DATASET_BASE_DIR) / "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect"
    device = "cuda:0"
    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    bspl_l, q_trajs_d, planning_task = load_spline(data_dir)
    #q_trajs_pos_ref = q_trajs_d["pos"]
    #q_trajs_vel_ref = q_trajs_d["vel"]
    #q_trajs_acc_ref = q_trajs_d["acc"]

    task_id = 0
    prob = [[0.0, 0.44], [0.40, 0.84]] # 5 * 0.04 => 0.2 = 6 overlaps
    partial_dur = 5.0 * 0.44 # len 16

    # create new spline 
    ans_spline = split_curve_scipy(bspl_l[task_id],prob[0][0], prob[1][1])
    ans_cps = ans_spline.c ## 22
    print("control point length : ",len(ans_cps))
    ans_cps = to_torch(ans_cps,**tensor_args)
    partial_traj = ParametricTrajectoryBspline(
            n_control_points=len(ans_cps),
            degree=5,
            num_T_pts=111,
            zero_vel_at_start_and_goal=False,
            zero_acc_at_start_and_goal=False,
            remove_outer_control_points=False,
            keep_last_control_point=False,
            trajectory_duration=5.0 * prob[1][1] , 
            tensor_args=tensor_args,
            phase_time_class="PhaseTimeLinear",
        )
    

    q_trajs_d = partial_traj.get_q_trajectory_in_phase(ans_cps, get_type=("pos","vel","acc"))
    # print(partial_traj.phase_time.rs[..., None])
    ans_scale = partial_traj.phase_time.rs[0]
    #scale = 0.5850 # 3/5 * 39/40
    #scale = 1.0 / partial_traj.trajectory_duration
    q_trajs_pos_ref = q_trajs_d["pos"] 
    q_trajs_vel_ref = q_trajs_d["vel"] * ans_scale
    q_trajs_acc_ref = q_trajs_d["acc"] * ans_scale * ans_scale # time 


    n_ovlp = 4
    spline_l = []
    scipy_l = []
    new_q_cps = []
    q_cps_all = []
    q_trajs_pos_all = []
    q_trajs_vel_all = []
    q_trajs_acc_all = []
    sml_num_T_pts = 56
    for i,p in enumerate(prob) : 
        tmp_spline = split_curve_scipy(bspl_l[task_id], p[0], p[1])
        #scipy_l.append(tmp_spline)
        new_q_cps = to_torch(tmp_spline.c, **tensor_args)
        tmp_traj = ParametricTrajectoryBspline(
            n_control_points=len(new_q_cps),
            degree=5,
            num_T_pts=sml_num_T_pts,
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
            new_right_cps = get_new_control_point(q_trajs_pos_all[-1][0,-1,:], 
                                                q_trajs_vel_all[-1][0,-1,:]/scale, 
                                                q_trajs_acc_all[-1][0,-1,:]/(scale*scale),
                                                    h, T)

            new_q_cps[:3] = new_right_cps["left"].T

        # add noise 
        if i == 0 : 
            new_q_cps[-n_ovlp:] = new_q_cps[-n_ovlp:] + torch.randn_like(new_q_cps[-n_ovlp:]) * 0.005
        else : 
            new_q_cps[:n_ovlp] = new_q_cps[:n_ovlp] + torch.randn_like(new_q_cps[:n_ovlp]) * 0.005
        
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

    sml_num_T_pts = 56 # (sml_n_pts - k ) * 5 + 1
    sml_n_pts = 16
    n_pts = 16 + 16 -6
    num_T_pts = 106 

    l_traj = ParametricTrajectoryBspline(
        n_control_points=sml_n_pts,
        degree=5,
        num_T_pts=sml_num_T_pts,
        zero_vel_at_start_and_goal=False,
        zero_acc_at_start_and_goal=False,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        trajectory_duration=partial_dur,
        tensor_args=tensor_args,
        phase_time_class="PhaseTimeLinear",
    )
    r_traj = ParametricTrajectoryBspline(
        n_control_points=sml_n_pts,
        degree=5,
        num_T_pts=sml_num_T_pts,
        zero_vel_at_start_and_goal=False,
        zero_acc_at_start_and_goal=False,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        trajectory_duration=partial_dur,
        tensor_args=tensor_args,
        phase_time_class="PhaseTimeLinear",
    )
    l_cps= to_torch(q_cps_all[0], **tensor_args)
    r_cps= to_torch(q_cps_all[1], **tensor_args)
    merged_q_cps, merged_traj, g_targe_l = merge_splines(l_cps, r_cps, l_traj, r_traj, tensor_args=tensor_args)

    # plot merged cps
    #plt.close(fig)
    q_trajs_merged_d = merged_traj.get_q_trajectory_in_phase(merged_q_cps, get_type=("pos","vel","acc"))
    scale = merged_traj.phase_time.rs[0]
    q_trajs_merged_pos = to_numpy(q_trajs_merged_d["pos"])
    q_trajs_merged_vel = to_numpy(q_trajs_merged_d["vel"] * scale)
    q_trajs_merged_acc = to_numpy(q_trajs_merged_d["acc"] * (scale * scale))
    q_trajs_merged = (q_trajs_merged_pos, q_trajs_merged_vel, q_trajs_merged_acc)

    q_trajs_pos_all = []
    q_trajs_vel_all = []
    q_trajs_acc_all = []
    sml_scale = l_traj.phase_time.rs[0]
    for i_hz in range(len(prob)) : 
        q_trajs_tmp_d = l_traj.get_q_trajectory_in_phase(to_torch(q_cps_all[i_hz], **tensor_args), get_type=("pos","vel","acc"))        
        q_trajs_tmp_pos = q_trajs_tmp_d["pos"].unsqueeze(0)
        q_trajs_tmp_vel = (q_trajs_tmp_d["vel"]*sml_scale).unsqueeze(0)
        q_trajs_tmp_acc = (q_trajs_tmp_d["acc"]*sml_scale * sml_scale).unsqueeze(0)

        q_trajs_pos_all.append(q_trajs_tmp_pos)
        q_trajs_vel_all.append(q_trajs_tmp_vel)
        q_trajs_acc_all.append(q_trajs_tmp_acc)

    q_trajs_pos_all = to_numpy(torch.cat(q_trajs_pos_all, axis = 0))
    q_trajs_vel_all = to_numpy(torch.cat(q_trajs_vel_all, axis = 0))
    q_trajs_acc_all = to_numpy(torch.cat(q_trajs_acc_all, axis = 0))
        
    q_trajs_filtered = (q_trajs_pos_all, q_trajs_vel_all, q_trajs_acc_all)

    dim = 2
    fig, axs = plt.subplots(dim, 3, squeeze=False, figsize=(18, 2.5 * dim))

    axs[0, 0].set_title("Position")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Acceleration")
    axs[-1, 1].set_xlabel("Time [s]")

    # h = 1.0 / (spline_l[-1].bspline.m - 2 * spline_l[-1].bspline.d)
    # T = spline_l[-1].bspline.num_T_pts
    # scale = spline_l[-1].phase_time.rs[0]
    t_start = 0
    t_goal = merged_traj.trajectory_duration
    print(merged_traj.trajectory_duration)
    #partial_dur * len(prob)
    dt = l_traj.trajectory_duration / (l_traj.num_T_pts-1)

    dt_merged = merged_traj.trajectory_duration / (merged_traj.num_T_pts-1)
    # Positions, velocities, accelerations

    offset = (l_traj.trajectory_duration - 0.2) # 0.2 = 2*sml_traj.trajectory_duration - final_traj.trajectory-duration 
    print(prob)
    print(dt*merged_traj.num_T_pts)

    colors = ['red', 'green', 'blue', 'orange']
    q_trajs_ref_flt = (to_numpy(q_trajs_pos_ref), to_numpy(q_trajs_vel_ref), to_numpy(q_trajs_acc_ref))
    for i, ax in enumerate(axs):
        for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
            for i_hz in range(len(prob)) :
                tmp_timesteps =  offset *(i_hz) + dt*(np.arange(l_traj.num_T_pts))
                ax[j].plot(tmp_timesteps, q_trajs_filtered_item[i_hz,:,i], c = colors[i_hz], linestyle="solid")
            # plot merged spline
            tmp_timesteps = 0 + dt_merged*(np.arange(merged_traj.num_T_pts))
            ax[j].plot(tmp_timesteps, q_trajs_merged[j][:,i], c = colors[-1], linestyle="solid")
            ax[j].plot(tmp_timesteps, q_trajs_ref_flt[j][:,i], c ="gray", linestyle="solid", alpha=0.4)
        ax[0].set_ylabel(f"$q_{i}$")

    t_eps = 0.1
    for ax in list(itertools.chain(*axs)):
        ax.set_xlim(t_start - t_eps, t_goal + t_eps)

    fig.savefig(os.path.join("figures/joint_space_trajectories.png"), bbox_inches="tight")
    
    # plot scatter 
    fig_cps, axs_cps = create_fig_and_axes(2, figsize=(6, 6))
    merged_q_cps_np = to_numpy(merged_q_cps)
    axs_cps.scatter(merged_q_cps_np[:,0], merged_q_cps_np[:,1], c="blue", marker="o", s=10**2, zorder=100)
    #axs_cps.scatter(ans_spline.c[:20,0], ans_spline.c[:20,1], c="blue", marker="o", s=10**2, zorder=109, alpha=0.3) # ground truth control point 
    g_target_l = to_numpy(torch.cat(g_targe_l, dim=0))
    print(g_target_l)
    axs_cps.scatter(g_target_l[:,0], g_target_l[:,1], c="blue", marker="o", s=10**2, zorder=109, alpha=0.3)
    ref_traj = to_numpy(q_trajs_pos_ref[0])
    axs_cps.plot(q_trajs_merged[0][:, 0], q_trajs_merged[0][:, 1], color="orange", linestyle="solid", linewidth=3, marker="x")
    axs_cps.plot(q_trajs_filtered[0][0, :, 0], q_trajs_filtered[0][0,:, 1], color="red", linestyle="solid", linewidth=3, marker="x", alpha=0.3)
    axs_cps.plot(q_trajs_filtered[0][1,:, 0], q_trajs_filtered[0][1,:, 1], color="green", linestyle="solid", linewidth=3, marker="x", alpha=0.3)
    #axs_cps.plot(ref_traj[:100, 0], ref_traj[:100, 1], color="red", linestyle="solid", linewidth=3, marker="x", alpha=0.3) # ground truth 
    
    fig_cps.savefig(os.path.join("figures/robot_trajectories.png"), bbox_inches="tight")
    

if __name__ == "__main__" :
    main()