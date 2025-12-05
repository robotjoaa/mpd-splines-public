import torch
import numpy as np

#from mpd.models.diffusion_models import CompDiffusionModel
from mpd.models.diffusion_models.helpers import (
    print_color,
)
from .utils import extract_ovlp_from_full
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from pb_ompl.pb_ompl import fit_bspline_to_path
from functools import partial

class Traj_Blender:
    def __init__(self, horizon,
                        len_ovlp_cd,
                        is_spline,
                        blend_type: str,
                        exp_beta=3,
                        trajs_info=None,
                        ):
        
        self.exp_beta = exp_beta
        self.blend_type = blend_type
        self.len_ovlp = len_ovlp_cd
        self.hzn_step_size = horizon - self.len_ovlp
        self.hzn = horizon
        self.gap_len = self.hzn - 2 * self.len_ovlp

        self.is_spline = is_spline
        self.trajs_info = trajs_info

        assert self.gap_len > 0

    def get_local_q_trajs(self, cps_list, device):
        assert self.trajs_info 

        local_d = self.trajs_info['local']
        n_comp = len(cps_list)
        result = {}
        pos_all = []
        vel_all = []
        acc_all = []
        for i, cps in enumerate(cps_list) :  
            cps = to_torch(cps, device=device)
            q_pos_start = None
            q_pos_end = None
            #import pdb; pdb.set_trace()
            if i== 0 : 
                tmp_traj = local_d['start']
                q_pos_start = cps[...,0,:].clone()#[...,None,:]
            elif i == n_comp -1 :
                tmp_traj = local_d['end']
                q_pos_end = cps[...,-1,:].clone()#[...,None,:]
            else :
                tmp_traj = local_d['mid']
                q_pos_start = cps[...,0,:].clone()#[...,None,:]
                q_pos_end = cps[...,-1,:].clone()#[...,None,:]
            # start, goal of tmp_traj must set 
            #import pdb; pdb.set_trace()
            tmp_q_trajs_d = tmp_traj.get_q_trajectory(cps, q_pos_start, q_pos_end, 
                                                    get_type=("pos","vel","acc"), get_time_representation=True)
            pos_all.append(tmp_q_trajs_d["pos"].unsqueeze(0))
            vel_all.append(tmp_q_trajs_d["vel"].unsqueeze(0))
            acc_all.append(tmp_q_trajs_d["acc"].unsqueeze(0))
            
        result["pos"] = torch.cat(pos_all)
        result["vel"] = torch.cat(vel_all)
        result["acc"] = torch.cat(acc_all)
        return result


    def blend_traj_lists(self, trajs_list):
        """
        trajs_list: list of len n_comp: [ (B,H,D),..., ]
        returns:
            - trajs_out: a list of [tot_hzn, dim]
        """
        ## to np 
        trajs_list = to_numpy(trajs_list)

        n_comp = len(trajs_list)
        b_s,_, dd = trajs_list[0].shape ## b h d

        tot_hzn = n_comp * self.hzn - \
                    (n_comp - 1) * self.len_ovlp
        
        print(f'{tot_hzn=}')
        # trajs_out = np.zeros( shape=(b_s, tot_hzn, dd) ) ## NOTE: default is float64
        trajs_out = np.zeros( shape=(b_s, tot_hzn, dd), dtype=np.float32 ) ## Dec 2: changed to float32
        cnt_v = np.zeros_like(trajs_out)
        ## copy non-ovlp parts
        for i_c in range(n_comp):
            tjs_p_i = trajs_list[i_c]

            if i_c == 0:
                tmp_idx_1 = 0
                tmp_idx_2 = self.hzn_step_size
                ## B,hstep,dim
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, :self.hzn_step_size, :]
            elif i_c < n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.gap_len
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:self.len_ovlp+self.gap_len, :]
                
            elif i_c == n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.hzn_step_size

                assert tmp_idx_2 == tot_hzn
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:, :]

            cnt_v[ :, tmp_idx_1:tmp_idx_2, : ] += 1
            print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}, {tot_hzn=}')

        ## handle and merge the ovlp parts
        for i_c in range(n_comp-1):
            tmp_idx_1 = (i_c + 1) * self.hzn_step_size
            tmp_idx_2 = tmp_idx_1 + self.len_ovlp

            ## b,sm_hzn,d
            tjs_p_i = trajs_list[i_c]
            _, end_tjs_i = extract_ovlp_from_full(tjs_p_i, self.len_ovlp)
            ## b,sm_hzn,d
            tjs_p_i_plus_1 = trajs_list[i_c+1]
            st_tjs_i_plus_1, _ = extract_ovlp_from_full(tjs_p_i_plus_1, self.len_ovlp)


            ## b,len_o,d
            if not self.is_spline : 
                trajs_blend = blend_2_np_trajs_23d(end_tjs_i, st_tjs_i_plus_1, 
                                               self.blend_type, self.exp_beta)

            else : 
                # blend bspline control points
                trajs_blend = blend_spline_np(end_tjs_i, st_tjs_i_plus_1)

            trajs_out[:, tmp_idx_1:tmp_idx_2, :] = trajs_blend
            cnt_v[:, tmp_idx_1:tmp_idx_2, :] += 1

            print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}')

        assert tmp_idx_2 == (tot_hzn - self.hzn_step_size)


        assert (cnt_v == 1).all()

        

        return trajs_out
    
    # def get_traj_timesteps(self, n_comp) : 
    #     if self.is_spline : 
    #         raise NotImplemented
    #     else : 
            


def blend_2_np_trajs_23d(traj_1: np.ndarray, traj_2: np.ndarray, blend_type='exponential', beta=5):
    """
    ** Only takes in the ovlp parts **, blend full traj_1 and traj_2
    ** Blend for multiple dim,
    ----[----
         ----]-----

    Parameters:
    - traj_1: np.ndarray, shape (N1, D), first trajectory positions
    - traj_2: np.ndarray, shape (N2, D), second trajectory positions
    - blend_type: str, type of blending function ('exponential', 'cosine', 'linear', 'smoothstep')
    - beta: float, parameter for the exponential blending function (controls sharpness)

    Returns:
    - traj_blend: np.ndarray, blended trajectory positions
    """

    # assert traj_1.ndim == 2 and traj_1.shape[1] == 1
    assert traj_1.ndim in [2,3] and traj_2.ndim in [2,3] 
    assert traj_1.shape and traj_2.shape

    if traj_1.ndim == 2:
        len_tj, _ = traj_1.shape
    else:
        b_s, len_tj, _ = traj_1.shape
    # Overlapping region from t = 8 to t = 10
    t_overlap_start = 0
    t_overlap_end = len_tj - 1
    t_overlap = np.arange(0, len_tj) ## 1D

    ## Blending function selection, Checked, correct formula
    if blend_type in ['exponential', 'exp']:
        # Exponential blending function
        def w(t):
            exponent = -beta * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
    elif blend_type == 'cosine':
        # Cosine blending function
        def w(t):
            return 0.5 * (1 + np.cos(np.pi * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)))
    elif blend_type == 'linear':
        # Linear blending function
        def w(t):
            return 1 - (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
    elif blend_type == 'smoothstep':
        # Smoothstep blending function
        def w(t):
            x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return 1 - (3 * x**2 - 2 * x**3)
    else:
        raise ValueError("Invalid blending function. Choose 'exponential', 'cosine', 'linear', or 'smoothstep'.")

    ## Compute weights, np 1d, from 0 to 1
    ## weights = w(t_overlap)[:, np.newaxis]  # Column vector for broadcasting
    weights = w(t_overlap)  # Column vector for broadcasting
    if traj_1.ndim == 2:
        weights = weights[:, None] ## (len_tj, 1)
    elif traj_1.ndim == 3:
        weights = weights[None, :, None,] ## (1, len_tj, 1)
    
    # print(f'{weights[(0,-1),]=},{weights.shape=}')

    ## Blend the overlapping region
    traj_blend = weights * traj_1 + (1 - weights) * traj_2

    # print(f'{traj_blend.shape=}')

    return traj_blend

def blend_spline_np( 
    left_traj : np.ndarray,
    right_traj : np.ndarray,
    #merged_spl,
    #overlap = 10,
) : 
    # blend trajectory 
    l_ovlp = left_traj.copy()
    r_ovlp = right_traj.copy()
    ovlp_traj = blend_2_np_trajs_23d(l_ovlp, r_ovlp, blend_type="linear")
    return ovlp_traj 
    # # print(blend_traj.shape)

    # # fit bspline
    # bspline_params = fit_bspline_to_path(
    #     blend_traj,
    #     bspline_degree=merged_spl.bspline.d,
    #     bspline_num_control_points=merged_spl.bspline.n_pts,
    #     bspline_zero_acc_at_start_and_goal=False,
    #     bspline_zero_vel_at_start_and_goal=False,
    #     debug=False,
    # )

    # _, cc, _ = bspline_params
    # return cc.T


