from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import pdb

# from mpd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from mpd.models.diffusion_models.helpers import (
    cosine_beta_schedule, Losses, exponential_beta_schedule, extract_2d,
    batch_repeat_tensor_in_dict, print_color
)
from mpd.models.diffusion_models.sample_functions import (
    extract,
    guide_gradient_steps,
    ddpm_sample_fn,
    apply_hard_conditioning,
    ddim_create_time_pairs,
    apply_dict
)

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, clip_grad_by_norm, clip_grad_by_value
from .comp_diffuser import utils, Traj_Blender, Unet1D_TjTi_Stgl_Cond_V3

from mpd.models.diffusion_models import CompDiffusionModel
from mpd.inference.cost_guides_comp import CostGuideManagerCompTrajectory

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class CompDiffusionModelv2(CompDiffusionModel):
    """
    Improved CompDiffusionModel with 
    Global, local loss, cfg
    Uses improved context model with global, local models

    training part modified
    
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert isinstance(self.model, Unet1D_TjTi_Stgl_Cond_V3)
        assert self.guide_mode != 'default' # train using cfg 
        assert self.predict_epsilon

        self.condition_guidance_l = kwargs.get('condition_guidance_l', 1.5)
        self.condition_guidance_g = kwargs.get('condition_guidance_g', 0.5)
        # Loss weight 
        self.lambda_l = kwargs.get('lambda_l', 1)
        self.lambda_g = kwargs.get('lambda_g', 1)

        print_color(f"CompDiffusionModelv2 {self.condition_guidance_g =}, {self.condition_guidance_l=}")
        print_color(f"CompDiffusionModelv2 {self.lambda_g=}, {self.lambda_l=}")

    # ------------------------------------------ training ------------------------------------------#
    # called from render_samples
    def get_tj_cond(self, x, g_cond, context_d, timesteps):
        """
        TODO: Directly copy from p_sample_loop, probably we can later use this func in the func
        generate the input for denoising
        - g_cond: a dict
        - timesteps: (B,H)
        """
        if g_cond['do_cond'] == 'both_ovlp': ## full_clean
            st_traj, end_traj = self.extract_ovlp_from_full(g_cond['traj_full'])
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=st_traj,
                end_traj=end_traj,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                # is_rand=True,
                t_type=g_cond['t_type'],
                is_noisy=False,
                context_d = context_d, 
                hard_conds={},
                )
            # pdb.set_trace()
            tj_cond['do_cond'] = True
            # pdb.set_trace()
        elif g_cond['do_cond'] == 'both_stgl':
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=None,
                end_traj=None,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                context_d = context_d, 
                hard_conds=g_cond['stgl_cond'],
                )
            tj_cond['do_cond'] = True

        elif g_cond['do_cond'] == 'st_endovlp':
            _, end_traj = self.extract_ovlp_from_full(g_cond['traj_full'])
            st_cond = self.split_hard_conds(g_cond['stgl_cond'], is_start=True)
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=None,
                end_traj=end_traj,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                context_d = context_d, 
                hard_conds= st_cond,
                )
            tj_cond['do_cond'] = True
        
        elif g_cond['do_cond'] == 'stovlp_gl':
            st_traj, _ = self.extract_ovlp_from_full(g_cond['traj_full'])
            end_cond = self.split_hard_conds(g_cond['stgl_cond'], is_start=False)
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=st_traj,
                end_traj=None,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                context_d = context_d, 
                hard_conds=end_cond,
                )
            tj_cond['do_cond'] = True


        elif g_cond['do_cond'] == False:
            ## drop everything
            tj_cond = dict(st_ovlp_is_drop=None, end_ovlp_is_drop=None, 
                            is_st_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            is_end_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            context_d = context_d, 
                            )
            tj_cond['do_cond'] = False
        else: 
            raise NotImplementedError

        
        tj_cond['context_d'] = context_d

        ## x is also modified!
        return x, tj_cond

    @torch.no_grad()
    def conditional_sample(self, g_cond, horizon=None, batch_size=1, method="ddim", **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        #device = self.betas.device
        #batch_size = len(g_cond['traj_full']) ## TODO: check

        # pdb.set_trace()

        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)

        # only implemented for ddim
        assert self.use_ddim

        if method == "ddim":
            #return self.ddim_sample_loop(shape, hard_conds, **sample_kwargs)
            return self.ddim_sample_loop(shape, g_cond, **sample_kwargs)
        else : 
            raise NotImplementedError
        
    @torch.no_grad()
    def ddim_sample(self, x, 
        tj_cond, 
        t, # 2d
        t_next, # 2d
        sampling_timesteps,
        current_step, 
        ddim_eta = 0.0, 
        t_start_guide=torch.inf,
        scale_grad_by_one_minus_alpha=False,
        guide=None,
        guide_lr=0.05,
        n_guide_steps=1,
        max_perturb_x=0.1,
        clip_grad_fn = lambda x: x,
        ddim_scale_grad_prior=1.0,
        ddim_final_add_noise=True, # _time_next >= 1
        use_clipped_model_output=True, # cfg
        return_x_recon=False,
        compute_costs_with_xrecon=False,
        results_ns=None,
        g_cond=None,
        cfg_zero_end_ovlp=False,
        **sample_kwargs,
    ) : 

        alpha = extract_2d(self.alphas_cumprod, t, x.shape)
        alpha_next = extract_2d(self.alphas_cumprod, t_next, x.shape)
        # compute std_dev_t
        sigma = ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma**2).sqrt()
        
        assert 'context_d' in tj_cond and 'progress' in tj_cond['context_d']

        # denoising noise
        with TimerCUDA() as t_generator:
            
            if tj_cond['do_cond']: # cfg
                x_3, t_2d_3, tj_cond_3 = batch_repeat_tensor_in_dict(x, t, tj_cond, n_rp=3, repeat_context=False)
                if cfg_zero_end_ovlp:
                    tj_cond_3['force_zero_end_ovlp'] = True
                assert (t_2d_3[0] == t_2d_3[0,0]).all(), 'sanity check'
                t_1d_3 = t_2d_3[:, 0]
                out = self.model(x_3, t_1d_3, tj_cond_3, only_uncond=False)
                # first 1/3 is global
                # second 1/3 is uncond
                # last 1/3 is local

                out_global = out[:len(x), :, :]
                out_uncd = out[len(x):-len(x), :, :]
                out_local = out[-len(x):, :, :]
                
                model_out = out_uncd \
                    + self.condition_guidance_g * (out_global - out_uncd) \
                    + self.condition_guidance_l * (out_local - out_uncd)
                
                
            else : # do_cond is false, local is all drop 
                model_out = self.small_model_pred(x, t, tj_cond)

        if results_ns is not None:
            results_ns.t_generator += t_generator.elapsed

        beta_prod_t = 1 - alpha
        grad_prior = model_out
        
        def update_x(_x, _grad_prior):
            _x_recon = self.predict_start_from_noise(_x, t, noise=_grad_prior)
            if self.clip_denoised:
                _x_recon.clamp_(-1.0, 1.0)
            else:
                assert RuntimeError()

            # # if use classifier guidance, use modified grad 
            if self.guide_mode != "cfg" : # hybrid
                # default self.predict_epsilon == True
                # _pred_noise = _grad_prior (modified gradient from classifier guidance)
                if self.predict_epsilon : 
                    _pred_noise = self.predict_noise_from_start(_x, t, x0=_grad_prior)
                else :
                    raise NotImplementedError

            else : 
                if use_clipped_model_output : # always true
                    _pred_noise = (_x - alpha.sqrt() * _x_recon) / beta_prod_t.sqrt()
                else : 
                    _pred_noise = self.predict_noise_from_start(_x, t, x0=_x_recon)

            _x = _x_recon * alpha_next.sqrt() + c * _pred_noise
            # _x = apply_hard_conditioning(_x, hard_conds)
            return _x, _x_recon

        # Modify the noise if guidance is active
        # https://arxiv.org/pdf/2105.05233.pdf - Algorithm 2
        if self.guide_mode == "cfg" :
            guide = None

        if guide is not None and (sampling_timesteps - current_step <= t_start_guide) : 
            with TimerCUDA() as t_guide:
                x_start = x.clone()
                for k_gd in range(n_guide_steps):
                    grad_prior_weighted = grad_prior * ddim_scale_grad_prior
                    if compute_costs_with_xrecon:
                        raise NotImplementedError("compute_costs_with_xrecon is not implemented")
                    else:
                        assert 'context_d' in tj_cond and 'idx' in tj_cond
                        context_d = tj_cond['context_d']
                        idx = tj_cond['idx']
                        # if guide is not none, it must have timestep
                        # Pass context_d as keyword to avoid toggling return_cost arg in CostGuideManagerParametricTrajectory
                        #pdb.set_trace()
                        grad_guide = guide(x, context_d=context_d, idx=idx) # tj_cond must include time range for guidance, g_cond
                        

                    grad_guide_clipped = clip_grad_fn(grad_guide)
                    grad_guide_clipped_weighted = guide_lr * grad_guide_clipped

                    # grad_prior_weighted_norm = torch.linalg.norm(grad_prior_weighted, dim=-1)
                    # grad_guide_clipped_weighted_norm = torch.linalg.norm(grad_guide_clipped_weighted, dim=-1)
                    # print(f'denoising epsilon (norm): {grad_prior_weighted_norm.mean():.4f} +- {grad_prior_weighted_norm.std():.4f}')
                    # print(f'guide grad (norm): {grad_guide_clipped_weighted_norm.mean():.4f} +- {grad_guide_clipped_weighted_norm.std():.4f}')

                    if scale_grad_by_one_minus_alpha:
                        # by default we skip it, because (1-alpha) -> 0 when t -> 0
                        grad_total = grad_prior_weighted - (1 - alpha).sqrt() * grad_guide_clipped_weighted
                    else:
                        grad_total = grad_prior_weighted - grad_guide_clipped_weighted
                    # print(f'{grad_prior=}, {grad_guide_clipped=}, {grad_total=}')
                    x_tmp, x_recon = update_x(x, grad_total)
                    # Clip the perturbation to avoid large changes from x_start_opt
                    x_delta = x_tmp - x_start
                    x_delta_clipped = torch.clip(x_delta, -max_perturb_x, max_perturb_x)
                    x = x_start + x_delta_clipped

                if results_ns is not None:
                    results_ns.t_guide += t_guide.elapsed
        else:
            x, x_recon = update_x(x, grad_prior)

        if ddim_final_add_noise: #_time_next >= 1
            # add noise
            noise = torch.randn_like(x) if ddim_eta != 0.0 else 0.0
            x = x + sigma * noise
            # x = apply_hard_conditioning(x, hard_conds)
        
        if return_x_recon : 
            return x, x_recon
    
        else : 
            return x

    ### must implement 
    @torch.no_grad()
    def run_inference(
        self,
        context_d=None,
        #hard_conds=None,
        g_cond=None,
        n_samples=1,
        return_chain=False,
        return_chain_x_recon=False,
        is_train=True,
        condition_guidance_g=None,
        condition_guidance_l=None,
        **diffusion_kwargs,
    ):
        # repeat hard conditions and contexts for n_samples
        # for k, v in hard_conds.items():
        #     new_state = einops.repeat(v, "... -> b ...", b=n_samples)
        #     hard_conds[k] = new_state

        for k, v in context_d.items():
            context_d[k] = einops.repeat(v, "... -> b ...", b=n_samples)

        # Sample from diffusion model
        # if diffusion_kwargs have n_comp
        if is_train :  # training
            samples, chain, chain_x_recon = self.conditional_sample(
                #hard_conds,
                g_cond = g_cond,
                context_d=context_d,
                batch_size=n_samples,
                return_chain=True,
                return_chain_x_recon=True,
                **diffusion_kwargs,
            )
        else :
            # need n_comps in kwargs
            # hard_conds = g_cond['stgl_cond']
            # stgl_cond = apply_dict(
            #                 einops.repeat,
            #                 hard_conds,
            #                 'b d -> (repeat b) d', repeat=n_samples,
            #             )
            # g_cond['st_gl'] = stgl_cond

            print_color(f"loaded trained model : {self.guide_mode=}, {self.len_ovlp_cd=}")

            self.len_ovlp_cd = diffusion_kwargs.get('len_ovlp_cd', self.len_ovlp_cd)

            #if not hasattr(self, "tj_blder") : 
            # should be in diffusion_kwargs
            self.is_spline = diffusion_kwargs.get('is_spline', False) # manual  
            self.blend_type = diffusion_kwargs.get('blend_type', "linear")
            self.blend_beta = diffusion_kwargs.get('blend_beta', 3)
            inference_mode = diffusion_kwargs.get("guide_mode", 'default')
            self.guide_mode = inference_mode
            do_cond = diffusion_kwargs.get("do_cond", False)
            print_color(f"run_inference : {inference_mode=}, {do_cond=}, {self.len_ovlp_cd=}")
            print(f"hard_conds : {g_cond['st_gl']}")

            trajs_info = None 
            # if self.is_spline :
            tmp_guide = diffusion_kwargs.get('guide', None) 
            if tmp_guide is not None and isinstance(tmp_guide, CostGuideManagerCompTrajectory) : 
                trajs_info = tmp_guide.get_all_parametric_trajectory()

            self.tj_blder = Traj_Blender(
                self.horizon,
                self.len_ovlp_cd,
                self.is_spline,
                self.blend_type,
                exp_beta = self.blend_beta,
                trajs_info = trajs_info,
            )
        
            if condition_guidance_g is not None and isinstance(condition_guidance_g, float):
                self.condition_guidance_g = condition_guidance_g

            if condition_guidance_l is not None and isinstance(condition_guidance_l, float) : 
                self.condition_guidance_l = condition_guidance_l
            # pdb.set_trace()
            print_color(f"run_inference : {self.condition_guidance_g=}, {self.condition_guidance_l=}")
            
            sample = self.gen_cond_stgl(g_cond, 
                                        context_d, 
                                        batch_size=n_samples,
                                        return_chain=return_chain,
                                        pick_type="all",
                                        **diffusion_kwargs,
                                        )
            return sample

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain
        trajs_x_recon_chain_normalized = chain_x_recon

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, "b diffsteps ... -> diffsteps b ...")
        trajs_x_recon_chain_normalized = einops.rearrange(
            trajs_x_recon_chain_normalized, "b diffsteps ... -> diffsteps b ..."
        )

        if return_chain and return_chain_x_recon:
            return trajs_chain_normalized, trajs_x_recon_chain_normalized
        elif return_chain:
            return trajs_chain_normalized

        # return the last denoising step
        return trajs_chain_normalized[-1]


    #------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_start, x_noisy, noise, t_2d, tj_cond):
        
        batch_loss_w = torch.ones_like(x_start)  # [B, H, D]

        def zero_mask(mask, idx):
            if idx is None or (isinstance(idx, bool) and not idx):
                return
            idx_t = torch.as_tensor(idx, device=batch_loss_w.device)
            rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx_t.dim() == 0:  # scalar
                batch_loss_w[rows, idx_t.item()] = 0.0
            else:
                # Apply the same column indices to every masked row.
                batch_loss_w[rows.unsqueeze(-1), idx_t.unsqueeze(0), :] = 0.0

        zero_mask(tj_cond.get("is_st_inpat", torch.zeros_like(batch_loss_w[:, 0, 0], dtype=torch.bool)),
                  tj_cond.get("idx_st_inpat", None))
        zero_mask(tj_cond.get("is_end_inpat", torch.zeros_like(batch_loss_w[:, 0, 0], dtype=torch.bool)),
                  tj_cond.get("idx_end_inpat", None))

        assert x_start.shape[1] == self.horizon 
        # diffusion model
        # hard_conds considered with batch_loss_w

        # get concatenation of three outputs

        x_noisy_3, t_2d_3, tj_cond_3 = batch_repeat_tensor_in_dict(
            x_noisy, t_2d, tj_cond, n_rp=3, repeat_context=False
        )

        assert (t_2d_3[0] == t_2d_3[0,0]).all(), 'sanity check'
        
        t_1d_3 = t_2d_3[:, 0]
        
        # local / uncond / global
        out = self.model(x_noisy_3, t_1d_3, tj_cond_3)
        x_recon_l = out[:len(x_noisy), :, :]
        x_recon_u = out[len(x_noisy):-len(x_noisy), :, :]
        x_recon_g = out[-len(x_noisy):,:,:]

        # repeat noise, batch_loss_w 3 times to match x_recon 
        # noise = noise.repeat([3] + [1,]*len(noise.shape[1:]))
        # batch_loss_w = batch_loss_w.repeat([3] + [1,]*len(batch_loss_w.shape[1:]))

        # create weight vector
        # lam_u = torch.ones_like([x_start.shape[0] + [1,]*len(x_start.shape[1:])])
        # lam_l = self.lambda_l * torch.ones_like([x_start.shape[0] + [1,]*len(x_start.shape[1:])])
        # lam_g = self.lambda_g * torch.ones_like([x_start.shape[0] + [1,]*len(x_start.shape[1:])])


        if self.predict_epsilon:
            loss_l, _ = self.loss_fn(x_recon_l, noise, ext_loss_w=batch_loss_w)
            loss_u, _ = self.loss_fn(x_recon_u, noise, ext_loss_w=batch_loss_w)
            loss_g, _ = self.loss_fn(x_recon_g, noise, ext_loss_w=batch_loss_w)
        else:
            raise NotImplementedError
            #loss, info = self.loss_fn(x_recon, x_start, ext_loss_w=batch_loss_w)
        info = dict(
            loss_uncond = loss_u,
            loss_local = loss_l,
            loss_global = loss_g,
        )
        loss = loss_u + loss_l * self.lambda_l + loss_g * self.lambda_g
        return loss, info
    
    # # should be this format to match GaussianDiffusionLoss
    # def loss(self, x_clean, context_d, hard_conds):
    #     # cond_st_gl is start, end hard conds. 
    #     # it can be infered from context_d

    #     batch_size = x_clean.shape[0]
    #     # pdb.set_trace() ## check x dim
    #     t_1d = torch.randint(0, self.n_diffusion_steps, (batch_size, 1), device=x_clean.device).long()
    #     ## B,H
    #     t_2d = t_1d.expand(-1, self.horizon)
    #     #t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

    #     noise = torch.randn_like(x_clean)
    #     x_noisy = self.q_sample(x_start=x_clean, t_2d=t_2d, noise=noise)

    #     ## choose training conditioning scheme
        
    #     x_noisy, tj_cond = self.create_train_tj_cond(
    #         x_clean, x_noisy, t_1d[:, 0], t_1d[:,0].clone(), context_d, hard_conds, is_rand=True)
        
    #         ### make each tj_cond_* from tj_cond output
    #     total_loss, total_info = self.p_losses(x_clean[:, :, :], x_noisy, noise, t_2d, tj_cond)
    
    #     return total_loss, total_info

    def create_train_tj_cond(self, x_clean: torch.Tensor, 
                             x_noisy: torch.Tensor, ## for model input, change inpainting part
                             t_1d_st: torch.Tensor, 
                             t_1d_end: torch.Tensor, 
                             context_d: dict,
                             hard_conds: dict,
                             is_rand):
        """
        t_1d: (B,)
        """
        batch_size = x_clean.shape[0]
        device = x_clean.device

        ## True if we want to do condition
        ## (B,); bool, True if this sample will use overlap as condition
        st_cd_use_ovlp = torch.rand( size=(batch_size,), device=device ) < self.tr_ovlp_prob
        end_cd_use_ovlp = torch.rand( size=(batch_size,), device=device ) < self.tr_ovlp_prob

        st_cd_use_inpat = ~ st_cd_use_ovlp
        end_cd_use_inpat = ~ end_cd_use_ovlp


        #### no dropout
        # if self.guide_mode != "default" : 
        #     all_drop_prob = self.tr_all_drop_prob ## 0.15

        #     st_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob
        #     end_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob

        #     ## set those to be dropout to False, so no condition at all for 0.15 * bs
        #     st_cd_use_ovlp[st_is_all_drop] = False
        #     st_cd_use_inpat[st_is_all_drop] = False

        #     end_cd_use_ovlp[end_is_all_drop] = False
        #     end_cd_use_inpat[end_is_all_drop] = False




        ###### TODO:
        ###### We need to modify the corresponding samples for inpainting
        # pdb.set_trace() ## check cond_st_gl --> 0: (B,2), horizon-1:(B,2)

        cond_st = self.split_hard_conds(hard_conds, is_start=True)
        num_st_cond = len(cond_st)

        for k,v in cond_st.items() : 
            cond_st[k] = v[st_cd_use_inpat]
        x_noisy[ st_cd_use_inpat ] = apply_hard_conditioning( x_noisy[ st_cd_use_inpat ], cond_st)

        idx_st_inpat = None 
        if num_st_cond > 0 : 
            idx_st_inpat = torch.as_tensor(list(cond_st.keys()), dtype=torch.long, device = device)
        ####
        cond_end = self.split_hard_conds(hard_conds, is_start=False)
        num_end_cond = len(cond_end)
        # {self.horizon-1: hard_conds[self.horizon-1][end_cd_use_inpat]}
        for k,v in cond_end.items() : 
            cond_end[k] = v[end_cd_use_inpat]
        x_noisy[ end_cd_use_inpat ] = apply_hard_conditioning( x_noisy[ end_cd_use_inpat ], cond_end)

        idx_end_inpat= None
        if num_end_cond > 0 :
            idx_end_inpat = torch.as_tensor(list(cond_end.keys()), dtype=torch.long, device = device)

        # pdb.set_trace() #### check if replace properly



        ## t range is [0, 255(self.n_diffusion_steps-1)], the only available range is -0 or -1
        if is_rand:
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2) # [0,2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        else:
            assert False
            # t_1d_st = t_1d.clone()
            # t_1d_end = t_1d.clone()
        ## Oct 6 newly Added
        t_1d_st = torch.clamp(t_1d_st, min=0, max=self.n_diffusion_steps-1)
        t_1d_end = t_1d_st.clone()
        t_1d_end = torch.clamp(t_1d_end, min=0, max=self.n_diffusion_steps-1)


        ## TODO: slow, can be improved
        # t_2d_st = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)
        # t_2d_end = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)
        t_2d_st = t_1d_st.unsqueeze(1).expand(-1, self.len_ovlp_cd)   # (B, H)
        t_2d_end = t_1d_end.unsqueeze(1).expand(-1, self.len_ovlp_cd) 
        # pdb.set_trace()

        ## add noise
        st_traj = x_clean[:, :self.len_ovlp_cd, :].detach().clone()
        st_traj = self.q_sample(x_start=st_traj, t_2d=t_2d_st, noise=None)

        end_traj = x_clean[:, -self.len_ovlp_cd:, :].detach().clone()
        end_traj = self.q_sample(x_start=end_traj, t_2d=t_2d_end, noise=None)

        # pdb.set_trace()

        tj_cond = {
            'st_ovlp_is_drop': ~st_cd_use_ovlp, # st_is_drop,
            'end_ovlp_is_drop': ~end_cd_use_ovlp, #end_is_drop,
            ##
            'st_ovlp_traj': st_traj,
            'end_ovlp_traj': end_traj,
            ##
            'st_ovlp_t': t_1d_st,
            'end_ovlp_t': t_1d_end,
            ## NEW
            'is_st_inpat': st_cd_use_inpat,
            'is_end_inpat': end_cd_use_inpat,
            'idx_st_inpat' : idx_st_inpat, 
            'idx_end_inpat' : idx_end_inpat,
            'context_d' : context_d
        }
        # pdb.set_trace() ## TODO: check

        return x_noisy, tj_cond

    def small_model_pred(self, x, t_2d, tj_cond: dict, ) -> torch.Tensor:
        
        ## simple model should take in only 1d
        assert (t_2d[0] == t_2d[0,0]).all(), 'sanity check'

        ## only use for uncond output
        assert tj_cond['do_cond'] in [None, False]

        t_1d = t_2d[:, 0]
        pred_out = self.model(x, t_1d, tj_cond, only_uncond=True)

        return pred_out

    def create_eval_tj_cond(self, 
                            x_et: torch.Tensor, ## x eval t
                            st_traj, # : torch.Tensor, 
                            end_traj,
                            t_1d_st: torch.Tensor, 
                            t_1d_end: torch.Tensor, 
                            t_type: str,
                            is_noisy: bool,
                            context_d:dict,
                            hard_conds:dict,
                            progress: Optional[Union[float, torch.Tensor]] = None):
        """
        t_1d: (B,)
        if st_traj is not None, then do st traj inpainting;
        if end_traj is not None, then do end traj inpainting;
        if 0 in stgl_cond, then do start inpainting;
        if hzn-1 in stgl_cond, then do end inpainting;
        """
        assert t_1d_st.ndim == 1 and t_1d_end.ndim == 1

        ## TODO: check the case where both end do inpainting, and using cls-free guidance

        batch_size = x_et.shape[0] # if st_traj is not None else end_traj.shape[0]
        device = x_et.device
        
        d_dim = x_et.shape[2]
        
        # batch_size = st_traj.shape[0]
        if st_traj == None:
            ## drop everything
            # st_traj = torch.zeros_like(end_traj)
            # st_traj = torch.zeros(size=(batch_size, self.len_ovlp_cd, d_dim), 
                                #   dtype=x_et.dtype, device=device)
            # st_is_drop = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
            st_is_drop = None
        else:
            ## keep everything ## From Here Oct 8 20:51 TODO: using tensor might be slower than np?
            # st_is_drop =  np.zeros(shape=(batch_size,), dtype=bool) ## no drop
            st_is_drop = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device) ## no drop

            # assert 0 not in hard_conds.keys()
            # batch_size = st_traj.shape[0]
        
        ## start conditioning
        #if 0 in hard_conds.keys():
        st_cond = self.split_hard_conds(hard_conds, is_start=True)
        num_st_cond = len(st_cond)


        if st_traj is None and num_st_cond > 0 : 
            # pdb.set_trace()
            # assert st_traj is None
            x_et = apply_hard_conditioning(x_et, st_cond)
            is_st_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
            idx_st_inpat =  torch.as_tensor(list(st_cond.keys()), dtype=torch.long, device = device)
        else:
            is_st_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
            idx_st_inpat = None



        
        hzn_minus1 = self.horizon - 1
        if end_traj == None:
            end_is_drop = None # no feature
        else:
            end_is_drop = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device) # no drop

        
        ## hzn_minus1 cannot be in hard_conds eventhough it is last segment
        ## when use ee_pose_goal 

        ## do inpainting conditioning
        # if hzn_minus1 in hard_conds.keys():
        end_cond = self.split_hard_conds(hard_conds, is_start=False)
        num_end_cond = len(end_cond)
        if end_traj is None and num_end_cond > 0 : 
            # pdb.set_trace()
            # assert end_traj is None
            x_et = apply_hard_conditioning(x_et, end_cond)
            ## create 2 dimension is_end_inpat compatible for multiple hard conditions
            is_end_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
            idx_end_inpat =  torch.as_tensor(list(end_cond.keys()), dtype=torch.long, device = device)
        else:
            is_end_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
            idx_end_inpat = None

        # ## each end should have some conditions
        # assert ( (st_traj is not None) or 0 in hard_conds ) and \
        #             ( (end_traj is not None) or hzn_minus1 in stgl_cond )

       # pdb.set_trace()
        ## t range is [0, 255(self.n_diffusion_steps-1)], the only available range is -0 or -1
        if t_type == 'rand':
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2) # [0,2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        elif t_type == '-1':
            t_1d_st = t_1d_st - torch.ones_like(t_1d_st,)
            t_1d_end = t_1d_end - torch.ones_like(t_1d_end,)
        elif t_type == '0':
            pass
        else:
            raise NotImplementedError
            # t_1d_st = t_1d.clone()
            # t_1d_end = t_1d.clone()
        t_1d_st = torch.clamp(t_1d_st, min=0, max=self.n_diffusion_steps-1)
        t_1d_end = torch.clamp(t_1d_end, min=0, max=self.n_diffusion_steps-1)


        # t_2d_st = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)
        # t_2d_end = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)
        t_2d_st = t_1d_st.unsqueeze(1).expand(-1, self.len_ovlp_cd)
        t_2d_end = t_1d_end.unsqueeze(1).expand(-1, self.len_ovlp_cd)
        
        ### -------
        if st_traj == None:
            pass
        elif not is_noisy:
            st_traj = self.q_sample(x_start=st_traj, t_2d=t_2d_st, noise=None)
        else:
            st_traj = st_traj.clone()
        
        if end_traj == None:
            pass
        elif not is_noisy:
            end_traj = self.q_sample(x_start=end_traj, t_2d=t_2d_end, noise=None)
        else:
            end_traj = end_traj.clone()

        # pdb.set_trace()

        # ensure progress matches batch size for context model (used in ContextModelGlobal)
        context_d = dict(context_d)
        progress_tensor = progress if progress is not None else context_d.get('progress', None)
        if progress_tensor is None:
            progress_tensor = torch.zeros(batch_size, device=device, dtype=x_et.dtype)
        else:
            if not torch.is_tensor(progress_tensor):
                progress_tensor = torch.tensor(progress_tensor, device=device, dtype=x_et.dtype)
            else:
                progress_tensor = progress_tensor.to(device=device, dtype=x_et.dtype)
            if progress_tensor.ndim == 0:
                progress_tensor = progress_tensor.expand(batch_size)
            elif progress_tensor.shape[0] == 1 and batch_size > 1:
                progress_tensor = progress_tensor.expand(batch_size, *progress_tensor.shape[1:])
            else:
                assert progress_tensor.shape[0] == batch_size, "progress must align with batch size"
            assert progress_tensor.numel() == batch_size, "progress must provide one value per batch element"
            # flatten any trailing singleton dims to 1D for the sinusoidal encoder
            progress_tensor = progress_tensor.reshape(batch_size)
        context_d['progress'] = progress_tensor

        ## TODO: Add guide time range to tj_cond
        tj_cond = {
            'st_ovlp_is_drop': st_is_drop,
            'end_ovlp_is_drop': end_is_drop,
            ## must be noisy
            'st_ovlp_traj': st_traj,
            'end_ovlp_traj': end_traj,
            ## 
            'st_ovlp_t': t_1d_st,
            'end_ovlp_t': t_1d_end,
            ##
            'is_st_inpat': is_st_inpat,
            'is_end_inpat': is_end_inpat,
            'idx_st_inpat': idx_st_inpat, # used for batch_loss_w
            'idx_end_inpat': idx_end_inpat,
            'context_d': context_d,

            ## eval_st
            'force_zero_end_ovlp' : self.eval_st_only
        }
        # pdb.set_trace()
        
        return x_et, tj_cond

    def gen_cond_stgl(self, 
                      g_cond, 
                      context_d,
                      batch_size=1,
                      n_comp=2,
                      horizon=None,
                      return_chain=False,
                      pick_type="first",
                      top_n=1,  
                      results_ns = None,
                      **diffusion_kwargs):
        """
        Jan 21: Default Version that Support Replan
        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        self.len_ovlp_cd = diffusion_kwargs.get('len_ovlp_cd', self.len_ovlp_cd)
        print(f"gen_cond_stgl {batch_size=}, {n_comp=}, {horizon=}, {self.len_ovlp_cd=}, {pick_type=}, {top_n=}")
        #import pdb; pdb.set_trace()
        
        hzn = horizon if horizon else self.horizon 
        o_dim = self.state_dim
        c_shape = [batch_size, hzn, o_dim] ## e.g.,(20,160,2)
        
        # pdb.set_trace() ## check format

        st_gl = g_cond['st_gl']

        ## shape: 2, n_probs, dim
        # assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        
        ## make sure return is not a view
        hard_conds = {}
        #pdb.set_trace()
        if st_gl :
            for k,v in st_gl.items() : 
                if v.ndim == 1 : 
                    tmp = v[None, ]
                else : 
                    tmp = v 
                assert tmp.ndim == 2
                hard_conds[k] = einops.repeat(tmp, 'n_p d -> (n_p rr) d', rr=batch_size).clone()

        if return_chain : 
            trajs_list, chain = self.comp_pred_p_loop_n(
                c_shape, context_d, hard_conds, n_comp=n_comp, return_chain=True, 
                results_ns = results_ns, **diffusion_kwargs)
        else : 
            trajs_list = self.comp_pred_p_loop_n(
                c_shape, context_d, hard_conds, n_comp=n_comp, return_chain=False, 
                results_ns = results_ns, **diffusion_kwargs)
        

        # if self.cp_infer_t_type == 'interleave': ## original our
        #     trajs_list = self.diffusion_model.comp_pred_p_loop_n(
        #         c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        # elif self.cp_infer_t_type == 'same_t': ## Same t denoising, but not parallel
        #     trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t(
        #         c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
            
        # elif self.cp_infer_t_type == 'gsc': ## baseline
        #     trajs_list = self.diffusion_model.comp_pred_p_loop_n_GSC(
        #         c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        # elif self.cp_infer_t_type == 'same_t_p': ## Same t denoising and *parallel*
        #     trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t_parallel(
        #         c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        # elif self.cp_infer_t_type == 'ar_back': ## backward autoregressive denosing
        #     trajs_list = self.diffusion_model.comp_pred_p_loop_n_ar_backward(
        #         c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        # else:
        #     raise NotImplementedError
        
        #self.ncp_pred_time_list.append( [self.n_comp,  time.time() - cur_time] ) ## unit: sec
        
        ## note that we can return a lof of stuff
        ## get unnormed numpy list, same format (unnormalized after this output)
        # trajs_list_np_un = utils.get_np_trajs_list(trajs_list, do_unnorm=True, 
        #                                            normalizer=self.normalizer)
        
        ## TODO : implement below in torch 
        ## ranking of all the traj candiates based on the distance of ovlp parts

        # numpy 
        device = trajs_list[0].device

        trajs_list_np = [to_numpy(t) for t in trajs_list]
        
        s_idxs, dist_per_sam = utils.compute_ovlp_dist(trajs_list_np, 
                                                    self.len_ovlp_cd, self.is_spline)
        print(f"{dist_per_sam=}")
        
        # top_n = batch_size
        top_n = min(10 , batch_size)
        ## list, pick out the topn, from un-normed traj
        trajs_list_topn = utils.pick_top_n_trajs(trajs_list_np, s_idxs, top_n)
        ## np, un-normed, shape (B, tot_hzn, dim)
        trajs_list_topn_bl = self.tj_blder.blend_traj_lists(trajs_list_topn)

        # back to torch
        trajs_list_topn_bl = torch.cat([to_torch(t, device=device).unsqueeze(0) for t in trajs_list_topn_bl])

        ## pick one traj to execute
        if pick_type == 'first':
            pick_traj = trajs_list_topn_bl[0].unsqueeze(0)
        elif pick_type == 'rand':
            p_idx = np.random.randint(low=0, high=top_n)
            pick_traj = trajs_list_topn_bl[p_idx].unsqueeze(0)
        elif pick_type == 'all':
            pick_traj = trajs_list_topn_bl
        else : 
            raise NotImplementedError
        
        if results_ns : 
            trajs_list_topn_q_trajs = self.tj_blder.get_local_q_trajs(trajs_list_topn, device)
            # Keep logging artifacts on CPU to avoid holding large GPU tensors during inference.
            trajs_list_topn_cpu = {k: v.detach().cpu() for k, v in trajs_list_topn_q_trajs.items()}
            
            results_ns.update(
                trajs_list_topn =  trajs_list_topn_cpu,
                trajs_info = self.tj_blder.trajs_info
                # trajs_list_topn_bl = trajs_list_topn_bl,
                # trajs_list = trajs_list 
            )
            if return_chain :
                results_ns.update(
                    trajs_list_iter = chain
                )
            
        # if return_chain : # return chains for trajectory before merging is meaningless

        return pick_traj

    # def update_context_d(self, tj_cond) :
    #     context_d = tj_cond['context_d']
    #     context_d['progress'] = n_comp

    @torch.no_grad()
    def comp_pred_p_loop_n(self, ##
                        shape,  # (batch_size, horizon)
                        context_d,  
                        hard_conds,
                        n_comp,
                        return_chain=False,
                        ddim_eta=0.0,
                        ddim_skip_type = "uniform",
                        ddim_sampling_timesteps = None, 
                        t_start_guide = torch.inf,
                        n_diffusion_steps_without_noise = 0,
                        guide = None,
                        results_ns=None,
                        **sample_kwargs):
        """assume compose n trajectories"""
        # assert n_comp >= 2 
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,] if return_chain else None

        # assert len(hard_conds[0]) == shape[0]

        if self.use_ddim:
            sampling_timesteps = ddim_sampling_timesteps if ddim_sampling_timesteps is not None else self.n_diffusion_steps
            time_pairs = ddim_create_time_pairs(
                self.n_diffusion_steps, sampling_timesteps, ddim_skip_type, n_diffusion_steps_without_noise
            )
        else:
            raise NotImplementedError
        ## -----------------
        
        do_cond = sample_kwargs.get('do_cond', False)

        from tqdm import tqdm
        for k_step, (_time, _time_next) in enumerate(time_pairs):
            if _time == _time_next:
                continue
            if _time_next < 0:
                _time = 1
                _time_next = 0
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            t = torch.full((batch_size, self.horizon), _time, device=device, dtype=torch.long)
            t_next = torch.full((batch_size, self.horizon), _time_next, device=device, dtype=torch.long)

            ## iteratively denoise each sub traj
            for i_tj in range(n_comp):
                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=t[:,0],
                        t_1d_end=t[:,0], 
                        t_type='0', 
                        is_noisy=True,
                        context_d=context_d,
                        hard_conds=hard_conds,
                        progress=i_tj/n_comp,
                        # i_tj, n_comp
                        )
                    
                    tj_cond_p_i['do_cond'] = do_cond # True 
                    tj_cond_p_i['idx'] = i_tj
                    # if self.use_ddim:
                    
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
                                results_ns=results_ns,
                                **sample_kwargs,
                            ) 


                    x_p_list[i_tj] = x_p_i
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=t_next[:,0], # t[:,0]-1
                        t_1d_end=t[:,0], 
                        t_type='0', 
                        is_noisy=True,
                        context_d=context_d,
                        hard_conds={},
                        progress=i_tj/n_comp,
                        # i_tj, n_comp
                    )
                    
                    tj_cond_p_i['do_cond'] = do_cond # True
                    tj_cond_p_i['idx'] = i_tj
                    # if self.use_ddim:
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
                                results_ns=results_ns,
                                **sample_kwargs,
                            ) 
                    
                    x_p_list[i_tj] = x_p_i

                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=t_next[:,0], # t[:,0]-1
                        t_1d_end=t[:,0], 
                        t_type='0',
                        is_noisy=True,
                        context_d=context_d,
                        hard_conds=hard_conds,
                        progress=i_tj/n_comp,
                        # i_tj, n_comp
                        )
                    
                    tj_cond_p_i['do_cond'] = do_cond # True
                    tj_cond_p_i['idx'] = i_tj
                    # if self.use_ddim:
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
                                results_ns=results_ns,
                                **sample_kwargs,
                            ) 
                    x_p_list[i_tj] = x_p_i
            
            if return_chain:
                x_dfu_all.append([_ for _ in x_p_list])

        
        #### -----------
        if isinstance(guide, CostGuideManagerCompTrajectory) : 
            guide.reset_task_trajectory()

        ## Finished
        #pdb.set_trace()
        st_cond = self.split_hard_conds(hard_conds, is_start=True)
        end_cond = self.split_hard_conds(hard_conds, is_start=False)
        x_p_list[0] = apply_hard_conditioning(x_p_list[0],st_cond)
        x_p_list[-1] = apply_hard_conditioning(x_p_list[-1],end_cond)

        if return_chain:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
        
