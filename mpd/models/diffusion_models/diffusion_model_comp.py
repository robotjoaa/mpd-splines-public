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
)

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, clip_grad_by_norm, clip_grad_by_value

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class CompDiffusionModel(nn.Module, ABC):
    """
    A compositional variant of GaussianDiffusionModel.

    - Uses the existing classifier-guidance aware sampling loops from the base
      class (ddpm/ddim) to generate individual trajectory segments.
    - Composes `n_components` segments with a fixed-length overlap and blends
      those overlaps into a single long trajectory.

    Notes:
    * The current implementation expects the caller to prepare a list of
      `hard_conds` dicts, one per component. This keeps us aligned with the
      existing API and avoids introducing a new CFG-style conditioner. A higher
      level wrapper (e.g., a policy/planner) can construct those dicts using
      start/goal/overlap context similar to comp_diffuser.
    * Overlap blending is adapted from `comp_diffuser/traj_blender.py`, but kept
      lightweight and numpy-based here.
    """

    def __init__(
        self,
        denoise_fn=None,
        variance_schedule="cosine",
        n_diffusion_steps=100,
        clip_denoised=True,
        predict_epsilon=False,
        loss_type="l2",
        # context_model=None,
        horizon=None,
        len_ovlp_cd=None,
        comp_config={},
        **kwargs,
    ):
        super().__init__()
        #self.model = MyDataParallel(denoise_fn)
        self.model = denoise_fn # Unet1D_TjTi_Stgl_Cond_V2

        # context_model defined inside denoise_fn
        # self.context_model = context_model

        self.n_diffusion_steps = n_diffusion_steps

        self.state_dim = self.model.state_dim

        self.horizon = horizon # sm_horizon
        self.len_ovlp_cd = len_ovlp_cd 
        if variance_schedule == "cosine":
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == "exponential":
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type]()

        ### for compose 
        self.comp_config = comp_config
        self.use_ddim = True
        self.guide_mode = comp_config.get('guide_mode', 'default') # default (cg), cfg, hybrid
        self.len_overlap = self.comp_config['len_ovlp_cd']
        self.hzn_step_size = self.n_diffusion_steps  # TODO: revisit; placeholder for API symmetry
        # self.use_cfg = self.comp_config.get('use_cfg', False)
        self.condition_guidance_w = self.comp_config.get('condition_guidance_w', 2.0)
        self.tr_inpat_prob = self.comp_config['tr_inpat_prob']
        self.tr_ovlp_prob = self.comp_config['tr_ovlp_prob']
        self.tr_all_drop_prob = self.comp_config['tr_1side_drop_prob']
        print_color(f"{self.comp_config['tr_1side_drop_prob']=}")

        self.train_st_only = comp_config.get('train_st_only', False)

        # self.tr_no_ovlp_none = self.comp_config.get('tr_no_ovlp_none', False)
        print_color(f"{self.train_st_only=}")
        assert self.tr_inpat_prob + self.tr_ovlp_prob == 1.0
        # self.blend_type = "exponential"
        # self.blend_beta = 3

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t_2d, x0):
        if self.predict_epsilon : 
            return x0
        else : 
            return (
                extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t - x0) / \
                        extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape
                )


    def predict_start_from_noise(self, x_t, t_2d, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        assert t_2d.ndim == 2
        # pdb.set_trace()
        ## x_t: B, H, dim
        if self.predict_epsilon:
            ## directly switch to 2d version
            return (
                ## B,H,1 * B,H,dim
                extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t -
                extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        '''from x_0 and x_t to x_{t-1}
        see equeation 6 and 7
        '''
        # pdb.set_trace() ## check buffer dim
        ## directly, e.g., 10,384,6
        posterior_mean = (
            extract_2d(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_2d(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        ## now 2D, not 1D in vanilla diffusion
        ## both two e.g., [B=10, H=384, 1]
        posterior_variance = extract_2d(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_2d(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    ### TODO : implement ddpm for classifier guidance
    #def p_mean_variance(self, x, hard_conds, context_d, t, prior_weight_with_guide=1.0, **kwargs):
    #def p_sample(self, x, tj_cond, timesteps, mask_same_t=None):
    #def p_sample_loop()
    '''
    g_cond : 
    do_cond 
    traj_full
    t_type
    stgl_cond
    '''
    
    def split_hard_conds(self, hard_conds, is_start = True, max_cond = 3) :
        res = {}
        if is_start : 
            for i, v in hard_conds.items() : 
                if i < max_cond :
                    res[i] = v
    
        else : 
            for i, v in hard_conds.items() : 
                if i > max_cond :
                    res[i] = v

        return res

    # called from render_samples
    # self.get_tj_cond(x, hard_conds, context_d, t)
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
                hard_conds=end_cond,
                )
            tj_cond['do_cond'] = True


        elif g_cond['do_cond'] == False:
            ## drop everything
            tj_cond = dict(st_ovlp_is_drop=None, end_ovlp_is_drop=None, 
                            is_st_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            is_end_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            )
            tj_cond['do_cond'] = False
        else: 
            raise NotImplementedError

        
        tj_cond['context_d'] = context_d

        ## x is also modified!
        return x, tj_cond

    @torch.no_grad()
    def conditional_sample(self, g_cond, horizon=None, batch_size=1, method="ddpm", **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        # assert False, 'not finished'
        #device = self.betas.device
        #batch_size = len(g_cond['traj_full']) ## TODO: check

        # if tj_cond['st_ovlp_is_drop'] is not None:
        #     batch_size = len(tj_cond['st_ovlp_traj'])
        # elif  tj_cond['end_ovlp_is_drop'] is not None:
        #     batch_size = len(tj_cond['end_ovlp_traj'])
        # else:
        #     assert False
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
        
    # @torch.no_grad()
    # def sample_unCond(self, batch_size, *args, horizon=None, **kwargs):
    #     '''
    #         batch_size : int
    #     '''
    #     device = self.betas.device
    #     horizon = horizon or self.horizon
    #     shape = (batch_size, horizon, self.observation_dim)
    #     g_cond = dict(do_cond=False)
    #     ## placeholder

    #     return self.p_sample_loop(shape, g_cond, *args, **kwargs)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        shape_x,
        # hard_conds,
        g_cond = None, 
        context_d=None,
        return_chain=False,
        return_chain_x_recon=False,
        ddim_eta=0.0,
        ddim_skip_type="uniform",
        ddim_sampling_timesteps=None,
        t_start_guide=torch.inf,
        scale_grad_by_one_minus_alpha=False,
        guide=None,
        guide_lr=0.05,
        n_guide_steps=1,
        max_perturb_x=0.1,
        clip_grad=False,
        clip_grad_rule="value",  # 'norm', 'value'
        max_grad_norm=1.0,  # clip the norm of the control point gradients
        max_grad_value=1.0,  # clip the control point gradients
        n_diffusion_steps_without_noise=0,
        ddim_scale_grad_prior=1.0,
        compute_costs_with_xrecon=False,
        results_ns=None,
        **sample_kwargs,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226

        device = self.betas.device
        batch_size = shape_x[0]
        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = ddim_sampling_timesteps if ddim_sampling_timesteps is not None else total_timesteps
        assert (
            sampling_timesteps <= total_timesteps
        ), f"sampling_timesteps={sampling_timesteps} > total_timesteps={total_timesteps}"

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        time_pairs = ddim_create_time_pairs(
            total_timesteps, sampling_timesteps, ddim_skip_type, n_diffusion_steps_without_noise
        )

        clip_grad_fn = lambda x: x
        if clip_grad and clip_grad_rule == "norm":
            clip_grad_fn = partial(clip_grad_by_norm, max_grad_norm=max_grad_norm)
        elif clip_grad and clip_grad_rule == "value":
            clip_grad_fn = partial(clip_grad_by_value, max_grad_value=max_grad_value)

        x = torch.randn(shape_x, device=device)
        # x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None
        chain_x_recon = [x] if return_chain_x_recon else None

        
        # context_emb = None
        # if context_d is not None:
        #     context_emb = self.context_model(**context_d)

        for k_step, (_time, _time_next) in enumerate(time_pairs):
            if _time == _time_next:
                continue
            if _time_next < 0:
                _time = 1
                _time_next = 0
            # should be 2d (batch_size, self.horizon)
            t = torch.full((batch_size, self.horizon), _time, device=device, dtype=torch.long)
            # prev_timesteps
            t_next = torch.full((batch_size, self.horizon), _time_next, device=device, dtype=torch.long)
            

            x, tj_cond = self.get_tj_cond(x, g_cond, context_d, t)

            assert guide is None # guide only applied when using ddim_sample alone

            tmp = self.ddim_sample(
                    x = x, 
                    tj_cond = tj_cond, 
                    t = t,
                    t_next = t_next, # 2d
                    sampling_timesteps=sampling_timesteps,
                    current_step = k_step, 
                    ddim_eta = ddim_eta, 
                    t_start_guide= t_start_guide,
                    scale_grad_by_one_minus_alpha=scale_grad_by_one_minus_alpha,
                    guide=guide,
                    guide_lr=guide_lr,
                    n_guide_steps=n_guide_steps,
                    max_perturb_x=max_perturb_x,
                    clip_grad_fn=clip_grad_fn,
                    ddim_final_add_noise= _time_next >= 1,
                    return_x_recon=return_chain_x_recon,
                    ddim_scale_grad_prior=ddim_scale_grad_prior,
                    compute_costs_with_xrecon=compute_costs_with_xrecon,
                    results_ns=results_ns,
                    g_cond=g_cond,
                    **sample_kwargs,
                )
            if return_chain_x_recon : 
                x, x_recon = tmp
            else :
                x = tmp

            if return_chain:
                chain.append(x.clone())
            if return_chain_x_recon:
                chain_x_recon.append(x_recon.clone())

        chains = []
        if return_chain:
            chain = torch.stack(chain, dim=1)
            chains.append(chain)

        if return_chain_x_recon:
            chain_x_recon = torch.stack(chain_x_recon, dim=1)
            chains.append(chain_x_recon)

        return x, *chains

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
        **sample_kwargs,
    ) : 

        alpha = extract_2d(self.alphas_cumprod, t, x.shape)
        alpha_next = extract_2d(self.alphas_cumprod, t_next, x.shape)
        # compute std_dev_t
        sigma = ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma**2).sqrt()

        # denoising noise
        with TimerCUDA() as t_generator:

            if self.guide_mode != "default" and tj_cond['do_cond']: # cfg
                x_2, t_2d_2, tj_cond_2 = batch_repeat_tensor_in_dict(x, t, tj_cond, n_rp=2)
                assert (t_2d_2[0] == t_2d_2[0,0]).all(), 'sanity check'
                t_1d_2 = t_2d_2[:, 0]
                out = self.model(x_2, t_1d_2, tj_cond_2, force_dropout=True, half_fd=True)
                out_cd = out[:len(x), :, :]
                out_uncd = out[len(x):, :, :]
                model_out = out_uncd + self.condition_guidance_w * (out_cd - out_uncd)
                
            else : # original 
                model_out = self.small_model_pred(x, t, tj_cond)

        if results_ns is not None:
            results_ns.t_generator += t_generator.elapsed

        beta_prod_t = 1 - alpha
        grad_prior = model_out
        
        def update_x(_x, _grad_prior):
            _x_recon = self.predict_start_from_noise(_x, t=t, noise=_grad_prior)
            if self.clip_denoised:
                _x_recon.clamp_(-1.0, 1.0)
            else:
                assert RuntimeError()

            # if use classifier guidance, use modified grad 
            if self.guide_mode != "cfg" : 
                # default self.predict_epsilon == True
                # _pred_noise = _grad_prior (modified gradient from classifier guidance)
                _pred_noise = self.predict_noise_from_start(_x, t=t, x0=_grad_prior)
            else : 
                if use_clipped_model_output : # always true
                    _pred_noise = (_x - alpha.sqrt() * _x_recon) / beta_prod_t.sqrt()
                else : 
                    _pred_noise = self.predict_noise_from_start(_x, t=t, x0=_x_recon)

            _x = _x_recon * alpha_next.sqrt() + c * _pred_noise
            # _x = apply_hard_conditioning(_x, hard_conds)
            return _x, _x_recon

        # Modify the noise if guidance is active
        # https://arxiv.org/pdf/2105.05233.pdf - Algorithm 2
        if guide is not None and (sampling_timesteps - current_step <= t_start_guide) : 
            with TimerCUDA() as t_guide:
                x_start = x.clone()
                for k_gd in range(n_guide_steps):
                    grad_prior_weighted = grad_prior * ddim_scale_grad_prior
                    if compute_costs_with_xrecon:
                        raise NotImplementedError("compute_costs_with_xrecon is not implemented")
                    else:
                        #grad_guide = guide(x, context_d=context_d)
                        grad_guide = guide(x, tj_cond) # tj_cond must include time range for guidance, g_cond

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
            raise NotImplementedError

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

    def q_sample(self, x_start, t_2d, noise=None):
        '''add noise to x_t from x_0
        mask_no_noise: bool (B,H), if True, then do not add any noise to x_start
        '''
        assert t_2d.ndim == 2 # B, horizon

        if noise is None:
            noise = torch.randn_like(x_start)
        
        ## vanilla diffusion: (B, 1, 1) * x_start: (B, H, dim) e.g., [32, 128, 6]
        # sample = (
        #     extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        #     extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        # )

        ## Ours: t (B, H, 1) * x_start (B, H, dim)
        q_coef1 = extract_2d(self.sqrt_alphas_cumprod, t_2d, x_start.shape)
        q_coef2 = extract_2d(self.sqrt_one_minus_alphas_cumprod, t_2d, x_start.shape)

        ## when t=0, 0.9999 * x_start + 0.0137 * noise 
        sample = q_coef1 * x_start + q_coef2 * noise

        return sample

    def p_losses(self, x_start, x_noisy, noise, t_2d, tj_cond):
        
        batch_loss_w = torch.ones_like(x_start[:,:,:1])
        batch_loss_w[tj_cond['is_st_inpat'],0] = 0.
        batch_loss_w[tj_cond['is_end_inpat'],self.horizon-1] = 0.
        assert x_start.shape[1] == self.horizon 

        # diffusion model
        #x_recon = self.model(x_noisy, t, context_emb)
        #x_recon = apply_hard_conditioning(x_recon, hard_conds)
        # hard_conds considered with batch_loss_w
        x_recon = self.small_model_pred(x_noisy, t_2d, tj_cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, ext_loss_w=batch_loss_w)
        else:
            loss, info = self.loss_fn(x_recon, x_start, ext_loss_w=batch_loss_w)

        return loss, info
    # should be this format to match GaussianDiffusionLoss
    #def loss(self, x, context_d, *args):
    def loss(self, x_clean, context_d, hard_conds):
        # cond_st_gl is start, end hard conds. 
        # it can be infered from context_d

        batch_size = x_clean.shape[0]
        # pdb.set_trace() ## check x dim
        t_1d = torch.randint(0, self.n_diffusion_steps, (batch_size, 1), device=x_clean.device).long()
        ## B,H
        t_2d = t_1d.expand(-1, self.horizon)
        #t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

        noise = torch.randn_like(x_clean)
        x_noisy = self.q_sample(x_start=x_clean, t_2d=t_2d, noise=noise)

        ## choose training conditioning scheme
        if self.train_st_only:
            raise NotImplementedError
            x_noisy, tj_cond = self.create_train_tj_cond_prev_only(
                x_clean, x_noisy, t_1d[:, 0], cond_st_gl, is_rand=True)
        else:
            x_noisy, tj_cond = self.create_train_tj_cond(
                x_clean, x_noisy, t_1d[:, 0], t_1d[:,0].clone(), context_d, hard_conds, is_rand=True)
        
        diffuse_loss, info = self.p_losses(x_clean[:, :, :], x_noisy, noise, t_2d, tj_cond)
            
        total_loss = diffuse_loss
        return total_loss, info

# ------------------------------------------ warmup ------------------------------------------#
    @torch.no_grad()
    def warmup(self, shape_x, device="cuda"):
        batch_size, n_support_points, state_dim = shape_x
        x = torch.randn(shape_x, device=device)
        t = make_timesteps(batch_size, 1, device)
        # context_emb = None
        #if self.context_model is not None:
        #context_emb = torch.randn(batch_size, self.context_model.out_dim, device=device)
        tj_cond = {}
        self.model(x, t, tj_cond, warm_up = True)


    def extract_ovlp_from_full(self, x: torch.Tensor):
        """x: either np or tensor"""
        st_traj = x[:, :self.len_ovlp_cd, :]
        end_traj = x[:, -self.len_ovlp_cd:, :]
        if torch.is_tensor(st_traj):
            assert torch.is_tensor(end_traj)
            st_traj = st_traj.detach().clone()
            end_traj = end_traj.detach().clone()
        else:
            assert type(st_traj) == np.ndarray
            assert type(end_traj) == np.ndarray

        return st_traj, end_traj


    '''
    original cond_st_gl
    return {
                0: observations[0],
                self.horizon - 1: observations[-1],
            }
    '''
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

        if self.guide_mode != "default" : 
            all_drop_prob = self.tr_all_drop_prob ## 0.15

            st_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob
            end_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob

            ## set those to be dropout to False, so no condition at all for 0.15 * bs
            st_cd_use_ovlp[st_is_all_drop] = False
            st_cd_use_inpat[st_is_all_drop] = False

            end_cd_use_ovlp[end_is_all_drop] = False
            end_cd_use_inpat[end_is_all_drop] = False

        ###### TODO:
        ###### We need to modify the corresponding samples for inpainting
        # pdb.set_trace() ## check cond_st_gl --> 0: (B,2), horizon-1:(B,2)

        cond_st = self.split_hard_conds(hard_conds, is_start=True)
        for k,v in cond_st.items() : 
            cond_st[k] = v[st_cd_use_inpat]
        x_noisy[ st_cd_use_inpat ] = apply_hard_conditioning( x_noisy[ st_cd_use_inpat ], cond_st)
        ####
        cond_end = self.split_hard_conds(hard_conds, is_start=False)
        # {self.horizon-1: hard_conds[self.horizon-1][end_cd_use_inpat]}
        for k,v in cond_end.items() : 
            cond_end[k] = v[end_cd_use_inpat]
        x_noisy[ end_cd_use_inpat ] = apply_hard_conditioning( x_noisy[ end_cd_use_inpat ], cond_end)

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
            'context_d' : context_d
        }
        # pdb.set_trace() ## TODO: check
        
        if self.guide_mode == "default" : # no drop
            tj_cond['st_ovlp_is_drop'] = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
            tj_cond['end_ovlp_is_drop'] = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

        return x_noisy, tj_cond

    def small_model_pred(self, x, t_2d, tj_cond: dict, ) -> torch.Tensor:
        
        ## simple model should take in only 1d
        assert (t_2d[0] == t_2d[0,0]).all(), 'sanity check'
        t_1d = t_2d[:, 0]
        pred_out = self.model(x, t_1d, tj_cond)

        return pred_out

    def get_total_hzn(self, num_comp):
        return num_comp * self.horizon - \
                    (num_comp - 1) * self.len_ovlp_cd

    def create_eval_tj_cond(self, 
                            x_et: torch.Tensor, ## x eval t
                            st_traj, # : torch.Tensor, 
                            end_traj,
                            t_1d_st: torch.Tensor, 
                            t_1d_end: torch.Tensor, 
                            t_type: str,
                            is_noisy: bool,
                            context_d:dict,
                            hard_conds:dict):
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
        if 0 in hard_conds.keys():
            # pdb.set_trace()
            assert st_traj is None
            st_cond = self.split_hard_conds(hard_conds, is_start=True)
            x_et = apply_hard_conditioning(x_et, st_cond)
            is_st_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
        else:
            is_st_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)



        
        hzn_minus1 = self.horizon - 1
        if end_traj == None:
            end_is_drop = None # no feature
        else:
            end_is_drop = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device) # no drop

        
        ## hzn_minus1 cannot be in hard_conds eventhough it is last segment
        ## when use ee_pose_goal 

        ## do inpainting conditioning
        if hzn_minus1 in hard_conds.keys():
            # pdb.set_trace()
            assert end_traj is None
            end_cond = self.split_hard_conds(hard_conds, is_start=False)
            x_et = apply_hard_conditioning(x_et, end_cond)
            is_end_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
        else:
            # end_traj can be None here
            is_end_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

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
            'context_d': context_d
        }
        # pdb.set_trace()

        return x_et, tj_cond

    @torch.no_grad()
    def comp_pred_p_loop_n(self, ##
                        shape,  # (batch_size, horizon)
                        context_d,  
                        hard_conds,
                        n_comp,
                        return_chain=False,
                        return_chain_x_recon=False,
                        ddim_eta=0.0,
                        ddim_skip_type = "uniform",
                        ddim_sampling_timesteps = None, 
                        t_start_guide = torch.inf,
                        n_diffusion_steps_without_noise = 0,
                        guide = None,
                        return_diffusion=False,
                        results_ns=None,
                        **sample_kwargs):
        """assume compose n trajectories"""
        assert n_comp >= 2 
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(hard_conds[0]) == shape[0]

        if self.use_ddim:
            sampling_timesteps = ddim_sampling_timesteps if ddim_sampling_timesteps is not None else self.n_diffusion_steps
            time_pairs = ddim_create_time_pairs(
                self.n_diffusion_steps, sampling_timesteps, ddim_skip_type, n_diffusion_steps_without_noise
            )
        else:
            raise NotImplementedError
        ## -----------------

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
                        hard_conds=hard_conds
                        # i_tj, n_comp
                        )
                    
                    tj_cond_p_i['do_cond'] = False # True

                    # if self.use_ddim:
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
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
                        # i_tj, n_comp
                    )
                    
                    tj_cond_p_i['do_cond'] = False # True
                    
                    # if self.use_ddim:
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
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
                        hard_conds=hard_conds
                        # i_tj, n_comp
                        )
                    
                    tj_cond_p_i['do_cond'] = False # True

                    # if self.use_ddim:
                    x_p_i = self.ddim_sample(x_p_i, tj_cond_p_i, t, t_next, 
                                sampling_timesteps=sampling_timesteps,
                                current_step = k_step, 
                                t_start_guide= t_start_guide, 
                                ddim_eta=ddim_eta,
                                guide = guide,
                                ddim_final_add_noise= (_time_next >= 1),
                                **sample_kwargs,
                            ) 
                    x_p_list[i_tj] = x_p_i
            
            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
        
        ## Finished
        st_cond = self.split_hard_conds(hard_conds, is_start=True)
        end_cond = self.split_hard_conds(hard_conds, is_start=False)
        x_p_list[0] = apply_hard_conditioning(x_p_list[0],st_cond)
        x_p_list[-1] = apply_hard_conditioning(x_p_list[-1],end_cond)

        ### TODO : change to return_chain, return_x_recons
        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
        