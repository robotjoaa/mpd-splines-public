import torch, pdb, einops
import torch.nn as nn
from einops.layers.torch import Rearrange

import numpy as np
from .sml_temporal_dd_v1 import ResidualTemporalBlock_dd
#from comp_diffuser.sml_helpers import Traj_Time_Encoder
from mpd.models.diffusion_models import ContextModelGlobal
from .sml_helpers_small import SmallOverlapEncoder

from mpd import models
from mpd.models.layers.layers import (
    Downsample1d,
    Conv1dBlock,
    Upsample1d,
    ResidualTemporalBlock,
    TimeEncoder,
    MLP,
    group_norm_n_groups,
    LinearAttention,
    PreNorm,
    Residual,
)
from mpd.models.layers.layers_attention import SpatialTransformer
from mpd.models.diffusion_models.helpers import (
    print_color
)

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from .stgl_sml_temporal_cond_v2 import Unet1D_TjTi_Stgl_Cond_V2

# --------------------------------------------------------

class Unet1D_TjTi_Stgl_Cond_V3(Unet1D_TjTi_Stgl_Cond_V2):

    def __init__(
        self,
        context_model=None,
        **kwargs,
    ):
        """
        the UNet Based Denoiser Backbone of CompDiffuser
        use inpainting for the start and goal state;
        also support noisy-sample conditioning.

        train with dual cfg
        """
        super().__init__(context_model=context_model,
                         **kwargs)

        assert self.context_model is not None and isinstance(self.context_model, ContextModelGlobal)

    def forward(self, x, time,
                tj_cond: dict, warm_up=False, only_uncond=False):
        '''
            x : [ batch x horizon x transition ]
            time: [batch,]
            walls_loc: [batch, 6], 2D
            force_dropout: first 1/3 used to compute global, (drop local)
            third_fd: drop the conditions for the last 1/3 in the input batch 

            ### requirement for tj_cond
            - context_d (normalized): context_qs, context_ee_pose_goal, progress, delta_to_goal 
        '''
        if not warm_up :
            ### comp_diffuser 
            is_st_inpat = tj_cond['is_st_inpat'] ## torch tensor gpu
            is_end_inpat = tj_cond['is_end_inpat']
            ## sanity check
            b_size = x.shape[0]
            assert is_st_inpat.shape[0] == b_size and is_st_inpat.ndim == 1 \
                and is_st_inpat.dtype == torch.bool
            assert is_end_inpat.shape[0] == b_size and is_end_inpat.ndim == 1 \
                and is_end_inpat.dtype == torch.bool
            
            ## ----------------

            x = einops.rearrange(x, 'b h t -> b t h')

            t_feat = self.time_mlp(time) ## e.g., (B,64) ## TODO:
            
            # pdb.set_trace()

            ## obtain feature
            st_ovlp_is_drop = tj_cond['st_ovlp_is_drop']
            end_ovlp_is_drop = tj_cond['end_ovlp_is_drop']
            force_zero_end_ovlp = tj_cond.get('force_zero_end_ovlp', False)

            # reuse single computation for CFG triplets (global / uncond / local)
            use_cfg_triplet = (not only_uncond) and (b_size % 3 == 0)
            if use_cfg_triplet:
                base_bs = b_size // 3

                def _slice_cond(val):
                    if torch.is_tensor(val) and val.shape[0] == b_size:
                        return val[:base_bs]
                    return val

                context_d_full = tj_cond['context_d']
                needs_slice = any(torch.is_tensor(v) and v.shape[0] == b_size for v in context_d_full.values())
                context_d_base = (
                    {k: _slice_cond(v) if torch.is_tensor(v) and v.shape[0] == b_size else v
                     for k, v in context_d_full.items()}
                    if needs_slice else context_d_full
                )
                global_emb_base = self.context_model(**context_d_base)

                if st_ovlp_is_drop is not None:
                    st_ovlp_feat_base = self.st_ovlp_model(
                        tj_cond['st_ovlp_traj'][:base_bs],
                        time=tj_cond['st_ovlp_t'][:base_bs],
                    )
                    assert st_ovlp_is_drop.dtype == torch.bool
                    st_ovlp_feat_base[st_ovlp_is_drop[:base_bs]] = 0.
                    assert not torch.logical_and(~st_ovlp_is_drop[:base_bs], is_st_inpat[:base_bs]).any()
                else:
                    st_ovlp_feat_base = torch.zeros((base_bs, self.st_ovlp_model.out_dim), device=x.device)

                if self.end_ovlp_model is not None:
                    if force_zero_end_ovlp:
                        end_ovlp_feat_base = torch.zeros((base_bs, self.end_ovlp_model.out_dim), device=x.device)
                    elif tj_cond['end_ovlp_is_drop'] is not None:
                        end_ovlp_feat_base = self.end_ovlp_model(
                            tj_cond['end_ovlp_traj'][:base_bs],
                            time=tj_cond['end_ovlp_t'][:base_bs],
                        )
                        end_ovlp_feat_base[end_ovlp_is_drop[:base_bs]] = 0.
                        assert end_ovlp_is_drop.dtype == torch.bool
                        assert not torch.logical_and(~end_ovlp_is_drop[:base_bs], is_end_inpat[:base_bs]).any()
                    else:
                        end_ovlp_feat_base = torch.zeros((base_bs, self.end_ovlp_model.out_dim), device=x.device)
                else:
                    end_ovlp_feat_base = None

                zero_global = torch.zeros_like(global_emb_base)
                global_emb = torch.cat((global_emb_base, zero_global, zero_global), dim=0)

                zero_st = torch.zeros_like(st_ovlp_feat_base)
                st_ovlp_feat = torch.cat((zero_st, zero_st, st_ovlp_feat_base), dim=0)

                if end_ovlp_feat_base is not None:
                    zero_end = torch.zeros_like(end_ovlp_feat_base)
                    end_ovlp_feat = torch.cat((zero_end, zero_end, end_ovlp_feat_base), dim=0)
                else:
                    end_ovlp_feat = None
            else:
                global_emb = self.context_model(**tj_cond['context_d'])

                if st_ovlp_is_drop is not None: ##
                    st_ovlp_feat = self.st_ovlp_model(tj_cond['st_ovlp_traj'], 
                                            time=tj_cond['st_ovlp_t'])
                    assert len(st_ovlp_is_drop) == len(st_ovlp_feat)
                    assert st_ovlp_is_drop.dtype == torch.bool ## a numpy array
                    st_ovlp_feat[ st_ovlp_is_drop ] = 0.
                    # (~st_ovlp_is_drop) == 
                    assert not torch.logical_and(~st_ovlp_is_drop, is_st_inpat).any() ## must be false
                    
                else:
                    ## no cond if None
                    # st_ovlp_feat = torch.zeros_like(st_ovlp_feat)
                    st_ovlp_feat = torch.zeros( (x.shape[0], self.st_ovlp_model.out_dim), device=x.device)

                if self.end_ovlp_model is not None: 
                    if force_zero_end_ovlp:
                        end_ovlp_feat = torch.zeros((x.shape[0], self.end_ovlp_model.out_dim), device=x.device)

                    elif tj_cond['end_ovlp_is_drop'] is not None:
                        end_ovlp_feat = self.end_ovlp_model(tj_cond['end_ovlp_traj'],
                                                            time=tj_cond['end_ovlp_t'])
                        end_ovlp_feat[ tj_cond['end_ovlp_is_drop'] ] = 0.
                        assert end_ovlp_is_drop.dtype == torch.bool
                        assert not torch.logical_and(~end_ovlp_is_drop, is_end_inpat).any()
                    else:
                        # end_ovlp_feat = torch.zeros_like(end_ovlp_feat)
                        end_ovlp_feat = torch.zeros( (x.shape[0], self.end_ovlp_model.out_dim), device=x.device)
                else : 
                    end_ovlp_feat = None

            ## Here we create corresponding condition feature to let the model know if we actually overwrite!
            if self.inpaint_token_type == 'const':
                # (B,token_dim)
                st_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
                num_st_inpt = torch.sum(is_st_inpat).item()
                ## assign value
                st_token[is_st_inpat] = self.st_use_inpaint_token.repeat( (num_st_inpt, 1) )
                st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat( (b_size - num_st_inpt, 1) )
                # pdb.set_trace()
                ### from Here Oct 10 14:38

                end_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
                num_end_inpt = torch.sum(is_end_inpat).item()
                end_token[is_end_inpat] = self.end_use_inpaint_token.repeat( (num_end_inpt, 1) )
                end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat( (b_size - num_end_inpt, 1) )

                st_token = self.st_inpaint_model(st_token)
                end_token = self.end_inpaint_model(end_token)
            else : 
                raise NotImplementedError

            ## NOTE: for one side, we can only either do inpainting or ovlp conditioning

            # use force_dropout during training 
            # if force_dropout and third_fd :
            #     # pdb.set_trace() ## important: do not drop the st_token?
            #     assert not self.training
            #     if third_fd:

            if not only_uncond : 
                if not use_cfg_triplet:
                    assert NotImplementedError, "Using inefficient method"
                    # b_s = len(st_ovlp_feat)
                    # # drop the second half
                    # assert b_s % 3 == 0
                    # tmp_batch = int(b_s)//3
                    # # first 1/3 is global
                    # # second 1/3 is uncond
                    # # last 1/3 is local
                    # st_ovlp_feat[:2*tmp_batch] = 0.
                    # if self.end_ovlp_model is not None : 
                    #     end_ovlp_feat[:2*tmp_batch] = 0. 
                    # else :
                    #     end_ovlp_feat = None
                    # global_emb[tmp_batch:] = 0.
            else : 
                st_ovlp_feat = torch.zeros( (x.shape[0], self.st_ovlp_model.out_dim), device=x.device)
                end_ovlp_feat = None
                if self.end_ovlp_model is not None : 
                    end_ovlp_feat = torch.zeros( (x.shape[0], self.end_ovlp_model.out_dim), device=x.device)
                global_emb = torch.zeros( (x.shape[0], self.context_model.out_dim), device=x.device)
            ## e.g., B, time_emb_dim+128+128
            if end_ovlp_feat is not None : 
                feat_list = [t_feat, st_ovlp_feat, end_ovlp_feat, st_token, end_token, global_emb]
            else : 
                feat_list = [t_feat, st_ovlp_feat, st_token, global_emb]
            t_feat = torch.cat(feat_list, dim=-1)

            
            # t_feat : b cond

        else : # warmup
            assert self.state_dim == x.shape[-1]
            b_size = x.shape[0]
            x = einops.rearrange(x, 'b h t -> b t h')
            t_feat = torch.randn((b_size, self.tot_cond_dim), device=x.device)

        h = []

        # pdb.set_trace()

        for resnet, resnet2, downsample in self.downs:

            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, t_feat)
        x = self.mid_block2(x, t_feat)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
