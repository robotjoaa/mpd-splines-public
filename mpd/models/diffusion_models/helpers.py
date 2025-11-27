import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_robotics.torch_utils.torch_utils import to_torch


# helpers from comp_diffuser
from colorama import Fore

def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)

def extract_2d(a, t, x_shape):
    """
    extract to t, to two dimension, e.g., return (B, H, 1)
    """
    assert a.ndim == 1 and t.ndim == 2
    b, h, *_ = t.shape
    ## NOTE: when t is also tensor, will create a new tensor
    out = a[t]
    # pdb.set_trace()
    out = out.reshape(b, h, *((1,) * (len(x_shape) - 2)))
    ## out: B, H, 1
    # pdb.set_trace()

    return out
    
def batch_repeat_tensor_in_dict(x: torch.Tensor, t_2d: torch.Tensor, cond_dd: dict, n_rp: int):
	'''
	FIXED: Return a new dict rather than modified the original Dict!
	'''
	if x is not None:
		## B H D
		x = x.repeat( (n_rp, 1, 1) )
	if t_2d is not None:
		assert t_2d.ndim == 2
		t_2d = t_2d.repeat( (n_rp, 1,) )
	
	new_dd = {}
	for k in cond_dd.keys():
		if torch.is_tensor(cond_dd[k]):
			new_dd[k] = cond_dd[k].repeat(   [n_rp] + [1,] * len(cond_dd[k].shape[1:])  )
		elif type(cond_dd[k]) == np.ndarray:
			# dd[k]
			new_dd[k] = einops.repeat(cond_dd[k], 'b ... -> (rr b) ...', rr=n_rp )
		else:
			new_dd[k] = cond_dd[k]
			assert type(cond_dd[k]) in [bool, type(None)]

			
	return x, t_2d, new_dd

# -----------------------------------------------------------------------------#
# ---------------------------- variance schedules -----------------------------#
# -----------------------------------------------------------------------------#


def linear_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_diffusion_steps)


def quadratic_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, n_diffusion_steps) ** 2


def sigmoid_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, n_diffusion_steps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = n_diffusion_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=a_min, a_max=a_max)
    return to_torch(betas_clipped, dtype=dtype)


def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # exponential increasing noise from t=0 to t=T
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    beta_start = to_torch(beta_start)
    beta_end = to_torch(beta_end)
    a = 1 / n_diffusion_steps * torch.log(beta_end / beta_start)
    return beta_start * torch.exp(a * x)


def constant_fraction_beta_schedule(n_diffusion_steps):
    # exponential increasing noise from t=0 to t=T
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    return 1 / (n_diffusion_steps - x + 1)


def variance_preserving_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # Works only with a small number of diffusion steps
    # https://arxiv.org/abs/2112.07804
    # https://openreview.net/pdf?id=AHvFDPi-FA
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    alphas = torch.exp(
        -beta_start * (1 / n_diffusion_steps) - 0.5 * (beta_end - beta_start) * (2 * x - 1) / (n_diffusion_steps**2)
    )
    betas = 1 - alphas
    return betas


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        if self.weights is not None:
            weighted_loss = (loss * self.weights).mean()
        else:
            weighted_loss = loss.mean()
        return weighted_loss, {}


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}
