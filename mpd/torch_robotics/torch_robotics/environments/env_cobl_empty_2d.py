import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvCoblEmpty2D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):

        limits = torch.tensor([[-1.0, -1.2], [2.0, 1.0]], **tensor_args)

        super().__init__(
            limits= limits,  # environments limits
            obj_fixed_list=None,
            obj_extra_list=None,
            precompute_sdf_obj_fixed=False,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs,
        )

if __name__ == "__main__":
    env = EnvCoblEmpty2D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()