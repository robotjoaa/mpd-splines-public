import numpy as np
import torch
from matplotlib import pyplot as plt

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_simple_2d import EnvSimple2D


from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


'''
    EnvSimple2DExtraObject with moving object field works
'''
class EnvDynSimple2DExtraObjects(EnvSimple2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [-0.15, 0.15],
                        [-0.075, -0.85],
                        [-0.1, -0.1],
                        [0.5, 0.35],
                        [-0.6, -0.85],
                        [0.05, 0.85],
                        [-0.8, 0.15],
                        [0.8, -0.8],
                    ]
                ),
                np.array(
                    [
                        0.05,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                    ]
                ),
                tensor_args=tensor_args,
            ),
            MultiBoxField(
                np.array(
                    [
                        [0.45, -0.1],
                        [-0.25, -0.5],
                        [0.8, 0.1],
                    ]
                ),
                np.array(
                    [
                        [0.15, 0.25],
                        [0.15, 0.25],
                        [0.15, 0.15],
                    ]
                ),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "dyn-simple2d-extraobjects")], tensor_args=tensor_args, **kwargs
        )
