import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes

from torch_robotics.environments.pb_diff_envs.rand_rec_group import RandRectangleGroup


class EnvRandMaze2D(EnvBase):

    def __init__(self, 
                 wall_locations, wall_hExts, limits, 
                 tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, 
                 sdf_cell_size=0.005, **kwargs):
        
        '''
        3 gaps: 
        a. gap between two walls to ensure connectivity
        b. gap between each wall and the four edges (same gap value as in a.)
        c. min gap between robot and wall
        we will pack wall info to a class and pass to the robot
        '''

        ## consider only rectangle wals
        self.wall_locations = wall_locations
        self.wall_hExts = wall_hExts
        self.num_walls = len(self.wall_locations)
        self.tensor_args = tensor_args

        # maze_size = np.array([5, 5]) 
        # robot_config = dict(maze_size=maze_size, min_to_wall_dist=0.01, collision_eps=0.02)
        # self.maze_size:np.ndarray = robot_config['maze_size']

        self.limits = torch.tensor(limits, **tensor_args)

        assert len(self.wall_locations) == len(self.wall_hExts)

        wall_field = MultiBoxField(centers = np.array(self.wall_locations),
                                   sizes = np.array(self.wall_hExts * 2),
                                   tensor_args = self.tensor_args)
        
        # sdf_cell_size needs to be multiplied by 2.5 2->5

        super().__init__(
            #limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            limits = self.limits,
            obj_fixed_list=[ObjectField(wall_field, "rand2d")],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size * 2.5,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=50)

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                "delta": 1e-2,
                "trust_region": True,
                "method": "cholesky",
            },
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError


# class EnvDense2DExtraObjects(EnvDense2D):

#     def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
#         obj_extra_list = [
#             MultiSphereField(
#                 np.array(
#                     [
#                         [-0.4, 0.1],
#                         [-0.075, -0.85],
#                         [-0.1, -0.1],
#                     ]
#                 ),
#                 np.array(
#                     [
#                         0.075,
#                         0.1,
#                         0.075,
#                     ]
#                 ),
#                 tensor_args=tensor_args,
#             ),
#             MultiBoxField(
#                 np.array(
#                     [
#                         [0.45, -0.1],
#                         [0.35, 0.35],
#                         [-0.6, -0.85],
#                         [-0.65, -0.25],
#                     ]
#                 ),
#                 np.array(
#                     [
#                         [0.2, 0.2],
#                         [0.1, 0.15],
#                         [0.1, 0.25],
#                         [0.15, 0.1],
#                     ]
#                 ),
#                 tensor_args=tensor_args,
#             ),
#         ]

#         super().__init__(
#             obj_extra_list=[ObjectField(obj_extra_list, "dense2d-extraobjects")], tensor_args=tensor_args, **kwargs
#         )


if __name__ == "__main__":

    env = EnvRandMaze2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    # env = EnvDense2DExtraObjects(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    # fig, ax = create_fig_and_axes(env.dim)
    # env.render(ax)
    # plt.show()

    # # Render sdf
    # fig, ax = create_fig_and_axes(env.dim)
    # env.render_sdf(ax, fig)

    # # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    # plt.show()
