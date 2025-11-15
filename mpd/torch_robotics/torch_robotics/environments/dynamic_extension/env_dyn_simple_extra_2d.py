"""
EnvSimple2D with dynamic extra objects using the wrapper-based EnvDynBase pattern.

This module demonstrates using MovingObjectField instances with a standard environment.
The dynamic objects are automatically handled by EnvDynBase's wrapper pattern.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_simple_2d import EnvSimple2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    MovingObjectField,
    LinearTrajectory
)
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvDynSimple2DExtraObjects(EnvDynBase):
    """
    EnvSimple2D environment with moving obstacles using the wrapper pattern.

    This class wraps EnvSimple2D (which inherits from EnvBase) and adds moving obstacles
    using MovingObjectField instances. The wrapper automatically handles time-dependent
    rendering and SDF computation.
    """

    def __init__(
        self,
        time_range=(0.0, 10.0),
        k_smooth=0.05,
        smoothing_method="Quadratic",
        no_smoothing=False,
        moving_object_trajectories=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs
    ):
        """
        Args:
            time_range: (t_min, t_max) for moving obstacles
            k_smooth: Smoothness parameter for smooth SDF union
            smoothing_method: "Quadratic" or "LSE"
            moving_object_trajectories: Optional dict with trajectory configs for moving objects
                                       e.g., {"sphere": {"start": [1, -1], "end": [-1, 1]},
                                              "box": {"start": [-1, 1], "end": [1, -1]}}
                                       If None, uses default hard-coded trajectories
            tensor_args: Tensor device and dtype
            **kwargs: Additional arguments passed to EnvSimple2D
        """

        # Create the base static environment (EnvSimple2D inherits from EnvBase)
        # We'll pass this as the wrapped environment through the limits and obj_fixed_list
        base_env = EnvSimple2D(tensor_args=tensor_args, **kwargs)

        # Create moving obstacles with trajectories
        # Moving sphere 1: Diagonal trajectory (2D primitives, 3D trajectory)
        prim_sphere = MultiSphereField(
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
            )

        # Configure sphere trajectory
        if moving_object_trajectories and "sphere" in moving_object_trajectories:
            sphere_config = moving_object_trajectories["sphere"]
            traj_sphere = self._create_trajectory_from_config(
                sphere_config, time_range, tensor_args
            )
        else:
            # Default trajectory
            traj_sphere = LinearTrajectory(
                keyframe_times=[time_range[0], time_range[1]],
                keyframe_positions=[
                    [1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0]
                ],
                tensor_args=tensor_args
            )

        moving_sphere = MovingObjectField(
            primitive_fields=[prim_sphere],
            trajectory=traj_sphere,
            name="dyn-simple2d-sphere"
        )

        # Moving box: Opposite diagonal trajectory
        prim_box = MultiBoxField(
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
        )

        # Configure box trajectory
        if moving_object_trajectories and "box" in moving_object_trajectories:
            box_config = moving_object_trajectories["box"]
            traj_box = self._create_trajectory_from_config(
                box_config, time_range, tensor_args
            )
        else:
            # Default trajectory
            traj_box = LinearTrajectory(
                keyframe_times=[time_range[0], time_range[1]],
                keyframe_positions=[
                    [-1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0]
                ],
                tensor_args=tensor_args
            )

        moving_box = MovingObjectField(
            primitive_fields=[prim_box],
            trajectory=traj_box,
            name="dyn-simple2d-box"
        )

        # Initialize EnvDynBase wrapper with the base environment's configuration
        # and add moving objects to obj_extra_list
        super().__init__(
            limits=base_env.limits,
            obj_fixed_list=base_env.obj_fixed_list,  # Static obstacles from EnvSimple2D
            obj_extra_list=[moving_sphere, moving_box],  # MovingObjectField instances
            precompute_sdf_obj_fixed=kwargs.get('precompute_sdf_obj_fixed', True),
            precompute_sdf_obj_extra=False,  # Don't precompute for moving objects
            sdf_cell_size=kwargs.get('sdf_cell_size', 0.01),
            time_range=time_range,
            k_smooth=k_smooth,
            smoothing_method=smoothing_method,
            no_smoothing=no_smoothing,
            tensor_args=tensor_args,
        )

    @staticmethod
    def _create_trajectory_from_config(config, time_range, tensor_args):
        """
        Create a LinearTrajectory from a configuration dict.

        Args:
            config: Dict with trajectory configuration, e.g.,
                   {"start": [x, y], "end": [x, y]} or
                   {"start": [x, y, z], "end": [x, y, z]}
            time_range: (t_start, t_end) for the trajectory
            tensor_args: Tensor device and dtype

        Returns:
            LinearTrajectory instance
        """
        start_pos = config.get("start", [0.0, 0.0])
        end_pos = config.get("end", [0.0, 0.0])

        # Ensure 3D positions (add z=0 if 2D)
        if len(start_pos) == 2:
            start_pos = [start_pos[0], start_pos[1], 0.0]
        if len(end_pos) == 2:
            end_pos = [end_pos[0], end_pos[1], 0.0]

        return LinearTrajectory(
            keyframe_times=[time_range[0], time_range[1]],
            keyframe_positions=[start_pos, end_pos],
            tensor_args=tensor_args
        )

    def configure_moving_objects_from_start_goal(self, q_start, q_goal, trajectory_duration=None):
        """
        Configure moving object trajectories based on robot start and goal positions.

        This is a helper method for potential-based diffusion or other methods that
        need to update dynamic obstacles based on the planning problem.

        Args:
            q_start: Robot start configuration (task space position for 2D point mass)
            q_goal: Robot goal configuration (task space position for 2D point mass)
            trajectory_duration: Optional duration override, uses env time_range if None

        Example usage in inference:
            q_pos_start, q_pos_goal, _ = evaluation_samples_generator.get_data_sample(idx)
            env.configure_moving_objects_from_start_goal(q_pos_start, q_pos_goal)
        """
        if trajectory_duration is None:
            trajectory_duration = self.time_range[1] - self.time_range[0]

        # Convert to numpy if tensor
        if torch.is_tensor(q_start):
            q_start = q_start.cpu().numpy()
        if torch.is_tensor(q_goal):
            q_goal = q_goal.cpu().numpy()

        # Example configuration: moving obstacles perpendicular to start-goal line
        ### TODO : implement load from file
        # Calculate perpendicular direction
        # direction = q_goal - q_start
        # perp_direction = np.array([-direction[1], direction[0]]) if len(direction) >= 2 else np.array([0, 0])
        # if np.linalg.norm(perp_direction) > 0:
        #     perp_direction = perp_direction / np.linalg.norm(perp_direction)

        # # Create trajectories that cross the path
        # offset = 0.5  # Distance from center
        # sphere_start = perp_direction * offset
        # sphere_end = -perp_direction * offset
        # box_start = -perp_direction * offset
        # box_end = perp_direction * offset

        theta = np.random.uniform(-1, 1, 2)
        offset = 0.5 * np.linalg.norm(q_goal - q_start)
        dir_sphere = np.array([np.cos(theta[0]), np.sin(theta[0])])
        dir_box = np.array([np.cos(theta[1]), np.sin(theta[1])])
        sphere_start = dir_sphere * offset
        sphere_end = -dir_sphere * offset
        box_start = dir_box * offset
        box_end = -dir_box * offset

        # Update trajectories
        trajectory_configs = {
            "dyn-simple2d-sphere": LinearTrajectory(
                keyframe_times=[self.time_range[0], self.time_range[1]],
                keyframe_positions=[
                    [sphere_start[0], sphere_start[1], 0.0],
                    [sphere_end[0], sphere_end[1], 0.0]
                ],
                tensor_args=self.tensor_args
            ),
            "dyn-simple2d-box": LinearTrajectory(
                keyframe_times=[self.time_range[0], self.time_range[1]],
                keyframe_positions=[
                    [box_start[0], box_start[1], 0.0],
                    [box_end[0], box_end[1], 0.0]
                ],
                tensor_args=self.tensor_args
            )
        }

        result = self.update_all_moving_object_trajectories(trajectory_configs)
        assert result['dyn-simple2d-sphere'] and result['dyn-simple2d-sphere']

        # save as numpy
        trajectory_configs_dict = {
            "dyn-simple2d-sphere":{
                "keyframe_times" : to_numpy([self.time_range[0], self.time_range[1]]),
                "keyframe_positions" : to_numpy(
                    [
                        [sphere_start[0], sphere_start[1], 0.0],
                        [sphere_end[0], sphere_end[1], 0.0]
                    ])
            },
            "dyn-simple2d-box" :{
                "keyframe_times" : to_numpy([self.time_range[0], self.time_range[1]]),
                "keyframe_positions" : to_numpy(
                    [
                        [box_start[0], box_start[1], 0.0],
                        [box_end[0], box_end[1], 0.0]
                    ])
            },
        }

        return trajectory_configs_dict
    




if __name__ == "__main__":
    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with moving obstacles
    env = EnvDynSimple2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size = 0.01,
        time_range = (0.0, 10.0),
        k_smooth=0.05,
        smoothing_method=None,
        no_smoothing=True,
        tensor_args=tensor_args
    )

    print(f"Environment created:")
    print(f"  - Dimension: {env.dim}")
    print(f"  - Fixed objects: {len(env.obj_fixed_list)}")
    print(f"  - Extra (moving) objects: {len(env.obj_extra_list)}")
    print(f"  - Has moving objects: {env._has_moving_objects()}")
    print(f"  - Time range: {env.time_range}")

    # Test 1: Render at different times
    # print("\nTest 1: Rendering at different times")
    #fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    #times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    # for ax, t in zip(axes.flat, times):
    #     env.render(ax, time=t)
    #     ax.set_title(f"t = {t:.1f}s")

    # plt.tight_layout()
    # plt.savefig('/tmp/env_dyn_extra_2d_rendering.png', dpi=150)
    # print("  Saved to /tmp/env_dyn_extra_2d_rendering.png")
    # plt.close()

    # Test 2: Render SDF at different times
    # print("\nTest 2: Time-varying SDF")
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # for ax, t in zip(axes.flat, times):
    #     env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
    #     ax.set_title(f"SDF at t = {t:.1f}s")

    # plt.tight_layout()
    # plt.savefig('/tmp/env_dyn_extra_2d_sdf.png', dpi=150)
    # print("  Saved to /tmp/env_dyn_extra_2d_sdf.png")
    # plt.close()

    # Test 3: Animate environment
    # print("\nTest 3: Creating animation")

    # # Create time steps array for animation
    # time_steps = torch.linspace(0.0, 10.0, 100, **tensor_args)

    # env.animate_with_time(
    #     trajectory_time_steps=time_steps,
    #     n_frames=50,
    #     video_filepath='/tmp/env_dyn_extra_2d_animation.mp4',
    #     show_time_label=True,
    #     anim_time=5,
    #     dpi=100,
    # )
    # print("  Saved animation to /tmp/env_dyn_extra_2d_animation.mp4")

    # # Test 4: Test SDF queries at specific points over time
    # print("\nTest 4: SDF queries over time")
    # test_points = torch.tensor([
    #     [0.0, 0.0],
    #     [0.5, 0.5],
    #     [-0.5, -0.5],
    # ], **tensor_args)

    # for point in test_points:
    #     print(f"\n  Point {point.cpu().numpy()}:")
    #     for t in [0.0, 2.5, 5.0]:
    #         sdf_val = env.compute_sdf(point.unsqueeze(0).unsqueeze(0), time=t)
    #         collision_str = " [COLLISION]" if sdf_val.item() < 0 else ""
    #         print(f"    t={t:.1f}s: SDF = {sdf_val.item():+.4f}{collision_str}")

    # print("\n" + "="*70)
    # print("All tests completed!")
    # print("="*70)
    # print("\nKey features demonstrated:")
    # print("  - EnvDynBase wraps existing EnvBase-based environments")
    # print("  - MovingObjectField automatically handled in rendering")
    # print("  - MovingObjectField automatically handled in SDF computation")
    # print("  - Time-aware animations")
    # print("  - Smooth union for overlapping obstacles")

    df_obj_list = env.get_df_obj_list()
    # shape_x = (5, 18, 2)
    shape_x = (1, 18, 2)
    link_pos = torch.randn(shape_x, device=tensor_args["device"])
    for df in df_obj_list:
        # df : GridMapSDF or MovingObjectField
        # print("CollisionObjectdistancefield", type(df))
        get_gradient = True
        if get_gradient:
            sdf_vals, sdf_gradient = df.compute_signed_distance(link_pos, get_gradient=get_gradient)
        else:
            sdf_vals = df.compute_signed_distance(link_pos, get_gradient=get_gradient)