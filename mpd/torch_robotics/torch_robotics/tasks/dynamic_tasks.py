"""
Dynamic environment planning tasks.

This module provides DynPlanningTask which extends PlanningTask to properly handle
time-varying environments with moving obstacles (EnvDynBase).
"""

import functools
import itertools
import sys
from abc import ABC
from functools import partial
import inspect

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from mpd.parametric_trajectory.trajectory_base import ParametricTrajectoryBase
from mpd.plotting.utils import remove_axes_labels_ticks
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionWorkspaceBoundariesDistanceField,
    CollisionObjectDistanceField,
)
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.configuration_free_space import plot_configuration_free_space
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video, plot_multiline

from .tasks import PlanningTask


class DynPlanningTask(PlanningTask):
    """
    Planning task for dynamic environments with time-varying obstacles.

    This class extends PlanningTask to properly handle environments where obstacles
    move over time (EnvDynBase). Key enhancements:
    - Passes timesteps to collision checking for accurate moving obstacle detection
    - Time-aware rendering that displays obstacles at correct positions
    - Compatible with CollisionObjectDistanceFieldTimeVarying

    Usage:
        # Create dynamic environment
        env = EnvDynBase(
            limits=limits,
            obj_fixed_list=static_obstacles,
            moving_obj_list_fn=moving_obstacles_fn,
            time_range=(0.0, 5.0),
        )

        # Create planning task
        task = DynPlanningTask(
            env=env,
            robot=robot,
            parametric_trajectory=trajectory,
            ...
        )

        # Collision checking automatically handles timesteps
        collisions = task.compute_collision(q_trajs)

        # Animation automatically renders at correct times
        task.animate_robot_trajectories(q_trajs)
    """

    def __init__(self, **kwargs):
        """
        Initialize dynamic planning task.

        Args:
            **kwargs: All arguments passed to parent PlanningTask constructor
        """
        super().__init__(**kwargs)
        # Detect if environment is dynamic
        self.is_dynamic = self._check_if_dynamic()

    def set_collision_fields(
        self,
        obstacle_cutoff_margin,
        use_field_collision_self,
        use_field_collision_objects,
        use_field_collision_ws_boundaries
    ):
        """
        Override to set time-varying collision fields for dynamic environments.

        Args:
            obstacle_cutoff_margin: Cutoff margin for obstacle collision
            use_field_collision_self: Whether to use self-collision field
            use_field_collision_objects: Whether to use object collision field
            use_field_collision_ws_boundaries: Whether to use workspace boundaries field
        """

        # For dynamic environments, use time-varying collision fields
        from torch_robotics.torch_planning_objectives.fields.distance_fields_time_varying import (
            CollisionObjectDistanceFieldTimeVarying
        )

        # Self-collision field (static)
        self.df_collision_self = self.robot.df_collision_self

        # Time-varying collision field for objects
        self.df_collision_objects = CollisionObjectDistanceFieldTimeVarying(
            self.robot,
            df_time_varying_obj_fn=self.env.get_df_obj_list,
            parametric_trajectory=self.parametric_trajectory,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=obstacle_cutoff_margin,
            tensor_args=self.tensor_args,
        )

        # Time-varying collision field for extra objects
        self.df_collision_extra_objects = None
        if self.env.obj_extra_list is not None:
            self.df_collision_extra_objects = CollisionObjectDistanceFieldTimeVarying(
                self.robot,
                df_time_varying_obj_fn=partial(self.env.get_df_obj_list, return_extra_objects_only=True),
                parametric_trajectory=self.parametric_trajectory,
                link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
                cutoff_margin=obstacle_cutoff_margin,
                tensor_args=self.tensor_args,
            )
            self._collision_fields_extra_objects = [self.df_collision_extra_objects]
        else:
            self._collision_fields_extra_objects = []

        # Workspace boundaries field (static)
        self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
            self.robot,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=obstacle_cutoff_margin,
            ws_bounds_min=self.ws_min,
            ws_bounds_max=self.ws_max,
            tensor_args=self.tensor_args,
        )

        # Apply usage flags
        self.df_collision_self = self.df_collision_self if use_field_collision_self else None
        self.df_collision_objects = self.df_collision_objects if use_field_collision_objects else None
        self.df_collision_ws_boundaries = self.df_collision_ws_boundaries if use_field_collision_ws_boundaries else None

        # Update collision fields list
        self._collision_fields = [
            self.df_collision_self,
            self.df_collision_objects,
            self.df_collision_ws_boundaries,
            ]
        # else:
        #     # # For static environments, use parent implementation
        #     # super().set_collision_fields(
        #     #     obstacle_cutoff_margin,
        #     #     use_field_collision_self,
        #     #     use_field_collision_objects,
        #     #     use_field_collision_ws_boundaries
        #     # )
        #     raise NotImplementedError

    def _check_if_dynamic(self):
        """
        Check if environment is dynamic (has time-varying obstacles).

        Returns:
            bool: True if environment is dynamic
        """
        # Check for EnvDynBase wrapper
        if hasattr(self.env, '_has_moving_objects'):
            return self.env._has_moving_objects()

        # Check for is_static attribute
        if hasattr(self.env, 'is_static'):
            return not self.env.is_static

        # Check if env is EnvDynBase by checking for time_range attribute
        if hasattr(self.env, 'time_range'):
            return True

        # Default: assume static
        return False

    def _get_timesteps_for_horizon(self, horizon_size=None):
        """
        Get timesteps corresponding to trajectory horizon.

        Args:
            horizon_size: Optional horizon size. If None, gets from parametric_trajectory

        Returns:
            torch.Tensor: Timesteps, shape (horizon,)
        """
        if hasattr(self.parametric_trajectory, 'get_timesteps'):
            timesteps = self.parametric_trajectory.get_timesteps()
            if horizon_size is not None and len(timesteps) != horizon_size:
                # Interpolate to match horizon
                t_start = timesteps[0]
                t_end = timesteps[-1]
                timesteps = torch.linspace(t_start, t_end, horizon_size, **self.tensor_args)
            return timesteps
        else:
            # Fallback: use time_range if available
            if hasattr(self.env, 'time_range'):
                t_start, t_end = self.env.time_range
                if horizon_size is None:
                    horizon_size = 64  # Default
                return torch.linspace(t_start, t_end, horizon_size, **self.tensor_args)
            else:
                return None

    ###############################################################################################################
    # Time-aware rendering methods
    ###############################################################################################################

    def render_robot_trajectories(
        self, fig=None, ax=None, q_pos_trajs=None, q_pos_trajs_best=None,
        time=None, color_collisions=True, **kwargs
    ):
        """
        Enhanced render that supports time parameter for dynamic environments.

        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            q_pos_trajs: Robot trajectories
            q_pos_trajs_best: Best trajectory (optional)
            time: Time value for rendering time-varying obstacles (optional)
            color_collisions: Whether to color trajectories by collision status
            **kwargs: Additional rendering arguments

        Returns:
            fig, ax
        """
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        # Render environment at specific time if supported
        if self.is_dynamic and hasattr(self.env, 'render'):
            sig = inspect.signature(self.env.render)
            if 'time' in sig.parameters and time is not None:
                self.env.render(ax, time=time)
            else:
                self.env.render(ax)
        else:
            self.env.render(ax)
        print("render_robot_trajectories")
        # Render trajectories
        if q_pos_trajs is not None:
            if color_collisions:
                _, q_trajs_coll_idxs, _, q_trajs_free_idxs, _ = self.get_trajs_unvalid_and_valid(
                    q_pos_trajs, return_indices=True, **kwargs
                )
                kwargs["colors"] = []
                for i in range(len(q_trajs_coll_idxs) + len(q_trajs_free_idxs)):
                    kwargs["colors"].append(
                        self.colors["collision"] if i in q_trajs_coll_idxs else self.colors["free"]
                    )
            else:
                kwargs["colors"] = [self.colors["free"]] * len(q_pos_trajs)

        self.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs, **kwargs)

        if q_pos_trajs_best is not None:
            kwargs["colors"] = ["blue"]
            self.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs_best.unsqueeze(0), **kwargs)

        return fig, ax

    def animate_robot_trajectories(
        self,
        q_pos_trajs=None,
        q_pos_start=None,
        q_pos_goal=None,
        plot_x_trajs=False,
        n_frames=100,
        remove_title=False,
        process_axes=lambda x: x,
        **kwargs,
    ):
        """
        Enhanced animation that renders moving obstacles at correct times.

        For dynamic environments, this method automatically:
        - Extracts timesteps from parametric_trajectory
        - Renders environment at the correct time for each frame
        - Displays time information in frame titles

        Args:
            q_pos_trajs: Robot trajectories (batch, horizon, q_dim)
            q_pos_start: Start configuration
            q_pos_goal: Goal configuration
            plot_x_trajs: Whether to plot full trajectories
            n_frames: Number of animation frames
            remove_title: Whether to remove title
            process_axes: Function to process axes
            **kwargs: Additional arguments passed to create_animation_video
        """
        if q_pos_trajs is None:
            return

        assert q_pos_trajs.ndim == 3
        B, H, D = q_pos_trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        q_pos_trajs_selection = q_pos_trajs[:, idxs, :]

        # Get time steps for the trajectory
        time_steps = None
        time_steps_tensor = None
        if self.is_dynamic:
            time_steps_full = self._get_timesteps_for_horizon(H)
            if time_steps_full is not None:
                # Select timesteps for selected frames
                time_steps_tensor = time_steps_full[idxs]  # (n_frames,)
                if isinstance(time_steps_full, torch.Tensor):
                    time_steps = to_numpy(time_steps_tensor)
                else:
                    time_steps = np.array(time_steps_tensor)

        # Precompute collisions for all frames at once (batch, n_frames)
        collision_check_kwargs = {}
        if self.is_dynamic and time_steps_tensor is not None:
            # Pass timesteps for time-varying collision checking
            collision_check_kwargs['timesteps'] = time_steps_tensor
        collisions_all = self.compute_collision(q_pos_trajs_selection, margin=0.0, **collision_check_kwargs)

        # Precompute trajectory colors and collision points if plotting full trajectories
        traj_colors = None
        collision_points = None
        if plot_x_trajs and q_pos_trajs is not None:
            # Compute collisions for full trajectory ONCE
            full_traj_collision_kwargs = {}
            if self.is_dynamic and time_steps_tensor is not None:
                # Need timesteps for full horizon
                full_time_steps = self._get_timesteps_for_horizon(H)
                if full_time_steps is not None:
                    full_traj_collision_kwargs['timesteps'] = full_time_steps

            _, q_trajs_coll_idxs, _, q_trajs_free_idxs, _ = self.get_trajs_unvalid_and_valid(
                q_pos_trajs, return_indices=True, **full_traj_collision_kwargs
            )
            traj_colors = []
            for i in range(len(q_trajs_coll_idxs) + len(q_trajs_free_idxs)):
                traj_colors.append(
                    self.colors["collision"] if i in q_trajs_coll_idxs else self.colors["free"]
                )

            # Extract collision points with DYNAMIC obstacles only
            collision_points = []
            # TODO : implement self.env.get_df_obj_list, return_dynamic_objects_only
            if self.is_dynamic and self.df_collision_extra_objects is not None and full_time_steps is not None:
                # Compute dynamic-only collisions
                # Forward kinematics
                fk_collision_pos = self.robot.fk_map_collision(q_pos_trajs)  # (batch, horizon, taskspaces, x_dim)

                # Compute collision with dynamic objects only
                dynamic_collision_kwargs = {'timesteps': full_time_steps, 'margin': 0.0}

                cost_dynamic_objects = self.df_collision_extra_objects.compute_cost(
                    q_pos_trajs, fk_collision_pos, field_type="occupancy", **dynamic_collision_kwargs
                )  # (batch, horizon)

                # Extract collision points for each trajectory
                for b in range(B):
                    q_traj = q_pos_trajs[b]  # (horizon, q_dim)
                    collision_mask = cost_dynamic_objects[b]  # (horizon,)

                    if collision_mask.any():
                        collision_idxs = torch.where(collision_mask)[0]
                        collision_pts = q_traj[collision_idxs]  # (num_collisions, q_dim)
                        collision_points.append(collision_pts)
                    else:
                        collision_points.append(None)
            else:
                # If not dynamic, no collision points
                collision_points = [None] * B

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i, ax):
            ax.clear()

            # Set title with time information
            if not remove_title:
                title = f"step: {idxs[i]}/{H-1}"
                if time_steps is not None:
                    title += f", time: {time_steps[i]:.3f}s"
                ax.set_title(title)

            # Render environment at current time
            current_time = time_steps[i] if time_steps is not None else None
            if self.is_dynamic and hasattr(self.env, 'render'):
                sig = inspect.signature(self.env.render)
                if 'time' in sig.parameters and current_time is not None:
                    self.env.render(ax, time=current_time)
                else:
                    self.env.render(ax)
            else:
                self.env.render(ax)
            if plot_x_trajs:
                # Render trajectories with precomputed colors (avoid recomputing collisions!)
                if traj_colors is not None:
                    render_kwargs = kwargs.copy()
                    render_kwargs["colors"] = traj_colors
                    self.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs, **render_kwargs)

            # Get precomputed collisions for current frame
            qs = q_pos_trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)

            # Render each robot with precomputed collision status
            for idx, q in enumerate(qs):
                is_collision = collisions_all[idx, i] if collisions_all.ndim > 1 else collisions_all

                self.robot.render(
                    ax,
                    q_pos=q,
                    color=(
                        self.colors_robot["collision"]
                        if is_collision
                        else self.colors_robot["free"]
                    ),
                    arrow_length=0.1,
                    arrow_alpha=0.5,
                    arrow_linewidth=1.0,
                    cmap=self.cmaps["collision"] if is_collision else self.cmaps["free"],
                    **kwargs,
                )

            # Render collision markers for trajectories
            if plot_x_trajs and collision_points is not None:
                for b, coll_pts in enumerate(collision_points):
                    if coll_pts is not None:
                        coll_pts_np = to_numpy(coll_pts)
                        if self.env.dim == 2:
                            ax.scatter(coll_pts_np[:, 0], coll_pts_np[:, 1],
                                      color='red', marker='x', s=7, linewidths=2, zorder=10, label='Collision' if b == 0 else '')
                        elif self.env.dim == 3:
                            ax.scatter(coll_pts_np[:, 0], coll_pts_np[:, 1], coll_pts_np[:, 2],
                                      color='red', marker='x', s=7, linewidths=2, zorder=10, label='Collision' if b == 0 else '')

            if q_pos_start is not None:
                self.robot.render(ax, q_pos_start, color="green", cmap="Greens", **kwargs)
            if q_pos_goal is not None:
                self.robot.render(ax, q_pos_goal, color="purple", cmap="Purples", **kwargs)

            process_axes(ax)

        create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,), **kwargs)

    def animate_opt_iters_robots(
        self,
        trajs_pos=None,
        traj_pos_best=None,
        start_state=None,
        goal_state=None,
        control_points=None,
        n_frames=10,
        remove_axes_labels_and_ticks=False,
        **kwargs,
    ):
        """
        Animate optimization iterations with time-aware rendering.

        Args:
            trajs_pos: Trajectories over optimization steps (steps, batch, horizon, q_dim)
            traj_pos_best: Best trajectory
            start_state: Start configuration
            goal_state: Goal configuration
            control_points: Control points for trajectories
            n_frames: Number of frames
            remove_axes_labels_and_ticks: Whether to remove axes labels
            **kwargs: Additional arguments
        """
        if trajs_pos is None:
            return

        assert trajs_pos.ndim == 4
        S, B, H, D = trajs_pos.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_pos_selection = trajs_pos[idxs]
        if control_points is None:
            control_points_selection = trajs_pos_selection
        else:
            control_points_selection = control_points[idxs]

        # Get middle time for static rendering
        render_time = None
        if self.is_dynamic:
            time_steps = self._get_timesteps_for_horizon(H)
            if time_steps is not None:
                mid_idx = H // 2
                render_time = time_steps[mid_idx]
                if isinstance(render_time, torch.Tensor):
                    render_time = float(render_time.cpu())

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i, ax):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            self.render_robot_trajectories(
                fig=fig,
                ax=ax,
                q_pos_trajs=trajs_pos_selection[i],
                control_points=control_points_selection[i],
                q_pos_trajs_best=traj_pos_best if i == n_frames - 1 else None,
                time=render_time,
                start_state=start_state,
                goal_state=goal_state,
                **kwargs,
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color="green", cmap="Greens")
            if goal_state is not None:
                self.robot.render(ax, goal_state, color="purple", cmap="Purples")

            if remove_axes_labels_and_ticks:
                remove_axes_labels_ticks(ax)

        create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,), **kwargs)

    def plot_joint_space_trajectories(
        self,
        fig=None,
        axs=None,
        q_pos_trajs=None,
        q_vel_trajs=None,
        q_acc_trajs=None,
        q_pos_traj_best=None,
        q_vel_traj_best=None,
        q_acc_traj_best=None,
        q_pos_start=None,
        q_pos_goal=None,
        q_vel_start=None,
        q_vel_goal=None,
        q_acc_start=None,
        q_acc_goal=None,
        set_q_pos_limits=True,
        set_q_vel_limits=True,
        set_q_acc_limits=True,
        control_points=None,
        **kwargs,
    ):
        """
        Override to show red segments for dynamic obstacle collisions.

        This method extends the base plot_joint_space_trajectories to highlight
        trajectory segments that collide with dynamic obstacles in red.
        """
        if q_pos_trajs is None:
            return

        q_pos_trajs_np = to_numpy(q_pos_trajs)
        assert q_pos_trajs_np.ndim == 3  # batch, horizon, q dimension

        B, H, D = q_pos_trajs.shape

        # Get timesteps for full horizon
        full_time_steps = self._get_timesteps_for_horizon(H)

        # Compute dynamic obstacle collisions per waypoint for each trajectory
        dynamic_collision_masks = None
        if self.is_dynamic and self.df_collision_extra_objects is not None and full_time_steps is not None:
            # Forward kinematics
            fk_collision_pos = self.robot.fk_map_collision(q_pos_trajs)  # (batch, horizon, taskspaces, x_dim)

            # Compute collision with dynamic objects only
            dynamic_collision_kwargs = {'timesteps': full_time_steps, 'margin':0.0}
            cost_dynamic_objects = self.df_collision_extra_objects.compute_cost(
                q_pos_trajs, fk_collision_pos, field_type="occupancy", **dynamic_collision_kwargs
            )  # (batch, horizon)

            dynamic_collision_masks = to_numpy(cost_dynamic_objects)  # (batch, horizon) boolean array

        # Filter trajectories not in collision and inside joint limits (for overall trajectory classification)
        q_trajs_l = [q_pos_trajs]
        if q_vel_trajs is not None:
            q_trajs_l.append(q_vel_trajs)
        if q_acc_trajs is not None:
            q_trajs_l.append(q_acc_trajs)
        q_trajs = torch.cat(q_trajs_l, dim=-1)

        q_trajs_coll, q_trajs_coll_idxs, q_trajs_free, q_trajs_free_idxs, _ = self.get_trajs_unvalid_and_valid(
            q_trajs, return_indices=True, **kwargs
        )

        q_pos_trajs_coll_np = None
        q_vel_trajs_coll_np = None
        q_acc_trajs_coll_np = None
        if q_trajs_coll is not None:
            q_pos_trajs_coll_np = to_numpy(self.get_position(q_trajs_coll))
            if q_pos_trajs_coll_np.ndim == 2:
                q_pos_trajs_coll_np = q_pos_trajs_coll_np[None, ...]
            if q_vel_trajs is not None:
                q_vel_trajs_coll_np = to_numpy(self.get_velocity(q_trajs_coll))
                if q_vel_trajs_coll_np.ndim == 2:
                    q_vel_trajs_coll_np = q_vel_trajs_coll_np[None, ...]
            if q_acc_trajs is not None:
                q_acc_trajs_coll_np = to_numpy(self.get_acceleration(q_trajs_coll))
                if q_acc_trajs_coll_np.ndim == 2:
                    q_acc_trajs_coll_np = q_acc_trajs_coll_np[None, ...]

        q_pos_trajs_free_np = None
        q_vel_trajs_free_np = None
        q_acc_trajs_free_np = None
        if q_trajs_free is not None:
            q_pos_trajs_free_np = to_numpy(self.get_position(q_trajs_free))
            if q_pos_trajs_free_np.ndim == 2:
                q_pos_trajs_free_np = q_pos_trajs_free_np[None, ...]
            if q_vel_trajs is not None:
                q_vel_trajs_free_np = to_numpy(self.get_velocity(q_trajs_free))
                if q_vel_trajs_free_np.ndim == 2:
                    q_vel_trajs_free_np = q_vel_trajs_free_np[None, ...]
            if q_acc_trajs is not None:
                q_acc_trajs_free_np = to_numpy(self.get_acceleration(q_trajs_free))
                if q_acc_trajs_free_np.ndim == 2:
                    q_acc_trajs_free_np = q_acc_trajs_free_np[None, ...]

        if q_pos_start is not None:
            q_pos_start = to_numpy(q_pos_start)
        if q_vel_start is not None:
            q_vel_start = to_numpy(q_vel_start)
        if q_acc_start is not None:
            q_acc_start = to_numpy(q_acc_start)
        if q_pos_goal is not None:
            q_pos_goal = to_numpy(q_pos_goal)
        if q_vel_goal is not None:
            q_vel_goal = to_numpy(q_vel_goal)
        if q_acc_goal is not None:
            q_acc_goal = to_numpy(q_acc_goal)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 3, squeeze=False, figsize=(18, 2.5 * self.robot.q_dim))

        axs[0, 0].set_title("Position")
        axs[0, 1].set_title("Velocity")
        axs[0, 2].set_title("Acceleration")
        axs[-1, 1].set_xlabel("Time [s]")
        timesteps = to_numpy(self.parametric_trajectory.get_timesteps().reshape(1, -1))
        t_start, t_goal = timesteps[0, 0], timesteps[0, -1]

        for i, ax in enumerate(axs):
            for q_trajs_filtered, traj_idxs, base_color in zip(
                [
                    (q_pos_trajs_coll_np, q_vel_trajs_coll_np, q_acc_trajs_coll_np),
                    (q_pos_trajs_free_np, q_vel_trajs_free_np, q_acc_trajs_free_np),
                ],
                [q_trajs_coll_idxs, q_trajs_free_idxs],
                ["black", "orange"],
            ):
                # Positions, velocities, accelerations
                for j, q_trajs_filtered_item in enumerate(q_trajs_filtered):
                    if q_trajs_filtered_item is not None:
                        # Plot each trajectory with segment coloring for dynamic collisions
                        for traj_idx_in_filtered, original_traj_idx in enumerate(traj_idxs):
                            traj_data = q_trajs_filtered_item[traj_idx_in_filtered, :, i]  # (horizon,)
                            traj_timesteps = timesteps[0]  # (horizon,)

                            # Determine segment colors based on dynamic collision
                            if dynamic_collision_masks is not None:
                                collision_mask = dynamic_collision_masks[original_traj_idx]  # (horizon,)
                                # Create segments with colors
                                self._plot_trajectory_with_collision_segments(
                                    ax[j], traj_timesteps, traj_data, collision_mask, base_color
                                )
                            else:
                                # No dynamic collision info, use base plotting
                                plot_multiline(
                                    ax[j],
                                    np.repeat(timesteps, 1, axis=0),
                                    traj_data.reshape(1, -1),
                                    color=base_color,
                                    **kwargs,
                                )

            if q_pos_traj_best is not None:
                q_pos_traj_best_np = to_numpy(q_pos_traj_best)
                plot_multiline(ax[0], timesteps, q_pos_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
            if q_vel_traj_best is not None:
                q_vel_traj_best_np = to_numpy(q_vel_traj_best)
                plot_multiline(ax[1], timesteps, q_vel_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)
            if q_acc_traj_best is not None:
                q_acc_traj_best_np = to_numpy(q_acc_traj_best)
                plot_multiline(ax[2], timesteps, q_acc_traj_best_np[..., i].reshape(1, -1), color="blue", **kwargs)

            # Start and goal
            for j, x in enumerate([q_pos_start, q_vel_start, q_acc_start]):
                if x is not None:
                    ax[j].scatter(t_start, x[i], color="green")

            for j, x in enumerate([q_pos_goal, q_vel_goal, q_acc_goal]):
                if x is not None:
                    ax[j].scatter(t_goal, x[i], color="purple")

            ax[0].set_ylabel(f"$q_{i}$")

            if set_q_pos_limits:
                q_pos_min, q_pos_max = self.robot.q_pos_min_np[i], self.robot.q_pos_max_np[i]
                padding = 0.1 * np.abs(q_pos_max - q_pos_min)
                ax[0].set_ylim(q_pos_min - padding, q_pos_max + padding)
                ax[0].plot([t_start, t_goal], [q_pos_max, q_pos_max], color="k", linestyle="--")
                ax[0].plot([t_start, t_goal], [q_pos_min, q_pos_min], color="k", linestyle="--")
            if set_q_vel_limits and self.robot.dq_max_np is not None:
                ax[1].plot(
                    [t_start, t_goal], [self.robot.dq_max_np[i], self.robot.dq_max_np[i]], color="k", linestyle="--"
                )
                ax[1].plot(
                    [t_start, t_goal], [-self.robot.dq_max_np[i], -self.robot.dq_max_np[i]], color="k", linestyle="--"
                )
            if set_q_acc_limits and self.robot.ddq_max_np is not None:
                ax[2].plot(
                    [t_start, t_goal], [self.robot.ddq_max_np[i], self.robot.ddq_max_np[i]], color="k", linestyle="--"
                )
                ax[2].plot(
                    [t_start, t_goal], [-self.robot.ddq_max_np[i], -self.robot.ddq_max_np[i]], color="k", linestyle="--"
                )

        # time limits
        t_eps = 0.1
        for ax in list(itertools.chain(*axs)):
            ax.set_xlim(t_start - t_eps, t_goal + t_eps)

        # plot control points
        if control_points is not None:
            control_points_np = to_numpy(control_points)
            control_points_timesteps = to_numpy(self.parametric_trajectory.get_phase_steps())
            for control_points_np_one in control_points_np:
                for i, ax in enumerate(axs):
                    ax[0].scatter(control_points_timesteps, control_points_np_one[:, i], color="red", s=2**2, zorder=10)

        return fig, axs

    def _plot_trajectory_with_collision_segments(self, ax, timesteps, traj_data, collision_mask, base_color):
        """
        Helper function to plot trajectory with red segments where dynamic collisions occur.

        Args:
            ax: Matplotlib axis
            timesteps: Time values (horizon,)
            traj_data: Trajectory data for one joint (horizon,)
            collision_mask: Boolean mask indicating dynamic collisions (horizon,)
            base_color: Base color for non-collision segments
        """
        import matplotlib.collections as mcoll

        # Create line segments
        points = np.column_stack([timesteps, traj_data])  # (horizon, 2)
        segments = np.stack([points[:-1], points[1:]], axis=1)  # (horizon-1, 2, 2)

        # Assign colors: red for collision segments, base color otherwise
        # A segment is in collision if either endpoint is in collision
        segment_colors = []
        for idx in range(len(segments)):
            if collision_mask[idx] or collision_mask[idx + 1]:
                segment_colors.append('red')
            else:
                segment_colors.append(base_color)

        # Create and add line collection
        line_collection = mcoll.LineCollection(segments, colors=segment_colors, linewidths=1.0)
        ax.add_collection(line_collection)

        # Update axis limits
        ax.autoscale()

