"""
Extensions to PlanningTask for time-varying environments.

This module provides helper functions and classes to integrate EnvDynBase
with the existing PlanningTask animation infrastructure.
"""

import numpy as np
import torch
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
from torch_robotics.torch_utils.torch_utils import to_numpy


def animate_robot_trajectories_with_time(
    task,
    q_pos_trajs=None,
    q_pos_start=None,
    q_pos_goal=None,
    plot_x_trajs=False,
    n_frames=10,
    remove_title=False,
    process_axes=lambda x: x,
    **kwargs,
):
    """
    Enhanced version of PlanningTask.animate_robot_trajectories that supports time-varying environments.

    This function is compatible with EnvDynBase and will render moving obstacles at the correct
    time corresponding to each trajectory step.

    Args:
        task: PlanningTask instance
        q_pos_trajs: Robot trajectories (batch, horizon, q_dim)
        q_pos_start: Start configuration
        q_pos_goal: Goal configuration
        plot_x_trajs: Whether to plot full trajectories
        n_frames: Number of animation frames
        remove_title: Whether to remove title
        process_axes: Function to process axes
        **kwargs: Additional arguments passed to create_animation_video

    Example:
        from torch_robotics.environments.dynamic_extension.task_extensions import (
            animate_robot_trajectories_with_time
        )

        # Create planning task with EnvDynBase
        env = EnvDynBase(
            limits=limits,
            obj_fixed_list=fixed_objs,
            moving_obj_list_fn=moving_objs_fn,
            time_range=(0.0, 1.0),
        )

        planning_task = PlanningTask(env=env, robot=robot, ...)

        # Animate with time-varying obstacles
        animate_robot_trajectories_with_time(
            planning_task,
            q_pos_trajs=trajectories,
            q_pos_start=q_start,
            q_pos_goal=q_goal,
            video_filepath='animation.mp4',
        )
    """
    if q_pos_trajs is None:
        return

    assert q_pos_trajs.ndim == 3
    B, H, D = q_pos_trajs.shape

    idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
    q_pos_trajs_selection = q_pos_trajs[:, idxs, :]

    # Get time steps for the trajectory
    time_steps = None
    if hasattr(task.parametric_trajectory, 'get_timesteps'):
        time_steps = task.parametric_trajectory.get_timesteps()
        if isinstance(time_steps, torch.Tensor):
            time_steps = to_numpy(time_steps)
        else:
            time_steps = np.array(time_steps)

    fig, ax = create_fig_and_axes(dim=task.env.dim)

    def animate_fn(i, ax):
        ax.clear()
        if not remove_title:
            title = f"step: {idxs[i]}/{H-1}"
            if time_steps is not None:
                title += f", time: {time_steps[idxs[i]]:.3f}s"
            ax.set_title(title)

        if plot_x_trajs:
            # Render with trajectories - need to handle time for environment
            # This is more complex, so we'll render environment separately
            if hasattr(task.env, 'render') and time_steps is not None:
                # Check if env.render accepts time parameter
                import inspect
                sig = inspect.signature(task.env.render)
                if 'time' in sig.parameters:
                    task.env.render(ax, time=time_steps[idxs[i]])
                else:
                    task.env.render(ax)
            else:
                task.env.render(ax)

            task.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs, **kwargs)
        else:
            # Render environment at current time
            if hasattr(task.env, 'render') and time_steps is not None:
                # Check if env.render accepts time parameter
                import inspect
                sig = inspect.signature(task.env.render)
                if 'time' in sig.parameters:
                    task.env.render(ax, time=time_steps[idxs[i]])
                else:
                    task.env.render(ax)
            else:
                task.env.render(ax)

        # Render robots at current timestep
        qs = q_pos_trajs_selection[:, i, :]  # batch, q_dim
        if qs.ndim == 1:
            qs = qs.unsqueeze(0)  # interface (batch, q_dim)

        for q in qs:
            task.robot.render(
                ax,
                q_pos=q,
                color=(
                    task.colors_robot["collision"]
                    if task.compute_collision(q, margin=0.0)
                    else task.colors_robot["free"]
                ),
                arrow_length=0.1,
                arrow_alpha=0.5,
                arrow_linewidth=1.0,
                cmap=task.cmaps["collision"] if task.compute_collision(q, margin=0.0) else task.cmaps["free"],
                **kwargs,
            )

        if q_pos_start is not None:
            task.robot.render(ax, q_pos_start, color="blue", cmap="Greens", **kwargs)
        if q_pos_goal is not None:
            task.robot.render(ax, q_pos_goal, color="red", cmap="Purples", **kwargs)

        process_axes(ax)

    create_animation_video(fig, animate_fn, n_frames=n_frames, fargs=(ax,), **kwargs)


def render_robot_trajectories_with_time(
    task,
    fig=None,
    ax=None,
    q_pos_trajs=None,
    q_pos_trajs_best=None,
    time=None,
    color_collisions=True,
    **kwargs
):
    """
    Enhanced version of PlanningTask.render_robot_trajectories that supports time parameter.

    Args:
        task: PlanningTask instance
        fig: Matplotlib figure
        ax: Matplotlib axis
        q_pos_trajs: Robot trajectories
        q_pos_trajs_best: Best trajectory (optional)
        time: Time value for rendering time-varying obstacles (optional)
        color_collisions: Whether to color trajectories by collision status
        **kwargs: Additional rendering arguments

    Returns:
        fig, ax

    Example:
        # Render at specific time
        fig, ax = render_robot_trajectories_with_time(
            planning_task,
            q_pos_trajs=trajectories,
            time=0.5,
        )
        plt.show()
    """
    if fig is None or ax is None:
        fig, ax = create_fig_and_axes(dim=task.env.dim)

    # Render environment at specific time if supported
    if hasattr(task.env, 'render'):
        import inspect
        sig = inspect.signature(task.env.render)
        if 'time' in sig.parameters and time is not None:
            task.env.render(ax, time=time)
        else:
            task.env.render(ax)

    # Render trajectories
    if q_pos_trajs is not None:
        if color_collisions:
            _, q_trajs_coll_idxs, _, q_trajs_free_idxs, _ = task.get_trajs_unvalid_and_valid(
                q_pos_trajs, return_indices=True, **kwargs
            )
            kwargs["colors"] = []
            for i in range(len(q_trajs_coll_idxs) + len(q_trajs_free_idxs)):
                kwargs["colors"].append(task.colors["collision"] if i in q_trajs_coll_idxs else task.colors["free"])
        else:
            kwargs["colors"] = [task.colors["free"]] * len(q_pos_trajs)

    task.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs, **kwargs)

    if q_pos_trajs_best is not None:
        kwargs["colors"] = ["blue"]
        task.robot.render_trajectories(ax, q_pos_trajs=q_pos_trajs_best.unsqueeze(0), **kwargs)

    return fig, ax
