"""
Dynamic environment base class with overlap detection and smooth SDF composition.

This module wraps EnvBase to handle potentially overlapping objects using
smooth SDF unions for differentiability. Movement trajectories are handled
automatically when MovingObjectField instances are detected in object lists.
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

from mpd.torch_robotics.torch_robotics.environments.env_base import EnvBase
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension.moving_primitives import MovingObjectField
from mpd.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from mpd.torch_robotics.torch_robotics.torch_utils.torch_timer import TimerCUDA


class EnvDynBase:
    """
    Dynamic environment wrapper class that handles overlapping objects with smooth SDF composition.

    This class wraps EnvBase and provides:
    - Smooth SDF unions (differentiable at overlaps)
    - Automatic handling of MovingObjectField instances in object lists
    - Time-aware rendering when MovingObjectField objects are detected
    """

    def __init__(
        self,
        k_smooth=20.0,
        smoothing_method="Quadratic",
        time_range=(0.0, 1.0),
        **kwargs
    ):
        """
        Args:
            k_smooth: Smoothness parameter for smooth union (higher = sharper transition)
            smoothing_method: "Quadratic" (smin) or "LSE" (LogSumExp)
            time_range: (t_min, t_max) for time-varying objects
            **kwargs: Arguments passed to EnvBase constructor
        """
        # Create wrapped EnvBase instance
        self.env = EnvBase(**kwargs)

        # Smooth SDF parameters
        self.k_smooth = k_smooth
        self.smoothing_method = smoothing_method

        # Time-varying obstacle support
        self.time_range = time_range

    # ===== Delegated properties from wrapped EnvBase =====

    @property
    def tensor_args(self):
        return self.env.tensor_args

    @property
    def name(self):
        return self.env.name

    @property
    def limits(self):
        return self.env.limits

    @property
    def limits_np(self):
        return self.env.limits_np

    @property
    def dim(self):
        return self.env.dim

    @property
    def obj_fixed_list(self):
        return self.env.obj_fixed_list

    @obj_fixed_list.setter
    def obj_fixed_list(self, value):
        self.env.obj_fixed_list = value

    @property
    def obj_extra_list(self):
        return self.env.obj_extra_list

    @obj_extra_list.setter
    def obj_extra_list(self, value):
        self.env.obj_extra_list = value

    @property
    def obj_all_list(self):
        return self.env.obj_all_list

    @property
    def grid_map_sdf_obj_fixed(self):
        return self.env.grid_map_sdf_obj_fixed

    @property
    def grid_map_sdf_obj_extra(self):
        return self.env.grid_map_sdf_obj_extra

    @property
    def sdf_cell_size(self):
        return self.env.sdf_cell_size

    # ===== Delegated methods from wrapped EnvBase =====

    def update_obj_all_list(self):
        return self.env.update_obj_all_list()

    def get_obj_list(self):
        return self.env.get_obj_list()

    def get_obj_fixed_list(self):
        return self.env.get_obj_fixed_list()

    def get_obj_extra_list(self):
        return self.env.get_obj_extra_list()

    def set_obj_extra_list(self, obj_extra_list):
        return self.env.set_obj_extra_list(obj_extra_list)

    def get_df_obj_list(self, return_extra_objects_only=False):
        return self.env.get_df_obj_list(return_extra_objects_only)

    def build_sdf_grid(self, compute_sdf_obj_fixed=False, compute_sdf_obj_extra=False):
        return self.env.build_sdf_grid(compute_sdf_obj_fixed, compute_sdf_obj_extra)

    def add_objects_extra(self, obj_l):
        return self.env.add_objects_extra(obj_l)

    def update_objects_extra(self):
        return self.env.update_objects_extra()

    def build_occupancy_map(self, cell_size=None):
        return self.env.build_occupancy_map(cell_size)

    def zero_grad(self):
        return self.env.zero_grad()

    # ===== Helper methods =====

    def _has_moving_objects(self):
        """Check if any objects in fixed or extra lists are MovingObjectField instances."""
        for obj in self.obj_fixed_list:
            if isinstance(obj, MovingObjectField):
                return True
        for obj in self.obj_extra_list:
            if isinstance(obj, MovingObjectField):
                return True
        return False

    def _update_moving_objects_at_time(self, time):
        """Update all MovingObjectField instances to the specified time."""
        for obj in self.obj_fixed_list:
            if isinstance(obj, MovingObjectField):
                obj.update_pose_at_time(time)
        for obj in self.obj_extra_list:
            if isinstance(obj, MovingObjectField):
                obj.update_pose_at_time(time)

    # ===== Enhanced rendering with automatic MovingObjectField support =====

    def render(self, ax=None, time=None):
        """
        Render the environment, automatically handling MovingObjectField instances.

        If time is provided or MovingObjectField objects are detected, updates their
        poses before rendering.

        Args:
            ax: Matplotlib axis
            time: Optional time parameter for MovingObjectField objects.
                  If None and moving objects exist, uses middle of time_range.
        """
        # Auto-detect moving objects and determine rendering time
        has_moving = self._has_moving_objects()

        if has_moving:
            # Use provided time, or default to middle of time range
            if time is None:
                time = (self.time_range[0] + self.time_range[1]) / 2.0
            # Clamp to valid range
            time = max(self.time_range[0], min(self.time_range[1], time))
            # Update all moving objects
            self._update_moving_objects_at_time(time)

        # Delegate to wrapped EnvBase for actual rendering
        self.env.render(ax)

    def compute_sdf(self, x, reshape_shape=None, use_smooth_union=True, smoothing_method=None, time=None):
        """
        Compute SDF with optional smooth union for all objects.

        Uses smooth union instead of EnvBase's hard minimum for differentiability.
        Automatically handles MovingObjectField instances at the specified time.

        Args:
            x: Query points, shape (..., dim)
            reshape_shape: Optional shape to reshape output
            use_smooth_union: If True, use smooth union; if False, use hard minimum
            smoothing_method: Override default smoothing method ("Quadratic" or "LSE")
            time: Time for MovingObjectField evaluation (auto-detected if None)

        Returns:
            SDF values at query points
        """
        # Update moving objects if present
        has_moving = self._has_moving_objects()
        if has_moving:
            if time is None:
                time = (self.time_range[0] + self.time_range[1]) / 2.0
            time = max(self.time_range[0], min(self.time_range[1], time))
            self._update_moving_objects_at_time(time)

        # Use instance default if not specified
        if smoothing_method is None:
            smoothing_method = self.smoothing_method

        if not use_smooth_union:
            # Delegate to wrapped EnvBase (hard minimum)
            return self.env.compute_sdf(x, reshape_shape=reshape_shape)


        if smoothing_method == "LSE":
            # Collect all SDFs using LogSumExp smooth union
            all_sdfs = []

            # Compute SDF from each fixed object
            for obj in self.obj_fixed_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                all_sdfs.append(sdf_obj)

            # Compute SDF from each extra object
            for obj in self.obj_extra_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                all_sdfs.append(sdf_obj)

            # Handle empty case
            if len(all_sdfs) == 0:
                return torch.ones(x.shape[:-1], **self.tensor_args) * float('inf')

            # Apply smooth union: φ_union = -k * logsumexp(-φ_i/k)
            if len(all_sdfs) == 1:
                return all_sdfs[0]
            else:
                sdfs_stacked = torch.stack(all_sdfs, dim=-1)
                return -self.k_smooth * torch.logsumexp(-sdfs_stacked / self.k_smooth, dim=-1)

        elif smoothing_method == "Quadratic":
            # Quadratic smooth minimum (smin) implementation
            # Based on: https://iquilezles.org/articles/smin/
            def smin(a, b, k):
                k *= 4.0  # Normalizing factor
                h = torch.maximum(k - torch.abs(a - b), torch.zeros_like(a)) / k
                return torch.minimum(a, b) - k * 0.25 * h * h

            sdf = None

            # Compute SDF from fixed objects
            for obj in self.obj_fixed_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape is not None:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                if sdf is None:
                    sdf = sdf_obj
                else:
                    sdf = smin(sdf, sdf_obj, self.k_smooth)

            # Compute SDF from extra objects
            for obj in self.obj_extra_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape is not None:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                if sdf is None:
                    sdf = sdf_obj
                else:
                    sdf = smin(sdf, sdf_obj, self.k_smooth)

            # Handle case with no objects
            if sdf is None:
                sdf = torch.ones(x.shape[:-1], **self.tensor_args) * float('inf')

            return sdf

        else:
            raise NotImplementedError(f"Smoothing method '{smoothing_method}' not implemented")

    def render_sdf(self, ax=None, fig=None, use_smooth_union=True, time=None):
        """
        Render SDF field with automatic MovingObjectField support.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
            use_smooth_union: Whether to use smooth union
            time: Time for MovingObjectField evaluation
        """
        if self.dim != 2:
            raise NotImplementedError("SDF rendering only implemented for 2D environments")

        # Create query grid
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=200, **self.tensor_args)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=200, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")

        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim)

        # Compute SDF
        sdf = self.compute_sdf(stacked_tensors, reshape_shape=X.shape,
                              use_smooth_union=use_smooth_union, time=time)

        sdf_np = to_numpy(sdf)
        ctf = ax.contourf(to_numpy(X), to_numpy(Y), sdf_np, levels=20, cmap='RdBu')
        ax.contour(to_numpy(X), to_numpy(Y), sdf_np, levels=[0], colors='black', linewidths=2)

        if fig is not None:
            fig.colorbar(ctf, ax=ax, orientation="vertical")

        ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
        ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def render_grad_sdf(self, ax=None, fig=None, use_smooth_union=True, time=None):
        """
        Render gradient of SDF field with automatic MovingObjectField support.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
            use_smooth_union: Whether to use smooth union
            time: Time for MovingObjectField evaluation
        """
        if self.dim != 2:
            raise NotImplementedError("Gradient rendering only implemented for 2D environments")

        # Create query grid (coarser for gradient visualization)
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=20, **self.tensor_args)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=20, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")

        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim)

        # Compute gradient using autograd
        stacked_tensors.requires_grad_(True)
        f_grad_sdf = lambda x: self.compute_sdf(x, reshape_shape=X.shape,
                                                use_smooth_union=use_smooth_union,
                                                time=time).sum()
        grad_sdf = jacobian(f_grad_sdf, stacked_tensors)

        grad_sdf_np = to_numpy(grad_sdf).squeeze()
        ax.quiver(
            to_numpy(X_flat), to_numpy(Y_flat),
            grad_sdf_np[:, 0], grad_sdf_np[:, 1],
            color="red"
        )

        ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
        ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def compare_sdf_methods(self, save_path=None):
        """
        Compare hard minimum vs smooth union visualization.

        Args:
            save_path: Path to save figure (optional)

        Returns:
            fig, axes
        """
        if self.dim != 2:
            raise NotImplementedError("Comparison only implemented for 2D environments")

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Hard minimum SDF
        self.render_sdf(ax=axes[0, 0], fig=fig, use_smooth_union=False)
        axes[0, 0].set_title(f'Hard Minimum (Non-differentiable at overlaps)')

        # Smooth union SDF
        self.render_sdf(ax=axes[0, 1], fig=fig, use_smooth_union=True)
        axes[0, 1].set_title(f'Smooth Union (k={self.k_smooth})')

        # Hard minimum gradient
        self.render_grad_sdf(ax=axes[1, 0], fig=fig, use_smooth_union=False)
        axes[1, 0].set_title('Gradient - Hard Minimum')

        # Smooth union gradient
        self.render_grad_sdf(ax=axes[1, 1], fig=fig, use_smooth_union=True)
        axes[1, 1].set_title(f'Gradient - Smooth Union (k={self.k_smooth})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")

        return fig, axes

    def create_render_fn_with_time(self, trajectory_time_steps):
        """
        Create a time-aware rendering function for use with animate_robot_trajectories.

        This function returns a wrapper that can replace self.env.render in animations
        to support time-varying obstacle visualization.

        Args:
            trajectory_time_steps: Tensor or array of time values for each trajectory step

        Returns:
            A function with signature (ax, frame_idx) that renders at the appropriate time

        Example:
            # In PlanningTask.animate_robot_trajectories, modify the animate_fn:
            time_steps = planning_task.parametric_trajectory.get_timesteps()
            render_fn = env.create_render_fn_with_time(time_steps)

            def animate_fn(i, ax):
                ax.clear()
                render_fn(ax, idxs[i])  # Render at time corresponding to frame i
                # ... rest of animation code
        """
        def render_at_frame(ax, frame_idx):
            """Render environment at specific frame index."""
            if frame_idx < len(trajectory_time_steps):
                time = trajectory_time_steps[frame_idx]
                if isinstance(time, torch.Tensor):
                    time = time.item()
                self.render(ax, time=time)
            else:
                # Fallback to last time or no time
                self.render(ax, time=self.time_range[1] if self.moving_obj_list_fn else None)

        return render_at_frame


    def animate_with_time(
        self,
        trajectory_time_steps,
        n_frames=50,
        video_filepath="time_varying_env.mp4",
        show_time_label=True,
        **kwargs
    ):
        """
        Create an animation of the time-varying environment.

        This is useful for visualizing moving obstacles without robot trajectories.

        Args:
            trajectory_time_steps: Tensor or array of time values for animation
            n_frames: Number of frames in the animation
            video_filepath: Path to save the video
            show_time_label: Whether to show time label on the plot
            **kwargs: Additional arguments passed to create_animation_video

        Example:
            # Animate moving obstacles over time
            time_steps = torch.linspace(0.0, 1.0, 100)
            env.animate_with_time(time_steps, n_frames=50, video_filepath='obstacles.mp4')
        """
        from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
        import numpy as np

        # Convert to numpy if needed
        if isinstance(trajectory_time_steps, torch.Tensor):
            trajectory_time_steps_np = to_numpy(trajectory_time_steps)
        else:
            trajectory_time_steps_np = np.array(trajectory_time_steps)

        # Select frame indices
        H = len(trajectory_time_steps_np)
        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        times_selected = trajectory_time_steps_np[idxs]

        fig, ax = create_fig_and_axes(dim=self.dim)

        def animate_fn(i, ax):
            ax.clear()
            time_current = times_selected[i]

            if show_time_label:
                ax.set_title(f"Time: {time_current:.3f}s")

            # Render environment at this time
            self.render(ax, time=time_current)

        create_animation_video(
            fig, animate_fn, n_frames=n_frames, video_filepath=video_filepath, fargs=(ax,), **kwargs
        )

    def animate_sdf_with_extra_objects(
        self,
        time_range=None,
        n_frames=50,
        video_filepath="sdf_animation.mp4",
        use_smooth_union=True,
        show_obstacles=True,
        vmin=None,
        vmax=None,
        **kwargs
    ):
        """
        Create an animation of SDF field as MovingObjectField instances move over time.

        Automatically detects and animates any MovingObjectField objects in the environment.

        Args:
            time_range: (t_min, t_max) time range for animation. If None, uses self.time_range
            n_frames: Number of frames in animation
            video_filepath: Path to save video
            use_smooth_union: Whether to use smooth union (True) or hard minimum (False)
            show_obstacles: Whether to overlay obstacle boundaries
            vmin, vmax: Color scale limits for SDF visualization
            **kwargs: Additional arguments passed to create_animation_video

        Example:
            env.animate_sdf_with_extra_objects(
                time_range=(0.0, 1.0),
                n_frames=30,
                video_filepath='sdf_moving_obstacle.mp4'
            )
        """
        from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
        import numpy as np

        if self.dim != 2:
            raise NotImplementedError("SDF animation only implemented for 2D environments")

        if not self._has_moving_objects():
            raise ValueError("No MovingObjectField instances found in environment")

        # Use provided time range or fallback to self.time_range
        if time_range is None:
            time_range = self.time_range

        # Generate time steps
        times = np.linspace(time_range[0], time_range[1], n_frames)

        # Create query grid for SDF
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=200, **self.tensor_args)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=200, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim)

        fig, ax = create_fig_and_axes(dim=self.dim)

        # Determine color scale from first frame if not provided
        if vmin is None or vmax is None:
            sdf_initial = self.compute_sdf(stacked_tensors, reshape_shape=X.shape,
                                          use_smooth_union=use_smooth_union,
                                          time=times[0])
            if vmin is None:
                vmin = max(sdf_initial.min().item(), -0.6)
            if vmax is None:
                vmax = min(sdf_initial.max().item(), 0.6)

        def animate_fn(i, ax):
            ax.clear()
            time_current = times[i]

            # Compute SDF at this time (automatically updates MovingObjectField poses)
            sdf = self.compute_sdf(stacked_tensors, reshape_shape=X.shape,
                                  use_smooth_union=use_smooth_union,
                                  time=time_current)
            sdf_np = to_numpy(sdf)

            # Plot SDF
            ctf = ax.contourf(to_numpy(X), to_numpy(Y), sdf_np,
                            levels=np.linspace(vmin, vmax, 21),
                            cmap='RdBu', vmin=vmin, vmax=vmax)
            ax.contour(to_numpy(X), to_numpy(Y), sdf_np, levels=[0],
                      colors='black', linewidths=2)

            # Overlay obstacles if requested
            if show_obstacles:
                # Render all objects (already at correct time)
                if self.obj_fixed_list:
                    for obj in self.obj_fixed_list:
                        obj.render(ax, alpha=0.3, color='gray')

                if self.obj_extra_list:
                    for obj in self.obj_extra_list:
                        obj.render(ax, alpha=0.5, color='red')

            ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
            ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            method_name = self.smoothing_method if use_smooth_union else "Hard Min"
            ax.set_title(f"SDF ({method_name}, k={self.k_smooth:.1f})\nTime: {time_current:.3f}s",
                        fontsize=12)

        create_animation_video(
            fig, animate_fn, n_frames=n_frames, video_filepath=video_filepath, fargs=(ax,), **kwargs
        )

        print(f"Saved SDF animation to {video_filepath}")

    def animate_grad_sdf_with_extra_objects(
        self,
        time_range=None,
        n_frames=50,
        video_filepath="grad_sdf_animation.mp4",
        use_smooth_union=True,
        show_obstacles=True,
        arrow_scale=None,
        **kwargs
    ):
        """
        Create an animation of gradient SDF field as MovingObjectField instances move.

        Automatically detects and animates gradients for any MovingObjectField objects.

        Args:
            time_range: (t_min, t_max) time range for animation. If None, uses self.time_range
            n_frames: Number of frames in animation
            video_filepath: Path to save video
            use_smooth_union: Whether to use smooth union (True) or hard minimum (False)
            show_obstacles: Whether to overlay obstacle boundaries
            arrow_scale: Scale for gradient arrows (None = auto)
            **kwargs: Additional arguments passed to create_animation_video

        Example:
            env.animate_grad_sdf_with_extra_objects(
                time_range=(0.0, 1.0),
                n_frames=30,
                video_filepath='grad_sdf_moving.mp4'
            )
        """
        from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
        import numpy as np

        if self.dim != 2:
            raise NotImplementedError("Gradient SDF animation only implemented for 2D environments")

        if not self._has_moving_objects():
            raise ValueError("No MovingObjectField instances found in environment")

        if time_range is None:
            time_range = self.time_range

        # Generate time steps
        times = np.linspace(time_range[0], time_range[1], n_frames)

        # Create coarser query grid for gradients
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=20, **self.tensor_args)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=20, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim)

        fig, ax = create_fig_and_axes(dim=self.dim)

        def animate_fn(i, ax):
            ax.clear()
            time_current = times[i]

            # Compute gradient using autograd (automatically updates MovingObjectField poses)
            stacked_tensors_grad = stacked_tensors.clone().detach().requires_grad_(True)
            f_grad_sdf = lambda x: self.compute_sdf(x, reshape_shape=X.shape,
                                                    use_smooth_union=use_smooth_union,
                                                    time=time_current).sum()
            grad_sdf = jacobian(f_grad_sdf, stacked_tensors_grad)
            grad_sdf_np = to_numpy(grad_sdf).squeeze()

            # Plot gradients
            ax.quiver(
                to_numpy(X_flat), to_numpy(Y_flat),
                grad_sdf_np[:, 0], grad_sdf_np[:, 1],
                color="red", scale=arrow_scale
            )

            # Overlay obstacles if requested (already at correct time)
            if show_obstacles:
                if self.obj_fixed_list:
                    for obj in self.obj_fixed_list:
                        obj.render(ax, alpha=0.3, color='gray')

                if self.obj_extra_list:
                    for obj in self.obj_extra_list:
                        obj.render(ax, alpha=0.5, color='red')

            ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
            ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            method_name = self.smoothing_method if use_smooth_union else "Hard Min"
            ax.set_title(f"Gradient SDF ({method_name}, k={self.k_smooth:.1f})\nTime: {time_current:.3f}s",
                        fontsize=12)

        create_animation_video(
            fig, animate_fn, n_frames=n_frames, video_filepath=video_filepath, fargs=(ax,), **kwargs
        )

        print(f"Saved gradient SDF animation to {video_filepath}")


if __name__ == "__main__":
    # Example usage demonstrating wrapper pattern with MovingObjectField
    from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField, ObjectField
    from torch_robotics.environments.dynamic_extension.trajectory import LinearTrajectory, CircularTrajectory
    import numpy as np

    tensor_args = DEFAULT_TENSOR_ARGS

    # Example 1: Static environment with overlapping objects
    obj_list = [
        MultiSphereField(
            centers=np.array([[0.0, 0.0], [0.3, 0.0]]),  # Overlapping spheres
            radii=np.array([0.25, 0.25]),
            tensor_args=tensor_args
        ),
        MultiBoxField(
            centers=np.array([[0.5, 0.5]]),
            sizes=np.array([[0.3, 0.3]]),
            tensor_args=tensor_args
        ),
    ]

    env = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "test_objects")],
        k_smooth=30.0,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    # Compare methods
    env.compare_sdf_methods(save_path='/tmp/env_dyn_base_comparison.png')
    print("Static environment visualization saved!")

    # Example 2: Dynamic environment with MovingObjectField
    # Create a moving sphere
    sphere_prim = MultiSphereField(
        centers=np.array([[0.0, 0.0]]),
        radii=np.array([0.2]),
        tensor_args=tensor_args
    )

    # Define a circular trajectory
    circular_traj = CircularTrajectory(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.6,
        angular_velocity=2 * np.pi,  # One full rotation per second
        initial_phase=0.0,
        axis='z',
        tensor_args=tensor_args
    )

    # Create moving object
    moving_sphere = MovingObjectField(
        primitive_fields=[sphere_prim],
        trajectory=circular_traj,
        name="moving_sphere"
    )

    # Create environment with both static and moving objects
    env_dynamic = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=[ObjectField(obj_list, "static_objects")],
        obj_extra_list=[moving_sphere],  # MovingObjectField in extra list
        k_smooth=30.0,
        smoothing_method="Quadratic",
        time_range=(0.0, 1.0),
        tensor_args=tensor_args
    )

    print("Dynamic environment created with MovingObjectField!")
    print("MovingObjectField is automatically handled in render() and compute_sdf()")
