"""
Dynamic environment base class with overlap detection and smooth SDF composition.

This module extends EnvBase to handle potentially overlapping objects using
smooth SDF unions for differentiability.
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

from torch_robotics.environments.env_base import EnvBase
#from torch_robotics.environments.dynamic_extension.grid_map_dyn import GridMapDynSDF
from torch_robotics.environments.grid_map_sdf import GridMapSDF
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics.torch_utils.torch_timer import TimerCUDA
# class EnvBase(ABC):

#     def __init__(
#         self,
#         limits=None,
#         obj_fixed_list=None,
#         obj_extra_list=None,
#         precompute_sdf_obj_fixed=True,
#         precompute_sdf_obj_extra=True,
#         sdf_cell_size=None,
#         tensor_args=DEFAULT_TENSOR_ARGS,
#         **kwargs,
#     ):

class EnvDynBase(EnvBase):
    """
    Dynamic environment base class that handles overlapping objects.

    Unlike EnvBase which uses torch.minimum (assumes no overlaps),
    this class detects overlaps and applies smooth union for differentiability.

    Supports time-varying rendering for moving obstacles.
    """

    def __init__(
        self,
        k_smooth=20.0,
        moving_obj_list_fn=None,
        smoothing_method="Quadratic",
        time_range=(0.0, 1.0),
        **kwargs
    ):
        """
        Args:
            k_smooth: Smoothness parameter for log-sum-exp union (higher = sharper)
            moving_obj_list_fn: Optional function that takes time t and returns list of ObjectField at that time
            time_range: (t_min, t_max) for time-varying objects
            **kwargs: Arguments passed to EnvBase
        """
        super().__init__(**kwargs)

        self.k_smooth = k_smooth
        self.smoothing_method=smoothing_method
        # Time-varying obstacle support

        self.moving_obj_list_fn = moving_obj_list_fn
        self.time_range = time_range

    # def build_sdf_grid(self, compute_sdf_obj_fixed=False, compute_sdf_obj_extra=False):
    #     if compute_sdf_obj_fixed:
    #         with TimerCUDA() as t:
    #             # Compute SDF grid
    #             self.grid_map_sdf_obj_fixed = GridMapDynSDF(
    #                 self.limits, self.sdf_cell_size, self.obj_fixed_list, tensor_args=self.tensor_args
    #             )
    #         print(f"Computing the SDF grid and gradients of FIXED objects took: {t.elapsed:.3f} sec")

    #     if self.obj_extra_list and compute_sdf_obj_extra:
    #         with TimerCUDA() as t:
    #             # Compute SDF grid
    #             self.grid_map_sdf_obj_extra = GridMapDynSDF(
    #                 self.limits, self.sdf_cell_size, self.obj_extra_list, tensor_args=self.tensor_args
    #             )
    #         print(f"Computing the SDF grid and gradients of EXTRA objects took: {t.elapsed:.3f} sec")

    def render(self, ax=None, time=None):
        """
        Render the environment, optionally at a specific time for moving obstacles.

        Args:
            ax: Matplotlib axis
            time: Optional time parameter for time-varying obstacles.
                  If provided, renders moving obstacles at this time.
                  If None, renders static objects only.
        """
        # Render fixed objects (always present)
        if self.obj_fixed_list:
            for obj in self.obj_fixed_list:
                obj.render(ax)

        # Render extra objects (static)
        if self.obj_extra_list:
            for obj in self.obj_extra_list:
                obj.render(ax, color="red", cmap="Reds")

        # Render moving objects at specific time
        if time is not None and self.moving_obj_list_fn is not None:
            # Clamp time to valid range
            time_clamped = max(self.time_range[0], min(self.time_range[1], time))

            # Get object configuration at this time
            moving_objs = self.moving_obj_list_fn(time_clamped)

            # Render each moving object
            for obj in moving_objs:
                obj.render(ax, color="purple", cmap="Purples", alpha=0.7)

        # Set axes limits and labels
        ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
        ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
        if self.dim == 3:
            ax.set_zlim(self.limits_np[0][2], self.limits_np[1][2])
            ax.set_zlabel("z")
        ax.set_aspect("equal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def compute_sdf(self, x, reshape_shape=None, use_smooth_union=True, smoothing_method="Quadratic"):
        """
        Compute SDF with optional smooth union for all objects.

        This overrides EnvBase.compute_sdf() to use smooth union instead of hard minimum.

        Args:
            x: Query points, shape (..., dim)
            reshape_shape: Optional shape to reshape output
            use_smooth_union: If True, use smooth union (logsumexp);
                            if False, fall back to hard minimum

        Returns:
            SDF values at query points
        """
        if not use_smooth_union:
            # Fall back to parent class behavior (hard minimum)
            return super().compute_sdf(x, reshape_shape=reshape_shape)


        if smoothing_method == "LSE" : 
            print("Smoothing method : ",smoothing_method)
            # Collect all SDFs
            all_sdfs = []

            # Compute SDF for fixed objects
            # if self.grid_map_sdf_obj_fixed is not None:
            #     # Grid-based SDF (already precomputed)
            #     sdf_fixed = self.grid_map_sdf_obj_fixed(x)
            #     if reshape_shape:
            #         sdf_fixed = sdf_fixed.reshape(reshape_shape)
            #     all_sdfs.append(sdf_fixed)
            # else:
                # Compute SDF from each object
            for obj in self.obj_fixed_list:
                print(obj)
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                all_sdfs.append(sdf_obj)

            # Compute SDF for extra objects
            # if self.grid_map_sdf_obj_extra is not None:
            #     # Grid-based SDF
            #     sdf_extra = self.grid_map_sdf_obj_extra(x)
            #     if reshape_shape:
            #         sdf_extra = sdf_extra.reshape(reshape_shape)
            #     all_sdfs.append(sdf_extra)

            # else :
                # Compute SDF from each object
            for obj in self.obj_extra_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                all_sdfs.append(sdf_obj)

            # Handle empty case
            if len(all_sdfs) == 0:
                return torch.ones(x.shape[:-1], **self.tensor_args) * float('inf')

            # Apply smooth union to all SDFs at once
            if len(all_sdfs) == 1:
                return all_sdfs[0]
            else:
                # Stack all SDFs and apply logsumexp
                # φ_union = -k * logsumexp(-φ_i/k)
                sdfs_stacked = torch.stack(all_sdfs, dim=-1)
                print(sdfs_stacked.shape)
                return -self.k_smooth * torch.logsumexp(-sdfs_stacked / self.k_smooth, dim=-1)
            
        elif smoothing_method == "Quadratic" : 
            print("Smoothing method : ",smoothing_method)
            sdf = None
            # input a, b : sdf gridmap, k : self.k_smooth
            def smin(a, b, k) : 
                k *= 4.0 #normalizing factor https://iquilezles.org/articles/smin/
                h = torch.maximum(k - torch.abs(a - b), torch.zeros_like(a)) / k; 
                return torch.minimum(a,b) - k*0.25*h*h
            
            # if the sdf of fixed objects is precomputed, then use it
            # if self.grid_map_sdf_obj_fixed is not None:
            #     sdf = self.grid_map_sdf_obj_fixed(x)
            #     if reshape_shape:
            #         sdf = sdf.reshape(reshape_shape)
            # else:
            if self.obj_fixed_list : 
                for obj in self.obj_fixed_list:
                    sdf_obj = obj.compute_signed_distance(x)
                    if reshape_shape is not None:
                        sdf_obj = sdf_obj.reshape(reshape_shape)
                    if sdf is None:
                        sdf = sdf_obj
                    else:
                        sdf = smin(sdf, sdf_obj, self.k_smooth)

            # compute sdf of extra objects
            sdf_extra_objects = None
            # if self.obj_extra_list:
            #     if self.grid_map_sdf_obj_extra is not None:
            #         sdf_extra_objects = self.grid_map_sdf_obj_extra(x)
            #         if reshape_shape:
            #             sdf_extra_objects = sdf_extra_objects.reshape(reshape_shape)
            #     else:
            if self.obj_extra_list : 
                for obj in self.obj_extra_list:
                    sdf_obj = obj.compute_signed_distance(x)
                    if reshape_shape is not None:
                        sdf_obj = sdf_obj.reshape(reshape_shape)
                    if sdf_extra_objects is None:
                        sdf_extra_objects = sdf_obj
                    else:
                        sdf_extra_objects = smin(sdf_extra_objects, sdf_obj, self.k_smooth)
                if sdf is None:
                    sdf = sdf_extra_objects
                else:
                    sdf = smin(sdf, sdf_extra_objects, self.k_smooth)

            return sdf

        else :
            raise NotImplementedError("smoothing method not implemented")

    def render_sdf(self, ax=None, fig=None, use_smooth_union=True):
        """
        Render SDF field. Compatible with env_base.py visualization.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
            use_smooth_union: Whether to use smooth union
        """
        if self.dim != 2:
            raise NotImplementedError("Rendering only implemented for 2D environments")

        # Create query grid
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=200, **self.tensor_args)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=200, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")

        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim)

        # Compute SDF
        sdf = self.compute_sdf(stacked_tensors, reshape_shape=X.shape, use_smooth_union=use_smooth_union, smoothing_method=self.smoothing_method)

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

    def render_grad_sdf(self, ax=None, fig=None, use_smooth_union=True):
        """
        Render gradient of SDF field. Compatible with env_base.py visualization.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
            use_smooth_union: Whether to use smooth union
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
        f_grad_sdf = lambda x: self.compute_sdf(x, reshape_shape=X.shape, use_smooth_union=use_smooth_union, smoothing_method=self.smoothing_method).sum()
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
        extra_obj_trajectory_fn=None,
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
        Create an animation of SDF field as extra objects move over time.

        This visualizes how the SDF composition (fixed + extra) changes with smoothing.

        Args:
            extra_obj_trajectory_fn: Function(t) that returns list of ObjectField for extra_list at time t
                                    If None, uses self.moving_obj_list_fn
            time_range: (t_min, t_max) time range for animation. If None, uses self.time_range
            n_frames: Number of frames in animation
            video_filepath: Path to save video
            use_smooth_union: Whether to use smooth union (True) or hard minimum (False)
            show_obstacles: Whether to overlay obstacle boundaries
            vmin, vmax: Color scale limits for SDF visualization
            **kwargs: Additional arguments passed to create_animation_video

        Example:
            # Animate SDF as obstacles move
            def moving_obstacles(t):
                pos_x = -0.5 + t  # Move from left to right
                sphere = MultiSphereField(centers=[[pos_x, 0.0]], radii=[0.2])
                return [ObjectField([sphere], "moving")]

            env.animate_sdf_with_extra_objects(
                extra_obj_trajectory_fn=moving_obstacles,
                time_range=(0.0, 1.0),
                n_frames=30,
                video_filepath='sdf_moving_obstacle.mp4'
            )
        """
        from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
        import numpy as np

        if self.dim != 2:
            raise NotImplementedError("SDF animation only implemented for 2D environments")

        # Use provided function or fallback to moving_obj_list_fn
        if extra_obj_trajectory_fn is None:
            if self.moving_obj_list_fn is None:
                raise ValueError("Either extra_obj_trajectory_fn or moving_obj_list_fn must be provided")
            extra_obj_trajectory_fn = self.moving_obj_list_fn

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
            # Temporarily set extra objects to compute initial SDF
            original_extra_list = self.obj_extra_list
            self.obj_extra_list = extra_obj_trajectory_fn(times[0])
            sdf_initial = self.compute_sdf(stacked_tensors, reshape_shape=X.shape,
                                          use_smooth_union=use_smooth_union,
                                          smoothing_method=self.smoothing_method)
            if vmin is None:
                vmin = max(sdf_initial.min().item(), -0.6)
            if vmax is None:
                vmax = min(sdf_initial.max().item(), 0.6)
            self.obj_extra_list = original_extra_list

        def animate_fn(i, ax):
            ax.clear()
            time_current = times[i]

            # Update extra objects for this time
            original_extra_list = self.obj_extra_list
            self.obj_extra_list = extra_obj_trajectory_fn(time_current)

            # Compute SDF
            sdf = self.compute_sdf(stacked_tensors, reshape_shape=X.shape,
                                  use_smooth_union=use_smooth_union,
                                  smoothing_method=self.smoothing_method)
            sdf_np = to_numpy(sdf)

            # Plot SDF
            ctf = ax.contourf(to_numpy(X), to_numpy(Y), sdf_np,
                            levels=np.linspace(vmin, vmax, 21),
                            cmap='RdBu', vmin=vmin, vmax=vmax)
            ax.contour(to_numpy(X), to_numpy(Y), sdf_np, levels=[0],
                      colors='black', linewidths=2)

            # Overlay obstacles if requested
            if show_obstacles:
                # Render fixed objects
                if self.obj_fixed_list:
                    for obj in self.obj_fixed_list:
                        obj.render(ax, alpha=0.3, color='blue')

                # Render extra objects
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

            # Restore original extra objects
            self.obj_extra_list = original_extra_list

        create_animation_video(
            fig, animate_fn, n_frames=n_frames, video_filepath=video_filepath, fargs=(ax,), **kwargs
        )

        print(f"✓ Saved SDF animation to {video_filepath}")

    def animate_grad_sdf_with_extra_objects(
        self,
        extra_obj_trajectory_fn=None,
        time_range=None,
        n_frames=50,
        video_filepath="grad_sdf_animation.mp4",
        use_smooth_union=True,
        show_obstacles=True,
        arrow_scale=None,
        **kwargs
    ):
        """
        Create an animation of gradient SDF field as extra objects move over time.

        This visualizes how gradients change with smoothing as obstacles overlap.

        Args:
            extra_obj_trajectory_fn: Function(t) that returns list of ObjectField for extra_list at time t
            time_range: (t_min, t_max) time range for animation
            n_frames: Number of frames in animation
            video_filepath: Path to save video
            use_smooth_union: Whether to use smooth union (True) or hard minimum (False)
            show_obstacles: Whether to overlay obstacle boundaries
            arrow_scale: Scale for gradient arrows (None = auto)
            **kwargs: Additional arguments passed to create_animation_video

        Example:
            env.animate_grad_sdf_with_extra_objects(
                extra_obj_trajectory_fn=moving_obstacles,
                time_range=(0.0, 1.0),
                n_frames=30,
                video_filepath='grad_sdf_moving.mp4'
            )
        """
        from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video
        import numpy as np

        if self.dim != 2:
            raise NotImplementedError("Gradient SDF animation only implemented for 2D environments")

        # Use provided function or fallback
        if extra_obj_trajectory_fn is None:
            if self.moving_obj_list_fn is None:
                raise ValueError("Either extra_obj_trajectory_fn or moving_obj_list_fn must be provided")
            extra_obj_trajectory_fn = self.moving_obj_list_fn

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

            # Update extra objects for this time
            original_extra_list = self.obj_extra_list
            self.obj_extra_list = extra_obj_trajectory_fn(time_current)

            # Compute gradient using autograd
            stacked_tensors_grad = stacked_tensors.clone().detach().requires_grad_(True)
            f_grad_sdf = lambda x: self.compute_sdf(x, reshape_shape=X.shape,
                                                    use_smooth_union=use_smooth_union,
                                                    smoothing_method=self.smoothing_method).sum()
            grad_sdf = jacobian(f_grad_sdf, stacked_tensors_grad)
            grad_sdf_np = to_numpy(grad_sdf).squeeze()

            # Plot gradients
            ax.quiver(
                to_numpy(X_flat), to_numpy(Y_flat),
                grad_sdf_np[:, 0], grad_sdf_np[:, 1],
                color="red", scale=arrow_scale
            )

            # Overlay obstacles if requested
            if show_obstacles:
                # Render fixed objects
                if self.obj_fixed_list:
                    for obj in self.obj_fixed_list:
                        obj.render(ax, alpha=0.3, color='blue')

                # Render extra objects
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

            # Restore original extra objects
            self.obj_extra_list = original_extra_list

        create_animation_video(
            fig, animate_fn, n_frames=n_frames, video_filepath=video_filepath, fargs=(ax,), **kwargs
        )

        print(f"✓ Saved gradient SDF animation to {video_filepath}")


if __name__ == "__main__":
    # Example usage
    from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField, ObjectField
    import numpy as np

    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with overlapping objects
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
        overlap_margin=0.05,
        tensor_args=tensor_args
    )

    # Compare methods
    env.compare_sdf_methods(save_path='/tmp/env_dyn_base_comparison.png')
    print("Visualization saved!")
