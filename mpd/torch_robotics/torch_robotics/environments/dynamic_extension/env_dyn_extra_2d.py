"""
EnvDense2D with dynamic extra objects using the wrapper-based EnvDynBase pattern.

This module demonstrates using MovingObjectField instances with a standard environment.
The dynamic objects are automatically handled by EnvDynBase's wrapper pattern.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from mpd.torch_robotics.torch_robotics.environments.env_dense_2d import EnvDense2D
from mpd.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    MovingObjectField,
    LinearTrajectory
)
from mpd.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from mpd.torch_robotics.torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvDense2DDynExtraObjects(EnvDynBase):
    """
    EnvDense2D environment with moving obstacles using the wrapper pattern.

    This class wraps EnvDense2D (which inherits from EnvBase) and adds moving obstacles
    using MovingObjectField instances. The wrapper automatically handles time-dependent
    rendering and SDF computation.
    """

    def __init__(
        self,
        time_range=(0.0, 5.0),
        k_smooth=30.0,
        smoothing_method="Quadratic",
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs
    ):
        """
        Args:
            time_range: (t_min, t_max) for moving obstacles
            k_smooth: Smoothness parameter for smooth SDF union
            smoothing_method: "Quadratic" or "LSE"
            tensor_args: Tensor device and dtype
            **kwargs: Additional arguments passed to EnvDense2D
        """

        # Create the base static environment (EnvDense2D inherits from EnvBase)
        # We'll pass this as the wrapped environment through the limits and obj_fixed_list
        base_env = EnvDense2D(tensor_args=tensor_args, **kwargs)

        # Create moving obstacles with trajectories
        # Moving sphere 1: Diagonal trajectory (2D primitives, 3D trajectory)
        prim_sphere = MultiSphereField(
            centers=np.array([
                [-0.4, 0.1],
                [-0.075, -0.85],
                [-0.1, -0.1],
            ]),
            radii=np.array([0.075, 0.1, 0.075]),
            tensor_args=tensor_args,
        )

        traj_sphere = LinearTrajectory(
            keyframe_times=[0.0, 5.0],
            keyframe_positions=[
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0]
            ],
            tensor_args=tensor_args
        )

        moving_sphere = MovingObjectField(
            primitive_fields=[prim_sphere],
            trajectory=traj_sphere,
            name="linear_sphere"
        )

        # Moving box: Opposite diagonal trajectory
        prim_box = MultiBoxField(
            centers=np.array([
                [0.45, -0.1],
                [0.35, 0.35],
                [-0.6, -0.85],
                [-0.65, -0.25],
            ]),
            sizes=np.array([
                [0.2, 0.2],
                [0.1, 0.15],
                [0.1, 0.25],
                [0.15, 0.1],
            ]),
            tensor_args=tensor_args,
        )

        traj_box = LinearTrajectory(
            keyframe_times=[0.0, 5.0],
            keyframe_positions=[
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0]
            ],
            tensor_args=tensor_args
        )

        moving_box = MovingObjectField(
            primitive_fields=[prim_box],
            trajectory=traj_box,
            name="linear_box"
        )

        # Initialize EnvDynBase wrapper with the base environment's configuration
        # and add moving objects to obj_extra_list
        super().__init__(
            limits=base_env.limits,
            obj_fixed_list=base_env.obj_fixed_list,  # Static obstacles from EnvDense2D
            obj_extra_list=[moving_sphere, moving_box],  # MovingObjectField instances
            precompute_sdf_obj_fixed=kwargs.get('precompute_sdf_obj_fixed', True),
            precompute_sdf_obj_extra=False,  # Don't precompute for moving objects
            sdf_cell_size=kwargs.get('sdf_cell_size', 0.01),
            time_range=time_range,
            k_smooth=k_smooth,
            smoothing_method=smoothing_method,
            tensor_args=tensor_args,
        )


if __name__ == "__main__":
    tensor_args = DEFAULT_TENSOR_ARGS

    # Create environment with moving obstacles
    env = EnvDense2DDynExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=tensor_args
    )

    print(f"Environment created:")
    print(f"  - Dimension: {env.dim}")
    print(f"  - Fixed objects: {len(env.obj_fixed_list)}")
    print(f"  - Extra (moving) objects: {len(env.obj_extra_list)}")
    print(f"  - Has moving objects: {env._has_moving_objects()}")
    print(f"  - Time range: {env.time_range}")

    # Test 1: Render at different times
    print("\nTest 1: Rendering at different times")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    for ax, t in zip(axes.flat, times):
        env.render(ax, time=t)
        ax.set_title(f"t = {t:.1f}s")

    plt.tight_layout()
    plt.savefig('/tmp/env_dyn_extra_2d_rendering.png', dpi=150)
    print("  Saved to /tmp/env_dyn_extra_2d_rendering.png")
    plt.close()

    # Test 2: Render SDF at different times
    print("\nTest 2: Time-varying SDF")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, t in zip(axes.flat, times):
        env.render_sdf(ax=ax, fig=None, use_smooth_union=True, time=t)
        ax.set_title(f"SDF at t = {t:.1f}s")

    plt.tight_layout()
    plt.savefig('/tmp/env_dyn_extra_2d_sdf.png', dpi=150)
    print("  Saved to /tmp/env_dyn_extra_2d_sdf.png")
    plt.close()

    # Test 3: Animate environment
    print("\nTest 3: Creating animation")
    env.animate_with_time(
        time_range=(0.0, 5.0),
        n_frames=50,
        video_filepath='/tmp/env_dyn_extra_2d_animation.mp4',
        show_time_label=True,
        anim_time=5,
        dpi=100,
    )
    print("  Saved animation to /tmp/env_dyn_extra_2d_animation.mp4")

    # Test 4: Test SDF queries at specific points over time
    print("\nTest 4: SDF queries over time")
    test_points = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
    ], **tensor_args)

    for point in test_points:
        print(f"\n  Point {point.cpu().numpy()}:")
        for t in [0.0, 2.5, 5.0]:
            sdf_val = env.compute_sdf(point.unsqueeze(0).unsqueeze(0), time=t)
            collision_str = " [COLLISION]" if sdf_val.item() < 0 else ""
            print(f"    t={t:.1f}s: SDF = {sdf_val.item():+.4f}{collision_str}")

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nKey features demonstrated:")
    print("  - EnvDynBase wraps existing EnvBase-based environments")
    print("  - MovingObjectField automatically handled in rendering")
    print("  - MovingObjectField automatically handled in SDF computation")
    print("  - Time-aware animations")
    print("  - Smooth union for overlapping obstacles")
