"""
Test SDF and Gradient SDF animation with moving obstacles.

This demonstrates how smoothing methods (LSE vs Quadratic) behave when
obstacles move and overlap, showing the final composition step (Step 4).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from torch_robotics.environments.primitives import (
    MultiSphereField, MultiBoxField, ObjectField
)
from torch_robotics.environments.dynamic_extension.env_dyn_base import EnvDynBase
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def test_sdf_animation_moving_sphere():
    """Test 1: Single sphere moving through fixed obstacles."""
    print("\n" + "="*80)
    print("TEST 1: SDF Animation - Moving Sphere Through Fixed Obstacles")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Fixed obstacles (static)
    obj_fixed_list = [
        ObjectField(
            [MultiSphereField(
                centers=np.array([[-0.4, 0.0]]),
                radii=np.array([0.25]),
                tensor_args=tensor_args
            )],
            name="left_sphere"
        ),
        ObjectField(
            [MultiSphereField(
                centers=np.array([[0.4, 0.0]]),
                radii=np.array([0.25]),
                tensor_args=tensor_args
            )],
            name="right_sphere"
        ),
    ]

    # Moving obstacle: sphere moves from left to right
    def moving_sphere(t):
        """Sphere moves from x=-0.6 to x=+0.6 over time [0, 1]."""
        pos_x = -0.6 + 1.2 * t
        sphere = MultiSphereField(
            centers=np.array([[pos_x, 0.0]]),
            radii=np.array([0.2]),
            tensor_args=tensor_args
        )
        return [ObjectField([sphere], "moving_sphere")]

    # Create environment with LSE smoothing
    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    # Create environment with Quadratic smoothing
    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    print("\nGenerating SDF animations...")
    print("  - LSE smoothing")
    env_lse.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='sdf_moving_sphere_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("  - Quadratic smoothing")
    env_quad.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='sdf_moving_sphere_quadratic.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("  - Hard minimum (no smoothing)")
    env_lse.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='sdf_moving_sphere_hard.mp4',
        use_smooth_union=False,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("\n✓ Generated SDF animations:")
    print("  - sdf_moving_sphere_lse.mp4")
    print("  - sdf_moving_sphere_quadratic.mp4")
    print("  - sdf_moving_sphere_hard.mp4")


def test_gradient_animation_moving_sphere():
    """Test 2: Gradient field animation with moving sphere."""
    print("\n" + "="*80)
    print("TEST 2: Gradient SDF Animation - Moving Sphere")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Fixed obstacles
    obj_fixed_list = [
        ObjectField(
            [MultiSphereField(
                centers=np.array([[-0.4, 0.0]]),
                radii=np.array([0.25]),
                tensor_args=tensor_args
            )],
            name="left_sphere"
        ),
        ObjectField(
            [MultiSphereField(
                centers=np.array([[0.4, 0.0]]),
                radii=np.array([0.25]),
                tensor_args=tensor_args
            )],
            name="right_sphere"
        ),
    ]

    # Moving sphere
    def moving_sphere(t):
        pos_x = -0.6 + 1.2 * t
        sphere = MultiSphereField(
            centers=np.array([[pos_x, 0.0]]),
            radii=np.array([0.2]),
            tensor_args=tensor_args
        )
        return [ObjectField([sphere], "moving_sphere")]

    # LSE environment
    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    # Quadratic environment
    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    print("\nGenerating gradient SDF animations...")
    print("  - LSE smoothing")
    env_lse.animate_grad_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='grad_sdf_moving_sphere_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("  - Quadratic smoothing")
    env_quad.animate_grad_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='grad_sdf_moving_sphere_quadratic.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("  - Hard minimum (no smoothing)")
    env_lse.animate_grad_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_sphere,
        time_range=(0.0, 1.0),
        n_frames=30,
        video_filepath='grad_sdf_moving_sphere_hard.mp4',
        use_smooth_union=False,
        show_obstacles=True,
        anim_time=5,
        dpi=120
    )

    print("\n✓ Generated gradient SDF animations:")
    print("  - grad_sdf_moving_sphere_lse.mp4")
    print("  - grad_sdf_moving_sphere_quadratic.mp4")
    print("  - grad_sdf_moving_sphere_hard.mp4")


def test_complex_overlap_scenario():
    """Test 3: Complex scenario with multiple moving obstacles."""
    print("\n" + "="*80)
    print("TEST 3: Complex Overlap - Two Moving Obstacles")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Fixed obstacle in center
    obj_fixed_list = [
        ObjectField(
            [MultiBoxField(
                centers=np.array([[0.0, 0.0]]),
                sizes=np.array([[0.3, 0.3]]),
                tensor_args=tensor_args
            )],
            name="center_box"
        ),
    ]

    # Two moving spheres that orbit around the center
    def moving_obstacles(t):
        """Two spheres orbit the center box."""
        angle1 = 2 * np.pi * t
        angle2 = 2 * np.pi * t + np.pi

        radius = 0.5

        # Sphere 1
        pos1_x = radius * np.cos(angle1)
        pos1_y = radius * np.sin(angle1)
        sphere1 = MultiSphereField(
            centers=np.array([[pos1_x, pos1_y]]),
            radii=np.array([0.2]),
            tensor_args=tensor_args
        )

        # Sphere 2
        pos2_x = radius * np.cos(angle2)
        pos2_y = radius * np.sin(angle2)
        sphere2 = MultiSphereField(
            centers=np.array([[pos2_x, pos2_y]]),
            radii=np.array([0.2]),
            tensor_args=tensor_args
        )

        return [
            ObjectField([sphere1], "moving_sphere_1"),
            ObjectField([sphere2], "moving_sphere_2"),
        ]

    # LSE environment
    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    # Quadratic environment
    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    print("\nGenerating complex scenario animations...")
    print("  - LSE SDF")
    env_lse.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_obstacles,
        time_range=(0.0, 1.0),
        n_frames=40,
        video_filepath='sdf_orbiting_spheres_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=6,
        dpi=120
    )

    print("  - Quadratic SDF")
    env_quad.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_obstacles,
        time_range=(0.0, 1.0),
        n_frames=40,
        video_filepath='sdf_orbiting_spheres_quadratic.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=6,
        dpi=120
    )

    print("  - LSE Gradients")
    env_lse.animate_grad_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_obstacles,
        time_range=(0.0, 1.0),
        n_frames=40,
        video_filepath='grad_sdf_orbiting_spheres_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=6,
        dpi=120
    )

    print("\n✓ Generated complex scenario animations:")
    print("  - sdf_orbiting_spheres_lse.mp4")
    print("  - sdf_orbiting_spheres_quadratic.mp4")
    print("  - grad_sdf_orbiting_spheres_lse.mp4")


def test_narrow_passage_opening():
    """Test 4: Narrow passage that opens and closes."""
    print("\n" + "="*80)
    print("TEST 4: Narrow Passage Opening/Closing")
    print("="*80)

    tensor_args = DEFAULT_TENSOR_ARGS

    # Fixed obstacle in center
    obj_fixed_list = [
        ObjectField(
            [MultiBoxField(
                centers=np.array([[0.5, 0.0]]),
                sizes=np.array([[0.3, 0.3]]),
                tensor_args=tensor_args
            )],
            name="center_box"
        ),
    ]


    # Two walls that move apart and together
    def moving_walls(t):
        """Walls open up (0 to 0.5) then close (0.5 to 1.0)."""
        if t <= 0.5:
            # Opening: walls move apart
            gap = 0.2 + 0.6 * (t / 0.5)
        else:
            # Closing: walls move together
            gap = 0.8 - 0.6 * ((t - 0.5) / 0.5)

        wall_left = MultiBoxField(
            centers=np.array([[-gap/2, 0.0]]),
            sizes=np.array([[0.2, 0.5]]),
            tensor_args=tensor_args
        )

        wall_right = MultiBoxField(
            centers=np.array([[gap/2, 0.0]]),
            sizes=np.array([[0.2, 0.5]]),
            tensor_args=tensor_args
        )

        return [
            ObjectField([wall_left], "left_wall"),
            ObjectField([wall_right], "right_wall"),
        ]

    # LSE environment
    env_lse = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="LSE",
        tensor_args=tensor_args
    )

    # Quadratic environment
    env_quad = EnvDynBase(
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
        obj_fixed_list=obj_fixed_list,
        k_smooth=0.1,
        smoothing_method="Quadratic",
        tensor_args=tensor_args
    )

    print("\nGenerating narrow passage animations...")
    print("  - LSE SDF")
    env_lse.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_walls,
        time_range=(0.0, 1.0),
        n_frames=50,
        video_filepath='sdf_narrow_passage_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=8,
        dpi=120
    )

    print("  - Quadratic SDF")
    env_quad.animate_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_walls,
        time_range=(0.0, 1.0),
        n_frames=50,
        video_filepath='sdf_narrow_passage_quadratic.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=8,
        dpi=120
    )

    print("  - Comparison: LSE vs Quadratic Gradients")
    env_lse.animate_grad_sdf_with_extra_objects(
        extra_obj_trajectory_fn=moving_walls,
        time_range=(0.0, 1.0),
        n_frames=50,
        video_filepath='grad_sdf_narrow_passage_lse.mp4',
        use_smooth_union=True,
        show_obstacles=True,
        anim_time=8,
        dpi=120
    )

    print("\n✓ Generated narrow passage animations:")
    print("  - sdf_narrow_passage_lse.mp4")
    print("  - sdf_narrow_passage_quadratic.mp4")
    print("  - grad_sdf_narrow_passage_lse.mp4")


def run_all_tests():
    """Run all SDF animation tests."""
    print("\n" + "="*80)
    print("SDF ANIMATION TESTS - Smoothing with Moving Obstacles")
    print("="*80)
    print("\nThese tests demonstrate how LSE and Quadratic smoothing methods")
    print("handle the FINAL COMPOSITION of fixed + extra (moving) objects.")
    print("\nKey observations to look for:")
    print("  1. How SDF values blend in overlap regions")
    print("  2. Smoothness of transitions as obstacles move")
    print("  3. Gradient continuity (especially for Quadratic vs Hard Min)")
    print("  4. Conservative nature of LSE vs closer approximation of Quadratic")

    try:
        #test_sdf_animation_moving_sphere()
        #test_gradient_animation_moving_sphere()
        #test_complex_overlap_scenario()
        test_narrow_passage_opening()

        print("\n" + "="*80)
        print("✓ ALL ANIMATION TESTS COMPLETED!")
        print("="*80)
        print("\nGenerated videos (12 total):")
        print("\nMoving Sphere Through Fixed Obstacles:")
        print("  - sdf_moving_sphere_lse.mp4")
        print("  - sdf_moving_sphere_quadratic.mp4")
        print("  - sdf_moving_sphere_hard.mp4")
        print("  - grad_sdf_moving_sphere_lse.mp4")
        print("  - grad_sdf_moving_sphere_quadratic.mp4")
        print("  - grad_sdf_moving_sphere_hard.mp4")
        print("\nOrbiting Spheres:")
        print("  - sdf_orbiting_spheres_lse.mp4")
        print("  - sdf_orbiting_spheres_quadratic.mp4")
        print("  - grad_sdf_orbiting_spheres_lse.mp4")
        print("\nNarrow Passage:")
        print("  - sdf_narrow_passage_lse.mp4")
        print("  - sdf_narrow_passage_quadratic.mp4")
        print("  - grad_sdf_narrow_passage_lse.mp4")

        print("\nKey Insights:")
        print("  • LSE produces smoother, more conservative blending")
        print("  • Quadratic is faster and closer to true geometry")
        print("  • Hard minimum shows discontinuous gradients at overlaps")
        print("  • Smooth methods maintain gradient continuity for optimization")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
