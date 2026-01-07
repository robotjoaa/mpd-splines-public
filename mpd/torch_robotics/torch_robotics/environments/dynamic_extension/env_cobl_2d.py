"""
Dynamic environment for COBL dataset evaluation.

This environment handles time-varying obstacles from COBL evaluation dataset.
Each scenario has multiple dynamic obstacles with full trajectories.

Author: Claude
Date: 2025-01-21
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Optional, Tuple

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiSphereField

from torch_robotics.environments.dynamic_extension import (
    EnvDynBase,
    MovingObjectField,
    WaypointTrajectory,
)
from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy, DEFAULT_TENSOR_ARGS

from mpd.paths import DATASET_BASE_DIR
import os 

class EnvCobl2D(EnvDynBase):
    """
    Dynamic 2D environment with COBL dataset obstacles.

    Each scenario contains multiple pedestrians (obstacles) with known trajectories
    over time. Obstacles are represented as moving spheres.

    Attributes:
        obstacle_trajectories: List of obstacle trajectories, each [T, 6] array
                              with [x, y, vx, vy, cos(θ), sin(θ)]
        obstacle_radius: Radius for obstacle spheres
        time_range: (t_min, t_max) for trajectory time range
    """

    def __init__(
        self,
        scenario_idx: int = 296, # isaac gym canont reset when secenario_idx change. 
        eval_dataset_path: Optional[str] = None,
        obstacle_trajectories: Optional[List[np.ndarray]] = None,
        obstacle_radius: float = 0.035,  # Scaled: 0.35m / 10 = 0.035
        dt: float = 0.1,
        horizon: int = 80,
        k_smooth: float = 20.0,
        name: str = 'EnvCobl2D',
        tensor_args: dict = DEFAULT_TENSOR_ARGS,
        **kwargs
    ):
        """
        Args:
            scenario_idx: Scenario index to load from eval dataset
            eval_dataset_path: Path to eval_dataset.pkl file
            obstacle_trajectories: Alternative to eval_dataset_path - directly provide trajectories
            obstacle_radius: Radius of obstacle spheres (meters)
            workspace_limits: (x_min, x_max, y_min, y_max) workspace bounds
            dt: Time step (seconds)
            horizon: Number of timesteps
            k_smooth: Smoothness parameter for SDF composition
            name: Environment name
            tensor_args: Device and dtype settings
            **kwargs: Additional arguments
        """
        # Load obstacle data
        self.eval_data = None
        if obstacle_trajectories is None:
            if eval_dataset_path is None:
                eval_dataset_path = os.path.join(DATASET_BASE_DIR, 'EnvCoblEmpty2D-RobotPointMass2DBig/eval_dataset.pkl')
                print("Using default eval_dataset_path : ", eval_dataset_path)
                # raise ValueError("Must provide either eval_dataset_path or obstacle_trajectories")

            import pickle
            with open(eval_dataset_path, 'rb') as f:
                eval_data = pickle.load(f)
            self.eval_data = eval_data

            if scenario_idx >= len(eval_data['obstacle_trajectories']):
                raise ValueError(f"Scenario {scenario_idx} not found. Dataset has {len(eval_data['obstacle_trajectories'])} scenarios")

            self.obstacle_trajectories = eval_data['obstacle_trajectories'][scenario_idx]
            self.ego_trajectory = eval_data['ego_trajectories'][scenario_idx]  # [80, 2]
            self.scale_factor = eval_data['scale_factor']
            print(f"{scenario_idx=}, {len(self.obstacle_trajectories)=}, {self.scale_factor=}")
            
        else:
            self.obstacle_trajectories = obstacle_trajectories
            self.ego_trajectory = None

        self.scenario_idx = scenario_idx
        self.obstacle_radius = obstacle_radius
        self.dt = dt
        self.horizon = horizon

        # Time range
        time_range = (0.0, (horizon - 1) * dt)

        # Workspace limits (similar to EnvSimple2D: [-1, -1] to [1, 1])
        # Data is scaled by 1/10, so original ~[-10, 20] x [-12, 10] becomes ~[-1, 2] x [-1.2, 1]
        # Use slightly expanded limits to accommodate all data
        # training [[-1.0, -1.2], [2.0, 1.0]]
        # evaluation 
        limits = torch.tensor([[-1.0, -1.5], [1.6, 1.2]], **tensor_args) # modified according to dataset statistics

        # Create moving obstacle fields
        obj_fixed_list = []  # No static obstacles
        obj_extra_list = []  # Moving obstacles go here

        for i, obs_traj in enumerate(self.obstacle_trajectories):
            # obs_traj: [T, 6] with [x, y, vx, vy, cos(θ), sin(θ)]
            positions = obs_traj[:, :2]  # [T, 2]
            prim_sphere= MultiSphereField(
                np.array(
                    # [positions[0]]
                    [[0.0, 0.0]]
                ),
                np.array(
                    [obstacle_radius]
                ),
                tensor_args=tensor_args,
            )
            traj_sphere = WaypointTrajectory(obs_traj, dt = self.dt, horizon = self.horizon)

            # Create moving sphere
            moving_sphere = MovingObjectField(
                primitive_fields=[prim_sphere],
                trajectory=traj_sphere,
                name=f"dyn-cobl2d-sphere-{self.scenario_idx}-{i}"
            )

            obj_extra_list.append(moving_sphere)

        # Call parent constructor
        super().__init__(
            name=name,
            limits=limits,
            obj_fixed_list=obj_fixed_list,
            obj_extra_list=obj_extra_list,
            precompute_sdf_obj_fixed=False, # no fixed obj
            precompute_sdf_obj_extra=False,
            time_range=time_range,
            k_smooth=k_smooth,
            tensor_args=tensor_args,
            **kwargs
        )

        self.n_obstacles = len(self.obstacle_trajectories)

    def get_obstacle_positions_at_time(self, t: float) -> torch.Tensor:
        """
        Get obstacle positions at a specific time.

        Args:
            t: Time value in [0, (horizon-1)*dt]

        Returns:
            Tensor of shape [n_obstacles, 2] with obstacle positions
        """
        # Compute time index
        t_idx = min(int(t / self.dt), self.horizon - 1)

        positions = []
        for obs_traj in self.obstacle_trajectories:
            pos = obs_traj[t_idx, :2]  # [2]
            positions.append(pos)

        return to_torch(np.array(positions), **self.tensor_args)
    
    def update_all_moving_object_trajectories(self, trajectory_configs):
        # change scenario id 
        self.scenario_idx = trajectory_configs.get('scenario_idx', 0)
        self.obstacle_trajectories = self.eval_data['obstacle_trajectories'][self.scenario_idx]
        self.ego_trajectory = self.eval_data['ego_trajectories'][self.scenario_idx]  # [80, 2]
        # tmp_res = []
        # for tj in self.eval_data['obstacle_trajectories'] : 
        #     tmp_res.append(any([np.isnan(t[0]).any() for t in tj]))
        # print(f"nan {all(tmp_res)}")
        # import pdb; pdb.set_trace()
        # update object field
        new_obj_extra = []  # Moving obstacles go here

        for i, obs_traj in enumerate(self.obstacle_trajectories):
            # obs_traj: [T, 6] with [x, y, vx, vy, cos(θ), sin(θ)]
            positions = obs_traj[:, :2]  # [T, 2]
            prim_sphere= MultiSphereField(
                np.array(
                    [[0.0, 0.0]]
                ),
                np.array(
                    [self.obstacle_radius]
                ),
                tensor_args=self.tensor_args,
            )
            traj_sphere = WaypointTrajectory(obs_traj, dt = self.dt, horizon = self.horizon)

            # Create moving sphere
            moving_sphere = MovingObjectField(
                primitive_fields=[prim_sphere],
                trajectory=traj_sphere,
                name=f"dyn-cobl2d-sphere-{self.scenario_idx}-{i}"
            )

            new_obj_extra.append(moving_sphere)
        self.set_obj_extra_list(new_obj_extra)
        # self.obj_extra_list = new_obj_extra
        self.n_obstacles = len(self.obstacle_trajectories)
        print(f"Updated Scenario {self.scenario_idx=}, {self.n_obstacles=}")

    def render(self, ax=None, time: Optional[float] = None, **kwargs):
        """
        Render environment at a specific time.

        Args:
            ax: Matplotlib axis
            time: Time value for rendering moving obstacles (if None, renders at t=0)
            **kwargs: Additional rendering arguments
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # if time is None:
        #     time = 0.0

        has_moving = self._has_moving_objects()

        if has_moving:
            # Use provided time, or default to middle of time range
            if time is None:
                time = (self.time_range[0] + self.time_range[1]) / 2.0
            # Clamp to valid range
            time = max(self.time_range[0], min(self.time_range[1], time))
            # Update all moving objects
            # print("EnvDynBase:",time)
            self._update_moving_objects_at_time(time)


        # Render workspace bounds
        x_min, y_min = to_numpy(self.limits[0])
        x_max, y_max = to_numpy(self.limits[1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        # # only show it in 
        # xticks = ax.get_xticks()
        # ax.set_xticklabels([f"{t*10:g}" for t in xticks])
        # yticks = ax.get_yticks()
        # ax.set_yticklabels([f"{t*10:g}" for t in yticks])
        # self._update_moving_objects_at_time(time)
        # Render obstacles at current time
        t_idx = min(int(time / self.dt), self.horizon - 1)

        for i, obs_traj in enumerate(self.obstacle_trajectories):
            pos = obs_traj[t_idx, :2]
            vel = obs_traj[t_idx, 2:4]

            # Draw obstacle circle
            circle = Circle(
                (pos[0], pos[1]),
                self.obstacle_radius,
                color='purple',
                fill=True,
                alpha=0.6,
                zorder=5
            )
            ax.add_patch(circle)

            # Draw velocity arrow
            if np.linalg.norm(vel) > 0.01:
                ax.arrow(
                    pos[0], pos[1],
                    vel[0], vel[1],
                    head_width=0.05,
                    head_length=0.05,
                    length_includes_head=True,
                    fc='purple',
                    ec='purple',
                    alpha=0.5,
                    zorder=6
                )

            # Optionally draw full trajectory
            if kwargs.get('show_trajectories', False):
                traj_positions = obs_traj[:, :2]
                ax.plot(
                    traj_positions[:, 0],
                    traj_positions[:, 1],
                    color='purple',
                    alpha=0.2,
                    linewidth=1.0,
                    zorder=3
                )

        # Add time label
        ax.text(
            0.02, 0.98,
            f"t = {time:.2f}s",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'COBL Environment - Scenario {self.scenario_idx} ({self.n_obstacles} obstacles)')

        return ax

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  scenario_idx={self.scenario_idx},\n"
            f"  n_obstacles={self.n_obstacles},\n"
            f"  obstacle_radius={self.obstacle_radius},\n"
            f"  time_range={self.time_range},\n"
            f"  workspace={to_numpy(self.limits).tolist()}\n"
            f")"
        )


def test_env_cobl_2d():
    """Test the COBL environment."""
    import os
    from pathlib import Path

    # Find eval dataset
    base_dir = os.path.join(DATASET_BASE_DIR, 'EnvCoblEmpty2D-RobotPointMass2D')
    eval_path = os.path.join(base_dir, 'eval_dataset.pkl')

    if not os.path.exists(eval_path):
        print(f"Error: Evaluation dataset not found at {eval_path}")
        print("Please run convert_cobl_to_mpd_hdf5.py first")
        return

    print("=" * 80)
    print("TESTING COBL ENVIRONMENT")
    print("=" * 80)

    # Create environment
    scenario_idx = 0
    env = EnvCobl2D(
        scenario_idx=scenario_idx,
        eval_dataset_path=str(eval_path),
        obstacle_radius=0.35,
    )

    print(f"\n{env}")

    # Test rendering at different times
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times = [0.0, 1.6, 3.2, 4.8, 6.4, 7.9]

    for ax, t in zip(axes.flat, times):
        env.render(ax=ax, time=t, show_trajectories=False)

    plt.tight_layout()
    plt.savefig('cobl_env_test.png', dpi=150)
    print(f"\n✅ Saved test visualization to /tmp/cobl_env_test.png")

    # Test obstacle position query
    print(f"\n  Testing obstacle position queries:")
    for t in [0.0, 4.0, 7.9]:
        positions = env.get_obstacle_positions_at_time(t)
        print(f"    t={t:.1f}s: {positions.shape[0]} obstacles")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_env_cobl_2d()
