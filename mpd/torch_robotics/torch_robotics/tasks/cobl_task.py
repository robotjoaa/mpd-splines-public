"""
Planning task configuration for COBL dataset with dynamic obstacles.

Author: Claude
Date: 2025-01-21
"""

import torch
from torch_robotics.robots import RobotPointMass2D
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from mpd.torch_robotics.torch_robotics.tasks.dynamic_tasks import DynPlanningTask
from mpd.torch_robotics.torch_robotics.environments.dynamic_extension.env_cobl_2d import EnvCobl2D


def create_cobl_planning_task(
    scenario_idx: int = 0,
    eval_dataset_path: str = None,
    obstacle_radius: float = 0.035,  # Scaled: 0.35m / 10
    parametric_trajectory=None,
    obstacle_cutoff_margin: float = 0.015,  # Scaled: 0.15m / 10
    moving_object_gradient_scale: float = 1.0,
    moving_object_margin_scale: float = 1.0,
    tensor_args: dict = DEFAULT_TENSOR_ARGS,
    **kwargs
):
    """
    Create a planning task for COBL evaluation scenario.

    Args:
        scenario_idx: Which scenario to load from evaluation dataset
        eval_dataset_path: Path to eval_dataset.pkl
        obstacle_radius: Radius of dynamic obstacles
        parametric_trajectory: Trajectory representation (B-spline, etc.)
        obstacle_cutoff_margin: Margin for obstacle collision detection
        moving_object_gradient_scale: Scale factor for dynamic obstacle gradients
        moving_object_margin_scale: Scale factor for dynamic obstacle margins
        tensor_args: Device and dtype settings
        **kwargs: Additional arguments

    Returns:
        DynPlanningTask configured for COBL scenario
    """

    # Create robot (2D point mass)
    robot = RobotPointMass2D(
        use_self_collision_storm=False,
        tensor_args=tensor_args
    )

    # Create dynamic environment
    env = EnvCobl2D(
        scenario_idx=scenario_idx,
        eval_dataset_path=eval_dataset_path,
        obstacle_radius=obstacle_radius,
        tensor_args=tensor_args,
    )

    # Create planning task
    task = DynPlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=parametric_trajectory,
        ws_limits=env.limits,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        use_field_collision_self=True,
        use_field_collision_objects=False,
        moving_object_gradient_scale=moving_object_gradient_scale,
        moving_object_margin_scale=moving_object_margin_scale,
        tensor_args=tensor_args,
        **kwargs
    )

    return task, env


def get_cobl_start_goal_from_scenario(
    scenario_idx: int,
    eval_dataset_path: str,
    tensor_args: dict = DEFAULT_TENSOR_ARGS,
):
    """
    Get start and goal positions from COBL evaluation scenario.

    Args:
        scenario_idx: Scenario index
        eval_dataset_path: Path to eval_dataset.pkl
        tensor_args: Device and dtype settings

    Returns:
        q_start: Start position [2]
        q_goal: Goal position [2]
    """
    import pickle

    with open(eval_dataset_path, 'rb') as f:
        eval_data = pickle.load(f)

    ego_traj = eval_data['ego_trajectories'][scenario_idx]  # [80, 2]

    q_start = torch.tensor(ego_traj[0], **tensor_args)  # First position
    q_goal = torch.tensor(ego_traj[-1], **tensor_args)  # Last position

    return q_start, q_goal


if __name__ == '__main__':
    """Test COBL planning task creation."""
    from pathlib import Path

    print("=" * 80)
    print("TESTING COBL PLANNING TASK")
    print("=" * 80)

    # Find eval dataset
    base_dir = Path(__file__).parent.parent.parent.parent / 'data_trajectories' / 'EnvCoblEmpty2D-RobotPointMass2D'
    eval_path = base_dir / 'eval_dataset.pkl'

    if not eval_path.exists():
        print(f"\n❌ Error: Evaluation dataset not found at {eval_path}")
        print("Please run convert_cobl_to_mpd_hdf5.py first")
        exit(1)

    # Get start/goal
    q_start, q_goal = get_cobl_start_goal_from_scenario(
        scenario_idx=0,
        eval_dataset_path=str(eval_path),
    )

    print(f"\nScenario 0:")
    print(f"  Start: {q_start.numpy()}")
    print(f"  Goal:  {q_goal.numpy()}")
    print(f"  Distance: {torch.norm(q_goal - q_start).item():.2f}m")

    # Create planning task (without parametric trajectory for now)
    print(f"\nCreating planning task...")
    task, env = create_cobl_planning_task(
        scenario_idx=0,
        eval_dataset_path=str(eval_path),
    )

    print(f"\n{task}")
    print(f"\n{env}")

    # Test collision checking at different times
    print(f"\nTesting collision checking...")

    # Create a simple straight-line trajectory
    n_waypoints = 80
    t = torch.linspace(0, 1, n_waypoints).unsqueeze(1)  # [80, 1]
    q_traj = q_start * (1 - t) + q_goal * t  # [80, 2]
    q_traj = q_traj.unsqueeze(0)  # [1, 80, 2]

    # Get timesteps
    timesteps = task._get_timesteps_for_horizon(n_waypoints)

    # Check collision with dynamic obstacles
    collision = task.compute_collision(q_traj, margin=0.0, timesteps=timesteps)

    print(f"  Trajectory shape: {q_traj.shape}")
    print(f"  Collision result: {collision}")
    print(f"  Has collision: {collision.any()}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
