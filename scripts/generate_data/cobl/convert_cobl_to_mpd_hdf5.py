"""
Convert COBL dataset (train80.pt, eval80_ego.pt, eval80_obs.pkl) to MPD HDF5 format.

This script creates:
1. dataset_merged_doubled.hdf5 - Training/validation dataset compatible with MPD
2. Dynamic environment class for evaluation with moving obstacles

Author: Claude
Date: 2025-01-21
"""

import os
import h5py
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from mpd.paths import DATASET_BASE_DIR
# COBL configuration
HORIZON = 80
DT = 0.1
SCALE_FACTOR = 10.0  # Fixed scale factor for normalization
OBSTACLE_RADIUS = 0.35  # Original obstacle radius in meters
CUTOFF_MARGIN = 0.15  # Cutoff margin to make effective obstacle radius 0.5m


def convert_cobl_to_hdf5(
    train_path: str,
    output_dir: str,
    dataset_name: str = "dataset_merged_doubled.hdf5",
    train_val_split: float = 0.9,
    downsample_factor: int = 1,
    scale_to_unit_workspace: bool = True,
    min_start_goal_distance: float = None,
):
    """
    Convert COBL train80.pt to MPD HDF5 format.

    Args:
        train_path: Path to train80.pt file
        output_dir: Output directory for HDF5 file
        dataset_name: Name of output HDF5 file
        train_val_split: Fraction for training (rest for validation)
        downsample_factor: Downsample trajectories (1 = no downsample, 2 = every other, etc.)
        scale_to_unit_workspace: Scale trajectories to [-1, 1] workspace like EnvSimple2D
        min_start_goal_distance: Minimum required start-goal distance (scaled units).
                                Trajectories with distance < this will be filtered out.
                                If None, no distance filtering is applied.
    """
    print("=" * 80)
    print("CONVERTING COBL DATASET TO MPD HDF5 FORMAT")
    print("=" * 80)

    # Load COBL data
    print(f"\nLoading COBL training data from: {train_path}")
    train_data = torch.load(train_path, map_location='cpu')
    print(f"  Shape: {train_data.shape}")
    print(f"  Original trajectories: {train_data.shape[0]:,}")

    # Downsample if requested
    if downsample_factor > 1:
        train_data = train_data[::downsample_factor]
        print(f"  After downsampling ({downsample_factor}x): {train_data.shape[0]:,}")

    n_total = train_data.shape[0]
    n_train = int(n_total * train_val_split)
    n_val = n_total - n_train

    print(f"\n  Split:")
    print(f"    Training:   {n_train:,} trajectories")
    print(f"    Validation: {n_val:,} trajectories")

    # Extract trajectory data
    # COBL format: [x, y, vx, vy, cos(θ), sin(θ), v, θ, ω]
    # We use [x, y] positions for sol_path
    positions = train_data[:, :2, :].numpy()  # [N, 2, 80]

    # Scale data by SCALE_FACTOR (divide by 10.0)
    if scale_to_unit_workspace:
        print(f"\n  Scaling trajectories by factor 1/{SCALE_FACTOR}")
        positions = positions / SCALE_FACTOR

        # Also scale velocities
        velocities = train_data[:, 2:4, :].numpy() / SCALE_FACTOR
        speeds = train_data[:, 6, :].numpy() / SCALE_FACTOR
    else:
        velocities = train_data[:, 2:4, :].numpy()
        speeds = train_data[:, 6, :].numpy()

    # Transpose to [N, 80, 2] for MPD format
    sol_path = positions.transpose(0, 2, 1)  # [N, 80, 2]

    # Filter trajectories by start-goal distance if requested
    if min_start_goal_distance is not None:
        print(f"\n  Filtering trajectories by start-goal distance...")
        print(f"    Min allowed distance: {min_start_goal_distance:.3f} ({min_start_goal_distance * SCALE_FACTOR:.1f}m original)")

        # Calculate start-goal distances
        start_positions = sol_path[:, 0, :]  # [N, 2]
        goal_positions = sol_path[:, -1, :]  # [N, 2]
        distances = np.linalg.norm(goal_positions - start_positions, axis=1)  # [N]

        # Create mask for valid trajectories (distance <= min_start_goal_distance)
        valid_mask = distances >= min_start_goal_distance
        n_valid = valid_mask.sum()
        n_filtered = len(valid_mask) - n_valid

        print(f"    Before filtering: {len(sol_path):,} trajectories")
        print(f"    Valid trajectories: {n_valid:,} ({100*n_valid/len(sol_path):.2f}%)")
        print(f"    Filtered out: {n_filtered:,} ({100*n_filtered/len(sol_path):.2f}%)")

        # Apply filter to all data arrays
        sol_path = sol_path[valid_mask]
        velocities = velocities[valid_mask]
        speeds = speeds[valid_mask]
        train_data = train_data[valid_mask]

        # Update counts
        n_total = len(sol_path)
        n_train = int(n_total * train_val_split)
        n_val = n_total - n_train

        print(f"\n    After filtering:")
        print(f"      Training:   {n_train:,} trajectories")
        print(f"      Validation: {n_val:,} trajectories")

    # Calculate trajectory limits for workspace bounds
    x_min, x_max = sol_path[:,:,0].min(), sol_path[:,:,0].max()
    y_min, y_max = sol_path[:,:,1].min(), sol_path[:,:,1].max()

    print(f"\n  Trajectory waypoints:")
    print(f"    Shape: {sol_path.shape}")
    print(f"    Format: [n_trajectories, n_waypoints=80, n_dims=2]")
    print(f"    X range: [{x_min:.3f}, {x_max:.3f}] (width: {x_max - x_min:.3f})")
    print(f"    Y range: [{y_min:.3f}, {y_max:.3f}] (height: {y_max - y_min:.3f})")

    # Create task IDs (each trajectory is a unique task for COBL)
    task_ids = np.arange(n_total, dtype=np.int64)

    # Create success flags (all COBL trajectories are successful by definition)
    success = np.ones(n_total, dtype=bool)

    # Create validity flags
    all_states_valid = np.ones(n_total, dtype=bool)
    sol_path_after_bspline_fit = np.ones(n_total, dtype=bool)

    # Save to HDF5
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, dataset_name)

    print(f"\n  Creating HDF5 file: {output_path}")
    with h5py.File(output_path, 'w') as f:
        # Main trajectory data
        f.create_dataset('sol_path', data=sol_path, compression='gzip', compression_opts=4)
        f.create_dataset('task_id', data=task_ids, compression='gzip')
        f.create_dataset('success', data=success, compression='gzip')
        # f.create_dataset('all_states_valid_after_bspline_fit', data=all_states_valid, compression='gzip')
        # f.create_dataset('sol_path_after_bspline_fit', data=sol_path_after_bspline_fit, compression='gzip')

        # Optional: Store additional COBL-specific data
        # Store velocities for reference (already scaled above)
        f.create_dataset('velocities', data=velocities.transpose(0, 2, 1), compression='gzip', compression_opts=4)

        # Store orientations (cos, sin) - no scaling needed
        orientations = train_data[:, 4:6, :].numpy().transpose(0, 2, 1)  # [N, 80, 2]
        f.create_dataset('orientations', data=orientations, compression='gzip', compression_opts=4)

        # Store speeds (already scaled above)
        f.create_dataset('speeds', data=speeds, compression='gzip', compression_opts=4)

        # Metadata
        f.attrs['source'] = 'COBL_diffusion'
        f.attrs['horizon'] = HORIZON
        f.attrs['dt'] = DT
        f.attrs['n_train'] = n_train
        f.attrs['n_val'] = n_val
        f.attrs['train_val_split'] = train_val_split
        f.attrs['scale_factor'] = SCALE_FACTOR
        f.attrs['obstacle_radius'] = OBSTACLE_RADIUS / SCALE_FACTOR  # Scaled radius
        f.attrs['cutoff_margin'] = CUTOFF_MARGIN / SCALE_FACTOR  # Scaled margin

        # Trajectory limits (for workspace bounds)
        f.attrs['x_min'] = float(x_min)
        f.attrs['x_max'] = float(x_max)
        f.attrs['y_min'] = float(y_min)
        f.attrs['y_max'] = float(y_max)

        # Distance filtering metadata
        if min_start_goal_distance is not None:
            f.attrs['min_start_goal_distance'] = float(min_start_goal_distance)
            f.attrs['min_start_goal_distance_original'] = float(min_start_goal_distance * SCALE_FACTOR)
            f.attrs['filtered'] = True
        else:
            f.attrs['filtered'] = False

        f.attrs['description'] = 'Converted from COBL train80.pt - 2D point mass trajectories with dynamic obstacles. Data scaled by 1/10.'

    print(f"\n✅ Successfully created HDF5 dataset!")
    print(f"   Location: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024**2):.1f} MB")

    return output_path


def create_eval_dataset(
    eval_ego_path: str,
    eval_obs_path: str,
    output_dir: str,
):
    """
    Create evaluation dataset with dynamic obstacles.

    Args:
        eval_ego_path: Path to eval80_ego.pt
        eval_obs_path: Path to eval80_obs.pkl
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("CREATING EVALUATION DATASET WITH DYNAMIC OBSTACLES")
    print("=" * 80)

    # Load evaluation data
    print(f"\nLoading evaluation data...")
    eval_ego = torch.load(eval_ego_path, map_location='cpu')
    with open(eval_obs_path, 'rb') as f:
        eval_obs = pickle.load(f)

    print(f"  Ego trajectories: {eval_ego.shape[0]}")
    print(f"  Obstacle scenarios: {len(eval_obs)}")

    # Scale positions by SCALE_FACTOR
    print(f"\n  Scaling evaluation data by factor 1/{SCALE_FACTOR}")
    eval_positions = eval_ego[:, :2, :].numpy() / SCALE_FACTOR  # [300, 2, 80]
    eval_positions = eval_positions.transpose(0, 2, 1)  # [300, 80, 2]

    eval_velocities = eval_ego[:, 2:4, :].numpy() / SCALE_FACTOR  # [300, 2, 80]
    eval_velocities = eval_velocities.transpose(0, 2, 1)  # [300, 80, 2]

    # Process obstacles
    print(f"\n  Processing obstacle trajectories...")
    obstacle_data_list = []

    for i, obs_tensor in enumerate(tqdm(eval_obs, desc="Processing obstacles")):
        # obs_tensor: [1, max_n_obs, 6, 80]
        # Extract valid obstacles (non-NaN)
        valid_obstacles = []

        for j in range(obs_tensor.shape[1]):
            if not torch.isnan(obs_tensor[0, j, 0, 0]):
                # Extract obstacle trajectory: [6, 80] -> [80, 6]
                obs_traj = obs_tensor[0, j, :, :].numpy().T  # [80, 6]
                # Format: [x, y, vx, vy, cos(θ), sin(θ)]
                # Scale positions and velocities
                obs_traj[:, :2] /= SCALE_FACTOR  # Scale positions
                obs_traj[:, 2:4] /= SCALE_FACTOR  # Scale velocities
                valid_obstacles.append(obs_traj)

        obstacle_data_list.append(valid_obstacles)

    # Calculate evaluation data limits
    eval_x_min, eval_x_max = eval_positions[:,:,0].min(), eval_positions[:,:,0].max()
    eval_y_min, eval_y_max = eval_positions[:,:,1].min(), eval_positions[:,:,1].max()

    print(f"\n  Evaluation data range:")
    print(f"    X: [{eval_x_min:.3f}, {eval_x_max:.3f}] (width: {eval_x_max - eval_x_min:.3f})")
    print(f"    Y: [{eval_y_min:.3f}, {eval_y_max:.3f}] (height: {eval_y_max - eval_y_min:.3f})")

    # Generate problems per scenario
    print(f"\n  Generating 1 problem per scenario (trajectory endpoints)...")
    problems = []

    for scenario_idx in range(eval_ego.shape[0]):
        # Problem 0: Original trajectory endpoints
        q_start = eval_positions[scenario_idx, 0, :]  # First waypoint
        q_goal = eval_positions[scenario_idx, -1, :]  # Last waypoint
        problems.append({'q_start': q_start, 'q_goal': q_goal})

    # Save to pickle for easy loading
    eval_output_path = os.path.join(output_dir, 'eval_dataset.pkl')
    eval_data = {
        'ego_trajectories': eval_positions,  # [300, 80, 2] - scaled
        'obstacle_trajectories': obstacle_data_list,  # List of 300 scenarios - scaled
        'ego_velocities': eval_velocities,  # [300, 80, 2] - scaled
        #'ego_orientations': eval_ego[:, 4:6, :].numpy().transpose(0, 2, 1),  # [300, 80, 2]
        'problems': problems,  # List of 300 problems (one per scenario)
        'horizon': HORIZON,
        'dt': DT,
        'n_scenarios': eval_ego.shape[0],
        'n_problems_per_scenario': 1,
        'scale_factor': SCALE_FACTOR,
        'obstacle_radius': OBSTACLE_RADIUS / SCALE_FACTOR,
        'cutoff_margin': CUTOFF_MARGIN / SCALE_FACTOR,
        # Data limits for workspace bounds
        # 'x_min': float(eval_x_min),
        # 'x_max': float(eval_x_max),
        # 'y_min': float(eval_y_min),
        # 'y_max': float(eval_y_max),
    }

    with open(eval_output_path, 'wb') as f:
        pickle.dump(eval_data, f)

    print(f"\n✅ Successfully created evaluation dataset!")
    print(f"   Location: {eval_output_path}")
    print(f"   Size: {os.path.getsize(eval_output_path) / (1024**2):.1f} MB")

    # Print statistics
    n_obstacles_per_scenario = [len(obs_list) for obs_list in obstacle_data_list]
    print(f"\n  Obstacle statistics:")
    print(f"    Min obstacles: {min(n_obstacles_per_scenario)}")
    print(f"    Max obstacles: {max(n_obstacles_per_scenario)}")
    print(f"    Mean obstacles: {np.mean(n_obstacles_per_scenario):.1f}")
    print(f"    Median obstacles: {np.median(n_obstacles_per_scenario):.0f}")

    return eval_output_path


def main():
    """Main conversion script."""
    base_dir = os.path.join(DATASET_BASE_DIR, 'EnvCoblEmpty2D-RobotPointMass2DBig')
    eval_path = os.path.join(base_dir, 'eval_dataset.pkl')
    # Paths
    train_path = os.path.join(base_dir, 'train80.pt')
    eval_ego_path = os.path.join(base_dir,'eval80_ego.pt')
    eval_obs_path = os.path.join(base_dir,'eval80_obs.pkl')

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(eval_ego_path):
        raise FileNotFoundError(f"Evaluation ego data not found: {eval_ego_path}")
    if not os.path.exists(eval_obs_path):
        raise FileNotFoundError(f"Evaluation obstacle data not found: {eval_obs_path}")

    print(f"Input files:")
    print(f"  Train: {train_path}")
    print(f"  Eval ego: {eval_ego_path}")
    print(f"  Eval obs: {eval_obs_path}")

    # Convert training data to HDF5
    output_dir = str(base_dir)
    hdf5_path = convert_cobl_to_hdf5(
        train_path=str(train_path),
        output_dir=output_dir,
        train_val_split=0.9,
        downsample_factor=1,  # Set to >1 to reduce dataset size
        min_start_goal_distance=0.4,  # 4m original, 0.4 scaled - filter out trajectories with distance < 4m (keep >= 4m)
    )

    # Create evaluation dataset
    eval_path = create_eval_dataset(
        eval_ego_path=str(eval_ego_path),
        eval_obs_path=str(eval_obs_path),
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"\nCreated files:")
    print(f"  1. Training/validation HDF5: {hdf5_path}")
    print(f"  2. Evaluation dataset: {eval_path}")
    print(f"\nNext steps:")
    print(f"  1. Create dynamic environment class for COBL")
    print(f"  2. Test with MPD training pipeline")
    print(f"  3. Evaluate on dynamic obstacle scenarios")
    print("=" * 80)


if __name__ == '__main__':
    main()
