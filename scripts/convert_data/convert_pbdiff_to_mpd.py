"""
Convert pb_diff_envs dataset to MPD HDF5 format.

This script converts datasets from pb_diff_envs SequenceDataset/GoalDataset format
to MPD's TrajectoryDatasetWaypoints/Bspline format.

Usage:
    python scripts/convert_data/convert_pbdiff_to_mpd.py \
        --env_name maze2d-randSmaze2d-v0 \
        --output_dir data_trajectories/maze2d_pbdiff_converted \
        --config config_pbdiff.yaml

Author: Claude Code
Date: 2025-10-28
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

# Add pb_diff_envs to path
# pb_diff_path = Path(__file__).parent.parent.parent / "potential-motion-plan-release"
# if pb_diff_path.exists():
#     sys.path.insert(0, str(pb_diff_path))

from torch_robotics.environments.pb_diff_envs import EnvRandMaze2D, Maze2DRandRecGroupList

try:
    from mpd.datasets.pb_diff_datasets.data_api import load_environment, sequence_dataset
    from mpd.datasets.pb_diff_datasets.preprocessing import get_preprocess_fn, maze2d_set_terminals_luo

except ImportError:
    print("ERROR: Cannot import pb_diff_envs modules.")
    print("Please ensure potential-motion-plan-release is in the correct location.")
    sys.exit(1)


def convert_pbdiff_dataset_to_mpd_format(
    env_name,
    output_hdf5_path,
    preprocess_fns=[],
    dataset_config={},
    save_args_yaml=True,
):
    """
    Convert pb_diff_envs dataset to MPD HDF5 format.

    Args:
        env_name: pb_diff_envs environment name (e.g., "maze2d-randSmaze2d-v0")
        output_hdf5_path: Path to save converted HDF5 file
        preprocess_fns: Preprocessing functions for pb_diff_envs
        dataset_config: Configuration dict for SequenceDataset
        save_args_yaml: If True, save args.yaml for MPD loader

    Output HDF5 structure:
        sol_path: (n_trajectories, variable_length, q_dim) - Full trajectories
        task_id: (n_trajectories,) - Task identifiers
        env_idx: (n_trajectories,) - Environment index for each trajectory

    Attributes:
        num_trajectories: Total number of trajectories
        num_environments: Number of unique environments
        source_dataset: Original pb_diff_envs dataset name
    """
    print("="*80)
    print("Converting pb_diff_envs dataset to MPD format")
    print("="*80)
    print(f"Environment: {env_name}")
    print(f"Output: {output_hdf5_path}")
    print(f"Dataset config: {dataset_config}")
    print()

    # Load environment and preprocessing
    print("[1/5] Loading pb_diff_envs environment...")
    # gym registered env name 
    # env = load_environment(env_name)
    env = Maze2DRandRecGroupList
    env.name = env_name 
    env.maze_change_as_terminal = dataset_config.get('maze_change_as_terminal', True)
    env.cut_episode_len = dataset_config.get('cut_episode_len', False)

    print(f"  Environment: {env.name}")
    print(f"  maze_change_as_terminal: {env.maze_change_as_terminal}")
    print(f"  cut_episode_len: {env.cut_episode_len}")


    # preprocess_fns
    preprocess_fns = [maze2d_set_terminals_luo]

    preprocess_fn = get_preprocess_fn(preprocess_fns, env)

    # Get episode iterator
    print("\n[2/5] Iterating through episodes...")
    episode_iter = sequence_dataset(env, preprocess_fn)

    # Collect all trajectories
    trajectories = []
    task_ids = []
    env_indices = []

    task_id = 0
    for episode in tqdm(episode_iter, desc="Processing episodes"):
        # Episode data
        observations = episode['observations']  # (T, obs_dim)

        # Extract environment index (constant within episode)
        if 'maze_idx' in episode:
            env_idx = int(episode['maze_idx'][0, 0])  # First timestep's maze_idx
        else:
            env_idx = 0

        # Skip empty episodes
        if len(observations) == 0:
            print(f"  Warning: Skipping empty episode (task_id={task_id})")
            continue

        # Store trajectory
        trajectories.append(observations)
        task_ids.append(task_id)
        env_indices.append(env_idx)

        task_id += 1

    print(f"\n  Collected {len(trajectories)} trajectories")
    print(f"  Environment indices: {min(env_indices)} to {max(env_indices)}")
    print(f"  Unique environments: {len(set(env_indices))}")

    # Compute statistics
    print("\n[3/5] Computing trajectory statistics...")
    traj_lengths = [len(traj) for traj in trajectories]
    print(f"  Min trajectory length: {min(traj_lengths)}")
    print(f"  Max trajectory length: {max(traj_lengths)}")
    print(f"  Mean trajectory length: {np.mean(traj_lengths):.1f}")
    print(f"  State dimension: {trajectories[0].shape[-1]}")

    # Save to HDF5 in MPD format
    print("\n[4/5] Saving to HDF5...")
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)

    with h5py.File(output_hdf5_path, 'w') as f:
        # Create ragged array dataset for trajectories (variable length)
        # Use variable-length dtype for efficient storage
        dt = h5py.vlen_dtype(np.dtype('float32'))
        sol_path_dataset = f.create_dataset(
            'sol_path',
            (len(trajectories),),
            dtype=dt
        )

        for i in tqdm(range(len(trajectories)), desc="Writing trajectories"):
            sol_path_dataset[i] = trajectories[i].astype(np.float32).flatten()

        # Save task IDs
        f.create_dataset('task_id', data=np.array(task_ids, dtype=np.int32))

        # Save environment indices (IMPORTANT for multi-env)
        f.create_dataset('env_idx', data=np.array(env_indices, dtype=np.int32))

        # Save metadata
        f.attrs['num_trajectories'] = len(trajectories)
        f.attrs['num_environments'] = len(set(env_indices))
        f.attrs['source_dataset'] = env_name
        f.attrs['state_dim'] = trajectories[0].shape[-1]
        f.attrs['min_trajectory_length'] = min(traj_lengths)
        f.attrs['max_trajectory_length'] = max(traj_lengths)
        f.attrs['mean_trajectory_length'] = float(np.mean(traj_lengths))

    print(f"  Saved to {output_hdf5_path}")
    print(f"  File size: {os.path.getsize(output_hdf5_path) / 1024 / 1024:.2f} MB")

    # Save args.yaml for MPD loader
    if save_args_yaml:
        print("\n[5/5] Saving args.yaml for MPD loader...")
        args_yaml_path = os.path.join(os.path.dirname(output_hdf5_path), "args.yaml")

        # Determine robot type based on environment
        if 'maze2d' in env_name.lower() or '2d' in env_name.lower():
            robot_id = 'RobotPointMass2D'
            env_id = 'EnvMaze2DBase'
        elif '3d' in env_name.lower():
            robot_id = 'RobotPointMass'
            env_id = 'EnvBase'
        else:
            robot_id = 'RobotPointMass2D'  # Default
            env_id = 'EnvMaze2DBase'

        args_dict = {
            'env_id': env_id,
            'robot_id': robot_id,
            'min_distance_robot_env': 0.01,
            'planner': 'pb_diff_envs_converted',
            'source_dataset': env_name,
            'num_trajectories': len(trajectories),
            'num_environments': len(set(env_indices)),
            'conversion_date': str(np.datetime64('today')),
        }

        with open(args_yaml_path, 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False)

        print(f"  Saved args.yaml to {args_yaml_path}")

    # Print summary
    print("\n" + "="*80)
    print("Conversion completed successfully!")
    print("="*80)
    print(f"Output file: {output_hdf5_path}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Environments: {len(set(env_indices))}")
    print(f"Tasks: {len(set(task_ids))}")
    print(f"State dimension: {trajectories[0].shape[-1]}")
    print()
    print("Usage with MPD:")
    print(f"  dataset_subdir = \"{os.path.basename(os.path.dirname(output_hdf5_path))}\"")
    print(f"  dataset_file_merged = \"{os.path.basename(output_hdf5_path)}\"")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert pb_diff_envs dataset to MPD HDF5 format"
    )
    parser.add_argument(
        '--env_name',
        type=str,
        required=True,
        help='pb_diff_envs environment name (e.g., maze2d-randSmaze2d-v0)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for converted dataset'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default='dataset_merged.hdf5',
        help='Output HDF5 filename (default: dataset_merged.hdf5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML config file for dataset conversion settings'
    )
    parser.add_argument(
        '--maze_change_as_terminal',
        action='store_true',
        default=True,
        help='Use maze_change_bool as episode termination (default: True)'
    )
    parser.add_argument(
        '--no_maze_change_as_terminal',
        action='store_false',
        dest='maze_change_as_terminal',
        help='Do not use maze_change_bool as episode termination'
    )
    parser.add_argument(
        '--cut_episode_len',
        type=int,
        default=None,
        help='Further cut episodes to specified length (default: None)'
    )

    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..', 'datasets')

    eval_ng_300 = 300
    eval_np_200 = 200
    dyn_eval_ng_100 = 100
    dyn_eval_np_2000 = 2000
    multi_w_train_ng = 300
    multi_w_train_spe = 25000

    hr = 0.50
    gym_dict = dict(
        env_id='randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp:Maze2DRandRecGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 3000,
            'samples_per_env': 25000,
            'num_walls': 6,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    # New
                                    min_rep_dist=1e-3,
                                    ball_rad_n=1.6,
                                    ),
                                    # min_episode_distance # default
                                    # robot_collision_eps # default
                                    # epi_dist_type
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 100,
            'dataset_url':f'{root_dir}/randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0.hdf5',
        }
    )


    # Load config if provided
    dataset_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            dataset_config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    else:
        dataset_config = {
            'maze_change_as_terminal': args.maze_change_as_terminal,
            'is_mazelist': True,
        }
        if args.cut_episode_len:
            dataset_config['cut_episode_len'] = args.cut_episode_len

    # Create output path
    output_hdf5_path = os.path.join(args.output_dir, args.output_filename)

    # Convert dataset
    convert_pbdiff_dataset_to_mpd_format(
        env_name=args.env_name,
        output_hdf5_path=output_hdf5_path,
        preprocess_fns=[],
        dataset_config=dataset_config,
        save_args_yaml=True,
    )


if __name__ == "__main__":
    main()
