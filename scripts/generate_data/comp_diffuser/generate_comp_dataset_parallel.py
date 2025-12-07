from multiprocessing import Pool
import argparse
import math
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def sample_segment_worker(args):
    (
        out_idx, base_idx, input_path,
        min_frac, max_frac, max_seg_len, seq_len,
        seed
    ) = args

    rng = np.random.default_rng(seed + out_idx)

    # Load full path inside worker (safe)
    with h5py.File(input_path, "r") as f:
        path = np.asarray(f["sol_path"][base_idx])
        task_id = f["task_id"][base_idx]

    horizon = path.shape[0]

    # Segment sampling
    min_len = max(1, int(math.ceil(horizon * min_frac)))
    max_len_val = max(min_len, int(math.floor(horizon * max_frac)))
    seg_len = int(rng.integers(min_len, max_len_val + 1))

    max_start = horizon - seg_len
    start_idx = int(rng.integers(0, max_start + 1))

    segment = path[start_idx:start_idx + seg_len]

    # padded segment
    padded = np.full((max_seg_len, *path.shape[1:]), np.nan, dtype=path.dtype)
    padded[:seg_len] = segment

    # delta-to-goal
    delta = path[-1] - segment[-1]

    return (
        out_idx,
        padded,
        path,                      # full path
        start_idx,
        seg_len,
        start_idx / horizon,       # progress
        delta,
        path[0],                   # start_state
        path[-1],                  # goal_state
        horizon,                   # full_traj_len
        task_id,
        base_idx
    )

def derive_output_path(input_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    base, ext = os.path.splitext(input_path)
    return f"{base}_comp_segments{ext}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-frac", type=float, default=0.3)
    parser.add_argument("--max-frac", type=float, default=0.6)
    parser.add_argument("--multiplier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input
    output_path = derive_output_path(input_path, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(input_path, "r") as src:
        sol_paths = src["sol_path"]
        num_trajs_original = len(sol_paths)
        num_trajs = num_trajs_original * args.multiplier

        seq_len = sol_paths.shape[1]
        base_shape = sol_paths.shape[2:]
        scalar_dtype = sol_paths.dtype.base
        max_seg_len = int(math.ceil(seq_len * args.max_frac))

        # --------------------------------------------
        # Preallocate arrays
        # --------------------------------------------
        seg_array = np.full((num_trajs, max_seg_len, *base_shape), np.nan, dtype=scalar_dtype)
        full_path_array = np.zeros((num_trajs, seq_len, *base_shape), dtype=scalar_dtype)

        progress_array = np.zeros(num_trajs, dtype=np.float32)
        delta_array = np.zeros((num_trajs, *base_shape), dtype=scalar_dtype)
        start_idx_array = np.zeros(num_trajs, dtype=np.int32)
        seg_len_array = np.zeros(num_trajs, dtype=np.int32)
        start_array = np.zeros((num_trajs, *base_shape), dtype=scalar_dtype)
        goal_array = np.zeros((num_trajs, *base_shape), dtype=scalar_dtype)
        full_len_array = np.zeros(num_trajs, dtype=np.int32)
        task_id_array = np.zeros(num_trajs, dtype=src["task_id"].dtype)

        # --------------------------------------------
        # Build job list
        # --------------------------------------------
        jobs = []
        for out_idx in range(num_trajs):
            base_idx = out_idx % num_trajs_original
            jobs.append((
                out_idx, base_idx, input_path,
                args.min_frac, args.max_frac,
                max_seg_len, seq_len,
                args.seed
            ))

        # --------------------------------------------
        # Run multiprocessing worker sampling
        # --------------------------------------------
        with Pool() as pool:
            for (
                out_idx,
                padded,
                full_path,
                start_idx,
                seg_len,
                progress,
                delta,
                start_state,
                goal_state,
                full_len,
                task_id,
                base_idx
            ) in tqdm(pool.imap(sample_segment_worker, jobs),
                      total=num_trajs, desc="Sampling segments (parallel)"):

                seg_array[out_idx] = padded
                full_path_array[out_idx] = full_path

                progress_array[out_idx] = progress
                delta_array[out_idx] = delta
                start_idx_array[out_idx] = start_idx
                seg_len_array[out_idx] = seg_len
                start_array[out_idx] = start_state
                goal_array[out_idx] = goal_state
                full_len_array[out_idx] = full_len
                task_id_array[out_idx] = task_id

        # --------------------------------------------
        # Now handle replicated passthrough datasets
        # --------------------------------------------
        replicated_data = {}
        for key, ds in src.items():
            if key in {
                "sol_path", "task_id",
                "progress", "delta_to_goal",
                "segment_start_idx", "segment_len",
                "start_state", "goal_state",
                "full_traj_len", "sol_path_full"
            }:
                continue

            if ds.shape and ds.shape[0] == num_trajs_original:
                replicated_data[key] = np.asarray(ds)[
                    np.arange(num_trajs) % num_trajs_original
                ]
            else:
                replicated_data[key] = ds[:]

    # --------------------------------------------
    # Write all datasets ONCE
    # --------------------------------------------
    with h5py.File(output_path, "w") as dst:
        dst.create_dataset("sol_path", data=seg_array, compression="lzf")
        dst.create_dataset("sol_path_full", data=full_path_array, compression="lzf")
        dst.create_dataset("progress", data=progress_array, compression="lzf")
        dst.create_dataset("delta_to_goal", data=delta_array, compression="lzf")
        dst.create_dataset("segment_start_idx", data=start_idx_array, compression="lzf")
        dst.create_dataset("segment_len", data=seg_len_array, compression="lzf")
        dst.create_dataset("start_state", data=start_array, compression="lzf")
        dst.create_dataset("goal_state", data=goal_array, compression="lzf")
        dst.create_dataset("full_traj_len", data=full_len_array, compression="lzf")
        dst.create_dataset("task_id", data=task_id_array, compression="lzf")

        # write passthrough datasets
        for key, arr in replicated_data.items():
            dst.create_dataset(key, data=arr, compression="lzf")

    print(f"Saved composed dataset with {num_trajs} segments to {output_path}")

if __name__ == "__main__":
    main()