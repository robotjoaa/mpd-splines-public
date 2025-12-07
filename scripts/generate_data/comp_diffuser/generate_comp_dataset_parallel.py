from multiprocessing import Pool
import argparse
import math
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def sample_segment_worker(args):
    """Executed in worker process."""
    (
        out_idx,      # output index
        base_idx,     # index into original dataset
        input_path,
        min_frac,
        max_frac,
        max_seg_len,
        seed,
    ) = args

    rng = np.random.default_rng(seed + out_idx)

    # Open file inside worker (safe)
    with h5py.File(input_path, "r") as f:
        path = np.asarray(f["sol_path"][base_idx])

    horizon = path.shape[0]
    min_len = max(1, int(math.ceil(horizon * min_frac)))
    max_len_val = max(min_len, int(math.floor(horizon * max_frac)))
    seg_len = rng.integers(min_len, max_len_val + 1)

    max_start = horizon - seg_len
    start_idx = int(rng.integers(0, max_start + 1))

    segment = path[start_idx:start_idx + seg_len]

    padded = np.full((max_seg_len, *path.shape[1:]),
                     np.nan, dtype=path.dtype)
    padded[:seg_len] = segment

    return (
        out_idx,
        padded,
        start_idx / horizon,
        path[0],      # start state
        path[-1],     # goal state
        base_idx,
    )

def derive_output_path(input_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    base, ext = os.path.splitext(input_path)
    return f"{base}_comp_segments{ext}"


def main():
    parser = argparse.ArgumentParser(description="Generate composed trajectory dataset with random segments.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source dataset_merged_doubled.hdf5 file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the augmented dataset. Defaults to <input>_comp_segments.hdf5.",
    )
    parser.add_argument("--min-frac", type=float, default=0.3, help="Minimum segment fraction (0,1].")
    parser.add_argument("--max-frac", type=float, default=0.6, help="Maximum segment fraction (0,1].")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Dataset size multiplier from original dataset size (default 3).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible sampling.")
    args = parser.parse_args()

    assert 0 < args.min_frac <= 1.0, "min-frac must be in (0, 1]."
    assert 0 < args.max_frac <= 1.0, "max-frac must be in (0, 1]."
    assert args.min_frac <= args.max_frac, "min-frac must be <= max-frac."

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
        # 🧠 Allocate output arrays ONCE in RAM
        # --------------------------------------------
        seg_array = np.full(
            (num_trajs, max_seg_len, *base_shape),
            np.nan,
            dtype=scalar_dtype
        )
        progress_array = np.zeros((num_trajs,), dtype=np.float32)
        start_array = np.zeros((num_trajs, *base_shape), dtype=scalar_dtype)
        goal_array = np.zeros((num_trajs, *base_shape), dtype=scalar_dtype)
        task_id_array = np.zeros((num_trajs,), dtype=src["task_id"].dtype)

        # --------------------------------------------
        # Build job list
        # --------------------------------------------
        jobs = []
        for out_idx in range(num_trajs):
            base_idx = out_idx % num_trajs_original
            jobs.append((
                out_idx,
                base_idx,
                input_path,
                args.min_frac,
                args.max_frac,
                max_seg_len,
                args.seed,
            ))

        # --------------------------------------------
        # Run worker processes
        # --------------------------------------------
        with Pool() as pool:
            for (
                out_idx,
                padded,
                prog,
                start_st,
                goal_st,
                base_idx,
            ) in tqdm(pool.imap(sample_segment_worker, jobs),
                      total=num_trajs,
                      desc="Sampling (parallel)"):

                # Store results into preallocated arrays
                seg_array[out_idx] = padded
                progress_array[out_idx] = prog
                start_array[out_idx] = start_st
                goal_array[out_idx] = goal_st
                task_id_array[out_idx] = src["task_id"][base_idx]

        # --------------------------------------------
        # 💾 ONE-SHOT HDF5 WRITE
        # --------------------------------------------
        with h5py.File(output_path, "w") as dst:
            dst.create_dataset(
                "sol_path",
                data=seg_array,
                dtype=scalar_dtype,
                compression="lzf"
            )
            dst.create_dataset(
                "progress",
                data=progress_array,
                dtype=np.float32,
                compression="lzf"
            )
            dst.create_dataset(
                "start_state",
                data=start_array,
                dtype=scalar_dtype,
                compression="lzf"
            )
            dst.create_dataset(
                "goal_state",
                data=goal_array,
                dtype=scalar_dtype,
                compression="lzf"
            )
            dst.create_dataset(
                "task_id",
                data=task_id_array,
                dtype=task_id_array.dtype,
                compression="lzf"
            )

    # print(f"Saved composed dataset with {num_trajs} segments to {output_path}")


    print(f"Saved composed dataset with {num_trajs} segments to {output_path}")

if __name__ == "__main__":
    main()