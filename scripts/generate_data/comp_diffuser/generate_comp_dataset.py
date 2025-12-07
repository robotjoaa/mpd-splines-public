import argparse
import math
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def sample_segment(
    path: np.ndarray, rng: np.random.Generator, min_frac: float, max_frac: float
) -> Tuple[np.ndarray, int]:
    """Sample a contiguous segment from a trajectory.

    Args:
        path: Trajectory of shape [H, D].
        rng: Random generator.
        min_frac: Minimum fraction of the trajectory to keep (0, 1].
        max_frac: Maximum fraction of the trajectory to keep (0, 1].

    Returns:
        segment: Sub-sequence of the trajectory.
        start_idx: Starting index of the segment in the original trajectory.
    """
    horizon = path.shape[0]
    assert horizon > 0, "Empty trajectory provided."

    min_len = max(1, int(math.ceil(horizon * min_frac)))
    max_len = max(min_len, int(math.floor(horizon * max_frac)))
    seg_len = rng.integers(min_len, max_len + 1)

    max_start = horizon - seg_len
    start_idx = int(rng.integers(0, max_start + 1))
    end_idx = start_idx + seg_len

    return path[start_idx:end_idx], start_idx


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

    rng = np.random.default_rng(args.seed)

    with h5py.File(input_path, "r") as src_h5:
        if "sol_path" not in src_h5:
            raise KeyError(f"'sol_path' dataset not found in {input_path}.")

        sol_paths = src_h5["sol_path"]
        num_trajs_original = len(sol_paths)
        num_trajs = len(sol_paths) * args.multiplier
        seq_len = sol_paths.shape[1]
        base_shape = tuple(sol_paths.shape[2:])  # shape of one waypoint
        scalar_dtype = sol_paths.dtype.base
        max_seg_len = int(math.ceil(seq_len * args.max_frac))

        with h5py.File(output_path, "w") as dst_h5:
            # Segment datasets
            seg_ds = dst_h5.create_dataset(
                "sol_path", shape=(num_trajs, max_seg_len, *base_shape), dtype=scalar_dtype, compression="gzip"
            )

            # Metadata / additional fields
            progress_ds = dst_h5.create_dataset("progress", shape=(num_trajs,), dtype=np.float32, compression="gzip")
            # delta_ds = dst_h5.create_dataset(
            #     "delta_to_goal", shape=(num_trajs, *base_shape), dtype=scalar_dtype, compression="gzip"
            # )
            # start_idx_ds = dst_h5.create_dataset(
            #     "segment_start_idx", shape=(num_trajs,), dtype=np.int32, compression="gzip"
            # )
            seg_len_ds = dst_h5.create_dataset("segment_len", shape=(num_trajs,), dtype=np.int32, compression="gzip")
            start_ds = dst_h5.create_dataset(
                "start_state", shape=(num_trajs, *base_shape), dtype=scalar_dtype, compression="gzip"
            )
            goal_ds = dst_h5.create_dataset(
                "goal_state", shape=(num_trajs, *base_shape), dtype=scalar_dtype, compression="gzip"
            )
            # full_len_ds = dst_h5.create_dataset("full_traj_len", shape=(num_trajs,), dtype=np.int32, compression="gzip")
            task_id_ds = dst_h5.create_dataset("task_id", shape=(num_trajs,), dtype=src_h5["task_id"].dtype, compression="gzip")
            # full_path_ds = dst_h5.create_dataset(
            #     "sol_path_full", shape=(num_trajs, seq_len, *base_shape), dtype=scalar_dtype, compression="gzip"
            # )

            # Prepare passthrough datasets that align with the new length.
            replicated_src = {}
            replicated_dst = {}
            for key, ds in src_h5.items():
                if key in {"sol_path", "task_id"}:
                    continue
                if key in dst_h5:
                    continue
                if ds.shape and ds.shape[0] == num_trajs_original:
                    replicated_src[key] = ds
                    replicated_dst[key] = dst_h5.create_dataset(
                        key, shape=(num_trajs, *ds.shape[1:]), dtype=ds.dtype, compression="gzip"
                    )
                else:
                    dst_h5.create_dataset(key, data=ds[:], dtype=ds.dtype, compression="gzip")

            for i in tqdm(range(num_trajs), desc="Sampling segments"):
                base_idx = int(rng.integers(0, num_trajs_original))
                path = np.asarray(sol_paths[base_idx])
                segment, start_idx = sample_segment(path, rng, args.min_frac, args.max_frac)

                # pad segment to fixed max length with NaNs for downstream masking
                padded = np.full((max_seg_len, *base_shape), np.nan, dtype=scalar_dtype)
                padded[: segment.shape[0]] = segment

                seg_ds[i] = padded
                # full_path_ds[i] = path.astype(scalar_dtype)
                progress_ds[i] = start_idx / path.shape[0]
                # delta_ds[i] = (path[-1] - segment[-1]).astype(scalar_dtype, copy=False)
                # start_idx_ds[i] = start_idx
                # seg_len_ds[i] = len(segment)
                start_ds[i] = path[0].astype(scalar_dtype, copy=False)
                goal_ds[i] = path[-1].astype(scalar_dtype, copy=False)
                # full_len_ds[i] = path.shape[0]
                task_id_ds[i] = src_h5["task_id"][base_idx]

                for key, dst in replicated_dst.items():
                    dst[i] = replicated_src[key][base_idx]

    print(f"Saved composed dataset with {num_trajs} segments to {output_path}")


if __name__ == "__main__":
    main()
