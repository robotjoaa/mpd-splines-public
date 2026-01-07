import os
import glob
import isaacgym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from torch_robotics.torch_utils.torch_utils import (
    to_numpy,
)
from dotmap import DotMap

# load results dict from experiment
LOG_DIR = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs"

from analyze_result import load_results, get_option_name_str
from pprint import pprint
import json 
import pdb
import types
from typing import Any, Dict, List, Mapping

# exp_name = "launch_inference-experiments-test_2025-11-07_17-35-52" # margin scale 4 
# exp_name = "launch_inference-experiments-test_2025-11-07_17-40-57" # margin scale 4 failed
# option_l = [["waypoints", "bspline"]] #bspline plot front

# EXP_NAME = "launch_inference-experiments-test_2025-11-07_17-35-52" # easy
# EXP_NAME = "launch_inference-experiments-test_2025-12-07_20-28-11" # hard
EXP_NAME = "launch_inference-experiments-test_2025-12-08_03-00-28" # static
OPTION_L = [["waypoints", "bspline"]]
# exp_name = "launch_inference-experiments-test_2025-11-07_14-32-00"
# Load results
# print("Loading results...")
# all_results, option_l = load_results(exp_name, option_l)

# # pdb.set_trace()

# # print config of 23
# for k, v in all_results.items():
#     option_name_str = get_option_name_str(k, option_l)
#     print(f"\nOption {option_name_str}: {len(v)} samples")
#     if len(v) > 0:
#         #print(f"  Sample result keys: {list(v[0].keys())}")
#         #if hasattr(v[0], 'dyn_obj_config') and v[0].dyn_obj_config is not None:
#             #print(f"  dyn_obj_config keys: {list(v[0].dyn_obj_config.keys())}")
#         pprint(v[1].dyn_obj_config)
#         print("q_pos_start")
#         pprint(v[1].q_pos_start)
#         print("q_pos_goal")
#         pprint(v[1].q_pos_goal)

'''
This script loads inference results for an experiment, pulls out the dynamic
obstacle configuration along with the start/goal joint positions, and writes
them to a JSON file. The JSON can then be inspected to verify that the problem
setups (e.g., waypoints vs. B-spline) are identical.
'''

# def load_results(exp_name: str, option_l: List[List[str]]):
#     """Load experiment results without depending on external utilities."""
#     all_results: Dict = {}

#     pattern = os.path.join(LOG_DIR, exp_name, "**/args.yaml")
#     exp_base = [os.path.dirname(p) for p in glob.glob(pattern, recursive=True)]
#     print("num settings : ", len(exp_base))

#     for exp in exp_base:
#         n_idx = []
#         for option in option_l:
#             match_idx = [i for i, p in enumerate(option) if p in exp]
#             assert len(match_idx) == 1
#             n_idx.append(match_idx[0])

#         selected_name = tuple(n_idx)
#         pattern = os.path.join(exp, "results_single_plan-*")
#         file_names = glob.glob(pattern)
#         result = []

#         for filename in file_names:
#             tmp_result = torch.load(filename, weights_only=False)
#             result.append(tmp_result)

#         print(selected_name)
#         all_results[selected_name] = result

#     return all_results, option_l


# def get_option_name_str(option_tuple, option_l):
#     """Convert option index tuple to readable string using option_l."""
#     parts = []
#     for idx, option_idx in enumerate(option_tuple):
#         if idx < len(option_l):
#             parts.append(option_l[idx][option_idx])
#     return "-".join(parts)


def _to_serializable(value: Any) -> Any:
    """Convert torch/NumPy/DotMap structures to JSON-friendly Python types."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (DotMap, Mapping)):
        return {k: _to_serializable(v) for k, v in dict(value).items()}
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def extract_problem_configs(all_results: Dict, option_l: List[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Collect dyn_obj_config, start, and goal for each option."""
    problem_configs: Dict[str, List[Dict[str, Any]]] = {}

    for option_tuple, results in all_results.items():
        option_name = get_option_name_str(option_tuple, option_l)
        option_data: List[Dict[str, Any]] = []

        for idx, result in enumerate(results):
            entry: Dict[str, Any] = {"idx": idx}

            if hasattr(result, "q_pos_start"):
                entry["q_pos_start"] = _to_serializable(result.q_pos_start)
            if hasattr(result, "q_pos_goal"):
                entry["q_pos_goal"] = _to_serializable(result.q_pos_goal)
            if hasattr(result, "dyn_obj_config") and result.dyn_obj_config is not None:
                entry["dyn_obj_config"] = _to_serializable(result.dyn_obj_config)

            option_data.append(entry)

        problem_configs[option_name] = option_data

    return problem_configs


def compare_options(problem_configs: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print a quick equality check across options for the stored fields."""
    option_names = list(problem_configs.keys())
    if len(option_names) < 2:
        print("Only one option available; skipping comparison.")
        return

    reference_name = option_names[0]
    reference_data = problem_configs[reference_name]

    def _configs_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # keys = ("q_pos_start", "q_pos_goal", "dyn_obj_config")
        keys = ("q_pos_start", "q_pos_goal")
        return all(a.get(k) == b.get(k) for k in keys)

    for other_name in option_names[1:]:
        other_data = problem_configs[other_name]
        min_len = min(len(reference_data), len(other_data))

        mismatches = []
        for i in range(min_len):
            if not _configs_match(reference_data[i], other_data[i]):
                mismatches.append(i)

        print(f"Comparing {other_name} vs {reference_name}:")
        print(f"  Problem counts -> {len(reference_data)} vs {len(other_data)}")
        print(f"  Mismatched indices -> {mismatches[:10]}{'...' if len(mismatches) > 10 else ''}")

        if len(reference_data) != len(other_data):
            print("  Warning: option counts differ; extra problems not compared.")


def main(exp_name: str = EXP_NAME, option_l: List[List[str]] = OPTION_L) -> None:
    print(f"Loading results for {exp_name}...")
    all_results, option_l = load_results(exp_name, option_l)

    problem_configs = extract_problem_configs(all_results, option_l)

    output_dir = os.path.join(LOG_DIR, exp_name, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "problem_configs.json")

    with open(output_path, "w") as f:
        json.dump(problem_configs, f, indent=2)

    print(f"Saved problem configs to {output_path}")
    compare_options(problem_configs)


if __name__ == "__main__":
    main()