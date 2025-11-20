import os
import glob
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
# exp_name = "launch_inference-experiments-test_2025-11-07_17-35-52" # margin scale 4 
# exp_name = "launch_inference-experiments-test_2025-11-07_17-40-57" # margin scale 4 failed
option_l = [["waypoints", "bspline"]] #bspline plot front
exp_name = "launch_inference-experiments-test_2025-11-07_14-32-00"
# Load results
print("Loading results...")
all_results, option_l = load_results(exp_name, option_l)

# print config of 23
for k, v in all_results.items():
    option_name_str = get_option_name_str(k, option_l)
    print(f"\nOption {option_name_str}: {len(v)} samples")
    if len(v) > 0:
        #print(f"  Sample result keys: {list(v[0].keys())}")
        #if hasattr(v[0], 'dyn_obj_config') and v[0].dyn_obj_config is not None:
            #print(f"  dyn_obj_config keys: {list(v[0].dyn_obj_config.keys())}")
        pprint(v[1].dyn_obj_config)
        print("q_pos_start")
        pprint(v[1].q_pos_start)
        print("q_pos_goal")
        pprint(v[1].q_pos_goal)