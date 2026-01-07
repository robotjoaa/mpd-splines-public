import os
import isaacgym
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


# path to the experiment folder (adjust if needed)
base = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs/launch_inference-experiments-comp_v2_2025-12-08_01-19-45/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/selection_start_goal___test/extra_objects___True/planner_alg___mpd_comp/model_selection___bspline/phase_time_class___PhaseTimeLinear/diffusion_sampling_method___ddim/n_diffusion_steps_without_noise___0/project_gradient_hierarchy___False/trajectory_duration___4.2/is_hard___False/guide_mode___hybrid/n_comp___3/len_ovlp_cd___8/use_end_ovlp_model___True/context_progress___True/global_context_qs___True/global_context_ee_goal_pose___False/condition_guidance_l___1.5/condition_guidance_g___0.5/0"
plan_ids = np.arange(0,100)
vals, missing, fail = {}, [], []
for pid in plan_ids:
    path = os.path.join(base, f"results_single_plan-{pid:03d}.pt")
    if not os.path.exists(path):
        missing.append(pid)
        continue
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        succ = d["metrics"]["trajs_all"]["success"] if isinstance(d, dict) else d.metrics.trajs_all.success
        vals[pid] = float(succ)
    except Exception as e:
        fail.append((pid, str(e)))

print("success values:", vals, sum(vals.values()))
print("missing:", missing)
print("failed:", fail)