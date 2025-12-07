import datetime
import os.path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch

from mpd.datasets.trajectories_dataset_bspline import TrajectoryDatasetBspline
from mpd.datasets.trajectories_dataset_waypoints import (
    TrajectoryDatasetWaypoints,
    subsample_waypoints,
)
from torch_robotics import robots
from torch_robotics.trajectory.utils import interpolate_points_v1
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy
from pb_ompl.pb_ompl import fit_bspline_to_path
import pickle
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
    link_rot_from_link_tensor,
    rmat_to_flat,
)
### Used only for training comp diffusion model, not for inference ### 

class _TrajectoryCompBase:
    """Mixin for shared comp-dataset field keys."""

    def _init_comp_fields(self):
        self.field_key_progress = "progress"
        self.field_key_delta_to_goal = "delta_to_goal"
        self.field_key_segment_start_idx = "segment_start_idx"
        self.field_key_segment_len = "segment_len"
        self.field_key_full_traj_len = "full_traj_len"
        self.field_key_full_start = "start_state"
        self.field_key_full_goal = "goal_state"
        self.field_key_task_id_comp = "task_id"
        self.field_key_full_sol_path = "sol_path_full"
        self.field_key_full_ee_goal_orientation = "full_ee_goal_orientation"
        self.field_key_full_ee_goal_position = "full_ee_goal_position"
        self.field_key_full_ee_goal_orientation = "full_ee_goal_orientation"
    def _attach_comp_fields(
        self,
        fields_d: Dict[str, Any],
        progress,
        delta_to_goal,
        segment_start_idx,
        segment_len,
        full_traj_len,
        full_start,
        full_goal,
        task_ids,
        device: Optional[str] = None,
    ):
        device = device if device is not None else "cpu"
        if progress is not None:
            fields_d[self.field_key_progress] = torch.as_tensor(
                progress, dtype=torch.float32, device=device
            )
        if delta_to_goal is not None:
            fields_d[self.field_key_delta_to_goal] = torch.as_tensor(
                delta_to_goal, dtype=self.tensor_args["dtype"], device=device
            )
        if segment_start_idx is not None:
            fields_d[self.field_key_segment_start_idx] = torch.as_tensor(
                segment_start_idx, dtype=torch.long, device=device
            )
        if segment_len is not None:
            fields_d[self.field_key_segment_len] = torch.as_tensor(
                segment_len, dtype=torch.long, device=device
            )
        if full_traj_len is not None:
            fields_d[self.field_key_full_traj_len] = torch.as_tensor(
                full_traj_len, dtype=torch.long, device=device
            )
        if full_start is not None:
            fields_d[self.field_key_full_start] = torch.as_tensor(
                full_start, dtype=self.tensor_args["dtype"], device=device
            )
        if full_goal is not None:
            fields_d[self.field_key_full_goal] = torch.as_tensor(
                full_goal, dtype=self.tensor_args["dtype"], device=device
            )
        if task_ids is not None:
            fields_d[self.field_key_task_id_comp] = torch.as_tensor(
                task_ids, dtype=torch.long, device=device
            )
        return fields_d


class TrajectoryCompDatasetBspline(_TrajectoryCompBase, TrajectoryDatasetBspline):
    """Compositional variant that loads partial segments plus metadata."""

    def __init__(self, **kwargs):
        self._init_comp_fields()
        comp_norm_keys = [
            self.field_key_delta_to_goal,
            self.field_key_full_start,
            self.field_key_full_goal,
            self.field_key_full_ee_goal_position,
            self.field_key_full_ee_goal_orientation,
        ]
        super().__init__(extra_keys = comp_norm_keys, **kwargs)

        # Normalize additional comp-specific continuous fields
        
        # self.normalizer_keys.extend([k for k in comp_norm_keys if k not in self.normalizer_keys])
        # self.normalize_all_data(*comp_norm_keys)

    def load_data(self, n_task_samples=-1):
        with TimerCUDA() as t_load_data:
            print("Loading composed bspline data ...")
            # File name for data reload
            data_reload_prefix = f'{self.dataset_file_merged.replace(".hdf5", "")}_reload'
            data_reload_prefix += f"-ntasks_{n_task_samples}"
            data_reload_prefix += f"--bspline"
            data_reload_prefix += f"-degree_{self.planning_task.parametric_trajectory.bspline.d}"
            data_reload_prefix += f"-n_pts_{self.planning_task.parametric_trajectory.bspline.n_pts}"
            data_reload_prefix += f"-zero_vel_{self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal}"
            data_reload_prefix += f"-zero_acc_{self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal}"
            data_reload_file_path = os.path.join(self.base_dir, f"{data_reload_prefix}_comp.pickle") # tmp without statistics

            if os.path.exists(data_reload_file_path) and not self.reload_data:
                # load the pre-processed dataset
                self.reload_data_fn(data_reload_file_path, n_task_samples=n_task_samples)
            else:
                dataset_h5 = h5py.File(os.path.join(self.base_dir, self.dataset_file_merged), "r")

                inner_control_points_all = []
                q_start_all = []
                q_goal_all = []

                progress_all = []
                delta_all = []
                seg_start_idx_all = []
                seg_len_all = []
                full_len_all = []
                full_start_all = []
                full_goal_all = []
                task_ids_all = []

                task_ids_processed = []
                cps_idx = 0
                n_discarded_trajectories = 0
                for i, path_padded in enumerate(dataset_h5["sol_path"]):
                    if n_task_samples != -1 and i > 0:
                        task_ids_processed.append(dataset_h5[self.field_key_task_id_comp][i])
                        task_ids_processed = list(set(task_ids_processed))
                        if len(task_ids_processed) >= n_task_samples:
                            break

                    # Remove NaN padding based on stored segment length
                    seg_len = int(dataset_h5[self.field_key_segment_len][i])
                    path = np.asarray(path_padded)[:seg_len]

                    try:
                        bspline_params = fit_bspline_to_path(
                            path,
                            bspline_degree=self.planning_task.parametric_trajectory.bspline.d,
                            bspline_num_control_points=self.planning_task.parametric_trajectory.bspline.n_pts,
                            bspline_zero_vel_at_start_and_goal=self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                            bspline_zero_acc_at_start_and_goal=self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                            debug=False,
                        )
                        _, cc_tmp, _ = bspline_params
                        if np.any(cc_tmp.min(1) <= 2 * to_numpy(self.planning_task.robot.q_pos_min)) or np.any(
                            cc_tmp.max(1) >= 2 * to_numpy(self.planning_task.robot.q_pos_max)
                        ):
                            n_discarded_trajectories += 1
                            raise Exception
                    except Exception:
                        continue

                    tt, cc, k = bspline_params
                    cc_np = np.array(cc)
                    if isinstance(self.planning_task.robot, robots.RobotPanda) and cc_np.shape[0] > 9:
                        cc_np = cc_np[:7, :]

                    control_points = to_torch(cc_np, dtype=self.tensor_args["dtype"], device="cpu").transpose(0, 1)
                    q_start_all.append(control_points[0])
                    q_goal_all.append(control_points[-1])

                    inner_control_points = self.planning_task.parametric_trajectory.remove_control_points_fn(
                        control_points
                    )

                    inner_control_points_all.append(inner_control_points)

                    task_id = dataset_h5[self.field_key_task_id_comp][i]
                    self.map_control_points_id_to_task_id[cps_idx] = task_id
                    if task_id in self.map_task_id_to_control_points_id:
                        self.map_task_id_to_control_points_id[task_id].append(cps_idx)
                    else:
                        self.map_task_id_to_control_points_id[task_id] = [cps_idx]

                    progress_all.append(dataset_h5[self.field_key_progress][i])
                    delta_all.append(dataset_h5[self.field_key_delta_to_goal][i])
                    seg_start_idx_all.append(dataset_h5[self.field_key_segment_start_idx][i])
                    seg_len_all.append(dataset_h5[self.field_key_segment_len][i])
                    full_len_all.append(dataset_h5[self.field_key_full_traj_len][i])
                    task_ids_all.append(task_id)
                    full_start_all.append(dataset_h5[self.field_key_full_start][i]) 
                    full_goal_all.append(dataset_h5[self.field_key_full_goal][i])

                    # np_start = dataset_h5[self.field_key_full_start][i] 
                    # np_goal = dataset_h5[self.field_key_full_goal][i]
                    # full_start_all.append(to_torch(np_start, dtype=self.tensor_args["dtype"], device="cpu")) 
                    # full_goal_all.append(to_torch(np_goal, dtype=self.tensor_args["dtype"], device="cpu"))

                    cps_idx += 1

                    if i % 20000 == 0 or i == len(dataset_h5["sol_path"]) - 1:
                        print(
                            f"Time spent: {str(datetime.timedelta(seconds=t_load_data.elapsed))} - "
                            f'loaded {i}/{len(dataset_h5["sol_path"])} '
                            f'({i/len(dataset_h5["sol_path"]):.2%}) trajectories.'
                            )

                #print(f'Number of discarded trajectories: {n_discarded_trajectories}/{len(dataset_h5["sol_path"])}')
                inner_control_points_tensor = torch.stack(inner_control_points_all)
                self.fields[self.field_key_control_points] = inner_control_points_tensor
                self.fields = self.build_fields_data_sample(
                    self.fields,
                    torch.stack(q_start_all),  # use start, goal from original path
                    torch.stack(q_goal_all),
                    device="cpu",
                    progress=progress_all,
                    delta_to_goal=delta_all,
                    segment_start_idx=seg_start_idx_all,
                    segment_len=seg_len_all,
                    full_traj_len=full_len_all,
                    full_start=full_start_all,
                    full_goal=full_goal_all,
                    task_ids=task_ids_all,
                )

                # self.run_collision_statistics()
                # Save data to disk to speed up loading the next time.
                data_to_save = {
                    "fields": self.fields,
                    "map_task_id_to_control_points_id": self.map_task_id_to_control_points_id,
                    "map_control_points_id_to_task_id": self.map_control_points_id_to_task_id,
                    # "percentage_free_trajs": (np.mean(percentage_free_trajs_l), np.std(percentage_free_trajs_l)),
                    # "percentage_collision_intensity": (
                        # np.mean(percentage_collision_intensity_l),
                        # np.std(percentage_collision_intensity_l),
                    # ),
                }
                pickle.dump(data_to_save, open(data_reload_file_path, "wb"))

            print("... done loading data.")
            print(f"Loading data took {t_load_data.elapsed:.2f} seconds.")

    def build_fields_data_sample(
        self,
        fields_d: Dict[str, Any],
        q_start,
        q_goal,
        ee_pose_goal=None,
        device=None,
        progress=None,
        delta_to_goal=None,
        segment_start_idx=None,
        segment_len=None,
        full_traj_len=None,
        full_start=None,
        full_goal=None,
        task_ids=None,
        **kwargs,
    ):
        fields_d = super().build_fields_data_sample(fields_d, q_start, q_goal, ee_pose_goal=ee_pose_goal, device=device)
        # compute end-effector goal for full trajectory goal if provided
        if full_goal is not None:
            ee_goal_full = self.planning_task.robot.get_EE_pose(
                to_torch(full_goal, **self.planning_task.robot.tensor_args)
            ).to(device if device is not None else self.tensor_args["device"])
            fields_d["full_" + self.field_key_context_ee_goal_pose] = ee_goal_full
            fields_d[self.field_key_full_ee_goal_orientation] = rmat_to_flat(
                link_rot_from_link_tensor(ee_goal_full)
            )
            fields_d[self.field_key_full_ee_goal_position] = link_pos_from_link_tensor(ee_goal_full)
            # fields_d["full_" + self.field_key_context_ee_goal_position] = link_pos_from_link_tensor(ee_goal_full)

        fields_d = self._attach_comp_fields(
            fields_d,
            progress,
            delta_to_goal,
            segment_start_idx,
            segment_len,
            full_traj_len,
            full_start,
            full_goal,
            task_ids,
            device=device if device is not None else "cpu",
        )
        return fields_d

    def build_context(self, data_sample, is_train=True):
        if is_train : 
            context_d = {
                self.field_key_progress: data_sample[self.field_key_progress], # already normalized
                # f"{self.field_key_delta_to_goal}_normalized": data_sample[f"{self.field_key_delta_to_goal}_normalized"],
                f"{self.field_key_full_start}_normalized": data_sample[f"{self.field_key_full_start}_normalized"],
                f"{self.field_key_full_goal}_normalized": data_sample[f"{self.field_key_full_goal}_normalized"],
                f"{self.field_key_full_ee_goal_position}_normalized": data_sample[f"{self.field_key_full_ee_goal_position}_normalized"],
                f"{self.field_key_full_ee_goal_orientation}_normalized": data_sample[f"{self.field_key_full_ee_goal_orientation}_normalized"]
            }
        else : 
            context_d = {
                # f"{self.field_key_delta_to_goal}_normalized": data_sample[f"{self.field_key_delta_to_goal}_normalized"],
                f"{self.field_key_full_start}_normalized": data_sample[f"{self.field_key_q_start}_normalized"],
                f"{self.field_key_full_goal}_normalized": data_sample[f"{self.field_key_q_goal}_normalized"],
                f"{self.field_key_full_ee_goal_position}_normalized": data_sample[f"{self.field_key_context_ee_goal_position}_normalized"],
                f"{self.field_key_full_ee_goal_orientation}_normalized": data_sample[f"{self.field_key_context_ee_goal_orientation}_normalized"]
            }
        return context_d

class TrajectoryCompDatasetWaypoint(_TrajectoryCompBase, TrajectoryDatasetWaypoints):
    """Compositional variant for waypoint-based trajectories."""

    def __init__(self, **kwargs):
        self._init_comp_fields()
        comp_norm_keys = [
            self.field_key_delta_to_goal,
            self.field_key_full_start,
            self.field_key_full_goal,
            self.field_key_full_ee_goal_position,
            self.field_key_full_ee_goal_orientation,
        ]
        super().__init__(extra_keys = comp_norm_keys, **kwargs)

        # self.normalizer_keys.extend([k for k in comp_norm_keys if k not in self.normalizer_keys])
        # self.normalize_all_data(*comp_norm_keys)

    def load_data(self, n_task_samples=-1):
        with TimerCUDA() as t_load_data:
            print("Loading composed waypoint data ...")
            data_reload_prefix = f'{self.dataset_file_merged.replace(".hdf5", "")}_reload'
            data_reload_prefix += f"-ntasks_{n_task_samples}"
            data_reload_prefix += f"--waypoints"
            data_reload_prefix += f"-n_pts_{self.planning_task.parametric_trajectory.n_control_points}"
            data_reload_pickle_path = os.path.join(self.base_dir, f"{data_reload_prefix}_comp.pickle")

            if os.path.exists(data_reload_pickle_path) and not self.reload_data:
                print(data_reload_pickle_path)
                self.reload_data_fn(data_reload_pickle_path, n_task_samples=n_task_samples)
                print("... done loading data.")
                print(f"Loading data took {t_load_data.elapsed:.2f} seconds.")
                return
            else : 
                dataset_h5 = h5py.File(os.path.join(self.base_dir, self.dataset_file_merged), "r")

                inner_control_points_all = []
                q_start_all = []
                q_goal_all = []

                progress_all = []
                delta_all = []
                seg_start_idx_all = []
                seg_len_all = []
                full_len_all = []
                full_start_all = []
                full_goal_all = []
                task_ids_all = []

                task_ids_processed = []
                cps_idx = 0
                #import pdb; pdb.set_trace()
                for i, path_padded in enumerate(dataset_h5["sol_path"]):
                    if n_task_samples != -1 and i > 0:
                        task_ids_processed.append(dataset_h5[self.field_key_task_id_comp][i])
                        task_ids_processed = list(set(task_ids_processed))
                        if len(task_ids_processed) >= n_task_samples:
                            break

                    seg_len = int(dataset_h5[self.field_key_segment_len][i])
                    sol_path = np.asarray(path_padded)[:seg_len]
                    n_target_points = self.planning_task.parametric_trajectory.n_control_points
                    if n_target_points <= sol_path.shape[0]:
                        control_points = subsample_waypoints(sol_path, n_target_points).to(dtype=self.tensor_args["dtype"])
                    else:
                        control_points = interpolate_points_v1(
                            torch.from_numpy(sol_path).to(dtype=self.tensor_args["dtype"])[None, ...],
                            n_target_points,
                        ).squeeze()

                    if isinstance(self.planning_task.robot, robots.RobotPanda) and control_points.shape[-1] == 9:
                        control_points = control_points[..., :7]

                    q_start_all.append(control_points[0])
                    q_goal_all.append(control_points[-1])

                    inner_control_points = self.planning_task.parametric_trajectory.remove_control_points_fn(
                        control_points
                    )
                    inner_control_points_all.append(inner_control_points)

                    task_id = dataset_h5[self.field_key_task_id_comp][i]
                    self.map_control_points_id_to_task_id[cps_idx] = task_id
                    if task_id in self.map_task_id_to_control_points_id:
                        self.map_task_id_to_control_points_id[task_id].append(cps_idx)
                    else:
                        self.map_task_id_to_control_points_id[task_id] = [cps_idx]

                    progress_all.append(dataset_h5[self.field_key_progress][i])
                    delta_all.append(dataset_h5[self.field_key_delta_to_goal][i])
                    seg_start_idx_all.append(dataset_h5[self.field_key_segment_start_idx][i])
                    seg_len_all.append(dataset_h5[self.field_key_segment_len][i])
                    full_len_all.append(dataset_h5[self.field_key_full_traj_len][i])
                    full_start_all.append(dataset_h5[self.field_key_full_start][i])
                    full_goal_all.append(dataset_h5[self.field_key_full_goal][i])
                    task_ids_all.append(task_id)

                    cps_idx += 1

                    if i % 20000 == 0 or i == len(dataset_h5["sol_path"]) - 1:
                        print(
                            f"Time spent: {str(datetime.timedelta(seconds=t_load_data.elapsed))} - "
                            f'loaded {i}/{len(dataset_h5["sol_path"])} '
                            f'({i/len(dataset_h5["sol_path"]):.2%}) trajectories.'
                        )

                inner_control_points_tensor = torch.stack(inner_control_points_all)
                self.fields[self.field_key_control_points] = inner_control_points_tensor

                self.fields = self.build_fields_data_sample(
                    self.fields,
                    torch.stack(q_start_all),
                    torch.stack(q_goal_all),
                    device="cpu",
                    progress=progress_all,
                    delta_to_goal=delta_all,
                    segment_start_idx=seg_start_idx_all,
                    segment_len=seg_len_all,
                    full_traj_len=full_len_all,
                    full_start=full_start_all,
                    full_goal=full_goal_all,
                    task_ids=task_ids_all,
                )

                data_to_save = {
                    "fields": self.fields,
                    "map_task_id_to_control_points_id": self.map_task_id_to_control_points_id,
                    "map_control_points_id_to_task_id": self.map_control_points_id_to_task_id,
                }
                pickle.dump(data_to_save, open(data_reload_pickle_path, "wb"))

            print("... done loading data.")
            print(f"Loading data took {t_load_data.elapsed:.2f} seconds.")

    def build_fields_data_sample(
        self,
        fields_d: Dict[str, Any],
        q_start,
        q_goal,
        ee_pose_goal=None,
        device=None,
        progress=None,
        delta_to_goal=None,
        segment_start_idx=None,
        segment_len=None,
        full_traj_len=None,
        full_start=None,
        full_goal=None,
        task_ids=None,
        **kwargs,
    ):
        fields_d = super().build_fields_data_sample(fields_d, q_start, q_goal, ee_pose_goal=ee_pose_goal, device=device)
        if full_goal is not None:
            ee_goal_full = self.planning_task.robot.get_EE_pose(
                to_torch(full_goal, **self.planning_task.robot.tensor_args)
            ).to(device if device is not None else self.tensor_args["device"])
            fields_d["full_" + self.field_key_context_ee_goal_pose] = ee_goal_full
            fields_d[self.field_key_full_ee_goal_orientation] = rmat_to_flat(
                link_rot_from_link_tensor(ee_goal_full)
            )
            fields_d[self.field_key_full_ee_goal_position] = link_pos_from_link_tensor(ee_goal_full)

        fields_d = self._attach_comp_fields(
            fields_d,
            progress,
            delta_to_goal,
            segment_start_idx,
            segment_len,
            full_traj_len,
            full_start,
            full_goal,
            task_ids,
            device=device if device is not None else "cpu",
        )
        return fields_d

    def build_context(self, data_sample, is_train=True):
        if is_train : 
            context_d = {
                self.field_key_progress: data_sample[self.field_key_progress], # already normalized
                # f"{self.field_key_delta_to_goal}_normalized": data_sample[f"{self.field_key_delta_to_goal}_normalized"],
                f"{self.field_key_full_start}_normalized": data_sample[f"{self.field_key_full_start}_normalized"],
                f"{self.field_key_full_goal}_normalized": data_sample[f"{self.field_key_full_goal}_normalized"],
                f"{self.field_key_full_ee_goal_position}_normalized": data_sample[f"{self.field_key_full_ee_goal_position}_normalized"],
                f"{self.field_key_full_ee_goal_orientation}_normalized": data_sample[f"{self.field_key_full_ee_goal_orientation}_normalized"]
            }
        else : 
            context_d = {
                # f"{self.field_key_delta_to_goal}_normalized": data_sample[f"{self.field_key_delta_to_goal}_normalized"],
                f"{self.field_key_full_start}_normalized": data_sample[f"{self.field_key_q_start}_normalized"],
                f"{self.field_key_full_goal}_normalized": data_sample[f"{self.field_key_q_goal}_normalized"],
                f"{self.field_key_full_ee_goal_position}_normalized": data_sample[f"{self.field_key_context_ee_goal_position}_normalized"],
                f"{self.field_key_full_ee_goal_orientation}_normalized": data_sample[f"{self.field_key_context_ee_goal_orientation}_normalized"]
            }
        return context_d