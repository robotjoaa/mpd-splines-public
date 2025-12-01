import time
from functools import partial

import einops
import torch
from dotmap import DotMap

from deps.theseus.torchlie.torchlie.functional.se3_impl import _adjoint_impl
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_utils.torch_timer import TimerCUDA

from torchlie.functional import SE3 as SE3_Func

from torch.func import vmap, jacrev, functional_call

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy
from torch_robotics.visualizers.plot_utils import create_fig_and_axes
from mpd.inference.cost_guides import (
    CostGuideManagerParametricTrajectory,
    NoCostException,
    project_hierarchical_gradients_fast,
)
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.datasets.trajectories_dataset_bspline import adjust_bspline_number_control_points
from mpd.datasets.trajectories_dataset_waypoints import adjust_waypoints

import pdb
class CostGuideManagerCompTrajectory(CostGuideManagerParametricTrajectory):

    def __init__(
        self,
        planning_task,
        dataset,
        args_inference,
        n_comp,
        len_ovlp_cd,
        tensor_args=DEFAULT_TENSOR_ARGS,
        debug=False,
        **kwargs,
    ):
        self.n_comp = n_comp
        self.len_ovlp_cd = len_ovlp_cd

        # Keep a handle on the original (local) parametric trajectory before the
        # parent class copies it – costs should be computed on these local
        # segments.
        self._parametric_trajectory_local = planning_task.parametric_trajectory
        self.global_start = planning_task.q_pos_start
        self.global_goal = planning_task.q_pos_goal
        print(f"{self.global_start=},{self.global_goal=}")
        self.n_control_points_local = self._parametric_trajectory_local.n_control_points
        self.num_T_pts_local = self._parametric_trajectory_local.num_T_pts
        self.traj_duration_local = self._parametric_trajectory_local.trajectory_duration

        print("CostGuideManagerCompTrajectory: ",self.n_control_points_local, self.num_T_pts_local, self.traj_duration_local)
        super().__init__(planning_task, dataset, args_inference, tensor_args=tensor_args, debug=debug, **kwargs)

        phase_time_class, phase_time_args = self._get_phase_time_config()
        self.common_kwargs = dict(
            num_T_pts=self.num_T_pts_local,
            trajectory_duration=self.traj_duration_local,
            phase_time_class=phase_time_class,
            phase_time_args=phase_time_args,
            tensor_args=self.tensor_args,
        )

        # Overlap portion expressed in the same units as control points and time steps.
        ovlp_frac = self.len_ovlp_cd / float(self.n_control_points_local)
        self.overlap_time = self.traj_duration_local * ovlp_frac
        self.overlap_T_pts = max(1, int(round(self.num_T_pts_local * ovlp_frac)))
        print(f'{ovlp_frac=}, {self.overlap_time=}, {self.overlap_T_pts=}')

        # Build the merged/global trajectory used for rendering / merged rollouts.
        self._parametric_trajectory_global = self._set_merged_trajectory()
       
        assert hasattr(self, 'traj_kwargs') # must be set in set_merged_trajectory 

        # self.planning_task.merged_trajectory = self._parametric_trajectory_global

        # modify parameteric_trajectory_local following setup.
        # use this per i_th 
        self.local_trajectory_d = self._adjust_local_trajectory()
        
        self.planning_task.parametric_trajectory = self._parametric_trajectory_global
        print(f"{self.planning_task.parametric_trajectory.n_control_points=}")
        # Ensure the guide keeps using the local segment trajectory for cost computation.
        self.parametric_trajectory = self._parametric_trajectory_local # placeholder
        
    def reset_task_trajectory(self) : 
        print(f"reset cost function parametric trajectory")
        for cost_key in self.costs:
            cost_fn = self.costs[cost_key].cost
            cost_fn.update_parametric_trajectory(self.planning_task.parametric_trajectory)

        # self.planning_task.parametric_trajectory = self._parametric_trajectory_global

    def get_parametric_trajectory(self, idx):  # key : "start", "end", ("mid")
        assert idx < self.n_comp and idx >= 0
        offset_info = {}
        # when current local trajectory starts
        offset_info["t_offset"] = idx* (self.traj_duration_local - self.overlap_time)
        # following num_points
        #int(self.n_comp * num_T_pts_local - (self.n_comp - 1) * overlap_T_pts
        offset_info["num_offset"] = idx * (self.num_T_pts_local - self.overlap_T_pts)

        if idx == 0 :
            key = "start"
        elif idx == self.n_comp - 1 :
            key = "end"
        else : 
            key = "mid"

        return self.local_trajectory_d[key], offset_info

    def _get_phase_time_config(self):
        phase_time = self._parametric_trajectory_local.phase_time
        phase_time_class = phase_time.__class__.__name__
        phase_time_args = {}
        if hasattr(phase_time, "alpha"):
            phase_time_args["alpha"] = phase_time.alpha
        return phase_time_class, phase_time_args

    def _clone_parametric_trajectory(self, n_control_points, num_T_pts, trajectory_duration):
        phase_time_class, phase_time_args = self._get_phase_time_config()
        common_kwargs = dict(
            num_T_pts=num_T_pts,
            trajectory_duration=trajectory_duration,
            phase_time_class=phase_time_class,
            phase_time_args=phase_time_args,
            tensor_args=self.tensor_args,
        )

        template = self._parametric_trajectory_local
        if isinstance(template, ParametricTrajectoryWaypoints):
            # reverse of augment control point
            _, num_add = adjust_waypoints(n_control_points, template.remove_outer_control_points, template.keep_last_control_points)
            print(f"control points added : {num_add}")
            self.traj_kwargs = dict(
                remove_outer_control_points=template.remove_outer_control_points,
                keep_last_control_point=template.keep_last_control_points,
                use_interpolation_matrix=template.use_interpolation_matrix,
            )
            return ParametricTrajectoryWaypoints(
                n_control_points=n_control_points + num_add,
                **self.traj_kwargs,
                **common_kwargs,
            )
        if isinstance(template, ParametricTrajectoryBspline):
            # reverse of augment control point
            _, num_add = adjust_bspline_number_control_points(n_control_points, template.remove_outer_control_points, template.keep_last_control_points, \
                                                              template.zero_vel_at_start_and_goal, template.zero_acc_at_start_and_goal)
            
            print(f"control points added : {num_add}")
            self.traj_kwargs = dict(
                degree=template.bspline.degree,
                zero_vel_at_start_and_goal=template.zero_vel_at_start_and_goal,
                zero_acc_at_start_and_goal=template.zero_acc_at_start_and_goal,
                remove_outer_control_points=template.remove_outer_control_points,
                keep_last_control_point=template.keep_last_control_point,
            )
        
            return ParametricTrajectoryBspline(
                n_control_points=n_control_points + num_add,
                **self.traj_kwargs,
                **common_kwargs,
            )

        raise NotImplementedError(f"Unsupported parametric trajectory type: {type(template)}")

    def _set_merged_trajectory(self):
        # Local statistics
        n_cp_local = getattr(self.dataset, "n_learnable_control_points", self.dataset.control_points_dim[0])
        
        # n_cp_local = self.n_control_points_local # 18
        # Overlap portion expressed in the same units as control points and time steps.
        ovlp_frac = self.len_ovlp_cd / float(n_cp_local)
        overlap_time = self.traj_duration_local * ovlp_frac
        overlap_T_pts = max(1, int(round(self.num_T_pts_local * ovlp_frac)))

        n_cp_global = max(1, int(self.n_comp * n_cp_local - (self.n_comp - 1) * self.len_ovlp_cd))
        num_T_pts_global = max(1, int(self.n_comp * self.num_T_pts_local - (self.n_comp - 1) * overlap_T_pts))
        traj_duration_global = self.n_comp * self.traj_duration_local - (self.n_comp - 1) * overlap_time

        merged_traj = self._clone_parametric_trajectory(
            n_control_points=n_cp_global,
            num_T_pts=num_T_pts_global,
            trajectory_duration=traj_duration_global,
        )
        merged_traj.q_pos_start = getattr(self.planning_task, "q_pos_start", None)
        merged_traj.q_pos_goal = getattr(self.planning_task, "q_pos_goal", None)
        print(f"set merged trajectory start : {merged_traj.q_pos_start}, goal : {merged_traj.q_pos_goal}")
        return merged_traj
    
    def _adjust_local_trajectory(self):
        # Local statistics
        template = self._parametric_trajectory_local
        
        name_l = ['start', 'mid', 'end'] if self.n_comp > 2 else ['start', 'end'] 
        l_traj_dict = {}
        # convert global adjustment to local adjustment

        # self.common_kwargs = dict(
        #     num_T_pts=self.num_T_pts_local,
        #     trajectory_duration=self.traj_duration_local,
        #     phase_time_class=phase_time_class,
        #     phase_time_args=phase_time_args,
        #     tensor_args=self.tensor_args,
        # )
        #pdb.set_trace()
        for name in name_l : 
            n_control_points = template.n_control_points # 18
            tmp_kwargs = self.traj_kwargs.copy()
            #tmp_same = self.common_kwargs.copy()
            if name == "start" : 
                tmp_kwargs["remove_from_start_control_points"] = True
                n_control_points = n_control_points - 1 # start from 17
            elif name == "end" : 
                tmp_kwargs["remove_from_start_control_points"] = False
                n_control_points = n_control_points - 1 # start from 17 
            elif name == "mid":
                #tmp_kwargs["remove_from_start_control_points"] = None
                tmp_kwargs["remove_outer_control_points"] = False
                n_control_points = n_control_points - 2 # start from 16 (no remove)
            else :
                raise NotImplementedError 

            tmp_class = template.__class__
            traj_tmp = tmp_class(n_control_points=n_control_points,
                      **tmp_kwargs,
                      **self.common_kwargs
                      )
            # if isinstance(template, ParametricTrajectoryWaypoints):
            #     traj_tmp = ParametricTrajectoryWaypoints(
            #         n_control_points=template.n_control_points,
            #         **self.traj_kwargs,
            #         **self.common_kwargs,
            #     )
            # if isinstance(template, ParametricTrajectoryBspline):
            #     traj_tmp = ParametricTrajectoryBspline(
            #         n_control_points=n_control_points,
            #         **self.traj_kwargs,
            #         **self.common_kwargs,
            #     )
            
            l_traj_dict[name] = traj_tmp
        #pdb.set_trace()
        return l_traj_dict

    @torch.enable_grad()
    def __call__(self, control_points_normalized, idx=0, return_cost=False, warmup=False, plot_gradients=False, **kwargs):
        """
        Args:
            control_points_normalized: (batch_size, n_control_points, q_dim)
            idx -th out of n_comps,
        """
        if self.debug:
            print()
            print(f"Guide step {self.step_guide_call}")

        # # Keep local trajectory boundary conditions in sync with the planner.
        # This will be adjusted by local parametric_trajectory options stored in local_trajectory_d
        # self.parametric_trajectory.q_pos_start = self.planning_task.q_pos_start
        # self.parametric_trajectory.q_pos_goal = self.planning_task.q_pos_goal



        # timesteps from the compositional diffusion model are forwarded so that
        # cost functions can restrict themselves to the appropriate temporal
        # window (e.g., dynamic environments).

        # timesteps computed here
        # if "timestep" in kwargs and "timesteps" not in kwargs:
        #     kwargs["timesteps"] = kwargs.pop("timestep")

        # Unnormalize the control points.
        # The generative model outputs normalized control points, but the costs are defined on the unnormalized
        # trajectory space.
        # use current_traj
        current_traj, offset_info = self.get_parametric_trajectory(idx)
        
        # otherwise, control_points_normalized meet current_traj's n_control_points


        control_points = self.dataset.unnormalize_control_points(control_points_normalized)

        # Get the trajectory (position, velocity, acceleration) from the control points, in phase.
        control_points.requires_grad_(True)
    
        # pdb.set_trace()

        # update cost's trajectory
        for cost_key in self.costs:
            cost_fn = self.costs[cost_key].cost
            cost_fn.update_parametric_trajectory(current_traj)
        
        q_traj_in_phase_d = current_traj.get_q_trajectory( # augment_control_points_fn
            control_points, self.global_start, self.global_goal, get_type=("pos", "vel", "acc"), get_time_representation=False
        ) 
        q_traj_pos_in_phase = q_traj_in_phase_d["pos"]
        q_traj_vel_in_phase = q_traj_in_phase_d["vel"]
        q_traj_acc_in_phase = q_traj_in_phase_d["acc"]

        # Compute forward kinematics and spatial (world) jacobians
        assert q_traj_pos_in_phase.ndim == 3
        q_traj_pos_in_phase_original_shape = q_traj_pos_in_phase.shape
        q_traj_pos_aux = einops.rearrange(q_traj_pos_in_phase, "... d -> (...) d")

        with TimerCUDA() as t_fk_jac:
            # collision links and jacobians
            jacs_spatial, link_poses = self.robot.jfk_s_collision_spheres(q_traj_pos_aux)
            jacs_spatial_th = torch.stack(jacs_spatial).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th = einops.rearrange(
                jacs_spatial_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th = torch.stack(link_poses).transpose(0, 1)  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th = einops.rearrange(
                link_poses_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

            # end effector links and jacobians
            jacs_spatial_ee, link_poses_ee = self.robot.jfk_s_ee(q_traj_pos_aux)
            jacs_spatial_th_ee = torch.stack(jacs_spatial_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th_ee = einops.rearrange(
                jacs_spatial_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th_ee = torch.stack(link_poses_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th_ee = einops.rearrange(
                link_poses_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

        if self.debug:
            print(f"FK and Jacobians (time): {t_fk_jac.elapsed:.4f} s")
            print("-" * 50)

        # Compute cost and gradients wrt to the control points normalized
        with TimerCUDA() as t_cost_grad_all:
            cost_all = 0.0
            grad_costs_wrt_cp_normalized_l = []
            #rs_inv = self.parametric_trajectory.phase_time.rs_inv
            #s = self.parametric_trajectory.phase_time.s
            rs_inv = current_traj.phase_time.rs_inv
            s = current_traj.phase_time.s

            for k, cost_key in enumerate(self.costs):
                s_time = time.perf_counter()

                cost_fn = self.costs[cost_key].cost
                weight = self.costs[cost_key].weight

                timesteps = current_traj.phase_time.t + offset_info['t_offset']

                assert control_points_normalized.shape[0] == control_points.shape[0]

                cost_single_in_phase, grad_cost_single_wrt_cp_normalized_in_phase = (
                    self.compute_cost_grad_cp_normalized(
                        cost_fn,
                        control_points_normalized,
                        control_points,
                        q_traj_pos_in_phase,
                        q_traj_vel_in_phase,
                        q_traj_acc_in_phase,
                        link_poses_th,
                        jacs_spatial_th,
                        link_poses_th_ee,
                        jacs_spatial_th_ee,
                        current_traj=current_traj,
                        # compose
                        timesteps=timesteps,
                        **kwargs,
                    )
                )

                if idx==self.n_comp -1 and self.dataset.context_ee_goal_pose and cost_key != "CostTaskSpaceEEGoalPose":
                    # If the EE pose goal context is set, the generative model determines the last joint position.
                    # Hence, we zero the gradient of the last control point for all costs except the EE pose goal cost,
                    # to avoid changing the last joint position.
                    # Note that the penultimate control point of the B-spline still affects the last portion of
                    # the trajectory, which means it can remove it from collisions, even though the last control point
                    # is fixed.
                    grad_cost_single_wrt_cp_normalized_in_phase[..., -1, :] = 0.0

                if idx == self.n_comp - 1 and self.dataset.context_ee_goal_pose and cost_key == "CostTaskSpaceEEGoalPose":
                    # The CostTaskSpaceEEGoalPose cost is defined only on the last point of the trajectory,
                    # so we use directly the cost and gradient at the last point, without integration.
                    cost_single = cost_single_in_phase[..., -1]
                    grad_cost_single_wrt_cp_normalized = grad_cost_single_wrt_cp_normalized_in_phase[:, -1, ...]
                else:
                    # Approximate integral in eq. 28 -- https://arxiv.org/pdf/2412.19948
                    cost_single = torch.trapezoid(
                        cost_single_in_phase * rs_inv,
                        s,
                        dim=-1,
                    )

                    grad_cost_single_wrt_cp_normalized = torch.trapezoid(
                        grad_cost_single_wrt_cp_normalized_in_phase * rs_inv[None, :, None, None],
                        s,
                        dim=-3,
                    )

                cost_single_weighted = weight * cost_single
                cost_all += cost_single_weighted 
                grad_cost_single_wrt_cp_normalized_weighted = weight * grad_cost_single_wrt_cp_normalized
                grad_costs_wrt_cp_normalized_l.append(grad_cost_single_wrt_cp_normalized_weighted)

                if self.debug:
                    print(f"{cost_key} (cost): {cost_single_weighted.mean():.4f} +- {cost_single_weighted.std():.4f}")
                    grad_cost_all_wrt_cp_normalized_norm_weighted = torch.linalg.norm(
                        grad_cost_single_wrt_cp_normalized_weighted, dim=-1
                    )
                    print(
                        f"{cost_key} (grad norm):"
                        f" {grad_cost_all_wrt_cp_normalized_norm_weighted.mean():.4f}"
                        f" +- {grad_cost_all_wrt_cp_normalized_norm_weighted.std():.4f}"
                    )
                    print(f"{cost_key} (time): {time.perf_counter() - s_time:.4f} s")
                    print(f"--------------------------------")

        if self.debug:
            print(f"Costs and gradients (time): {t_cost_grad_all.elapsed:.4f} s")

        # Project gradients respecting hierarchy
        if self.args_inference.project_gradient_hierarchy:
            with TimerCUDA() as t_project_gradients:
                grad_costs_all_wrt_cp_normalized, grad_costs_all_wrt_cp_normalized_projected_l = (
                    project_hierarchical_gradients_fast(grad_costs_wrt_cp_normalized_l)
                )
            if self.debug:
                print(f"Project gradients (time): {t_project_gradients.elapsed:.4f} s")
        else:
            grad_costs_all_wrt_cp_normalized = torch.stack(grad_costs_wrt_cp_normalized_l).sum(dim=0)

        # -1 because the denoising gradient methods expect an objective function to maximize, but we want to minimize
        # the cost
        grad_costs_all_wrt_cp_normalized = -1.0 * grad_costs_all_wrt_cp_normalized

        # scatter plot of gradients in 2D
        if plot_gradients and self.debug and control_points_normalized.shape[-1] == 2:
            import matplotlib.pyplot as plt

            fig, ax = create_fig_and_axes(self.planning_task.env.dim)
            ax.scatter(to_numpy(control_points)[..., 0], to_numpy(control_points[..., 1]))
            self.planning_task.env.render(ax)
            self.planning_task.robot.render_trajectories(
                ax, q_traj_pos_in_phase, plot_points_scatter=False, control_points=control_points
            )

            grad_costs_wrt_cp_normalized_l += [-1.0 * grad_costs_all_wrt_cp_normalized]
            colors = ["g", "y", "c", "m", "k"]
            for k, grad in enumerate(grad_costs_wrt_cp_normalized_l):
                grad *= -1  # flip the gradient direction
                ax.quiver(
                    to_numpy(control_points[..., 0]),
                    to_numpy(control_points[..., 1]),
                    to_numpy(grad[..., 0]),
                    to_numpy(grad[..., 1]),
                    color=colors[k % len(colors)] if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "blue",
                    label=f"{list(self.costs.keys())[k]}" if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "Total",
                    scale=25,
                    width=0.005,
                )
            ax.legend()
            plt.show()

        # Increment step counter
        if not warmup:
            self.step_guide_call += 1
        if return_cost:
            return cost_all, grad_costs_all_wrt_cp_normalized
        return grad_costs_all_wrt_cp_normalized

    def compute_cost_grad_cp_normalized(
        self,
        cost_fn,
        control_points_normalized,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        current_traj=None,
        **kwargs,
    ):
        # compute cost gradients wrt to the control points normalized
        # The cost C can be a function of the trajectory q(s), the control points cp, or the task space x.
        # dC/dcp_norm = dC/dx * dx/dq * dq/dcp * dcp/dcp_normalized

        # We compute the gradient of the cost wrt to the joint space q
        # For TaskSpace costs, we compute dC/dq = dC/dx * dx/dq * dq/dcp
        # For JointSpace costs, we compute dC/dq = dC/dq * dq/dcp
        # pdb.set_trace()
        print(f"before cost fn : {control_points.shape=},")
        cost_value_in_phase, grad_cost_wrt_cp_in_phase = cost_fn.compute_cost_grad_wrt_cp( # 10, 128, 17, 2
            control_points, # [10, 16, 2]
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            link_poses_th,
            jacs_spatial_th,
            link_poses_th_ee,
            jacs_spatial_th_ee,
            **kwargs,
        )
        print(f"after cost fn :  {cost_value_in_phase.shape=} {grad_cost_wrt_cp_in_phase.shape=}")
        # Gradient of the control points wrt to the control points normalized
        # dcp/dcp_norm
        grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(control_points_normalized) # [10, 16, 2]
        # pdb.set_trace()
        # Align shape if the current local trajectory removed a boundary control point
        n_cp_cost = grad_cost_wrt_cp_in_phase.shape[-2]
        n_cp_norm = grad_cp_wrt_cp_normalized.shape[-2]
        assert n_cp_cost == n_cp_norm
        # if n_cp_cost != n_cp_norm:
        #     diff = n_cp_norm - n_cp_cost
        #     if diff == 1 and current_traj is not None:
        #         drop_start = getattr(current_traj, "remove_from_start_control_points", None)
        #         if drop_start is True:
        #             grad_cp_wrt_cp_normalized = grad_cp_wrt_cp_normalized[..., 1:, :]
        #         elif drop_start is False:
        #             grad_cp_wrt_cp_normalized = grad_cp_wrt_cp_normalized[..., :-1, :]
        #         else:
        #             grad_cp_wrt_cp_normalized = grad_cp_wrt_cp_normalized[..., :-1, :]
        #     elif diff == 2 and getattr(current_traj, "remove_outer_control_points", False):
        #         grad_cp_wrt_cp_normalized = grad_cp_wrt_cp_normalized[..., 1:-1, :]
        #     elif diff == -1 and current_traj is not None:
        #         drop_start = getattr(current_traj, "remove_from_start_control_points", None)
        #         if drop_start is True:
        #             grad_cost_wrt_cp_in_phase = grad_cost_wrt_cp_in_phase[..., 1:, :]
        #         elif drop_start is False:
        #             grad_cost_wrt_cp_in_phase = grad_cost_wrt_cp_in_phase[..., :-1, :]
        #         else:
        #             grad_cost_wrt_cp_in_phase = grad_cost_wrt_cp_in_phase[..., :-1, :]
        #     elif diff == -2 and getattr(current_traj, "remove_outer_control_points", False):
        #         grad_cost_wrt_cp_in_phase = grad_cost_wrt_cp_in_phase[..., 1:-1, :]

        # assert (
        #     grad_cp_wrt_cp_normalized.shape[-2] == grad_cost_wrt_cp_in_phase.shape[-2]
        # ), f"Control point gradient shape mismatch after alignment: cost_grad {grad_cost_wrt_cp_in_phase.shape} vs norm_grad {grad_cp_wrt_cp_normalized.shape}"

        # Gradient of the cost wrt to the control points normalized
        # dC/dcp_norm = dC/dcp * dcp/dcp_norm
        # In matrix form -- Hadamard product (the normalization is done element-wise)
        grad_cost_wrt_cp_normalized_per_shape_step = torch.einsum(
            "...jkn,...kn->...jkn", grad_cost_wrt_cp_in_phase, grad_cp_wrt_cp_normalized
        )

        return cost_value_in_phase, grad_cost_wrt_cp_normalized_per_shape_step

    def warmup(self, shape_x, **kwargs):
        # assert len(shape_x) == 3
        # b,h,d  = shape_x
        # current_traj, _ = self.get_parametric_trajectory(0)
        # x = torch.randn((b,current_traj.n_control_points,d), **self.tensor_args)
        # print(x.shape)
        x = torch.randn(shape_x, **self.tensor_args)
        idx = torch.randint(0, self.n_comp, (1,))[0].item()
        print("warmup ",idx)
        self.__call__(x, idx=idx, warmup=True)
