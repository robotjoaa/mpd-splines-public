"""
Composable diffusion model.

This mirrors the compositional trajectory sampling flow from the comp_diffuser
codepath, but reuses the existing GaussianDiffusionModel (classifier-guided)
infrastructure. The intent is to:
- sample multiple overlapping trajectory segments with the base diffusion model
  (one call per segment, using standard hard_conds + classifier guidance),
  rather than writing a bespoke CFG denoiser; and
- stitch those segments together with a simple overlap blending routine.

Some pieces (e.g., automatic creation of per-segment hard_conds, richer
overlap-aware conditioning) are left as TODOs to be filled in once the desired
API is finalized.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mpd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from mpd.models.diffusion_models.sample_functions import apply_hard_conditioning


class CompDiffusionModel(GaussianDiffusionModel):
    """
    A compositional variant of GaussianDiffusionModel.

    - Uses the existing classifier-guidance aware sampling loops from the base
      class (ddpm/ddim) to generate individual trajectory segments.
    - Composes `n_components` segments with a fixed-length overlap and blends
      those overlaps into a single long trajectory.

    Notes:
    * The current implementation expects the caller to prepare a list of
      `hard_conds` dicts, one per component. This keeps us aligned with the
      existing API and avoids introducing a new CFG-style conditioner. A higher
      level wrapper (e.g., a policy/planner) can construct those dicts using
      start/goal/overlap context similar to comp_diffuser.
    * Overlap blending is adapted from `comp_diffuser/traj_blender.py`, but kept
      lightweight and numpy-based here.
    """

    def __init__(
        self,
        *args,
        overlap_len: int,
        blend_type: str = "exponential",
        blend_beta: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.len_overlap = overlap_len
        self.hzn_step_size = self.n_diffusion_steps  # TODO: revisit; placeholder for API symmetry
        self.blend_type = blend_type
        self.blend_beta = blend_beta

    # ---------------------------------------------------------------------- #
    # Public API                                                            #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def comp_sample(
        self,
        hard_conds_per_comp: Sequence[Dict[int, torch.Tensor]],
        horizon: Optional[int] = None,
        blend: bool = True,
        method: str = "ddpm",
        **sample_kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[np.ndarray]]:
        """
        Run the base diffusion sampler for each component, then optionally blend.

        Args:
            hard_conds_per_comp: list of hard_conds dicts, one per component.
                Each entry is fed to `conditional_sample` directly. The caller is
                responsible for ensuring start/end alignment across components.
            horizon: optional override per-component horizon.
            blend: whether to merge the segments into a single long trajectory.
            method/sample_kwargs: forwarded to `conditional_sample`.

        Returns:
            - List of per-component trajectories (torch tensors shaped
              [B x horizon x state_dim]).
            - If blend=True, a numpy array with the stitched trajectory
              [B x total_horizon x state_dim]; otherwise None.
        """
        trajectories: List[torch.Tensor] = []

        for hard_conds in hard_conds_per_comp:
            traj = self.conditional_sample(
                hard_conds=hard_conds,
                horizon=horizon,
                method=method,
                **sample_kwargs,
            )
            # Defensive: ensure conditioning is applied on the returned sample.
            traj = apply_hard_conditioning(traj, hard_conds)
            trajectories.append(traj)

        blended = None
        if blend and len(trajectories) > 1:
            blended = self._blend_components(trajectories)

        return trajectories, blended

    # ---------------------------------------------------------------------- #
    # Helpers                                                               #
    # ---------------------------------------------------------------------- #
    def _to_numpy(self, traj: torch.Tensor) -> np.ndarray:
        return traj.detach().cpu().numpy()

    def _blend_components(self, trajs: List[torch.Tensor]) -> np.ndarray:
        """
        Blend a list of component trajectories with fixed-length overlap.
        """
        trajs_np = [self._to_numpy(t) for t in trajs]
        b_s, h, d = trajs_np[0].shape
        n_comp = len(trajs_np)

        assert h > self.len_overlap, "Component horizon must exceed overlap length."
        gap_len = h - 2 * self.len_overlap
        assert gap_len > 0, "Overlap too large relative to component horizon."

        total_hzn = n_comp * h - (n_comp - 1) * self.len_overlap
        trajs_out = np.zeros((b_s, total_hzn, d), dtype=np.float32)
        cnt_v = np.zeros_like(trajs_out)

        # Copy non-overlap regions.
        for i_c, comp in enumerate(trajs_np):
            if i_c == 0:
                idx1, idx2 = 0, h - self.len_overlap
                trajs_out[:, idx1:idx2, :] = comp[:, : idx2 - idx1, :]
            elif i_c < n_comp - 1:
                idx1 = h + (i_c - 1) * (h - self.len_overlap)
                idx2 = idx1 + gap_len
                trajs_out[:, idx1:idx2, :] = comp[:, self.len_overlap : self.len_overlap + gap_len, :]
            else:
                idx1 = h + (i_c - 1) * (h - self.len_overlap)
                idx2 = idx1 + (h - self.len_overlap)
                trajs_out[:, idx1:idx2, :] = comp[:, self.len_overlap :, :]
            cnt_v[:, idx1:idx2, :] += 1

        # Blend overlaps.
        for i_c in range(n_comp - 1):
            idx1 = (i_c + 1) * (h - self.len_overlap)
            idx2 = idx1 + self.len_overlap

            _, end_traj = self._extract_overlap(trajs_np[i_c])
            start_traj, _ = self._extract_overlap(trajs_np[i_c + 1])
            blended = self._blend_two_trajs(end_traj, start_traj, self.blend_type, self.blend_beta)
            trajs_out[:, idx1:idx2, :] = blended
            cnt_v[:, idx1:idx2, :] += 1

        assert (cnt_v == 1).all(), "Overlap blending produced inconsistent counts."
        return trajs_out

    def _extract_overlap(self, traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return traj[:, : self.len_overlap, :], traj[:, -self.len_overlap :, :]

    def _blend_two_trajs(self, traj_1: np.ndarray, traj_2: np.ndarray, blend_type: str, beta: float):
        """
        Blend the overlapping windows of two trajectories. Adapted from
        comp_diffuser/blender; kept numpy-only to avoid torch graph pollution.
        """
        assert traj_1.shape == traj_2.shape
        len_tj = traj_1.shape[1]
        t_overlap_start = 0
        t_overlap_end = len_tj - 1
        t_overlap = np.arange(0, len_tj)

        if blend_type in ["exponential", "exp"]:
            def w(t):
                exponent = -beta * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
                return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
        elif blend_type == "cosine":
            def w(t):
                return 0.5 * (1 + np.cos(np.pi * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)))
        elif blend_type == "linear":
            def w(t):
                return 1 - (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
        elif blend_type == "smoothstep":
            def w(t):
                x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
                return 1 - (3 * x ** 2 - 2 * x ** 3)
        else:
            raise ValueError(f"Invalid blend_type {blend_type}")

        weights = w(t_overlap)[None, :, None]  # (1, len_tj, 1) for broadcasting
        return weights * traj_1 + (1 - weights) * traj_2

class CompDiffusionPlanner:
    """
    Utility to turn environment problems into per-component hard conditions.

    The planner mirrors the logic of the comp_diffuser planning stack but
    reuses the classifier-guided sampler exposed by ``CompDiffusionModel``.
    It interpolates between start/goal states, adds overlap hints, and
    produces the ``hard_conds_per_comp`` list consumed by ``comp_sample``.
    """

    def __init__(
        self,
        diffusion_model: CompDiffusionModel,
        component_horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.diffusion_model = diffusion_model
        self.len_overlap = diffusion_model.len_overlap
        self.device = device or diffusion_model.betas.device
        self.dtype = diffusion_model.betas.dtype
        self.state_dim = diffusion_model.state_dim
        self.component_horizon = component_horizon or getattr(diffusion_model, "horizon", None)
        if self.component_horizon is None:
            raise ValueError("component_horizon must be provided when the diffusion model has no horizon attr")

    # ------------------------------------------------------------------ #
    def build_hard_conds_per_comp(
        self,
        start_state: Union[np.ndarray, torch.Tensor],
        goal_state: Union[np.ndarray, torch.Tensor],
        num_components: int,
        overlap_hints: bool = True,
        horizon: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[int, torch.Tensor]]:
        """Create a list of hard_conds dicts (one per component)."""

        if num_components < 1:
            raise ValueError("num_components must be at least 1")

        horizon = horizon or self.component_horizon
        step_size = horizon - self.len_overlap
        if step_size <= 0:
            raise ValueError("overlap must be smaller than the component horizon")

        start = self._prepare_state(start_state, batch_size)
        goal = self._prepare_state(goal_state, batch_size or start.shape[0])
        if start.shape[0] != goal.shape[0]:
            if start.shape[0] == 1:
                start = start.expand(goal.shape[0], -1)
            elif goal.shape[0] == 1:
                goal = goal.expand(start.shape[0], -1)
            else:
                raise ValueError("start_state and goal_state batch dims are incompatible")

        total_horizon = num_components * horizon - (num_components - 1) * self.len_overlap
        anchors = self._interpolate(start, goal, total_horizon)

        hard_conds_per_comp: List[Dict[int, torch.Tensor]] = []
        for comp_idx in range(num_components):
            start_idx = comp_idx * step_size
            segment = anchors[:, start_idx : start_idx + horizon, :]

            conds: Dict[int, torch.Tensor] = {
                0: segment[:, 0, :],
                horizon - 1: segment[:, -1, :],
            }

            if overlap_hints and self.len_overlap > 0:
                prefix = segment[:, : self.len_overlap, :]
                suffix = segment[:, -self.len_overlap :, :]
                for idx in range(self.len_overlap):
                    conds[idx] = prefix[:, idx, :]
                offset = horizon - self.len_overlap
                for idx in range(self.len_overlap):
                    conds[offset + idx] = suffix[:, idx, :]

            hard_conds_per_comp.append(conds)

        return hard_conds_per_comp

    def plan(
        self,
        problem: Union[Dict[str, Any], Tuple[Any, Any]],
        num_components: int,
        blend: bool = True,
        overlap_hints: bool = True,
        horizon: Optional[int] = None,
        method: str = "ddpm",
        **sample_kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[np.ndarray]]:
        """Plan a trajectory by constructing hard_conds and calling comp_sample."""

        start_state, goal_state = self._extract_problem_states(problem)
        hard_conds_per_comp = self.build_hard_conds_per_comp(
            start_state=start_state,
            goal_state=goal_state,
            num_components=num_components,
            overlap_hints=overlap_hints,
            horizon=horizon,
        )

        if "batch_size" not in sample_kwargs:
            first_comp_conds = hard_conds_per_comp[0]
            sample_kwargs["batch_size"] = next(iter(first_comp_conds.values())).shape[0]

        return self.diffusion_model.comp_sample(
            hard_conds_per_comp,
            horizon=horizon or self.component_horizon,
            blend=blend,
            method=method,
            **sample_kwargs,
        )

    # ------------------------------------------------------------------ #
    def _prepare_state(
        self, state: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        tensor = state if torch.is_tensor(state) else torch.as_tensor(state)
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] != self.state_dim:
            tensor = tensor.reshape(tensor.shape[0], self.state_dim)
        if batch_size is not None and tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor

    def _interpolate(self, start: torch.Tensor, goal: torch.Tensor, total_horizon: int) -> torch.Tensor:
        alphas = torch.linspace(0.0, 1.0, total_horizon, device=self.device, dtype=self.dtype)
        alphas = alphas.view(1, -1, 1)
        start_exp = start[:, None, :]
        goal_exp = goal[:, None, :]
        return (1.0 - alphas) * start_exp + alphas * goal_exp

    def _extract_problem_states(self, problem: Union[Dict[str, Any], Tuple[Any, Any]]):
        if isinstance(problem, tuple) and len(problem) == 2:
            return problem

        if not isinstance(problem, dict):
            raise TypeError("problem must be a dict or (start, goal) tuple")

        start = problem.get("start_state") or problem.get("start") or problem.get("start_pos")
        goal = problem.get("goal_state") or problem.get("goal") or problem.get("goal_pos")
        if start is None or goal is None:
            raise KeyError("problem dict must provide start and goal states")
        return start, goal
