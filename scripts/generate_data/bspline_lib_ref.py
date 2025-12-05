import torch
import numpy as np

def merge_splines_with_overlap(
    final_traj,
    left_traj,
    right_traj,
    overlap=6,
    K=3,
    j0=None,
):
    """
    Blend left_traj and right_traj into final_traj using an overlap band
    of 'overlap' control points in final_traj, enforcing C² continuity
    (pos/vel/acc) at K sample points inside the overlap (for each side),
    solved in least-squares sense.

    Assumptions:
        - degree = 5 (support width = 6) and overlap == 6 recommended.
        - left_traj, right_traj, final_traj all use knots in [0,1].
        - left/right have same n_control_points and degree.
        - final_traj has n_control_points = n_left + n_right - overlap.
        - final_traj.control_points initially contain:
              [0..n_sml-1]   = left control points
              [n_sml..end]   = right control points (with overlap already
                                structurally arranged, you will overwrite
                                the overlap band).
    Args:
        final_traj : ParametricTrajectoryBspline for merged spline
        left_traj  : ParametricTrajectoryBspline for left spline
        right_traj : ParametricTrajectoryBspline for right spline
        overlap    : number of overlapping / blending control points
        K          : number of continuity samples per side
        j0         : starting index of overlap band in final_traj.control_points.
                     If None, defaults to n_left - overlap.

    Returns:
        merged_cps : (n_merged, dim) tensor of merged control points
    """

    device = final_traj.control_points.device
    dtype  = final_traj.control_points.dtype
    tensor_args = {"device": device, "dtype": dtype}

    # --- Basic objects and sizes ---
    final_spl = final_traj.bspline
    sml_spl_L = left_traj.bspline
    sml_spl_R = right_traj.bspline

    deg = final_traj.degree
    assert deg == 5, "This function is currently specialized for degree 5."

    # left / right cps
    l_cps = left_traj.control_points.to(**tensor_args)   # [n_sml, dim]
    r_cps = right_traj.control_points.to(**tensor_args)
    n_sml, dim = l_cps.shape

    # merged cps (initial guess from final_traj)
    merged_cps = final_traj.control_points.to(**tensor_args)  # [n_merged, dim]
    n_merged = merged_cps.shape[0]

    # default overlap start index
    if j0 is None:
        j0 = n_sml - overlap          # e.g. 16 - 6 = 10

    assert j0 >= 0 and j0 + overlap <= n_merged

    unknown_idx = torch.arange(j0, j0 + overlap, device=device, dtype=torch.long)

    # --- span counts (just in case you need them later) ---
    spans_left  = n_sml    - deg
    spans_right = n_sml    - deg
    spans_merge = n_merged - deg

    # --- phase-time scaling (for vel/acc conversion) ---
    sml_scale_L = left_traj.phase_time.rs[0]
    sml_scale_R = right_traj.phase_time.rs[0]
    merge_scale = final_traj.phase_time.rs[0]

    # For simplicity assume left/right use same phase-time scale:
    assert torch.allclose(
        sml_scale_L, sml_scale_R
    ), "Left and right phase_time.rs[0] differ; handle separately if needed."
    sml_scale = sml_scale_L

    # --- reference jets (in phase domain) ---
    # q_*: dict("pos": [T_s, dim], "vel": [T_s, dim], "acc": [T_s, dim])
    qL = left_traj.get_q_trajectory_in_phase(
        l_cps, get_type=("pos", "vel", "acc")
    )
    qR = right_traj.get_q_trajectory_in_phase(
        r_cps, get_type=("pos", "vel", "acc")
    )

    # --- time grids in phase space of small and merged trajs ---
    Tm = np.linspace(0.0, 1.0, final_traj.num_T_pts)
    Ts = np.linspace(0.0, 1.0, left_traj.num_T_pts)  # same for right

    # --- local overlap support in merged spline ---
    # Overlap control points j0..j0+overlap-1 have support in approx [u[j0], u[j0+overlap]]
    u = final_spl.u.cpu().numpy()
    t0 = u[j0]                  # start of overlap in merged param
    t1 = u[j0 + overlap]        # end   of overlap in merged param

    # Choose K interior samples in [t0, t1]
    ts = np.linspace(t0, t1, K + 2)[1:-1]   # (K points, skipping endpoints)

    A_rows = []
    B_rows = []

    def add_constraints_at_sample(t_m, target_dict):
        """
        Add C, C', C'' constraints at merged parameter t_m
        for one "side" (left or right), using target_dict with keys
        'pos', 'vel', 'acc' already scaled to merged-time units.
        """
        # 1) find merged knot span index
        k = int(np.searchsorted(u, t_m, side="right") - 1)
        k = max(deg, min(k, len(u) - deg - 2))  # clamp safely

        # 2) local support control-point indices
        idx_block = torch.arange(k - deg, k + 1, device=device)  # 6 indices

        # 3) unknown vs fixed mask
        mask_block = torch.isin(idx_block, unknown_idx)
        idx_unknown = idx_block[mask_block]
        idx_fixed   = idx_block[~mask_block]

        # 4) find nearest merged time-sample index
        t_idx = int(np.searchsorted(Tm, t_m))

        for deriv_level, key in zip([0, 1, 2], ["pos", "vel", "acc"]):
            # basis coefficients at this time & derivative order
            if deriv_level == 0:
                coeffs = final_spl.N[0, t_idx, idx_block]    # [6]
            elif deriv_level == 1:
                coeffs = final_spl.dN[0, t_idx, idx_block]
            else:
                coeffs = final_spl.ddN[0, t_idx, idx_block]

            # fixed contribution
            if idx_fixed.numel() > 0:
                fc = (coeffs[~mask_block].unsqueeze(1) *
                      merged_cps[idx_fixed]).sum(dim=0)       # [dim]
            else:
                fc = torch.zeros(dim, **tensor_args)

            # A row: contributions from unknown CPs only
            A_row = torch.zeros(overlap, **tensor_args)
            if idx_unknown.numel() > 0:
                cols = (idx_unknown - j0).long()              # map to [0..overlap-1]
                A_row[cols] = coeffs[mask_block]

            # b row: target - fixed contribution
            target = target_dict[key]                         # [dim]
            B_row = target - fc

            A_rows.append(A_row)
            B_rows.append(B_row)

    # --- build constraints for left & right sides using local u mapping ---
    for t_m in ts:
        # Local coordinate u in [0,1] inside overlap window
        u_loc = (t_m - t0) / (t1 - t0)

        # Left should use near END of its domain: t_L = 1 - u_loc
        t_L = 1.0 - u_loc
        # find nearest sample in left phase grid
        sl_idx = int(np.abs(Ts - t_L).argmin())

        # jets scaled from small-phase to merged-time domain
        target_l = {
            "pos": qL["pos"][sl_idx].to(**tensor_args),
            "vel": qL["vel"][sl_idx].to(**tensor_args) * (sml_scale / merge_scale),
            "acc": qL["acc"][sl_idx].to(**tensor_args) * (sml_scale / merge_scale) ** 2,
        }

        add_constraints_at_sample(t_m, target_l)

    for t_m in ts:
        u_loc = (t_m - t0) / (t1 - t0)

        # Right should use near START of its domain: t_R = u_loc
        t_R = u_loc
        sr_idx = int(np.abs(Ts - t_R).argmin())

        target_r = {
            "pos": qR["pos"][sr_idx].to(**tensor_args),
            "vel": qR["vel"][sr_idx].to(**tensor_args) * (sml_scale / merge_scale),
            "acc": qR["acc"][sr_idx].to(**tensor_args) * (sml_scale / merge_scale) ** 2,
        }

        add_constraints_at_sample(t_m, target_r)

    # --- stack A and b ---
    A_big = torch.stack(A_rows, dim=0)   # [M, overlap]
    B_big = torch.stack(B_rows, dim=0)   # [M, dim]

    # --- least-squares solve ---
    lsq = torch.linalg.lstsq(A_big, B_big)
    x_overlap = lsq.solution   # [overlap, dim]

    # --- write back into merged cps ---
    merged_cps = merged_cps.clone()
    merged_cps[unknown_idx] = x_overlap

    # update final_traj if there's a setter
    if hasattr(final_traj, "set_control_points"):
        final_traj.set_control_points(merged_cps)

    return merged_cps
