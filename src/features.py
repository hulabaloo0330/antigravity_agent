"""Strictly causal pre-stimulus feature extraction, temporal basis kernels, and circuit state summaries.

Enforces `MODEL_REQUIREMENTS.md` §1, §3, §4, §7:
- All inputs computed strictly on frames `< onset_idx` in physical seconds (`fps`-invariant).
- Fold-internal neuron scaling `s_n` and fold-internal `Neuron x Stimulus` empirical-Bayes priors
  (fixing Historical Trap #4).
- Explicit exclusion of target neuron `i` and its bilateral homolog `i_LR` from all cross-neuron
  network summaries (`H1`), with configurable preservation of `ASEL/ASER` and `AWCON/AWCOFF` asymmetries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


LAYER1_INTEGRATORS = ("AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR", "RIAL", "RIAR")
BACKWARD_COMMAND = ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER", "RIML", "RIMR")
FORWARD_COMMAND = ("AVBL", "AVBR", "PVCL", "PVCR", "RIBL", "RIBR", "RID")
ASYMMETRIC_CHEMOSENSORY = ("ASEL", "ASER", "AWCON", "AWCOFF")
CORE_AMPHID_SENSORY = (
    "ASEL", "ASER", "AWCON", "AWCOFF", "AWCL", "AWCR",
    "ASHL", "ASHR", "AWAL", "AWAR", "AWBL", "AWBR",
    "ADFL", "ADFR", "ADLL", "ADLR", "ASGL", "ASGR",
    "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
)


def get_bilateral_exclusion_set(
    neuron_name: str,
    canonical_neurons: Sequence[str],
    exclude_bilateral: bool = True,
    preserve_sensory_asymmetry: bool = False,
) -> List[int]:
    """Return canonical indices to exclude when summarizing network state for target `neuron_name`."""
    idx_map = {n: i for i, n in enumerate(canonical_neurons)}
    excl: Set[str] = {neuron_name}
    # Always exclude exact anatomical aliases for AWC so AWCL never leaks into AWCON/AWCOFF
    if neuron_name in ("AWCON", "AWCR"):
        excl.update({"AWCON", "AWCR"})
    elif neuron_name in ("AWCOFF", "AWCL"):
        excl.update({"AWCOFF", "AWCL"})

    if exclude_bilateral:
        if neuron_name in ("AWCON", "AWCOFF", "AWCL", "AWCR"):
            if not preserve_sensory_asymmetry:
                excl.update({"AWCON", "AWCOFF", "AWCL", "AWCR"})
        elif neuron_name in ("ASEL", "ASER"):
            if not preserve_sensory_asymmetry:
                excl.update({"ASEL", "ASER"})
        elif neuron_name not in ("AQR", "RIR", "PQR") and len(neuron_name) >= 4 and neuron_name[-1] in ("L", "R"):
            partner = neuron_name[:-1] + ("R" if neuron_name[-1] == "L" else "L")
            excl.add(partner)

    return sorted(idx_map[n] for n in excl if n in idx_map)


def build_post_temporal_basis(
    n_out: int,
    fps: float,
    u_post: np.ndarray,
    taus_sec: Sequence[float] = (0.75, 2.0, 5.0, 10.0),
) -> np.ndarray:
    """Build orthonormal post-stimulus temporal basis `Phi` `[T_out, B]` from physical time & `u_post`.

    Captures immediate step, fast/medium/slow exponential rise & post-offset decay, plus biphasic adaptation.
    """
    dt = 1.0 / max(float(fps), 1e-6)
    u = np.asarray(u_post, dtype=np.float64).ravel()
    if u.size != n_out:
        u = np.ones(n_out, dtype=np.float64)

    cols: List[np.ndarray] = [np.ones(n_out, dtype=np.float64)]
    exp_responses: List[np.ndarray] = []
    for tau in taus_sec:
        alpha = 1.0 - np.exp(-dt / max(float(tau), 1e-3))
        r = np.zeros(n_out, dtype=np.float64)
        state = 0.0
        for t in range(n_out):
            state = (1.0 - alpha) * state + alpha * u[t]
            r[t] = state
        peak = np.max(np.abs(r))
        if peak > 1e-8:
            r = r / peak
        cols.append(r)
        exp_responses.append(r)

    if len(exp_responses) >= 2:
        # Biphasic transient (fast rise minus slow adaptation)
        cols.append(exp_responses[0] - exp_responses[-1])

    raw_mat = np.column_stack(cols)
    # Orthonormalize columns via economy QR with deterministic sign convention
    q, r_mat = np.linalg.qr(raw_mat, mode="reduced")
    signs = np.sign(np.diag(r_mat))
    signs[signs == 0] = 1.0
    q = q * signs[None, :]
    return q.astype(np.float32)


def fit_fold_neuron_stim_priors(
    train_trials: Sequence[Mapping[str, Any]],
    n_canon: int,
    taus_sec: Sequence[float] = (0.75, 2.0, 5.0, 10.0),
    shrinkage_counts: float = 5.0,
) -> Dict[str, Any]:
    """Fit training-fold-only neuron baseline means and `Neuron x Stimulus` basis-projected priors.

    Expressing the training-fold mean response trajectory in the physical-second basis `Phi` ensures
    clean smoothing, exact frame-count alignment, and fold-internal consistency for Ridge and MLP!
    """
    # 1. Compute training-fold per-neuron baseline mean from causal pre-anchors
    pre_vals: List[List[float]] = [[] for _ in range(n_canon)]
    stim_vocab: Set[str] = set()

    # Basis-coefficient accumulators: (neuron, stim) and global (stim)
    b_dim: Optional[int] = None
    nc_sum: Dict[Tuple[int, str], np.ndarray] = {}
    nc_cnt: Dict[Tuple[int, str], float] = {}
    c_sum: Dict[str, np.ndarray] = {}
    c_cnt: Dict[str, float] = {}

    for tr in train_trials:
        if bool(tr.get("is_buffer_control", False)):
            continue
        stim = str(tr["stimulus_id"])
        stim_vocab.add(stim)
        n_out = tr["y_true"].shape[1]
        phi = build_post_temporal_basis(n_out, float(tr["fps"]), tr["u_post"], taus_sec=taus_sec)
        if b_dim is None:
            b_dim = phi.shape[1]

        vm = tr["valid_mask"]
        y_true = tr["y_true"]
        y_anc = tr["y_pre_anchor"]

        for n_i in np.where(tr["has_pre"])[0]:
            if np.isfinite(y_anc[n_i]):
                pre_vals[n_i].append(float(y_anc[n_i]))

        # Project valid neuron trajectories onto orthonormal Phi
        full_rows = np.where(np.all(vm, axis=1))[0]
        if full_rows.size > 0:
            dy = (y_true[full_rows] - y_anc[full_rows, None]).astype(np.float64)
            coeffs = dy @ phi  # [M_rows, B]
            c_sum[stim] = c_sum.get(stim, np.zeros(b_dim, dtype=np.float64)) + np.sum(coeffs, axis=0)
            c_cnt[stim] = c_cnt.get(stim, 0.0) + float(full_rows.size)
            for k_row, n_i in enumerate(full_rows):
                key = (int(n_i), stim)
                if key not in nc_sum:
                    nc_sum[key] = coeffs[k_row].copy()
                    nc_cnt[key] = 1.0
                else:
                    nc_sum[key] += coeffs[k_row]
                    nc_cnt[key] += 1.0

        # Handle partially observed post-stimulus rows via ridge-stabilized least squares
        part_rows = np.where(np.any(vm, axis=1) & (~np.all(vm, axis=1)))[0]
        for n_i in part_rows:
            rm = vm[n_i]
            if int(np.sum(rm)) >= 3:
                phi_sub = phi[rm].astype(np.float64)
                dy_sub = (y_true[n_i, rm] - y_anc[n_i]).astype(np.float64)
                coef, _, _, _ = np.linalg.lstsq(
                    phi_sub.T @ phi_sub + 1e-2 * np.eye(phi.shape[1]),
                    phi_sub.T @ dy_sub,
                    rcond=None,
                )
                key = (int(n_i), stim)
                nc_sum[key] = nc_sum.get(key, np.zeros(phi.shape[1], dtype=np.float64)) + coef
                nc_cnt[key] = nc_cnt.get(key, 0.0) + 1.0
                c_sum[stim] = c_sum.get(stim, np.zeros(phi.shape[1], dtype=np.float64)) + coef
                c_cnt[stim] = c_cnt.get(stim, 0.0) + 1.0

    if b_dim is None:
        b_dim = len(taus_sec) + 2

    mu_train = np.zeros(n_canon, dtype=np.float32)
    obs_counts = np.zeros(n_canon, dtype=np.int32)
    for n_i in range(n_canon):
        if pre_vals[n_i]:
            mu_train[n_i] = float(np.mean(pre_vals[n_i]))
            obs_counts[n_i] = len(pre_vals[n_i])

    stim_global_coef: Dict[str, np.ndarray] = {}
    for stim in sorted(stim_vocab):
        if c_cnt.get(stim, 0.0) > 0:
            stim_global_coef[stim] = (c_sum[stim] / c_cnt[stim]).astype(np.float32)
        else:
            stim_global_coef[stim] = np.zeros(b_dim, dtype=np.float32)

    neuron_stim_coef: Dict[Tuple[int, str], np.ndarray] = {}
    for (n_i, stim), s_vec in nc_sum.items():
        cnt_val = nc_cnt[(n_i, stim)]
        g_vec = stim_global_coef[stim].astype(np.float64)
        shrunk = (cnt_val * (s_vec / cnt_val) + shrinkage_counts * g_vec) / (cnt_val + shrinkage_counts)
        neuron_stim_coef[(n_i, stim)] = shrunk.astype(np.float32)

    return {
        "mu_train": mu_train,
        "obs_counts": obs_counts,
        "stim_vocab": sorted(stim_vocab),
        "stim_global_coef": stim_global_coef,
        "neuron_stim_coef": neuron_stim_coef,
        "b_dim": b_dim,
        "taus_sec": list(taus_sec),
    }


def predict_prior_delta_trajectory(
    trial: Mapping[str, Any],
    priors: Mapping[str, Any],
    n_canon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(prior_delta [N, T_out], prior_coef [N, B], phi [T_out, B])` for a trial."""
    n_out = trial["y_true"].shape[1]
    phi = build_post_temporal_basis(
        n_out, float(trial["fps"]), trial["u_post"], taus_sec=priors["taus_sec"]
    )
    b_dim = phi.shape[1]
    coef_mat = np.zeros((n_canon, b_dim), dtype=np.float32)

    if not bool(trial.get("is_buffer_control", False)):
        stim = str(trial["stimulus_id"])
        g_coef = priors["stim_global_coef"].get(stim, np.zeros(b_dim, dtype=np.float32))
        ns_map = priors["neuron_stim_coef"]
        for n_i in range(n_canon):
            coef_mat[n_i] = ns_map.get((n_i, stim), g_coef)

    prior_delta = coef_mat @ phi.T  # [N, T_out]
    return prior_delta, coef_mat, phi


def extract_self_history_features_matrix(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    lag_bins_sec: Sequence[Tuple[float, float]] = ((0.0, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 10.0)),
) -> np.ndarray:
    """Extract causal self-history feature matrix `[N_canon, D_self]` (`D_self = 10`) for all neurons in `trial`.

    Features per neuron `i` (all strictly `< onset_idx`, scaled by training-fold `s_n[i]`):
      0..3: Binned causal pre-stimulus history relative to `y_pre_anchor[i]` over `lag_bins_sec`
      4: Fast causal derivative `dx/dt` over `[-2.5, 0)s`
      5: Slow causal linear trend `dx/dt` over `[-10.0, 0)s`
      6: Within-worm calibration deviation `(y_pre_anchor[i] - mu_cal_worm[i]) / s_n[i]` (session mean-reversion)
      7: Cross-worm level deviation `(y_pre_anchor[i] - mu_train[i]) / s_n[i]` (population mean-reversion)
      8: Pre-stimulus fluctuation amplitude `std_pre[i] / s_n[i]`
      9: Pre-stimulus causal observation availability fraction in `[0, 1]`
    """
    x_pre = np.asarray(trial["x_pre"], dtype=np.float64)  # [T_in, N]
    m_pre = np.asarray(trial["m_pre"], dtype=bool)
    x_cal = np.asarray(trial["x_cal"], dtype=np.float64)
    m_cal = np.asarray(trial["m_cal"], dtype=bool)
    y_anc = np.asarray(trial["y_pre_anchor"], dtype=np.float64)
    scales = np.asarray(s_n, dtype=np.float64)
    fps = float(trial["fps"])
    t_in, n_canon = x_pre.shape

    d_self = len(lag_bins_sec) + 6
    feats = np.zeros((n_canon, d_self), dtype=np.float32)

    # Time before onset in seconds for each row of x_pre: t_back[k] > 0
    t_back = (np.arange(t_in, 0, -1, dtype=np.float64) - 0.5) / max(fps, 1e-6)

    for b_idx, (t_lo, t_hi) in enumerate(lag_bins_sec):
        win_rows = (t_back >= t_lo) & (t_back < t_hi)
        if not np.any(win_rows):
            win_rows[-1] = True
        sub_x = x_pre[win_rows]
        sub_m = m_pre[win_rows]
        cnt = np.sum(sub_m, axis=0)
        valid_n = cnt > 0
        if np.any(valid_n):
            means = np.sum(np.where(sub_m[:, valid_n], sub_x[:, valid_n], 0.0), axis=0) / cnt[valid_n]
            feats[valid_n, b_idx] = ((means - y_anc[valid_n]) / scales[valid_n]).astype(np.float32)

    base_col = len(lag_bins_sec)
    # Col base+0: Fast derivative ([-1,0)s minus [-2.5,-1)s)
    feats[:, base_col + 0] = feats[:, 0] - feats[:, min(1, base_col - 1)]
    # Col base+1: Slow linear trend ([-2.5,0)s minus [-10,-5)s)
    feats[:, base_col + 1] = 0.5 * (feats[:, 0] + feats[:, min(1, base_col - 1)]) - feats[:, base_col - 1]

    # Col base+2: Deviation from worm's causal [-30, -10)s calibration window
    if x_cal.shape[0] > 0:
        cal_cnt = np.sum(m_cal, axis=0)
        has_c = (cal_cnt > 0) & np.isfinite(y_anc)
        if np.any(has_c):
            cal_mu = np.sum(np.where(m_cal[:, has_c], x_cal[:, has_c], 0.0), axis=0) / cal_cnt[has_c]
            feats[has_c, base_col + 2] = ((y_anc[has_c] - cal_mu) / scales[has_c]).astype(np.float32)

    # Col base+3: Deviation from training-fold population neuron mean (clipped for outlier safety)
    has_a = np.isfinite(y_anc)
    feats[has_a, base_col + 3] = np.clip(
        (y_anc[has_a] - mu_train[has_a]) / scales[has_a], -6.0, 6.0
    ).astype(np.float32)

    # Col base+4: Pre-stimulus local standard deviation / scale
    pre_cnt = np.sum(m_pre, axis=0)
    has_2 = pre_cnt >= 2
    if np.any(has_2):
        centered = np.where(m_pre[:, has_2], x_pre[:, has_2] - y_anc[None, has_2], 0.0)
        var_pre = np.sum(centered ** 2, axis=0) / pre_cnt[has_2]
        feats[has_2, base_col + 4] = np.clip(np.sqrt(var_pre) / scales[has_2], 0.0, 6.0).astype(np.float32)

    # Col base+5: Causal observation availability fraction
    feats[:, base_col + 5] = (pre_cnt / max(float(t_in), 1.0)).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


_ALLOW_MASK_CACHE: Dict[Tuple[Tuple[str, ...], bool, bool], np.ndarray] = {}


def _get_allow_mask_matrix(
    canonical_neurons: Sequence[str],
    exclude_bilateral: bool,
    preserve_sensory_asymmetry: bool,
) -> np.ndarray:
    """Return cached boolean `[N_canon, N_canon]` where `allow[i, j]` is True iff `j` is allowed for target `i`."""
    key = (tuple(canonical_neurons), bool(exclude_bilateral), bool(preserve_sensory_asymmetry))
    cached = _ALLOW_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    n_canon = len(canonical_neurons)
    allow = np.ones((n_canon, n_canon), dtype=bool)
    for target_i, t_name in enumerate(canonical_neurons):
        excl_indices = get_bilateral_exclusion_set(
            t_name,
            canonical_neurons,
            exclude_bilateral=exclude_bilateral,
            preserve_sensory_asymmetry=preserve_sensory_asymmetry,
        )
        allow[target_i, excl_indices] = False
    _ALLOW_MASK_CACHE[key] = allow
    return allow


def extract_causal_network_state_per_target(
    trial: Mapping[str, Any],
    canonical_neurons: Sequence[str],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    exclude_bilateral: bool = True,
    preserve_sensory_asymmetry: bool = False,
    precision_kappa: float = 0.0,
) -> np.ndarray:
    """Extract target-specific causal slow-state vectors `Z_state` `[N_canon, D_state]` strictly excluding `i` (and `i_LR`).

    Computes within-worm centered normalized pre-stimulus activity (`(x_pre - mu_worm_causal) / s_n`)
    across functionally defined C. elegans circuit modules (`Layer-1 AIA/AIB/AIY/AIZ`, `Backward AVA/AVD/AVE/RIM`,
    `Forward AVB/PVC/RIB`, `Asymmetric ASEL/ASER/AWCON/AWCOFF`) and their causal derivatives (`dAVA/dt`, `dL1/dt`),
    strictly zeroing out `i` and `i_LR` BEFORE averaging!
    """
    x_pre = np.asarray(trial["x_pre"], dtype=np.float64)
    m_pre = np.asarray(trial["m_pre"], dtype=bool)
    x_cal = np.asarray(trial["x_cal"], dtype=np.float64)
    m_cal = np.asarray(trial["m_cal"], dtype=bool)
    scales = np.asarray(s_n, dtype=np.float64)
    fps = float(trial["fps"])
    t_in, n_canon = x_pre.shape

    # Within-worm causal baseline per neuron (combining [-30,-10)s and [-10,0)s, strictly < onset_idx)
    if x_cal.shape[0] > 0:
        x_bg = np.vstack([x_cal, x_pre])
        m_bg = np.vstack([m_cal, m_pre])
    else:
        x_bg = x_pre
        m_bg = m_pre
    bg_cnt = np.sum(m_bg, axis=0)
    bg_mu = np.where(
        bg_cnt > 0,
        np.sum(np.where(m_bg, x_bg, 0.0), axis=0) / np.maximum(bg_cnt, 1),
        mu_train,
    )

    # Normalized causal trace z_pre [T_in, N_canon], zero where unobserved
    z_pre = np.where(m_pre, np.clip((x_pre - bg_mu[None, :]) / scales[None, :], -5.0, 5.0), 0.0)

    n_fast = max(1, min(t_in, int(round(2.5 * fps))))
    n_early = max(1, t_in - n_fast)

    # Per-neuron fast level [-2.5, 0)s, early level [-10, -2.5)s, and derivative
    cnt_fast = np.sum(m_pre[-n_fast:], axis=0)
    obs_j = cnt_fast > 0
    z_fast = np.where(obs_j, np.sum(z_pre[-n_fast:], axis=0) / np.maximum(cnt_fast, 1), 0.0)

    cnt_early = np.sum(m_pre[:n_early], axis=0)
    obs_e = cnt_early > 0
    z_early = np.where(obs_e, np.sum(z_pre[:n_early], axis=0) / np.maximum(cnt_early, 1), z_fast)
    dz_dt = z_fast - z_early  # Causal pre-stimulus slope (e.g. dAVA/dt, dRIM/dt!)

    idx_map = {n: i for i, n in enumerate(canonical_neurons)}
    l1_idx = [idx_map[n] for n in LAYER1_INTEGRATORS if n in idx_map]
    bwd_idx = [idx_map[n] for n in BACKWARD_COMMAND if n in idx_map]
    fwd_idx = [idx_map[n] for n in FORWARD_COMMAND if n in idx_map]
    sens_idx = [idx_map[n] for n in CORE_AMPHID_SENSORY if n in idx_map]
    asym_idx = [idx_map[n] for n in ASYMMETRIC_CHEMOSENSORY if n in idx_map]

    allow_mat = _get_allow_mask_matrix(
        canonical_neurons,
        exclude_bilateral=exclude_bilateral,
        preserve_sensory_asymmetry=preserve_sensory_asymmetry,
    )
    valid_excl_mat = allow_mat & obs_j[None, :]  # [N_canon, N_canon]

    def _vec_group_stat(vals: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        if not indices:
            return np.zeros(n_canon, dtype=np.float64)
        sub_mask = valid_excl_mat[:, indices]  # [N_canon, |G|]
        n_v = np.sum(sub_mask, axis=1).astype(np.float64)
        raw_sum = sub_mask.astype(np.float64) @ vals[list(indices)]
        raw_mean = np.where(n_v > 0.0, raw_sum / np.maximum(n_v, 1.0), 0.0)
        if precision_kappa > 0.0:
            raw_mean = raw_mean * (n_v / (n_v + precision_kappa))
        return raw_mean

    b_lev = _vec_group_stat(z_fast, bwd_idx)
    b_der = _vec_group_stat(dz_dt, bwd_idx)
    f_lev = _vec_group_stat(z_fast, fwd_idx)
    f_der = _vec_group_stat(dz_dt, fwd_idx)
    l1_lev = _vec_group_stat(z_fast, l1_idx)
    l1_der = _vec_group_stat(dz_dt, l1_idx)

    d_state = 12
    out = np.zeros((n_canon, d_state), dtype=np.float32)
    out[:, 0] = b_lev.astype(np.float32)
    out[:, 1] = b_der.astype(np.float32)
    out[:, 2] = f_lev.astype(np.float32)
    out[:, 3] = f_der.astype(np.float32)
    out[:, 4] = (b_lev - f_lev).astype(np.float32)
    out[:, 5] = np.tanh(b_lev - f_lev).astype(np.float32)
    out[:, 6] = l1_lev.astype(np.float32)
    out[:, 7] = l1_der.astype(np.float32)

    if preserve_sensory_asymmetry and len(asym_idx) == 4:
        for k_a, a_idx in enumerate(asym_idx):
            out[:, 8 + k_a] = np.where(valid_excl_mat[:, a_idx], z_fast[a_idx], 0.0).astype(np.float32)
    else:
        s_lev = _vec_group_stat(z_fast, sens_idx)
        s_der = _vec_group_stat(dz_dt, sens_idx)
        all_n_v = np.sum(valid_excl_mat, axis=1).astype(np.float64)
        all_sum = valid_excl_mat.astype(np.float64) @ z_fast
        all_mean = np.where(all_n_v > 0.0, all_sum / np.maximum(all_n_v, 1.0), 0.0)
        if precision_kappa > 0.0:
            all_mean = all_mean * (all_n_v / (all_n_v + precision_kappa))
        out[:, 8] = s_lev.astype(np.float32)
        out[:, 9] = s_der.astype(np.float32)
        out[:, 10] = all_mean.astype(np.float32)
        out[:, 11] = (all_n_v / max(n_canon, 1)).astype(np.float32)

    return out

