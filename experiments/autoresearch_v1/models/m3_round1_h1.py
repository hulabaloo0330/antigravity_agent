"""Milestone M3 — Round 1: H1 Slow-State Increment + 7 Confound Controls & Residualized Architecture.

Implements and evaluates 4 genuinely trained causal trajectory models across all 3 immobilized Dev splits
(`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`) under unified `configs/lambda_global.json`
(zero `dataset_id`-specific hyperparameter branches):

1. `M3R1_01_self_anchor_multiscale`:
   Strengthened self-only anchor combining `B4_ridge_self_stim` + multi-scale causal self-history basis
   (`[-10,-5)s`, `[-5,-2)s`, `[-2,0)s` mean/slope/last-value + causal orthonormal spline basis +
   stimulus-conditioned self-response residualization) with shrunk per-neuron residual coupling.
2. `M3R1_02_h1_confounds_only`:
   Anchored on `M3R1_01_self_anchor_multiscale`, adds ONLY the 7 strictly causal confound controls
   (fold-internally standardized):
     (1) global mean pre-stimulus fluorescence (excluding target `i` & bilateral `i_LR`),
     (2) neuron missingness/observation fraction `m`,
     (3) stimulus round/habituation index (`round_idx`),
     (4) prior cue identity (`prev_stimulus_id`),
     (5) recording batch indicator (fold-safe OOF empirical-Bayes shrunk encoding, zero leakage),
     (6) animal sex (`hermaphrodite`/`male`/`unknown`),
     (7) self-baseline pre-stimulus drift (`[-30,-10)s` and `[-10,-5)s` vs `[-5,0)s`).
3. `M3R1_03_h1_slow_state_circuit`:
   Two-stage residual learning anchored on `M3R1_02_h1_confounds_only`, adding missingness-precision-weighted
   (`sqrt(n_obs / (n_obs + kappa))`) pre-stimulus slow-state features strictly excluding target `i` and
   its bilateral counterpart:
     - Sensory integration interneurons (`AIA`, `AIB`, `AIY`, `AIZ`, L/R preserved),
     - Command / reversal-forward attractors (`AVA`, `AVB`, `RIM`, `RIB` + `AVE`/`AVD`/`RMD`),
     - Asymmetric chemosensory context (`ASEL`/`ASER`, `AWCON`/`AWCOFF`, L/R preserved),
     - Multi-window causal slow-state (`[-10,-5)s`, `[-5,0)s`, slow slope) + fold-internal causal PCA.
4. `M3R1_04_h1_slow_state_mlp_residual`:
   Compact strongly regularized PyTorch residual MLP predicting the `M3R1_02_h1_confounds_only` residual
   from the same excluded-self circuit slow-state features (inner animal-grouped validation early stopping,
   deterministic seed, unified `lambda_global`).
"""

from __future__ import annotations

import copy
import datetime as dt
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from evaluators.scorer import score_saved_predictions_file
from src.baselines import (
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
    _solve_ridge_sample_scaled,
)
from src.data_adapters import DATA_ROOT_DEFAULT, EXP_ROOT_DEFAULT, compute_file_sha256, load_dataset
from src.evaluator import (
    align_records_to_canonical,
    apply_sensor_bounds_to_aligned,
    build_grouped_cv_splits,
    compare_against_reference,
    compute_training_fold_neuron_scales,
    compute_training_fold_sensor_bounds,
    extract_causal_trial_records,
    save_CompactPredictions,
)
from src.features import (
    _get_allow_mask_matrix,
    extract_self_history_features_matrix,
    fit_fold_neuron_stim_priors,
    predict_prior_delta_trajectory,
)

# ---------------------------------------------------------------------------
# Circuit Module Definitions (Preserving L/R & Sensory Asymmetry)
# ---------------------------------------------------------------------------
SENSORY_INTEGRATION_LR: Tuple[str, ...] = (
    "AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
)
SENSORY_INTEGRATION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AIAL", "AIAR"), ("AIBL", "AIBR"), ("AIYL", "AIYR"), ("AIZL", "AIZR"),
)
COMMAND_BACKWARD_EXT: Tuple[str, ...] = (
    "AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER", "RIML", "RIMR",
    "RMDL", "RMDR", "RMDDL", "RMDDR", "RMDVL", "RMDVR",
)
COMMAND_FORWARD_EXT: Tuple[str, ...] = (
    "AVBL", "AVBR", "RIBL", "RIBR", "PVCL", "PVCR", "RID",
)
COMMAND_CORE_PAIRS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("AVA", ("AVAL", "AVAR")),
    ("AVB", ("AVBL", "AVBR")),
    ("RIM", ("RIML", "RIMR")),
    ("RIB", ("RIBL", "RIBR")),
)
ASYMMETRIC_SENSORY_NEURONS: Tuple[str, ...] = ("ASEL", "ASER", "AWCON", "AWCOFF")

M3R1_MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "run_id": "r1_01_m3r1_01_self_anchor_multiscale",
        "model_id": "M3R1_01_self_anchor_multiscale",
        "parent_id": "r0_04_b4_ridge_self_stim",
        "is_self_only": True,
        "uses_confounds": False,
        "uses_network_state": False,
        "is_stochastic": False,
        "single_factor_changed": "multiscale_self_history_basis_plus_shrunk_per_neuron_residual",
        "hypothesis": "Strengthened self-only anchor (B4_ridge_self_stim + [-10,-5), [-5,-2), [-2,0)s mean/slope/last + causal spline basis + shrunk per-neuron residual) to verify whether H1 gains exceed improved self-history.",
    },
    {
        "run_id": "r1_02_m3r1_02_h1_confounds_only",
        "model_id": "M3R1_02_h1_confounds_only",
        "parent_id": "r1_01_m3r1_01_self_anchor_multiscale",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": False,
        "is_stochastic": False,
        "single_factor_changed": "add_7_causal_confound_controls_residualized",
        "hypothesis": "Residualized 7 confound controls (excluded-self global mean, missingness fraction m, round_idx, prev_stim, fold-safe OOF batch encoding, sex, self pre-stim drift) anchored on M3R1_01.",
    },
    {
        "run_id": "r1_03_m3r1_03_h1_slow_state_circuit",
        "model_id": "M3R1_03_h1_slow_state_circuit",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "is_stochastic": False,
        "single_factor_changed": "add_kappa_shrunk_excluded_self_circuit_slow_state_ridge_residual",
        "hypothesis": "H1: Two-stage residualized excluded-self/bilateral circuit slow-state (AIA/AIB/AIY/AIZ, AVA/AVB/RIM/RIB, ASEL/ASER/AWCON/AWCOFF + causal PCA, sqrt(n/(n+kappa)) weighted) beyond M3R1_02 confounds.",
    },
    {
        "run_id": "r1_04_m3r1_04_h1_slow_state_mlp_residual",
        "model_id": "M3R1_04_h1_slow_state_mlp_residual",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "is_stochastic": True,
        "single_factor_changed": "nonlinear_residual_mlp_on_excluded_self_circuit_slow_state",
        "hypothesis": "H1 nonlinear: Compact strongly regularized PyTorch residual MLP predicting M3R1_02 residuals from excluded-self circuit slow-state features.",
    },
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2)


def _code_bundle_sha256(exp_root: Path) -> str:
    h = hashlib.sha256()
    rel_files = [
        "protocol.json",
        "configs/lambda_global.json",
        "src/evaluator.py",
        "src/features.py",
        "src/baselines.py",
        "evaluators/scorer.py",
        "models/m3_round1_h1.py",
    ]
    for rel in rel_files:
        p = exp_root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Multi-Scale Causal Self-History Feature Extractor (Self-Only!)
# ---------------------------------------------------------------------------
_PRE_SPLINE_CACHE: Dict[Tuple[int, float, Tuple[float, ...]], np.ndarray] = {}


def _get_causal_pre_spline_basis(
    t_in: int,
    fps: float,
    taus_sec: Sequence[float] = (0.75, 2.0, 5.0, 10.0),
) -> np.ndarray:
    """Build fps-invariant orthonormal causal temporal basis `[T_in, K]` on `[-10, 0)s`."""
    key = (int(t_in), round(float(fps), 4), tuple(float(x) for x in taus_sec))
    cached = _PRE_SPLINE_CACHE.get(key)
    if cached is not None:
        return cached
    t_back = (np.arange(t_in, 0, -1, dtype=np.float64) - 0.5) / max(float(fps), 1e-6)
    cols = [np.exp(-t_back / max(float(tau), 1e-3)) for tau in taus_sec]
    raw = np.column_stack(cols)
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = (q * signs[None, :]).astype(np.float64)
    _PRE_SPLINE_CACHE[key] = q
    return q


def _window_mean_and_slope(
    z_pre: np.ndarray,
    m_pre: np.ndarray,
    t_sec: np.ndarray,
    win_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-neuron mean and OLS slope (`dz/dt` in units/sec) over a causal sub-window."""
    sub_z = z_pre[win_mask]
    sub_m = m_pre[win_mask]
    sub_t = t_sec[win_mask]
    n_canon = z_pre.shape[1]

    cnt = np.sum(sub_m, axis=0).astype(np.float64)
    valid = cnt > 0
    mean_v = np.zeros(n_canon, dtype=np.float64)
    if np.any(valid):
        mean_v[valid] = np.sum(np.where(sub_m[:, valid], sub_z[:, valid], 0.0), axis=0) / cnt[valid]

    slope_v = np.zeros(n_canon, dtype=np.float64)
    valid_s = cnt >= 2
    if np.any(valid_s):
        t_col = sub_t[:, None]
        tm = np.where(sub_m[:, valid_s], t_col, 0.0)
        t_bar = np.sum(tm, axis=0) / cnt[valid_s]
        dt_c = np.where(sub_m[:, valid_s], t_col - t_bar[None, :], 0.0)
        dz_c = np.where(sub_m[:, valid_s], sub_z[:, valid_s] - mean_v[None, valid_s], 0.0)
        denom = np.sum(dt_c ** 2, axis=0)
        good_d = denom > 1e-6
        sl = np.zeros(int(np.sum(valid_s)), dtype=np.float64)
        sl[good_d] = np.sum(dt_c[:, good_d] * dz_c[:, good_d], axis=0) / denom[good_d]
        slope_v[valid_s] = np.clip(sl, -3.0, 3.0)

    return mean_v, slope_v


def extract_multiscale_self_features(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    taus_sec: Sequence[float] = (0.75, 2.0, 5.0, 10.0),
) -> np.ndarray:
    """Strictly causal multi-scale self-history features `[N_canon, D_ms]` (zero cross-neuron info).

    Combines standard `h_self` (10 dims) + multi-scale `[-10,-5)s`, `[-5,-2)s`, `[-2,0)s` means & slopes
    (6 dims) + last observed value relative to anchor & curvature (2 dims) + 4-dim causal spline basis
    projection (4 dims) = 22 dims.
    """
    h_base = extract_self_history_features_matrix(trial, s_n, mu_train)  # [N, 10]

    x_pre = np.asarray(trial["x_pre"], dtype=np.float64)
    m_pre = np.asarray(trial["m_pre"], dtype=bool)
    y_anc = np.asarray(trial["y_pre_anchor"], dtype=np.float64)
    scales = np.asarray(s_n, dtype=np.float64)
    fps = float(trial["fps"])
    t_in, n_canon = x_pre.shape

    safe_anc = np.where(np.isfinite(y_anc), y_anc, mu_train)
    z_rel = np.where(
        m_pre,
        np.clip((x_pre - safe_anc[None, :]) / scales[None, :], -6.0, 6.0),
        0.0,
    )

    # Physical time in seconds relative to onset (strictly negative in [-10, 0))
    t_back = (np.arange(t_in, 0, -1, dtype=np.float64) - 0.5) / max(fps, 1e-6)
    t_sec = -t_back

    w_10_5 = (t_back >= 5.0) & (t_back <= 10.5)
    w_5_2 = (t_back >= 2.0) & (t_back < 5.0)
    w_2_0 = (t_back >= 0.0) & (t_back < 2.0)
    if not np.any(w_10_5):
        w_10_5[0] = True
    if not np.any(w_5_2):
        w_5_2[t_in // 2] = True
    if not np.any(w_2_0):
        w_2_0[-1] = True

    m_10_5, sl_10_5 = _window_mean_and_slope(z_rel, m_pre, t_sec, w_10_5)
    m_5_2, sl_5_2 = _window_mean_and_slope(z_rel, m_pre, t_sec, w_5_2)
    m_2_0, sl_2_0 = _window_mean_and_slope(z_rel, m_pre, t_sec, w_2_0)

    # Last observed causal value relative to y_pre_anchor
    last_val = np.zeros(n_canon, dtype=np.float64)
    has_any = np.any(m_pre, axis=0)
    if np.any(has_any):
        # Index of last True along axis 0
        rev_idx = np.argmax(m_pre[::-1, has_any], axis=0)
        last_rows = (t_in - 1) - rev_idx
        cols = np.where(has_any)[0]
        last_val[cols] = z_rel[last_rows, cols]

    # Curvature / acceleration between [-2,0)s and [-5,-2)s
    curv = np.clip(sl_2_0 - sl_5_2, -3.0, 3.0)

    # Causal orthonormal spline basis projection of [-10, 0)s self-history
    q_pre = _get_causal_pre_spline_basis(t_in, fps, taus_sec=taus_sec)  # [T_in, 4]
    # Projection normalized by observed fraction to stay invariant to missing frames
    obs_frac = np.maximum(np.mean(m_pre, axis=0), 0.1)
    spline_proj = (z_rel.T @ q_pre) / np.sqrt(max(float(t_in), 1.0)) / obs_frac[:, None]
    spline_proj = np.clip(spline_proj, -5.0, 5.0)

    extra = np.column_stack(
        [m_10_5, m_5_2, m_2_0, sl_10_5, sl_5_2, sl_2_0, last_val, curv, spline_proj]
    ).astype(np.float32)
    out = np.hstack([h_base, extra]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 2. All 7 Strictly Causal Confound Controls (Zero Leakage)
# ---------------------------------------------------------------------------
def _compute_causal_within_worm_z(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(z_pre [T_in, N], m_pre [T_in, N], bg_mu [N])` using strictly `< onset_idx` frames."""
    x_pre = np.asarray(trial["x_pre"], dtype=np.float64)
    m_pre = np.asarray(trial["m_pre"], dtype=bool)
    x_cal = np.asarray(trial["x_cal"], dtype=np.float64)
    m_cal = np.asarray(trial["m_cal"], dtype=bool)
    scales = np.asarray(s_n, dtype=np.float64)

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
    z_pre = np.where(
        m_pre,
        np.clip((x_pre - bg_mu[None, :]) / scales[None, :], -5.0, 5.0),
        0.0,
    )
    return z_pre, m_pre, bg_mu


def extract_confound_features_matrix(
    trial: Mapping[str, Any],
    canonical_neurons: Sequence[str],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    prev_stim_vocab: Sequence[str],
    batch_enc_vec: np.ndarray,
    batch_seen: float,
    kappa: float = 4.0,
    net_trial_override: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Extract all 7 causal confound controls `[N_canon, D_conf]`.

    Confound (1) global pre-stimulus mean strictly excludes target `i` AND its bilateral homolog `i_LR`!
    `net_trial_override` (used ONLY in invariance probes) overrides the cross-neuron global mean source
    while keeping target self-drift fixed.
    """
    n_canon = len(canonical_neurons)
    net_src = net_trial_override if net_trial_override is not None else trial

    # Confound (1): Global mean pre-stimulus fluorescence (strictly excluding target i & bilateral i_LR!)
    z_pre_net, m_pre_net, _ = _compute_causal_within_worm_z(net_src, s_n, mu_train)
    t_in_net = z_pre_net.shape[0]
    fps_net = float(net_src["fps"])
    n_fast = max(1, min(t_in_net, int(round(2.5 * fps_net))))

    cnt_fast = np.sum(m_pre_net[-n_fast:], axis=0)
    obs_fast = cnt_fast > 0
    z_fast = np.where(obs_fast, np.sum(z_pre_net[-n_fast:], axis=0) / np.maximum(cnt_fast, 1), 0.0)

    cnt_all = np.sum(m_pre_net, axis=0)
    obs_all = cnt_all > 0
    z_all = np.where(obs_all, np.sum(z_pre_net, axis=0) / np.maximum(cnt_all, 1), 0.0)

    allow_mat = _get_allow_mask_matrix(
        canonical_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=False
    )
    vm_fast = allow_mat & obs_fast[None, :]
    nv_fast = np.sum(vm_fast, axis=1).astype(np.float64)
    gmean_fast = np.where(nv_fast > 0.0, (vm_fast.astype(np.float64) @ z_fast) / np.maximum(nv_fast, 1.0), 0.0)
    gmean_fast = gmean_fast * np.sqrt(nv_fast / (nv_fast + max(kappa, 1e-6)))

    vm_all = allow_mat & obs_all[None, :]
    nv_all = np.sum(vm_all, axis=1).astype(np.float64)
    gmean_all = np.where(nv_all > 0.0, (vm_all.astype(np.float64) @ z_all) / np.maximum(nv_all, 1.0), 0.0)
    gmean_all = gmean_all * np.sqrt(nv_all / (nv_all + max(kappa, 1e-6)))

    # Confound (2): Neuron missingness / observation fraction m (trial-wide excl-self + self obs frac)
    trial_obs_frac = nv_all / max(float(n_canon - 2), 1.0)
    m_pre_self = np.asarray(trial["m_pre"], dtype=bool)
    self_obs_frac = np.mean(m_pre_self, axis=0).astype(np.float64)

    # Confound (3): Stimulus round / habituation index
    rep_idx = float(trial.get("repeat_idx", 1))
    t_ord = float(trial.get("trial_order", 1))
    f_round = np.full(n_canon, rep_idx / 5.0, dtype=np.float64)
    f_log_round = np.full(n_canon, np.log1p(max(rep_idx, 0.0)), dtype=np.float64)
    f_tord = np.full(n_canon, t_ord / 20.0, dtype=np.float64)

    # Confound (4): Prior cue identity (prev_stimulus_id one-hot + same_as_prev_stim)
    prev_s = str(trial.get("prev_stim_id", "none"))
    cur_s = str(trial.get("stimulus_id", "unknown"))
    p_oh = np.zeros(len(prev_stim_vocab) + 1, dtype=np.float64)
    if prev_s in prev_stim_vocab:
        p_oh[list(prev_stim_vocab).index(prev_s)] = 1.0
    else:
        p_oh[-1] = 1.0
    p_mat = np.repeat(p_oh[None, :], n_canon, axis=0)
    same_prev = np.full(n_canon, 1.0 if prev_s == cur_s else 0.0, dtype=np.float64)

    # Confound (5): Recording batch indicator (fold-safe shrunk OOF residual encoding + batch_seen)
    b_mat = np.repeat(np.asarray(batch_enc_vec, dtype=np.float64)[None, :], n_canon, axis=0)
    b_seen = np.full(n_canon, float(batch_seen), dtype=np.float64)

    # Confound (6): Animal sex (hermaphrodite / male / unknown)
    sx = str(trial.get("sex", "unknown")).strip().lower()
    sx_oh = np.array(
        [
            1.0 if sx == "hermaphrodite" else 0.0,
            1.0 if sx == "male" else 0.0,
            1.0 if sx not in ("hermaphrodite", "male") else 0.0,
        ],
        dtype=np.float64,
    )
    sx_mat = np.repeat(sx_oh[None, :], n_canon, axis=0)

    # Confound (7): Self-baseline pre-stimulus drift ([-5,0)s vs [-10,-5)s AND [-30,-10)s calibration drift)
    x_pre_s = np.asarray(trial["x_pre"], dtype=np.float64)
    x_cal_s = np.asarray(trial["x_cal"], dtype=np.float64)
    m_cal_s = np.asarray(trial["m_cal"], dtype=bool)
    y_anc_s = np.asarray(trial["y_pre_anchor"], dtype=np.float64)
    scales = np.asarray(s_n, dtype=np.float64)
    t_in_s = x_pre_s.shape[0]
    fps_s = float(trial["fps"])
    t_back_s = (np.arange(t_in_s, 0, -1, dtype=np.float64) - 0.5) / max(fps_s, 1e-6)
    w_5_0 = t_back_s < 5.0
    w_10_5 = t_back_s >= 5.0
    if not np.any(w_5_0):
        w_5_0[-1] = True
    if not np.any(w_10_5):
        w_10_5[0] = True

    c5 = np.sum(m_pre_self[w_5_0], axis=0)
    c10 = np.sum(m_pre_self[w_10_5], axis=0)
    mu5 = np.where(c5 > 0, np.sum(np.where(m_pre_self[w_5_0], x_pre_s[w_5_0], 0.0), axis=0) / np.maximum(c5, 1), 0.0)
    mu10 = np.where(c10 > 0, np.sum(np.where(m_pre_self[w_10_5], x_pre_s[w_10_5], 0.0), axis=0) / np.maximum(c10, 1), mu5)
    drift_10_5 = np.where((c5 > 0) & (c10 > 0), np.clip((mu5 - mu10) / scales, -5.0, 5.0), 0.0)

    drift_cal = np.zeros(n_canon, dtype=np.float64)
    if x_cal_s.shape[0] > 0:
        ccal = np.sum(m_cal_s, axis=0)
        has_c = (ccal > 0) & np.isfinite(y_anc_s)
        if np.any(has_c):
            cal_mu = np.sum(np.where(m_cal_s[:, has_c], x_cal_s[:, has_c], 0.0), axis=0) / ccal[has_c]
            drift_cal[has_c] = np.clip((y_anc_s[has_c] - cal_mu) / scales[has_c], -5.0, 5.0)

    cols = np.column_stack(
        [
            gmean_fast,
            gmean_all,
            trial_obs_frac,
            self_obs_frac,
            f_round,
            f_log_round,
            f_tord,
            same_prev,
            b_seen,
            drift_10_5,
            drift_cal,
        ]
    )
    out = np.hstack([cols, p_mat, b_mat, sx_mat]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 3. H1 Excluded-Self Circuit Slow-State & Fold-Internal Causal PCA
# ---------------------------------------------------------------------------
def _extract_trial_slow_vectors(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-neuron causal `[-10,-5)s` level, `[-5,0)s` level, slow slope, and observation masks."""
    z_pre, m_pre, _ = _compute_causal_within_worm_z(trial, s_n, mu_train)
    t_in, n_canon = z_pre.shape
    fps = float(trial["fps"])
    t_back = (np.arange(t_in, 0, -1, dtype=np.float64) - 0.5) / max(fps, 1e-6)

    w_recent = t_back < 5.0
    w_early = t_back >= 5.0
    if not np.any(w_recent):
        w_recent[-1] = True
    if not np.any(w_early):
        w_early[0] = True

    c_rec = np.sum(m_pre[w_recent], axis=0)
    obs_rec = c_rec > 0
    z_rec = np.where(obs_rec, np.sum(z_pre[w_recent], axis=0) / np.maximum(c_rec, 1), 0.0)

    c_ear = np.sum(m_pre[w_early], axis=0)
    obs_ear = c_ear > 0
    z_ear = np.where(obs_ear, np.sum(z_pre[w_early], axis=0) / np.maximum(c_ear, 1), z_rec)

    obs_both = obs_rec & obs_ear
    z_slope = np.where(obs_both, z_rec - z_ear, 0.0)
    return z_ear, z_rec, z_slope, obs_rec, obs_ear


def fit_causal_slow_state_pca(
    train_trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    n_canon: int,
    n_comp: int = 6,
    n_slope_comp: int = 3,
) -> Dict[str, np.ndarray]:
    """Fit causal population slow-state PCA strictly inside training fold (zero-imputed + mask)."""
    rec_rows: List[np.ndarray] = []
    slp_rows: List[np.ndarray] = []
    for tr in train_trials:
        if bool(tr.get("is_buffer_control", False)):
            continue
        _, z_rec, z_slp, obs_rec, _ = _extract_trial_slow_vectors(tr, s_n, mu_train)
        rec_rows.append(np.where(obs_rec, z_rec, 0.0))
        slp_rows.append(np.where(obs_rec, z_slp, 0.0))

    if len(rec_rows) < 4:
        return {
            "v_rec": np.zeros((n_canon, n_comp), dtype=np.float64),
            "mu_rec": np.zeros(n_canon, dtype=np.float64),
            "v_slp": np.zeros((n_canon, n_slope_comp), dtype=np.float64),
            "mu_slp": np.zeros(n_canon, dtype=np.float64),
        }

    x_rec = np.vstack(rec_rows).astype(np.float64)
    mu_rec = np.mean(x_rec, axis=0)
    _, _, vt_rec = np.linalg.svd(x_rec - mu_rec[None, :], full_matrices=False)
    k1 = min(n_comp, vt_rec.shape[0])
    v_rec = np.zeros((n_canon, n_comp), dtype=np.float64)
    v_rec[:, :k1] = vt_rec[:k1].T

    x_slp = np.vstack(slp_rows).astype(np.float64)
    mu_slp = np.mean(x_slp, axis=0)
    _, _, vt_slp = np.linalg.svd(x_slp - mu_slp[None, :], full_matrices=False)
    k2 = min(n_slope_comp, vt_slp.shape[0])
    v_slp = np.zeros((n_canon, n_slope_comp), dtype=np.float64)
    v_slp[:, :k2] = vt_slp[:k2].T

    return {"v_rec": v_rec, "mu_rec": mu_rec, "v_slp": v_slp, "mu_slp": mu_slp}


def extract_circuit_slow_state_features(
    trial: Mapping[str, Any],
    canonical_neurons: Sequence[str],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    pca_state: Mapping[str, np.ndarray],
    kappa: float = 4.0,
) -> np.ndarray:
    """Extract missingness-precision-weighted excluded-self/bilateral circuit slow-state `[N_canon, D_circ]`.

    - Strictly masks out target `i` AND its bilateral homolog `i_LR` BEFORE any pooling or PCA projection.
    - Uses `sqrt(n_obs / (n_obs + kappa))` precision weighting.
    - Preserves L/R identity for `AIA/AIB/AIY/AIZ` and strictly preserves `ASEL`/`ASER`, `AWCON`/`AWCOFF`.
    """
    n_canon = len(canonical_neurons)
    idx_map = {n: i for i, n in enumerate(canonical_neurons)}
    z_ear, z_rec, z_slp, obs_rec, obs_ear = _extract_trial_slow_vectors(trial, s_n, mu_train)

    allow_mat = _get_allow_mask_matrix(
        canonical_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=False
    )
    vm_rec = allow_mat & obs_rec[None, :]  # [N_canon, N_canon]
    vm_ear = allow_mat & obs_ear[None, :]
    vm_both = vm_rec & vm_ear

    def _precision_group_mean(vals: np.ndarray, vmat: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        if not indices:
            return np.zeros(n_canon, dtype=np.float64)
        sub_m = vmat[:, list(indices)]
        n_v = np.sum(sub_m, axis=1).astype(np.float64)
        raw_mean = np.where(n_v > 0.0, (sub_m.astype(np.float64) @ vals[list(indices)]) / np.maximum(n_v, 1.0), 0.0)
        return raw_mean * np.sqrt(n_v / (n_v + max(kappa, 1e-6)))

    feats: List[np.ndarray] = []

    # (A) Sensory integration interneurons: AIA, AIB, AIY, AIZ (group multi-window + L/R preserved)
    l1_indices = [idx_map[n] for n in SENSORY_INTEGRATION_LR if n in idx_map]
    feats.append(_precision_group_mean(z_ear, vm_ear, l1_indices))
    feats.append(_precision_group_mean(z_rec, vm_rec, l1_indices))
    feats.append(_precision_group_mean(z_slp, vm_both, l1_indices))

    # L/R preserved per-neuron level [-5,0)s and L-R contrast for AIA, AIB, AIY, AIZ
    for n_l, n_r in SENSORY_INTEGRATION_PAIRS:
        il = idx_map.get(n_l)
        ir = idx_map.get(n_r)
        vl = np.where(vm_rec[:, il], z_rec[il], 0.0) if il is not None else np.zeros(n_canon, dtype=np.float64)
        vr = np.where(vm_rec[:, ir], z_rec[ir], 0.0) if ir is not None else np.zeros(n_canon, dtype=np.float64)
        # Shrink single-neuron reading gently by sqrt(1/(1+kappa/4)) when observed
        s_single = np.sqrt(1.0 / (1.0 + max(kappa, 0.0) * 0.25))
        feats.append(vl * s_single)
        feats.append(vr * s_single)
        both_lr = (vm_rec[:, il] & vm_rec[:, ir]) if (il is not None and ir is not None) else np.zeros(n_canon, dtype=bool)
        feats.append(np.where(both_lr, (vl - vr) * s_single, 0.0))

    # (B) Command / reversal-forward attractors: AVA, AVB, RIM, RIB (+ AVE, AVD, RMD)
    bwd_idx = [idx_map[n] for n in COMMAND_BACKWARD_EXT if n in idx_map]
    fwd_idx = [idx_map[n] for n in COMMAND_FORWARD_EXT if n in idx_map]
    b_ear = _precision_group_mean(z_ear, vm_ear, bwd_idx)
    b_rec = _precision_group_mean(z_rec, vm_rec, bwd_idx)
    b_slp = _precision_group_mean(z_slp, vm_both, bwd_idx)
    f_ear = _precision_group_mean(z_ear, vm_ear, fwd_idx)
    f_rec = _precision_group_mean(z_rec, vm_rec, fwd_idx)
    f_slp = _precision_group_mean(z_slp, vm_both, fwd_idx)
    feats.extend(
        [
            b_ear,
            b_rec,
            b_slp,
            f_ear,
            f_rec,
            f_slp,
            b_rec - f_rec,
            np.tanh(b_rec - f_rec),
            b_slp - f_slp,
        ]
    )

    for _, pair_names in COMMAND_CORE_PAIRS:
        p_idx = [idx_map[n] for n in pair_names if n in idx_map]
        feats.append(_precision_group_mean(z_rec, vm_rec, p_idx))
        feats.append(_precision_group_mean(z_slp, vm_both, p_idx))

    # (C) Asymmetric chemosensory context: ASEL, ASER, AWCON, AWCOFF (NEVER collapsing L/R!)
    asym_rec: Dict[str, np.ndarray] = {}
    asym_obs: Dict[str, np.ndarray] = {}
    s_single = np.sqrt(1.0 / (1.0 + max(kappa, 0.0) * 0.25))
    for aname in ASYMMETRIC_SENSORY_NEURONS:
        ai = idx_map.get(aname)
        if ai is not None:
            v_rec = np.where(vm_rec[:, ai], z_rec[ai], 0.0) * s_single
            v_slp = np.where(vm_both[:, ai], z_slp[ai], 0.0) * s_single
            asym_rec[aname] = v_rec
            asym_obs[aname] = vm_rec[:, ai]
        else:
            v_rec = np.zeros(n_canon, dtype=np.float64)
            v_slp = np.zeros(n_canon, dtype=np.float64)
            asym_rec[aname] = v_rec
            asym_obs[aname] = np.zeros(n_canon, dtype=bool)
        feats.append(v_rec)
        feats.append(v_slp)

    # Explicit ASEL-ASER and AWCON-AWCOFF asymmetry contrasts (zeroed if either excluded or unobserved)
    feats.append(np.where(asym_obs["ASEL"] & asym_obs["ASER"], asym_rec["ASEL"] - asym_rec["ASER"], 0.0))
    feats.append(np.where(asym_obs["AWCON"] & asym_obs["AWCOFF"], asym_rec["AWCON"] - asym_rec["AWCOFF"], 0.0))

    # (D) Fold-internal Causal PCA / low-rank population slow-state score (strictly excluding i & i_LR!)
    v_rec_mat = pca_state["v_rec"]  # [N_canon, K1]
    mu_rec_vec = pca_state["mu_rec"]
    v_slp_mat = pca_state["v_slp"]  # [N_canon, K2]
    mu_slp_vec = pca_state["mu_slp"]

    nv_all = np.sum(vm_rec, axis=1).astype(np.float64)
    prec_all = np.sqrt(nv_all / (nv_all + max(kappa, 1e-6)))

    z_rec_c = np.where(obs_rec, z_rec - mu_rec_vec, 0.0)
    vm_rec_f = vm_rec.astype(np.float64)
    # Strictly excludes target i & bilateral i_LR because vm_rec_f[i, excl] == 0.0!
    raw_pc_rec = vm_rec_f @ (z_rec_c[:, None] * v_rec_mat)  # [N_canon, K1]
    norm_pc_rec = np.sqrt(vm_rec_f @ (v_rec_mat ** 2) + 1e-3)
    pc_rec = (raw_pc_rec / norm_pc_rec) * prec_all[:, None]

    z_slp_c = np.where(obs_rec, z_slp - mu_slp_vec, 0.0)
    raw_pc_slp = vm_rec_f @ (z_slp_c[:, None] * v_slp_mat)  # [N_canon, K2]
    norm_pc_slp = np.sqrt(vm_rec_f @ (v_slp_mat ** 2) + 1e-3)
    pc_slp = (raw_pc_slp / norm_pc_slp) * prec_all[:, None]

    out = np.hstack([np.column_stack(feats), pc_rec, pc_slp]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 4. Unified M3 Round 1 Model Class (Two-Stage Residualized Architecture)
# ---------------------------------------------------------------------------
class M3R1Model:
    """Implements all 4 M3 Round 1 models under unified `configs/lambda_global.json`."""

    def __init__(
        self,
        model_id: str,
        config: Mapping[str, Any],
        canonical_neurons: Sequence[str],
        functional_tiers: Sequence[str],
    ) -> None:
        self.model_id = str(model_id)
        self.config = dict(config)
        self.hp = dict(config.get("hyperparameters", config))
        self.canonical_neurons = list(canonical_neurons)
        self.functional_tiers = list(functional_tiers)
        self.n_canon = len(self.canonical_neurons)
        self.state: Dict[str, Any] = {}

        tier_names = ("sensory", "interneuron", "premotor_command", "motor", "pharyngeal")
        self._tier_oh = np.zeros((self.n_canon, len(tier_names)), dtype=np.float32)
        for i, t_str in enumerate(self.functional_tiers):
            if t_str in tier_names:
                self._tier_oh[i, tier_names.index(t_str)] = 1.0

    # --- Stage 1: Strengthened Self-Only Anchor (`M3R1_01_self_anchor_multiscale`) ---
    def _build_self_anchor_features(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(x_self_global [N, D_g], x_self_indiv [N, D_i])` strictly from target self + stimulus."""
        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("h_ms", {}):
            h_ms = fold_cache["h_ms"][tid]
        else:
            h_ms = extract_multiscale_self_features(
                trial, s_n, priors["mu_train"], taus_sec=self.state["taus"]
            )
        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), priors["stim_vocab"]
        )
        stim_mat = np.repeat(stim_oh[None, :], self.n_canon, axis=0)
        _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, self.n_canon)
        prior_norm = prior_coef / s_n[:, None].astype(np.float32)

        # Stimulus-conditioned self-response interactions (level, calibration dev, fast/recent slope x prior)
        # h_ms cols: 0..3 lag bins, 4 fast_d, 5 slow_d, 6 cal_dev, 7 pop_dev, 8 std, 9 obs_frac,
        #            10..12 (m_10_5, m_5_2, m_2_0), 13..15 (sl_10_5, sl_5_2, sl_2_0), 16 last, 17 curv, 18..21 spline
        self_x_prior = h_ms[:, [4, 6, 7, 15, 16]] * prior_norm[:, [0]]
        x_global = np.hstack([h_ms, self._tier_oh, stim_mat, prior_norm, self_x_prior]).astype(np.float32)
        # Compact self-only features for shrunk per-neuron residual coupling
        x_indiv = h_ms[:, [0, 1, 2, 3, 6, 7, 10, 11, 12, 14, 15, 16]].astype(np.float32)
        return x_global, x_indiv

    def _fit_stage1_self_anchor(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        l2 = float(self.hp.get("l2_penalty", 25.0))
        shrink_n = float(self.hp.get("prior_shrinkage_counts", 5.0))
        b_dim = int(priors["b_dim"])

        xg_list, y_list = [], []
        per_n_x: List[List[np.ndarray]] = [[] for _ in range(self.n_canon)]
        per_n_y: List[List[np.ndarray]] = [[] for _ in range(self.n_canon)]
        trial_cache_s1: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
                c_norm, rval, _, _ = fold_cache["proj_targets"][tid]
            else:
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors, s_n, self.n_canon)
            xg, xi = self._build_self_anchor_features(tr, s_n, priors, fold_cache=fold_cache)
            trial_cache_s1[tid] = (xg, xi, c_norm, rval)
            idx = np.where(rval)[0]
            if idx.size > 0:
                xg_list.append(xg[idx])
                y_list.append(c_norm[idx])

        w_self_global = _solve_ridge_sample_scaled(np.vstack(xg_list), np.vstack(y_list), l2)

        # Stage 1b: Empirical-Bayes shrunk per-neuron self-history residual coupling
        d_indiv = 12
        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            xg, xi, c_norm, rval = trial_cache_s1[tid]
            res_1a = c_norm - (xg @ w_self_global)
            idx = np.where(rval)[0]
            for n_i in idx:
                per_n_x[int(n_i)].append(xi[int(n_i)])
                per_n_y[int(n_i)].append(res_1a[int(n_i)])

        w_self_indiv = np.zeros((self.n_canon, d_indiv, b_dim), dtype=np.float32)
        for n_i in range(self.n_canon):
            cnt = len(per_n_x[n_i])
            if cnt >= 6:
                xn = np.vstack(per_n_x[n_i])
                yn = np.vstack(per_n_y[n_i])
                w_raw = _solve_ridge_sample_scaled(xn, yn, l2 * 6.0)
                eb_shrink = float(cnt / (cnt + shrink_n))
                w_self_indiv[n_i] = (w_raw * eb_shrink).astype(np.float32)

        return {
            "w_self_global": w_self_global,
            "w_self_indiv": w_self_indiv,
            "trial_cache_s1": trial_cache_s1,
        }

    def _predict_stage1_coef(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        s1: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        xg, xi = self._build_self_anchor_features(trial, s_n, priors, fold_cache=fold_cache)
        c_g = xg @ s1["w_self_global"]
        c_i = np.einsum("nd,ndb->nb", xi, s1["w_self_indiv"])
        return (c_g + c_i).astype(np.float32)

    # --- Stage 2: 7 Confound Controls (`M3R1_02_h1_confounds_only`) ---
    def _fit_stage2_confounds(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        s1: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        l2_conf = float(self.hp.get("state_l2_penalty", 50.0))
        b_dim = int(priors["b_dim"])
        prev_vocab = sorted({str(t.get("prev_stim_id", "none")) for t in non_buf_trials})

        # Compute Stage-1 residuals on training trials & fold-safe OOF batch residual encoding
        # Group sums by (date_batch) and (date_batch, animal_id) so each animal's encoding excludes itself!
        batch_sum: Dict[str, np.ndarray] = {}
        batch_cnt: Dict[str, float] = {}
        ba_sum: Dict[Tuple[str, str], np.ndarray] = {}
        ba_cnt: Dict[Tuple[str, str], float] = {}
        r1_by_tid: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            xg, xi, c_norm, rval = s1["trial_cache_s1"][tid]
            pred_s1 = (xg @ s1["w_self_global"]) + np.einsum("nd,ndb->nb", xi, s1["w_self_indiv"])
            r1 = (c_norm - pred_s1).astype(np.float64)
            r1_by_tid[tid] = (r1.astype(np.float32), rval)

            idx = np.where(rval)[0]
            if idx.size > 0:
                b_id = str(tr.get("date_batch", "unknown"))
                a_id = str(tr.get("animal_id", "unknown"))
                s_vec = np.sum(r1[idx], axis=0)
                cnt = float(idx.size)
                batch_sum[b_id] = batch_sum.get(b_id, np.zeros(b_dim, dtype=np.float64)) + s_vec
                batch_cnt[b_id] = batch_cnt.get(b_id, 0.0) + cnt
                ba_key = (b_id, a_id)
                ba_sum[ba_key] = ba_sum.get(ba_key, np.zeros(b_dim, dtype=np.float64)) + s_vec
                ba_cnt[ba_key] = ba_cnt.get(ba_key, 0.0) + cnt

        batch_shrink = 200.0
        batch_enc_full: Dict[str, np.ndarray] = {}
        for b_id, s_vec in batch_sum.items():
            batch_enc_full[b_id] = (s_vec / (batch_cnt[b_id] + batch_shrink)).astype(np.float32)

        xc_list, y_list = [], []
        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            r1, rval = r1_by_tid[tid]
            b_id = str(tr.get("date_batch", "unknown"))
            a_id = str(tr.get("animal_id", "unknown"))
            # Leave-one-animal-out batch encoding inside training fold
            rem_cnt = batch_cnt.get(b_id, 0.0) - ba_cnt.get((b_id, a_id), 0.0)
            if rem_cnt > 0.0:
                rem_sum = batch_sum[b_id] - ba_sum.get((b_id, a_id), np.zeros(b_dim, dtype=np.float64))
                b_enc_oof = (rem_sum / (rem_cnt + batch_shrink)).astype(np.float32)
                b_seen = 1.0
            else:
                b_enc_oof = np.zeros(b_dim, dtype=np.float32)
                b_seen = 0.0

            xc = extract_confound_features_matrix(
                tr,
                self.canonical_neurons,
                s_n,
                priors["mu_train"],
                prev_vocab,
                b_enc_oof,
                b_seen,
                kappa=kappa,
            )
            idx = np.where(rval)[0]
            if idx.size > 0:
                xc_list.append(xc[idx])
                y_list.append(r1[idx])

        x_all = np.vstack(xc_list).astype(np.float64)
        y_all = np.vstack(y_list).astype(np.float64)
        c_mean = np.mean(x_all, axis=0, keepdims=True)
        c_std = np.maximum(np.std(x_all, axis=0, keepdims=True), 1e-4)
        x_norm = ((x_all - c_mean) / c_std).astype(np.float32)

        w_conf = _solve_ridge_sample_scaled(x_norm, y_all.astype(np.float32), l2_conf)
        return {
            "prev_stim_vocab": prev_vocab,
            "batch_enc_full": batch_enc_full,
            "b_dim": b_dim,
            "c_mean": c_mean.astype(np.float32),
            "c_std": c_std.astype(np.float32),
            "w_conf": w_conf,
            "r1_by_tid": r1_by_tid,
        }

    def _predict_stage2_delta_coef(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        s2: Mapping[str, Any],
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        b_id = str(trial.get("date_batch", "unknown"))
        if b_id in s2["batch_enc_full"]:
            b_enc = s2["batch_enc_full"][b_id]
            b_seen = 1.0
        else:
            b_enc = np.zeros(int(s2["b_dim"]), dtype=np.float32)
            b_seen = 0.0
        xc = extract_confound_features_matrix(
            trial,
            self.canonical_neurons,
            s_n,
            priors["mu_train"],
            s2["prev_stim_vocab"],
            b_enc,
            b_seen,
            kappa=kappa,
            net_trial_override=net_trial_override,
        )
        xc_norm = (xc - s2["c_mean"]) / s2["c_std"]
        return (xc_norm @ s2["w_conf"]).astype(np.float32)

    # --- Stage 3: H1 Excluded-Self Circuit Slow-State (Ridge or Residual MLP) ---
    def _assemble_stage3_data(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        s1: Mapping[str, Any],
        s2: Mapping[str, Any],
        pca_state: Mapping[str, np.ndarray],
        include_tier: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        xz_list, y_list = [], []
        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            r1, rval = s2["r1_by_tid"][tid]
            # Stage-2 residual: target - (Stage1 + Stage2)
            d2 = self._predict_stage2_delta_coef(tr, s_n, priors, s2)
            r2 = r1 - d2
            z_circ = extract_circuit_slow_state_features(
                tr, self.canonical_neurons, s_n, priors["mu_train"], pca_state, kappa=kappa
            )
            if include_tier:
                z_in = np.hstack([z_circ, self._tier_oh]).astype(np.float32)
            else:
                z_in = z_circ
            idx = np.where(rval)[0]
            if idx.size > 0:
                xz_list.append(z_in[idx])
                y_list.append(r2[idx])
        return np.vstack(xz_list), np.vstack(y_list)

    def _fit_stage3_circuit_ridge(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        s1: Mapping[str, Any],
        s2: Mapping[str, Any],
    ) -> Dict[str, Any]:
        rank = int(self.hp.get("rank", 6))
        l2_state = float(self.hp.get("state_l2_penalty", 50.0))
        pca_state = fit_causal_slow_state_pca(
            non_buf_trials, s_n, priors["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )
        x_z, y_r2 = self._assemble_stage3_data(
            non_buf_trials, s_n, priors, s1, s2, pca_state, include_tier=False
        )
        z_mean = np.mean(x_z, axis=0, keepdims=True).astype(np.float32)
        z_std = np.maximum(np.std(x_z, axis=0, keepdims=True), 1e-4).astype(np.float32)
        x_norm = ((x_z - z_mean) / z_std).astype(np.float32)
        # Scale penalty proportionally to feature count relative to baseline 12-dim state so variance cannot blow up
        eff_l2 = l2_state * max(1.0, float(x_norm.shape[1]) / 12.0)
        w_circ = _solve_ridge_sample_scaled(x_norm, y_r2, eff_l2)
        return {
            "pca_state": pca_state,
            "z_mean": z_mean,
            "z_std": z_std,
            "w_circ": w_circ,
        }

    def _fit_stage3_circuit_mlp(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        outer_priors: Mapping[str, Any],
        outer_s1: Mapping[str, Any],
        outer_s2: Mapping[str, Any],
    ) -> Dict[str, Any]:
        seed = int(self.hp.get("seed", 20261117))
        torch.manual_seed(seed)
        np.random.seed(seed)

        rank = int(self.hp.get("rank", 6))
        h_dim = int(self.hp.get("mlp_hidden_dim", 48))
        drop = float(self.hp.get("dropout", 0.15))
        lr = float(self.hp.get("learning_rate", 0.003))
        wd = float(self.hp.get("weight_decay", 0.01)) * 2.0  # Strong regularization on residual MLP
        max_ep = int(self.hp.get("max_epochs", 18))
        patience = int(self.hp.get("early_stopping_patience", 4))
        bsz = int(self.hp.get("batch_size", 2048))
        b_dim = int(outer_priors["b_dim"])

        # Inner animal-grouped validation split for early stopping (zero leakage!)
        worms = sorted({str(t["animal_id"]) for t in non_buf_trials})
        n_val_w = max(1, len(worms) // 5)
        val_worms = set(worms[-n_val_w:])
        inner_tr = [t for t in non_buf_trials if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in non_buf_trials if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(non_buf_trials)
            inner_va = list(non_buf_trials[:1])

        in_priors = fit_fold_neuron_stim_priors(
            inner_tr,
            self.n_canon,
            taus_sec=self.state["taus"],
            shrinkage_counts=float(self.hp.get("prior_shrinkage_counts", 5.0)),
        )
        in_s1 = self._fit_stage1_self_anchor(inner_tr, s_n, in_priors, fold_cache=None)
        in_s2 = self._fit_stage2_confounds(inner_tr, s_n, in_priors, in_s1, fold_cache=None)
        in_pca = fit_causal_slow_state_pca(
            inner_tr, s_n, in_priors["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )
        x_itr, y_itr = self._assemble_stage3_data(
            inner_tr, s_n, in_priors, in_s1, in_s2, in_pca, include_tier=True
        )

        # Assemble inner validation targets strictly using inner anchor
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        x_iva_l, y_iva_l = [], []
        for tr in inner_va:
            c_norm, rval, _, _ = _project_trial_residual_targets(tr, in_priors, s_n, self.n_canon)
            p1 = self._predict_stage1_coef(tr, s_n, in_priors, in_s1)
            d2 = self._predict_stage2_delta_coef(tr, s_n, in_priors, in_s2)
            r2 = c_norm - (p1 + d2)
            z_circ = extract_circuit_slow_state_features(
                tr, self.canonical_neurons, s_n, in_priors["mu_train"], in_pca, kappa=kappa
            )
            z_in = np.hstack([z_circ, self._tier_oh]).astype(np.float32)
            idx = np.where(rval)[0]
            if idx.size > 0:
                x_iva_l.append(z_in[idx])
                y_iva_l.append(r2[idx])
        x_iva = np.vstack(x_iva_l)
        y_iva = np.vstack(y_iva_l)

        xm_i = np.mean(x_itr, axis=0, keepdims=True).astype(np.float32)
        xs_i = np.maximum(np.std(x_itr, axis=0, keepdims=True), 1e-4).astype(np.float32)
        d_in = x_itr.shape[1]

        def _make_res_mlp() -> nn.Module:
            net = nn.Sequential(
                nn.Linear(d_in, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(h_dim, max(16, h_dim // 2)),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(max(16, h_dim // 2), b_dim),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        probe = _make_res_mlp()
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        xt_t = torch.from_numpy(((x_itr - xm_i) / xs_i).astype(np.float32))
        yt_t = torch.from_numpy(y_itr.astype(np.float32))
        xv_t = torch.from_numpy(((x_iva - xm_i) / xs_i).astype(np.float32))
        yv_t = torch.from_numpy(y_iva.astype(np.float32))

        best_val = float(torch.mean(yv_t ** 2).item())
        best_ep = 0
        bad = 0
        n_rows = xt_t.shape[0]
        for ep in range(1, max_ep + 1):
            probe.train()
            perm = torch.randperm(n_rows)
            for s0 in range(0, n_rows, bsz):
                bi = perm[s0 : s0 + bsz]
                opt.zero_grad()
                loss = torch.mean((probe(xt_t[bi]) - yt_t[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            probe.eval()
            with torch.no_grad():
                vloss = float(torch.mean((probe(xv_t) - yv_t) ** 2).item())
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        # Outer refit on all training-fold animals for `best_ep` epochs
        outer_pca = fit_causal_slow_state_pca(
            non_buf_trials, s_n, outer_priors["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )
        x_full, y_full = self._assemble_stage3_data(
            non_buf_trials, s_n, outer_priors, outer_s1, outer_s2, outer_pca, include_tier=True
        )
        xm_o = np.mean(x_full, axis=0, keepdims=True).astype(np.float32)
        xs_o = np.maximum(np.std(x_full, axis=0, keepdims=True), 1e-4).astype(np.float32)

        torch.manual_seed(seed)
        final_net = _make_res_mlp()
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            xf_t = torch.from_numpy(((x_full - xm_o) / xs_o).astype(np.float32))
            yf_t = torch.from_numpy(y_full.astype(np.float32))
            nf = xf_t.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for s0 in range(0, nf, bsz):
                    bi = perm[s0 : s0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(xf_t[bi]) - yf_t[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()

        return {
            "pca_state": outer_pca,
            "z_mean": xm_o,
            "z_std": xs_o,
            "mlp_net": final_net,
            "selected_epochs": best_ep,
        }

    def fit(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> "M3R1Model":
        taus = tuple(float(x) for x in self.hp.get("basis_taus_sec", [0.75, 2.0, 5.0, 10.0]))
        shrink = float(self.hp.get("prior_shrinkage_counts", 5.0))
        if fold_cache is not None and "priors" in fold_cache:
            priors = fold_cache["priors"]
        else:
            priors = fit_fold_neuron_stim_priors(
                train_trials, self.n_canon, taus_sec=taus, shrinkage_counts=shrink
            )

        self.state = {
            "s_n": np.asarray(s_n, dtype=np.float32).copy(),
            "priors": priors,
            "taus": taus,
        }
        non_buf_trials = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        s1 = self._fit_stage1_self_anchor(non_buf_trials, s_n, priors, fold_cache=fold_cache)
        self.state["s1"] = {
            "w_self_global": s1["w_self_global"],
            "w_self_indiv": s1["w_self_indiv"],
        }
        if self.model_id == "M3R1_01_self_anchor_multiscale":
            return self

        s2 = self._fit_stage2_confounds(non_buf_trials, s_n, priors, s1, fold_cache=fold_cache)
        self.state["s2"] = {
            "prev_stim_vocab": s2["prev_stim_vocab"],
            "batch_enc_full": s2["batch_enc_full"],
            "b_dim": s2["b_dim"],
            "c_mean": s2["c_mean"],
            "c_std": s2["c_std"],
            "w_conf": s2["w_conf"],
        }
        if self.model_id == "M3R1_02_h1_confounds_only":
            return self

        if self.model_id == "M3R1_03_h1_slow_state_circuit":
            self.state["s3"] = self._fit_stage3_circuit_ridge(non_buf_trials, s_n, priors, s1, s2)
        elif self.model_id == "M3R1_04_h1_slow_state_mlp_residual":
            self.state["s3"] = self._fit_stage3_circuit_mlp(non_buf_trials, s_n, priors, s1, s2)
        else:
            raise ValueError(f"Unknown M3R1 model_id: {self.model_id}")
        return self

    def predict_trial(
        self,
        trial: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        s_n = self.state["s_n"]
        priors = self.state["priors"]
        mu_train = priors["mu_train"]
        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
            _, _, prior_delta, phi = fold_cache["proj_targets"][tid]
        else:
            prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, self.n_canon)

        pred_c = self._predict_stage1_coef(trial, s_n, priors, self.state["s1"], fold_cache=fold_cache)

        if self.model_id != "M3R1_01_self_anchor_multiscale":
            pred_c = pred_c + self._predict_stage2_delta_coef(
                trial, s_n, priors, self.state["s2"], net_trial_override=net_trial_override
            )

        if self.model_id in ("M3R1_03_h1_slow_state_circuit", "M3R1_04_h1_slow_state_mlp_residual"):
            s3 = self.state["s3"]
            kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
            net_src = net_trial_override if net_trial_override is not None else trial
            z_circ = extract_circuit_slow_state_features(
                net_src, self.canonical_neurons, s_n, mu_train, s3["pca_state"], kappa=kappa
            )
            if self.model_id == "M3R1_03_h1_slow_state_circuit":
                z_norm = (z_circ - s3["z_mean"]) / s3["z_std"]
                pred_c = pred_c + (z_norm @ s3["w_circ"])
            else:
                z_in = np.hstack([z_circ, self._tier_oh]).astype(np.float32)
                z_norm = (z_in - s3["z_mean"]) / s3["z_std"]
                net: nn.Module = s3["mlp_net"]
                net.eval()
                with torch.no_grad():
                    pred_c = pred_c + net(torch.from_numpy(z_norm.astype(np.float32))).numpy()

        return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

    def parameter_vector(self) -> np.ndarray:
        parts: List[np.ndarray] = [
            self.state["s_n"].ravel().astype(np.float64),
            self.state["priors"]["mu_train"].ravel().astype(np.float64),
            self.state["s1"]["w_self_global"].ravel().astype(np.float64),
            self.state["s1"]["w_self_indiv"].ravel().astype(np.float64),
        ]
        if "s2" in self.state:
            parts.append(self.state["s2"]["w_conf"].ravel().astype(np.float64))
        if "s3" in self.state:
            s3 = self.state["s3"]
            parts.append(s3["pca_state"]["v_rec"].ravel().astype(np.float64))
            if "w_circ" in s3:
                parts.append(s3["w_circ"].ravel().astype(np.float64))
            if "mlp_net" in s3:
                for p in s3["mlp_net"].parameters():
                    parts.append(p.detach().cpu().numpy().ravel().astype(np.float64))
        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 5. Explicit Leakage Invariance Verification on M3R1_03_h1_slow_state_circuit
# ---------------------------------------------------------------------------
def verify_m3_r1_invariance(
    model_id: str = "M3R1_03_h1_slow_state_circuit",
    dataset_id: str = "yemini2021",
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
) -> Dict[str, Any]:
    """Verify future-frame poisoning < 1e-7, OOF tamper invariance < 1e-7, and target+bilateral exclusion."""
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    raw_recs = load_dataset(dataset_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
    sub_recs = raw_recs[: min(8, len(raw_recs))]
    aligned, canon, tiers = align_records_to_canonical(sub_recs)
    splits = build_grouped_cv_splits(aligned)
    train_idx, val_idx = splits[0]

    s_n_clean, _, _ = compute_training_fold_neuron_scales(
        [aligned[i].traces for i in train_idx],
        [aligned[i].obs_mask for i in train_idx],
        floor_percentile=10.0,
    )
    train_trials_clean = [t for i in train_idx for t in extract_causal_trial_records(aligned[i])]
    val_rec_clean = aligned[val_idx[0]]
    val_trials_clean = extract_causal_trial_records(val_rec_clean)

    model_clean = M3R1Model(model_id, cfg, canon, tiers).fit(train_trials_clean, s_n_clean)
    y_pred_clean = model_clean.predict_trial(val_trials_clean[0])

    # Probe 1: Future-frame poisoning t >= onset_idx
    val_rec_poison = copy.deepcopy(val_rec_clean)
    ev0_onset = int(val_rec_poison.stim_events[0]["onset_idx"])
    val_rec_poison.traces[ev0_onset:, :] = np.where(
        val_rec_poison.obs_mask[ev0_onset:, :],
        val_rec_poison.traces[ev0_onset:, :] * -25.0 + 1.0e6,
        np.nan,
    ).astype(np.float32)
    val_trials_poison = extract_causal_trial_records(val_rec_poison)
    y_pred_poison = model_clean.predict_trial(val_trials_poison[0])
    future_diff = float(np.max(np.abs(y_pred_clean - y_pred_poison)))
    target_changed = float(np.nanmax(np.abs(val_trials_clean[0]["y_true"] - val_trials_poison[0]["y_true"])))

    # Probe 2: OOF validation label/trace tampering
    aligned_tamper = copy.deepcopy(aligned)
    for vi in val_idx:
        aligned_tamper[vi].traces = np.where(
            aligned_tamper[vi].obs_mask,
            aligned_tamper[vi].traces * -50.0 + 9876.5,
            np.nan,
        ).astype(np.float32)
    s_n_tamper, _, _ = compute_training_fold_neuron_scales(
        [aligned_tamper[i].traces for i in train_idx],
        [aligned_tamper[i].obs_mask for i in train_idx],
        floor_percentile=10.0,
    )
    train_trials_tamper = [t for i in train_idx for t in extract_causal_trial_records(aligned_tamper[i])]
    model_tamper = M3R1Model(model_id, cfg, canon, tiers).fit(train_trials_tamper, s_n_tamper)
    oof_weight_diff = float(np.max(np.abs(model_clean.parameter_vector() - model_tamper.parameter_vector())))
    oof_pred_diff = float(np.max(np.abs(y_pred_clean - model_tamper.predict_trial(val_trials_clean[0]))))

    # Probe 3: Target + bilateral exclusion across AVA, AIY, ASE, AWC + asymmetry preservation for non-self target
    idx_map = {n: i for i, n in enumerate(canon)}

    def _pair_exclusion_diff(target_name: str, partner_name: str) -> float:
        ti, pi = idx_map[target_name], idx_map[partner_name]
        r0 = copy.deepcopy(val_rec_clean)
        r0.traces[:ev0_onset, pi] = 0.0
        r0.obs_mask[:ev0_onset, pi] = True
        r1 = copy.deepcopy(val_rec_clean)
        r1.traces[:ev0_onset, pi] = 50.0
        r1.obs_mask[:ev0_onset, pi] = True
        p0 = model_clean.predict_trial(extract_causal_trial_records(r0)[0])
        p1 = model_clean.predict_trial(extract_causal_trial_records(r1)[0])
        return float(np.max(np.abs(p0[ti] - p1[ti])))

    ava_bilat_diff = _pair_exclusion_diff("AVAL", "AVAR")
    aiy_bilat_diff = _pair_exclusion_diff("AIYL", "AIYR")
    ase_bilat_diff = _pair_exclusion_diff("ASEL", "ASER")
    awc_bilat_diff = _pair_exclusion_diff("AWCON", "AWCOFF")

    # Direct target-self perturbation inside network/confound pathway (holding self-history fixed)
    base_tr = extract_causal_trial_records(val_rec_clean)[0]
    net_ov = copy.deepcopy(base_tr)
    aval_i = idx_map["AVAL"]
    net_ov["x_pre"] = net_ov["x_pre"].copy()
    net_ov["m_pre"] = net_ov["m_pre"].copy()
    net_ov["x_pre"][:, aval_i] = 50.0
    net_ov["m_pre"][:, aval_i] = True
    yp_base = model_clean.predict_trial(base_tr)
    yp_ov = model_clean.predict_trial(base_tr, net_trial_override=net_ov)
    target_self_net_diff = float(np.max(np.abs(yp_base[aval_i] - yp_ov[aval_i])))

    # Verify ASEL & ASER remain distinct (non-collapsed) for an unrelated target (AIYL)
    aiyl_i = idx_map["AIYL"]
    asel_i, aser_i = idx_map["ASEL"], idx_map["ASER"]
    t_a = copy.deepcopy(base_tr)
    t_a["x_pre"] = t_a["x_pre"].copy()
    t_a["m_pre"] = t_a["m_pre"].copy()
    t_a["x_pre"][:, [asel_i, aser_i]] = 0.0
    t_a["m_pre"][:, [asel_i, aser_i]] = True
    t_b = copy.deepcopy(t_a)
    t_b["x_pre"] = t_b["x_pre"].copy()
    t_b["x_pre"][-8:, asel_i] = 3.0 * float(s_n_clean[asel_i])
    t_b["x_pre"][-8:, aser_i] = -3.0 * float(s_n_clean[aser_i])
    zf_a = extract_circuit_slow_state_features(
        t_a, canon, s_n_clean, model_clean.state["priors"]["mu_train"], model_clean.state["s3"]["pca_state"]
    )
    zf_b = extract_circuit_slow_state_features(
        t_b, canon, s_n_clean, model_clean.state["priors"]["mu_train"], model_clean.state["s3"]["pca_state"]
    )
    aiyl_sees_ase_asymmetry = float(np.max(np.abs(zf_b[aiyl_i] - zf_a[aiyl_i])))

    passed = (
        future_diff < 1e-7
        and target_changed > 1e4
        and oof_weight_diff < 1e-7
        and oof_pred_diff < 1e-7
        and ava_bilat_diff < 1e-7
        and aiy_bilat_diff < 1e-7
        and ase_bilat_diff < 1e-7
        and awc_bilat_diff < 1e-7
        and target_self_net_diff < 1e-7
        and aiyl_sees_ase_asymmetry > 1e-3
    )
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "passed": bool(passed),
        "future_poison_max_abs_diff": future_diff,
        "future_poison_target_shift_verified": target_changed,
        "oof_tamper_weight_diff": oof_weight_diff,
        "oof_tamper_pred_diff": oof_pred_diff,
        "bilateral_exclusion_AVAR_to_AVAL_diff": ava_bilat_diff,
        "bilateral_exclusion_AIYR_to_AIYL_diff": aiy_bilat_diff,
        "bilateral_exclusion_ASER_to_ASEL_diff": ase_bilat_diff,
        "bilateral_exclusion_AWCOFF_to_AWCON_diff": awc_bilat_diff,
        "target_self_network_pathway_diff": target_self_net_diff,
        "aiyl_sees_asel_aser_asymmetry_delta": aiyl_sees_ase_asymmetry,
    }


# ---------------------------------------------------------------------------
# 6. Animal-Level Bootstrap CIs for I_d and I_macro (1000 Resamples)
# ---------------------------------------------------------------------------
def compute_macro_bootstrap_ci(
    cand_by_ds: Mapping[str, Mapping[str, Any]],
    ref_by_ds: Mapping[str, Mapping[str, Any]],
    ds_list: Sequence[str],
    n_boot: int = 1000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compute paired animal-level bootstrap 95% CI for `I_macro` across `ds_list`."""
    rng = np.random.default_rng(seed)
    boot_i_d: List[np.ndarray] = []
    point_i_d: List[float] = []
    for ds_id in ds_list:
        c_lw = cand_by_ds[ds_id]["L_w"]
        r_lw = ref_by_ds[ds_id]["L_w"]
        worms = sorted(c_lw.keys())
        c_arr = np.array([float(c_lw[w]) for w in worms], dtype=np.float64)
        r_arr = np.array([float(r_lw[w]) for w in worms], dtype=np.float64)
        point_i_d.append(float(1.0 - np.mean(c_arr) / max(np.mean(r_arr), 1e-12)))
        idx = rng.integers(0, len(worms), size=(n_boot, len(worms)))
        b_gains = 1.0 - np.mean(c_arr[idx], axis=1) / np.maximum(np.mean(r_arr[idx], axis=1), 1e-12)
        boot_i_d.append(b_gains)
    macro_point = float(np.mean(point_i_d))
    macro_boot = np.mean(np.vstack(boot_i_d), axis=0)
    return {
        "I_macro": round(macro_point, 6),
        "I_macro_ci95": [round(float(np.percentile(macro_boot, 2.5)), 6), round(float(np.percentile(macro_boot, 97.5)), 6)],
    }


# ---------------------------------------------------------------------------
# 7. Memory-Safe Runner (1 Dataset at a Time + gc.collect() + Standalone Scorer)
# ---------------------------------------------------------------------------
def _build_m3_fold_cache(
    all_trials: Sequence[Mapping[str, Any]],
    train_trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    n_canon: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    hp = config.get("hyperparameters", config)
    taus = tuple(float(x) for x in hp.get("basis_taus_sec", [0.75, 2.0, 5.0, 10.0]))
    shrink = float(hp.get("prior_shrinkage_counts", 5.0))
    priors = fit_fold_neuron_stim_priors(train_trials, n_canon, taus_sec=taus, shrinkage_counts=shrink)
    h_ms_map: Dict[str, np.ndarray] = {}
    proj_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for tr in all_trials:
        tid = str(tr.get("trial_id", ""))
        h_ms_map[tid] = extract_multiscale_self_features(tr, s_n, priors["mu_train"], taus_sec=taus)
        proj_map[tid] = _project_trial_residual_targets(tr, priors, s_n, n_canon)
    return {"priors": priors, "h_ms": h_ms_map, "proj_targets": proj_map}


def run_m3_round1_h1(
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
    n_boot: int = 1000,
    resume: bool = False,
) -> Dict[str, Any]:
    t_start = time.time()
    cfg_path = exp_root / "configs" / "lambda_global.json"
    prot_path = exp_root / "protocol.json"
    sm_path = exp_root / "split_manifest.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]

    protocol_sha = compute_file_sha256(prot_path)
    config_sha = compute_file_sha256(cfg_path)
    split_sha = compute_file_sha256(sm_path)
    code_sha = _code_bundle_sha256(exp_root)

    datasets = ["yemini2021", "seyedolmohadesin2025", "toyoshima2024"]
    main_sources = ["seyedolmohadesin2025", "toyoshima2024"]

    runs_dir = exp_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Track per-model timing & prediction paths
    model_wall: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R1_MODEL_SPECS}
    model_cpu: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R1_MODEL_SPECS}
    model_reg_ts: Dict[str, str] = {s["model_id"]: _utc_now() for s in M3R1_MODEL_SPECS}
    model_pred_files: Dict[str, Dict[str, str]] = {s["model_id"]: {} for s in M3R1_MODEL_SPECS}

    # Step 1: Process ONE dataset at a time to keep peak RSS < 3.0 GiB!
    for ds_id in datasets:
        all_exist = all(
            (runs_dir / spec["run_id"] / f"predictions_dev_{ds_id}.npz").exists()
            for spec in M3R1_MODEL_SPECS
        )
        if resume and all_exist:
            for spec in M3R1_MODEL_SPECS:
                mid = spec["model_id"]
                rid = spec["run_id"]
                pred_file = runs_dir / rid / f"predictions_dev_{ds_id}.npz"
                model_pred_files[mid][ds_id] = str(pred_file.relative_to(exp_root))
            print(f"[M3R1] Resumed existing predictions for {ds_id} (4/4 models on disk)", flush=True)
            continue

        t_ds0 = time.time()
        print(f"[M3R1] Loading Dev split: {ds_id} (RSS={_peak_rss_mb():.1f} MB)", flush=True)
        raw_recs = load_dataset(ds_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
        aligned, canon, tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        del raw_recs

        fold_bundles: List[Dict[str, Any]] = []
        train_fold_session_ids: List[List[str]] = []
        for f_idx, (tr_idx, va_idx) in enumerate(splits):
            tr_phys = {aligned[i].physical_animal_id for i in tr_idx}
            va_phys = {aligned[i].physical_animal_id for i in va_idx}
            if tr_phys & va_phys:
                raise RuntimeError(f"Physical animal leakage in {ds_id} fold {f_idx}: {tr_phys & va_phys}")

            lo_b, hi_b = compute_training_fold_sensor_bounds(
                [aligned[i].traces for i in tr_idx],
                [aligned[i].obs_mask for i in tr_idx],
            )
            tr_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in tr_idx]
            va_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in va_idx]

            s_n, _, _ = compute_training_fold_neuron_scales(
                [r.traces for r in tr_bounded],
                [r.obs_mask for r in tr_bounded],
                floor_percentile=10.0,
            )
            tr_trials = [
                t
                for r in tr_bounded
                for t in extract_causal_trial_records(
                    r,
                    input_sec=float(hp["input_window_sec"]),
                    target_sec=float(hp["target_window_sec"]),
                    calibration_sec=float(hp["calibration_window_sec"]),
                    fast_anchor_sec=float(hp["fast_anchor_sec"]),
                    include_synthetic_buffer_windows=True,
                )
            ]
            va_trials = [
                t
                for r in va_bounded
                for t in extract_causal_trial_records(
                    r,
                    input_sec=float(hp["input_window_sec"]),
                    target_sec=float(hp["target_window_sec"]),
                    calibration_sec=float(hp["calibration_window_sec"]),
                    fast_anchor_sec=float(hp["fast_anchor_sec"]),
                    include_synthetic_buffer_windows=True,
                )
            ]
            f_cache = _build_m3_fold_cache(tr_trials + va_trials, tr_trials, s_n, len(canon), cfg)
            fold_bundles.append(
                {
                    "fold_idx": f_idx,
                    "train_trials": tr_trials,
                    "val_trials": va_trials,
                    "s_n": s_n,
                    "cache": f_cache,
                }
            )
            train_fold_session_ids.append([aligned[i].animal_id for i in tr_idx])

        del aligned

        for spec in M3R1_MODEL_SPECS:
            mid = spec["model_id"]
            rid = spec["run_id"]
            rdir = runs_dir / rid
            rdir.mkdir(parents=True, exist_ok=True)
            pred_file = rdir / f"predictions_dev_{ds_id}.npz"

            t_m0 = time.time()
            c_m0 = time.process_time()
            oof_trials: List[Dict[str, Any]] = []
            oof_preds: List[np.ndarray] = []
            oof_fold_ids: List[int] = []

            for fb in fold_bundles:
                model = M3R1Model(mid, cfg, canon, tiers)
                model.fit(fb["train_trials"], fb["s_n"], fold_cache=fb["cache"])
                for vtr in fb["val_trials"]:
                    yp = model.predict_trial(vtr, fold_cache=fb["cache"])
                    oof_trials.append(vtr)
                    oof_preds.append(yp)
                    oof_fold_ids.append(int(fb["fold_idx"]))

            save_CompactPredictions(
                out_path=pred_file,
                trials=oof_trials,
                y_preds=oof_preds,
                train_fold_animal_ids=train_fold_session_ids,
                fold_ids=np.array(oof_fold_ids, dtype=np.int32),
            )
            dt_w = time.time() - t_m0
            dt_c = time.process_time() - c_m0
            model_wall[mid] += dt_w
            model_cpu[mid] += dt_c
            model_pred_files[mid][ds_id] = str(pred_file.relative_to(exp_root))
            print(f"[M3R1]   {mid} on {ds_id}: {dt_w:.2f}s (RSS={_peak_rss_mb():.1f} MB)", flush=True)

        del fold_bundles, train_fold_session_ids
        gc.collect()
        print(f"[M3R1] Finished {ds_id} in {time.time() - t_ds0:.1f}s (post-GC RSS={_peak_rss_mb():.1f} MB)", flush=True)

    # Step 2: Score EVERY run via standalone `evaluators/scorer.py` (one dataset at a time, no redundant ref reload!)
    print("[M3R1] Scoring all 4 runs with standalone evaluators/scorer.py...", flush=True)
    scored_by_model: Dict[str, Dict[str, Any]] = {s["model_id"]: {} for s in M3R1_MODEL_SPECS}
    ref_b4_scored: Dict[str, Any] = {}

    for ds_id in datasets:
        t_s0 = time.time()
        b4_path = exp_root / "runs" / "r0_04_b4_ridge_self_stim" / f"predictions_dev_{ds_id}.npz"
        ref_b4_scored[ds_id] = score_saved_predictions_file(
            pred_path=b4_path,
            split="dev",
            ref_pred_path=None,
            data_root=data_root,
            exp_root=exp_root,
            allow_vault_unseal=False,
            resolve_registry_ref=False,
        )
        for spec in M3R1_MODEL_SPECS:
            mid = spec["model_id"]
            p_path = exp_root / model_pred_files[mid][ds_id]
            sc = score_saved_predictions_file(
                pred_path=p_path,
                split="dev",
                ref_pred_path=None,
                data_root=data_root,
                exp_root=exp_root,
                allow_vault_unseal=False,
                resolve_registry_ref=False,
            )
            scored_by_model[mid][ds_id] = sc
        gc.collect()
        print(f"[M3R1]   Scored {ds_id} (B4 + 4 models) in {time.time() - t_s0:.1f}s (RSS={_peak_rss_mb():.1f} MB)", flush=True)

    # Step 3: Compute pairwise comparisons vs B4_ridge_self_stim, M3R1_01, and M3R1_02 with 1000-resample bootstrap CIs
    m01_id = "M3R1_01_self_anchor_multiscale"
    m02_id = "M3R1_02_h1_confounds_only"
    summary_models: Dict[str, Any] = {}
    run_jsonl_entries: List[Dict[str, Any]] = []

    for spec in M3R1_MODEL_SPECS:
        mid = spec["model_id"]
        rid = spec["run_id"]
        ds_out: Dict[str, Any] = {}
        for ds_id in datasets:
            sc = scored_by_model[mid][ds_id]
            cmp_b4 = compare_against_reference(sc, ref_b4_scored[ds_id], n_boot=n_boot, seed=int(hp["seed"]))
            cmp_01 = compare_against_reference(sc, scored_by_model[m01_id][ds_id], n_boot=n_boot, seed=int(hp["seed"]))
            cmp_02 = compare_against_reference(sc, scored_by_model[m02_id][ds_id], n_boot=n_boot, seed=int(hp["seed"]))
            ds_out[ds_id] = {
                "E_d": round(float(sc["E_d"]), 6),
                "rmse_raw": round(float(sc["rmse_raw"]), 6),
                "post_R2": round(float(sc["post_R2"]), 6),
                "post_R2_ci95": [round(float(x), 6) for x in sc["post_R2_ci95"]],
                "delta_R2": round(float(sc["delta_R2"]), 6),
                "delta_R2_ci95": [round(float(x), 6) for x in sc["delta_R2_ci95"]],
                "buffer_E_d": round(float(sc["buffer_E_d"]), 6) if sc.get("buffer_E_d") is not None else None,
                "n_worms_total": int(sc["n_worms_total"]),
                "vs_B4_ridge_self_stim": {
                    "I_d": round(float(cmp_b4["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_b4["I_d_ci95"]],
                    "n_worms_improved": int(cmp_b4["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_b4["fraction_worms_improved"]), 4),
                },
                "vs_M3R1_01_self_anchor_multiscale": {
                    "I_d": round(float(cmp_01["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_01["I_d_ci95"]],
                    "n_worms_improved": int(cmp_01["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_01["fraction_worms_improved"]), 4),
                },
                "vs_M3R1_02_h1_confounds_only": {
                    "I_d": round(float(cmp_02["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_02["I_d_ci95"]],
                    "n_worms_improved": int(cmp_02["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_02["fraction_worms_improved"]), 4),
                },
                "temporal_horizon_errors": {
                    k: round(float(v), 6) if v is not None else None
                    for k, v in sc.get("temporal_horizon_errors", {}).items()
                },
                "cell_class_errors": {
                    k: round(float(v), 6) for k, v in sc.get("cell_class_errors", {}).items()
                },
                "sex_errors": {
                    k: round(float(v), 6) for k, v in sc.get("sex_errors", {}).items()
                },
                "stimulus_errors": {
                    k: round(float(v), 6) for k, v in sc.get("stimulus_errors", {}).items()
                },
            }

        macro_b4_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_b4_scored, main_sources, n_boot=n_boot, seed=int(hp["seed"]))
        macro_b4_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_b4_scored, datasets, n_boot=n_boot, seed=int(hp["seed"]))
        macro_01_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m01_id], main_sources, n_boot=n_boot, seed=int(hp["seed"]))
        macro_01_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m01_id], datasets, n_boot=n_boot, seed=int(hp["seed"]))
        macro_02_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m02_id], main_sources, n_boot=n_boot, seed=int(hp["seed"]))
        macro_02_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m02_id], datasets, n_boot=n_boot, seed=int(hp["seed"]))

        worst_main_b4 = min(ds_out[d]["vs_B4_ridge_self_stim"]["I_d"] for d in main_sources)

        summary_models[mid] = {
            "run_id": rid,
            "parent_id": spec["parent_id"],
            "is_self_only": spec["is_self_only"],
            "uses_confounds": spec["uses_confounds"],
            "uses_network_state": spec["uses_network_state"],
            "is_stochastic": spec["is_stochastic"],
            "hypothesis": spec["hypothesis"],
            "I_macro_main2_vs_B4": macro_b4_main2["I_macro"],
            "I_macro_main2_vs_B4_ci95": macro_b4_main2["I_macro_ci95"],
            "I_macro_all3_vs_B4": macro_b4_all3["I_macro"],
            "I_macro_all3_vs_B4_ci95": macro_b4_all3["I_macro_ci95"],
            "I_macro_main2_vs_M3R1_01": macro_01_main2["I_macro"],
            "I_macro_main2_vs_M3R1_01_ci95": macro_01_main2["I_macro_ci95"],
            "I_macro_all3_vs_M3R1_01": macro_01_all3["I_macro"],
            "I_macro_all3_vs_M3R1_01_ci95": macro_01_all3["I_macro_ci95"],
            "I_macro_main2_vs_M3R1_02": macro_02_main2["I_macro"],
            "I_macro_main2_vs_M3R1_02_ci95": macro_02_main2["I_macro_ci95"],
            "I_macro_all3_vs_M3R1_02": macro_02_all3["I_macro"],
            "I_macro_all3_vs_M3R1_02_ci95": macro_02_all3["I_macro_ci95"],
            "worst_main_source_I_d_vs_B4": round(float(worst_main_b4), 6),
            "prediction_files_dev": model_pred_files[mid],
            "wall_clock_sec": round(model_wall[mid], 3),
            "cpu_time_sec": round(model_cpu[mid], 3),
            "datasets": ds_out,
        }

        # Write per-run metrics.json into runs/<run_id>/
        rdir = runs_dir / rid
        (rdir / "metrics_dev.json").write_text(json.dumps(summary_models[mid], indent=2), encoding="utf-8")

        # Prepare runs.jsonl record matching exact Round 0 schema
        src_metrics_jsonl = {}
        for ds_id in datasets:
            d_m = ds_out[ds_id]
            src_metrics_jsonl[ds_id] = {
                "E_d": d_m["E_d"],
                "I_d": d_m["vs_B4_ridge_self_stim"]["I_d"],
                "I_d_ci95": d_m["vs_B4_ridge_self_stim"]["I_d_ci95"],
                "post_R2": d_m["post_R2"],
                "post_R2_ci95": d_m["post_R2_ci95"],
                "delta_R2": d_m["delta_R2"],
                "delta_R2_ci95": d_m["delta_R2_ci95"],
                "rmse_raw": d_m["rmse_raw"],
                "n_worms_improved": d_m["vs_B4_ridge_self_stim"]["n_worms_improved"],
                "n_worms_total": d_m["n_worms_total"],
            }

        run_jsonl_entries.append(
            {
                "run_id": rid,
                "parent_id": spec["parent_id"],
                "round": 1,
                "category": "M3_round1_H1",
                "model_id": mid,
                "family_id": "M3R1_H1",
                "is_self_only": spec["is_self_only"],
                "hypothesis": spec["hypothesis"],
                "single_factor_changed": spec["single_factor_changed"],
                "registered_before_training_utc": model_reg_ts[mid],
                "completed_at_utc": _utc_now(),
                "config_id": cfg["config_id"],
                "seed": int(hp["seed"]),
                "checksums": {
                    "protocol_sha256": protocol_sha,
                    "lambda_global_sha256": config_sha,
                    "split_manifest_sha256": split_sha,
                    "code_bundle_sha256": code_sha,
                },
                "reference_baseline_id": "B4_ridge_self_stim",
                "I_macro": macro_b4_main2["I_macro"],
                "I_macro_ci95": macro_b4_main2["I_macro_ci95"],
                "I_macro_all3_immobilized": macro_b4_all3["I_macro"],
                "I_macro_all3_ci95": macro_b4_all3["I_macro_ci95"],
                "I_macro_vs_M3R1_01": macro_01_main2["I_macro"],
                "I_macro_vs_M3R1_02": macro_02_main2["I_macro"],
                "worst_main_source_I_d": round(float(worst_main_b4), 6),
                "source_metrics": src_metrics_jsonl,
                "prediction_files_dev": model_pred_files[mid],
                "wall_clock_sec": round(model_wall[mid], 3),
                "cpu_time_sec": round(model_cpu[mid], 3),
                "peak_rss_mb": _peak_rss_mb(),
                "exit_code": 0,
                "status": "completed",
                "decision": "RETAINED_M3_R1_DIAGNOSTIC",
            }
        )

    # Step 4: Explicit leakage invariance verification on M3R1_03_h1_slow_state_circuit
    print("[M3R1] Running leakage invariance check on M3R1_03_h1_slow_state_circuit...", flush=True)
    invariance_res = verify_m3_r1_invariance(
        model_id="M3R1_03_h1_slow_state_circuit",
        dataset_id="yemini2021",
        data_root=data_root,
        exp_root=exp_root,
    )
    if not invariance_res["passed"]:
        raise RuntimeError(f"Leakage invariance FAILED on M3R1_03_h1_slow_state_circuit: {invariance_res}")
    print(
        f"[M3R1] Invariance PASSED: future_diff={invariance_res['future_poison_max_abs_diff']:.2e}, "
        f"oof_diff={invariance_res['oof_tamper_weight_diff']:.2e}, "
        f"bilat_AIY={invariance_res['bilateral_exclusion_AIYR_to_AIYL_diff']:.2e}",
        flush=True,
    )

    # Step 5: Append to runs.jsonl (idempotent by run_id)
    runs_jsonl_path = exp_root / "runs.jsonl"
    existing_lines: List[str] = []
    new_rids = {e["run_id"] for e in run_jsonl_entries}
    if runs_jsonl_path.exists():
        for line in runs_jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("run_id") in new_rids:
                    continue
            except Exception:
                pass
            existing_lines.append(line)
    for entry in run_jsonl_entries:
        existing_lines.append(json.dumps(entry))
    runs_jsonl_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")

    # Step 6: Save results/m3_round1_h1_summary.json
    summary_payload = {
        "schema_version": "1.0.0",
        "round": 1,
        "milestone": "M3_round1_H1",
        "created_at_utc": _utc_now(),
        "config_id": cfg["config_id"],
        "lambda_global_path": "configs/lambda_global.json",
        "lambda_global_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "split_manifest_sha256": split_sha,
        "code_bundle_sha256": code_sha,
        "n_bootstrap_resamples": n_boot,
        "reference_baseline_id": "B4_ridge_self_stim",
        "reference_baseline_E_d": {d: round(float(ref_b4_scored[d]["E_d"]), 6) for d in datasets},
        "invariance_verification": invariance_res,
        "models": summary_models,
        "total_wall_clock_sec": round(time.time() - t_start, 3),
        "peak_rss_mb": _peak_rss_mb(),
    }
    out_summary = exp_root / "results" / "m3_round1_h1_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(
        f"[M3R1] Complete in {summary_payload['total_wall_clock_sec']:.1f}s "
        f"(Peak RSS={summary_payload['peak_rss_mb']:.1f} MB). Saved {out_summary}",
        flush=True,
    )
    return summary_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run M3 Round 1 (H1 slow-state + 7 confound controls)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing saved predictions_dev_*.npz if present")
    args = parser.parse_args()
    run_m3_round1_h1(resume=bool(args.resume))
