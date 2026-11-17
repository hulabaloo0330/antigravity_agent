"""
R3 Minimal Falsifiable Pre-Stimulus State Probe (probe_prestim_v1)
==================================================================
Tests whether strictly causal pre-stimulus neural/network state improves
cross-animal / cross-date out-of-sample prediction of post-stimulus neural
responses over stimulus-only and un-gated superposition baselines, strictly
isolated by experimental modality:

  - P1 (immobilized, 5 Hz): Yang et al. (2022) single-neuron/circuit calcium
    under IAA, 2-nonanone, and mixture (Leave-One-Date-Out CV).
  - P2 (freely_moving, ~1.7 Hz): Atanas et al. (2023) acute heat pulse (30 worms)
    and Dag et al. (2023) food patch encounter (6 worms) (Leave-One-Worm-Out CV).
  - P3 (immobilized, 4 Hz): Yemini et al. (2021) NeuroPAL whole-brain head
    recordings (21 worms x 3 chemical stimuli: butanone, pentanedione, NaCl)
    (Leave-One-Worm-Out CV).

Hierarchy of 5 strictly causal out-of-sample predictors on post-stimulus level `y_post`
(normalized solely by disjoint early pre-stimulus anchor window to prevent shared-denominator artifacts):
  - B0 `zero_change` (persistence):      y_hat = y_pre
  - B1 `stim_only_mean`:                 y_hat = mu_post_train(neuron, stim)  [ignores pre-stim state]
  - B2 `population_mean_response`:       y_hat = y_pre + mu_delta_train(neuron, stim) [un-gated additive shift]
  - M1 `ridge_prestim` (NO backprop):    y_hat = mu_post_train + Ridge(X_state, mu_post_train, mu_delta_train, state x stim interactions)
  - M2 `mlp_prestim_backprop` (Backprop): 2-layer ReLU MLP trained via Adam + backpropagation + L2 weight decay on the same features.
"""

from __future__ import annotations
import csv
import hashlib
import json
import math
import pathlib
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

DEPS_DIR = r"C:\projects\bio\source_data\behavior_forecast_5papers_2026-09-15\.deps"
if DEPS_DIR not in sys.path:
    sys.path.insert(0, DEPS_DIR)

import numpy as np
import scipy.io as sio
import scipy.linalg as sla

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(r"C:\projects\bio")
EXP_DIR = ROOT / "experiments" / "probe_prestim_v1"
EXT_DIR = EXP_DIR / "external_data" / "yemini2021"
RES_DIR = EXP_DIR / "results"
EXT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20261117
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
MLP_DECAYS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


# ============================================================================
# 1. Pure-NumPy Models: Closed-Form Ridge (No Backprop) & 2-Layer MLP (Backprop)
# ============================================================================

class TrainOnlyStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "TrainOnlyStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form Ridge solution w = (X^T X + alpha I)^{-1} X^T (y - y_mean). Zero backprop."""
    y_mean = float(np.mean(y))
    yc = y - y_mean
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d, dtype=np.float64)
    b = X.T @ yc
    w = sla.solve(A, b, assume_a="pos")
    return w, y_mean


def predict_ridge(X: np.ndarray, w: np.ndarray, y_mean: float) -> np.ndarray:
    return X @ w + y_mean


def fit_mlp_backprop(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    weight_decay: float = 1e-1,
    hidden_dim: int = 16,
    lr: float = 0.008,
    max_epochs: int = 150,
    seed: int = RNG_SEED,
) -> dict[str, np.ndarray | float]:
    """2-layer ReLU MLP trained via explicit forward pass + backpropagation + Adam."""
    rng = np.random.default_rng(seed)
    n, d = X_tr.shape
    y_mean = float(np.mean(y_tr))
    yc = (y_tr - y_mean).reshape(-1, 1)

    W1 = rng.normal(0.0, math.sqrt(2.0 / max(d, 1)), size=(d, hidden_dim))
    b1 = np.zeros((1, hidden_dim), dtype=np.float64)
    W2 = rng.normal(0.0, math.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1))
    b2 = np.zeros((1, 1), dtype=np.float64)

    params = [W1, b1, W2, b2]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val = float("inf")
    best_snapshot = [p.copy() for p in params]
    patience, wait = 20, 0

    batch_size = min(128, max(16, n))
    step = 0
    for _epoch in range(max_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr[idx]
            yb = yc[idx]
            bs = float(len(idx))

            # Forward pass
            z1 = xb @ W1 + b1
            h1 = np.maximum(0.0, z1)
            pred = h1 @ W2 + b2
            err = (pred - yb) * (2.0 / bs)

            # Backpropagation (chain rule)
            gW2 = h1.T @ err + weight_decay * W2
            gb2 = np.sum(err, axis=0, keepdims=True)
            dh1 = err @ W2.T
            dz1 = dh1 * (z1 > 0.0)
            gW1 = xb.T @ dz1 + weight_decay * W1
            gb1 = np.sum(dz1, axis=0, keepdims=True)

            grads = [gW1, gb1, gW2, gb2]
            step += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * g
                v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * (g * g)
                m_hat = m_adam[i] / (1.0 - beta1**step)
                v_hat = v_adam[i] / (1.0 - beta2**step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_pred = np.maximum(0.0, X_val @ W1 + b1) @ W2 + b2 + y_mean
            val_mse = float(np.mean((val_pred.ravel() - y_val) ** 2))
            if val_mse < best_val - 1e-7:
                best_val = val_mse
                best_snapshot = [p.copy() for p in params]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            best_snapshot = [p.copy() for p in params]

    return {
        "W1": best_snapshot[0],
        "b1": best_snapshot[1],
        "W2": best_snapshot[2],
        "b2": best_snapshot[3],
        "y_mean": y_mean,
    }


def predict_mlp(X: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    h1 = np.maximum(0.0, X @ model["W1"] + model["b1"])
    out = h1 @ model["W2"] + model["b2"] + model["y_mean"]
    return out.ravel()


# ============================================================================
# 2. Grouped Cross-Validation, Residual Design Matrix & Statistical Tests
# ============================================================================

def make_group_folds(groups: np.ndarray, max_folds: int = 25) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = sorted(set(groups.tolist()))
    n_g = len(uniq)
    assert n_g >= 3, f"Need >=3 independent groups for grouped CV, got {n_g}"
    if n_g <= max_folds:
        return [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in uniq]
    buckets = [uniq[i::max_folds] for i in range(max_folds)]
    folds = []
    for b in buckets:
        bset = set(b)
        mask = np.array([g in bset for g in groups], dtype=bool)
        folds.append((np.where(~mask)[0], np.where(mask)[0]))
    return folds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 1e-12 else 0.0
    if np.std(y_true) > 1e-10 and np.std(y_pred) > 1e-10:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = 0.0
    return {"rmse": rmse, "mse": mse, "r2": r2, "pearson_r": r}


def group_stats_comparison(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_base: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 2000,
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> dict[str, float | list[float] | int]:
    """Group-level paired permutation test & group-cluster bootstrap 95% CI on MSE reduction & R2 gain."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    g_err_m = np.array([np.mean((y_true[groups == g] - pred_model[groups == g]) ** 2) for g in uniq])
    g_err_b = np.array([np.mean((y_true[groups == g] - pred_base[groups == g]) ** 2) for g in uniq])
    diffs = g_err_b - g_err_m  # positive => model has LOWER MSE than baseline
    obs_mean_diff = float(np.mean(diffs))

    n_g = len(uniq)
    if n_g <= 12:
        signs = np.array([[1.0 if (mask >> j) & 1 else -1.0 for j in range(n_g)] for mask in range(1 << n_g)])
        perm_means = (signs @ diffs) / n_g
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n_g))
        perm_means = (signs @ diffs) / n_g
    p_two_sided = float((np.sum(np.abs(perm_means) >= abs(obs_mean_diff) - 1e-15) + 1) / (len(perm_means) + 1))

    idx_by_g = [np.where(groups == g)[0] for g in uniq]
    boot_r2_gain = np.empty(n_boot, dtype=np.float64)
    boot_rmse_diff = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n_g, size=n_g)
        b_idx = np.concatenate([idx_by_g[c] for c in chosen])
        yt = y_true[b_idx]
        m_mod = compute_metrics(yt, pred_model[b_idx])
        m_bas = compute_metrics(yt, pred_base[b_idx])
        boot_r2_gain[b] = m_mod["r2"] - m_bas["r2"]
        boot_rmse_diff[b] = m_bas["rmse"] - m_mod["rmse"]

    return {
        "group_mse_reduction_mean": obs_mean_diff,
        "paired_perm_p_two_sided": p_two_sided,
        "r2_gain_boot_ci95": [float(np.percentile(boot_r2_gain, 2.5)), float(np.percentile(boot_r2_gain, 97.5))],
        "rmse_reduction_boot_ci95": [float(np.percentile(boot_rmse_diff, 2.5)), float(np.percentile(boot_rmse_diff, 97.5))],
        "n_groups_improved": int(np.sum(diffs > 0)),
        "n_groups_total": int(n_g),
    }


def compute_train_group_means(
    keys: list[str], y_post: np.ndarray, y_delta: np.ndarray, idx: np.ndarray
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Fits per-(neuron, stim) mean post level and mean delta strictly on `idx` (shrinkage-free & zero test leakage)."""
    g_post = float(np.mean(y_post[idx]))
    g_delta = float(np.mean(y_delta[idx]))
    sums_p: dict[str, float] = {}
    sums_d: dict[str, float] = {}
    cnts: dict[str, int] = {}
    for i in idx:
        k = keys[i]
        sums_p[k] = sums_p.get(k, 0.0) + float(y_post[i])
        sums_d[k] = sums_d.get(k, 0.0) + float(y_delta[i])
        cnts[k] = cnts.get(k, 0) + 1
    # Empirical Bayes James-Stein style mild shrinkage (tau=2) for rare neuron-stim keys
    tau = 2.0
    mu_p = {k: (sums_p[k] + tau * g_post) / (cnts[k] + tau) for k in cnts}
    mu_d = {k: (sums_d[k] + tau * g_delta) / (cnts[k] + tau) for k in cnts}
    return mu_p, mu_d, g_post, g_delta


def build_state_design(
    keys: list[str],
    X_state: np.ndarray,
    y_pre: np.ndarray,
    mu_p: dict[str, float],
    mu_d: dict[str, float],
    g_post: float,
    g_delta: float,
    sub_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds compact, low-dimensional state-gating feature matrix strictly using
    training-fold stimulus priors `mu_p` & `mu_d` (zero high-dimensional one-hot bloat!).
    Returns (X_design, base_stim_post, base_add_post).
    """
    m = len(sub_idx)
    s_post = np.array([mu_p.get(keys[i], g_post) for i in sub_idx], dtype=np.float64)
    s_delta = np.array([mu_d.get(keys[i], g_delta) for i in sub_idx], dtype=np.float64)
    pre_sub = y_pre[sub_idx]
    st_sub = X_state[sub_idx]

    # Deviation of pre-stimulus state from expected stimulus post-level, plus multiplicative valence gating
    # (captures Gordus 2015 / Ray 2025 mechanism: inhibitory stimulus can only suppress if pre-state is high!)
    k_int = min(4, st_sub.shape[1])
    gating_terms = s_delta[:, None] * st_sub[:, :k_int]
    abs_gating = np.abs(s_delta)[:, None] * st_sub[:, :2]
    X_des = np.column_stack([
        pre_sub,
        s_post,
        s_delta,
        pre_sub - s_post,
        st_sub,
        gating_terms,
        abs_gating,
    ])
    return X_des, s_post, pre_sub + s_delta


def evaluate_cohort(
    cohort_name: str,
    modality: str,
    split_scheme: str,
    X_state: np.ndarray,
    stim_keys: list[str],
    y_delta: np.ndarray,
    y_pre_level: np.ndarray,
    groups: np.ndarray,
    causal_cutoff_desc: str,
    tamper_fn,
) -> dict:
    """
    Strict grouped CV comparing:
      - B0 `zero_change`:             y_post_hat = y_pre
      - B1 `stim_only_mean`:          y_post_hat = mu_post_train(neuron, stim)
      - B2 `population_mean_response`: y_post_hat = y_pre + mu_delta_train(neuron, stim)  [= mu_delta on y_delta!]
      - M1 `ridge_prestim`:           closed-form Ridge residual over B2
      - M2 `mlp_prestim_backprop`:    2-layer ReLU MLP (backprop + Adam) residual over B2
    Notice: Because M1 and M2 predict `y_post = base_add + r_hat`, as regularization -> inf, `r_hat -> 0`,
    so B2 (`population_mean_response` on delta) is strictly nested inside M1/M2 without shrinking stimulus intercepts!
    """
    n = len(y_delta)
    y_post = y_pre_level + y_delta
    folds = make_group_folds(groups)

    pred_b0_post = np.zeros(n, dtype=np.float64)
    pred_b1_post = np.zeros(n, dtype=np.float64)
    pred_b2_post = np.zeros(n, dtype=np.float64)
    pred_ridge_post = np.zeros(n, dtype=np.float64)
    pred_mlp_post = np.zeros(n, dtype=np.float64)
    chosen_alphas = []
    chosen_decays = []

    for fold_i, (tr, te) in enumerate(folds):
        assert set(groups[tr].tolist()).isdisjoint(set(groups[te].tolist())), "Group leakage!"

        mu_p, mu_d, g_p, g_d = compute_train_group_means(stim_keys, y_post, y_delta, tr)
        X_tr_raw, b1_tr, b2_tr = build_state_design(stim_keys, X_state, y_pre_level, mu_p, mu_d, g_p, g_d, tr)
        X_te_raw, b1_te, b2_te = build_state_design(stim_keys, X_state, y_pre_level, mu_p, mu_d, g_p, g_d, te)

        pred_b0_post[te] = y_pre_level[te]
        pred_b1_post[te] = b1_te
        pred_b2_post[te] = b2_te

        # Residual target relative to B2 (`y_post - (y_pre + mu_delta) == y_delta - mu_delta`)
        res_tr = y_post[tr] - b2_tr

        # Inner grouped CV on training groups ONLY to tune Ridge alpha & MLP weight decay
        tr_groups = groups[tr]
        uniq_trg = sorted(set(tr_groups.tolist()))
        inner_folds = make_group_folds(tr_groups, max_folds=min(5, len(uniq_trg))) if len(uniq_trg) >= 3 else [(np.arange(len(tr)), np.arange(len(tr)))]

        best_alpha, best_alpha_mse = 1000.0, float("inf")
        for alpha in RIDGE_ALPHAS:
            in_errs = []
            for itr, ival in inner_folds:
                tr_sub_idx = tr[itr]
                val_sub_idx = tr[ival]
                mp_i, md_i, gp_i, gd_i = compute_train_group_means(stim_keys, y_post, y_delta, tr_sub_idx)
                Xi_tr, _, b2i_tr = build_state_design(stim_keys, X_state, y_pre_level, mp_i, md_i, gp_i, gd_i, tr_sub_idx)
                Xi_val, _, b2i_val = build_state_design(stim_keys, X_state, y_pre_level, mp_i, md_i, gp_i, gd_i, val_sub_idx)
                sc_i = TrainOnlyStandardScaler().fit(Xi_tr)
                w_i, ym_i = fit_ridge_closed_form(sc_i.transform(Xi_tr), y_post[tr_sub_idx] - b2i_tr, alpha)
                p_i = b2i_val + predict_ridge(sc_i.transform(Xi_val), w_i, ym_i)
                in_errs.append(float(np.mean((y_post[val_sub_idx] - p_i) ** 2)))
            m_err = float(np.mean(in_errs))
            if m_err < best_alpha_mse:
                best_alpha_mse = m_err
                best_alpha = alpha
        chosen_alphas.append(best_alpha)

        scaler = TrainOnlyStandardScaler().fit(X_tr_raw)
        X_tr_s = scaler.transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)

        w_f, ym_f = fit_ridge_closed_form(X_tr_s, res_tr, best_alpha)
        pred_ridge_post[te] = b2_te + predict_ridge(X_te_s, w_f, ym_f)

        # Inner tune & fit 2-layer MLP with backpropagation
        itr0, ival0 = inner_folds[0]
        mp0, md0, gp0, gd0 = compute_train_group_means(stim_keys, y_post, y_delta, tr[itr0])
        X0_tr, _, b20_tr = build_state_design(stim_keys, X_state, y_pre_level, mp0, md0, gp0, gd0, tr[itr0])
        X0_val, _, b20_val = build_state_design(stim_keys, X_state, y_pre_level, mp0, md0, gp0, gd0, tr[ival0])
        sc0 = TrainOnlyStandardScaler().fit(X0_tr)
        X0_tr_s = sc0.transform(X0_tr)
        X0_val_s = sc0.transform(X0_val)
        res0_tr = y_post[tr[itr0]] - b20_tr
        res0_val = y_post[tr[ival0]] - b20_val

        best_wd, best_wd_mse = 1.0, float("inf")
        best_mlp = None
        for wd in MLP_DECAYS:
            mlp_cand = fit_mlp_backprop(
                X0_tr_s, res0_tr, X0_val_s, res0_val,
                weight_decay=wd, seed=RNG_SEED + fold_i
            )
            v_mse = float(np.mean((res0_val - predict_mlp(X0_val_s, mlp_cand)) ** 2))
            if v_mse < best_wd_mse:
                best_wd_mse = v_mse
                best_wd = wd
                best_mlp = mlp_cand
        chosen_decays.append(best_wd)
        assert best_mlp is not None
        # Predict on test using the inner-validated early-stopped model (using sc0 so input distribution matches!)
        pred_mlp_post[te] = b2_te + predict_mlp(sc0.transform(X_te_raw), best_mlp)

    # Future-tamper invariance verification
    X_state_tampered = tamper_fn()
    max_feat_diff = float(np.max(np.abs(X_state - X_state_tampered)))
    future_invariance_ok = (max_feat_diff == 0.0)

    # Convert post predictions to implied delta predictions (`y_post_hat - y_pre`)
    pred_b0_delta = pred_b0_post - y_pre_level
    pred_b1_delta = pred_b1_post - y_pre_level
    pred_b2_delta = pred_b2_post - y_pre_level
    pred_ridge_delta = pred_ridge_post - y_pre_level
    pred_mlp_delta = pred_mlp_post - y_pre_level

    m_post = {
        "zero_change": compute_metrics(y_post, pred_b0_post),
        "stim_only_post_mean": compute_metrics(y_post, pred_b1_post),
        "population_mean_response": compute_metrics(y_post, pred_b2_post),
        "ridge_prestim": compute_metrics(y_post, pred_ridge_post),
        "mlp_prestim_backprop": compute_metrics(y_post, pred_mlp_post),
    }
    m_delta = {
        "zero_change": compute_metrics(y_delta, pred_b0_delta),
        "stim_only_post_mean": compute_metrics(y_delta, pred_b1_delta),
        "population_mean_response": compute_metrics(y_delta, pred_b2_delta),
        "ridge_prestim": compute_metrics(y_delta, pred_ridge_delta),
        "mlp_prestim_backprop": compute_metrics(y_delta, pred_mlp_delta),
    }

    # Crucial dual comparisons:
    #   (A) Ridge(pre-stim + stim) vs Pure Stimulus-Only Post Mean (B1: ignores pre-stim state completely!)
    #   (B) Ridge(pre-stim + stim) vs Un-gated Additive Shift (B2: `y_pre + mu_delta`, tests nonlinear/damped state-gating!)
    comp_ridge_vs_stim_only = group_stats_comparison(y_post, pred_ridge_post, pred_b1_post, groups)
    comp_ridge_vs_pop_delta = group_stats_comparison(y_post, pred_ridge_post, pred_b2_post, groups)
    comp_ridge_vs_zero = group_stats_comparison(y_post, pred_ridge_post, pred_b0_post, groups)
    comp_mlp_vs_pop_delta = group_stats_comparison(y_post, pred_mlp_post, pred_b2_post, groups)
    comp_ridge_vs_mlp = group_stats_comparison(y_post, pred_ridge_post, pred_mlp_post, groups)

    # Decision rule on state-gating gain over BOTH B1 (stim-only) and B2 (un-gated additive shift)
    r2_gain_vs_b2 = m_delta["ridge_prestim"]["r2"] - m_delta["population_mean_response"]["r2"]
    p_vs_b2 = float(comp_ridge_vs_pop_delta["paired_perm_p_two_sided"])
    ci_lo_b2 = comp_ridge_vs_pop_delta["r2_gain_boot_ci95"][0]
    if r2_gain_vs_b2 > 0.02 and p_vs_b2 < 0.05 and ci_lo_b2 > 0.0:
        flag = "positive"
    elif r2_gain_vs_b2 < -0.02 and p_vs_b2 < 0.05:
        flag = "negative"
    else:
        flag = "null"

    return {
        "dataset_name": cohort_name,
        "modality": modality,
        "split_scheme": split_scheme,
        "n_groups": int(len(set(groups.tolist()))),
        "n_samples_or_trials": int(n),
        "n_features_design": int(X_tr_raw.shape[1]),
        "n_stim_categories": int(len(set(stim_keys))),
        "causal_cutoff_frame": causal_cutoff_desc,
        "baselines": {
            "zero_change": {"delta": m_delta["zero_change"], "post_level": m_post["zero_change"]},
            "stim_only_post_mean": {"delta": m_delta["stim_only_post_mean"], "post_level": m_post["stim_only_post_mean"]},
            "population_mean_response": {"delta": m_delta["population_mean_response"], "post_level": m_post["population_mean_response"]},
        },
        "models": {
            "ridge_prestim": {
                "delta": m_delta["ridge_prestim"],
                "post_level": m_post["ridge_prestim"],
                "median_alpha": float(np.median(chosen_alphas)),
            },
            "mlp_prestim_backprop": {
                "delta": m_delta["mlp_prestim_backprop"],
                "post_level": m_post["mlp_prestim_backprop"],
                "median_weight_decay": float(np.median(chosen_decays)),
            },
        },
        "comparisons": {
            "ridge_vs_stim_only_post_mean": comp_ridge_vs_stim_only,
            "ridge_vs_population_mean_delta": comp_ridge_vs_pop_delta,
            "ridge_vs_zero_change": comp_ridge_vs_zero,
            "mlp_backprop_vs_population_mean_delta": comp_mlp_vs_pop_delta,
            "ridge_vs_mlp_backprop": comp_ridge_vs_mlp,
        },
        "leakage_checks": {
            "future_tamper_invariance": future_invariance_ok,
            "future_tamper_max_diff": max_feat_diff,
            "scaler_fit_train_only": True,
            "group_disjoint_folds": True,
        },
        "conclusion_flag": flag,
    }


# ============================================================================
# 3. Cohort P1 Loader: Yang et al. (2022) Immobilized 5 Hz Odor Calcium
# ============================================================================

def col_letters_to_idx(col_str: str) -> int:
    idx = 0
    for ch in col_str:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def parse_yang_raw_sheet(xlsx_path: pathlib.Path, sheet_name: str) -> list[tuple[str, np.ndarray]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path) as zf:
        ss = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot:
                ss.append("".join(si.itertext()))
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = [el.attrib["name"] for el in wb.findall(f".//{ns}sheet")]
        s_idx = sheets.index(sheet_name) + 1
        wroot = ET.fromstring(zf.read(f"xl/worksheets/sheet{s_idx}.xml"))
        rows = wroot.findall(f".//{ns}row")

        first_formula_col = 10**6
        grid: dict[tuple[int, int], str] = {}
        max_row = 0
        for r in rows:
            r_num = int(r.attrib["r"])
            max_row = max(max_row, r_num)
            for c in r:
                ref = c.attrib.get("r", "")
                c_idx = col_letters_to_idx("".join(ch for ch in ref if ch.isalpha()))
                if c.find(f"{ns}f") is not None:
                    first_formula_col = min(first_formula_col, c_idx)
                    continue
                v_el = c.find(f"{ns}v")
                if v_el is None or v_el.text is None:
                    continue
                val = v_el.text
                if c.attrib.get("t") == "s" and val.isdigit():
                    val = ss[int(val)]
                grid[(r_num, c_idx)] = val

        traces = []
        cur_date = None
        for c_idx in range(first_formula_col):
            if (2, c_idx) not in grid:
                break
            hdr = grid.get((1, c_idx), "").strip()
            if hdr.endswith(".0"):
                hdr = hdr[:-2]
            if hdr.isdigit() and len(hdr) == 6:
                cur_date = hdr
            assert cur_date is not None
            vals = []
            valid_col = True
            for r_num in range(2, max_row + 1):
                sval = grid.get((r_num, c_idx))
                if sval is None:
                    valid_col = False
                    break
                try:
                    vals.append(float(sval))
                except ValueError:
                    valid_col = False
                    break
            if valid_col and len(vals) >= 200:
                traces.append((cur_date, np.array(vals, dtype=np.float64)))
        return traces


def extract_yang_features(raw_traces: list[tuple[str, str, str, np.ndarray]], tamper_future: bool = False):
    X_list, stim_keys, y_delta_list, y_pre_list, groups = [], [], [], [], []
    for date_str, neuron, stim, f_orig in raw_traces:
        f = f_orig.copy()
        if tamper_future:
            f[100:] = f[100:] + 1e6

        f_anchor = float(np.mean(f[0:40]))
        if abs(f_anchor) < 1e-6:
            continue
        df_early = (f - f_anchor) / f_anchor

        pre_win = df_early[50:100]
        pre_last2s = df_early[90:100]
        pre_prev2s = df_early[50:60]
        t_ax = np.linspace(-1.0, 1.0, 50)
        slope = float(np.polyfit(t_ax, pre_win, 1)[0])
        pre_mean = float(np.mean(pre_win))
        pre_std = float(np.std(pre_win))
        pre_min = float(np.min(pre_win))
        pre_max = float(np.max(pre_win))
        pre_momentum = float(np.mean(pre_last2s) - np.mean(pre_prev2s))
        log_raw_f0 = math.log(max(f_anchor, 1.0))

        X_list.append([pre_mean, float(pre_last2s.mean()), slope, pre_momentum, pre_std, pre_max - pre_min, log_raw_f0])
        stim_keys.append(f"{neuron}__{stim}")

        df_true = (f_orig - f_anchor) / f_anchor
        post_end = 150 if neuron in ("AWB", "ASH") else 125
        y_pre = float(np.mean(df_true[50:100]))
        y_post = float(np.mean(df_true[100:post_end]))
        y_pre_list.append(y_pre)
        y_delta_list.append(y_post - y_pre)
        groups.append(date_str)

    return (
        np.array(X_list, dtype=np.float64),
        stim_keys,
        np.array(y_delta_list, dtype=np.float64),
        np.array(y_pre_list, dtype=np.float64),
        np.array(groups),
    )


def run_p1_yang2022() -> list[dict]:
    yroot = ROOT / "research" / "circuit_feasibility_20260916" / "sources" / "yang2022"
    specs = [
        ("Fig5BCD_IAA.xlsx", "N2-I4", "AWB", "IAA"),
        ("Fig5EFG_Nona.xlsx", "N2", "AWB", "Nona"),
        ("Fig5IJK_Nona IAA.xlsx", "N2", "AWB", "Nona_IAA"),
        ("Fig6ACD.xlsx", "N2-ZC2905", "ASH", "Nona"),
        ("Fig6B.xlsx", "N2-ZC2905", "ASH", "Nona"),
        ("Fig6E.xlsx", "ASH", "ASH", "IAA"),
        ("Fig7ADG.xlsx", "AVA-N2", "AVA", "IAA"),
        ("Fig7ADG.xlsx", "AIB-N2", "AIB", "IAA"),
        ("Fig7ADG.xlsx", "RIM-N2", "RIM", "IAA"),
        ("Fig7BEH.xlsx", "AVA-N2", "AVA", "Nona"),
        ("Fig7BEH.xlsx", "AIB-N2", "AIB", "Nona"),
        ("Fig7BEH.xlsx", "RIM-N2", "RIM", "Nona"),
        ("Fig7CFI.xlsx", "AVA-N2", "AVA", "Nona_IAA"),
        ("Fig7CFI.xlsx", "AIB-N2", "AIB", "Nona_IAA"),
        ("Fig7CFI.xlsx", "RIM-N2", "RIM", "Nona_IAA"),
    ]
    seen_hashes: set[str] = set()
    all_wt: list[tuple[str, str, str, np.ndarray]] = []
    for fname, sname, neuron, stim in specs:
        cols = parse_yang_raw_sheet(yroot / fname, sname)
        for d_str, arr in cols:
            h = hashlib.sha256(arr.tobytes()).hexdigest()
            key_h = f"{neuron}:{stim}:{h}"
            if key_h in seen_hashes:
                continue
            seen_hashes.add(key_h)
            all_wt.append((d_str, neuron, stim, arr))

    sensory_wt = [x for x in all_wt if x[1] in ("AWB", "ASH")]
    inter_wt = [x for x in all_wt if x[1] in ("AVA", "AIB", "RIM")]

    out = []
    for name, subset in [
        ("P1a_yang2022_WT_all_5neurons", all_wt),
        ("P1b_yang2022_WT_sensory_AWB_ASH", sensory_wt),
        ("P1c_yang2022_WT_interneurons_AVA_AIB_RIM", inter_wt),
    ]:
        X, sk, yd, yp, grp = extract_yang_features(subset, tamper_future=False)
        out.append(evaluate_cohort(
            cohort_name=name,
            modality="immobilized",
            split_scheme="date_grouped_leave_one_date_out_cv",
            X_state=X,
            stim_keys=sk,
            y_delta=yd,
            y_pre_level=yp,
            groups=grp,
            causal_cutoff_desc="frame_100_at_5Hz_strictly_pre_onset",
            tamper_fn=lambda sub=subset: extract_yang_features(sub, tamper_future=True)[0],
        ))
    return out


# ============================================================================
# 4. Cohort P2 Loader: Atanas 2023 Heat & Dag 2023 PatchEncounter (Freely Moving)
# ============================================================================

def load_nf_v2_cohort(cond_keyword: str) -> list[dict]:
    nf_root = ROOT / "source_data" / "neural_forecast_v2_2026-09-15"
    records = []
    for mpath in sorted(nf_root.rglob("metadata.json")):
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        if not meta.get("training_eligible", False):
            continue
        if cond_keyword not in meta.get("condition", []):
            continue
        evs = meta.get("events_source_index") or {}
        if cond_keyword not in evs or not evs[cond_keyword]:
            continue
        rdir = mpath.parent
        sig = np.load(rdir / "signal.npy")
        val = np.load(rdir / "valid.npy")
        beh = np.load(rdir / "behavior.npy")
        beh_val = np.load(rdir / "behavior_valid.npy")
        neurons = json.loads((rdir / "neurons.json").read_text(encoding="utf-8"))
        records.append({
            "id": meta["source_record_id"],
            "event_frame": int(evs[cond_keyword][0]),
            "dt": float(meta.get("median_dt_s", 0.6)),
            "sig": sig.astype(np.float64),
            "val": val.astype(bool),
            "beh": beh.astype(np.float64),
            "beh_val": beh_val.astype(bool),
            "neurons": neurons,
        })
    return records


def extract_atanas_named_features(records: list[dict], shared_names: set[str], tamper_future: bool = False):
    X_list, stim_keys, yd_list, yp_list, groups = [], [], [], [], []
    for rec in records:
        ev = rec["event_frame"]
        t_cut = ev - 2
        if t_cut < 75 or ev + 21 > rec["sig"].shape[0]:
            continue
        sig = rec["sig"].copy()
        beh = rec["beh"].copy()
        if tamper_future:
            sig[t_cut:, :] = sig[t_cut:, :] + 1e6
            beh[t_cut:, :] = beh[t_cut:, :] + 1e6

        anchor_slice = sig[t_cut - 70 : t_cut - 20, :]
        has_fin = np.sum(np.isfinite(anchor_slice), axis=0) >= 20
        anchor_mean = np.zeros(sig.shape[1], dtype=np.float64)
        anchor_std = np.zeros(sig.shape[1], dtype=np.float64)
        if np.any(has_fin):
            anchor_mean[has_fin] = np.nanmean(anchor_slice[:, has_fin], axis=0)
            anchor_std[has_fin] = np.nanstd(anchor_slice[:, has_fin], axis=0)
        valid_anchor = has_fin & (anchor_std > 1e-6)
        if np.sum(valid_anchor) < 10:
            continue

        z_causal = np.zeros_like(sig)
        z_causal[:, valid_anchor] = (sig[:, valid_anchor] - anchor_mean[valid_anchor]) / anchor_std[valid_anchor]

        wb_pre = z_causal[t_cut - 20 : t_cut, valid_anchor]
        wb_mean = float(np.nanmean(wb_pre))
        wb_std = float(np.nanstd(wb_pre))
        wb_high_frac = float(np.nanmean(wb_pre > 1.0))

        b_slice = beh[t_cut - 20 : t_cut, 0]
        b_v = rec["beh_val"][t_cut - 20 : t_cut, 0]
        vel_pre = float(np.mean(b_slice[b_v])) if np.any(b_v) else 0.0
        rev_frac_pre = float(np.mean(b_slice[b_v] < 0.0)) if np.any(b_v) else 0.0

        z_true = np.zeros_like(rec["sig"])
        z_true[:, valid_anchor] = (rec["sig"][:, valid_anchor] - anchor_mean[valid_anchor]) / anchor_std[valid_anchor]

        for j, ninfo in enumerate(rec["neurons"]):
            if not valid_anchor[j]:
                continue
            lbl = ninfo.get("label") or ninfo.get("neuron_class") or ninfo.get("name") if isinstance(ninfo, dict) else str(ninfo)
            if not lbl or lbl not in shared_names:
                continue
            if np.mean(rec["val"][t_cut - 20 : t_cut, j]) < 0.8 or np.mean(rec["val"][ev + 1 : ev + 21, j]) < 0.8:
                continue
            n_pre_seq = np.where(np.isfinite(z_causal[t_cut - 20 : t_cut, j]), z_causal[t_cut - 20 : t_cut, j], 0.0)
            n_pre_mean = float(np.mean(n_pre_seq))
            n_pre_last5 = float(np.mean(n_pre_seq[-5:]))
            n_slope = float(np.polyfit(np.linspace(-1.0, 1.0, len(n_pre_seq)), n_pre_seq, 1)[0])
            n_std = float(np.std(n_pre_seq))

            X_list.append([n_pre_mean, n_pre_last5, n_slope, n_std, wb_mean, wb_std, wb_high_frac, vel_pre, rev_frac_pre])
            stim_keys.append(lbl)

            y_pre = float(np.nanmean(z_true[t_cut - 20 : t_cut, j]))
            y_post = float(np.nanmean(z_true[ev + 1 : ev + 21, j]))
            yp_list.append(y_pre)
            yd_list.append(y_post - y_pre)
            groups.append(rec["id"])

    return (
        np.array(X_list, dtype=np.float64),
        stim_keys,
        np.array(yd_list, dtype=np.float64),
        np.array(yp_list, dtype=np.float64),
        np.array(groups),
    )


def extract_pop_summary_features(records: list[dict], stim_tag: str, tamper_future: bool = False):
    X_list, stim_keys, yd_list, yp_list, groups = [], [], [], [], []
    for rec in records:
        ev = rec["event_frame"]
        t_cut = ev - 2
        if t_cut < 75 or ev + 21 > rec["sig"].shape[0]:
            continue
        sig = rec["sig"].copy()
        beh = rec["beh"].copy()
        if tamper_future:
            sig[t_cut:, :] = sig[t_cut:, :] + 1e6
            beh[t_cut:, :] = beh[t_cut:, :] + 1e6

        anchor_slice = sig[t_cut - 70 : t_cut - 20, :]
        has_fin = np.sum(np.isfinite(anchor_slice), axis=0) >= 20
        anchor_mean = np.zeros(sig.shape[1], dtype=np.float64)
        anchor_std = np.zeros(sig.shape[1], dtype=np.float64)
        if np.any(has_fin):
            anchor_mean[has_fin] = np.nanmean(anchor_slice[:, has_fin], axis=0)
            anchor_std[has_fin] = np.nanstd(anchor_slice[:, has_fin], axis=0)
        valid_anchor = has_fin & (anchor_std > 1e-6)
        if np.sum(valid_anchor) < 10:
            continue

        z_c = (sig[:, valid_anchor] - anchor_mean[valid_anchor]) / anchor_std[valid_anchor]
        z_t = (rec["sig"][:, valid_anchor] - anchor_mean[valid_anchor]) / anchor_std[valid_anchor]
        wb_pre = np.nan_to_num(z_c[t_cut - 20 : t_cut, :], nan=0.0)

        wb_mean = float(np.mean(wb_pre))
        wb_last5 = float(np.mean(wb_pre[-5:, :]))
        wb_std = float(np.std(wb_pre))
        wb_abs = float(np.mean(np.abs(wb_pre)))
        b_slice = beh[t_cut - 20 : t_cut, 0]
        b_v = rec["beh_val"][t_cut - 20 : t_cut, 0]
        vel_pre = float(np.mean(b_slice[b_v])) if np.any(b_v) else 0.0

        X_list.append([wb_mean, wb_last5, wb_std, wb_abs, vel_pre])
        stim_keys.append(stim_tag)

        y_pre = float(np.mean(np.abs(np.nan_to_num(z_t[t_cut - 20 : t_cut, :], nan=0.0))))
        y_post = float(np.mean(np.abs(np.nan_to_num(z_t[ev + 1 : ev + 21, :], nan=0.0))))
        yp_list.append(y_pre)
        yd_list.append(y_post - y_pre)
        groups.append(rec["id"])

    return (
        np.array(X_list, dtype=np.float64),
        stim_keys,
        np.array(yd_list, dtype=np.float64),
        np.array(yp_list, dtype=np.float64),
        np.array(groups),
    )


def run_p2_freely_moving() -> list[dict]:
    heat_recs = load_nf_v2_cohort("heat")
    patch_recs = load_nf_v2_cohort("patchEncounter")

    counts: dict[str, int] = {}
    for rec in heat_recs:
        seen_in_worm = set()
        for ninfo in rec["neurons"]:
            lbl = ninfo.get("label") or ninfo.get("neuron_class") or ninfo.get("name") if isinstance(ninfo, dict) else str(ninfo)
            if lbl and not lbl.isdigit() and not lbl.startswith("neuron_"):
                seen_in_worm.add(lbl)
        for lbl in seen_in_worm:
            counts[lbl] = counts.get(lbl, 0) + 1
    shared_heat_neurons = {k for k, c in counts.items() if c >= 8}

    out = []
    if len(shared_heat_neurons) >= 5:
        X, sk, yd, yp, grp = extract_atanas_named_features(heat_recs, shared_heat_neurons, tamper_future=False)
        out.append(evaluate_cohort(
            cohort_name="P2a_atanas2023_heat_named_neurons_19worms",
            modality="freely_moving",
            split_scheme="animal_grouped_leave_one_worm_out_cv",
            X_state=X,
            stim_keys=sk,
            y_delta=yd,
            y_pre_level=yp,
            groups=grp,
            causal_cutoff_desc="event_frame_minus_2_strictly_pre_heat",
            tamper_fn=lambda: extract_atanas_named_features(heat_recs, shared_heat_neurons, tamper_future=True)[0],
        ))

    X_p, sk_p, yd_p, yp_p, grp_p = [], [], [], [], []
    for tag, sub in [("heat", heat_recs), ("patchEncounter", patch_recs)]:
        Xi, ski, ydi, ypi, grpi = extract_pop_summary_features(sub, tag, tamper_future=False)
        X_p.append(Xi)
        sk_p.extend(ski)
        yd_p.append(ydi)
        yp_p.append(ypi)
        grp_p.append(grpi)
    out.append(evaluate_cohort(
        cohort_name="P2b_atanas_heat30_dag_patch6_wholebrain_arousal",
        modality="freely_moving",
        split_scheme="animal_grouped_leave_one_worm_out_cv",
        X_state=np.vstack(X_p),
        stim_keys=sk_p,
        y_delta=np.concatenate(yd_p),
        y_pre_level=np.concatenate(yp_p),
        groups=np.concatenate(grp_p),
        causal_cutoff_desc="event_frame_minus_2_strictly_pre_stimulus",
        tamper_fn=lambda: np.vstack([
            extract_pop_summary_features(heat_recs, "heat", tamper_future=True)[0],
            extract_pop_summary_features(patch_recs, "patchEncounter", tamper_future=True)[0],
        ]),
    ))
    return out


# ============================================================================
# 5. Cohort P3 Loader: Yemini et al. (2021) 21 NeuroPAL Worms x 3 Chemical Stimuli
# ============================================================================

def ensure_yemini_mat() -> pathlib.Path:
    mat_path = EXT_DIR / "all_traces_Heads_new.mat"
    if mat_path.exists() and mat_path.stat().st_size == 52362462:
        return mat_path
    url = "https://raw.githubusercontent.com/TuragaLab/wormvae/master/data/worm_activity/all_traces_Heads_new.mat"
    print(f"Downloading Yemini 2021 NeuroPAL dataset ({url}) ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    assert len(data) == 52362462, f"Unexpected byte size {len(data)}"
    mat_path.write_bytes(data)
    return mat_path


def load_yemini_data():
    mat_path = ensure_yemini_mat()
    tv = sio.loadmat(str(mat_path))
    trace_arr = tv["traces"]
    is_L = tv["is_L"]
    neurons_name = tv["neurons"]
    stim_times = tv["stim_times"]
    stims = tv["stims"]

    n_worms = 21
    n_cell = 189
    T = 960
    dt = 0.25
    activity = np.full((n_worms, n_cell, T), np.nan, dtype=np.float64)

    neuron_names = []
    for ifile in range(109):
        nm = str(neurons_name[ifile][0][0])
        if is_L[ifile][0][0].shape[0] == 42:
            neuron_names.append(nm + "L")
            neuron_names.append(nm + "R")
        else:
            neuron_names.append(nm)

    for w in range(n_worms):
        ineuron = 0
        for ifile in range(109):
            if trace_arr[ifile][0].shape[1] == 42:
                for offset in (0, 21):
                    d = trace_arr[ifile][0][0][w + offset]
                    if d.shape[0] >= 1 and d[0].shape[0] > 0:
                        L = min(T, d[0].shape[0])
                        activity[w, ineuron, :L] = d[0][:L]
                    ineuron += 1
            else:
                d = trace_arr[ifile][0][0][w]
                if d.shape[0] >= 1 and d[0].shape[0] > 0:
                    L = min(T, d[0].shape[0])
                    activity[w, ineuron, :L] = d[0][:L]
                ineuron += 1

    odor_names = ["butanone", "pentanedione", "NaCl"]
    return activity, neuron_names, stim_times, stims, odor_names, dt


def extract_yemini_features(activity, neuron_names, stim_times, stims, odor_names, dt, shared_mask, tamper_future=False):
    rev_set = {"AVAL", "AVAR", "RIML", "RIMR", "AIBL", "AIBR", "AVEL", "AVER"}
    fwd_set = {"AVBL", "AVBR", "RIBL", "RIBR"}
    rev_idx = [i for i, nm in enumerate(neuron_names) if nm in rev_set]
    fwd_idx = [i for i, nm in enumerate(neuron_names) if nm in fwd_set]

    X_list, stim_keys, yd_list, yp_list, groups = [], [], [], [], []
    n_worms = activity.shape[0]

    for w in range(n_worms):
        for k in range(stim_times.shape[0]):
            f_on = int(round(float(stim_times[k, 0]) / dt))
            if f_on < 80 or f_on + 40 > activity.shape[2]:
                continue
            act_w = activity[w].copy()
            if tamper_future:
                act_w[:, f_on:] = act_w[:, f_on:] + 1e6

            init_slice = act_w[:, 40 : min(160, f_on - 40)]
            has_obs = np.sum(np.isfinite(init_slice), axis=1) >= 20
            mu0 = np.zeros((act_w.shape[0], 1), dtype=np.float64)
            sd0 = np.zeros((act_w.shape[0], 1), dtype=np.float64)
            if np.any(has_obs):
                mu0[has_obs, 0] = np.nanmean(init_slice[has_obs], axis=1)
                sd0[has_obs, 0] = np.nanstd(init_slice[has_obs], axis=1)
            ok = has_obs & (sd0.ravel() > 1e-6)

            z_c = np.full_like(act_w, np.nan)
            z_c[ok] = (act_w[ok] - mu0[ok]) / sd0[ok]

            z_t = np.full_like(activity[w], np.nan)
            z_t[ok] = (activity[w, ok] - mu0[ok]) / sd0[ok]

            rev_pre = z_c[rev_idx, f_on - 40 : f_on]
            fwd_pre = z_c[fwd_idx, f_on - 40 : f_on]
            rev_state = float(np.nanmean(rev_pre)) if np.any(np.isfinite(rev_pre)) else 0.0
            fwd_state = float(np.nanmean(fwd_pre)) if np.any(np.isfinite(fwd_pre)) else 0.0
            wb_pre = z_c[ok, f_on - 40 : f_on]
            wb_mean = float(np.nanmean(wb_pre)) if np.any(np.isfinite(wb_pre)) else 0.0
            wb_std = float(np.nanstd(wb_pre)) if np.any(np.isfinite(wb_pre)) else 0.0

            odor_id = int(stims[w, k]) - 1
            odor_nm = odor_names[odor_id] if 0 <= odor_id < len(odor_names) else f"stim{odor_id}"

            for i_n, nm in enumerate(neuron_names):
                if not shared_mask[i_n] or not ok[i_n]:
                    continue
                pre_seq = z_c[i_n, f_on - 40 : f_on]
                post_seq_true = z_t[i_n, f_on : f_on + 40]
                if np.mean(np.isfinite(pre_seq)) < 0.9 or np.mean(np.isfinite(post_seq_true)) < 0.9:
                    continue
                pre_seq = np.nan_to_num(pre_seq, nan=0.0)
                n_mean = float(np.mean(pre_seq))
                n_last8 = float(np.mean(pre_seq[-8:]))
                n_slope = float(np.polyfit(np.linspace(-1.0, 1.0, len(pre_seq)), pre_seq, 1)[0])
                n_std = float(np.std(pre_seq))

                X_list.append([n_mean, n_last8, n_slope, n_std, rev_state, fwd_state, rev_state - fwd_state, wb_mean, wb_std])
                stim_keys.append(f"{nm}__{odor_nm}")

                y_pre = float(np.nanmean(z_t[i_n, f_on - 40 : f_on]))
                y_post = float(np.nanmean(post_seq_true))
                yp_list.append(y_pre)
                yd_list.append(y_post - y_pre)
                groups.append(f"worm_{w:02d}")

    return (
        np.array(X_list, dtype=np.float64),
        stim_keys,
        np.array(yd_list, dtype=np.float64),
        np.array(y_pre_list := yp_list, dtype=np.float64),
        np.array(groups),
    )


def run_p3_yemini2021() -> list[dict]:
    activity, neuron_names, stim_times, stims, odor_names, dt = load_yemini_data()
    obs_per_neuron = np.sum(np.any(np.isfinite(activity), axis=2), axis=0)
    shared_mask = obs_per_neuron >= 12
    core_circuit_names = {
        "AWCL", "AWCR", "AWAL", "AWAR", "ASEL", "ASER", "ASHL", "ASHR", "AWBL", "AWBR",
        "AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
        "AVAL", "AVAR", "AVBL", "AVBR", "AVEL", "AVER", "RIML", "RIMR", "RIBL", "RIBR"
    }
    core_mask = shared_mask & np.array([nm in core_circuit_names for nm in neuron_names], dtype=bool)
    inter_motor_names = {
        "AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
        "AVAL", "AVAR", "AVBL", "AVBR", "AVEL", "AVER", "RIML", "RIMR", "RIBL", "RIBR"
    }
    inter_mask = shared_mask & np.array([nm in inter_motor_names for nm in neuron_names], dtype=bool)

    out = []
    for cname, msk in [
        ("P3a_yemini2021_21worms_all_shared_head_neurons", shared_mask),
        ("P3b_yemini2021_21worms_chemosensory_command_circuit", core_mask),
        ("P3c_yemini2021_21worms_interneurons_only", inter_mask),
    ]:
        X, sk, yd, yp, grp = extract_yemini_features(
            activity, neuron_names, stim_times, stims, odor_names, dt, msk, tamper_future=False
        )
        out.append(evaluate_cohort(
            cohort_name=cname,
            modality="immobilized",
            split_scheme="animal_grouped_leave_one_worm_out_cv",
            X_state=X,
            stim_keys=sk,
            y_delta=yd,
            y_pre_level=yp,
            groups=grp,
            causal_cutoff_desc="f_on_minus_1_at_4Hz_strictly_pre_odor_onset",
            tamper_fn=lambda m=msk: extract_yemini_features(
                activity, neuron_names, stim_times, stims, odor_names, dt, m, tamper_future=True
            )[0],
        ))
    return out


# ============================================================================
# 6. Main Runner & Summary Artifact Writer
# ============================================================================

def main() -> None:
    t0 = time.time()
    print("[1/3] Running P1 (Yang 2022 immobilized 5 Hz odor calcium) ...")
    res_p1 = run_p1_yang2022()
    print("[2/3] Running P2 (Atanas 2023 heat + Dag 2023 patchEncounter freely moving) ...")
    res_p2 = run_p2_freely_moving()
    print("[3/3] Running P3 (Yemini 2021 21 NeuroPAL worms x 3 odors immobilized 4 Hz) ...")
    res_p3 = run_p3_yemini2021()

    all_cohorts = res_p1 + res_p2 + res_p3
    elapsed = round(time.time() - t0, 2)

    summary = {
        "experiment_id": "probe_prestim_v1",
        "verified_date": "2026-11-17",
        "elapsed_seconds": elapsed,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "cohorts": all_cohorts,
    }
    json_path = RES_DIR / "probe_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = RES_DIR / "probe_per_cohort_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "dataset_name", "modality", "n_groups", "n_samples",
            "b0_persist_post_r2", "b1_stim_only_post_r2", "b2_pop_delta_post_r2",
            "ridge_post_r2", "mlp_backprop_post_r2",
            "b2_pop_delta_r2", "ridge_delta_r2", "mlp_backprop_delta_r2",
            "r2_gain_ridge_vs_b2", "ci95_low_b2", "ci95_high_b2", "perm_p_vs_b2",
            "perm_p_vs_b1_stim_only", "future_tamper_invariance", "conclusion_flag"
        ])
        for c in all_cohorts:
            cb2 = c["comparisons"]["ridge_vs_population_mean_delta"]
            cb1 = c["comparisons"]["ridge_vs_stim_only_post_mean"]
            writer.writerow([
                c["dataset_name"], c["modality"], c["n_groups"], c["n_samples_or_trials"],
                round(c["baselines"]["zero_change"]["post_level"]["r2"], 4),
                round(c["baselines"]["stim_only_post_mean"]["post_level"]["r2"], 4),
                round(c["baselines"]["population_mean_response"]["post_level"]["r2"], 4),
                round(c["models"]["ridge_prestim"]["post_level"]["r2"], 4),
                round(c["models"]["mlp_prestim_backprop"]["post_level"]["r2"], 4),
                round(c["baselines"]["population_mean_response"]["delta"]["r2"], 4),
                round(c["models"]["ridge_prestim"]["delta"]["r2"], 4),
                round(c["models"]["mlp_prestim_backprop"]["delta"]["r2"], 4),
                round(c["models"]["ridge_prestim"]["delta"]["r2"] - c["baselines"]["population_mean_response"]["delta"]["r2"], 4),
                round(cb2["r2_gain_boot_ci95"][0], 4),
                round(cb2["r2_gain_boot_ci95"][1], 4),
                round(cb2["paired_perm_p_two_sided"], 5),
                round(cb1["paired_perm_p_two_sided"], 5),
                c["leakage_checks"]["future_tamper_invariance"],
                c["conclusion_flag"],
            ])

    print(f"\n=== ALL COHORTS COMPLETE in {elapsed}s ===")
    for c in all_cohorts:
        cb2 = c["comparisons"]["ridge_vs_population_mean_delta"]
        cb1 = c["comparisons"]["ridge_vs_stim_only_post_mean"]
        print(
            f"  {c['dataset_name']} [{c['modality']}, G={c['n_groups']}, N={c['n_samples_or_trials']}]:\n"
            f"    post_R2 : B0_persist={c['baselines']['zero_change']['post_level']['r2']:.4f} | "
            f"B1_stimOnly={c['baselines']['stim_only_post_mean']['post_level']['r2']:.4f} | "
            f"B2_addShift={c['baselines']['population_mean_response']['post_level']['r2']:.4f} | "
            f"M1_ridge={c['models']['ridge_prestim']['post_level']['r2']:.4f} | "
            f"M2_mlpBP={c['models']['mlp_prestim_backprop']['post_level']['r2']:.4f}\n"
            f"    delta_R2: B2_popMean={c['baselines']['population_mean_response']['delta']['r2']:.4f} | "
            f"M1_ridge={c['models']['ridge_prestim']['delta']['r2']:.4f} | "
            f"M2_mlpBP={c['models']['mlp_prestim_backprop']['delta']['r2']:.4f} | "
            f"p_vs_B2={cb2['paired_perm_p_two_sided']:.4g} (CI95={cb2['r2_gain_boot_ci95']}), "
            f"p_vs_B1={cb1['paired_perm_p_two_sided']:.4g} -> {c['conclusion_flag']}"
        )


if __name__ == "__main__":
    main()
