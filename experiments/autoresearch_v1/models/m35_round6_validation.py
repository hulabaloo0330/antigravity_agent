"""Milestone M3.5 — Round 6: Complete Multi-Dimensional Rigor Suite for Upgraded M3.5 Models.

Implements 5 resumable stages (`--stage {seeds,oat,robustness,null199,summarize}` + `--resume`)
across all 3 immobilized Dev splits (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`)
under unified `configs/lambda_global.json` (STRICTLY ZERO `dataset_id` hyperparameter branches,
ZERO access to `vault_sealed/`, processing 1 dataset at a time + `gc.collect()` so peak RSS < 3.0 GiB):

1. `--stage seeds`:
   5-seed multi-seed stability (`[20261117, 20261118, 20261119, 20261120, 20261121]`) for
   `M35_01_zeroinit_nonlinear_self_anchor`, `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film`,
   and `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl`. Reuses verified seed `20261117`
   predictions from Round 5 and trains 4 new seeds with precomputed deterministic fold caches.

2. `--stage oat`:
   `0.5x / 1x / 2x` one-at-a-time hyperparameter neighborhood stability (9 configs) perturbing
   4 core global hyperparameters of `M35_03` (`self_weight_decay`, `self_lr`, `film_weight_decay`,
   `film_lr`) across all 3 Dev splits (seed `20261117`), reporting fraction positive vs both
   `B4_ridge_self_stim` and `M35_01` (verifying `>= 80%` threshold) plus min/max of `I_macro`.

3. `--stage robustness`:
   (a) F11 Missingness dropout curve (`+0%`, `+15%`, `+30%`, `+50%` extra random non-target
       neuron masking, strictly causal, zero pre-split PPCA) for `B4_ridge_self_stim`, `M35_01`,
       and `M35_03` across all 3 Dev splits.
   (b) Strictly causal `[-5, 0)s` short-window check (zero frames before `-5.0s`) retraining
       `B4_ridge_self_stim`, `M35_01`, and `M35_03` from scratch across all 3 Dev splits.

4. `--stage null199`:
   (a) `K = 199` constrained retrained random null graphs (degree-, 3D-soma-distance-, and
       cell-type-preserving via `ConstrainedNullConnectomeGenerator`: `K = 199` full 8-layer nulls
       + `K = 199` `W_gap`-only nulls) genuinely retraining the 1-hop+2-hop signed connectome
       residual model from scratch on top of frozen `M35_01` nonlinear anchor residuals across all
       3 Dev splits, computing exact empirical one-sided `p = (1 + k) / (1 + 199)` and `z`-score.
   (b) `K_film = 19` genuine end-to-end `M35_02` Patch-FiLM retrainings on constrained null graphs
       `k = 1..19` on Main 2 (`seyedolmohadesin2025`, `toyoshima2024`).

5. `--stage summarize`:
   Consolidates Stages 1–4 into `results/m35_round6_full_validation.json` and appends Round 6
   to `research_loop_log.md`.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
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
from models.m3_round1_h1 import (
    M3R1Model,
    _build_m3_fold_cache,
    _peak_rss_mb,
    _utc_now,
    extract_circuit_slow_state_features,
    fit_causal_slow_state_pca,
)
from models.m3_round2_h2 import (
    _apply_svd_compressor,
    _build_bilinear_features,
    _fit_reduced_rank_ridge,
    _fit_svd_compressor,
)
from models.m3_round3_h3 import (
    _solve_weighted_ridge_sample_scaled,
    extract_multiscale_presynaptic_state,
)
from models.m3_round4_f11 import (
    DATASETS_ALL3,
    DATASETS_MAIN2,
    PRIMARY_SEED,
    SEEDS_5,
    _extract_fold_trials,
    make_causal_network_dropout_override,
    truncate_trial_to_short_window,
)
from models.m35_round5_nonlinear import (
    M35_MODEL_SPECS,
    TIER_NAMES,
    M35Model,
    ZeroInitNonlinearSelfNet,
    ZeroInitPatchFiLMGate,
    _extract_excluded_self_pop_pca_features,
    _fit_excluded_self_pop_pca,
    _train_torch_residual_with_inner_cv,
    build_1hop_2hop_connectome_atlas,
    build_patch_to_basis_matrix,
    extract_1hop_2hop_lagmatched_drives,
    extract_nonlinear_self_features,
)
from models.m4_null_graphs import (
    ConstrainedNullConnectomeGenerator,
    ensure_null_bank,
)
from src.baselines import (
    CausalBaselineModel,
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
    build_fold_feature_cache,
)
from src.data_adapters import DATA_ROOT_DEFAULT, EXP_ROOT_DEFAULT, load_dataset
from src.evaluator import (
    align_records_to_canonical,
    build_grouped_cv_splits,
    evaluate_predictions,
)
from src.features import predict_prior_delta_trajectory

SCRATCH_DEFAULT = Path("/home/antigravity/.gemini/jetski/scratch/teamwork_tmp/worker_m35_r6_full_validation")
DEFAULT_K_NULL = 199
DEFAULT_K_FILM = 19


def _get_scratch_dir() -> Path:
    env_tmp = os.environ.get("TMPDIR", "")
    p = Path(env_tmp) if env_tmp else SCRATCH_DEFAULT
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tier_onehot(functional_tiers: Sequence[str]) -> np.ndarray:
    n_canon = len(functional_tiers)
    tier_oh = np.zeros((n_canon, len(TIER_NAMES)), dtype=np.float32)
    for i, t_str in enumerate(functional_tiers):
        if t_str in TIER_NAMES:
            tier_oh[i, TIER_NAMES.index(t_str)] = 1.0
    return tier_oh


def _extract_u_stim(trial: Mapping[str, Any], s_n: np.ndarray, priors: Mapping[str, Any], n_canon: int) -> np.ndarray:
    stim_id = str(trial["stimulus_id"])
    is_buf = bool(trial.get("is_buffer_control", False))
    stim_oh = _encode_stim_onehot(stim_id, is_buf, priors["stim_vocab"])[:-1]
    stim_mat = np.repeat(stim_oh[None, :], n_canon, axis=0)
    _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, n_canon)
    prior_norm_2 = np.clip((prior_coef / np.maximum(s_n[:, None].astype(np.float32), 1e-6))[:, :2], -4.0, 4.0)
    return np.hstack([stim_mat, prior_norm_2]).astype(np.float32)


def _load_b4_and_r5_reference_eds(exp_root: Path, data_root: Path) -> Dict[str, Dict[str, float]]:
    """Load exact frozen Dev E_d for B4_ridge_self_stim and Round 5 primary seed (M35_01/02/03)."""
    r5_path = exp_root / "results" / "m35_round5_summary.json"
    r5 = json.loads(r5_path.read_text(encoding="utf-8"))
    refs_cache = exp_root / "results" / "_m3_r4_refs_cache.json"
    out: Dict[str, Dict[str, float]] = {"B4_ridge_self_stim": {}}
    if refs_cache.exists():
        rc = json.loads(refs_cache.read_text(encoding="utf-8"))
        for ds in DATASETS_ALL3:
            out["B4_ridge_self_stim"][ds] = float(rc["B4_ridge_self_stim"][ds]["E_d"])
    else:
        for ds in DATASETS_ALL3:
            sc = score_saved_predictions_file(
                exp_root / "runs" / "r0_04_b4_ridge_self_stim" / f"predictions_dev_{ds}.npz",
                split="dev", data_root=data_root, exp_root=exp_root, resolve_registry_ref=False,
            )
            out["B4_ridge_self_stim"][ds] = float(sc["E_d"])

    for spec in M35_MODEL_SPECS:
        mid = spec["model_id"]
        out[mid] = {}
        for ds in DATASETS_ALL3:
            out[mid][ds] = float(r5["models"][mid]["source_metrics"][ds]["E_d"])
    return out


# ---------------------------------------------------------------------------
# Fast Deterministic Fold Cache for M35_01 / M35_02 / M35_03
# (Precomputes all seed-independent linear anchors, nonlinear self features,
#  connectome drives, and excluded-self population PCA ONCE per fold!)
# ---------------------------------------------------------------------------
def _build_single_fold_m35_cache(
    fold_idx: int,
    aligned: Sequence[Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    canon_neurons: Sequence[str],
    functional_tiers: Sequence[str],
    tier_oh: np.ndarray,
    atlas: Mapping[str, Any],
    cfg: Mapping[str, Any],
    include_connectome: bool = True,
    include_graphfree: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    """Build deterministic fold tensors for ONE fold so callers can run all seeds/OAT configs and free memory immediately."""
    n_canon = len(canon_neurons)
    hp = cfg["hyperparameters"]
    taus = tuple(float(x) for x in hp.get("basis_taus_sec", (0.75, 2.0, 5.0, 10.0)))
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    wl_rank = int(hp.get("wireless_rank", 6))
    rank = int(hp.get("rank", 6))
    p_c, p_z = 8, 6
    d_state = p_c + p_z

    s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
    non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]
    fc_m3 = _build_m3_fold_cache(train_trials + val_trials, train_trials, s_n, n_canon, cfg)

    anchor_out = M3R1Model("M3R1_02_h1_confounds_only", cfg, canon_neurons, functional_tiers)
    anchor_out.fit(train_trials, s_n, fold_cache=fc_m3)
    priors_out = anchor_out.state["priors"]
    mu_out_tr = priors_out["mu_train"]
    b_dim = int(priors_out["b_dim"])
    patch_mat = build_patch_to_basis_matrix(b_dim=b_dim, taus_sec=taus)

    worms = sorted({str(t["animal_id"]) for t in non_buf})
    n_val_w = max(1, len(worms) // 5)
    val_worms = set(worms[-n_val_w:])
    inner_tr = [t for t in non_buf if str(t["animal_id"]) not in val_worms]
    inner_va = [t for t in non_buf if str(t["animal_id"]) in val_worms]
    if not inner_tr:
        inner_tr = list(non_buf)
        inner_va = list(non_buf[:1])

    anchor_in = M3R1Model("M3R1_02_h1_confounds_only", cfg, canon_neurons, functional_tiers)
    anchor_in.fit(inner_tr, s_n, fold_cache=None)
    priors_in = anchor_in.state["priors"]
    mu_in_tr = priors_in["mu_train"]

    pca_in = pca_out = None
    if include_connectome:
        pca_in = fit_causal_slow_state_pca(
            inner_tr, s_n, mu_in_tr, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )
        pca_out = fit_causal_slow_state_pca(
            non_buf, s_n, mu_out_tr, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

    pop_pca_in = pop_pca_out = None
    if include_graphfree:
        pop_pca_in = _fit_excluded_self_pop_pca(inner_tr, s_n, mu_in_tr, n_comp=d_state)
        pop_pca_out = _fit_excluded_self_pop_pca(non_buf, s_n, mu_out_tr, n_comp=d_state)

    def _collect_split(tr_list: Sequence[Mapping[str, Any]], anc: M3R1Model, pr: Mapping[str, Any], pca_st, pop_pca):
        xl, yl, nl, tl, ul = [], [], [], [], []
        dw_l, dwl_l, z_l, gf_l = [], [], [], []
        mu = pr["mu_train"]
        for tr in tr_list:
            c_norm, rval, _, _ = _project_trial_residual_targets(tr, pr, s_n, n_canon)
            p1 = anc._predict_stage1_coef(tr, s_n, pr, anc.state["s1"])
            d2 = anc._predict_stage2_delta_coef(tr, s_n, pr, anc.state["s2"])
            r_lin = (c_norm - (p1 + d2)).astype(np.float32)
            x_nl = extract_nonlinear_self_features(tr, s_n, pr, tier_oh)
            u_st = _extract_u_stim(tr, s_n, pr, n_canon)
            idx = np.where(rval)[0]
            if idx.size == 0:
                continue
            xl.append(x_nl[idx])
            yl.append(r_lin[idx])
            nl.append(idx.astype(np.int64))
            tl.append(tier_oh[idx])
            ul.append(u_st[idx])
            if include_connectome:
                d_w, d_wl = extract_1hop_2hop_lagmatched_drives(tr, atlas, s_n, mu, kappa=kappa)
                z_c = extract_circuit_slow_state_features(tr, canon_neurons, s_n, mu, pca_st, kappa=kappa)
                dw_l.append(d_w[idx])
                dwl_l.append(d_wl[idx])
                z_l.append(z_c[idx])
            if include_graphfree:
                s_gf = _extract_excluded_self_pop_pca_features(tr, s_n, mu, atlas["allow_mask"], pop_pca)
                gf_l.append(s_gf[idx])
        res = {
            "x": np.vstack(xl).astype(np.float32),
            "y_lin": np.vstack(yl).astype(np.float32),
            "n": np.concatenate(nl).astype(np.int64),
            "t": np.vstack(tl).astype(np.float32),
            "u": np.vstack(ul).astype(np.float32),
        }
        if include_connectome:
            res["dw"] = np.vstack(dw_l).astype(np.float32)
            res["dwl"] = np.vstack(dwl_l).astype(np.float32)
            res["z"] = np.vstack(z_l).astype(np.float32)
        if include_graphfree:
            res["s_gf"] = np.vstack(gf_l).astype(np.float32)
        return res

    sp_itr = _collect_split(inner_tr, anchor_in, priors_in, pca_in, pop_pca_in)
    sp_iva = _collect_split(inner_va, anchor_in, priors_in, pca_in, pop_pca_in)
    sp_all = _collect_split(non_buf, anchor_out, priors_out, pca_out, pop_pca_out)

    mu_in = np.mean(sp_itr["x"], axis=0, keepdims=True).astype(np.float32)
    sd_in = np.maximum(np.std(sp_itr["x"], axis=0, keepdims=True), 1e-4).astype(np.float32)
    x_itr_n = np.clip((sp_itr["x"] - mu_in) / sd_in, -5.0, 5.0).astype(np.float32)
    x_iva_n = np.clip((sp_iva["x"] - mu_in) / sd_in, -5.0, 5.0).astype(np.float32)

    mu_out = np.mean(sp_all["x"], axis=0, keepdims=True).astype(np.float32)
    sd_out = np.maximum(np.std(sp_all["x"], axis=0, keepdims=True), 1e-4).astype(np.float32)
    x_all_n = np.clip((sp_all["x"] - mu_out) / sd_out, -5.0, 5.0).astype(np.float32)

    s_conn_itr = s_conn_iva = s_conn_all = None
    wl_comp_out = c_comp_out = z_comp_out = None
    if include_connectome:
        wl_comp_in = _fit_svd_compressor(sp_itr["dwl"], n_comp=min(wl_rank, sp_itr["dwl"].shape[1]))
        c_itr = np.hstack([sp_itr["dw"], _apply_svd_compressor(sp_itr["dwl"], wl_comp_in)])
        c_iva = np.hstack([sp_iva["dw"], _apply_svd_compressor(sp_iva["dwl"], wl_comp_in)])
        c_comp_in = _fit_svd_compressor(c_itr, n_comp=p_c)
        z_comp_in = _fit_svd_compressor(sp_itr["z"], n_comp=p_z)
        s_conn_itr = np.hstack([_apply_svd_compressor(c_itr, c_comp_in), _apply_svd_compressor(sp_itr["z"], z_comp_in)]).astype(np.float32)
        s_conn_iva = np.hstack([_apply_svd_compressor(c_iva, c_comp_in), _apply_svd_compressor(sp_iva["z"], z_comp_in)]).astype(np.float32)

        wl_comp_out = _fit_svd_compressor(sp_all["dwl"], n_comp=min(wl_rank, sp_all["dwl"].shape[1]))
        c_all = np.hstack([sp_all["dw"], _apply_svd_compressor(sp_all["dwl"], wl_comp_out)])
        c_comp_out = _fit_svd_compressor(c_all, n_comp=p_c)
        z_comp_out = _fit_svd_compressor(sp_all["z"], n_comp=p_z)
        s_conn_all = np.hstack([_apply_svd_compressor(c_all, c_comp_out), _apply_svd_compressor(sp_all["z"], z_comp_out)]).astype(np.float32)

    n_va = len(val_trials)
    x_va_list, p_lin_va_list, u_va_list = [], [], []
    s_conn_va_list, s_gf_va_list = [], []
    prior_delta_va_list, phi_va_list = [], []

    for tr in val_trials:
        prior_delta, _, phi = predict_prior_delta_trajectory(tr, priors_out, n_canon)
        p1 = anchor_out._predict_stage1_coef(tr, s_n, priors_out, anchor_out.state["s1"])
        d2 = anchor_out._predict_stage2_delta_coef(tr, s_n, priors_out, anchor_out.state["s2"])
        p_lin_va_list.append((p1 + d2).astype(np.float32))
        x_nl = extract_nonlinear_self_features(tr, s_n, priors_out, tier_oh)
        x_va_list.append(np.clip((x_nl - mu_out) / sd_out, -5.0, 5.0).astype(np.float32))
        u_va_list.append(_extract_u_stim(tr, s_n, priors_out, n_canon))
        prior_delta_va_list.append(prior_delta.astype(np.float32))
        phi_va_list.append(phi.astype(np.float32))

        if include_connectome:
            d_w, d_wl = extract_1hop_2hop_lagmatched_drives(tr, atlas, s_n, mu_out_tr, kappa=kappa)
            z_c = extract_circuit_slow_state_features(tr, canon_neurons, s_n, mu_out_tr, pca_out, kappa=kappa)
            c_v = np.hstack([d_w, _apply_svd_compressor(d_wl, wl_comp_out)])
            s_conn_va_list.append(
                np.hstack([_apply_svd_compressor(c_v, c_comp_out), _apply_svd_compressor(z_c, z_comp_out)]).astype(np.float32)
            )
        if include_graphfree:
            s_gf_va_list.append(
                _extract_excluded_self_pop_pca_features(tr, s_n, mu_out_tr, atlas["allow_mask"], pop_pca_out)
            )

    fc = {
        "fold_idx": fold_idx,
        "s_n": np.asarray(s_n, dtype=np.float32),
        "mu_train": np.asarray(mu_out_tr, dtype=np.float32),
        "b_dim": b_dim,
        "patch_mat": patch_mat,
        "d_state": d_state,
        "x_itr_n": x_itr_n, "y_itr": sp_itr["y_lin"], "n_itr": sp_itr["n"], "t_itr": sp_itr["t"], "u_itr": sp_itr["u"],
        "x_iva_n": x_iva_n, "y_iva": sp_iva["y_lin"], "n_iva": sp_iva["n"], "t_iva": sp_iva["t"], "u_iva": sp_iva["u"],
        "x_all_n": x_all_n, "y_all": sp_all["y_lin"], "n_all": sp_all["n"], "t_all": sp_all["t"], "u_all": sp_all["u"],
        "s_conn_itr": s_conn_itr, "s_conn_iva": s_conn_iva, "s_conn_all": s_conn_all,
        "s_gf_itr": sp_itr.get("s_gf"), "s_gf_iva": sp_iva.get("s_gf"), "s_gf_all": sp_all.get("s_gf"),
        "n_va": n_va,
        "x_va_cat": np.vstack(x_va_list).astype(np.float32),
        "p_lin_va_list": p_lin_va_list,
        "u_va_cat": np.vstack(u_va_list).astype(np.float32),
        "s_conn_va_cat": np.vstack(s_conn_va_list).astype(np.float32) if include_connectome else None,
        "s_gf_va_cat": np.vstack(s_gf_va_list).astype(np.float32) if include_graphfree else None,
        "prior_delta_va_list": prior_delta_va_list,
        "phi_va_list": phi_va_list,
        "val_trials": val_trials,
    }
    del fc_m3, anchor_out, anchor_in, train_trials, sp_itr, sp_iva, sp_all
    gc.collect()
    return fc, val_trials, s_n


def _fit_m35_01_from_cache(
    fc: Mapping[str, Any],
    n_canon: int,
    tier_oh: np.ndarray,
    seed: int,
    self_lr: float,
    self_wd: float,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> Dict[str, Any]:
    """Train M35_01 from precomputed fold cache (bit-for-bit identical to M35Model.fit_shared_nonlinear_anchor)."""
    d_in = fc["x_itr_n"].shape[1]
    b_dim = int(fc["b_dim"])
    patch_mat = fc["patch_mat"]

    def _build_self_net() -> ZeroInitNonlinearSelfNet:
        return ZeroInitNonlinearSelfNet(
            d_in=d_in, n_canon=n_canon, n_tiers=len(TIER_NAMES),
            b_dim=b_dim, patch_to_basis=patch_mat,
            hidden_dim=hidden_dim, emb_dim=8, dropout=dropout,
        )

    net_full, alpha_self, best_ep, _, _ = _train_torch_residual_with_inner_cv(
        build_net_fn=_build_self_net,
        tr_tensors=(torch.from_numpy(fc["x_itr_n"]), torch.from_numpy(fc["n_itr"]), torch.from_numpy(fc["t_itr"])),
        y_tr=torch.from_numpy(fc["y_itr"]),
        va_tensors=(torch.from_numpy(fc["x_iva_n"]), torch.from_numpy(fc["n_iva"]), torch.from_numpy(fc["t_iva"])),
        y_va=torch.from_numpy(fc["y_iva"]),
        full_tensors=(torch.from_numpy(fc["x_all_n"]), torch.from_numpy(fc["n_all"]), torch.from_numpy(fc["t_all"])),
        y_full=torch.from_numpy(fc["y_all"]),
        lr=self_lr, weight_decay=self_wd, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, seed=seed,
    )

    torch.manual_seed(seed + 55)
    np.random.seed(seed + 55)
    net_in = _build_self_net()
    if best_ep > 0 and alpha_self > 0.02:
        opt_i = torch.optim.AdamW(net_in.parameters(), lr=self_lr, weight_decay=self_wd)
        loss_fn = nn.MSELoss()
        x_itr_t = torch.from_numpy(fc["x_itr_n"])
        n_itr_t = torch.from_numpy(fc["n_itr"])
        t_itr_t = torch.from_numpy(fc["t_itr"])
        y_itr_t = torch.from_numpy(fc["y_itr"])
        n_rows = x_itr_t.shape[0]
        for _ in range(best_ep):
            net_in.train()
            perm = torch.randperm(n_rows)
            for st in range(0, n_rows, batch_size):
                idx = perm[st : st + batch_size]
                opt_i.zero_grad()
                p = net_in(x_itr_t[idx], n_itr_t[idx], t_itr_t[idx])
                loss = loss_fn(p, y_itr_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net_in.parameters(), 2.0)
                opt_i.step()
    net_in.eval()
    net_full.eval()

    if alpha_self > 0.0:
        with torch.no_grad():
            d_itr = net_in(torch.from_numpy(fc["x_itr_n"]), torch.from_numpy(fc["n_itr"]), torch.from_numpy(fc["t_itr"])).numpy()
            d_iva = net_in(torch.from_numpy(fc["x_iva_n"]), torch.from_numpy(fc["n_iva"]), torch.from_numpy(fc["t_iva"])).numpy()
            d_all = net_full(torch.from_numpy(fc["x_all_n"]), torch.from_numpy(fc["n_all"]), torch.from_numpy(fc["t_all"])).numpy()
            n_va = int(fc["n_va"])
            n_va_idx = torch.arange(n_canon, dtype=torch.long).repeat(n_va)
            t_va_rep = torch.from_numpy(np.tile(tier_oh, (n_va, 1)))
            d_va = net_full(torch.from_numpy(fc["x_va_cat"]), n_va_idx, t_va_rep).numpy().reshape(n_va, n_canon, b_dim)
        y_nl_itr = (fc["y_itr"] - alpha_self * d_itr).astype(np.float32)
        y_nl_iva = (fc["y_iva"] - alpha_self * d_iva).astype(np.float32)
        y_nl_all = (fc["y_all"] - alpha_self * d_all).astype(np.float32)
        p_01_va = [
            (fc["p_lin_va_list"][i] + alpha_self * d_va[i]).astype(np.float32)
            for i in range(n_va)
        ]
    else:
        y_nl_itr = fc["y_itr"]
        y_nl_iva = fc["y_iva"]
        y_nl_all = fc["y_all"]
        p_01_va = [p.copy() for p in fc["p_lin_va_list"]]

    return {
        "alpha_self": float(alpha_self),
        "best_ep_self": int(best_ep),
        "y_nl_itr": y_nl_itr,
        "y_nl_iva": y_nl_iva,
        "y_nl_all": y_nl_all,
        "p_01_va": p_01_va,
    }


def _fit_film_from_cache(
    fc: Mapping[str, Any],
    m01_res: Mapping[str, Any],
    s_itr: np.ndarray,
    s_iva: np.ndarray,
    s_all: np.ndarray,
    s_va_cat: np.ndarray,
    n_canon: int,
    tier_oh: np.ndarray,
    seed: int,
    film_lr: float,
    film_wd: float,
    film_dim: int,
    dropout: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> Tuple[List[np.ndarray], float, int]:
    """Train ZeroInitPatchFiLMGate from precomputed fold cache & return validation coef predictions."""
    d_state = s_itr.shape[1]
    d_u = fc["u_itr"].shape[1]
    b_dim = int(fc["b_dim"])
    patch_mat = fc["patch_mat"]

    def _build_film_net() -> ZeroInitPatchFiLMGate:
        return ZeroInitPatchFiLMGate(
            d_state=d_state, d_u=d_u, n_canon=n_canon, n_tiers=len(TIER_NAMES),
            b_dim=b_dim, patch_to_basis=patch_mat,
            film_dim=film_dim, emb_dim=8, dropout=dropout,
        )

    net_film, alpha_film, best_ep_film, _, _ = _train_torch_residual_with_inner_cv(
        build_net_fn=_build_film_net,
        tr_tensors=(torch.from_numpy(s_itr), torch.from_numpy(fc["u_itr"]), torch.from_numpy(fc["t_itr"]), torch.from_numpy(fc["n_itr"])),
        y_tr=torch.from_numpy(m01_res["y_nl_itr"]),
        va_tensors=(torch.from_numpy(s_iva), torch.from_numpy(fc["u_iva"]), torch.from_numpy(fc["t_iva"]), torch.from_numpy(fc["n_iva"])),
        y_va=torch.from_numpy(m01_res["y_nl_iva"]),
        full_tensors=(torch.from_numpy(s_all), torch.from_numpy(fc["u_all"]), torch.from_numpy(fc["t_all"]), torch.from_numpy(fc["n_all"])),
        y_full=torch.from_numpy(m01_res["y_nl_all"]),
        lr=film_lr, weight_decay=film_wd, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, seed=seed,
    )

    n_va = int(fc["n_va"])
    if alpha_film > 0.0:
        n_va_idx = torch.arange(n_canon, dtype=torch.long).repeat(n_va)
        t_va_rep = torch.from_numpy(np.tile(tier_oh, (n_va, 1)))
        net_film.eval()
        with torch.no_grad():
            d_va = net_film(
                torch.from_numpy(s_va_cat),
                torch.from_numpy(fc["u_va_cat"]),
                t_va_rep,
                n_va_idx,
            ).numpy().reshape(n_va, n_canon, b_dim)
        p_va = [(m01_res["p_01_va"][i] + alpha_film * d_va[i]).astype(np.float32) for i in range(n_va)]
    else:
        p_va = [p.copy() for p in m01_res["p_01_va"]]

    return p_va, float(alpha_film), int(best_ep_film)


def _reconstruct_fold_preds(fc: Mapping[str, Any], p_coef_va: Sequence[np.ndarray]) -> List[np.ndarray]:
    s_n = fc["s_n"]
    mu_train = fc["mu_train"]
    out = []
    for i, tr in enumerate(fc["val_trials"]):
        out.append(
            _reconstruct_y_pred_from_coef(
                tr, fc["prior_delta_va_list"][i], fc["phi_va_list"][i], p_coef_va[i], s_n, mu_train
            )
        )
    return out


# ---------------------------------------------------------------------------
# STAGE 1: 5-Seed Multi-Seed Stability (M35_01, M35_02, M35_03) — 1 Fold at a Time
# ---------------------------------------------------------------------------
def run_stage_seeds(
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    hp = cfg["hyperparameters"]
    refs = _load_b4_and_r5_reference_eds(exp_root, data_root)
    ds_list = list(datasets) if datasets else list(DATASETS_ALL3)
    new_seeds = [s for s in SEEDS_5 if s != PRIMARY_SEED]
    models = [s["model_id"] for s in M35_MODEL_SPECS]

    lr = float(hp.get("learning_rate", 0.003))
    wd = float(hp.get("weight_decay", 0.01))
    h_dim = int(hp.get("mlp_hidden_dim", 48))
    film_dim = int(hp.get("poco_film_dim", 24))
    drop = float(hp.get("dropout", 0.15))
    bs = int(hp.get("batch_size", 2048))
    max_ep = int(hp.get("max_epochs", 18))
    patience = int(hp.get("early_stopping_patience", 4))

    per_ds_results: Dict[str, Any] = {}
    for ds in ds_list:
        ckpt = exp_root / "results" / f"_m35_r6_seeds_{ds}.json"
        if resume and ckpt.exists():
            print(f"[R6:seeds] Resuming cached 5-seed results for {ds} from {ckpt}", flush=True)
            per_ds_results[ds] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue

        t_ds = time.time()
        print(f"\n[R6:seeds] === Dataset: {ds} (1-fold-at-a-time streaming for <1.5 GiB RSS) ===", flush=True)
        raw_recs = load_dataset(
            dataset_id=ds, split="dev", allow_vault_unseal=False,
            data_root=data_root, exp_root=exp_root, caller_reason="m35_r6_seeds",
        )
        aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        atlas = build_1hop_2hop_connectome_atlas(canon_neurons, data_root=data_root)
        n_canon = len(canon_neurons)
        tier_oh = _tier_onehot(functional_tiers)

        scratch = _get_scratch_dir()
        val_trials_all: List[Dict[str, Any]] = []
        val_sn_all: List[np.ndarray] = []
        preds_by_sd_m: Dict[int, Dict[str, List[np.ndarray]]] = {
            sd: {m: [] for m in models} for sd in new_seeds
        }

        stopped_early = False
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_ckpt = scratch / f"seeds_{ds}_fold{fold_idx}.pkl"
            if resume and fold_ckpt.exists():
                with fold_ckpt.open("rb") as f:
                    fdat = pickle.load(f)
                val_trials_all.extend(fdat["val_trials"])
                val_sn_all.extend([fdat["s_n"]] * len(fdat["val_trials"]))
                for sd in new_seeds:
                    for m in models:
                        preds_by_sd_m[sd][m].extend(fdat["preds"][sd][m])
                print(f"  [R6:seeds:{ds}] fold {fold_idx+1}/{len(splits)} resumed from {fold_ckpt.name}", flush=True)
                continue

            t_f = time.time()
            fc, val_trials, s_n = _build_single_fold_m35_cache(
                fold_idx, aligned, train_idx, val_idx, canon_neurons, functional_tiers,
                tier_oh, atlas, cfg, include_connectome=True, include_graphfree=True,
            )
            val_trials_all.extend(val_trials)
            val_sn_all.extend([s_n] * len(val_trials))

            fold_preds: Dict[int, Dict[str, List[np.ndarray]]] = {sd: {} for sd in new_seeds}
            for sd in new_seeds:
                m01_res = _fit_m35_01_from_cache(
                    fc, n_canon, tier_oh, seed=sd, self_lr=lr, self_wd=wd,
                    hidden_dim=h_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
                )
                p02_va, _, _ = _fit_film_from_cache(
                    fc, m01_res, fc["s_conn_itr"], fc["s_conn_iva"], fc["s_conn_all"], fc["s_conn_va_cat"],
                    n_canon, tier_oh, seed=sd + 200, film_lr=lr, film_wd=wd,
                    film_dim=film_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
                )
                p03_va, _, _ = _fit_film_from_cache(
                    fc, m01_res, fc["s_gf_itr"], fc["s_gf_iva"], fc["s_gf_all"], fc["s_gf_va_cat"],
                    n_canon, tier_oh, seed=sd + 300, film_lr=lr, film_wd=wd,
                    film_dim=film_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
                )
                p01_rec = _reconstruct_fold_preds(fc, m01_res["p_01_va"])
                p02_rec = _reconstruct_fold_preds(fc, p02_va)
                p03_rec = _reconstruct_fold_preds(fc, p03_va)
                fold_preds[sd][models[0]] = p01_rec
                fold_preds[sd][models[1]] = p02_rec
                fold_preds[sd][models[2]] = p03_rec
                preds_by_sd_m[sd][models[0]].extend(p01_rec)
                preds_by_sd_m[sd][models[1]].extend(p02_rec)
                preds_by_sd_m[sd][models[2]].extend(p03_rec)

            with fold_ckpt.open("wb") as f:
                pickle.dump({"val_trials": val_trials, "s_n": s_n, "preds": fold_preds}, f)

            print(
                f"  [R6:seeds:{ds}] fold {fold_idx+1}/{len(splits)} done & checkpointed in {time.time() - t_f:.1f}s | "
                f"elapsed={time.time() - t0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                flush=True,
            )
            del fc, fold_preds
            gc.collect()

            if fold_idx + 1 < len(splits) and (time.time() - t0) > 400.0:
                print(f"  [R6:seeds:{ds}] Soft 400s wall-clock cap reached after fold {fold_idx+1}; resume cleanly with --resume!", flush=True)
                stopped_early = True
                break

        if stopped_early:
            del raw_recs, aligned, splits, atlas, val_trials_all, val_sn_all, preds_by_sd_m
            gc.collect()
            return {"stage": "seeds", "status": "partial_checkpointed", "dataset": ds}

        seed_ed: Dict[str, Dict[str, float]] = {m: {str(PRIMARY_SEED): float(refs[m][ds])} for m in models}
        seed_r2: Dict[str, Dict[str, float]] = {m: {} for m in models}
        for sd in new_seeds:
            for m in models:
                ev = evaluate_predictions(
                    val_trials_all, preds_by_sd_m[sd][m], val_sn_all,
                    functional_tiers=functional_tiers, compute_bootstrap=False, seed=sd,
                )
                seed_ed[m][str(sd)] = float(ev["E_d"])
                seed_r2[m][str(sd)] = float(ev["post_R2"])
            print(
                f"  [R6:seeds:{ds}] seed={sd} | "
                f"M35_01={seed_ed[models[0]][str(sd)]:.6f} | "
                f"M35_02={seed_ed[models[1]][str(sd)]:.6f} | "
                f"M35_03={seed_ed[models[2]][str(sd)]:.6f}",
                flush=True,
            )

        ds_rec = {
            "dataset_id": ds,
            "B4_E_d": float(refs["B4_ridge_self_stim"][ds]),
            "seeds": SEEDS_5,
            "seed_E_d": seed_ed,
            "seed_post_R2_new4": seed_r2,
            "wall_sec": round(time.time() - t_ds, 2),
            "peak_rss_mb": _peak_rss_mb(),
        }
        ckpt.write_text(json.dumps(ds_rec, indent=2), encoding="utf-8")
        per_ds_results[ds] = ds_rec
        # Clean up per-fold scratch pickles for completed dataset
        for fi in range(len(splits)):
            fp = scratch / f"seeds_{ds}_fold{fi}.pkl"
            if fp.exists():
                fp.unlink()
        del raw_recs, aligned, splits, atlas, val_trials_all, val_sn_all, preds_by_sd_m
        gc.collect()

    summary: Dict[str, Any] = {
        "stage": "seeds",
        "generated_at_utc": _utc_now(),
        "seeds": SEEDS_5,
        "reused_primary_seed_from_round5": PRIMARY_SEED,
        "per_dataset": per_ds_results,
        "wall_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }

    if all(ds in per_ds_results for ds in DATASETS_ALL3):
        model_stats: Dict[str, Any] = {}
        for m in models:
            per_d_stat: Dict[str, Any] = {}
            m2_b4_seeds, a3_b4_seeds = [], []
            m2_01_seeds, a3_01_seeds = [], []
            for sd in SEEDS_5:
                sk = str(sd)
                id_b4 = {ds: 1.0 - per_ds_results[ds]["seed_E_d"][m][sk] / per_ds_results[ds]["B4_E_d"] for ds in DATASETS_ALL3}
                id_01 = {ds: 1.0 - per_ds_results[ds]["seed_E_d"][m][sk] / per_ds_results[ds]["seed_E_d"][models[0]][sk] for ds in DATASETS_ALL3}
                m2_b4_seeds.append(float(np.mean([id_b4[ds] for ds in DATASETS_MAIN2])))
                a3_b4_seeds.append(float(np.mean([id_b4[ds] for ds in DATASETS_ALL3])))
                m2_01_seeds.append(float(np.mean([id_01[ds] for ds in DATASETS_MAIN2])))
                a3_01_seeds.append(float(np.mean([id_01[ds] for ds in DATASETS_ALL3])))

            for ds in DATASETS_ALL3:
                eds = [per_ds_results[ds]["seed_E_d"][m][str(sd)] for sd in SEEDS_5]
                b4_e = per_ds_results[ds]["B4_E_d"]
                ids_b4 = [1.0 - e / b4_e for e in eds]
                eds_01 = [per_ds_results[ds]["seed_E_d"][models[0]][str(sd)] for sd in SEEDS_5]
                ids_01 = [1.0 - e / e01 for e, e01 in zip(eds, eds_01)]
                per_d_stat[ds] = {
                    "E_d_mean": round(float(np.mean(eds)), 6),
                    "E_d_std": round(float(np.std(eds, ddof=1)), 6),
                    "I_d_vs_B4_mean": round(float(np.mean(ids_b4)), 6),
                    "I_d_vs_B4_std": round(float(np.std(ids_b4, ddof=1)), 6),
                    "I_d_vs_M35_01_mean": round(float(np.mean(ids_01)), 6),
                    "I_d_vs_M35_01_std": round(float(np.std(ids_01, ddof=1)), 6),
                }

            model_stats[m] = {
                "I_macro_main2_vs_B4_mean": round(float(np.mean(m2_b4_seeds)), 6),
                "I_macro_main2_vs_B4_std": round(float(np.std(m2_b4_seeds, ddof=1)), 6),
                "I_macro_all3_vs_B4_mean": round(float(np.mean(a3_b4_seeds)), 6),
                "I_macro_all3_vs_B4_std": round(float(np.std(a3_b4_seeds, ddof=1)), 6),
                "I_macro_main2_vs_M35_01_mean": round(float(np.mean(m2_01_seeds)), 6),
                "I_macro_main2_vs_M35_01_std": round(float(np.std(m2_01_seeds, ddof=1)), 6),
                "I_macro_all3_vs_M35_01_mean": round(float(np.mean(a3_01_seeds)), 6),
                "I_macro_all3_vs_M35_01_std": round(float(np.std(a3_01_seeds, ddof=1)), 6),
                "per_seed_I_macro_all3_vs_B4": [round(x, 6) for x in a3_b4_seeds],
                "per_seed_I_macro_main2_vs_B4": [round(x, 6) for x in m2_b4_seeds],
                "per_seed_I_macro_all3_vs_M35_01": [round(x, 6) for x in a3_01_seeds],
                "per_seed_I_macro_main2_vs_M35_01": [round(x, 6) for x in m2_01_seeds],
                "per_dataset": per_d_stat,
            }
        summary["models"] = model_stats
        out_path = exp_root / "results" / "_m35_r6_stage1_seeds.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[R6:seeds] Saved {out_path}", flush=True)
        for m in models:
            ms = model_stats[m]
            print(
                f"  {m}: All3 vs B4 = {100*ms['I_macro_all3_vs_B4_mean']:+.2f}% ± {100*ms['I_macro_all3_vs_B4_std']:.2f}% | "
                f"Main2 vs B4 = {100*ms['I_macro_main2_vs_B4_mean']:+.2f}% ± {100*ms['I_macro_main2_vs_B4_std']:.2f}% | "
                f"Net Main2 vs M35_01 = {100*ms['I_macro_main2_vs_M35_01_mean']:+.2f}% ± {100*ms['I_macro_main2_vs_M35_01_std']:.2f}%",
                flush=True,
            )

    return summary


# ---------------------------------------------------------------------------
# STAGE 2: 0.5x / 1x / 2x OAT Hyperparameter Neighborhood Stability (9 configs)
# ---------------------------------------------------------------------------
def run_stage_oat(
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    hp = cfg["hyperparameters"]
    refs = _load_b4_and_r5_reference_eds(exp_root, data_root)
    ds_list = list(datasets) if datasets else list(DATASETS_ALL3)

    base_lr = float(hp.get("learning_rate", 0.003))
    base_wd = float(hp.get("weight_decay", 0.01))
    h_dim = int(hp.get("mlp_hidden_dim", 48))
    film_dim = int(hp.get("poco_film_dim", 24))
    drop = float(hp.get("dropout", 0.15))
    bs = int(hp.get("batch_size", 2048))
    max_ep = int(hp.get("max_epochs", 18))
    patience = int(hp.get("early_stopping_patience", 4))

    grid: List[Dict[str, Any]] = [
        {"point_id": "1x_baseline", "param": "baseline", "multiplier": 1.0, "self_wd": base_wd, "self_lr": base_lr, "film_wd": base_wd, "film_lr": base_lr, "changes_self": False},
        {"point_id": "self_weight_decay_0.5x", "param": "self_weight_decay", "multiplier": 0.5, "self_wd": base_wd * 0.5, "self_lr": base_lr, "film_wd": base_wd, "film_lr": base_lr, "changes_self": True},
        {"point_id": "self_weight_decay_2.0x", "param": "self_weight_decay", "multiplier": 2.0, "self_wd": base_wd * 2.0, "self_lr": base_lr, "film_wd": base_wd, "film_lr": base_lr, "changes_self": True},
        {"point_id": "self_lr_0.5x", "param": "self_lr", "multiplier": 0.5, "self_wd": base_wd, "self_lr": base_lr * 0.5, "film_wd": base_wd, "film_lr": base_lr, "changes_self": True},
        {"point_id": "self_lr_2.0x", "param": "self_lr", "multiplier": 2.0, "self_wd": base_wd, "self_lr": base_lr * 2.0, "film_wd": base_wd, "film_lr": base_lr, "changes_self": True},
        {"point_id": "film_weight_decay_0.5x", "param": "film_weight_decay", "multiplier": 0.5, "self_wd": base_wd, "self_lr": base_lr, "film_wd": base_wd * 0.5, "film_lr": base_lr, "changes_self": False},
        {"point_id": "film_weight_decay_2.0x", "param": "film_weight_decay", "multiplier": 2.0, "self_wd": base_wd, "self_lr": base_lr, "film_wd": base_wd * 2.0, "film_lr": base_lr, "changes_self": False},
        {"point_id": "film_lr_0.5x", "param": "film_lr", "multiplier": 0.5, "self_wd": base_wd, "self_lr": base_lr, "film_wd": base_wd, "film_lr": base_lr * 0.5, "changes_self": False},
        {"point_id": "film_lr_2.0x", "param": "film_lr", "multiplier": 2.0, "self_wd": base_wd, "self_lr": base_lr, "film_wd": base_wd, "film_lr": base_lr * 2.0, "changes_self": False},
    ]

    scratch = _get_scratch_dir()
    per_ds_oat: Dict[str, Any] = {}
    for ds in ds_list:
        ckpt = exp_root / "results" / f"_m35_r6_oat_{ds}.json"
        if resume and ckpt.exists():
            print(f"[R6:oat] Resuming cached 9-config OAT results for {ds} from {ckpt}", flush=True)
            per_ds_oat[ds] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue

        t_ds = time.time()
        print(f"\n[R6:oat] === Dataset: {ds} (9-config OAT, 1-fold-at-a-time) ===", flush=True)
        raw_recs = load_dataset(
            dataset_id=ds, split="dev", allow_vault_unseal=False,
            data_root=data_root, exp_root=exp_root, caller_reason="m35_r6_oat",
        )
        aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        atlas = build_1hop_2hop_connectome_atlas(canon_neurons, data_root=data_root)
        n_canon = len(canon_neurons)
        tier_oh = _tier_onehot(functional_tiers)

        val_trials_all: List[Dict[str, Any]] = []
        val_sn_all: List[np.ndarray] = []
        preds_03_by_pt: Dict[str, List[np.ndarray]] = {g["point_id"]: [] for g in grid}
        preds_01_by_pt: Dict[str, List[np.ndarray]] = {g["point_id"]: [] for g in grid}

        stopped_early = False
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_ckpt = scratch / f"oat_{ds}_fold{fold_idx}.pkl"
            if resume and fold_ckpt.exists():
                with fold_ckpt.open("rb") as f:
                    fdat = pickle.load(f)
                val_trials_all.extend(fdat["val_trials"])
                val_sn_all.extend([fdat["s_n"]] * len(fdat["val_trials"]))
                for g in grid:
                    pid = g["point_id"]
                    preds_03_by_pt[pid].extend(fdat["p03"][pid])
                    preds_01_by_pt[pid].extend(fdat["p01"][pid])
                print(f"  [R6:oat:{ds}] fold {fold_idx+1}/{len(splits)} resumed from {fold_ckpt.name}", flush=True)
                continue

            t_f = time.time()
            fc, val_trials, s_n = _build_single_fold_m35_cache(
                fold_idx, aligned, train_idx, val_idx, canon_neurons, functional_tiers,
                tier_oh, atlas, cfg, include_connectome=False, include_graphfree=True,
            )
            val_trials_all.extend(val_trials)
            val_sn_all.extend([s_n] * len(val_trials))

            m01_base = _fit_m35_01_from_cache(
                fc, n_canon, tier_oh, seed=PRIMARY_SEED, self_lr=base_lr, self_wd=base_wd,
                hidden_dim=h_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
            )
            p01_base_rec = _reconstruct_fold_preds(fc, m01_base["p_01_va"])

            fold_p03: Dict[str, List[np.ndarray]] = {}
            fold_p01: Dict[str, List[np.ndarray]] = {}
            for g in grid:
                pid = g["point_id"]
                if g["changes_self"]:
                    m01_cur = _fit_m35_01_from_cache(
                        fc, n_canon, tier_oh, seed=PRIMARY_SEED, self_lr=g["self_lr"], self_wd=g["self_wd"],
                        hidden_dim=h_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
                    )
                    p01_cur_rec = _reconstruct_fold_preds(fc, m01_cur["p_01_va"])
                else:
                    m01_cur = m01_base
                    p01_cur_rec = p01_base_rec

                p03_va, _, _ = _fit_film_from_cache(
                    fc, m01_cur, fc["s_gf_itr"], fc["s_gf_iva"], fc["s_gf_all"], fc["s_gf_va_cat"],
                    n_canon, tier_oh, seed=PRIMARY_SEED + 300, film_lr=g["film_lr"], film_wd=g["film_wd"],
                    film_dim=film_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
                )
                p03_rec = _reconstruct_fold_preds(fc, p03_va)
                fold_p01[pid] = p01_cur_rec
                fold_p03[pid] = p03_rec
                preds_01_by_pt[pid].extend(p01_cur_rec)
                preds_03_by_pt[pid].extend(p03_rec)

            with fold_ckpt.open("wb") as f:
                pickle.dump({"val_trials": val_trials, "s_n": s_n, "p01": fold_p01, "p03": fold_p03}, f)

            print(
                f"  [R6:oat:{ds}] fold {fold_idx+1}/{len(splits)} done & checkpointed in {time.time() - t_f:.1f}s | "
                f"elapsed={time.time() - t0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                flush=True,
            )
            del fc, fold_p01, fold_p03
            gc.collect()

            if fold_idx + 1 < len(splits) and (time.time() - t0) > 400.0:
                print(f"  [R6:oat:{ds}] Soft 400s wall-clock cap reached after fold {fold_idx+1}; resume cleanly with --resume!", flush=True)
                stopped_early = True
                break

        if stopped_early:
            del raw_recs, aligned, splits, atlas, val_trials_all, val_sn_all, preds_03_by_pt, preds_01_by_pt
            gc.collect()
            return {"stage": "oat", "status": "partial_checkpointed", "dataset": ds}

        ds_pts: Dict[str, Any] = {}
        for g in grid:
            pid = g["point_id"]
            ev03 = evaluate_predictions(
                val_trials_all, preds_03_by_pt[pid], val_sn_all,
                functional_tiers=functional_tiers, compute_bootstrap=False, seed=PRIMARY_SEED,
            )
            ev01 = evaluate_predictions(
                val_trials_all, preds_01_by_pt[pid], val_sn_all,
                functional_tiers=functional_tiers, compute_bootstrap=False, seed=PRIMARY_SEED,
            )
            ds_pts[pid] = {
                "M35_03_E_d": float(ev03["E_d"]),
                "M35_01_paired_E_d": float(ev01["E_d"]),
                "M35_03_post_R2": float(ev03["post_R2"]),
            }

        diff_1x = abs(ds_pts["1x_baseline"]["M35_03_E_d"] - refs["M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"][ds])
        print(
            f"[R6:oat] {ds} done in {time.time() - t_ds:.1f}s | "
            f"1x_baseline E_d={ds_pts['1x_baseline']['M35_03_E_d']:.6f} "
            f"(diff vs R5={diff_1x:.2e}) | RSS={_peak_rss_mb():.1f} MB",
            flush=True,
        )
        ds_rec = {
            "dataset_id": ds,
            "B4_E_d": float(refs["B4_ridge_self_stim"][ds]),
            "M35_01_1x_E_d": float(refs["M35_01_zeroinit_nonlinear_self_anchor"][ds]),
            "baseline_1x_diff_vs_R5": float(diff_1x),
            "points": ds_pts,
            "wall_sec": round(time.time() - t_ds, 2),
        }
        ckpt.write_text(json.dumps(ds_rec, indent=2), encoding="utf-8")
        per_ds_oat[ds] = ds_rec
        for fi in range(len(splits)):
            fp = scratch / f"oat_{ds}_fold{fi}.pkl"
            if fp.exists():
                fp.unlink()
        del raw_recs, aligned, splits, atlas, val_trials_all, val_sn_all, preds_03_by_pt, preds_01_by_pt
        gc.collect()

    summary: Dict[str, Any] = {
        "stage": "oat",
        "generated_at_utc": _utc_now(),
        "n_points": len(grid),
        "perturbed_hyperparameters": ["self_weight_decay", "self_lr", "film_weight_decay", "film_lr"],
        "wall_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }

    if all(ds in per_ds_oat for ds in DATASETS_ALL3):
        pts_out: List[Dict[str, Any]] = []
        for g in grid:
            pid = g["point_id"]
            per_d = {}
            ib4_m2, ib4_a3 = [], []
            i01_paired_m2, i01_paired_a3 = [], []
            i01_1x_m2, i01_1x_a3 = [], []
            for ds in DATASETS_ALL3:
                ed03 = per_ds_oat[ds]["points"][pid]["M35_03_E_d"]
                ed01_p = per_ds_oat[ds]["points"][pid]["M35_01_paired_E_d"]
                ed01_1x = per_ds_oat[ds]["M35_01_1x_E_d"]
                edb4 = per_ds_oat[ds]["B4_E_d"]
                ib4 = 1.0 - ed03 / edb4
                i01_p = 1.0 - ed03 / ed01_p
                i01_f = 1.0 - ed03 / ed01_1x
                per_d[ds] = {
                    "E_d": round(ed03, 6),
                    "I_d_vs_B4": round(ib4, 6),
                    "I_d_vs_M35_01_paired": round(i01_p, 6),
                    "I_d_vs_M35_01_1x": round(i01_f, 6),
                }
                ib4_a3.append(ib4)
                i01_paired_a3.append(i01_p)
                i01_1x_a3.append(i01_f)
                if ds in DATASETS_MAIN2:
                    ib4_m2.append(ib4)
                    i01_paired_m2.append(i01_p)
                    i01_1x_m2.append(i01_f)

            pts_out.append({
                "point_id": pid,
                "param": g["param"],
                "multiplier": g["multiplier"],
                "I_macro_main2_vs_B4": round(float(np.mean(ib4_m2)), 6),
                "I_macro_all3_vs_B4": round(float(np.mean(ib4_a3)), 6),
                "I_macro_main2_vs_M35_01_paired": round(float(np.mean(i01_paired_m2)), 6),
                "I_macro_all3_vs_M35_01_paired": round(float(np.mean(i01_paired_a3)), 6),
                "I_macro_main2_vs_M35_01_1x": round(float(np.mean(i01_1x_m2)), 6),
                "I_macro_all3_vs_M35_01_1x": round(float(np.mean(i01_1x_a3)), 6),
                "per_dataset": per_d,
            })

        n_pts = len(pts_out)
        m2_b4_vals = [p["I_macro_main2_vs_B4"] for p in pts_out]
        a3_b4_vals = [p["I_macro_all3_vs_B4"] for p in pts_out]
        m2_01p_vals = [p["I_macro_main2_vs_M35_01_paired"] for p in pts_out]
        a3_01p_vals = [p["I_macro_all3_vs_M35_01_paired"] for p in pts_out]

        pos_b4_m2 = sum(v > 0.0 for v in m2_b4_vals)
        pos_b4_a3 = sum(v > 0.0 for v in a3_b4_vals)
        pos_01p_m2 = sum(v > 0.0 for v in m2_01p_vals)
        pos_01p_a3 = sum(v > 0.0 for v in a3_01p_vals)

        summary.update({
            "positive_fraction_main2_vs_B4": round(pos_b4_m2 / n_pts, 4),
            "positive_count_main2_vs_B4": f"{pos_b4_m2}/{n_pts}",
            "positive_fraction_all3_vs_B4": round(pos_b4_a3 / n_pts, 4),
            "positive_count_all3_vs_B4": f"{pos_b4_a3}/{n_pts}",
            "passes_80pct_vs_B4": bool((pos_b4_m2 / n_pts) >= 0.80 and (pos_b4_a3 / n_pts) >= 0.80),
            "positive_fraction_main2_vs_M35_01": round(pos_01p_m2 / n_pts, 4),
            "positive_count_main2_vs_M35_01": f"{pos_01p_m2}/{n_pts}",
            "positive_fraction_all3_vs_M35_01": round(pos_01p_a3 / n_pts, 4),
            "positive_count_all3_vs_M35_01": f"{pos_01p_a3}/{n_pts}",
            "passes_80pct_vs_M35_01": bool((pos_01p_m2 / n_pts) >= 0.80 and (pos_01p_a3 / n_pts) >= 0.80),
            "I_macro_main2_vs_B4_min": round(min(m2_b4_vals), 6),
            "I_macro_main2_vs_B4_max": round(max(m2_b4_vals), 6),
            "I_macro_all3_vs_B4_min": round(min(a3_b4_vals), 6),
            "I_macro_all3_vs_B4_max": round(max(a3_b4_vals), 6),
            "I_macro_main2_vs_M35_01_min": round(min(m2_01p_vals), 6),
            "I_macro_main2_vs_M35_01_max": round(max(m2_01p_vals), 6),
            "I_macro_all3_vs_M35_01_min": round(min(a3_01p_vals), 6),
            "I_macro_all3_vs_M35_01_max": round(max(a3_01p_vals), 6),
            "points": pts_out,
        })
        out_path = exp_root / "results" / "_m35_r6_stage2_oat.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"\n[R6:oat] Saved {out_path} | vs B4: {pos_b4_a3}/{n_pts} All3 "
            f"[{100*min(a3_b4_vals):+.2f}%, {100*max(a3_b4_vals):+.2f}%], Main2 [{100*min(m2_b4_vals):+.2f}%, {100*max(m2_b4_vals):+.2f}%] | "
            f"vs M35_01: {pos_01p_m2}/{n_pts} Main2, {pos_01p_a3}/{n_pts} All3",
            flush=True,
        )

    return summary


# ---------------------------------------------------------------------------
# STAGE 3: F11 Missingness Dropout Curve (+0%/+15%/+30%/+50%) + [-5, 0)s Check
# ---------------------------------------------------------------------------
def _predict_m35_with_net_override(
    mdl: M35Model,
    trial: Mapping[str, Any],
    net_ov: Mapping[str, Any],
) -> np.ndarray:
    """Strictly causal prediction where target-self uses `trial` and non-target network pathways use `net_ov`."""
    s_n = mdl.state["s_n"]
    st01 = mdl.state["st01"]
    priors = st01["priors"]
    mu_train = priors["mu_train"]
    prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, mdl.n_canon)

    # M35_01: stage 1 is strictly self-only; stage 2 (confounds) uses net_ov; nonlinear self-net is strictly self-only
    anchor = st01["anchor"]
    p1 = anchor._predict_stage1_coef(trial, s_n, priors, anchor.state["s1"])
    d2 = anchor._predict_stage2_delta_coef(trial, s_n, priors, anchor.state["s2"], net_trial_override=net_ov)
    p_coef = (p1 + d2).astype(np.float32)

    alpha_s = float(st01["alpha_self"])
    if alpha_s > 0.0:
        x_nl = extract_nonlinear_self_features(trial, s_n, priors, mdl._tier_oh)
        x_norm = np.clip((x_nl - st01["x_mu"]) / st01["x_sd"], -5.0, 5.0).astype(np.float32)
        net_s: ZeroInitNonlinearSelfNet = st01["net_self"]
        net_s.eval()
        with torch.no_grad():
            delta_s = net_s(
                torch.from_numpy(x_norm),
                torch.arange(mdl.n_canon, dtype=torch.long),
                torch.from_numpy(mdl._tier_oh),
            ).numpy().astype(np.float32)
        p_coef = (p_coef + alpha_s * delta_s).astype(np.float32)

    if mdl.model_id == "M35_03_nonlinear_anchor_graphfree_patch_film_ctrl":
        alpha_f = float(mdl.state.get("alpha_film", 0.0))
        if alpha_f > 0.0:
            s_state = _extract_excluded_self_pop_pca_features(
                net_ov, s_n, mu_train, mdl.atlas["allow_mask"], mdl.state["pop_pca_out"]
            )
            u_st = mdl._extract_u_stim(trial, s_n, priors)
            net_f: ZeroInitPatchFiLMGate = mdl.state["net_film"]
            net_f.eval()
            with torch.no_grad():
                delta_f = net_f(
                    torch.from_numpy(s_state),
                    torch.from_numpy(u_st),
                    torch.from_numpy(mdl._tier_oh),
                    torch.arange(mdl.n_canon, dtype=torch.long),
                ).numpy().astype(np.float32)
            p_coef = (p_coef + alpha_f * delta_f).astype(np.float32)

    return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, p_coef, s_n, mu_train)


def run_stage_robustness(
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    hp = cfg["hyperparameters"]
    ds_list = list(datasets) if datasets else list(DATASETS_ALL3)
    drop_rates = [0.0, 0.15, 0.30, 0.50]
    models = [
        "B4_ridge_self_stim",
        "M35_01_zeroinit_nonlinear_self_anchor",
        "M35_03_nonlinear_anchor_graphfree_patch_film_ctrl",
    ]

    per_ds_rob: Dict[str, Any] = {}
    for d_i, ds in enumerate(ds_list):
        ckpt = exp_root / "results" / f"_m35_r6_robustness_{ds}.json"
        if resume and ckpt.exists():
            print(f"[R6:robustness] Resuming cached F11 robustness for {ds} from {ckpt}", flush=True)
            per_ds_rob[ds] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue

        t_ds = time.time()
        print(f"\n[R6:robustness] === Dataset: {ds} (Missingness Dropout + [-5,0)s Short-Window) ===", flush=True)
        raw_recs = load_dataset(
            dataset_id=ds, split="dev", allow_vault_unseal=False,
            data_root=data_root, exp_root=exp_root, caller_reason="m35_r6_robustness",
        )
        aligned, canon, tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        atlas = build_1hop_2hop_connectome_atlas(canon, data_root=data_root)

        # (a) Artificial missingness dropout curve (+0%, +15%, +30%, +50%)
        drop_ckpt = exp_root / "results" / f"_m35_r6_rob_drop_{ds}.json"
        if resume and drop_ckpt.exists():
            print(f"[R6:robustness:{ds}] Resuming cached dropout curve from {drop_ckpt}", flush=True)
            drop_eval = json.loads(drop_ckpt.read_text(encoding="utf-8"))
        else:
            val_trials_all: List[Dict[str, Any]] = []
            val_sn_all: List[np.ndarray] = []
            drop_preds: Dict[str, Dict[str, List[np.ndarray]]] = {
                f"{int(round(r * 100))}pct": {m: [] for m in models} for r in drop_rates
            }

            for fold_i, (train_idx, val_idx) in enumerate(splits):
                s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
                all_tr = train_trials + val_trials
                b4_cache = build_fold_feature_cache(all_tr, train_trials, s_n, canon, cfg)
                b4 = CausalBaselineModel("B4_ridge_self_stim", cfg, canon, tiers).fit(train_trials, s_n, fold_cache=b4_cache)

                m3_cache = _build_m3_fold_cache(all_tr, train_trials, s_n, len(canon), cfg)
                m01 = M35Model(models[1], cfg, canon, tiers, atlas)
                shared_01 = m01.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=m3_cache)
                m01.fit(train_trials, s_n, fold_cache=m3_cache, shared_m35_01=shared_01)
                m03 = M35Model(models[2], cfg, canon, tiers, atlas)
                m03.fit(train_trials, s_n, fold_cache=m3_cache, shared_m35_01=shared_01)

                val_trials_all.extend(val_trials)
                val_sn_all.extend([s_n] * len(val_trials))

                for r_i, rate in enumerate(drop_rates):
                    rkey = f"{int(round(rate * 100))}pct"
                    rng = np.random.default_rng(PRIMARY_SEED + 1000 * d_i + 100 * fold_i + 17 * r_i)
                    for tr in val_trials:
                        net_ov = make_causal_network_dropout_override(tr, rate, rng)
                        drop_preds[rkey][models[0]].append(b4.predict_trial(tr, fold_cache=b4_cache))
                        drop_preds[rkey][models[1]].append(_predict_m35_with_net_override(m01, tr, net_ov))
                        drop_preds[rkey][models[2]].append(_predict_m35_with_net_override(m03, tr, net_ov))

                del b4_cache, m3_cache, b4, m01, m03, shared_01, train_trials, val_trials
                gc.collect()

            drop_eval = {}
            for rate in drop_rates:
                rkey = f"{int(round(rate * 100))}pct"
                drop_eval[rkey] = {}
                for m in models:
                    ev = evaluate_predictions(
                        val_trials_all, drop_preds[rkey][m], val_sn_all,
                        functional_tiers=tiers, compute_bootstrap=False, seed=PRIMARY_SEED,
                    )
                    drop_eval[rkey][m] = {"E_d": float(ev["E_d"]), "post_R2": float(ev["post_R2"])}
            drop_ckpt.write_text(json.dumps(drop_eval, indent=2), encoding="utf-8")
            print(
                f"[R6:robustness:{ds}] Saved dropout curve {drop_ckpt.name} in {time.time() - t_ds:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                flush=True,
            )
            del val_trials_all, val_sn_all, drop_preds
            gc.collect()
            if (time.time() - t0) > 280.0:
                print(f"[R6:robustness:{ds}] Soft 280s cap reached after dropout curve; resume cleanly for [-5,0)s!", flush=True)
                del raw_recs, aligned, splits, atlas
                gc.collect()
                return {"stage": "robustness", "status": "partial_dropout_done", "dataset": ds}

        # (b) Strictly causal [-5, 0)s short-window retrain & eval
        val_5s_all: List[Dict[str, Any]] = []
        sn_5s_all: List[np.ndarray] = []
        preds_5s: Dict[str, List[np.ndarray]] = {m: [] for m in models}

        for train_idx, val_idx in splits:
            s_n, tr_raw, va_raw = _extract_fold_trials(aligned, train_idx, val_idx, hp)
            train_trials = [truncate_trial_to_short_window(t, window_sec=5.0) for t in tr_raw]
            val_trials = [truncate_trial_to_short_window(t, window_sec=5.0) for t in va_raw]
            all_tr = train_trials + val_trials

            b4_cache = build_fold_feature_cache(all_tr, train_trials, s_n, canon, cfg)
            b4 = CausalBaselineModel("B4_ridge_self_stim", cfg, canon, tiers).fit(train_trials, s_n, fold_cache=b4_cache)

            m3_cache = _build_m3_fold_cache(all_tr, train_trials, s_n, len(canon), cfg)
            m01 = M35Model(models[1], cfg, canon, tiers, atlas)
            shared_01 = m01.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=m3_cache)
            m01.fit(train_trials, s_n, fold_cache=m3_cache, shared_m35_01=shared_01)
            m03 = M35Model(models[2], cfg, canon, tiers, atlas)
            m03.fit(train_trials, s_n, fold_cache=m3_cache, shared_m35_01=shared_01)

            val_5s_all.extend(val_trials)
            sn_5s_all.extend([s_n] * len(val_trials))
            for tr in val_trials:
                preds_5s[models[0]].append(b4.predict_trial(tr, fold_cache=b4_cache))
                preds_5s[models[1]].append(m01.predict_trial(tr))
                preds_5s[models[2]].append(m03.predict_trial(tr))

            del b4_cache, m3_cache, b4, m01, m03, shared_01, train_trials, val_trials
            gc.collect()

        short5s_eval: Dict[str, Any] = {}
        for m in models:
            ev = evaluate_predictions(
                val_5s_all, preds_5s[m], sn_5s_all,
                functional_tiers=tiers, compute_bootstrap=False, seed=PRIMARY_SEED,
            )
            short5s_eval[m] = {
                "E_d": float(ev["E_d"]),
                "post_R2": float(ev["post_R2"]),
                "delta_R2": float(ev["delta_R2"]),
            }

        ds_rec = {
            "dataset_id": ds,
            "dropout_curve": drop_eval,
            "short_window_5s": short5s_eval,
            "wall_sec": round(time.time() - t_ds, 2),
            "peak_rss_mb": _peak_rss_mb(),
        }
        ckpt.write_text(json.dumps(ds_rec, indent=2), encoding="utf-8")
        per_ds_rob[ds] = ds_rec
        print(
            f"[R6:robustness] {ds} done in {ds_rec['wall_sec']}s | "
            f"+0% M35_03={drop_eval['0pct'][models[2]]['E_d']:.5f} -> +50%={drop_eval['50pct'][models[2]]['E_d']:.5f} | "
            f"[-5,0)s B4={short5s_eval[models[0]]['E_d']:.5f}, M35_01={short5s_eval[models[1]]['E_d']:.5f}, "
            f"M35_03={short5s_eval[models[2]]['E_d']:.5f} | RSS={_peak_rss_mb():.1f} MB",
            flush=True,
        )
        del raw_recs, aligned, splits, atlas, val_5s_all, sn_5s_all, preds_5s
        gc.collect()

    summary: Dict[str, Any] = {
        "stage": "robustness",
        "generated_at_utc": _utc_now(),
        "protocol": "strictly_causal_nontarget_dropout_and_5s_short_window_zero_presplit_ppca",
        "wall_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }

    if all(ds in per_ds_rob for ds in DATASETS_ALL3):
        # Format dropout curve summary
        drop_levels: Dict[str, Any] = {}
        for rate in drop_rates:
            rkey = f"{int(round(rate * 100))}pct"
            entry: Dict[str, Any] = {"extra_dropout_frac": rate, "models": {}}
            for m in models:
                per_d = {}
                ib4_m2, ib4_a3, i01_m2, i01_a3 = [], [], [], []
                for ds in DATASETS_ALL3:
                    ed = per_ds_rob[ds]["dropout_curve"][rkey][m]["E_d"]
                    b4_e = per_ds_rob[ds]["dropout_curve"][rkey][models[0]]["E_d"]
                    m01_e = per_ds_rob[ds]["dropout_curve"][rkey][models[1]]["E_d"]
                    ib4 = 1.0 - ed / b4_e
                    i01 = 1.0 - ed / m01_e
                    per_d[ds] = {"E_d": round(ed, 6), "I_d_vs_B4": round(ib4, 6), "I_d_vs_M35_01": round(i01, 6)}
                    ib4_a3.append(ib4)
                    i01_a3.append(i01)
                    if ds in DATASETS_MAIN2:
                        ib4_m2.append(ib4)
                        i01_m2.append(i01)
                entry["models"][m] = {
                    "I_macro_main2_vs_B4": round(float(np.mean(ib4_m2)), 6),
                    "I_macro_all3_vs_B4": round(float(np.mean(ib4_a3)), 6),
                    "I_macro_main2_vs_M35_01": round(float(np.mean(i01_m2)), 6),
                    "I_macro_all3_vs_M35_01": round(float(np.mean(i01_a3)), 6),
                    "per_dataset": per_d,
                }
            drop_levels[rkey] = entry

        # Format [-5, 0)s short-window summary
        short_models: Dict[str, Any] = {}
        for m in models:
            per_d = {}
            ib4_m2, ib4_a3, i01_m2, i01_a3 = [], [], [], []
            for ds in DATASETS_ALL3:
                rec_m = per_ds_rob[ds]["short_window_5s"][m]
                ed = rec_m["E_d"]
                b4_e = per_ds_rob[ds]["short_window_5s"][models[0]]["E_d"]
                m01_e = per_ds_rob[ds]["short_window_5s"][models[1]]["E_d"]
                ib4 = 1.0 - ed / b4_e
                i01 = 1.0 - ed / m01_e
                per_d[ds] = {
                    "E_d": round(ed, 6),
                    "I_d_vs_B4_5s": round(ib4, 6),
                    "I_d_vs_M35_01_5s": round(i01, 6),
                    "post_R2": round(float(rec_m["post_R2"]), 6),
                    "delta_R2": round(float(rec_m["delta_R2"]), 6),
                }
                ib4_a3.append(ib4)
                i01_a3.append(i01)
                if ds in DATASETS_MAIN2:
                    ib4_m2.append(ib4)
                    i01_m2.append(i01)
            short_models[m] = {
                "I_macro_main2_vs_B4_5s": round(float(np.mean(ib4_m2)), 6),
                "I_macro_all3_vs_B4_5s": round(float(np.mean(ib4_a3)), 6),
                "I_macro_main2_vs_M35_01_5s": round(float(np.mean(i01_m2)), 6),
                "I_macro_all3_vs_M35_01_5s": round(float(np.mean(i01_a3)), 6),
                "per_dataset": per_d,
            }

        summary["dropout_curve"] = {"dropout_rates": drop_rates, "levels": drop_levels}
        summary["short_window_5s"] = {"window_sec": [-5.0, 0.0], "calibration_window_excluded": True, "models": short_models}
        out_path = exp_root / "results" / "_m35_r6_stage3_robustness.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[R6:robustness] Saved {out_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# STAGE 4: K=199 Constrained Null Graphs on top of M35_01 Nonlinear Anchor
#          + K_film=19 End-to-End M35_02 Patch-FiLM Retrainings (Main 2)
# ---------------------------------------------------------------------------
def _expand_stack_with_2hop(w_stack: np.ndarray, allow_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From 8-layer stack `[8, N, N]`, build lag-matched 1-hop+2-hop matrices strictly masked AFTER squaring.

    w_stack order: 0=chem_pos, 1=chem_neg, 2=chem_unk, 3=gap, 4=witvliet_core, 5=npp_short, 6=npp_mid, 7=monoamine.
    Returns:
      - `fast_wired`: `[8, N, N]` (gap, chem_pos, chem_neg, chem_unk, witvliet, wired_2hop, chem_2hop_pos, chem_2hop_neg)
      - `slow_wired`: `[2, N, N]` (gap, wired_2hop)
      - `slow_wl`:    `[3, N, N]` (npp_short, npp_mid, monoamine)
    """
    w_pos, w_neg, w_unk, w_gap, w_wit = w_stack[0], w_stack[1], w_stack[2], w_stack[3], w_stack[4]
    w_chem_unsigned = w_pos + w_neg + w_unk

    w_wired = (w_chem_unsigned + w_gap).astype(np.float32)
    rs_w = np.maximum(np.sum(w_wired, axis=1, keepdims=True), 1e-6)
    w_wired_norm = w_wired / rs_w
    w_wired_2hop = np.maximum((w_wired_norm @ w_wired_norm) * allow_f, 0.0).astype(np.float32)

    rs_c = np.maximum(np.sum(w_chem_unsigned, axis=1, keepdims=True), 1e-6)
    w_chem_signed = (w_pos - w_neg) / rs_c
    w_chem_2hop_s = ((w_chem_signed @ w_chem_signed) * allow_f).astype(np.float32)
    w_chem_2hop_pos = np.maximum(w_chem_2hop_s, 0.0).astype(np.float32)
    w_chem_2hop_neg = np.maximum(-w_chem_2hop_s, 0.0).astype(np.float32)

    fast_wired = np.stack(
        [w_gap, w_pos, w_neg, w_unk, w_wit, w_wired_2hop, w_chem_2hop_pos, w_chem_2hop_neg], axis=0
    ).astype(np.float32)
    slow_wired = np.stack([w_gap, w_wired_2hop], axis=0).astype(np.float32)
    slow_wl = w_stack[5:8].astype(np.float32)
    return fast_wired, slow_wired, slow_wl


def _compute_1hop2hop_drives_batched(
    fast_wired: np.ndarray,
    slow_wired: np.ndarray,
    slow_wl: np.ndarray,
    sm_fast_cat: np.ndarray,
    sm_slow_cat: np.ndarray,
    n_trials: int,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lag-matched 1-hop+2-hop wired `[n_trials, N, 20]` and wireless `[n_trials, N, 6]` drives via BLAS."""
    n_canon = fast_wired.shape[1]
    d_wired = np.empty((n_trials, n_canon, 20), dtype=np.float32)
    for li in range(8):
        w = fast_wired[li]
        denom = np.maximum(np.sum(w, axis=1, dtype=np.float32) + float(kappa), 1e-6)[:, None]
        drv = np.clip((w @ sm_fast_cat) / denom, -6.0, 6.0)
        d_wired[:, :, li * 2 : (li + 1) * 2] = drv.reshape(n_canon, n_trials, 2).transpose(1, 0, 2)
    for li in range(2):
        w = slow_wired[li]
        denom = np.maximum(np.sum(w, axis=1, dtype=np.float32) + float(kappa), 1e-6)[:, None]
        drv = np.clip((w @ sm_slow_cat) / denom, -6.0, 6.0)
        d_wired[:, :, 16 + li * 2 : 16 + (li + 1) * 2] = drv.reshape(n_canon, n_trials, 2).transpose(1, 0, 2)

    d_wl = np.empty((n_trials, n_canon, 6), dtype=np.float32)
    for li in range(3):
        w = slow_wl[li]
        denom = np.maximum(np.sum(w, axis=1, dtype=np.float32) + float(kappa), 1e-6)[:, None]
        drv = np.clip((w @ sm_slow_cat) / denom, -6.0, 6.0)
        d_wl[:, :, li * 2 : (li + 1) * 2] = drv.reshape(n_canon, n_trials, 2).transpose(1, 0, 2)
    return d_wired, d_wl


def _extract_sm_fast_slow(tr: Mapping[str, Any], s_n: np.ndarray, mu_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
    s_fast = s_mat[:, [2, 3]] * m_obs[:, None]
    s_slow_lvl = 0.5 * (s_mat[:, 0] + s_mat[:, 1])
    s_slow_slp = np.clip(s_mat[:, 1] - s_mat[:, 0], -6.0, 6.0)
    s_slow = np.column_stack([s_slow_lvl, s_slow_slp]).astype(np.float32) * m_obs[:, None]
    return s_fast.astype(np.float32), s_slow.astype(np.float32)


def _build_m35_01_anchored_null_cache(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    include_inner_for_film: bool = False,
) -> Dict[str, Any]:
    """Precompute frozen M35_01 nonlinear anchor residuals + graph-independent presynaptic states ONCE per fold."""
    raw_recs = load_dataset(
        dataset_id=dataset_id, split="dev", allow_vault_unseal=False,
        data_root=data_root, exp_root=exp_root, caller_reason="m35_r6_null199_cache",
    )
    aligned, canon, tiers = align_records_to_canonical(raw_recs)
    splits = build_grouped_cv_splits(aligned)
    atlas = build_1hop_2hop_connectome_atlas(canon, data_root=data_root)
    n_canon = len(canon)
    tier_oh = _tier_onehot(tiers)

    hp = cfg["hyperparameters"]
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    rank = int(hp.get("rank", 6))
    p_z = 6

    fold_caches: List[Dict[str, Any]] = []
    val_trials_all: List[Dict[str, Any]] = []
    val_sn_all: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
        fc_m3 = _build_m3_fold_cache(train_trials + val_trials, train_trials, s_n, n_canon, cfg)

        # Fit frozen M35_01 nonlinear anchor ONCE per fold (primary seed)
        m01 = M35Model("M35_01_zeroinit_nonlinear_self_anchor", cfg, canon, tiers, atlas)
        shared_01 = m01.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=fc_m3)
        st01_out = shared_01["outer"]
        st01_in = shared_01["inner"]
        priors_out = st01_out["priors"]
        mu_out = priors_out["mu_train"]
        non_buf = shared_01["non_buf"]

        pca_out = None
        if include_inner_for_film:
            pca_out = fit_causal_slow_state_pca(non_buf, s_n, mu_out, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2))

        smf_tr, sms_tr, tr_ti, tr_ni, u_tr, y_tr, z_tr = [], [], [], [], [], [], []
        for ti, tr in enumerate(non_buf):
            c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors_out, s_n, n_canon)
            p_nl = m01._predict_m35_01_coef(tr, s_n, st01_out)
            r_nl = (c_norm - p_nl).astype(np.float32)
            sf, ss = _extract_sm_fast_slow(tr, s_n, mu_out)
            smf_tr.append(sf)
            sms_tr.append(ss)
            idx = np.where(rval)[0]
            if idx.size > 0:
                tr_ti.append(np.full(idx.size, ti, dtype=np.int32))
                tr_ni.append(idx.astype(np.int32))
                u_tr.append(_extract_u_stim(tr, s_n, priors_out, n_canon)[idx])
                y_tr.append(r_nl[idx])
                if include_inner_for_film:
                    z_tr.append(extract_circuit_slow_state_features(tr, canon, s_n, mu_out, pca_out, kappa=kappa)[idx])

        n_out_i = np.concatenate(tr_ni).astype(np.int64)
        z_all_low = outer_z_comp = None
        if include_inner_for_film:
            z_all_arr = np.vstack(z_tr).astype(np.float32)
            outer_z_comp = _fit_svd_compressor(z_all_arr, n_comp=p_z)
            z_all_low = _apply_svd_compressor(z_all_arr, outer_z_comp)

        smf_va, sms_va, u_va, anc_va, pdel_va, phi_va, z_va_low = [], [], [], [], [], [], []
        for tr in val_trials:
            prior_delta, _, phi = predict_prior_delta_trajectory(tr, priors_out, n_canon)
            p_nl = m01._predict_m35_01_coef(tr, s_n, st01_out)
            sf, ss = _extract_sm_fast_slow(tr, s_n, mu_out)
            smf_va.append(sf)
            sms_va.append(ss)
            u_va.append(_extract_u_stim(tr, s_n, priors_out, n_canon))
            anc_va.append(p_nl)
            pdel_va.append(prior_delta.astype(np.float32))
            phi_va.append(phi.astype(np.float32))
            if include_inner_for_film:
                zc = extract_circuit_slow_state_features(tr, canon, s_n, mu_out, pca_out, kappa=kappa)
                z_va_low.append(_apply_svd_compressor(zc, outer_z_comp))

        fc_entry: Dict[str, Any] = {
            "fold_idx": fold_idx,
            "s_n": np.asarray(s_n, dtype=np.float32),
            "mu_train": np.asarray(mu_out, dtype=np.float32),
            "b_dim": int(st01_out["b_dim"]),
            "patch_mat": st01_out["patch_mat"],
            "n_tr": len(non_buf),
            "smf_tr_cat": np.hstack(smf_tr).astype(np.float32),
            "sms_tr_cat": np.hstack(sms_tr).astype(np.float32),
            "tr_ti": np.concatenate(tr_ti),
            "tr_ni": n_out_i,
            "u_tr": np.vstack(u_tr).astype(np.float32),
            "y_tr": np.vstack(y_tr).astype(np.float32),
            "t_tr": tier_oh[n_out_i],
            "n_va": len(val_trials),
            "smf_va_cat": np.hstack(smf_va).astype(np.float32),
            "sms_va_cat": np.hstack(sms_va).astype(np.float32),
            "u_va_list": u_va,
            "u_va_cat": np.vstack(u_va).astype(np.float32),
            "anc_va_list": anc_va,
            "pdel_va_list": pdel_va,
            "phi_va_list": phi_va,
            "val_trials": val_trials,
        }

        if include_inner_for_film:
            inner_tr = shared_01["inner_tr"]
            inner_va = shared_01["inner_va"]
            pr_in = st01_in["priors"]
            mu_in = pr_in["mu_train"]
            pca_in = fit_causal_slow_state_pca(inner_tr, s_n, mu_in, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2))

            def _collect_in(tr_sub):
                sf_l, ss_l, ti_l, ni_l, z_l, u_l, y_l = [], [], [], [], [], [], []
                for ti, tr in enumerate(tr_sub):
                    c_norm, rval, _, _ = _project_trial_residual_targets(tr, pr_in, s_n, n_canon)
                    p_nl = m01._predict_m35_01_coef(tr, s_n, st01_in)
                    r_nl = (c_norm - p_nl).astype(np.float32)
                    sf, ss = _extract_sm_fast_slow(tr, s_n, mu_in)
                    sf_l.append(sf)
                    ss_l.append(ss)
                    idx = np.where(rval)[0]
                    if idx.size > 0:
                        ti_l.append(np.full(idx.size, ti, dtype=np.int32))
                        ni_l.append(idx.astype(np.int64))
                        z_l.append(extract_circuit_slow_state_features(tr, canon, s_n, mu_in, pca_in, kappa=kappa)[idx])
                        u_l.append(_extract_u_stim(tr, s_n, pr_in, n_canon)[idx])
                        y_l.append(r_nl[idx])
                ni_arr = np.concatenate(ni_l).astype(np.int64)
                return {
                    "n": len(tr_sub),
                    "smf_cat": np.hstack(sf_l).astype(np.float32),
                    "sms_cat": np.hstack(ss_l).astype(np.float32),
                    "ti": np.concatenate(ti_l),
                    "ni": ni_arr,
                    "z": np.vstack(z_l).astype(np.float32),
                    "u": np.vstack(u_l).astype(np.float32),
                    "y": np.vstack(y_l).astype(np.float32),
                    "t": tier_oh[ni_arr],
                }

            in_tr_c = _collect_in(inner_tr)
            in_va_c = _collect_in(inner_va)
            in_z_comp = _fit_svd_compressor(in_tr_c["z"], n_comp=p_z)
            in_tr_c["z_low"] = _apply_svd_compressor(in_tr_c["z"], in_z_comp)
            in_va_c["z_low"] = _apply_svd_compressor(in_va_c["z"], in_z_comp)
            del in_tr_c["z"], in_va_c["z"]

            fc_entry["z_all_low"] = z_all_low
            fc_entry["z_va_low_cat"] = np.vstack(z_va_low).astype(np.float32)
            fc_entry["in_tr"] = in_tr_c
            fc_entry["in_va"] = in_va_c

        fold_caches.append(fc_entry)
        val_trials_all.extend(val_trials)
        val_sn_all.extend([s_n] * len(val_trials))
        del fc_m3, m01, shared_01, train_trials
        gc.collect()

    del raw_recs, aligned, splits
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "canonical_neurons": canon,
        "functional_tiers": tiers,
        "n_canon": n_canon,
        "tier_oh": tier_oh,
        "allow_f": atlas["allow_mask"].astype(np.float32),
        "fold_caches": fold_caches,
        "val_trials_all": val_trials_all,
        "val_sn_all": val_sn_all,
    }


def _retrain_1hop2hop_ridge_on_m35_01_anchor(
    ds_cache: Mapping[str, Any],
    w_stack: np.ndarray,
    hp: Mapping[str, Any],
) -> Dict[str, Any]:
    """Genuinely retrain 1-hop+2-hop signed connectome Ridge + RRR-Bilinear from scratch on top of frozen M35_01."""
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    conn_l2 = float(hp.get("connectome_l2_penalty", 60.0))
    wl_rank = int(hp.get("wireless_rank", 6))
    wl_mult = float(hp.get("wireless_l2_multiplier", 2.5))
    bilin_rank = int(hp.get("bilinear_rank", 4))

    fast_w, slow_w, slow_wl = _expand_stack_with_2hop(w_stack, ds_cache["allow_f"])
    preds_all: List[np.ndarray] = []

    for fc in ds_cache["fold_caches"]:
        dw_tr_all, dwl_tr_all = _compute_1hop2hop_drives_batched(
            fast_w, slow_w, slow_wl, fc["smf_tr_cat"], fc["sms_tr_cat"], fc["n_tr"], kappa
        )
        dw_all = dw_tr_all[fc["tr_ti"], fc["tr_ni"], :]
        dwl_all = dwl_tr_all[fc["tr_ti"], fc["tr_ni"], :]
        u_all = fc["u_tr"]
        y_all = fc["y_tr"]

        w_mu = np.mean(dw_all, axis=0, keepdims=True).astype(np.float32)
        w_sd = np.maximum(np.std(dw_all, axis=0, keepdims=True), 1e-4).astype(np.float32)
        dw_n = np.clip((dw_all - w_mu) / w_sd, -5.0, 5.0).astype(np.float32)

        wl_k = min(wl_rank, dwl_all.shape[1])
        wl_comp = _fit_svd_compressor(dwl_all, n_comp=wl_k)
        dwl_low = _apply_svd_compressor(dwl_all, wl_comp)
        d_lin = np.hstack([dw_n, dwl_low]).astype(np.float32)

        col_l2 = np.concatenate([
            np.full(dw_n.shape[1], conn_l2, dtype=np.float64),
            np.full(dwl_low.shape[1], conn_l2 * wl_mult, dtype=np.float64),
        ])
        eff_col_l2 = col_l2 * max(1.0, float(d_lin.shape[1]) / 12.0)

        w_lin = _solve_weighted_ridge_sample_scaled(d_lin, y_all, eff_col_l2)
        y_res_lin = (y_all - d_lin @ w_lin).astype(np.float32)

        p_c = min(4, max(1, d_lin.shape[1]))
        c_comp = _fit_svd_compressor(d_lin, n_comp=p_c)
        c_low = _apply_svd_compressor(d_lin, c_comp)
        x_bilin = _build_bilinear_features(c_low, u_all)
        xb_mu = np.mean(x_bilin, axis=0, keepdims=True).astype(np.float32)
        xb_sd = np.maximum(np.std(x_bilin, axis=0, keepdims=True), 1e-4).astype(np.float32)
        xb_n = np.clip((x_bilin - xb_mu) / xb_sd, -5.0, 5.0).astype(np.float32)
        bilin_eff_l2 = conn_l2 * max(1.0, float(xb_n.shape[1]) / 12.0) * 1.5
        rrr = _fit_reduced_rank_ridge(xb_n, y_res_lin, bilin_eff_l2, rank=bilin_rank)
        w_rrr = rrr["w"]

        dw_va, dwl_va = _compute_1hop2hop_drives_batched(
            fast_w, slow_w, slow_wl, fc["smf_va_cat"], fc["sms_va_cat"], fc["n_va"], kappa
        )
        s_n = fc["s_n"]
        mu_train = fc["mu_train"]
        for vi, tr in enumerate(fc["val_trials"]):
            dw_n_v = np.clip((dw_va[vi] - w_mu) / w_sd, -5.0, 5.0).astype(np.float32)
            dwl_low_v = _apply_svd_compressor(dwl_va[vi], wl_comp)
            d_lin_v = np.hstack([dw_n_v, dwl_low_v]).astype(np.float32)
            delta_lin = (d_lin_v @ w_lin).astype(np.float32)
            c_low_v = _apply_svd_compressor(d_lin_v, c_comp)
            xb_v = _build_bilinear_features(c_low_v, fc["u_va_list"][vi])
            xb_n_v = np.clip((xb_v - xb_mu) / xb_sd, -5.0, 5.0).astype(np.float32)
            delta_bilin = (xb_n_v @ w_rrr).astype(np.float32)
            pred_c = fc["anc_va_list"][vi] + delta_lin + delta_bilin
            preds_all.append(
                _reconstruct_y_pred_from_coef(
                    tr, fc["pdel_va_list"][vi], fc["phi_va_list"][vi], pred_c, s_n, mu_train
                )
            )

    ev = evaluate_predictions(
        ds_cache["val_trials_all"], preds_all, ds_cache["val_sn_all"],
        functional_tiers=ds_cache["functional_tiers"], compute_bootstrap=False, seed=PRIMARY_SEED,
    )
    return {"E_d": float(ev["E_d"]), "post_R2": float(ev["post_R2"])}


def _retrain_m35_02_film_on_graph(
    ds_cache: Mapping[str, Any],
    w_stack: np.ndarray,
    hp: Mapping[str, Any],
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Genuinely retrain M35_02 ZeroInitPatchFiLMGate from scratch inside EVERY CV fold on `w_stack`."""
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    wl_rank = int(hp.get("wireless_rank", 6))
    film_dim = int(hp.get("poco_film_dim", 24))
    drop = float(hp.get("dropout", 0.15))
    lr = float(hp.get("learning_rate", 0.003))
    wd = float(hp.get("weight_decay", 0.01))
    max_ep = int(hp.get("max_epochs", 18))
    patience = int(hp.get("early_stopping_patience", 4))
    bs = int(hp.get("batch_size", 2048))
    n_canon = ds_cache["n_canon"]
    tier_oh = ds_cache["tier_oh"]
    p_c, p_z = 8, 6
    d_state = p_c + p_z

    fast_w, slow_w, slow_wl = _expand_stack_with_2hop(w_stack, ds_cache["allow_f"])
    preds_all: List[np.ndarray] = []

    for fc in ds_cache["fold_caches"]:
        in_tr = fc["in_tr"]
        in_va = fc["in_va"]
        b_dim = int(fc["b_dim"])
        patch_mat = fc["patch_mat"]

        # Inner drives & SVD compressors
        dw_itr_all, dwl_itr_all = _compute_1hop2hop_drives_batched(
            fast_w, slow_w, slow_wl, in_tr["smf_cat"], in_tr["sms_cat"], in_tr["n"], kappa
        )
        dw_itr = dw_itr_all[in_tr["ti"], in_tr["ni"], :]
        dwl_itr = dwl_itr_all[in_tr["ti"], in_tr["ni"], :]

        dw_iva_all, dwl_iva_all = _compute_1hop2hop_drives_batched(
            fast_w, slow_w, slow_wl, in_va["smf_cat"], in_va["sms_cat"], in_va["n"], kappa
        )
        dw_iva = dw_iva_all[in_va["ti"], in_va["ni"], :]
        dwl_iva = dwl_iva_all[in_va["ti"], in_va["ni"], :]

        wl_comp_in = _fit_svd_compressor(dwl_itr, n_comp=min(wl_rank, dwl_itr.shape[1]))
        c_itr = np.hstack([dw_itr, _apply_svd_compressor(dwl_itr, wl_comp_in)])
        c_iva = np.hstack([dw_iva, _apply_svd_compressor(dwl_iva, wl_comp_in)])
        c_comp_in = _fit_svd_compressor(c_itr, n_comp=p_c)
        s_itr = np.hstack([_apply_svd_compressor(c_itr, c_comp_in), in_tr["z_low"]]).astype(np.float32)
        s_iva = np.hstack([_apply_svd_compressor(c_iva, c_comp_in), in_va["z_low"]]).astype(np.float32)

        # Outer drives & SVD compressors
        dw_out_all, dwl_out_all = _compute_1hop2hop_drives_batched(
            fast_w, slow_w, slow_wl, fc["smf_tr_cat"], fc["sms_tr_cat"], fc["n_tr"], kappa
        )
        dw_all = dw_out_all[fc["tr_ti"], fc["tr_ni"], :]
        dwl_all = dwl_out_all[fc["tr_ti"], fc["tr_ni"], :]
        wl_comp_out = _fit_svd_compressor(dwl_all, n_comp=min(wl_rank, dwl_all.shape[1]))
        c_all = np.hstack([dw_all, _apply_svd_compressor(dwl_all, wl_comp_out)])
        c_comp_out = _fit_svd_compressor(c_all, n_comp=p_c)
        s_all = np.hstack([_apply_svd_compressor(c_all, c_comp_out), fc["z_all_low"]]).astype(np.float32)

        d_u = in_tr["u"].shape[1]

        def _build_film_net() -> ZeroInitPatchFiLMGate:
            return ZeroInitPatchFiLMGate(
                d_state=d_state, d_u=d_u, n_canon=n_canon, n_tiers=len(TIER_NAMES),
                b_dim=b_dim, patch_to_basis=patch_mat, film_dim=film_dim, emb_dim=8, dropout=drop,
            )

        net_film, alpha_film, _, _, _ = _train_torch_residual_with_inner_cv(
            build_net_fn=_build_film_net,
            tr_tensors=(torch.from_numpy(s_itr), torch.from_numpy(in_tr["u"]), torch.from_numpy(in_tr["t"]), torch.from_numpy(in_tr["ni"])),
            y_tr=torch.from_numpy(in_tr["y"]),
            va_tensors=(torch.from_numpy(s_iva), torch.from_numpy(in_va["u"]), torch.from_numpy(in_va["t"]), torch.from_numpy(in_va["ni"])),
            y_va=torch.from_numpy(in_va["y"]),
            full_tensors=(torch.from_numpy(s_all), torch.from_numpy(fc["u_tr"]), torch.from_numpy(fc["t_tr"]), torch.from_numpy(fc["tr_ni"])),
            y_full=torch.from_numpy(fc["y_tr"]),
            lr=lr, weight_decay=wd, batch_size=bs, max_epochs=max_ep, patience=patience,
            seed=int(seed) + 200,
        )

        # Validation prediction
        n_va = int(fc["n_va"])
        s_n = fc["s_n"]
        mu_train = fc["mu_train"]
        if alpha_film > 0.0:
            dw_va, dwl_va = _compute_1hop2hop_drives_batched(
                fast_w, slow_w, slow_wl, fc["smf_va_cat"], fc["sms_va_cat"], n_va, kappa
            )
            dw_va_flat = dw_va.reshape(n_va * n_canon, 20)
            dwl_va_flat = dwl_va.reshape(n_va * n_canon, 6)
            c_va_flat = np.hstack([dw_va_flat, _apply_svd_compressor(dwl_va_flat, wl_comp_out)])
            s_va_cat = np.hstack([_apply_svd_compressor(c_va_flat, c_comp_out), fc["z_va_low_cat"]]).astype(np.float32)

            n_va_idx = torch.arange(n_canon, dtype=torch.long).repeat(n_va)
            t_va_rep = torch.from_numpy(np.tile(tier_oh, (n_va, 1)))
            net_film.eval()
            with torch.no_grad():
                d_va = net_film(
                    torch.from_numpy(s_va_cat),
                    torch.from_numpy(fc["u_va_cat"]),
                    t_va_rep,
                    n_va_idx,
                ).numpy().reshape(n_va, n_canon, b_dim)
            for vi, tr in enumerate(fc["val_trials"]):
                pred_c = (fc["anc_va_list"][vi] + alpha_film * d_va[vi]).astype(np.float32)
                preds_all.append(
                    _reconstruct_y_pred_from_coef(
                        tr, fc["pdel_va_list"][vi], fc["phi_va_list"][vi], pred_c, s_n, mu_train
                    )
                )
        else:
            for vi, tr in enumerate(fc["val_trials"]):
                preds_all.append(
                    _reconstruct_y_pred_from_coef(
                        tr, fc["pdel_va_list"][vi], fc["phi_va_list"][vi], fc["anc_va_list"][vi], s_n, mu_train
                    )
                )

    ev = evaluate_predictions(
        ds_cache["val_trials_all"], preds_all, ds_cache["val_sn_all"],
        functional_tiers=ds_cache["functional_tiers"], compute_bootstrap=False, seed=PRIMARY_SEED,
    )
    return {"E_d": float(ev["E_d"]), "post_R2": float(ev["post_R2"])}


def _compute_null_stats(real_ed: float, null_eds: Sequence[float], anchor_ed: float) -> Dict[str, Any]:
    arr = np.asarray(null_eds, dtype=np.float64)
    k_tot = int(arr.size)
    # One-sided: how many null graphs achieve E_d <= real_ed (i.e., beat or tie the real connectome)
    k_better_or_eq = int(np.sum(arr <= real_ed))
    p_emp = float((1 + k_better_or_eq) / (1 + k_tot))
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if k_tot > 1 else 0.0
    z_score = float((mu - real_ed) / sd) if sd > 1e-12 else 0.0
    real_gain_vs_anchor = float(1.0 - real_ed / max(anchor_ed, 1e-8))
    null_gains = 1.0 - arr / max(anchor_ed, 1e-8)
    return {
        "k_null": k_tot,
        "real_E_d": round(float(real_ed), 6),
        "anchor_M35_01_E_d": round(float(anchor_ed), 6),
        "null_E_d_mean": round(mu, 6),
        "null_E_d_std": round(sd, 6),
        "real_gain_vs_M35_01": round(real_gain_vs_anchor, 6),
        "null_gain_vs_M35_01_mean": round(float(np.mean(null_gains)), 6),
        "null_gain_vs_M35_01_std": round(float(np.std(null_gains, ddof=1)) if k_tot > 1 else 0.0, 6),
        "k_null_le_real": k_better_or_eq,
        "p_empirical_onesided": round(p_emp, 6),
        "z_score_real_better_positive": round(z_score, 4),
    }


def run_stage_null199(
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    k_null: int = DEFAULT_K_NULL,
    k_film: int = DEFAULT_K_FILM,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    scratch = _get_scratch_dir()
    hp = cfg["hyperparameters"]
    refs = _load_b4_and_r5_reference_eds(exp_root, data_root)

    raw_bundle = np.load(data_root / "connectomes" / "connectome_priors_bundle.npz", allow_pickle=True)
    canon_ref = [str(n) for n in raw_bundle["canonical_neurons"]]
    gen = ConstrainedNullConnectomeGenerator(canon_ref, data_root=data_root)
    bank_path, constraint_report = ensure_null_bank(gen, k_null=k_null, scratch_dir=scratch, resume=resume)

    ds_list = list(datasets) if datasets else list(DATASETS_ALL3)
    ridge_by_ds: Dict[str, Any] = {}
    film_by_ds: Dict[str, Any] = {}

    for ds in ds_list:
        ridge_ckpt = exp_root / "results" / f"_m35_r6_null199_ridge_{ds}.json"
        film_ckpt = exp_root / "results" / f"_m35_r6_null19_film_{ds}.json"
        need_ridge = not (resume and ridge_ckpt.exists())
        need_film = (ds in DATASETS_MAIN2) and not (resume and film_ckpt.exists())

        if not need_ridge:
            print(f"[R6:null199] Resuming cached K={k_null} ridge nulls for {ds} from {ridge_ckpt}", flush=True)
            ridge_by_ds[ds] = json.loads(ridge_ckpt.read_text(encoding="utf-8"))
        if ds in DATASETS_MAIN2 and not need_film:
            print(f"[R6:null199] Resuming cached K_film={k_film} M35_02 nulls for {ds} from {film_ckpt}", flush=True)
            film_by_ds[ds] = json.loads(film_ckpt.read_text(encoding="utf-8"))

        if not need_ridge and not need_film:
            continue

        t_ds = time.time()
        cache_pkl = scratch / f"m35_01_null_cache_{ds}_film{int(ds in DATASETS_MAIN2)}.pkl"
        if resume and cache_pkl.exists():
            print(f"\n[R6:null199] === Resuming M35_01-anchored null cache for {ds} from {cache_pkl.name} ===", flush=True)
            with cache_pkl.open("rb") as f:
                ds_cache = pickle.load(f)
        else:
            print(f"\n[R6:null199] === Building M35_01-anchored null cache for {ds} (film={ds in DATASETS_MAIN2}) ===", flush=True)
            ds_cache = _build_m35_01_anchored_null_cache(
                ds, cfg, data_root, exp_root, include_inner_for_film=(ds in DATASETS_MAIN2)
            )
            with cache_pkl.open("wb") as f:
                pickle.dump(ds_cache, f)
        if ds_cache["canonical_neurons"] != canon_ref:
            raise AssertionError(f"Canonical neuron order mismatch on {ds}")

        anchor_ed = float(refs["M35_01_zeroinit_nonlinear_self_anchor"][ds])

        if need_ridge:
            t_r0 = time.time()
            real_r = _retrain_1hop2hop_ridge_on_m35_01_anchor(ds_cache, gen.real_stack, hp)
            full_ckpt = scratch / f"null199_full_{ds}.json"
            if resume and full_ckpt.exists():
                full_eds = json.loads(full_ckpt.read_text(encoding="utf-8"))
                print(f"  [R6:null199:{ds}] Resumed {len(full_eds)} Full-8L ridge nulls from {full_ckpt.name}", flush=True)
            else:
                mm = np.load(bank_path, mmap_mode="r")
                full_eds = []
                for k in range(k_null):
                    w_k = np.asarray(mm[k], dtype=np.float32)
                    rk = _retrain_1hop2hop_ridge_on_m35_01_anchor(ds_cache, w_k, hp)
                    full_eds.append(float(rk["E_d"]))
                    if (k + 1) % 50 == 0 or (k + 1) == k_null:
                        print(
                            f"  [R6:null199:{ds}] Full-8L {k+1}/{k_null} | real={real_r['E_d']:.6f} | "
                            f"null_mean={np.mean(full_eds):.6f} | elapsed={time.time() - t_r0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                            flush=True,
                        )
                del mm
                full_ckpt.write_text(json.dumps(full_eds), encoding="utf-8")

            if (time.time() - t0) > 400.0:
                print(f"  [R6:null199:{ds}] Soft 400s cap reached after Full-8L; resume cleanly with --resume!", flush=True)
                del ds_cache
                gc.collect()
                return {"stage": "null199", "status": "partial_full_ridge_done", "dataset": ds}

            t_g0 = time.time()
            gap_eds: List[float] = []
            for k in range(1, k_null + 1):
                w_g = gen.generate_gap_only_null_stack(PRIMARY_SEED + 900000 + 1000 * k)
                rg = _retrain_1hop2hop_ridge_on_m35_01_anchor(ds_cache, w_g, hp)
                gap_eds.append(float(rg["E_d"]))
                if k % 50 == 0 or k == k_null:
                    print(
                        f"  [R6:null199:{ds}] W_gap-only {k}/{k_null} | real={real_r['E_d']:.6f} | "
                        f"null_mean={np.mean(gap_eds):.6f} | elapsed={time.time() - t_g0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                        flush=True,
                    )

            rec_r = {
                "dataset_id": ds,
                "k_null": k_null,
                "anchor_M35_01_E_d": anchor_ed,
                "real_E_d": float(real_r["E_d"]),
                "full_null_E_d": full_eds,
                "gap_only_null_E_d": gap_eds,
                "full_null_stats": _compute_null_stats(real_r["E_d"], full_eds, anchor_ed),
                "gap_only_stats": _compute_null_stats(real_r["E_d"], gap_eds, anchor_ed),
                "wall_sec": round(time.time() - t_r0, 2),
                "peak_rss_mb": _peak_rss_mb(),
            }
            ridge_ckpt.write_text(json.dumps(rec_r), encoding="utf-8")
            ridge_by_ds[ds] = rec_r
            print(
                f"  [R6:null199:{ds}] Ridge summary: Full p={rec_r['full_null_stats']['p_empirical_onesided']:.4f} "
                f"(z={rec_r['full_null_stats']['z_score_real_better_positive']:+.2f}) | "
                f"Gap-only p={rec_r['gap_only_stats']['p_empirical_onesided']:.4f} "
                f"(z={rec_r['gap_only_stats']['z_score_real_better_positive']:+.2f})",
                flush=True,
            )
            if (time.time() - t0) > 360.0 and need_film:
                print(f"  [R6:null199:{ds}] Soft 360s cap reached after Ridge; resume cleanly for Patch-FiLM nulls!", flush=True)
                del ds_cache
                gc.collect()
                return {"stage": "null199", "status": "partial_ridge_done", "dataset": ds}

        if need_film:
            t_f0 = time.time()
            real_f = _retrain_m35_02_film_on_graph(ds_cache, gen.real_stack, hp, seed=PRIMARY_SEED)
            r5_m02_ed = float(refs["M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film"][ds])
            diff_f = abs(real_f["E_d"] - r5_m02_ed)
            print(
                f"  [R6:null19_film:{ds}] Real M35_02 E_d={real_f['E_d']:.6f} vs R5={r5_m02_ed:.6f} (diff={diff_f:.2e})",
                flush=True,
            )
            film_part_ckpt = scratch / f"null19_film_part_{ds}.json"
            film_eds: List[float] = []
            if resume and film_part_ckpt.exists():
                film_eds = json.loads(film_part_ckpt.read_text(encoding="utf-8"))
                print(f"  [R6:null19_film:{ds}] Resumed {len(film_eds)}/{k_film} Patch-FiLM nulls from {film_part_ckpt.name}", flush=True)

            mm = np.load(bank_path, mmap_mode="r")
            stopped_film = False
            for k in range(len(film_eds), k_film):
                w_k = np.asarray(mm[k], dtype=np.float32)
                rf = _retrain_m35_02_film_on_graph(ds_cache, w_k, hp, seed=PRIMARY_SEED)
                film_eds.append(float(rf["E_d"]))
                film_part_ckpt.write_text(json.dumps(film_eds), encoding="utf-8")
                if (k + 1) % 5 == 0 or (k + 1) == k_film:
                    print(
                        f"  [R6:null19_film:{ds}] M35_02 null {k+1}/{k_film} | real={real_f['E_d']:.6f} | "
                        f"null_mean={np.mean(film_eds):.6f} | elapsed={time.time() - t_f0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                        flush=True,
                    )
                if (k + 1) < k_film and (time.time() - t0) > 400.0:
                    print(f"  [R6:null19_film:{ds}] Soft 400s cap reached after {k+1}/{k_film}; resume cleanly with --resume!", flush=True)
                    stopped_film = True
                    break
            del mm
            if stopped_film:
                del ds_cache
                gc.collect()
                return {"stage": "null199", "status": "partial_film_done", "dataset": ds}
            rec_f = {
                "dataset_id": ds,
                "k_film": k_film,
                "anchor_M35_01_E_d": anchor_ed,
                "real_E_d": float(real_f["E_d"]),
                "saved_R5_M35_02_E_d": r5_m02_ed,
                "k0_abs_diff": float(diff_f),
                "null_E_d": film_eds,
                "stats": _compute_null_stats(real_f["E_d"], film_eds, anchor_ed),
                "wall_sec": round(time.time() - t_f0, 2),
                "peak_rss_mb": _peak_rss_mb(),
            }
            film_ckpt.write_text(json.dumps(rec_f), encoding="utf-8")
            film_by_ds[ds] = rec_f
            print(
                f"  [R6:null19_film:{ds}] M35_02 Patch-FiLM summary: p={rec_f['stats']['p_empirical_onesided']:.4f} "
                f"(z={rec_f['stats']['z_score_real_better_positive']:+.2f})",
                flush=True,
            )

        if cache_pkl.exists():
            cache_pkl.unlink()
        del ds_cache
        gc.collect()

    summary: Dict[str, Any] = {
        "stage": "null199",
        "generated_at_utc": _utc_now(),
        "k_null": k_null,
        "k_film": k_film,
        "constraints": constraint_report,
        "wall_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }

    if all(ds in ridge_by_ds for ds in DATASETS_ALL3):
        def _macro_null(ds_subset: Sequence[str], key: str) -> Dict[str, Any]:
            real_gains = [1.0 - ridge_by_ds[d]["real_E_d"] / ridge_by_ds[d]["anchor_M35_01_E_d"] for d in ds_subset]
            real_macro = float(np.mean(real_gains))
            null_macros = []
            for k in range(k_null):
                gk = [1.0 - ridge_by_ds[d][key][k] / ridge_by_ds[d]["anchor_M35_01_E_d"] for d in ds_subset]
                null_macros.append(float(np.mean(gk)))
            arr = np.asarray(null_macros, dtype=np.float64)
            k_ge = int(np.sum(arr >= real_macro))
            p_emp = float((1 + k_ge) / (1 + k_null))
            sd = float(np.std(arr, ddof=1))
            z = float((real_macro - float(np.mean(arr))) / sd) if sd > 1e-12 else 0.0
            return {
                "real_I_macro_vs_M35_01": round(real_macro, 6),
                "null_I_macro_vs_M35_01_mean": round(float(np.mean(arr)), 6),
                "null_I_macro_vs_M35_01_std": round(sd, 6),
                "k_null_ge_real": k_ge,
                "p_empirical_onesided": round(p_emp, 6),
                "z_score_real_better_positive": round(z, 4),
            }

        summary["ridge199_on_M35_01_anchor"] = {
            "per_dataset": {
                ds: {
                    "full_8layer": ridge_by_ds[ds]["full_null_stats"],
                    "gap_only": ridge_by_ds[ds]["gap_only_stats"],
                    "wall_sec": ridge_by_ds[ds]["wall_sec"],
                }
                for ds in DATASETS_ALL3
            },
            "macro_main2_full_8layer": _macro_null(DATASETS_MAIN2, "full_null_E_d"),
            "macro_all3_full_8layer": _macro_null(DATASETS_ALL3, "full_null_E_d"),
            "macro_main2_gap_only": _macro_null(DATASETS_MAIN2, "gap_only_null_E_d"),
            "macro_all3_gap_only": _macro_null(DATASETS_ALL3, "gap_only_null_E_d"),
        }

    if all(ds in film_by_ds for ds in DATASETS_MAIN2):
        real_gains = [1.0 - film_by_ds[d]["real_E_d"] / film_by_ds[d]["anchor_M35_01_E_d"] for d in DATASETS_MAIN2]
        real_macro = float(np.mean(real_gains))
        null_macros = [
            float(np.mean([1.0 - film_by_ds[d]["null_E_d"][k] / film_by_ds[d]["anchor_M35_01_E_d"] for d in DATASETS_MAIN2]))
            for k in range(k_film)
        ]
        arr = np.asarray(null_macros, dtype=np.float64)
        k_ge = int(np.sum(arr >= real_macro))
        p_emp = float((1 + k_ge) / (1 + k_film))
        sd = float(np.std(arr, ddof=1)) if k_film > 1 else 0.0
        z = float((real_macro - float(np.mean(arr))) / sd) if sd > 1e-12 else 0.0
        summary["m35_02_patch_film_null19_main2"] = {
            "per_dataset": {
                ds: {
                    "k0_abs_diff_vs_R5": film_by_ds[ds]["k0_abs_diff"],
                    "stats": film_by_ds[ds]["stats"],
                    "wall_sec": film_by_ds[ds]["wall_sec"],
                }
                for ds in DATASETS_MAIN2
            },
            "macro_main2": {
                "real_I_macro_vs_M35_01": round(real_macro, 6),
                "null_I_macro_vs_M35_01_mean": round(float(np.mean(arr)), 6),
                "null_I_macro_vs_M35_01_std": round(sd, 6),
                "k_null_ge_real": k_ge,
                "p_empirical_onesided": round(p_emp, 6),
                "z_score_real_better_positive": round(z, 4),
            },
        }

    if "ridge199_on_M35_01_anchor" in summary and "m35_02_patch_film_null19_main2" in summary:
        out_path = exp_root / "results" / "_m35_r6_stage4_null199.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[R6:null199] Saved {out_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# STAGE 5: Consolidation into results/m35_round6_full_validation.json & Log
# ---------------------------------------------------------------------------
def _append_round6_log(exp_root: Path, consolidated: Mapping[str, Any]) -> None:
    log_path = exp_root / "research_loop_log.md"
    marker = "## Round 6 (`M3.5` Full Multi-Seed, OAT, Robustness & 199-Null-Graph Validation)"
    if log_path.exists() and marker in log_path.read_text(encoding="utf-8"):
        return

    s1 = consolidated["stage1_five_seed_stability"]["models"]
    s2 = consolidated["stage2_oat_05x_1x_2x"]
    s3_d = consolidated["stage3_robustness"]["dropout_curve"]["levels"]
    s3_5 = consolidated["stage3_robustness"]["short_window_5s"]["models"]
    s4_r = consolidated["stage4_null_graphs"]["ridge199_on_M35_01_anchor"]
    s4_f = consolidated["stage4_null_graphs"]["m35_02_patch_film_null19_main2"]

    lines: List[str] = [
        "",
        "---",
        "",
        marker,
        f"- **Date (UTC)**: `{consolidated['generated_at_utc']}`",
        "- **Unified Config**: `configs/lambda_global.json` (ZERO `dataset_id` branches, ZERO `vault_sealed/` access)",
        "",
        "### 1. 5-Seed Multi-Seed Stability (`[20261117..20261121]`)",
        "| Model | `Seyed I_d` vs `B4` (mean±std) | `Toyo I_d` vs `B4` | `Yemini I_d` vs `B4` | `I_macro` Main 2 vs `B4` | `I_macro` All 3 vs `B4` | Net Main 2 vs `M35_01` | Net All 3 vs `M35_01` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in M35_MODEL_SPECS:
        m = spec["model_id"]
        ms = s1[m]
        pd = ms["per_dataset"]
        lines.append(
            f"| `{m}` | `{100*pd['seyedolmohadesin2025']['I_d_vs_B4_mean']:+.2f}% ± {100*pd['seyedolmohadesin2025']['I_d_vs_B4_std']:.2f}%` | "
            f"`{100*pd['toyoshima2024']['I_d_vs_B4_mean']:+.2f}% ± {100*pd['toyoshima2024']['I_d_vs_B4_std']:.2f}%` | "
            f"`{100*pd['yemini2021']['I_d_vs_B4_mean']:+.2f}% ± {100*pd['yemini2021']['I_d_vs_B4_std']:.2f}%` | "
            f"`{100*ms['I_macro_main2_vs_B4_mean']:+.2f}% ± {100*ms['I_macro_main2_vs_B4_std']:.2f}%` | "
            f"`{100*ms['I_macro_all3_vs_B4_mean']:+.2f}% ± {100*ms['I_macro_all3_vs_B4_std']:.2f}%` | "
            f"`{100*ms['I_macro_main2_vs_M35_01_mean']:+.2f}% ± {100*ms['I_macro_main2_vs_M35_01_std']:.2f}%` | "
            f"`{100*ms['I_macro_all3_vs_M35_01_mean']:+.2f}% ± {100*ms['I_macro_all3_vs_M35_01_std']:.2f}%` |"
        )

    lines.extend([
        "",
        "### 2. `0.5x / 1x / 2x` OAT Hyperparameter Neighborhood Stability (`M35_03`, 9 configs)",
        f"- **Positive vs `B4_ridge`**: Main 2 `{s2['positive_count_main2_vs_B4']}` (`{100*s2['positive_fraction_main2_vs_B4']:.1f}%`), "
        f"All 3 `{s2['positive_count_all3_vs_B4']}` (`{100*s2['positive_fraction_all3_vs_B4']:.1f}%`), passes `>=80%`: `{s2['passes_80pct_vs_B4']}`.",
        f"- **Range vs `B4_ridge`**: Main 2 `[{100*s2['I_macro_main2_vs_B4_min']:+.2f}%, {100*s2['I_macro_main2_vs_B4_max']:+.2f}%]`, "
        f"All 3 `[{100*s2['I_macro_all3_vs_B4_min']:+.2f}%, {100*s2['I_macro_all3_vs_B4_max']:+.2f}%]`.",
        f"- **Positive vs paired `M35_01`**: Main 2 `{s2['positive_count_main2_vs_M35_01']}` (`{100*s2['positive_fraction_main2_vs_M35_01']:.1f}%`), "
        f"All 3 `{s2['positive_count_all3_vs_M35_01']}` (`{100*s2['positive_fraction_all3_vs_M35_01']:.1f}%`), passes `>=80%`: `{s2['passes_80pct_vs_M35_01']}`.",
        f"- **Range vs paired `M35_01`**: Main 2 `[{100*s2['I_macro_main2_vs_M35_01_min']:+.2f}%, {100*s2['I_macro_main2_vs_M35_01_max']:+.2f}%]`, "
        f"All 3 `[{100*s2['I_macro_all3_vs_M35_01_min']:+.2f}%, {100*s2['I_macro_all3_vs_M35_01_max']:+.2f}%]`.",
        "",
        "### 3. F11 Missingness Dropout Curve (`+0%/+15%/+30%/+50%`) & `[-5, 0)s` Short-Window Check",
        "| Condition | `M35_01` Main 2 vs `B4` | `M35_01` All 3 vs `B4` | `M35_03` Main 2 vs `B4` | `M35_03` All 3 vs `B4` | Net `M35_03 vs M35_01` Main 2 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for rkey in ("0pct", "15pct", "30pct", "50pct"):
        m1 = s3_d[rkey]["models"]["M35_01_zeroinit_nonlinear_self_anchor"]
        m3 = s3_d[rkey]["models"]["M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"]
        lines.append(
            f"| Dropout `+{rkey}` | `{100*m1['I_macro_main2_vs_B4']:+.2f}%` | `{100*m1['I_macro_all3_vs_B4']:+.2f}%` | "
            f"`{100*m3['I_macro_main2_vs_B4']:+.2f}%` | `{100*m3['I_macro_all3_vs_B4']:+.2f}%` | `{100*m3['I_macro_main2_vs_M35_01']:+.2f}%` |"
        )
    m1_5 = s3_5["M35_01_zeroinit_nonlinear_self_anchor"]
    m3_5 = s3_5["M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"]
    lines.append(
        f"| Short-window `[-5,0)s` | `{100*m1_5['I_macro_main2_vs_B4_5s']:+.2f}%` | `{100*m1_5['I_macro_all3_vs_B4_5s']:+.2f}%` | "
        f"`{100*m3_5['I_macro_main2_vs_B4_5s']:+.2f}%` | `{100*m3_5['I_macro_all3_vs_B4_5s']:+.2f}%` | `{100*m3_5['I_macro_main2_vs_M35_01_5s']:+.2f}%` |"
    )

    lines.extend([
        "",
        "### 4. `K = 199` Constrained Null Graphs on Frozen `M35_01` Nonlinear Anchor + `K_film = 19` End-to-End `M35_02`",
        "| Test Family | `Seyed2025` (`p`, `z`) | `Toyo2024` (`p`, `z`) | `Yemini2021` (`p`, `z`) | Main 2 Macro (`p`, `z`) | All 3 Macro (`p`, `z`) |",
        "|---|---|---|---|---|---|",
    ])
    for fam_label, fam_key, m2_key, a3_key in [
        ("1-hop+2-hop Ridge on `M35_01` (`K=199` Full 8L)", "full_8layer", "macro_main2_full_8layer", "macro_all3_full_8layer"),
        ("1-hop+2-hop Ridge on `M35_01` (`K=199` `W_gap`-only)", "gap_only", "macro_main2_gap_only", "macro_all3_gap_only"),
    ]:
        ps = s4_r["per_dataset"]
        m2 = s4_r[m2_key]
        a3 = s4_r[a3_key]
        lines.append(
            f"| {fam_label} | `p={ps['seyedolmohadesin2025'][fam_key]['p_empirical_onesided']:.4f}, z={ps['seyedolmohadesin2025'][fam_key]['z_score_real_better_positive']:+.2f}` | "
            f"`p={ps['toyoshima2024'][fam_key]['p_empirical_onesided']:.4f}, z={ps['toyoshima2024'][fam_key]['z_score_real_better_positive']:+.2f}` | "
            f"`p={ps['yemini2021'][fam_key]['p_empirical_onesided']:.4f}, z={ps['yemini2021'][fam_key]['z_score_real_better_positive']:+.2f}` | "
            f"`p={m2['p_empirical_onesided']:.4f}, z={m2['z_score_real_better_positive']:+.2f}` | "
            f"`p={a3['p_empirical_onesided']:.4f}, z={a3['z_score_real_better_positive']:+.2f}` |"
        )
    pf = s4_f["per_dataset"]
    mf = s4_f["macro_main2"]
    lines.append(
        f"| End-to-End `M35_02` Patch-FiLM (`K=19` Full 8L) | "
        f"`p={pf['seyedolmohadesin2025']['stats']['p_empirical_onesided']:.4f}, z={pf['seyedolmohadesin2025']['stats']['z_score_real_better_positive']:+.2f}` | "
        f"`p={pf['toyoshima2024']['stats']['p_empirical_onesided']:.4f}, z={pf['toyoshima2024']['stats']['z_score_real_better_positive']:+.2f}` | "
        f"N/A (Main 2) | "
        f"`p={mf['p_empirical_onesided']:.4f}, z={mf['z_score_real_better_positive']:+.2f}` | N/A |"
    )
    lines.append("")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_stage_summarize(exp_root: Path) -> Dict[str, Any]:
    s1 = json.loads((exp_root / "results" / "_m35_r6_stage1_seeds.json").read_text(encoding="utf-8"))
    s2 = json.loads((exp_root / "results" / "_m35_r6_stage2_oat.json").read_text(encoding="utf-8"))
    s3 = json.loads((exp_root / "results" / "_m35_r6_stage3_robustness.json").read_text(encoding="utf-8"))
    s4 = json.loads((exp_root / "results" / "_m35_r6_stage4_null199.json").read_text(encoding="utf-8"))
    r5 = json.loads((exp_root / "results" / "m35_round5_summary.json").read_text(encoding="utf-8"))

    consolidated = {
        "generated_at_utc": _utc_now(),
        "config_id": "lambda_global_v1",
        "zero_dataset_id_branches": True,
        "zero_vault_access": True,
        "round5_primary_seed_leakage_invariance": r5["leakage_invariance"],
        "stage1_five_seed_stability": s1,
        "stage2_oat_05x_1x_2x": s2,
        "stage3_robustness": s3,
        "stage4_null_graphs": s4,
    }

    out_path = exp_root / "results" / "m35_round6_full_validation.json"
    out_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    _append_round6_log(exp_root, consolidated)
    print(f"[R6:summarize] Saved {out_path} and updated research_loop_log.md", flush=True)
    return consolidated


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.5 Round 6: Complete Multi-Dimensional Rigor Suite")
    parser.add_argument("--stage", choices=["seeds", "oat", "robustness", "null199", "summarize", "all"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k-null", type=int, default=DEFAULT_K_NULL)
    parser.add_argument("--k-film", type=int, default=DEFAULT_K_FILM)
    parser.add_argument("--config", type=str, default=str(EXP_ROOT / "configs" / "lambda_global.json"))
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--exp-root", type=str, default=str(EXP_ROOT_DEFAULT))
    args = parser.parse_args()

    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    ds_arg = [args.dataset] if args.dataset else None

    if args.stage in ("seeds", "all"):
        run_stage_seeds(cfg, data_root, exp_root, resume=args.resume, datasets=ds_arg)
    if args.stage in ("oat", "all"):
        run_stage_oat(cfg, data_root, exp_root, resume=args.resume, datasets=ds_arg)
    if args.stage in ("robustness", "all"):
        run_stage_robustness(cfg, data_root, exp_root, resume=args.resume, datasets=ds_arg)
    if args.stage in ("null199", "all"):
        run_stage_null199(cfg, data_root, exp_root, k_null=args.k_null, k_film=args.k_film, resume=args.resume, datasets=ds_arg)
    if args.stage in ("summarize", "all"):
        run_stage_summarize(exp_root)


if __name__ == "__main__":
    main()
