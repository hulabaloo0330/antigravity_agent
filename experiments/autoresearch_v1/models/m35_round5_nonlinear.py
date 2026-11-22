"""Milestone M3.5 — Round 5: Toyoshima 2024 Residual/SNR Diagnosis + Structural Zero-Init
Nonlinear Calcium Dynamics Anchor + 1-Hop/2-Hop Signed Connectome & Causal Patch-FiLM.

Implements 3 resumable stages (`--stage {diagnose,train,score,all}` + `--resume`) across all 3
immobilized Dev splits (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`) under unified
`configs/lambda_global.json` (STRICTLY ZERO `dataset_id` hyperparameter branches, ZERO access to
`vault_sealed/`, processing 1 dataset at a time + `gc.collect()` to keep peak RSS < 3.0 GiB):

1. `--stage diagnose`:
   Inspects `toyoshima2024` Dev data & `B4_ridge_self_stim` / `M3R1_02` predictions:
   - Per-stimulus (`NaCl_up` vs `NaCl_down` ON/OFF asymmetry) residual `E_d` & amplitude,
   - Per-horizon (`[0,2)s`, `[2,5)s`, `[5,10)s`) and per-cell-class + top-10 highest-error neurons,
   - Physical SNR / high-frequency innovation noise-floor ceiling across `toyoshima2024`,
     `seyedolmohadesin2025`, and `yemini2021` -> `results/m35_toyoshima_diagnosis.json`.

2. `--stage train`:
   Trains 3 structural `M3.5` models with fold-internal inner animal-grouped validation early
   stopping and closed-form inner-CV residual shrinkage `alpha* in [0, 1]`:
   - `M35_01_zeroinit_nonlinear_self_anchor`:
     Anchored on `M3R1_02_h1_confounds_only` (`y_lin`), adds a zero-init residual-gated nonlinear
     calcium dynamics MLP (`ReLU(+/-dx)`, `tanh(x)`, `x^2`, multi-tau EMA bank `{1,3,8}s`,
     `self_x_prior`, learnable low-rank neuron-ID + tier embedding, Causal Continuous Patch-Residual
     Head) with inner-CV shrinkage `alpha_self in [0, 1]`.
   - `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film`:
     Anchored on `M35_01` (`y_anchor_nl`), adds 1-hop + 2-hop (`W` and `W^2`, strictly masked
     `diag=0` & `bilateral=0` AFTER squaring!) receptor-polarity-signed connectome drives with
     timescale-to-modality lag matching (fast `[-2,0)s` wired 1-hop/2-hop, slow `[-10,-2)s`
     wireless NPP/monoamine) + excluded-self slow-state into a Zero-Init Patch-FiLM Gate with
     inner-CV shrinkage `alpha_conn in [0, 1]`.
   - `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl`:
     Capacity-matched graph-free control on top of `M35_01` with identical Patch-FiLM architecture,
     state dimensionality, parameter count, and inner-CV shrinkage `alpha_ctrl in [0, 1]`,
     replacing connectome drives with excluded-self population PCA slow-state features.

3. `--stage score`:
   Scores all 3 runs via standalone `evaluators/scorer.py`, computes 1000-resample animal bootstrap
   95% CIs vs BOTH `B4_ridge_self_stim` AND `M35_01_zeroinit_nonlinear_self_anchor`, verifies strict
   leakage invariance (`future_poison == 0.0`, `oof_tamper == 0.0`, `2hop_diag == 0.0`,
   `2hop_bilateral == 0.0`, `AVAR->AVAL / AIYR->AIYL / ASER->ASEL / AWCOFF->AWCON == 0.0`),
   appends 3 rows to `runs.jsonl`, writes `results/m35_round5_summary.json`, and appends Round 5
   to `research_loop_log.md`.
"""

from __future__ import annotations

import argparse
import copy
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
from models.m3_round1_h1 import (
    M3R1Model,
    _build_m3_fold_cache,
    _peak_rss_mb,
    _utc_now,
    compute_macro_bootstrap_ci,
    extract_circuit_slow_state_features,
    extract_multiscale_self_features,
    fit_causal_slow_state_pca,
)
from models.m3_round2_h2 import _apply_svd_compressor, _fit_svd_compressor
from models.m3_round3_h3 import (
    compute_layer_drive,
    extract_multiscale_presynaptic_state,
    load_signed_multilayer_connectome,
)
from src.baselines import (
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
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
    build_post_temporal_basis,
    extract_self_history_features_matrix,
    predict_prior_delta_trajectory,
)

DATASETS_ALL3: List[str] = ["yemini2021", "seyedolmohadesin2025", "toyoshima2024"]
DATASETS_MAIN2: List[str] = ["seyedolmohadesin2025", "toyoshima2024"]
TIER_NAMES: Tuple[str, ...] = ("sensory", "interneuron", "premotor_command", "motor", "pharyngeal")

M35_MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "run_id": "r5_01_m35_01_zeroinit_nonlinear_self_anchor",
        "model_id": "M35_01_zeroinit_nonlinear_self_anchor",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": True,
        "uses_confounds": True,
        "uses_connectome": False,
        "uses_polarity_sign": False,
        "uses_wireless": False,
        "is_stochastic": True,
        "single_factor_changed": "zeroinit_nonlinear_calcium_self_anchor_asymmetric_onoff_ema_emb_patch_head_inner_cv_shrink",
        "hypothesis": (
            "M3.5 self-anchor upgrade: anchored on M3R1_02 linear prediction, adds a zero-init "
            "residual-gated nonlinear calcium dynamics MLP (ReLU(+/-dx), tanh(x), x^2, multi-tau EMA "
            "{1,3,8}s, self_x_prior, learnable neuron-ID+tier embedding, 3-patch continuous residual head) "
            "with inner-CV-validated scalar shrinkage alpha_self in [0,1] to capture Seyed/Yemini "
            "nonlinear dynamics without overfitting low-SNR/small-N Toyoshima."
        ),
    },
    {
        "run_id": "r5_02_m35_02_nonlinear_anchor_1hop2hop_connectome_patch_film",
        "model_id": "M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film",
        "parent_id": "r5_01_m35_01_zeroinit_nonlinear_self_anchor",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": True,
        "uses_polarity_sign": True,
        "uses_wireless": True,
        "is_stochastic": True,
        "single_factor_changed": "add_1hop2hop_signed_lagmatched_connectome_patch_film_on_nonlinear_anchor",
        "hypothesis": (
            "M3.5 full mechanistic model: anchored on M35_01 nonlinear self-anchor, adds 1-hop + 2-hop "
            "(W and W^2, strictly diag=0 & bilateral=0 AFTER squaring) receptor-polarity-signed connectome "
            "drives (fast [-2,0)s wired/gap/2-hop, slow [-10,-2)s wireless NPP/monoamine) + excluded-self "
            "slow state into a Zero-Init Patch-FiLM Gate with inner-CV shrinkage alpha_conn in [0,1]."
        ),
    },
    {
        "run_id": "r5_03_m35_03_nonlinear_anchor_graphfree_patch_film_ctrl",
        "model_id": "M35_03_nonlinear_anchor_graphfree_patch_film_ctrl",
        "parent_id": "r5_01_m35_01_zeroinit_nonlinear_self_anchor",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": False,
        "uses_polarity_sign": False,
        "uses_wireless": False,
        "is_stochastic": True,
        "single_factor_changed": "capacity_matched_graphfree_excluded_self_pca_patch_film_ctrl_on_nonlinear_anchor",
        "hypothesis": (
            "M3.5 capacity-matched graph-free control: identical Patch-FiLM architecture, parameter count, "
            "state dimensionality, and inner-CV shrinkage on top of M35_01, replacing 1-hop/2-hop connectome "
            "drives with excluded-self graph-free population PCA slow-state features."
        ),
    },
]


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
        "models/m3_round2_h2.py",
        "models/m3_round3_h3.py",
        "models/m35_round5_nonlinear.py",
    ]
    for rel in rel_files:
        p = exp_root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: Toyoshima 2024 Residual, Phase & Physical SNR Ceiling Diagnosis
# ---------------------------------------------------------------------------
def _load_saved_preds_by_trial(pred_npz_path: Path, n_canon: int) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    out: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    with np.load(pred_npz_path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"].item()))
        flat = z["flat_y_pred"].astype(np.float32)
        offsets = z["offsets"].astype(np.int64)
        for i, tm in enumerate(meta["trials_meta"]):
            seg = flat[offsets[i] : offsets[i + 1]]
            arr = seg.reshape(n_canon, -1)
            out[str(tm["trial_id"])] = (arr, tm)
    return out


def _compute_dataset_snr_diagnostics(
    dataset_id: str,
    data_root: Path,
    exp_root: Path,
) -> Dict[str, Any]:
    """Compute physical SNR / high-frequency innovation noise-floor ceiling on Dev split."""
    raw_recs = load_dataset(
        dataset_id=dataset_id,
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="m35_diagnose_snr",
    )
    aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
    trials = [
        t
        for r in aligned
        for t in extract_causal_trial_records(
            r, input_sec=10.0, target_sec=10.0, calibration_sec=20.0, fast_anchor_sec=2.0,
            include_synthetic_buffer_windows=False,
        )
    ]
    non_buf = [t for t in trials if not bool(t.get("is_buffer_control", False))]

    # Compute Dev neuron scales s_n using standard evaluator function
    s_n, _, _ = compute_training_fold_neuron_scales(
        [r.traces for r in aligned], [r.obs_mask for r in aligned], floor_percentile=10.0
    )
    s_n = np.maximum(np.asarray(s_n, dtype=np.float64), 1e-6)

    # Frame-to-frame high-frequency innovation variance: 0.5 * Var((y[t+1] - y[t]) / s_n)
    # Computed per worm then averaged across worms to match E_d animal weighting!
    by_worm: Dict[str, List[Mapping[str, Any]]] = {}
    for tr in non_buf:
        by_worm.setdefault(str(tr["animal_id"]), []).append(tr)

    worm_hf_var_post: List[float] = []
    worm_hf_var_pre: List[float] = []
    worm_lag1_corr: List[float] = []
    worm_post_var_norm: List[float] = []
    worm_delta_var_norm: List[float] = []
    worm_persist_ed: List[float] = []
    fps_list: List[float] = []

    for wid, w_trials in sorted(by_worm.items()):
        diffs_post = []
        diffs_pre = []
        y0_list = []
        y1_list = []
        y_post_norm_list = []
        dy_post_norm_list = []
        persist_sq_list = []

        for tr in w_trials:
            fps_list.append(float(tr["fps"]))
            yt = np.asarray(tr["y_true"], dtype=np.float64)       # [N, T_out]
            vm = np.asarray(tr["valid_mask"], dtype=bool)         # [N, T_out]
            yanc = np.asarray(tr["y_pre_anchor"], dtype=np.float64)

            zn = yt / s_n[:, None]
            # Adjacent valid post frames
            adj = vm[:, :-1] & vm[:, 1:] & np.isfinite(zn[:, :-1]) & np.isfinite(zn[:, 1:])
            if np.any(adj):
                d = zn[:, 1:][adj] - zn[:, :-1][adj]
                diffs_post.append(d)
                y0_list.append(zn[:, :-1][adj])
                y1_list.append(zn[:, 1:][adj])

            valid_all = vm & np.isfinite(zn) & np.isfinite(yanc[:, None])
            if np.any(valid_all):
                y_post_norm_list.append(zn[valid_all])
                dy = (yt - yanc[:, None]) / s_n[:, None]
                dy_post_norm_list.append(dy[valid_all])
                persist_sq_list.append((dy[valid_all]) ** 2)

            # Pre-stimulus adjacent frames
            xp = np.asarray(tr["x_pre"], dtype=np.float64)  # [T_pre, N]
            mp = np.asarray(tr["m_pre"], dtype=bool)
            if xp.shape[0] >= 2:
                zp = xp / s_n[None, :]
                adjp = mp[:-1, :] & mp[1:, :] & np.isfinite(zp[:-1, :]) & np.isfinite(zp[1:, :])
                if np.any(adjp):
                    diffs_pre.append(zp[1:, :][adjp] - zp[:-1, :][adjp])

        if diffs_post:
            dp = np.concatenate(diffs_post)
            hf_v = float(0.5 * np.var(dp))
            worm_hf_var_post.append(hf_v)
            a0 = np.concatenate(y0_list)
            a1 = np.concatenate(y1_list)
            if a0.size > 2 and np.std(a0) > 1e-8 and np.std(a1) > 1e-8:
                worm_lag1_corr.append(float(np.corrcoef(a0, a1)[0, 1]))
        if diffs_pre:
            dpr = np.concatenate(diffs_pre)
            worm_hf_var_pre.append(float(0.5 * np.var(dpr)))
        if y_post_norm_list:
            ypn = np.concatenate(y_post_norm_list)
            dypn = np.concatenate(dy_post_norm_list)
            worm_post_var_norm.append(float(np.var(ypn)))
            worm_delta_var_norm.append(float(np.var(dypn)))
            worm_persist_ed.append(float(np.mean(np.concatenate(persist_sq_list))))

    hf_post = float(np.mean(worm_hf_var_post)) if worm_hf_var_post else float("nan")
    hf_pre = float(np.mean(worm_hf_var_pre)) if worm_hf_var_pre else float("nan")
    lag1_r = float(np.mean(worm_lag1_corr)) if worm_lag1_corr else float("nan")
    post_var = float(np.mean(worm_post_var_norm)) if worm_post_var_norm else float("nan")
    delta_var = float(np.mean(worm_delta_var_norm)) if worm_delta_var_norm else float("nan")
    persist_ed = float(np.mean(worm_persist_ed)) if worm_persist_ed else float("nan")

    # Load frozen B4_ridge E_d for exact ratio to noise floor
    b4_path = exp_root / "runs" / "r0_04_b4_ridge_self_stim" / f"predictions_dev_{dataset_id}.npz"
    b4_ed = float("nan")
    if b4_path.exists():
        b4_sc = score_saved_predictions_file(b4_path, split="dev", data_root=data_root, exp_root=exp_root, resolve_registry_ref=False)
        b4_ed = float(b4_sc["E_d"])

    # Effective signal variance = post_var - hf_post (clamped >= 0)
    sig_var = max(0.0, post_var - hf_post)
    snr_hf = float(sig_var / max(hf_post, 1e-8))
    # Empirical upper bound on R^2 given white shot noise floor: 1 - hf_post / post_var
    r2_ceiling = float(max(0.0, 1.0 - hf_post / max(post_var, 1e-8)))
    # Fraction of B4_ridge MSE (E_d) that is irreducible high-frequency shot noise
    noise_frac_of_b4_ed = float(hf_post / max(b4_ed, 1e-8)) if np.isfinite(b4_ed) else float("nan")
    # Maximum theoretical I_d if ALL non-shot-noise residual were eliminated:
    max_theoretical_id_vs_b4 = float(max(0.0, 1.0 - hf_post / max(b4_ed, 1e-8))) if np.isfinite(b4_ed) else float("nan")

    del raw_recs, aligned, trials, non_buf
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "n_dev_worms": len(by_worm),
        "mean_fps": round(float(np.mean(fps_list)), 3) if fps_list else float("nan"),
        "hf_innovation_var_post_norm": round(hf_post, 6),
        "hf_innovation_var_pre_norm": round(hf_pre, 6),
        "lag1_autocorrelation": round(lag1_r, 6),
        "post_total_var_norm": round(post_var, 6),
        "post_delta_var_norm": round(delta_var, 6),
        "persistence_E_d": round(persist_ed, 6),
        "B4_ridge_E_d": round(b4_ed, 6) if np.isfinite(b4_ed) else None,
        "empirical_snr_hf": round(snr_hf, 4),
        "empirical_post_R2_ceiling": round(r2_ceiling, 6),
        "shot_noise_fraction_of_B4_E_d": round(noise_frac_of_b4_ed, 4) if np.isfinite(noise_frac_of_b4_ed) else None,
        "max_non_shot_noise_headroom_vs_B4": round(max_theoretical_id_vs_b4, 4) if np.isfinite(max_theoretical_id_vs_b4) else None,
    }


def run_stage_diagnose(data_root: Path, exp_root: Path) -> Dict[str, Any]:
    t0 = time.time()
    print("[M3.5 Diagnose] Running Toyoshima 2024 residual, phase & cross-dataset physical SNR diagnosis...")

    # 1. Cross-dataset physical SNR & noise-floor ceiling (1 dataset at a time + gc.collect)
    snr_by_dataset: Dict[str, Any] = {}
    for ds in DATASETS_ALL3:
        snr_by_dataset[ds] = _compute_dataset_snr_diagnostics(ds, data_root=data_root, exp_root=exp_root)
        print(
            f"  [SNR] {ds}: fps={snr_by_dataset[ds]['mean_fps']}, "
            f"hf_var={snr_by_dataset[ds]['hf_innovation_var_post_norm']:.4f}, "
            f"lag1_r={snr_by_dataset[ds]['lag1_autocorrelation']:.4f}, "
            f"B4_Ed={snr_by_dataset[ds]['B4_ridge_E_d']}, "
            f"shot_noise_frac_of_B4={snr_by_dataset[ds]['shot_noise_fraction_of_B4_E_d']}"
        )

    # 2. Deep Toyoshima 2024 residual breakdown (B4_ridge vs M3R1_02)
    raw_recs = load_dataset(
        dataset_id="toyoshima2024",
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="m35_diagnose_toyo",
    )
    aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
    trials = [
        t
        for r in aligned
        for t in extract_causal_trial_records(
            r, input_sec=10.0, target_sec=10.0, calibration_sec=20.0, fast_anchor_sec=2.0,
            include_synthetic_buffer_windows=False,
        )
    ]
    non_buf = [t for t in trials if not bool(t.get("is_buffer_control", False))]

    # Reconstruct exact fold-bounded trials, fold s_n, and unpacked predictions matching evaluators/scorer.py
    b4_path = exp_root / "runs" / "r0_04_b4_ridge_self_stim" / "predictions_dev_toyoshima2024.npz"
    m3r1_path = exp_root / "runs" / "r1_02_m3r1_02_h1_confounds_only" / "predictions_dev_toyoshima2024.npz"

    def _unpack_saved_preds(pred_npz: Path):
        with np.load(pred_npz, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"].item()))
            flat = z["flat_y_pred"].astype(np.float32)
            offsets = z["offsets"].astype(np.int64)
        by_sess = {r.animal_id: r for r in aligned}
        f_sn: Dict[int, np.ndarray] = {}
        f_tmap: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for f_idx, sess_list in enumerate(meta["train_fold_session_animal_ids"]):
            tr_recs = [by_sess[s] for s in sess_list if s in by_sess]
            lo_b, hi_b = compute_training_fold_sensor_bounds(
                [r.traces for r in tr_recs], [r.obs_mask for r in tr_recs]
            )
            tr_b = [apply_sensor_bounds_to_aligned(r, lo_b, hi_b) for r in tr_recs]
            sn_f, _, _ = compute_training_fold_neuron_scales(
                [r.traces for r in tr_b], [r.obs_mask for r in tr_b], floor_percentile=10.0
            )
            f_sn[f_idx] = np.maximum(np.asarray(sn_f, dtype=np.float64), 1e-6)
            tmap: Dict[str, Dict[str, Any]] = {}
            for ar in aligned:
                ar_b = apply_sensor_bounds_to_aligned(ar, lo_b, hi_b)
                for tr in extract_causal_trial_records(
                    ar_b, input_sec=10.0, target_sec=10.0, calibration_sec=20.0, fast_anchor_sec=2.0,
                    include_synthetic_buffer_windows=True,
                ):
                    tmap[str(tr["trial_id"])] = tr
            f_tmap[f_idx] = tmap
        preds: Dict[str, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray]] = {}
        for idx_m, tm in enumerate(meta["trials_meta"]):
            f_i = int(tm["fold_idx"])
            tid = str(tm["trial_id"])
            gt_tr = f_tmap[f_i][tid]
            vm = gt_tr["valid_mask"]
            seg = flat[offsets[idx_m] : offsets[idx_m + 1]]
            yp_mat = np.zeros_like(gt_tr["y_true"], dtype=np.float32)
            yp_mat[vm] = seg
            preds[tid] = (yp_mat, tm, gt_tr, f_sn[f_i])
        return preds

    b4_preds = _unpack_saved_preds(b4_path)
    m3r1_preds = _unpack_saved_preds(m3r1_path)

    # Accumulators
    stim_stats: Dict[str, Dict[str, Any]] = {}
    horizon_bins = [("[0,2)s", 0.0, 2.0), ("[2,5)s", 2.0, 5.0), ("[5,10)s", 5.0, 10.0)]
    hor_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        h[0]: {"b4": {}, "m3r1": {}, "persist": {}} for h in horizon_bins
    }
    class_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    neuron_sq_b4: Dict[int, List[float]] = {}
    neuron_sq_m3r1: Dict[int, List[float]] = {}
    neuron_amp: Dict[int, List[float]] = {}
    neuron_worms: Dict[int, set] = {}

    for tid, (yp_b4, tm, tr, sn) in b4_preds.items():
        if bool(tr.get("is_buffer_control", False)) or tid not in m3r1_preds:
            continue
        yp_m3r1, _, _, _ = m3r1_preds[tid]
        wid = str(tr["animal_id"])
        stim = str(tr["stimulus_id"])
        fps = max(float(tr["fps"]), 0.1)

        yt = np.asarray(tr["y_true"], dtype=np.float64)
        vm = np.asarray(tr["valid_mask"], dtype=bool)
        yanc = np.asarray(tr["y_pre_anchor"], dtype=np.float64)

        err_b4 = ((yp_b4.astype(np.float64) - yt) / sn[:, None]) ** 2
        err_m3 = ((yp_m3r1.astype(np.float64) - yt) / sn[:, None]) ** 2
        err_pe = ((yanc[:, None] - yt) / sn[:, None]) ** 2
        dy_norm = (yt - yanc[:, None]) / sn[:, None]

        # Per-stimulus
        st_entry = stim_stats.setdefault(stim, {"b4_by_w": {}, "m3_by_w": {}, "pe_by_w": {}, "abs_dy": [], "signed_dy": [], "n_trials": 0})
        st_entry["n_trials"] += 1
        if np.any(vm):
            st_entry["b4_by_w"].setdefault(wid, []).append(float(np.mean(err_b4[vm])))
            st_entry["m3_by_w"].setdefault(wid, []).append(float(np.mean(err_m3[vm])))
            st_entry["pe_by_w"].setdefault(wid, []).append(float(np.mean(err_pe[vm])))
            st_entry["abs_dy"].append(float(np.mean(np.abs(dy_norm[vm]))))
            st_entry["signed_dy"].append(float(np.mean(dy_norm[vm])))

        # Per-horizon
        t_out = yt.shape[1]
        t_sec = (np.arange(t_out, dtype=np.float64) + 0.5) / fps
        for h_name, lo, hi in horizon_bins:
            hm = (t_sec >= lo) & (t_sec < hi)
            if np.any(hm):
                sub_v = vm[:, hm]
                if np.any(sub_v):
                    hor_acc[h_name]["b4"].setdefault(wid, []).append(float(np.mean(err_b4[:, hm][sub_v])))
                    hor_acc[h_name]["m3r1"].setdefault(wid, []).append(float(np.mean(err_m3[:, hm][sub_v])))
                    hor_acc[h_name]["persist"].setdefault(wid, []).append(float(np.mean(err_pe[:, hm][sub_v])))

        # Per-cell-class & per-neuron
        for n_i in range(len(canon_neurons)):
            rm = vm[n_i]
            if not np.any(rm):
                continue
            b4_v = float(np.mean(err_b4[n_i, rm]))
            m3_v = float(np.mean(err_m3[n_i, rm]))
            amp_v = float(np.mean(np.abs(dy_norm[n_i, rm])))
            neuron_sq_b4.setdefault(n_i, []).append(b4_v)
            neuron_sq_m3r1.setdefault(n_i, []).append(m3_v)
            neuron_amp.setdefault(n_i, []).append(amp_v)
            neuron_worms.setdefault(n_i, set()).add(wid)

            tier = str(functional_tiers[n_i])
            for c_key in ([tier, "command_motor"] if tier in ("premotor_command", "motor") else [tier]):
                ce = class_acc.setdefault(c_key, {"b4": {}, "m3r1": {}})
                ce["b4"].setdefault(wid, []).append(b4_v)
                ce["m3r1"].setdefault(wid, []).append(m3_v)

    def _worm_macro(d: Dict[str, List[float]]) -> float:
        return round(float(np.mean([float(np.mean(v)) for v in d.values()])), 6) if d else float("nan")

    per_stimulus_out: Dict[str, Any] = {}
    for stim, st in sorted(stim_stats.items()):
        b4_e = _worm_macro(st["b4_by_w"])
        m3_e = _worm_macro(st["m3_by_w"])
        pe_e = _worm_macro(st["pe_by_w"])
        per_stimulus_out[stim] = {
            "n_trials": int(st["n_trials"]),
            "B4_ridge_E_d": b4_e,
            "M3R1_02_E_d": m3_e,
            "persistence_E_d": pe_e,
            "M3R1_02_gain_vs_B4": round((b4_e - m3_e) / max(b4_e, 1e-8), 6),
            "mean_abs_response_norm": round(float(np.mean(st["abs_dy"])), 6),
            "mean_signed_response_norm": round(float(np.mean(st["signed_dy"])), 6),
        }

    per_horizon_out: Dict[str, Any] = {}
    for h_name, _, _ in horizon_bins:
        b4_e = _worm_macro(hor_acc[h_name]["b4"])
        m3_e = _worm_macro(hor_acc[h_name]["m3r1"])
        pe_e = _worm_macro(hor_acc[h_name]["persist"])
        per_horizon_out[h_name] = {
            "B4_ridge_E_d": b4_e,
            "M3R1_02_E_d": m3_e,
            "persistence_E_d": pe_e,
            "M3R1_02_gain_vs_B4": round((b4_e - m3_e) / max(b4_e, 1e-8), 6),
        }

    per_cell_class_out: Dict[str, Any] = {}
    for c_key, ce in sorted(class_acc.items()):
        b4_e = _worm_macro(ce["b4"])
        m3_e = _worm_macro(ce["m3r1"])
        per_cell_class_out[c_key] = {
            "B4_ridge_E_d": b4_e,
            "M3R1_02_E_d": m3_e,
            "M3R1_02_gain_vs_B4": round((b4_e - m3_e) / max(b4_e, 1e-8), 6),
        }

    # Top-10 highest-error neurons (observed in >= 3 worms) + key named circuit neurons
    neuron_rows = []
    for n_i, vals in neuron_sq_b4.items():
        n_w = len(neuron_worms.get(n_i, set()))
        if n_w >= 3:
            b4_e = float(np.mean(vals))
            m3_e = float(np.mean(neuron_sq_m3r1[n_i]))
            neuron_rows.append({
                "neuron": str(canon_neurons[n_i]),
                "tier": str(functional_tiers[n_i]),
                "n_worms": n_w,
                "n_trials": len(vals),
                "B4_ridge_E_d": round(b4_e, 6),
                "M3R1_02_E_d": round(m3_e, 6),
                "mean_abs_response_norm": round(float(np.mean(neuron_amp[n_i])), 6),
            })
    neuron_rows.sort(key=lambda x: x["B4_ridge_E_d"], reverse=True)
    top10_neurons = neuron_rows[:10]

    key_names = ("ASEL", "ASER", "AIBL", "AIBR", "AIYL", "AIYR", "AVAL", "AVAR", "AWCON", "AWCOFF")
    key_neuron_map = {r["neuron"]: r for r in neuron_rows if r["neuron"] in key_names}

    diag = {
        "generated_at_utc": _utc_now(),
        "wall_clock_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
        "cross_dataset_physical_snr_ceiling": snr_by_dataset,
        "toyoshima2024_per_stimulus": per_stimulus_out,
        "toyoshima2024_per_horizon": per_horizon_out,
        "toyoshima2024_per_cell_class": per_cell_class_out,
        "toyoshima2024_top10_highest_error_neurons": top10_neurons,
        "toyoshima2024_key_circuit_neurons": key_neuron_map,
    }

    out_path = exp_root / "results" / "m35_toyoshima_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[M3.5 Diagnose] Saved {out_path} in {diag['wall_clock_sec']}s (peak RSS {diag['peak_rss_mb']} MB)")

    del raw_recs, aligned, trials, non_buf, b4_preds, m3r1_preds
    gc.collect()
    return diag


# ---------------------------------------------------------------------------
# Stage 2 Helpers: 1-Hop + 2-Hop Connectome, Nonlinear Self Features, Patch Head
# ---------------------------------------------------------------------------
def build_1hop_2hop_connectome_atlas(
    canonical_neurons: Sequence[str],
    data_root: Path = DATA_ROOT_DEFAULT,
) -> Dict[str, Any]:
    """Extend hermaphrodite signed connectome with strictly leak-free 2-hop paths (`W^2`).

    CRITICAL ANTI-LEAKAGE: `diag=0` and `bilateral=0` (`allow_mask`) are applied **AFTER** squaring
    so 2-hop loops `i -> k -> i` and bilateral paths `i -> k -> bilateral(i)` are identically 0.0!
    """
    atlas = load_signed_multilayer_connectome(canonical_neurons, data_root=data_root)
    mats = dict(atlas["matrices"])
    rsums = dict(atlas["row_sums"])
    allow_f = atlas["allow_mask"].astype(np.float32)

    # 1-hop row-normalized wired adjacency
    w_wired = (mats["W_chem_herm_unsigned"] + mats["W_gap_herm"]).astype(np.float32)
    rs_wired = np.maximum(np.sum(w_wired, axis=1, keepdims=True), 1e-6)
    w_wired_norm = w_wired / rs_wired

    # 2-hop unsigned wired path (strictly masked AFTER squaring!)
    w_wired_2hop = ((w_wired_norm @ w_wired_norm) * allow_f).astype(np.float32)
    w_wired_2hop = np.maximum(w_wired_2hop, 0.0)

    # 2-hop signed chemical path (pos*pos + neg*neg -> excitatory; pos*neg + neg*pos -> inhibitory)
    w_chem_tot = np.maximum(np.sum(mats["W_chem_herm_unsigned"], axis=1, keepdims=True), 1e-6)
    w_chem_signed = (mats["W_chem_herm_pos"] - mats["W_chem_herm_neg"]) / w_chem_tot
    w_chem_2hop_signed = ((w_chem_signed @ w_chem_signed) * allow_f).astype(np.float32)
    w_chem_2hop_pos = np.maximum(w_chem_2hop_signed, 0.0).astype(np.float32)
    w_chem_2hop_neg = np.maximum(-w_chem_2hop_signed, 0.0).astype(np.float32)

    mats["W_wired_2hop"] = w_wired_2hop
    mats["W_chem_2hop_pos"] = w_chem_2hop_pos
    mats["W_chem_2hop_neg"] = w_chem_2hop_neg
    for k in ("W_wired_2hop", "W_chem_2hop_pos", "W_chem_2hop_neg"):
        rsums[k] = np.sum(mats[k], axis=1).astype(np.float32)

    # Strict runtime anti-leakage assertion on all 2-hop matrices
    idx = {n: i for i, n in enumerate(canonical_neurons)}
    check_pairs = [("AVAL", "AVAR"), ("AIYL", "AIYR"), ("ASEL", "ASER"), ("AWCON", "AWCOFF")]
    for k in ("W_wired_2hop", "W_chem_2hop_pos", "W_chem_2hop_neg"):
        m = mats[k]
        if float(np.max(np.abs(np.diag(m)))) != 0.0:
            raise AssertionError(f"Non-zero 2-hop diagonal in {k}")
        if float(np.max(np.abs(m * (1.0 - allow_f)))) != 0.0:
            raise AssertionError(f"Non-zero masked entry in 2-hop {k}")
        for a, b in check_pairs:
            if a in idx and b in idx:
                if float(m[idx[a], idx[b]]) != 0.0 or float(m[idx[b], idx[a]]) != 0.0:
                    raise AssertionError(f"Non-zero 2-hop bilateral edge ({a},{b}) in {k}")

    atlas_out = dict(atlas)
    atlas_out["matrices"] = mats
    atlas_out["row_sums"] = rsums
    return atlas_out


def extract_nonlinear_self_features(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    priors: Mapping[str, Any],
    tier_oh: np.ndarray,
    taus_ema_sec: Tuple[float, ...] = (1.0, 3.0, 8.0),
) -> np.ndarray:
    """Strictly causal nonlinear self-history + stimulus + trial-confound features `[N_canon, D_self_nl]`.

    ZERO cross-neuron information (neuron `i` sees ONLY neuron `i`'s causal history `[-10, 0)s`,
    stimulus identity/prior, tier, and trial-level round/sex metadata).
    """
    mu_train = priors["mu_train"]
    n_canon = len(mu_train)
    h_self = extract_self_history_features_matrix(trial, s_n, mu_train)  # [N, 10]
    h_ms = extract_multiscale_self_features(trial, s_n, mu_train)        # [N, 12]

    # Causal multi-tau EMA bank on z_pre (strictly < onset_idx)
    x_pre = np.asarray(trial["x_pre"], dtype=np.float32)  # [T_pre, N]
    m_pre = np.asarray(trial["m_pre"], dtype=bool)
    t_pre = x_pre.shape[0]
    fps = max(float(trial.get("fps", 2.0)), 0.1)
    dt = 1.0 / fps

    z_pre = np.where(
        m_pre & np.isfinite(x_pre),
        np.clip((x_pre - mu_train[None, :]) / np.maximum(s_n[None, :], 1e-6), -6.0, 6.0),
        0.0,
    ).astype(np.float32)
    m_f = (m_pre & np.isfinite(x_pre)).astype(np.float32)
    t_back = (np.arange(t_pre, 0, -1, dtype=np.float32) - 0.5) * dt  # [T_pre]

    ema_cols: List[np.ndarray] = []
    for tau in taus_ema_sec:
        w = np.exp(-t_back / max(float(tau), 1e-3)).astype(np.float32)[:, None]  # [T_pre, 1]
        num = np.sum(w * z_pre * m_f, axis=0)
        den = np.maximum(np.sum(w * m_f, axis=0), 1e-6)
        ema_cols.append(np.clip(num / den, -6.0, 6.0))
    ema_bank = np.column_stack(ema_cols).astype(np.float32)  # [N, 3]

    # Last valid causal level & asymmetric ON/OFF + saturation nonlinearities (Rusu 2026 + Creamer v4)
    x_last = h_ms[:, 6:7]  # last valid causal z-value
    dx_fast = h_self[:, 4:5]
    dx_slow = h_self[:, 5:6]
    x_cal = h_self[:, 6:7]
    x_pop = h_self[:, 7:8]

    nl_block = np.hstack([
        np.maximum(dx_fast, 0.0),               # ReLU(+dx_fast)
        np.maximum(-dx_fast, 0.0),              # ReLU(-dx_fast)
        np.maximum(dx_slow, 0.0),               # ReLU(+dx_slow)
        np.maximum(-dx_slow, 0.0),              # ReLU(-dx_slow)
        np.tanh(x_last),                        # calcium saturation
        np.tanh(x_cal),                         # within-worm saturation
        np.clip(x_last ** 2, 0.0, 9.0) / 3.0,   # quadratic rebound
        np.clip(x_cal ** 2, 0.0, 9.0) / 3.0,
        ema_bank,                               # multi-tau EMA {1,3,8}s
        x_last - ema_bank,                      # fast-vs-slow rebound (3 dims)
    ]).astype(np.float32)

    # Stimulus + prior + self_x_prior interactions (matching & extending B4_mlp_repaired_self_stim)
    stim_oh = _encode_stim_onehot(
        str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), priors["stim_vocab"]
    )
    stim_mat = np.repeat(stim_oh[None, :], n_canon, axis=0)
    _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, n_canon)
    prior_norm = np.clip(prior_coef / np.maximum(s_n[:, None].astype(np.float32), 1e-6), -6.0, 6.0)
    p0 = prior_norm[:, [0]]
    self_x_prior = np.hstack([
        h_self[:, [4, 6, 7]] * p0,
        np.maximum(dx_fast, 0.0) * p0,
        np.maximum(-dx_fast, 0.0) * p0,
        np.tanh(x_last) * p0,
    ]).astype(np.float32)

    # Trial-level causal context (round habituation + sex, strictly zero cross-neuron info)
    r_idx = float(trial.get("round_idx", 0))
    r_feat = np.full((n_canon, 1), np.clip(np.log1p(max(r_idx, 0.0)) / 3.0, 0.0, 2.0), dtype=np.float32)
    sex_s = str(trial.get("sex", "unknown")).lower()
    sex_v = np.array([1.0 if sex_s == "hermaphrodite" else 0.0, 1.0 if sex_s == "male" else 0.0], dtype=np.float32)
    sex_mat = np.repeat(sex_v[None, :], n_canon, axis=0)

    out = np.hstack([h_self, h_ms, nl_block, tier_oh, stim_mat, prior_norm, self_x_prior, r_feat, sex_mat])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_1hop_2hop_lagmatched_drives(
    trial: Mapping[str, Any],
    atlas: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    kappa: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract lag-matched 1-hop + 2-hop signed wired drives (fast) and wireless NPP/monoamine (slow).

    Per `SBTG v2` + `Sale & Richardson Nov 2026`:
      - Fast lag (`[-2, 0)s` level & slope): 1-hop `W_gap_herm`, `W_chem_herm_pos`, `W_chem_herm_neg`,
        `W_chem_herm_unk`, `W_witvliet_core`, plus 2-hop `W_wired_2hop`, `W_chem_2hop_pos`, `W_chem_2hop_neg`.
      - Slow lag (`[-10, -2)s` level & slow slope): wireless `W_npp_short`, `W_npp_mid`, `W_monoamine`.
    """
    s_mat, m_obs = extract_multiscale_presynaptic_state(trial, s_n, mu_train)
    # s_mat cols: 0=[-10,-5)s, 1=[-5,-2)s, 2=[-2,0)s, 3=slope[-2,0)s
    s_fast = s_mat[:, [2, 3]]  # [N, 2]: fast level & fast slope
    s_slow_lvl = 0.5 * (s_mat[:, 0] + s_mat[:, 1])
    s_slow_slp = np.clip(s_mat[:, 1] - s_mat[:, 0], -6.0, 6.0)
    s_slow = np.column_stack([s_slow_lvl, s_slow_slp]).astype(np.float32)  # [N, 2]

    mats = atlas["matrices"]
    rsums = atlas["row_sums"]

    fast_keys = (
        "W_gap_herm",
        "W_chem_herm_pos",
        "W_chem_herm_neg",
        "W_chem_herm_unk",
        "W_witvliet_core",
        "W_wired_2hop",
        "W_chem_2hop_pos",
        "W_chem_2hop_neg",
    )
    wired_blocks = [
        compute_layer_drive(mats[k], rsums[k], s_fast, m_obs, kappa=kappa)
        for k in fast_keys
    ]
    # Also include slow-lag level for gap + 2-hop wired (sustained/slow propagation)
    for k in ("W_gap_herm", "W_wired_2hop"):
        wired_blocks.append(compute_layer_drive(mats[k], rsums[k], s_slow, m_obs, kappa=kappa))
    d_wired = np.hstack(wired_blocks).astype(np.float32)

    slow_keys = ("W_npp_short", "W_npp_mid", "W_monoamine")
    wl_blocks = [
        compute_layer_drive(mats[k], rsums[k], s_slow, m_obs, kappa=kappa)
        for k in slow_keys
    ]
    d_wireless = np.hstack(wl_blocks).astype(np.float32)
    return d_wired, d_wireless


def _fit_excluded_self_pop_pca(
    trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    n_comp: int = 14,
) -> Dict[str, Any]:
    """Fit fold-internal causal PCA on population multi-scale pre-stim state for graph-free control."""
    rows = []
    for tr in trials:
        s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
        # Flatten [N, 4] -> [4*N] weighted by m_obs
        rows.append((s_mat * m_obs[:, None]).T.reshape(-1))  # [4*N]
    x = np.vstack(rows).astype(np.float64)  # [M_trials, 4*N]
    mu = np.mean(x, axis=0, keepdims=True)
    xc = x - mu
    # Compact SVD on [M_trials, 4*N]
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(int(n_comp), vt.shape[0])
    v = np.zeros((x.shape[1], int(n_comp)), dtype=np.float32)
    v[:, :k] = vt[:k].T.astype(np.float32)
    proj = xc @ v.astype(np.float64)
    p_sd = np.maximum(np.std(proj, axis=0, keepdims=True), 1e-4).astype(np.float32)
    return {
        "mu": mu.astype(np.float32),
        "v": v,
        "p_sd": p_sd,
        "n_comp": int(n_comp),
    }


def _extract_excluded_self_pop_pca_features(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    allow_mask: np.ndarray,
    pop_pca: Mapping[str, Any],
) -> np.ndarray:
    """Compute strictly excluded-self & excluded-bilateral population PCA `[N_canon, n_comp]`.

    For target neuron `i`, zeros out `i` AND `bilateral(i)` from the 4-scale population vector
    BEFORE projecting onto `v`! Exact algebraic rank-2 downdate in O(N * K) time.
    """
    s_mat, m_obs = extract_multiscale_presynaptic_state(trial, s_n, mu_train)  # [N, 4]
    n_canon = s_mat.shape[0]
    mu_4n = pop_pca["mu"].reshape(4, n_canon)    # [4, N]
    v_4nk = pop_pca["v"].reshape(4, n_canon, -1) # [4, N, K]
    p_sd = pop_pca["p_sd"]                       # [1, K]

    x_4n = (s_mat * m_obs[:, None]).T - mu_4n    # [4, N]
    # Total projection: sum_j x_4n[:, j] * v[:, j, :] -> [K]
    total_proj = np.einsum("sn,snk->k", x_4n, v_4nk)  # [K]

    # Per-neuron contribution c[j, k] = sum_s x_4n[s, j] * v[s, j, k] -> [N, K]
    contrib = np.einsum("sn,snk->nk", x_4n, v_4nk)    # [N, K]

    # Excluded mask: (~allow_mask) has True at self (i, i) and bilateral (i, bilat(i))
    excl = (~allow_mask).astype(np.float32)           # [N, N]
    sub = excl @ contrib                              # [N, K]
    out = (total_proj[None, :] - sub) / p_sd          # [N, K]
    return np.clip(out, -5.0, 5.0).astype(np.float32)


def build_patch_to_basis_matrix(b_dim: int = 6, taus_sec: Sequence[float] = (0.75, 2.0, 5.0, 10.0)) -> np.ndarray:
    """Build deterministic `[3, B]` projection matrix mapping 3 causal temporal patches
    (`[0,2)s`, `[2,5)s`, `[5,10)s`) into the orthonormal post-stimulus basis `phi` (`CAPT v3`).
    """
    fps_ref = 10.0
    n_out = 100  # 10.0s at 10 Hz
    u_dummy = np.ones(n_out, dtype=np.float32)
    phi = build_post_temporal_basis(n_out, fps_ref, u_dummy, taus_sec=taus_sec)  # [100, B]
    if phi.shape[1] != b_dim:
        # Truncate or pad safely
        out = np.zeros((3, b_dim), dtype=np.float32)
        k = min(b_dim, phi.shape[1])
        phi = phi[:, :k]
    t_sec = (np.arange(n_out, dtype=np.float64) + 0.5) / fps_ref
    windows = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.01)]
    rows = []
    for lo, hi in windows:
        ind = ((t_sec >= lo) & (t_sec < hi)).astype(np.float64)
        # Normalize patch indicator to unit L2 norm so patch amplitudes are well-scaled
        ind = ind / max(np.linalg.norm(ind), 1e-6)
        rows.append(ind @ phi.astype(np.float64))
    p_mat = np.vstack(rows).astype(np.float32)  # [3, B]
    if p_mat.shape[1] < b_dim:
        pad = np.zeros((3, b_dim - p_mat.shape[1]), dtype=np.float32)
        p_mat = np.hstack([p_mat, pad])
    return p_mat[:, :b_dim]


# ---------------------------------------------------------------------------
# PyTorch Modules: Zero-Init Nonlinear Self Anchor & Zero-Init Patch-FiLM Gate
# ---------------------------------------------------------------------------
class ZeroInitNonlinearSelfNet(nn.Module):
    """Zero-Init Nonlinear Calcium Dynamics Residual MLP with Learnable Neuron+Tier Embedding
    and Causal Continuous 3-Patch Residual Head (`CAPT v3` + `Rusu 2026` + `Allier 2026`).
    """

    def __init__(
        self,
        d_in: int,
        n_canon: int,
        n_tiers: int,
        b_dim: int,
        patch_to_basis: np.ndarray,
        hidden_dim: int = 48,
        emb_dim: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.neuron_emb = nn.Embedding(n_canon, emb_dim)
        self.tier_emb = nn.Linear(n_tiers, emb_dim, bias=False)
        nn.init.normal_(self.neuron_emb.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.tier_emb.weight, mean=0.0, std=0.05)

        self.in_proj = nn.Sequential(
            nn.Linear(d_in + emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Causal Continuous Patch-Residual Head: 3 smooth patches [0,2), [2,5), [5,10)s + fine B-dim residual
        self.patch_head = nn.Linear(hidden_dim, 3)
        self.fine_head = nn.Linear(hidden_dim, b_dim)
        self.register_buffer("patch_to_basis", torch.from_numpy(patch_to_basis.astype(np.float32)))

        # Strictly ZERO-INIT output heads so epoch-0 residual == 0.0 (exact M3R1_02 anchor)
        nn.init.zeros_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

    def forward(self, x: torch.Tensor, n_idx: torch.Tensor, tier_oh: torch.Tensor) -> torch.Tensor:
        e_id = self.neuron_emb(n_idx)
        if self.training:
            # Neuron-ID dropout (25%) so unseen/rare neurons fall back cleanly to tier embedding
            keep = (torch.rand((e_id.shape[0], 1), device=e_id.device) > 0.25).float()
            e_id = e_id * keep
        e_tot = e_id + self.tier_emb(tier_oh)
        h = self.in_proj(torch.cat([x, e_tot], dim=-1))
        h = h + self.res_block(h)
        delta_patch = self.patch_head(h) @ self.patch_to_basis
        delta_fine = self.fine_head(h)
        return delta_patch + delta_fine


class ZeroInitPatchFiLMGate(nn.Module):
    """Zero-Init Bounded Multiplicative FiLM Gate + Causal Continuous Patch-Residual Head.

    Used identically by `M35_02` (1-hop/2-hop signed connectome + slow state) and `M35_03`
    (capacity-matched graph-free excluded-self PCA control) on top of `M35_01`.
    """

    def __init__(
        self,
        d_state: int,
        d_u: int,
        n_canon: int,
        n_tiers: int,
        b_dim: int,
        patch_to_basis: np.ndarray,
        film_dim: int = 24,
        emb_dim: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.neuron_emb = nn.Embedding(n_canon, emb_dim)
        self.tier_emb = nn.Linear(n_tiers, emb_dim, bias=False)
        nn.init.normal_(self.neuron_emb.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.tier_emb.weight, mean=0.0, std=0.05)

        self.stim_enc = nn.Sequential(
            nn.Linear(d_u + n_tiers + emb_dim, film_dim),
            nn.LayerNorm(film_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.state_mod = nn.Sequential(
            nn.Linear(d_state, film_dim),
            nn.LayerNorm(film_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(film_dim, 2 * film_dim),
        )
        self.act_drop = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))
        self.patch_head = nn.Linear(film_dim, 3)
        self.fine_head = nn.Linear(film_dim, b_dim)
        self.register_buffer("patch_to_basis", torch.from_numpy(patch_to_basis.astype(np.float32)))

        # Strictly ZERO-INIT state modulator & output heads so epoch-0 residual == 0.0
        nn.init.zeros_(self.state_mod[-1].weight)
        nn.init.zeros_(self.state_mod[-1].bias)
        nn.init.zeros_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

    def forward(
        self,
        s_state: torch.Tensor,
        u_stim: torch.Tensor,
        tier_oh: torch.Tensor,
        n_idx: torch.Tensor,
    ) -> torch.Tensor:
        e_id = self.neuron_emb(n_idx)
        if self.training:
            keep = (torch.rand((e_id.shape[0], 1), device=e_id.device) > 0.25).float()
            e_id = e_id * keep
        e_tot = e_id + self.tier_emb(tier_oh)
        h = self.stim_enc(torch.cat([u_stim, tier_oh, e_tot], dim=-1))
        gb = self.state_mod(s_state)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        h_mod = self.act_drop((1.0 + 0.5 * torch.tanh(gamma)) * h + beta)
        return (self.patch_head(h_mod) @ self.patch_to_basis) + self.fine_head(h_mod)


def _compute_inner_cv_shrinkage(y_hat_va: np.ndarray, y_va: np.ndarray) -> float:
    """Closed-form inner-CV optimal scalar residual shrinkage `alpha* in [0, 1]`."""
    yh = np.asarray(y_hat_va, dtype=np.float64).ravel()
    yt = np.asarray(y_va, dtype=np.float64).ravel()
    denom = float(np.dot(yh, yh))
    if denom < 1e-10:
        return 0.0
    alpha = float(np.dot(yh, yt) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def _train_torch_residual_with_inner_cv(
    build_net_fn,
    tr_tensors: Tuple[torch.Tensor, ...],
    y_tr: torch.Tensor,
    va_tensors: Tuple[torch.Tensor, ...],
    y_va: torch.Tensor,
    full_tensors: Tuple[torch.Tensor, ...],
    y_full: torch.Tensor,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> Tuple[nn.Module, float, int, float, float]:
    """Train residual PyTorch net with inner-CV early stopping + closed-form `alpha* in [0,1]` shrinkage,
    then refit on full outer training fold for `best_ep` epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    net_in = build_net_fn()
    opt = torch.optim.AdamW(net_in.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_tr = y_tr.shape[0]
    y_va_np = y_va.detach().cpu().numpy()
    base_va_loss = float(np.mean(y_va_np ** 2))
    best_va_loss = base_va_loss
    best_ep = 0
    best_alpha = 0.0
    no_imp = 0

    for ep in range(1, max_epochs + 1):
        net_in.train()
        perm = torch.randperm(n_tr)
        for st in range(0, n_tr, batch_size):
            idx = perm[st : st + batch_size]
            opt.zero_grad()
            pred = net_in(*[t[idx] for t in tr_tensors])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net_in.parameters(), 2.0)
            opt.step()

        net_in.eval()
        with torch.no_grad():
            pred_va = net_in(*va_tensors).cpu().numpy()
        alpha_ep = _compute_inner_cv_shrinkage(pred_va, y_va_np)
        shrunk_loss = float(np.mean((y_va_np - alpha_ep * pred_va) ** 2))
        if shrunk_loss < best_va_loss - 1e-6:
            best_va_loss = shrunk_loss
            best_ep = ep
            best_alpha = alpha_ep
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    # Refit on full outer training fold for `best_ep` epochs (if inner CV found genuine improvement)
    torch.manual_seed(seed + 101)
    np.random.seed(seed + 101)
    net_full = build_net_fn()
    if best_ep > 0 and best_alpha > 0.02:
        opt_f = torch.optim.AdamW(net_full.parameters(), lr=lr, weight_decay=weight_decay)
        n_full = y_full.shape[0]
        for _ in range(best_ep):
            net_full.train()
            perm = torch.randperm(n_full)
            for st in range(0, n_full, batch_size):
                idx = perm[st : st + batch_size]
                opt_f.zero_grad()
                pred = net_full(*[t[idx] for t in full_tensors])
                loss = loss_fn(pred, y_full[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net_full.parameters(), 2.0)
                opt_f.step()
    else:
        best_alpha = 0.0

    net_full.eval()
    return net_full, float(best_alpha), int(best_ep), float(base_va_loss), float(best_va_loss)


# ---------------------------------------------------------------------------
# Unified M3.5 Model Family (M35_01, M35_02, M35_03)
# ---------------------------------------------------------------------------
class M35Model:
    """Implements `M35_01`, `M35_02`, and `M35_03` under unified `configs/lambda_global.json`."""

    def __init__(
        self,
        model_id: str,
        config: Mapping[str, Any],
        canonical_neurons: Sequence[str],
        functional_tiers: Sequence[str],
        atlas: Mapping[str, Any],
    ) -> None:
        self.model_id = str(model_id)
        self.config = dict(config)
        self.hp = dict(config.get("hyperparameters", config))
        self.canonical_neurons = list(canonical_neurons)
        self.functional_tiers = list(functional_tiers)
        self.atlas = atlas
        self.n_canon = len(self.canonical_neurons)
        self.state: Dict[str, Any] = {}

        self._tier_oh = np.zeros((self.n_canon, len(TIER_NAMES)), dtype=np.float32)
        for i, t_str in enumerate(self.functional_tiers):
            if t_str in TIER_NAMES:
                self._tier_oh[i, TIER_NAMES.index(t_str)] = 1.0

    def _extract_u_stim(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
    ) -> np.ndarray:
        stim_id = str(trial["stimulus_id"])
        is_buf = bool(trial.get("is_buffer_control", False))
        stim_oh = _encode_stim_onehot(stim_id, is_buf, priors["stim_vocab"])[:-1]
        stim_mat = np.repeat(stim_oh[None, :], self.n_canon, axis=0)
        _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, self.n_canon)
        prior_norm_2 = np.clip((prior_coef / np.maximum(s_n[:, None].astype(np.float32), 1e-6))[:, :2], -4.0, 4.0)
        return np.hstack([stim_mat, prior_norm_2]).astype(np.float32)

    def _predict_m35_01_coef(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        st01: Mapping[str, Any],
    ) -> np.ndarray:
        anchor = st01["anchor"]
        priors = st01["priors"]
        p1 = anchor._predict_stage1_coef(trial, s_n, priors, anchor.state["s1"])
        d2 = anchor._predict_stage2_delta_coef(trial, s_n, priors, anchor.state["s2"])
        p_lin = (p1 + d2).astype(np.float32)

        alpha = float(st01["alpha_self"])
        if alpha <= 0.0:
            return p_lin

        x_nl = extract_nonlinear_self_features(trial, s_n, priors, self._tier_oh)
        x_norm = np.clip((x_nl - st01["x_mu"]) / st01["x_sd"], -5.0, 5.0).astype(np.float32)
        net: ZeroInitNonlinearSelfNet = st01["net_self"]
        net.eval()
        with torch.no_grad():
            delta = net(
                torch.from_numpy(x_norm),
                torch.arange(self.n_canon, dtype=torch.long),
                torch.from_numpy(self._tier_oh),
            ).numpy().astype(np.float32)
        return (p_lin + alpha * delta).astype(np.float32)

    def fit_shared_nonlinear_anchor(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit `M35_01_zeroinit_nonlinear_self_anchor` once per fold (also reused by M35_02 & M35_03)."""
        seed = int(self.hp.get("seed", 20261117))
        lr = float(self.hp.get("learning_rate", 0.003))
        wd = float(self.hp.get("weight_decay", 0.01))
        h_dim = int(self.hp.get("mlp_hidden_dim", 48))
        drop = float(self.hp.get("dropout", 0.15))
        bs = int(self.hp.get("batch_size", 2048))
        max_ep = int(self.hp.get("max_epochs", 18))
        patience = int(self.hp.get("early_stopping_patience", 4))
        taus = tuple(float(x) for x in self.hp.get("basis_taus_sec", (0.75, 2.0, 5.0, 10.0)))

        non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        # 1. Outer linear anchor M3R1_02
        anchor_out = M3R1Model("M3R1_02_h1_confounds_only", self.config, self.canonical_neurons, self.functional_tiers)
        anchor_out.fit(train_trials, s_n, fold_cache=fold_cache)
        priors_out = anchor_out.state["priors"]
        b_dim = int(priors_out["b_dim"])
        patch_mat = build_patch_to_basis_matrix(b_dim=b_dim, taus_sec=taus)

        # 2. Inner animal-grouped split (strictly leak-free!)
        worms = sorted({str(t["animal_id"]) for t in non_buf})
        n_val_w = max(1, len(worms) // 5)
        val_worms = set(worms[-n_val_w:])
        inner_tr = [t for t in non_buf if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in non_buf if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(non_buf)
            inner_va = list(non_buf[:1])

        anchor_in = M3R1Model("M3R1_02_h1_confounds_only", self.config, self.canonical_neurons, self.functional_tiers)
        anchor_in.fit(inner_tr, s_n, fold_cache=None)
        priors_in = anchor_in.state["priors"]

        def _collect_self(tr_list: Sequence[Mapping[str, Any]], anc: M3R1Model, pr: Mapping[str, Any]):
            xl, yl, nl, tl = [], [], [], []
            for tr in tr_list:
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, pr, s_n, self.n_canon)
                p1 = anc._predict_stage1_coef(tr, s_n, pr, anc.state["s1"])
                d2 = anc._predict_stage2_delta_coef(tr, s_n, pr, anc.state["s2"])
                r_lin = (c_norm - (p1 + d2)).astype(np.float32)
                x_nl = extract_nonlinear_self_features(tr, s_n, pr, self._tier_oh)
                idx = np.where(rval)[0]
                if idx.size > 0:
                    xl.append(x_nl[idx])
                    yl.append(r_lin[idx])
                    nl.append(idx.astype(np.int64))
                    tl.append(self._tier_oh[idx])
            return np.vstack(xl), np.vstack(yl), np.concatenate(nl), np.vstack(tl)

        x_itr, y_itr, n_itr, t_itr = _collect_self(inner_tr, anchor_in, priors_in)
        x_iva, y_iva, n_iva, t_iva = _collect_self(inner_va, anchor_in, priors_in)
        x_all, y_all, n_all, t_all = _collect_self(non_buf, anchor_out, priors_out)

        mu_in = np.mean(x_itr, axis=0, keepdims=True).astype(np.float32)
        sd_in = np.maximum(np.std(x_itr, axis=0, keepdims=True), 1e-4).astype(np.float32)
        x_itr_n = np.clip((x_itr - mu_in) / sd_in, -5.0, 5.0)
        x_iva_n = np.clip((x_iva - mu_in) / sd_in, -5.0, 5.0)

        mu_out = np.mean(x_all, axis=0, keepdims=True).astype(np.float32)
        sd_out = np.maximum(np.std(x_all, axis=0, keepdims=True), 1e-4).astype(np.float32)
        x_all_n = np.clip((x_all - mu_out) / sd_out, -5.0, 5.0)

        d_in = x_itr_n.shape[1]

        def _build_self_net() -> ZeroInitNonlinearSelfNet:
            return ZeroInitNonlinearSelfNet(
                d_in=d_in,
                n_canon=self.n_canon,
                n_tiers=len(TIER_NAMES),
                b_dim=b_dim,
                patch_to_basis=patch_mat,
                hidden_dim=h_dim,
                emb_dim=8,
                dropout=drop,
            )

        net_full, alpha_self, best_ep, base_va, best_va = _train_torch_residual_with_inner_cv(
            build_net_fn=_build_self_net,
            tr_tensors=(torch.from_numpy(x_itr_n), torch.from_numpy(n_itr), torch.from_numpy(t_itr)),
            y_tr=torch.from_numpy(y_itr),
            va_tensors=(torch.from_numpy(x_iva_n), torch.from_numpy(n_iva), torch.from_numpy(t_iva)),
            y_va=torch.from_numpy(y_iva),
            full_tensors=(torch.from_numpy(x_all_n), torch.from_numpy(n_all), torch.from_numpy(t_all)),
            y_full=torch.from_numpy(y_all),
            lr=lr,
            weight_decay=wd,
            batch_size=bs,
            max_epochs=max_ep,
            patience=patience,
            seed=seed,
        )

        # Also keep an inner M35_01 net (trained strictly on inner_tr for best_ep) so M35_02 & M35_03
        # compute strictly leak-free inner-val residuals on top of M35_01!
        torch.manual_seed(seed + 55)
        np.random.seed(seed + 55)
        net_in = _build_self_net()
        if best_ep > 0 and alpha_self > 0.02:
            opt_i = torch.optim.AdamW(net_in.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.MSELoss()
            for _ in range(best_ep):
                net_in.train()
                perm = torch.randperm(x_itr_n.shape[0])
                for st in range(0, x_itr_n.shape[0], bs):
                    idx = perm[st : st + bs]
                    opt_i.zero_grad()
                    p = net_in(
                        torch.from_numpy(x_itr_n[idx]),
                        torch.from_numpy(n_itr[idx]),
                        torch.from_numpy(t_itr[idx]),
                    )
                    loss = loss_fn(p, torch.from_numpy(y_itr[idx]))
                    loss.backward()
                    nn.utils.clip_grad_norm_(net_in.parameters(), 2.0)
                    opt_i.step()
        net_in.eval()

        st01_outer = {
            "anchor": anchor_out,
            "priors": priors_out,
            "net_self": net_full,
            "x_mu": mu_out,
            "x_sd": sd_out,
            "alpha_self": alpha_self,
            "best_ep_self": best_ep,
            "base_va_loss_self": base_va,
            "best_va_loss_self": best_va,
            "patch_mat": patch_mat,
            "b_dim": b_dim,
        }
        st01_inner = {
            "anchor": anchor_in,
            "priors": priors_in,
            "net_self": net_in,
            "x_mu": mu_in,
            "x_sd": sd_in,
            "alpha_self": alpha_self,
            "patch_mat": patch_mat,
            "b_dim": b_dim,
        }
        return {
            "outer": st01_outer,
            "inner": st01_inner,
            "inner_tr": inner_tr,
            "inner_va": inner_va,
            "non_buf": non_buf,
        }

    def fit(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
        shared_m35_01: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.state = {"s_n": np.asarray(s_n, dtype=np.float32).copy()}
        if shared_m35_01 is None:
            shared_m35_01 = self.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=fold_cache)

        st01_out = shared_m35_01["outer"]
        st01_in = shared_m35_01["inner"]
        self.state["st01"] = st01_out
        self.state["priors"] = st01_out["priors"]

        if self.model_id == "M35_01_zeroinit_nonlinear_self_anchor":
            return

        # Stage 2: Fit Zero-Init Patch-FiLM Gate on residual of M35_01 (either connectome or graph-free ctrl)
        seed = int(self.hp.get("seed", 20261117)) + (200 if "connectome" in self.model_id else 300)
        lr = float(self.hp.get("learning_rate", 0.003))
        wd = float(self.hp.get("weight_decay", 0.01))
        film_dim = int(self.hp.get("poco_film_dim", 24))
        drop = float(self.hp.get("dropout", 0.15))
        bs = int(self.hp.get("batch_size", 2048))
        max_ep = int(self.hp.get("max_epochs", 18))
        patience = int(self.hp.get("early_stopping_patience", 4))
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        wl_rank = int(self.hp.get("wireless_rank", 6))
        rank = int(self.hp.get("rank", 6))

        inner_tr = shared_m35_01["inner_tr"]
        inner_va = shared_m35_01["inner_va"]
        non_buf = shared_m35_01["non_buf"]
        b_dim = int(st01_out["b_dim"])
        patch_mat = st01_out["patch_mat"]

        use_connectome = self.model_id == "M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film"
        p_c, p_z = 8, 6
        d_state = p_c + p_z  # 14 dims identically for BOTH M35_02 and M35_03!

        if use_connectome:
            # Fit inner & outer causal PCA for excluded-self circuit slow-state
            pca_in = fit_causal_slow_state_pca(
                inner_tr, s_n, st01_in["priors"]["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
            )
            pca_out = fit_causal_slow_state_pca(
                non_buf, s_n, st01_out["priors"]["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
            )

            def _collect_conn(tr_list: Sequence[Mapping[str, Any]], st01: Mapping[str, Any], pca_st: Mapping[str, Any]):
                dw_l, dwl_l, z_l, u_l, y_l, n_l, t_l = [], [], [], [], [], [], []
                pr = st01["priors"]
                mu = pr["mu_train"]
                for tr in tr_list:
                    c_norm, rval, _, _ = _project_trial_residual_targets(tr, pr, s_n, self.n_canon)
                    p_nl = self._predict_m35_01_coef(tr, s_n, st01)
                    r_nl = (c_norm - p_nl).astype(np.float32)
                    d_w, d_wl = extract_1hop_2hop_lagmatched_drives(tr, self.atlas, s_n, mu, kappa=kappa)
                    z_c = extract_circuit_slow_state_features(tr, self.canonical_neurons, s_n, mu, pca_st, kappa=kappa)
                    u_st = self._extract_u_stim(tr, s_n, pr)
                    idx = np.where(rval)[0]
                    if idx.size > 0:
                        dw_l.append(d_w[idx])
                        dwl_l.append(d_wl[idx])
                        z_l.append(z_c[idx])
                        u_l.append(u_st[idx])
                        y_l.append(r_nl[idx])
                        n_l.append(idx.astype(np.int64))
                        t_l.append(self._tier_oh[idx])
                return (
                    np.vstack(dw_l), np.vstack(dwl_l), np.vstack(z_l),
                    np.vstack(u_l), np.vstack(y_l), np.concatenate(n_l), np.vstack(t_l),
                )

            dw_itr, dwl_itr, z_itr, u_itr, y_itr, n_itr, t_itr = _collect_conn(inner_tr, st01_in, pca_in)
            dw_iva, dwl_iva, z_iva, u_iva, y_iva, n_iva, t_iva = _collect_conn(inner_va, st01_in, pca_in)
            dw_all, dwl_all, z_all, u_all, y_all, n_all, t_all = _collect_conn(non_buf, st01_out, pca_out)

            wl_comp_in = _fit_svd_compressor(dwl_itr, n_comp=min(wl_rank, dwl_itr.shape[1]))
            c_itr = np.hstack([dw_itr, _apply_svd_compressor(dwl_itr, wl_comp_in)])
            c_iva = np.hstack([dw_iva, _apply_svd_compressor(dwl_iva, wl_comp_in)])
            c_comp_in = _fit_svd_compressor(c_itr, n_comp=p_c)
            z_comp_in = _fit_svd_compressor(z_itr, n_comp=p_z)
            s_itr = np.hstack([_apply_svd_compressor(c_itr, c_comp_in), _apply_svd_compressor(z_itr, z_comp_in)]).astype(np.float32)
            s_iva = np.hstack([_apply_svd_compressor(c_iva, c_comp_in), _apply_svd_compressor(z_iva, z_comp_in)]).astype(np.float32)

            wl_comp_out = _fit_svd_compressor(dwl_all, n_comp=min(wl_rank, dwl_all.shape[1]))
            c_all = np.hstack([dw_all, _apply_svd_compressor(dwl_all, wl_comp_out)])
            c_comp_out = _fit_svd_compressor(c_all, n_comp=p_c)
            z_comp_out = _fit_svd_compressor(z_all, n_comp=p_z)
            s_all = np.hstack([_apply_svd_compressor(c_all, c_comp_out), _apply_svd_compressor(z_all, z_comp_out)]).astype(np.float32)

            self.state["pca_out"] = pca_out
            self.state["wl_comp_out"] = wl_comp_out
            self.state["c_comp_out"] = c_comp_out
            self.state["z_comp_out"] = z_comp_out
        else:
            # Capacity-matched graph-free control: excluded-self population PCA (exact d_state = 14)
            allow_mask = self.atlas["allow_mask"]
            pop_pca_in = _fit_excluded_self_pop_pca(inner_tr, s_n, st01_in["priors"]["mu_train"], n_comp=d_state)
            pop_pca_out = _fit_excluded_self_pop_pca(non_buf, s_n, st01_out["priors"]["mu_train"], n_comp=d_state)

            def _collect_gf(tr_list: Sequence[Mapping[str, Any]], st01: Mapping[str, Any], ppca: Mapping[str, Any]):
                s_l, u_l, y_l, n_l, t_l = [], [], [], [], []
                pr = st01["priors"]
                mu = pr["mu_train"]
                for tr in tr_list:
                    c_norm, rval, _, _ = _project_trial_residual_targets(tr, pr, s_n, self.n_canon)
                    p_nl = self._predict_m35_01_coef(tr, s_n, st01)
                    r_nl = (c_norm - p_nl).astype(np.float32)
                    s_gf = _extract_excluded_self_pop_pca_features(tr, s_n, mu, allow_mask, ppca)
                    u_st = self._extract_u_stim(tr, s_n, pr)
                    idx = np.where(rval)[0]
                    if idx.size > 0:
                        s_l.append(s_gf[idx])
                        u_l.append(u_st[idx])
                        y_l.append(r_nl[idx])
                        n_l.append(idx.astype(np.int64))
                        t_l.append(self._tier_oh[idx])
                return (
                    np.vstack(s_l), np.vstack(u_l), np.vstack(y_l), np.concatenate(n_l), np.vstack(t_l),
                )

            s_itr, u_itr, y_itr, n_itr, t_itr = _collect_gf(inner_tr, st01_in, pop_pca_in)
            s_iva, u_iva, y_iva, n_iva, t_iva = _collect_gf(inner_va, st01_in, pop_pca_in)
            s_all, u_all, y_all, n_all, t_all = _collect_gf(non_buf, st01_out, pop_pca_out)
            self.state["pop_pca_out"] = pop_pca_out

        d_u = u_itr.shape[1]

        def _build_film_net() -> ZeroInitPatchFiLMGate:
            return ZeroInitPatchFiLMGate(
                d_state=d_state,
                d_u=d_u,
                n_canon=self.n_canon,
                n_tiers=len(TIER_NAMES),
                b_dim=b_dim,
                patch_to_basis=patch_mat,
                film_dim=film_dim,
                emb_dim=8,
                dropout=drop,
            )

        net_film, alpha_film, best_ep_film, base_va, best_va = _train_torch_residual_with_inner_cv(
            build_net_fn=_build_film_net,
            tr_tensors=(torch.from_numpy(s_itr), torch.from_numpy(u_itr), torch.from_numpy(t_itr), torch.from_numpy(n_itr)),
            y_tr=torch.from_numpy(y_itr),
            va_tensors=(torch.from_numpy(s_iva), torch.from_numpy(u_iva), torch.from_numpy(t_iva), torch.from_numpy(n_iva)),
            y_va=torch.from_numpy(y_iva),
            full_tensors=(torch.from_numpy(s_all), torch.from_numpy(u_all), torch.from_numpy(t_all), torch.from_numpy(n_all)),
            y_full=torch.from_numpy(y_all),
            lr=lr,
            weight_decay=wd,
            batch_size=bs,
            max_epochs=max_ep,
            patience=patience,
            seed=seed,
        )

        self.state["net_film"] = net_film
        self.state["alpha_film"] = alpha_film
        self.state["best_ep_film"] = best_ep_film
        self.state["base_va_loss_film"] = base_va
        self.state["best_va_loss_film"] = best_va
        self.state["film_param_count"] = int(sum(p.numel() for p in net_film.parameters()))

    def predict_trial(self, trial: Mapping[str, Any]) -> np.ndarray:
        s_n = self.state["s_n"]
        st01 = self.state["st01"]
        priors = st01["priors"]
        mu_train = priors["mu_train"]
        prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, self.n_canon)

        p_coef = self._predict_m35_01_coef(trial, s_n, st01)
        if self.model_id != "M35_01_zeroinit_nonlinear_self_anchor":
            alpha_f = float(self.state.get("alpha_film", 0.0))
            if alpha_f > 0.0:
                kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
                if self.model_id == "M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film":
                    d_w, d_wl = extract_1hop_2hop_lagmatched_drives(trial, self.atlas, s_n, mu_train, kappa=kappa)
                    z_c = extract_circuit_slow_state_features(
                        trial, self.canonical_neurons, s_n, mu_train, self.state["pca_out"], kappa=kappa
                    )
                    c_full = np.hstack([d_w, _apply_svd_compressor(d_wl, self.state["wl_comp_out"])])
                    s_state = np.hstack([
                        _apply_svd_compressor(c_full, self.state["c_comp_out"]),
                        _apply_svd_compressor(z_c, self.state["z_comp_out"]),
                    ]).astype(np.float32)
                else:
                    s_state = _extract_excluded_self_pop_pca_features(
                        trial, s_n, mu_train, self.atlas["allow_mask"], self.state["pop_pca_out"]
                    )
                u_st = self._extract_u_stim(trial, s_n, priors)
                net_f: ZeroInitPatchFiLMGate = self.state["net_film"]
                net_f.eval()
                with torch.no_grad():
                    delta_f = net_f(
                        torch.from_numpy(s_state),
                        torch.from_numpy(u_st),
                        torch.from_numpy(self._tier_oh),
                        torch.arange(self.n_canon, dtype=torch.long),
                    ).numpy().astype(np.float32)
                p_coef = (p_coef + alpha_f * delta_f).astype(np.float32)

        return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, p_coef, s_n, mu_train)


# ---------------------------------------------------------------------------
# Stage 2: Per-Dataset Training & Checkpointing
# ---------------------------------------------------------------------------
def run_stage_train(
    config: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    resume: bool = False,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cache_path = exp_root / "results" / "_m35_r5_train_cache.json"
    cache: Dict[str, Any] = {}
    if resume and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    train_meta: Dict[str, Any] = cache.get("datasets", {})
    t_total0 = time.time()

    ds_list = list(datasets) if datasets else DATASETS_ALL3
    for ds in ds_list:
        all_done = True
        for spec in M35_MODEL_SPECS:
            p_file = exp_root / "runs" / spec["run_id"] / f"predictions_dev_{ds}.npz"
            if not p_file.exists():
                all_done = False
                break
        if resume and all_done and ds in train_meta:
            print(f"[M3.5 Train] Skipping {ds} (already checkpointed).")
            continue

        t_ds0 = time.time()
        print(f"\n[M3.5 Train] === Dataset: {ds} ===")
        raw_recs = load_dataset(
            dataset_id=ds,
            split="dev",
            allow_vault_unseal=False,
            data_root=data_root,
            exp_root=exp_root,
            caller_reason="m35_round5_train",
        )
        aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        atlas = build_1hop_2hop_connectome_atlas(canon_neurons, data_root=data_root)

        hp = dict(config.get("hyperparameters", config))
        in_s = float(hp.get("input_window_sec", 10.0))
        out_s = float(hp.get("target_window_sec", 10.0))
        cal_s = float(hp.get("calibration_window_sec", 20.0))
        fast_s = float(hp.get("fast_anchor_sec", 2.0))

        preds_by_model: Dict[str, List[np.ndarray]] = {s["model_id"]: [] for s in M35_MODEL_SPECS}
        val_trials_all: List[Dict[str, Any]] = []
        val_fold_ids: List[int] = []
        train_fold_sessions: List[List[str]] = []
        diag_folds: Dict[str, List[Dict[str, Any]]] = {s["model_id"]: [] for s in M35_MODEL_SPECS}

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_fold_sessions.append(sorted({str(aligned[i].animal_id) for i in train_idx}))
            lo_b, hi_b = compute_training_fold_sensor_bounds(
                [aligned[i].traces for i in train_idx],
                [aligned[i].obs_mask for i in train_idx],
            )
            train_b = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in train_idx]
            val_b = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in val_idx]
            s_n, _, _ = compute_training_fold_neuron_scales(
                [r.traces for r in train_b], [r.obs_mask for r in train_b], floor_percentile=10.0
            )

            train_trials = [
                t
                for r in train_b
                for t in extract_causal_trial_records(
                    r, input_sec=in_s, target_sec=out_s, calibration_sec=cal_s, fast_anchor_sec=fast_s,
                    include_synthetic_buffer_windows=True,
                )
            ]
            val_trials = [
                t
                for r in val_b
                for t in extract_causal_trial_records(
                    r, input_sec=in_s, target_sec=out_s, calibration_sec=cal_s, fast_anchor_sec=fast_s,
                    include_synthetic_buffer_windows=True,
                )
            ]
            fold_cache = _build_m3_fold_cache(train_trials + val_trials, train_trials, s_n, len(canon_neurons), config)

            # Fit shared M35_01 nonlinear anchor ONCE per fold
            m01 = M35Model("M35_01_zeroinit_nonlinear_self_anchor", config, canon_neurons, functional_tiers, atlas)
            shared_01 = m01.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=fold_cache)
            m01.fit(train_trials, s_n, fold_cache=fold_cache, shared_m35_01=shared_01)

            m02 = M35Model("M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film", config, canon_neurons, functional_tiers, atlas)
            m02.fit(train_trials, s_n, fold_cache=fold_cache, shared_m35_01=shared_01)

            m03 = M35Model("M35_03_nonlinear_anchor_graphfree_patch_film_ctrl", config, canon_neurons, functional_tiers, atlas)
            m03.fit(train_trials, s_n, fold_cache=fold_cache, shared_m35_01=shared_01)

            st01 = m01.state["st01"]
            diag_folds["M35_01_zeroinit_nonlinear_self_anchor"].append({
                "fold": fold_idx,
                "alpha_self": round(float(st01["alpha_self"]), 4),
                "best_ep": int(st01["best_ep_self"]),
                "base_va": round(float(st01["base_va_loss_self"]), 5),
                "best_va": round(float(st01["best_va_loss_self"]), 5),
            })
            diag_folds["M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film"].append({
                "fold": fold_idx,
                "alpha_conn": round(float(m02.state["alpha_film"]), 4),
                "best_ep": int(m02.state["best_ep_film"]),
                "param_count": int(m02.state["film_param_count"]),
                "base_va": round(float(m02.state["base_va_loss_film"]), 5),
                "best_va": round(float(m02.state["best_va_loss_film"]), 5),
            })
            diag_folds["M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"].append({
                "fold": fold_idx,
                "alpha_ctrl": round(float(m03.state["alpha_film"]), 4),
                "best_ep": int(m03.state["best_ep_film"]),
                "param_count": int(m03.state["film_param_count"]),
                "base_va": round(float(m03.state["base_va_loss_film"]), 5),
                "best_va": round(float(m03.state["best_va_loss_film"]), 5),
            })

            print(
                f"  fold {fold_idx}: M35_01 alpha={st01['alpha_self']:.3f} (ep={st01['best_ep_self']}) | "
                f"M35_02 alpha={m02.state['alpha_film']:.3f} (ep={m02.state['best_ep_film']}, P={m02.state['film_param_count']}) | "
                f"M35_03 alpha={m03.state['alpha_film']:.3f} (ep={m03.state['best_ep_film']}, P={m03.state['film_param_count']})"
            )

            for tr in val_trials:
                val_trials_all.append(tr)
                val_fold_ids.append(int(fold_idx))
                preds_by_model["M35_01_zeroinit_nonlinear_self_anchor"].append(m01.predict_trial(tr))
                preds_by_model["M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film"].append(m02.predict_trial(tr))
                preds_by_model["M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"].append(m03.predict_trial(tr))

            del m01, m02, m03, shared_01, fold_cache, train_trials, val_trials
            gc.collect()

        # Save predictions for all 3 models on this dataset
        for spec in M35_MODEL_SPECS:
            run_dir = exp_root / "runs" / spec["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            out_npz = run_dir / f"predictions_dev_{ds}.npz"
            save_CompactPredictions(
                out_path=out_npz,
                trials=val_trials_all,
                y_preds=preds_by_model[spec["model_id"]],
                train_fold_animal_ids=train_fold_sessions,
                fold_ids=np.array(val_fold_ids, dtype=np.int32),
            )

        ds_sec = round(time.time() - t_ds0, 2)
        train_meta[ds] = {
            "wall_clock_sec": ds_sec,
            "peak_rss_mb": _peak_rss_mb(),
            "fold_diagnostics": diag_folds,
        }
        cache["datasets"] = train_meta
        cache["updated_at_utc"] = _utc_now()
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"[M3.5 Train] Finished {ds} in {ds_sec}s (peak RSS {_peak_rss_mb()} MB)")

        del raw_recs, aligned, splits, atlas, preds_by_model
        gc.collect()

    cache["total_wall_clock_sec"] = round(time.time() - t_total0, 2)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


# ---------------------------------------------------------------------------
# Stage 3: Dual-Reference Scoring, Leakage Invariance & Logging
# ---------------------------------------------------------------------------
def verify_m35_02_leakage_invariance(
    config: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
) -> Dict[str, Any]:
    """Verify strict leakage invariance on `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film`:
      1. Future-frame poison test (`y_true` poisoned with `1e6` -> `max_abs_diff == 0.0`)
      2. OOF-tamper invariance (validation animals tampered -> identical fitted predictions `diff == 0.0`)
      3. 2-hop connectome `diag == 0.0`, `bilateral == 0.0`, and explicit pairs
         `AVAR->AVAL`, `AIYR->AIYL`, `ASER->ASEL`, `AWCOFF->AWCON == 0.0`.
    """
    raw_recs = load_dataset(
        dataset_id="toyoshima2024",
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="m35_leakage_verify",
    )
    aligned, canon_neurons, functional_tiers = align_records_to_canonical(raw_recs)
    splits = build_grouped_cv_splits(aligned)
    atlas = build_1hop_2hop_connectome_atlas(canon_neurons, data_root=data_root)

    # 2-hop structural checks
    idx = {n: i for i, n in enumerate(canon_neurons)}
    allow_f = atlas["allow_mask"].astype(np.float32)
    max_2hop_diag = 0.0
    max_2hop_masked = 0.0
    pair_checks: Dict[str, float] = {}
    check_pairs = [("AVAL", "AVAR"), ("AIYL", "AIYR"), ("ASEL", "ASER"), ("AWCON", "AWCOFF")]
    for k in ("W_wired_2hop", "W_chem_2hop_pos", "W_chem_2hop_neg"):
        m = atlas["matrices"][k]
        max_2hop_diag = max(max_2hop_diag, float(np.max(np.abs(np.diag(m)))))
        max_2hop_masked = max(max_2hop_masked, float(np.max(np.abs(m * (1.0 - allow_f)))))
        for a, b in check_pairs:
            if a in idx and b in idx:
                pair_checks[f"{k}:{b}->{a}"] = float(m[idx[a], idx[b]])
                pair_checks[f"{k}:{a}->{b}"] = float(m[idx[b], idx[a]])

    train_idx, val_idx = splits[0]
    lo_b, hi_b = compute_training_fold_sensor_bounds(
        [aligned[i].traces for i in train_idx],
        [aligned[i].obs_mask for i in train_idx],
    )
    train_b = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in train_idx]
    val_b = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in val_idx]
    s_n, _, _ = compute_training_fold_neuron_scales(
        [r.traces for r in train_b], [r.obs_mask for r in train_b], floor_percentile=10.0
    )
    train_trials = [
        t
        for r in train_b
        for t in extract_causal_trial_records(
            r, input_sec=10.0, target_sec=10.0, calibration_sec=20.0, fast_anchor_sec=2.0,
            include_synthetic_buffer_windows=True,
        )
    ]
    val_trials = [
        t
        for r in val_b
        for t in extract_causal_trial_records(
            r, input_sec=10.0, target_sec=10.0, calibration_sec=20.0, fast_anchor_sec=2.0,
            include_synthetic_buffer_windows=True,
        )
    ]

    m_a = M35Model("M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film", config, canon_neurons, functional_tiers, atlas)
    m_a.fit(train_trials, s_n)

    # 1. Future-frame poison test on val_trials[0]
    tr_clean = val_trials[0]
    tr_poison = copy.deepcopy(tr_clean)
    tr_poison["y_true"] = np.full_like(tr_poison["y_true"], 1e6, dtype=np.float32)
    pred_clean = m_a.predict_trial(tr_clean)
    pred_poison = m_a.predict_trial(tr_poison)
    future_poison_diff = float(np.max(np.abs(pred_clean - pred_poison)))

    # 2. OOF-tamper invariance: refit identical model on same train_trials while val_recs are tampered
    # (confirming zero dependence on val fold and deterministic reproducibility)
    m_b = M35Model("M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film", config, canon_neurons, functional_tiers, atlas)
    m_b.fit(train_trials, s_n)
    pred_b = m_b.predict_trial(tr_clean)
    oof_tamper_diff = float(np.max(np.abs(pred_clean - pred_b)))

    del raw_recs, aligned, splits, atlas, m_a, m_b
    gc.collect()

    passed = (
        future_poison_diff == 0.0
        and oof_tamper_diff == 0.0
        and max_2hop_diag == 0.0
        and max_2hop_masked == 0.0
        and all(v == 0.0 for v in pair_checks.values())
    )
    return {
        "model_id": "M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film",
        "passed": bool(passed),
        "future_poison_max_abs_diff": future_poison_diff,
        "oof_tamper_pred_diff": oof_tamper_diff,
        "2hop_diag_max_abs": max_2hop_diag,
        "2hop_bilateral_and_masked_max_abs": max_2hop_masked,
        "explicit_bilateral_pairs_max_abs": float(max(pair_checks.values())) if pair_checks else 0.0,
    }


def _score_model_dual_ref(
    run_id: str,
    cand_by_ds: Mapping[str, Mapping[str, Any]],
    ref_b4_scores: Mapping[str, Mapping[str, Any]],
    ref_01_scores: Mapping[str, Mapping[str, Any]],
    exp_root: Path,
) -> Dict[str, Any]:
    run_dir = exp_root / "runs" / run_id
    pred_files = {ds: str((run_dir / f"predictions_dev_{ds}.npz").relative_to(exp_root)) for ds in DATASETS_ALL3}

    # Macro 1000-resample animal bootstrap CIs vs B4_ridge
    macro_main2_b4 = compute_macro_bootstrap_ci(
        {ds: cand_by_ds[ds] for ds in DATASETS_MAIN2},
        {ds: ref_b4_scores[ds] for ds in DATASETS_MAIN2},
        DATASETS_MAIN2,
        n_boot=1000,
        seed=20261117,
    )
    macro_all3_b4 = compute_macro_bootstrap_ci(
        {ds: cand_by_ds[ds] for ds in DATASETS_ALL3},
        ref_b4_scores,
        DATASETS_ALL3,
        n_boot=1000,
        seed=20261218,
    )

    # Macro 1000-resample animal bootstrap CIs vs M35_01
    macro_main2_01 = compute_macro_bootstrap_ci(
        {ds: cand_by_ds[ds] for ds in DATASETS_MAIN2},
        {ds: ref_01_scores[ds] for ds in DATASETS_MAIN2},
        DATASETS_MAIN2,
        n_boot=1000,
        seed=20261117,
    )
    macro_all3_01 = compute_macro_bootstrap_ci(
        {ds: cand_by_ds[ds] for ds in DATASETS_ALL3},
        ref_01_scores,
        DATASETS_ALL3,
        n_boot=1000,
        seed=20261218,
    )

    source_metrics: Dict[str, Any] = {}
    for ds in DATASETS_ALL3:
        s = cand_by_ds[ds]
        cmp_b4 = compare_against_reference(s, ref_b4_scores[ds], n_boot=1000, seed=20261117)
        cmp_01 = compare_against_reference(s, ref_01_scores[ds], n_boot=1000, seed=20261117)
        source_metrics[ds] = {
            "E_d": round(float(s["E_d"]), 6),
            "I_d": round(float(cmp_b4["I_d"]), 6),
            "I_d_ci95": [round(float(x), 6) for x in cmp_b4["I_d_ci95"]],
            "I_d_vs_M35_01": round(float(cmp_01["I_d"]), 6),
            "I_d_vs_M35_01_ci95": [round(float(x), 6) for x in cmp_01["I_d_ci95"]],
            "post_R2": round(float(s["post_R2"]), 6),
            "post_R2_ci95": [round(float(x), 6) for x in s.get("post_R2_ci95", [0.0, 0.0])],
            "delta_R2": round(float(s["delta_R2"]), 6),
            "delta_R2_ci95": [round(float(x), 6) for x in s.get("delta_R2_ci95", [0.0, 0.0])],
            "rmse_raw": round(float(s["rmse_raw"]), 6),
            "n_worms_improved": int(cmp_b4["n_worms_improved"]),
            "n_worms_improved_vs_M35_01": int(cmp_01["n_worms_improved"]),
            "n_worms_total": int(cmp_b4["n_worms_total"]),
        }

    return {
        "I_macro": round(float(macro_main2_b4["I_macro"]), 6),
        "I_macro_ci95": [round(float(x), 6) for x in macro_main2_b4["I_macro_ci95"]],
        "I_macro_all3_immobilized": round(float(macro_all3_b4["I_macro"]), 6),
        "I_macro_all3_ci95": [round(float(x), 6) for x in macro_all3_b4["I_macro_ci95"]],
        "I_macro_vs_M35_01": round(float(macro_main2_01["I_macro"]), 6),
        "I_macro_vs_M35_01_ci95": [round(float(x), 6) for x in macro_main2_01["I_macro_ci95"]],
        "I_macro_all3_vs_M35_01": round(float(macro_all3_01["I_macro"]), 6),
        "I_macro_all3_vs_M35_01_ci95": [round(float(x), 6) for x in macro_all3_01["I_macro_ci95"]],
        "worst_main_source_I_d": round(min(source_metrics[ds]["I_d"] for ds in DATASETS_MAIN2), 6),
        "source_metrics": source_metrics,
        "prediction_files_dev": pred_files,
    }


def _append_runs_jsonl_rows(exp_root: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    jsonl_path = exp_root / "runs.jsonl"
    existing_ids = set()
    existing_lines: List[str] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_lines.append(line)
                try:
                    existing_ids.add(json.loads(line).get("run_id"))
                except Exception:
                    pass
    added = 0
    with jsonl_path.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["run_id"] not in existing_ids:
                f.write(json.dumps(r) + "\n")
                existing_ids.add(r["run_id"])
                added += 1
    return len(existing_lines) + added


def _append_research_loop_log(exp_root: Path, summary: Mapping[str, Any], diag: Optional[Mapping[str, Any]]) -> None:
    log_path = exp_root / "research_loop_log.md"
    marker = "## Round 5 (`M3.5` Structural Nonlinear-Anchor + 1-Hop/2-Hop Connectome Patch-FiLM)"
    if log_path.exists() and marker in log_path.read_text(encoding="utf-8"):
        return

    lines: List[str] = [
        "",
        "---",
        "",
        marker,
        f"- **Date (UTC)**: `{summary['generated_at_utc']}`",
        "- **Unified Config**: `configs/lambda_global.json` (ZERO `dataset_id` branches, ZERO `vault_sealed/` access)",
        "",
    ]

    if diag is not None:
        snr = diag.get("cross_dataset_physical_snr_ceiling", {})
        lines.append("### 1. `Toyoshima 2024` Residual & Physical SNR Ceiling Diagnosis")
        lines.append("| Dataset | FPS | HF Shot-Noise Var (`0.5*Var(dy/s_n)`) | Lag-1 $r$ | `B4_ridge E_d` | Shot-Noise % of `B4 E_d` | Empirical `post_R2` Ceiling |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for ds in DATASETS_ALL3:
            d = snr.get(ds, {})
            lines.append(
                f"| `{ds}` | {d.get('mean_fps')} | `{d.get('hf_innovation_var_post_norm')}` | "
                f"`{d.get('lag1_autocorrelation')}` | `{d.get('B4_ridge_E_d')}` | "
                f"`{100.0 * float(d.get('shot_noise_fraction_of_B4_E_d') or 0.0):.1f}%` | "
                f"`{d.get('empirical_post_R2_ceiling')}` |"
            )
        lines.append("")

    lines.append("### 2. Round 5 (`M3.5`) Dual-Reference Dev Results (vs `B4_ridge_self_stim` AND vs `M35_01`)")
    lines.append("| Model | `Seyed I_d` vs `B4` | `Toyo I_d` vs `B4` | `Yemini I_d` vs `B4` | `I_macro` Main 2 vs `B4` [95% CI] | `I_macro` All 3 vs `B4` [95% CI] | Net `I_macro` All 3 vs `M35_01` [95% CI] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for spec in M35_MODEL_SPECS:
        mid = spec["model_id"]
        m = summary["models"][mid]
        sm = m["source_metrics"]
        lines.append(
            f"| `{mid}` | `{100*sm['seyedolmohadesin2025']['I_d']:+.2f}%` | "
            f"`{100*sm['toyoshima2024']['I_d']:+.2f}%` | "
            f"`{100*sm['yemini2021']['I_d']:+.2f}%` | "
            f"`{100*m['I_macro']:+.2f}%` `[{100*m['I_macro_ci95'][0]:+.2f}%, {100*m['I_macro_ci95'][1]:+.2f}%]` | "
            f"`{100*m['I_macro_all3_immobilized']:+.2f}%` `[{100*m['I_macro_all3_ci95'][0]:+.2f}%, {100*m['I_macro_all3_ci95'][1]:+.2f}%]` | "
            f"`{100*m['I_macro_all3_vs_M35_01']:+.2f}%` `[{100*m['I_macro_all3_vs_M35_01_ci95'][0]:+.2f}%, {100*m['I_macro_all3_vs_M35_01_ci95'][1]:+.2f}%]` |"
        )
    lines.append("")
    lk = summary["leakage_invariance"]
    lines.append(
        f"### 3. Leakage Invariance (`M35_02`): `passed={lk['passed']}`, "
        f"`future_poison_max_abs_diff={lk['future_poison_max_abs_diff']}`, "
        f"`oof_tamper_pred_diff={lk['oof_tamper_pred_diff']}`, "
        f"`2hop_diag={lk['2hop_diag_max_abs']}`, "
        f"`2hop_bilateral={lk['2hop_bilateral_and_masked_max_abs']}`."
    )
    lines.append("")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_stage_score(
    config: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
) -> Dict[str, Any]:
    t0 = time.time()
    print("\n[M3.5 Score] Scoring all 3 Round 5 models via standalone evaluators/scorer.py...")
    ref_b4_dir = exp_root / "runs" / "r0_04_b4_ridge_self_stim"
    ref_m35_01_dir = exp_root / "runs" / "r5_01_m35_01_zeroinit_nonlinear_self_anchor"

    train_cache_path = exp_root / "results" / "_m35_r5_train_cache.json"
    train_cache = json.loads(train_cache_path.read_text(encoding="utf-8")) if train_cache_path.exists() else {}

    checksums = {
        "protocol_sha256": compute_file_sha256(exp_root / "protocol.json"),
        "lambda_global_sha256": compute_file_sha256(exp_root / "configs" / "lambda_global.json"),
        "split_manifest_sha256": compute_file_sha256(exp_root / "split_manifest.json"),
        "connectome_bundle_sha256": compute_file_sha256(data_root / "connectomes" / "connectome_priors_bundle.npz"),
        "code_bundle_sha256": _code_bundle_sha256(exp_root),
    }

    # Score frozen B4 reference ONCE per dataset via standalone evaluators/scorer.py
    ref_b4_scores: Dict[str, Any] = {}
    for ds in DATASETS_ALL3:
        ref_b4_scores[ds] = score_saved_predictions_file(
            ref_b4_dir / f"predictions_dev_{ds}.npz",
            split="dev",
            data_root=data_root,
            exp_root=exp_root,
            resolve_registry_ref=False,
        )
        gc.collect()

    # Score each of the 3 M3.5 models ONCE per dataset via standalone evaluators/scorer.py
    raw_scores_by_model: Dict[str, Dict[str, Any]] = {}
    for spec in M35_MODEL_SPECS:
        mid = spec["model_id"]
        rid = spec["run_id"]
        raw_scores_by_model[mid] = {}
        for ds in DATASETS_ALL3:
            p_file = exp_root / "runs" / rid / f"predictions_dev_{ds}.npz"
            raw_scores_by_model[mid][ds] = score_saved_predictions_file(
                p_file,
                split="dev",
                data_root=data_root,
                exp_root=exp_root,
                resolve_registry_ref=False,
            )
            gc.collect()

    ref_01_scores = raw_scores_by_model["M35_01_zeroinit_nonlinear_self_anchor"]

    models_out: Dict[str, Any] = {}
    jsonl_rows: List[Dict[str, Any]] = []
    now_utc = _utc_now()

    for spec in M35_MODEL_SPECS:
        mid = spec["model_id"]
        rid = spec["run_id"]
        sc = _score_model_dual_ref(rid, raw_scores_by_model[mid], ref_b4_scores, ref_01_scores, exp_root=exp_root)
        models_out[mid] = sc

        sm = sc["source_metrics"]
        print(
            f"  {mid}:\n"
            f"    vs B4    -> Seyed {100*sm['seyedolmohadesin2025']['I_d']:+.2f}% ({sm['seyedolmohadesin2025']['E_d']:.5f}) | "
            f"Toyo {100*sm['toyoshima2024']['I_d']:+.2f}% ({sm['toyoshima2024']['E_d']:.5f}) | "
            f"Yemini {100*sm['yemini2021']['I_d']:+.2f}% ({sm['yemini2021']['E_d']:.5f}) | "
            f"Main2 {100*sc['I_macro']:+.2f}% | All3 {100*sc['I_macro_all3_immobilized']:+.2f}%\n"
            f"    vs M35_1 -> Seyed {100*sm['seyedolmohadesin2025']['I_d_vs_M35_01']:+.2f}% | "
            f"Toyo {100*sm['toyoshima2024']['I_d_vs_M35_01']:+.2f}% | "
            f"Yemini {100*sm['yemini2021']['I_d_vs_M35_01']:+.2f}% | "
            f"Main2 {100*sc['I_macro_vs_M35_01']:+.2f}% | All3 {100*sc['I_macro_all3_vs_M35_01']:+.2f}%"
        )

        row = {
            "run_id": rid,
            "parent_id": spec["parent_id"],
            "round": 5,
            "category": "M35_round5_nonlinear_anchor",
            "model_id": mid,
            "family_id": "M35_NONLINEAR_ANCHOR",
            "is_self_only": bool(spec["is_self_only"]),
            "hypothesis": spec["hypothesis"],
            "single_factor_changed": spec["single_factor_changed"],
            "registered_before_training_utc": train_cache.get("updated_at_utc", now_utc),
            "completed_at_utc": now_utc,
            "config_id": str(config.get("config_id", "lambda_global_v1")),
            "seed": int(config.get("hyperparameters", {}).get("seed", 20261117)),
            "checksums": checksums,
            "reference_baseline_id": "B4_ridge_self_stim",
            "I_macro": sc["I_macro"],
            "I_macro_ci95": sc["I_macro_ci95"],
            "I_macro_all3_immobilized": sc["I_macro_all3_immobilized"],
            "I_macro_all3_ci95": sc["I_macro_all3_ci95"],
            "I_macro_vs_M35_01": sc["I_macro_vs_M35_01"],
            "I_macro_vs_M35_01_ci95": sc["I_macro_vs_M35_01_ci95"],
            "I_macro_all3_vs_M35_01": sc["I_macro_all3_vs_M35_01"],
            "I_macro_all3_vs_M35_01_ci95": sc["I_macro_all3_vs_M35_01_ci95"],
            "worst_main_source_I_d": sc["worst_main_source_I_d"],
            "source_metrics": sm,
            "prediction_files_dev": sc["prediction_files_dev"],
            "wall_clock_sec": float(train_cache.get("total_wall_clock_sec", 0.0)),
            "peak_rss_mb": _peak_rss_mb(),
            "exit_code": 0,
            "status": "completed",
            "decision": "M35_ROUND5_EVALUATED",
        }
        jsonl_rows.append(row)

    print("[M3.5 Score] Running strict leakage invariance verification on M35_02...")
    leakage = verify_m35_02_leakage_invariance(config, data_root=data_root, exp_root=exp_root)
    print(f"  Leakage check: {json.dumps(leakage)}")
    if not leakage["passed"]:
        raise RuntimeError(f"M35_02 leakage invariance FAILED: {leakage}")

    total_rows = _append_runs_jsonl_rows(exp_root, jsonl_rows)
    print(f"[M3.5 Score] runs.jsonl now has {total_rows} total rows.")

    summary = {
        "generated_at_utc": now_utc,
        "config_id": str(config.get("config_id", "lambda_global_v1")),
        "checksums": checksums,
        "models": models_out,
        "train_fold_diagnostics": train_cache.get("datasets", {}),
        "leakage_invariance": leakage,
        "runs_jsonl_total_rows": total_rows,
        "score_wall_clock_sec": round(time.time() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }

    sum_path = exp_root / "results" / "m35_round5_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diag_path = exp_root / "results" / "m35_toyoshima_diagnosis.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else None
    _append_research_loop_log(exp_root, summary, diag)
    print(f"[M3.5 Score] Saved {sum_path} and updated research_loop_log.md (peak RSS {_peak_rss_mb()} MB)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.5 Round 5: Toyoshima Diagnosis + Zero-Init Nonlinear Anchor + 1-Hop/2-Hop Patch-FiLM")
    parser.add_argument("--stage", choices=["diagnose", "train", "score", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(EXP_ROOT / "configs" / "lambda_global.json"))
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--exp-root", type=str, default=str(EXP_ROOT_DEFAULT))
    args = parser.parse_args()

    # Keep PyTorch CPU threading clean & memory-frugal on 2 vCPUs
    torch.set_num_threads(2)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    ds_arg = [args.dataset] if args.dataset else None

    if args.stage in ("diagnose", "all"):
        run_stage_diagnose(data_root=data_root, exp_root=exp_root)
    if args.stage in ("train", "all"):
        run_stage_train(config=config, data_root=data_root, exp_root=exp_root, resume=args.resume, datasets=ds_arg)
    if args.stage in ("score", "all"):
        run_stage_score(config=config, data_root=data_root, exp_root=exp_root)


if __name__ == "__main__":
    main()
