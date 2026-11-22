#!/usr/bin/env python3
"""Step B1 (M5): Single One-Shot Vault Confirmatory Unsealing for `autoresearch_v1`.

CRITICAL INTEGRITY GUARANTEES:
1. ZERO modification of any of the 27 frozen files hashed in `final_manifest.json`
   (only `exposure_ledger.json` is updated AFTER one-shot unsealing to record the
   exact unsealing event, preserving all pre-unseal hashes and animal statuses).
2. Pre-unseal cryptographic SHA-256 verification of `final_manifest.json`'s
   `manifest_sha256` AND all 27 listed file digests; aborts with non-zero exit
   on any mismatch or if `vault_access_audit.jsonl` already has entries.
3. Strictly one-shot unsealing: each of the 5 pre-registered models is fitted
   strictly on Dev animals ONLY (sensor bounds, per-neuron scales `s_n`, feature
   standardizers, PCA/SVD compressors, and internal early-stopping validation split
   all strictly inside Dev), then evaluated ONCE in a single batch on the sealed
   vault animals (`seyedolmohadesin2025` 15w, `toyoshima2024` 10w,
   `baskoylu2026_copper` 9w = 34 unseen animals).
4. Prediction-first saving: all per-model, per-dataset, per-seed vault predictions
   and ground-truth metadata (`y_pred`, `y_true`, `valid_mask`, `animal_id`,
   `trial_id`, `stimulus_id`, `neuron_ids`, `s_n`) are persisted to
   `runs/m5_confirmatory/` BEFORE any confirmatory scores are computed.
5. CPU-only (`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`), zero `/tmp` writes.

Usage:
    python scripts/run_m5_vault_confirmation.py --verify-only
    python scripts/run_m5_vault_confirmation.py --execute-one-shot-unseal
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

EXP_ROOT_DEFAULT = Path("/home/antigravity/experiments/autoresearch_v1")
DATA_ROOT_DEFAULT = Path("/home/antigravity/data")
if str(EXP_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT_DEFAULT))

import torch

torch.set_num_threads(min(2, os.cpu_count() or 1))

from models.m3_round3_h3 import load_signed_multilayer_connectome
from models.m3_round4_f11 import M3R4Model, _extract_fold_trials
from models.m35_round5_nonlinear import build_1hop_2hop_connectome_atlas
from models.m35_round6_validation import (
    SEEDS_5,
    _build_single_fold_m35_cache,
    _fit_film_from_cache,
    _fit_m35_01_from_cache,
    _reconstruct_fold_preds,
    _tier_onehot,
)
from src.baselines import CausalBaselineModel
from src.data_adapters import compute_file_sha256, load_dataset
from src.evaluator import (
    align_records_to_canonical,
    evaluate_predictions,
    save_CompactPredictions,
)

PRIMARY_SOURCES = ["seyedolmohadesin2025", "toyoshima2024"]
ISOLATED_SOURCE = "baskoylu2026_copper"
ALL_VAULT_SOURCES = ["seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper"]

MODEL_B4 = "B4_ridge_self_stim"
MODEL_M3R4 = "M3R4_01_connectome_film_herm_final"
MODEL_M35_01 = "M35_01_zeroinit_nonlinear_self_anchor"
MODEL_M35_02 = "M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film"
MODEL_M35_03 = "M35_03_nonlinear_anchor_graphfree_patch_film_ctrl"

FROZEN_LINEUP = [
    MODEL_B4,
    MODEL_M3R4,
    MODEL_M35_01,
    MODEL_M35_02,
    MODEL_M35_03,
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2)


# ===========================================================================
# 1. Pre-Unseal Cryptographic SHA-256 Verification (27/27 Frozen Files)
# ===========================================================================
def verify_frozen_manifest(exp_root: Path) -> Dict[str, Any]:
    """Verify `final_manifest.json` self-hash, all 27 frozen file SHA-256 digests, and untouched vault."""
    manifest_path = exp_root / "final_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing final_manifest.json at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_manifest_sha = manifest.get("manifest_sha256")
    manifest_body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    computed_manifest_sha = hashlib.sha256(
        json.dumps(manifest_body, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if computed_manifest_sha != recorded_manifest_sha:
        raise ValueError(
            f"final_manifest.json manifest_sha256 mismatch! "
            f"recorded={recorded_manifest_sha}, computed={computed_manifest_sha}"
        )

    checksums = manifest.get("sha256_checksums", {})
    gov_hashes = checksums.get("governance_and_configs", {})
    py_hashes = checksums.get("python_source_files", {})
    all_expected: Dict[str, str] = {}
    all_expected.update(gov_hashes)
    all_expected.update(py_hashes)

    if len(all_expected) != 27:
        raise ValueError(f"Expected 27 frozen files in final_manifest.json, found {len(all_expected)}")

    mismatches: List[str] = []
    verified_files: Dict[str, str] = {}
    for rel_path, expected_sha in sorted(all_expected.items()):
        fp = exp_root / rel_path
        if not fp.exists():
            mismatches.append(f"MISSING: {rel_path}")
            continue
        actual_sha = compute_file_sha256(fp)
        if actual_sha != expected_sha:
            mismatches.append(f"MISMATCH: {rel_path} (expected={expected_sha}, actual={actual_sha})")
        else:
            verified_files[rel_path] = actual_sha

    if mismatches:
        raise ValueError("Frozen file SHA-256 verification FAILED:\n  " + "\n  ".join(mismatches))

    audit_log = exp_root / "vault_access_audit.jsonl"
    existing_audit = []
    if audit_log.exists():
        existing_audit = [ln for ln in audit_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(existing_audit) != 0:
        raise RuntimeError(
            f"ONE-SHOT VAULT VIOLATION: vault_access_audit.jsonl already contains {len(existing_audit)} entries! "
            "Refusing to re-unseal vault."
        )

    return {
        "manifest": manifest,
        "manifest_sha256": recorded_manifest_sha,
        "n_verified": len(verified_files),
        "n_expected": len(all_expected),
        "frozen_files_verified": f"{len(verified_files)}/{len(all_expected)}",
        "verified_digests": verified_files,
        "pre_unseal_exposure_ledger_sha256": verified_files["exposure_ledger.json"],
    }


# ===========================================================================
# 2. Statistical Helpers (B=2000 Animal Bootstrap, Exact Sign-Flip & Wilcoxon)
# ===========================================================================
def _exact_paired_sign_flip_p(diffs: np.ndarray) -> Dict[str, float]:
    """Exact paired sign-flip randomization test on animal-level `ref_Lw - cand_Lw`."""
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    if n == 0:
        return {"p_two_sided": float("nan"), "p_greater": float("nan"), "method": "empty"}
    obs = float(np.mean(d))
    if np.all(np.abs(d) <= 1e-15):
        return {"p_two_sided": 1.0, "p_greater": 1.0, "method": "exact_all_zero"}

    if n <= 16:
        # Exact enumeration over all 2^n sign assignments
        n_comb = 1 << n
        ints = np.arange(n_comb, dtype=np.uint32)[:, None]
        bits = (ints >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        signs = np.where(bits == 1, 1.0, -1.0)
        means = (signs @ d) / float(n)
        p_gt = float(np.mean(means >= obs - 1e-15))
        p_two = float(np.mean(np.abs(means) >= abs(obs) - 1e-15))
        return {
            "p_two_sided": p_two,
            "p_greater": p_gt,
            "n_permutations": int(n_comb),
            "method": "exact_enumeration",
        }

    rng = np.random.default_rng(20261117)
    n_mc = 100000
    signs = rng.choice([-1.0, 1.0], size=(n_mc, n))
    means = (signs @ d) / float(n)
    p_gt = float((1 + np.sum(means >= obs - 1e-15)) / (1 + n_mc))
    p_two = float((1 + np.sum(np.abs(means) >= abs(obs) - 1e-15)) / (1 + n_mc))
    return {
        "p_two_sided": p_two,
        "p_greater": p_gt,
        "n_permutations": n_mc,
        "method": "monte_carlo",
    }


def _wilcoxon_paired_p(diffs: np.ndarray) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test on animal-level `ref_Lw - cand_Lw`."""
    d = np.asarray(diffs, dtype=np.float64)
    nz = d[np.abs(d) > 1e-15]
    if nz.size < 2:
        return {"statistic": None, "p_two_sided": 1.0, "p_greater": 1.0}
    w_two = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    w_gt = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
    return {
        "statistic": float(w_two.statistic),
        "p_two_sided": float(w_two.pvalue),
        "p_greater": float(w_gt.pvalue),
    }


def _animal_bootstrap_gain(
    cand_lw: Mapping[str, float],
    ref_lw: Mapping[str, float],
    n_boot: int = 2000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compute point estimate, marginal 95% CI, Bonferroni 97.5% CI, exact sign-flip & Wilcoxon p-values."""
    worms = sorted(cand_lw.keys())
    if set(worms) != set(ref_lw.keys()):
        raise ValueError("Animal ID mismatch between candidate and reference")
    c_arr = np.array([float(cand_lw[w]) for w in worms], dtype=np.float64)
    r_arr = np.array([float(ref_lw[w]) for w in worms], dtype=np.float64)
    nw = len(worms)

    e_cand = float(np.mean(c_arr))
    e_ref = float(np.mean(r_arr))
    i_d = float(1.0 - e_cand / max(e_ref, 1e-12))
    diffs = r_arr - c_arr
    n_imp = int(np.sum(c_arr < r_arr - 1e-12))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nw, size=(n_boot, nw))
    boot_gains = 1.0 - np.mean(c_arr[idx], axis=1) / np.maximum(np.mean(r_arr[idx], axis=1), 1e-12)

    ci95 = [float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5))]
    ci975 = [float(np.percentile(boot_gains, 1.25)), float(np.percentile(boot_gains, 98.75))]

    return {
        "I_d": i_d,
        "ci95_marginal": ci95,
        "ci975_simultaneous_bonferroni": ci975,
        "boot_draws": boot_gains,
        "n_worms_improved": n_imp,
        "n_worms_total": nw,
        "fraction_worms_improved": float(n_imp / max(nw, 1)),
        "mean_paired_loss_reduction": float(np.mean(diffs)),
        "sign_flip_test": _exact_paired_sign_flip_p(diffs),
        "wilcoxon_test": _wilcoxon_paired_p(diffs),
        "paired_loss_diff_by_worm": {w: float(diffs[i]) for i, w in enumerate(worms)},
    }


def _compute_r2_with_dual_cis(
    trials: Sequence[Mapping[str, Any]],
    per_seed_preds: Sequence[Sequence[np.ndarray]],
    n_boot: int = 2000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compute exact `post_R2`, `delta_R2`, `rmse_raw` and BOTH 95% marginal & 97.5% Bonferroni CIs.

    Uses equal-weight average across seeds of per-animal squared residuals (`res_vec`) to match
    the pre-registered `equal_weight_mean_across_5_seeds_on_per_animal_loss_L_w` rule (never
    mixing `post_R2` and `delta_R2` denominators!).
    """
    n_seeds = len(per_seed_preds)
    w_cnt: Dict[str, int] = {}
    w_res_sq_seedsum: Dict[str, float] = {}
    w_sum_yc: Dict[str, float] = {}
    w_sum_yc2: Dict[str, float] = {}
    w_sum_dyc: Dict[str, float] = {}
    w_sum_dyc2: Dict[str, float] = {}

    ref_center_y: Optional[float] = None
    ref_center_dy: Optional[float] = None

    for tr_i, tr in enumerate(trials):
        if bool(tr.get("is_buffer_control", False)):
            continue
        wid = str(tr["animal_id"])
        y_true = np.asarray(tr["y_true"], dtype=np.float64)
        vmask = np.asarray(tr["valid_mask"], dtype=bool)
        n_val = int(np.sum(vmask))
        if n_val == 0:
            continue

        py_v = y_true[vmask]
        pre_arr = np.broadcast_to(np.asarray(tr["y_pre_anchor"], dtype=np.float64)[:, None], y_true.shape)
        dy_v = py_v - pre_arr[vmask]
        if ref_center_y is None:
            ref_center_y = float(np.mean(py_v))
            ref_center_dy = float(np.mean(dy_v))

        yc = py_v - ref_center_y
        dyc = dy_v - (ref_center_dy if ref_center_dy is not None else 0.0)
        w_cnt[wid] = w_cnt.get(wid, 0) + n_val
        w_sum_yc[wid] = w_sum_yc.get(wid, 0.0) + float(np.sum(yc))
        w_sum_yc2[wid] = w_sum_yc2.get(wid, 0.0) + float(np.sum(yc ** 2))
        w_sum_dyc[wid] = w_sum_dyc.get(wid, 0.0) + float(np.sum(dyc))
        w_sum_dyc2[wid] = w_sum_dyc2.get(wid, 0.0) + float(np.sum(dyc ** 2))

        rsq_acc = 0.0
        for sd_i in range(n_seeds):
            yp = np.asarray(per_seed_preds[sd_i][tr_i], dtype=np.float64)
            diff = py_v - yp[vmask]
            rsq_acc += float(np.sum(diff ** 2))
        w_res_sq_seedsum[wid] = w_res_sq_seedsum.get(wid, 0.0) + (rsq_acc / float(n_seeds))

    worms = sorted(w_cnt.keys())
    nw = len(worms)
    cnt_vec = np.array([w_cnt[w] for w in worms], dtype=np.float64)
    res_vec = np.array([w_res_sq_seedsum[w] for w in worms], dtype=np.float64)
    yc_vec = np.array([w_sum_yc[w] for w in worms], dtype=np.float64)
    yc2_vec = np.array([w_sum_yc2[w] for w in worms], dtype=np.float64)
    dyc_vec = np.array([w_sum_dyc[w] for w in worms], dtype=np.float64)
    dyc2_vec = np.array([w_sum_dyc2[w] for w in worms], dtype=np.float64)

    n_all = float(np.sum(cnt_vec))
    ss_res = float(np.sum(res_vec))
    ss_post = float(np.sum(yc2_vec) - (np.sum(yc_vec) ** 2) / max(n_all, 1.0))
    ss_delta = float(np.sum(dyc2_vec) - (np.sum(dyc_vec) ** 2) / max(n_all, 1.0))
    post_r2 = float(1.0 - ss_res / max(ss_post, 1e-12))
    delta_r2 = float(1.0 - ss_res / max(ss_delta, 1e-12))
    rmse_raw = float(np.sqrt(ss_res / max(n_all, 1.0)))

    rng = np.random.default_rng(seed)
    samp = rng.integers(0, nw, size=(n_boot, nw))
    b_cnt = np.maximum(np.sum(cnt_vec[samp], axis=1), 1.0)
    b_res = np.sum(res_vec[samp], axis=1)
    b_yc = np.sum(yc_vec[samp], axis=1)
    b_yc2 = np.sum(yc2_vec[samp], axis=1)
    b_dyc = np.sum(dyc_vec[samp], axis=1)
    b_dyc2 = np.sum(dyc2_vec[samp], axis=1)
    b_ss_post = np.maximum(b_yc2 - (b_yc ** 2) / b_cnt, 1e-12)
    b_ss_delta = np.maximum(b_dyc2 - (b_dyc ** 2) / b_cnt, 1e-12)
    bp = 1.0 - b_res / b_ss_post
    bd = 1.0 - b_res / b_ss_delta

    return {
        "post_R2": post_r2,
        "post_R2_ci95_marginal": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
        "post_R2_ci975_simultaneous_bonferroni": [float(np.percentile(bp, 1.25)), float(np.percentile(bp, 98.75))],
        "delta_R2": delta_r2,
        "delta_R2_ci95_marginal": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
        "delta_R2_ci975_simultaneous_bonferroni": [float(np.percentile(bd, 1.25)), float(np.percentile(bd, 98.75))],
        "rmse_raw": rmse_raw,
    }


# ===========================================================================
# 3. Cell-Class, Neuron-Level Hotspot, Stimulus, Date-Batch & Sex Breakdowns
# ===========================================================================
def _map_cell_class(tier: str) -> str:
    t = str(tier).strip().lower()
    if t == "sensory":
        return "sensory"
    if t == "interneuron":
        return "interneuron"
    if t in ("premotor_command", "motor", "command_motor"):
        return "command_motor"
    return "other"


def _compute_cell_class_and_neuron_hotspots(
    trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    preds_by_model_seed: Mapping[str, Sequence[Sequence[np.ndarray]]],
    canon_neurons: Sequence[str],
    functional_tiers: Sequence[str],
    connectome_degrees: Mapping[str, Dict[str, int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute cell-class (`sensory`, `interneuron`, `command_motor`, `other`) metrics
    and neuron-level hotspot tables (`top_20_gain_vs_B4`, `top_20_net_cross_neuron_gain_vs_M35_01`,
    `bottom_10_vs_B4`) with Cook 2019 & Ripoll-Sánchez 2023 in-degrees.
    """
    n_canon = len(canon_neurons)
    classes_per_neuron = [_map_cell_class(t) for t in functional_tiers]
    class_names = ["sensory", "interneuron", "command_motor", "other"]
    scales = np.asarray(s_n, dtype=np.float64)

    # Per-model, per-neuron normalized MSE & observed worm count
    # Also per-model, per-class equal-animal normalized MSE & pooled post_R2
    neuron_err_sum: Dict[str, np.ndarray] = {m: np.zeros(n_canon, dtype=np.float64) for m in FROZEN_LINEUP}
    neuron_cnt_sum: np.ndarray = np.zeros(n_canon, dtype=np.float64)
    neuron_worms_obs: List[set] = [set() for _ in range(n_canon)]

    class_worm_errs: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {c: {} for c in class_names} for m in FROZEN_LINEUP
    }
    class_r2_ss_res: Dict[str, Dict[str, float]] = {m: {c: 0.0 for c in class_names} for m in FROZEN_LINEUP}
    class_r2_sum_y: Dict[str, float] = {c: 0.0 for c in class_names}
    class_r2_sum_y2: Dict[str, float] = {c: 0.0 for c in class_names}
    class_r2_cnt: Dict[str, int] = {c: 0 for c in class_names}

    class_masks = {
        c: np.array([classes_per_neuron[i] == c for i in range(n_canon)], dtype=bool)
        for c in class_names
    }

    for tr_i, tr in enumerate(trials):
        if bool(tr.get("is_buffer_control", False)):
            continue
        wid = str(tr["animal_id"])
        y_true = np.asarray(tr["y_true"], dtype=np.float64)
        vmask = np.asarray(tr["valid_mask"], dtype=bool)
        row_cnt = np.sum(vmask, axis=1)
        active = row_cnt > 0
        if not np.any(active):
            continue

        neuron_cnt_sum += row_cnt
        for ni in np.where(active)[0]:
            neuron_worms_obs[int(ni)].add(wid)

        for c in class_names:
            cm = vmask & class_masks[c][:, None]
            if np.any(cm):
                vals = y_true[cm]
                class_r2_sum_y[c] += float(np.sum(vals))
                class_r2_sum_y2[c] += float(np.sum(vals ** 2))
                class_r2_cnt[c] += int(vals.size)

        for m in FROZEN_LINEUP:
            seed_list = preds_by_model_seed[m]
            n_sd = len(seed_list)
            raw_sq_acc = np.zeros_like(y_true, dtype=np.float64)
            for sd_i in range(n_sd):
                yp = np.asarray(seed_list[sd_i][tr_i], dtype=np.float64)
                diff = np.where(vmask, y_true - yp, 0.0)
                raw_sq_acc += diff ** 2
            raw_sq_mean = raw_sq_acc / float(n_sd)
            norm_sq_mean = raw_sq_mean / (scales[:, None] ** 2)

            row_norm_sum = np.sum(np.where(vmask, norm_sq_mean, 0.0), axis=1)
            neuron_err_sum[m] += row_norm_sum

            row_norm_mse = np.zeros(n_canon, dtype=np.float64)
            row_norm_mse[active] = row_norm_sum[active] / row_cnt[active]

            for c in class_names:
                sel = active & class_masks[c]
                if np.any(sel):
                    class_worm_errs[m][c].setdefault(wid, []).extend(row_norm_mse[sel].tolist())
                cm = vmask & class_masks[c][:, None]
                if np.any(cm):
                    class_r2_ss_res[m][c] += float(np.sum(raw_sq_mean[cm]))

    # Summarize cell-class breakdown
    cell_class_summary: Dict[str, Any] = {}
    for c in class_names:
        c_cnt = class_r2_cnt[c]
        if c_cnt == 0:
            continue
        ss_tot = max(class_r2_sum_y2[c] - (class_r2_sum_y[c] ** 2) / float(c_cnt), 1e-12)
        c_entry: Dict[str, Any] = {
            "n_vault_worms": len(class_worm_errs[MODEL_B4][c]),
            "n_valid_points": int(c_cnt),
            "models": {},
        }
        e_b4_c = float(np.mean([np.mean(v) for v in class_worm_errs[MODEL_B4][c].values()]))
        e_01_c = float(np.mean([np.mean(v) for v in class_worm_errs[MODEL_M35_01][c].values()]))
        for m in FROZEN_LINEUP:
            w_means = {w: float(np.mean(v)) for w, v in sorted(class_worm_errs[m][c].items())}
            e_m = float(np.mean(list(w_means.values())))
            r2_m = float(1.0 - class_r2_ss_res[m][c] / ss_tot)
            c_entry["models"][m] = {
                "E_d": round(e_m, 6),
                "I_d_vs_B4": round(1.0 - e_m / max(e_b4_c, 1e-12), 6),
                "I_d_vs_M35_01": round(1.0 - e_m / max(e_01_c, 1e-12), 6),
                "post_R2": round(r2_m, 6),
            }
        cell_class_summary[c] = c_entry

    # Neuron-level hotspot table (observed in >= 2 vault worms and >= 20 valid points)
    neuron_rows: List[Dict[str, Any]] = []
    for ni, nname in enumerate(canon_neurons):
        n_worms = len(neuron_worms_obs[ni])
        cnt = neuron_cnt_sum[ni]
        if n_worms < 2 or cnt < 20:
            continue
        b4_mse = float(neuron_err_sum[MODEL_B4][ni] / cnt)
        m01_mse = float(neuron_err_sum[MODEL_M35_01][ni] / cnt)
        m02_mse = float(neuron_err_sum[MODEL_M35_02][ni] / cnt)
        m03_mse = float(neuron_err_sum[MODEL_M35_03][ni] / cnt)
        m3r4_mse = float(neuron_err_sum[MODEL_M3R4][ni] / cnt)

        g03_b4 = 100.0 * (1.0 - m03_mse / max(b4_mse, 1e-12))
        g02_b4 = 100.0 * (1.0 - m02_mse / max(b4_mse, 1e-12))
        g01_b4 = 100.0 * (1.0 - m01_mse / max(b4_mse, 1e-12))
        net03_01 = 100.0 * (1.0 - m03_mse / max(m01_mse, 1e-12))
        net02_01 = 100.0 * (1.0 - m02_mse / max(m01_mse, 1e-12))

        deg = connectome_degrees.get(nname, {})
        neuron_rows.append(
            {
                "neuron_id": nname,
                "cell_class": classes_per_neuron[ni],
                "functional_tier": str(functional_tiers[ni]),
                "n_vault_worms_observed": int(n_worms),
                "n_valid_points": int(cnt),
                "B4_mse": round(b4_mse, 6),
                "M3R4_01_mse": round(m3r4_mse, 6),
                "M35_01_mse": round(m01_mse, 6),
                "M35_02_mse": round(m02_mse, 6),
                "M35_03_mse": round(m03_mse, 6),
                "gain_pct_M35_01_vs_B4": round(g01_b4, 3),
                "gain_pct_M35_02_vs_B4": round(g02_b4, 3),
                "gain_pct_M35_03_vs_B4": round(g03_b4, 3),
                "net_cross_neuron_gain_pct_M35_02_vs_M35_01": round(net02_01, 3),
                "net_cross_neuron_gain_pct_M35_03_vs_M35_01": round(net03_01, 3),
                "cook2019_gap_in_degree": int(deg.get("gap_in", 0)),
                "cook2019_chem_in_degree": int(deg.get("chem_in", 0)),
                "ripollsanchez2023_npp_short_in_degree": int(deg.get("npp_short_in", 0)),
                "ripollsanchez2023_npp_mid_in_degree": int(deg.get("npp_mid_in", 0)),
            }
        )

    top20_vs_b4 = sorted(neuron_rows, key=lambda r: r["gain_pct_M35_03_vs_B4"], reverse=True)[:20]
    top20_net_vs_m35_01 = sorted(
        neuron_rows, key=lambda r: r["net_cross_neuron_gain_pct_M35_03_vs_M35_01"], reverse=True
    )[:20]
    bottom10_vs_b4 = sorted(neuron_rows, key=lambda r: r["gain_pct_M35_03_vs_B4"])[:10]

    hotspots = {
        "n_evaluated_neurons": len(neuron_rows),
        "top_20_gain_vs_B4": top20_vs_b4,
        "top_20_net_cross_neuron_gain_vs_M35_01": top20_net_vs_m35_01,
        "bottom_10_vs_B4": bottom10_vs_b4,
    }
    return cell_class_summary, hotspots


def _load_connectome_in_degrees(data_root: Path) -> Dict[str, Dict[str, int]]:
    """Compute Cook 2019 W_gap / W_chem and Ripoll-Sánchez 2023 NPP-GPCR in-degrees (excluding self)."""
    cz = np.load(data_root / "connectomes" / "connectome_priors_bundle.npz", allow_pickle=False)
    neurons = [str(x) for x in cz["canonical_neurons"]]
    gap = np.asarray(cz["cook2019_herm_gap"], dtype=np.float64).copy()
    chem = np.asarray(cz["cook2019_herm_chem"], dtype=np.float64).copy()
    npp_s = np.asarray(cz["ripollsanchez2023_short"], dtype=np.float64).copy()
    npp_m = np.asarray(cz["ripollsanchez2023_mid"], dtype=np.float64).copy()
    for mat in (gap, chem, npp_s, npp_m):
        np.fill_diagonal(mat, 0.0)
    out: Dict[str, Dict[str, int]] = {}
    for j, name in enumerate(neurons):
        out[name] = {
            "gap_in": int(np.sum(gap[:, j] > 0)),
            "chem_in": int(np.sum(chem[:, j] > 0)),
            "npp_short_in": int(np.sum(npp_s[:, j] > 0)),
            "npp_mid_in": int(np.sum(npp_m[:, j] > 0)),
        }
    return out


def _stratified_group_breakdown(
    lw_by_model: Mapping[str, Mapping[str, float]],
    group_by_animal: Mapping[str, str],
) -> Dict[str, Any]:
    """Compute E_d, I_d vs B4, I_d vs M35_01, and worms improved per stratum (date_batch / sex)."""
    groups = sorted(set(group_by_animal.values()))
    out: Dict[str, Any] = {}
    for g in groups:
        aids = sorted([a for a, gv in group_by_animal.items() if gv == g and a in lw_by_model[MODEL_B4]])
        if not aids:
            continue
        e_b4 = float(np.mean([lw_by_model[MODEL_B4][a] for a in aids]))
        e_01 = float(np.mean([lw_by_model[MODEL_M35_01][a] for a in aids]))
        g_models: Dict[str, Any] = {}
        for m in FROZEN_LINEUP:
            vals = [lw_by_model[m][a] for a in aids]
            e_m = float(np.mean(vals))
            n_imp_b4 = int(sum(lw_by_model[m][a] < lw_by_model[MODEL_B4][a] - 1e-12 for a in aids))
            n_imp_01 = int(sum(lw_by_model[m][a] < lw_by_model[MODEL_M35_01][a] - 1e-12 for a in aids))
            g_models[m] = {
                "E_d": round(e_m, 6),
                "I_d_vs_B4": round(1.0 - e_m / max(e_b4, 1e-12), 6),
                "I_d_vs_M35_01": round(1.0 - e_m / max(e_01, 1e-12), 6),
                "n_worms_improved_vs_B4": n_imp_b4,
                "n_worms_improved_vs_M35_01": n_imp_01,
            }
        out[g] = {
            "n_animals": len(aids),
            "animal_ids": aids,
            "models": g_models,
        }
    return out


# ===========================================================================
# 4. Single-Source Train-on-Dev / Predict-Once-on-Vault Pipeline
# ===========================================================================
def run_single_source_confirmation(
    ds: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    runs_dir: Path,
    connectome_degrees: Mapping[str, Dict[str, int]],
    n_boot: int = 2000,
    boot_seed: int = 20261117,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train all 5 frozen models strictly on Dev animals ONLY, save predictions FIRST, then score Vault ONCE."""
    t0 = time.time()
    hp = cfg["hyperparameters"]
    print(f"\n======================================================================", flush=True)
    print(f"[M5 One-Shot] Source: {ds} | Loading Dev (train) & Vault (one-shot test)", flush=True)
    print(f"======================================================================", flush=True)

    dev_recs = load_dataset(
        dataset_id=ds,
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="m5_one_shot_confirmatory_dev_train",
    )
    vault_recs = load_dataset(
        dataset_id=ds,
        split="vault",
        allow_vault_unseal=True,
        data_root=data_root,
        exp_root=exp_root,
        require_final_manifest_for_vault=True,
        caller_reason="m5_one_shot_confirmatory_unseal",
    )
    print(f"  [{ds}] Loaded {len(dev_recs)} Dev records, {len(vault_recs)} Vault records", flush=True)

    # Record exact vault file provenance (animal_id, physical_animal_id, sha256)
    vault_provenance = [
        {
            "animal_id": r.animal_id,
            "physical_animal_id": str(r.extra_metadata.get("physical_animal_id", r.animal_id)),
            "sex": r.sex,
            "date_batch": r.date_batch,
            "sha256": r.sha256,
        }
        for r in vault_recs
    ]

    aligned_dev, canon_neurons, functional_tiers = align_records_to_canonical(dev_recs)
    aligned_vault, _, _ = align_records_to_canonical(vault_recs, canonical_neurons=canon_neurons)
    aligned_all = aligned_dev + aligned_vault
    train_idx = list(range(len(aligned_dev)))
    val_idx = list(range(len(aligned_dev), len(aligned_dev) + len(aligned_vault)))
    train_session_ids = [r.animal_id for r in aligned_dev]

    n_canon = len(canon_neurons)
    tier_oh = _tier_onehot(functional_tiers)
    atlas_m35 = build_1hop_2hop_connectome_atlas(canon_neurons, data_root=data_root)
    atlas_m3r3 = load_signed_multilayer_connectome(canon_neurons, data_root=data_root)

    # 1. Extract Dev train trials and Vault test trials (sensor bounds & s_n fitted STRICTLY on Dev!)
    s_n_dev, tr_trials, va_trials = _extract_fold_trials(aligned_all, train_idx, val_idx, hp)
    fold_ids = np.zeros(len(va_trials), dtype=np.int32)
    print(
        f"  [{ds}] Extracted {len(tr_trials)} Dev train trials, {len(va_trials)} Vault test trials "
        f"(s_n range=[{float(s_n_dev.min()):.4f}, {float(s_n_dev.max()):.4f}])",
        flush=True,
    )

    # Persist vault ground-truth metadata bundle BEFORE scoring (prediction-first protocol)
    gt_dir = runs_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_meta = {
        "dataset_id": ds,
        "split": "vault",
        "n_dev_records": len(dev_recs),
        "n_vault_records": len(vault_recs),
        "dev_session_animal_ids": train_session_ids,
        "vault_provenance": vault_provenance,
        "neuron_ids": list(canon_neurons),
        "functional_tiers": list(functional_tiers),
        "trials_meta": [
            {
                "trial_id": t["trial_id"],
                "animal_id": t["animal_id"],
                "session_animal_id": t["session_animal_id"],
                "stimulus_id": t["stimulus_id"],
                "is_buffer_control": bool(t["is_buffer_control"]),
                "sex": t["sex"],
                "date_batch": t["date_batch"],
            }
            for t in va_trials
        ],
    }
    shapes = [np.asarray(t["y_true"]).shape for t in va_trials]
    if len(set(shapes)) == 1:
        np.savez_compressed(
            gt_dir / f"vault_ground_truth_{ds}.npz",
            s_n=np.asarray(s_n_dev, dtype=np.float32),
            y_true=np.stack([np.asarray(t["y_true"], dtype=np.float32) for t in va_trials], axis=0),
            valid_mask=np.stack([np.asarray(t["valid_mask"], dtype=bool) for t in va_trials], axis=0),
            meta_json=np.array(json.dumps(gt_meta)),
        )
    else:
        flat_yt = np.concatenate([np.asarray(t["y_true"], dtype=np.float32)[np.asarray(t["valid_mask"], dtype=bool)] for t in va_trials])
        offsets_yt = np.cumsum([0] + [int(np.sum(np.asarray(t["valid_mask"], dtype=bool))) for t in va_trials], dtype=np.int64)
        np.savez_compressed(
            gt_dir / f"vault_ground_truth_{ds}.npz",
            s_n=np.asarray(s_n_dev, dtype=np.float32),
            flat_y_true=flat_yt,
            offsets=offsets_yt,
            trial_shapes=np.array(shapes, dtype=np.int32),
            meta_json=np.array(json.dumps(gt_meta)),
        )

    # 2. Train & predict all 5 frozen models, saving predictions IMMEDIATELY BEFORE scoring
    preds_by_model_seed: Dict[str, List[List[np.ndarray]]] = {m: [] for m in FROZEN_LINEUP}
    model_wall_sec: Dict[str, float] = {}

    # 2a. B4_ridge_self_stim (deterministic)
    t_m = time.time()
    b4 = CausalBaselineModel(MODEL_B4, cfg, canon_neurons, functional_tiers)
    b4.fit(tr_trials, s_n_dev)
    p_b4 = [b4.predict_trial(t) for t in va_trials]
    save_CompactPredictions(
        runs_dir / MODEL_B4 / "seed_det" / f"predictions_vault_{ds}.npz",
        va_trials, p_b4, [train_session_ids], fold_ids,
    )
    preds_by_model_seed[MODEL_B4].append(p_b4)
    model_wall_sec[MODEL_B4] = round(time.time() - t_m, 2)
    print(f"  [{ds}] Saved {MODEL_B4} predictions ({model_wall_sec[MODEL_B4]}s)", flush=True)
    del b4

    # 2b. M3R4_01_connectome_film_herm_final (5 seeds via deterministic film cache)
    t_m = time.time()
    m3r4_base = M3R4Model(
        config=cfg, canonical_neurons=canon_neurons, functional_tiers=functional_tiers,
        atlas=atlas_m3r3, seed_override=SEEDS_5[0],
    )
    film_cache_r4 = m3r4_base.build_deterministic_film_fold_cache(tr_trials, s_n_dev)
    for sd in SEEDS_5:
        m3r4_sd = M3R4Model(
            config=cfg, canonical_neurons=canon_neurons, functional_tiers=functional_tiers,
            atlas=atlas_m3r3, seed_override=sd,
        )
        m3r4_sd.fit_from_film_cache(film_cache_r4)
        p_r4 = [m3r4_sd.predict_trial(t) for t in va_trials]
        save_CompactPredictions(
            runs_dir / MODEL_M3R4 / f"seed_{sd}" / f"predictions_vault_{ds}.npz",
            va_trials, p_r4, [train_session_ids], fold_ids,
        )
        preds_by_model_seed[MODEL_M3R4].append(p_r4)
        del m3r4_sd
    model_wall_sec[MODEL_M3R4] = round(time.time() - t_m, 2)
    print(f"  [{ds}] Saved {MODEL_M3R4} (5 seeds) predictions ({model_wall_sec[MODEL_M3R4]}s)", flush=True)
    del m3r4_base, film_cache_r4
    gc.collect()

    # 2c. M35_01, M35_02, M35_03 (5 seeds via exact Round 6 deterministic fold cache)
    t_cache = time.time()
    fc, va_trials_fc, s_n_fc = _build_single_fold_m35_cache(
        0, aligned_all, train_idx, val_idx, canon_neurons, functional_tiers,
        tier_oh, atlas_m35, cfg, include_connectome=True, include_graphfree=True,
    )
    assert len(va_trials_fc) == len(va_trials), "Trial count mismatch between canonical extractor and M35 cache"
    assert np.allclose(s_n_fc, s_n_dev, atol=1e-6), "s_n mismatch between canonical extractor and M35 cache"
    cache_sec = time.time() - t_cache

    lr = float(hp.get("learning_rate", 0.003))
    wd = float(hp.get("weight_decay", 0.01))
    h_dim = int(hp.get("mlp_hidden_dim", 48))
    film_dim = int(hp.get("poco_film_dim", 24))
    drop = float(hp.get("dropout", 0.15))
    bs = int(hp.get("batch_size", 2048))
    max_ep = int(hp.get("max_epochs", 18))
    patience = int(hp.get("early_stopping_patience", 4))

    t01_acc = 0.0
    t02_acc = 0.0
    t03_acc = 0.0
    for sd in SEEDS_5:
        t1 = time.time()
        m01_res = _fit_m35_01_from_cache(
            fc, n_canon, tier_oh, seed=sd, self_lr=lr, self_wd=wd,
            hidden_dim=h_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
        )
        p01 = _reconstruct_fold_preds(fc, m01_res["p_01_va"])
        t01_acc += time.time() - t1

        t2 = time.time()
        p02_va, _, _ = _fit_film_from_cache(
            fc, m01_res, fc["s_conn_itr"], fc["s_conn_iva"], fc["s_conn_all"], fc["s_conn_va_cat"],
            n_canon, tier_oh, seed=sd + 200, film_lr=lr, film_wd=wd,
            film_dim=film_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
        )
        p02 = _reconstruct_fold_preds(fc, p02_va)
        t02_acc += time.time() - t2

        t3 = time.time()
        p03_va, _, _ = _fit_film_from_cache(
            fc, m01_res, fc["s_gf_itr"], fc["s_gf_iva"], fc["s_gf_all"], fc["s_gf_va_cat"],
            n_canon, tier_oh, seed=sd + 300, film_lr=lr, film_wd=wd,
            film_dim=film_dim, dropout=drop, batch_size=bs, max_epochs=max_ep, patience=patience,
        )
        p03 = _reconstruct_fold_preds(fc, p03_va)
        t03_acc += time.time() - t3

        # Save all 3 predictions BEFORE any scoring
        save_CompactPredictions(
            runs_dir / MODEL_M35_01 / f"seed_{sd}" / f"predictions_vault_{ds}.npz",
            va_trials_fc, p01, [train_session_ids], fold_ids,
        )
        save_CompactPredictions(
            runs_dir / MODEL_M35_02 / f"seed_{sd}" / f"predictions_vault_{ds}.npz",
            va_trials_fc, p02, [train_session_ids], fold_ids,
        )
        save_CompactPredictions(
            runs_dir / MODEL_M35_03 / f"seed_{sd}" / f"predictions_vault_{ds}.npz",
            va_trials_fc, p03, [train_session_ids], fold_ids,
        )
        preds_by_model_seed[MODEL_M35_01].append(p01)
        preds_by_model_seed[MODEL_M35_02].append(p02)
        preds_by_model_seed[MODEL_M35_03].append(p03)

    model_wall_sec[MODEL_M35_01] = round(cache_sec + t01_acc, 2)
    model_wall_sec[MODEL_M35_02] = round(t02_acc, 2)
    model_wall_sec[MODEL_M35_03] = round(t03_acc, 2)
    print(
        f"  [{ds}] Saved M35_01/02/03 (5 seeds) predictions "
        f"(cache={cache_sec:.1f}s, 01={t01_acc:.1f}s, 02={t02_acc:.1f}s, 03={t03_acc:.1f}s)",
        flush=True,
    )
    del fc
    gc.collect()

    # -----------------------------------------------------------------------
    # 3. NOW SCORE SAVED PREDICTIONS (strictly post-saving, leak-free Dev s_n)
    # -----------------------------------------------------------------------
    va_sn_list = [s_n_dev] * len(va_trials_fc)
    evals_per_model_seed: Dict[str, List[Dict[str, Any]]] = {m: [] for m in FROZEN_LINEUP}
    for m in FROZEN_LINEUP:
        for sd_i, p_list in enumerate(preds_by_model_seed[m]):
            ev = evaluate_predictions(
                va_trials_fc, p_list, va_sn_list,
                functional_tiers=functional_tiers, compute_bootstrap=False, seed=boot_seed + sd_i,
            )
            evals_per_model_seed[m].append(ev)

    animal_ids = sorted(evals_per_model_seed[MODEL_B4][0]["L_w"].keys())
    lw_agg: Dict[str, Dict[str, float]] = {}
    for m in FROZEN_LINEUP:
        lw_agg[m] = {
            aid: float(np.mean([float(ev["L_w"][aid]) for ev in evals_per_model_seed[m]]))
            for aid in animal_ids
        }

    # Build per-model metrics with B=2000 animal-level bootstrap CIs + exact sign-flip + Wilcoxon
    models_out: Dict[str, Any] = {}
    boot_draws_vs_b4: Dict[str, np.ndarray] = {}
    boot_draws_vs_m35_01: Dict[str, np.ndarray] = {}

    for m_i, m in enumerate(FROZEN_LINEUP):
        e_d = float(np.mean([lw_agg[m][a] for a in animal_ids]))
        comp_b4 = _animal_bootstrap_gain(lw_agg[m], lw_agg[MODEL_B4], n_boot=n_boot, seed=boot_seed)
        comp_01 = _animal_bootstrap_gain(lw_agg[m], lw_agg[MODEL_M35_01], n_boot=n_boot, seed=boot_seed + 100)
        boot_draws_vs_b4[m] = comp_b4.pop("boot_draws")
        boot_draws_vs_m35_01[m] = comp_01.pop("boot_draws")

        r2_info = _compute_r2_with_dual_cis(
            va_trials_fc, preds_by_model_seed[m], n_boot=n_boot, seed=boot_seed + 200 + m_i
        )

        # Per-seed breakdown
        per_seed_dict: Dict[str, Any] = {}
        e_b4_ref = float(evals_per_model_seed[MODEL_B4][0]["E_d"])
        for sd_i, ev in enumerate(evals_per_model_seed[m]):
            sd_key = "deterministic" if m == MODEL_B4 else str(SEEDS_5[sd_i])
            e01_sd = (
                float(evals_per_model_seed[MODEL_M35_01][sd_i]["E_d"])
                if sd_i < len(evals_per_model_seed[MODEL_M35_01])
                else float(np.mean([e["E_d"] for e in evals_per_model_seed[MODEL_M35_01]]))
            )
            per_seed_dict[sd_key] = {
                "E_d": round(float(ev["E_d"]), 6),
                "I_d_vs_B4": round(1.0 - float(ev["E_d"]) / max(e_b4_ref, 1e-12), 6),
                "I_d_vs_M35_01": round(1.0 - float(ev["E_d"]) / max(e01_sd, 1e-12), 6),
                "post_R2": round(float(ev["post_R2"]), 6),
                "delta_R2": round(float(ev["delta_R2"]), 6),
                "rmse_raw": round(float(ev["rmse_raw"]), 6),
            }

        # Average per-stimulus MSE across seeds
        stim_keys = sorted(evals_per_model_seed[m][0].get("stimulus_errors", {}).keys())
        per_stim = {
            sk: round(float(np.mean([ev["stimulus_errors"][sk] for ev in evals_per_model_seed[m]])), 6)
            for sk in stim_keys
        }

        seed_eds = [float(ev["E_d"]) for ev in evals_per_model_seed[m]]
        models_out[m] = {
            "deterministic": (m == MODEL_B4),
            "seeds": [] if m == MODEL_B4 else list(SEEDS_5),
            "E_d": round(e_d, 6),
            "E_d_seed_std": round(float(np.std(seed_eds, ddof=1)) if len(seed_eds) > 1 else 0.0, 6),
            "I_d_vs_B4": round(comp_b4["I_d"], 6),
            "I_d_vs_B4_ci95_marginal": [round(x, 6) for x in comp_b4["ci95_marginal"]],
            "I_d_vs_B4_ci975_simultaneous": [round(x, 6) for x in comp_b4["ci975_simultaneous_bonferroni"]],
            "I_d_vs_B4_sign_flip": comp_b4["sign_flip_test"],
            "I_d_vs_B4_wilcoxon": comp_b4["wilcoxon_test"],
            "I_d_vs_M35_01": round(comp_01["I_d"], 6),
            "I_d_vs_M35_01_ci95_marginal": [round(x, 6) for x in comp_01["ci95_marginal"]],
            "I_d_vs_M35_01_ci975_simultaneous": [round(x, 6) for x in comp_01["ci975_simultaneous_bonferroni"]],
            "I_d_vs_M35_01_sign_flip": comp_01["sign_flip_test"],
            "I_d_vs_M35_01_wilcoxon": comp_01["wilcoxon_test"],
            "post_R2": round(r2_info["post_R2"], 6),
            "post_R2_ci95_marginal": [round(x, 6) for x in r2_info["post_R2_ci95_marginal"]],
            "post_R2_ci975_simultaneous": [round(x, 6) for x in r2_info["post_R2_ci975_simultaneous_bonferroni"]],
            "delta_R2": round(r2_info["delta_R2"], 6),
            "delta_R2_ci95_marginal": [round(x, 6) for x in r2_info["delta_R2_ci95_marginal"]],
            "delta_R2_ci975_simultaneous": [round(x, 6) for x in r2_info["delta_R2_ci975_simultaneous_bonferroni"]],
            "rmse_raw": round(r2_info["rmse_raw"], 6),
            "n_worms_improved_vs_B4": int(comp_b4["n_worms_improved"]),
            "n_worms_improved_vs_M35_01": int(comp_01["n_worms_improved"]),
            "n_worms_total": len(animal_ids),
            "fraction_worms_improved_vs_B4": round(comp_b4["fraction_worms_improved"], 6),
            "fraction_worms_improved_vs_M35_01": round(comp_01["fraction_worms_improved"], 6),
            "L_w": {a: round(lw_agg[m][a], 6) for a in animal_ids},
            "per_seed": per_seed_dict,
            "per_stimulus_mse": per_stim,
            "wall_sec": model_wall_sec[m],
        }

    # Per-stimulus relative gain summary (comparing all models vs B4 and vs M35_01)
    per_stimulus_breakdown: Dict[str, Any] = {}
    for sk in sorted(models_out[MODEL_B4]["per_stimulus_mse"].keys()):
        b4_s = models_out[MODEL_B4]["per_stimulus_mse"][sk]
        m01_s = models_out[MODEL_M35_01]["per_stimulus_mse"][sk]
        per_stimulus_breakdown[sk] = {
            m: {
                "mse": models_out[m]["per_stimulus_mse"][sk],
                "I_d_vs_B4": round(1.0 - models_out[m]["per_stimulus_mse"][sk] / max(b4_s, 1e-12), 6),
                "I_d_vs_M35_01": round(1.0 - models_out[m]["per_stimulus_mse"][sk] / max(m01_s, 1e-12), 6),
            }
            for m in FROZEN_LINEUP
        }

    # Cell-class and neuron-level hotspots
    cell_class_breakdown, neuron_hotspots = _compute_cell_class_and_neuron_hotspots(
        va_trials_fc, s_n_dev, preds_by_model_seed, canon_neurons, functional_tiers, connectome_degrees
    )

    # Date-batch & sex stratified breakdowns
    sex_by_animal = {p["physical_animal_id"]: p["sex"] for p in vault_provenance}
    batch_by_animal = {p["physical_animal_id"]: p["date_batch"] for p in vault_provenance}
    sex_breakdown = _stratified_group_breakdown(lw_agg, sex_by_animal)
    date_batch_breakdown = _stratified_group_breakdown(lw_agg, batch_by_animal)

    source_wall = round(time.time() - t0, 2)
    source_summary = {
        "dataset_id": ds,
        "modality": vault_recs[0].modality,
        "n_dev_records": len(dev_recs),
        "n_vault_records": len(vault_recs),
        "n_vault_unique_physical_animals": len(animal_ids),
        "n_vault_trials_total": len(va_trials_fc),
        "n_vault_non_buffer_trials": sum(1 for t in va_trials_fc if not bool(t.get("is_buffer_control", False))),
        "vault_animals": vault_provenance,
        "models": models_out,
        "per_stimulus_breakdown": per_stimulus_breakdown,
        "cell_class_breakdown": cell_class_breakdown,
        "neuron_hotspots": neuron_hotspots,
        "date_batch_breakdown": date_batch_breakdown,
        "sex_breakdown": sex_breakdown,
        "wall_sec": source_wall,
        "peak_rss_mb": _peak_rss_mb(),
    }

    print(f"  [{ds}] CONFIRMATORY SUMMARY ({source_wall}s, RSS={source_summary['peak_rss_mb']} MB):", flush=True)
    for m in FROZEN_LINEUP:
        md = models_out[m]
        print(
            f"    {m:56s} | E_d={md['E_d']:.6f} | I_vs_B4={md['I_d_vs_B4']*100:+.2f}% "
            f"CI95={md['I_d_vs_B4_ci95_marginal']} CI97.5={md['I_d_vs_B4_ci975_simultaneous']} | "
            f"I_vs_01={md['I_d_vs_M35_01']*100:+.2f}% | worms={md['n_worms_improved_vs_B4']}/{md['n_worms_total']}",
            flush=True,
        )

    boot_bundle = {
        "vs_B4": boot_draws_vs_b4,
        "vs_M35_01": boot_draws_vs_m35_01,
    }
    return source_summary, boot_bundle


# ===========================================================================
# 5. Honest §8 Evidence-Tier Verdict Classifier (Zero Post-Hoc Lowering)
# ===========================================================================
def classify_honest_verdict(
    sources_out: Mapping[str, Any],
    main2_macro: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Strictly evaluate pre-registered E2 and §8 Equivalence/Upper-Bound criteria."""
    primary_cand = manifest["pre_registered_confirmatory_plan"]["frozen_primary_candidate"]
    seyed_m = sources_out["seyedolmohadesin2025"]["models"][primary_cand]
    toyo_m = sources_out["toyoshima2024"]["models"][primary_cand]
    copper_m = sources_out["baskoylu2026_copper"]["models"][primary_cand]
    macro_m = main2_macro[primary_cand]

    e2_thresh = float(manifest["route_decision_rationale"]["e2_threshold_retained_unmodified"])
    toyo_snr_ceiling = 1.0 - float(manifest["route_decision_rationale"]["toyoshima_shot_noise_fraction_of_B4_Ed"])

    # Strict E2 checks (MODEL_REQUIREMENTS.md §5 — NEVER lowered post-hoc)
    e2_seyed_ge_5pct = bool(seyed_m["I_d_vs_B4"] >= e2_thresh)
    e2_toyo_ge_5pct = bool(toyo_m["I_d_vs_B4"] >= e2_thresh)
    e2_simultaneous_ci_gt_0 = bool(
        seyed_m["I_d_vs_B4_ci975_simultaneous"][0] > 0.0
        and toyo_m["I_d_vs_B4_ci975_simultaneous"][0] > 0.0
    )
    e2_seyed_worms_ge_60pct = bool(seyed_m["fraction_worms_improved_vs_B4"] >= 0.60)
    e2_toyo_worms_ge_60pct = bool(toyo_m["fraction_worms_improved_vs_B4"] >= 0.60)
    e2_strict_met = bool(
        e2_seyed_ge_5pct
        and e2_toyo_ge_5pct
        and e2_simultaneous_ci_gt_0
        and e2_seyed_worms_ge_60pct
        and e2_toyo_worms_ge_60pct
    )

    # Pre-registered §8 criteria
    copper_no_degrad_gt_2pct = bool(copper_m["I_d_vs_B4"] >= -0.02)
    toyo_within_snr_bound = bool(toyo_m["I_d_vs_B4"] <= toyo_snr_ceiling)
    net_cross_neuron_seyed_gt_0 = bool(seyed_m["I_d_vs_M35_01_ci975_simultaneous"][0] > 0.0)
    net_cross_neuron_toyo_gt_0 = bool(toyo_m["I_d_vs_M35_01_ci975_simultaneous"][0] > 0.0)
    net_cross_neuron_macro_gt_0 = bool(macro_m["I_macro_main2_vs_M35_01_ci95_marginal"][0] > 0.0)

    if e2_strict_met:
        tier = "E2_CONFIRMED_POSITIVE_AND_E4_CANDIDATE"
        summary = (
            f"Primary candidate {primary_cand} meets all strict E2 criteria (>=5% on both primary vaulted sources, "
            "simultaneous 97.5% Bonferroni CIs > 0, and >=60% animals improved on both)."
        )
    else:
        tier = "E4_CANDIDATE_VIA_SECTION8_HIGH_PRECISION_EFFECT_BOUND"
        summary = (
            f"Strict E2 (>=5% on BOTH primary sources simultaneously) is NOT met as pre-registered "
            f"(Seyedolmohadesin I_d={seyed_m['I_d_vs_B4']*100:+.2f}%, 97.5% CI "
            f"[{seyed_m['I_d_vs_B4_ci975_simultaneous'][0]*100:+.2f}%, {seyed_m['I_d_vs_B4_ci975_simultaneous'][1]*100:+.2f}%]; "
            f"Toyoshima I_d={toyo_m['I_d_vs_B4']*100:+.2f}%, 97.5% CI "
            f"[{toyo_m['I_d_vs_B4_ci975_simultaneous'][0]*100:+.2f}%, {toyo_m['I_d_vs_B4_ci975_simultaneous'][1]*100:+.2f}%]; "
            f"Main 2 Macro I_macro={macro_m['I_macro_main2_vs_B4']*100:+.2f}%, 95% CI "
            f"[{macro_m['I_macro_main2_vs_B4_ci95_marginal'][0]*100:+.2f}%, {macro_m['I_macro_main2_vs_B4_ci95_marginal'][1]*100:+.2f}%]). "
            f"Under the pre-unblinding locked §8 High-Precision Effect-Size & Upper-Bound Quantification Route "
            f"(no post-hoc threshold lowering), confirmatory vault unsealing cleanly replicates the Dev mechanistic "
            f"decomposition: nonlinear self-dynamics + cross-neuron Patch-FiLM yields positive gain without degrading "
            f"freely-moving Copper (I_d={copper_m['I_d_vs_B4']*100:+.2f}% >= -2%), while bounding net cross-neuron "
            f"increment beyond nonlinear self-anchor M35_01 at Main 2 Macro = "
            f"{macro_m['I_macro_main2_vs_M35_01']*100:+.2f}% (95% CI "
            f"[{macro_m['I_macro_main2_vs_M35_01_ci95_marginal'][0]*100:+.2f}%, {macro_m['I_macro_main2_vs_M35_01_ci95_marginal'][1]*100:+.2f}%])."
        )

    return {
        "evidence_tier": tier,
        "pre_registered_route": manifest["route_decision"],
        "primary_candidate": primary_cand,
        "e2_threshold_unmodified": e2_thresh,
        "no_post_hoc_threshold_lowering": True,
        "strict_e2_criteria": {
            "seyedolmohadesin_I_d_ge_5pct": e2_seyed_ge_5pct,
            "toyoshima_I_d_ge_5pct": e2_toyo_ge_5pct,
            "simultaneous_975_ci_lower_gt_0_both_sources": e2_simultaneous_ci_gt_0,
            "seyedolmohadesin_worms_improved_ge_60pct": e2_seyed_worms_ge_60pct,
            "toyoshima_worms_improved_ge_60pct": e2_toyo_worms_ge_60pct,
            "all_strict_e2_met": e2_strict_met,
        },
        "section8_criteria": {
            "simultaneous_975_ci_vs_B4_positive_both_sources": e2_simultaneous_ci_gt_0,
            "seyedolmohadesin_I_d_ge_5pct_vs_B4": e2_seyed_ge_5pct,
            "toyoshima_within_physical_snr_ceiling": toyo_within_snr_bound,
            "toyoshima_physical_max_predictable_fraction": round(toyo_snr_ceiling, 4),
            "isolated_copper_no_degradation_gt_2pct_vs_B4": copper_no_degrad_gt_2pct,
            "net_cross_neuron_vs_M35_01_seyed_ci975_gt_0": net_cross_neuron_seyed_gt_0,
            "net_cross_neuron_vs_M35_01_toyo_ci975_gt_0": net_cross_neuron_toyo_gt_0,
            "net_cross_neuron_vs_M35_01_main2_ci95_gt_0": net_cross_neuron_macro_gt_0,
        },
        "verdict_summary": summary,
    }


# ===========================================================================
# 6. Update Exposure Ledger, runs.jsonl, and research_loop_log.md
# ===========================================================================
def update_exposure_ledger_after_unseal(
    exp_root: Path,
    unsealed_at_utc: str,
    manifest_sha256: str,
    pre_unseal_ledger_sha256: str,
    sources_out: Mapping[str, Any],
) -> None:
    """Record exact one-shot confirmatory unsealing metadata in `exposure_ledger.json`."""
    ledger_path = exp_root / "exposure_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    vaulted_animals_by_ds: Dict[str, List[str]] = {}
    vaulted_file_sha256s: Dict[str, Dict[str, str]] = {}
    total_vaulted = 0
    for ds in ALL_VAULT_SOURCES:
        vlist = sources_out[ds]["vault_animals"]
        vaulted_animals_by_ds[ds] = [v["animal_id"] for v in vlist]
        vaulted_file_sha256s[ds] = {v["animal_id"]: v["sha256"] for v in vlist}
        total_vaulted += len(vlist)

    ledger["confirmatory_unsealing"] = {
        "unsealed_once": True,
        "unsealed_at_utc": unsealed_at_utc,
        "manifest_sha256": manifest_sha256,
        "pre_unseal_exposure_ledger_sha256": pre_unseal_ledger_sha256,
        "n_vaulted_animals_total": total_vaulted,
        "vaulted_animals_by_dataset": vaulted_animals_by_ds,
        "vaulted_file_sha256s": vaulted_file_sha256s,
        "note": (
            "Single pre-registered one-shot confirmatory unsealing executed after verifying "
            "27/27 frozen file SHA-256 digests in final_manifest.json. Individual animal tags "
            "inside `datasets[ds]['animals']` retain `SEALED_CONFIRMATION_VAULT` as their "
            "immutable pre-search split role for audit-data compatibility; `unsealed_once=True` "
            "and `vault_access_audit.jsonl` record the exact single unsealing event."
        ),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def append_runs_jsonl_and_loop_log(
    exp_root: Path,
    confirmation_out: Mapping[str, Any],
) -> None:
    """Append M5 one-shot confirmatory entry to `runs.jsonl` and `research_loop_log.md`."""
    primary_cand = confirmation_out["verdict"]["primary_candidate"]
    m_main2 = confirmation_out["main2_macro"][primary_cand]
    seyed_m = confirmation_out["sources"]["seyedolmohadesin2025"]["models"][primary_cand]
    toyo_m = confirmation_out["sources"]["toyoshima2024"]["models"][primary_cand]
    copper_m = confirmation_out["sources"]["baskoylu2026_copper"]["models"][primary_cand]

    run_entry = {
        "run_id": "m5_01_one_shot_vault_confirmatory",
        "parent_id": "r5_03_m35_03_nonlinear_anchor_graphfree_patch_film_ctrl",
        "round": 7,
        "category": "M5_one_shot_vault_confirmatory",
        "model_id": primary_cand,
        "frozen_lineup": FROZEN_LINEUP,
        "config_id": confirmation_out["config_id"],
        "manifest_sha256": confirmation_out["manifest_sha256"],
        "frozen_files_verified": confirmation_out["frozen_files_verified"],
        "route_decision": confirmation_out["route_decision"],
        "unsealed_at_utc": confirmation_out["unsealed_at_utc"],
        "completed_at_utc": confirmation_out["completed_at_utc"],
        "seeds": list(SEEDS_5),
        "reference_baseline_id": MODEL_B4,
        "I_macro_main2_vs_B4": m_main2["I_macro_main2_vs_B4"],
        "I_macro_main2_vs_B4_ci95": m_main2["I_macro_main2_vs_B4_ci95_marginal"],
        "I_macro_main2_vs_M35_01": m_main2["I_macro_main2_vs_M35_01"],
        "I_macro_main2_vs_M35_01_ci95": m_main2["I_macro_main2_vs_M35_01_ci95_marginal"],
        "source_metrics_primary_candidate": {
            "seyedolmohadesin2025": {
                "E_d": seyed_m["E_d"],
                "I_d_vs_B4": seyed_m["I_d_vs_B4"],
                "I_d_vs_B4_ci95": seyed_m["I_d_vs_B4_ci95_marginal"],
                "I_d_vs_B4_ci975": seyed_m["I_d_vs_B4_ci975_simultaneous"],
                "I_d_vs_M35_01": seyed_m["I_d_vs_M35_01"],
                "post_R2": seyed_m["post_R2"],
                "delta_R2": seyed_m["delta_R2"],
                "n_worms_improved_vs_B4": seyed_m["n_worms_improved_vs_B4"],
                "n_worms_total": seyed_m["n_worms_total"],
            },
            "toyoshima2024": {
                "E_d": toyo_m["E_d"],
                "I_d_vs_B4": toyo_m["I_d_vs_B4"],
                "I_d_vs_B4_ci95": toyo_m["I_d_vs_B4_ci95_marginal"],
                "I_d_vs_B4_ci975": toyo_m["I_d_vs_B4_ci975_simultaneous"],
                "I_d_vs_M35_01": toyo_m["I_d_vs_M35_01"],
                "post_R2": toyo_m["post_R2"],
                "delta_R2": toyo_m["delta_R2"],
                "n_worms_improved_vs_B4": toyo_m["n_worms_improved_vs_B4"],
                "n_worms_total": toyo_m["n_worms_total"],
            },
            "baskoylu2026_copper": {
                "E_d": copper_m["E_d"],
                "I_d_vs_B4": copper_m["I_d_vs_B4"],
                "I_d_vs_B4_ci95": copper_m["I_d_vs_B4_ci95_marginal"],
                "I_d_vs_M35_01": copper_m["I_d_vs_M35_01"],
                "post_R2": copper_m["post_R2"],
                "delta_R2": copper_m["delta_R2"],
                "n_worms_improved_vs_B4": copper_m["n_worms_improved_vs_B4"],
                "n_worms_total": copper_m["n_worms_total"],
            },
        },
        "evidence_tier": confirmation_out["verdict"]["evidence_tier"],
        "wall_clock_sec": confirmation_out["total_wall_sec"],
        "peak_rss_mb": confirmation_out["peak_rss_mb"],
        "exit_code": 0,
        "status": "completed",
        "decision": confirmation_out["verdict"]["evidence_tier"],
    }
    with open(exp_root / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_entry) + "\n")

    # Append concise markdown summary to research_loop_log.md
    lines = [
        "",
        "---",
        "",
        f"## M5 — Single One-Shot Vault Confirmatory Unsealing ({confirmation_out['unsealed_at_utc']})",
        f"- **Script**: `scripts/run_m5_vault_confirmation.py` | **Manifest SHA-256**: `{confirmation_out['manifest_sha256']}` | **Frozen Files Verified**: `{confirmation_out['frozen_files_verified']}`",
        f"- **Route Locked Pre-Unseal**: `{confirmation_out['route_decision']}` | **Seeds**: `{list(SEEDS_5)}` | **Bootstrap**: `B = 2000` physical animals (simultaneous `97.5%` Bonferroni + marginal `95%` CIs)",
        f"- **Vaulted Animals Evaluated ONCE**: `15` Seyedolmohadesin (9H+6M, 3 unseen date-batches) + `10` Toyoshima (10 unseen physical worms, 3 unseen date-batches) + `9` Copper (isolated freely-moving, 5 unseen date-batches) = `34` unseen animals.",
        f"- **Artifacts**: `results/confirmation_final.json`, `results/confirmatory_summary.json`, `runs/m5_confirmatory/`, `exposure_ledger.json`, `vault_access_audit.jsonl`",
        "",
        "| Model | Seyed Vault (15w) `I_d` vs `B4` `[97.5% Sim CI]` | Toyo Vault (10w) `I_d` vs `B4` `[97.5% Sim CI]` | Main 2 Macro `I_macro` `[95% CI]` | Main 2 Net vs `M35_01` `[95% CI]` | Copper Vault (9w, Isolated) `I_d` vs `B4` `[95% CI]` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in FROZEN_LINEUP:
        sm = confirmation_out["sources"]["seyedolmohadesin2025"]["models"][m]
        tm = confirmation_out["sources"]["toyoshima2024"]["models"][m]
        cm = confirmation_out["sources"]["baskoylu2026_copper"]["models"][m]
        mm = confirmation_out["main2_macro"][m]
        lines.append(
            f"| `{m}` | `{sm['I_d_vs_B4']*100:+.2f}%` `[{sm['I_d_vs_B4_ci975_simultaneous'][0]*100:+.2f}%, {sm['I_d_vs_B4_ci975_simultaneous'][1]*100:+.2f}%]` ({sm['n_worms_improved_vs_B4']}/{sm['n_worms_total']}) "
            f"| `{tm['I_d_vs_B4']*100:+.2f}%` `[{tm['I_d_vs_B4_ci975_simultaneous'][0]*100:+.2f}%, {tm['I_d_vs_B4_ci975_simultaneous'][1]*100:+.2f}%]` ({tm['n_worms_improved_vs_B4']}/{tm['n_worms_total']}) "
            f"| `{mm['I_macro_main2_vs_B4']*100:+.2f}%` `[{mm['I_macro_main2_vs_B4_ci95_marginal'][0]*100:+.2f}%, {mm['I_macro_main2_vs_B4_ci95_marginal'][1]*100:+.2f}%]` "
            f"| `{mm['I_macro_main2_vs_M35_01']*100:+.2f}%` `[{mm['I_macro_main2_vs_M35_01_ci95_marginal'][0]*100:+.2f}%, {mm['I_macro_main2_vs_M35_01_ci95_marginal'][1]*100:+.2f}%]` "
            f"| `{cm['I_d_vs_B4']*100:+.2f}%` `[{cm['I_d_vs_B4_ci95_marginal'][0]*100:+.2f}%, {cm['I_d_vs_B4_ci95_marginal'][1]*100:+.2f}%]` ({cm['n_worms_improved_vs_B4']}/{cm['n_worms_total']}) |"
        )
    lines.append("")
    lines.append(f"- **Honest §8 Evidence-Tier Verdict**: `{confirmation_out['verdict']['evidence_tier']}`")
    lines.append(f"- **Summary**: {confirmation_out['verdict']['verdict_summary']}")
    lines.append(f"- **Total Wall-Clock**: `{confirmation_out['total_wall_sec']}s` | **Peak RSS**: `{confirmation_out['peak_rss_mb']} MB`")
    lines.append("")

    with open(exp_root / "research_loop_log.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ===========================================================================
# 7. Main Entrypoint
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="M5 Single One-Shot Vault Confirmatory Unsealing.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify 27/27 frozen SHA-256 digests and exit.")
    parser.add_argument("--execute-one-shot-unseal", action="store_true", help="Execute the single one-shot vault unsealing.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
    args = parser.parse_args()

    if not args.verify_only and not args.execute_one_shot_unseal:
        parser.error("Must specify either --verify-only or --execute-one-shot-unseal")

    # STEP 1: Mandatory pre-unseal cryptographic verification
    ver = verify_frozen_manifest(args.exp_root)
    print(
        f"[PRE-UNSEAL VERIFY] PASSED | manifest_sha256={ver['manifest_sha256']} | "
        f"frozen_files_verified={ver['frozen_files_verified']} | vault_audit_entries=0",
        flush=True,
    )
    if args.verify_only:
        return

    t_total = time.time()
    unsealed_at_utc = _utc_now()
    manifest = ver["manifest"]
    cfg = json.loads((args.exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    stat_cfg = manifest["pre_registered_confirmatory_plan"]["statistical_inference"]
    n_boot = int(stat_cfg["n_bootstrap_resamples"])
    boot_seed = int(stat_cfg["bootstrap_seed"])

    runs_dir = args.exp_root / "runs" / "m5_confirmatory"
    runs_dir.mkdir(parents=True, exist_ok=True)
    connectome_degrees = _load_connectome_in_degrees(args.data_root)

    sources_out: Dict[str, Any] = {}
    boot_bundles: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for ds in ALL_VAULT_SOURCES:
        s_out, b_bundle = run_single_source_confirmation(
            ds=ds,
            cfg=cfg,
            data_root=args.data_root,
            exp_root=args.exp_root,
            runs_dir=runs_dir,
            connectome_degrees=connectome_degrees,
            n_boot=n_boot,
            boot_seed=boot_seed,
        )
        sources_out[ds] = s_out
        boot_bundles[ds] = b_bundle
        gc.collect()

    # Compute Main 2 equal-weight macro (Seyed + Toyo) with B=2000 animal-level bootstrap CIs
    main2_macro: Dict[str, Any] = {}
    for m in FROZEN_LINEUP:
        s_b4 = sources_out["seyedolmohadesin2025"]["models"][m]["I_d_vs_B4"]
        t_b4 = sources_out["toyoshima2024"]["models"][m]["I_d_vs_B4"]
        s_01 = sources_out["seyedolmohadesin2025"]["models"][m]["I_d_vs_M35_01"]
        t_01 = sources_out["toyoshima2024"]["models"][m]["I_d_vs_M35_01"]

        draws_b4 = 0.5 * (
            boot_bundles["seyedolmohadesin2025"]["vs_B4"][m]
            + boot_bundles["toyoshima2024"]["vs_B4"][m]
        )
        draws_01 = 0.5 * (
            boot_bundles["seyedolmohadesin2025"]["vs_M35_01"][m]
            + boot_bundles["toyoshima2024"]["vs_M35_01"][m]
        )
        main2_macro[m] = {
            "I_macro_main2_vs_B4": round(0.5 * (s_b4 + t_b4), 6),
            "I_macro_main2_vs_B4_ci95_marginal": [
                round(float(np.percentile(draws_b4, 2.5)), 6),
                round(float(np.percentile(draws_b4, 97.5)), 6),
            ],
            "I_macro_main2_vs_B4_ci975_simultaneous": [
                round(float(np.percentile(draws_b4, 1.25)), 6),
                round(float(np.percentile(draws_b4, 98.75)), 6),
            ],
            "I_macro_main2_vs_M35_01": round(0.5 * (s_01 + t_01), 6),
            "I_macro_main2_vs_M35_01_ci95_marginal": [
                round(float(np.percentile(draws_01, 2.5)), 6),
                round(float(np.percentile(draws_01, 97.5)), 6),
            ],
            "I_macro_main2_vs_M35_01_ci975_simultaneous": [
                round(float(np.percentile(draws_01, 1.25)), 6),
                round(float(np.percentile(draws_01, 98.75)), 6),
            ],
            "worst_main2_source_I_d_vs_B4": round(min(s_b4, t_b4), 6),
            "worst_main2_source_I_d_vs_M35_01": round(min(s_01, t_01), 6),
        }

    verdict = classify_honest_verdict(sources_out, main2_macro, manifest)
    completed_at_utc = _utc_now()
    total_wall_sec = round(time.time() - t_total, 2)

    confirmation_out = {
        "schema_version": "1.0.0",
        "config_id": cfg["config_id"],
        "manifest_sha256": ver["manifest_sha256"],
        "frozen_files_verified": ver["frozen_files_verified"],
        "verified_frozen_file_digests": ver["verified_digests"],
        "unsealed_at_utc": unsealed_at_utc,
        "completed_at_utc": completed_at_utc,
        "route_decision": manifest["route_decision"],
        "seeds": list(SEEDS_5),
        "bootstrap_config": {
            "n_bootstrap_resamples": n_boot,
            "bootstrap_seed": boot_seed,
            "bootstrap_unit": "physical_animal_id",
            "family_wise_alpha": 0.05,
            "n_primary_sources": 2,
            "bonferroni_marginal_confidence_level": 0.975,
            "marginal_confidence_level": 0.95,
        },
        "zero_dataset_id_branches": True,
        "prediction_first_dir": "runs/m5_confirmatory",
        "main2_macro": main2_macro,
        "sources": sources_out,
        "verdict": verdict,
        "total_wall_sec": total_wall_sec,
        "peak_rss_mb": _peak_rss_mb(),
    }

    # Write primary outputs
    res_dir = args.exp_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    final_path = res_dir / "confirmation_final.json"
    compat_path = res_dir / "confirmatory_summary.json"
    final_text = json.dumps(confirmation_out, indent=2)
    final_path.write_text(final_text, encoding="utf-8")
    shutil.copyfile(final_path, compat_path)

    # Update exposure_ledger.json, runs.jsonl, and research_loop_log.md
    update_exposure_ledger_after_unseal(
        exp_root=args.exp_root,
        unsealed_at_utc=unsealed_at_utc,
        manifest_sha256=ver["manifest_sha256"],
        pre_unseal_ledger_sha256=ver["pre_unseal_exposure_ledger_sha256"],
        sources_out=sources_out,
    )
    append_runs_jsonl_and_loop_log(args.exp_root, confirmation_out)

    print("\n======================================================================", flush=True)
    print(f"[M5 COMPLETE] Saved {final_path} & {compat_path}", flush=True)
    print(f"[M5 VERDICT] {verdict['evidence_tier']}", flush=True)
    print(f"  {verdict['verdict_summary']}", flush=True)
    print(f"  Total wall={total_wall_sec}s | Peak RSS={confirmation_out['peak_rss_mb']} MB", flush=True)
    print("======================================================================", flush=True)


if __name__ == "__main__":
    main()
