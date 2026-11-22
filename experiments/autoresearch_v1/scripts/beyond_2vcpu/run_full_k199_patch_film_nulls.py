#!/usr/bin/env python3
"""Standalone CPU-Only PyTorch 2.8 Extension Script 1:
Full K_film=199 End-to-End M35_02 Patch-FiLM Constrained Null-Graph Retraining.

Scales the in-container K_film=19 end-to-end PyTorch M35_02 Patch-FiLM null-graph
pilot (which hit the p=1/(1+19)=0.05 empirical resolution floor on Seyedolmohadesin
2025 with 0/19 nulls beating the real connectome, z=+2.25) to full K_film=199
(empirical p-value resolution floor 0.005) across Dev datasets (`yemini2021`,
`seyedolmohadesin2025`, `toyoshima2024`).

Key guarantees:
- STRICTLY Dev splits only (`allow_vault_unseal=False`; zero `vault_sealed/` access).
- Strictly causal `[-10, 0)s -> [0, 10)s` window, fold-internal `s_n`, zero `dataset_id` branches.
- CPU-only default (`--device cpu`), multi-process `--n-jobs` parallelization across
  null graphs `k = 1..K` with `OMP_NUM_THREADS=1` & `torch.set_num_threads(1)` per worker.
- Atomic per-null JSON checkpointing (`null_k{k:03d}.json`) + `--resume` support.
- Exact empirical one-sided permutation p-value `p = (1 + count(I_null >= I_real)) / (1 + K)`
  (equivalently `count(E_null <= E_real)`) and z-score output.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Enforce single-threaded BLAS before importing numpy/torch in worker processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from models.m3_round1_h1 import (
    M3R1Model,
    _build_m3_fold_cache,
    extract_circuit_slow_state_features,
    fit_causal_slow_state_pca,
)
from models.m3_round2_h2 import _apply_svd_compressor, _fit_svd_compressor
from models.m3_round4_f11 import DATASETS_ALL3, DATASETS_MAIN2, PRIMARY_SEED, _extract_fold_trials
from models.m35_round5_nonlinear import (
    M35Model,
    TIER_NAMES,
    ZeroInitPatchFiLMGate,
    _train_torch_residual_with_inner_cv,
    build_1hop_2hop_connectome_atlas,
)
from models.m35_round6_validation import (
    _compute_1hop2hop_drives_batched,
    _compute_null_stats,
    _retrain_m35_02_film_on_graph,
    _expand_stack_with_2hop,
    _extract_sm_fast_slow,
    _extract_u_stim,
    _tier_onehot,
)
from models.m4_null_graphs import ConstrainedNullConnectomeGenerator, ensure_null_bank
from src.baselines import _project_trial_residual_targets, _reconstruct_y_pred_from_coef
from src.data_adapters import DATA_ROOT_DEFAULT, load_dataset
from src.evaluator import align_records_to_canonical, build_grouped_cv_splits, evaluate_predictions


# Global worker state initialized via fork or pool initializer
_WORKER_STATE: Dict[str, Any] = {}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _init_worker(ds_cache: Dict[str, Any], hp: Dict[str, Any], seed: int, device: str) -> None:
    """Configure single-threaded CPU execution inside each pool worker."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _WORKER_STATE["ds_cache"] = ds_cache
    _WORKER_STATE["hp"] = hp
    _WORKER_STATE["seed"] = int(seed)
    _WORKER_STATE["device"] = str(device)


def _eval_single_null_worker(args: Tuple[int, np.ndarray, str]) -> Dict[str, Any]:
    """Worker entrypoint: genuinely retrain M35_02 Patch-FiLM on null graph k."""
    k_idx, w_stack_k, ckpt_path_str = args
    t0 = time.time()
    ds_cache = _WORKER_STATE["ds_cache"]
    hp = _WORKER_STATE["hp"]
    seed = _WORKER_STATE["seed"]

    res = _retrain_m35_02_film_on_graph(ds_cache, w_stack_k, hp, seed=seed)
    wall = round(time.time() - t0, 3)
    record = {
        "k": int(k_idx),
        "E_d": float(res["E_d"]),
        "post_R2": float(res["post_R2"]),
        "wall_sec": wall,
        "worker_pid": os.getpid(),
        "completed_utc": _utc_now(),
    }
    _atomic_write_json(Path(ckpt_path_str), record)
    return record


def _build_dev_anchored_cache(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    max_folds: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """Build deterministic Dev-only M35_01 nonlinear anchor cache (strictly allow_vault_unseal=False)."""
    raw_recs = load_dataset(
        dataset_id=dataset_id,
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="beyond_2vcpu_k199_patch_film_nulls_dev_only",
    )
    aligned, canon, tiers = align_records_to_canonical(raw_recs)
    splits = build_grouped_cv_splits(aligned)
    if max_folds is not None and max_folds > 0:
        splits = splits[: int(max_folds)]

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
    anchor_preds_all: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
        fc_m3 = _build_m3_fold_cache(train_trials + val_trials, train_trials, s_n, n_canon, cfg)

        m01 = M35Model("M35_01_zeroinit_nonlinear_self_anchor", cfg, canon, tiers, atlas)
        shared_01 = m01.fit_shared_nonlinear_anchor(train_trials, s_n, fold_cache=fc_m3)
        st01_out = shared_01["outer"]
        st01_in = shared_01["inner"]
        priors_out = st01_out["priors"]
        mu_out = priors_out["mu_train"]
        non_buf = shared_01["non_buf"]

        pca_out = fit_causal_slow_state_pca(
            non_buf, s_n, mu_out, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

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
                z_tr.append(extract_circuit_slow_state_features(tr, canon, s_n, mu_out, pca_out, kappa=kappa)[idx])

        n_out_i = np.concatenate(tr_ni).astype(np.int64)
        z_all_arr = np.vstack(z_tr).astype(np.float32)
        outer_z_comp = _fit_svd_compressor(z_all_arr, n_comp=p_z)
        z_all_low = _apply_svd_compressor(z_all_arr, outer_z_comp)

        smf_va, sms_va, anc_va, pdel_va, phi_va, u_va, z_va_low = [], [], [], [], [], [], []
        for tr in val_trials:
            sf, ss = _extract_sm_fast_slow(tr, s_n, mu_out)
            smf_va.append(sf)
            sms_va.append(ss)
            p_nl_va = m01._predict_m35_01_coef(tr, s_n, st01_out)
            anc_va.append(p_nl_va)
            pdel, _, phi = predict_prior_delta_trajectory(tr, priors_out, n_canon)
            pdel_va.append(pdel)
            phi_va.append(phi)
            u_va.append(_extract_u_stim(tr, s_n, priors_out, n_canon))
            z_raw = extract_circuit_slow_state_features(tr, canon, s_n, mu_out, pca_out, kappa=kappa)
            z_va_low.append(_apply_svd_compressor(z_raw, outer_z_comp))
            anchor_preds_all.append(
                _reconstruct_y_pred_from_coef(tr, pdel, phi, p_nl_va, s_n, mu_out)
            )

        # Inner split for Patch-FiLM early stopping & alpha_film shrinkage
        priors_in = st01_in["priors"]
        mu_in = priors_in["mu_train"]
        inner_tr = shared_01["inner_tr"]
        inner_va = shared_01["inner_va"]
        pca_in = fit_causal_slow_state_pca(
            inner_tr, s_n, mu_in, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

        def _collect_in(tr_sub: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            sf_l, ss_l, ti_l, ni_l, z_l, u_l, y_l = [], [], [], [], [], [], []
            for ti, tr in enumerate(tr_sub):
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors_in, s_n, n_canon)
                p_nl = m01._predict_m35_01_coef(tr, s_n, st01_in)
                r_nl = (c_norm - p_nl).astype(np.float32)
                sf, ss = _extract_sm_fast_slow(tr, s_n, mu_in)
                sf_l.append(sf)
                ss_l.append(ss)
                idx = np.where(rval)[0]
                if idx.size > 0:
                    ti_l.append(np.full(idx.size, ti, dtype=np.int32))
                    ni_l.append(idx.astype(np.int32))
                    z_l.append(extract_circuit_slow_state_features(tr, canon, s_n, mu_in, pca_in, kappa=kappa)[idx])
                    u_l.append(_extract_u_stim(tr, s_n, priors_in, n_canon)[idx])
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

        fc_entry: Dict[str, Any] = {
            "fold_idx": fold_idx,
            "s_n": s_n,
            "mu_train": mu_out,
            "b_dim": int(priors_out["b_dim"]),
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
            "anc_va_list": anc_va,
            "pdel_va_list": pdel_va,
            "phi_va_list": phi_va,
            "u_va_cat": np.vstack(u_va).astype(np.float32),
            "val_trials": val_trials,
            "z_all_low": z_all_low,
            "z_va_low_cat": np.vstack(z_va_low).astype(np.float32),
            "in_tr": in_tr_c,
            "in_va": in_va_c,
        }

        fold_caches.append(fc_entry)
        val_trials_all.extend(val_trials)
        val_sn_all.extend([s_n] * len(val_trials))
        del fc_m3, m01, shared_01, train_trials
        gc.collect()

    ev_anc = evaluate_predictions(
        val_trials_all, anchor_preds_all, val_sn_all,
        functional_tiers=tiers, compute_bootstrap=False, seed=PRIMARY_SEED,
    )
    anchor_ed = float(ev_anc["E_d"])

    ds_cache = {
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
    return ds_cache, anchor_ed


def predict_prior_delta_trajectory(trial: Mapping[str, Any], priors: Mapping[str, Any], n_canon: int):
    from src.features import predict_prior_delta_trajectory as _pred
    return _pred(trial, priors, n_canon)


def run_dataset_k_nulls(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    out_dir: Path,
    k_null: int,
    n_jobs: int,
    seed: int,
    device: str,
    resume: bool,
    max_folds: Optional[int],
    null_bank_path: Path,
) -> Dict[str, Any]:
    """Run real connectome + K constrained null-graph M35_02 Patch-FiLM retrains for one Dev dataset."""
    t_ds = time.time()
    torch.set_num_threads(1)

    print(f"[K{k_null}-PatchFiLM:{dataset_id}] Building frozen Dev M35_01 anchor cache (allow_vault_unseal=False)...", flush=True)
    ds_cache, anchor_ed = _build_dev_anchored_cache(
        dataset_id, cfg, data_root, exp_root, max_folds=max_folds
    )
    canon = ds_cache["canonical_neurons"]
    hp = copy.deepcopy(cfg["hyperparameters"])

    # Load or generate constrained null connectome bank (degree, 3D distance, cell-type matched)
    gen = ConstrainedNullConnectomeGenerator(canon, data_root=data_root)
    bank_dir = null_bank_path.parent
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_npy_path, constraint_report = ensure_null_bank(
        gen, k_null=k_null, scratch_dir=bank_dir, resume=resume
    )
    w_real = gen.real_stack.astype(np.float32)

    ds_ckpt_dir = out_dir / "checkpoints" / dataset_id
    ds_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real connectome k=0 evaluation
    real_ckpt = ds_ckpt_dir / "real_k000.json"
    if resume and real_ckpt.exists():
        real_rec = json.loads(real_ckpt.read_text(encoding="utf-8"))
        print(f"[K{k_null}-PatchFiLM:{dataset_id}] Resumed k=0 real E_d={real_rec['E_d']:.6f}", flush=True)
    else:
        t0 = time.time()
        real_res = _retrain_m35_02_film_on_graph(ds_cache, w_real, hp, seed=seed)
        real_rec = {
            "k": 0,
            "E_d": float(real_res["E_d"]),
            "post_R2": float(real_res["post_R2"]),
            "wall_sec": round(time.time() - t0, 3),
            "completed_utc": _utc_now(),
        }
        _atomic_write_json(real_ckpt, real_rec)
        print(
            f"[K{k_null}-PatchFiLM:{dataset_id}] k=0 real E_d={real_rec['E_d']:.6f} "
            f"(anchor={anchor_ed:.6f}, I_vs_anchor={1.0 - real_rec['E_d']/max(anchor_ed,1e-8):+.4%}, "
            f"wall={real_rec['wall_sec']:.1f}s)",
            flush=True,
        )

    # 2. Identify remaining nulls k = 1..k_null
    completed_records: Dict[int, Dict[str, Any]] = {}
    pending_tasks: List[Tuple[int, np.ndarray, str]] = []
    mm_bank = np.load(bank_npy_path, mmap_mode="r")
    for k in range(1, k_null + 1):
        ckpt_p = ds_ckpt_dir / f"null_k{k:03d}.json"
        if resume and ckpt_p.exists():
            completed_records[k] = json.loads(ckpt_p.read_text(encoding="utf-8"))
        else:
            w_k = np.array(mm_bank[k - 1], dtype=np.float32, copy=True)
            pending_tasks.append((k, w_k, str(ckpt_p)))

    print(
        f"[K{k_null}-PatchFiLM:{dataset_id}] Resumed {len(completed_records)}/{k_null}; "
        f"dispatching {len(pending_tasks)} nulls across n_jobs={n_jobs} (device={device})...",
        flush=True,
    )

    if pending_tasks:
        eff_jobs = max(1, min(int(n_jobs), len(pending_tasks)))
        if eff_jobs == 1:
            _init_worker(ds_cache, hp, seed, device)
            for task in pending_tasks:
                rec = _eval_single_null_worker(task)
                completed_records[rec["k"]] = rec
        else:
            ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
            with ctx.Pool(
                processes=eff_jobs,
                initializer=_init_worker,
                initargs=(ds_cache, hp, seed, device),
            ) as pool:
                for idx_done, rec in enumerate(pool.imap_unordered(_eval_single_null_worker, pending_tasks), start=1):
                    completed_records[rec["k"]] = rec
                    if idx_done % max(1, min(10, len(pending_tasks))) == 0 or idx_done == len(pending_tasks):
                        print(
                            f"  [K{k_null}-PatchFiLM:{dataset_id}] Completed {len(completed_records)}/{k_null} "
                            f"(last k={rec['k']:03d}, E_d={rec['E_d']:.6f}, wall={rec['wall_sec']:.1f}s)",
                            flush=True,
                        )

    null_eds = [float(completed_records[k]["E_d"]) for k in range(1, k_null + 1)]
    null_walls = [float(completed_records[k]["wall_sec"]) for k in range(1, k_null + 1)]
    stats = _compute_null_stats(real_rec["E_d"], null_eds, anchor_ed)

    # Explicit I_null >= I_real verification
    real_i = 1.0 - float(real_rec["E_d"]) / max(anchor_ed, 1e-8)
    null_is = [1.0 - e / max(anchor_ed, 1e-8) for e in null_eds]
    k_i_ge_real = int(sum(1 for x in null_is if x >= real_i))
    p_i = float((1 + k_i_ge_real) / (1 + k_null))
    stats["k_I_null_ge_I_real"] = k_i_ge_real
    stats["p_empirical_I_ge_real"] = round(p_i, 6)

    del ds_cache, mm_bank, gen
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "anchor_M35_01_E_d": round(anchor_ed, 6),
        "real_M35_02_E_d": round(float(real_rec["E_d"]), 6),
        "real_M35_02_post_R2": round(float(real_rec["post_R2"]), 6),
        "stats": stats,
        "null_E_d_array": [round(x, 6) for x in null_eds],
        "null_I_vs_anchor_array": [round(x, 6) for x in null_is],
        "per_null_wall_sec_mean": round(float(np.mean(null_walls)) if null_walls else 0.0, 3),
        "dataset_wall_sec": round(time.time() - t_ds, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }


def parse_args() -> argparse.Namespace:
    default_jobs = min(8, os.cpu_count() or 2)
    p = argparse.ArgumentParser(
        description="Full K_film=199 CPU-only PyTorch 2.8 M35_02 Patch-FiLM null-graph retraining (Dev splits only)."
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu"], help="Execution device (CPU-only).")
    p.add_argument("--n-jobs", type=int, default=default_jobs, help=f"Number of CPU worker processes (default: {default_jobs}).")
    p.add_argument("--k-null", type=int, default=199, help="Number of constrained randomized null graphs (default: 199).")
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS_ALL3),
        help="Comma-separated Dev dataset IDs (default: yemini2021,seyedolmohadesin2025,toyoshima2024).",
    )
    p.add_argument("--seed", type=int, default=PRIMARY_SEED, help="Random seed for Patch-FiLM training.")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT), help="Path to data/ root.")
    p.add_argument("--exp-root", type=str, default=str(EXP_ROOT), help="Path to experiments/autoresearch_v1/.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(EXP_ROOT / "results" / "beyond_2vcpu_k199_patch_film_nulls"),
        help="Directory for atomic per-null checkpoints and final JSON summary.",
    )
    p.add_argument("--resume", action="store_true", help="Resume from existing per-null JSON checkpoints.")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast end-to-end verification (yemini2021, 1 Dev fold, K=2 nulls, max_epochs=3, <45s on 2 vCPUs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    k_null = int(args.k_null)
    max_folds: Optional[int] = None

    if args.smoke_test:
        datasets = ["seyedolmohadesin2025"]
        k_null = 2
        max_folds = 1
        cfg = copy.deepcopy(cfg)
        cfg["hyperparameters"]["max_epochs"] = 4
        cfg["hyperparameters"]["early_stopping_patience"] = 2
        if args.output_dir == str(EXP_ROOT / "results" / "beyond_2vcpu_k199_patch_film_nulls"):
            env_tmp = os.environ.get("TMPDIR", str(out_dir))
            out_dir = Path(env_tmp) / "smoke_k199_patch_film"
            out_dir.mkdir(parents=True, exist_ok=True)

    null_bank_path = out_dir / f"constrained_null_bank_k{k_null}.npz"

    print(
        f"=== Beyond-2-vCPU Script 1: M35_02 Patch-FiLM Null Retraining (K={k_null}) ===\n"
        f"torch={torch.__version__} | device={args.device} | n_jobs={args.n_jobs} | "
        f"datasets={datasets} | smoke_test={args.smoke_test} | vault_access=FORBIDDEN",
        flush=True,
    )

    per_dataset: Dict[str, Any] = {}
    for ds in datasets:
        per_dataset[ds] = run_dataset_k_nulls(
            dataset_id=ds,
            cfg=cfg,
            data_root=data_root,
            exp_root=exp_root,
            out_dir=out_dir,
            k_null=k_null,
            n_jobs=args.n_jobs,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            max_folds=max_folds,
            null_bank_path=null_bank_path,
        )

    # Macro summaries across available datasets
    def _macro_summary(ds_subset: Sequence[str]) -> Optional[Dict[str, Any]]:
        if not all(d in per_dataset for d in ds_subset):
            return None
        real_macro = float(np.mean([per_dataset[d]["stats"]["real_gain_vs_M35_01"] for d in ds_subset]))
        null_macro_arr = np.array(
            [
                float(np.mean([per_dataset[d]["null_I_vs_anchor_array"][k] for d in ds_subset]))
                for k in range(k_null)
            ],
            dtype=np.float64,
        )
        k_ge = int(np.sum(null_macro_arr >= real_macro))
        p_emp = float((1 + k_ge) / (1 + k_null))
        sd = float(np.std(null_macro_arr, ddof=1)) if k_null > 1 else 0.0
        z = float((real_macro - float(np.mean(null_macro_arr))) / sd) if sd > 1e-12 else 0.0
        return {
            "datasets": list(ds_subset),
            "k_null": k_null,
            "real_I_macro_vs_M35_01": round(real_macro, 6),
            "null_I_macro_vs_M35_01_mean": round(float(np.mean(null_macro_arr)), 6),
            "null_I_macro_vs_M35_01_std": round(sd, 6),
            "k_null_ge_real": k_ge,
            "p_empirical_onesided": round(p_emp, 6),
            "z_score": round(z, 4),
        }

    summary = {
        "script": "run_full_k199_patch_film_nulls.py",
        "generated_utc": _utc_now(),
        "torch_version": torch.__version__,
        "device": args.device,
        "n_jobs": int(args.n_jobs),
        "k_null": k_null,
        "seed": int(args.seed),
        "smoke_test": bool(args.smoke_test),
        "zero_vault_access": True,
        "zero_dataset_id_branches": True,
        "config_id": cfg["config_id"],
        "per_dataset": per_dataset,
        "macro_main2": _macro_summary(DATASETS_MAIN2),
        "macro_all3": _macro_summary(DATASETS_ALL3),
        "total_wall_sec": round(time.time() - t_start, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }

    summary_path = out_dir / f"patch_film_null_k{k_null}_summary.json"
    _atomic_write_json(summary_path, summary)
    print(
        f"[DONE] Saved summary to {summary_path} | total_wall_sec={summary['total_wall_sec']}s | "
        f"peak_rss_mb={summary['peak_rss_mb']} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
