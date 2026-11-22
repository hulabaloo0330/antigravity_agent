"""Standalone read-only scorer CLI (`evaluators/scorer.py`) implementing `PROJECT.md` Contract 2 & 3.

Computes per-neuron scale `s_n = max(std_train_n, s_floor)` directly from canonical training-fold
animal traces on disk (never trusting model-supplied scales!), and emits `E_d`, `L_w`, `I_d`, `I_macro`,
`post_R2`, `delta_R2`, separate animal-bootstrap CIs, `rmse_raw`, `n_worms_improved`, `n_worms_total`,
and cell-class / stimulus / sex / temporal-horizon breakdowns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data_adapters import DATA_ROOT_DEFAULT, load_dataset
from src.evaluator import (
    align_records_to_canonical,
    apply_sensor_bounds_to_aligned,
    compare_against_reference,
    compute_training_fold_neuron_scales,
    compute_training_fold_sensor_bounds,
    evaluate_predictions,
    extract_causal_trial_records,
)


def _load_protocol_windows(exp_root: Path) -> Dict[str, float]:
    prot_path = exp_root / "protocol.json"
    if prot_path.exists():
        p = json.loads(prot_path.read_text(encoding="utf-8"))
        in_w = p.get("input_window_sec", [-10.0, 0.0])
        tg_w = p.get("target_window_sec", [0.0, 10.0])
        cal_w = p.get("calibration_window_sec", [-30.0, -10.0])
        return {
            "input_sec": float(abs(in_w[1] - in_w[0])),
            "target_sec": float(abs(tg_w[1] - tg_w[0])),
            "calibration_sec": float(abs(cal_w[1] - cal_w[0])),
            "floor_percentile": float(p.get("scale_floor_percentile", 10.0)),
            "fallback_floor": float(p.get("scale_floor_fallback", 1e-6)),
        }
    return {
        "input_sec": 10.0,
        "target_sec": 10.0,
        "calibration_sec": 20.0,
        "floor_percentile": 10.0,
        "fallback_floor": 1e-6,
    }


def _resolve_default_ref_pred(dataset_id: str, split: str, exp_root: Path) -> Optional[Path]:
    reg_path = exp_root / "baseline_registry.json"
    if not reg_path.exists():
        return None
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        ref_info = reg.get("reference_baseline", {})
        pred_map = ref_info.get("prediction_files", {}).get(split, {})
        if dataset_id in pred_map:
            cand = Path(pred_map[dataset_id])
            if not cand.is_absolute():
                cand = exp_root / cand
            if cand.exists():
                return cand
    except Exception:
        return None
    return None


def score_saved_predictions_file(
    pred_path: Path,
    split: str = "dev",
    ref_pred_path: Path | None = None,
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT,
    allow_vault_unseal: bool = False,
    resolve_registry_ref: bool = True,
) -> Dict[str, Any]:
    """Read saved `predictions_*.npz` (or run directory) and score strictly against canonical ground-truth traces."""
    pred_path = Path(pred_path)
    if pred_path.is_dir():
        return score_run_directory(
            run_dir=pred_path,
            split=split,
            ref_run_dir=ref_pred_path,
            data_root=data_root,
            exp_root=exp_root,
            allow_vault_unseal=allow_vault_unseal,
        )

    win_cfg = _load_protocol_windows(exp_root)
    dataset_id: Optional[str] = None

    with np.load(pred_path, allow_pickle=False) as z:
        if "meta_json" in z.files:
            payload = json.loads(str(z["meta_json"].item()))
            trials_meta = payload["trials_meta"]
            train_fold_sessions = payload["train_fold_session_animal_ids"]
            flat_y_pred = z["flat_y_pred"].astype(np.float32)
            offsets = z["offsets"].astype(np.int64)
            if not trials_meta:
                raise ValueError(f"Empty trials_meta in {pred_path}")
            dataset_id = str(trials_meta[0]["dataset_id"])

            raw_recs = load_dataset(
                dataset_id=dataset_id,
                split=split,
                allow_vault_unseal=allow_vault_unseal,
                data_root=data_root,
                exp_root=exp_root,
                require_final_manifest_for_vault=(split != "dev"),
                caller_reason="evaluators.scorer",
            )
            dev_recs = (
                raw_recs
                if split == "dev"
                else load_dataset(
                    dataset_id=dataset_id,
                    split="dev",
                    allow_vault_unseal=False,
                    data_root=data_root,
                    exp_root=exp_root,
                )
            )
            aligned_eval, canon_neurons, tiers = align_records_to_canonical(raw_recs)
            aligned_dev, _, _ = align_records_to_canonical(dev_recs, canonical_neurons=canon_neurons)
            dev_by_session = {r.animal_id: r for r in aligned_dev}

            fold_scales_cache: Dict[int, np.ndarray] = {}
            fold_trial_lookup: Dict[int, Dict[str, Dict[str, Any]]] = {}
            for f_idx, sess_list in enumerate(train_fold_sessions):
                raw_tr_list = [dev_by_session[s].traces for s in sess_list if s in dev_by_session]
                raw_mk_list = [dev_by_session[s].obs_mask for s in sess_list if s in dev_by_session]
                lo_b, hi_b = compute_training_fold_sensor_bounds(raw_tr_list, raw_mk_list)
                bounded_tr_recs = [
                    apply_sensor_bounds_to_aligned(dev_by_session[s], lo_b, hi_b)
                    for s in sess_list
                    if s in dev_by_session
                ]
                s_n, _, _ = compute_training_fold_neuron_scales(
                    [r.traces for r in bounded_tr_recs],
                    [r.obs_mask for r in bounded_tr_recs],
                    floor_percentile=win_cfg["floor_percentile"],
                    fallback_floor=win_cfg["fallback_floor"],
                )
                fold_scales_cache[f_idx] = s_n
                f_tmap: Dict[str, Dict[str, Any]] = {}
                for ar in aligned_eval:
                    ar_b = apply_sensor_bounds_to_aligned(ar, lo_b, hi_b)
                    for tr in extract_causal_trial_records(
                        ar_b,
                        input_sec=win_cfg["input_sec"],
                        target_sec=win_cfg["target_sec"],
                        calibration_sec=win_cfg["calibration_sec"],
                        include_synthetic_buffer_windows=True,
                    ):
                        f_tmap[tr["trial_id"]] = tr
                fold_trial_lookup[f_idx] = f_tmap

            trials_ordered: List[Dict[str, Any]] = []
            y_preds_ordered: List[np.ndarray] = []
            fold_sn_ordered: List[np.ndarray] = []
            for idx_m, mrow in enumerate(trials_meta):
                f_i = int(mrow["fold_idx"])
                tid = str(mrow["trial_id"])
                if tid not in fold_trial_lookup[f_i]:
                    raise KeyError(f"Trial {tid} not found in canonical {dataset_id}/{split} ground truth")
                gt_tr = fold_trial_lookup[f_i][tid]
                vm = gt_tr["valid_mask"]
                seg = flat_y_pred[offsets[idx_m] : offsets[idx_m + 1]]
                if int(np.sum(vm)) != int(seg.size):
                    raise ValueError(f"Valid mask count mismatch on {tid}: {np.sum(vm)} vs {seg.size}")
                yp_mat = np.zeros_like(gt_tr["y_true"], dtype=np.float32)
                yp_mat[vm] = seg
                trials_ordered.append(gt_tr)
                y_preds_ordered.append(yp_mat)
                fold_sn_ordered.append(fold_scales_cache[f_i])

            metrics = evaluate_predictions(
                trials=trials_ordered,
                y_preds=y_preds_ordered,
                fold_s_n_list=fold_sn_ordered,
                functional_tiers=tiers,
                compute_bootstrap=True,
            )
            metrics["dataset_id"] = dataset_id
            metrics["split"] = split
        else:
            # Direct dense array prediction bundle (used in unit/CLI standalone tests)
            y_pred_all = z["y_pred"].astype(np.float32)
            y_true_all = z["y_true"].astype(np.float32)
            valid_all = z["valid_mask"].astype(bool)
            animal_ids = [str(x) for x in z["animal_id"].tolist()]
            stim_ids = [str(x) for x in z["stimulus_id"].tolist()]
            is_bufs = (
                z["is_buffer_control"].astype(bool).tolist()
                if "is_buffer_control" in z.files
                else [False] * len(animal_ids)
            )
            y_pre = (
                z["y_pre_anchor"].astype(np.float32)
                if "y_pre_anchor" in z.files
                else np.zeros((len(animal_ids), y_pred_all.shape[1]), dtype=np.float32)
            )
            tiers_dense = (
                [str(x) for x in z["functional_tiers"].tolist()]
                if "functional_tiers" in z.files
                else None
            )
            if "train_traces" in z.files and "train_masks" in z.files:
                s_n, _, _ = compute_training_fold_neuron_scales(
                    list(z["train_traces"]),
                    list(z["train_masks"]),
                    floor_percentile=win_cfg["floor_percentile"],
                    fallback_floor=win_cfg["fallback_floor"],
                )
            else:
                s_n, _, _ = compute_training_fold_neuron_scales(
                    [y_true_all[:, :, 0]],
                    [np.any(valid_all, axis=2)],
                    floor_percentile=win_cfg["floor_percentile"],
                    fallback_floor=win_cfg["fallback_floor"],
                )
            trials_ordered = []
            y_preds_ordered = []
            fold_sn_ordered = []
            for i in range(len(animal_ids)):
                trials_ordered.append(
                    {
                        "animal_id": animal_ids[i],
                        "stimulus_id": stim_ids[i],
                        "is_buffer_control": bool(is_bufs[i]),
                        "y_true": y_true_all[i],
                        "valid_mask": valid_all[i],
                        "y_pre_anchor": y_pre[i],
                    }
                )
                y_preds_ordered.append(y_pred_all[i])
                fold_sn_ordered.append(s_n)
            metrics = evaluate_predictions(
                trials_ordered,
                y_preds_ordered,
                fold_sn_ordered,
                functional_tiers=tiers_dense,
                compute_bootstrap=True,
            )
            metrics["split"] = split

    eff_ref = ref_pred_path
    if eff_ref is None and resolve_registry_ref and dataset_id is not None:
        eff_ref = _resolve_default_ref_pred(dataset_id=dataset_id, split=split, exp_root=exp_root)

    if eff_ref is not None and Path(eff_ref).exists():
        ref_metrics = score_saved_predictions_file(
            Path(eff_ref),
            split=split,
            ref_pred_path=None,
            data_root=data_root,
            exp_root=exp_root,
            allow_vault_unseal=allow_vault_unseal,
            resolve_registry_ref=False,
        )
        comp = compare_against_reference(metrics, ref_metrics)
        metrics.update(
            {
                "I_d": comp["I_d"],
                "I_d_ci95": comp["I_d_ci95"],
                "E_d_reference": comp["E_d_reference"],
                "n_worms_improved": comp["n_worms_improved"],
                "n_worms_total": comp["n_worms_total"],
                "fraction_worms_improved": comp["fraction_worms_improved"],
            }
        )
    else:
        metrics.setdefault("I_d", None)
        metrics.setdefault("I_d_ci95", None)
        metrics.setdefault("n_worms_improved", None)

    # If sibling main-source prediction files exist in the same run folder and reference_baseline is registered, compute I_macro
    if resolve_registry_ref and dataset_id is not None:
        main_sources = ["seyedolmohadesin2025", "toyoshima2024"]
        sib_gains: List[float] = []
        for m_ds in main_sources:
            sib_p = pred_path.parent / f"predictions_{split}_{m_ds}.npz"
            if m_ds == dataset_id and metrics.get("I_d") is not None:
                sib_gains.append(float(metrics["I_d"]))
            elif sib_p.exists():
                m_ref = _resolve_default_ref_pred(dataset_id=m_ds, split=split, exp_root=exp_root)
                if m_ref is not None and m_ref.exists():
                    sib_m = score_saved_predictions_file(
                        sib_p,
                        split=split,
                        ref_pred_path=m_ref,
                        data_root=data_root,
                        exp_root=exp_root,
                        allow_vault_unseal=allow_vault_unseal,
                        resolve_registry_ref=False,
                    )
                    if sib_m.get("I_d") is not None:
                        sib_gains.append(float(sib_m["I_d"]))
        metrics["I_macro"] = float(np.mean(sib_gains)) if len(sib_gains) == len(main_sources) else None
    else:
        metrics.setdefault("I_macro", None)

    return metrics


def score_run_directory(
    run_dir: Path,
    split: str = "dev",
    ref_run_dir: Path | None = None,
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT,
    allow_vault_unseal: bool = False,
) -> Dict[str, Any]:
    """Score all `predictions_<split>_*.npz` inside a run directory and aggregate `I_macro`."""
    npz_files = sorted(run_dir.glob(f"predictions_{split}_*.npz"))
    if not npz_files:
        single = run_dir / f"predictions_{split}.npz"
        if single.exists():
            return score_saved_predictions_file(
                single,
                split=split,
                ref_pred_path=ref_run_dir,
                data_root=data_root,
                exp_root=exp_root,
                allow_vault_unseal=allow_vault_unseal,
            )
        raise FileNotFoundError(f"No predictions_{split}_*.npz found in {run_dir}")

    by_ds: Dict[str, Any] = {}
    for p in npz_files:
        ref_p = None
        if ref_run_dir is not None:
            cand = Path(ref_run_dir) / p.name if Path(ref_run_dir).is_dir() else Path(ref_run_dir)
            if cand.exists():
                ref_p = cand
        res = score_saved_predictions_file(
            p,
            split=split,
            ref_pred_path=ref_p,
            data_root=data_root,
            exp_root=exp_root,
            allow_vault_unseal=allow_vault_unseal,
            resolve_registry_ref=(ref_p is None),
        )
        ds = str(res.get("dataset_id", p.stem.replace(f"predictions_{split}_", "")))
        by_ds[ds] = res

    main_sources = ["seyedolmohadesin2025", "toyoshima2024"]
    main_gains = [float(by_ds[d]["I_d"]) for d in main_sources if d in by_ds and by_ds[d].get("I_d") is not None]
    i_macro = float(np.mean(main_gains)) if len(main_gains) == len(main_sources) else None
    worst_main = float(np.min(main_gains)) if main_gains else None
    return {
        "run_dir": str(run_dir),
        "split": split,
        "I_macro": i_macro,
        "worst_main_source_I_d": worst_main,
        "datasets": by_ds,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone read-only trajectory scorer")
    parser.add_argument("--pred", required=True, type=Path, help="Path to saved predictions .npz or run directory")
    parser.add_argument("--ref-pred", default=None, type=Path, help="Optional reference baseline predictions .npz or run directory")
    parser.add_argument("--split", default="dev", choices=["dev", "confirmatory", "vault"])
    parser.add_argument("--out", default=None, type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)

    sp = "vault" if args.split in ("confirmatory", "vault") else "dev"
    res = score_saved_predictions_file(
        pred_path=args.pred,
        split=sp,
        ref_pred_path=args.ref_pred,
        allow_vault_unseal=(sp == "vault"),
    )
    out_str = json.dumps(res, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_str, encoding="utf-8")
    print(out_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
