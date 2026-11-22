"""Frozen leak-free causal trajectory evaluator and animal-level bootstrap scorer (`M2`).

Implements `PROJECT.md` Interface Contract 2 and `MODEL_REQUIREMENTS.md` §1, §5, §6:
- Physical-second causal windowing: `[-10.0, 0.0)s` input (`ceil(10*fps)` frames strictly `< onset_idx`),
  optional `[-30.0, -10.0)s` calibration, `[0.0, 10.0)s` post-stimulus trajectory (`floor(10*fps)` frames).
- Strict training-fold-only per-neuron scale `s_n = max(std_train_n, s_floor)` where `s_floor` is the
  10th percentile of strictly positive training neuron standard deviations.
- Equal-per-animal primary loss `L_w` (averaging repeats within each stimulus condition first, then
  equal-weight across non-buffer stimulus conditions in animal `w`), `E_d = mean_w L_w`, `I_d`, `I_macro`.
- Separate animal-cluster bootstrap CIs for `post_R2` (`Var(y_post)` denominator) vs `delta_R2`
  (`Var(y_post - y_pre_mean)` denominator), preventing Historical Trap #3.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.connectome_priors import canonicalize_neuron_name, load_connectome_priors
from src.data_adapters import AnimalRecord


@dataclass
class AlignedAnimal:
    """AnimalRecord aligned to the 310 canonical C. elegans neurons."""

    dataset_id: str
    animal_id: str
    physical_animal_id: str
    sex: str
    modality: str
    date_batch: str
    fps: float
    traces: np.ndarray  # float32 [T, N_canon], NaN where unobserved
    obs_mask: np.ndarray  # bool [T, N_canon]
    observed_neuron_indices: np.ndarray  # int32 indices into canonical_neurons
    stim_events: List[Dict[str, Any]]
    extra_metadata: Dict[str, Any]


def get_physical_animal_id(rec: AnimalRecord) -> str:
    """Return unique physical worm ID, grouping multi-session worms (`MODEL_REQUIREMENTS §1, §6.1`)."""
    phys = rec.extra_metadata.get("physical_animal_id")
    if phys is not None and str(phys).strip():
        return str(phys).strip()
    return str(rec.animal_id)


def compute_causal_window_frames(
    fps: float,
    input_sec: float = 10.0,
    target_sec: float = 10.0,
    calibration_sec: Optional[float] = 20.0,
    fast_anchor_sec: float = 2.0,
) -> Dict[str, int]:
    """Compute exact frame counts from physical seconds (`PROJECT.md` Contract 2)."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    out = {
        "n_input_frames": int(math.ceil(input_sec * fps)),
        "n_target_frames": int(math.floor(target_sec * fps)),
        "n_fast_anchor_frames": max(1, int(math.ceil(fast_anchor_sec * fps))),
    }
    if calibration_sec is not None and calibration_sec > 0.0:
        out["n_calibration_frames"] = int(math.ceil(calibration_sec * fps))
    else:
        out["n_calibration_frames"] = 0
    return out


def align_records_to_canonical(
    records: Sequence[AnimalRecord],
    canonical_neurons: Optional[Sequence[str]] = None,
) -> Tuple[List[AlignedAnimal], List[str], List[str]]:
    """Align `AnimalRecord`s onto the 310 canonical C. elegans neuron vocabulary."""
    cp = load_connectome_priors(canonical_neurons=canonical_neurons)
    canon_list = list(cp["canonical_neurons"])
    tiers = list(cp["functional_tiers"])
    idx_map = {n: i for i, n in enumerate(canon_list)}
    n_canon = len(canon_list)

    aligned: List[AlignedAnimal] = []
    for rec in records:
        t_len = rec.traces_raw.shape[0]
        traces = np.full((t_len, n_canon), np.nan, dtype=np.float32)
        mask = np.zeros((t_len, n_canon), dtype=bool)
        obs_cols: List[int] = []
        for col_j, raw_name in enumerate(rec.neuron_ids):
            cname = canonicalize_neuron_name(raw_name)
            if cname not in idx_map:
                # Non-neuronal ROI (e.g. GLR/HYPL glia/hypodermis in Toyoshima) -> strictly excluded
                continue
            ci = idx_map[cname]
            col_tr = rec.traces_raw[:, col_j]
            col_mk = rec.obs_mask[:, col_j] & np.isfinite(col_tr)
            if np.any(col_mk):
                traces[:, ci] = np.where(col_mk, col_tr, np.nan)
                mask[:, ci] = col_mk
                obs_cols.append(ci)
        aligned.append(
            AlignedAnimal(
                dataset_id=rec.dataset_id,
                animal_id=rec.animal_id,
                physical_animal_id=get_physical_animal_id(rec),
                sex=rec.sex,
                modality=rec.modality,
                date_batch=rec.date_batch,
                fps=float(rec.fps),
                traces=traces,
                obs_mask=mask,
                observed_neuron_indices=np.array(sorted(set(obs_cols)), dtype=np.int32),
                stim_events=list(rec.stim_events),
                extra_metadata=dict(rec.extra_metadata),
            )
        )
    return aligned, canon_list, tiers


def build_grouped_cv_splits(aligned_records: Sequence[AlignedAnimal]) -> List[Tuple[List[int], List[int]]]:
    """Build deterministic leak-free animal/batch-grouped CV folds on Dev animals.

    - Never splits two sessions of the same `physical_animal_id` across train and validation.
    - When `n_physical_animals > 25` and `n_batches >= 5`, groups by `date_batch` so date/chip
      batch confounds cannot leak across folds; otherwise performs Leave-One-Physical-Animal-Out CV.
    """
    n_rec = len(aligned_records)
    phys_ids = [r.physical_animal_id for r in aligned_records]
    unique_phys = sorted(set(phys_ids))
    batches = [r.date_batch for r in aligned_records]
    unique_batches = sorted(set(batches))

    if len(unique_phys) > 25 and len(unique_batches) >= 5:
        # Ensure physical_animal_id never spans batches; if it does, merge groups conservatively
        groups = batches
        unique_groups = unique_batches
    else:
        groups = phys_ids
        unique_groups = unique_phys

    splits: List[Tuple[List[int], List[int]]] = []
    for g in unique_groups:
        val_idx = [i for i in range(n_rec) if groups[i] == g]
        val_phys = {phys_ids[i] for i in val_idx}
        train_idx = [i for i in range(n_rec) if phys_ids[i] not in val_phys]
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))
    return splits


def compute_training_fold_neuron_scales(
    train_traces: Sequence[np.ndarray],
    train_masks: Sequence[np.ndarray],
    floor_percentile: float = 10.0,
    fallback_floor: float = 1e-6,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute training-fold-only `s_n = max(std_train_n, s_floor)` and observation counts."""
    if not train_traces or len(train_traces) != len(train_masks):
        raise ValueError("Non-empty aligned train_traces and train_masks required")

    n_neurons = int(train_traces[0].shape[1])
    pooled: List[List[np.ndarray]] = [[] for _ in range(n_neurons)]
    obs_worm_counts = np.zeros(n_neurons, dtype=np.int32)

    for tr, mk in zip(train_traces, train_masks):
        arr = np.asarray(tr, dtype=np.float64)
        msk = np.asarray(mk, dtype=bool)
        for n_i in range(n_neurons):
            vals = arr[msk[:, n_i], n_i]
            if vals.size > 0:
                pooled[n_i].append(vals)
                obs_worm_counts[n_i] += 1

    std_train = np.zeros(n_neurons, dtype=np.float64)
    for n_i in range(n_neurons):
        if pooled[n_i]:
            cat = np.concatenate(pooled[n_i], axis=0)
            if cat.size >= 2:
                std_train[n_i] = float(np.std(cat, ddof=0))

    pos_stds = std_train[std_train > 1e-12]
    if pos_stds.size > 0:
        s_floor = max(float(np.percentile(pos_stds, floor_percentile)), fallback_floor)
    else:
        s_floor = float(fallback_floor)

    s_n = np.maximum(std_train, s_floor)
    return s_n, s_floor, obs_worm_counts


def extract_causal_trial_records(
    rec: AlignedAnimal,
    input_sec: float = 10.0,
    target_sec: float = 10.0,
    calibration_sec: float = 20.0,
    fast_anchor_sec: float = 2.0,
    include_synthetic_buffer_windows: bool = False,
) -> List[Dict[str, Any]]:
    """Extract strictly causal pre-stimulus inputs `[-10, 0)s` and post-stimulus targets `[0, 10)s`.

    CRITICAL CAUSALITY INVARIANT:
    `x_pre`, `m_pre`, `x_cal`, `m_cal`, and `y_pre_mean` read ONLY indices `< onset_idx`.
    Tampering `rec.traces[onset_idx:]` with arbitrary values (`+1e6`, `-999`, `NaN`) leaves
    every causal input array bit-for-bit identical!
    """
    win = compute_causal_window_frames(
        rec.fps,
        input_sec=input_sec,
        target_sec=target_sec,
        calibration_sec=calibration_sec,
        fast_anchor_sec=fast_anchor_sec,
    )
    n_in = win["n_input_frames"]
    n_out = win["n_target_frames"]
    n_cal = win["n_calibration_frames"]
    n_fast = win["n_fast_anchor_frames"]
    t_total, n_canon = rec.traces.shape

    events_to_extract = list(rec.stim_events)
    if include_synthetic_buffer_windows:
        # Also extract genuine pre-stimulus quiet baseline pseudo-events strictly >=15s before stimulus onsets
        has_native_ctrl = any(
            bool(ev.get("is_control", False) or ev.get("is_buffer_control", False))
            for ev in events_to_extract
        )
        if not has_native_ctrl:
            occupied = [(int(e["onset_idx"]), int(e.get("offset_idx", e["onset_idx"] + n_out))) for e in events_to_extract]
            cand_on = n_in + max(1, int(round(5.0 * rec.fps)))
            step_f = max(n_in + n_out, int(round(25.0 * rec.fps)))
            b_ord = 0
            while cand_on + n_out <= t_total and b_ord < 4:
                overlaps = any(not (cand_on + n_out <= s0 - 2 or cand_on >= s1 + int(round(10.0 * rec.fps))) for s0, s1 in occupied)
                if not overlaps and cand_on >= n_in:
                    b_ord += 1
                    events_to_extract.append(
                        {
                            "stim_id": "pseudo_buffer_45s",
                            "onset_idx": cand_on,
                            "offset_idx": cand_on + n_out,
                            "onset_sec": float(cand_on / rec.fps),
                            "offset_sec": float((cand_on + n_out) / rec.fps),
                            "trial_order": b_ord,
                            "is_control": True,
                            "is_buffer_control": True,
                        }
                    )
                cand_on += step_f

    trials: List[Dict[str, Any]] = []
    seen_stim_counts: Dict[str, int] = {}

    for ev_i, ev in enumerate(events_to_extract):
        onset = int(ev["onset_idx"])
        offset = int(ev.get("offset_idx", onset + n_out))
        if onset < n_in or onset + n_out > t_total:
            continue

        stim_id = str(ev.get("stim_id", ev.get("stimulus_id", "unknown")))
        is_buf = bool(ev.get("is_buffer_control", False) or ev.get("is_control", False))
        seen_stim_counts[stim_id] = seen_stim_counts.get(stim_id, 0) + 1
        repeat_idx = int(ev.get("round_idx", seen_stim_counts[stim_id]))
        prev_stim = str(events_to_extract[ev_i - 1].get("stim_id", "none")) if ev_i > 0 else "none"

        # STRICTLY CAUSAL SLICES (< onset)
        x_pre = np.array(rec.traces[onset - n_in : onset, :], dtype=np.float32, copy=True)
        m_pre = np.array(rec.obs_mask[onset - n_in : onset, :], dtype=bool, copy=True)

        cal_start = max(0, onset - n_in - n_cal)
        x_cal = np.array(rec.traces[cal_start : onset - n_in, :], dtype=np.float32, copy=True)
        m_cal = np.array(rec.obs_mask[cal_start : onset - n_in, :], dtype=bool, copy=True)

        # Causal pre-stimulus anchor per neuron ([-2s, 0)s with fallback to [-10s, 0)s)
        x_fast = x_pre[-n_fast:, :]
        m_fast = m_pre[-n_fast:, :]
        y_pre_anchor = np.full(n_canon, np.nan, dtype=np.float32)
        y_pre_10s_mean = np.full(n_canon, np.nan, dtype=np.float32)
        has_pre = np.any(m_pre, axis=0)
        has_fast = np.any(m_fast, axis=0)

        for n_i in np.where(has_pre)[0]:
            vals_10 = x_pre[m_pre[:, n_i], n_i]
            y_pre_10s_mean[n_i] = float(np.mean(vals_10))
            if has_fast[n_i]:
                vals_f = x_fast[m_fast[:, n_i], n_i]
                y_pre_anchor[n_i] = float(np.mean(vals_f))
            else:
                y_pre_anchor[n_i] = y_pre_10s_mean[n_i]

        # Post-stimulus target [onset : onset + n_out] -> transposed to [N_canon, T_out]
        y_post_raw = np.array(rec.traces[onset : onset + n_out, :].T, dtype=np.float32, copy=True)
        m_post_raw = np.array(rec.obs_mask[onset : onset + n_out, :].T, dtype=bool, copy=True)
        # Valid evaluation targets require both finite post observation AND causal pre-anchor availability
        valid_mask = m_post_raw & has_pre[:, None]

        # Known stimulus schedule waveform u_post [T_out] in [0, 1] (1 during pulse [onset, offset), 0 after offset)
        u_post = np.zeros(n_out, dtype=np.float32)
        pulse_frames = max(1, min(n_out, offset - onset))
        u_post[:pulse_frames] = 1.0

        trials.append(
            {
                "dataset_id": rec.dataset_id,
                "session_animal_id": rec.animal_id,
                "animal_id": rec.physical_animal_id,  # Physical worm grouping unit!
                "sex": rec.sex,
                "date_batch": rec.date_batch,
                "fps": rec.fps,
                "trial_id": f"{rec.animal_id}::ev{ev_i:02d}",
                "trial_order": int(ev.get("trial_order", ev_i + 1)),
                "repeat_idx": repeat_idx,
                "prev_stim_id": prev_stim,
                "onset_idx": onset,
                "onset_sec": float(ev.get("onset_sec", onset / rec.fps)),
                "stimulus_id": stim_id,
                "is_buffer_control": is_buf,
                "u_post": u_post,
                "x_pre": x_pre,
                "m_pre": m_pre,
                "x_cal": x_cal,
                "m_cal": m_cal,
                "has_pre": has_pre,
                "y_pre_anchor": y_pre_anchor,
                "y_pre_mean": y_pre_10s_mean,
                "y_true": y_post_raw,
                "valid_mask": valid_mask,
            }
        )
    return trials


def compute_training_fold_sensor_bounds(
    train_traces: Sequence[np.ndarray],
    train_masks: Sequence[np.ndarray],
    iqr_mult: float = 15.0,
) -> Tuple[float, float]:
    """Compute strictly training-fold physiological trace bounds `[lo, hi]` to reject tracking / `CFP->0` spikes."""
    chunks = [tr[mk] for tr, mk in zip(train_traces, train_masks) if np.any(mk)]
    if not chunks:
        return -1.0e6, 1.0e6
    vals = np.concatenate(chunks).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        return -1.0e6, 1.0e6
    q01, q25, q75, q999 = np.percentile(vals, [0.1, 25.0, 75.0, 99.9])
    iqr = max(float(q75 - q25), 1e-3)
    return float(q01 - iqr_mult * iqr), float(q999 + iqr_mult * iqr)


def apply_sensor_bounds_to_aligned(
    rec: AlignedAnimal,
    lo_bound: float,
    hi_bound: float,
) -> AlignedAnimal:
    """Mask instantaneous sensor glitches outside training-fold `[lo_bound, hi_bound]` identically for all models."""
    in_bounds = rec.obs_mask & (rec.traces >= lo_bound) & (rec.traces <= hi_bound)
    if np.array_equal(in_bounds, rec.obs_mask):
        return rec
    clean_traces = np.where(in_bounds, rec.traces, np.nan).astype(np.float32)
    obs_idx = np.where(np.any(in_bounds, axis=0))[0]
    return AlignedAnimal(
        dataset_id=rec.dataset_id,
        animal_id=rec.animal_id,
        physical_animal_id=rec.physical_animal_id,
        sex=rec.sex,
        modality=rec.modality,
        date_batch=rec.date_batch,
        fps=rec.fps,
        traces=clean_traces,
        obs_mask=in_bounds,
        observed_neuron_indices=obs_idx,
        stim_events=rec.stim_events,
        extra_metadata=rec.extra_metadata,
    )


def evaluate_predictions(
    trials: Sequence[Mapping[str, Any]],
    y_preds: Sequence[np.ndarray],
    fold_s_n_list: Sequence[np.ndarray],
    functional_tiers: Optional[Sequence[str]] = None,
    compute_bootstrap: bool = True,
    n_boot: int = 500,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Evaluate out-of-fold trajectory predictions strictly per `MODEL_REQUIREMENTS.md §5`.

    Each trial uses the `s_n` vector computed EXCLUSIVELY on its own training fold!
    """
    if len(trials) != len(y_preds) or len(trials) != len(fold_s_n_list):
        raise ValueError("Aligned trials, y_preds, and fold_s_n_list required")

    by_worm_cond: Dict[str, Dict[str, List[float]]] = {}
    early_by_worm_cond: Dict[str, Dict[str, List[float]]] = {}
    late_by_worm_cond: Dict[str, Dict[str, List[float]]] = {}
    buffer_by_worm: Dict[str, List[float]] = {}
    sex_by_worm: Dict[str, str] = {}
    stim_errors: Dict[str, List[float]] = {}
    class_errors: Dict[str, List[float]] = {}
    worm_class_errors: Dict[str, Dict[str, List[float]]] = {}

    total_raw_sq_err = 0.0
    total_nonbuf_pts = 0
    ref_center_y: Optional[float] = None
    ref_center_dy: Optional[float] = None
    w_cnt: Dict[str, int] = {}
    w_res_sq: Dict[str, float] = {}
    w_sum_yc: Dict[str, float] = {}
    w_sum_yc2: Dict[str, float] = {}
    w_sum_dyc: Dict[str, float] = {}
    w_sum_dyc2: Dict[str, float] = {}

    tier_masks: Dict[str, np.ndarray] = {}
    if functional_tiers is not None:
        ft_arr = np.asarray([str(t) for t in functional_tiers])
        for u_t in sorted(set(ft_arr.tolist())):
            tier_masks[u_t] = ft_arr == u_t
        cmd_m = (ft_arr == "premotor_command") | (ft_arr == "motor") | (ft_arr == "command_motor")
        if np.any(cmd_m):
            tier_masks["command_motor"] = cmd_m

    total_points = 0
    valid_points = 0
    skipped_trials = 0

    for tr, y_hat_raw, s_n_raw in zip(trials, y_preds, fold_s_n_list):
        worm_id = str(tr["animal_id"])
        stim_id = str(tr["stimulus_id"])
        is_buf = bool(tr.get("is_buffer_control", False))
        sex_by_worm[worm_id] = str(tr.get("sex", "unknown"))

        y_true = np.asarray(tr["y_true"], dtype=np.float64)
        y_pred = np.asarray(y_hat_raw, dtype=np.float64)
        vmask = np.asarray(tr["valid_mask"], dtype=bool)
        scales = np.asarray(s_n_raw, dtype=np.float64)

        total_points += int(vmask.size)
        n_val = int(np.sum(vmask))
        valid_points += n_val
        if n_val == 0:
            skipped_trials += 1
            continue

        if not np.all(np.isfinite(y_pred[vmask])):
            raise ValueError(f"Non-finite y_pred on valid points in {tr.get('trial_id')}")

        diff_full = np.where(vmask, y_true - y_pred, 0.0)
        norm_sq = (diff_full / scales[:, None]) ** 2
        trial_mse = float(np.sum(norm_sq[vmask]) / n_val)

        if is_buf:
            buffer_by_worm.setdefault(worm_id, []).append(trial_mse)
            continue

        by_worm_cond.setdefault(worm_id, {}).setdefault(stim_id, []).append(trial_mse)
        stim_errors.setdefault(stim_id, []).append(trial_mse)

        # Split post-stimulus window into early onset transient [0, 3)s vs late sustained/drift [3, 10)s
        fps_tr = float(tr.get("fps", max(1.0, y_true.shape[1] / 10.0)))
        t_split = max(1, min(y_true.shape[1] - 1, int(round(3.0 * fps_tr))))
        if y_true.shape[1] >= 2:
            vm_e = vmask[:, :t_split]
            vm_l = vmask[:, t_split:]
            if np.any(vm_e):
                early_by_worm_cond.setdefault(worm_id, {}).setdefault(stim_id, []).append(
                    float(np.sum(norm_sq[:, :t_split][vm_e]) / int(np.sum(vm_e)))
                )
            if np.any(vm_l):
                late_by_worm_cond.setdefault(worm_id, {}).setdefault(stim_id, []).append(
                    float(np.sum(norm_sq[:, t_split:][vm_l]) / int(np.sum(vm_l)))
                )

        py_v = y_true[vmask]
        raw_diff = diff_full[vmask]
        pre_arr = np.broadcast_to(np.asarray(tr["y_pre_anchor"], dtype=np.float64)[:, None], y_true.shape)
        dy_v = py_v - pre_arr[vmask]

        if ref_center_y is None:
            ref_center_y = float(np.mean(py_v))
            ref_center_dy = float(np.mean(dy_v))

        rsq = float(np.sum(raw_diff ** 2))
        total_raw_sq_err += rsq
        total_nonbuf_pts += n_val

        yc = py_v - ref_center_y
        dyc = dy_v - (ref_center_dy if ref_center_dy is not None else 0.0)
        w_cnt[worm_id] = w_cnt.get(worm_id, 0) + n_val
        w_res_sq[worm_id] = w_res_sq.get(worm_id, 0.0) + rsq
        w_sum_yc[worm_id] = w_sum_yc.get(worm_id, 0.0) + float(np.sum(yc))
        w_sum_yc2[worm_id] = w_sum_yc2.get(worm_id, 0.0) + float(np.sum(yc ** 2))
        w_sum_dyc[worm_id] = w_sum_dyc.get(worm_id, 0.0) + float(np.sum(dyc))
        w_sum_dyc2[worm_id] = w_sum_dyc2.get(worm_id, 0.0) + float(np.sum(dyc ** 2))

        if tier_masks and y_true.shape[0] == len(functional_tiers or ()):
            row_counts = np.sum(vmask, axis=1)
            active_rows = row_counts > 0
            row_sums = np.sum(np.where(vmask, norm_sq, 0.0), axis=1)
            row_errs = np.zeros(y_true.shape[0], dtype=np.float64)
            row_errs[active_rows] = row_sums[active_rows] / row_counts[active_rows]
            for t_name, t_m in tier_masks.items():
                sel = active_rows & t_m
                if np.any(sel):
                    vals_t = row_errs[sel].tolist()
                    class_errors.setdefault(t_name, []).extend(vals_t)
                    worm_class_errors.setdefault(t_name, {}).setdefault(worm_id, []).extend(vals_t)

    def _equal_animal_mean(wmap: Mapping[str, Mapping[str, List[float]]]) -> Optional[float]:
        wvals = []
        for _, cmap in sorted(wmap.items()):
            cmeans = [float(np.mean(v)) for _, v in sorted(cmap.items()) if v]
            if cmeans:
                wvals.append(float(np.mean(cmeans)))
        return float(np.mean(wvals)) if wvals else None

    worm_losses: Dict[str, float] = {}
    for wid, cmap in sorted(by_worm_cond.items()):
        cmeans = [float(np.mean(v)) for _, v in sorted(cmap.items()) if v]
        if cmeans:
            worm_losses[wid] = float(np.mean(cmeans))

    if not worm_losses:
        raise ValueError("No valid non-buffer animals evaluated")

    worms_sorted = sorted(worm_losses.keys())
    e_d = float(np.mean([worm_losses[w] for w in worms_sorted]))
    rmse = float(np.sqrt(total_raw_sq_err / max(total_nonbuf_pts, 1))) if total_nonbuf_pts > 0 else float("nan")

    cnt_vec = np.array([w_cnt[w] for w in worms_sorted], dtype=np.float64)
    res_vec = np.array([w_res_sq[w] for w in worms_sorted], dtype=np.float64)
    yc_vec = np.array([w_sum_yc[w] for w in worms_sorted], dtype=np.float64)
    yc2_vec = np.array([w_sum_yc2[w] for w in worms_sorted], dtype=np.float64)
    dyc_vec = np.array([w_sum_dyc[w] for w in worms_sorted], dtype=np.float64)
    dyc2_vec = np.array([w_sum_dyc2[w] for w in worms_sorted], dtype=np.float64)

    n_all = float(np.sum(cnt_vec))
    ss_res_all = float(np.sum(res_vec))
    ss_post_all = float(np.sum(yc2_vec) - (np.sum(yc_vec) ** 2) / max(n_all, 1.0))
    ss_delta_all = float(np.sum(dyc2_vec) - (np.sum(dyc_vec) ** 2) / max(n_all, 1.0))
    post_r2 = float(1.0 - ss_res_all / max(ss_post_all, 1e-12))
    delta_r2 = float(1.0 - ss_res_all / max(ss_delta_all, 1e-12))

    post_ci = [post_r2, post_r2]
    delta_ci = [delta_r2, delta_r2]
    if compute_bootstrap and len(worms_sorted) >= 2:
        rng = np.random.default_rng(seed)
        nw = len(worms_sorted)
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
        post_ci = [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))]
        delta_ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]

    sex_losses: Dict[str, float] = {}
    for sx in sorted(set(sex_by_worm.values())):
        sw = [worm_losses[w] for w in worms_sorted if sex_by_worm.get(w) == sx]
        if sw:
            sex_losses[sx] = float(np.mean(sw))

    worm_cell_means = {
        tier: {w: float(np.mean(vals)) for w, vals in sorted(wmap.items())}
        for tier, wmap in sorted(worm_class_errors.items())
    }
    equal_animal_cell_errors = {
        tier: float(np.mean(list(wdict.values())))
        for tier, wdict in sorted(worm_cell_means.items())
        if wdict
    }

    buffer_e_d = (
        float(np.mean([np.mean(v) for v in buffer_by_worm.values()]))
        if buffer_by_worm
        else None
    )

    return {
        "E_d": e_d,
        "L_w": worm_losses,
        "rmse_raw": rmse,
        "RMSE": rmse,
        "post_R2": post_r2,
        "post_R2_ci95": post_ci,
        "delta_R2": delta_r2,
        "delta_R2_ci95": delta_ci,
        "buffer_E_d": buffer_e_d,
        "buffer_L_w": {w: float(np.mean(v)) for w, v in sorted(buffer_by_worm.items())},
        "temporal_horizon_errors": {
            "early_0_3s_E_d": _equal_animal_mean(early_by_worm_cond),
            "late_3_10s_E_d": _equal_animal_mean(late_by_worm_cond),
        },
        "stimulus_errors": {k: float(np.mean(v)) for k, v in sorted(stim_errors.items())},
        "cell_class_errors": equal_animal_cell_errors if equal_animal_cell_errors else {k: float(np.mean(v)) for k, v in sorted(class_errors.items())},
        "worm_cell_class_errors": worm_cell_means,
        "sex_errors": sex_losses,
        "coverage_fraction": float(valid_points / max(total_points, 1)),
        "skipped_trials": skipped_trials,
        "n_worms_total": len(worms_sorted),
    }


def compare_against_reference(
    cand_metrics: Mapping[str, Any],
    ref_metrics: Mapping[str, Any],
    n_boot: int = 2000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compute paired animal-level relative gain `I_d`, worm count improved, and bootstrap CI."""
    cand_lw = cand_metrics["L_w"]
    ref_lw = ref_metrics["L_w"]
    worms = sorted(cand_lw.keys())
    if set(worms) != set(ref_lw.keys()):
        raise ValueError("Candidate and reference evaluated different animal sets")
    nw = len(worms)
    c_arr = np.array([float(cand_lw[w]) for w in worms], dtype=np.float64)
    r_arr = np.array([float(ref_lw[w]) for w in worms], dtype=np.float64)

    e_cand = float(np.mean(c_arr))
    e_ref = float(np.mean(r_arr))
    i_d = float(1.0 - e_cand / max(e_ref, 1e-12))
    n_imp = int(np.sum(c_arr < r_arr - 1e-12))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nw, size=(n_boot, nw))
    boot_gains = 1.0 - np.mean(c_arr[idx], axis=1) / np.maximum(np.mean(r_arr[idx], axis=1), 1e-12)

    return {
        "E_d_candidate": e_cand,
        "E_d_reference": e_ref,
        "I_d": i_d,
        "I_d_ci95": [float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5))],
        "n_worms_improved": n_imp,
        "n_worms_total": nw,
        "fraction_worms_improved": float(n_imp / max(nw, 1)),
        "paired_loss_diff_by_worm": {w: float(ref_lw[w] - cand_lw[w]) for w in worms},
    }


def save_CompactPredictions(
    out_path: Path,
    trials: Sequence[Mapping[str, Any]],
    y_preds: Sequence[np.ndarray],
    train_fold_animal_ids: Sequence[Sequence[str]],
    fold_ids: np.ndarray,
) -> None:
    """Save compact per-trial predictions on valid points + training-fold animal provenance for isolated scoring."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store sparse valid predictions + compact metadata so 40 runs take < 40 MB disk total
    flat_preds = []
    offsets = [0]
    meta_rows = []
    for i, (tr, yp) in enumerate(zip(trials, y_preds)):
        vm = np.asarray(tr["valid_mask"], dtype=bool)
        vals = np.asarray(yp, dtype=np.float32)[vm]
        flat_preds.append(vals)
        offsets.append(offsets[-1] + int(vals.size))
        meta_rows.append(
            {
                "dataset_id": tr["dataset_id"],
                "session_animal_id": tr["session_animal_id"],
                "animal_id": tr["animal_id"],
                "trial_id": tr["trial_id"],
                "onset_idx": int(tr["onset_idx"]),
                "stimulus_id": tr["stimulus_id"],
                "is_buffer_control": bool(tr["is_buffer_control"]),
                "fold_idx": int(fold_ids[i]),
            }
        )
    concat_preds = np.concatenate(flat_preds) if flat_preds else np.zeros(0, dtype=np.float32)
    payload = {
        "trials_meta": meta_rows,
        "train_fold_session_animal_ids": [list(x) for x in train_fold_animal_ids],
    }
    np.savez_compressed(
        out_path,
        flat_y_pred=concat_preds,
        offsets=np.array(offsets, dtype=np.int64),
        meta_json=np.array(json.dumps(payload)),
    )
