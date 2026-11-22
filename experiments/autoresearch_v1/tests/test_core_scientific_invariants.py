"""Core Scientific Invariants Test Suite (`M2`, `DISPATCH.md §4.2` — 14 high-precision checks).

Focuses strictly on genuine mathematical & scientific anti-leakage guarantees:
1. Physical-second causal windowing across heterogeneous frame rates (`4.0`, `2.66`, `5.719` Hz).
2. End-to-end future-frame poisoning invariance (`t >= onset_idx` -> `max|diff| < 1e-7`).
3. Out-of-fold validation animal tampering invariance (`oof_tamper_weight_diff < 1e-7`).
4. Training-fold-only per-neuron scale `s_n = max(std_train_n, s_floor)` (10th percentile floor).
5. Equal-per-animal primary loss `L_w` and `E_d` invariance to trial-repeat duplication.
6. Multi-session `physical_animal_id` CV grouping on `toyoshima2024` (`14` sessions = `11` worms).
7. Strict separation of buffer/control trials (`buffer_E_d`) from primary `E_d`.
8. Separate variance denominators & animal-bootstrap CIs for `post_R2` vs `delta_R2` (Trap #3).
9. Target-self `i` and bilateral homolog `i_LR` exclusion from cross-neuron state & subspace pathways.
10. Configurable preservation of asymmetric `ASEL/ASER` and `AWCON/AWCOFF` sensory channels.
11. Static AST check enforcing zero `dataset_id` hyperparameter branching under `lambda_global`.
12. Sealed confirmation vault read-lock (`PermissionError` when `allow_vault_unseal=False`).
13. Standalone read-only `scorer.py` recomputing `s_n` from disk ground truth.
14. End-to-end `verify_model_invariance` pass across linear, subspace/LDS, and neural baselines.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluators.scorer import score_saved_predictions_file
from models.m2_pipeline import verify_single_model_invariance
from src.baselines import CausalBaselineModel
from src.data_adapters import load_dataset
from src.evaluator import (
    align_records_to_canonical,
    apply_sensor_bounds_to_aligned,
    build_grouped_cv_splits,
    compute_causal_window_frames,
    compute_training_fold_neuron_scales,
    compute_training_fold_sensor_bounds,
    evaluate_predictions,
    extract_causal_trial_records,
    save_CompactPredictions,
)
from src.features import extract_causal_network_state_per_target, get_bilateral_exclusion_set

DATA_ROOT = Path("/home/antigravity/data")
EXP_ROOT = Path("/home/antigravity/experiments/autoresearch_v1")


def _load_cfg() -> dict:
    cfg = json.loads((EXP_ROOT / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    cfg["hyperparameters"]["max_epochs"] = 5
    return cfg


def test_01_physical_second_window_frame_counts() -> None:
    w4 = compute_causal_window_frames(4.0, input_sec=10.0, target_sec=10.0, calibration_sec=20.0)
    assert w4["n_input_frames"] == 40 and w4["n_target_frames"] == 40 and w4["n_calibration_frames"] == 80

    w_seyed = compute_causal_window_frames(2.66, input_sec=10.0, target_sec=10.0, calibration_sec=20.0)
    assert w_seyed["n_input_frames"] == 27 and w_seyed["n_target_frames"] == 26 and w_seyed["n_calibration_frames"] == 54

    w_toyo = compute_causal_window_frames(5.719, input_sec=10.0, target_sec=10.0, calibration_sec=20.0)
    assert w_toyo["n_input_frames"] == 58 and w_toyo["n_target_frames"] == 57 and w_toyo["n_calibration_frames"] == 115


def test_02_future_frame_poisoning_leaves_predictions_identical() -> None:
    cfg = _load_cfg()
    recs = load_dataset("yemini2021", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)[:6]
    aligned, canon, tiers = align_records_to_canonical(recs)
    s_n, _, _ = compute_training_fold_neuron_scales(
        [r.traces for r in aligned[:5]], [r.obs_mask for r in aligned[:5]]
    )
    train_trials = [t for r in aligned[:5] for t in extract_causal_trial_records(r)]
    val_clean = aligned[5]
    val_poison = copy.deepcopy(val_clean)
    onset0 = int(val_poison.stim_events[0]["onset_idx"])
    val_poison.traces[onset0:, :] = np.where(val_poison.obs_mask[onset0:, :], 1.0e6, np.nan).astype(np.float32)

    tr_clean = extract_causal_trial_records(val_clean)[0]
    tr_poison = extract_causal_trial_records(val_poison)[0]
    assert float(np.nanmax(np.abs(tr_clean["y_true"] - tr_poison["y_true"]))) > 1e5

    for mid in ("B3_self_arfir_spline", "B4_mlp_repaired_self_stim", "B5_lds_stim", "B5_causal_tcn", "B6_ray_gordus_state_gain"):
        m = CausalBaselineModel(mid, cfg, canon, tiers).fit(train_trials, s_n)
        yp_a = m.predict_trial(tr_clean)
        yp_b = m.predict_trial(tr_poison)
        assert float(np.max(np.abs(yp_a - yp_b))) < 1e-7, f"Future leak in {mid}"


def test_03_oof_validation_label_tampering_leaves_weights_identical() -> None:
    res = verify_single_model_invariance("B4_mlp_repaired_cross_neuron", dataset_id="yemini2021")
    assert res["oof_tamper_weight_diff"] < 1e-7
    assert res["oof_tamper_pred_diff"] < 1e-7


def test_04_training_fold_scale_s_n_and_s_floor_never_touch_test_animals() -> None:
    rng = np.random.default_rng(42)
    tr1 = rng.normal(0.0, 2.0, size=(50, 5)).astype(np.float32)
    tr1[:, 0] = rng.normal(0.0, 0.01, size=50).astype(np.float32)  # very quiet neuron with tiny positive std
    tr1[:, 4] = np.nan  # unobserved neuron
    mk1 = np.isfinite(tr1)
    s_a, floor_a, counts_a = compute_training_fold_neuron_scales([tr1], [mk1], floor_percentile=10.0)

    # Unobserved neuron 4 and quiet neuron 0 receive exact s_floor; normal neuron 1 exceeds s_floor
    assert np.isclose(s_a[4], floor_a)
    assert np.isclose(s_a[0], floor_a)
    assert counts_a[4] == 0 and counts_a[1] == 1
    assert s_a[1] > floor_a


def test_05_equal_per_animal_loss_invariant_to_repeat_duplication() -> None:
    y_t = np.ones((4, 10), dtype=np.float32) * 2.0
    y_p1 = np.ones((4, 10), dtype=np.float32) * 1.0  # sq err = 1.0
    y_p2 = np.ones((4, 10), dtype=np.float32) * 0.0  # sq err = 4.0
    vm = np.ones((4, 10), dtype=bool)
    sn = np.ones(4, dtype=np.float64)
    anc = np.zeros(4, dtype=np.float32)

    t_w1_c1 = {"animal_id": "w1", "stimulus_id": "nacl", "is_buffer_control": False, "y_true": y_t, "valid_mask": vm, "y_pre_anchor": anc}
    t_w1_c2 = {"animal_id": "w1", "stimulus_id": "iaa", "is_buffer_control": False, "y_true": y_t, "valid_mask": vm, "y_pre_anchor": anc}
    t_w2_c1 = {"animal_id": "w2", "stimulus_id": "nacl", "is_buffer_control": False, "y_true": y_t, "valid_mask": vm, "y_pre_anchor": anc}

    m_base = evaluate_predictions([t_w1_c1, t_w1_c2, t_w2_c1], [y_p1, y_p2, y_p1], [sn, sn, sn], compute_bootstrap=False)
    # Duplicate w1's nacl repeat 5 times -> w1's nacl condition mean stays 1.0, iaa stays 4.0, L_w1 stays 2.5!
    trials_dup = [t_w1_c1] * 5 + [t_w1_c2, t_w2_c1]
    preds_dup = [y_p1] * 5 + [y_p2, y_p1]
    sn_dup = [sn] * 7
    m_dup = evaluate_predictions(trials_dup, preds_dup, sn_dup, compute_bootstrap=False)

    assert np.isclose(m_base["L_w"]["w1"], 2.5)
    assert np.isclose(m_dup["L_w"]["w1"], 2.5)
    assert np.isclose(m_base["E_d"], m_dup["E_d"])


def test_06_physical_animal_id_cv_grouping_on_toyoshima2024() -> None:
    recs = load_dataset("toyoshima2024", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)
    aligned, _, tiers = align_records_to_canonical(recs)
    assert len(aligned) == 14
    assert len({r.physical_animal_id for r in aligned}) == 11

    splits = build_grouped_cv_splits(aligned)
    assert len(splits) == 11
    for tr_i, va_i in splits:
        tr_phys = {aligned[i].physical_animal_id for i in tr_i}
        va_phys = {aligned[i].physical_animal_id for i in va_i}
        assert tr_phys.isdisjoint(va_phys)

    trials = [t for r in aligned for t in extract_causal_trial_records(r)]
    zero_preds = [np.zeros_like(t["y_true"]) for t in trials]
    sn = [np.ones(310, dtype=np.float64) for _ in trials]
    ev = evaluate_predictions(trials, zero_preds, sn, functional_tiers=tiers, compute_bootstrap=False)
    assert ev["n_worms_total"] == 11


def test_07_buffer_controls_excluded_from_primary_E_d() -> None:
    y_t = np.ones((4, 8), dtype=np.float32)
    vm = np.ones((4, 8), dtype=bool)
    sn = np.ones(4, dtype=np.float64)
    anc = np.zeros(4, dtype=np.float32)
    t_stim = {"animal_id": "w1", "stimulus_id": "nacl", "is_buffer_control": False, "y_true": y_t, "valid_mask": vm, "y_pre_anchor": anc}
    t_buf = {"animal_id": "w1", "stimulus_id": "Control", "is_buffer_control": True, "y_true": y_t, "valid_mask": vm, "y_pre_anchor": anc}

    m1 = evaluate_predictions([t_stim, t_buf], [y_t, np.zeros_like(y_t)], [sn, sn], compute_bootstrap=False)
    m2 = evaluate_predictions([t_stim, t_buf], [y_t, y_t * 500.0], [sn, sn], compute_bootstrap=False)
    assert np.isclose(m1["E_d"], 0.0) and np.isclose(m2["E_d"], 0.0)
    assert m2["buffer_E_d"] > m1["buffer_E_d"] * 100.0


def test_08_separate_post_R2_vs_delta_R2_denominators_and_cis() -> None:
    rng = np.random.default_rng(7)
    trials, preds, sns = [], [], []
    for w in range(6):
        anc = rng.normal(0.0, 5.0, size=10).astype(np.float32)  # large baseline spread across neurons
        dy = rng.normal(0.0, 0.5, size=(10, 12)).astype(np.float32)  # smaller evoked delta
        yt = anc[:, None] + dy
        yp = anc[:, None] + 0.5 * dy
        trials.append({
            "animal_id": f"w{w}",
            "stimulus_id": "s1",
            "is_buffer_control": False,
            "y_true": yt,
            "valid_mask": np.ones((10, 12), dtype=bool),
            "y_pre_anchor": anc,
        })
        preds.append(yp)
        sns.append(np.ones(10, dtype=np.float64))

    ev = evaluate_predictions(trials, preds, sns, compute_bootstrap=True, n_boot=300, seed=11)
    assert ev["post_R2"] > ev["delta_R2"] + 0.10
    assert ev["post_R2_ci95"] != ev["delta_R2_ci95"]


def test_09_target_self_and_bilateral_homolog_excluded_from_network_summaries() -> None:
    recs = load_dataset("yemini2021", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)[:2]
    aligned, canon, _ = align_records_to_canonical(recs)
    idx_map = {n: i for i, n in enumerate(canon)}
    aval_i, avar_i = idx_map["AVAL"], idx_map["AVAR"]

    ex_aval = get_bilateral_exclusion_set("AVAL", canon, exclude_bilateral=True)
    assert aval_i in ex_aval and avar_i in ex_aval

    tr0 = extract_causal_trial_records(aligned[0])[0]
    sn = np.ones(len(canon), dtype=np.float64)
    mu = np.zeros(len(canon), dtype=np.float32)
    st_base = extract_causal_network_state_per_target(tr0, canon, sn, mu, exclude_bilateral=True)

    tr_mod = copy.deepcopy(tr0)
    tr_mod["x_pre"][:, [aval_i, avar_i]] = 999.0
    tr_mod["m_pre"][:, [aval_i, avar_i]] = True
    st_mod = extract_causal_network_state_per_target(tr_mod, canon, sn, mu, exclude_bilateral=True)
    assert float(np.max(np.abs(st_base[aval_i] - st_mod[aval_i]))) < 1e-7


def test_10_asymmetric_chemosensory_identities_preserved_when_enabled() -> None:
    recs = load_dataset("yemini2021", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)[:1]
    aligned, canon, _ = align_records_to_canonical(recs)
    idx_map = {n: i for i, n in enumerate(canon)}
    asel_i, aser_i = idx_map["ASEL"], idx_map["ASER"]
    awcon_i, awcoff_i, awcr_i = idx_map["AWCON"], idx_map["AWCOFF"], idx_map["AWCR"]

    # Anatomical alias AWCR must ALWAYS be excluded from AWCON even when preserve_sensory_asymmetry=True
    ex_awcon = get_bilateral_exclusion_set("AWCON", canon, exclude_bilateral=True, preserve_sensory_asymmetry=True)
    assert awcr_i in ex_awcon and awcoff_i not in ex_awcon

    tr_a = extract_causal_trial_records(aligned[0])[0]
    tr_a["x_pre"][:, [asel_i, aser_i, awcon_i, awcoff_i]] = 0.0
    tr_a["m_pre"][:, [asel_i, aser_i, awcon_i, awcoff_i]] = True
    tr_b = copy.deepcopy(tr_a)
    tr_b["x_pre"][-5:, aser_i] = 3.5
    tr_b["x_pre"][-5:, awcoff_i] = -3.5

    sn = np.ones(len(canon), dtype=np.float64)
    mu = np.zeros(len(canon), dtype=np.float32)
    s0 = extract_causal_network_state_per_target(tr_a, canon, sn, mu, exclude_bilateral=True, preserve_sensory_asymmetry=True)
    s1 = extract_causal_network_state_per_target(tr_b, canon, sn, mu, exclude_bilateral=True, preserve_sensory_asymmetry=True)
    assert float(np.max(np.abs(s1[asel_i] - s0[asel_i]))) > 1e-2
    assert float(np.max(np.abs(s1[awcon_i] - s0[awcon_i]))) > 1e-2


def test_11_ast_audit_enforces_zero_dataset_id_hyperparameter_branching() -> None:
    ds_literals = {"yemini2021", "seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper"}
    for rel in ("src/features.py", "src/baselines.py", "src/evaluator.py"):
        tree = ast.parse((EXP_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                consts = {
                    c.value for c in ast.walk(node.test) if isinstance(c, ast.Constant) and isinstance(c.value, str)
                }
                overlap = consts & ds_literals
                assert not overlap, f"Prohibited dataset_id branch {overlap} found in {rel}"


def test_12_vault_seal_remains_locked_during_dev_evaluation() -> None:
    for ds_id in ("seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper", "yemini2021"):
        with pytest.raises(PermissionError, match="SEALED VAULT ACCESS DENIED"):
            load_dataset(ds_id, split="vault", allow_vault_unseal=False, data_root=DATA_ROOT, exp_root=EXP_ROOT)


def test_13_standalone_scorer_recomputes_scales_from_ground_truth(tmp_path: Path) -> None:
    recs = load_dataset("toyoshima2024", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)
    aligned, _, tiers = align_records_to_canonical(recs)
    splits = build_grouped_cv_splits(aligned)

    trials_all, preds_all, sn_all, fids = [], [], [], []
    train_sess = []
    for f_i, (tr_idx, va_idx) in enumerate(splits):
        lo_b, hi_b = compute_training_fold_sensor_bounds(
            [aligned[i].traces for i in tr_idx], [aligned[i].obs_mask for i in tr_idx]
        )
        tr_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in tr_idx]
        va_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in va_idx]
        sn, _, _ = compute_training_fold_neuron_scales(
            [r.traces for r in tr_bounded], [r.obs_mask for r in tr_bounded]
        )
        train_sess.append([aligned[i].animal_id for i in tr_idx])
        for vb in va_bounded:
            for tr in extract_causal_trial_records(vb, include_synthetic_buffer_windows=True):
                anc = np.where(np.isfinite(tr["y_pre_anchor"]), tr["y_pre_anchor"], 0.0).astype(np.float32)
                yp = np.repeat(anc[:, None], tr["y_true"].shape[1], axis=1)
                trials_all.append(tr)
                preds_all.append(yp)
                sn_all.append(sn)
                fids.append(f_i)

    direct = evaluate_predictions(trials_all, preds_all, sn_all, functional_tiers=tiers, compute_bootstrap=False)
    pred_npz = tmp_path / "predictions_dev_toyoshima2024.npz"
    save_CompactPredictions(pred_npz, trials_all, preds_all, train_sess, np.array(fids, dtype=np.int32))

    scored = score_saved_predictions_file(
        pred_npz, split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT, resolve_registry_ref=False
    )
    assert np.isclose(direct["E_d"], scored["E_d"], rtol=1e-5, atol=1e-6)
    assert np.isclose(direct["rmse_raw"], scored["rmse_raw"], rtol=1e-5, atol=1e-6)
    assert "command_motor" in scored["cell_class_errors"]
    assert scored["n_worms_total"] == 11


def test_14_verify_invariance_passes_across_baseline_families() -> None:
    for mid in ("B3_self_arfir_spline", "B5_pop_rrr", "B5_lds_stim", "B6_poco_film"):
        res = verify_single_model_invariance(mid, dataset_id="yemini2021")
        assert res["passed"] is True, f"Invariance check failed for {mid}: {res}"
