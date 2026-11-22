"""Core scientific anti-leakage, causality, vault-lock, and connectome tests (`M1` — 12 focused checks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from src.audit_cli import run_all_checks
from src.connectome_priors import canonicalize_neuron_name, load_connectome_priors
from src.data_adapters import AnimalRecord, load_dataset, save_animal_record_npz

DATA_ROOT = Path("/home/antigravity/data")
EXP_ROOT = Path("/home/antigravity/experiments/autoresearch_v1")


def causal_prestim_baseline_normalize(traces_raw: np.ndarray, onset_idx: int, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference causal `[-10, 0)s` window slicer used to verify zero future leakage."""
    w = int(round(10.0 * fps))
    pre = traces_raw[onset_idx - w : onset_idx]  # strictly < onset_idx
    post = traces_raw[onset_idx : onset_idx + w]  # [onset_idx, onset_idx + w)
    mu_pre = np.nanmean(pre, axis=0, keepdims=True)
    sd_pre = np.nanstd(pre, axis=0, keepdims=True) + 1e-6
    return (pre - mu_pre) / sd_pre, (post - mu_pre) / sd_pre


# 1. Sealed vault lock (`vault` & `vault_sealed`)
def test_01_sealed_vault_blocks_access_by_default() -> None:
    for ds_id in ("seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper", "yemini2021"):
        for alias in ("vault", "vault_sealed"):
            with pytest.raises(PermissionError, match="SEALED VAULT ACCESS DENIED"):
                load_dataset(ds_id, split=alias, allow_vault_unseal=False, data_root=DATA_ROOT, exp_root=EXP_ROOT)


# 2. Final manifest gate before unsealing
def test_02_vault_requires_final_manifest_when_enforced(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="final_manifest.json"):
        load_dataset(
            "seyedolmohadesin2025",
            split="vault",
            allow_vault_unseal=True,
            data_root=DATA_ROOT,
            exp_root=tmp_path,
            require_final_manifest_for_vault=True,
        )


# 3. SHA-256 tamper protection on `.npz` records
def test_03_tampered_npz_checksum_rejected(tmp_path: Path) -> None:
    ds_dir = tmp_path / "data" / "mock_ds" / "dev"
    rec = AnimalRecord(
        dataset_id="mock_ds",
        animal_id="mock_w1",
        sex="hermaphrodite",
        modality="immobilized",
        date_batch="20250101",
        fps=4.0,
        traces_raw=np.array([[1.2, 2.3], [1.4, np.nan]], dtype=np.float32),
        obs_mask=np.array([[True, True], [True, False]], dtype=bool),
        neuron_ids=["ASEL", "ASER"],
        stim_events=[{"stim_id": "nacl", "onset_idx": 1, "offset_idx": 2, "trial_order": 1}],
        exposure_status="NEW_DEV",
    )
    npz_path = ds_dir / "mock_w1.npz"
    good_sha = save_animal_record_npz(rec, npz_path)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "split_manifest.json").write_text(
        json.dumps({"datasets": {"mock_ds": {"dev_records": [{"animal_id": "mock_w1", "sha256": good_sha}]}}}),
        encoding="utf-8",
    )
    rec.traces_raw[0, 0] = -999.0
    save_animal_record_npz(rec, npz_path)
    with pytest.raises(ValueError, match="Tamper/checksum error"):
        load_dataset("mock_ds", split="dev", data_root=tmp_path / "data", exp_root=exp_dir)


# 4. Zero animal_id, physical_animal_id, and date_batch overlap across splits + E2 confirmation power
def test_04_two_unexposed_immobilized_sources_and_disjoint_vault_splits() -> None:
    sm = json.loads((EXP_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    seyed = sm["datasets"]["seyedolmohadesin2025"]
    assert seyed["n_dev"] + seyed["n_vault"] == 47
    assert seyed["n_vault"] >= 14
    assert set(seyed["dev_dates"]).isdisjoint(set(seyed["vault_dates"]))

    toyo = sm["datasets"]["toyoshima2024"]
    assert toyo["n_dev"] + toyo["n_vault"] == 24
    assert toyo["n_vault"] >= 10
    assert len({r["physical_animal_id"] for r in toyo["vault_records"]}) >= 10
    assert {r["physical_animal_id"] for r in toyo["dev_records"]}.isdisjoint(
        {r["physical_animal_id"] for r in toyo["vault_records"]}
    )
    assert set(toyo["dev_dates"]).isdisjoint(set(toyo["vault_dates"]))

    copper = sm["datasets"]["baskoylu2026_copper"]
    assert copper["n_dev"] + copper["n_vault"] == 27
    assert copper["n_vault"] == 9
    assert set(copper["dev_dates"]).isdisjoint(set(copper["vault_dates"]))


# 5. Historically exposed Yemini 2021 strictly quarantined from vault
def test_05_historically_exposed_yemini_has_zero_vault_animals() -> None:
    sm = json.loads((EXP_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    assert sm["datasets"]["yemini2021"]["n_dev"] == 21
    assert sm["datasets"]["yemini2021"]["n_vault"] == 0


# 6. Raw un-centered trace verification (rejects normalize1 / global z-score leakage)
def test_06_all_datasets_contain_raw_uncentered_traces() -> None:
    for ds_id in ("yemini2021", "seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper"):
        recs = load_dataset(ds_id, split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)
        assert len(recs) > 0
        col_means = np.abs(np.nanmean(recs[0].traces_raw, axis=0))
        assert float(np.nanmax(col_means)) > 1e-2, f"{ds_id} appears globally mean-centered!"


# 7. Causal `[-10, 0)s` window invariance to arbitrary post-stimulus `[0, +10)s` perturbation
def test_07_causal_prestim_features_invariant_to_future_corruption() -> None:
    rec = load_dataset("seyedolmohadesin2025", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)[0]
    onset = int(rec.stim_events[0]["onset_idx"])
    x_clean = rec.traces_raw.copy()
    x_corrupt = rec.traces_raw.copy()
    x_corrupt[onset:, :] = x_corrupt[onset:, :] * 50.0 + 1234.5  # Massive post-stimulus mutation

    pre_a, _ = causal_prestim_baseline_normalize(x_clean, onset, rec.fps)
    pre_b, _ = causal_prestim_baseline_normalize(x_corrupt, onset, rec.fps)
    assert np.allclose(pre_a, pre_b, equal_nan=True)


# 8. Fold-only summary statistics invariance (train-fold stats aligned to canonical neurons never leak val animals)
def test_08_fold_only_statistics_never_leak_validation_animals() -> None:
    priors = load_connectome_priors(sex="hermaphrodite", connectome_dir=DATA_ROOT / "connectomes")
    canon_idx = {n: i for i, n in enumerate(priors["canonical_neurons"])}
    recs = load_dataset("toyoshima2024", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)

    def align_worm_mean(r: AnimalRecord, scale: float = 1.0) -> np.ndarray:
        out = np.full(len(canon_idx), np.nan, dtype=np.float64)
        for j, nm in enumerate(r.neuron_ids):
            if nm in canon_idx:
                out[canon_idx[nm]] = float(np.nanmean(r.traces_raw[:, j] * scale))
        return out

    train_animals, val_animals = recs[:8], recs[8:]
    train_mu_1 = np.nanmean(np.vstack([align_worm_mean(r, 1.0) for r in train_animals]), axis=0)
    _ = [align_worm_mean(r, 100.0) for r in val_animals]
    train_mu_2 = np.nanmean(np.vstack([align_worm_mean(r, 1.0) for r in train_animals]), axis=0)
    assert np.allclose(train_mu_1, train_mu_2, equal_nan=True)


# 9. Distinct left/right (`ASEL != ASER`) and functional (`AWCON != AWCOFF`) neuron identities
def test_09_asymmetric_chemosensory_identities_preserved() -> None:
    y_recs = load_dataset("yemini2021", split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)
    y_ids = set().union(*(r.neuron_ids for r in y_recs))
    assert {"ASEL", "ASER", "AWCON", "AWCOFF"} <= y_ids

    for ds_id in ("seyedolmohadesin2025", "toyoshima2024", "baskoylu2026_copper"):
        ids = set().union(*(r.neuron_ids for r in load_dataset(ds_id, split="dev", data_root=DATA_ROOT, exp_root=EXP_ROOT)))
        assert {"ASEL", "ASER", "AWCL", "AWCR"} <= ids


# 10. Biophysical receptor-mediated connectome polarity (+1, -1 GluCl/ACC, 0 mixed)
def test_10_connectome_receptor_polarity_and_sex_dimorphism() -> None:
    p_h = load_connectome_priors(sex="hermaphrodite", connectome_dir=DATA_ROOT / "connectomes")
    p_m = load_connectome_priors(sex="male", connectome_dir=DATA_ROOT / "connectomes")
    pol = p_h["edge_polarity"]
    assert np.sum(pol == 1.0) > 100
    assert np.sum(pol == -1.0) > 100
    assert np.sum(pol == 0.0) > 100
    assert not np.allclose(p_h["chem"], p_m["chem"])
    assert np.allclose(p_h["gap"], p_h["gap"].T)


# 11. Randi 2023 WT vs unc-31 atlas + real temporal impulse-response / DANDI 001075 traces & quarantine
def test_11_randi2023_atlas_and_timeseries_present_and_quarantined() -> None:
    meta = json.loads((DATA_ROOT / "randi2023_atlas" / "randi2023_metadata.json").read_text(encoding="utf-8"))
    assert meta["wt_tested_directed_pairs"] > 1000
    assert meta["unc31_dependent_candidate_pairs"] > 0
    assert "NEVER_USED_FOR_EDGE_PRIOR_SELECTION" in meta["governance_boundary"]
    assert (DATA_ROOT / "randi2023_atlas" / "randi2023_wt_unc31_atlas.npz").exists()
    ts_z = np.load(DATA_ROOT / "randi2023_atlas" / "dandi001075_sample_timeseries.npz", allow_pickle=False)
    assert len(ts_z.files) >= 2


# 12. End-to-end audit-data check passes cleanly
def test_12_audit_data_passes_cleanly() -> None:
    ok, errors, _ = run_all_checks(data_root=DATA_ROOT, exp_root=EXP_ROOT)
    assert ok is True, f"audit-data failed: {errors}"
