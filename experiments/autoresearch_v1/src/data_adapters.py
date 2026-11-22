"""Standardized raw/unsmoothed multi-dataset animal record loader with strict test-vault sealing.

Implements Interface Contract 1 (`PROJECT.md`) and `MODEL_REQUIREMENTS.md` §1, §2, §6:
- Loads standardized per-animal `.npz` records from `/home/antigravity/data/<dataset_id>/<split>/`.
- Strictly raises `PermissionError` if `split == "vault"` and `allow_vault_unseal` is `False` (or if
  `final_manifest.json` does not exist when `require_final_manifest=True`).
- Verifies per-file SHA-256 checksums against `experiments/autoresearch_v1/split_manifest.json` and logs
  every authorized vault access to `experiments/autoresearch_v1/vault_access_audit.jsonl`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DATA_ROOT_DEFAULT = Path("/home/antigravity/data")
EXP_ROOT_DEFAULT = Path("/home/antigravity/experiments/autoresearch_v1")


@dataclass
class AnimalRecord:
    """Canonical in-memory representation of a single C. elegans whole-brain recording."""

    dataset_id: str
    animal_id: str
    sex: str  # "hermaphrodite" | "male"
    modality: str  # "immobilized" | "freely_moving"
    date_batch: str  # Recording date / experimental batch group ID
    fps: float  # Sampling rate in Hz
    traces_raw: np.ndarray  # float32 [T, N_obs], raw/unsmoothed dF/F0 or fluorescence (NaN where unobserved)
    obs_mask: np.ndarray  # bool [T, N_obs], True iff finite & observed
    neuron_ids: List[str]  # length N_obs, canonical NeuroPAL names preserving ASEL/R & AWCON/OFF
    stim_events: List[Dict[str, Any]]  # [{"stim_id": str, "onset_idx": int, "offset_idx": int, "trial_order": int, ...}]
    exposure_status: str  # "HISTORICALLY_EXPOSED_DEV_ONLY" | "NEW_DEV" | "SEALED_CONFIRMATION_VAULT"
    sha256: str = ""
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save_animal_record_npz(record: AnimalRecord, out_path: Path) -> str:
    """Serialize an `AnimalRecord` to compressed `.npz` and return its exact file SHA-256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "dataset_id": record.dataset_id,
        "animal_id": record.animal_id,
        "sex": record.sex,
        "modality": record.modality,
        "date_batch": record.date_batch,
        "fps": float(record.fps),
        "neuron_ids": list(record.neuron_ids),
        "stim_events": list(record.stim_events),
        "exposure_status": record.exposure_status,
        "extra_metadata": record.extra_metadata,
    }
    np.savez_compressed(
        out_path,
        traces_raw=np.asarray(record.traces_raw, dtype=np.float32),
        obs_mask=np.asarray(record.obs_mask, dtype=bool),
        metadata_json=np.array(json.dumps(meta_payload, sort_keys=True)),
    )
    sha = compute_file_sha256(out_path)
    record.sha256 = sha
    return sha


def load_animal_record_npz(npz_path: Path, expected_sha256: Optional[str] = None) -> AnimalRecord:
    """Load a single `.npz` file into an `AnimalRecord`, optionally verifying SHA-256."""
    actual_sha = compute_file_sha256(npz_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {npz_path}: expected {expected_sha256}, got {actual_sha}"
        )
    with np.load(npz_path, allow_pickle=False) as z:
        traces_raw = z["traces_raw"].astype(np.float32)
        obs_mask = z["obs_mask"].astype(bool)
        meta = json.loads(str(z["metadata_json"].item()))
    return AnimalRecord(
        dataset_id=meta["dataset_id"],
        animal_id=meta["animal_id"],
        sex=meta["sex"],
        modality=meta["modality"],
        date_batch=meta["date_batch"],
        fps=float(meta["fps"]),
        traces_raw=traces_raw,
        obs_mask=obs_mask,
        neuron_ids=list(meta["neuron_ids"]),
        stim_events=list(meta["stim_events"]),
        exposure_status=meta["exposure_status"],
        sha256=actual_sha,
        extra_metadata=dict(meta.get("extra_metadata", {})),
    )


def load_dataset(
    dataset_id: str,
    split: str = "dev",
    allow_vault_unseal: bool = False,
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
    require_final_manifest_for_vault: bool = False,
    caller_reason: str = "unspecified",
) -> List[AnimalRecord]:
    """Load all standardized `AnimalRecord`s for `(dataset_id, split)` (Interface Contract 1).

    Raises
    ------
    PermissionError
         whenever `split == "vault"` and `allow_vault_unseal` is `False` (preventing any accidental
        test-set snooping during M1–M3 development and hyperparameter search).
    """
    split_norm = split.strip().lower()
    if split_norm == "vault_sealed":
        split_norm = "vault"
    if split_norm not in ("dev", "vault"):
        raise ValueError(f"Invalid split={split!r}; must be 'dev', 'vault', or 'vault_sealed'.")

    if split_norm == "vault":
        if not allow_vault_unseal:
            raise PermissionError(
                f"SEALED VAULT ACCESS DENIED for dataset={dataset_id!r}: split='vault' cannot be loaded "
                "when allow_vault_unseal=False. Vault animals are strictly quarantined until pre-registered "
                "M4 confirmation after freezing final_manifest.json."
            )
        final_manifest_path = Path(exp_root) / "final_manifest.json"
        if require_final_manifest_for_vault and not final_manifest_path.exists():
            raise PermissionError(
                f"SEALED VAULT ACCESS DENIED: final_manifest.json not found at {final_manifest_path}."
            )
        # Log authorized unseal event to append-only query audit log (`MODEL_REQUIREMENTS.md` §6.9)
        audit_log = Path(exp_root) / "vault_access_audit.jsonl"
        audit_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_id": dataset_id,
            "split": split_norm,
            "allow_vault_unseal": True,
            "final_manifest_present": final_manifest_path.exists(),
            "caller_reason": caller_reason,
        }
        with open(audit_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    split_manifest_path = Path(exp_root) / "split_manifest.json"
    expected_hashes: Dict[str, str] = {}
    if split_manifest_path.exists():
        sm = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        ds_entry = sm.get("datasets", {}).get(dataset_id, {})
        for rec_meta in ds_entry.get(f"{split_norm}_records", []):
            expected_hashes[rec_meta["animal_id"]] = rec_meta["sha256"]

    split_dir = Path(data_root) / dataset_id / split_norm
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory does not exist: {split_dir}")

    npz_files = sorted(split_dir.glob("*.npz"))
    records: List[AnimalRecord] = []
    for p in npz_files:
        rec = load_animal_record_npz(p)
        if rec.animal_id in expected_hashes and rec.sha256 != expected_hashes[rec.animal_id]:
            raise ValueError(
                f"Tamper/checksum error on {p} (animal_id={rec.animal_id}): "
                f"manifest={expected_hashes[rec.animal_id]}, disk={rec.sha256}"
            )
        if split_norm == "vault" and rec.exposure_status != "SEALED_CONFIRMATION_VAULT":
            raise ValueError(f"Invalid exposure_status={rec.exposure_status!r} inside vault split for {rec.animal_id}")
        if split_norm == "dev" and rec.exposure_status == "SEALED_CONFIRMATION_VAULT":
            raise ValueError(f"Vault animal {rec.animal_id} leaked into dev split!")
        records.append(rec)

    return records
