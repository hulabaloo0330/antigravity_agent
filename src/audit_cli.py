"""Lean Scientific Anti-Leakage & Causality Audit (`Interface Contract 5`, `PROJECT.md`).

Usage:
    python -m src.cli audit-data
    python -m src.audit_cli --check-all

Focuses strictly on core scientific validity (no artificial MB caps or bureaucratic bloat):
1. Governance files (`data_registry.json`, `exposure_ledger.json`, `split_manifest.json`) exist and self-verify.
2. Zero cross-animal (`animal_id` AND `physical_animal_id`) overlap AND zero `date_batch` overlap between
   `dev` and `vault` across all datasets (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`, `baskoylu2026_copper`).
3. Sealed vault lock: `load_dataset(..., split="vault", allow_vault_unseal=False)` strictly raises `PermissionError`,
   and every `.npz` file on disk matches its `split_manifest.json` SHA-256 checksum.
4. Raw/unsmoothed causal fidelity: standardized traces are NOT globally z-scored/mean-centered, `obs_mask`
   matches `np.isfinite(traces_raw)`, and every stimulus event has valid causal pre-stimulus indices.
5. Biological identity & connectome fidelity: `ASEL != ASER` and `AWCON != AWCOFF` / `AWCL != AWCR` preserved,
   and CeNGEN receptor-mediated edge polarity distinguishes `+1`, `-1` (Cl-channel), and `0` (mixed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.connectome_priors import load_connectome_priors
from src.data_adapters import compute_file_sha256, load_dataset

DATA_ROOT_DEFAULT = Path("/home/antigravity/data")
EXP_ROOT_DEFAULT = Path("/home/antigravity/experiments/autoresearch_v1")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for fname in files:
            fp = Path(root) / fname
            if not fp.is_symlink():
                total += fp.stat().st_size
    return total


def run_all_checks(
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
) -> Tuple[bool, List[str], Dict[str, object]]:
    """Run core scientific anti-leakage, causality, vault-lock, and biological-prior checks."""
    errors: List[str] = []
    report: Dict[str, object] = {}

    reg_path = exp_root / "data_registry.json"
    led_path = exp_root / "exposure_ledger.json"
    man_path = exp_root / "split_manifest.json"
    for p in (reg_path, led_path, man_path):
        if not p.exists():
            errors.append(f"Missing governance artifact: {p}")
    if errors:
        return False, errors, report

    exposure_ledger = json.loads(led_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(man_path.read_text(encoding="utf-8"))

    sm_copy = {k: v for k, v in split_manifest.items() if k != "manifest_sha256"}
    expected_sm_hash = hashlib.sha256(json.dumps(sm_copy, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
    if split_manifest.get("manifest_sha256") != expected_sm_hash:
        errors.append("split_manifest.json self-hash mismatch.")

    ds_summaries = {}
    for ds_id, ds_info in split_manifest.get("datasets", {}).items():
        dev_recs = ds_info.get("dev_records", [])
        vault_recs = ds_info.get("vault_records", [])

        # Check 1: Zero animal_id, physical_animal_id, and date_batch overlap
        dev_ids = {r["animal_id"] for r in dev_recs}
        vault_ids = {r["animal_id"] for r in vault_recs}
        if dev_ids & vault_ids:
            errors.append(f"[{ds_id}] animal_id overlap between dev and vault: {sorted(dev_ids & vault_ids)}")

        dev_phys = {r.get("physical_animal_id", r["animal_id"]) for r in dev_recs}
        vault_phys = {r.get("physical_animal_id", r["animal_id"]) for r in vault_recs}
        if dev_phys & vault_phys:
            errors.append(f"[{ds_id}] physical_animal_id overlap between dev and vault: {sorted(dev_phys & vault_phys)}")

        dev_dates = {r["date_batch"] for r in dev_recs}
        vault_dates = {r["date_batch"] for r in vault_recs}
        if dev_dates & vault_dates:
            errors.append(f"[{ds_id}] date_batch confound overlap between dev and vault: {sorted(dev_dates & vault_dates)}")

        if ds_id == "yemini2021" and len(vault_recs) != 0:
            errors.append(f"[yemini2021] Historically exposed dataset must have 0 vault records, found {len(vault_recs)}")

        # Check 2: Disk SHA-256 integrity & exposure ledger consistency (without unsealing vault arrays)
        led_animals = exposure_ledger.get("datasets", {}).get(ds_id, {}).get("animals", {})
        for split_label, recs in (("dev", dev_recs), ("vault", vault_recs)):
            for r in recs:
                fpath = Path(r["path"])
                if not fpath.exists():
                    errors.append(f"[{ds_id}/{split_label}] Missing .npz file: {fpath}")
                    continue
                if compute_file_sha256(fpath) != r["sha256"]:
                    errors.append(f"[{ds_id}/{split_label}] SHA-256 mismatch for {r['animal_id']}")
                led_status = led_animals.get(r["animal_id"])
                if split_label == "vault" and led_status != "SEALED_CONFIRMATION_VAULT":
                    errors.append(f"[{ds_id}] Vault animal {r['animal_id']} ledger status={led_status!r}")
                if split_label == "dev" and led_status == "SEALED_CONFIRMATION_VAULT":
                    errors.append(f"[{ds_id}] Dev animal {r['animal_id']} marked SEALED_CONFIRMATION_VAULT!")

        # Check 3: Vault lock raises PermissionError by default (both 'vault' and 'vault_sealed')
        for v_alias in ("vault", "vault_sealed"):
            try:
                load_dataset(ds_id, split=v_alias, allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
                errors.append(f"[{ds_id}] load_dataset(split={v_alias!r}, allow_vault_unseal=False) failed to block!")
            except PermissionError:
                pass

        # Check 4: Dev records raw un-centered fidelity, causal stimulus indices, and asymmetric ID preservation
        loaded_dev = load_dataset(ds_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
        observed_union = set()
        max_abs_mean = 0.0
        for rec in loaded_dev:
            if rec.traces_raw.ndim != 2 or rec.obs_mask.shape != rec.traces_raw.shape:
                errors.append(f"[{rec.animal_id}] Invalid traces_raw/obs_mask shape: {rec.traces_raw.shape}")
            if not np.array_equal(rec.obs_mask, np.isfinite(rec.traces_raw)):
                errors.append(f"[{rec.animal_id}] obs_mask != np.isfinite(traces_raw)")
            if len(set(rec.neuron_ids)) != len(rec.neuron_ids):
                errors.append(f"[{rec.animal_id}] Duplicate neuron_ids in record")
            bad_names = [n for n in rec.neuron_ids if not n.isalnum() or len(n) > 10]
            if bad_names:
                errors.append(f"[{rec.animal_id}] Malformed neuron_ids: {bad_names[:5]}")
            if len(rec.stim_events) == 0:
                errors.append(f"[{rec.animal_id}] Zero stimulus events recorded")
            for ev in rec.stim_events:
                if int(ev["onset_idx"]) <= 0 or int(ev["onset_idx"]) >= rec.traces_raw.shape[0]:
                    errors.append(f"[{rec.animal_id}] Invalid stimulus onset_idx={ev['onset_idx']}")
            col_means = np.abs(np.nanmean(rec.traces_raw, axis=0))
            max_abs_mean = max(max_abs_mean, float(np.nanmax(col_means)))
            observed_union.update(rec.neuron_ids)

        if max_abs_mean < 1e-4:
            errors.append(f"[{ds_id}] Suspiciously zero-centered full-sequence traces (max|mean|={max_abs_mean:.3e})!")
        if "ASEL" not in observed_union or "ASER" not in observed_union:
            errors.append(f"[{ds_id}] Asymmetric pair ASEL / ASER not distinctly preserved!")

        ds_summaries[ds_id] = {
            "n_dev": len(dev_recs),
            "n_vault": len(vault_recs),
            "n_vault_unique_animals": len(vault_phys),
            "dev_unique_neurons": len(observed_union),
            "max_abs_trace_mean": round(max_abs_mean, 4),
        }

    report["datasets"] = ds_summaries

    # Check 5: Multi-layer signed connectome & biophysical receptor polarity
    priors_h = load_connectome_priors(sex="hermaphrodite", connectome_dir=data_root / "connectomes")
    priors_m = load_connectome_priors(sex="male", connectome_dir=data_root / "connectomes")
    c_neurons = set(priors_h["canonical_neurons"])
    for req_n in ("ASEL", "ASER", "AWCON", "AWCOFF", "AWCL", "AWCR", "AIAL", "AIAR", "AIBL", "AIBR", "AVAL", "AVAR"):
        if req_n not in c_neurons:
            errors.append(f"Connectome missing distinct neuron: {req_n}")

    pol = priors_h["edge_polarity"]
    n_pos, n_neg, n_mix = int(np.sum(pol == 1.0)), int(np.sum(pol == -1.0)), int(np.sum(pol == 0.0))
    if n_pos == 0 or n_neg == 0 or n_mix == 0:
        errors.append(f"Connectome polarity invalid (+1:{n_pos}, -1:{n_neg}, 0:{n_mix})")
    if np.array_equal(priors_h["chem"], priors_m["chem"]):
        errors.append("Hermaphrodite and male Cook 2019 chemical connectomes identical!")

    report["connectome"] = {
        "n_canonical_neurons": len(c_neurons),
        "polarity_plus1": n_pos,
        "polarity_minus1": n_neg,
        "polarity_mixed0": n_mix,
    }
    report["data_dir_mb"] = round(_dir_size_bytes(data_root) / (1024 * 1024), 2)

    return len(errors) == 0, errors, report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scientific anti-leakage, causality, and vault audit.")
    parser.add_argument("--check-all", action="store_true", help="Run all scientific data governance checks.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
    args = parser.parse_args(argv)

    if not args.check_all:
        parser.print_help()
        return 1

    ok, errors, report = run_all_checks(data_root=args.data_root, exp_root=args.exp_root)
    print(json.dumps({"passed": ok, "errors": errors, "summary": report}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
