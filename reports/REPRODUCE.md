# Reproducibility, SHA-256 Verification & Artifact Map (`autoresearch_v1`)

- **Environment**: Linux, CPU-only (`2 vCPUs`, `7.8 GiB RAM`, `0 B Swap`), Python 3.11 virtualenv `/home/antigravity/.venv` (PyTorch `2.8.*` CPU-only, `scipy`, `scikit-learn`, `numpy`, `h5py`, `cect`, `pytest`).
- **Disk Hygiene**: Zero writes to `/tmp`. Always set `TMPDIR` to a scratch directory under `/home/antigravity/.gemini/jetski/scratch/teamwork_tmp/` when running scratch jobs.
- **Frozen Global Configuration**: `experiments/autoresearch_v1/configs/lambda_global.json` (`config_id: lambda_global_v1`, `zero_dataset_id_branches: true`).

---

## 1. Cryptographic Integrity & Governance Audit Commands

Run from `/home/antigravity/experiments/autoresearch_v1`:

### 1.1 Verify `final_manifest.json` SHA-256 Digests (`27/27` Frozen Files)
```bash
/home/antigravity/.venv/bin/python -c "
import hashlib, json, pathlib
root = pathlib.Path('/home/antigravity/experiments/autoresearch_v1')
mf = json.loads((root / 'final_manifest.json').read_text())
print('Manifest SHA-256:', mf['manifest_sha256'])
all_files = {}
all_files.update(mf['sha256_checksums']['governance_and_configs'])
all_files.update(mf['sha256_checksums']['python_source_files'])
mismatches = []
for rel, expected in sorted(all_files.items()):
    actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if actual != expected:
        mismatches.append((rel, expected, actual))
print(f'Verified {len(all_files) - len(mismatches)}/{len(all_files)} frozen files.')
if mismatches:
    for rel, exp, act in mismatches:
        print(f'  DIFF: {rel} expected={exp[:12]} actual={act[:12]}')
"
```
*(Note: `exposure_ledger.json` was intentionally updated once during the pre-registered M5 confirmatory unsealing at `2026-11-22T00:59:44Z` to record `unsealed_once: true` and `pre_unseal_exposure_ledger_sha256: 5f76a5099f1d4c831a03320106ef5c461169415bb2ab0181ddaea2586a8ad5ee`; all 26 other frozen files retain exact pre-unseal SHA-256 digests.)*

### 1.2 Data Governance, Vault & Anti-Leakage CLI Checks
```bash
# 1. Validate data_registry.json, split_manifest.json, exposure_ledger.json, & connectome bundle
PYTHONPATH=/home/antigravity/experiments/autoresearch_v1 \
  /home/antigravity/.venv/bin/python -m src.cli audit-data

# 2. Run M1 lean scientific anti-leakage unit tests (12/12 PASS)
PYTHONPATH=/home/antigravity/experiments/autoresearch_v1 \
  /home/antigravity/.venv/bin/pytest -v \
  /home/antigravity/experiments/autoresearch_v1/tests/test_m1_data.py

# 3. Verify live model causal & OOF leakage invariance (future poison == 0.0, OOF tamper == 0.0)
PYTHONPATH=/home/antigravity/experiments/autoresearch_v1 \
  /home/antigravity/.venv/bin/python -m src.cli verify-invariance --model B4_ridge_self_stim
```

---

## 2. End-to-End Pipeline Execution Commands (CPU-Only)

All commands assume `CWD=/home/antigravity/experiments/autoresearch_v1`, `PYTHONPATH=/home/antigravity/experiments/autoresearch_v1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

| Milestone | Description | Exact CPU-Only Command | Persisted Output Artifact |
|---|---|---|---|
| **M2** | Historical Yemini audit + 15 baselines (6 families) on 3 immobilized Dev splits | `/home/antigravity/.venv/bin/python -m models.m2_pipeline` | `results/yemini_historical_audit.json`, `results/m2_baselines_summary.json` |
| **M3 R1–R4** | H1 slow states & 7 confounds, H2 bilinear vs additive + buffer/perm controls, H3 4-layer connectome & sex dimorphism, F11 robustness | `/home/antigravity/.venv/bin/python -m models.m3_round1_h1` (then `m3_round2_h2`, `m3_round3_h3`, `m3_round4_f11`) | `results/m3_mechanistic_and_robustness.json` |
| **M4 Stage A** | `K=199` constrained retrained null graphs on linear base (`W_gap` $p=0.020$) + `K_film=19` | `/home/antigravity/.venv/bin/python -m models.m4_null_graphs` | `results/null_graphs_summary.json` |
| **M3.5 R5–R6** | Toyoshima SNR diagnosis, zero-init nonlinear calcium self-anchor + 1/2-hop Patch-FiLM, 5-seed, 9/9 OAT, F11 dropout/5s, `K=199` nonlinear-base nulls + `K_film=19` | `/home/antigravity/.venv/bin/python -m models.m35_round5_nonlinear` && `/home/antigravity/.venv/bin/python -m models.m35_round6_validation --stage all` | `results/m35_toyoshima_diagnosis.json`, `results/m35_round6_full_validation.json` |
| **M4 Stage B** | Non-circular `Randi 2023` WT vs `unc-31` + isolated `Copper` Dev + §8 cryptographic freeze (`final_manifest.json`) | `/home/antigravity/.venv/bin/python scripts/run_m4b_randi_copper_freeze.py` | `results/randi_unc31_validation.json`, `results/copper_stress_test.json`, `final_manifest.json` |
| **M5** | **One-shot confirmatory unsealing** (34 unseen vaulted worms, prediction-first save, 2,000× animal bootstrap, hotspots) | `/home/antigravity/.venv/bin/python scripts/run_m5_vault_confirmation.py` *(already executed once at `2026-11-22T00:59:44Z`; do not re-run to tune)* | `results/confirmation_final.json`, `runs/m5_confirmatory/` |
| **Standalone Scorer** | Re-score any saved prediction `.npz` independently from canonical traces | `/home/antigravity/.venv/bin/python -m src.cli score --pred <path_to_npz> --split <dev\|confirmatory>` | JSON metrics (`E_d`, `L_w`, `I_d`, `post_R2`, `delta_R2`, `rmse_raw`) |

---

## 3. Beyond-2-vCPU Multi-Core CPU-Only Extensions

Standalone PyTorch `2.8.*` multi-core CPU scripts (`--device cpu --n-jobs`) for heavier post-2-vCPU experiments (`K_film=199` Patch-FiLM null retraining, 20-seed confirmatory ensembles, multi-hop graph attention) live in:
- `/home/antigravity/experiments/autoresearch_v1/scripts/beyond_2vcpu/`
- Documentation & empirical timing extrapolations: `/home/antigravity/research/autoresearch_v1/BEYOND_CPU_PROPOSALS.md`

---

## 4. Complete Workspace Artifact Map

- **Data & Connectomes (`/home/antigravity/data/`, `203.24 MiB`)**: `yemini2021/`, `seyedolmohadesin2025/{dev,vault_sealed}/`, `toyoshima2024/{dev,vault_sealed}/`, `baskoylu2026_copper/{dev,vault_sealed}/`, `randi2023_atlas/`, `connectomes/connectome_priors_bundle.npz`.
- **Governance & Manifests (`/home/antigravity/experiments/autoresearch_v1/`)**: `configs/lambda_global.json`, `data_registry.json`, `split_manifest.json`, `toyoshima2024_manifest.json`, `exposure_ledger.json`, `protocol.json`, `baseline_registry.json`, `runs.jsonl`, `final_manifest.json`, `research_loop_log.md`.
- **Persisted Quantitative Results (`/home/antigravity/experiments/autoresearch_v1/results/`)**: `yemini_historical_audit.json`, `m2_baselines_summary.json`, `m3_mechanistic_and_robustness.json`, `null_graphs_summary.json`, `m35_toyoshima_diagnosis.json`, `m35_round6_full_validation.json`, `randi_unc31_validation.json`, `copper_stress_test.json`, `confirmation_final.json`.
- **Publication-Grade Reports (`/home/antigravity/experiments/autoresearch_v1/reports/`)**: `FINAL_SCIENTIFIC_REPORT.md`, `NEURON_HOTSPOT_AND_SNR_ANALYSIS.md`, `LITERATURE_COMPARISON.md`, `REPRODUCE.md`.
- **Research Summary Mirrors (`/home/antigravity/research/autoresearch_v1/`)**: `STATUS.md`, `HYPOTHESES.md`, `RESULTS.md`, `LITERATURE_COMPARISON.md`, `REPRODUCE.md`, `BEYOND_CPU_PROPOSALS.md`.
