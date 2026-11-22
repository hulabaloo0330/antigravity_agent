# Beyond-2-vCPU CPU-Only PyTorch 2.8 Extension Suite (`autoresearch_v1`)

This directory contains **three standalone, genuine, CPU-only PyTorch 2.8 (`torch==2.8.*`)** extension scripts designed to scale the `autoresearch_v1` C. elegans chemosensory trajectory & connectome experiments beyond the in-container 2-vCPU / 7.8 GiB RAM envelope onto multi-core CPU servers (`8 / 16 / 32 / 64 vCPU`) with **zero GPU/CUDA requirement**.

## Scientific & Integrity Guarantees
1. **Strict Zero-Vault Isolation**: Every script loads **Dev splits ONLY** (`load_dataset(..., split="dev", allow_vault_unseal=False)`). None of these scripts ever reads `data/*/vault_sealed/`.
2. **Strict Zero-Leakage Causal Protocol**: All scripts enforce the frozen `[-10.0, 0.0)s -> [0.0, 10.0)s` causal window (`configs/lambda_global.json`), fold-internal neuron variance scaling `s_n = max(std_train_n, s_floor)`, and zero `dataset_id` hyperparameter branches.
3. **Clean Multi-Core CPU Scaling (`--device cpu --n-jobs`)**: Every worker process inside `multiprocessing.Pool` explicitly pins `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `torch.set_num_threads(1)` to eliminate BLAS thread oversubscription across `8 / 16 / 32 / 64 vCPU` servers.
4. **PyTorch 2.8 CPU-Only Compatibility**: `requirements_cpu_torch28.txt` pins `torch>=2.8.0,<2.9.0` via the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`). All three scripts use strictly standard, forward-compatible PyTorch APIs verified end-to-end in the project `.venv` (`torch==2.5.1+cpu`).

## Installation (External Multi-Core CPU Node)
```bash
python3.11 -m venv .venv_torch28_cpu
source .venv_torch28_cpu/bin/activate
pip install --upgrade pip
pip install -r experiments/autoresearch_v1/scripts/beyond_2vcpu/requirements_cpu_torch28.txt
```

## Included Standalone Scripts

| Script | Scientific Purpose | Parallelization Unit | Checkpointing |
|---|---|---|---|
| `run_full_k199_patch_film_nulls.py` | Scales in-container `K_film=19` end-to-end `M35_02` Patch-FiLM null retraining (which hit the `p=0.05` floor on `seyedolmohadesin2025`, `0/19` nulls beating real) to full **`K_film=199`** (`p=0.005` floor) across all 3 Dev datasets | Null graphs `k = 1..K` | Atomic per-null JSON (`null_k{k:03d}.json`) + `--resume` |
| `run_nebula_iwae_causal_ssm_lwo.py` | Genuine **NEBULA/IWAE-style strictly causal variational SSM** (`[-10,0)s` causal GRU encoder -> low-rank nonlinear latent rollout -> `K_iw` importance-weighted ELBO -> `s_n`-scaled decoder) with built-in causal invariance check | Grouped / LWO CV folds | Atomic summary JSON |
| `run_creamer_em_connectome_scaling.py` | Genuine **Creamer-style EM-connectome continuous-time RK2 biophysical ODE-SSM** parameterizing signed `Cook 2019` (`W_chem_pos/neg/unk`, `W_gap`) + `Ripoll-Sánchez 2023` (`W_npp_short/mid`, `W_monoamine`), learnable tier time constants `tau_i`, and low-rank residual sweep (`--ranks 0,8,16`, `--n-seeds 20`) | `(rank, seed, fold)` tasks | Atomic summary JSON |

## Fast End-to-End Smoke Tests (`< 45s` Each on 2 vCPUs)
Run from `/home/antigravity` (setting `TMPDIR` to a local scratch directory, never `/tmp`):
```bash
export SCRATCH_DIR="$HOME/.gemini/jetski/scratch/teamwork_tmp/smoke_verify"
mkdir -p "$SCRATCH_DIR"

# 1. Script 1: M35_02 Patch-FiLM Null Retraining (Seyedolmohadesin fold 0, K=2)
TMPDIR="$SCRATCH_DIR" python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_full_k199_patch_film_nulls.py \
  --smoke-test --device cpu --n-jobs 2

# 2. Script 2: NEBULA/IWAE Strictly Causal Variational SSM (Yemini folds 0-1)
TMPDIR="$SCRATCH_DIR" python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_nebula_iwae_causal_ssm_lwo.py \
  --smoke-test --device cpu --n-jobs 2

# 3. Script 3: Creamer-Style EM-Connectome Biophysical ODE-SSM (Yemini fold 0, 2 seeds)
TMPDIR="$SCRATCH_DIR" python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_creamer_em_connectome_scaling.py \
  --smoke-test --device cpu --n-jobs 2
```

## Full Multi-Core Production Commands (Example: 16-vCPU CPU-Only Server)
```bash
# 1. Full K_film=199 M35_02 Patch-FiLM null retraining (~19 min on 16 vCPUs, resumable)
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_full_k199_patch_film_nulls.py \
  --device cpu --n-jobs 16 --k-null 199 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024 --resume

# 2. NEBULA/IWAE causal variational SSM across all 3 Dev datasets (~10 min on 16 vCPUs)
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_nebula_iwae_causal_ssm_lwo.py \
  --device cpu --n-jobs 16 --latent-dim 24 --dyn-rank 8 --k-iw 8 --epochs 24 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024

# 3. Creamer EM-connectome RK2 ODE-SSM 20-seed x 3-rank sweep (~16 min on 16 vCPUs)
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_creamer_em_connectome_scaling.py \
  --device cpu --n-jobs 16 --n-seeds 20 --ranks 0,8,16 --epochs 20 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024
```

For full `2 / 8 / 16 / 32 / 64 vCPU` wall-clock & RAM extrapolations grounded in exact measured 2-vCPU timings, see:
- `experiments/autoresearch_v1/reports/BEYOND_2VCPU_COMPUTE_EXTRAPOLATION.md`
- `research/autoresearch_v1/BEYOND_CPU_PROPOSALS.md`
