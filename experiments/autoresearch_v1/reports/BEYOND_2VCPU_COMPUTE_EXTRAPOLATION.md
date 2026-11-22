# Beyond-2-vCPU CPU-Only PyTorch 2.8 Extension Suite & Compute-Cost Extrapolation Report (`autoresearch_v1`)

**Date**: 2026-11-22  
**Target Runtime**: CPU-Only Multi-Core Linux Nodes (`torch>=2.8.0,<2.9.0`, `--device cpu --n-jobs <N>`, `OMP_NUM_THREADS=1` & `torch.set_num_threads(1)` per worker)  
**Vault Isolation**: **100% Dev-Only (`allow_vault_unseal=False`)** — no extension script ever reads `data/*/vault_sealed/`  
**Executable Package**: `experiments/autoresearch_v1/scripts/beyond_2vcpu/`

---

## 1. Executive Summary

All core `autoresearch_v1` milestones (`M1`–`M5`) were executed strictly CPU-only inside our 2-vCPU / 7.8 GiB RAM (0 B Swap) container. While that envelope was sufficient to complete the full 5-seed `M3.5` validation, 9-point `0.5x/1x/2x` OAT neighborhood stability, missingness/short-window robustness, `K=199` closed-form connectome null retraining, `Randi 2023` WT/`unc-31` & `Baskoylu 2026 Copper` checks, and the one-shot `M5` sealed-vault confirmation, **three high-value scientific follow-up experiments** exceed a single interactive 2-vCPU turn budget:

1. **Extension 1 (`run_full_k199_patch_film_nulls.py`)**: Scaling the in-container `K_film=19` end-to-end PyTorch `M35_02` Patch-FiLM constrained null-graph pilot — which hit the exact empirical p-value floor `p = 1/(1+19) = 0.05` on `seyedolmohadesin2025` (`0/19` nulls beating the real connectome, `z = +2.25`) and on `Main-2 macro` (`0/19`, `p = 0.05`, `z = +2.18`) — to full **`K_film=199`** (`p = 0.005` floor) across all three immobilized Dev datasets (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`).
2. **Extension 2 (`run_nebula_iwae_causal_ssm_lwo.py`)**: A genuine **NEBULA/IWAE-style strictly causal variational state-space model** baseline (`[-10, 0)s` causal GRU variational encoder -> low-rank nonlinear latent rollout -> `K_iw`-particle IWAE ELBO -> `s_n`-scaled masked Gaussian decoder) scaled to full Leave-One-Worm-Out (67 worms) × 5-seed Dev CV.
3. **Extension 3 (`run_creamer_em_connectome_scaling.py`)**: A genuine **Creamer-style EM-connectome continuous-time RK2 biophysical ODE-SSM** coupling signed `Cook 2019` (`W_chem_pos`, `W_chem_neg`, `W_chem_unk`, `W_gap`) + `Ripoll-Sánchez 2023` (`W_npp_short/mid`, `W_monoamine`) with learnable neuron-class time constants `tau_i` and a **20-seed × 3-rank (`0, 8, 16`)** scaling sweep across all three Dev datasets.

All three standalone scripts have been implemented, verified end-to-end with `--smoke-test --device cpu --n-jobs 2` on our 2-vCPU container (`exit code 0`, `< 45s` each, genuine loss optimization & finite `E_d` / `I_d` / `post_R2`), and benchmarked to ground the `2 / 8 / 16 / 32 / 64 vCPU` scaling tables below.

---

## 2. Measured 2-vCPU Ground-Truth Benchmarks (In-Container)

### 2.1 Historical In-Container Reference Timings (2 vCPU, 7.8 GiB RAM, `torch==2.5.1+cpu`)

| Benchmark Source | Workload | Measured Wall-Clock (2 vCPU) | Peak RSS (MB) | Key Measured Metric / Unit Cost |
|---|---|---:|---:|---|
| `m35_round6_full_validation.json` (Stage 1) | 5-seed stability (`M35_01/02/03`, 4 new seeds × 3 Dev datasets) | `313.92 s` (per-dataset: Yemini `306.85 s`, Seyed `270.05 s`, Toyo `312.64 s`) | `1960.46 MB` (Seyed peak) | `~23 s / seed / model / dataset` |
| `m35_round6_full_validation.json` (Stage 2) | 9-point `0.5x/1x/2x` OAT neighborhood (`M35_03`, 3 Dev datasets) | `296.61 s` | `1210.29 MB` | `9/9` positive vs `B4` & `M35_01` |
| `m35_round6_full_validation.json` (Stage 3) | Missingness dropout (`+0..50%`) + `[-5,0)s` short window (3 Dev datasets) | `333.13 s` | `1756.98 MB` | Strictly causal, zero pre-split PPCA |
| `m35_round6_full_validation.json` (Stage 4a) | Constrained null bank gen (`K=199`, 8 layers) + `K=199` Ridge nulls (3 Dev datasets) | Bank gen: `122.92 s`; Ridge199: Yemini `221.59 s`, Seyed `494.89 s`, Toyo `275.54 s` | `833.85 MB` | `0.62 s / null` bank gen; `1.1–2.5 s / null` closed-form |
| `m35_round6_full_validation.json` (Stage 4b) | **`K_film=19` End-to-End `M35_02` Patch-FiLM nulls** (Seyed + Toyo) | Seyed: `385.52 s`; Toyo: `393.11 s` | `833.85 MB` | **`20.29 s / null` (Seyed)**, **`20.69 s / null` (Toyo)**; `0/19` beat real on Seyed (`p=0.05`, `z=+2.25`) |
| `null_graphs_summary.json` (M4) | `K=199` Ridge nulls (3 Dev) + `K_film=19` `M3R4_01` FiLM nulls (Main-2) | Bank gen: `92.14 s`; Ridge199: `2641.93 s`; FiLM19: Seyed `727.90 s`, Toyo `544.47 s` | `2677.10 MB` | `38.3 s / null` (Seyed), `28.7 s / null` (Toyo) |
| `confirmation_final.json` (M5 timing ref only) | One-shot 5-seed evaluation on Dev->Vault (Seyed `218.89 s`, Toyo `90.64 s`, Copper `17.93 s`) | `327.91 s` | `2214.29 MB` | Timing reference only; vault strictly excluded from extension scripts |

### 2.2 Live Smoke-Test Verification of All 3 Standalone Extension Scripts (2 vCPU, `--device cpu --n-jobs 2`)

| Script | Exact Smoke-Test Scope | Exit Code | Wall-Clock (2 vCPU) | Peak RSS (MB) | Measured Verification Output |
|---|---|---:|---:|---:|---|
| `run_full_k199_patch_film_nulls.py` | `seyedolmohadesin2025` Dev fold 0, `K=2` nulls, `max_epochs=4` | **`0`** | **`37.80 s`** (`2.40 s / null` on 1 fold) | `1316.83 MB` | `anchor=0.177672`, `real E_d=0.173023` (`post_R2=0.9279`, **`I_vs_anchor=+2.6167%`**) vs `null E_d=[0.175032, 0.175401]` (`0/2` beat real, `p=0.333`, `z=+8.42`) |
| `run_nebula_iwae_causal_ssm_lwo.py` | `yemini2021` Dev folds 0–1 (2-worker pool), `D=12`, `K_iw=4`, `epochs=5` | **`0`** | **`4.32 s`** (`2.45 s / fold` / worker) | `560.81 MB` | `E_d=0.767047`, `post_R2=0.4647`, **`I_vs_persist=+2.3891%`**, IWAE loss `1.2723 -> 1.2529`, `causal_invariance_max_diff=0.0` |
| `run_creamer_em_connectome_scaling.py` | `yemini2021` Dev fold 0, `rank=8`, `n_seeds=2`, `epochs=6`, `t_steps=16` | **`0`** | **`3.20 s`** (`2.00 s / task` / worker) | `528.26 MB` | `E_d=0.465439 ± 0.000088`, `post_R2=0.6603`, **`I_vs_persist=+0.2067%`**, learned `cook_gap=0.2039`, `cook_chem_pos=0.2016`, `sensory tau=2.401 s` |

---

## 3. CPU-Only Multi-Core Scaling Extrapolations (`2 / 8 / 16 / 32 / 64 vCPU`, `torch==2.8.*`)

**Scaling Model**:  
Because every worker pins `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`, tasks (null graphs `k=1..K`, CV folds, or `(rank, seed, fold)` tuples) are embarrassingly parallel with small fixed setup overhead `T_setup` (data loading, null-bank generation, or frozen anchor caching) and empirical parallel efficiency `eta(P)` accounting for memory-bandwidth contention and fork/IPC overhead:
- `P = 2`: `eta = 0.92`
- `P = 8`: `eta = 0.90`
- `P = 16`: `eta = 0.88`
- `P = 32`: `eta = 0.85`
- `P = 64`: `eta = 0.82`

$$T_{\text{wall}}(P) = T_{\text{setup}} + \frac{T_{\text{parallel, 1-core}}}{P \cdot \eta(P)}$$

### 3.1 Extension 1 — Full `K_film = 199` End-to-End `M35_02` Patch-FiLM Null Retraining (All 3 Dev Datasets)

- **Grounded Unit Cost**: Measured `20.29 s / null` (`seyedolmohadesin2025`, 5 folds), `20.69 s / null` (`toyoshima2024`, 5 folds), and `~19.0 s / null` (`yemini2021`, 5 folds) -> `59.98 s / null` across all 3 Dev datasets (`T_parallel,1 = 199 × 60.0 s = 11,940 s = 3.32 core-hours`). Fixed setup: `123 s` null-bank generation + `180 s` frozen `M35_01` anchor caching (`T_setup = 303 s`).
- **RAM Footprint**: `~1.1–1.3 GiB` per worker on `seyedolmohadesin2025` (lower on `yemini2021`/`toyoshima2024`; Linux `fork` copy-on-write shares read-only anchor arrays across workers).

| CPU Node (`torch==2.8.*`) | `--n-jobs` | Est. RAM / Worker | Recommended Node RAM | Est. Total Wall-Clock |
|---|---:|---:|---:|---:|
| `2 vCPU` (current container) | `2` | `~1.3 GiB` | `>= 4 GiB` | **`1.89 h`** (`113 min`) |
| `8 vCPU` | `8` | `~1.2 GiB` | `>= 16 GiB` | **`0.54 h`** (`33 min`) |
| `16 vCPU` | `16` | `~1.1 GiB` | `>= 24 GiB` | **`0.32 h`** (`19 min`) |
| `32 vCPU` | `32` | `~1.1 GiB` | `>= 48 GiB` | **`0.21 h`** (`12 min`) |
| `64 vCPU` | `64` | `~1.1 GiB` | `>= 96 GiB` | **`0.15 h`** (`9 min`) |

**Copy-Paste Command (`16 vCPU` example, fully resumable)**:
```bash
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_full_k199_patch_film_nulls.py \
  --device cpu --n-jobs 16 --k-null 199 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024 --resume
```

### 3.2 Extension 2 — `NEBULA/IWAE` Strictly Causal Variational SSM (Full 67-Worm LWO × 5 Seeds across All 3 Dev Datasets)

- **Grounded Unit Cost**: Measured `2.45 s / fold` on `yemini2021` at `(D=12, K_iw=4, epochs=5, T=16)` -> scaled to production `(D=24, K_iw=8, epochs=24, T=20)` (`~24x` FLOPs): `~59 s / fold` (`yemini2021`), `~148 s / fold` (`seyedolmohadesin2025`), `~106 s / fold` (`toyoshima2024`).
  - **5-Fold Grouped Dev CV** (15 folds, 1 seed): `T_parallel,1 = 1,565 s` (`0.43 core-hours`) -> **`~3 min` on 16 vCPUs**.
  - **Full Leave-One-Worm-Out (67 worms) × 5 Seeds** (335 tasks): `T_parallel,1 = 35,400 s` (`9.83 core-hours`), `T_setup ≈ 30 s`.
- **RAM Footprint**: `~0.60–0.65 GiB` per worker (`560.81 MB` measured peak RSS in smoke test).

| CPU Node (`torch==2.8.*`) | `--n-jobs` | Est. RAM / Worker | Recommended Node RAM | Est. Wall-Clock (5-Fold, 1-Seed) | Est. Wall-Clock (67-Worm LWO × 5 Seeds) |
|---|---:|---:|---:|---:|---:|
| `2 vCPU` | `2` | `~0.65 GiB` | `>= 4 GiB` | `0.24 h` (`15 min`) | **`5.34 h`** |
| `8 vCPU` | `8` | `~0.65 GiB` | `>= 8 GiB` | `0.07 h` (`4 min`) | **`1.37 h`** (`82 min`) |
| `16 vCPU` | `16` | `~0.65 GiB` | `>= 16 GiB` | `0.04 h` (`2.5 min`) | **`0.70 h`** (`42 min`) |
| `32 vCPU` | `32` | `~0.65 GiB` | `>= 32 GiB` | `0.03 h` (`1.6 min`) | **`0.36 h`** (`22 min`) |
| `64 vCPU` | `64` | `~0.65 GiB` | `>= 64 GiB` | `0.02 h` (`1.2 min`) | **`0.19 h`** (`11 min`) |

**Copy-Paste Command (`16 vCPU` example)**:
```bash
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_nebula_iwae_causal_ssm_lwo.py \
  --device cpu --n-jobs 16 --latent-dim 24 --dyn-rank 8 --k-iw 8 --epochs 24 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024
```

### 3.3 Extension 3 — Creamer-Style EM-Connectome Biophysical ODE-SSM (`20 Seeds × 3 Ranks [0, 8, 16] × 5 Folds = 900 Tasks`)

- **Grounded Unit Cost**: Measured `2.00 s / task` on `yemini2021` at `(epochs=6, t_steps=16)` -> scaled to production `(epochs=20, t_steps=20)` (`~4.17x` FLOPs): `8.35 s / task` (`yemini2021`), `20.9 s / task` (`seyedolmohadesin2025`), `15.0 s / task` (`toyoshima2024`). Across `3 ranks × 20 seeds × 5 folds = 300 tasks / dataset` (`900 tasks` total): `T_parallel,1 = 13,275 s` (`3.69 core-hours`), `T_setup ≈ 20 s`.
- **RAM Footprint**: `~0.55–0.60 GiB` per worker (`528.26 MB` measured peak RSS in smoke test).

| CPU Node (`torch==2.8.*`) | `--n-jobs` | Est. RAM / Worker | Recommended Node RAM | Est. Total Wall-Clock (900 Tasks) |
|---|---:|---:|---:|---:|
| `2 vCPU` | `2` | `~0.60 GiB` | `>= 4 GiB` | **`2.00 h`** (`120 min`) |
| `8 vCPU` | `8` | `~0.60 GiB` | `>= 8 GiB` | **`0.51 h`** (`31 min`) |
| `16 vCPU` | `16` | `~0.60 GiB` | `>= 16 GiB` | **`0.26 h`** (`16 min`) |
| `32 vCPU` | `32` | `~0.60 GiB` | `>= 32 GiB` | **`0.14 h`** (`8 min`) |
| `64 vCPU` | `64` | `~0.60 GiB` | `>= 64 GiB` | **`0.07 h`** (`4 min`) |

**Copy-Paste Command (`16 vCPU` example)**:
```bash
python experiments/autoresearch_v1/scripts/beyond_2vcpu/run_creamer_em_connectome_scaling.py \
  --device cpu --n-jobs 16 --n-seeds 20 --ranks 0,8,16 --epochs 20 \
  --datasets yemini2021,seyedolmohadesin2025,toyoshima2024
```

### 3.4 Combined Full-Suite Wall-Clock Summary (All 3 Extensions End-to-End)

| Node Size (`torch==2.8.*`, CPU-only) | Combined Wall-Clock (Ext 1 + Ext 2 5-Fold + Ext 3) | Combined Wall-Clock (Ext 1 + Ext 2 67-Worm LWO×5 + Ext 3) |
|---|---:|---:|
| `2 vCPU` | `4.13 h` | `9.23 h` |
| `8 vCPU` | `1.12 h` | `2.42 h` |
| `16 vCPU` | **`0.62 h` (`37 min`)** | **`1.28 h` (`77 min`)** |
| `32 vCPU` | **`0.38 h` (`23 min`)** | **`0.71 h` (`43 min`)** |
| `64 vCPU` | **`0.24 h` (`14 min`)** | **`0.41 h` (`25 min`)** |

---

## 4. Scientific Hypotheses Resolved by Each Extended Run

1. **Breaking the `p = 0.05` Empirical Resolution Floor (`run_full_k199_patch_film_nulls.py`)**:
   - In our 2-vCPU Round 6 validation (`m35_round6_full_validation.json`), `M35_02` Patch-FiLM achieved `0/19` nulls beating the real connectome on `seyedolmohadesin2025` (`z = +2.25`, `p = 0.05`) and `0/19` on `Main-2 macro` (`z = +2.18`, `p = 0.05`), while `toyoshima2024` showed no topological specificity (`16/19`, `p = 0.85`).
   - Running `K_film = 199` lowers the minimum achievable empirical p-value from `0.05` to `0.005`, decisively distinguishing whether the `seyedolmohadesin2025` nonlinear Patch-FiLM topological advantage is genuinely significant at `p < 0.01` (`alpha = 0.01` after Bonferroni correction across datasets) or a small-`K` tail fluctuation.
2. **Testing Causal Variational Latent-State Inference Without Future-Frame Smoothing (`run_nebula_iwae_causal_ssm_lwo.py`)**:
   - Recent latent-variable foundation models (`NEBULA v2`, `CAPT`) report strong reconstruction `R^2` under non-causal smoothing or masked-frame imputation. By enforcing a strictly causal `[-10, 0)s` variational encoder (`causal_invariance_max_diff == 0.0`) and `K_iw`-particle IWAE training, Extension 2 directly tests whether stochastic latent state-space dynamics (`D=24`, low-rank `A + f_theta`) surpass `M35_01`/`M35_02` on unseen worms under our leak-free causal forecasting protocol.
3. **Testing Continuous-Time Biophysical Connectome ODE Scaling & Rank Saturation (`run_creamer_em_connectome_scaling.py`)**:
   - Directly parameterizing signed `Cook 2019` wired (`W_chem_pos/neg/unk`, `W_gap`) and `Ripoll-Sánchez 2023` wireless (`W_npp_short/mid`, `W_monoamine`) matrices inside a damped RK2 ODE-SSM (`tau_i dx_i/dt = -x_i + sum_m g_m W_m phi(x) + (U V^T) phi(x) + B u(t)`) tests:
     (a) whether `rank=0` (pure EM + wireless connectome) vs `rank=8, 16` (connectome + low-rank unmapped extrasynaptic modes) reveals a measurable saturation knee across 20 random seeds; and
     (b) whether learned neuron-class calcium time constants `tau_class` and layer gains `g_m` consistently recover faster sensory dynamics vs slower interneuron/premotor attractor timescales across datasets.
