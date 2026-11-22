# Auto-Research Iterative Loop Log (`autoresearch_v1`)

- **Protocol**: `celegans_chemosensory_causal_trajectory_v1` (`[-10.0, 0.0)s` causal input $\to$ `[0.0, 10.0)s` post-stimulus trajectory, optional `[-30.0, -10.0)s` causal calibration)
- **Unified Global Config**: `configs/lambda_global.json` (`b449c039d14d...`, zero `dataset_id` branches)
- **Frozen Reference Baseline (`MODEL_REQUIREMENTS.md §4–§5`)**: **`B4_ridge_self_stim`** (strongest `self-history + stimulus` model on unexposed main Dev splits `seyedolmohadesin2025` + `toyoshima2024`)
- **Evaluation Splits (Dev ONLY — Sealed Vaults Untouched)**:
  - `seyedolmohadesin2025` (Main Unexposed #1): `32` Dev worms (`6` date-batch-grouped folds, `768` non-buffer chemical trials + `192` buffer controls)
  - `toyoshima2024` (Main Unexposed #2): `14` Dev sessions = **`11` unique physical worms** (`11` physical-worm-grouped folds, `360` NaCl step trials)
  - `yemini2021` (Historical Dev / Audit): `21` Dev worms (`21` LOWO folds, `63` odor/salt trials)

---

## Round 0 (`M2` Baseline Anchor): All 6 Competitive Baseline Families

### 0.1 Cross-Source Primary Metric Summary (`E_d`, `I_d`, `I_macro`, Worm Improvement Counts)

| Family | Model ID | Self-Only | `Seyed2025 E_d` | `Seyed2025 I_d` (`Worms`) | `Toyo2024 E_d` | `Toyo2024 I_d` (`Worms`) | **`I_macro` (Main 2)** | `Yemini2021 E_d` | `Yemini2021 I_d` (`Worms`) | Time (s) |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1` | `B1_persistence` | Yes | 0.2312 | -36.61% (`2/32`) | 0.3787 | -11.38% (`0/11`) | **-23.99%** | 1.0183 | -27.46% (`4/21`) | 0.6 |
| `B2` | `B2_neuron_stim_mean` | Yes | 0.1884 | -11.35% (`3/32`) | 0.3771 | -10.91% (`0/11`) | **-11.13%** | 0.8771 | -9.78% (`4/21`) | 1.2 |
| `B3` | `B3_self_arfir_spline` | Yes | 0.1739 | -2.77% (`5/32`) | 0.3409 | -0.26% (`2/11`) | **-1.52%** | 0.7789 | +2.50% (`15/21`) | 3.8 |
| `B4` | `B4_ridge_self_stim` **(REF)** | Yes | 0.1692 | +0.00% (`0/32`) | 0.3400 | +0.00% (`0/11`) | **+0.00%** | 0.7989 | +0.00% (`0/21`) | 3.7 |
| `B4` | `B4_ridge_cross_neuron` | No | 0.1690 | +0.11% (`19/32`) | 0.3401 | -0.04% (`7/11`) | **+0.04%** | 0.7958 | +0.39% (`12/21`) | 4.0 |
| `B4` | `B4_mlp_repaired_self_stim` | Yes | 0.1635 | +3.39% (`23/32`) | 0.3475 | -2.23% (`5/11`) | **+0.58%** | 0.7829 | +2.01% (`15/21`) | 55.4 |
| `B4` | `B4_mlp_repaired_cross_neuron` | No | 0.1644 | +2.85% (`18/32`) | 0.3491 | -2.69% (`5/11`) | **+0.08%** | 0.7847 | +1.78% (`13/21`) | 53.8 |
| `B5` | `B5_pop_rrr` | No | 0.1801 | -6.40% (`4/32`) | 0.3412 | -0.36% (`2/11`) | **-3.38%** | 0.7943 | +0.58% (`12/21`) | 3.9 |
| `B5` | `B5_lds_stim` | No | 0.1806 | -6.72% (`4/32`) | 0.3412 | -0.37% (`2/11`) | **-3.55%** | 0.7947 | +0.52% (`12/21`) | 17.8 |
| `B5` | `B5_causal_gru` | No | 0.1805 | -6.63% (`3/32`) | 0.3526 | -3.72% (`0/11`) | **-5.17%** | 0.8036 | -0.58% (`10/21`) | 311.0 |
| `B5` | `B5_causal_tcn` | No | 0.1807 | -6.78% (`4/32`) | 0.3550 | -4.43% (`0/11`) | **-5.61%** | 0.7895 | +1.18% (`14/21`) | 333.4 |
| `B6` | `B6_poco_univariate_ms_mlp` | Yes | 0.1795 | -6.04% (`4/32`) | 0.3459 | -1.75% (`2/11`) | **-3.90%** | 0.7898 | +1.14% (`15/21`) | 42.0 |
| `B6` | `B6_poco_film` | No | 0.1781 | -5.24% (`6/32`) | 0.3461 | -1.80% (`3/11`) | **-3.52%** | 0.7860 | +1.61% (`15/21`) | 58.0 |
| `B6` | `B6_ray_gordus_state_gain` | No | 0.1804 | -6.61% (`5/32`) | 0.3411 | -0.34% (`2/11`) | **-3.48%** | 0.7910 | +0.99% (`7/21`) | 4.0 |
| `B6` | `B6_capacity_matched_graphfree_control` | No | 0.1803 | -6.52% (`4/32`) | 0.3416 | -0.47% (`2/11`) | **-3.50%** | 0.8035 | -0.58% (`11/21`) | 4.7 |

### 0.2 Trajectory `post_R2` and `delta_R2` with Separate Animal-Cluster 95% Bootstrap CIs

| Model ID | `Seyed2025 post_R2 [95% CI]` | `Seyed2025 delta_R2 [95% CI]` | `Toyo2024 post_R2 [95% CI]` | `Toyo2024 delta_R2 [95% CI]` | `Yemini2021 post_R2 [95% CI]` | `Yemini2021 delta_R2 [95% CI]` |
|---|---|---|---|---|---|---|
| `B1_persistence` | 0.8522 [0.8337, 0.8703] | -0.0005 [-0.0011, -0.0001] | 0.7186 [0.5888, 0.8457] | -0.0000 [-0.0000, -0.0000] | 0.3413 [0.1236, 0.4854] | -0.0083 [-0.0146, -0.0041] |
| `B2_neuron_stim_mean` | 0.8642 [0.8453, 0.8821] | 0.0810 [0.0602, 0.1036] | 0.7189 [0.5898, 0.8451] | 0.0010 [-0.0085, 0.0051] | 0.4218 [0.2270, 0.5381] | 0.1149 [0.0747, 0.1666] |
| `B3_self_arfir_spline` | 0.8902 [0.8764, 0.9025] | 0.2571 [0.2075, 0.2984] | 0.7463 [0.6353, 0.8554] | 0.0982 [0.0641, 0.1148] | 0.5148 [0.3852, 0.5916] | 0.2573 [0.1941, 0.3136] |
| `B4_ridge_self_stim` | 0.8873 [0.8732, 0.8999] | 0.2372 [0.1807, 0.2840] | 0.7470 [0.6364, 0.8555] | 0.1008 [0.0669, 0.1173] | 0.4967 [0.3649, 0.5762] | 0.2295 [0.1596, 0.2910] |
| `B4_ridge_cross_neuron` | 0.8875 [0.8733, 0.9001] | 0.2383 [0.1838, 0.2839] | 0.7469 [0.6361, 0.8556] | 0.1005 [0.0668, 0.1168] | 0.4970 [0.3615, 0.5794] | 0.2300 [0.1642, 0.2880] |
| `B4_mlp_repaired_self_stim` | 0.8904 [0.8760, 0.9035] | 0.2584 [0.2026, 0.3035] | 0.7410 [0.6236, 0.8558] | 0.0797 [0.0655, 0.0858] | 0.4956 [0.3625, 0.5710] | 0.2278 [0.1431, 0.3055] |
| `B4_mlp_repaired_cross_neuron` | 0.8913 [0.8765, 0.9045] | 0.2644 [0.2116, 0.3082] | 0.7396 [0.6203, 0.8551] | 0.0744 [0.0627, 0.0793] | 0.4931 [0.3499, 0.5774] | 0.2241 [0.1453, 0.3007] |
| `B5_pop_rrr` | 0.8834 [0.8695, 0.8958] | 0.2107 [0.1646, 0.2486] | 0.7462 [0.6353, 0.8553] | 0.0981 [0.0639, 0.1147] | 0.5043 [0.3731, 0.5846] | 0.2413 [0.1833, 0.2922] |
| `B5_lds_stim` | 0.8830 [0.8689, 0.8959] | 0.2084 [0.1609, 0.2468] | 0.7462 [0.6352, 0.8552] | 0.0980 [0.0634, 0.1146] | 0.5022 [0.3675, 0.5845] | 0.2379 [0.1827, 0.2870] |
| `B5_causal_gru` | 0.8813 [0.8650, 0.8969] | 0.1964 [0.1642, 0.2308] | 0.7376 [0.6204, 0.8528] | 0.0675 [0.0452, 0.0771] | 0.4883 [0.3383, 0.5776] | 0.2167 [0.1682, 0.2618] |
| `B5_causal_tcn` | 0.8811 [0.8647, 0.8963] | 0.1954 [0.1587, 0.2292] | 0.7368 [0.6206, 0.8506] | 0.0646 [0.0363, 0.0778] | 0.5006 [0.3490, 0.5892] | 0.2355 [0.1856, 0.2916] |
| `B6_poco_univariate_ms_mlp` | 0.8830 [0.8683, 0.8961] | 0.2083 [0.1552, 0.2538] | 0.7427 [0.6276, 0.8555] | 0.0855 [0.0648, 0.0957] | 0.4990 [0.3482, 0.5878] | 0.2330 [0.1839, 0.2829] |
| `B6_poco_film` | 0.8866 [0.8714, 0.8992] | 0.2328 [0.1706, 0.2896] | 0.7424 [0.6267, 0.8554] | 0.0844 [0.0652, 0.0936] | 0.5035 [0.3613, 0.5878] | 0.2400 [0.1812, 0.2971] |
| `B6_ray_gordus_state_gain` | 0.8829 [0.8687, 0.8962] | 0.2075 [0.1566, 0.2493] | 0.7463 [0.6353, 0.8554] | 0.0984 [0.0646, 0.1148] | 0.4976 [0.3607, 0.5827] | 0.2309 [0.1641, 0.2827] |
| `B6_capacity_matched_graphfree_control` | 0.8830 [0.8690, 0.8957] | 0.2080 [0.1644, 0.2440] | 0.7459 [0.6347, 0.8553] | 0.0970 [0.0639, 0.1132] | 0.4967 [0.3640, 0.5784] | 0.2296 [0.1714, 0.2808] |

### 0.3 Subpopulation & Residual Failure Diagnosis (Empirical Guidance for `M3` Round 1)

Inspecting out-of-fold residual errors of the frozen self-history reference baseline (`B4_ridge_self_stim`) vs the strongest Round-0 network baseline (`B4_mlp_repaired_cross_neuron`, `I_macro = +0.08%`):

1. **Cell-Class Stratification (`sensory` vs `interneuron` vs `command_motor`)**:
   - `seyedolmohadesin2025` (`B4_ridge_self_stim` `E_d` by tier): `sensory = 0.3789`, `interneuron = 0.1051`, `command_motor = 0.1095` (vs `B4_mlp_repaired_cross_neuron`: `sensory = 0.3685`, `interneuron = 0.1006`, `command_motor = 0.1087`).
   - `toyoshima2024` (`B4_ridge_self_stim` `E_d` by tier): `sensory = 0.5198`, `interneuron = 0.3601`, `command_motor = 0.3854` (vs `B4_mlp_repaired_cross_neuron`: `sensory = 0.5427`, `interneuron = 0.3743`, `command_motor = 0.3949`).
   - `yemini2021` (`B4_ridge_self_stim` `E_d` by tier): `sensory = 1.4219`, `interneuron = 0.8813`, `command_motor = 0.5065` (vs `B4_mlp_repaired_cross_neuron`: `sensory = 1.3712`, `interneuron = 0.8383`, `command_motor = 0.5932`).
2. **Sex Dimorphism Stratification (`hermaphrodite` vs `male` in `seyedolmohadesin2025`)**:
   - `B4_ridge_self_stim`: `hermaphrodite = 0.1920`, `male = 0.1537`.
   - `B4_mlp_repaired_cross_neuron`: `hermaphrodite = 0.1832`, `male = 0.1515`.
3. **Temporal Horizon Breakdown — Stimulus Onset Transient (`[0, 3)s`) vs Slow-State Drift (`[3, 10)s`)**:
   - `seyedolmohadesin2025` (`B4_ridge_self_stim`): early `[0, 3)s` `E_d = 0.0759` vs late `[3, 10)s` `E_d = 0.2114` (`late/early ratio = 2.78x`; `B4_mlp_repaired_cross_neuron` late `E_d = 0.2052`).
   - `toyoshima2024` (`B4_ridge_self_stim`): early `[0, 3)s` `E_d = 0.3260` vs late `[3, 10)s` `E_d = 0.3460` (`late/early ratio = 1.06x`; `B4_mlp_repaired_cross_neuron` late `E_d = 0.3564`).
   - `yemini2021` (`B4_ridge_self_stim`): early `[0, 3)s` `E_d = 0.3369` vs late `[3, 10)s` `E_d = 0.9972` (`late/early ratio = 2.96x`; `B4_mlp_repaired_cross_neuron` late `E_d = 0.9784`).
4. **Actionable Mechanistic Diagnosis & Headroom for `M3` (`H1` Slow-State, `H2` Bilinear Gating, `H3` Signed Connectome Priors)**:
   - **Where errors concentrate by neuron class**: Across all three immobilized datasets, normalized post-stimulus MSE (`E_d`) is highest in **sensory neurons** (`Seyed = 0.3789`, `Toyo = 0.5198`, `Yemini = 1.4219`), reflecting large stimulus-evoked transients and trial-to-trial gain variability. In `seyedolmohadesin2025`, `interneuron` (`0.1051`) and `command_motor` (`0.1095`) errors are ~3.5x smaller than sensory errors in absolute normalized MSE, whereas in `toyoshima2024` (high-frequency NaCl steps) `interneuron` (`0.3601`) and `command_motor` (`0.3854`) carry ~70% as much normalized error as sensory neurons, and in `yemini2021` `interneuron` (`0.8813`) is 1.74x higher than `command_motor` (`0.5065`).
   - **Temporal horizon & H1 slow-state headroom**: Residual error grows sharply in the **late `[3, 10)s` window** in `seyedolmohadesin2025` (`2.78x` late/early: `0.2114` vs `0.0759`) and `yemini2021` (`2.96x` late/early: `0.9972` vs `0.3369`), while `toyoshima2024` is nearly uniform (`1.06x`: `0.3460` vs `0.3260`) due to rapid NaCl step cycling. This directly points to **H1 slow internal-state drift** (`AIA/AIB/AIY/AIZ` + `AVA/AVB/RIM/RIB`) as the primary target for late-window error reduction in `Seyedolmohadesin` and `Yemini`.
   - **Nonlinear self-history & H2 bilinear gating headroom**: `B4_mlp_repaired_self_stim` achieves the highest `I_macro` (`+0.58%` vs `B4_ridge_self_stim`, `+2.02%` vs `B3_self_arfir_spline`) with strong gains on `Seyedolmohadesin` (`+3.39%`, `23/32` worms) and `Yemini` (`+2.01%`, `15/21` worms), but overfits on smaller-N `Toyoshima` (`-2.23%`, `5/11` worms). Crucially, `B6_poco_film` outperforms its univariate control `B6_poco_univariate_ms_mlp` on both `Seyedolmohadesin` (`0.1781` vs `0.1795`) and `Yemini` (`0.7860` vs `0.7898`), providing an early empirical signal that **multiplicative/FiLM state-by-stimulus gating (H2)** extracts cross-neuron signal that additive concatenation misses — while naive unconstrained cross-neuron features (`B4_mlp_repaired_cross_neuron`: `+0.08%`, `B5_causal_gru`: `-5.17%`, `B5_causal_tcn`: `-5.61%`) suffer variance inflation under 30%–50% neuron missingness.
   - **H3 signed connectome & sexual dimorphism headroom**: In `seyedolmohadesin2025`, hermaphrodites exhibit 25% higher residual error than males (`0.1920` vs `0.1537` under `B4_ridge_self_stim`), and `B4_ridge_cross_neuron` already improves `19/32` Seyed worms (`+0.11%`), `7/11` Toyo worms (`-0.04%`), and `12/21` Yemini worms (`+0.39%`) with only 8 coarse circuit features. Replacing unconstrained dense cross-neuron projections with **missingness-precision-weighted (`kappa`-shrunk) receptor-polarity-signed connectome priors (`Cook 2019` H/M chem/gap + `Ripoll-Sánchez 2023` 92 NPP-GPCR + `CeNGEN` `EAT-4` `GLR+`/`AVR-` signs)** and low-rank bilinear `State x Stimulus` gating (`H2`) is the highest-leverage next step for **`M3` Round 1** (paired with all 7 confound controls: `global_mean`, `obs_mask`, `trial_order`, `prev_stim`, `repeat_adaptation`, `elapsed_time`, `sex_batch`).


---

## Round 1 (`M3` H1 Slow-State Increment & 7 Confound Controls — Residualized Architecture)

- **Code**: `models/m3_round1_h1.py` | **Summary**: `results/m3_round1_h1_summary.json` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, zero `dataset_id` branches)
- **Peak RSS**: `2425.0 MB` (`2.37 GiB`, strictly `< 3.0 GiB` via 1-dataset-at-a-time processing + `gc.collect()`).
- **Leakage Invariance Audit (`M3R1_03_h1_slow_state_circuit` on `yemini2021`)**: **PASSED** — future-frame poisoning `t>=0` max abs diff `= 0.00e+00` (`target_shift = 1.00e+06`), OOF validation tamper weight/pred diff `= 0.00e+00`, bilateral exclusion diffs `AVAR->AVAL = 0.0`, `AIYR->AIYL = 0.0`, `ASER->ASEL = 0.0`, `AWCOFF->AWCON = 0.0`, target-self network pathway diff `= 0.0`, and asymmetric chemosensory preservation verified (`AIYL` sees `ASEL`/`ASER` contrast delta `= 1.4142`).

### 1.1 Cross-Source Primary Metric Summary (`E_d`, `I_d`, `I_macro` with 1000-Resample Animal Bootstrap 95% CIs)

| Model ID | Type | `Seyed2025 E_d` | `Seyed2025 I_d vs B4 [95% CI]` (`Worms`) | `Toyo2024 E_d` | `Toyo2024 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` Main 2 vs B4 [95% CI]** | `Yemini2021 E_d` | `Yemini2021 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` All 3 vs B4 [95% CI]** | Train Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4_ridge_self_stim` *(REF)* | Self-only | 0.169241 | +0.00% (`0/32`) | 0.339974 | +0.00% (`0/11`) | **+0.00%** | 0.798889 | +0.00% (`0/21`) | **+0.00%** | 3.7 |
| `M3R1_01_self_anchor_multiscale` | Self-only anchor | 0.166983 | +1.33% [+0.91%, +1.76%] (`30/32`) | 0.340323 | -0.10% [-0.22%, +0.14%] (`5/11`) | **+0.62% [+0.41%, +0.86%]** | 0.768538 | +3.80% [+1.86%, +4.71%] (`18/21`) | **+1.68% [+1.02%, +2.01%]** | 6.5 |
| `M3R1_02_h1_confounds_only` | Self + 7 confounds | 0.167000 | +1.32% [+0.75%, +1.87%] (`25/32`) | 0.339912 | +0.02% [-0.34%, +0.33%] (`8/11`) | **+0.67% [+0.35%, +0.99%]** | 0.772943 | +3.25% [+1.34%, +4.06%] (`18/21`) | **+1.53% [+0.87%, +1.87%]** | 17.6 |
| `M3R1_03_h1_slow_state_circuit` | + H1 circuit (Ridge) | 0.167214 | +1.20% [+0.56%, +1.84%] (`25/32`) | 0.340191 | -0.06% [-0.42%, +0.25%] (`7/11`) | **+0.57% [+0.22%, +0.94%]** | 0.774176 | +3.09% [+0.59%, +4.16%] (`14/21`) | **+1.41% [+0.54%, +1.83%]** | 40.3 |
| `M3R1_04_h1_slow_state_mlp_residual` | + H1 circuit (MLP) | 0.167298 | +1.15% [+0.51%, +1.80%] (`23/32`) | 0.339967 | +0.00% [-0.30%, +0.27%] (`8/11`) | **+0.57% [+0.24%, +0.92%]** | 0.773106 | +3.23% [+1.37%, +4.05%] (`18/21`) | **+1.46% [+0.78%, +1.80%]** | 107.1 |

### 1.2 Incremental Contrast Table (`vs M3R1_01` Self-Anchor & `vs M3R1_02` Confounds-Only)

| Model ID | `Seyed2025 I_d vs M3R1_02 [95% CI]` | `Toyo2024 I_d vs M3R1_02 [95% CI]` | `Yemini2021 I_d vs M3R1_02 [95% CI]` | **`I_macro` Main 2 vs M3R1_02 [95% CI]** | **`I_macro` All 3 vs M3R1_02 [95% CI]** | `Seyed post_R2 / delta_R2` | `Toyo post_R2 / delta_R2` | `Yemini post_R2 / delta_R2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M3R1_01_self_anchor_multiscale` | +0.01% [-0.31%, +0.44%] | -0.12% [-0.26%, +0.15%] | +0.57% [-0.20%, +1.10%] | -0.06% [-0.24%, +0.19%] | +0.15% [-0.11%, +0.40%] | 0.8915 / 0.2654 | 0.7465 / 0.0989 | 0.5228 / 0.2696 |
| `M3R1_02_h1_confounds_only` | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | 0.8915 / 0.2658 | 0.7466 / 0.0995 | 0.5186 / 0.2631 |
| `M3R1_03_h1_slow_state_circuit` | -0.13% [-0.31%, +0.05%] (`10/32`) | -0.08% [-0.11%, -0.05%] (`2/11`) | -0.16% [-1.10%, +0.29%] (`9/21`) | **-0.11% [-0.19%, -0.02%]** | **-0.12% [-0.44%, +0.04%]** | 0.8916 / 0.2664 | 0.7464 / 0.0986 | 0.5161 / 0.2593 |
| `M3R1_04_h1_slow_state_mlp_residual` | -0.18% [-0.45%, +0.07%] (`12/32`) | -0.02% [-0.10%, +0.06%] (`3/11`) | -0.02% [-0.05%, +0.01%] (`3/21`) | **-0.10% [-0.24%, +0.03%]** | **-0.07% [-0.17%, +0.01%]** | 0.8917 / 0.2669 | 0.7466 / 0.0993 | 0.5186 / 0.2631 |

### 1.3 Subpopulation, Horizon & Sex Diagnostics (Round 1)

1. **Strengthened Self-Anchor (`M3R1_01`) Captures Large Real Self-Dynamics Headroom**:
   - Compared to frozen `B4_ridge_self_stim`, adding multi-scale `[-10,-5)s`, `[-5,-2)s`, `[-2,0)s` means/slopes/last-value + causal orthonormal spline basis + shrunk per-neuron residual coupling improves `30/32` Seyed worms (`+1.33%`, `delta_R2: 0.2372 -> 0.2654`) and `18/21` Yemini worms (`+3.80%`, `delta_R2: 0.2295 -> 0.2696`), lifting `I_macro` (Main 2) to **`+0.62% [+0.41%, +0.86%]`** and `I_macro` (All 3) to **`+1.68% [+1.02%, +2.01%]`** — already surpassing every Round 0 model (`B4_mlp_repaired_self_stim` was `+0.58%` Main 2 with `-2.23%` Toyo regression).
   - Error reductions in `M3R1_01` span all tiers on `Seyed2025` (`sensory: 0.3789 -> 0.3751`, `interneuron: 0.1051 -> 0.1037`, `command_motor: 0.1095 -> 0.1053`) and `Yemini2021` (`sensory: 1.4219 -> 1.3801`, `interneuron: 0.8813 -> 0.8408`, `command_motor: 0.5065 -> 0.4826`), and reduce both early `[0,3)s` (`0.0759 -> 0.0749` Seyed; `0.3369 -> 0.3182` Yemini) and late `[3,10)s` (`0.2114 -> 0.2086` Seyed; `0.9972 -> 0.9619` Yemini) errors.
2. **All 7 Confound Controls (`M3R1_02`)**:
   - Adding the 7 causal confounds lifts `Toyo2024` from `-0.10%` to **`+0.02%`** (`9/11` worms improved vs `M3R1_01`, `I_d = +0.12%`), making `M3R1_02` the **only model so far with strictly positive point gain on all 3 datasets** (`I_macro` Main 2 `= +0.67% [+0.35%, +0.99%]`, `worst_main_source_I_d = +0.02%`). On `Seyed2025`, confounds improve `20/32` worms (`-0.01%` net) and lower `hermaphrodite` error (`0.1901 -> 0.1899`) and `sensory` error (`0.3751 -> 0.3744`).
3. **Key H1 Mechanistic Finding — Additive Stimulus-Independent Circuit Slow-State Adds Zero Net Gain**:
   - Once self-history (`M3R1_01`) and all 7 confounds (`M3R1_02`) are properly controlled, **purely additive excluded-self circuit slow-state** (`M3R1_03_h1_slow_state_circuit`: `I_macro` Main 2 vs `M3R1_02` `= -0.11% [-0.19%, -0.02%]`) and **stimulus-unconditioned nonlinear residual MLP** (`M3R1_04_h1_slow_state_mlp_residual`: `-0.10% [-0.24%, +0.03%]`) do **not** reduce equal-animal `E_d` beyond `M3R1_02` (tight upper 95% CI `< +0.07%` on `Seyed2025`, `< +0.06%` on `Toyo2024`, `< +0.29%` on `Yemini2021`).
   - Crucially, note the diagnostic dissociation on `Seyed2025`: `M3R1_04` achieves the highest pooled `post_R2` (`0.8917`) and `delta_R2` (`0.2669`) and lowers `premotor_command` MSE (`0.1368 -> 0.1361`), while `M3R1_03` lowers `Yemini` `sensory` MSE (`1.3631 -> 1.3607`) — yet **stimulus-blind additive slow-state shifts** slightly worsen equal-animal `E_d` across multi-cue panels (`Seyed` 8 cues, `Yemini` 3 cues) because attractive vs aversive cues induce **opposite-sign state-gated responses** that cancel out in an additive `W @ z_state` term!
   - **Sex dimorphism (`Seyed2025`)**: The hermaphrodite-vs-male error gap persists across all Round 1 models (`M3R1_02`: `hermaphrodite = 0.1899` vs `male = 0.1513`, `+25.5%` higher error in hermaphrodites), indicating that sex-invariant circuit features do not yet resolve sex-specific chemosensory routing (e.g., `1uM ascr#3` pheromone `E_d = 0.2360`).

### 1.4 Concrete Mechanistic Design Recommendations for M3 Round 2 (`H2`: Low-Rank Bilinear `State × Stimulus` Gating)

Round 1 rules out naive additive slow-state shifts (`I_macro vs M3R1_02 <= +0.03%` 95% CI upper bound) and establishes `M3R1_02_h1_confounds_only` (`+0.67%` Main 2 vs `B4`) as our rock-solid two-stage residual anchor. For **Round 2 (H2)**, we should directly test **multiplicative / bilinear `State × Stimulus` gating**:
1. **Anchor on `M3R1_02_h1_confounds_only`** (strictly inside each fold) so every H2 model starts at zero variance inflation.
2. **Low-Rank Bilinear `State × Stimulus` Gating (`H2_bilinear_state_x_stim`)**:
   - Interact compact excluded-self slow-state `z_state` (command `AVA/AVB/RIM/RIB`, Layer-1 `AIA/AIB/AIY/AIZ`, asymmetric `ASEL/ASER/AWCON/AWCOFF`, top-4 causal PCs) with **stimulus drive** (`prior_norm` + stimulus one-hot) via a low-rank bilinear factorization (`rank = hp["bilinear_rank"] = 4`, `l2 = hp["bilinear_l2_penalty"] = 60.0`), so pre-stimulus internal state modulates the *gain and temporal phase* of stimulus-evoked trajectories rather than adding a stimulus-blind offset!
3. **Capacity-Matched Additive Control (`H2_additive_capacity_matched`)**:
   - Exact same parameter count / rank / regularization on `[z_state, stim]` WITHOUT multiplicative `State × Stimulus` interaction, to isolate true gating from capacity.
4. **Layer Dissociation & Difference-in-Differences (`Delta_inter - Delta_sensory`)**:
   - Explicitly test `Sensory -> Interneuron` and `Interneuron <-> Interneuron` gating vs uniform gating, computing formal animal-level DiD (`I_d(interneuron) - I_d(sensory)`).
5. **Buffer Pseudo-Event & Within-Condition Permutation Controls**:
   - Evaluate `H2_bilinear` on `45s` buffer pseudo-events (where stimulus-gating gain MUST collapse toward 0 vs real chemical cues) and within-condition trial-state permutations to verify genuine trial-specific state gating rather than static neuron-stimulus memorization.


---

## Round 2 (`M3` H2 Bilinear `State × Stimulus` Gating, Capacity-Matched Additive Control & Layer Dissociation)

- **Code**: `models/m3_round2_h2.py` | **Summary**: `results/m3_round2_h2_summary.json` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, `bilinear_rank = 4`, `bilinear_l2_penalty = 60.0`, `poco_film_dim = 24`, zero `dataset_id` branches)
- **Peak RSS**: `2930.5 MB` (`2.86 GiB`, strictly `< 3.0 GiB` via 1-dataset-at-a-time processing + `gc.collect()`).
- **Leakage Invariance Audit (`M3R2_02_h2_bilinear_hierarchical` on `yemini2021`)**: **PASSED** — future-frame poisoning `t>=0` max abs diff `= 0.00e+00` (`target_shift = 1.00e+06`), OOF validation tamper weight/pred diff `= 0.00e+00`, bilateral exclusion diffs `AVAR->AVAL = 0.0`, `AIYR->AIYL = 0.0`, `ASER->ASEL = 0.0`, `AWCOFF->AWCON = 0.0`, target-self network pathway diff `= 0.0`, and asymmetric chemosensory preservation verified (`AIYL` sees `ASEL`/`ASER` contrast delta `= 1.4142`).

### 2.1 Cross-Source Primary Metric Summary (`E_d`, `I_d`, `I_macro` with 1000-Resample Animal Bootstrap 95% CIs)

| Model ID | Type | `Seyed2025 E_d` | `Seyed2025 I_d vs B4 [95% CI]` (`Worms`) | `Toyo2024 E_d` | `Toyo2024 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` Main 2 vs B4 [95% CI]** | `Yemini2021 E_d` | `Yemini2021 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` All 3 vs B4 [95% CI]** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M3R1_02_h1_confounds_only` *(ANCHOR)* | Self + 7 confounds | 0.167000 | +1.32% [+0.75%, +1.87%] (`25/32`) | 0.339912 | +0.02% [-0.34%, +0.33%] (`8/11`) | **+0.67% [+0.35%, +0.99%]** | 0.772943 | +3.25% [+1.34%, +4.06%] (`18/21`) | **+1.53% [+0.87%, +1.87%]** |
| `M3R2_01_h2_bilinear_uniform` | Uniform Bilinear RRR (`r=4`) | 0.166995 | +1.33% [+0.48%, +2.15%] (`24/32`) | 0.340105 | -0.04% [-0.37%, +0.24%] (`8/11`) | **+0.64% [+0.20%, +1.08%]** | 0.776051 | +2.86% [+1.41%, +3.86%] (`16/21`) | **+1.38% [+0.81%, +1.82%]** |
| `M3R2_02_h2_bilinear_hierarchical` | Hierarchical Bilinear RRR | 0.167195 | +1.21% [+0.36%, +1.95%] (`23/32`) | 0.340187 | -0.06% [-0.41%, +0.22%] (`7/11`) | **+0.57% [+0.14%, +0.98%]** | 0.774647 | +3.03% [+1.75%, +3.78%] (`18/21`) | **+1.39% [+0.90%, +1.77%]** |
| `M3R2_03_h2_capacity_matched_additive` | Capacity-Matched Additive (`r=4`) | 0.167147 | +1.24% [+0.66%, +1.80%] (`24/32`) | 0.339811 | +0.05% [-0.31%, +0.40%] (`9/11`) | **+0.64% [+0.33%, +0.98%]** | 0.768510 | +3.80% [+1.84%, +4.70%] (`17/21`) | **+1.70% [+1.02%, +2.04%]** |
| `M3R2_04_h2_film_gating_residual` | **Compact FiLM Gate Residual** | **0.166596** | **+1.56% [+0.67%, +2.50%]** (`26/32`) | **0.339891** | **+0.02% [-0.33%, +0.36%]** (`9/11`) | **+0.79% [+0.32%, +1.31%]** | **0.770208** | **+3.59% [+2.03%, +4.33%]** (`17/21`) | **+1.73% [+1.09%, +2.14%]** |

### 2.2 Incremental Contrasts (`vs M3R1_02` Anchor & `vs M3R2_03` Capacity-Matched Additive) & Capacity Audit

| Model ID | Params (`Seyed/Toyo/Yem`) | `I_macro` Main 2 vs `M3R1_02` [95% CI] | `I_macro` All 3 vs `M3R1_02` [95% CI] | `I_macro` Main 2 vs `M3R2_03` [95% CI] | `Seyed` vs `M3R1_02` (`Worms`) | `Toyo` vs `M3R1_02` (`Worms`) | `Yemini` vs `M3R1_02` (`Worms`) | `Seyed post_R2 / delta_R2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M3R2_01_h2_bilinear_uniform` | `480 / 192 / 240` (`r=4`) | -0.03% [-0.28%, +0.20%] | -0.15% [-0.49%, +0.33%] | +0.00% [-0.26%, +0.22%] | +0.00% (`15/32`) | -0.06% (`3/11`) | -0.40% (`10/21`) | 0.8913 / 0.2645 |
| `M3R2_02_h2_bilinear_hierarchical` | `1080 / 432 / 540` | -0.10% [-0.37%, +0.13%] | -0.14% [-0.44%, +0.28%] | -0.07% [-0.37%, +0.17%] | -0.12% (`16/32`) | -0.08% (`3/11`) | -0.22% (`9/21`) | 0.8913 / 0.2645 |
| `M3R2_03_h2_capacity_matched_additive` | `480 / 192 / 240` (`r=4`) | -0.03% [-0.10%, +0.04%] | +0.17% [-0.04%, +0.32%] | +0.00% | -0.09% (`15/32`) | +0.03% (`8/11`) | +0.57% (`14/21`) | 0.8916 / 0.2664 |
| `M3R2_04_h2_film_gating_residual` | `2046 / 1902 / 1926` | **+0.12% [-0.20%, +0.50%]** | **+0.20% [-0.05%, +0.51%]** | **+0.15% [-0.18%, +0.53%]** | **+0.24%** (`17/32`) | **+0.01%** (`6/11`) | **+0.35% [+0.09%, +0.98%]** (`16/21`) | **0.8918 / 0.2679** |

### 2.3 Mandatory H2 Mechanistic Controls (`PROJECT.md F9`)

1. **Within-Condition Permutation Control (`20` deterministic permutations of `z_state` within identical `stimulus_id`)**:
   - **`M3R2_01_h2_bilinear_uniform`**: Breaking trial-specific state–stimulus alignment drops `I_macro` by **`+0.22%` (Main 2)** and **`+0.27%` (All 3)** — degrading **`20/20` (100%)** permutations on `Seyed2025` (`+0.003% -> -0.417%`, drop **`+0.42%`**), **`19/20` (95%)** on `Toyo2024` (`drop +0.03%`), and **`17/20` (85%)** on `Yemini2021` (`drop +0.36%`).
   - **`M3R2_02_h2_bilinear_hierarchical`**: Breaking trial-specific state–stimulus alignment drops `I_macro` by **`+0.24%` (Main 2)** and **`+0.35%` (All 3)** — degrading **`20/20` (100%)** permutations on `Seyed2025` (`drop +0.40%`), **`20/20` (100%)** on `Toyo2024` (`drop +0.08%`), and **`19/20` (95%)** on `Yemini2021` (`drop +0.55%`).
   - **Interpretation**: Genuine trial-specific `State × Stimulus` coupling is robustly present across all 3 sources (`85–100%` of permutations degrade performance), proving the bilinear gating signal is NOT static stimulus-mean memorization; however, dense unguided outer-product Ridge incurs estimation variance that offsets its `+0.24%–+0.42%` real signal, whereas bounded multiplicative FiLM (`M3R2_04`) begins to convert that signal into net OOF gains across all 3 datasets!
2. **45s Buffer Pseudo-Event Negative Control**:
   - **Natural buffer windows (`buffer_E_d` vs `M3R1_02`)**: Gain collapses to `~0%` (`M3R2_01`: `-0.06%` Yemini, `-0.03%` Seyed, `+0.006%` Toyo; `M3R2_02`: `-0.04%` Yemini, `-0.02%` Seyed, `+0.002%` Toyo).
   - **Sham-stimulus buffer probe** (applying trained H2 gating residual to real buffer `z_state` with sham chemical stimulus): Strongly negative on ALL 3 sources (`M3R2_01`: `-1.31% [-3.30%, -0.76%]` Yemini, `-1.26% [-1.91%, -0.72%]` Seyed, `-0.11% [-0.22%, -0.03%]` Toyo; `M3R2_02`: `-2.05%` Yemini, `-0.83%` Seyed, `-0.19%` Toyo — all 95% CIs strictly `< 0`), confirming gating specificity to true stimulus-evoked trajectories.
3. **Layer Dissociation Difference-in-Differences (`DiD`, 1000-resample animal bootstrap 95% CIs)**:
   - `M3R2_01 vs M3R2_03` (Main 2): `DiD(Inter - Sensory) = -0.05% [-0.38%, +0.28%]`, `DiD(Cmd - Sensory) = -0.37% [-0.69%, +0.01%]` (uniform bilinear helps `Seyed` `sensory` `+0.28%` and `interneuron` `+0.16%`, but hurts `command_motor` `-0.46%`).
   - `M3R2_02 vs M3R2_03` (Main 2): `DiD(Inter - Sensory) = +0.06% [-0.50%, +0.63%]`, `DiD(Cmd - Sensory) = -0.03% [-0.48%, +0.46%]` — hierarchical tier separation eliminates the `Seyed` `command_motor` variance penalty (`-0.46% -> -0.01%`), flipping `DiD(Inter - Sensory)` positive (`+0.08%` Seyed, `+0.04%` Toyo), though all 95% CIs span zero.
4. **Best-Model (`M3R2_04`) Subpopulation, Horizon & Sex Breakdown**:
   - `M3R2_04_h2_film_gating_residual` is the **new overall best Dev model** (`I_macro` Main 2 `= +0.79% [+0.32%, +1.31%]` vs `B4`, All 3 `= +1.73% [+1.09%, +2.14%]`), improving ALL 3 sources simultaneously over both `B4` and `M3R1_02` (`Seyed +0.24%`, `Toyo +0.01%`, `Yemini +0.35% [+0.09%, +0.98%]`).
   - On `Seyed2025`, `M3R2_04` reduces late `[3,10)s` error (`0.2085 -> 0.2079`), `sensory` (`0.3744 -> 0.3726`), `interneuron` (`0.1038 -> 0.1034`), `premotor_command` (`0.1368 -> 0.1361`), and both `hermaphrodite` (`0.1899 -> 0.1894`) and `male` (`0.1513 -> 0.1510`) errors — while the `+25.4%` hermaphrodite-vs-male error gap (`0.1894` vs `0.1510`) remains open for sex-specific connectome routing!

### 2.4 Concrete Mechanistic Design for M3 Round 3 (`H3`: Receptor-Polarity-Signed Multi-Layer Connectome Priors + Sex Dimorphism)

Round 2 proves two critical facts: (i) genuine trial-specific `State × Stimulus` coupling exists (`100%` permutation drop on `Seyed2025`, `+0.12%` Main 2 / `+0.20%` All 3 OOF gain under bounded FiLM `M3R2_04`), and (ii) dense unguided cross-neuron projections suffer variance inflation under 30%–50% neuron missingness. For **Round 3 (H3)**, replace unguided population projections with **biologically structured, missingness-precision-weighted (`kappa`) receptor-polarity-signed multi-layer connectome priors** anchored on `M3R1_02` (and tested both as additive connectome-filtered drives and connectome-gated FiLM/bilinear):
1. **4-Layer Connectome Ablation Ladder (`data/connectomes/connectome_priors_bundle.npz`)** under unified `configs/lambda_global.json` (`connectome_l2_penalty = 60.0`, `wireless_rank = 6`, `wireless_l2_multiplier = 2.5`, `connectome_taus_sec = [2.0, 5.0, 10.0]`, strictly excluding target `i` & bilateral `i_LR` and preserving `ASEL/ASER`, `AWCON/OFF`):
   - **`M3R3_01_h3_wired_unsigned`**: `Cook 2019` wired chemical + gap junctions (`W_chem + W_gap`), unsigned (`abs`), multi-tau causal convolution (`[2, 5, 10]s`) with `kappa` missingness-precision weighting.
   - **`M3R3_02_h3_wired_signed_cengen`**: Multiply `W_chem` by **`CeNGEN` receptor polarity sign matrix `S_sign`** (`EAT-4` Glu `+1` on `GLR/NMR` e.g. `ASH->AVA`, `-1` on `AVR-14/15/GLC` e.g. `AWC->AIY`), separating excitatory vs inhibitory synaptic drives (`W_chem_pos`, `W_chem_neg`, `W_gap`).
   - **`M3R3_03_h3_multilayer_wired_plus_wireless_npp`**: Add **`Ripoll-Sánchez 2023` 92 NPP-GPCR short/mid-range wireless neuropeptide + monoamine layers** (low-rank `wireless_rank = 6`, `2.5x` L2 multiplier), plus unique-vs-shared variance decomposition across correlated wired/wireless graphs.
   - **`M3R3_04_h3_sex_dimorphic_connectome_film`**: Select **`Cook 2019` sex-specific matrices (`W_chem_herm`/`W_gap_herm` for hermaphrodites vs `W_chem_male`/`W_gap_male` for males)** inside `seyedolmohadesin2025` (zero leakage, `sex` is a known pre-recording covariate!) combined with bounded FiLM `State × Stimulus` gating (`M3R2_04` winner), directly testing whether sex-specific wiring narrows the `hermaphrodite (0.1894)` vs `male (0.1510)` gap and `1uM ascr#3` pheromone error!
2. **Pre-M4 Diagnostic Readiness**: Also run F11 robustness checks (artificial extra missingness dropout curve + `0.5x/1x/2x` hyperparameter neighborhood stability) on the winning Round 3 candidate to prepare for M4 (`>=199` degree/distance/cell-type-matched retrained random null graphs + `Randi unc-31` + `Baskoylu Copper` + §8 route freeze).


---

## Round 3 (`M3` H3 Receptor-Polarity-Signed Multi-Layer Connectome Priors & Sex Dimorphism)

- **Code**: `models/m3_round3_h3.py` | **Summary**: `results/m3_round3_h3_summary.json` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, `connectome_l2_penalty = 60.0`, `wireless_rank = 6`, `wireless_l2_multiplier = 2.5`, `bilinear_rank = 4`, `missingness_precision_kappa = 4.0`, zero `dataset_id` branches)
- **Peak RSS**: `3012.3 MB` (`2.94 GiB`, strictly `< 3.0 GiB` via 1-dataset-at-a-time processing + `gc.collect()`).
- **Leakage Invariance Audit (`M3R3_03_h3_multilayer_wired_plus_wireless_npp` on `yemini2021`)**: **PASSED** — future-frame poisoning `t>=0` max abs diff `= 0.00e+00` (`target_shift = 1.00e+06`), OOF validation tamper weight/pred diff `= 0.00e+00`, connectome matrix `max|diag| = 0.0`, `max|bilateral| = 0.0`, end-to-end bilateral exclusion diffs `AVAR->AVAL = 0.0`, `AIYR->AIYL = 0.0`, `ASER->ASEL = 0.0`, `AWCOFF->AWCON = 0.0`, target-self network pathway diff `= 0.0`, and asymmetric chemosensory connectome preservation verified (`AIYL` sees `ASEL`/`ASER` signed drive delta `= 1.4082`).

### 3.1 Cross-Source Primary Metric Summary (`E_d`, `I_d`, `I_macro` with 1000-Resample Animal Bootstrap 95% CIs)

| Model ID | Type | `Seyed2025 E_d` | `Seyed2025 I_d vs B4 [95% CI]` (`Worms`) | `Toyo2024 E_d` | `Toyo2024 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` Main 2 vs B4 [95% CI]** | `Yemini2021 E_d` | `Yemini2021 I_d vs B4 [95% CI]` (`Worms`) | **`I_macro` All 3 vs B4 [95% CI]** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M3R1_02_h1_confounds_only` *(ANCHOR)* | Self + 7 confounds | 0.167000 | +1.32% [+0.75%, +1.87%] (`25/32`) | 0.339912 | +0.02% [-0.34%, +0.33%] (`8/11`) | **+0.67% [+0.35%, +0.99%]** | 0.772943 | +3.25% [+1.34%, +4.06%] (`18/21`) | **+1.53% [+0.87%, +1.87%]** |
| `M3R2_04_h2_film_gating_residual` *(R2 BEST)* | Compact FiLM Gate | 0.166596 | +1.56% [+0.67%, +2.50%] (`26/32`) | 0.339891 | +0.02% [-0.33%, +0.36%] (`9/11`) | **+0.79% [+0.32%, +1.31%]** | 0.770208 | +3.59% [+2.03%, +4.33%] (`17/21`) | **+1.73% [+1.09%, +2.14%]** |
| `M3R3_01_h3_wired_unsigned` | Cook 2019 Chem+Gap Unsigned | 0.166422 | +1.67% [+0.76%, +2.56%] (`26/32`) | 0.339956 | +0.01% [-0.35%, +0.34%] (`7/11`) | **+0.84% [+0.37%, +1.32%]** | 0.773918 | +3.13% [+1.13%, +4.01%] (`16/21`) | **+1.60% [+0.87%, +2.03%]** |
| `M3R3_02_h3_wired_signed_cengen` | CeNGEN Signed Chem+Gap | 0.166363 | +1.70% [+0.76%, +2.63%] (`25/32`) | 0.340034 | -0.02% [-0.38%, +0.27%] (`7/11`) | **+0.84% [+0.34%, +1.37%]** | 0.774216 | +3.09% [+1.08%, +3.96%] (`16/21`) | **+1.59% [+0.86%, +2.02%]** |
| `M3R3_03_h3_multilayer_wired_plus_wireless_npp` | 4-Layer Signed + 92 NPP/Mono | 0.166318 | +1.73% [+0.71%, +2.69%] (`25/32`) | 0.340044 | -0.02% [-0.39%, +0.27%] (`7/11`) | **+0.85% [+0.32%, +1.39%]** | 0.775935 | +2.87% [+0.96%, +3.71%] (`15/21`) | **+1.53% [+0.83%, +1.96%]** |
| `M3R3_04_h3_sex_dimorphic_connectome_film` | **Sex-Dimorphic Connectome FiLM** | **0.166281** | **+1.75% [+0.69%, +2.81%]** (`25/32`) | **0.339866** | **+0.03% [-0.31%, +0.38%]** (`7/11`) | **+0.89% [+0.35%, +1.48%]** | **0.770368** | **+3.57% [+1.93%, +4.31%]** (`18/21`) | **+1.78% [+1.14%, +2.24%]** |

### 3.2 Incremental Contrasts (`vs M3R1_02` Anchor & `vs M3R2_04` Round-2 Best)

| Model ID | `I_macro` Main 2 vs `M3R1_02` [95% CI] | `I_macro` All 3 vs `M3R1_02` [95% CI] | `I_macro` Main 2 vs `M3R2_04` [95% CI] | `Seyed` vs `M3R1_02` (`Worms`) | `Toyo` vs `M3R1_02` (`Worms`) | `Yemini` vs `M3R1_02` (`Worms`) | `Seyed post_R2 / delta_R2` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M3R3_01_h3_wired_unsigned` | +0.17% [-0.22%, +0.51%] | +0.07% [-0.20%, +0.31%] | +0.04% [-0.47%, +0.46%] | +0.35% (`25/32`) | -0.01% (`6/11`) | -0.13% (`7/21`) | 0.8912 / 0.2637 |
| `M3R3_02_h3_wired_signed_cengen` | +0.17% [-0.24%, +0.56%] | +0.06% [-0.23%, +0.32%] | +0.05% [-0.47%, +0.48%] | +0.38% (`23/32`) | -0.04% (`5/11`) | -0.16% (`6/21`) | **0.8917 / 0.2667** |
| `M3R3_03_h3_multilayer_wired_plus_wireless_npp` | +0.18% [-0.25%, +0.59%] | -0.01% [-0.30%, +0.28%] | +0.06% [-0.49%, +0.51%] | +0.41% (`24/32`) | -0.04% (`5/11`) | -0.39% (`5/21`) | 0.8916 / 0.2662 |
| `M3R3_04_h3_sex_dimorphic_connectome_film` | **+0.22% [-0.24%, +0.70%]** | **+0.26% [-0.07%, +0.60%]** | **+0.10% [-0.36%, +0.51%]** | **+0.43%** (`23/32`) | **+0.01%** (`8/11`) | **+0.33%** (`14/21`) | 0.8915 / 0.2654 |

### 3.3 Mandatory H3 Mechanistic Diagnostics (`PROJECT.md F10`)

1. **4-Layer Inter-Layer Correlation (Edge-Level vs Incoming Trial-Drive Dissociation)**:
   - **Anatomical edge level** (`47,269` union non-zero allowed edges): All 5 layers are nearly orthogonal (`|Spearman r| <= 0.069` between wired and wireless layers; `W_npp_wireless <-> W_monoamine` `r = -0.239`).
   - **Incoming trial-level drive features (`seyedolmohadesin2025` `[-2,0)s` drives)**: Dense wireless drives become **strongly collinear with each other and with electrical gap junctions** (`W_npp_wireless <-> W_monoamine` Spearman `r = +0.780`, `W_gap <-> W_npp_wireless` `r = +0.651`, `W_gap <-> W_monoamine` `r = +0.606`), whereas sparse polarity-split chemical drives stay distinct (`W_chem_pos <-> W_chem_neg` `r = +0.105`; `W_chem_pos <-> W_gap` `r = +0.202`).
2. **Leave-One-Layer-Out Ablation & Unique vs Shared Variance Partition (`M3R3_03` vs `M3R1_02` Anchor)**:
   - `full` 4-layer: Main 2 `= +0.185% [-0.254%, +0.593%]` | All 3 `= -0.006%`
   - `-W_gap` (drop gap junctions): Main 2 drops to **`+0.080%`** (`unique_W_gap = +0.105%` Main 2, `+0.073%` All 3 — **largest unique positive layer contribution!**)
   - `-W_chem_signed`: Main 2 `= +0.210%` (`unique_W_chem_signed = -0.025%` Main 2, `+0.013%` All 3)
   - `-W_wireless_npp`: Main 2 `= +0.196%` (`unique_W_wireless_npp = -0.011%` Main 2, `+0.020%` All 3)
   - `-W_monoamine`: Main 2 `= +0.190%` (`unique_W_monoamine = -0.005%` Main 2, `-0.017%` All 3)
   - **Wired-only (`-all_wireless`) vs Wireless-only (`-all_wired`) Partition**:
     - Main 2: `gain_wired_only = +0.173%`, `gain_wireless_only = +0.250%`, `gain_full = +0.185%` $\implies$ **`shared_wired_wireless = +0.238%`** (shared network synchrony dominates unique wired `-0.065%` / unique wireless `+0.012%`!).
     - All 3: `gain_wired_only = +0.060%`, `gain_wireless_only = -0.137%`, `gain_full = -0.006%` $\implies$ `unique_wired = +0.132%`, `unique_wireless = -0.066%` (adding dense wireless drives hurts smaller-N `Yemini` due to variance inflation).
3. **Sex Dimorphism Analysis (`seyedolmohadesin2025`: `13` Hermaphrodites vs `19` Males)**:
   - Under `M3R3_04` (`sex_matched`), **males** achieve stronger gains (`E_d = 0.150012`, `+2.38% [+0.80%, +3.70%]` vs `B4`, `15/19` worms; `+0.88% [-0.48%, +2.06%]` vs `M3R1_02`, `16/19` worms) than **hermaphrodites** (`E_d = 0.190060`, `+1.01% [-0.53%, +2.58%]` vs `B4`, `10/13` worms; `-0.09% [-1.27%, +1.13%]` vs `M3R1_02`, `7/13` worms), while `1uM ascr#3` pheromone error drops from `0.2360` (`M3R1_02`) to **`0.2295`** (`-2.75%`).
   - **Critical Falsification Control (`sex_matched` vs `herm_only` vs `sex_swapped`)**:
     - `sex_matched` vs `herm_only` (hermaphrodite connectome applied to males): **`+0.001% [-0.44%, +0.47%]`** on males (`0.150012` vs `0.150013`) and **`-0.12% [-0.47%, +0.25%]`** overall (`0.166281` vs `0.166084`, `16/32` worms).
     - `sex_matched` vs `sex_swapped`: `-0.35% [-0.65%, +0.02%]` overall (`0.166281` vs `0.165702`), with `herm_only` (`+0.55%` vs `M3R1_02`, `23/32` worms) and `sex_swapped` (`+0.78%` vs `M3R1_02`, `24/32` worms) matching or slightly beating `sex_matched`!
   - **Honest Mechanistic Diagnosis**: `M3R3_04` sets a new overall Dev record (`+0.89% [+0.35%, +1.48%]` Main 2, `+1.78% [+1.14%, +2.24%]` All 3 vs `B4`; strictly positive on ALL 3 datasets vs both `B4` and `M3R1_02`), driven by **multi-layer connectome drives + bounded FiLM gating** (especially in males and on `1uM ascr#3` pheromone / `sensory` neurons `0.3744 -> 0.3704` / late `[3,10)s` `0.2085 -> 0.2076`), **NOT** by sex-specific `Cook 2019 H/M` head-wiring substitution (where male head EM has sparser edge coverage `2315` vs `3768` and sex-specific tail circuits lie outside the head imaging FOV).

### 3.4 Concrete Recommendations for M3 Round 4 (`F11` Robustness & Best Candidate Synthesis)

1. **Synthesize Clean Best Candidate (`M3R4_01_connectome_herm_film_best`)**:
   - Combine Round 3's two cleanest findings: use the complete **`Cook 2019` hermaphrodite signed wired + `W_gap` + `Ripoll-Sánchez` wireless** connectome (`herm_only` routing, avoiding sparser male head EM noise — already `0.166084` on Seyed!) + excluded-self slow-state into the bounded FiLM residual gate (`M3R3_04` architecture), and compare directly against `M3R3_04`, `M3R2_04`, and `M3R3_02` (closed-form signed wired).
2. **Mandatory F11 Stress Tests on Dev (under frozen `lambda_global`)**:
   - **Artificial Missingness Dropout Curve (`+0%`, `+15%`, `+30%`, `+50%` extra random neuron dropout)**: Verify graceful `kappa`-precision shrinkage without variance blowup.
   - **Short-Window Check (`[-5, 0)s` vs `[-10, 0)s`)**: Quantify whether slow `[ -10, -5)s` history contributes incremental gain beyond `[-5, 0)s`.
   - **`0.5x / 1x / 2x` One-at-a-Time Hyperparameter Neighborhood Stability**: Perturb `connectome_l2_penalty`, `missingness_precision_kappa`, `weight_decay`, and `poco_film_dim` / `bilinear_rank` across `0.5x/1x/2x` to verify stability and check the `>= 80%` positive criterion.
3. **Pre-M4 §8 Route Readiness**:
   - Across Rounds 1–3, the tight animal-bootstrap 95% CI upper bound for `I_macro` (Main 2) is **`+1.48%` vs `B4`** and **`+0.70%` vs `M3R1_02`** (well below `5%`). After Round 4 F11 robustness checks, freeze the best candidate for M4 (`>=199` retrained degree/distance/cell-type-matched null graphs on the closed-form connectome model `M3R3_02`/`M3R3_03` + `Randi unc-31` + `Baskoylu Copper` + §8 Equivalence / Effect Upper-Bound route lock in `final_manifest.json`).



---

## Round 4 (`M3` Final Candidate Synthesis, F11 Robustness Suite & M3 Closure)

- **Code**: `models/m3_round4_f11.py` | **Consolidated M3 Artifact**: `results/m3_mechanistic_and_robustness.json` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, zero `dataset_id` branches)
- **Frozen Dev Candidate**: **`M3R4_01_connectome_film_herm_final`** (`runs/r4_01_m3r4_01_connectome_film_herm_final/`, appended to `runs.jsonl` as row 28) — unified `herm_only` receptor-polarity-signed multi-layer connectome (`W_chem_pos`, `W_chem_neg`, `W_chem_unk`, `W_gap`, `W_witvliet_core`, `W_npp_short/mid`, `W_monoamine`, strictly `diag=0` and `bilateral=0`, preserving `ASEL/ASER` & `AWCON/OFF` asymmetry) + excluded-self circuit slow-state into the bounded multiplicative FiLM residual gate anchored on `M3R1_02_h1_confounds_only`.
- **Leakage Invariance Audit (`M3R4_01` + All 4 M3 Rounds)**: **PASSED (`4/4` rounds)** — `future_poison_max_abs_diff = 0.00e+00`, `oof_tamper_pred_diff = 0.00e+00`, `bilateral_exclusion_AIYR_to_AIYL_diff = 0.00e+00`, `max_diag = 0.0`, `max_bilat = 0.0`. `vault_sealed/` untouched.

### 4.1 `M3R4_01_connectome_film_herm_final` Primary Seed (`20261117`) & 5-Seed Stability (`[20261117..20261121]`)

| Metric / Split | `M3R4_01` Primary (`20261117`) | 5-Seed Mean ± Std | Gain vs `B4_ridge_self_stim` [95% CI] (`Worms`) | Gain vs `M3R1_02` [95% CI] (`Worms`) |
|---|---:|---:|---:|---:|
| **`seyedolmohadesin2025` (`E_d`)** | **0.166084** | `0.166126 ± 0.000204` | **+1.87% [+0.80%, +2.84%]** (`25/32`) | +0.55% [-0.21%, +1.33%] (`23/32`) |
| **`toyoshima2024` (`E_d`)** | **0.339866** | `0.339790 ± 0.000064` | **+0.03% [-0.27%, +0.34%]** (`7/11`) | +0.01% [-0.09%, +0.14%] (`8/11`) |
| **`yemini2021` (`E_d`)** | **0.770368** | `0.771331 ± 0.000677` | **+3.57% [+1.88%, +4.36%]** (`18/21`) | **+0.33% [+0.14%, +0.70%]** (`14/21`) |
| **`I_macro` Main 2** | **+0.95%** | **`+0.95% ± 0.07%`** | **+0.95% [+0.46%, +1.46%]** (`worst +0.03%`) | +0.28% [-0.12%, +0.66%] (`+0.28% ± 0.07%`) |
| **`I_macro` All 3** | **+1.82%** | **`+1.78% ± 0.04%`** | **+1.82% [+1.14%, +2.28%]** | **+0.30% [+0.05%, +0.59%]** (`+0.26% ± 0.04%`) |

- `M3R4_01` sets the **new overall M3 Dev record** (`+0.95% [+0.46%, +1.46%]` Main 2, `+1.82% [+1.14%, +2.28%]` All 3 vs `B4`, beating `M3R3_04`'s `+0.89%` by eliminating sparse male head EM noise on `Seyed`: `0.166084` vs `0.166281`) with ultra-tight 5-seed reproducibility (`± 0.07%` Main 2, `± 0.04%` All 3).

### 4.2 F11(a) Artificial Missingness Dropout Curve (`+0%`, `+15%`, `+30%`, `+50%` Extra Non-Target Neuron Dropout)

| Extra Dropout | `M3R4_01` `Seyed E_d` | `M3R4_01` `Toyo E_d` | `M3R4_01` `Yemini E_d` | `M3R4_01` `I_macro` Main 2 vs `B4` | `M3R4_01` `I_macro` All 3 vs `B4` | `M3R4_01` `I_macro` Main 2 vs `M3R1_02` | `M3R1_02` `I_macro` Main 2 vs `B4` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`+0%`** | 0.166084 | 0.339866 | 0.770368 | **+0.95%** | **+1.82%** | **+0.28%** | +0.67% |
| **`+15%`** | 0.166026 | 0.340024 | 0.773297 | **+0.94%** | **+1.70%** | **+0.35%** | +0.60% |
| **`+30%`** | 0.166364 | 0.340352 | 0.778694 | **+0.79%** | **+1.37%** | **+0.32%** | +0.48% |
| **`+50%`** | 0.166756 | 0.340925 | 0.790745 | **+0.59%** | **+0.74%** | **+0.36%** | +0.24% |

- **Verdict**: Strict graceful degradation under heavy artificial missingness (on top of native 30–50% missingness), with zero variance blowup; `M3R4_01` stays strictly positive vs `B4` (`+0.59%` Main 2 even at `+50%` extra dropout) and preserves a steady `+0.32%–+0.36%` incremental advantage over `M3R1_02` across all dropout levels thanks to `kappa`-precision shrinkage.

### 4.3 F11(b) Short-Window Check (`[-5, 0)s` vs Standard `[-10, 0)s`, Zero Frames Before `-5s`)

| Model ID | `Seyed E_d` (`5s` vs `10s`) | `Toyo E_d` (`5s` vs `10s`) | `Yemini E_d` (`5s` vs `10s`) | `I_macro` Main 2 vs `B4_5s` | `I_macro` Main 2 vs `M3R1_02_5s` | `I_macro` All 3 vs `M3R1_02_5s` |
|---|---:|---:|---:|---:|---:|---:|
| `B4_ridge_self_stim` | `0.173608` (`+0.0044`) | `0.353809` (`+0.0138`) | `0.769764` (`-0.0291`) | +0.00% | -0.47% | +0.17% |
| `M3R1_02_h1_confounds_only` | `0.171165` (`+0.0042`) | `0.355559` (`+0.0156`) | `0.780941` (`+0.0080`) | +0.46% | +0.00% | +0.00% |
| **`M3R4_01_connectome_film_herm_final`** | **`0.171039`** (`+0.0050`) | **`0.354641`** (`+0.0148`) | **`0.780065`** (`+0.0097`) | **+0.62%** | **+0.17%** | **+0.15%** |

- **Verdict**: Removing `[-10, -5)s` + calibration history degrades absolute `E_d` across all 3 models on both main datasets (`+0.004–0.016`), confirming slow pre-stimulus history carries genuine signal; nevertheless, `M3R4_01` still outperforms `B4_5s` (`+0.62%` Main 2) and beats `M3R1_02_5s` on **all 3 datasets** (`Seyed +0.07%`, `Toyo +0.26%`, `Yemini +0.11%`).

### 4.4 F11(c) `0.5x / 1x / 2x` One-at-a-Time Hyperparameter Neighborhood Stability (9 Configs)

| Point ID | `Seyed E_d` | `Toyo E_d` | `Yemini E_d` | `I_macro` Main 2 vs `B4` | `I_macro` All 3 vs `B4` | `I_macro` Main 2 vs `M3R1_02` |
|---|---:|---:|---:|---:|---:|---:|
| `1x_baseline` | 0.166084 | 0.339866 | 0.770368 | **+0.95%** | +1.82% | **+0.28%** |
| `l2_penalty_0.5x` | 0.165506 | 0.340020 | 0.767723 | **+1.10%** | +2.03% | **+0.43%** |
| `l2_penalty_2.0x` | 0.166275 | 0.339979 | 0.774989 | **+0.88%** | +1.58% | **+0.21%** |
| `kappa_0.5x` | 0.166104 | 0.339840 | 0.770356 | **+0.95%** | +1.82% | **+0.28%** |
| `kappa_2.0x` | 0.166277 | 0.339898 | 0.770266 | **+0.89%** | +1.79% | **+0.22%** |
| `weight_decay_0.5x` | 0.166083 | 0.339866 | 0.770367 | **+0.95%** | +1.82% | **+0.28%** |
| `weight_decay_2.0x` | 0.166087 | 0.339866 | 0.770371 | **+0.95%** | +1.82% | **+0.28%** |
| `learning_rate_0.5x` | 0.165905 | 0.339748 | 0.770407 | **+1.02%** | +1.87% | **+0.35%** |
| `learning_rate_2.0x` | 0.166156 | 0.339877 | 0.771100 | **+0.93%** | +1.78% | **+0.26%** |

- **Neighborhood Stability Verdict**: **`9/9 (100.0%)` positive vs `B4_ridge_self_stim`** (`+0.88%` to `+1.10%` Main 2; `+1.58%` to `+2.03%` All 3) AND **`9/9 (100.0%)` positive vs `M3R1_02`** (`+0.21%` to `+0.43%` Main 2) — cleanly **PASSING** the `>= 80%` stability requirement (`MODEL_REQUIREMENTS.md §3, §5`).

### 4.5 M3 Closure & Pre-M4 `§8` Route Readiness
- **Milestone M3 is COMPLETE** across H1, H2, H3, and F11 under single frozen `configs/lambda_global.json`.
- Because Dev `I_macro` Main 2 `= +0.95% [+0.46%, +1.46%]` has its upper 95% CI (`+1.46%`) far below the `5%` positive-confirmation threshold, **M4 MUST proceed under the pre-registered `MODEL_REQUIREMENTS §8` Equivalence / High-Precision Effect Upper-Bound Route**: execute `>= 199` degree/distance/cell-type-preserving retrained random null graphs, non-circular `Randi 2023` WT vs `unc-31` peptidergic validation, isolated `Baskoylu 2026 Copper` freely-moving stress test under `lambda_global`, and cryptographically freeze `final_manifest.json` prior to one-shot M5 vault unsealing.

---

## Milestone M4 — Stage A (`>= 199` Fully Retrained Constrained Random Null Graphs, `PROJECT.md F12`)

- **Code**: `models/m4_null_graphs.py` | **Artifact**: `results/null_graphs_summary.json` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, zero `dataset_id` branches) | **Vault**: `vault_sealed/` untouched.
- **Retraining Integrity**: Every null graph (`k = 1..199` ridge, `k = 1..19` PyTorch FiLM) was **genuinely retrained from scratch inside every CV fold** after recomputing missingness-precision-weighted causal connectome drives from the rewired null graph (zero prediction shuffling, zero real-graph weight reuse). `k = 0` real-graph exact verification matched saved `M3R3_03` to `abs_diff <= 1.64e-09` across all 3 Dev datasets and saved `M3R4_01` to `6` decimal places (`0.166084` `Seyed`, `0.339866` `Toyo`). Peak RSS: `2677.1 MB` (`< 3.0 GiB`).

### A.1 Constrained Null Generator Verification (`K = 199` Full 8-Layer Null Bank)
- **Exact Preserved Constraints (`199/199` nulls, all 8 layers)**:
  - In-degree & out-degree correlation: `min r = 1.0000` (`100%` exact).
  - Cell-type-pair block counts `(tier_post, tier_pre)`: `100.0%` exact.
  - 3D soma Euclidean distance-bin edge counts (5 quantile bins): `100.0%` exact.
  - Zero diagonal (`0.0`), zero bilateral-partner edges (`0.0`), exact symmetry on `W_gap_herm` (`0.0`).
- **Genuine Randomization Fidelity**:
  - Wired layers (`W_chem_pos/neg/unk`, `W_gap_herm`, `W_witvliet_core`): edge Jaccard `0.215–0.360`, weight Spearman `r = -0.436 to -0.231`.
  - Dense wireless layers (`W_npp_short`, `W_npp_mid`, `W_monoamine`): topology rewired + within-`(tier, tier, dist_bin)` weight permutation $\to$ weight Spearman `r = 0.102–0.152`.

### A.2 Retrained Null Distributions vs Real Biological Connectome (`vs B4_ridge_self_stim` & `vs M3R1_02`)

| Null Control Family & Model | Scope | Real `I` (vs `B4`) | Null Mean ± Std `[p05, p95]` (vs `B4`) | Real `I` (vs `M3R1_02`) | `k_ge_real / K` | Empirical One-Sided `p` | `z`-score |
|---|---|---:|---:|---:|---:|---:|---:|
| **Primary `K=199` Full 8-Layer Rewired** (`M3R3_03_herm_ridge`) | **Main 2** | **+0.853%** | `+0.833% ± 0.061%` `[+0.735%, +0.923%]` | +0.185% | `83 / 199` | **0.4200** | **+0.33** |
| Primary `K=199` Full 8-Layer Rewired (`M3R3_03_herm_ridge`) | All 3 | +1.527% | `+1.563% ± 0.065%` `[+1.453%, +1.678%]` | -0.006% | `142 / 199` | 0.7150 | -0.57 |
| Primary `K=199` Full 8-Layer Rewired (`M3R3_03_herm_ridge`) | `Seyed` / `Toyo` / `Yemini` | `+1.727%` / `-0.021%` / `+2.873%` | `+1.691%` / `-0.024%` / `+3.024%` | `+0.408%` / `-0.039%` / `-0.387%` | `84` / `61` / `177` | `0.425` / `0.310` / `0.890` | `+0.30` / `+0.42` / `-1.09` |
| **`K=199` Polarity-Sign-Only Permutation** (`S_sign` on fixed topology) | **Main 2** | **+0.853%** | `+0.847% ± 0.031%` `[+0.802%, +0.903%]` | +0.185% | `70 / 199` | **0.3550** | **+0.20** |
| `K=199` Polarity-Sign-Only Permutation (`S_sign` on fixed topology) | All 3 | +1.527% | `+1.527% ± 0.039%` `[+1.470%, +1.591%]` | -0.006% | `95 / 199` | 0.4800 | -0.00 |
| **`K=199` Gap-Junction-Only Rewired** (only `W_gap_herm` rewired) | **Main 2** | **+0.853%** | `+0.791% ± 0.030%` `[+0.743%, +0.847%]` | +0.185% | **`3 / 199`** | **0.0200** | **+2.07** |
| `K=199` Gap-Junction-Only Rewired (only `W_gap_herm` rewired) | `Seyed` / `Toyo` / `Yemini` | `+1.727%` / `-0.021%` / `+2.873%` | `+1.604%` / `-0.022%` / `+2.936%` | `+0.408%` / `-0.039%` / `-0.387%` | **`5`** / `59` / `155` | **`0.030`** / `0.300` / `0.780` | **`+2.06`** / `+0.47` / `-0.76` |
| **Matched `K=19` PyTorch FiLM Retrained** (`M3R4_01_film_herm`) | **Main 2** | **+0.948%** | `+0.907% ± 0.094%` `[+0.777%, +1.043%]` | +0.281% | `7 / 19` | **0.4000** | **+0.45** |
| Matched `K=19` PyTorch FiLM Retrained (`M3R4_01_film_herm`) | `Seyed` / `Toyo` | `+1.865%` / `+0.032%` | `+1.754% ± 0.180%` / `+0.060% ± 0.014%` | `+0.548%` / `+0.014%` | `7/19` / `19/19` | `0.400` / `1.000` | `+0.62` / `-1.99` |

- **Mechanistic Verdict**:
  1. **Full 8-layer connectome vs degree+distance+cell-type nulls is NOT statistically significant** under both exact `K=199` Ridge (`p = 0.420`, `z = +0.33` Main 2; `p = 0.715` All 3) and matched `K=19` PyTorch FiLM (`p = 0.400`, `z = +0.45` Main 2): the bulk of the `+0.17%–+0.24%` incremental gain over `M3R1_02` is captured by degree, spatial-proximity, and cell-type-tier aggregation alone.
  2. **Polarity-sign-only permutation** shows no significant drop (`p = 0.355` Main 2, `p = 0.480` All 3).
  3. **Electrical gap-junction (`W_gap_herm`) wiring specificity IS statistically significant** on Main 2 (`k_ge = 3/199`, `p = 0.020`, `z = +2.07`), driven by `seyedolmohadesin2025` (`k_ge = 5/199`, `p = 0.030`, `z = +2.06`, `+0.124%` gain over gap-rewired nulls), confirming Round 3's LOLO ablation that electrical coupling carries a small but genuine neuron-specific signal.

---

## Round 5 (`M3.5` Structural Nonlinear-Anchor + 1-Hop/2-Hop Connectome Patch-FiLM)
- **Date (UTC)**: `2026-11-21T12:49:51Z`
- **Unified Config**: `configs/lambda_global.json` (ZERO `dataset_id` branches, ZERO `vault_sealed/` access)

### 1. `Toyoshima 2024` Residual & Physical SNR Ceiling Diagnosis
| Dataset | FPS | HF Shot-Noise Var (`0.5*Var(dy/s_n)`) | Lag-1 $r$ | `B4_ridge E_d` | Shot-Noise % of `B4 E_d` | Empirical `post_R2` Ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `yemini2021` | 4.0 | `0.011446` | `0.989505` | `0.798889` | `1.4%` | `0.989151` |
| `seyedolmohadesin2025` | 2.66 | `0.009079` | `0.989427` | `0.169241` | `5.4%` | `0.990203` |
| `toyoshima2024` | 4.273 | `0.279239` | `0.928056` | `0.339974` | `82.1%` | `0.918025` |

### 2. Round 5 (`M3.5`) Dual-Reference Dev Results (vs `B4_ridge_self_stim` AND vs `M35_01`)
| Model | `Seyed I_d` vs `B4` | `Toyo I_d` vs `B4` | `Yemini I_d` vs `B4` | `I_macro` Main 2 vs `B4` [95% CI] | `I_macro` All 3 vs `B4` [95% CI] | Net `I_macro` All 3 vs `M35_01` [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| `M35_01_zeroinit_nonlinear_self_anchor` | `+4.03%` | `+0.54%` | `+7.53%` | `+2.29%` `[+1.56%, +3.17%]` | `+4.03%` `[+2.98%, +5.49%]` | `+0.00%` `[+0.00%, +0.00%]` |
| `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film` | `+4.76%` | `+0.55%` | `+7.51%` | `+2.66%` `[+1.81%, +3.67%]` | `+4.28%` `[+3.18%, +5.76%]` | `+0.25%` `[-0.08%, +0.63%]` |
| `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl` | `+4.94%` | `+0.58%` | `+7.60%` | `+2.76%` `[+1.89%, +3.81%]` | `+4.37%` `[+3.25%, +5.83%]` | `+0.35%` `[+0.10%, +0.64%]` |

### 3. Leakage Invariance (`M35_02`): `passed=True`, `future_poison_max_abs_diff=0.0`, `oof_tamper_pred_diff=0.0`, `2hop_diag=0.0`, `2hop_bilateral=0.0`.

---

## Round 6 (`M3.5` Full Multi-Seed, OAT, Robustness & 199-Null-Graph Validation)
- **Date (UTC)**: `2026-11-21T16:20:04Z`
- **Unified Config**: `configs/lambda_global.json` (ZERO `dataset_id` branches, ZERO `vault_sealed/` access)

### 1. 5-Seed Multi-Seed Stability (`[20261117..20261121]`)
| Model | `Seyed I_d` vs `B4` (mean±std) | `Toyo I_d` vs `B4` | `Yemini I_d` vs `B4` | `I_macro` Main 2 vs `B4` | `I_macro` All 3 vs `B4` | Net Main 2 vs `M35_01` | Net All 3 vs `M35_01` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M35_01_zeroinit_nonlinear_self_anchor` | `+4.60% ± 0.88%` | `+0.50% ± 0.03%` | `+6.90% ± 0.41%` | `+2.55% ± 0.43%` | `+4.00% ± 0.20%` | `+0.00% ± 0.00%` | `+0.00% ± 0.00%` |
| `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film` | `+5.09% ± 0.73%` | `+0.52% ± 0.02%` | `+6.85% ± 0.47%` | `+2.80% ± 0.36%` | `+4.15% ± 0.19%` | `+0.27% ± 0.11%` | `+0.16% ± 0.11%` |
| `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl` | `+5.50% ± 0.80%` | `+0.53% ± 0.03%` | `+6.85% ± 0.55%` | `+3.01% ± 0.39%` | `+4.29% ± 0.15%` | `+0.48% ± 0.07%` | `+0.31% ± 0.10%` |

### 2. `0.5x / 1x / 2x` OAT Hyperparameter Neighborhood Stability (`M35_03`, 9 configs)
- **Positive vs `B4_ridge`**: Main 2 `9/9` (`100.0%`), All 3 `9/9` (`100.0%`), passes `>=80%`: `True`.
- **Range vs `B4_ridge`**: Main 2 `[+2.57%, +3.19%]`, All 3 `[+3.98%, +4.53%]`.
- **Positive vs paired `M35_01`**: Main 2 `9/9` (`100.0%`), All 3 `9/9` (`100.0%`), passes `>=80%`: `True`.
- **Range vs paired `M35_01`**: Main 2 `[+0.38%, +0.61%]`, All 3 `[+0.25%, +0.38%]`.

### 3. F11 Missingness Dropout Curve (`+0%/+15%/+30%/+50%`) & `[-5, 0)s` Short-Window Check
| Condition | `M35_01` Main 2 vs `B4` | `M35_01` All 3 vs `B4` | `M35_03` Main 2 vs `B4` | `M35_03` All 3 vs `B4` | Net `M35_03 vs M35_01` Main 2 |
|---|---:|---:|---:|---:|---:|
| Dropout `+0pct` | `+2.29%` | `+4.03%` | `+2.76%` | `+4.37%` | `+0.49%` |
| Dropout `+15pct` | `+2.23%` | `+3.86%` | `+2.65%` | `+4.16%` | `+0.44%` |
| Dropout `+30pct` | `+2.12%` | `+3.52%` | `+2.49%` | `+3.79%` | `+0.38%` |
| Dropout `+50pct` | `+1.91%` | `+2.85%` | `+2.18%` | `+3.05%` | `+0.28%` |
| Short-window `[-5,0)s` | `+3.06%` | `+2.32%` | `+3.32%` | `+2.51%` | `+0.27%` |

### 4. `K = 199` Constrained Null Graphs on Frozen `M35_01` Nonlinear Anchor + `K_film = 19` End-to-End `M35_02`
| Test Family | `Seyed2025` (`p`, `z`) | `Toyo2024` (`p`, `z`) | `Yemini2021` (`p`, `z`) | Main 2 Macro (`p`, `z`) | All 3 Macro (`p`, `z`) |
|---|---|---|---|---|---|
| 1-hop+2-hop Ridge on `M35_01` (`K=199` Full 8L) | `p=0.3200, z=+0.51` | `p=0.0500, z=+1.65` | `p=0.1050, z=+1.14` | `p=0.2600, z=+0.65` | `p=0.1050, z=+1.25` |
| 1-hop+2-hop Ridge on `M35_01` (`K=199` `W_gap`-only) | `p=0.0650, z=+1.60` | `p=0.0400, z=+1.88` | `p=0.1400, z=+0.97` | `p=0.0500, z=+1.73` | `p=0.0850, z=+1.58` |
| End-to-End `M35_02` Patch-FiLM (`K=19` Full 8L) | `p=0.0500, z=+2.25` | `p=0.8500, z=-0.88` | N/A (Main 2) | `p=0.0500, z=+2.18` | N/A |

---

## M4 Stage B: Non-Circular `Randi 2023` WT vs `unc-31` Validation, Modality-Isolated `Baskoylu 2026 Copper` Dev Stress Test & §8 Cryptographic Freeze
- **Date (UTC)**: `2026-11-22T00:49:29Z`
- **Code**: `scripts/run_m4b_randi_copper_freeze.py` | **Config**: `configs/lambda_global.json` (`lambda_global_v1`, ZERO `dataset_id` branches, ZERO `vault_sealed/` access)
- **Artifacts**: `results/randi_unc31_validation.json`, `results/copper_stress_test.json`, `final_manifest.json`

### B.1 Non-Circular `Randi 2023` WT (113w) vs `unc-31` (18w) Peptidergic Validation
- **Provenance & Anti-Circularity**: `0` pairs in `connectome_priors_bundle.npz` or Dev training were derived from `Randi 2023`. Matrix orientation mathematically verified (`raw_nonzero_cols_stimulated = raw_nonzero_diag = 173`, `raw_nonzero_rows_recorded = 208`; transposed `[post, pre] -> [pre, post]` to match `Cook 2019` / `Ripoll-Sánchez 2023` `W[pre, post]`). Evaluated across `299` shared neurons (`7,939` co-tested WT & `unc-31` non-bilateral directed pairs; ultra-conservative complement of `7,747` pairs also verified after explicitly excluding all `192` wired-absent `unc-31`-selected wireless candidates).

| Layer / Contrast (Co-Tested Non-Bilateral Pairs) | `n_edge` / `n_no_edge` | `|R_WT| - |R_unc31|` Mean Diff `[95% Boot CI]` | Cliff's $\delta$ | MWU `p` (2-sided) | WT-Only Abolished OR (`Fisher p`) |
|---|---:|---:|---:|---:|---:|
| `ripollsanchez2023_short` (All co-tested) | `3,125 / 4,814` | `+0.00368` `[-0.00328, +0.01176]` | `+0.0265` | **0.0459** | `1.479` (`7.53e-05`) |
| `ripollsanchez2023_short` **ONLY (Wired-Absent)** vs Neither | `2,696 / 4,569` | `+0.00660` (Med `+0.00282` `[+0.00010, +0.00524]`) | **`+0.0371`** | **0.0081** | **`1.481` (`3.43e-04`)** |
| `ripollsanchez2023_mid` (All co-tested) | `3,646 / 4,293` | `+0.00271` `[-0.00455, +0.01230]` | `+0.0024` | `0.8510` | `0.997` (`1.00`) |
| `monoamine` (All co-tested control) | `298 / 7,641` | `+0.00370` `[-0.01107, +0.01625]` | `+0.0245` | `0.4370` | `0.641` (`0.152`) |
| `cook2019_herm_chem` (Wired chemical control) | `495 / 7,444` | `-0.00568` `[-0.01951, +0.00719]` | `+0.0120` | `0.6410` | `2.006` (`2.30e-05`)* |
| `cook2019_herm_gap` (Wired electrical control) | `234 / 7,705` | `-0.01661` `[-0.04154, +0.00532]` | `+0.0043` | `0.9130` | `3.848` (`1.00e-10`)* |

*\*Note: Wired controls have high `WT-only OR` due to higher baseline WT responsiveness combined with lower statistical power in 18 `unc-31` vs 113 WT animals; crucially, ONLY short-range `Ripoll-Sánchez 2023` NPP-GPCR edges exhibit positive continuous `|R_WT| - |R_unc31|` attenuation (`p = 0.0081` on wired-absent pairs), whereas wired chemical/gap controls show slightly negative continuous attenuation.*
- **Non-Circular Dev Cross-Reference (`seyedolmohadesin2025` Dev, `155` shared neurons)**: `M35_02 vs M35_03` per-neuron gain correlates weakly and non-significantly with Randi incoming `unc-31` attenuation (`Spearman rho = +0.130, p = 0.107, 95% CI [-0.031, +0.302]`), and `M35_02 vs M35_01` correlates weakly with gap-junction in-degree (`rho = +0.136, p = 0.092`).

### B.2 Modality-Isolated `Baskoylu 2026 Copper` Freely-Moving Stress Test (18 Dev Worms ONLY)
- **Protocol**: Strict Dev-only (`18` worms, `18` LOO folds, `108` valid causal trials, `1.667 Hz`, `9` vaulted worms untouched), frozen `lambda_global_v1`, `5` seeds `[20261117..20261121]` for `M35_01/02/03`, `1000x` animal-level Bootstrap 95% CIs. Wall-clock: `211.62s`, peak RSS: `684.2 MB`.
- **Leakage Invariance (`M35_02` on Copper Dev fold 0)**: `passed = True`, `future_poison_max_abs_diff = 0.0`, `oof_tamper_pred_diff = 0.0`.

| Model | `E_d` | `I_d` vs `B4` `[95% Boot CI]` | `I_d` vs `M35_01` `[95% Boot CI]` | `post_R2` | `delta_R2` | `RMSE_raw` | Worms Improved vs `B4` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4_ridge_self_stim` | `0.357946` | `+0.00%` `[0.00%, 0.00%]` | `-1.58%` | `0.8839` | `0.1983` | `0.1593` | `0 / 18` |
| `M3R4_01_connectome_film_herm_final` | `0.355865` | `+0.58%` `[-0.68%, +1.76%]` | `-0.99%` `[-1.60%, -0.26%]` | `0.8852` | `0.2072` | `0.1584` | `12 / 18` |
| `M35_01_zeroinit_nonlinear_self_anchor` (5-seed) | `0.352381` | **`+1.55%`** `[-0.03%, +3.05%]` | `+0.00%` `[0.00%, 0.00%]` | `0.8866` | `0.2169` | `0.1574` | `14 / 18` |
| `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film` (5-seed) | `0.352557` | `+1.51%` `[-0.13%, +2.92%]` | `-0.05%` `[-0.19%, +0.08%]` | `0.8863` | `0.2148` | `0.1576` | `14 / 18` |
| `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl` (5-seed) | `0.354101` | `+1.07%` `[-0.77%, +2.66%]` | `-0.49%` `[-0.78%, -0.21%]` | `0.8858` | `0.2117` | `0.1579` | `14 / 18` |

- **Stress-Test Verdict**: Zero degradation (`> +1.0%` improvement over `B4` across all `M3.5` models, `14/18` worms improved, well within the `-2%` non-degradation guardrail). In freely-moving worms, nonlinear self-dynamics (`M35_01`, `+1.55%`) transfer cleanly without retuning, while unmeasured locomotion/proprioceptive variability limits additional cross-neuron Patch-FiLM gains (`-0.05%` connectome `M35_02` vs `-0.49%` graph-free `M35_03` relative to `M35_01`).

### B.3 Pre-Unblinding Cryptographic Freeze (`final_manifest.json`, §6 & §8 Lock)
- **Route Locked**: `section8_equivalence_bound` (per `MODEL_REQUIREMENTS.md §8` & User A3 — Dev 5-seed `M35_03` reaches `+4.29% ± 0.15%` All 3 and `+3.01% ± 0.39%` Main 2 with `Seyed = +5.50%` vs `Toyo = +0.53%` bounded by `82.14%` white optical shot-noise floor; `5%` E2 threshold is NOT lowered post-hoc).
- **Manifest SHA-256**: `db5cd2a4fe67d471092a865cbd5dc66a75ad6b7b6cb8731b1aae2249eeecfb85` (`6` governance/config files + `21` Python source files hashed; `vault_untouched_at_freeze = True`, `0` vault audit entries).

---

## M5 — Single One-Shot Vault Confirmatory Unsealing (2026-11-22T00:59:44Z)
- **Script**: `scripts/run_m5_vault_confirmation.py` | **Manifest SHA-256**: `db5cd2a4fe67d471092a865cbd5dc66a75ad6b7b6cb8731b1aae2249eeecfb85` | **Frozen Files Verified**: `27/27`
- **Route Locked Pre-Unseal**: `section8_equivalence_bound` | **Seeds**: `[20261117, 20261118, 20261119, 20261120, 20261121]` | **Bootstrap**: `B = 2000` physical animals (simultaneous `97.5%` Bonferroni + marginal `95%` CIs)
- **Vaulted Animals Evaluated ONCE**: `15` Seyedolmohadesin (9H+6M, 3 unseen date-batches) + `10` Toyoshima (10 unseen physical worms, 3 unseen date-batches) + `9` Copper (isolated freely-moving, 5 unseen date-batches) = `34` unseen animals.
- **Artifacts**: `results/confirmation_final.json`, `results/confirmatory_summary.json`, `runs/m5_confirmatory/`, `exposure_ledger.json`, `vault_access_audit.jsonl`

| Model | Seyed Vault (15w) `I_d` vs `B4` `[97.5% Sim CI]` | Toyo Vault (10w) `I_d` vs `B4` `[97.5% Sim CI]` | Main 2 Macro `I_macro` `[95% CI]` | Main 2 Net vs `M35_01` `[95% CI]` | Copper Vault (9w, Isolated) `I_d` vs `B4` `[95% CI]` |
|---|---:|---:|---:|---:|---:|
| `B4_ridge_self_stim` | `+0.00%` `[+0.00%, +0.00%]` (0/15) | `+0.00%` `[+0.00%, +0.00%]` (0/10) | `+0.00%` `[+0.00%, +0.00%]` | `-2.55%` `[-3.64%, -1.68%]` | `+0.00%` `[+0.00%, +0.00%]` (0/9) |
| `M3R4_01_connectome_film_herm_final` | `+2.82%` `[+1.21%, +4.18%]` (13/15) | `+0.02%` `[-0.35%, +0.34%]` (7/10) | `+1.42%` `[+0.71%, +2.05%]` | `-1.06%` `[-2.31%, -0.20%]` | `+1.76%` `[+0.93%, +2.36%]` (8/9) |
| `M35_01_zeroinit_nonlinear_self_anchor` | `+4.40%` `[+2.46%, +6.31%]` (13/15) | `+0.50%` `[+0.15%, +1.40%]` (10/10) | `+2.45%` `[+1.66%, +3.34%]` | `+0.00%` `[+0.00%, +0.00%]` | `+2.29%` `[+1.14%, +3.60%]` (8/9) |
| `M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film` | `+5.60%` `[+3.33%, +7.64%]` (12/15) | `+0.55%` `[+0.21%, +1.43%]` (10/10) | `+3.07%` `[+2.09%, +4.03%]` | `+0.66%` `[+0.14%, +1.12%]` | `+2.39%` `[+1.40%, +3.45%]` (8/9) |
| `M35_03_nonlinear_anchor_graphfree_patch_film_ctrl` | `+5.40%` `[+3.04%, +7.86%]` (14/15) | `+0.52%` `[+0.17%, +1.42%]` (9/10) | `+2.96%` `[+1.92%, +4.08%]` | `+0.54%` `[-0.00%, +1.04%]` | `+2.10%` `[+0.93%, +3.16%]` (8/9) |

- **Honest §8 Evidence-Tier Verdict**: `E4_CANDIDATE_VIA_SECTION8_HIGH_PRECISION_EFFECT_BOUND`
- **Summary**: Strict E2 (>=5% on BOTH primary sources simultaneously) is NOT met as pre-registered (Seyedolmohadesin I_d=+5.40%, 97.5% CI [+3.04%, +7.86%]; Toyoshima I_d=+0.52%, 97.5% CI [+0.17%, +1.42%]; Main 2 Macro I_macro=+2.96%, 95% CI [+1.92%, +4.08%]). Under the pre-unblinding locked §8 High-Precision Effect-Size & Upper-Bound Quantification Route (no post-hoc threshold lowering), confirmatory vault unsealing cleanly replicates the Dev mechanistic decomposition: nonlinear self-dynamics + cross-neuron Patch-FiLM yields positive gain without degrading freely-moving Copper (I_d=+2.10% >= -2%), while bounding net cross-neuron increment beyond nonlinear self-anchor M35_01 at Main 2 Macro = +0.54% (95% CI [-0.00%, +1.04%]).
- **Total Wall-Clock**: `327.91s` | **Peak RSS**: `2214.29 MB`
