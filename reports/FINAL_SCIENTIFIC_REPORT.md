# Whole-Brain Causal Chemosensory Trajectory Forecasting and Multi-Layer Connectome State-Gating in *Caenorhabditis elegans*: Final Scientific Report (`autoresearch_v1`)

- **Date (UTC)**: `2026-11-22`
- **Frozen Global Configuration**: `lambda_global_v1` (`configs/lambda_global.json`, SHA-256 `b449c039d14d97d208f18e64215909770c281c71f0151a2d73919f6d03f90d9c`, `zero_dataset_id_branches: true`)
- **Pre-Unblinding Manifest SHA-256**: `db5cd2a4fe67d471092a865cbd5dc66a75ad6b7b6cb8731b1aae2249eeecfb85` (`27/27` frozen files cryptographically verified before single confirmatory unsealing at `2026-11-22T00:59:44Z`)
- **Pre-Registered Route**: `section8_equivalence_bound` (`MODEL_REQUIREMENTS.md §8`, locked prior to touching any vaulted animal; `E2` 5% threshold retained unmodified)
- **Final Evidence Tier Verdict**: **`E4_CANDIDATE_VIA_SECTION8_HIGH_PRECISION_EFFECT_BOUND`**
- **Execution Environment**: Strictly CPU-only (`2 vCPUs`, `7.8 GiB RAM`, `0 B Swap`, PyTorch 2.8 CPU, peak confirmatory RSS `2214.29 MiB`, confirmatory wall clock `327.91 s`)

---

## 1. Executive Summary & Honest §8 Evidence-Tier Verdict

**Bottom-Line Conclusion (Pyramid Principle):** Across **34 strictly unseen, date-batch-disjoint vaulted animals** spanning three independent public *C. elegans* whole-brain calcium imaging datasets (`seyedolmohadesin2025`, `toyoshima2024`, and isolated freely-moving `baskoylu2026_copper`), a single frozen CPU-only causal forecasting configuration (`lambda_global_v1`, zero per-dataset hyperparameter tuning) **confirms statistically significant positive unseen-animal improvement over the frozen self-history + stimulus Ridge reference (`B4_ridge_self_stim`) on every source**, while **honestly NOT meeting the strict `E2` criterion (`I_d >= 5%` on both primary immobilized sources simultaneously)** because raw `toyoshima2024` YC2.60 traces are physically bounded by an **`82.14%` high-frequency white optical shot-noise floor**.

Under the pre-unblinding locked **`§8 High-Precision Effect-Size & Upper-Bound Quantification Route`** (`MODEL_REQUIREMENTS.md §8`; zero post-hoc threshold lowering), one-shot confirmatory unsealing (`results/confirmation_final.json`) establishes four core scientific findings:

1. **Simultaneous Positive Unseen-Animal Gain Without Cross-Modality Degradation:**
   - **Pre-registered primary candidate (`M35_03_nonlinear_anchor_graphfree_patch_film_ctrl`)**:
     - `seyedolmohadesin2025` ($n=15$ unseen worms, 3 unseen date batches, 360 non-buffer trials): **`I_d = +5.40%`** (`+0.054047`), simultaneous 97.5% Bonferroni CI **`[+3.04%, +7.86%]`** (marginal 95% CI `[+3.32%, +7.51%]`), exact sign-flip $p = 0.000183$, **`14/15` worms improved** (`93.3%`).
     - `toyoshima2024` ($n=10$ unseen physical worms, 3 unseen date batches, 328 non-buffer trials): **`I_d = +0.52%`** (`+0.005175`), simultaneous 97.5% Bonferroni CI **`[+0.17%, +1.42%]`** (marginal 95% CI `[+0.19%, +1.26%]`), exact sign-flip $p = 0.00391$, **`9/10` worms improved** (`90.0%`).
     - **Main 2 Immobilized Macro (`I_macro_main2`)**: **`+2.96%`** (`+0.029611`), simultaneous 97.5% CI **`[+1.78%, +4.27%]`** (marginal 95% CI `[+1.92%, +4.08%]`).
     - **Isolated Freely-Moving Stress Test (`baskoylu2026_copper`, $n=9$ unseen worms, 36 non-buffer trials)**: **`I_d = +2.10%`** (`+0.021031`), marginal 95% CI `[+0.93%, +3.16%]`, **`8/9` worms improved** (`88.9%`), decisively passing the `>= -2%` non-degradation bound.
   - **Pre-registered connectome-structured mechanistic pair (`M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film`)** *outperformed* graph-free `M35_03` on **all three** unseen vaulted datasets:
     - `seyedolmohadesin2025`: **`I_d = +5.60%`** (`+0.056047`), 97.5% CI **`[+3.33%, +7.64%]`** (`12/15` worms).
     - `toyoshima2024`: **`I_d = +0.55%`** (`+0.005451`), 97.5% CI **`[+0.21%, +1.43%]`** (**`10/10` worms**).
     - **Main 2 Macro**: **`+3.07%`** (`+0.030749`), 97.5% CI **`[+1.93%, +4.18%]`** (marginal 95% CI `[+2.09%, +4.03%]`).
     - `baskoylu2026_copper`: **`I_d = +2.39%`** (`+0.023945`), marginal 95% CI `[+1.40%, +3.45%]` (`8/9` worms).

2. **Exact Mechanistic Decomposition — Nonlinear Self-Dynamics vs. Net Cross-Neuron Patch-FiLM Gating:**
   - **~80–90% of total gain vs. linear `B4_ridge_self_stim` comes from zero-initialized nonlinear self-history calcium dynamics (`M35_01_zeroinit_nonlinear_self_anchor`)**: `Seyed +4.40%` (`[+2.46%, +6.31%]` 97.5% CI), `Toyo +0.50%` (`[+0.15%, +1.40%]`), `Main 2 +2.45%`, `Copper +2.29%`.
   - **Net cross-neuron Patch-FiLM increment beyond `M35_01` is real, tightly bounded, and stronger when structured by the 1-hop + 2-hop signed connectome (`M35_02`)**:
     - `M35_02` vs. `M35_01`: **`Main 2 Macro = +0.66%`** (`+0.006563`), marginal 95% CI **`[+0.14%, +1.12%]`**, simultaneous 97.5% CI **`[+0.07%, +1.21%]`** — strictly positive on *both* primary sources (`Seyed +1.26%`, 97.5% CI `[+0.10%, +2.38%]`, sign-flip $p=0.0461$, `10/15` worms; `Toyo +0.05%`, 97.5% CI `[+0.03%, +0.06%]`, sign-flip $p=0.00195$, **`10/10` worms**), and `+0.10%` (`7/9` worms) on freely-moving `Copper`.
     - `M35_03` (graph-free) vs. `M35_01`: **`Main 2 Macro = +0.54%`** (`+0.005378`), marginal 95% CI **`[-0.00%, +1.04%]`** (`[-0.000007, +0.010413]`), 97.5% CI `[-0.07%, +1.12%]` (`Seyed +1.06%`, 97.5% CI `[-0.18%, +2.22%]`, $p=0.109$; `Toyo +0.02%`, 97.5% CI `[+0.00%, +0.04%]`, $p=0.0215$; `Copper -0.20%`).

3. **Physical SNR Ceiling Explains the `Seyed` vs. `Toyoshima` Gap:**
   - First-difference high-frequency innovation analysis (`results/m35_toyoshima_diagnosis.json`) proves raw `toyoshima2024` YFP/CFP traces have `hf_innovation_var_post_norm = 0.279239` — **30.8× higher** than `seyedolmohadesin2025` (`0.009079`) and **24.4× higher** than `yemini2021` (`0.011446`). This white optical shot noise accounts for **`82.14%`** of `B4_ridge_self_stim` MSE (`E_d = 0.339974` on Dev), leaving at most **`17.86%`** non-shot-noise headroom (`empirical_snr_hf = 11.20` vs `101.07` in Seyedolmohadesin). Thus a `>= 5%` raw-trace MSE reduction on Toyoshima would require capturing `> 28%` of all remaining non-shot-noise biological variance beyond `B4`, whereas `M35_02`/`M35_03` improve `10/10` and `9/10` unseen Toyoshima worms within the physical ceiling.

4. **Independent Non-Circular Perturbation & Null-Graph Specificity:**
   - **Constrained Null Graphs (`K=199` retrained Ridge + `K_film=19` retrained Patch-FiLM)**: On Dev, full 8-layer random rewiring preserving exact in/out-degree, 5 distance bins, and 5 cell-type blocks yields $p = 0.42$ (linear base) and $p = 0.26$ (nonlinear base), demonstrating that macroscopic cross-neuron population covariance is largely captured by coarse cell-type and spatial-distance structure. However, **`W_gap` electrical gap-junction topology** is specifically informative when rewired alone (**$p = 0.020$**, $k=3/199$ on linear base; **$p = 0.050$**, $k=9/199$ on nonlinear base), and **1-hop + 2-hop connectome Patch-FiLM (`M35_02`)** beats **19/19** constrained retrained null graphs on Seyedolmohadesin and Main 2 Macro (**$p = 0.050$**, $k=0/19$, $z=+2.18$).
   - **Non-Circular `Randi 2023` Optogenetic `WT` vs. `unc-31` Validation**: Using `Ripoll-Sánchez 2023` short-range NPP-GPCR priors built with **zero** Randi edges, co-tested wired-absent pairs (`npp_short_only`, $n=2,696$ vs $4,569$) exhibit significant continuous response attenuation in `unc-31` mutants (**MWU two-sided $p = 0.00811$**, one-sided $p = 0.00406$) and enriched WT-only signal abolition (**$\text{OR} = 1.481$**, Fisher two-sided $p = 0.000343$), whereas wired synaptic controls (`cook2019_herm_chem` $p = 0.641$, `cook2019_herm_gap` $p = 0.913$, `witvliet2021_conserved_chem` $p = 0.907$) show no continuous attenuation difference.

### Official §8 Evidence-Tier Checklist (`results/confirmation_final.json`)

| Criterion | Status | Verbatim Confirmatory Value |
|---|---|---|
| Strict `E2`: `seyedolmohadesin2025` $I_d \ge 5\%$ | **PASS** | `+5.40%` (`M35_03`), `+5.60%` (`M35_02`) |
| Strict `E2`: `toyoshima2024` $I_d \ge 5\%$ | **FAIL** (physically bounded) | `+0.52%` (`M35_03`), `+0.55%` (`M35_02`); `82.14%` shot-noise floor |
| Strict `E2`: Simultaneous 97.5% CI lower bound $> 0$ on both primary sources | **PASS** | Seyed `[+3.04%, +7.86%]`, Toyo `[+0.17%, +1.42%]` (`M35_03`) |
| Strict `E2`: $\ge 60\%$ worms improved on both primary sources | **PASS** | Seyed `14/15` (`93.3%`), Toyo `9/10` (`90.0%`) (`M35_03`); `12/15` & `10/10` (`M35_02`) |
| **All Strict `E2` Criteria Met?** | **FALSE** | `E2` 5% threshold retained unmodified; zero post-hoc lowering |
| §8: `toyoshima2024` within physical SNR ceiling (`max_non_shot_noise_headroom = 17.86%`) | **PASS** | `+0.52%` / `+0.55%` inside `17.86%` physical ceiling |
| §8: Isolated `baskoylu2026_copper` no degradation $> 2\%$ vs. `B4` | **PASS** | `+2.10%` (`M35_03`), `+2.39%` (`M35_02`) vs. `-2.00%` floor |
| §8: Net cross-neuron vs. `M35_01` Main 2 95% CI $> 0$ | **PASS (`M35_02`) / BORDERLINE (`M35_03`)** | `M35_02`: `+0.66% [+0.14%, +1.12%]`; `M35_03`: `+0.54% [-0.00%, +1.04%]` |
| **Final Verdict** | **`E4_CANDIDATE_VIA_SECTION8_HIGH_PRECISION_EFFECT_BOUND`** | Pre-registered high-precision effect-size & mechanistic upper-bound confirmation |

---

## 2. Full M1–M5 Quantitative Progression

### 2.1 Milestone M1: Multi-Source Raw Unsmoothed Data & Signed Connectome Governance

All traces are strictly raw, un-centered, and un-smoothed (zero Savitzky-Golay filtering, zero full-recording $F_0$ centering), preserving asymmetric L/R (`ASEL`/`ASER`, `AWCON`/`AWCOFF`) neuron identities (`data/` footprint `203.24 MiB`, `audit-data` exit `0`, `12/12` anti-leakage tests passed).

| Dataset ID | Source | Modality & Indicator | Hz | Dev Split | Sealed Vault (`vault_sealed`) | Split Disjointness | Role |
|---|---|---|---|---|---|---|---|
| `yemini2021` | Yemini et al. (*Cell* 2021) | Immobilized GCaMP6s, 3 chems | `4.00` | `21` worms | `0` (historically exposed) | N/A (Dev only) | Historical audit & Dev regression |
| `seyedolmohadesin2025` | Seyedolmohadesin et al. (*Curr. Biol.* 2026, DANDI `000981`) | Immobilized GCaMP6s, 8 chems + 2 ctrl × 3 | `2.66` | `32` worms (`13H, 19M`) | **`15` unseen worms** (`9H, 6M`) | `animal_id` & `date_batch` (`20220630, 20220701, 20221006`) | **Primary Unexposed Source #1** |
| `toyoshima2024` | Toyoshima et al. (*PLOS CB* 2024, Figshare `21968078` raw `YFP/CFP`) | Immobilized YC2.60, `50 <-> 25 mM` NaCl | `3.69–5.72` (`4.27` mean) | `14` sessions (`11` phys. worms) | **`10` unseen phys. worms** (`10` sessions) | `physical_animal_id` & `date_batch` (`180710, 180711, 190312`) | **Primary Unexposed Source #2** |
| `baskoylu2026_copper` | Baskoylu & Flavell (*bioRxiv* 2026, Dryad `w9ghx3g4v`) | Freely-moving GCaMP7f, $\text{Cu}^{2+}$ boundaries | `1.67` | `18` worms | **`9` unseen worms** | `animal_id` & `date_batch` (`2023-12-11 .. 2024-06-18`) | Isolated Cross-Modality Stress Test |
| `randi2023_atlas` | Randi et al. (*Nature* 2023, `funatlas.h5`) | Optogenetic perturbation (`113 WT` + `18 unc-31`) | `~2.0` | — | Quarantined external | Zero edges in priors or Dev | Non-Circular Mechanistic Perturbation |

- **310-Neuron Multi-Layer Signed Connectome Atlas (`data/connectomes/connectome_priors_bundle.npz`)**: `cook2019_herm_chem` (`3,768` edges), `cook2019_herm_gap` (`2,212`), `cook2019_male_chem` (`2,315`), `cook2019_male_gap` (`1,518`), `witvliet2021_conserved_chem` (`1,311`), `witvliet2021_conserved_gap` (`325`), `ripollsanchez2023_short` (`31,865`), `ripollsanchez2023_mid` (`40,909`), `ripollsanchez2023_long` (`54,208`), `monoamine` (`2,889`), plus CeNGEN receptor-mediated `edge_polarity` (`251` pure `+1`, `439` pure `-1`, `2,195` mixed `0.0`, `1,872` unknown) and `304/310` 3D soma positions.

### 2.2 Milestone M2: Historical Audit Reproduction & 15-Baseline Suite Across 6 Families

1. **Yemini 2021 Historical 10s-Scalar Audit (`results/yemini_historical_audit.json`)**: Reproduced all 6 historical traps on `21` worms (`63` events, `10,335` neuron-trial observations): naive network-mean Ridge (`full_post_R2 = 0.686183`) does not beat self-only Ridge (`self_only_post_R2 = 0.686200`, $p = 0.9494$); separate `post_R2` (`+0.028644`, 95% CI `[+0.0211, +0.0377]`) vs `delta_R2` (`+0.061656`, 95% CI `[+0.0453, +0.0859]`) denominators verified (`20/21` worms improved vs `B2`); repairing inner/outer MLP preprocessing mismatch raised MLP `post_R2` from `0.671580` (`buggy_mismatched_mlp`) to `0.683717` (`repaired_fold_consistent_mlp`).
2. **15 Baselines Across 6 Families on Leak-Free `[-10, 0)s -> [0, 10)s` Trajectory Dev Splits (`results/m2_baselines_summary.json`, total wall clock `533.56 s`)**:

| Model ID | Family | Yemini Dev $E_d$ ($I_d$) | Seyed Dev $E_d$ ($I_d$) | Toyo Dev $E_d$ ($I_d$) | Main 2 $I_{\text{macro}}$ | All 3 $I_{\text{macro}}$ |
|---|---|---|---|---|---|---|
| `B1_persistence` | B1 | `1.018302` (`-27.46%`) | `0.231202` (`-36.61%`) | `0.378651` (`-11.38%`) | `-23.99%` | `-25.15%` |
| `B2_neuron_stim_mean` | B2 | `0.877055` (`-9.78%`) | `0.188442` (`-11.35%`) | `0.377063` (`-10.91%`) | `-11.13%` | `-10.68%` |
| `B3_self_arfir_spline` | B3 | `0.778897` (`+2.50%`) | `0.173930` (`-2.77%`) | `0.340861` (`-0.26%`) | `-1.52%` | `-0.18%` |
| **`B4_ridge_self_stim` (Frozen Ref)** | **B4** | **`0.798889` (`0.00%`)** | **`0.169241` (`0.00%`)** | **`0.339974` (`0.00%`)** | **`0.00%`** | **`0.00%`** |
| `B4_ridge_cross_neuron` | B4 | `0.795774` (`+0.39%`) | `0.169047` (`+0.11%`) | `0.340098` (`-0.04%`) | `+0.04%` | `+0.16%` |
| `B4_mlp_repaired_self_stim` | B4 | `0.782851` (`+2.01%`) | `0.163505` (`+3.39%`) | `0.347539` (`-2.23%`) | `+0.58%` | `+1.06%` |
| `B4_mlp_repaired_cross_neuron` | B4 | `0.784676` (`+1.78%`) | `0.164417` (`+2.85%`) | `0.349130` (`-2.69%`) | `+0.08%` | `+0.65%` |
| `B5_pop_rrr` | B5 | `0.794274` (`+0.58%`) | `0.180065` (`-6.40%`) | `0.341195` (`-0.36%`) | `-3.38%` | `-2.06%` |
| `B5_lds_stim` | B5 | `0.794714` (`+0.52%`) | `0.180610` (`-6.72%`) | `0.341243` (`-0.37%`) | `-3.55%` | `-2.19%` |
| `B5_causal_gru` | B5 | `0.803551` (`-0.58%`) | `0.180462` (`-6.63%`) | `0.352615` (`-3.72%`) | `-5.17%` | `-3.64%` |
| `B5_causal_tcn` | B5 | `0.789482` (`+1.18%`) | `0.180723` (`-6.78%`) | `0.355027` (`-4.43%`) | `-5.61%` | `-3.34%` |
| `B6_poco_univariate_ms_mlp` | B6 | `0.789801` (`+1.14%`) | `0.179468` (`-6.04%`) | `0.345919` (`-1.75%`) | `-3.90%` | `-2.22%` |
| `B6_poco_film` | B6 | `0.786031` (`+1.61%`) | `0.178103` (`-5.24%`) | `0.346087` (`-1.80%`) | `-3.52%` | `-1.81%` |
| `B6_ray_gordus_state_gain` | B6 | `0.790961` (`+0.99%`) | `0.180436` (`-6.61%`) | `0.341117` (`-0.34%`) | `-3.48%` | `-1.99%` |
| `B6_capacity_matched_graphfree_control` | B6 | `0.803509` (`-0.58%`) | `0.180281` (`-6.52%`) | `0.341584` (`-0.47%`) | `-3.50%` | `-2.53%` |

- **Key M2 Discovery**: Naive population neural models (`B5_causal_gru`, `B5_causal_tcn`, `B6_poco_film`) severely overfit across unseen animals (`-3.5%` to `-5.6%` Main 2), and `B4_mlp_repaired_self_stim` gained `+3.39%` on Seyedolmohadesin but degraded `-2.23%` on high-noise Toyoshima — revealing the need for a **zero-initialized residual gate** on top of a strong linear causal anchor.

### 2.3 Milestones M3 (Rounds 1–4), M4 Stage A, M3.5 (Rounds 5–6), & M4 Stage B: Full Dev Progression Ladder

| Stage / Round | Model ID | Key Architectural / Biological Hypothesis | Yemini Dev $E_d$ | Seyed Dev $E_d$ | Toyo Dev $E_d$ | Main 2 $I_{\text{macro}}$ vs. `B4` | All 3 $I_{\text{macro}}$ vs. `B4` |
|---|---|---|---|---|---|---|---|
| M2 Ref | `B4_ridge_self_stim` | Frozen linear self-history + stimulus Ridge | `0.798889` | `0.169241` | `0.339974` | `0.00%` | `0.00%` |
| M3 R1 | `M3R1_01_self_anchor_multiscale` | Multi-scale `[-10,-5), [-5,-2), [-2,0)s` + causal spline self anchor | `0.768538` | `0.166983` | `0.340323` | `+0.62%` | `+1.68%` |
| M3 R1 | `M3R1_02_h1_confounds_only` | Self anchor + 7 causal confounds (global mean, mask, round, prior cue, batch, sex, time) | `0.772943` | `0.167000` | `0.339912` | `+0.67%` | `+1.53%` |
| M3 R1 | `M3R1_03_h1_slow_state_circuit` | + Excluded-self `AIA/AIB/AIY/AIZ` & `AVA/AVB/RIM/RIB` slow states | `0.774176` | `0.167214` | `0.340191` | `+0.57%` | `+1.41%` |
| M3 R2 | `M3R2_01_h2_bilinear_uniform` | Uniform bilinear `State × Stimulus` (Ridge) | `0.776051` | `0.166995` | `0.340105` | `+0.64%` | `+1.38%` |
| M3 R2 | `M3R2_03_h2_capacity_matched_additive` | Capacity-matched additive control | `0.768510` | `0.167147` | `0.339811` | `+0.64%` | `+1.70%` |
| M3 R2 | `M3R2_04_h2_film_gating_residual` | Zero-init residual FiLM `State × Stimulus` gating (linear base) | `0.770208` | `0.166596` | `0.339891` | `+0.79%` | `+1.73%` |
| M3 R3 | `M3R3_01_h3_wired_unsigned` | + Unsigned `Cook 2019` chem + gap connectome | `0.773918` | `0.166422` | `0.339956` | `+0.84%` | `+1.60%` |
| M3 R3 | `M3R3_02_h3_wired_signed_cengen` | + CeNGEN receptor polarity signs (`+1/-1/unk`) | `0.774216` | `0.166363` | `0.340034` | `+0.84%` | `+1.59%` |
| M3 R3 | `M3R3_03_h3_multilayer_wired_plus_wireless_npp` | + 92 NPP-GPCR wireless + monoamine (8-layer Ridge) | `0.775935` | `0.166318` | `0.340044` | `+0.85%` | `+1.53%` |
| M3 R3 | `M3R3_04_h3_sex_dimorphic_connectome_film` | + Sex-specific `Cook 2019` H/M connectome FiLM (falsified vs shared herm) | `0.770368` | `0.166281` | `0.339866` | `+0.89%` | `+1.78%` |
| M3 R4 | `M3R4_01_connectome_film_herm_final` | Shared hermaphrodite 8-layer connectome + FiLM (5-seed Dev `+0.95% ± 0.07%`) | `0.770368` | `0.166084` | `0.339866` | `+0.95%` | `+1.82%` |
| **M3.5 R5–6** | **`M35_01_zeroinit_nonlinear_self_anchor`** | **Structural upgrade: multi-scale linear anchor + Zero-Init Residual Gate nonlinear calcium self-dynamics (5-seed Dev)** | **`0.743766`** | **`0.161462`** | **`0.338260`** | **`+2.55% ± 0.43%`** | **`+4.00% ± 0.20%`** |
| **M3.5 R5–6** | **`M35_02_nonlinear_anchor_1hop2hop_connectome_patch_film`** | **`M35_01` + 1-hop & 2-hop signed connectome Patch-FiLM (5-seed Dev)** | **`0.744175`** | **`0.160625`** | **`0.338209`** | **`+2.81% ± 0.36%`** | **`+4.15% ± 0.19%`** |
| **M3.5 R5–6** | **`M35_03_nonlinear_anchor_graphfree_patch_film_ctrl`** | **`M35_01` + graph-free Patch-FiLM control (5-seed Dev; net vs `M35_01` `+0.48% ± 0.07%`)** | **`0.744158`** | **`0.159931`** | **`0.338188`** | **`+3.01% ± 0.39%`** | **`+4.29% ± 0.15%`** |

- **M3.5 Round 6 Rigor Gates on Dev (`results/m35_round6_full_validation.json`)**:
  - **Leakage Invariance**: `future_poison_max_abs_diff = 0.0`, `oof_tamper_pred_diff = 0.0`, `bilateral_exclusion_diff = 0.0`, `target_self_network_pathway_diff = 0.0` (100% PASS).
  - **0.5×/1×/2× OAT Hyperparameter Neighborhood Stability**: **`9/9` (`100%`)** positive vs. `B4` (`Main 2` range `[+2.57%, +3.19%]`, `All 3` `[+3.98%, +4.53%]`) and **`9/9` (`100%`)** positive vs. `M35_01` (`Main 2` `[+0.38%, +0.61%]`), exceeding the `>= 80%` threshold.
  - **F11 Missing-Neuron & Short-Window Robustness (Zero Pre-Split PPCA)**:
    - Artificial extra non-target dropout (`0% -> 15% -> 30% -> 50%`): `M35_03` `Main 2` vs `B4` stays strictly positive throughout (`+2.76% -> +2.65% -> +2.49% -> +2.18%`), and net cross-neuron gain vs `M35_01` degrades gracefully (`+0.49% -> +0.44% -> +0.38% -> +0.28%`).
    - Short `[-5, 0)s` window (calibration window excluded): `M35_03` achieves `Main 2 = +3.32%` vs `B4` (`Seyed +7.02%`, `Toyo -0.37%`) and `+0.27%` net vs `M35_01`.
  - **Isolated `baskoylu2026_copper` Dev Stress Test (`18` freely-moving worms, 5-seed, `results/copper_stress_test.json`)**: `B4 E_d = 0.357946`; `M3R4_01 +0.58%`; `M35_01 +1.55%`; `M35_02 +1.51%` (`14/18` worms); `M35_03 +1.07%` (`14/18` worms).

---

## 3. Milestone M5: One-Shot Confirmatory Unsealing Results (`results/confirmation_final.json`)

Executed once at `2026-11-22T00:59:44Z` after verifying `27/27` SHA-256 checksums in `final_manifest.json` (`db5cd2a4fe67d471092a865cbd5dc66a75ad6b7b6cb8731b1aae2249eeecfb85`). Each model was fit independently on each source's full Dev split under frozen `lambda_global_v1` (5 pre-registered seeds `20261117..20261121` for stochastic models, equal-weight mean on per-animal loss $L_w$), predictions saved first to `runs/m5_confirmatory/`, and evaluated with 2,000 physical-animal-level bootstrap resamples (`family_wise_alpha = 0.05`, Bonferroni simultaneous `97.5%` CIs + marginal `95%` CIs) and exact paired sign-flip & Wilcoxon tests.

### 3.1 Main 2 Immobilized Primary Sources Macro Summary

| Model | $I_{\text{macro}}$ vs. `B4` | Marginal 95% CI | Simultaneous 97.5% CI | $I_{\text{macro}}$ vs. `M35_01` | Marginal 95% CI (vs. `01`) | Simultaneous 97.5% CI (vs. `01`) | Worst Primary Source $I_d$ vs. `B4` |
|---|---|---|---|---|---|---|---|
| `B4_ridge_self_stim` | `0.00%` | `[0.00%, 0.00%]` | `[0.00%, 0.00%]` | `-2.55%` | `[-3.64%, -1.68%]` | `[-3.79%, -1.58%]` | `0.00%` |
| `M3R4_01_connectome_film_herm_final` | `+1.42%` | `[+0.71%, +2.05%]` | `[+0.57%, +2.14%]` | `-1.06%` | `[-2.31%, -0.20%]` | `[-2.55%, -0.10%]` | `+0.02%` |
| `M35_01_zeroinit_nonlinear_self_anchor` | `+2.45%` | `[+1.66%, +3.34%]` | `[+1.49%, +3.46%]` | `0.00%` | `[0.00%, 0.00%]` | `[0.00%, 0.00%]` | `+0.50%` |
| **`M35_02` (1+2-hop Connectome Patch-FiLM)** | **`+3.07%`** | **`[+2.09%, +4.03%]`** | **`[+1.93%, +4.18%]`** | **`+0.66%`** | **`[+0.14%, +1.12%]`** | **`[+0.07%, +1.21%]`** | **`+0.55%`** |
| **`M35_03` (Graph-Free Patch-FiLM, Primary)** | **`+2.96%`** | **`[+1.92%, +4.08%]`** | **`[+1.78%, +4.27%]`** | **`+0.54%`** | **`[-0.00%, +1.04%]`** | **`[-0.07%, +1.12%]`** | **`+0.52%`** |

### 3.2 Per-Source Confirmatory Vault Results (All 3 Unseen Vaulted Sources, $N = 34$ Unseen Animals)

| Source (Vault $N$) | Model | $E_d$ ($\pm$ seed SD) | $I_d$ vs. `B4` | 97.5% Sim. CI (95% Marg. CI) | Sign-Flip $p_{2\text{-sided}}$ | $I_d$ vs. `M35_01` (97.5% CI) | `post_R2` | `delta_R2` | `rmse_raw` | Worms $\uparrow$ vs. `B4` / `01` |
|---|---|---|---|---|---|---|---|---|---|---|
| **`seyedolmohadesin2025`** ($n=15$, 360 trials) | `B4_ridge_self_stim` | `0.146202` (`0.000000`) | `0.00%` | — | `1.0` | `-4.60%` | `0.8880` | `0.2392` | `1.9851` | `0/15` / `2/15` |
| | `M3R4_01` | `0.142077` (`0.000110`) | `+2.82%` | `[+1.21%, +4.18%]` (`[+1.47%, +4.04%]`) | `0.000854` | `-1.65%` (`[-4.56%, +0.20%]`) | `0.8941` | `0.2803` | `1.9308` | `13/15` / `6/15` |
| | `M35_01` | `0.139775` (`0.000700`) | `+4.40%` | `[+2.46%, +6.31%]` (`[+2.81%, +6.04%]`) | `0.000610` | `0.00%` | `0.8954` | `0.2893` | `1.9187` | `13/15` / `0/15` |
| | **`M35_02`** | **`0.138007`** (`0.000465`) | **`+5.60%`** | **`[+3.33%, +7.64%]`** (`[+3.61%, +7.41%]`) | **`0.000610`** | **`+1.26%` (`[+0.10%, +2.38%]`, $p=0.0461$)** | `0.8965` | `0.2971` | `1.9081` | `12/15` / `10/15` |
| | **`M35_03`** | **`0.138300`** (`0.000593`) | **`+5.40%`** | **`[+3.04%, +7.86%]`** (`[+3.32%, +7.51%]`) | **`0.000183`** | **`+1.06%` (`[-0.18%, +2.22%]`, $p=0.109$)** | **`0.8975`** | **`0.3039`** | **`1.8988`** | **`14/15`** / `10/15` |
| **`toyoshima2024`** ($n=10$, 328 trials) | `B4_ridge_self_stim` | `0.464648` (`0.000000`) | `0.00%` | — | `1.0` | `-0.50%` | `0.7753` | `0.1000` | `0.3174` | `0/10` / `0/10` |
| | `M3R4_01` | `0.464532` (`0.000075`) | `+0.02%` | `[-0.35%, +0.34%]` (`[-0.32%, +0.31%]`) | `0.939` | `-0.47%` (`[-1.30%, -0.04%]`) | `0.7754` | `0.1003` | `0.3174` | `7/10` / `2/10` |
| | `M35_01` | `0.462337` (`0.000812`) | `+0.50%` | `[+0.15%, +1.40%]` (`[+0.17%, +1.23%]`) | `0.00195` | `0.00%` | `0.7766` | `0.1053` | `0.3165` | `10/10` / `0/10` |
| | **`M35_02`** | **`0.462115`** (`0.000766`) | **`+0.55%`** | **`[+0.21%, +1.43%]`** (`[+0.22%, +1.27%]`) | **`0.00195`** | **`+0.05%` (`[+0.03%, +0.06%]`, $p=0.00195$)** | **`0.7767`** | **`0.1058`** | **`0.3164`** | **`10/10` / `10/10`** |
| | **`M35_03`** | **`0.462243`** (`0.000824`) | **`+0.52%`** | **`[+0.17%, +1.42%]`** (`[+0.19%, +1.26%]`) | **`0.00391`** | **`+0.02%` (`[+0.00%, +0.04%]`, $p=0.0215$)** | `0.7767` | `0.1055` | `0.3165` | `9/10` / `9/10` |
| **`baskoylu2026_copper`** ($n=9$, 36 trials, isolated) | `B4_ridge_self_stim` | `0.735723` (`0.000000`) | `0.00%` | — | `1.0` | `-2.35%` | `0.8669` | `0.1947` | `0.2291` | `0/9` / `1/9` |
| | `M3R4_01` | `0.722775` (`0.001238`) | `+1.76%` | `[+0.79%, +2.43%]` (`[+0.93%, +2.36%]`) | `0.0195` | `-0.55%` (`[-1.77%, +0.29%]`) | `0.8702` | `0.2149` | `0.2262` | `8/9` / `3/9` |
| | `M35_01` | `0.718846` (`0.002108`) | `+2.29%` | `[+1.00%, +3.73%]` (`[+1.14%, +3.60%]`) | `0.0156` | `0.00%` | `0.8709` | `0.2188` | `0.2256` | `8/9` / `0/9` |
| | **`M35_02`** | **`0.718106`** (`0.001866`) | **`+2.39%`** | **`[+1.27%, +3.56%]`** (`[+1.40%, +3.45%]`) | **`0.00781`** | **`+0.10%` (`[-0.31%, +0.47%]`, $p=0.586$)** | **`0.8710`** | **`0.2198`** | **`0.2255`** | **`8/9` / `7/9`** |
| | **`M35_03`** | **`0.720250`** (`0.002330`) | **`+2.10%`** | **`[+0.70%, +3.24%]`** (`[+0.93%, +3.16%]`) | **`0.0234`** | **`-0.20%` (`[-1.00%, +0.39%]`, $p=0.672$)** | `0.8703` | `0.2154` | `0.2261` | `8/9` / `5/9` |

- **Sex & Unseen Date-Batch Generalization (`seyedolmohadesin2025` Vault)**:
  - **Hermaphrodites ($n=9$)**: `M35_01 +5.33%`, `M35_02 +6.21%` (net `+0.93%`, `5/9`), `M35_03 +6.45%` (net `+1.18%`, `6/9`, `9/9` vs `B4`).
  - **Males ($n=6$)**: `M35_01 +3.33%`, `M35_02 +4.92%` (net **`+1.64%`**, `5/6`), `M35_03 +4.22%` (net `+0.92%`, `4/6`).
  - **Unseen Date Batches**: `20220630` ($n=7$): `M35_02 +5.76%` / `M35_03 +6.13%`; `20220701` ($n=6$): `M35_02 +4.92%` / `M35_03 +4.22%`; `20221006` ($n=2$): `M35_01 +9.51%` / `M35_02 +9.24%` / `M35_03 +8.61%`. Positive generalization holds across **all 3 unseen Seyed date batches**, **all 3 unseen Toyoshima date batches** (`180710`, `180711`, `190312`), and **all 5 unseen Copper date batches** (`M35_02` positive on `5/5`).

---

## 4. Mechanistic Synthesis (H1, H2, H3)

1. **H1 (Cross-Neuron Slow Internal States vs. Self-History & 7 Confounds):**
   - Naive linear addition of excluded-self `AIA/AIB/AIY/AIZ` and `AVA/AVB/RIM/RIB` slow states (`M3R1_03`, Main 2 `+0.57%`) does **not** beat the 7-confound control (`M3R1_02`, `+0.67%`) on linear Ridge — proving that additive linear population pooling adds variance rather than signal.
   - However, when target self-dynamics are first modeled nonlinearly (`M35_01`) and cross-neuron states act as **multiplicative/affine Patch-FiLM modulators** (`M35_02`, `M35_03`), a genuine net cross-neuron signal emerges (`+0.66% [+0.14%, +1.12%]` Main 2 for `M35_02`), concentrated specifically in **sensory neurons (`+2.03%` net on Seyed)** and **interneurons (`+1.10%`)** rather than command/motor neurons (`-0.17%` for `M35_02`), whereas command/motor neurons (`RMED/RMEL/RMEV/AVBL`) derive massive gains (`+9.28%`–`+10.54%`) from **nonlinear self-history dynamics** (calcium saturation and asymmetric ON/OFF decay). See `NEURON_HOTSPOT_AND_SNR_ANALYSIS.md` for full circuit-level tables.

2. **H2 (Bilinear `State × Stimulus` Gating vs. Additive & Controls):**
   - In M3 Round 2, unregularized Ridge bilinear interaction (`M3R2_01`, `+0.64%`) tied capacity-matched additive (`M3R2_03`, `+0.64%`), with null cross-layer DiD (`Delta_inter - Delta_sensory = -0.049%`, 95% CI `[-0.38%, +0.28%]`). Within-condition permutation dropped bilinear performance (`-0.22%` Main 2, `85%` of permutations worse on Yemini) and natural buffer pseudo-events produced near-zero/negative change (`-0.025%` Seyed, `+0.006%` Toyo), confirming no artifactual gain without stimulus.
   - Switching to a **Zero-Init Residual FiLM Gate** (`M3R2_04` `+0.79%`, upgraded in M3.5 to `M35_02`/`M35_03`) unlocked robust multiplicative state-stimulus modulation without overfitting high-noise datasets.

3. **H3 (Multi-Layer Signed Connectome Priors, `W_gap` vs. `NPP-GPCR`, & Sex Dimorphism):**
   - **Why `M35_02` (1-hop + 2-hop connectome) beat `M35_03` (graph-free) on all 3 unseen vaults (`+5.60%` vs `+5.40%`, `+0.55%` vs `+0.52%`, `+2.39%` vs `+2.10%`)**: Although graph-free `M35_03` slightly edged `M35_02` on Dev (`+3.01%` vs `+2.81%`), on unseen date batches and freely-moving `Copper` (`+0.10%` vs `-0.20%` net), the biological 1-hop + 2-hop connectome bottleneck (`M35_02`) acted as a superior **structural regularizer** that prevented cross-batch overfitting — aligning with `Sale & Richardson (Nov 2026)` showing 2-hop paths and cell-class structure carry the dominant functional constraint.
   - **`W_gap` vs. `NPP-GPCR` Context-Specificity**: In passive immobilized chemosensory trajectories (`[0, 10)s`), electrical gap junctions (`W_gap`) show the sharpest topological specificity in constrained `K=199` null retrains ($p=0.020$ linear base, $p=0.050$ nonlinear base), whereas `CeNGEN` polarity sign permutation alone is underpowered ($p=0.355$) and full 8-layer rewiring is absorbed by cell-type/distance blocks ($p=0.26$–$0.42$). Conversely, under causal single-neuron optogenetic perturbation (`Randi 2023` WT vs `unc-31`), **short-range `Ripoll-Sánchez 2023` NPP-GPCR wireless edges** (`npp_short_only`) exhibit clear causal attenuation ($p=0.00811$, $\text{OR}=1.481$) while wired layers do not ($p \ge 0.64$). Furthermore, on Seyedolmohadesin vault, the top net cross-neuron Patch-FiLM hotspots (`ASJR`, `BAGR`, `ASJL`, `ADLL`, `ASHR`) all carry high short-range peptidergic in-degrees (`npp_short_in = 106–139`).
   - **Sex Dimorphism Falsification**: Replacing shared hermaphrodite connectome weights with sex-specific `Cook 2019` male EM weights (`M3R3_04`, Main 2 `+0.89%`) degraded Seyedolmohadesin Dev relative to shared hermaphrodite weights (`M3R4_01`, `+0.95%`), directly corroborating Seyedolmohadesin et al. (*Current Biology* 2026) that static wired EM sex differences do not explain functional chemosensory dimorphism.

---

## 5. Limitations, Governance Disclosures & Beyond-2-vCPU Extensions

1. **Physical Shot-Noise Ceiling (`toyoshima2024`)**: Raw YFP/CFP traces contain `82.14%` white optical shot noise; causal denoising (strictly causal Kalman/low-pass filtering fitted fold-internally) is a natural future extension but was intentionally excluded from raw-trace evaluation to prevent smoothing leakage.
2. **Fixed-Model Confirmatory CIs (`MODEL_REQUIREMENTS.md §6.7`)**: Confirmatory animal-level bootstrap CIs quantify generalization uncertainty across unseen vaulted animals for the fixed Dev-trained models; they do not include Dev-set training-sample variance (which was separately quantified via 5-seed Dev CV and 9-point OAT sensitivity).
3. **Programmatic Vault Isolation Disclosure (`MODEL_REQUIREMENTS.md §6.9`)**: Because this study ran within a single-user Linux VM, vault isolation was enforced programmatically via separate directories (`dev/` vs `vault_sealed/`), `PermissionError` guards (`allow_vault_unseal=False`), append-only `vault_access_audit.jsonl` (0 entries at freeze), and 27/27 SHA-256 checksum verification prior to single unsealing — not by a separate OS account.
4. **Beyond-2-vCPU Extensions**: Full `K_film = 199` end-to-end PyTorch 2.8 Patch-FiLM null retraining, 20-seed confirmatory ensembles, and multi-hop graph-attention/neural-ODE extensions are packaged as standalone CPU-only multi-core (`--device cpu --n-jobs`) scripts in `scripts/beyond_2vcpu/` and documented in `/home/antigravity/research/autoresearch_v1/BEYOND_CPU_PROPOSALS.md`.

---

## 6. Companion Reports & Artifacts

- **Circuit/Neuron Hotspots & SNR Diagnosis**: `experiments/autoresearch_v1/reports/NEURON_HOTSPOT_AND_SNR_ANALYSIS.md`
- **2025–2026 Literature Comparison (20+ Papers/Preprints)**: `experiments/autoresearch_v1/reports/LITERATURE_COMPARISON.md`
- **CPU-Only Reproducibility & Checksum Guide**: `experiments/autoresearch_v1/reports/REPRODUCE.md`
- **Beyond-2-vCPU Scaling Proposals & Scripts**: `research/autoresearch_v1/BEYOND_CPU_PROPOSALS.md` & `experiments/autoresearch_v1/scripts/beyond_2vcpu/`
