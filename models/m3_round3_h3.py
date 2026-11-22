"""Milestone M3 — Round 3: H3 Receptor-Polarity-Signed Multi-Layer Connectome Priors & Sex Dimorphism.

Implements and genuinely trains/evaluates 4 models anchored on `M3R1_02_h1_confounds_only`
across all 3 immobilized Dev splits (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`)
under unified `configs/lambda_global.json` (zero `dataset_id`-specific hyperparameter branches):

1. `M3R3_01_h3_wired_unsigned`:
   Cook 2019 hermaphrodite wired chemical + electrical gap junctions (+ Witvliet 2021 conserved core),
   unsigned, missingness-precision-weighted (`sum_j W_ij * m_j / (sum_j W_ij + kappa)`), strictly
   excluding target self `W[i,i]=0` AND bilateral partner `W[i, bilateral(i)]=0` (while preserving
   `ASEL/ASER` & `AWCON/OFF` asymmetry as distinct presynaptic sources), over multi-scale causal
   windows (`[-10,-5)s`, `[-5,-2)s`, `[-2,0)s` level & slope), predicting residual over `M3R1_02`
   via strongly regularized Ridge + low-rank stimulus gating (`rank = bilinear_rank = 4`).
2. `M3R3_02_h3_wired_signed_cengen`:
   Receptor-polarity-signed `CeNGEN` chemical + gap junctions: splits `W_chem` explicitly into
   excitatory (`W_chem_pos`, `S_sign > 0`), inhibitory (`W_chem_neg`, `S_sign < 0`), and
   mixed/unknown (`W_chem_unk`) + `W_gap` + `W_witvliet_core`.
3. `M3R3_03_h3_multilayer_wired_plus_wireless_npp`:
   Full 4-layer connectome adding `Ripoll-Sánchez 2023` 92 NPP-GPCR short/mid-range neuropeptide
   and monoamine wireless drives (low-rank SVD compressed to `wireless_rank = 6` with stronger
   `wireless_l2_multiplier = 2.5` L2 regularization) on top of `M3R3_02_h3_wired_signed_cengen`.
4. `M3R3_04_h3_sex_dimorphic_connectome_film`:
   Sex-specific `Cook 2019 H/M` signed multi-layer connectome (`herm` matrices for hermaphrodites
   and unknown; `male` matrices for males) + wireless NPP/monoamine drives, feeding both compact
   connectome drives and excluded-self circuit slow-state into the bounded multiplicative FiLM
   residual gate (from `M3R2_04`) anchored on `M3R1_02`.

Also computes all mandatory F10 mechanistic diagnostics:
  1. 4-Layer Connectome Inter-Layer Correlation (Spearman & Pearson over non-zero edges and over
     incoming trial-level drive features) among `W_chem_pos`, `W_chem_neg`, `W_gap`,
     `W_npp_wireless`, and `W_monoamine`.
  2. Leave-One-Layer-Out Ablation (`-W_chem_signed`, `-W_gap`, `-W_wireless_npp`, `-W_monoamine`,
     plus wired-only vs wireless-only unique/shared variance partition) across all 3 Dev splits.
  3. Sex Dimorphism Analysis on `seyedolmohadesin2025` Dev animals (`sex_matched` vs `herm_only`
     vs `sex_swapped`) with separate `hermaphrodite` and `male` `E_d`, `I_d`, and 1000-resample
     animal-level bootstrap 95% CIs.
  4. End-to-End Leakage Invariance Verification on `M3R3_03_h3_multilayer_wired_plus_wireless_npp`.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from evaluators.scorer import score_saved_predictions_file
from models.m3_round1_h1 import (
    M3R1Model,
    _build_m3_fold_cache,
    _peak_rss_mb,
    _utc_now,
    compute_macro_bootstrap_ci,
    extract_circuit_slow_state_features,
    fit_causal_slow_state_pca,
)
from models.m3_round2_h2 import (
    ResidualFiLMGate,
    _apply_svd_compressor,
    _build_bilinear_features,
    _fit_reduced_rank_ridge,
    _fit_svd_compressor,
)
from src.baselines import (
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
)
from src.data_adapters import DATA_ROOT_DEFAULT, EXP_ROOT_DEFAULT, compute_file_sha256, load_dataset
from src.evaluator import (
    align_records_to_canonical,
    apply_sensor_bounds_to_aligned,
    build_grouped_cv_splits,
    compare_against_reference,
    compute_training_fold_neuron_scales,
    compute_training_fold_sensor_bounds,
    evaluate_predictions,
    extract_causal_trial_records,
    save_CompactPredictions,
)
from src.features import _get_allow_mask_matrix, predict_prior_delta_trajectory

M3R3_MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "run_id": "r3_01_m3r3_01_h3_wired_unsigned",
        "model_id": "M3R3_01_h3_wired_unsigned",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": True,
        "uses_polarity_sign": False,
        "uses_wireless": False,
        "uses_sex_connectome": False,
        "is_stochastic": False,
        "single_factor_changed": "wired_unsigned_cook2019_chem_gap_witvliet_kappa_drive_plus_lowrank_stim_gate",
        "hypothesis": "H3 unsigned wired: Cook 2019 herm chem + gap (+ Witvliet conserved core) missingness-precision-weighted causal drives with strict self+bilateral exclusion + low-rank stimulus gating anchored on M3R1_02.",
    },
    {
        "run_id": "r3_02_m3r3_02_h3_wired_signed_cengen",
        "model_id": "M3R3_02_h3_wired_signed_cengen",
        "parent_id": "r3_01_m3r3_01_h3_wired_unsigned",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": True,
        "uses_polarity_sign": True,
        "uses_wireless": False,
        "uses_sex_connectome": False,
        "is_stochastic": False,
        "single_factor_changed": "split_w_chem_by_cengen_receptor_polarity_pos_neg_unk",
        "hypothesis": "H3 signed wired: Splits Cook 2019 W_chem by CeNGEN receptor polarity S_sign into excitatory (S>0), inhibitory (S<0), and mixed/unknown drives + W_gap + Witvliet core.",
    },
    {
        "run_id": "r3_03_m3r3_03_h3_multilayer_wired_plus_wireless_npp",
        "model_id": "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
        "parent_id": "r3_02_m3r3_02_h3_wired_signed_cengen",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": True,
        "uses_polarity_sign": True,
        "uses_wireless": True,
        "uses_sex_connectome": False,
        "is_stochastic": False,
        "single_factor_changed": "add_ripollsanchez2023_92npp_short_mid_and_monoamine_lowrank_wireless",
        "hypothesis": "H3 4-layer: Adds Ripoll-Sanchez 2023 92 NPP-GPCR short/mid-range neuropeptide and monoamine wireless drives (SVD wireless_rank=6, 2.5x L2) on top of M3R3_02 signed wired connectome.",
    },
    {
        "run_id": "r3_04_m3r3_04_h3_sex_dimorphic_connectome_film",
        "model_id": "M3R3_04_h3_sex_dimorphic_connectome_film",
        "parent_id": "r2_04_m3r2_04_h2_film_gating_residual",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_connectome": True,
        "uses_polarity_sign": True,
        "uses_wireless": True,
        "uses_sex_connectome": True,
        "is_stochastic": True,
        "single_factor_changed": "sex_specific_cook2019_hm_signed_multilayer_connectome_plus_bounded_film_gate",
        "hypothesis": "H3 sex-dimorphic FiLM: Routes hermaphrodites/unknown through Cook 2019 herm matrices and males through Cook 2019 male matrices + CeNGEN polarity + wireless NPP/monoamine + excluded-self slow state into bounded multiplicative FiLM residual gate anchored on M3R1_02.",
    },
]


def _code_bundle_sha256(exp_root: Path) -> str:
    h = hashlib.sha256()
    rel_files = [
        "protocol.json",
        "configs/lambda_global.json",
        "src/evaluator.py",
        "src/features.py",
        "src/baselines.py",
        "evaluators/scorer.py",
        "models/m3_round1_h1.py",
        "models/m3_round2_h2.py",
        "models/m3_round3_h3.py",
    ]
    for rel in rel_files:
        p = exp_root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Connectome Atlas Loader (Strict Self & Bilateral-Pair Exclusion)
# ---------------------------------------------------------------------------
def load_signed_multilayer_connectome(
    canonical_neurons: Sequence[str],
    data_root: Path = DATA_ROOT_DEFAULT,
) -> Dict[str, Any]:
    """Load Cook 2019 H/M, Witvliet 2021, Ripoll-Sánchez 2023 NPP/monoamine & CeNGEN polarity.

    Bundle convention verified directly on biology: `W_raw[pre, post]`.
    We transpose to `W_in[post, pre] = W_raw.T * allow` so row `i` sums presynaptic inputs `j -> i`,
    strictly enforcing `W_in[i, i] == 0` AND `W_in[i, bilateral(i)] == 0` (including `AVAR->AVAL`,
    `AIYR->AIYL`, `ASER->ASEL`, `AWCOFF->AWCON`), while preserving `ASEL` vs `ASER` and `AWCON` vs
    `AWCOFF` as distinct presynaptic columns onto downstream neurons (e.g. `AIYL`).
    """
    bundle_path = data_root / "connectomes" / "connectome_priors_bundle.npz"
    raw = np.load(bundle_path, allow_pickle=True)
    bundle_neurons = [str(n) for n in raw["canonical_neurons"]]
    if list(canonical_neurons) != bundle_neurons:
        idx_map = {n: i for i, n in enumerate(bundle_neurons)}
        perm = np.array([idx_map[n] for n in canonical_neurons], dtype=np.int64)
        def _reindex(mat: np.ndarray) -> np.ndarray:
            return mat[np.ix_(perm, perm)].astype(np.float32)
    else:
        def _reindex(mat: np.ndarray) -> np.ndarray:
            return np.asarray(mat, dtype=np.float32)

    allow = _get_allow_mask_matrix(
        canonical_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=False
    )
    allow_f = allow.astype(np.float32)

    def _to_incoming(key: str) -> np.ndarray:
        w = np.nan_to_num(_reindex(raw[key]), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        # Transpose W[pre, post] -> W_in[post, pre] and strictly mask self + bilateral
        return (w.T * allow_f).astype(np.float32)

    s_sign_in = _reindex(raw["edge_polarity"]).T  # [post, pre], finite values in {-1, 0, +1}, else NaN
    pos_mask = (np.isfinite(s_sign_in) & (s_sign_in > 0.0)).astype(np.float32)
    neg_mask = (np.isfinite(s_sign_in) & (s_sign_in < 0.0)).astype(np.float32)
    unk_mask = (1.0 - np.clip(pos_mask + neg_mask, 0.0, 1.0)).astype(np.float32)

    w_hc = _to_incoming("cook2019_herm_chem")
    w_hg = _to_incoming("cook2019_herm_gap")
    w_mc = _to_incoming("cook2019_male_chem")
    w_mg = _to_incoming("cook2019_male_gap")
    w_wit = (_to_incoming("witvliet2021_conserved_chem") + _to_incoming("witvliet2021_conserved_gap")).astype(np.float32)
    w_npp_short = _to_incoming("ripollsanchez2023_short")
    w_npp_mid = _to_incoming("ripollsanchez2023_mid")
    w_mono = _to_incoming("monoamine")

    mats: Dict[str, np.ndarray] = {
        "W_chem_herm_unsigned": w_hc,
        "W_chem_herm_pos": (w_hc * pos_mask).astype(np.float32),
        "W_chem_herm_neg": (w_hc * neg_mask).astype(np.float32),
        "W_chem_herm_unk": (w_hc * unk_mask).astype(np.float32),
        "W_gap_herm": w_hg,
        "W_chem_male_unsigned": w_mc,
        "W_chem_male_pos": (w_mc * pos_mask).astype(np.float32),
        "W_chem_male_neg": (w_mc * neg_mask).astype(np.float32),
        "W_chem_male_unk": (w_mc * unk_mask).astype(np.float32),
        "W_gap_male": w_mg,
        "W_witvliet_core": w_wit,
        "W_npp_short": w_npp_short,
        "W_npp_mid": w_npp_mid,
        "W_monoamine": w_mono,
    }

    # Strict runtime assertion: zero diagonal & zero bilateral pairs on every layer
    idx = {n: i for i, n in enumerate(canonical_neurons)}
    check_pairs = [("AVAL", "AVAR"), ("AIYL", "AIYR"), ("ASEL", "ASER"), ("AWCON", "AWCOFF")]
    for k, m in mats.items():
        if float(np.max(np.abs(np.diag(m)))) != 0.0:
            raise AssertionError(f"Non-zero diagonal in {k}")
        for a, b in check_pairs:
            if a in idx and b in idx:
                if float(m[idx[a], idx[b]]) != 0.0 or float(m[idx[b], idx[a]]) != 0.0:
                    raise AssertionError(f"Non-zero bilateral edge ({a},{b}) in {k}")

    row_sums = {k: np.sum(m, axis=1).astype(np.float32) for k, m in mats.items()}
    return {
        "canonical_neurons": list(canonical_neurons),
        "allow_mask": allow,
        "matrices": mats,
        "row_sums": row_sums,
        "bundle_sha256": compute_file_sha256(bundle_path),
    }


# ---------------------------------------------------------------------------
# 2. Multi-Scale Causal Presynaptic Activity & Precision-Weighted Drives
# ---------------------------------------------------------------------------
def extract_multiscale_presynaptic_state(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract strictly causal multi-scale pre-stimulus activity `S_mat [N, 4]` and mask `M_obs [N]`.

    Windows (all strictly `< onset_idx`, within `[-10.0, 0.0)s`):
      col 0: `[-10, -5)s` mean (slow baseline level)
      col 1: `[-5, -2)s` mean (mid pre-stimulus level)
      col 2: `[-2, 0)s` mean (fast pre-stimulus level)
      col 3: `[-2, 0)s` causal slope (`[-1, 0)s - [-2, -1)s`, fallback `w_2_0 - w_5_2`)
    """
    x_pre = np.asarray(trial["x_pre"], dtype=np.float32)  # [T_pre, N]
    m_pre = np.asarray(trial["m_pre"], dtype=bool)        # [T_pre, N]
    t_pre, n_canon = x_pre.shape
    fps = max(float(trial.get("fps", 2.0)), 0.1)
    dt_sec = 1.0 / fps

    z_pre = np.where(
        m_pre & np.isfinite(x_pre),
        np.clip((x_pre - mu_train[None, :]) / np.maximum(s_n[None, :], 1e-6), -6.0, 6.0),
        0.0,
    ).astype(np.float32)
    m_f = (m_pre & np.isfinite(x_pre)).astype(np.float32)

    # Frame timestamps relative to onset_idx: t_rel in [-T_pre*dt, 0)
    t_rel = -(np.arange(t_pre, 0, -1, dtype=np.float32)) * dt_sec

    def _win_mean(lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
        sel = (t_rel >= lo) & (t_rel < hi)
        if not np.any(sel):
            return np.zeros(n_canon, dtype=np.float32), np.zeros(n_canon, dtype=np.float32)
        cnt = np.sum(m_f[sel], axis=0)
        val = np.sum(z_pre[sel], axis=0) / np.maximum(cnt, 1.0)
        return val.astype(np.float32), (cnt > 0.0).astype(np.float32)

    w_10_5, obs_10_5 = _win_mean(-10.01, -5.0)
    w_5_2, obs_5_2 = _win_mean(-5.0, -2.0)
    w_2_0, obs_2_0 = _win_mean(-2.0, 0.01)
    w_1_0, obs_1_0 = _win_mean(-1.0, 0.01)
    w_2_1, obs_2_1 = _win_mean(-2.0, -1.0)

    # Causal slope in [-2, 0)s
    has_fine = (obs_1_0 > 0.0) & (obs_2_1 > 0.0)
    slope_2_0 = np.where(has_fine, w_1_0 - w_2_1, np.where((obs_2_0 > 0.0) & (obs_5_2 > 0.0), w_2_0 - w_5_2, 0.0))

    m_obs = ((obs_2_0 + obs_5_2 + obs_10_5) > 0.0).astype(np.float32)
    s_mat = np.column_stack([w_10_5, w_5_2, w_2_0, slope_2_0]).astype(np.float32)
    s_mat = np.clip(s_mat * m_obs[:, None], -6.0, 6.0)
    return s_mat, m_obs


def compute_layer_drive(
    w_in: np.ndarray,
    tot_w: np.ndarray,
    s_mat: np.ndarray,
    m_obs: np.ndarray,
    kappa: float = 4.0,
) -> np.ndarray:
    """Missingness-precision-weighted incoming presynaptic drive:
    `drive[i, :] = (sum_j W_in[i, j] * m_j * S[j, :]) / (sum_j W_in[i, j] + kappa)`.
    """
    denom = (tot_w + float(kappa))[:, None]
    num = w_in @ (s_mat * m_obs[:, None])
    return np.clip(num / np.maximum(denom, 1e-6), -6.0, 6.0).astype(np.float32)


def extract_connectome_drive_blocks(
    trial: Mapping[str, Any],
    atlas: Mapping[str, Any],
    s_n: np.ndarray,
    mu_train: np.ndarray,
    kappa: float = 4.0,
    mode: str = "wired_signed",
    sex_routing_mode: str = "sex_matched",
    ablate_layer: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute `(wired_drives [N, D_w], wireless_drives [N, D_wl] or None)` for a single trial."""
    s_mat, m_obs = extract_multiscale_presynaptic_state(trial, s_n, mu_train)
    mats = atlas["matrices"]
    rsums = atlas["row_sums"]

    use_male = False
    if mode == "sex_dimorphic_multilayer":
        raw_sex = str(trial.get("sex", "unknown")).lower()
        if sex_routing_mode == "sex_matched":
            use_male = raw_sex == "male"
        elif sex_routing_mode == "herm_only":
            use_male = False
        elif sex_routing_mode == "sex_swapped":
            use_male = raw_sex != "male"
        else:
            raise ValueError(f"Unknown sex_routing_mode: {sex_routing_mode}")

    sx = "male" if use_male else "herm"
    wired_blocks: List[np.ndarray] = []

    if mode == "wired_unsigned":
        for k in ("W_chem_herm_unsigned", "W_gap_herm", "W_witvliet_core"):
            wired_blocks.append(compute_layer_drive(mats[k], rsums[k], s_mat, m_obs, kappa=kappa))
        return np.hstack(wired_blocks).astype(np.float32), None

    # Signed wired layers (subject to optional leave-one-layer-out ablation)
    if ablate_layer not in ("W_chem_signed", "all_wired"):
        for k in (f"W_chem_{sx}_pos", f"W_chem_{sx}_neg", f"W_chem_{sx}_unk"):
            wired_blocks.append(compute_layer_drive(mats[k], rsums[k], s_mat, m_obs, kappa=kappa))
    if ablate_layer not in ("W_gap", "all_wired"):
        k_gap = f"W_gap_{sx}"
        wired_blocks.append(compute_layer_drive(mats[k_gap], rsums[k_gap], s_mat, m_obs, kappa=kappa))
    if ablate_layer != "all_wired":
        wired_blocks.append(compute_layer_drive(mats["W_witvliet_core"], rsums["W_witvliet_core"], s_mat, m_obs, kappa=kappa))

    wired_arr = np.hstack(wired_blocks).astype(np.float32) if wired_blocks else np.zeros((s_mat.shape[0], 0), dtype=np.float32)

    if mode == "wired_signed" or ablate_layer == "all_wireless":
        return wired_arr, None

    # Wireless layers (Ripoll-Sánchez 2023 short/mid NPP-GPCR + monoamine)
    wireless_blocks: List[np.ndarray] = []
    if ablate_layer != "W_wireless_npp":
        wireless_blocks.append(compute_layer_drive(mats["W_npp_short"], rsums["W_npp_short"], s_mat, m_obs, kappa=kappa))
        wireless_blocks.append(compute_layer_drive(mats["W_npp_mid"], rsums["W_npp_mid"], s_mat, m_obs, kappa=kappa))
    if ablate_layer != "W_monoamine":
        wireless_blocks.append(compute_layer_drive(mats["W_monoamine"], rsums["W_monoamine"], s_mat, m_obs, kappa=kappa))

    wireless_arr = (
        np.hstack(wireless_blocks).astype(np.float32)
        if wireless_blocks
        else np.zeros((s_mat.shape[0], 0), dtype=np.float32)
    )
    return wired_arr, wireless_arr


# ---------------------------------------------------------------------------
# 3. Diagonal-Penalty Sample-Scaled Ridge Helper (Honors wireless_l2_multiplier)
# ---------------------------------------------------------------------------
def _solve_weighted_ridge_sample_scaled(
    x: np.ndarray,
    y: np.ndarray,
    col_l2_penalties: np.ndarray,
) -> np.ndarray:
    """Closed-form Ridge `(X^T X + diag(alpha_vec))^{-1} X^T Y` with sample-size scaling."""
    n_samp, d = x.shape
    b_dim = y.shape[1]
    if n_samp == 0 or d == 0:
        return np.zeros((d, b_dim), dtype=np.float32)
    scale = max(1.0, float(np.sqrt(float(n_samp) / 100.0)))
    alpha_vec = np.asarray(col_l2_penalties, dtype=np.float64) * scale
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    gram = x64.T @ x64
    gram[np.diag_indices(d)] += alpha_vec
    rhs = x64.T @ y64
    w = np.linalg.solve(gram, rhs)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Unified M3 Round 3 Connectome Model Class (Anchored on M3R1_02)
# ---------------------------------------------------------------------------
class M3R3Model:
    """Implements all 4 M3 Round 3 H3 connectome models anchored on `M3R1_02_h1_confounds_only`."""

    def __init__(
        self,
        model_id: str,
        config: Mapping[str, Any],
        canonical_neurons: Sequence[str],
        functional_tiers: Sequence[str],
        atlas: Mapping[str, Any],
        sex_routing_mode: str = "sex_matched",
        ablate_layer: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.config = dict(config)
        self.hp = dict(config.get("hyperparameters", config))
        self.canonical_neurons = list(canonical_neurons)
        self.functional_tiers = list(functional_tiers)
        self.atlas = atlas
        self.sex_routing_mode = str(sex_routing_mode)
        self.ablate_layer = ablate_layer
        self.n_canon = len(self.canonical_neurons)
        self.state: Dict[str, Any] = {}

        tier_names = ("sensory", "interneuron", "premotor_command", "motor", "pharyngeal")
        self._tier_oh = np.zeros((self.n_canon, len(tier_names)), dtype=np.float32)
        for i, t_str in enumerate(self.functional_tiers):
            if t_str in tier_names:
                self._tier_oh[i, tier_names.index(t_str)] = 1.0

    def _drive_mode(self) -> str:
        if self.model_id == "M3R3_01_h3_wired_unsigned":
            return "wired_unsigned"
        if self.model_id == "M3R3_02_h3_wired_signed_cengen":
            return "wired_signed"
        if self.model_id == "M3R3_03_h3_multilayer_wired_plus_wireless_npp":
            return "multilayer_signed"
        if self.model_id == "M3R3_04_h3_sex_dimorphic_connectome_film":
            return "sex_dimorphic_multilayer"
        raise ValueError(f"Unknown M3R3 model_id: {self.model_id}")

    def _extract_u_stim(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
    ) -> np.ndarray:
        stim_id = str(trial["stimulus_id"])
        is_buf = bool(trial.get("is_buffer_control", False))
        stim_oh_nonbuf = _encode_stim_onehot(stim_id, is_buf, priors["stim_vocab"])[:-1]
        stim_mat = np.repeat(stim_oh_nonbuf[None, :], self.n_canon, axis=0)
        _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, self.n_canon)
        prior_norm_2 = np.clip((prior_coef / s_n[:, None].astype(np.float32))[:, :2], -4.0, 4.0)
        return np.hstack([stim_mat, prior_norm_2]).astype(np.float32)

    def _fit_or_reuse_anchor(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
        anchor_cache: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[M3R1Model, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        if anchor_cache is not None:
            return anchor_cache["anchor"], anchor_cache["pca_state"], anchor_cache["train_res_map"]

        anchor = M3R1Model("M3R1_02_h1_confounds_only", self.config, self.canonical_neurons, self.functional_tiers)
        anchor.fit(train_trials, s_n, fold_cache=fold_cache)
        priors = anchor.state["priors"]
        rank = int(self.hp.get("rank", 6))
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        pca_state = fit_causal_slow_state_pca(
            non_buf, s_n, priors["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )
        train_res_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for tr in non_buf:
            tid = str(tr.get("trial_id", ""))
            if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
                c_norm, rval, _, _ = fold_cache["proj_targets"][tid]
            else:
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors, s_n, self.n_canon)
            p1 = anchor._predict_stage1_coef(tr, s_n, priors, anchor.state["s1"], fold_cache=fold_cache)
            d2 = anchor._predict_stage2_delta_coef(tr, s_n, priors, anchor.state["s2"])
            r2 = (c_norm - (p1 + d2)).astype(np.float32)
            z_circ = extract_circuit_slow_state_features(
                tr, self.canonical_neurons, s_n, priors["mu_train"], pca_state, kappa=kappa
            )
            train_res_map[tid] = (r2, rval, z_circ)

        return anchor, pca_state, train_res_map

    def fit(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
        anchor_cache: Optional[Mapping[str, Any]] = None,
    ) -> "M3R3Model":
        anchor, pca_state, train_res_map = self._fit_or_reuse_anchor(
            train_trials, s_n, fold_cache=fold_cache, anchor_cache=anchor_cache
        )
        priors = anchor.state["priors"]
        mu_train = priors["mu_train"]
        non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]
        b_dim = int(priors["b_dim"])
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        conn_l2 = float(self.hp.get("connectome_l2_penalty", 60.0))
        wl_rank = int(self.hp.get("wireless_rank", 6))
        wl_mult = float(self.hp.get("wireless_l2_multiplier", 2.5))
        bilin_rank = int(self.hp.get("bilinear_rank", 4))
        d_mode = self._drive_mode()

        self.state = {
            "s_n": np.asarray(s_n, dtype=np.float32).copy(),
            "anchor": anchor,
            "priors": priors,
            "pca_state": pca_state,
            "b_dim": b_dim,
            "drive_mode": d_mode,
            "capacity_info": {},
        }

        dw_rows, dwl_rows, z_rows, u_rows, y_rows, n_rows = [], [], [], [], [], []
        has_wl = False
        for tr in non_buf:
            tid = str(tr.get("trial_id", ""))
            r2, rval, z_circ = train_res_map[tid]
            d_w, d_wl = extract_connectome_drive_blocks(
                tr,
                self.atlas,
                s_n,
                mu_train,
                kappa=kappa,
                mode=d_mode,
                sex_routing_mode=self.sex_routing_mode,
                ablate_layer=self.ablate_layer,
            )
            u_stim = self._extract_u_stim(tr, s_n, priors)
            idx = np.where(rval)[0]
            if idx.size > 0:
                dw_rows.append(d_w[idx])
                if d_wl is not None and d_wl.shape[1] > 0:
                    has_wl = True
                    dwl_rows.append(d_wl[idx])
                z_rows.append(z_circ[idx])
                u_rows.append(u_stim[idx])
                y_rows.append(r2[idx])
                n_rows.append(idx)

        dw_all = np.vstack(dw_rows).astype(np.float32)
        dwl_all = np.vstack(dwl_rows).astype(np.float32) if has_wl else None
        z_all = np.vstack(z_rows).astype(np.float32)
        u_all = np.vstack(u_rows).astype(np.float32)
        y_all = np.vstack(y_rows).astype(np.float32)
        n_all = np.concatenate(n_rows).astype(np.int32)

        if self.model_id in (
            "M3R3_01_h3_wired_unsigned",
            "M3R3_02_h3_wired_signed_cengen",
            "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
        ):
            # Standardize wired block; compress wireless block to wireless_rank via fold-internal SVD
            w_mu = np.mean(dw_all, axis=0, keepdims=True).astype(np.float32) if dw_all.shape[1] > 0 else np.zeros((1, 0), dtype=np.float32)
            w_sd = np.maximum(np.std(dw_all, axis=0, keepdims=True), 1e-4).astype(np.float32) if dw_all.shape[1] > 0 else np.ones((1, 0), dtype=np.float32)
            dw_n = np.clip((dw_all - w_mu) / w_sd, -5.0, 5.0).astype(np.float32) if dw_all.shape[1] > 0 else dw_all

            wl_comp = None
            if dwl_all is not None and dwl_all.shape[1] > 0:
                wl_k = min(wl_rank, dwl_all.shape[1])
                wl_comp = _fit_svd_compressor(dwl_all, n_comp=wl_k)
                dwl_low = _apply_svd_compressor(dwl_all, wl_comp)
                d_lin = np.hstack([dw_n, dwl_low]).astype(np.float32)
                col_l2 = np.concatenate([
                    np.full(dw_n.shape[1], conn_l2, dtype=np.float64),
                    np.full(dwl_low.shape[1], conn_l2 * wl_mult, dtype=np.float64),
                ])
            else:
                d_lin = dw_n
                col_l2 = np.full(dw_n.shape[1], conn_l2, dtype=np.float64)

            # Stage 3a: Strongly regularized direct connectome drive Ridge
            eff_col_l2 = col_l2 * max(1.0, float(d_lin.shape[1]) / 12.0)
            w_lin = _solve_weighted_ridge_sample_scaled(d_lin, y_all, eff_col_l2)
            y_res_lin = (y_all - d_lin @ w_lin).astype(np.float32)

            # Stage 3b: Low-rank stimulus-gated connectome interaction (c_low (x) u_stim, rank = bilinear_rank)
            p_c = min(4, max(1, d_lin.shape[1]))
            c_comp = _fit_svd_compressor(d_lin, n_comp=p_c)
            c_low = _apply_svd_compressor(d_lin, c_comp)
            x_bilin = _build_bilinear_features(c_low, u_all)
            xb_mu = np.mean(x_bilin, axis=0, keepdims=True).astype(np.float32)
            xb_sd = np.maximum(np.std(x_bilin, axis=0, keepdims=True), 1e-4).astype(np.float32)
            xb_n = np.clip((x_bilin - xb_mu) / xb_sd, -5.0, 5.0).astype(np.float32)
            bilin_eff_l2 = conn_l2 * max(1.0, float(xb_n.shape[1]) / 12.0) * 1.5
            rrr = _fit_reduced_rank_ridge(xb_n, y_res_lin, bilin_eff_l2, rank=bilin_rank)

            self.state["h3"] = {
                "w_mu": w_mu,
                "w_sd": w_sd,
                "wl_comp": wl_comp,
                "w_lin": w_lin,
                "c_comp": c_comp,
                "xb_mu": xb_mu,
                "xb_sd": xb_sd,
                "w_rrr": rrr["w"],
            }
            self.state["capacity_info"] = {
                "wired_dim": int(dw_n.shape[1]),
                "wireless_dim": int(wl_comp["v"].shape[1]) if wl_comp is not None else 0,
                "bilin_dim": int(xb_n.shape[1]),
                "param_count": int(w_lin.size + rrr["param_count"]),
                "bilin_rank": int(rrr["rank"]),
            }

        elif self.model_id == "M3R3_04_h3_sex_dimorphic_connectome_film":
            self.state["h3"] = self._fit_connectome_film_residual(
                non_buf_trials=non_buf,
                s_n=s_n,
                dw_all=dw_all,
                dwl_all=dwl_all,
                z_all=z_all,
                u_all=u_all,
                y_all=y_all,
                n_all=n_all,
                b_dim=b_dim,
            )
            net: nn.Module = self.state["h3"]["film_net"]
            self.state["capacity_info"] = {
                "param_count": int(sum(p.numel() for p in net.parameters())),
                "selected_epochs": int(self.state["h3"]["selected_epochs"]),
            }
        else:
            raise ValueError(f"Unknown M3R3 model_id: {self.model_id}")

        return self

    def _fit_connectome_film_residual(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        dw_all: np.ndarray,
        dwl_all: Optional[np.ndarray],
        z_all: np.ndarray,
        u_all: np.ndarray,
        y_all: np.ndarray,
        n_all: np.ndarray,
        b_dim: int,
    ) -> Dict[str, Any]:
        seed = int(self.hp.get("seed", 20261117))
        torch.manual_seed(seed)
        np.random.seed(seed)

        film_dim = int(self.hp.get("poco_film_dim", 24))
        drop = float(self.hp.get("dropout", 0.15))
        lr = float(self.hp.get("learning_rate", 0.003))
        wd = float(self.hp.get("weight_decay", 0.01)) * 2.0
        max_ep = int(self.hp.get("max_epochs", 18))
        patience = int(self.hp.get("early_stopping_patience", 4))
        bsz = int(self.hp.get("batch_size", 2048))
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        wl_rank = int(self.hp.get("wireless_rank", 6))
        p_c = 6
        p_z = 6
        d_mode = self._drive_mode()

        # Inner animal-grouped validation split for zero-leakage early stopping
        worms = sorted({str(t["animal_id"]) for t in non_buf_trials})
        n_val_w = max(1, len(worms) // 5)
        val_worms = set(worms[-n_val_w:])
        inner_tr = [t for t in non_buf_trials if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in non_buf_trials if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(non_buf_trials)
            inner_va = list(non_buf_trials[:1])

        in_anchor, in_pca, in_res_map = self._fit_or_reuse_anchor(inner_tr, s_n)
        in_priors = in_anchor.state["priors"]
        in_mu = in_priors["mu_train"]

        def _collect_inner(trials_sub: Sequence[Mapping[str, Any]], use_res_map: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            cw_l, cwl_l, z_l, u_l, y_l, t_l = [], [], [], [], [], []
            for tr in trials_sub:
                tid = str(tr.get("trial_id", ""))
                if use_res_map:
                    r2, rval, z_circ = in_res_map[tid]
                else:
                    c_norm, rval, _, _ = _project_trial_residual_targets(tr, in_priors, s_n, self.n_canon)
                    p1 = in_anchor._predict_stage1_coef(tr, s_n, in_priors, in_anchor.state["s1"])
                    d2 = in_anchor._predict_stage2_delta_coef(tr, s_n, in_priors, in_anchor.state["s2"])
                    r2 = (c_norm - (p1 + d2)).astype(np.float32)
                    z_circ = extract_circuit_slow_state_features(
                        tr, self.canonical_neurons, s_n, in_mu, in_pca, kappa=kappa
                    )
                d_w, d_wl = extract_connectome_drive_blocks(
                    tr, self.atlas, s_n, in_mu, kappa=kappa, mode=d_mode,
                    sex_routing_mode=self.sex_routing_mode, ablate_layer=self.ablate_layer,
                )
                u_stim = self._extract_u_stim(tr, s_n, in_priors)
                idx = np.where(rval)[0]
                if idx.size > 0:
                    cw_l.append(d_w[idx])
                    cwl_l.append(d_wl[idx] if d_wl is not None else np.zeros((idx.size, 0), dtype=np.float32))
                    z_l.append(z_circ[idx])
                    u_l.append(u_stim[idx])
                    y_l.append(r2[idx])
                    t_l.append(self._tier_oh[idx])
            return (
                np.vstack(cw_l), np.vstack(cwl_l), np.vstack(z_l),
                np.vstack(u_l), np.vstack(y_l), np.vstack(t_l),
            )

        cw_tr, cwl_tr, z_tr, u_tr, y_tr, t_tr = _collect_inner(inner_tr, use_res_map=True)
        cw_va, cwl_va, z_va, u_va, y_va, t_va = _collect_inner(inner_va, use_res_map=False)

        in_wl_comp = _fit_svd_compressor(cwl_tr, n_comp=min(wl_rank, cwl_tr.shape[1])) if cwl_tr.shape[1] > 0 else None
        c_tr_full = np.hstack([cw_tr, _apply_svd_compressor(cwl_tr, in_wl_comp)]) if in_wl_comp is not None else cw_tr
        c_va_full = np.hstack([cw_va, _apply_svd_compressor(cwl_va, in_wl_comp)]) if in_wl_comp is not None else cw_va

        in_c_comp = _fit_svd_compressor(c_tr_full, n_comp=p_c)
        in_z_comp = _fit_svd_compressor(z_tr, n_comp=p_z)
        s_tr = np.hstack([_apply_svd_compressor(c_tr_full, in_c_comp), _apply_svd_compressor(z_tr, in_z_comp)]).astype(np.float32)
        s_va = np.hstack([_apply_svd_compressor(c_va_full, in_c_comp), _apply_svd_compressor(z_va, in_z_comp)]).astype(np.float32)

        probe = ResidualFiLMGate(p_c + p_z, u_tr.shape[1], self._tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

        st_t, ut_t, tt_t, yt_t = torch.from_numpy(s_tr), torch.from_numpy(u_tr), torch.from_numpy(t_tr), torch.from_numpy(y_tr)
        sv_t, uv_t, tv_t, yv_t = torch.from_numpy(s_va), torch.from_numpy(u_va), torch.from_numpy(t_va), torch.from_numpy(y_va)

        best_val = float(torch.mean(yv_t ** 2).item())
        best_ep = 0
        bad = 0
        n_rows = st_t.shape[0]
        for ep in range(1, max_ep + 1):
            probe.train()
            perm = torch.randperm(n_rows)
            for s0 in range(0, n_rows, bsz):
                bi = perm[s0 : s0 + bsz]
                opt.zero_grad()
                loss = torch.mean((probe(st_t[bi], ut_t[bi], tt_t[bi]) - yt_t[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            probe.eval()
            with torch.no_grad():
                vloss = float(torch.mean((probe(sv_t, uv_t, tv_t) - yv_t) ** 2).item())
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        # Outer refit on full training-fold animals for `best_ep` epochs
        outer_wl_comp = (
            _fit_svd_compressor(dwl_all, n_comp=min(wl_rank, dwl_all.shape[1]))
            if dwl_all is not None and dwl_all.shape[1] > 0
            else None
        )
        c_full = np.hstack([dw_all, _apply_svd_compressor(dwl_all, outer_wl_comp)]) if outer_wl_comp is not None else dw_all
        outer_c_comp = _fit_svd_compressor(c_full, n_comp=p_c)
        outer_z_comp = _fit_svd_compressor(z_all, n_comp=p_z)
        s_full = np.hstack([_apply_svd_compressor(c_full, outer_c_comp), _apply_svd_compressor(z_all, outer_z_comp)]).astype(np.float32)
        t_full = self._tier_oh[n_all]

        torch.manual_seed(seed)
        final_net = ResidualFiLMGate(p_c + p_z, u_all.shape[1], self._tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            sf_t, uf_t, tf_t, yf_t = torch.from_numpy(s_full), torch.from_numpy(u_all), torch.from_numpy(t_full), torch.from_numpy(y_all)
            nf = sf_t.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for s0 in range(0, nf, bsz):
                    bi = perm[s0 : s0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(sf_t[bi], uf_t[bi], tf_t[bi]) - yf_t[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()

        return {
            "wl_comp": outer_wl_comp,
            "c_comp": outer_c_comp,
            "z_comp": outer_z_comp,
            "film_net": final_net,
            "selected_epochs": best_ep,
        }

    def predict_trial(
        self,
        trial: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
        z_circ_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        s_n = self.state["s_n"]
        anchor: M3R1Model = self.state["anchor"]
        priors = self.state["priors"]
        mu_train = priors["mu_train"]
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        d_mode = self.state["drive_mode"]

        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
            _, _, prior_delta, phi = fold_cache["proj_targets"][tid]
        else:
            prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, self.n_canon)

        pred_c = anchor._predict_stage1_coef(trial, s_n, priors, anchor.state["s1"], fold_cache=fold_cache)
        pred_c = pred_c + anchor._predict_stage2_delta_coef(
            trial, s_n, priors, anchor.state["s2"], net_trial_override=net_trial_override
        )

        net_src = net_trial_override if net_trial_override is not None else trial
        d_w, d_wl = extract_connectome_drive_blocks(
            net_src,
            self.atlas,
            s_n,
            mu_train,
            kappa=kappa,
            mode=d_mode,
            sex_routing_mode=self.sex_routing_mode,
            ablate_layer=self.ablate_layer,
        )
        u_stim = self._extract_u_stim(trial, s_n, priors)
        h3 = self.state["h3"]

        if self.model_id in (
            "M3R3_01_h3_wired_unsigned",
            "M3R3_02_h3_wired_signed_cengen",
            "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
        ):
            dw_n = np.clip((d_w - h3["w_mu"]) / h3["w_sd"], -5.0, 5.0).astype(np.float32) if d_w.shape[1] > 0 else d_w
            if h3["wl_comp"] is not None and d_wl is not None and d_wl.shape[1] > 0:
                dwl_low = _apply_svd_compressor(d_wl, h3["wl_comp"])
                d_lin = np.hstack([dw_n, dwl_low]).astype(np.float32)
            else:
                d_lin = dw_n
            delta_lin = (d_lin @ h3["w_lin"]).astype(np.float32)
            c_low = _apply_svd_compressor(d_lin, h3["c_comp"])
            xb = _build_bilinear_features(c_low, u_stim)
            xb_n = np.clip((xb - h3["xb_mu"]) / h3["xb_sd"], -5.0, 5.0).astype(np.float32)
            delta_bilin = (xb_n @ h3["w_rrr"]).astype(np.float32)
            pred_c = pred_c + delta_lin + delta_bilin
        else:
            if z_circ_override is not None:
                z_circ = z_circ_override
            else:
                z_circ = extract_circuit_slow_state_features(
                    net_src, self.canonical_neurons, s_n, mu_train, self.state["pca_state"], kappa=kappa
                )
            c_full = np.hstack([d_w, _apply_svd_compressor(d_wl, h3["wl_comp"])]) if (h3["wl_comp"] is not None and d_wl is not None) else d_w
            s_vec = np.hstack([
                _apply_svd_compressor(c_full, h3["c_comp"]),
                _apply_svd_compressor(z_circ, h3["z_comp"]),
            ]).astype(np.float32)
            net: nn.Module = h3["film_net"]
            net.eval()
            with torch.no_grad():
                delta_film = net(
                    torch.from_numpy(s_vec),
                    torch.from_numpy(u_stim.astype(np.float32)),
                    torch.from_numpy(self._tier_oh.astype(np.float32)),
                ).numpy()
            pred_c = pred_c + delta_film.astype(np.float32)

        return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

    def parameter_vector(self) -> np.ndarray:
        parts: List[np.ndarray] = [
            self.state["anchor"].parameter_vector(),
            self.state["pca_state"]["v_rec"].ravel().astype(np.float64),
        ]
        h3 = self.state["h3"]
        for k in ("w_lin", "w_rrr"):
            if k in h3:
                parts.append(h3[k].ravel().astype(np.float64))
        if "film_net" in h3:
            for p in h3["film_net"].parameters():
                parts.append(p.detach().cpu().numpy().ravel().astype(np.float64))
        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 5. Mandatory H3 Mechanistic Diagnostics (PROJECT.md F10)
# ---------------------------------------------------------------------------
def compute_connectome_edge_correlations(atlas: Mapping[str, Any]) -> Dict[str, Any]:
    """Pairwise Spearman & Pearson correlations over union of non-zero allowed edges among 5 layers."""
    mats = atlas["matrices"]
    allow = atlas["allow_mask"]
    layers = {
        "W_chem_pos": mats["W_chem_herm_pos"],
        "W_chem_neg": mats["W_chem_herm_neg"],
        "W_gap": mats["W_gap_herm"],
        "W_npp_wireless": (mats["W_npp_short"] + mats["W_npp_mid"]).astype(np.float32),
        "W_monoamine": mats["W_monoamine"],
    }
    names = list(layers.keys())
    union_nz = np.zeros_like(allow, dtype=bool)
    for m in layers.values():
        union_nz |= (m > 0.0) & allow

    edge_stats = {
        k: {
            "nnz_allowed": int(np.sum((m > 0.0) & allow)),
            "total_weight": round(float(np.sum(m[allow])), 2),
        }
        for k, m in layers.items()
    }
    vecs = {k: m[union_nz].astype(np.float64) for k, m in layers.items()}
    spearman_mat: Dict[str, Dict[str, float]] = {}
    pearson_mat: Dict[str, Dict[str, float]] = {}
    for a in names:
        spearman_mat[a] = {}
        pearson_mat[a] = {}
        for b in names:
            if a == b:
                spearman_mat[a][b] = 1.0
                pearson_mat[a][b] = 1.0
            else:
                r_s = float(spearmanr(vecs[a], vecs[b]).statistic)
                r_p = float(pearsonr(vecs[a], vecs[b]).statistic)
                spearman_mat[a][b] = round(r_s if np.isfinite(r_s) else 0.0, 4)
                pearson_mat[a][b] = round(r_p if np.isfinite(r_p) else 0.0, 4)

    return {
        "union_nonzero_allowed_edges": int(np.sum(union_nz)),
        "layer_edge_counts": edge_stats,
        "spearman": spearman_mat,
        "pearson": pearson_mat,
    }


def compute_trial_drive_correlations(
    val_trials: Sequence[Mapping[str, Any]],
    val_sn_list: Sequence[np.ndarray],
    val_mu_list: Sequence[np.ndarray],
    atlas: Mapping[str, Any],
    kappa: float = 4.0,
) -> Dict[str, Any]:
    """Pairwise Spearman & Pearson correlations over incoming trial-level `[-2, 0)s` drive features."""
    mats = atlas["matrices"]
    rsums = atlas["row_sums"]
    w_npp = (mats["W_npp_short"] + mats["W_npp_mid"]).astype(np.float32)
    r_npp = np.sum(w_npp, axis=1).astype(np.float32)
    layer_defs = [
        ("W_chem_pos", mats["W_chem_herm_pos"], rsums["W_chem_herm_pos"]),
        ("W_chem_neg", mats["W_chem_herm_neg"], rsums["W_chem_herm_neg"]),
        ("W_gap", mats["W_gap_herm"], rsums["W_gap_herm"]),
        ("W_npp_wireless", w_npp, r_npp),
        ("W_monoamine", mats["W_monoamine"], rsums["W_monoamine"]),
    ]
    names = [x[0] for x in layer_defs]
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in names}

    for tr, s_n, mu_tr in zip(val_trials, val_sn_list, val_mu_list):
        if bool(tr.get("is_buffer_control", False)):
            continue
        obs_post = np.any(np.asarray(tr["valid_mask"], dtype=bool), axis=1)
        idx = np.where(obs_post)[0]
        if idx.size == 0:
            continue
        s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_tr)
        for lname, wm, rs in layer_defs:
            dr = compute_layer_drive(wm, rs, s_mat, m_obs, kappa=kappa)
            cols[lname].append(dr[idx, 2])  # [-2, 0)s recent incoming drive

    vecs = {k: np.concatenate(cols[k]).astype(np.float64) for k in names}
    n_obs_rows = int(len(vecs[names[0]])) if names else 0
    spearman_mat: Dict[str, Dict[str, float]] = {}
    pearson_mat: Dict[str, Dict[str, float]] = {}
    for a in names:
        spearman_mat[a] = {}
        pearson_mat[a] = {}
        for b in names:
            if a == b:
                spearman_mat[a][b] = 1.0
                pearson_mat[a][b] = 1.0
            else:
                r_s = float(spearmanr(vecs[a], vecs[b]).statistic)
                r_p = float(pearsonr(vecs[a], vecs[b]).statistic)
                spearman_mat[a][b] = round(r_s if np.isfinite(r_s) else 0.0, 4)
                pearson_mat[a][b] = round(r_p if np.isfinite(r_p) else 0.0, 4)

    return {
        "n_neuron_trial_observations": n_obs_rows,
        "spearman": spearman_mat,
        "pearson": pearson_mat,
    }


def compute_sex_stratified_comparison(
    cand_eval: Mapping[str, Any],
    ref_eval: Mapping[str, Any],
    worm_sex_map: Mapping[str, str],
    n_boot: int = 1000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compare candidate vs reference separately on hermaphrodite and male animals with 95% CIs."""
    cand_lw = cand_eval["L_w"]
    ref_lw = ref_eval["L_w"]
    out: Dict[str, Any] = {}
    for sex_label in ("hermaphrodite", "male", "all"):
        if sex_label == "all":
            worms = sorted(set(cand_lw.keys()) & set(ref_lw.keys()))
        else:
            worms = sorted(w for w in (set(cand_lw.keys()) & set(ref_lw.keys())) if worm_sex_map.get(w) == sex_label)
        if not worms:
            continue
        c_arr = np.array([float(cand_lw[w]) for w in worms], dtype=np.float64)
        r_arr = np.array([float(ref_lw[w]) for w in worms], dtype=np.float64)
        e_c = float(np.mean(c_arr))
        e_r = float(np.mean(r_arr))
        i_d = float(1.0 - e_c / max(e_r, 1e-12))
        rng = np.random.default_rng(seed + (11 if sex_label == "hermaphrodite" else (23 if sex_label == "male" else 37)))
        idx = rng.integers(0, len(worms), size=(n_boot, len(worms)))
        b_i = 1.0 - np.mean(c_arr[idx], axis=1) / np.maximum(np.mean(r_arr[idx], axis=1), 1e-12)
        out[sex_label] = {
            "n_worms": len(worms),
            "E_d_cand": round(e_c, 6),
            "E_d_ref": round(e_r, 6),
            "I_d": round(i_d, 6),
            "I_d_ci95": [round(float(np.percentile(b_i, 2.5)), 6), round(float(np.percentile(b_i, 97.5)), 6)],
            "n_worms_improved": int(np.sum(c_arr < r_arr - 1e-12)),
        }
    return out


def verify_m3_r3_invariance(
    model_id: str = "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
    dataset_id: str = "yemini2021",
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
) -> Dict[str, Any]:
    """Verify future-frame poisoning < 1e-7, OOF tamper < 1e-7, W[i,i]=0 & W[i,bilateral(i)]=0, and asymmetry."""
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    raw_recs = load_dataset(dataset_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
    sub_recs = raw_recs[: min(8, len(raw_recs))]
    aligned, canon, tiers = align_records_to_canonical(sub_recs)
    atlas = load_signed_multilayer_connectome(canon, data_root=data_root)
    splits = build_grouped_cv_splits(aligned)
    train_idx, val_idx = splits[0]

    s_n_clean, _, _ = compute_training_fold_neuron_scales(
        [aligned[i].traces for i in train_idx],
        [aligned[i].obs_mask for i in train_idx],
        floor_percentile=10.0,
    )
    train_trials_clean = [t for i in train_idx for t in extract_causal_trial_records(aligned[i])]
    val_rec_clean = aligned[val_idx[0]]
    val_trials_clean = extract_causal_trial_records(val_rec_clean)

    model_clean = M3R3Model(model_id, cfg, canon, tiers, atlas).fit(train_trials_clean, s_n_clean)
    y_pred_clean = model_clean.predict_trial(val_trials_clean[0])

    # Probe 1: Future-frame poisoning t >= onset_idx
    val_rec_poison = copy.deepcopy(val_rec_clean)
    ev0_onset = int(val_rec_poison.stim_events[0]["onset_idx"])
    val_rec_poison.traces[ev0_onset:, :] = np.where(
        val_rec_poison.obs_mask[ev0_onset:, :],
        val_rec_poison.traces[ev0_onset:, :] * -25.0 + 1.0e6,
        np.nan,
    ).astype(np.float32)
    val_trials_poison = extract_causal_trial_records(val_rec_poison)
    y_pred_poison = model_clean.predict_trial(val_trials_poison[0])
    future_diff = float(np.max(np.abs(y_pred_clean - y_pred_poison)))
    target_changed = float(np.nanmax(np.abs(val_trials_clean[0]["y_true"] - val_trials_poison[0]["y_true"])))

    # Probe 2: OOF validation label/trace tampering
    aligned_tamper = copy.deepcopy(aligned)
    for vi in val_idx:
        aligned_tamper[vi].traces = np.where(
            aligned_tamper[vi].obs_mask,
            aligned_tamper[vi].traces * -50.0 + 9876.5,
            np.nan,
        ).astype(np.float32)
    s_n_tamper, _, _ = compute_training_fold_neuron_scales(
        [aligned_tamper[i].traces for i in train_idx],
        [aligned_tamper[i].obs_mask for i in train_idx],
        floor_percentile=10.0,
    )
    train_trials_tamper = [t for i in train_idx for t in extract_causal_trial_records(aligned_tamper[i])]
    model_tamper = M3R3Model(model_id, cfg, canon, tiers, atlas).fit(train_trials_tamper, s_n_tamper)
    oof_weight_diff = float(np.max(np.abs(model_clean.parameter_vector() - model_tamper.parameter_vector())))
    oof_pred_diff = float(np.max(np.abs(y_pred_clean - model_tamper.predict_trial(val_trials_clean[0]))))

    # Probe 3: Direct matrix check + end-to-end bilateral exclusion (AVAR->AVAL, AIYR->AIYL, ASER->ASEL, AWCOFF->AWCON)
    idx_map = {n: i for i, n in enumerate(canon)}
    max_diag = max(float(np.max(np.abs(np.diag(m)))) for m in atlas["matrices"].values())
    max_bilat_mat = max(
        max(
            float(m[idx_map[a], idx_map[b]]),
            float(m[idx_map[b], idx_map[a]]),
        )
        for m in atlas["matrices"].values()
        for a, b in (("AVAL", "AVAR"), ("AIYL", "AIYR"), ("ASEL", "ASER"), ("AWCON", "AWCOFF"))
    )

    def _pair_exclusion_diff(target_name: str, partner_name: str) -> float:
        ti, pi = idx_map[target_name], idx_map[partner_name]
        r0 = copy.deepcopy(val_rec_clean)
        r0.traces[:ev0_onset, pi] = 0.0
        r0.obs_mask[:ev0_onset, pi] = True
        r1 = copy.deepcopy(val_rec_clean)
        r1.traces[:ev0_onset, pi] = 50.0
        r1.obs_mask[:ev0_onset, pi] = True
        p0 = model_clean.predict_trial(extract_causal_trial_records(r0)[0])
        p1 = model_clean.predict_trial(extract_causal_trial_records(r1)[0])
        return float(np.max(np.abs(p0[ti] - p1[ti])))

    ava_bilat_diff = _pair_exclusion_diff("AVAL", "AVAR")
    aiy_bilat_diff = _pair_exclusion_diff("AIYL", "AIYR")
    ase_bilat_diff = _pair_exclusion_diff("ASEL", "ASER")
    awc_bilat_diff = _pair_exclusion_diff("AWCON", "AWCOFF")

    # Target-self perturbation inside connectome/network pathway (holding self-history fixed)
    base_tr = extract_causal_trial_records(val_rec_clean)[0]
    net_ov = copy.deepcopy(base_tr)
    aval_i = idx_map["AVAL"]
    net_ov["x_pre"] = net_ov["x_pre"].copy()
    net_ov["m_pre"] = net_ov["m_pre"].copy()
    net_ov["x_pre"][:, aval_i] = 50.0
    net_ov["m_pre"][:, aval_i] = True
    yp_base = model_clean.predict_trial(base_tr)
    yp_ov = model_clean.predict_trial(base_tr, net_trial_override=net_ov)
    target_self_net_diff = float(np.max(np.abs(yp_base[aval_i] - yp_ov[aval_i])))

    # Verify ASEL & ASER asymmetry preserved for AIYL via signed connectome drives
    aiyl_i = idx_map["AIYL"]
    asel_i, aser_i = idx_map["ASEL"], idx_map["ASER"]
    t_a = copy.deepcopy(base_tr)
    t_a["x_pre"] = t_a["x_pre"].copy()
    t_a["m_pre"] = t_a["m_pre"].copy()
    t_a["x_pre"][:, [asel_i, aser_i]] = 0.0
    t_a["m_pre"][:, [asel_i, aser_i]] = True
    t_b = copy.deepcopy(t_a)
    t_b["x_pre"] = t_b["x_pre"].copy()
    t_b["x_pre"][-8:, asel_i] = 3.0 * float(s_n_clean[asel_i])
    t_b["x_pre"][-8:, aser_i] = -3.0 * float(s_n_clean[aser_i])
    dw_a, _ = extract_connectome_drive_blocks(t_a, atlas, s_n_clean, model_clean.state["priors"]["mu_train"], mode="wired_signed")
    dw_b, _ = extract_connectome_drive_blocks(t_b, atlas, s_n_clean, model_clean.state["priors"]["mu_train"], mode="wired_signed")
    aiyl_sees_ase_asym = float(np.max(np.abs(dw_b[aiyl_i] - dw_a[aiyl_i])))

    passed = (
        future_diff < 1e-7
        and target_changed > 1e4
        and oof_weight_diff < 1e-7
        and oof_pred_diff < 1e-7
        and max_diag == 0.0
        and max_bilat_mat == 0.0
        and ava_bilat_diff < 1e-7
        and aiy_bilat_diff < 1e-7
        and ase_bilat_diff < 1e-7
        and awc_bilat_diff < 1e-7
        and target_self_net_diff < 1e-7
        and aiyl_sees_ase_asym > 1e-3
    )
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "passed": bool(passed),
        "future_poison_max_abs_diff": future_diff,
        "future_poison_target_shift_verified": target_changed,
        "oof_tamper_weight_diff": oof_weight_diff,
        "oof_tamper_pred_diff": oof_pred_diff,
        "connectome_max_abs_diag": max_diag,
        "connectome_max_bilateral_entry": max_bilat_mat,
        "bilateral_exclusion_AVAR_to_AVAL_diff": ava_bilat_diff,
        "bilateral_exclusion_AIYR_to_AIYL_diff": aiy_bilat_diff,
        "bilateral_exclusion_ASER_to_ASEL_diff": ase_bilat_diff,
        "bilateral_exclusion_AWCOFF_to_AWCON_diff": awc_bilat_diff,
        "target_self_network_pathway_diff": target_self_net_diff,
        "aiyl_sees_asel_aser_connectome_asymmetry_delta": aiyl_sees_ase_asym,
    }


# ---------------------------------------------------------------------------
# 6. Memory-Safe Runner (1 Dataset at a Time + gc.collect() + F10 Diagnostics)
# ---------------------------------------------------------------------------
def run_m3_round3_h3(
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
    n_boot: int = 1000,
    resume: bool = False,
    stage: str = "all",
) -> Dict[str, Any]:
    t_start = time.time()
    cfg_path = exp_root / "configs" / "lambda_global.json"
    prot_path = exp_root / "protocol.json"
    sm_path = exp_root / "split_manifest.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]
    seed = int(hp["seed"])
    kappa = float(hp.get("missingness_precision_kappa", 4.0))

    protocol_sha = compute_file_sha256(prot_path)
    config_sha = compute_file_sha256(cfg_path)
    split_sha = compute_file_sha256(sm_path)
    code_sha = _code_bundle_sha256(exp_root)

    datasets = ["yemini2021", "seyedolmohadesin2025", "toyoshima2024"]
    main_sources = ["seyedolmohadesin2025", "toyoshima2024"]
    runs_dir = exp_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    diag_cache_path = exp_root / "results" / "m3_round3_diag_cache.json"

    model_wall: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R3_MODEL_SPECS}
    model_cpu: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R3_MODEL_SPECS}
    model_reg_ts: Dict[str, str] = {s["model_id"]: _utc_now() for s in M3R3_MODEL_SPECS}
    model_pred_files: Dict[str, Dict[str, str]] = {s["model_id"]: {} for s in M3R3_MODEL_SPECS}
    capacity_audit: Dict[str, Dict[str, Any]] = {s["model_id"]: {} for s in M3R3_MODEL_SPECS}

    # F10 Diagnostic collectors
    edge_corr_summary: Optional[Dict[str, Any]] = None
    drive_corr_by_ds: Dict[str, Any] = {}
    ablation_tags = ["full", "-W_chem_signed", "-W_gap", "-W_wireless_npp", "-W_monoamine", "-all_wired", "-all_wireless"]
    ablation_tag_to_arg = {
        "full": None,
        "-W_chem_signed": "W_chem_signed",
        "-W_gap": "W_gap",
        "-W_wireless_npp": "W_wireless_npp",
        "-W_monoamine": "W_monoamine",
        "-all_wired": "all_wired",
        "-all_wireless": "all_wireless",
    }
    ablation_evals: Dict[str, Dict[str, Any]] = {tag: {} for tag in ablation_tags}
    anchor_evals_inmem: Dict[str, Any] = {}
    sex_dimorphism_evals: Dict[str, Any] = {}
    seyed_worm_sex_map: Dict[str, str] = {}
    connectome_sha: str = ""

    if stage == "score":
        cached = json.loads(diag_cache_path.read_text(encoding="utf-8"))
        model_wall = cached["model_wall"]
        model_cpu = cached["model_cpu"]
        model_reg_ts = cached["model_reg_ts"]
        model_pred_files = cached["model_pred_files"]
        capacity_audit = cached["capacity_audit"]
        edge_corr_summary = cached["edge_corr_summary"]
        drive_corr_by_ds = cached["drive_corr_by_ds"]
        ablation_evals = cached["ablation_evals"]
        anchor_evals_inmem = cached["anchor_evals_inmem"]
        sex_dimorphism_evals = cached["sex_dimorphism_evals"]
        seyed_worm_sex_map = cached["seyed_worm_sex_map"]
        connectome_sha = cached["connectome_sha"]
        print(f"[M3R3] Loaded diagnostics cache from {diag_cache_path}", flush=True)

    # Step 1: Process ONE dataset at a time + gc.collect() (Peak RSS < 3.0 GiB)
    for ds_id in ([] if stage == "score" else datasets):
        t_ds0 = time.time()
        print(f"[M3R3] Loading Dev split: {ds_id} (RSS={_peak_rss_mb():.1f} MB)", flush=True)
        raw_recs = load_dataset(ds_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
        aligned, canon, tiers = align_records_to_canonical(raw_recs)
        if ds_id == "seyedolmohadesin2025":
            for rec in aligned:
                seyed_worm_sex_map[rec.physical_animal_id] = str(rec.sex).lower()
        splits = build_grouped_cv_splits(aligned)
        del raw_recs

        atlas = load_signed_multilayer_connectome(canon, data_root=data_root)
        connectome_sha = atlas["bundle_sha256"]
        if edge_corr_summary is None:
            edge_corr_summary = compute_connectome_edge_correlations(atlas)

        fold_bundles: List[Dict[str, Any]] = []
        train_fold_session_ids: List[List[str]] = []
        for f_idx, (tr_idx, va_idx) in enumerate(splits):
            tr_phys = {aligned[i].physical_animal_id for i in tr_idx}
            va_phys = {aligned[i].physical_animal_id for i in va_idx}
            if tr_phys & va_phys:
                raise RuntimeError(f"Physical animal leakage in {ds_id} fold {f_idx}: {tr_phys & va_phys}")

            lo_b, hi_b = compute_training_fold_sensor_bounds(
                [aligned[i].traces for i in tr_idx],
                [aligned[i].obs_mask for i in tr_idx],
            )
            tr_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in tr_idx]
            va_bounded = [apply_sensor_bounds_to_aligned(aligned[i], lo_b, hi_b) for i in va_idx]

            s_n, _, _ = compute_training_fold_neuron_scales(
                [r.traces for r in tr_bounded],
                [r.obs_mask for r in tr_bounded],
                floor_percentile=10.0,
            )
            tr_trials = [
                t
                for r in tr_bounded
                for t in extract_causal_trial_records(
                    r,
                    input_sec=float(hp["input_window_sec"]),
                    target_sec=float(hp["target_window_sec"]),
                    calibration_sec=float(hp["calibration_window_sec"]),
                    fast_anchor_sec=float(hp["fast_anchor_sec"]),
                    include_synthetic_buffer_windows=True,
                )
            ]
            va_trials = [
                t
                for r in va_bounded
                for t in extract_causal_trial_records(
                    r,
                    input_sec=float(hp["input_window_sec"]),
                    target_sec=float(hp["target_window_sec"]),
                    calibration_sec=float(hp["calibration_window_sec"]),
                    fast_anchor_sec=float(hp["fast_anchor_sec"]),
                    include_synthetic_buffer_windows=True,
                )
            ]
            f_cache = _build_m3_fold_cache(tr_trials + va_trials, tr_trials, s_n, len(canon), cfg)

            helper = M3R3Model("M3R3_01_h3_wired_unsigned", cfg, canon, tiers, atlas)
            anchor, pca_state, train_res_map = helper._fit_or_reuse_anchor(tr_trials, s_n, fold_cache=f_cache)
            val_z_circ = [
                extract_circuit_slow_state_features(
                    vtr, canon, s_n, anchor.state["priors"]["mu_train"], pca_state, kappa=kappa
                )
                for vtr in va_trials
            ]
            val_anchor_preds = [anchor.predict_trial(vtr, fold_cache=f_cache) for vtr in va_trials]

            fold_bundles.append(
                {
                    "fold_idx": f_idx,
                    "train_trials": tr_trials,
                    "val_trials": va_trials,
                    "s_n": s_n,
                    "mu_train": anchor.state["priors"]["mu_train"],
                    "cache": f_cache,
                    "anchor_cache": {
                        "anchor": anchor,
                        "pca_state": pca_state,
                        "train_res_map": train_res_map,
                    },
                    "val_z_circ": val_z_circ,
                    "val_anchor_preds": val_anchor_preds,
                }
            )
            train_fold_session_ids.append([aligned[i].animal_id for i in tr_idx])

        del aligned

        all_val_trials: List[Dict[str, Any]] = []
        all_val_sn: List[np.ndarray] = []
        all_val_mu: List[np.ndarray] = []
        all_anchor_preds: List[np.ndarray] = []
        for fb in fold_bundles:
            for vtr, ap in zip(fb["val_trials"], fb["val_anchor_preds"]):
                all_val_trials.append(vtr)
                all_val_sn.append(fb["s_n"])
                all_val_mu.append(fb["mu_train"])
                all_anchor_preds.append(ap)

        anchor_evals_inmem[ds_id] = evaluate_predictions(
            all_val_trials, all_anchor_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
        )
        drive_corr_by_ds[ds_id] = compute_trial_drive_correlations(
            all_val_trials, all_val_sn, all_val_mu, atlas, kappa=kappa
        )

        # Train & evaluate all 4 primary Round 3 models on this dataset
        for spec in M3R3_MODEL_SPECS:
            mid = spec["model_id"]
            rid = spec["run_id"]
            rdir = runs_dir / rid
            rdir.mkdir(parents=True, exist_ok=True)
            pred_file = rdir / f"predictions_dev_{ds_id}.npz"

            if resume and mid == "M3R3_04_h3_sex_dimorphic_connectome_film" and pred_file.exists():
                model_pred_files[mid][ds_id] = str(pred_file.relative_to(exp_root))
                print(f"[M3R3]   Resumed existing {mid} on {ds_id} (saved on disk)", flush=True)
                continue

            t_m0 = time.time()
            c_m0 = time.process_time()
            oof_trials: List[Dict[str, Any]] = []
            oof_preds: List[np.ndarray] = []
            oof_fold_ids: List[int] = []
            fold_caps: List[Dict[str, Any]] = []

            for fb in fold_bundles:
                model = M3R3Model(mid, cfg, canon, tiers, atlas, sex_routing_mode="sex_matched")
                model.fit(fb["train_trials"], fb["s_n"], fold_cache=fb["cache"], anchor_cache=fb["anchor_cache"])
                fold_caps.append(model.state.get("capacity_info", {}))
                for vtr, z_c in zip(fb["val_trials"], fb["val_z_circ"]):
                    yp = model.predict_trial(vtr, fold_cache=fb["cache"], z_circ_override=z_c)
                    oof_trials.append(vtr)
                    oof_preds.append(yp)
                    oof_fold_ids.append(int(fb["fold_idx"]))

            save_CompactPredictions(
                out_path=pred_file,
                trials=oof_trials,
                y_preds=oof_preds,
                train_fold_animal_ids=train_fold_session_ids,
                fold_ids=np.array(oof_fold_ids, dtype=np.int32),
            )
            dt_w = time.time() - t_m0
            dt_c = time.process_time() - c_m0
            model_wall[mid] += dt_w
            model_cpu[mid] += dt_c
            model_pred_files[mid][ds_id] = str(pred_file.relative_to(exp_root))

            cap_summary: Dict[str, Any] = {}
            if fold_caps:
                for k in ("wired_dim", "wireless_dim", "bilin_dim", "param_count", "bilin_rank", "selected_epochs"):
                    vals = [fc[k] for fc in fold_caps if k in fc]
                    if vals:
                        cap_summary[k] = round(float(np.mean(vals)), 3)
            capacity_audit[mid][ds_id] = cap_summary
            print(f"[M3R3]   {mid} on {ds_id}: {dt_w:.2f}s | cap={cap_summary} (RSS={_peak_rss_mb():.1f} MB)", flush=True)

            if mid == "M3R3_03_h3_multilayer_wired_plus_wireless_npp":
                ablation_evals["full"][ds_id] = evaluate_predictions(
                    oof_trials, oof_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                )
            if mid == "M3R3_04_h3_sex_dimorphic_connectome_film" and ds_id == "seyedolmohadesin2025":
                sex_dimorphism_evals["sex_matched"] = evaluate_predictions(
                    oof_trials, oof_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                )

        # F10 Diagnostic A: Leave-One-Layer-Out Ablation on M3R3_03 (fast closed-form Ridge!)
        t_ab0 = time.time()
        for tag in ablation_tags:
            if tag == "full":
                continue
            ab_arg = ablation_tag_to_arg[tag]
            ab_preds: List[np.ndarray] = []
            for fb in fold_bundles:
                m_ab = M3R3Model(
                    "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
                    cfg, canon, tiers, atlas, ablate_layer=ab_arg,
                )
                m_ab.fit(fb["train_trials"], fb["s_n"], fold_cache=fb["cache"], anchor_cache=fb["anchor_cache"])
                for vtr in fb["val_trials"]:
                    ab_preds.append(m_ab.predict_trial(vtr, fold_cache=fb["cache"]))
            ablation_evals[tag][ds_id] = evaluate_predictions(
                all_val_trials, ab_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
            )
        print(f"[M3R3]   Leave-one-layer-out ablations on {ds_id}: {time.time() - t_ab0:.2f}s", flush=True)

        # F10 Diagnostic B: Sex Dimorphism Controls (`herm_only`, `sex_swapped`) on seyedolmohadesin2025
        if ds_id == "seyedolmohadesin2025":
            t_sx0 = time.time()
            for s_mode in ("herm_only", "sex_swapped"):
                sx_preds: List[np.ndarray] = []
                for fb in fold_bundles:
                    m_sx = M3R3Model(
                        "M3R3_04_h3_sex_dimorphic_connectome_film",
                        cfg, canon, tiers, atlas, sex_routing_mode=s_mode,
                    )
                    m_sx.fit(fb["train_trials"], fb["s_n"], fold_cache=fb["cache"], anchor_cache=fb["anchor_cache"])
                    for vtr, z_c in zip(fb["val_trials"], fb["val_z_circ"]):
                        sx_preds.append(m_sx.predict_trial(vtr, fold_cache=fb["cache"], z_circ_override=z_c))
                sex_dimorphism_evals[s_mode] = evaluate_predictions(
                    all_val_trials, sx_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                )
            print(f"[M3R3]   Sex-dimorphism controls (herm_only, sex_swapped) on {ds_id}: {time.time() - t_sx0:.2f}s", flush=True)

        del fold_bundles, train_fold_session_ids, all_val_trials, all_val_sn, all_val_mu, all_anchor_preds, atlas
        gc.collect()
        print(f"[M3R3] Finished {ds_id} in {time.time() - t_ds0:.1f}s (post-GC RSS={_peak_rss_mb():.1f} MB)", flush=True)

    if stage != "score":
        def _slim_eval(ev: Mapping[str, Any]) -> Dict[str, Any]:
            return {
                "E_d": float(ev["E_d"]),
                "L_w": {str(k): float(v) for k, v in ev["L_w"].items()},
            }

        diag_cache_path.parent.mkdir(parents=True, exist_ok=True)
        diag_cache_path.write_text(
            json.dumps(
                {
                    "model_wall": model_wall,
                    "model_cpu": model_cpu,
                    "model_reg_ts": model_reg_ts,
                    "model_pred_files": model_pred_files,
                    "capacity_audit": capacity_audit,
                    "edge_corr_summary": edge_corr_summary,
                    "drive_corr_by_ds": drive_corr_by_ds,
                    "ablation_evals": {
                        tag: {d: _slim_eval(ev) for d, ev in dmap.items()}
                        for tag, dmap in ablation_evals.items()
                    },
                    "anchor_evals_inmem": {d: _slim_eval(ev) for d, ev in anchor_evals_inmem.items()},
                    "sex_dimorphism_evals": {k: _slim_eval(ev) for k, ev in sex_dimorphism_evals.items()},
                    "seyed_worm_sex_map": seyed_worm_sex_map,
                    "connectome_sha": connectome_sha,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[M3R3] Saved diagnostics cache to {diag_cache_path}", flush=True)
        if stage == "diagnostics":
            return {"status": "diagnostics_done", "diag_cache": str(diag_cache_path)}

    # Step 2: Score EVERY run with standalone `evaluators/scorer.py` (1 dataset at a time!)
    print("[M3R3] Scoring all 4 Round 3 runs + B4, M3R1_02 & M3R2_04 anchors via evaluators/scorer.py...", flush=True)
    scored_by_model: Dict[str, Dict[str, Any]] = {s["model_id"]: {} for s in M3R3_MODEL_SPECS}
    ref_b4_scored: Dict[str, Any] = {}
    ref_m02_scored: Dict[str, Any] = {}
    ref_r204_scored: Dict[str, Any] = {}

    for ds_id in datasets:
        t_s0 = time.time()
        b4_path = exp_root / "runs" / "r0_04_b4_ridge_self_stim" / f"predictions_dev_{ds_id}.npz"
        m02_path = exp_root / "runs" / "r1_02_m3r1_02_h1_confounds_only" / f"predictions_dev_{ds_id}.npz"
        r204_path = exp_root / "runs" / "r2_04_m3r2_04_h2_film_gating_residual" / f"predictions_dev_{ds_id}.npz"
        ref_b4_scored[ds_id] = score_saved_predictions_file(
            pred_path=b4_path, split="dev", ref_pred_path=None,
            data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
        )
        ref_m02_scored[ds_id] = score_saved_predictions_file(
            pred_path=m02_path, split="dev", ref_pred_path=None,
            data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
        )
        ref_r204_scored[ds_id] = score_saved_predictions_file(
            pred_path=r204_path, split="dev", ref_pred_path=None,
            data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
        )
        for spec in M3R3_MODEL_SPECS:
            mid = spec["model_id"]
            p_path = exp_root / model_pred_files[mid][ds_id]
            scored_by_model[mid][ds_id] = score_saved_predictions_file(
                pred_path=p_path, split="dev", ref_pred_path=None,
                data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
            )
        gc.collect()
        print(f"[M3R3]   Scored {ds_id} in {time.time() - t_s0:.1f}s (RSS={_peak_rss_mb():.1f} MB)", flush=True)

    # Step 3: Pairwise Comparisons (vs B4, vs M3R1_02, vs M3R2_04) with 1000-resample animal bootstrap 95% CIs
    summary_models: Dict[str, Any] = {}
    run_jsonl_entries: List[Dict[str, Any]] = []

    for spec in M3R3_MODEL_SPECS:
        mid = spec["model_id"]
        rid = spec["run_id"]
        ds_out: Dict[str, Any] = {}
        for ds_id in datasets:
            sc = scored_by_model[mid][ds_id]
            cmp_b4 = compare_against_reference(sc, ref_b4_scored[ds_id], n_boot=n_boot, seed=seed)
            cmp_m02 = compare_against_reference(sc, ref_m02_scored[ds_id], n_boot=n_boot, seed=seed)
            cmp_r204 = compare_against_reference(sc, ref_r204_scored[ds_id], n_boot=n_boot, seed=seed)
            ds_out[ds_id] = {
                "E_d": round(float(sc["E_d"]), 6),
                "rmse_raw": round(float(sc["rmse_raw"]), 6),
                "post_R2": round(float(sc["post_R2"]), 6),
                "post_R2_ci95": [round(float(x), 6) for x in sc["post_R2_ci95"]],
                "delta_R2": round(float(sc["delta_R2"]), 6),
                "delta_R2_ci95": [round(float(x), 6) for x in sc["delta_R2_ci95"]],
                "buffer_E_d": round(float(sc["buffer_E_d"]), 6) if sc.get("buffer_E_d") is not None else None,
                "n_worms_total": int(sc["n_worms_total"]),
                "vs_B4_ridge_self_stim": {
                    "I_d": round(float(cmp_b4["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_b4["I_d_ci95"]],
                    "n_worms_improved": int(cmp_b4["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_b4["fraction_worms_improved"]), 4),
                },
                "vs_M3R1_02_h1_confounds_only": {
                    "I_d": round(float(cmp_m02["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_m02["I_d_ci95"]],
                    "n_worms_improved": int(cmp_m02["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_m02["fraction_worms_improved"]), 4),
                },
                "vs_M3R2_04_h2_film_gating_residual": {
                    "I_d": round(float(cmp_r204["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_r204["I_d_ci95"]],
                    "n_worms_improved": int(cmp_r204["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_r204["fraction_worms_improved"]), 4),
                },
                "temporal_horizon_errors": {
                    k: round(float(v), 6) if v is not None else None
                    for k, v in sc.get("temporal_horizon_errors", {}).items()
                },
                "cell_class_errors": {
                    k: round(float(v), 6) for k, v in sc.get("cell_class_errors", {}).items()
                },
                "sex_errors": {
                    k: round(float(v), 6) for k, v in sc.get("sex_errors", {}).items()
                },
                "stimulus_errors": {
                    k: round(float(v), 6) for k, v in sc.get("stimulus_errors", {}).items()
                },
            }

        macro_b4_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_b4_scored, main_sources, n_boot=n_boot, seed=seed)
        macro_b4_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_b4_scored, datasets, n_boot=n_boot, seed=seed)
        macro_m02_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_m02_scored, main_sources, n_boot=n_boot, seed=seed)
        macro_m02_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_m02_scored, datasets, n_boot=n_boot, seed=seed)
        macro_r204_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_r204_scored, main_sources, n_boot=n_boot, seed=seed)
        macro_r204_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], ref_r204_scored, datasets, n_boot=n_boot, seed=seed)
        worst_main_b4 = min(ds_out[d]["vs_B4_ridge_self_stim"]["I_d"] for d in main_sources)

        summary_models[mid] = {
            "run_id": rid,
            "parent_id": spec["parent_id"],
            "is_self_only": spec["is_self_only"],
            "uses_confounds": spec["uses_confounds"],
            "uses_connectome": spec["uses_connectome"],
            "uses_polarity_sign": spec["uses_polarity_sign"],
            "uses_wireless": spec["uses_wireless"],
            "uses_sex_connectome": spec["uses_sex_connectome"],
            "is_stochastic": spec["is_stochastic"],
            "hypothesis": spec["hypothesis"],
            "capacity_audit": capacity_audit[mid],
            "I_macro_main2_vs_B4": macro_b4_main2["I_macro"],
            "I_macro_main2_vs_B4_ci95": macro_b4_main2["I_macro_ci95"],
            "I_macro_all3_vs_B4": macro_b4_all3["I_macro"],
            "I_macro_all3_vs_B4_ci95": macro_b4_all3["I_macro_ci95"],
            "I_macro_main2_vs_M3R1_02": macro_m02_main2["I_macro"],
            "I_macro_main2_vs_M3R1_02_ci95": macro_m02_main2["I_macro_ci95"],
            "I_macro_all3_vs_M3R1_02": macro_m02_all3["I_macro"],
            "I_macro_all3_vs_M3R1_02_ci95": macro_m02_all3["I_macro_ci95"],
            "I_macro_main2_vs_M3R2_04": macro_r204_main2["I_macro"],
            "I_macro_main2_vs_M3R2_04_ci95": macro_r204_main2["I_macro_ci95"],
            "I_macro_all3_vs_M3R2_04": macro_r204_all3["I_macro"],
            "I_macro_all3_vs_M3R2_04_ci95": macro_r204_all3["I_macro_ci95"],
            "worst_main_source_I_d_vs_B4": round(float(worst_main_b4), 6),
            "prediction_files_dev": model_pred_files[mid],
            "wall_clock_sec": round(model_wall[mid], 3),
            "cpu_time_sec": round(model_cpu[mid], 3),
            "datasets": ds_out,
        }
        (runs_dir / rid / "metrics_dev.json").write_text(json.dumps(summary_models[mid], indent=2), encoding="utf-8")

        src_metrics_jsonl = {}
        for ds_id in datasets:
            d_m = ds_out[ds_id]
            src_metrics_jsonl[ds_id] = {
                "E_d": d_m["E_d"],
                "I_d": d_m["vs_B4_ridge_self_stim"]["I_d"],
                "I_d_ci95": d_m["vs_B4_ridge_self_stim"]["I_d_ci95"],
                "post_R2": d_m["post_R2"],
                "post_R2_ci95": d_m["post_R2_ci95"],
                "delta_R2": d_m["delta_R2"],
                "delta_R2_ci95": d_m["delta_R2_ci95"],
                "rmse_raw": d_m["rmse_raw"],
                "n_worms_improved": d_m["vs_B4_ridge_self_stim"]["n_worms_improved"],
                "n_worms_total": d_m["n_worms_total"],
            }

        run_jsonl_entries.append(
            {
                "run_id": rid,
                "parent_id": spec["parent_id"],
                "round": 3,
                "category": "M3_round3_H3",
                "model_id": mid,
                "family_id": "M3R3_H3",
                "is_self_only": spec["is_self_only"],
                "hypothesis": spec["hypothesis"],
                "single_factor_changed": spec["single_factor_changed"],
                "registered_before_training_utc": model_reg_ts[mid],
                "completed_at_utc": _utc_now(),
                "config_id": cfg["config_id"],
                "seed": seed,
                "checksums": {
                    "protocol_sha256": protocol_sha,
                    "lambda_global_sha256": config_sha,
                    "split_manifest_sha256": split_sha,
                    "connectome_bundle_sha256": connectome_sha,
                    "code_bundle_sha256": code_sha,
                },
                "reference_baseline_id": "B4_ridge_self_stim",
                "I_macro": macro_b4_main2["I_macro"],
                "I_macro_ci95": macro_b4_main2["I_macro_ci95"],
                "I_macro_all3_immobilized": macro_b4_all3["I_macro"],
                "I_macro_all3_ci95": macro_b4_all3["I_macro_ci95"],
                "I_macro_vs_M3R1_02": macro_m02_main2["I_macro"],
                "I_macro_vs_M3R2_04": macro_r204_main2["I_macro"],
                "worst_main_source_I_d": round(float(worst_main_b4), 6),
                "source_metrics": src_metrics_jsonl,
                "prediction_files_dev": model_pred_files[mid],
                "wall_clock_sec": round(model_wall[mid], 3),
                "cpu_time_sec": round(model_cpu[mid], 3),
                "peak_rss_mb": _peak_rss_mb(),
                "exit_code": 0,
                "status": "completed",
                "decision": "RETAINED_M3_R3_DIAGNOSTIC",
            }
        )

    # Step 4: Compile F10 Leave-One-Layer-Out Ablation & Unique/Shared Variance Partition
    ablation_summary: Dict[str, Any] = {}
    for tag in ablation_tags:
        per_ds: Dict[str, Any] = {}
        for ds_id in datasets:
            cmp_anc = compare_against_reference(ablation_evals[tag][ds_id], anchor_evals_inmem[ds_id], n_boot=n_boot, seed=seed)
            cmp_full = compare_against_reference(ablation_evals[tag][ds_id], ablation_evals["full"][ds_id], n_boot=n_boot, seed=seed)
            per_ds[ds_id] = {
                "E_d": round(float(ablation_evals[tag][ds_id]["E_d"]), 6),
                "I_d_vs_M3R1_02": round(float(cmp_anc["I_d"]), 6),
                "I_d_vs_M3R1_02_ci95": [round(float(x), 6) for x in cmp_anc["I_d_ci95"]],
                "I_d_vs_full_M3R3_03": round(float(cmp_full["I_d"]), 6),
                "I_d_vs_full_M3R3_03_ci95": [round(float(x), 6) for x in cmp_full["I_d_ci95"]],
            }
        m2_anc = compute_macro_bootstrap_ci(ablation_evals[tag], anchor_evals_inmem, main_sources, n_boot=n_boot, seed=seed)
        a3_anc = compute_macro_bootstrap_ci(ablation_evals[tag], anchor_evals_inmem, datasets, n_boot=n_boot, seed=seed)
        m2_full = compute_macro_bootstrap_ci(ablation_evals[tag], ablation_evals["full"], main_sources, n_boot=n_boot, seed=seed)
        a3_full = compute_macro_bootstrap_ci(ablation_evals[tag], ablation_evals["full"], datasets, n_boot=n_boot, seed=seed)
        ablation_summary[tag] = {
            "I_macro_main2_vs_M3R1_02": m2_anc["I_macro"],
            "I_macro_main2_vs_M3R1_02_ci95": m2_anc["I_macro_ci95"],
            "I_macro_all3_vs_M3R1_02": a3_anc["I_macro"],
            "I_macro_all3_vs_M3R1_02_ci95": a3_anc["I_macro_ci95"],
            "delta_vs_full_main2": m2_full["I_macro"],
            "delta_vs_full_main2_ci95": m2_full["I_macro_ci95"],
            "delta_vs_full_all3": a3_full["I_macro"],
            "delta_vs_full_all3_ci95": a3_full["I_macro_ci95"],
            "datasets": per_ds,
        }

    # Unique vs Shared decomposition (Wired vs Wireless) relative to M3R1_02 anchor
    def _partition(metric_key: str) -> Dict[str, float]:
        g_full = float(ablation_summary["full"][metric_key])
        g_wired_only = float(ablation_summary["-all_wireless"][metric_key])
        g_wireless_only = float(ablation_summary["-all_wired"][metric_key])
        unique_wired = g_full - g_wireless_only
        unique_wireless = g_full - g_wired_only
        shared = g_wired_only + g_wireless_only - g_full
        return {
            "gain_full_4layer": round(g_full, 6),
            "gain_wired_only": round(g_wired_only, 6),
            "gain_wireless_only": round(g_wireless_only, 6),
            "unique_wired_contribution": round(unique_wired, 6),
            "unique_wireless_contribution": round(unique_wireless, 6),
            "shared_wired_wireless_contribution": round(shared, 6),
            "unique_W_chem_signed": round(g_full - float(ablation_summary["-W_chem_signed"][metric_key]), 6),
            "unique_W_gap": round(g_full - float(ablation_summary["-W_gap"][metric_key]), 6),
            "unique_W_wireless_npp": round(g_full - float(ablation_summary["-W_wireless_npp"][metric_key]), 6),
            "unique_W_monoamine": round(g_full - float(ablation_summary["-W_monoamine"][metric_key]), 6),
        }

    unique_shared_partition = {
        "macro_main2": _partition("I_macro_main2_vs_M3R1_02"),
        "macro_all3": _partition("I_macro_all3_vs_M3R1_02"),
    }

    # Step 5: Compile F10 Sex Dimorphism Analysis on seyedolmohadesin2025
    if "sex_matched" not in sex_dimorphism_evals:
        sex_dimorphism_evals["sex_matched"] = scored_by_model["M3R3_04_h3_sex_dimorphic_connectome_film"]["seyedolmohadesin2025"]
    sex_dimorphism_report = {
        "dataset_id": "seyedolmohadesin2025",
        "n_hermaphrodite": int(sum(1 for v in seyed_worm_sex_map.values() if v == "hermaphrodite")),
        "n_male": int(sum(1 for v in seyed_worm_sex_map.values() if v == "male")),
        "sex_matched_vs_B4": compute_sex_stratified_comparison(
            sex_dimorphism_evals["sex_matched"], ref_b4_scored["seyedolmohadesin2025"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
        "sex_matched_vs_M3R1_02": compute_sex_stratified_comparison(
            sex_dimorphism_evals["sex_matched"], ref_m02_scored["seyedolmohadesin2025"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
        "sex_matched_vs_herm_only": compute_sex_stratified_comparison(
            sex_dimorphism_evals["sex_matched"], sex_dimorphism_evals["herm_only"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
        "sex_matched_vs_sex_swapped": compute_sex_stratified_comparison(
            sex_dimorphism_evals["sex_matched"], sex_dimorphism_evals["sex_swapped"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
        "herm_only_vs_M3R1_02": compute_sex_stratified_comparison(
            sex_dimorphism_evals["herm_only"], ref_m02_scored["seyedolmohadesin2025"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
        "sex_swapped_vs_M3R1_02": compute_sex_stratified_comparison(
            sex_dimorphism_evals["sex_swapped"], ref_m02_scored["seyedolmohadesin2025"], seyed_worm_sex_map, n_boot=n_boot, seed=seed
        ),
    }

    # Step 6: Explicit Leakage Invariance Verification on M3R3_03_h3_multilayer_wired_plus_wireless_npp
    print("[M3R3] Running leakage invariance check on M3R3_03_h3_multilayer_wired_plus_wireless_npp...", flush=True)
    invariance_res = verify_m3_r3_invariance(
        model_id="M3R3_03_h3_multilayer_wired_plus_wireless_npp",
        dataset_id="yemini2021",
        data_root=data_root,
        exp_root=exp_root,
    )
    if not invariance_res["passed"]:
        raise RuntimeError(f"Leakage invariance FAILED on M3R3_03: {invariance_res}")
    print(
        f"[M3R3] Invariance PASSED: future_diff={invariance_res['future_poison_max_abs_diff']:.2e}, "
        f"oof_p_diff={invariance_res['oof_tamper_pred_diff']:.2e}, "
        f"bilat_AIY={invariance_res['bilateral_exclusion_AIYR_to_AIYL_diff']:.2e}, "
        f"bilat_ASE={invariance_res['bilateral_exclusion_ASER_to_ASEL_diff']:.2e}, "
        f"bilat_AWC={invariance_res['bilateral_exclusion_AWCOFF_to_AWCON_diff']:.2e}",
        flush=True,
    )

    # Step 7: Append 4 rows to runs.jsonl (idempotent by run_id -> total 27 rows)
    runs_jsonl_path = exp_root / "runs.jsonl"
    existing_lines: List[str] = []
    new_rids = {e["run_id"] for e in run_jsonl_entries}
    if runs_jsonl_path.exists():
        for line in runs_jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("run_id") in new_rids:
                    continue
            except Exception:
                pass
            existing_lines.append(line)
    for entry in run_jsonl_entries:
        existing_lines.append(json.dumps(entry))
    runs_jsonl_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")

    # Step 8: Save results/m3_round3_h3_summary.json
    summary_payload = {
        "schema_version": "1.0.0",
        "round": 3,
        "milestone": "M3_round3_H3",
        "created_at_utc": _utc_now(),
        "config_id": cfg["config_id"],
        "lambda_global_path": "configs/lambda_global.json",
        "lambda_global_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "split_manifest_sha256": split_sha,
        "connectome_bundle_sha256": connectome_sha,
        "code_bundle_sha256": code_sha,
        "n_bootstrap_resamples": n_boot,
        "reference_baseline_id": "B4_ridge_self_stim",
        "anchor_model_id": "M3R1_02_h1_confounds_only",
        "round2_best_model_id": "M3R2_04_h2_film_gating_residual",
        "reference_baseline_E_d": {d: round(float(ref_b4_scored[d]["E_d"]), 6) for d in datasets},
        "anchor_M3R1_02_E_d": {d: round(float(ref_m02_scored[d]["E_d"]), 6) for d in datasets},
        "round2_best_M3R2_04_E_d": {d: round(float(ref_r204_scored[d]["E_d"]), 6) for d in datasets},
        "invariance_verification": invariance_res,
        "connectome_interlayer_correlations": {
            "edge_level": edge_corr_summary,
            "trial_drive_level_by_dataset": drive_corr_by_ds,
        },
        "leave_one_layer_out_ablation": ablation_summary,
        "unique_shared_variance_partition": unique_shared_partition,
        "sex_dimorphism_analysis": sex_dimorphism_report,
        "models": summary_models,
        "total_wall_clock_sec": round(time.time() - t_start, 3),
        "peak_rss_mb": _peak_rss_mb(),
        "runs_jsonl_total_rows": len(existing_lines),
    }
    out_summary = exp_root / "results" / "m3_round3_h3_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(
        f"[M3R3] Complete in {summary_payload['total_wall_clock_sec']:.1f}s "
        f"(Peak RSS={summary_payload['peak_rss_mb']:.1f} MB, runs.jsonl={len(existing_lines)} rows). "
        f"Saved {out_summary}",
        flush=True,
    )
    return summary_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run M3 Round 3 (H3 signed connectome priors & sex dimorphism)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing saved M3R3_04 predictions_dev_*.npz if present")
    parser.add_argument("--stage", choices=["all", "diagnostics", "score"], default="all", help="Run full pipeline, diagnostics only, or scoring only")
    args = parser.parse_args()
    run_m3_round3_h3(resume=bool(args.resume), stage=str(args.stage))
