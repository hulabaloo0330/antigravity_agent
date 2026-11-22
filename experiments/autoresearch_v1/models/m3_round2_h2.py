"""Milestone M3 — Round 2: H2 Low-Rank Bilinear State x Stimulus Gating & Layer Dissociation.

Implements and genuinely trains/evaluates 4 models anchored on `M3R1_02_h1_confounds_only`
across all 3 immobilized Dev splits (`yemini2021`, `seyedolmohadesin2025`, `toyoshima2024`)
under unified `configs/lambda_global.json` (zero `dataset_id`-specific hyperparameter branches):

1. `M3R2_01_h2_bilinear_uniform`:
   Uniform low-rank bilinear `State x Stimulus` gating predicting the `M3R1_02` residual via
   fold-internal SVD compression of missingness-precision-weighted (`kappa`) excluded-self/bilateral
   circuit slow-state `z_state` interacted with stimulus drive `u_stim` (`z_low (x) u_stim`),
   solved via closed-form Reduced-Rank Ridge (`rank = bilinear_rank = 4`, `l2 = bilinear_l2_penalty`).
2. `M3R2_02_h2_bilinear_hierarchical`:
   Hierarchical circuit-layer bilinear gating differentiating biological target tiers:
     - Primary sensory neurons receive only strongly regularized asymmetric sensory context
       (`ASEL/ASER/AWCON/AWCOFF`) x `u_stim` (insulated from downstream command-state variance),
     - Interneurons receive full `Sensory -> Inter` + `Inter <-> Inter/Command` x `u_stim` gating,
     - Command/motor/pharyngeal neurons receive `Inter <-> Command` x `u_stim` gating.
3. `M3R2_03_h2_capacity_matched_additive`:
   Mandatory F9 capacity-matched additive control strictly matched in feature dimension, parameter
   count, rank (`rank = 4`), and L2 penalty to `M3R2_01_h2_bilinear_uniform`, using nonlinear
   additive bases on `z_state` alone and `u_stim` alone with ZERO `z_state x u_stim` cross-terms.
4. `M3R2_04_h2_film_gating_residual`:
   Compact PyTorch stimulus-conditioned FiLM multiplicative gating residual network
   (`(1 + 0.5*tanh(gamma(z_state))) * h(u_stim, tier) + beta(z_state)`) anchored on `M3R1_02`,
   early-stopped on an inner animal-grouped validation split with deterministic seed.

Also computes all mandatory F9 mechanistic controls & diagnostics:
  1. Layer Dissociation Difference-in-Differences (`Delta_inter - Delta_sensory`,
     `Delta_command - Delta_sensory`) with 1000-resample animal-level bootstrap 95% CIs.
  2. 45s Buffer Pseudo-Event Negative Control (natural `buffer_E_d` + sham-stimulus buffer probe).
  3. Within-Condition Permutation Control (20 deterministic permutations of `z_state` within
     the same `stimulus_id`, preserving stimulus-mean trajectory while breaking state-stimulus coupling).
  4. End-to-End Leakage Invariance Verification on `M3R2_02_h2_bilinear_hierarchical`.
"""

from __future__ import annotations

import copy
import datetime as dt
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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
from src.baselines import (
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
    _solve_ridge_sample_scaled,
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
from src.features import fit_fold_neuron_stim_priors, predict_prior_delta_trajectory

# Column partitions of `extract_circuit_slow_state_features` (51 dims total):
#   0..14  (15 dims): L1 sensory integration interneurons (AIA, AIB, AIY, AIZ, L/R preserved)
#   15..31 (17 dims): Command / reversal-forward attractors (AVA, AVB, RIM, RIB + AVE/AVD/RMD)
#   32..41 (10 dims): Asymmetric chemosensory context (ASEL, ASER, AWCON, AWCOFF + contrasts)
#   42..50 ( 9 dims): Fold-internal causal population PCA (6 recent + 3 slope)
IDX_INTER: Tuple[int, ...] = tuple(range(0, 15))
IDX_CMD: Tuple[int, ...] = tuple(range(15, 32))
IDX_SENS: Tuple[int, ...] = tuple(range(32, 42))
IDX_PCA: Tuple[int, ...] = tuple(range(42, 51))
IDX_INTER_CMD_PCA: Tuple[int, ...] = IDX_INTER + IDX_CMD + IDX_PCA

M3R2_MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "run_id": "r2_01_m3r2_01_h2_bilinear_uniform",
        "model_id": "M3R2_01_h2_bilinear_uniform",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "uses_bilinear_gating": True,
        "is_stochastic": False,
        "single_factor_changed": "uniform_low_rank_bilinear_state_x_stimulus_gating_rrr_rank4",
        "hypothesis": "H2 uniform: Low-rank bilinear (z_state (x) u_stim, rank=4 reduced-rank ridge) residual gating anchored on M3R1_02.",
    },
    {
        "run_id": "r2_02_m3r2_02_h2_bilinear_hierarchical",
        "model_id": "M3R2_02_h2_bilinear_hierarchical",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "uses_bilinear_gating": True,
        "is_stochastic": False,
        "single_factor_changed": "hierarchical_circuit_layer_bilinear_gating_sensory_vs_inter_vs_command",
        "hypothesis": "H2 hierarchical: Circuit-layer bilinear gating insulating primary sensory neurons from downstream command noise while gating Sensory->Inter and Inter<->Command.",
    },
    {
        "run_id": "r2_03_m3r2_03_h2_capacity_matched_additive",
        "model_id": "M3R2_03_h2_capacity_matched_additive",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "uses_bilinear_gating": False,
        "is_stochastic": False,
        "single_factor_changed": "capacity_matched_additive_nonlinear_z_and_u_zero_cross_terms",
        "hypothesis": "H2 F9 control: Strictly capacity-matched additive nonlinear basis on z_state and u_stim separately (same param count, rank=4, L2; ZERO z_state x u_stim cross-terms).",
    },
    {
        "run_id": "r2_04_m3r2_04_h2_film_gating_residual",
        "model_id": "M3R2_04_h2_film_gating_residual",
        "parent_id": "r1_02_m3r1_02_h1_confounds_only",
        "is_self_only": False,
        "uses_confounds": True,
        "uses_network_state": True,
        "uses_bilinear_gating": True,
        "is_stochastic": True,
        "single_factor_changed": "compact_stimulus_conditioned_film_multiplicative_gating_residual_net",
        "hypothesis": "H2 nonlinear FiLM: Compact PyTorch stimulus-conditioned FiLM gate ((1+0.5*tanh(gamma(z)))*h(u_stim,tier)+beta(z)) anchored on M3R1_02 with inner animal-grouped early stopping.",
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
    ]
    for rel in rel_files:
        p = exp_root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Core Linear Algebra Helpers (SVD Compression, Reduced-Rank Ridge, Additive Basis)
# ---------------------------------------------------------------------------
def _fit_svd_compressor(x: np.ndarray, n_comp: int) -> Dict[str, np.ndarray]:
    """Fit fold-internal mean/std + top-k SVD projection + post-projection scaling."""
    x64 = np.asarray(x, dtype=np.float64)
    mu = np.mean(x64, axis=0, keepdims=True)
    sd = np.maximum(np.std(x64, axis=0, keepdims=True), 1e-4)
    xn = np.clip((x64 - mu) / sd, -5.0, 5.0)
    d = xn.shape[1]
    k = min(int(n_comp), d, max(1, xn.shape[0] - 1))
    cov = (xn.T @ xn) / max(float(xn.shape[0]), 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:k]
    v = np.zeros((d, int(n_comp)), dtype=np.float64)
    v[:, :k] = eigvecs[:, order]
    proj = xn @ v
    p_sd = np.maximum(np.std(proj, axis=0, keepdims=True), 1e-4)
    return {
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "v": v.astype(np.float32),
        "p_sd": p_sd.astype(np.float32),
    }


def _apply_svd_compressor(x: np.ndarray, comp: Mapping[str, np.ndarray]) -> np.ndarray:
    xn = np.clip((np.asarray(x, dtype=np.float32) - comp["mu"]) / comp["sd"], -5.0, 5.0)
    proj = (xn @ comp["v"]) / comp["p_sd"]
    return np.clip(proj, -5.0, 5.0).astype(np.float32)


def _fit_reduced_rank_ridge(
    x: np.ndarray,
    y: np.ndarray,
    l2_penalty: float,
    rank: int,
) -> Dict[str, Any]:
    """Closed-form sample-scaled Reduced-Rank Ridge + exact effective degrees of freedom."""
    n_samp, d = x.shape
    b_dim = y.shape[1]
    if n_samp == 0 or d == 0:
        return {
            "w": np.zeros((d, b_dim), dtype=np.float32),
            "rank": 0,
            "param_count": int(d * b_dim),
            "eff_dof": 0.0,
        }
    w_ridge = _solve_ridge_sample_scaled(x, y, l2_penalty)
    y_hat = (x.astype(np.float64) @ w_ridge.astype(np.float64))
    gram_y = y_hat.T @ y_hat  # [B, B] tiny matrix
    eigvals, eigvecs = np.linalg.eigh(gram_y)
    r = min(int(rank), b_dim, d)
    top_idx = np.argsort(eigvals)[::-1][:r]
    v_r = eigvecs[:, top_idx]  # [B, r]
    proj_r = v_r @ v_r.T
    w_rrr = (w_ridge.astype(np.float64) @ proj_r).astype(np.float32)

    # Exact effective DoF of sample-scaled Ridge projected to rank r
    alpha = float(l2_penalty) * max(1.0, np.sqrt(float(n_samp) / 100.0))
    gram_x = x.astype(np.float64).T @ x.astype(np.float64)
    ev_x = np.maximum(np.linalg.eigvalsh(gram_x), 0.0)
    tr_h = float(np.sum(ev_x / (ev_x + alpha)))
    eff_dof = tr_h * float(r) / float(max(b_dim, 1))
    return {
        "w": w_rrr,
        "rank": int(r),
        "param_count": int(d * b_dim),
        "eff_dof": float(eff_dof),
    }


def _build_bilinear_features(z_low: np.ndarray, u_stim: np.ndarray) -> np.ndarray:
    """Outer product `(z_low (x) u_stim)` flattened to `[N, D_z * D_u]`."""
    n = z_low.shape[0]
    return (z_low[:, :, None] * u_stim[:, None, :]).reshape(n, -1).astype(np.float32)


def _build_capacity_matched_additive_features(
    z_low: np.ndarray,
    u_stim: np.ndarray,
    target_dim: int,
    seed: int = 20261117,
) -> np.ndarray:
    """Build purely additive nonlinear basis `[N, target_dim]` on `z_low` and `u_stim` separately.

    STRICTLY ZERO `z_low x u_stim` multiplicative cross-terms! Matches `target_dim == D_z * D_u`
    exactly so parameter count, rank, and L2 regularization match `M3R2_01_h2_bilinear_uniform`.
    """
    n, dz = z_low.shape
    du = u_stim.shape[1]

    # Purely u_stim terms (no z_low)
    u_blocks: List[np.ndarray] = [
        u_stim,
        np.tanh(u_stim),
        np.clip(u_stim ** 2, 0.0, 9.0),
    ]
    u_feat = np.hstack(u_blocks).astype(np.float32)
    # Keep at most min(u_feat.shape[1], max(du, target_dim // 4)) columns from u_stim
    n_u_keep = min(u_feat.shape[1], max(du, target_dim // 4))
    u_feat = u_feat[:, :n_u_keep]

    # Purely z_low terms (no u_stim)
    n_z_need = max(0, target_dim - n_u_keep)
    z_blocks: List[np.ndarray] = [
        z_low,
        np.tanh(z_low),
        np.clip(z_low ** 2, 0.0, 9.0),
    ]
    # Pairwise within-state interactions z_i * z_j (i < j) — still strictly zero u_stim!
    pair_cols = []
    for i in range(dz):
        for j in range(i + 1, dz):
            pair_cols.append(np.clip(z_low[:, i] * z_low[:, j], -9.0, 9.0))
    if pair_cols:
        z_blocks.append(np.column_stack(pair_cols).astype(np.float32))
    z_feat = np.hstack(z_blocks).astype(np.float32)

    if z_feat.shape[1] < n_z_need:
        extra = n_z_need - z_feat.shape[1]
        rng = np.random.default_rng(seed + 7919)
        r_mat = rng.normal(0.0, 1.0 / np.sqrt(max(dz, 1)), size=(dz, extra)).astype(np.float32)
        b_vec = rng.uniform(-0.5, 0.5, size=(1, extra)).astype(np.float32)
        z_extra = np.tanh(z_low @ r_mat + b_vec).astype(np.float32)
        z_feat = np.hstack([z_feat, z_extra]).astype(np.float32)

    out = np.hstack([u_feat, z_feat[:, :n_z_need]]).astype(np.float32)
    if out.shape[1] < target_dim:
        pad = np.zeros((n, target_dim - out.shape[1]), dtype=np.float32)
        out = np.hstack([out, pad]).astype(np.float32)
    return out[:, :target_dim]


# ---------------------------------------------------------------------------
# 2. Compact Stimulus-Conditioned FiLM Multiplicative Gating Residual Net
# ---------------------------------------------------------------------------
class ResidualFiLMGate(nn.Module):
    """Compact FiLM gate: `((1 + 0.5 * tanh(gamma(z_state))) * h(u_stim, tier) + beta(z_state)) -> delta_c`."""

    def __init__(self, d_z: int, d_u: int, d_tier: int, b_dim: int, film_dim: int = 24, dropout: float = 0.15) -> None:
        super().__init__()
        self.stim_enc = nn.Sequential(
            nn.Linear(d_u + d_tier, film_dim),
            nn.LayerNorm(film_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.state_mod = nn.Sequential(
            nn.Linear(d_z, film_dim),
            nn.LayerNorm(film_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(film_dim, 2 * film_dim),
        )
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(film_dim, b_dim),
        )
        # Initialize state modulator & output head near zero so epoch-0 prediction == M3R1_02 anchor
        nn.init.zeros_(self.state_mod[-1].weight)
        nn.init.zeros_(self.state_mod[-1].bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z_low: torch.Tensor, u_stim: torch.Tensor, tier_oh: torch.Tensor) -> torch.Tensor:
        h = self.stim_enc(torch.cat([u_stim, tier_oh], dim=-1))
        gb = self.state_mod(z_low)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        h_mod = (1.0 + 0.5 * torch.tanh(gamma)) * h + beta
        return self.head(h_mod)


# ---------------------------------------------------------------------------
# 3. Unified M3 Round 2 Model Class (Anchored on M3R1_02_h1_confounds_only)
# ---------------------------------------------------------------------------
class M3R2Model:
    """Implements all 4 M3 Round 2 H2 models anchored on `M3R1_02_h1_confounds_only`."""

    def __init__(
        self,
        model_id: str,
        config: Mapping[str, Any],
        canonical_neurons: Sequence[str],
        functional_tiers: Sequence[str],
    ) -> None:
        self.model_id = str(model_id)
        self.config = dict(config)
        self.hp = dict(config.get("hyperparameters", config))
        self.canonical_neurons = list(canonical_neurons)
        self.functional_tiers = list(functional_tiers)
        self.n_canon = len(self.canonical_neurons)
        self.state: Dict[str, Any] = {}

        tier_names = ("sensory", "interneuron", "premotor_command", "motor", "pharyngeal")
        self._tier_oh = np.zeros((self.n_canon, len(tier_names)), dtype=np.float32)
        for i, t_str in enumerate(self.functional_tiers):
            if t_str in tier_names:
                self._tier_oh[i, tier_names.index(t_str)] = 1.0

        ft_arr = np.asarray(self.functional_tiers)
        self._mask_sensory = ft_arr == "sensory"
        self._mask_inter = ft_arr == "interneuron"
        self._mask_cmd_other = ~(self._mask_sensory | self._mask_inter)

    def _extract_u_stim(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        stim_override: Optional[Tuple[str, bool]] = None,
    ) -> np.ndarray:
        """Build stimulus drive matrix `[N_canon, K_stim + 2]` (strictly 0 on buffer controls)."""
        if stim_override is not None:
            stim_id, is_buf = stim_override
            tr_use = dict(trial)
            tr_use["stimulus_id"] = stim_id
            tr_use["is_buffer_control"] = is_buf
        else:
            stim_id = str(trial["stimulus_id"])
            is_buf = bool(trial.get("is_buffer_control", False))
            tr_use = trial
        stim_oh_nonbuf = _encode_stim_onehot(stim_id, is_buf, priors["stim_vocab"])[:-1]
        stim_mat = np.repeat(stim_oh_nonbuf[None, :], self.n_canon, axis=0)
        _, prior_coef, _ = predict_prior_delta_trajectory(tr_use, priors, self.n_canon)
        prior_norm_2 = np.clip((prior_coef / s_n[:, None].astype(np.float32))[:, :2], -4.0, 4.0)
        return np.hstack([stim_mat, prior_norm_2]).astype(np.float32)

    def _fit_or_reuse_anchor(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
        anchor_cache: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[M3R1Model, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Fit `M3R1_02_h1_confounds_only` anchor + causal PCA + per-trial Stage-2 residuals (or reuse fold cache)."""
        if anchor_cache is not None:
            return anchor_cache["anchor"], anchor_cache["pca_state"], anchor_cache["train_res_map"]

        anchor = M3R1Model("M3R1_02_h1_confounds_only", self.config, self.canonical_neurons, self.functional_tiers)
        anchor.fit(train_trials, s_n, fold_cache=fold_cache)

        priors = anchor.state["priors"]
        rank = int(self.hp.get("rank", 6))
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
        non_buf_trials = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        pca_state = fit_causal_slow_state_pca(
            non_buf_trials, s_n, priors["mu_train"], self.n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

        train_res_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for tr in non_buf_trials:
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
    ) -> "M3R2Model":
        anchor, pca_state, train_res_map = self._fit_or_reuse_anchor(
            train_trials, s_n, fold_cache=fold_cache, anchor_cache=anchor_cache
        )
        priors = anchor.state["priors"]
        non_buf_trials = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]
        b_dim = int(priors["b_dim"])
        bilin_rank = int(self.hp.get("bilinear_rank", 4))
        bilin_l2 = float(self.hp.get("bilinear_l2_penalty", 60.0))
        seed = int(self.hp.get("seed", 20261117))

        self.state = {
            "s_n": np.asarray(s_n, dtype=np.float32).copy(),
            "anchor": anchor,
            "priors": priors,
            "pca_state": pca_state,
            "b_dim": b_dim,
            "capacity_info": {},
        }

        # Assemble valid training rows `(z_circ, u_stim, r2, neuron_idx)`
        z_rows, u_rows, y_rows, n_rows = [], [], [], []
        for tr in non_buf_trials:
            tid = str(tr.get("trial_id", ""))
            r2, rval, z_circ = train_res_map[tid]
            u_stim = self._extract_u_stim(tr, s_n, priors)
            idx = np.where(rval)[0]
            if idx.size > 0:
                z_rows.append(z_circ[idx])
                u_rows.append(u_stim[idx])
                y_rows.append(r2[idx])
                n_rows.append(idx)

        z_all = np.vstack(z_rows).astype(np.float32)
        u_all = np.vstack(u_rows).astype(np.float32)
        y_all = np.vstack(y_rows).astype(np.float32)
        n_all = np.concatenate(n_rows).astype(np.int32)
        d_u = u_all.shape[1]

        if self.model_id == "M3R2_01_h2_bilinear_uniform":
            p_z = 8
            z_comp = _fit_svd_compressor(z_all, n_comp=p_z)
            z_low = _apply_svd_compressor(z_all, z_comp)
            x_bilin = _build_bilinear_features(z_low, u_all)
            x_mu = np.mean(x_bilin, axis=0, keepdims=True).astype(np.float32)
            x_sd = np.maximum(np.std(x_bilin, axis=0, keepdims=True), 1e-4).astype(np.float32)
            x_norm = np.clip((x_bilin - x_mu) / x_sd, -5.0, 5.0).astype(np.float32)
            eff_l2 = bilin_l2 * max(1.0, float(x_norm.shape[1]) / 12.0)
            rrr = _fit_reduced_rank_ridge(x_norm, y_all, eff_l2, rank=bilin_rank)
            self.state["h2"] = {
                "z_comp": z_comp,
                "x_mu": x_mu,
                "x_sd": x_sd,
                "w": rrr["w"],
            }
            self.state["capacity_info"] = {
                "feature_dim": int(x_norm.shape[1]),
                "param_count": rrr["param_count"],
                "rank": rrr["rank"],
                "eff_dof": round(rrr["eff_dof"], 3),
                "eff_l2": round(eff_l2, 3),
            }

        elif self.model_id == "M3R2_02_h2_bilinear_hierarchical":
            tier_blocks: Dict[str, Dict[str, Any]] = {}
            tier_cfgs = [
                ("sensory", self._mask_sensory, list(IDX_SENS), 4, min(bilin_rank, 2), 2.0),
                ("interneuron", self._mask_inter, list(range(z_all.shape[1])), 8, bilin_rank, 1.0),
                ("command_other", self._mask_cmd_other, list(IDX_INTER_CMD_PCA), 6, bilin_rank, 1.0),
            ]
            total_params = 0
            total_dof = 0.0
            for t_name, t_mask, src_cols, p_k, r_k, l2_mult in tier_cfgs:
                row_sel = t_mask[n_all]
                if int(np.sum(row_sel)) < 20:
                    tier_blocks[t_name] = {"active": False, "src_cols": src_cols}
                    continue
                z_sub = z_all[row_sel][:, src_cols]
                u_sub = u_all[row_sel]
                y_sub = y_all[row_sel]
                z_comp = _fit_svd_compressor(z_sub, n_comp=p_k)
                z_low = _apply_svd_compressor(z_sub, z_comp)
                x_bilin = _build_bilinear_features(z_low, u_sub)
                x_mu = np.mean(x_bilin, axis=0, keepdims=True).astype(np.float32)
                x_sd = np.maximum(np.std(x_bilin, axis=0, keepdims=True), 1e-4).astype(np.float32)
                x_norm = np.clip((x_bilin - x_mu) / x_sd, -5.0, 5.0).astype(np.float32)
                eff_l2 = l2_mult * bilin_l2 * max(1.0, float(x_norm.shape[1]) / 12.0)
                rrr = _fit_reduced_rank_ridge(x_norm, y_sub, eff_l2, rank=r_k)
                tier_blocks[t_name] = {
                    "active": True,
                    "src_cols": src_cols,
                    "z_comp": z_comp,
                    "x_mu": x_mu,
                    "x_sd": x_sd,
                    "w": rrr["w"],
                }
                total_params += rrr["param_count"]
                total_dof += rrr["eff_dof"]
            self.state["h2"] = {"tier_blocks": tier_blocks}
            self.state["capacity_info"] = {
                "param_count": int(total_params),
                "eff_dof": round(total_dof, 3),
                "tiers": {
                    k: (int(v["w"].shape[0]) if v.get("active") else 0)
                    for k, v in tier_blocks.items()
                },
            }

        elif self.model_id == "M3R2_03_h2_capacity_matched_additive":
            p_z = 8
            z_comp = _fit_svd_compressor(z_all, n_comp=p_z)
            z_low = _apply_svd_compressor(z_all, z_comp)
            target_dim = p_z * d_u
            x_add = _build_capacity_matched_additive_features(z_low, u_all, target_dim=target_dim, seed=seed)
            x_mu = np.mean(x_add, axis=0, keepdims=True).astype(np.float32)
            x_sd = np.maximum(np.std(x_add, axis=0, keepdims=True), 1e-4).astype(np.float32)
            x_norm = np.clip((x_add - x_mu) / x_sd, -5.0, 5.0).astype(np.float32)
            eff_l2 = bilin_l2 * max(1.0, float(x_norm.shape[1]) / 12.0)
            rrr = _fit_reduced_rank_ridge(x_norm, y_all, eff_l2, rank=bilin_rank)
            self.state["h2"] = {
                "z_comp": z_comp,
                "target_dim": int(target_dim),
                "seed": seed,
                "x_mu": x_mu,
                "x_sd": x_sd,
                "w": rrr["w"],
            }
            self.state["capacity_info"] = {
                "feature_dim": int(x_norm.shape[1]),
                "param_count": rrr["param_count"],
                "rank": rrr["rank"],
                "eff_dof": round(rrr["eff_dof"], 3),
                "eff_l2": round(eff_l2, 3),
            }

        elif self.model_id == "M3R2_04_h2_film_gating_residual":
            self.state["h2"] = self._fit_film_residual(
                non_buf_trials=non_buf_trials,
                s_n=s_n,
                z_all=z_all,
                u_all=u_all,
                y_all=y_all,
                n_all=n_all,
                b_dim=b_dim,
            )
            net: nn.Module = self.state["h2"]["film_net"]
            n_params = int(sum(p.numel() for p in net.parameters()))
            self.state["capacity_info"] = {
                "param_count": n_params,
                "selected_epochs": int(self.state["h2"]["selected_epochs"]),
            }
        else:
            raise ValueError(f"Unknown M3R2 model_id: {self.model_id}")

        return self

    def _fit_film_residual(
        self,
        non_buf_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
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
        p_z = 8

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
        kappa = float(self.hp.get("missingness_precision_kappa", 4.0))

        z_itr_l, u_itr_l, y_itr_l, t_itr_l = [], [], [], []
        for tr in inner_tr:
            tid = str(tr.get("trial_id", ""))
            r2, rval, z_circ = in_res_map[tid]
            u_stim = self._extract_u_stim(tr, s_n, in_priors)
            idx = np.where(rval)[0]
            if idx.size > 0:
                z_itr_l.append(z_circ[idx])
                u_itr_l.append(u_stim[idx])
                y_itr_l.append(r2[idx])
                t_itr_l.append(self._tier_oh[idx])
        z_itr = np.vstack(z_itr_l)
        u_itr = np.vstack(u_itr_l)
        y_itr = np.vstack(y_itr_l)
        t_itr = np.vstack(t_itr_l)

        z_iva_l, u_iva_l, y_iva_l, t_iva_l = [], [], [], []
        for tr in inner_va:
            c_norm, rval, _, _ = _project_trial_residual_targets(tr, in_priors, s_n, self.n_canon)
            p1 = in_anchor._predict_stage1_coef(tr, s_n, in_priors, in_anchor.state["s1"])
            d2 = in_anchor._predict_stage2_delta_coef(tr, s_n, in_priors, in_anchor.state["s2"])
            r2 = (c_norm - (p1 + d2)).astype(np.float32)
            z_circ = extract_circuit_slow_state_features(
                tr, self.canonical_neurons, s_n, in_priors["mu_train"], in_pca, kappa=kappa
            )
            u_stim = self._extract_u_stim(tr, s_n, in_priors)
            idx = np.where(rval)[0]
            if idx.size > 0:
                z_iva_l.append(z_circ[idx])
                u_iva_l.append(u_stim[idx])
                y_iva_l.append(r2[idx])
                t_iva_l.append(self._tier_oh[idx])
        z_iva = np.vstack(z_iva_l)
        u_iva = np.vstack(u_iva_l)
        y_iva = np.vstack(y_iva_l)
        t_iva = np.vstack(t_iva_l)

        in_comp = _fit_svd_compressor(z_itr, n_comp=p_z)
        zl_itr = _apply_svd_compressor(z_itr, in_comp)
        zl_iva = _apply_svd_compressor(z_iva, in_comp)

        probe = ResidualFiLMGate(p_z, u_itr.shape[1], self._tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

        zt_t = torch.from_numpy(zl_itr)
        ut_t = torch.from_numpy(u_itr)
        tt_t = torch.from_numpy(t_itr)
        yt_t = torch.from_numpy(y_itr)

        zv_t = torch.from_numpy(zl_iva)
        uv_t = torch.from_numpy(u_iva)
        tv_t = torch.from_numpy(t_iva)
        yv_t = torch.from_numpy(y_iva)

        best_val = float(torch.mean(yv_t ** 2).item())
        best_ep = 0
        bad = 0
        n_rows = zt_t.shape[0]
        for ep in range(1, max_ep + 1):
            probe.train()
            perm = torch.randperm(n_rows)
            for s0 in range(0, n_rows, bsz):
                bi = perm[s0 : s0 + bsz]
                opt.zero_grad()
                pred = probe(zt_t[bi], ut_t[bi], tt_t[bi])
                loss = torch.mean((pred - yt_t[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            probe.eval()
            with torch.no_grad():
                vloss = float(torch.mean((probe(zv_t, uv_t, tv_t) - yv_t) ** 2).item())
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        # Outer refit on full training-fold animals for `best_ep` epochs
        outer_comp = _fit_svd_compressor(z_all, n_comp=p_z)
        zl_full = _apply_svd_compressor(z_all, outer_comp)
        t_full = self._tier_oh[n_all]

        torch.manual_seed(seed)
        final_net = ResidualFiLMGate(p_z, u_all.shape[1], self._tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            zf_t = torch.from_numpy(zl_full)
            uf_t = torch.from_numpy(u_all)
            tf_t = torch.from_numpy(t_full)
            yf_t = torch.from_numpy(y_all)
            nf = zf_t.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for s0 in range(0, nf, bsz):
                    bi = perm[s0 : s0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(zf_t[bi], uf_t[bi], tf_t[bi]) - yf_t[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()

        return {
            "z_comp": outer_comp,
            "film_net": final_net,
            "selected_epochs": best_ep,
        }

    def predict_h2_delta_coef(
        self,
        z_circ: np.ndarray,
        u_stim: np.ndarray,
    ) -> np.ndarray:
        """Compute ONLY the Stage-3 H2 residual basis coefficient delta `[N_canon, B]`."""
        h2 = self.state["h2"]
        b_dim = int(self.state["b_dim"])
        if self.model_id == "M3R2_01_h2_bilinear_uniform":
            z_low = _apply_svd_compressor(z_circ, h2["z_comp"])
            x_b = _build_bilinear_features(z_low, u_stim)
            x_n = np.clip((x_b - h2["x_mu"]) / h2["x_sd"], -5.0, 5.0)
            return (x_n @ h2["w"]).astype(np.float32)

        if self.model_id == "M3R2_02_h2_bilinear_hierarchical":
            delta = np.zeros((self.n_canon, b_dim), dtype=np.float32)
            mask_map = {
                "sensory": self._mask_sensory,
                "interneuron": self._mask_inter,
                "command_other": self._mask_cmd_other,
            }
            for t_name, t_mask in mask_map.items():
                blk = h2["tier_blocks"].get(t_name, {})
                if not blk.get("active", False) or not np.any(t_mask):
                    continue
                idx = np.where(t_mask)[0]
                z_sub = z_circ[idx][:, blk["src_cols"]]
                u_sub = u_stim[idx]
                z_low = _apply_svd_compressor(z_sub, blk["z_comp"])
                x_b = _build_bilinear_features(z_low, u_sub)
                x_n = np.clip((x_b - blk["x_mu"]) / blk["x_sd"], -5.0, 5.0)
                delta[idx] = (x_n @ blk["w"]).astype(np.float32)
            return delta

        if self.model_id == "M3R2_03_h2_capacity_matched_additive":
            z_low = _apply_svd_compressor(z_circ, h2["z_comp"])
            x_a = _build_capacity_matched_additive_features(
                z_low, u_stim, target_dim=int(h2["target_dim"]), seed=int(h2["seed"])
            )
            x_n = np.clip((x_a - h2["x_mu"]) / h2["x_sd"], -5.0, 5.0)
            return (x_n @ h2["w"]).astype(np.float32)

        if self.model_id == "M3R2_04_h2_film_gating_residual":
            z_low = _apply_svd_compressor(z_circ, h2["z_comp"])
            net: nn.Module = h2["film_net"]
            net.eval()
            with torch.no_grad():
                out = net(
                    torch.from_numpy(z_low.astype(np.float32)),
                    torch.from_numpy(u_stim.astype(np.float32)),
                    torch.from_numpy(self._tier_oh.astype(np.float32)),
                ).numpy()
            return out.astype(np.float32)

        raise ValueError(f"Unknown M3R2 model_id: {self.model_id}")

    def predict_trial(
        self,
        trial: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
        z_circ_override: Optional[np.ndarray] = None,
        stim_override_for_h2: Optional[Tuple[str, bool]] = None,
    ) -> np.ndarray:
        s_n = self.state["s_n"]
        anchor: M3R1Model = self.state["anchor"]
        priors = self.state["priors"]
        mu_train = priors["mu_train"]
        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
            _, _, prior_delta, phi = fold_cache["proj_targets"][tid]
        else:
            prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, self.n_canon)

        pred_c = anchor._predict_stage1_coef(trial, s_n, priors, anchor.state["s1"], fold_cache=fold_cache)
        pred_c = pred_c + anchor._predict_stage2_delta_coef(
            trial, s_n, priors, anchor.state["s2"], net_trial_override=net_trial_override
        )

        if z_circ_override is not None:
            z_circ = z_circ_override
        else:
            kappa = float(self.hp.get("missingness_precision_kappa", 4.0))
            net_src = net_trial_override if net_trial_override is not None else trial
            z_circ = extract_circuit_slow_state_features(
                net_src, self.canonical_neurons, s_n, mu_train, self.state["pca_state"], kappa=kappa
            )

        u_stim = self._extract_u_stim(trial, s_n, priors, stim_override=stim_override_for_h2)
        pred_c = pred_c + self.predict_h2_delta_coef(z_circ, u_stim)
        return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

    def parameter_vector(self) -> np.ndarray:
        parts: List[np.ndarray] = [
            self.state["anchor"].parameter_vector(),
            self.state["pca_state"]["v_rec"].ravel().astype(np.float64),
        ]
        h2 = self.state["h2"]
        if "w" in h2:
            parts.append(h2["w"].ravel().astype(np.float64))
        if "tier_blocks" in h2:
            for k in sorted(h2["tier_blocks"].keys()):
                blk = h2["tier_blocks"][k]
                if blk.get("active", False):
                    parts.append(blk["w"].ravel().astype(np.float64))
        if "film_net" in h2:
            for p in h2["film_net"].parameters():
                parts.append(p.detach().cpu().numpy().ravel().astype(np.float64))
        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 4. Mandatory H2 Mechanistic Controls & Diagnostics (PROJECT.md F9)
# ---------------------------------------------------------------------------
def compute_layer_dissociation_did(
    cand_sc: Mapping[str, Any],
    ref_sc: Mapping[str, Any],
    n_boot: int = 1000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Compute cross-layer Difference-in-Differences with paired 1000-resample animal bootstrap 95% CIs.

    `Delta_c = 1 - mean_w(E_cand[w, c]) / mean_w(E_ref[w, c])`
    `DiD_inter_minus_sensory = Delta_interneuron - Delta_sensory`
    `DiD_command_minus_sensory = Delta_command_motor - Delta_sensory`
    """
    c_wcc = cand_sc.get("worm_cell_class_errors", {})
    r_wcc = ref_sc.get("worm_cell_class_errors", {})

    def _tier_gain_and_boot(tier: str, worms: List[str], idx_mat: np.ndarray) -> Tuple[float, np.ndarray]:
        c_arr = np.array([float(c_wcc[tier][w]) for w in worms], dtype=np.float64)
        r_arr = np.array([float(r_wcc[tier][w]) for w in worms], dtype=np.float64)
        pt = float(1.0 - np.mean(c_arr) / max(np.mean(r_arr), 1e-12))
        b = 1.0 - np.mean(c_arr[idx_mat], axis=1) / np.maximum(np.mean(r_arr[idx_mat], axis=1), 1e-12)
        return pt, b

    common_worms = sorted(
        set(c_wcc.get("sensory", {}).keys())
        & set(c_wcc.get("interneuron", {}).keys())
        & set(c_wcc.get("command_motor", {}).keys())
        & set(r_wcc.get("sensory", {}).keys())
    )
    if len(common_worms) < 2:
        return {"n_worms": len(common_worms), "error": "insufficient_common_worms"}

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, len(common_worms), size=(n_boot, len(common_worms)))

    d_sens, b_sens = _tier_gain_and_boot("sensory", common_worms, idx_mat)
    d_inter, b_inter = _tier_gain_and_boot("interneuron", common_worms, idx_mat)
    d_cmd, b_cmd = _tier_gain_and_boot("command_motor", common_worms, idx_mat)

    did_is = d_inter - d_sens
    b_did_is = b_inter - b_sens
    did_cs = d_cmd - d_sens
    b_did_cs = b_cmd - b_sens

    return {
        "n_worms": len(common_worms),
        "Delta_sensory": round(d_sens, 6),
        "Delta_sensory_ci95": [round(float(np.percentile(b_sens, 2.5)), 6), round(float(np.percentile(b_sens, 97.5)), 6)],
        "Delta_interneuron": round(d_inter, 6),
        "Delta_interneuron_ci95": [round(float(np.percentile(b_inter, 2.5)), 6), round(float(np.percentile(b_inter, 97.5)), 6)],
        "Delta_command_motor": round(d_cmd, 6),
        "Delta_command_motor_ci95": [round(float(np.percentile(b_cmd, 2.5)), 6), round(float(np.percentile(b_cmd, 97.5)), 6)],
        "DiD_inter_minus_sensory": round(did_is, 6),
        "DiD_inter_minus_sensory_ci95": [round(float(np.percentile(b_did_is, 2.5)), 6), round(float(np.percentile(b_did_is, 97.5)), 6)],
        "DiD_command_minus_sensory": round(did_cs, 6),
        "DiD_command_minus_sensory_ci95": [round(float(np.percentile(b_did_cs, 2.5)), 6), round(float(np.percentile(b_did_cs, 97.5)), 6)],
        "_boot_inter_minus_sensory": b_did_is,
        "_boot_command_minus_sensory": b_did_cs,
    }


def compute_macro_did(
    did_by_ds: Mapping[str, Mapping[str, Any]],
    ds_list: Sequence[str],
) -> Dict[str, Any]:
    pt_is = float(np.mean([did_by_ds[d]["DiD_inter_minus_sensory"] for d in ds_list]))
    b_is = np.mean(np.vstack([did_by_ds[d]["_boot_inter_minus_sensory"] for d in ds_list]), axis=0)
    pt_cs = float(np.mean([did_by_ds[d]["DiD_command_minus_sensory"] for d in ds_list]))
    b_cs = np.mean(np.vstack([did_by_ds[d]["_boot_command_minus_sensory"] for d in ds_list]), axis=0)
    return {
        "DiD_inter_minus_sensory": round(pt_is, 6),
        "DiD_inter_minus_sensory_ci95": [round(float(np.percentile(b_is, 2.5)), 6), round(float(np.percentile(b_is, 97.5)), 6)],
        "DiD_command_minus_sensory": round(pt_cs, 6),
        "DiD_command_minus_sensory_ci95": [round(float(np.percentile(b_cs, 2.5)), 6), round(float(np.percentile(b_cs, 97.5)), 6)],
    }


def compute_buffer_gain_ci(
    cand_buf_lw: Mapping[str, float],
    ref_buf_lw: Mapping[str, float],
    n_boot: int = 1000,
    seed: int = 20261117,
) -> Dict[str, Any]:
    """Paired animal-level bootstrap 95% CI for relative gain on 45s buffer pseudo-events."""
    worms = sorted(set(cand_buf_lw.keys()) & set(ref_buf_lw.keys()))
    if not worms:
        return {"n_worms": 0, "I_d_buffer": None, "I_d_buffer_ci95": None}
    c_arr = np.array([float(cand_buf_lw[w]) for w in worms], dtype=np.float64)
    r_arr = np.array([float(ref_buf_lw[w]) for w in worms], dtype=np.float64)
    i_d = float(1.0 - np.mean(c_arr) / max(np.mean(r_arr), 1e-12))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(worms), size=(n_boot, len(worms)))
    b_g = 1.0 - np.mean(c_arr[idx], axis=1) / np.maximum(np.mean(r_arr[idx], axis=1), 1e-12)
    return {
        "n_worms": len(worms),
        "buffer_E_d_cand": round(float(np.mean(c_arr)), 6),
        "buffer_E_d_ref": round(float(np.mean(r_arr)), 6),
        "I_d_buffer": round(i_d, 6),
        "I_d_buffer_ci95": [round(float(np.percentile(b_g, 2.5)), 6), round(float(np.percentile(b_g, 97.5)), 6)],
        "n_worms_improved": int(np.sum(c_arr < r_arr - 1e-12)),
    }


def verify_m3_r2_invariance(
    model_id: str = "M3R2_02_h2_bilinear_hierarchical",
    dataset_id: str = "yemini2021",
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
) -> Dict[str, Any]:
    """Verify future-frame poisoning < 1e-7, OOF tamper < 1e-7, and target+bilateral exclusion = 0.0."""
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    raw_recs = load_dataset(dataset_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
    sub_recs = raw_recs[: min(8, len(raw_recs))]
    aligned, canon, tiers = align_records_to_canonical(sub_recs)
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

    model_clean = M3R2Model(model_id, cfg, canon, tiers).fit(train_trials_clean, s_n_clean)
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
    model_tamper = M3R2Model(model_id, cfg, canon, tiers).fit(train_trials_tamper, s_n_tamper)
    oof_weight_diff = float(np.max(np.abs(model_clean.parameter_vector() - model_tamper.parameter_vector())))
    oof_pred_diff = float(np.max(np.abs(y_pred_clean - model_tamper.predict_trial(val_trials_clean[0]))))

    # Probe 3: Target + bilateral exclusion (AVAR->AVAL, AIYR->AIYL, ASER->ASEL, AWCOFF->AWCON)
    idx_map = {n: i for i, n in enumerate(canon)}

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

    # Direct target-self perturbation inside network/gating pathway (holding self-history fixed)
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

    # Verify ASEL & ASER asymmetry preserved for AIYL interneuron
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
    zf_a = extract_circuit_slow_state_features(
        t_a, canon, s_n_clean, model_clean.state["priors"]["mu_train"], model_clean.state["pca_state"]
    )
    zf_b = extract_circuit_slow_state_features(
        t_b, canon, s_n_clean, model_clean.state["priors"]["mu_train"], model_clean.state["pca_state"]
    )
    aiyl_sees_ase_asymmetry = float(np.max(np.abs(zf_b[aiyl_i] - zf_a[aiyl_i])))

    passed = (
        future_diff < 1e-7
        and target_changed > 1e4
        and oof_weight_diff < 1e-7
        and oof_pred_diff < 1e-7
        and ava_bilat_diff < 1e-7
        and aiy_bilat_diff < 1e-7
        and ase_bilat_diff < 1e-7
        and awc_bilat_diff < 1e-7
        and target_self_net_diff < 1e-7
        and aiyl_sees_ase_asymmetry > 1e-3
    )
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "passed": bool(passed),
        "future_poison_max_abs_diff": future_diff,
        "future_poison_target_shift_verified": target_changed,
        "oof_tamper_weight_diff": oof_weight_diff,
        "oof_tamper_pred_diff": oof_pred_diff,
        "bilateral_exclusion_AVAR_to_AVAL_diff": ava_bilat_diff,
        "bilateral_exclusion_AIYR_to_AIYL_diff": aiy_bilat_diff,
        "bilateral_exclusion_ASER_to_ASEL_diff": ase_bilat_diff,
        "bilateral_exclusion_AWCOFF_to_AWCON_diff": awc_bilat_diff,
        "target_self_network_pathway_diff": target_self_net_diff,
        "aiyl_sees_asel_aser_asymmetry_delta": aiyl_sees_ase_asymmetry,
    }


# ---------------------------------------------------------------------------
# 5. Memory-Safe Runner (1 Dataset at a Time + gc.collect() + Controls + Scorer)
# ---------------------------------------------------------------------------
def run_m3_round2_h2(
    data_root: Path = DATA_ROOT_DEFAULT,
    exp_root: Path = EXP_ROOT_DEFAULT,
    n_boot: int = 1000,
    n_perm: int = 20,
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

    protocol_sha = compute_file_sha256(prot_path)
    config_sha = compute_file_sha256(cfg_path)
    split_sha = compute_file_sha256(sm_path)
    code_sha = _code_bundle_sha256(exp_root)

    datasets = ["yemini2021", "seyedolmohadesin2025", "toyoshima2024"]
    main_sources = ["seyedolmohadesin2025", "toyoshima2024"]
    runs_dir = exp_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    controls_cache_path = exp_root / "results" / "m3_round2_controls_cache.json"

    model_wall: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R2_MODEL_SPECS}
    model_cpu: Dict[str, float] = {s["model_id"]: 0.0 for s in M3R2_MODEL_SPECS}
    model_reg_ts: Dict[str, str] = {s["model_id"]: _utc_now() for s in M3R2_MODEL_SPECS}
    model_pred_files: Dict[str, Dict[str, str]] = {s["model_id"]: {} for s in M3R2_MODEL_SPECS}
    capacity_audit: Dict[str, Dict[str, Any]] = {s["model_id"]: {} for s in M3R2_MODEL_SPECS}

    # Diagnostic collectors (computed per dataset inside the memory-safe loop!)
    perm_results: Dict[str, Dict[str, Any]] = {
        "M3R2_01_h2_bilinear_uniform": {},
        "M3R2_02_h2_bilinear_hierarchical": {},
    }
    sham_buffer_results: Dict[str, Dict[str, Any]] = {
        "M3R2_01_h2_bilinear_uniform": {},
        "M3R2_02_h2_bilinear_hierarchical": {},
    }

    if stage == "score":
        cached = json.loads(controls_cache_path.read_text(encoding="utf-8"))
        model_wall = cached["model_wall"]
        model_cpu = cached["model_cpu"]
        model_reg_ts = cached["model_reg_ts"]
        model_pred_files = cached["model_pred_files"]
        capacity_audit = cached["capacity_audit"]
        perm_results = cached["perm_results"]
        sham_buffer_results = cached["sham_buffer_results"]
        print(f"[M3R2] Loaded controls cache from {controls_cache_path}", flush=True)

    # Step 1: Process ONE dataset at a time + gc.collect() (Peak RSS < 3.0 GiB)
    for ds_id in ([] if stage == "score" else datasets):
        t_ds0 = time.time()
        print(f"[M3R2] Loading Dev split: {ds_id} (RSS={_peak_rss_mb():.1f} MB)", flush=True)
        raw_recs = load_dataset(ds_id, split="dev", allow_vault_unseal=False, data_root=data_root, exp_root=exp_root)
        aligned, canon, tiers = align_records_to_canonical(raw_recs)
        splits = build_grouped_cv_splits(aligned)
        del raw_recs

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

            # Fit shared M3R1_02 anchor + causal PCA once per fold so all 4 H2 models share identical anchor
            helper = M3R2Model("M3R2_01_h2_bilinear_uniform", cfg, canon, tiers)
            anchor, pca_state, train_res_map = helper._fit_or_reuse_anchor(tr_trials, s_n, fold_cache=f_cache)
            # Precompute validation z_circ and anchor predictions for fast within-condition permutation & sham-buffer controls
            kappa = float(hp.get("missingness_precision_kappa", 4.0))
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

        # Evaluate anchor on validation trials (needed for exact paired permutation & sham-buffer deltas)
        all_val_trials: List[Dict[str, Any]] = []
        all_val_sn: List[np.ndarray] = []
        all_anchor_preds: List[np.ndarray] = []
        for fb in fold_bundles:
            for vtr, ap in zip(fb["val_trials"], fb["val_anchor_preds"]):
                all_val_trials.append(vtr)
                all_val_sn.append(fb["s_n"])
                all_anchor_preds.append(ap)
        anchor_eval_inmem = evaluate_predictions(
            all_val_trials, all_anchor_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
        )

        # Train & evaluate all 4 Round 2 models on this dataset
        for spec in M3R2_MODEL_SPECS:
            mid = spec["model_id"]
            rid = spec["run_id"]
            rdir = runs_dir / rid
            rdir.mkdir(parents=True, exist_ok=True)
            pred_file = rdir / f"predictions_dev_{ds_id}.npz"

            if resume and mid == "M3R2_04_h2_film_gating_residual" and pred_file.exists():
                model_pred_files[mid][ds_id] = str(pred_file.relative_to(exp_root))
                print(f"[M3R2]   Resumed existing {mid} on {ds_id} (saved on disk)", flush=True)
                continue

            t_m0 = time.time()
            c_m0 = time.process_time()
            oof_trials: List[Dict[str, Any]] = []
            oof_preds: List[np.ndarray] = []
            oof_fold_ids: List[int] = []
            fold_models: List[M3R2Model] = []
            fold_caps: List[Dict[str, Any]] = []

            for fb in fold_bundles:
                model = M3R2Model(mid, cfg, canon, tiers)
                model.fit(
                    fb["train_trials"],
                    fb["s_n"],
                    fold_cache=fb["cache"],
                    anchor_cache=fb["anchor_cache"],
                )
                fold_caps.append(model.state.get("capacity_info", {}))
                if mid in perm_results:
                    fold_models.append(model)
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

            # Record fold-averaged capacity info
            cap_summary: Dict[str, Any] = {}
            if fold_caps:
                for k in ("feature_dim", "param_count", "rank", "eff_dof", "eff_l2", "selected_epochs"):
                    vals = [fc[k] for fc in fold_caps if k in fc]
                    if vals:
                        cap_summary[k] = round(float(np.mean(vals)), 3)
            capacity_audit[mid][ds_id] = cap_summary
            print(f"[M3R2]   {mid} on {ds_id}: {dt_w:.2f}s | cap={cap_summary} (RSS={_peak_rss_mb():.1f} MB)", flush=True)

            # Run Mandatory Controls (Within-Condition Permutation & Sham-Stimulus Buffer Probe) for M3R2_01 & M3R2_02
            if mid in perm_results:
                # (A) Within-Condition Permutation Control (20 deterministic permutations)
                unperm_eval = evaluate_predictions(
                    oof_trials, oof_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                )
                unperm_e_d = float(unperm_eval["E_d"])
                anc_e_d = float(anchor_eval_inmem["E_d"])
                unperm_i_d_vs_anc = float(1.0 - unperm_e_d / max(anc_e_d, 1e-12))

                # Flatten all validation trial metadata + z_circ + fold index for within-condition permutation
                flat_z: List[np.ndarray] = []
                flat_fidx: List[int] = []
                stim_to_indices: Dict[str, List[int]] = {}
                g_i = 0
                for fi_pos, fb in enumerate(fold_bundles):
                    for vtr, z_c in zip(fb["val_trials"], fb["val_z_circ"]):
                        flat_z.append(z_c)
                        flat_fidx.append(fi_pos)
                        if not bool(vtr.get("is_buffer_control", False)):
                            stim_to_indices.setdefault(str(vtr["stimulus_id"]), []).append(g_i)
                        g_i += 1

                perm_e_ds: List[float] = []
                perm_i_ds: List[float] = []
                for p_idx in range(n_perm):
                    rng_p = np.random.default_rng(seed + 1000 + p_idx)
                    perm_map = list(range(len(flat_z)))
                    for s_key in sorted(stim_to_indices.keys()):
                        idxs = stim_to_indices[s_key]
                        if len(idxs) >= 2:
                            shuf = rng_p.permutation(idxs).tolist()
                            for orig_i, new_i in zip(idxs, shuf):
                                perm_map[orig_i] = new_i

                    p_preds: List[np.ndarray] = []
                    for gi_cur, (vtr, fi_pos) in enumerate(zip(oof_trials, flat_fidx)):
                        if bool(vtr.get("is_buffer_control", False)):
                            p_preds.append(oof_preds[gi_cur])
                        else:
                            z_perm = flat_z[perm_map[gi_cur]]
                            yp_p = fold_models[fi_pos].predict_trial(
                                vtr, fold_cache=fold_bundles[fi_pos]["cache"], z_circ_override=z_perm
                            )
                            p_preds.append(yp_p)

                    pe = evaluate_predictions(
                        oof_trials, p_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                    )
                    pe_ed = float(pe["E_d"])
                    perm_e_ds.append(pe_ed)
                    perm_i_ds.append(float(1.0 - pe_ed / max(anc_e_d, 1e-12)))

                perm_mean_i = float(np.mean(perm_i_ds))
                perm_std_i = float(np.std(perm_i_ds))
                perm_results[mid][ds_id] = {
                    "n_permutations": n_perm,
                    "unpermuted_E_d": round(unperm_e_d, 6),
                    "anchor_M3R1_02_E_d": round(anc_e_d, 6),
                    "unpermuted_I_d_vs_M3R1_02": round(unperm_i_d_vs_anc, 6),
                    "permuted_mean_E_d": round(float(np.mean(perm_e_ds)), 6),
                    "permuted_mean_I_d_vs_M3R1_02": round(perm_mean_i, 6),
                    "permuted_std_I_d_vs_M3R1_02": round(perm_std_i, 6),
                    "I_d_drop_on_permutation": round(unperm_i_d_vs_anc - perm_mean_i, 6),
                    "frac_perms_worse_than_unpermuted": round(
                        float(np.mean([x < unperm_i_d_vs_anc for x in perm_i_ds])), 4
                    ),
                }

                # (B) Sham-Stimulus Buffer Pseudo-Event Probe:
                # Apply trained H2 gating residual to buffer pseudo-events paired with a sham real stimulus
                rng_b = np.random.default_rng(seed + 5555)
                sham_preds: List[np.ndarray] = []
                for gi_cur, (vtr, fi_pos) in enumerate(zip(oof_trials, flat_fidx)):
                    if not bool(vtr.get("is_buffer_control", False)):
                        sham_preds.append(oof_preds[gi_cur])
                    else:
                        vocab = fold_models[fi_pos].state["priors"]["stim_vocab"]
                        sham_stim = str(vocab[int(rng_b.integers(0, len(vocab)))]) if vocab else "unknown"
                        yp_sham = fold_models[fi_pos].predict_trial(
                            vtr,
                            fold_cache=fold_bundles[fi_pos]["cache"],
                            z_circ_override=flat_z[gi_cur],
                            stim_override_for_h2=(sham_stim, False),
                        )
                        sham_preds.append(yp_sham)
                sham_eval = evaluate_predictions(
                    oof_trials, sham_preds, all_val_sn, functional_tiers=tiers, compute_bootstrap=False
                )
                sham_buffer_results[mid][ds_id] = compute_buffer_gain_ci(
                    sham_eval.get("buffer_L_w", {}),
                    anchor_eval_inmem.get("buffer_L_w", {}),
                    n_boot=n_boot,
                    seed=seed,
                )

                del fold_models

        del fold_bundles, train_fold_session_ids, all_val_trials, all_val_sn, all_anchor_preds
        gc.collect()
        print(f"[M3R2] Finished {ds_id} in {time.time() - t_ds0:.1f}s (post-GC RSS={_peak_rss_mb():.1f} MB)", flush=True)

    if stage != "score":
        controls_cache_path.parent.mkdir(parents=True, exist_ok=True)
        controls_cache_path.write_text(
            json.dumps(
                {
                    "model_wall": model_wall,
                    "model_cpu": model_cpu,
                    "model_reg_ts": model_reg_ts,
                    "model_pred_files": model_pred_files,
                    "capacity_audit": capacity_audit,
                    "perm_results": perm_results,
                    "sham_buffer_results": sham_buffer_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[M3R2] Saved controls cache to {controls_cache_path}", flush=True)
        if stage == "controls":
            return {"status": "controls_done", "controls_cache": str(controls_cache_path)}

    # Step 2: Score EVERY run with standalone `evaluators/scorer.py` (1 dataset at a time!)
    print("[M3R2] Scoring all 4 Round 2 runs + B4 & M3R1_02 anchors via evaluators/scorer.py...", flush=True)
    scored_by_model: Dict[str, Dict[str, Any]] = {s["model_id"]: {} for s in M3R2_MODEL_SPECS}
    ref_b4_scored: Dict[str, Any] = {}
    ref_m02_scored: Dict[str, Any] = {}

    for ds_id in datasets:
        t_s0 = time.time()
        b4_path = exp_root / "runs" / "r0_04_b4_ridge_self_stim" / f"predictions_dev_{ds_id}.npz"
        m02_path = exp_root / "runs" / "r1_02_m3r1_02_h1_confounds_only" / f"predictions_dev_{ds_id}.npz"
        ref_b4_scored[ds_id] = score_saved_predictions_file(
            pred_path=b4_path, split="dev", ref_pred_path=None,
            data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
        )
        ref_m02_scored[ds_id] = score_saved_predictions_file(
            pred_path=m02_path, split="dev", ref_pred_path=None,
            data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
        )
        for spec in M3R2_MODEL_SPECS:
            mid = spec["model_id"]
            p_path = exp_root / model_pred_files[mid][ds_id]
            scored_by_model[mid][ds_id] = score_saved_predictions_file(
                pred_path=p_path, split="dev", ref_pred_path=None,
                data_root=data_root, exp_root=exp_root, allow_vault_unseal=False, resolve_registry_ref=False,
            )
        gc.collect()
        print(f"[M3R2]   Scored {ds_id} in {time.time() - t_s0:.1f}s (RSS={_peak_rss_mb():.1f} MB)", flush=True)

    # Step 3: Pairwise Comparisons (vs B4, vs M3R1_02, vs M3R2_03) with 1000-resample animal bootstrap 95% CIs
    m03_id = "M3R2_03_h2_capacity_matched_additive"
    summary_models: Dict[str, Any] = {}
    run_jsonl_entries: List[Dict[str, Any]] = []

    for spec in M3R2_MODEL_SPECS:
        mid = spec["model_id"]
        rid = spec["run_id"]
        ds_out: Dict[str, Any] = {}
        for ds_id in datasets:
            sc = scored_by_model[mid][ds_id]
            cmp_b4 = compare_against_reference(sc, ref_b4_scored[ds_id], n_boot=n_boot, seed=seed)
            cmp_m02 = compare_against_reference(sc, ref_m02_scored[ds_id], n_boot=n_boot, seed=seed)
            cmp_m03 = compare_against_reference(sc, scored_by_model[m03_id][ds_id], n_boot=n_boot, seed=seed)
            buf_vs_m02 = compute_buffer_gain_ci(
                sc.get("buffer_L_w", {}), ref_m02_scored[ds_id].get("buffer_L_w", {}), n_boot=n_boot, seed=seed
            )
            ds_out[ds_id] = {
                "E_d": round(float(sc["E_d"]), 6),
                "rmse_raw": round(float(sc["rmse_raw"]), 6),
                "post_R2": round(float(sc["post_R2"]), 6),
                "post_R2_ci95": [round(float(x), 6) for x in sc["post_R2_ci95"]],
                "delta_R2": round(float(sc["delta_R2"]), 6),
                "delta_R2_ci95": [round(float(x), 6) for x in sc["delta_R2_ci95"]],
                "buffer_E_d": round(float(sc["buffer_E_d"]), 6) if sc.get("buffer_E_d") is not None else None,
                "buffer_vs_M3R1_02": buf_vs_m02,
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
                "vs_M3R2_03_h2_capacity_matched_additive": {
                    "I_d": round(float(cmp_m03["I_d"]), 6),
                    "I_d_ci95": [round(float(x), 6) for x in cmp_m03["I_d_ci95"]],
                    "n_worms_improved": int(cmp_m03["n_worms_improved"]),
                    "fraction_worms_improved": round(float(cmp_m03["fraction_worms_improved"]), 4),
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
        macro_m03_main2 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m03_id], main_sources, n_boot=n_boot, seed=seed)
        macro_m03_all3 = compute_macro_bootstrap_ci(scored_by_model[mid], scored_by_model[m03_id], datasets, n_boot=n_boot, seed=seed)

        worst_main_b4 = min(ds_out[d]["vs_B4_ridge_self_stim"]["I_d"] for d in main_sources)

        summary_models[mid] = {
            "run_id": rid,
            "parent_id": spec["parent_id"],
            "is_self_only": spec["is_self_only"],
            "uses_confounds": spec["uses_confounds"],
            "uses_network_state": spec["uses_network_state"],
            "uses_bilinear_gating": spec["uses_bilinear_gating"],
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
            "I_macro_main2_vs_M3R2_03": macro_m03_main2["I_macro"],
            "I_macro_main2_vs_M3R2_03_ci95": macro_m03_main2["I_macro_ci95"],
            "I_macro_all3_vs_M3R2_03": macro_m03_all3["I_macro"],
            "I_macro_all3_vs_M3R2_03_ci95": macro_m03_all3["I_macro_ci95"],
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
                "round": 2,
                "category": "M3_round2_H2",
                "model_id": mid,
                "family_id": "M3R2_H2",
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
                    "code_bundle_sha256": code_sha,
                },
                "reference_baseline_id": "B4_ridge_self_stim",
                "I_macro": macro_b4_main2["I_macro"],
                "I_macro_ci95": macro_b4_main2["I_macro_ci95"],
                "I_macro_all3_immobilized": macro_b4_all3["I_macro"],
                "I_macro_all3_ci95": macro_b4_all3["I_macro_ci95"],
                "I_macro_vs_M3R1_02": macro_m02_main2["I_macro"],
                "I_macro_vs_M3R2_03": macro_m03_main2["I_macro"],
                "worst_main_source_I_d": round(float(worst_main_b4), 6),
                "source_metrics": src_metrics_jsonl,
                "prediction_files_dev": model_pred_files[mid],
                "wall_clock_sec": round(model_wall[mid], 3),
                "cpu_time_sec": round(model_cpu[mid], 3),
                "peak_rss_mb": _peak_rss_mb(),
                "exit_code": 0,
                "status": "completed",
                "decision": "RETAINED_M3_R2_DIAGNOSTIC",
            }
        )

    # Step 4: Layer Dissociation Difference-in-Differences (DiD) with 1000-resample animal bootstrap CIs
    did_report: Dict[str, Any] = {}
    did_contrasts = [
        ("M3R2_01_vs_M3R2_03", "M3R2_01_h2_bilinear_uniform", scored_by_model[m03_id]),
        ("M3R2_02_vs_M3R2_03", "M3R2_02_h2_bilinear_hierarchical", scored_by_model[m03_id]),
        ("M3R2_01_vs_M3R1_02", "M3R2_01_h2_bilinear_uniform", ref_m02_scored),
        ("M3R2_02_vs_M3R1_02", "M3R2_02_h2_bilinear_hierarchical", ref_m02_scored),
    ]
    for label, cand_id, ref_map in did_contrasts:
        raw_by_ds: Dict[str, Dict[str, Any]] = {}
        for ds_id in datasets:
            raw_by_ds[ds_id] = compute_layer_dissociation_did(
                scored_by_model[cand_id][ds_id], ref_map[ds_id], n_boot=n_boot, seed=seed
            )
        macro_main2 = compute_macro_did(raw_by_ds, main_sources)
        macro_all3 = compute_macro_did(raw_by_ds, datasets)
        clean_by_ds = {
            ds_id: {k: v for k, v in d.items() if not k.startswith("_")}
            for ds_id, d in raw_by_ds.items()
        }
        did_report[label] = {
            "macro_main2": macro_main2,
            "macro_all3": macro_all3,
            "datasets": clean_by_ds,
        }

    # Aggregate Within-Condition Permutation Control across Main 2 & All 3
    perm_summary: Dict[str, Any] = {}
    for mid in ("M3R2_01_h2_bilinear_uniform", "M3R2_02_h2_bilinear_hierarchical"):
        dmap = perm_results[mid]
        m2_unp = float(np.mean([dmap[d]["unpermuted_I_d_vs_M3R1_02"] for d in main_sources]))
        m2_prm = float(np.mean([dmap[d]["permuted_mean_I_d_vs_M3R1_02"] for d in main_sources]))
        a3_unp = float(np.mean([dmap[d]["unpermuted_I_d_vs_M3R1_02"] for d in datasets]))
        a3_prm = float(np.mean([dmap[d]["permuted_mean_I_d_vs_M3R1_02"] for d in datasets]))
        perm_summary[mid] = {
            "I_macro_main2_unpermuted_vs_M3R1_02": round(m2_unp, 6),
            "I_macro_main2_permuted_vs_M3R1_02": round(m2_prm, 6),
            "I_macro_main2_drop": round(m2_unp - m2_prm, 6),
            "I_macro_all3_unpermuted_vs_M3R1_02": round(a3_unp, 6),
            "I_macro_all3_permuted_vs_M3R1_02": round(a3_prm, 6),
            "I_macro_all3_drop": round(a3_unp - a3_prm, 6),
            "datasets": dmap,
        }

    # Aggregate 45s Buffer Pseudo-Event Negative Control (natural + sham-stimulus probe)
    buffer_control_summary: Dict[str, Any] = {}
    for mid in ("M3R2_01_h2_bilinear_uniform", "M3R2_02_h2_bilinear_hierarchical"):
        buffer_control_summary[mid] = {
            "natural_buffer_vs_M3R1_02": {
                ds_id: summary_models[mid]["datasets"][ds_id]["buffer_vs_M3R1_02"] for ds_id in datasets
            },
            "sham_stimulus_buffer_probe_vs_M3R1_02": sham_buffer_results[mid],
        }

    # Step 5: Explicit Leakage Invariance Verification on M3R2_02_h2_bilinear_hierarchical
    print("[M3R2] Running leakage invariance check on M3R2_02_h2_bilinear_hierarchical...", flush=True)
    invariance_res = verify_m3_r2_invariance(
        model_id="M3R2_02_h2_bilinear_hierarchical",
        dataset_id="yemini2021",
        data_root=data_root,
        exp_root=exp_root,
    )
    if not invariance_res["passed"]:
        raise RuntimeError(f"Leakage invariance FAILED on M3R2_02_h2_bilinear_hierarchical: {invariance_res}")
    print(
        f"[M3R2] Invariance PASSED: future_diff={invariance_res['future_poison_max_abs_diff']:.2e}, "
        f"oof_w_diff={invariance_res['oof_tamper_weight_diff']:.2e}, "
        f"oof_p_diff={invariance_res['oof_tamper_pred_diff']:.2e}, "
        f"bilat_AIY={invariance_res['bilateral_exclusion_AIYR_to_AIYL_diff']:.2e}",
        flush=True,
    )

    # Step 6: Append 4 rows to runs.jsonl (idempotent by run_id -> total 23 rows)
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

    # Step 7: Save results/m3_round2_h2_summary.json
    summary_payload = {
        "schema_version": "1.0.0",
        "round": 2,
        "milestone": "M3_round2_H2",
        "created_at_utc": _utc_now(),
        "config_id": cfg["config_id"],
        "lambda_global_path": "configs/lambda_global.json",
        "lambda_global_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "split_manifest_sha256": split_sha,
        "code_bundle_sha256": code_sha,
        "n_bootstrap_resamples": n_boot,
        "n_within_condition_permutations": n_perm,
        "reference_baseline_id": "B4_ridge_self_stim",
        "anchor_model_id": "M3R1_02_h1_confounds_only",
        "reference_baseline_E_d": {d: round(float(ref_b4_scored[d]["E_d"]), 6) for d in datasets},
        "anchor_M3R1_02_E_d": {d: round(float(ref_m02_scored[d]["E_d"]), 6) for d in datasets},
        "invariance_verification": invariance_res,
        "layer_dissociation_did": did_report,
        "buffer_pseudo_event_control": buffer_control_summary,
        "within_condition_permutation_control": perm_summary,
        "models": summary_models,
        "total_wall_clock_sec": round(time.time() - t_start, 3),
        "peak_rss_mb": _peak_rss_mb(),
        "runs_jsonl_total_rows": len(existing_lines),
    }
    out_summary = exp_root / "results" / "m3_round2_h2_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(
        f"[M3R2] Complete in {summary_payload['total_wall_clock_sec']:.1f}s "
        f"(Peak RSS={summary_payload['peak_rss_mb']:.1f} MB, runs.jsonl={len(existing_lines)} rows). "
        f"Saved {out_summary}",
        flush=True,
    )
    return summary_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run M3 Round 2 (H2 bilinear State x Stimulus gating & controls)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing saved M3R2_04 predictions_dev_*.npz if present")
    parser.add_argument("--stage", choices=["all", "controls", "score"], default="all", help="Run full pipeline, controls only, or scoring only")
    args = parser.parse_args()
    run_m3_round2_h2(resume=bool(args.resume), stage=str(args.stage))
