"""Milestone M4 — Stage A: >= 199 Fully Retrained Constrained Random Null Graphs.

Implements `PROJECT.md F12` and `MODEL_REQUIREMENTS.md §1, §7`:
Tests whether the biological C. elegans multi-layer receptor-polarity-signed connectome
(`W_chem_pos`, `W_chem_neg`, `W_chem_unk`, `W_gap`, `W_witvliet_core`, `W_npp_short`,
`W_npp_mid`, `W_monoamine`) genuinely outperforms `K = 199` degree-, spatial-distance-,
and cell-type-preserving randomized null graphs under identical CV fold splits and
unified `configs/lambda_global.json`.

STRICT INTEGRITY GUARANTEE:
Every null graph `k = 1..199` is genuinely retrained from scratch inside EVERY CV fold
after recomputing incoming causal connectome drives from the rewired null graph.
Predictions are NEVER shuffled, and weights trained on the real graph are NEVER reused.

Staged resumable CLI (`--stage {benchmark,ridge199,film_null,summarize,all}`, `--resume`):
  1. `benchmark`: Verify null-graph constraints on sample, verify `k=0` real-graph
     exact agreement (`< 1e-6`) with saved `M3R3_03`, and time 1 ridge null + 1 FiLM null.
  2. `ridge199`: Generate & verify all `K = 199` constrained null graphs, then run full
     fold-by-fold retraining across all 3 Dev datasets (`yemini2021`, `seyedolmohadesin2025`,
     `toyoshima2024`) for:
       (a) Primary `K = 199` full 8-layer constrained rewired null (`M3R3_03_herm_ridge`),
       (b) `K = 199` CeNGEN polarity-sign-only permutation null (fixed topology),
       (c) `K = 199` gap-junction-only rewired null (fixed chemical & wireless layers).
  3. `film_null`: Run `K_film` (default 19, configurable via `--k-film`) genuine end-to-end
     PyTorch FiLM retrainings of `M3R4_01_connectome_film_herm_final` on constrained null
     graphs `k = 1..K_film` across Main 2 (`seyedolmohadesin2025`, `toyoshima2024`).
  4. `summarize`: Aggregate all checkpoints into `results/null_graphs_summary.json`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from evaluators.scorer import score_saved_predictions_file
from models.m3_round1_h1 import (
    M3R1Model,
    _build_m3_fold_cache,
    _peak_rss_mb,
    _utc_now,
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
from models.m3_round3_h3 import (
    _solve_weighted_ridge_sample_scaled,
    extract_multiscale_presynaptic_state,
    load_signed_multilayer_connectome,
)
from models.m3_round4_f11 import (
    DATASETS_ALL3,
    DATASETS_MAIN2,
    PRIMARY_SEED,
    _extract_fold_trials,
    _load_reference_scored,
    _prepare_dataset_folds,
)
from src.baselines import (
    _encode_stim_onehot,
    _project_trial_residual_targets,
    _reconstruct_y_pred_from_coef,
)
from src.data_adapters import DATA_ROOT_DEFAULT, EXP_ROOT_DEFAULT, compute_file_sha256
from src.evaluator import evaluate_predictions
from src.features import predict_prior_delta_trajectory

LAYER_NAMES: Tuple[str, ...] = (
    "W_chem_herm_pos",
    "W_chem_herm_neg",
    "W_chem_herm_unk",
    "W_gap_herm",
    "W_witvliet_core",
    "W_npp_short",
    "W_npp_mid",
    "W_monoamine",
)
WIRED_IDX: Tuple[int, ...] = (0, 1, 2, 3, 4)
WIRELESS_IDX: Tuple[int, ...] = (5, 6, 7)
DENSE_WIRELESS_NAMES = {"W_npp_short", "W_npp_mid", "W_monoamine"}
DEFAULT_K_NULL: int = 199
DEFAULT_K_FILM: int = 19
SCRATCH_DEFAULT = Path("/home/antigravity/.gemini/jetski/scratch/teamwork_tmp/worker_m4_null_graphs")


def _get_scratch_dir() -> Path:
    env_tmp = os.environ.get("TMPDIR", "")
    p = Path(env_tmp) if env_tmp else SCRATCH_DEFAULT
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    if a64.size < 2 or np.std(a64) < 1e-12 or np.std(b64) < 1e-12:
        return 1.0 if np.allclose(a64, b64) else 0.0
    r = float(np.corrcoef(a64, b64)[0, 1])
    return r if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# 1. Constrained Degree-, Spatial-Distance-, and Cell-Type-Preserving Nulls
# ---------------------------------------------------------------------------
class ConstrainedNullConnectomeGenerator:
    """Generates degree-, 3D-distance-bin-, and cell-type-pair-preserving null connectomes."""

    def __init__(
        self,
        canonical_neurons: Sequence[str],
        data_root: Path = DATA_ROOT_DEFAULT,
        n_dist_bins: int = 5,
    ) -> None:
        self.canonical_neurons = list(canonical_neurons)
        self.n_canon = len(self.canonical_neurons)
        self.n_dist_bins = int(n_dist_bins)
        self.atlas = load_signed_multilayer_connectome(self.canonical_neurons, data_root=data_root)
        self.allow: np.ndarray = np.asarray(self.atlas["allow_mask"], dtype=bool)

        bundle_path = data_root / "connectomes" / "connectome_priors_bundle.npz"
        raw = np.load(bundle_path, allow_pickle=True)
        bundle_neurons = [str(n) for n in raw["canonical_neurons"]]
        idx_map = {n: i for i, n in enumerate(bundle_neurons)}
        perm = np.array([idx_map[n] for n in self.canonical_neurons], dtype=np.int64)

        tiers_raw = [str(t) for t in raw["functional_tiers"]]
        self.functional_tiers = [tiers_raw[i] for i in perm]
        tier_uniq = sorted(set(self.functional_tiers))
        self.tier_names = tier_uniq
        self.tier_id = np.array([tier_uniq.index(t) for t in self.functional_tiers], dtype=np.int32)

        dist_raw = np.asarray(raw["soma_distance"], dtype=np.float32)
        dist = dist_raw[np.ix_(perm, perm)]
        # Enforce exact symmetry on finite entries
        fin_pos = dist[np.isfinite(dist) & (dist > 0.0)]
        qs = np.linspace(0.0, 1.0, self.n_dist_bins + 1)[1:-1]
        self.dist_bin_edges = np.quantile(fin_pos, qs).astype(np.float32)
        dist_bin = np.full((self.n_canon, self.n_canon), self.n_dist_bins, dtype=np.int32)
        fin_m = np.isfinite(dist)
        dist_bin[fin_m] = np.digitize(dist[fin_m], self.dist_bin_edges).astype(np.int32)
        self.dist_bin = dist_bin

        # Stack real 8 layers in canonical order: shape [8, N, N]
        self.real_stack = np.stack(
            [np.asarray(self.atlas["matrices"][k], dtype=np.float32) for k in LAYER_NAMES],
            axis=0,
        )
        self.chem_unsigned = np.asarray(self.atlas["matrices"]["W_chem_herm_unsigned"], dtype=np.float32)

    @staticmethod
    def _rewire_directed(
        w: np.ndarray,
        allow: np.ndarray,
        tier_id: np.ndarray,
        dist_bin: np.ndarray,
        rng: np.random.Generator,
        target_swaps_per_edge: float = 2.0,
        max_rounds: int = 25,
        permute_weights_in_bucket: bool = False,
    ) -> Tuple[np.ndarray, int]:
        rows, cols = np.nonzero(w)
        n_e = int(rows.size)
        if n_e < 2:
            return w.copy(), 0
        weights = w[rows, cols].copy()
        adj = (w > 0.0).copy()

        keys = tier_id[rows] * 100 + tier_id[cols] * 10 + dist_bin[rows, cols]
        uniq_k, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        buckets = [np.where(inv == i)[0] for i in range(uniq_k.size) if counts[i] >= 2]

        accepted = 0
        for b_idx in buckets:
            m = int(b_idx.size)
            target = int(m * target_swaps_per_edge)
            acc_b = 0
            batch_sz = max(m * 3, 96)
            for _ in range(max_rounds):
                if acc_b >= target:
                    break
                a = rng.integers(0, m, size=batch_sz)
                b = rng.integers(0, m, size=batch_sz)
                e1 = b_idx[a]
                e2 = b_idx[b]
                r1, c1 = rows[e1], cols[e1]
                r2, c2 = rows[e2], cols[e2]
                db = dist_bin[r1, c1]
                ok = (
                    (a != b)
                    & (r1 != r2)
                    & (c1 != c2)
                    & allow[r1, c2]
                    & allow[r2, c1]
                    & (~adj[r1, c2])
                    & (~adj[r2, c1])
                    & (dist_bin[r1, c2] == db)
                    & (dist_bin[r2, c1] == db)
                )
                for ci in np.nonzero(ok)[0]:
                    i1 = e1[ci]
                    i2 = e2[ci]
                    rr1, cc1 = rows[i1], cols[i1]
                    rr2, cc2 = rows[i2], cols[i2]
                    if rr1 == rr2 or cc1 == cc2:
                        continue
                    if adj[rr1, cc2] or adj[rr2, cc1]:
                        continue
                    if not (allow[rr1, cc2] and allow[rr2, cc1]):
                        continue
                    db0 = dist_bin[rr1, cc1]
                    if dist_bin[rr1, cc2] != db0 or dist_bin[rr2, cc1] != db0:
                        continue
                    adj[rr1, cc1] = False
                    adj[rr2, cc2] = False
                    adj[rr1, cc2] = True
                    adj[rr2, cc1] = True
                    cols[i1] = cc2
                    cols[i2] = cc1
                    acc_b += 1
            accepted += acc_b
            if permute_weights_in_bucket:
                weights[b_idx] = rng.permutation(weights[b_idx])

        out = np.zeros_like(w, dtype=np.float32)
        out[rows, cols] = weights
        return out, accepted

    @staticmethod
    def _rewire_symmetric(
        w: np.ndarray,
        allow: np.ndarray,
        tier_id: np.ndarray,
        dist_bin: np.ndarray,
        rng: np.random.Generator,
        target_swaps_per_edge: float = 3.0,
        max_rounds: int = 35,
    ) -> Tuple[np.ndarray, int]:
        ru, cu = np.nonzero(np.triu(w, k=1))
        n_e = int(ru.size)
        if n_e < 2:
            return w.copy(), 0
        weights = w[ru, cu].copy()
        u = np.where(tier_id[ru] <= tier_id[cu], ru, cu).copy()
        v = np.where(tier_id[ru] <= tier_id[cu], cu, ru).copy()
        adj = (w > 0.0).copy()

        keys = tier_id[u] * 100 + tier_id[v] * 10 + dist_bin[u, v]
        uniq_k, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        buckets = [np.where(inv == i)[0] for i in range(uniq_k.size) if counts[i] >= 2]

        accepted = 0
        for b_idx in buckets:
            m = int(b_idx.size)
            target = int(m * target_swaps_per_edge)
            acc_b = 0
            same_tier = bool(tier_id[u[b_idx[0]]] == tier_id[v[b_idx[0]]])
            batch_sz = max(m * 4, 128)
            for _ in range(max_rounds):
                if acc_b >= target:
                    break
                a = rng.integers(0, m, size=batch_sz)
                b = rng.integers(0, m, size=batch_sz)
                e1 = b_idx[a]
                e2 = b_idx[b]
                u1, v1 = u[e1], v[e1]
                u2, v2 = u[e2], v[e2]
                if same_tier:
                    flip = rng.random(batch_sz) < 0.5
                    u2_new = np.where(flip, v2, u2)
                    v2_new = np.where(flip, u2, v2)
                    u2, v2 = u2_new, v2_new
                db = dist_bin[u1, v1]
                ok = (
                    (a != b)
                    & (u1 != u2)
                    & (u1 != v2)
                    & (v1 != u2)
                    & (v1 != v2)
                    & allow[u1, v2]
                    & allow[u2, v1]
                    & (~adj[u1, v2])
                    & (~adj[u2, v1])
                    & (dist_bin[u1, v2] == db)
                    & (dist_bin[u2, v1] == db)
                )
                for ci in np.nonzero(ok)[0]:
                    i1 = e1[ci]
                    i2 = e2[ci]
                    uu1, vv1 = u[i1], v[i1]
                    uu2, vv2 = u[i2], v[i2]
                    if same_tier and rng.random() < 0.5:
                        uu2, vv2 = vv2, uu2
                    if uu1 == uu2 or uu1 == vv2 or vv1 == uu2 or vv1 == vv2:
                        continue
                    if adj[uu1, vv2] or adj[uu2, vv1]:
                        continue
                    if not (allow[uu1, vv2] and allow[uu2, vv1]):
                        continue
                    db0 = dist_bin[uu1, vv1]
                    if dist_bin[uu1, vv2] != db0 or dist_bin[uu2, vv1] != db0:
                        continue
                    adj[uu1, vv1] = adj[vv1, uu1] = False
                    adj[uu2, vv2] = adj[vv2, uu2] = False
                    adj[uu1, vv2] = adj[vv2, uu1] = True
                    adj[uu2, vv1] = adj[vv1, uu2] = True
                    u[i1], v[i1] = uu1, vv2
                    u[i2], v[i2] = uu2, vv1
                    acc_b += 1
            accepted += acc_b

        out = np.zeros_like(w, dtype=np.float32)
        out[u, v] = weights
        out[v, u] = weights
        return out, accepted

    def generate_full_null_stack(self, seed: int) -> Tuple[np.ndarray, Dict[str, int]]:
        """Generate one full 8-layer constrained null connectome stack `[8, N, N]`."""
        rng = np.random.default_rng(int(seed))
        layers: List[np.ndarray] = []
        swaps: Dict[str, int] = {}
        for li, name in enumerate(LAYER_NAMES):
            w = self.real_stack[li]
            if name == "W_gap_herm":
                wn, acc = self._rewire_symmetric(
                    w, self.allow, self.tier_id, self.dist_bin, rng,
                    target_swaps_per_edge=3.0, max_rounds=35,
                )
            elif name in DENSE_WIRELESS_NAMES:
                wn, acc = self._rewire_directed(
                    w, self.allow, self.tier_id, self.dist_bin, rng,
                    target_swaps_per_edge=1.5, max_rounds=20,
                    permute_weights_in_bucket=True,
                )
            else:
                wn, acc = self._rewire_directed(
                    w, self.allow, self.tier_id, self.dist_bin, rng,
                    target_swaps_per_edge=2.5, max_rounds=30,
                    permute_weights_in_bucket=False,
                )
            layers.append(wn)
            swaps[name] = int(acc)
        return np.stack(layers, axis=0).astype(np.float32), swaps

    def generate_sign_permutation_stack(self, seed: int) -> np.ndarray:
        """Permute CeNGEN polarity labels (pos/neg/unk) on fixed chemical topology within (tier, tier, dist_bin)."""
        rng = np.random.default_rng(int(seed))
        stack = self.real_stack.copy()
        w_pos, w_neg, w_unk = self.real_stack[0], self.real_stack[1], self.real_stack[2]
        w_tot = self.chem_unsigned
        rows, cols = np.nonzero(w_tot)
        # Existing polarity label per edge: 0=pos, 1=neg, 2=unk
        labels = np.full(rows.size, 2, dtype=np.int32)
        labels[w_pos[rows, cols] > 0.0] = 0
        labels[w_neg[rows, cols] > 0.0] = 1

        keys = self.tier_id[rows] * 100 + self.tier_id[cols] * 10 + self.dist_bin[rows, cols]
        uniq_k, inv = np.unique(keys, return_inverse=True)
        perm_labels = labels.copy()
        for i in range(uniq_k.size):
            b_idx = np.where(inv == i)[0]
            if b_idx.size >= 2:
                perm_labels[b_idx] = rng.permutation(labels[b_idx])

        weights = w_tot[rows, cols]
        new_pos = np.zeros_like(w_pos)
        new_neg = np.zeros_like(w_neg)
        new_unk = np.zeros_like(w_unk)
        m0 = perm_labels == 0
        m1 = perm_labels == 1
        m2 = perm_labels == 2
        new_pos[rows[m0], cols[m0]] = weights[m0]
        new_neg[rows[m1], cols[m1]] = weights[m1]
        new_unk[rows[m2], cols[m2]] = weights[m2]
        stack[0] = new_pos
        stack[1] = new_neg
        stack[2] = new_unk
        return stack

    def generate_gap_only_null_stack(self, seed: int) -> np.ndarray:
        """Rewire ONLY W_gap_herm symmetrically (preserving degree, cell-type, dist-bin); keep other 7 layers fixed."""
        rng = np.random.default_rng(int(seed))
        stack = self.real_stack.copy()
        wn, _ = self._rewire_symmetric(
            self.real_stack[3], self.allow, self.tier_id, self.dist_bin, rng,
            target_swaps_per_edge=3.0, max_rounds=35,
        )
        stack[3] = wn
        return stack

    def verify_null_stack(self, null_stack: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute full constraint-preservation and randomization-fidelity metrics per layer."""
        out: Dict[str, Dict[str, float]] = {}
        n_tiers = len(self.tier_names)
        n_bins = self.n_dist_bins + 1
        for li, name in enumerate(LAYER_NAMES):
            wr = self.real_stack[li]
            wn = null_stack[li]
            er = wr > 0.0
            en = wn > 0.0
            jacc = float(np.sum(er & en) / max(1, int(np.sum(er | en))))
            union = (er | en) & self.allow
            sp_r = float(spearmanr(wr[union], wn[union]).statistic) if np.sum(union) >= 2 else 0.0
            if not np.isfinite(sp_r):
                sp_r = 0.0

            in_deg_r = _safe_corr(er.sum(axis=1), en.sum(axis=1))
            out_deg_r = _safe_corr(er.sum(axis=0), en.sum(axis=0))
            in_str_r = _safe_corr(wr.sum(axis=1), wn.sum(axis=1))
            out_str_r = _safe_corr(wr.sum(axis=0), wn.sum(axis=0))

            # Cell-type-pair block counts
            rr, cr = np.nonzero(er)
            rn, cn = np.nonzero(en)
            ct_r = np.bincount(self.tier_id[rr] * n_tiers + self.tier_id[cr], minlength=n_tiers * n_tiers)
            ct_n = np.bincount(self.tier_id[rn] * n_tiers + self.tier_id[cn], minlength=n_tiers * n_tiers)
            ct_exact = float(1.0 - np.sum(np.abs(ct_r - ct_n)) / max(1, int(np.sum(ct_r))))

            # Distance-bin edge counts
            db_r = np.bincount(self.dist_bin[rr, cr], minlength=n_bins)
            db_n = np.bincount(self.dist_bin[rn, cn], minlength=n_bins)
            db_exact = float(1.0 - np.sum(np.abs(db_r - db_n)) / max(1, int(np.sum(db_r))))

            diag_max = float(np.max(np.abs(np.diag(wn))))
            disallowed_max = float(np.max(np.abs(wn[~self.allow])))
            sym_err = float(np.max(np.abs(wn - wn.T))) if name == "W_gap_herm" else 0.0

            out[name] = {
                "edge_jaccard_vs_real": jacc,
                "weight_spearman_vs_real": sp_r,
                "in_degree_corr": in_deg_r,
                "out_degree_corr": out_deg_r,
                "in_strength_corr": in_str_r,
                "out_strength_corr": out_str_r,
                "cell_type_block_match": ct_exact,
                "distance_bin_match": db_exact,
                "diag_max_abs": diag_max,
                "disallowed_max_abs": disallowed_max,
                "symmetry_max_abs": sym_err,
            }
        return out


# ---------------------------------------------------------------------------
# 2. Precomputed Graph-Independent Fold Cache (Exact Anchor & Presynaptic State)
# ---------------------------------------------------------------------------
def _extract_u_stim_for_trial(
    trial: Mapping[str, Any],
    s_n: np.ndarray,
    priors: Mapping[str, Any],
    n_canon: int,
) -> np.ndarray:
    stim_id = str(trial["stimulus_id"])
    is_buf = bool(trial.get("is_buffer_control", False))
    stim_oh_nonbuf = _encode_stim_onehot(stim_id, is_buf, priors["stim_vocab"])[:-1]
    stim_mat = np.repeat(stim_oh_nonbuf[None, :], n_canon, axis=0)
    _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, n_canon)
    prior_norm_2 = np.clip((prior_coef / s_n[:, None].astype(np.float32))[:, :2], -4.0, 4.0)
    return np.hstack([stim_mat, prior_norm_2]).astype(np.float32)


def _compute_all_layer_drives_batched(
    w_stack: np.ndarray,
    sm_cat: np.ndarray,
    n_trials: int,
    kappa: float,
) -> np.ndarray:
    """Compute missingness-precision-weighted drives for all 8 layers & all trials via BLAS.

    w_stack: [8, N, N]
    sm_cat:  [N, n_trials * 4]
    Returns: [n_trials, N, 32] (8 layers x 4 multiscale features).
    """
    n_layers, n_canon, _ = w_stack.shape
    out = np.empty((n_trials, n_canon, n_layers * 4), dtype=np.float32)
    for li in range(n_layers):
        w = w_stack[li]
        denom = np.maximum(np.sum(w, axis=1, dtype=np.float32) + float(kappa), 1e-6)[:, None]
        num = w @ sm_cat  # [N, n_trials * 4]
        drv = np.clip(num / denom, -6.0, 6.0)
        # Reshape [N, n_trials, 4] -> [n_trials, N, 4]
        out[:, :, li * 4 : (li + 1) * 4] = drv.reshape(n_canon, n_trials, 4).transpose(1, 0, 2)
    return out


def build_dataset_ridge_fold_caches(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
) -> Dict[str, Any]:
    """Precompute graph-independent M3R1_02 anchor, residuals, and presynaptic states ONCE per dataset."""
    aligned, canon, tiers, _, splits = _prepare_dataset_folds(dataset_id, data_root, exp_root)
    n_canon = len(canon)
    hp = cfg["hyperparameters"]
    fold_caches: List[Dict[str, Any]] = []
    val_trials_all: List[Dict[str, Any]] = []
    val_sn_all: List[np.ndarray] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
        all_fold_trials = train_trials + val_trials
        fc = _build_m3_fold_cache(all_fold_trials, train_trials, s_n, n_canon, cfg)

        anchor = M3R1Model("M3R1_02_h1_confounds_only", cfg, canon, tiers)
        anchor.fit(train_trials, s_n, fold_cache=fc)
        priors = anchor.state["priors"]
        mu_train = priors["mu_train"]
        non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        sm_tr_list: List[np.ndarray] = []
        tr_t_idx: List[np.ndarray] = []
        tr_n_idx: List[np.ndarray] = []
        u_tr_rows: List[np.ndarray] = []
        y_tr_rows: List[np.ndarray] = []

        for ti, tr in enumerate(non_buf):
            tid = str(tr.get("trial_id", ""))
            if tid in fc.get("proj_targets", {}):
                c_norm, rval, _, _ = fc["proj_targets"][tid]
            else:
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors, s_n, n_canon)
            p1 = anchor._predict_stage1_coef(tr, s_n, priors, anchor.state["s1"], fold_cache=fc)
            d2 = anchor._predict_stage2_delta_coef(tr, s_n, priors, anchor.state["s2"])
            r2 = (c_norm - (p1 + d2)).astype(np.float32)

            s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
            sm_tr_list.append((s_mat * m_obs[:, None]).astype(np.float32))
            idx = np.where(rval)[0]
            if idx.size > 0:
                tr_t_idx.append(np.full(idx.size, ti, dtype=np.int32))
                tr_n_idx.append(idx.astype(np.int32))
                u_tr_rows.append(_extract_u_stim_for_trial(tr, s_n, priors, n_canon)[idx])
                y_tr_rows.append(r2[idx])

        sm_va_list: List[np.ndarray] = []
        u_va_list: List[np.ndarray] = []
        anchor_c_va_list: List[np.ndarray] = []
        prior_delta_va_list: List[np.ndarray] = []
        phi_va_list: List[np.ndarray] = []
        for tr in val_trials:
            tid = str(tr.get("trial_id", ""))
            if tid in fc.get("proj_targets", {}):
                _, _, prior_delta, phi = fc["proj_targets"][tid]
            else:
                prior_delta, _, phi = predict_prior_delta_trajectory(tr, priors, n_canon)
            pred_c = anchor._predict_stage1_coef(tr, s_n, priors, anchor.state["s1"], fold_cache=fc)
            pred_c = (pred_c + anchor._predict_stage2_delta_coef(tr, s_n, priors, anchor.state["s2"])).astype(np.float32)
            s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
            sm_va_list.append((s_mat * m_obs[:, None]).astype(np.float32))
            u_va_list.append(_extract_u_stim_for_trial(tr, s_n, priors, n_canon))
            anchor_c_va_list.append(pred_c)
            prior_delta_va_list.append(prior_delta.astype(np.float32))
            phi_va_list.append(phi.astype(np.float32))

        fold_caches.append({
            "fold_i": fold_i,
            "s_n": np.asarray(s_n, dtype=np.float32),
            "mu_train": np.asarray(mu_train, dtype=np.float32),
            "n_tr": len(non_buf),
            "sm_tr_cat": np.hstack(sm_tr_list).astype(np.float32),
            "tr_t_idx": np.concatenate(tr_t_idx),
            "tr_n_idx": np.concatenate(tr_n_idx),
            "u_tr": np.vstack(u_tr_rows).astype(np.float32),
            "y_tr": np.vstack(y_tr_rows).astype(np.float32),
            "n_va": len(val_trials),
            "sm_va_cat": np.hstack(sm_va_list).astype(np.float32),
            "u_va_list": u_va_list,
            "anchor_c_va_list": anchor_c_va_list,
            "prior_delta_va_list": prior_delta_va_list,
            "phi_va_list": phi_va_list,
            "val_trials": val_trials,
        })
        val_trials_all.extend(val_trials)
        val_sn_all.extend([s_n] * len(val_trials))
        del fc, anchor, train_trials
        gc.collect()

    del aligned
    gc.collect()
    return {
        "dataset_id": dataset_id,
        "canonical_neurons": canon,
        "functional_tiers": tiers,
        "fold_caches": fold_caches,
        "val_trials_all": val_trials_all,
        "val_sn_all": val_sn_all,
    }


def retrain_and_eval_ridge_on_graph(
    ds_cache: Mapping[str, Any],
    w_stack: np.ndarray,
    hp: Mapping[str, Any],
) -> Dict[str, Any]:
    """Genuinely retrain M3R3_03_herm_ridge from scratch inside EVERY CV fold on `w_stack`."""
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    conn_l2 = float(hp.get("connectome_l2_penalty", 60.0))
    wl_rank = int(hp.get("wireless_rank", 6))
    wl_mult = float(hp.get("wireless_l2_multiplier", 2.5))
    bilin_rank = int(hp.get("bilinear_rank", 4))

    preds_all: List[np.ndarray] = []
    for fc in ds_cache["fold_caches"]:
        # 1. Recompute incoming connectome drives from `w_stack`
        drv_tr = _compute_all_layer_drives_batched(w_stack, fc["sm_tr_cat"], fc["n_tr"], kappa)
        tr_all = drv_tr[fc["tr_t_idx"], fc["tr_n_idx"], :]  # [M_obs, 32]
        dw_all = tr_all[:, :20]
        dwl_all = tr_all[:, 20:]
        u_all = fc["u_tr"]
        y_all = fc["y_tr"]

        # 2. Fold-internal standardization + wireless SVD compression
        w_mu = np.mean(dw_all, axis=0, keepdims=True).astype(np.float32)
        w_sd = np.maximum(np.std(dw_all, axis=0, keepdims=True), 1e-4).astype(np.float32)
        dw_n = np.clip((dw_all - w_mu) / w_sd, -5.0, 5.0).astype(np.float32)

        wl_k = min(wl_rank, dwl_all.shape[1])
        wl_comp = _fit_svd_compressor(dwl_all, n_comp=wl_k)
        dwl_low = _apply_svd_compressor(dwl_all, wl_comp)
        d_lin = np.hstack([dw_n, dwl_low]).astype(np.float32)

        col_l2 = np.concatenate([
            np.full(dw_n.shape[1], conn_l2, dtype=np.float64),
            np.full(dwl_low.shape[1], conn_l2 * wl_mult, dtype=np.float64),
        ])
        eff_col_l2 = col_l2 * max(1.0, float(d_lin.shape[1]) / 12.0)

        # 3. Stage 3a: Closed-form Ridge retrained from scratch
        w_lin = _solve_weighted_ridge_sample_scaled(d_lin, y_all, eff_col_l2)
        y_res_lin = (y_all - d_lin @ w_lin).astype(np.float32)

        # 4. Stage 3b: Reduced-Rank Bilinear Ridge retrained from scratch
        p_c = min(4, max(1, d_lin.shape[1]))
        c_comp = _fit_svd_compressor(d_lin, n_comp=p_c)
        c_low = _apply_svd_compressor(d_lin, c_comp)
        x_bilin = _build_bilinear_features(c_low, u_all)
        xb_mu = np.mean(x_bilin, axis=0, keepdims=True).astype(np.float32)
        xb_sd = np.maximum(np.std(x_bilin, axis=0, keepdims=True), 1e-4).astype(np.float32)
        xb_n = np.clip((x_bilin - xb_mu) / xb_sd, -5.0, 5.0).astype(np.float32)
        bilin_eff_l2 = conn_l2 * max(1.0, float(xb_n.shape[1]) / 12.0) * 1.5
        rrr = _fit_reduced_rank_ridge(xb_n, y_res_lin, bilin_eff_l2, rank=bilin_rank)
        w_rrr = rrr["w"]

        # 5. Predict validation trials for this fold
        drv_va = _compute_all_layer_drives_batched(w_stack, fc["sm_va_cat"], fc["n_va"], kappa)
        s_n = fc["s_n"]
        mu_train = fc["mu_train"]
        for vi, tr in enumerate(fc["val_trials"]):
            d_w = drv_va[vi, :, :20]
            d_wl = drv_va[vi, :, 20:]
            dw_n_v = np.clip((d_w - w_mu) / w_sd, -5.0, 5.0).astype(np.float32)
            dwl_low_v = _apply_svd_compressor(d_wl, wl_comp)
            d_lin_v = np.hstack([dw_n_v, dwl_low_v]).astype(np.float32)
            delta_lin = (d_lin_v @ w_lin).astype(np.float32)
            c_low_v = _apply_svd_compressor(d_lin_v, c_comp)
            xb_v = _build_bilinear_features(c_low_v, fc["u_va_list"][vi])
            xb_n_v = np.clip((xb_v - xb_mu) / xb_sd, -5.0, 5.0).astype(np.float32)
            delta_bilin = (xb_n_v @ w_rrr).astype(np.float32)
            pred_c = fc["anchor_c_va_list"][vi] + delta_lin + delta_bilin
            preds_all.append(
                _reconstruct_y_pred_from_coef(
                    tr, fc["prior_delta_va_list"][vi], fc["phi_va_list"][vi], pred_c, s_n, mu_train
                )
            )

    ev = evaluate_predictions(
        ds_cache["val_trials_all"],
        preds_all,
        ds_cache["val_sn_all"],
        functional_tiers=ds_cache["functional_tiers"],
        compute_bootstrap=False,
        seed=PRIMARY_SEED,
    )
    return {"E_d": float(ev["E_d"]), "post_R2": float(ev["post_R2"])}


# ---------------------------------------------------------------------------
# 3. Precomputed Graph-Independent Fold Cache for FiLM (`M3R4_01`)
# ---------------------------------------------------------------------------
def build_dataset_film_fold_caches(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
) -> Dict[str, Any]:
    """Precompute graph-independent outer & inner M3R1_02 anchors, slow states z, and targets for FiLM."""
    aligned, canon, tiers, _, splits = _prepare_dataset_folds(dataset_id, data_root, exp_root)
    n_canon = len(canon)
    hp = cfg["hyperparameters"]
    rank = int(hp.get("rank", 6))
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    tier_names = ("sensory", "interneuron", "premotor_command", "motor", "pharyngeal")
    tier_oh = np.zeros((n_canon, len(tier_names)), dtype=np.float32)
    for i, t_str in enumerate(tiers):
        if t_str in tier_names:
            tier_oh[i, tier_names.index(t_str)] = 1.0

    fold_caches: List[Dict[str, Any]] = []
    val_trials_all: List[Dict[str, Any]] = []
    val_sn_all: List[np.ndarray] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
        all_fold_trials = train_trials + val_trials
        fc = _build_m3_fold_cache(all_fold_trials, train_trials, s_n, n_canon, cfg)

        anchor = M3R1Model("M3R1_02_h1_confounds_only", cfg, canon, tiers)
        anchor.fit(train_trials, s_n, fold_cache=fc)
        priors = anchor.state["priors"]
        mu_train = priors["mu_train"]
        b_dim = int(priors["b_dim"])
        non_buf = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        pca_state = fit_causal_slow_state_pca(
            non_buf, s_n, mu_train, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

        # Outer training arrays (graph-independent)
        sm_out, t_out_i, n_out_i, z_out, u_out, y_out = [], [], [], [], [], []
        for ti, tr in enumerate(non_buf):
            tid = str(tr.get("trial_id", ""))
            if tid in fc.get("proj_targets", {}):
                c_norm, rval, _, _ = fc["proj_targets"][tid]
            else:
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, priors, s_n, n_canon)
            p1 = anchor._predict_stage1_coef(tr, s_n, priors, anchor.state["s1"], fold_cache=fc)
            d2 = anchor._predict_stage2_delta_coef(tr, s_n, priors, anchor.state["s2"])
            r2 = (c_norm - (p1 + d2)).astype(np.float32)
            z_circ = extract_circuit_slow_state_features(tr, canon, s_n, mu_train, pca_state, kappa=kappa)
            s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
            sm_out.append((s_mat * m_obs[:, None]).astype(np.float32))
            idx = np.where(rval)[0]
            if idx.size > 0:
                t_out_i.append(np.full(idx.size, ti, dtype=np.int32))
                n_out_i.append(idx.astype(np.int32))
                z_out.append(z_circ[idx])
                u_out.append(_extract_u_stim_for_trial(tr, s_n, priors, n_canon)[idx])
                y_out.append(r2[idx])

        z_all = np.vstack(z_out).astype(np.float32)
        outer_z_comp = _fit_svd_compressor(z_all, n_comp=6)
        z_all_low = _apply_svd_compressor(z_all, outer_z_comp)
        n_all = np.concatenate(n_out_i).astype(np.int32)

        # Inner animal-grouped split for zero-leakage early stopping (exact match to M3R4Model)
        worms = sorted({str(t["animal_id"]) for t in non_buf})
        n_val_w = max(1, len(worms) // 5)
        val_worms = set(worms[-n_val_w:])
        inner_tr = [t for t in non_buf if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in non_buf if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(non_buf)
            inner_va = list(non_buf[:1])

        in_anchor = M3R1Model("M3R1_02_h1_confounds_only", cfg, canon, tiers)
        in_anchor.fit(inner_tr, s_n)
        in_priors = in_anchor.state["priors"]
        in_mu = in_priors["mu_train"]
        in_pca = fit_causal_slow_state_pca(
            inner_tr, s_n, in_mu, n_canon, n_comp=rank, n_slope_comp=max(2, rank // 2)
        )

        def _collect_inner_base(trials_sub: Sequence[Mapping[str, Any]]):
            sm_l, ti_l, ni_l, z_l, u_l, y_l, t_l = [], [], [], [], [], [], []
            for ti, tr in enumerate(trials_sub):
                c_norm, rval, _, _ = _project_trial_residual_targets(tr, in_priors, s_n, n_canon)
                p1 = in_anchor._predict_stage1_coef(tr, s_n, in_priors, in_anchor.state["s1"])
                d2 = in_anchor._predict_stage2_delta_coef(tr, s_n, in_priors, in_anchor.state["s2"])
                r2 = (c_norm - (p1 + d2)).astype(np.float32)
                z_circ = extract_circuit_slow_state_features(tr, canon, s_n, in_mu, in_pca, kappa=kappa)
                s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, in_mu)
                sm_l.append((s_mat * m_obs[:, None]).astype(np.float32))
                idx = np.where(rval)[0]
                if idx.size > 0:
                    ti_l.append(np.full(idx.size, ti, dtype=np.int32))
                    ni_l.append(idx.astype(np.int32))
                    z_l.append(z_circ[idx])
                    u_l.append(_extract_u_stim_for_trial(tr, s_n, in_priors, n_canon)[idx])
                    y_l.append(r2[idx])
                    t_l.append(tier_oh[idx])
            return (
                len(trials_sub),
                np.hstack(sm_l).astype(np.float32),
                np.concatenate(ti_l),
                np.concatenate(ni_l),
                np.vstack(z_l).astype(np.float32),
                np.vstack(u_l).astype(np.float32),
                np.vstack(y_l).astype(np.float32),
                np.vstack(t_l).astype(np.float32),
            )

        n_itr, sm_itr, ti_itr, ni_itr, z_itr, u_itr, y_itr, t_itr = _collect_inner_base(inner_tr)
        n_iva, sm_iva, ti_iva, ni_iva, z_iva, u_iva, y_iva, t_iva = _collect_inner_base(inner_va)
        in_z_comp = _fit_svd_compressor(z_itr, n_comp=6)
        z_itr_low = _apply_svd_compressor(z_itr, in_z_comp)
        z_iva_low = _apply_svd_compressor(z_iva, in_z_comp)

        # Outer validation trials (graph-independent)
        sm_va, u_va, z_va_low, anc_va, pdel_va, phi_va = [], [], [], [], [], []
        for tr in val_trials:
            tid = str(tr.get("trial_id", ""))
            if tid in fc.get("proj_targets", {}):
                _, _, prior_delta, phi = fc["proj_targets"][tid]
            else:
                prior_delta, _, phi = predict_prior_delta_trajectory(tr, priors, n_canon)
            pred_c = anchor._predict_stage1_coef(tr, s_n, priors, anchor.state["s1"], fold_cache=fc)
            pred_c = (pred_c + anchor._predict_stage2_delta_coef(tr, s_n, priors, anchor.state["s2"])).astype(np.float32)
            s_mat, m_obs = extract_multiscale_presynaptic_state(tr, s_n, mu_train)
            z_circ = extract_circuit_slow_state_features(tr, canon, s_n, mu_train, pca_state, kappa=kappa)
            sm_va.append((s_mat * m_obs[:, None]).astype(np.float32))
            u_va.append(_extract_u_stim_for_trial(tr, s_n, priors, n_canon))
            z_va_low.append(_apply_svd_compressor(z_circ, outer_z_comp))
            anc_va.append(pred_c)
            pdel_va.append(prior_delta.astype(np.float32))
            phi_va.append(phi.astype(np.float32))

        fold_caches.append({
            "s_n": np.asarray(s_n, dtype=np.float32),
            "mu_train": np.asarray(mu_train, dtype=np.float32),
            "b_dim": b_dim,
            "n_out": len(non_buf),
            "sm_out_cat": np.hstack(sm_out).astype(np.float32),
            "t_out_i": np.concatenate(t_out_i),
            "n_out_i": n_all,
            "z_all_low": z_all_low,
            "u_all": np.vstack(u_out).astype(np.float32),
            "t_full": tier_oh[n_all],
            "y_all": np.vstack(y_out).astype(np.float32),
            "n_itr": n_itr, "sm_itr_cat": sm_itr, "ti_itr": ti_itr, "ni_itr": ni_itr,
            "z_itr_low": z_itr_low, "u_itr": u_itr, "t_itr": t_itr, "y_itr": y_itr,
            "n_iva": n_iva, "sm_iva_cat": sm_iva, "ti_iva": ti_iva, "ni_iva": ni_iva,
            "z_iva_low": z_iva_low, "u_iva": u_iva, "t_iva": t_iva, "y_iva": y_iva,
            "n_va": len(val_trials), "sm_va_cat": np.hstack(sm_va).astype(np.float32),
            "u_va_list": u_va, "z_va_low_list": z_va_low, "anc_va_list": anc_va,
            "pdel_va_list": pdel_va, "phi_va_list": phi_va,
            "val_trials": val_trials,
        })
        val_trials_all.extend(val_trials)
        val_sn_all.extend([s_n] * len(val_trials))
        del fc, anchor, in_anchor, train_trials
        gc.collect()

    del aligned
    gc.collect()
    return {
        "dataset_id": dataset_id,
        "canonical_neurons": canon,
        "functional_tiers": tiers,
        "tier_oh": tier_oh,
        "fold_caches": fold_caches,
        "val_trials_all": val_trials_all,
        "val_sn_all": val_sn_all,
    }


def retrain_and_eval_film_on_graph(
    ds_cache: Mapping[str, Any],
    w_stack: np.ndarray,
    hp: Mapping[str, Any],
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Genuinely retrain M3R4_01 PyTorch ResidualFiLMGate from scratch inside EVERY CV fold on `w_stack`."""
    kappa = float(hp.get("missingness_precision_kappa", 4.0))
    wl_rank = int(hp.get("wireless_rank", 6))
    film_dim = int(hp.get("poco_film_dim", 24))
    drop = float(hp.get("dropout", 0.15))
    lr = float(hp.get("learning_rate", 0.003))
    wd = float(hp.get("weight_decay", 0.01)) * 2.0
    max_ep = int(hp.get("max_epochs", 18))
    patience = int(hp.get("early_stopping_patience", 4))
    bsz = int(hp.get("batch_size", 2048))
    tier_oh = ds_cache["tier_oh"]
    p_c, p_z = 6, 6

    preds_all: List[np.ndarray] = []
    for fc in ds_cache["fold_caches"]:
        b_dim = int(fc["b_dim"])
        # Inner drives & SVD compressors
        drv_itr = _compute_all_layer_drives_batched(w_stack, fc["sm_itr_cat"], fc["n_itr"], kappa)
        itr_obs = drv_itr[fc["ti_itr"], fc["ni_itr"], :]
        cw_tr, cwl_tr = itr_obs[:, :20], itr_obs[:, 20:]

        drv_iva = _compute_all_layer_drives_batched(w_stack, fc["sm_iva_cat"], fc["n_iva"], kappa)
        iva_obs = drv_iva[fc["ti_iva"], fc["ni_iva"], :]
        cw_va, cwl_va = iva_obs[:, :20], iva_obs[:, 20:]

        in_wl_comp = _fit_svd_compressor(cwl_tr, n_comp=min(wl_rank, cwl_tr.shape[1]))
        c_tr_full = np.hstack([cw_tr, _apply_svd_compressor(cwl_tr, in_wl_comp)])
        c_va_full = np.hstack([cw_va, _apply_svd_compressor(cwl_va, in_wl_comp)])
        in_c_comp = _fit_svd_compressor(c_tr_full, n_comp=p_c)

        s_tr = np.hstack([_apply_svd_compressor(c_tr_full, in_c_comp), fc["z_itr_low"]]).astype(np.float32)
        s_va = np.hstack([_apply_svd_compressor(c_va_full, in_c_comp), fc["z_iva_low"]]).astype(np.float32)

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        st_t, ut_t, tt_t, yt_t = (
            torch.from_numpy(s_tr), torch.from_numpy(fc["u_itr"]),
            torch.from_numpy(fc["t_itr"]), torch.from_numpy(fc["y_itr"]),
        )
        sv_t, uv_t, tv_t, yv_t = (
            torch.from_numpy(s_va), torch.from_numpy(fc["u_iva"]),
            torch.from_numpy(fc["t_iva"]), torch.from_numpy(fc["y_iva"]),
        )
        probe = ResidualFiLMGate(p_c + p_z, ut_t.shape[1], tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

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

        # Outer refit on full training fold for `best_ep` epochs
        drv_out = _compute_all_layer_drives_batched(w_stack, fc["sm_out_cat"], fc["n_out"], kappa)
        out_obs = drv_out[fc["t_out_i"], fc["n_out_i"], :]
        dw_all, dwl_all = out_obs[:, :20], out_obs[:, 20:]
        outer_wl_comp = _fit_svd_compressor(dwl_all, n_comp=min(wl_rank, dwl_all.shape[1]))
        c_full = np.hstack([dw_all, _apply_svd_compressor(dwl_all, outer_wl_comp)])
        outer_c_comp = _fit_svd_compressor(c_full, n_comp=p_c)
        s_full = np.hstack([_apply_svd_compressor(c_full, outer_c_comp), fc["z_all_low"]]).astype(np.float32)

        torch.manual_seed(int(seed))
        sf_t, uf_t, tf_t, yf_t = (
            torch.from_numpy(s_full), torch.from_numpy(fc["u_all"]),
            torch.from_numpy(fc["t_full"]), torch.from_numpy(fc["y_all"]),
        )
        final_net = ResidualFiLMGate(p_c + p_z, uf_t.shape[1], tier_oh.shape[1], b_dim, film_dim=film_dim, dropout=drop)
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
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

        # Predict validation trials
        drv_va = _compute_all_layer_drives_batched(w_stack, fc["sm_va_cat"], fc["n_va"], kappa)
        s_n = fc["s_n"]
        mu_train = fc["mu_train"]
        t_va_t = torch.from_numpy(tier_oh)
        with torch.no_grad():
            for vi, tr in enumerate(fc["val_trials"]):
                d_w, d_wl = drv_va[vi, :, :20], drv_va[vi, :, 20:]
                c_v = np.hstack([d_w, _apply_svd_compressor(d_wl, outer_wl_comp)])
                s_vec = np.hstack([_apply_svd_compressor(c_v, outer_c_comp), fc["z_va_low_list"][vi]]).astype(np.float32)
                delta_film = final_net(
                    torch.from_numpy(s_vec),
                    torch.from_numpy(fc["u_va_list"][vi]),
                    t_va_t,
                ).numpy().astype(np.float32)
                pred_c = fc["anc_va_list"][vi] + delta_film
                preds_all.append(
                    _reconstruct_y_pred_from_coef(
                        tr, fc["pdel_va_list"][vi], fc["phi_va_list"][vi], pred_c, s_n, mu_train
                    )
                )

    ev = evaluate_predictions(
        ds_cache["val_trials_all"],
        preds_all,
        ds_cache["val_sn_all"],
        functional_tiers=ds_cache["functional_tiers"],
        compute_bootstrap=False,
        seed=PRIMARY_SEED,
    )
    return {"E_d": float(ev["E_d"]), "post_R2": float(ev["post_R2"])}


# ---------------------------------------------------------------------------
# 4. Null-Bank Caching (float16 memmap in scratch dir, zero redundant rewiring)
# ---------------------------------------------------------------------------
def ensure_null_bank(
    gen: ConstrainedNullConnectomeGenerator,
    k_null: int,
    scratch_dir: Path,
    resume: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    bank_path = scratch_dir / f"null_bank_k{k_null}.npy"
    stats_path = scratch_dir / f"null_bank_k{k_null}_constraints.json"
    if resume and bank_path.exists() and stats_path.exists():
        print(f"[M4:nulls] Reusing cached {k_null}-graph constrained null bank at {bank_path}", flush=True)
        return bank_path, json.loads(stats_path.read_text(encoding="utf-8"))

    t0 = time.time()
    n = gen.n_canon
    mm = np.lib.format.open_memmap(bank_path, mode="w+", dtype=np.float16, shape=(k_null, len(LAYER_NAMES), n, n))
    per_layer_accum: Dict[str, Dict[str, List[float]]] = {
        name: {
            "edge_jaccard_vs_real": [],
            "weight_spearman_vs_real": [],
            "in_degree_corr": [],
            "out_degree_corr": [],
            "in_strength_corr": [],
            "out_strength_corr": [],
            "cell_type_block_match": [],
            "distance_bin_match": [],
            "diag_max_abs": [],
            "disallowed_max_abs": [],
            "symmetry_max_abs": [],
        }
        for name in LAYER_NAMES
    }

    for k in range(1, k_null + 1):
        seed_k = PRIMARY_SEED + 1000 * k
        null_stack, _ = gen.generate_full_null_stack(seed_k)
        mm[k - 1] = null_stack.astype(np.float16)
        v = gen.verify_null_stack(null_stack)
        for name in LAYER_NAMES:
            for mk, mv in v[name].items():
                per_layer_accum[name][mk].append(float(mv))
        if k % 50 == 0 or k == k_null:
            print(
                f"[M4:nulls] Generated & verified {k}/{k_null} constrained null graphs "
                f"in {time.time() - t0:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                flush=True,
            )

    mm.flush()
    del mm

    summary_layers: Dict[str, Dict[str, Any]] = {}
    for name in LAYER_NAMES:
        d = per_layer_accum[name]
        summary_layers[name] = {
            "edge_jaccard_mean": round(float(np.mean(d["edge_jaccard_vs_real"])), 4),
            "edge_jaccard_std": round(float(np.std(d["edge_jaccard_vs_real"])), 4),
            "weight_spearman_mean": round(float(np.mean(d["weight_spearman_vs_real"])), 4),
            "weight_spearman_std": round(float(np.std(d["weight_spearman_vs_real"])), 4),
            "in_degree_corr_min": round(float(np.min(d["in_degree_corr"])), 6),
            "out_degree_corr_min": round(float(np.min(d["out_degree_corr"])), 6),
            "in_strength_corr_mean": round(float(np.mean(d["in_strength_corr"])), 4),
            "out_strength_corr_mean": round(float(np.mean(d["out_strength_corr"])), 4),
            "cell_type_block_match_min": round(float(np.min(d["cell_type_block_match"])), 6),
            "distance_bin_match_min": round(float(np.min(d["distance_bin_match"])), 6),
            "diag_max_abs": float(np.max(d["diag_max_abs"])),
            "disallowed_max_abs": float(np.max(d["disallowed_max_abs"])),
            "symmetry_max_abs": float(np.max(d["symmetry_max_abs"])),
        }

    constraint_report = {
        "k_null": k_null,
        "n_dist_bins": gen.n_dist_bins,
        "dist_bin_edges_um": [round(float(x), 4) for x in gen.dist_bin_edges],
        "tier_names": gen.tier_names,
        "generation_wall_sec": round(time.time() - t0, 2),
        "all_in_degree_corr_ge_099": all(v["in_degree_corr_min"] >= 0.99 for v in summary_layers.values()),
        "all_out_degree_corr_ge_099": all(v["out_degree_corr_min"] >= 0.99 for v in summary_layers.values()),
        "all_cell_type_exact_100pct": all(v["cell_type_block_match_min"] >= 0.999999 for v in summary_layers.values()),
        "all_dist_bin_exact_100pct": all(v["distance_bin_match_min"] >= 0.999999 for v in summary_layers.values()),
        "all_zero_diag_and_bilateral": all(
            v["diag_max_abs"] == 0.0 and v["disallowed_max_abs"] == 0.0 for v in summary_layers.values()
        ),
        "gap_exact_symmetry": summary_layers["W_gap_herm"]["symmetry_max_abs"] == 0.0,
        "per_layer": summary_layers,
    }
    stats_path.write_text(json.dumps(constraint_report, indent=2), encoding="utf-8")
    return bank_path, constraint_report


# ---------------------------------------------------------------------------
# 5. Stages: benchmark, ridge199, film_null, summarize
# ---------------------------------------------------------------------------
def run_stage_benchmark(data_root: Path, exp_root: Path) -> Dict[str, Any]:
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]
    t0 = time.time()

    # 1. Build yemini cache & verify k=0 real graph matches saved r3_03 predictions
    ds_y = build_dataset_ridge_fold_caches("yemini2021", cfg, data_root, exp_root)
    gen = ConstrainedNullConnectomeGenerator(ds_y["canonical_neurons"], data_root=data_root)
    real_y = retrain_and_eval_ridge_on_graph(ds_y, gen.real_stack, hp)

    r3_03_pred_path = exp_root / "runs" / "r3_03_m3r3_03_h3_multilayer_wired_plus_wireless_npp" / "predictions_dev_yemini2021.npz"
    saved_y = score_saved_predictions_file(
        r3_03_pred_path, split="dev", data_root=data_root, exp_root=exp_root, resolve_registry_ref=False
    )
    diff_y = abs(real_y["E_d"] - float(saved_y["E_d"]))
    print(
        f"[M4:bench] Yemini k=0 real E_d={real_y['E_d']:.6f} vs saved M3R3_03 E_d={saved_y['E_d']:.6f} "
        f"(abs_diff={diff_y:.2e})",
        flush=True,
    )
    if diff_y > 1e-5:
        raise AssertionError(f"Vectorized M3R3_03 k=0 mismatch on yemini2021: diff={diff_y}")

    # 2. Time 1 null generation + 1 ridge retraining on yemini
    t_gen = time.time()
    null1, _ = gen.generate_full_null_stack(PRIMARY_SEED + 1000)
    dt_gen = time.time() - t_gen
    ver = gen.verify_null_stack(null1)
    t_r = time.time()
    null_y = retrain_and_eval_ridge_on_graph(ds_y, null1, hp)
    dt_r_y = time.time() - t_r
    print(
        f"[M4:bench] 1 null gen={dt_gen:.3f}s | Yemini 1 ridge null={dt_r_y:.3f}s (E_d={null_y['E_d']:.6f}) | "
        f"gap_jacc={ver['W_gap_herm']['edge_jaccard_vs_real']:.3f} | "
        f"npp_short_sp={ver['W_npp_short']['weight_spearman_vs_real']:.3f}",
        flush=True,
    )
    del ds_y
    gc.collect()

    # 3. Time 1 FiLM retraining on seyedolmohadesin2025 + toyoshima2024
    film_times: Dict[str, float] = {}
    for d in DATASETS_MAIN2:
        fc_d = build_dataset_film_fold_caches(d, cfg, data_root, exp_root)
        t_f0 = time.time()
        res_real = retrain_and_eval_film_on_graph(fc_d, gen.real_stack, hp, seed=PRIMARY_SEED)
        t_f1 = time.time()
        res_null = retrain_and_eval_film_on_graph(fc_d, null1, hp, seed=PRIMARY_SEED)
        dt_f = time.time() - t_f1
        film_times[d] = round(dt_f, 3)
        print(
            f"[M4:bench] FiLM {d}: real E_d={res_real['E_d']:.6f} ({t_f1 - t_f0:.2f}s) | "
            f"null1 E_d={res_null['E_d']:.6f} ({dt_f:.2f}s) | RSS={_peak_rss_mb():.1f} MB",
            flush=True,
        )
        del fc_d
        gc.collect()

    est_19_film_sec = 19.0 * sum(film_times.values())
    est_39_film_sec = 39.0 * sum(film_times.values())
    print(
        f"[M4:bench] Est 19 FiLM nulls (Main 2): {est_19_film_sec:.1f}s | "
        f"Est 39 FiLM nulls (Main 2): {est_39_film_sec:.1f}s | Total bench: {time.time() - t0:.1f}s",
        flush=True,
    )
    return {
        "yemini_k0_diff": diff_y,
        "dt_null_gen_sec": round(dt_gen, 3),
        "dt_ridge_yemini_sec": round(dt_r_y, 3),
        "dt_film_main2_sec": film_times,
        "est_19_film_sec": round(est_19_film_sec, 1),
        "est_39_film_sec": round(est_39_film_sec, 1),
    }


def run_stage_ridge199(
    data_root: Path,
    exp_root: Path,
    k_null: int = DEFAULT_K_NULL,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
    families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    scratch = _get_scratch_dir()
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]

    # Load canonical neurons once to build/verify the K=199 null bank
    raw_bundle = np.load(data_root / "connectomes" / "connectome_priors_bundle.npz", allow_pickle=True)
    canon_ref = [str(n) for n in raw_bundle["canonical_neurons"]]
    gen = ConstrainedNullConnectomeGenerator(canon_ref, data_root=data_root)
    bank_path, constraint_report = ensure_null_bank(gen, k_null=k_null, scratch_dir=scratch, resume=resume)

    ds_list = list(datasets) if datasets else list(DATASETS_ALL3)
    fam_list = list(families) if families else ["full", "sign", "gap"]
    out_by_ds: Dict[str, Any] = {}
    for d in ds_list:
        ckpt = scratch / f"ridge199_{d}_k{k_null}.json"
        if resume and ckpt.exists():
            print(f"[M4:ridge199] Resuming complete cached {d} from {ckpt}", flush=True)
            out_by_ds[d] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue

        # Check which requested families are still missing
        fam_ckpts = {f: scratch / f"ridge199_{d}_k{k_null}_{f}.json" for f in ("full", "sign", "gap")}
        missing_fams = [f for f in fam_list if not (resume and fam_ckpts[f].exists())]
        if not missing_fams:
            print(f"[M4:ridge199] All requested families {fam_list} already cached for {d}", flush=True)
        else:
            t0 = time.time()
            print(f"[M4:ridge199] Preparing frozen anchor cache for {d} (families={missing_fams})...", flush=True)
            ds_cache = build_dataset_ridge_fold_caches(d, cfg, data_root, exp_root)
            if ds_cache["canonical_neurons"] != canon_ref:
                raise AssertionError(f"Canonical neuron order mismatch on {d}")

            # 1. Real graph k=0 (verified against saved r3_03)
            real_res = retrain_and_eval_ridge_on_graph(ds_cache, gen.real_stack, hp)
            r3_03_p = exp_root / "runs" / "r3_03_m3r3_03_h3_multilayer_wired_plus_wireless_npp" / f"predictions_dev_{d}.npz"
            saved_res = score_saved_predictions_file(
                r3_03_p, split="dev", data_root=data_root, exp_root=exp_root, resolve_registry_ref=False
            )
            k0_diff = abs(real_res["E_d"] - float(saved_res["E_d"]))
            if k0_diff > 1e-5:
                raise AssertionError(f"k=0 mismatch on {d}: {real_res['E_d']} vs {saved_res['E_d']}")

            base_meta = {
                "dataset_id": d,
                "k_null": k_null,
                "real_E_d": float(real_res["E_d"]),
                "saved_r3_03_E_d": float(saved_res["E_d"]),
                "k0_abs_diff": float(k0_diff),
            }

            if "full" in missing_fams:
                mm = np.load(bank_path, mmap_mode="r")
                full_null_ed: List[float] = []
                t_full = time.time()
                for k in range(k_null):
                    w_k = np.asarray(mm[k], dtype=np.float32)
                    r_k = retrain_and_eval_ridge_on_graph(ds_cache, w_k, hp)
                    full_null_ed.append(float(r_k["E_d"]))
                    if (k + 1) % 50 == 0 or (k + 1) == k_null:
                        print(
                            f"[M4:ridge199:{d}] Full-null {k + 1}/{k_null} | "
                            f"real E_d={real_res['E_d']:.6f} | null mean={np.mean(full_null_ed):.6f} | "
                            f"elapsed={time.time() - t_full:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                            flush=True,
                        )
                del mm
                fam_ckpts["full"].write_text(json.dumps({
                    **base_meta,
                    "family": "full",
                    "E_d_list": full_null_ed,
                    "wall_sec": round(time.time() - t_full, 2),
                    "peak_rss_mb": round(_peak_rss_mb(), 1),
                }), encoding="utf-8")

            if "sign" in missing_fams:
                sign_null_ed: List[float] = []
                t_sign = time.time()
                for k in range(1, k_null + 1):
                    w_s = gen.generate_sign_permutation_stack(PRIMARY_SEED + 500000 + 1000 * k)
                    r_s = retrain_and_eval_ridge_on_graph(ds_cache, w_s, hp)
                    sign_null_ed.append(float(r_s["E_d"]))
                    if k % 50 == 0 or k == k_null:
                        print(
                            f"[M4:ridge199:{d}] Sign-only {k}/{k_null} | "
                            f"real E_d={real_res['E_d']:.6f} | null mean={np.mean(sign_null_ed):.6f} | "
                            f"elapsed={time.time() - t_sign:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                            flush=True,
                        )
                fam_ckpts["sign"].write_text(json.dumps({
                    **base_meta,
                    "family": "sign",
                    "E_d_list": sign_null_ed,
                    "wall_sec": round(time.time() - t_sign, 2),
                    "peak_rss_mb": round(_peak_rss_mb(), 1),
                }), encoding="utf-8")

            if "gap" in missing_fams:
                gap_null_ed: List[float] = []
                t_gap = time.time()
                for k in range(1, k_null + 1):
                    w_g = gen.generate_gap_only_null_stack(PRIMARY_SEED + 900000 + 1000 * k)
                    r_g = retrain_and_eval_ridge_on_graph(ds_cache, w_g, hp)
                    gap_null_ed.append(float(r_g["E_d"]))
                    if k % 50 == 0 or k == k_null:
                        print(
                            f"[M4:ridge199:{d}] Gap-only {k}/{k_null} | "
                            f"real E_d={real_res['E_d']:.6f} | null mean={np.mean(gap_null_ed):.6f} | "
                            f"elapsed={time.time() - t_gap:.1f}s | RSS={_peak_rss_mb():.1f} MB",
                            flush=True,
                        )
                fam_ckpts["gap"].write_text(json.dumps({
                    **base_meta,
                    "family": "gap",
                    "E_d_list": gap_null_ed,
                    "wall_sec": round(time.time() - t_gap, 2),
                    "peak_rss_mb": round(_peak_rss_mb(), 1),
                }), encoding="utf-8")

            del ds_cache
            gc.collect()

        # Merge into unified per-dataset checkpoint when all 3 families are present
        if all(fam_ckpts[f].exists() for f in ("full", "sign", "gap")):
            f_full = json.loads(fam_ckpts["full"].read_text(encoding="utf-8"))
            f_sign = json.loads(fam_ckpts["sign"].read_text(encoding="utf-8"))
            f_gap = json.loads(fam_ckpts["gap"].read_text(encoding="utf-8"))
            rec = {
                "dataset_id": d,
                "k_null": k_null,
                "real_E_d": float(f_full["real_E_d"]),
                "saved_r3_03_E_d": float(f_full["saved_r3_03_E_d"]),
                "k0_abs_diff": float(f_full["k0_abs_diff"]),
                "full_null_E_d": f_full["E_d_list"],
                "sign_only_null_E_d": f_sign["E_d_list"],
                "gap_only_null_E_d": f_gap["E_d_list"],
                "wall_sec": round(float(f_full["wall_sec"]) + float(f_sign["wall_sec"]) + float(f_gap["wall_sec"]), 2),
                "peak_rss_mb": max(float(f_full["peak_rss_mb"]), float(f_sign["peak_rss_mb"]), float(f_gap["peak_rss_mb"])),
            }
            ckpt.write_text(json.dumps(rec), encoding="utf-8")
            out_by_ds[d] = rec

    return {"constraints": constraint_report, "by_dataset": out_by_ds}


def run_stage_film_null(
    data_root: Path,
    exp_root: Path,
    k_null: int = DEFAULT_K_NULL,
    k_film: int = DEFAULT_K_FILM,
    resume: bool = True,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    scratch = _get_scratch_dir()
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]

    raw_bundle = np.load(data_root / "connectomes" / "connectome_priors_bundle.npz", allow_pickle=True)
    canon_ref = [str(n) for n in raw_bundle["canonical_neurons"]]
    gen = ConstrainedNullConnectomeGenerator(canon_ref, data_root=data_root)
    bank_path, _ = ensure_null_bank(gen, k_null=k_null, scratch_dir=scratch, resume=resume)

    ds_list = list(datasets) if datasets else list(DATASETS_MAIN2)
    out_by_ds: Dict[str, Any] = {}
    for d in ds_list:
        ckpt = scratch / f"film_null_{d}_k{k_film}.json"
        part_ckpt = scratch / f"film_null_{d}_k{k_film}_partial.json"
        if resume and ckpt.exists():
            print(f"[M4:film_null] Resuming cached {d} from {ckpt}", flush=True)
            out_by_ds[d] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue

        t0 = time.time()
        null_ed: List[float] = []
        prior_wall = 0.0
        real_ed_val: Optional[float] = None
        if resume and part_ckpt.exists():
            pc = json.loads(part_ckpt.read_text(encoding="utf-8"))
            null_ed = [float(x) for x in pc.get("null_E_d", [])]
            prior_wall = float(pc.get("wall_sec", 0.0))
            real_ed_val = float(pc["real_E_d"]) if "real_E_d" in pc else None
            print(f"[M4:film_null] Resuming partial {d} from {len(null_ed)}/{k_film}...", flush=True)

        print(f"[M4:film_null] Preparing FiLM anchor cache for {d} (K_film={k_film})...", flush=True)
        ds_cache = build_dataset_film_fold_caches(d, cfg, data_root, exp_root)
        if real_ed_val is None:
            real_res = retrain_and_eval_film_on_graph(ds_cache, gen.real_stack, hp, seed=PRIMARY_SEED)
            real_ed_val = float(real_res["E_d"])

        mm = np.load(bank_path, mmap_mode="r")
        start_k = len(null_ed)
        for k in range(start_k, k_film):
            w_k = np.asarray(mm[k], dtype=np.float32)
            r_k = retrain_and_eval_film_on_graph(ds_cache, w_k, hp, seed=PRIMARY_SEED)
            null_ed.append(float(r_k["E_d"]))
            if (k + 1) % 3 == 0 or (k + 1) == k_film:
                part_ckpt.write_text(json.dumps({
                    "dataset_id": d,
                    "k_film": k_film,
                    "real_E_d": real_ed_val,
                    "null_E_d": null_ed,
                    "wall_sec": round(prior_wall + (time.time() - t0), 2),
                    "peak_rss_mb": round(_peak_rss_mb(), 1),
                }), encoding="utf-8")
                print(
                    f"[M4:film_null:{d}] {k + 1}/{k_film} | real E_d={real_ed_val:.6f} | "
                    f"null mean={np.mean(null_ed):.6f} | elapsed={time.time() - t0:.1f}s | "
                    f"RSS={_peak_rss_mb():.1f} MB",
                    flush=True,
                )
        del mm

        rec = {
            "dataset_id": d,
            "k_film": k_film,
            "real_E_d": float(real_ed_val),
            "null_E_d": null_ed,
            "wall_sec": round(prior_wall + (time.time() - t0), 2),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
        }
        ckpt.write_text(json.dumps(rec), encoding="utf-8")
        out_by_ds[d] = rec

        del ds_cache
        gc.collect()

    return out_by_ds


def _summarize_null_vector(
    real_val: float,
    null_vals: Sequence[float],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    arr = np.asarray(null_vals, dtype=np.float64)
    k_tot = int(arr.size)
    if higher_is_better:
        k_ge = int(np.sum(arr >= float(real_val) - 1e-12))
        z = float((float(real_val) - np.mean(arr)) / max(float(np.std(arr, ddof=1)), 1e-12))
    else:
        k_ge = int(np.sum(arr <= float(real_val) + 1e-12))
        z = float((np.mean(arr) - float(real_val)) / max(float(np.std(arr, ddof=1)), 1e-12))
    p_emp = float((1 + k_ge) / (1 + k_tot))
    return {
        "real": round(float(real_val), 6),
        "null_mean": round(float(np.mean(arr)), 6),
        "null_std": round(float(np.std(arr, ddof=1)), 6),
        "null_median": round(float(np.median(arr)), 6),
        "null_p05": round(float(np.percentile(arr, 5)), 6),
        "null_p95": round(float(np.percentile(arr, 95)), 6),
        "real_minus_null_mean": round(float(real_val - np.mean(arr)), 6),
        "k_ge_real": k_ge,
        "k_total": k_tot,
        "p_empirical_one_sided": round(p_emp, 6),
        "z_score": round(z, 3),
    }


def _build_null_family_summary(
    by_ds: Mapping[str, Mapping[str, Any]],
    key_null: str,
    refs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    datasets_scope: Sequence[str],
    include_all3: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"vs_B4_ridge_self_stim": {}, "vs_M3R1_02_h1_confounds_only": {}, "per_dataset_E_d": {}}
    for d in datasets_scope:
        out["per_dataset_E_d"][d] = {
            **_summarize_null_vector(by_ds[d]["real_E_d"], by_ds[d][key_null], higher_is_better=False),
            "null_E_d_array": [round(float(x), 6) for x in by_ds[d][key_null]],
        }

    for ref_name, out_key in (
        ("B4_ridge_self_stim", "vs_B4_ridge_self_stim"),
        ("M3R1_02_h1_confounds_only", "vs_M3R1_02_h1_confounds_only"),
    ):
        per_ds_i_real: Dict[str, float] = {}
        per_ds_i_null: Dict[str, np.ndarray] = {}
        per_ds_sum: Dict[str, Any] = {}
        for d in datasets_scope:
            ref_ed = float(refs[ref_name][d]["E_d"])
            r_i = 1.0 - float(by_ds[d]["real_E_d"]) / ref_ed
            n_i = 1.0 - np.asarray(by_ds[d][key_null], dtype=np.float64) / ref_ed
            per_ds_i_real[d] = r_i
            per_ds_i_null[d] = n_i
            per_ds_sum[d] = _summarize_null_vector(r_i, n_i, higher_is_better=True)

        m2_real = float(np.mean([per_ds_i_real[d] for d in DATASETS_MAIN2]))
        m2_null = np.mean(np.vstack([per_ds_i_null[d] for d in DATASETS_MAIN2]), axis=0)
        res_block: Dict[str, Any] = {
            "I_macro_main2": {
                **_summarize_null_vector(m2_real, m2_null, higher_is_better=True),
                "null_I_macro_main2_array": [round(float(x), 6) for x in m2_null],
            },
            "per_dataset_I_d": per_ds_sum,
        }
        if include_all3 and all(d in per_ds_i_real for d in DATASETS_ALL3):
            a3_real = float(np.mean([per_ds_i_real[d] for d in DATASETS_ALL3]))
            a3_null = np.mean(np.vstack([per_ds_i_null[d] for d in DATASETS_ALL3]), axis=0)
            res_block["I_macro_all3"] = {
                **_summarize_null_vector(a3_real, a3_null, higher_is_better=True),
                "null_I_macro_all3_array": [round(float(x), 6) for x in a3_null],
            }
        out[out_key] = res_block
    return out


def run_stage_summarize(
    data_root: Path,
    exp_root: Path,
    k_null: int = DEFAULT_K_NULL,
    k_film: int = DEFAULT_K_FILM,
) -> Dict[str, Any]:
    scratch = _get_scratch_dir()
    refs = _load_reference_scored(DATASETS_ALL3, exp_root, data_root)
    constraints = json.loads((scratch / f"null_bank_k{k_null}_constraints.json").read_text(encoding="utf-8"))

    ridge_by_ds = {
        d: json.loads((scratch / f"ridge199_{d}_k{k_null}.json").read_text(encoding="utf-8"))
        for d in DATASETS_ALL3
    }
    film_by_ds = {
        d: json.loads((scratch / f"film_null_{d}_k{k_film}.json").read_text(encoding="utf-8"))
        for d in DATASETS_MAIN2
    }

    summary = {
        "schema_version": "m4_null_graphs_v1",
        "generated_utc": _utc_now(),
        "config_id": "lambda_global_v1",
        "config_sha256": compute_file_sha256(exp_root / "configs" / "lambda_global.json"),
        "connectome_bundle_sha256": compute_file_sha256(data_root / "connectomes" / "connectome_priors_bundle.npz"),
        "vault_sealed_untouched": True,
        "retraining_integrity_guarantee": (
            "Every null graph k=1..K was genuinely retrained from scratch inside every CV fold "
            "after recomputing incoming causal connectome drives from the rewired null graph "
            "(zero prediction shuffling, zero real-graph weight reuse)."
        ),
        "null_generator_constraints": constraints,
        "primary_ridge_k199_full_rewired": {
            "model_id": "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
            "k0_verification": {
                d: {
                    "real_E_d": round(float(ridge_by_ds[d]["real_E_d"]), 6),
                    "saved_r3_03_E_d": round(float(ridge_by_ds[d]["saved_r3_03_E_d"]), 6),
                    "abs_diff": float(ridge_by_ds[d]["k0_abs_diff"]),
                }
                for d in DATASETS_ALL3
            },
            **_build_null_family_summary(ridge_by_ds, "full_null_E_d", refs, DATASETS_ALL3, include_all3=True),
        },
        "polarity_sign_only_permutation_k199": {
            "model_id": "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
            "description": "Fixed biological topology & weights; CeNGEN polarity labels (pos/neg/unk) permuted within (tier_post, tier_pre, dist_bin).",
            **_build_null_family_summary(ridge_by_ds, "sign_only_null_E_d", refs, DATASETS_ALL3, include_all3=True),
        },
        "gap_junction_only_rewired_k199": {
            "model_id": "M3R3_03_h3_multilayer_wired_plus_wireless_npp",
            "description": "Only W_gap_herm symmetrically rewired (preserving degree, cell-type, dist-bin); chemical & wireless layers fixed.",
            **_build_null_family_summary(ridge_by_ds, "gap_only_null_E_d", refs, DATASETS_ALL3, include_all3=True),
        },
        "matched_film_retrained_null": {
            "model_id": "M3R4_01_connectome_film_herm_final",
            "k_film": k_film,
            "datasets": list(DATASETS_MAIN2),
            **_build_null_family_summary(film_by_ds, "null_E_d", refs, DATASETS_MAIN2, include_all3=False),
        },
        "runtime_and_memory": {
            "null_bank_gen_wall_sec": float(constraints["generation_wall_sec"]),
            "ridge199_wall_sec_by_dataset": {d: float(ridge_by_ds[d]["wall_sec"]) for d in DATASETS_ALL3},
            "film_null_wall_sec_by_dataset": {d: float(film_by_ds[d]["wall_sec"]) for d in DATASETS_MAIN2},
            "peak_rss_mb_max": round(
                max(
                    [float(ridge_by_ds[d]["peak_rss_mb"]) for d in DATASETS_ALL3]
                    + [float(film_by_ds[d]["peak_rss_mb"]) for d in DATASETS_MAIN2]
                ),
                1,
            ),
        },
    }

    out_path = exp_root / "results" / "null_graphs_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[M4:summarize] Wrote {out_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 Stage A: >= 199 Retrained Constrained Random Null Graphs")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["benchmark", "ridge199", "film_null", "summarize", "all"],
    )
    parser.add_argument("--k-null", type=int, default=DEFAULT_K_NULL)
    parser.add_argument("--k-film", type=int, default=DEFAULT_K_FILM)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--family", type=str, default="all", choices=["full", "sign", "gap", "all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--exp-root", type=str, default=str(EXP_ROOT_DEFAULT))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    ds_arg = [args.dataset] if args.dataset else None
    fam_arg = None if args.family == "all" else [args.family]

    if args.stage in ("benchmark", "all"):
        run_stage_benchmark(data_root, exp_root)
    if args.stage in ("ridge199", "all"):
        run_stage_ridge199(data_root, exp_root, k_null=args.k_null, resume=args.resume, datasets=ds_arg, families=fam_arg)
    if args.stage in ("film_null", "all"):
        run_stage_film_null(data_root, exp_root, k_null=args.k_null, k_film=args.k_film, resume=args.resume, datasets=ds_arg)
    if args.stage in ("summarize", "all"):
        run_stage_summarize(data_root, exp_root, k_null=args.k_null, k_film=args.k_film)


if __name__ == "__main__":
    main()
