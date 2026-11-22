"""All 6 competitive causal baseline families (`M2`, `MODEL_REQUIREMENTS.md §4` & `DISPATCH.md §4.4`).

Every baseline:
1. Consumes strictly causal `[-10, 0)s` (and optional `[-30, -10)s` calibration) frames `< onset_idx`.
2. Fits all scaling, shrinkage priors, latent subspaces, and weights strictly inside the training fold.
3. Excludes target self `i` and symmetric bilateral homolog `i_LR` from all cross-neuron summaries.
4. Uses a single unified `lambda_global` hyperparameter dictionary (zero `dataset_id` tuning branches).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import (
    _get_allow_mask_matrix,
    build_post_temporal_basis,
    extract_causal_network_state_per_target,
    extract_self_history_features_matrix,
    fit_fold_neuron_stim_priors,
    predict_prior_delta_trajectory,
)


# Canonical M2 Baseline Registry Metadata (6 Families per MODEL_REQUIREMENTS.md §4 & DISPATCH.md §4.4)
M2_BASELINE_SPECS: List[Dict[str, Any]] = [
    {
        "model_id": "B1_persistence",
        "family_id": "B1",
        "family_name": "Persistence / Causal Pre-Stimulus Anchor",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": False,
        "description": "Zero-change causal pre-stimulus anchor [-2, 0)s (fallback [-10, 0)s / mu_train) held over [0, 10)s.",
    },
    {
        "model_id": "B2_neuron_stim_mean",
        "family_id": "B2",
        "family_name": "Neuron x Stimulus Empirical Mean Trajectory",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": False,
        "description": "Causal pre-stimulus anchor plus empirical-Bayes shrunk training-fold Neuron x Stimulus basis trajectory.",
    },
    {
        "model_id": "B3_self_arfir_spline",
        "family_id": "B3",
        "family_name": "Regularized AR-FIR / Orthonormal Temporal Basis Self-History",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": False,
        "description": "Multi-lag [-10, 0)s self-history + derivatives + [-30, -10)s calibration mean-reversion + stimulus schedule ONLY.",
    },
    {
        "model_id": "B4_ridge_self_stim",
        "family_id": "B4",
        "family_name": "Regularized Tabular Ridge & Repaired Causal MLP",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": False,
        "description": "Sample-scaled Ridge on causal self-history, functional tier, stimulus schedule, and fold-internal prior drive.",
    },
    {
        "model_id": "B4_ridge_cross_neuron",
        "family_id": "B4",
        "family_name": "Regularized Tabular Ridge & Repaired Causal MLP",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": False,
        "description": "Sample-scaled Ridge adding target-self- and bilateral-excluded causal circuit state features.",
    },
    {
        "model_id": "B4_mlp_repaired_self_stim",
        "family_id": "B4",
        "family_name": "Regularized Tabular Ridge & Repaired Causal MLP",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": True,
        "description": "Repaired fold-consistent 2-layer residual MLP (self+stim only) using inner-fold priors for early stopping.",
    },
    {
        "model_id": "B4_mlp_repaired_cross_neuron",
        "family_id": "B4",
        "family_name": "Regularized Tabular Ridge & Repaired Causal MLP",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": True,
        "description": "Repaired fold-consistent 2-layer residual MLP adding excluded-self/bilateral cross-neuron circuit state.",
    },
    {
        "model_id": "B5_pop_rrr",
        "family_id": "B5",
        "family_name": "Graph-Free Low-Rank Dynamical Baselines (RRR, LDS+u, GRU, TCN)",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": False,
        "description": "Graph-free Reduced-Rank Regression (rank=6) over self-history + excluded-self SVD population subspace.",
    },
    {
        "model_id": "B5_lds_stim",
        "family_id": "B5",
        "family_name": "Graph-Free Low-Rank Dynamical Baselines (RRR, LDS+u, GRU, TCN)",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": False,
        "description": "Stimulus-driven Linear Dynamical System (LDS+u, latent_dim=8, 0.5s steps) rolled out from target-excluded z_0.",
    },
    {
        "model_id": "B5_causal_gru",
        "family_id": "B5",
        "family_name": "Graph-Free Low-Rank Dynamical Baselines (RRR, LDS+u, GRU, TCN)",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": True,
        "description": "Compact causal GRU (hidden_dim=24) filtering [-10, 0)s binned trace + obs mask + excluded-self context.",
    },
    {
        "model_id": "B5_causal_tcn",
        "family_id": "B5",
        "family_name": "Graph-Free Low-Rank Dynamical Baselines (RRR, LDS+u, GRU, TCN)",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": True,
        "description": "Compact strictly causal dilated 1D TCN (channels=24, k=3, dilations=1,2) over [-10, 0)s + excluded-self context.",
    },
    {
        "model_id": "B6_poco_univariate_ms_mlp",
        "family_id": "B6",
        "family_name": "Literature State-Modulation & Capacity-Matched Controls",
        "is_self_only": True,
        "uses_network_state": False,
        "is_stochastic": True,
        "description": "Univariate multi-scale MLP baseline control from POCO (Duan et al. NeurIPS 2025) without population FiLM.",
    },
    {
        "model_id": "B6_poco_film",
        "family_id": "B6",
        "family_name": "Literature State-Modulation & Capacity-Matched Controls",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": True,
        "description": "Causal unseen-animal adaptation of POCO (Duan et al. 2025) modulating self-history via excluded-population FiLM.",
    },
    {
        "model_id": "B6_ray_gordus_state_gain",
        "family_id": "B6",
        "family_name": "Literature State-Modulation & Capacity-Matched Controls",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": False,
        "description": "Causal leaky-ODE state-gain baseline (Ray & Gordus 2025) gating multi-tau stimulus drive by AVA/RIM & dAVA/dt.",
    },
    {
        "model_id": "B6_capacity_matched_graphfree_control",
        "family_id": "B6",
        "family_name": "Literature State-Modulation & Capacity-Matched Controls",
        "is_self_only": False,
        "uses_network_state": True,
        "is_stochastic": False,
        "description": "Capacity-matched dense multi-lag low-rank graph-free population control against connectome priors.",
    },
]

MODEL_ID_ALIASES: Dict[str, str] = {
    "B0_persistence": "B1_persistence",
    "B1_mean_traj": "B2_neuron_stim_mean",
    "B2_self_arfir": "B3_self_arfir_spline",
    "B3_tabular_ridge": "B4_ridge_self_stim",
    "B3_repaired_mlp": "B4_mlp_repaired_self_stim",
    "B4_pop_rrr": "B5_pop_rrr",
    "B4_lds_stim": "B5_lds_stim",
    "B4_causal_gru": "B5_causal_gru",
    "B5_dense_lowrank_control": "B6_capacity_matched_graphfree_control",
    "B6_ray_gordus_cmd": "B6_ray_gordus_state_gain",
}


def resolve_canonical_model_id(model_id: str) -> str:
    s = str(model_id).strip()
    return MODEL_ID_ALIASES.get(s, s)


def _solve_ridge_sample_scaled(x: np.ndarray, y: np.ndarray, l2_penalty: float) -> np.ndarray:
    """Solve sample-scaled Ridge in closed form (`MODEL_REQUIREMENTS.md §3`)."""
    n_samp, d = x.shape
    if n_samp == 0:
        return np.zeros((d, y.shape[1]), dtype=np.float32)
    scale = float(l2_penalty) * max(1.0, np.sqrt(n_samp / 100.0))
    gram = (x.T @ x).astype(np.float64) + scale * np.eye(d, dtype=np.float64)
    rhs = (x.T @ y).astype(np.float64)
    w = np.linalg.solve(gram, rhs)
    return w.astype(np.float32)


def _project_trial_residual_targets(
    trial: Mapping[str, Any],
    priors: Mapping[str, Any],
    s_n: np.ndarray,
    n_canon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(target_coef_norm [N, B], row_valid [N], prior_delta [N, T_out], phi [T_out, B])`."""
    prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, n_canon)
    vm = trial["valid_mask"]
    y_true = trial["y_true"]
    y_anc = trial["y_pre_anchor"]
    b_dim = phi.shape[1]

    coef_norm = np.zeros((n_canon, b_dim), dtype=np.float32)
    row_valid = np.zeros(n_canon, dtype=bool)

    full_rows = np.where(np.all(vm, axis=1))[0]
    if full_rows.size > 0:
        res_norm = (
            (y_true[full_rows] - y_anc[full_rows, None] - prior_delta[full_rows])
            / s_n[full_rows, None]
        ).astype(np.float64)
        coef_norm[full_rows] = (res_norm @ phi).astype(np.float32)
        row_valid[full_rows] = True

    part_rows = np.where(np.any(vm, axis=1) & (~np.all(vm, axis=1)))[0]
    for n_i in part_rows:
        rm = vm[n_i]
        if int(np.sum(rm)) >= 4:
            ps = phi[rm].astype(np.float64)
            rn = ((y_true[n_i, rm] - y_anc[n_i] - prior_delta[n_i, rm]) / s_n[n_i]).astype(np.float64)
            c_hat = np.linalg.solve(ps.T @ ps + 1e-2 * np.eye(b_dim), ps.T @ rn)
            coef_norm[n_i] = c_hat.astype(np.float32)
            row_valid[n_i] = True

    return coef_norm, row_valid, prior_delta, phi


def _reconstruct_y_pred_from_coef(
    trial: Mapping[str, Any],
    prior_delta: np.ndarray,
    phi: np.ndarray,
    pred_coef_norm: np.ndarray,
    s_n: np.ndarray,
    mu_train: np.ndarray,
) -> np.ndarray:
    """Reconstruct `[N_canon, T_out]` trajectory prediction from normalized residual basis coefficients."""
    y_anc = np.where(np.isfinite(trial["y_pre_anchor"]), trial["y_pre_anchor"], mu_train).astype(np.float32)
    res_traj = (pred_coef_norm.astype(np.float32) @ phi.T) * s_n[:, None].astype(np.float32)
    return (y_anc[:, None] + prior_delta + res_traj).astype(np.float32)


def _encode_stim_onehot(stim_id: str, is_buf: bool, stim_vocab: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(stim_vocab) + 1, dtype=np.float32)
    if is_buf:
        vec[-1] = 1.0
    elif stim_id in stim_vocab:
        vec[list(stim_vocab).index(stim_id)] = 1.0
    return vec


class _CausalConv1dBlock(nn.Module):
    """Strictly causal 1D dilated convolution block (left-padding ONLY; zero right-padding)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T_in]
        x_pad = F.pad(x, (self.left_pad, 0))
        h = F.silu(self.norm(self.conv(x_pad)))
        h = self.drop(h)
        return h + self.proj(x)


class CausalBaselineModel:
    """Unified interface for all 6 M2 baseline families (`B1`..`B6`) under `lambda_global`."""

    def __init__(
        self,
        model_id: str,
        config: Mapping[str, Any],
        canonical_neurons: Sequence[str],
        functional_tiers: Sequence[str],
    ) -> None:
        self.requested_model_id = str(model_id)
        self.model_id = resolve_canonical_model_id(model_id)
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

    def _get_h_self(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and tid in fold_cache.get("h_self", {}):
            return fold_cache["h_self"][tid]
        lag_bins = tuple((float(a), float(b)) for a, b in self.hp.get("pre_ar_lags_sec", [[0.0, 1.0], [1.0, 2.5], [2.5, 5.0], [5.0, 10.0]]))
        return extract_self_history_features_matrix(trial, s_n, mu_train, lag_bins_sec=lag_bins)

    def _get_net_state(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        src_tr = net_trial_override if net_trial_override is not None else trial
        tid = str(src_tr.get("trial_id", ""))
        if net_trial_override is None and fold_cache is not None and tid in fold_cache.get("net_state", {}):
            return fold_cache["net_state"][tid]
        return extract_causal_network_state_per_target(
            src_tr,
            self.canonical_neurons,
            s_n,
            mu_train,
            exclude_bilateral=True,
            preserve_sensory_asymmetry=False,
        )

    def _get_projected_targets(
        self,
        trial: Mapping[str, Any],
        priors: Mapping[str, Any],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tid = str(trial.get("trial_id", ""))
        if fold_cache is not None and priors is fold_cache.get("priors") and tid in fold_cache.get("proj_targets", {}):
            return fold_cache["proj_targets"][tid]
        return _project_trial_residual_targets(trial, priors, s_n, self.n_canon)

    def fit(
        self,
        train_trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> "CausalBaselineModel":
        taus = tuple(float(x) for x in self.hp.get("basis_taus_sec", [0.75, 2.0, 5.0, 10.0]))
        shrink = float(self.hp.get("prior_shrinkage_counts", 5.0))
        if fold_cache is not None and "priors" in fold_cache:
            priors = fold_cache["priors"]
        else:
            priors = fit_fold_neuron_stim_priors(
                train_trials, self.n_canon, taus_sec=taus, shrinkage_counts=shrink
            )
        self.state = {
            "s_n": np.asarray(s_n, dtype=np.float32).copy(),
            "priors": priors,
            "taus": taus,
        }
        non_buf_trials = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]

        if self.model_id in ("B1_persistence", "B2_neuron_stim_mean"):
            return self
        if self.model_id == "B3_self_arfir_spline":
            self._fit_b3_self_arfir(non_buf_trials, s_n, priors, fold_cache=fold_cache)
        elif self.model_id == "B4_ridge_self_stim":
            self._fit_b4_tabular_ridge(non_buf_trials, s_n, priors, include_network=False, fold_cache=fold_cache)
        elif self.model_id == "B4_ridge_cross_neuron":
            self._fit_b4_tabular_ridge(non_buf_trials, s_n, priors, include_network=True, fold_cache=fold_cache)
        elif self.model_id == "B4_mlp_repaired_self_stim":
            self._fit_b4_repaired_mlp(non_buf_trials, s_n, priors, include_network=False, fold_cache=fold_cache)
        elif self.model_id == "B4_mlp_repaired_cross_neuron":
            self._fit_b4_repaired_mlp(non_buf_trials, s_n, priors, include_network=True, fold_cache=fold_cache)
        elif self.model_id == "B5_pop_rrr":
            self._fit_b5_pop_rrr(non_buf_trials, s_n, priors, use_dense_lags=False, fold_cache=fold_cache)
        elif self.model_id == "B5_lds_stim":
            self._fit_b5_lds_stim(non_buf_trials, s_n, priors, fold_cache=fold_cache)
        elif self.model_id == "B5_causal_gru":
            self._fit_b5_seq_net(non_buf_trials, s_n, priors, arch="gru", fold_cache=fold_cache)
        elif self.model_id == "B5_causal_tcn":
            self._fit_b5_seq_net(non_buf_trials, s_n, priors, arch="tcn", fold_cache=fold_cache)
        elif self.model_id in ("B6_poco_film", "B6_poco_univariate_ms_mlp"):
            self._fit_b6_poco_film(
                non_buf_trials,
                s_n,
                priors,
                enable_pop_film=(self.model_id == "B6_poco_film"),
                fold_cache=fold_cache,
            )
        elif self.model_id == "B6_ray_gordus_state_gain":
            self._fit_b6_ray_gordus(non_buf_trials, s_n, priors, fold_cache=fold_cache)
        elif self.model_id == "B6_capacity_matched_graphfree_control":
            self._fit_b5_pop_rrr(non_buf_trials, s_n, priors, use_dense_lags=True, fold_cache=fold_cache)
        else:
            raise ValueError(f"Unknown baseline model_id: {self.requested_model_id!r} (resolved={self.model_id!r})")
        return self

    def predict_trial(
        self,
        trial: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        """Predict `[N_canon, T_out]` trajectory.

        `net_trial_override` is used ONLY by `verify-invariance` to perturb target neuron `i` inside
        the cross-neuron pathway while holding target self-history `trial` fixed, proving zero self-leakage.
        """
        s_n = self.state["s_n"]
        priors = self.state["priors"]
        mu_train = priors["mu_train"]
        y_anc = np.where(np.isfinite(trial["y_pre_anchor"]), trial["y_pre_anchor"], mu_train).astype(np.float32)
        n_out = trial["y_true"].shape[1]

        if self.model_id == "B1_persistence":
            return np.repeat(y_anc[:, None], n_out, axis=1)

        if fold_cache is not None and priors is fold_cache.get("priors") and str(trial.get("trial_id", "")) in fold_cache.get("proj_targets", {}):
            _, _, prior_delta, phi = fold_cache["proj_targets"][str(trial["trial_id"])]
        else:
            prior_delta, _, phi = predict_prior_delta_trajectory(trial, priors, self.n_canon)

        if self.model_id == "B2_neuron_stim_mean":
            return (y_anc[:, None] + prior_delta).astype(np.float32)

        if self.model_id == "B3_self_arfir_spline":
            h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=fold_cache)
            c_global = h_self @ self.state["w_global"]
            c_indiv = np.einsum("nd,ndb->nb", h_self, self.state["w_neuron"])
            pred_c = c_global + c_indiv
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id in ("B4_ridge_self_stim", "B4_ridge_cross_neuron"):
            inc_net = self.model_id == "B4_ridge_cross_neuron"
            x_tab = self._build_tabular_features(
                trial, s_n, priors, include_network=inc_net, fold_cache=fold_cache, net_trial_override=net_trial_override
            )
            pred_c = x_tab @ self.state["w_tab"]
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id in ("B4_mlp_repaired_self_stim", "B4_mlp_repaired_cross_neuron"):
            inc_net = self.model_id == "B4_mlp_repaired_cross_neuron"
            x_tab = self._build_tabular_features(
                trial, s_n, priors, include_network=inc_net, fold_cache=fold_cache, net_trial_override=net_trial_override
            )
            x_norm = (x_tab - self.state["x_mean"]) / self.state["x_std"]
            net: nn.Module = self.state["mlp_net"]
            net.eval()
            with torch.no_grad():
                pred_c = net(torch.from_numpy(x_norm.astype(np.float32))).numpy()
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id in ("B5_pop_rrr", "B6_capacity_matched_graphfree_control"):
            x_pop = self._build_excluded_pop_rrr_features(
                trial,
                s_n,
                priors,
                use_dense_lags=(self.model_id == "B6_capacity_matched_graphfree_control"),
                fold_cache=fold_cache,
                net_trial_override=net_trial_override,
            )
            pred_c = (x_pop @ self.state["u_rrr"]) @ self.state["v_rrr"]
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id == "B5_lds_stim":
            ro_feat = self._rollout_lds_features(
                trial,
                s_n,
                mu_train,
                priors["stim_vocab"],
                fold_cache=fold_cache,
                net_trial_override=net_trial_override,
            )
            pred_c = ro_feat @ self.state["lds_readout"]
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id in ("B5_causal_gru", "B5_causal_tcn"):
            sq, cx = self._build_seq_inputs(
                trial, s_n, mu_train, priors["stim_vocab"], fold_cache=fold_cache, net_trial_override=net_trial_override
            )
            net_s: nn.Module = self.state["seq_net"]
            net_s.eval()
            with torch.no_grad():
                pred_c = net_s(torch.from_numpy(sq), torch.from_numpy(cx)).numpy()
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id in ("B6_poco_film", "B6_poco_univariate_ms_mlp"):
            ux, px = self._build_poco_inputs(
                trial, s_n, priors, fold_cache=fold_cache, net_trial_override=net_trial_override
            )
            net_p: nn.Module = self.state["poco_net"]
            net_p.eval()
            with torch.no_grad():
                pred_c = net_p(torch.from_numpy(ux), torch.from_numpy(px)).numpy()
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        if self.model_id == "B6_ray_gordus_state_gain":
            x_rg = self._build_ray_gordus_features(
                trial, s_n, priors, fold_cache=fold_cache, net_trial_override=net_trial_override
            )
            pred_c = x_rg @ self.state["w_rg"]
            return _reconstruct_y_pred_from_coef(trial, prior_delta, phi, pred_c, s_n, mu_train)

        raise ValueError(f"Unknown model_id {self.model_id}")

    def parameter_vector(self) -> np.ndarray:
        """Flatten all fitted model parameters into a 1D float64 vector for OOF tamper verification."""
        parts: List[np.ndarray] = [
            self.state["s_n"].ravel().astype(np.float64),
            self.state["priors"]["mu_train"].ravel().astype(np.float64),
        ]
        for (_, _), coef in sorted(self.state["priors"]["neuron_stim_coef"].items()):
            parts.append(coef.ravel().astype(np.float64))
        for k in sorted(self.state.keys()):
            if k in ("s_n", "priors", "taus", "selected_epochs"):
                continue
            val = self.state[k]
            if isinstance(val, np.ndarray):
                parts.append(val.ravel().astype(np.float64))
            elif isinstance(val, nn.Module):
                for p in val.parameters():
                    parts.append(p.detach().cpu().numpy().ravel().astype(np.float64))
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)

    # ------------------------------------------------------------------
    # Family B3: Multi-lag / spline AR-FIR Self-Dynamics
    # ------------------------------------------------------------------
    def _fit_b3_self_arfir(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        mu_train = priors["mu_train"]
        b_dim = int(priors["b_dim"])
        l2 = float(self.hp.get("l2_penalty", 25.0))

        xs_all: List[np.ndarray] = []
        ys_all: List[np.ndarray] = []
        per_n_x: List[List[np.ndarray]] = [[] for _ in range(self.n_canon)]
        per_n_y: List[List[np.ndarray]] = [[] for _ in range(self.n_canon)]

        for tr in trials:
            coef_norm, rval, _, _ = self._get_projected_targets(tr, priors, s_n, fold_cache=fold_cache)
            h_self = self._get_h_self(tr, s_n, mu_train, fold_cache=fold_cache)
            idx = np.where(rval)[0]
            if idx.size > 0:
                xs_all.append(h_self[idx])
                ys_all.append(coef_norm[idx])
                for n_i in idx:
                    per_n_x[int(n_i)].append(h_self[int(n_i)])
                    per_n_y[int(n_i)].append(coef_norm[int(n_i)])

        d_self = xs_all[0].shape[1] if xs_all else 10
        w_global = (
            _solve_ridge_sample_scaled(np.vstack(xs_all), np.vstack(ys_all), l2)
            if xs_all
            else np.zeros((d_self, b_dim), dtype=np.float32)
        )

        w_neuron = np.zeros((self.n_canon, d_self, b_dim), dtype=np.float32)
        for n_i in range(self.n_canon):
            if len(per_n_x[n_i]) >= 8:
                xn = np.vstack(per_n_x[n_i])
                yn = np.vstack(per_n_y[n_i]) - xn @ w_global
                w_neuron[n_i] = _solve_ridge_sample_scaled(xn, yn, l2 * 4.0)

        self.state["w_global"] = w_global
        self.state["w_neuron"] = w_neuron

    # ------------------------------------------------------------------
    # Family B4: Tabular Ridge + Repaired Fold-Consistent Residual MLP
    # ------------------------------------------------------------------
    def _build_tabular_features(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        include_network: bool = False,
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        mu_train = priors["mu_train"]
        h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=(fold_cache if priors is (fold_cache or {}).get("priors") else None))
        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), priors["stim_vocab"]
        )
        stim_mat = np.repeat(stim_oh[None, :], self.n_canon, axis=0)

        _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, self.n_canon)
        prior_norm = prior_coef / s_n[:, None].astype(np.float32)
        self_x_prior = h_self[:, [4, 6, 7]] * prior_norm[:, [0]]
        blocks = [h_self, self._tier_oh, stim_mat, prior_norm, self_x_prior]
        if include_network:
            net_st = self._get_net_state(
                trial,
                s_n,
                mu_train,
                fold_cache=(fold_cache if priors is (fold_cache or {}).get("priors") else None),
                net_trial_override=net_trial_override,
            )
            blocks.append(net_st)
        return np.hstack(blocks).astype(np.float32)

    def _fit_b4_tabular_ridge(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        include_network: bool = False,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        l2 = float(self.hp.get("l2_penalty", 25.0)) * (1.25 if include_network else 1.0)
        xs, ys = [], []
        for tr in trials:
            c_norm, rval, _, _ = self._get_projected_targets(tr, priors, s_n, fold_cache=fold_cache)
            x_tab = self._build_tabular_features(tr, s_n, priors, include_network=include_network, fold_cache=fold_cache)
            idx = np.where(rval)[0]
            if idx.size > 0:
                xs.append(x_tab[idx])
                ys.append(c_norm[idx])
        self.state["w_tab"] = _solve_ridge_sample_scaled(np.vstack(xs), np.vstack(ys), l2)

    def _fit_b4_repaired_mlp(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        outer_priors: Mapping[str, Any],
        include_network: bool = False,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Repaired MLP (`ANTIGRAVITY_CONTEXT` Trap #4):
        Inner train/val split fits `inner_priors` strictly on `inner_train_worms` for early stopping,
        then refits for `best_ep` on outer training worms with `outer_priors`.
        """
        seed = int(self.hp.get("seed", 20261117))
        torch.manual_seed(seed)
        np.random.seed(seed)

        worms = sorted({str(t["animal_id"]) for t in trials})
        n_val_w = max(1, len(worms) // 5)
        val_worms = set(worms[-n_val_w:])
        inner_tr = [t for t in trials if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in trials if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(trials)
            inner_va = list(trials[:1])

        inner_priors = fit_fold_neuron_stim_priors(
            inner_tr,
            self.n_canon,
            taus_sec=self.state["taus"],
            shrinkage_counts=float(self.hp.get("prior_shrinkage_counts", 5.0)),
        )

        def _assemble(
            tr_list: Sequence[Mapping[str, Any]],
            pr_dict: Mapping[str, Any],
            f_cache: Optional[Mapping[str, Any]] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            xl, yl = [], []
            for t in tr_list:
                cn, rv, _, _ = self._get_projected_targets(t, pr_dict, s_n, fold_cache=f_cache)
                xt = self._build_tabular_features(t, s_n, pr_dict, include_network=include_network, fold_cache=f_cache)
                ix = np.where(rv)[0]
                if ix.size > 0:
                    xl.append(xt[ix])
                    yl.append(cn[ix])
            return np.vstack(xl), np.vstack(yl)

        x_itr, y_itr = _assemble(inner_tr, inner_priors, None)
        x_iva, y_iva = _assemble(inner_va, inner_priors, None)
        xm = np.mean(x_itr, axis=0, keepdims=True)
        xs = np.maximum(np.std(x_itr, axis=0, keepdims=True), 1e-4)

        d_in = x_itr.shape[1]
        b_dim = y_itr.shape[1]
        h_dim = int(self.hp.get("mlp_hidden_dim", 48))
        drop = float(self.hp.get("dropout", 0.15))
        lr = float(self.hp.get("learning_rate", 0.002))
        wd = float(self.hp.get("weight_decay", 0.01))
        max_ep = int(self.hp.get("max_epochs", 30))
        patience = int(self.hp.get("early_stopping_patience", 6))
        bsz = int(self.hp.get("batch_size", 512))

        def _make_mlp(in_features: int) -> nn.Module:
            net = nn.Sequential(
                nn.Linear(in_features, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(h_dim, h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, b_dim),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        probe_net = _make_mlp(d_in)
        opt = torch.optim.AdamW(probe_net.parameters(), lr=lr, weight_decay=wd)
        xt_t = torch.from_numpy(((x_itr - xm) / xs).astype(np.float32))
        yt_t = torch.from_numpy(y_itr.astype(np.float32))
        xv_t = torch.from_numpy(((x_iva - xm) / xs).astype(np.float32))
        yv_t = torch.from_numpy(y_iva.astype(np.float32))

        best_val = float(torch.mean(yv_t ** 2).item())
        best_ep = 0
        bad = 0
        n_rows = xt_t.shape[0]
        for ep in range(1, max_ep + 1):
            probe_net.train()
            perm = torch.randperm(n_rows)
            for s0 in range(0, n_rows, bsz):
                bi = perm[s0 : s0 + bsz]
                opt.zero_grad()
                pred = probe_net(xt_t[bi])
                loss = torch.mean((pred - yt_t[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe_net.parameters(), 1.0)
                opt.step()
            probe_net.eval()
            with torch.no_grad():
                vloss = float(torch.mean((probe_net(xv_t) - yv_t) ** 2).item())
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        x_full, y_full = _assemble(trials, outer_priors, fold_cache)
        xm_full = np.mean(x_full, axis=0, keepdims=True)
        xs_full = np.maximum(np.std(x_full, axis=0, keepdims=True), 1e-4)
        torch.manual_seed(seed)
        final_net = _make_mlp(x_full.shape[1])
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            xf_t = torch.from_numpy(((x_full - xm_full) / xs_full).astype(np.float32))
            yf_t = torch.from_numpy(y_full.astype(np.float32))
            nf = xf_t.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for s0 in range(0, nf, bsz):
                    bi = perm[s0 : s0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(xf_t[bi]) - yf_t[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()
        self.state["mlp_net"] = final_net
        self.state["x_mean"] = xm_full.astype(np.float32)
        self.state["x_std"] = xs_full.astype(np.float32)
        self.state["selected_epochs"] = best_ep

    # ------------------------------------------------------------------
    # Family B5a & B6c: Graph-Free Population RRR & Dense Lag Control
    # ------------------------------------------------------------------
    def _fit_population_subspace(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        k_sub: int = 8,
    ) -> np.ndarray:
        pop_vecs = []
        for tr in trials:
            mp = tr["m_pre"]
            obs = np.sum(mp, axis=0) > 0
            v = np.zeros(self.n_canon, dtype=np.float32)
            v[obs] = np.clip((tr["y_pre_anchor"][obs] - mu_train[obs]) / s_n[obs], -5.0, 5.0)
            pop_vecs.append(v)
        mat = np.vstack(pop_vecs).astype(np.float64)
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        k_eff = min(k_sub, vt.shape[0])
        basis = np.zeros((self.n_canon, k_sub), dtype=np.float32)
        basis[:, :k_eff] = vt[:k_eff].T.astype(np.float32)
        return basis

    def _build_excluded_pop_rrr_features(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        use_dense_lags: bool = False,
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        """Build `[N_canon, D]` using exact `_get_allow_mask_matrix` matrix multiplication (zero self/LR leak)."""
        mu_train = priors["mu_train"]
        h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=fold_cache)
        p_basis = self.state["pop_basis"].astype(np.float64)  # [N_canon, K]
        allow_mat = _get_allow_mask_matrix(
            self.canonical_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=False
        ).astype(np.float64)

        net_tr = net_trial_override if net_trial_override is not None else trial
        xp = np.asarray(net_tr["x_pre"], dtype=np.float64)
        mp = np.asarray(net_tr["m_pre"], dtype=bool)
        fps = float(net_tr["fps"])
        n_fast = max(1, min(xp.shape[0], int(round(2.5 * fps))))

        cnt_f = np.sum(mp[-n_fast:], axis=0)
        obs_f = cnt_f > 0
        z_f = np.zeros(self.n_canon, dtype=np.float64)
        z_f[obs_f] = np.clip((net_tr["y_pre_anchor"][obs_f] - mu_train[obs_f]) / s_n[obs_f], -5.0, 5.0)

        # Exact target-excluded projection: allow_mat [N, N] @ (z_f[:, None] * p_basis [N, K]) -> [N, K]
        sub_f = allow_mat @ (z_f[:, None] * p_basis)
        if use_dense_lags:
            cnt_e = np.sum(mp[:-n_fast], axis=0) if xp.shape[0] > n_fast else cnt_f
            obs_e = cnt_e > 0
            z_e = np.zeros(self.n_canon, dtype=np.float64)
            if xp.shape[0] > n_fast and np.any(obs_e):
                em = np.sum(np.where(mp[:-n_fast, obs_e], xp[:-n_fast, obs_e], 0.0), axis=0) / cnt_e[obs_e]
                z_e[obs_e] = np.clip((em - mu_train[obs_e]) / s_n[obs_e], -5.0, 5.0)
            sub_e = allow_mat @ (z_e[:, None] * p_basis)
            pop_feats = np.hstack([sub_f, sub_f - sub_e]).astype(np.float32)
        else:
            pop_feats = sub_f.astype(np.float32)

        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), priors["stim_vocab"]
        )
        stim_mat = np.repeat(stim_oh[None, :], self.n_canon, axis=0)
        return np.hstack([h_self, pop_feats, stim_mat]).astype(np.float32)

    def _fit_b5_pop_rrr(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        use_dense_lags: bool = False,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        rank = int(self.hp.get("rank", 6))
        l2 = float(self.hp.get("l2_penalty", 25.0)) * (2.0 if use_dense_lags else 1.5)
        self.state["pop_basis"] = self._fit_population_subspace(trials, s_n, priors["mu_train"], k_sub=8)
        xs, ys = [], []
        for tr in trials:
            cn, rv, _, _ = self._get_projected_targets(tr, priors, s_n, fold_cache=fold_cache)
            xp = self._build_excluded_pop_rrr_features(tr, s_n, priors, use_dense_lags=use_dense_lags, fold_cache=fold_cache)
            ix = np.where(rv)[0]
            if ix.size > 0:
                xs.append(xp[ix])
                ys.append(cn[ix])
        x_mat = np.vstack(xs)
        y_mat = np.vstack(ys)
        w_full = _solve_ridge_sample_scaled(x_mat, y_mat, l2)
        y_hat = (x_mat @ w_full).astype(np.float64)
        _, _, vt = np.linalg.svd(y_hat, full_matrices=False)
        r_eff = min(rank, vt.shape[0])
        v_r = vt[:r_eff]
        u_r = (w_full.astype(np.float64) @ v_r.T).astype(np.float32)
        self.state["u_rrr"] = u_r
        self.state["v_rrr"] = v_r.astype(np.float32)

    # ------------------------------------------------------------------
    # Family B5b: Stimulus-Driven Linear Dynamical System (LDS+u)
    # ------------------------------------------------------------------
    def _fit_b5_lds_stim(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        d_lat = int(self.hp.get("lds_latent_dim", 8))
        l2 = float(self.hp.get("l2_penalty", 25.0))
        mu_train = priors["mu_train"]
        stim_vocab = priors["stim_vocab"]

        c_basis = self._fit_population_subspace(trials, s_n, mu_train, k_sub=d_lat)
        z_prev_list, z_next_list = [], []
        for tr in trials:
            z_seq, u_seq = self._extract_lds_binned_latents(tr, s_n, mu_train, c_basis, stim_vocab, include_post=True)
            inp = np.hstack([z_seq[:-1], u_seq[1:]])
            z_prev_list.append(inp)
            z_next_list.append(z_seq[1:])

        x_dyn = np.vstack(z_prev_list)
        y_dyn = np.vstack(z_next_list)
        w_ab = _solve_ridge_sample_scaled(x_dyn, y_dyn, l2)
        a_mat = w_ab[:d_lat, :].T.astype(np.float64)
        eigvals = np.linalg.eigvals(a_mat)
        max_abs = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        if max_abs > 0.96:
            a_mat = a_mat * (0.96 / max_abs)
        b_mat = w_ab[d_lat:, :].T.astype(np.float64)

        self.state["c_basis"] = c_basis
        self.state["lds_A"] = a_mat.astype(np.float32)
        self.state["lds_B"] = b_mat.astype(np.float32)

        xs_ro, ys_ro = [], []
        for tr in trials:
            cn, rv, _, _ = self._get_projected_targets(tr, priors, s_n, fold_cache=fold_cache)
            ro_feat = self._rollout_lds_features(tr, s_n, mu_train, stim_vocab, fold_cache=fold_cache)
            ix = np.where(rv)[0]
            if ix.size > 0:
                xs_ro.append(ro_feat[ix])
                ys_ro.append(cn[ix])
        self.state["lds_readout"] = _solve_ridge_sample_scaled(np.vstack(xs_ro), np.vstack(ys_ro), l2 * 1.5)

    def _extract_lds_binned_latents(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        c_basis: np.ndarray,
        stim_vocab: Sequence[str],
        include_post: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt_bin = 0.5
        n_bins = 20
        fps = float(trial["fps"])
        xp = trial["x_pre"]
        mp = trial["m_pre"]
        y_anc = np.where(np.isfinite(trial["y_pre_anchor"]), trial["y_pre_anchor"], mu_train)
        total_cb_energy = max(float(np.sum(c_basis ** 2)), 1e-6)

        def _bin_window(arr_tn: np.ndarray, msk_tn: np.ndarray) -> np.ndarray:
            t_len = arr_tn.shape[0]
            out_z = np.zeros((n_bins, c_basis.shape[1]), dtype=np.float32)
            for b in range(n_bins):
                f0 = min(int(round(b * dt_bin * fps)), max(0, t_len - 1))
                f1 = max(f0 + 1, min(t_len, int(round((b + 1) * dt_bin * fps))))
                sub_x = arr_tn[f0:f1]
                sub_m = msk_tn[f0:f1]
                cnt = np.sum(sub_m, axis=0)
                obs = cnt > 0
                if np.any(obs):
                    zn = np.zeros(self.n_canon, dtype=np.float32)
                    means = np.sum(np.where(sub_m[:, obs], sub_x[:, obs], 0.0), axis=0) / cnt[obs]
                    zn[obs] = np.clip((means - y_anc[obs]) / s_n[obs], -5.0, 5.0)
                    energy = float(np.sum(c_basis[obs] ** 2)) / total_cb_energy
                    out_z[b] = (zn @ c_basis) / max(energy, 0.25)
            return out_z

        z_pre_bins = _bin_window(xp, mp)
        u_zero = np.zeros((n_bins, len(stim_vocab) + 1), dtype=np.float32)
        if not include_post:
            return z_pre_bins, u_zero

        yp_tn = trial["y_true"].T
        mp_tn = trial["valid_mask"].T
        z_post_bins = _bin_window(yp_tn, mp_tn)
        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), stim_vocab
        )
        u_post_bins = np.zeros((n_bins, len(stim_vocab) + 1), dtype=np.float32)
        u_raw = trial["u_post"]
        for b in range(n_bins):
            f0 = min(len(u_raw) - 1, int(round(b * dt_bin * fps)))
            u_post_bins[b] = stim_oh * float(u_raw[max(0, f0)])
        return np.vstack([z_pre_bins, z_post_bins]), np.vstack([u_zero, u_post_bins])

    def _rollout_lds_features(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        stim_vocab: Sequence[str],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        """Compute exact target-excluded initial latent `z0_excl[i]` via `_get_allow_mask_matrix` and roll out `(A, B)`."""
        h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=fold_cache)
        c_basis = self.state["c_basis"].astype(np.float64)  # [N, d_lat]
        a_mat = self.state["lds_A"].astype(np.float64)
        b_mat = self.state["lds_B"].astype(np.float64)
        allow_mat = _get_allow_mask_matrix(
            self.canonical_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=False
        )

        net_tr = net_trial_override if net_trial_override is not None else trial
        xp = np.asarray(net_tr["x_pre"], dtype=np.float64)
        mp = np.asarray(net_tr["m_pre"], dtype=bool)
        fps = float(net_tr["fps"])
        t_len = xp.shape[0]
        f0 = min(int(round(19 * 0.5 * fps)), max(0, t_len - 1))
        f1 = max(f0 + 1, min(t_len, int(round(20 * 0.5 * fps))))

        y_anc_net = np.where(np.isfinite(net_tr["y_pre_anchor"]), net_tr["y_pre_anchor"], mu_train).astype(np.float64)
        cnt = np.sum(mp[f0:f1], axis=0)
        obs = cnt > 0
        zn_last = np.zeros(self.n_canon, dtype=np.float64)
        if np.any(obs):
            means = np.sum(np.where(mp[f0:f1, obs], xp[f0:f1, obs], 0.0), axis=0) / cnt[obs]
            zn_last[obs] = np.clip((means - y_anc_net[obs]) / s_n[obs], -5.0, 5.0)

        # Target-specific observed subspace energy strictly excluding self + bilateral!
        cb_row_sq = np.sum(c_basis ** 2, axis=1)  # [N]
        total_cb_sq = max(float(np.sum(cb_row_sq)), 1e-6)
        valid_excl = allow_mat & obs[None, :]  # [N, N]
        excl_energy = np.maximum((valid_excl.astype(np.float64) @ cb_row_sq) / total_cb_sq, 0.25)  # [N]
        z0_excl_mat = (allow_mat.astype(np.float64) @ (zn_last[:, None] * c_basis)) / excl_energy[:, None]  # [N, d_lat]

        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), stim_vocab
        ).astype(np.float64)
        u_raw = trial["u_post"]

        d_lat = a_mat.shape[0]
        a_sum_early = np.zeros((d_lat, d_lat), dtype=np.float64)
        a_sum_late = np.zeros((d_lat, d_lat), dtype=np.float64)
        forced_early = np.zeros(d_lat, dtype=np.float64)
        forced_late = np.zeros(d_lat, dtype=np.float64)
        z_forced = np.zeros(d_lat, dtype=np.float64)
        a_cur = np.eye(d_lat, dtype=np.float64)

        for step in range(20):
            a_cur = a_mat @ a_cur
            idx_u = min(len(u_raw) - 1, int(round(step * 0.5 * fps)))
            u_vec = stim_oh * float(u_raw[max(0, idx_u)])
            z_forced = a_mat @ z_forced + b_mat @ u_vec
            if step < 6:
                a_sum_early += a_cur / 6.0
                forced_early += z_forced / 6.0
            else:
                a_sum_late += a_cur / 14.0
                forced_late += z_forced / 14.0

        early_mat = z0_excl_mat @ a_sum_early.T + forced_early[None, :]
        late_mat = z0_excl_mat @ a_sum_late.T + forced_late[None, :]
        return np.hstack([h_self, early_mat.astype(np.float32), late_mat.astype(np.float32)]).astype(np.float32)

    # ------------------------------------------------------------------
    # Family B5c & B5d: Compact Causal GRU & Dilated Causal TCN
    # ------------------------------------------------------------------
    def _build_seq_inputs(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        mu_train: np.ndarray,
        stim_vocab: Sequence[str],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tid = str(trial.get("trial_id", ""))
        if net_trial_override is None and fold_cache is not None and tid in fold_cache.get("seq_pre", {}):
            seq = fold_cache["seq_pre"][tid]
        else:
            xp = trial["x_pre"]
            mp = trial["m_pre"]
            fps = float(trial["fps"])
            y_anc = np.where(np.isfinite(trial["y_pre_anchor"]), trial["y_pre_anchor"], mu_train)
            seq = np.zeros((self.n_canon, 10, 3), dtype=np.float32)
            t_len = xp.shape[0]
            for b in range(10):
                f0 = min(max(0, t_len - 1), int(round(b * 1.0 * fps)))
                f1 = max(f0 + 1, min(t_len, int(round((b + 1) * 1.0 * fps))))
                sx = xp[f0:f1]
                sm = mp[f0:f1]
                cnt = np.sum(sm, axis=0)
                obs = cnt > 0
                if np.any(obs):
                    mval = np.sum(np.where(sm[:, obs], sx[:, obs], 0.0), axis=0) / cnt[obs]
                    seq[obs, b, 0] = np.clip((mval - y_anc[obs]) / s_n[obs], -5.0, 5.0)
                    seq[obs, b, 1] = np.clip((y_anc[obs] - mu_train[obs]) / s_n[obs], -5.0, 5.0)
                    seq[obs, b, 2] = 1.0

        net_st = self._get_net_state(
            trial, s_n, mu_train, fold_cache=fold_cache, net_trial_override=net_trial_override
        )[:, [0, 2, 4, 6]]
        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), stim_vocab
        )
        ctx = np.hstack([net_st, np.repeat(stim_oh[None, :], self.n_canon, axis=0)]).astype(np.float32)
        return seq, ctx

    def _fit_b5_seq_net(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        outer_priors: Mapping[str, Any],
        arch: str = "gru",
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        seed = int(self.hp.get("seed", 20261117))
        torch.manual_seed(seed)
        np.random.seed(seed)

        b_dim = int(outer_priors["b_dim"])
        h_dim = int(self.hp.get("gru_hidden_dim", 24) if arch == "gru" else self.hp.get("tcn_channels", 24))
        k_sz = int(self.hp.get("tcn_kernel_size", 3))
        drop = float(self.hp.get("dropout", 0.15))
        lr = float(self.hp.get("learning_rate", 0.002))
        wd = float(self.hp.get("weight_decay", 0.01))
        max_ep = min(22, int(self.hp.get("max_epochs", 25)))
        patience = int(self.hp.get("early_stopping_patience", 5))
        bsz = int(self.hp.get("batch_size", 512))

        worms = sorted({str(t["animal_id"]) for t in trials})
        val_worms = set(worms[-max(1, len(worms) // 5) :])
        inner_tr = [t for t in trials if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in trials if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(trials)
            inner_va = list(trials[:1])

        inner_priors = fit_fold_neuron_stim_priors(
            inner_tr,
            self.n_canon,
            taus_sec=self.state["taus"],
            shrinkage_counts=float(self.hp.get("prior_shrinkage_counts", 5.0)),
        )

        def _assemble_seq(
            tr_list: Sequence[Mapping[str, Any]],
            pr_dict: Mapping[str, Any],
            f_cache: Optional[Mapping[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sl, cl, yl = [], [], []
            for t in tr_list:
                cn, rv, _, _ = self._get_projected_targets(t, pr_dict, s_n, fold_cache=f_cache)
                sq, cx = self._build_seq_inputs(
                    t, s_n, pr_dict["mu_train"], outer_priors["stim_vocab"], fold_cache=f_cache
                )
                ix = np.where(rv)[0]
                if ix.size > 0:
                    sl.append(sq[ix])
                    cl.append(cx[ix])
                    yl.append(cn[ix])
            return (
                torch.from_numpy(np.vstack(sl)),
                torch.from_numpy(np.vstack(cl)),
                torch.from_numpy(np.vstack(yl)),
            )

        s_tr, c_tr, y_tr = _assemble_seq(inner_tr, inner_priors, None)
        s_va, c_va, y_va = _assemble_seq(inner_va, inner_priors, None)

        class _CausalSeqModel(nn.Module):
            def __init__(self, ctx_dim: int, mode: str) -> None:
                super().__init__()
                self.mode = mode
                if mode == "gru":
                    self.enc = nn.GRU(input_size=3, hidden_size=h_dim, num_layers=1, batch_first=True)
                else:
                    self.enc = nn.Sequential(
                        _CausalConv1dBlock(3, h_dim, kernel_size=k_sz, dilation=1, dropout=drop),
                        _CausalConv1dBlock(h_dim, h_dim, kernel_size=k_sz, dilation=2, dropout=drop),
                    )
                self.head = nn.Sequential(
                    nn.Linear(h_dim + ctx_dim, h_dim),
                    nn.SiLU(),
                    nn.Linear(h_dim, b_dim),
                )
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

            def forward(self, seq_x: torch.Tensor, ctx_x: torch.Tensor) -> torch.Tensor:
                if self.mode == "gru":
                    _, h_n = self.enc(seq_x)
                    h_last = h_n[-1]
                else:
                    # Transpose [B, T, 3] -> [B, 3, T], run causal dilated convs, read final causal step [-1]
                    feat_t = self.enc(seq_x.transpose(1, 2))
                    h_last = feat_t[:, :, -1]
                return self.head(torch.cat([h_last, ctx_x], dim=-1))

        probe = _CausalSeqModel(c_tr.shape[1], arch)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        best_v = float(torch.mean(y_va ** 2).item())
        best_ep = 0
        bad = 0
        nrows = s_tr.shape[0]

        for ep in range(1, max_ep + 1):
            probe.train()
            perm = torch.randperm(nrows)
            for i0 in range(0, nrows, bsz):
                bi = perm[i0 : i0 + bsz]
                opt.zero_grad()
                loss = torch.mean((probe(s_tr[bi], c_tr[bi]) - y_tr[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            probe.eval()
            with torch.no_grad():
                vl = float(torch.mean((probe(s_va, c_va) - y_va) ** 2).item())
            if vl < best_v - 1e-5:
                best_v = vl
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        sf_tr, cf_tr, yf_tr = _assemble_seq(trials, outer_priors, fold_cache)
        torch.manual_seed(seed)
        final_net = _CausalSeqModel(cf_tr.shape[1], arch)
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            nf = sf_tr.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for i0 in range(0, nf, bsz):
                    bi = perm[i0 : i0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(sf_tr[bi], cf_tr[bi]) - yf_tr[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()
        self.state["seq_net"] = final_net
        self.state["selected_epochs"] = best_ep

    # ------------------------------------------------------------------
    # Family B6a: Causal POCO-FiLM & Univariate MS-MLP (Duan et al. NeurIPS 2025)
    # ------------------------------------------------------------------
    def _build_poco_inputs(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_train = priors["mu_train"]
        h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=(fold_cache if priors is (fold_cache or {}).get("priors") else None))
        stim_oh = _encode_stim_onehot(
            str(trial["stimulus_id"]), bool(trial.get("is_buffer_control", False)), priors["stim_vocab"]
        )
        univ = np.hstack([h_self, np.repeat(stim_oh[None, :], self.n_canon, axis=0)]).astype(np.float32)
        if self.model_id == "B6_poco_univariate_ms_mlp":
            pop = np.zeros((self.n_canon, 12), dtype=np.float32)
        else:
            pop = self._get_net_state(
                trial,
                s_n,
                mu_train,
                fold_cache=(fold_cache if priors is (fold_cache or {}).get("priors") else None),
                net_trial_override=net_trial_override,
            )
        return univ, pop

    def _fit_b6_poco_film(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        outer_priors: Mapping[str, Any],
        enable_pop_film: bool = True,
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        seed = int(self.hp.get("seed", 20261117))
        torch.manual_seed(seed)
        np.random.seed(seed)

        b_dim = int(outer_priors["b_dim"])
        f_dim = int(self.hp.get("poco_film_dim", 24))
        lr = float(self.hp.get("learning_rate", 0.002))
        wd = float(self.hp.get("weight_decay", 0.01))
        max_ep = min(25, int(self.hp.get("max_epochs", 30)))
        patience = int(self.hp.get("early_stopping_patience", 5))
        bsz = int(self.hp.get("batch_size", 512))

        worms = sorted({str(t["animal_id"]) for t in trials})
        val_worms = set(worms[-max(1, len(worms) // 5) :])
        inner_tr = [t for t in trials if str(t["animal_id"]) not in val_worms]
        inner_va = [t for t in trials if str(t["animal_id"]) in val_worms]
        if not inner_tr:
            inner_tr = list(trials)
            inner_va = list(trials[:1])

        inner_priors = fit_fold_neuron_stim_priors(
            inner_tr,
            self.n_canon,
            taus_sec=self.state["taus"],
            shrinkage_counts=float(self.hp.get("prior_shrinkage_counts", 5.0)),
        )
        # Keep identical stim_vocab width between inner and outer
        inner_priors["stim_vocab"] = list(outer_priors["stim_vocab"])

        def _assemble_poco(
            tr_list: Sequence[Mapping[str, Any]],
            pr_dict: Mapping[str, Any],
            f_cache: Optional[Mapping[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ul, pl, yl = [], [], []
            for t in tr_list:
                cn, rv, _, _ = self._get_projected_targets(t, pr_dict, s_n, fold_cache=f_cache)
                uf, pf = self._build_poco_inputs(t, s_n, pr_dict, fold_cache=f_cache)
                ix = np.where(rv)[0]
                if ix.size > 0:
                    ul.append(uf[ix])
                    pl.append(pf[ix])
                    yl.append(cn[ix])
            return (
                torch.from_numpy(np.vstack(ul)),
                torch.from_numpy(np.vstack(pl)),
                torch.from_numpy(np.vstack(yl)),
            )

        u_tr, p_tr, y_tr = _assemble_poco(inner_tr, inner_priors, None)
        u_va, p_va, y_va = _assemble_poco(inner_va, inner_priors, None)

        class _CausalPOCOFiLM(nn.Module):
            def __init__(self, d_univ: int, d_pop: int, use_film: bool) -> None:
                super().__init__()
                self.use_film = use_film
                self.univ_enc = nn.Sequential(nn.Linear(d_univ, f_dim), nn.LayerNorm(f_dim), nn.SiLU())
                self.film_gen = nn.Sequential(nn.Linear(d_pop, f_dim), nn.SiLU(), nn.Linear(f_dim, 2 * f_dim))
                nn.init.zeros_(self.film_gen[-1].weight)
                nn.init.zeros_(self.film_gen[-1].bias)
                self.out_head = nn.Sequential(nn.Linear(f_dim, f_dim), nn.SiLU(), nn.Linear(f_dim, b_dim))
                nn.init.zeros_(self.out_head[-1].weight)
                nn.init.zeros_(self.out_head[-1].bias)

            def forward(self, ux: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
                h = self.univ_enc(ux)
                if self.use_film:
                    gb = self.film_gen(px)
                    gamma, beta = torch.chunk(gb, 2, dim=-1)
                    h = (1.0 + 0.25 * torch.tanh(gamma)) * h + 0.25 * beta
                return self.out_head(h)

        probe = _CausalPOCOFiLM(u_tr.shape[1], p_tr.shape[1], enable_pop_film)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        best_v = float(torch.mean(y_va ** 2).item())
        best_ep = 0
        bad = 0
        nrows = u_tr.shape[0]
        for ep in range(1, max_ep + 1):
            probe.train()
            perm = torch.randperm(nrows)
            for i0 in range(0, nrows, bsz):
                bi = perm[i0 : i0 + bsz]
                opt.zero_grad()
                loss = torch.mean((probe(u_tr[bi], p_tr[bi]) - y_tr[bi]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            probe.eval()
            with torch.no_grad():
                vl = float(torch.mean((probe(u_va, p_va) - y_va) ** 2).item())
            if vl < best_v - 1e-5:
                best_v = vl
                best_ep = ep
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        uf_tr, pf_tr, yf_tr = _assemble_poco(trials, outer_priors, fold_cache)
        torch.manual_seed(seed)
        final_net = _CausalPOCOFiLM(uf_tr.shape[1], pf_tr.shape[1], enable_pop_film)
        if best_ep > 0:
            opt_f = torch.optim.AdamW(final_net.parameters(), lr=lr, weight_decay=wd)
            nf = uf_tr.shape[0]
            final_net.train()
            for _ in range(best_ep):
                perm = torch.randperm(nf)
                for i0 in range(0, nf, bsz):
                    bi = perm[i0 : i0 + bsz]
                    opt_f.zero_grad()
                    loss = torch.mean((final_net(uf_tr[bi], pf_tr[bi]) - yf_tr[bi]) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_net.parameters(), 1.0)
                    opt_f.step()
        final_net.eval()
        self.state["poco_net"] = final_net
        self.state["selected_epochs"] = best_ep

    # ------------------------------------------------------------------
    # Family B6b: Ray & Gordus (2025) Leaky-ODE Command-Derivative State-Gain
    # ------------------------------------------------------------------
    def _build_ray_gordus_features(
        self,
        trial: Mapping[str, Any],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
        net_trial_override: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        """Causal leaky-ODE state-gain baseline (`Ray & Gordus, Curr. Biol. 2025`):
        Combines leaky initial-condition self-decay + AVA/RIM level, causal derivative `dAVA/dt`,
        and `tanh(AVA - AVB)` reversal-attractor state multiplicatively gating multi-timescale stimulus drive.
        """
        mu_train = priors["mu_train"]
        h_self = self._get_h_self(trial, s_n, mu_train, fold_cache=fold_cache)
        net_st = self._get_net_state(
            trial, s_n, mu_train, fold_cache=fold_cache, net_trial_override=net_trial_override
        )
        # Columns 0, 1, 4, 5: AVA/RIM level, dAVA/dt derivative, bwd-fwd balance, tanh(bwd-fwd) attractor
        cmd = net_st[:, [0, 1, 4, 5]]
        _, prior_coef, _ = predict_prior_delta_trajectory(trial, priors, self.n_canon)
        drive_multi = prior_coef[:, :3] / s_n[:, None].astype(np.float32)  # Step + fast/med leaky ODE modes
        is_downstream = np.array(
            [1.0 if t in ("interneuron", "premotor_command") else 0.0 for t in self.functional_tiers],
            dtype=np.float32,
        )[:, None]
        # Outer-product multiplicative state-gain interactions on downstream interneurons/command cells
        rg_gate = (cmd[:, :, None] * drive_multi[:, None, :]).reshape(self.n_canon, -1) * is_downstream
        return np.hstack([h_self, cmd, drive_multi, rg_gate]).astype(np.float32)

    def _fit_b6_ray_gordus(
        self,
        trials: Sequence[Mapping[str, Any]],
        s_n: np.ndarray,
        priors: Mapping[str, Any],
        fold_cache: Optional[Mapping[str, Any]] = None,
    ) -> None:
        l2 = float(self.hp.get("l2_penalty", 25.0)) * 1.5
        xs, ys = [], []
        for tr in trials:
            cn, rv, _, _ = self._get_projected_targets(tr, priors, s_n, fold_cache=fold_cache)
            xr = self._build_ray_gordus_features(tr, s_n, priors, fold_cache=fold_cache)
            ix = np.where(rv)[0]
            if ix.size > 0:
                xs.append(xr[ix])
                ys.append(cn[ix])
        self.state["w_rg"] = _solve_ridge_sample_scaled(np.vstack(xs), np.vstack(ys), l2)


def build_fold_feature_cache(
    all_trials: Sequence[Mapping[str, Any]],
    train_trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    canonical_neurons: Sequence[str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Precompute strictly fold-internal features shared across baselines within a single CV fold."""
    hp = dict(config.get("hyperparameters", config))
    taus = tuple(float(x) for x in hp.get("basis_taus_sec", [0.75, 2.0, 5.0, 10.0]))
    shrink = float(hp.get("prior_shrinkage_counts", 5.0))
    lag_bins = tuple((float(a), float(b)) for a, b in hp.get("pre_ar_lags_sec", [[0.0, 1.0], [1.0, 2.5], [2.5, 5.0], [5.0, 10.0]]))
    n_canon = len(canonical_neurons)

    priors = fit_fold_neuron_stim_priors(train_trials, n_canon, taus_sec=taus, shrinkage_counts=shrink)
    mu_train = priors["mu_train"]

    h_self_map: Dict[str, np.ndarray] = {}
    net_state_map: Dict[str, np.ndarray] = {}
    proj_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    seq_map: Dict[str, np.ndarray] = {}

    for tr in all_trials:
        tid = str(tr["trial_id"])
        h_self_map[tid] = extract_self_history_features_matrix(tr, s_n, mu_train, lag_bins_sec=lag_bins)
        net_state_map[tid] = extract_causal_network_state_per_target(
            tr, canonical_neurons, s_n, mu_train, exclude_bilateral=True, preserve_sensory_asymmetry=False
        )
        proj_map[tid] = _project_trial_residual_targets(tr, priors, s_n, n_canon)

        xp = tr["x_pre"]
        mp = tr["m_pre"]
        fps = float(tr["fps"])
        y_anc = np.where(np.isfinite(tr["y_pre_anchor"]), tr["y_pre_anchor"], mu_train)
        seq = np.zeros((n_canon, 10, 3), dtype=np.float32)
        t_len = xp.shape[0]
        for b in range(10):
            f0 = min(max(0, t_len - 1), int(round(b * 1.0 * fps)))
            f1 = max(f0 + 1, min(t_len, int(round((b + 1) * 1.0 * fps))))
            sx = xp[f0:f1]
            sm = mp[f0:f1]
            cnt = np.sum(sm, axis=0)
            obs = cnt > 0
            if np.any(obs):
                mval = np.sum(np.where(sm[:, obs], sx[:, obs], 0.0), axis=0) / cnt[obs]
                seq[obs, b, 0] = np.clip((mval - y_anc[obs]) / s_n[obs], -5.0, 5.0)
                seq[obs, b, 1] = np.clip((y_anc[obs] - mu_train[obs]) / s_n[obs], -5.0, 5.0)
                seq[obs, b, 2] = 1.0
        seq_map[tid] = seq

    return {
        "priors": priors,
        "h_self": h_self_map,
        "net_state": net_state_map,
        "proj_targets": proj_map,
        "seq_pre": seq_map,
    }
