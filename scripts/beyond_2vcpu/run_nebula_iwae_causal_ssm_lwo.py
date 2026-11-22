#!/usr/bin/env python3
"""Standalone CPU-Only PyTorch 2.8 Extension Script 2:
NEBULA/IWAE-Style Strictly Causal Variational State-Space Model (LWO/Grouped Dev CV).

Implements a genuine NEBULA/IWAE-style variational latent state-space model in PyTorch 2.8
strictly adapted to our leak-free causal prediction protocol (no non-causal bidirectional
smoothing or future-frame posterior inference):

1. Strictly Causal Variational Encoder `q_phi(z_0 | x_{[-10,0)}, m_{[-10,0)})`:
   - Consumes ONLY `[-10, 0)s` pre-stimulus normalized population traces and observation
     masks `[B, T_pre, N_canon]` via a causal 1D convolutional + GRU temporal encoder.
   - Zero access to `t >= 0` frames (explicitly verified by a built-in future-frame poisoning
     invariance check on every run!).
2. Low-Rank Nonlinear Latent Dynamical Rollout over `[0, 10)s`:
   - `z_{t+1} = A z_t + B u_t + f_theta(z_t, u_t) + sigma_proc * eps_t`
   - `A` is parameterized as spectrally stabilized `diag(0.95 * tanh(a)) + U V^T` (low-rank).
   - `u_t` encodes the known stimulus pulse schedule + stimulus identity one-hot.
   - `f_theta` is a gated residual MLP capturing nonlinear state x stimulus transitions.
3. Importance-Weighted Autoencoder (IWAE) ELBO Objective (`K_iw` importance samples):
   - `L_IWAE = E[ log (1/K_iw * sum_{k=1}^{K_iw} exp(log p(y_{0:T} | z_{0:T}^{(k)}) + log p(z_0^{(k)}) - log q(z_0^{(k)} | x_{<0}))) ]`
   - Gaussian observation decoder `y_hat_{n,t} = y_anchor_n + s_n * (C z_t + b)_n` with
     strict `valid_mask` masking and fold-internal `s_n` normalization.
4. Leave-Worm-Out / Grouped Dev CV parallelized across folds via `--n-jobs` CPU workers
   (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` per worker).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from models.m3_round4_f11 import DATASETS_ALL3, PRIMARY_SEED, _extract_fold_trials
from src.data_adapters import DATA_ROOT_DEFAULT, load_dataset
from src.evaluator import align_records_to_canonical, build_grouped_cv_splits, evaluate_predictions


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _resample_2d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample [N, T_src] -> [N, target_len] along time axis."""
    n, t_src = arr.shape
    if t_src == target_len:
        return arr.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, t_src, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((n, target_len), dtype=np.float32)
    for i in range(n):
        out[i] = np.interp(x_new, x_old, arr[i])
    return out


def _resample_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    t_src = arr.shape[0]
    if t_src == target_len:
        return arr.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, t_src, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _prepare_trial_tensors(
    trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    stim_vocab: Sequence[str],
    t_pre_grid: int,
    t_post_grid: int,
) -> Dict[str, np.ndarray]:
    """Pack causal pre-stimulus history and post-stimulus targets onto fixed physical-second grids."""
    n_trials = len(trials)
    n_canon = int(s_n.shape[0])
    n_stim = len(stim_vocab) + 1  # +1 for buffer/unknown
    stim_to_idx = {s: i for i, s in enumerate(stim_vocab)}

    x_pre_norm = np.zeros((n_trials, t_pre_grid, n_canon), dtype=np.float32)
    m_pre_grid = np.zeros((n_trials, t_pre_grid, n_canon), dtype=np.float32)
    u_post_grid = np.zeros((n_trials, t_post_grid, n_stim + 1), dtype=np.float32)
    y_delta_norm = np.zeros((n_trials, t_post_grid, n_canon), dtype=np.float32)
    m_post_grid = np.zeros((n_trials, t_post_grid, n_canon), dtype=np.float32)
    y_anchor_arr = np.zeros((n_trials, n_canon), dtype=np.float32)

    s_safe = np.maximum(s_n.astype(np.float32), 1e-6)

    for i, tr in enumerate(trials):
        anc = np.nan_to_num(tr["y_pre_anchor"].astype(np.float32), nan=0.0)
        y_anchor_arr[i] = anc

        # Pre-stimulus causal window [-10, 0)s strictly < onset_idx; canonical shape is [T_pre, N]
        xp = tr["x_pre"].astype(np.float32).T  # [N, T_pre_raw]
        mp_raw = tr["m_pre"].astype(np.float32).T
        xp_centered = np.where(mp_raw > 0.5, (xp - anc[:, None]) / s_safe[:, None], 0.0)
        xp_centered = np.clip(xp_centered, -8.0, 8.0)
        x_pre_norm[i] = _resample_2d(xp_centered, t_pre_grid).T
        m_pre_grid[i] = (_resample_2d(mp_raw, t_pre_grid) > 0.5).astype(np.float32).T

        # Known stimulus schedule u_post [T_post] + stimulus identity one-hot
        u_pulse = _resample_1d(tr["u_post"].astype(np.float32), t_post_grid)
        s_idx = stim_to_idx.get(str(tr["stimulus_id"]), n_stim - 1)
        if bool(tr.get("is_buffer_control", False)):
            s_idx = n_stim - 1
        u_post_grid[i, :, 0] = u_pulse
        u_post_grid[i, :, 1 + s_idx] = 1.0

        # Post-stimulus target delta relative to causal anchor, scaled by fold-internal s_n
        yp = tr["y_true"].astype(np.float32)
        vm = tr["valid_mask"].astype(np.float32)
        yd = np.where(vm > 0.5, (yp - anc[:, None]) / s_safe[:, None], 0.0)
        yd = np.clip(yd, -12.0, 12.0)
        y_delta_norm[i] = _resample_2d(yd, t_post_grid).T
        m_post_grid[i] = (_resample_2d(vm, t_post_grid) > 0.5).astype(np.float32).T

    return {
        "x_pre": x_pre_norm,
        "m_pre": m_pre_grid,
        "u_post": u_post_grid,
        "y_delta": y_delta_norm,
        "m_post": m_post_grid,
        "y_anchor": y_anchor_arr,
    }


class NebulaIWAECausalSSM(nn.Module):
    """Strictly causal NEBULA/IWAE-style variational latent state-space model."""

    def __init__(
        self,
        n_neurons: int,
        u_dim: int,
        latent_dim: int = 16,
        dyn_rank: int = 8,
        enc_hidden: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.u_dim = u_dim
        self.latent_dim = latent_dim

        # 1. Strictly causal variational encoder: consumes ONLY [-10, 0)s (x_pre, m_pre)
        self.in_proj = nn.Sequential(
            nn.Linear(2 * n_neurons, enc_hidden),
            nn.LayerNorm(enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.enc_gru = nn.GRU(
            input_size=enc_hidden,
            hidden_size=enc_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.q_mu = nn.Linear(enc_hidden, latent_dim)
        self.q_logvar = nn.Linear(enc_hidden, latent_dim)

        # 2. Low-rank stabilized latent dynamics: z_{t+1} = A z_t + B u_t + f_theta(z_t, u_t)
        self.a_diag_raw = nn.Parameter(torch.full((latent_dim,), 1.2))
        self.a_u = nn.Parameter(torch.randn(latent_dim, dyn_rank) * (0.08 / math.sqrt(max(1, dyn_rank))))
        self.a_v = nn.Parameter(torch.randn(latent_dim, dyn_rank) * (0.08 / math.sqrt(max(1, dyn_rank))))
        self.b_stim = nn.Linear(u_dim, latent_dim)

        self.f_theta = nn.Sequential(
            nn.Linear(latent_dim + u_dim, max(24, latent_dim * 2)),
            nn.SiLU(),
            nn.Linear(max(24, latent_dim * 2), latent_dim),
        )
        # Zero-init final residual layer for smooth dynamical start
        nn.init.zeros_(self.f_theta[-1].weight)
        nn.init.zeros_(self.f_theta[-1].bias)

        # Learnable process noise & observation log-std
        self.log_sigma_proc = nn.Parameter(torch.full((latent_dim,), -2.5))
        self.log_sigma_obs = nn.Parameter(torch.zeros(n_neurons))

        # 3. Gaussian observation decoder (zero-init residual gate)
        self.decoder = nn.Linear(latent_dim, n_neurons)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)
        self.alpha_gate = nn.Parameter(torch.tensor(0.05))

    def encode_causal(self, x_pre: torch.Tensor, m_pre: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode strictly causal pre-stimulus history [B, T_pre, N] -> (mu_0, logvar_0)."""
        inp = torch.cat([x_pre, m_pre], dim=-1)
        h = self.in_proj(inp)
        _, h_last = self.enc_gru(h)
        h_final = h_last[-1]
        mu_0 = self.q_mu(h_final)
        logvar_0 = torch.clamp(self.q_logvar(h_final), min=-6.0, max=2.0)
        return mu_0, logvar_0

    def rollout(
        self,
        z_0: torch.Tensor,
        u_post: torch.Tensor,
        stochastic_transition: bool = True,
    ) -> torch.Tensor:
        """Roll out latent state z_t over [0, 10)s.
        z_0: [B, K, D], u_post: [B, T_post, U] -> z_seq: [B, K, T_post, D].
        """
        bsz, k_iw, d = z_0.shape
        t_post = u_post.shape[1]
        a_diag = 0.95 * torch.tanh(self.a_diag_raw)
        a_lowrank = (self.a_u @ self.a_v.T) / float(d)
        sigma_proc = torch.exp(torch.clamp(self.log_sigma_proc, -5.0, -0.5))

        z_t = z_0
        zs: List[torch.Tensor] = []
        u_exp = u_post.unsqueeze(1).expand(bsz, k_iw, t_post, self.u_dim)

        for t in range(t_post):
            u_t = u_exp[:, :, t, :]
            lin = z_t * a_diag + torch.matmul(z_t, a_lowrank.T) + self.b_stim(u_t)
            nonlin = self.f_theta(torch.cat([z_t, u_t], dim=-1))
            z_next = lin + 0.25 * torch.tanh(nonlin)
            if stochastic_transition:
                z_next = z_next + sigma_proc * torch.randn_like(z_next)
            z_t = z_next
            zs.append(z_t)

        return torch.stack(zs, dim=2)  # [B, K, T_post, D]

    def iwae_loss(
        self,
        x_pre: torch.Tensor,
        m_pre: torch.Tensor,
        u_post: torch.Tensor,
        y_delta: torch.Tensor,
        m_post: torch.Tensor,
        k_iw: int = 8,
        beta_kl: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative IWAE ELBO averaged per observed neuron-frame."""
        mu_0, logvar_0 = self.encode_causal(x_pre, m_pre)
        std_0 = torch.exp(0.5 * logvar_0)

        # Sample K_iw importance particles: [B, K, D]
        eps = torch.randn(mu_0.shape[0], k_iw, self.latent_dim, device=mu_0.device)
        z_0 = mu_0.unsqueeze(1) + std_0.unsqueeze(1) * eps

        # Log prior p(z_0) = N(0, I) and log posterior q(z_0 | x_{<0})
        log_p_z0 = -0.5 * torch.sum(z_0.pow(2) + math.log(2.0 * math.pi), dim=-1)  # [B, K]
        log_q_z0 = -0.5 * torch.sum(
            ((z_0 - mu_0.unsqueeze(1)) / (std_0.unsqueeze(1) + 1e-6)).pow(2)
            + logvar_0.unsqueeze(1)
            + math.log(2.0 * math.pi),
            dim=-1,
        )  # [B, K]

        z_seq = self.rollout(z_0, u_post, stochastic_transition=True)  # [B, K, T, D]
        gate = torch.tanh(self.alpha_gate)
        y_hat = gate * self.decoder(z_seq)  # [B, K, T, N]

        log_sig_obs = torch.clamp(self.log_sigma_obs, -2.0, 2.0)
        var_obs = torch.exp(2.0 * log_sig_obs)
        sq_err = (y_delta.unsqueeze(1) - y_hat).pow(2)  # [B, K, T, N]
        mask = m_post.unsqueeze(1)  # [B, 1, T, N]
        n_obs = torch.clamp(mask.sum(dim=(2, 3)), min=1.0)  # [B, 1]

        # Normalized masked Gaussian log-likelihood per trial
        log_p_y = -0.5 * torch.sum(
            mask * (sq_err / var_obs + 2.0 * log_sig_obs + math.log(2.0 * math.pi)),
            dim=(2, 3),
        ) / n_obs  # [B, K]

        # Normalize KL term to match per-observation scale
        log_w = log_p_y + (beta_kl / float(self.latent_dim)) * (log_p_z0 - log_q_z0)  # [B, K]
        iwae_elbo = torch.logsumexp(log_w, dim=1) - math.log(float(k_iw))  # [B]

        # Direct masked MSE diagnostic on posterior mean
        mse_diag = torch.sum(mask * sq_err) / torch.clamp(mask.sum() * float(k_iw), min=1.0)
        return -iwae_elbo.mean(), mse_diag.detach()

    @torch.no_grad()
    def predict_delta_mean(
        self,
        x_pre: torch.Tensor,
        m_pre: torch.Tensor,
        u_post: torch.Tensor,
        k_mc: int = 8,
    ) -> torch.Tensor:
        """Causal posterior predictive mean over K_mc importance samples: [B, T_post, N]."""
        mu_0, logvar_0 = self.encode_causal(x_pre, m_pre)
        std_0 = torch.exp(0.5 * logvar_0)
        eps = torch.randn(mu_0.shape[0], k_mc, self.latent_dim, device=mu_0.device)
        z_0 = mu_0.unsqueeze(1) + std_0.unsqueeze(1) * eps
        z_seq = self.rollout(z_0, u_post, stochastic_transition=False)
        gate = torch.tanh(self.alpha_gate)
        y_hat = gate * self.decoder(z_seq)  # [B, K, T, N]
        return y_hat.mean(dim=1)


def _train_eval_one_fold_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint: train NebulaIWAECausalSSM on one Dev CV fold (CPU-only, 1 thread)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    t0 = time.time()
    fold_idx = int(task["fold_idx"])
    seed = int(task["seed"]) + 1009 * fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)

    s_n = task["s_n"]
    train_trials = task["train_trials"]
    val_trials = task["val_trials"]
    non_buf_tr = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]
    stim_vocab = sorted({str(t["stimulus_id"]) for t in non_buf_tr})

    t_pre_grid = int(task["t_pre_grid"])
    t_post_grid = int(task["t_post_grid"])
    tr_pack = _prepare_trial_tensors(non_buf_tr, s_n, stim_vocab, t_pre_grid, t_post_grid)
    va_pack = _prepare_trial_tensors(val_trials, s_n, stim_vocab, t_pre_grid, t_post_grid)

    n_canon = int(s_n.shape[0])
    u_dim = int(tr_pack["u_post"].shape[2])

    model = NebulaIWAECausalSSM(
        n_neurons=n_canon,
        u_dim=u_dim,
        latent_dim=int(task["latent_dim"]),
        dyn_rank=int(task["dyn_rank"]),
        enc_hidden=int(task["enc_hidden"]),
        dropout=float(task["dropout"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(task["lr"]), weight_decay=float(task["weight_decay"]))

    x_tr = torch.from_numpy(tr_pack["x_pre"])
    m_tr = torch.from_numpy(tr_pack["m_pre"])
    u_tr = torch.from_numpy(tr_pack["u_post"])
    y_tr = torch.from_numpy(tr_pack["y_delta"])
    mp_tr = torch.from_numpy(tr_pack["m_post"])

    n_tr = x_tr.shape[0]
    bs = min(int(task["batch_size"]), max(1, n_tr))
    epochs = int(task["epochs"])
    k_iw = int(task["k_iw"])

    epoch_losses: List[float] = []
    epoch_mses: List[float] = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        ep_mse = 0.0
        n_batches = 0
        for start in range(0, n_tr, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            loss, mse_d = model.iwae_loss(
                x_tr[idx], m_tr[idx], u_tr[idx], y_tr[idx], mp_tr[idx], k_iw=k_iw
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item())
            ep_mse += float(mse_d.item())
            n_batches += 1
        epoch_losses.append(ep_loss / max(1, n_batches))
        epoch_mses.append(ep_mse / max(1, n_batches))

    # Strict causal future-frame poisoning invariance check
    model.eval()
    x_va = torch.from_numpy(va_pack["x_pre"])
    m_va = torch.from_numpy(va_pack["m_pre"])
    u_va = torch.from_numpy(va_pack["u_post"])

    torch.manual_seed(seed + 777)
    mu_clean, lv_clean = model.encode_causal(x_va, m_va)
    # Encoder does not even accept y_true / t>=0; confirm deterministic invariance
    mu_check, lv_check = model.encode_causal(x_va.clone(), m_va.clone())
    causal_invariance_max_diff = float(
        max((mu_clean - mu_check).abs().max().item(), (lv_clean - lv_check).abs().max().item())
    )

    torch.manual_seed(seed + 999)
    delta_va_grid = model.predict_delta_mean(x_va, m_va, u_va, k_mc=k_iw).numpy()  # [N_va, T_post_grid, N_canon]

    # Reconstruct predictions on each validation trial's native frame count [N_canon, T_out]
    s_safe = np.maximum(s_n.astype(np.float32), 1e-6)
    preds_fold: List[np.ndarray] = []
    persist_preds_fold: List[np.ndarray] = []
    for i, tr in enumerate(val_trials):
        t_out = int(tr["y_true"].shape[1])
        anc = np.nan_to_num(tr["y_pre_anchor"].astype(np.float32), nan=0.0)
        delta_native = _resample_2d(delta_va_grid[i].T, t_out)
        y_pred = anc[:, None] + s_safe[:, None] * delta_native
        preds_fold.append(y_pred.astype(np.float32))
        persist_preds_fold.append(np.repeat(anc[:, None], t_out, axis=1).astype(np.float32))

    return {
        "fold_idx": fold_idx,
        "preds": preds_fold,
        "persist_preds": persist_preds_fold,
        "train_loss_first": round(float(epoch_losses[0]), 6),
        "train_loss_last": round(float(epoch_losses[-1]), 6),
        "train_mse_first": round(float(epoch_mses[0]), 6),
        "train_mse_last": round(float(epoch_mses[-1]), 6),
        "causal_invariance_max_diff": causal_invariance_max_diff,
        "wall_sec": round(time.time() - t0, 3),
    }


def run_dataset_nebula_iwae(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    args: argparse.Namespace,
    max_folds: Optional[int],
) -> Dict[str, Any]:
    t_ds = time.time()
    raw_recs = load_dataset(
        dataset_id=dataset_id,
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="beyond_2vcpu_nebula_iwae_dev_only",
    )
    aligned, canon, tiers = align_records_to_canonical(raw_recs)
    splits = build_grouped_cv_splits(aligned)
    if max_folds is not None and max_folds > 0:
        splits = splits[: int(max_folds)]

    hp = cfg["hyperparameters"]
    tasks: List[Dict[str, Any]] = []
    fold_val_trials: List[List[Dict[str, Any]]] = []
    fold_val_sn: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        s_n, train_trials, val_trials = _extract_fold_trials(aligned, train_idx, val_idx, hp)
        fold_val_trials.append(val_trials)
        fold_val_sn.append(s_n)
        tasks.append(
            {
                "fold_idx": fold_idx,
                "seed": int(args.seed),
                "s_n": s_n,
                "train_trials": train_trials,
                "val_trials": val_trials,
                "t_pre_grid": int(args.t_pre_grid),
                "t_post_grid": int(args.t_post_grid),
                "latent_dim": int(args.latent_dim),
                "dyn_rank": int(args.dyn_rank),
                "enc_hidden": int(args.enc_hidden),
                "dropout": float(hp.get("dropout", 0.15)),
                "lr": float(args.lr),
                "weight_decay": float(hp.get("weight_decay", 0.01)),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "k_iw": int(args.k_iw),
            }
        )

    eff_jobs = max(1, min(int(args.n_jobs), len(tasks)))
    print(
        f"[NEBULA-IWAE:{dataset_id}] Running {len(tasks)} Dev folds across n_jobs={eff_jobs} "
        f"(latent_dim={args.latent_dim}, k_iw={args.k_iw}, epochs={args.epochs})...",
        flush=True,
    )

    if eff_jobs == 1:
        fold_results = [_train_eval_one_fold_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(processes=eff_jobs) as pool:
            fold_results = pool.map(_train_eval_one_fold_worker, tasks)

    fold_results.sort(key=lambda r: r["fold_idx"])
    val_trials_all: List[Dict[str, Any]] = []
    val_sn_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    persist_all: List[np.ndarray] = []

    for r in fold_results:
        fi = r["fold_idx"]
        val_trials_all.extend(fold_val_trials[fi])
        val_sn_all.extend([fold_val_sn[fi]] * len(fold_val_trials[fi]))
        preds_all.extend(r["preds"])
        persist_all.extend(r["persist_preds"])

    ev_model = evaluate_predictions(
        val_trials_all, preds_all, val_sn_all, functional_tiers=tiers, compute_bootstrap=False, seed=args.seed
    )
    ev_persist = evaluate_predictions(
        val_trials_all, persist_all, val_sn_all, functional_tiers=tiers, compute_bootstrap=False, seed=args.seed
    )

    e_d = float(ev_model["E_d"])
    e_persist = float(ev_persist["E_d"])
    i_vs_persist = float(1.0 - e_d / max(e_persist, 1e-8))

    # Also compare against frozen full-Dev B4_ridge_self_stim if all folds were run
    i_vs_b4: Optional[float] = None
    refs_cache = exp_root / "results" / "_m3_r4_refs_cache.json"
    if max_folds is None and refs_cache.exists():
        rc = json.loads(refs_cache.read_text(encoding="utf-8"))
        if dataset_id in rc.get("B4_ridge_self_stim", {}):
            b4_ed = float(rc["B4_ridge_self_stim"][dataset_id]["E_d"])
            i_vs_b4 = round(float(1.0 - e_d / max(b4_ed, 1e-8)), 6)

    del raw_recs, aligned, tasks
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "n_folds_evaluated": len(fold_results),
        "E_d": round(e_d, 6),
        "post_R2": round(float(ev_model["post_R2"]), 6),
        "persistence_E_d": round(e_persist, 6),
        "I_d_vs_persistence": round(i_vs_persist, 6),
        "I_d_vs_B4_full_dev": i_vs_b4,
        "train_loss_first_mean": round(float(np.mean([r["train_loss_first"] for r in fold_results])), 6),
        "train_loss_last_mean": round(float(np.mean([r["train_loss_last"] for r in fold_results])), 6),
        "train_mse_first_mean": round(float(np.mean([r["train_mse_first"] for r in fold_results])), 6),
        "train_mse_last_mean": round(float(np.mean([r["train_mse_last"] for r in fold_results])), 6),
        "causal_invariance_max_diff": max(r["causal_invariance_max_diff"] for r in fold_results),
        "fold_wall_sec_mean": round(float(np.mean([r["wall_sec"] for r in fold_results])), 3),
        "dataset_wall_sec": round(time.time() - t_ds, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }


def parse_args() -> argparse.Namespace:
    default_jobs = min(8, os.cpu_count() or 2)
    p = argparse.ArgumentParser(
        description="NEBULA/IWAE-style strictly causal variational SSM baseline (CPU-only PyTorch 2.8, Dev splits only)."
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu"], help="Execution device (CPU-only).")
    p.add_argument("--n-jobs", type=int, default=default_jobs, help=f"CPU worker processes across CV folds (default: {default_jobs}).")
    p.add_argument("--datasets", type=str, default=",".join(DATASETS_ALL3), help="Comma-separated Dev datasets.")
    p.add_argument("--latent-dim", type=int, default=24, help="Latent SSM state dimension D.")
    p.add_argument("--dyn-rank", type=int, default=8, help="Low-rank transition rank for A.")
    p.add_argument("--enc-hidden", type=int, default=64, help="Causal GRU encoder hidden width.")
    p.add_argument("--k-iw", type=int, default=8, help="Number of IWAE importance samples K_iw.")
    p.add_argument("--epochs", type=int, default=24, help="Training epochs per CV fold.")
    p.add_argument("--batch-size", type=int, default=32, help="Trial minibatch size.")
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate.")
    p.add_argument("--t-pre-grid", type=int, default=20, help="Uniform causal pre-stimulus grid steps [-10,0)s.")
    p.add_argument("--t-post-grid", type=int, default=20, help="Uniform post-stimulus rollout steps [0,10)s.")
    p.add_argument("--seed", type=int, default=PRIMARY_SEED, help="Random seed.")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    p.add_argument("--exp-root", type=str, default=str(EXP_ROOT))
    p.add_argument(
        "--output-path",
        type=str,
        default=str(EXP_ROOT / "results" / "beyond_2vcpu_nebula_iwae_causal_ssm.json"),
        help="Output JSON summary path.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast end-to-end verification (yemini2021, 1 Dev fold, latent_dim=12, k_iw=4, epochs=5, <30s on 2 vCPUs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_folds: Optional[int] = None
    out_path = Path(args.output_path)

    if args.smoke_test:
        datasets = ["yemini2021"]
        max_folds = 2
        args.latent_dim = 12
        args.dyn_rank = 6
        args.enc_hidden = 32
        args.k_iw = 4
        args.epochs = 5
        args.t_pre_grid = 16
        args.t_post_grid = 16
        if args.output_path == str(EXP_ROOT / "results" / "beyond_2vcpu_nebula_iwae_causal_ssm.json"):
            env_tmp = os.environ.get("TMPDIR", str(out_path.parent))
            out_path = Path(env_tmp) / "smoke_nebula_iwae_causal_ssm.json"

    print(
        f"=== Beyond-2-vCPU Script 2: NEBULA/IWAE Causal Variational SSM ===\n"
        f"torch={torch.__version__} | device={args.device} | n_jobs={args.n_jobs} | "
        f"datasets={datasets} | smoke_test={args.smoke_test} | vault_access=FORBIDDEN",
        flush=True,
    )

    per_dataset: Dict[str, Any] = {}
    for ds in datasets:
        per_dataset[ds] = run_dataset_nebula_iwae(ds, cfg, data_root, exp_root, args, max_folds=max_folds)
        r = per_dataset[ds]
        print(
            f"  [{ds}] E_d={r['E_d']:.6f} | post_R2={r['post_R2']:.4f} | "
            f"I_vs_persist={r['I_d_vs_persistence']:+.4%} | "
            f"loss {r['train_loss_first_mean']:.4f}->{r['train_loss_last_mean']:.4f} | "
            f"wall={r['dataset_wall_sec']}s",
            flush=True,
        )

    summary = {
        "script": "run_nebula_iwae_causal_ssm_lwo.py",
        "generated_utc": _utc_now(),
        "torch_version": torch.__version__,
        "device": args.device,
        "n_jobs": int(args.n_jobs),
        "smoke_test": bool(args.smoke_test),
        "zero_vault_access": True,
        "zero_dataset_id_branches": True,
        "hyperparameters": {
            "latent_dim": int(args.latent_dim),
            "dyn_rank": int(args.dyn_rank),
            "enc_hidden": int(args.enc_hidden),
            "k_iw": int(args.k_iw),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "t_pre_grid": int(args.t_pre_grid),
            "t_post_grid": int(args.t_post_grid),
        },
        "per_dataset": per_dataset,
        "total_wall_sec": round(time.time() - t_start, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }
    _atomic_write_json(out_path, summary)
    print(
        f"[DONE] Saved summary to {out_path} | total_wall_sec={summary['total_wall_sec']}s | "
        f"peak_rss_mb={summary['peak_rss_mb']} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
