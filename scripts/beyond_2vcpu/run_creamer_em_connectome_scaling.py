#!/usr/bin/env python3
"""Standalone CPU-Only PyTorch 2.8 Extension Script 3:
Creamer-Style EM-Connectome Continuous-Time Biophysical ODE-SSM Scaling Sweep.

Parameterizes signed Cook 2019 (`W_chem_pos`, `W_chem_neg`, `W_chem_unk`, `W_gap`) and
Ripoll-Sánchez 2023 (`W_npp_short`, `W_npp_mid`, `W_monoamine`) connectome couplings
inside a continuous-time damped RK2 (Heun) biophysical calcium ODE-SSM over `[0, 10)s`:

    tau_i * dx_i/dt = -x_i + sum_m g_m (W_m * S_sign) phi(x) + (U V^T) phi(x) + B u(t)

with:
- Learnable neuron-class + per-neuron calcium time constants `tau_i = tau_min + softplus(...)`,
- Learnable per-layer biophysical coupling gains `g_m` across the 6 wired + wireless layers,
- Optional low-rank unmapped/extrasynaptic residual coupling `U V^T` swept over `--ranks`,
- Zero-initialized residual gate `tanh(alpha_gate)` anchored on the causal pre-stimulus state,
- Multi-seed (`--n-seeds 20`) and rank/hidden-dim scaling sweeps parallelized across CPU
  workers via `--n-jobs` (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` per worker).

Strictly Dev splits ONLY (`allow_vault_unseal=False`; zero `vault_sealed/` access).
"""

from __future__ import annotations

import argparse
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
from models.m35_round5_nonlinear import TIER_NAMES
from src.data_adapters import DATA_ROOT_DEFAULT, load_dataset
from src.evaluator import align_records_to_canonical, build_grouped_cv_splits, evaluate_predictions
from src.features import _get_allow_mask_matrix


LAYER_NAMES: Tuple[str, ...] = (
    "cook_chem_pos",
    "cook_chem_neg",
    "cook_chem_unk",
    "cook_gap",
    "rippoll_npp_short_mid",
    "rippoll_monoamine",
)


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


def _row_normalize(mat: np.ndarray, allow_mask: np.ndarray) -> np.ndarray:
    m = mat.astype(np.float32) * allow_mask.astype(np.float32)
    row_sum = np.sum(np.abs(m), axis=1, keepdims=True)
    return np.divide(m, np.maximum(row_sum, 1.0), dtype=np.float32)


def load_em_biophysical_connectome(
    canon_neurons: Sequence[str],
    data_root: Path,
) -> np.ndarray:
    """Load signed Cook 2019 + Ripoll-Sánchez 2023 matrices aligned to `canon_neurons`: [6, N, N]."""
    bundle_path = Path(data_root) / "connectomes" / "connectome_priors_bundle.npz"
    with np.load(bundle_path, allow_pickle=False) as z:
        b_neurons = [str(x) for x in z["canonical_neurons"].tolist()]
        idx = np.array([b_neurons.index(n) for n in canon_neurons], dtype=np.int64)

        w_chem = z["cook2019_herm_chem"][idx][:, idx].astype(np.float32)
        w_gap = z["cook2019_herm_gap"][idx][:, idx].astype(np.float32)
        s_sign = z["edge_polarity"][idx][:, idx].astype(np.float32)
        w_npp_s = z["ripollsanchez2023_short"][idx][:, idx].astype(np.float32)
        w_npp_m = z["ripollsanchez2023_mid"][idx][:, idx].astype(np.float32)
        w_mono = z["monoamine"][idx][:, idx].astype(np.float32)

    allow_mask = _get_allow_mask_matrix(
        canon_neurons, exclude_bilateral=True, preserve_sensory_asymmetry=True
    ).astype(np.float32)
    # Signed chemical decomposition (preserving receptor polarity sign S_sign)
    w_chem_pos = np.where(s_sign > 0.5, w_chem, 0.0)
    w_chem_neg = -np.where(s_sign < -0.5, w_chem, 0.0)  # inhibitory negative sign
    w_chem_unk = np.where(np.abs(s_sign) <= 0.5, w_chem, 0.0)

    layers = [
        _row_normalize(w_chem_pos, allow_mask),
        _row_normalize(w_chem_neg, allow_mask),
        _row_normalize(w_chem_unk, allow_mask),
        _row_normalize(w_gap, allow_mask),
        _row_normalize(0.5 * (w_npp_s + w_npp_m), allow_mask),
        _row_normalize(w_mono, allow_mask),
    ]
    return np.stack(layers, axis=0).astype(np.float32)  # [6, N, N]


def _prepare_ode_trials(
    trials: Sequence[Mapping[str, Any]],
    s_n: np.ndarray,
    stim_vocab: Sequence[str],
    t_steps: int,
) -> Dict[str, np.ndarray]:
    """Extract causal initial condition x_0 (from [-2,0)s & [-10,0)s), stimulus u(t), and post targets."""
    n_trials = len(trials)
    n_canon = int(s_n.shape[0])
    n_stim = len(stim_vocab) + 1
    stim_to_idx = {s: i for i, s in enumerate(stim_vocab)}
    s_safe = np.maximum(s_n.astype(np.float32), 1e-6)

    x0_arr = np.zeros((n_trials, n_canon), dtype=np.float32)
    dx0_arr = np.zeros((n_trials, n_canon), dtype=np.float32)
    u_arr = np.zeros((n_trials, t_steps, n_stim + 1), dtype=np.float32)
    y_delta = np.zeros((n_trials, t_steps, n_canon), dtype=np.float32)
    m_post = np.zeros((n_trials, t_steps, n_canon), dtype=np.float32)

    for i, tr in enumerate(trials):
        anc = np.nan_to_num(tr["y_pre_anchor"].astype(np.float32), nan=0.0)
        mean10 = np.nan_to_num(tr["y_pre_mean"].astype(np.float32), nan=0.0)
        has_pre = tr["has_pre"].astype(np.float32)

        # Causal initial state deviation & pre-stimulus trend strictly from [-10, 0)s ([T_pre, N])
        x0_arr[i] = np.clip(has_pre * (anc - mean10) / s_safe, -6.0, 6.0)
        xp = tr["x_pre"].astype(np.float32).T  # [N, T_pre]
        mp = tr["m_pre"].astype(np.float32).T
        t_pre = xp.shape[1]
        half = max(1, t_pre // 2)
        early = np.sum(np.where(mp[:, :half] > 0.5, xp[:, :half], 0.0), axis=1) / np.maximum(mp[:, :half].sum(axis=1), 1.0)
        late = np.sum(np.where(mp[:, half:] > 0.5, xp[:, half:], 0.0), axis=1) / np.maximum(mp[:, half:].sum(axis=1), 1.0)
        dx0_arr[i] = np.clip(has_pre * (late - early) / s_safe, -6.0, 6.0)

        u_pulse = _resample_1d(tr["u_post"].astype(np.float32), t_steps)
        s_idx = stim_to_idx.get(str(tr["stimulus_id"]), n_stim - 1)
        if bool(tr.get("is_buffer_control", False)):
            s_idx = n_stim - 1
        u_arr[i, :, 0] = u_pulse
        u_arr[i, :, 1 + s_idx] = 1.0

        yp = tr["y_true"].astype(np.float32)
        vm = tr["valid_mask"].astype(np.float32)
        yd = np.where(vm > 0.5, (yp - anc[:, None]) / s_safe[:, None], 0.0)
        yd = np.clip(yd, -12.0, 12.0)
        y_delta[i] = _resample_2d(yd, t_steps).T
        m_post[i] = (_resample_2d(vm, t_steps) > 0.5).astype(np.float32).T

    return {
        "x0": x0_arr,
        "dx0": dx0_arr,
        "u_post": u_arr,
        "y_delta": y_delta,
        "m_post": m_post,
    }


class CreamerEMConnectomeODESSM(nn.Module):
    """Continuous-time damped RK2 biophysical calcium ODE-SSM with Cook 2019 + Ripoll-Sánchez 2023 priors."""

    def __init__(
        self,
        w_layers: np.ndarray,  # [6, N, N]
        tier_indices: np.ndarray,  # [N]
        u_dim: int,
        rank: int = 8,
        dt_sec: float = 0.5,
        tau_min: float = 0.5,
    ) -> None:
        super().__init__()
        n_layers, n_neurons, _ = w_layers.shape
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.rank = int(rank)
        self.dt_sec = float(dt_sec)
        self.tau_min = float(tau_min)

        self.register_buffer("w_layers", torch.from_numpy(w_layers))  # [6, N, N]
        self.register_buffer("tier_idx", torch.from_numpy(tier_indices.astype(np.int64)))

        # Learnable neuron-class + per-neuron time constants (initialized around tau ~ 2.5s)
        self.tau_class_raw = nn.Parameter(torch.full((len(TIER_NAMES),), 1.8))
        self.tau_neuron_delta = nn.Parameter(torch.zeros(n_neurons))

        # Per-layer biophysical coupling gains g_m
        self.layer_gains = nn.Parameter(torch.full((n_layers,), 0.15))

        # Optional low-rank unmapped residual coupling U V^T
        if self.rank > 0:
            scale = 0.05 / math.sqrt(max(1, self.rank))
            self.u_lowrank = nn.Parameter(torch.randn(n_neurons, self.rank) * scale)
            self.v_lowrank = nn.Parameter(torch.randn(n_neurons, self.rank) * scale)
        else:
            self.register_parameter("u_lowrank", None)
            self.register_parameter("v_lowrank", None)

        # Stimulus drive B u(t) + initial-velocity injection from causal pre-stimulus trend
        self.b_stim = nn.Linear(u_dim, n_neurons)
        nn.init.normal_(self.b_stim.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.b_stim.bias)
        self.w_dx0 = nn.Parameter(torch.full((n_neurons,), 0.1))

        # Near-zero-init residual gate so ODE starts near causal anchor while allowing immediate gradient flow
        self.alpha_gate = nn.Parameter(torch.tensor(0.05))

    def get_tau(self) -> torch.Tensor:
        raw = self.tau_class_raw[self.tier_idx] + 0.25 * torch.tanh(self.tau_neuron_delta)
        return self.tau_min + F.softplus(raw)  # [N], strictly >= tau_min

    def _effective_w(self) -> torch.Tensor:
        # Weighted sum of the 6 signed EM/wireless connectome matrices: [N, N]
        gains = torch.tanh(self.layer_gains).view(self.n_layers, 1, 1)
        w_eff = torch.sum(gains * self.w_layers, dim=0)
        if self.rank > 0 and self.u_lowrank is not None and self.v_lowrank is not None:
            w_eff = w_eff + (self.u_lowrank @ self.v_lowrank.T) / float(self.n_neurons)
        return w_eff

    def _rhs(self, x: torch.Tensor, u_drive: torch.Tensor, w_eff: torch.Tensor, inv_tau: torch.Tensor) -> torch.Tensor:
        # tau_i dx_i/dt = -x_i + W_eff @ tanh(x) + u_drive
        syn = torch.matmul(torch.tanh(x), w_eff.T)
        return (-x + syn + u_drive) * inv_tau

    def forward(self, x0: torch.Tensor, dx0: torch.Tensor, u_post: torch.Tensor) -> torch.Tensor:
        """Integrate biophysical ODE via RK2 (Heun) over T_steps: returns predicted delta [B, T, N]."""
        bsz, t_steps, _ = u_post.shape
        tau = self.get_tau()
        inv_tau = (1.0 / tau).unsqueeze(0)  # [1, N]
        w_eff = self._effective_w()  # [N, N]
        u_drive_all = self.b_stim(u_post)  # [B, T, N]

        x = x0 + self.w_dx0.unsqueeze(0) * dx0
        x_init = x
        dt_s = self.dt_sec
        traj: List[torch.Tensor] = []

        for t in range(t_steps):
            u_t = u_drive_all[:, t, :]
            k1 = self._rhs(x, u_t, w_eff, inv_tau)
            x_euler = x + dt_s * k1
            u_next = u_drive_all[:, min(t + 1, t_steps - 1), :]
            k2 = self._rhs(x_euler, u_next, w_eff, inv_tau)
            x = x + 0.5 * dt_s * (k1 + k2)
            traj.append(x - x_init)

        delta_seq = torch.stack(traj, dim=1)  # [B, T, N]
        gate = torch.tanh(self.alpha_gate)
        return gate * delta_seq


def _train_eval_ode_fold_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint: train CreamerEMConnectomeODESSM for one (rank, seed, fold) on 1 CPU thread."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    t0 = time.time()
    fold_idx = int(task["fold_idx"])
    seed = int(task["seed"])
    rank = int(task["rank"])
    torch.manual_seed(seed + 997 * fold_idx)
    np.random.seed(seed + 997 * fold_idx)

    s_n = task["s_n"]
    train_trials = task["train_trials"]
    val_trials = task["val_trials"]
    non_buf_tr = [t for t in train_trials if not bool(t.get("is_buffer_control", False))]
    stim_vocab = sorted({str(t["stimulus_id"]) for t in non_buf_tr})

    t_steps = int(task["t_steps"])
    dt_sec = 10.0 / float(t_steps)
    tr_pack = _prepare_ode_trials(non_buf_tr, s_n, stim_vocab, t_steps)
    va_pack = _prepare_ode_trials(val_trials, s_n, stim_vocab, t_steps)

    model = CreamerEMConnectomeODESSM(
        w_layers=task["w_layers"],
        tier_indices=task["tier_indices"],
        u_dim=int(tr_pack["u_post"].shape[2]),
        rank=rank,
        dt_sec=dt_sec,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(task["lr"]), weight_decay=float(task["weight_decay"]))

    x0_tr = torch.from_numpy(tr_pack["x0"])
    dx0_tr = torch.from_numpy(tr_pack["dx0"])
    u_tr = torch.from_numpy(tr_pack["u_post"])
    y_tr = torch.from_numpy(tr_pack["y_delta"])
    m_tr = torch.from_numpy(tr_pack["m_post"])

    n_tr = x0_tr.shape[0]
    bs = min(int(task["batch_size"]), max(1, n_tr))
    epochs = int(task["epochs"])

    losses: List[float] = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        nb = 0
        for start in range(0, n_tr, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            pred = model(x0_tr[idx], dx0_tr[idx], u_tr[idx])
            mask = m_tr[idx]
            loss = torch.sum(mask * (pred - y_tr[idx]).pow(2)) / torch.clamp(mask.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        losses.append(ep_loss / max(1, nb))

    model.eval()
    with torch.no_grad():
        delta_va = model(
            torch.from_numpy(va_pack["x0"]),
            torch.from_numpy(va_pack["dx0"]),
            torch.from_numpy(va_pack["u_post"]),
        ).numpy()
        learned_gains = {
            LAYER_NAMES[m]: round(float(torch.tanh(model.layer_gains[m]).item()), 4)
            for m in range(len(LAYER_NAMES))
        }
        learned_tau_class = {
            TIER_NAMES[i]: round(float((model.tau_min + F.softplus(model.tau_class_raw[i])).item()), 3)
            for i in range(len(TIER_NAMES))
        }
        alpha_val = round(float(torch.tanh(model.alpha_gate).item()), 4)

    s_safe = np.maximum(s_n.astype(np.float32), 1e-6)
    preds_fold: List[np.ndarray] = []
    persist_fold: List[np.ndarray] = []
    for i, tr in enumerate(val_trials):
        t_out = int(tr["y_true"].shape[1])
        anc = np.nan_to_num(tr["y_pre_anchor"].astype(np.float32), nan=0.0)
        d_nat = _resample_2d(delta_va[i].T, t_out)
        preds_fold.append((anc[:, None] + s_safe[:, None] * d_nat).astype(np.float32))
        persist_fold.append(np.repeat(anc[:, None], t_out, axis=1).astype(np.float32))

    return {
        "rank": rank,
        "seed": seed,
        "fold_idx": fold_idx,
        "preds": preds_fold,
        "persist_preds": persist_fold,
        "train_loss_first": round(float(losses[0]), 6),
        "train_loss_last": round(float(losses[-1]), 6),
        "alpha_gate": alpha_val,
        "learned_gains": learned_gains,
        "learned_tau_class": learned_tau_class,
        "wall_sec": round(time.time() - t0, 3),
    }


def run_dataset_creamer_scaling(
    dataset_id: str,
    cfg: Mapping[str, Any],
    data_root: Path,
    exp_root: Path,
    args: argparse.Namespace,
    ranks: Sequence[int],
    seeds: Sequence[int],
    max_folds: Optional[int],
) -> Dict[str, Any]:
    t_ds = time.time()
    raw_recs = load_dataset(
        dataset_id=dataset_id,
        split="dev",
        allow_vault_unseal=False,
        data_root=data_root,
        exp_root=exp_root,
        caller_reason="beyond_2vcpu_creamer_em_ode_dev_only",
    )
    aligned, canon, tiers = align_records_to_canonical(raw_recs)
    splits = build_grouped_cv_splits(aligned)
    if max_folds is not None and max_folds > 0:
        splits = splits[: int(max_folds)]

    w_layers = load_em_biophysical_connectome(canon, data_root=data_root)
    tier_indices = np.array([TIER_NAMES.index(t) if t in TIER_NAMES else 0 for t in tiers], dtype=np.int64)
    hp = cfg["hyperparameters"]

    fold_data: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for train_idx, val_idx in splits:
        fold_data.append(_extract_fold_trials(aligned, train_idx, val_idx, hp))

    tasks: List[Dict[str, Any]] = []
    for rank in ranks:
        for seed in seeds:
            for fold_idx, (s_n, tr_trials, va_trials) in enumerate(fold_data):
                tasks.append(
                    {
                        "rank": int(rank),
                        "seed": int(seed),
                        "fold_idx": fold_idx,
                        "s_n": s_n,
                        "train_trials": tr_trials,
                        "val_trials": va_trials,
                        "w_layers": w_layers,
                        "tier_indices": tier_indices,
                        "t_steps": int(args.t_steps),
                        "lr": float(args.lr),
                        "weight_decay": float(hp.get("weight_decay", 0.01)),
                        "batch_size": int(args.batch_size),
                        "epochs": int(args.epochs),
                    }
                )

    eff_jobs = max(1, min(int(args.n_jobs), len(tasks)))
    print(
        f"[Creamer-EM-ODE:{dataset_id}] Running {len(tasks)} tasks "
        f"(ranks={list(ranks)}, n_seeds={len(seeds)}, folds={len(fold_data)}) across n_jobs={eff_jobs}...",
        flush=True,
    )

    if eff_jobs == 1:
        results = [_train_eval_ode_fold_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(processes=eff_jobs) as pool:
            results = pool.map(_train_eval_ode_fold_worker, tasks)

    # Aggregate by (rank, seed)
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault((r["rank"], r["seed"]), []).append(r)

    refs_cache = exp_root / "results" / "_m3_r4_refs_cache.json"
    b4_ed: Optional[float] = None
    if max_folds is None and refs_cache.exists():
        rc = json.loads(refs_cache.read_text(encoding="utf-8"))
        if dataset_id in rc.get("B4_ridge_self_stim", {}):
            b4_ed = float(rc["B4_ridge_self_stim"][dataset_id]["E_d"])

    by_rank: Dict[str, Any] = {}
    for rank in ranks:
        seed_eds: List[float] = []
        seed_r2s: List[float] = []
        seed_i_persist: List[float] = []
        seed_loss_first: List[float] = []
        seed_loss_last: List[float] = []
        sample_gains: Dict[str, float] = {}
        sample_tau: Dict[str, float] = {}

        for seed in seeds:
            f_recs = sorted(grouped[(int(rank), int(seed))], key=lambda x: x["fold_idx"])
            val_trials_all, val_sn_all, preds_all, persist_all = [], [], [], []
            for fr in f_recs:
                fi = fr["fold_idx"]
                s_n_f, _, va_f = fold_data[fi]
                val_trials_all.extend(va_f)
                val_sn_all.extend([s_n_f] * len(va_f))
                preds_all.extend(fr["preds"])
                persist_all.extend(fr["persist_preds"])

            ev_m = evaluate_predictions(val_trials_all, preds_all, val_sn_all, functional_tiers=tiers, compute_bootstrap=False, seed=seed)
            ev_p = evaluate_predictions(val_trials_all, persist_all, val_sn_all, functional_tiers=tiers, compute_bootstrap=False, seed=seed)
            ed_val = float(ev_m["E_d"])
            ep_val = float(ev_p["E_d"])
            seed_eds.append(ed_val)
            seed_r2s.append(float(ev_m["post_R2"]))
            seed_i_persist.append(float(1.0 - ed_val / max(ep_val, 1e-8)))
            seed_loss_first.append(float(np.mean([fr["train_loss_first"] for fr in f_recs])))
            seed_loss_last.append(float(np.mean([fr["train_loss_last"] for fr in f_recs])))
            sample_gains = f_recs[0]["learned_gains"]
            sample_tau = f_recs[0]["learned_tau_class"]

        ed_mean = float(np.mean(seed_eds))
        by_rank[f"rank_{rank}"] = {
            "rank": int(rank),
            "n_seeds": len(seeds),
            "E_d_mean": round(ed_mean, 6),
            "E_d_std": round(float(np.std(seed_eds, ddof=1)) if len(seed_eds) > 1 else 0.0, 6),
            "post_R2_mean": round(float(np.mean(seed_r2s)), 6),
            "I_d_vs_persistence_mean": round(float(np.mean(seed_i_persist)), 6),
            "I_d_vs_B4_full_dev": round(float(1.0 - ed_mean / max(b4_ed, 1e-8)), 6) if b4_ed is not None else None,
            "train_loss_first_mean": round(float(np.mean(seed_loss_first)), 6),
            "train_loss_last_mean": round(float(np.mean(seed_loss_last)), 6),
            "sample_learned_layer_gains": sample_gains,
            "sample_learned_tau_class_sec": sample_tau,
        }

    del raw_recs, aligned, tasks
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "n_folds_evaluated": len(fold_data),
        "by_rank": by_rank,
        "task_wall_sec_mean": round(float(np.mean([r["wall_sec"] for r in results])), 3),
        "dataset_wall_sec": round(time.time() - t_ds, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }


def parse_args() -> argparse.Namespace:
    default_jobs = min(8, os.cpu_count() or 2)
    p = argparse.ArgumentParser(
        description="Creamer-style EM-connectome continuous-time biophysical ODE-SSM scaling sweep (CPU-only PyTorch 2.8)."
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu"], help="Execution device (CPU-only).")
    p.add_argument("--n-jobs", type=int, default=default_jobs, help=f"CPU worker processes (default: {default_jobs}).")
    p.add_argument("--datasets", type=str, default=",".join(DATASETS_ALL3), help="Comma-separated Dev datasets.")
    p.add_argument("--n-seeds", type=int, default=20, help="Number of random seeds per rank (default: 20).")
    p.add_argument("--ranks", type=str, default="0,8,16", help="Comma-separated low-rank residual dims (default: 0,8,16).")
    p.add_argument("--t-steps", type=int, default=20, help="RK2 integration steps over [0, 10)s (dt=10/t_steps).")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs per (rank, seed, fold).")
    p.add_argument("--batch-size", type=int, default=32, help="Trial minibatch size.")
    p.add_argument("--lr", type=float, default=5e-3, help="AdamW learning rate.")
    p.add_argument("--base-seed", type=int, default=PRIMARY_SEED, help="Base seed.")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    p.add_argument("--exp-root", type=str, default=str(EXP_ROOT))
    p.add_argument(
        "--output-path",
        type=str,
        default=str(EXP_ROOT / "results" / "beyond_2vcpu_creamer_em_ode_scaling.json"),
        help="Output JSON summary path.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast end-to-end verification (yemini2021, 1 Dev fold, n_seeds=2, ranks=8, epochs=6, <30s on 2 vCPUs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    cfg = json.loads((exp_root / "configs" / "lambda_global.json").read_text(encoding="utf-8"))

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    n_seeds = int(args.n_seeds)
    max_folds: Optional[int] = None
    out_path = Path(args.output_path)

    if args.smoke_test:
        datasets = ["yemini2021"]
        ranks = [8]
        n_seeds = 2
        max_folds = 1
        args.t_steps = 16
        args.epochs = 6
        if args.output_path == str(EXP_ROOT / "results" / "beyond_2vcpu_creamer_em_ode_scaling.json"):
            env_tmp = os.environ.get("TMPDIR", str(out_path.parent))
            out_path = Path(env_tmp) / "smoke_creamer_em_ode_scaling.json"

    seeds = [int(args.base_seed) + i for i in range(n_seeds)]

    print(
        f"=== Beyond-2-vCPU Script 3: Creamer-Style EM-Connectome Biophysical ODE-SSM ===\n"
        f"torch={torch.__version__} | device={args.device} | n_jobs={args.n_jobs} | "
        f"datasets={datasets} | ranks={ranks} | n_seeds={n_seeds} | smoke_test={args.smoke_test} | vault_access=FORBIDDEN",
        flush=True,
    )

    per_dataset: Dict[str, Any] = {}
    for ds in datasets:
        per_dataset[ds] = run_dataset_creamer_scaling(
            ds, cfg, data_root, exp_root, args, ranks=ranks, seeds=seeds, max_folds=max_folds
        )
        for rk, rv in per_dataset[ds]["by_rank"].items():
            print(
                f"  [{ds} | {rk}] E_d={rv['E_d_mean']:.6f}±{rv['E_d_std']:.6f} | "
                f"post_R2={rv['post_R2_mean']:.4f} | I_vs_persist={rv['I_d_vs_persistence_mean']:+.4%} | "
                f"loss {rv['train_loss_first_mean']:.4f}->{rv['train_loss_last_mean']:.4f}",
                flush=True,
            )

    summary = {
        "script": "run_creamer_em_connectome_scaling.py",
        "generated_utc": _utc_now(),
        "torch_version": torch.__version__,
        "device": args.device,
        "n_jobs": int(args.n_jobs),
        "smoke_test": bool(args.smoke_test),
        "zero_vault_access": True,
        "zero_dataset_id_branches": True,
        "ranks": ranks,
        "n_seeds": n_seeds,
        "t_steps": int(args.t_steps),
        "epochs": int(args.epochs),
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
