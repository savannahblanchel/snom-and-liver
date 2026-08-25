"""Physics-plus-ML SNOM fitting for processed organoid and tonsil spectra.

This is a self-contained inheritance of the senior semi-infinite workflow:

1. Generate synthetic spectra from a semi-infinite S-SNOM forward kernel.
2. Pretrain an MLP to predict bounded physical parameters from amplitude shape.
3. Use Latin-hypercube candidates and the MLP prediction as initial guesses.
4. Refine each experimental sample with gradient-based physical fitting.

The input is the existing sample-level summary. Point spectra are still
aggregated before this stage. Phase is retained for diagnostics and is excluded
from the main loss by default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
from scipy.stats import qmc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


EPS = 1e-8
DEFAULT_WMIN = 690.0
DEFAULT_WMAX = 1750.0
DEFAULT_STRIDE = 4


@dataclass(frozen=True)
class PlatformBounds:
    n_oscillators: int
    wt_min: float = 650.0
    wt_max: float = 1650.0
    gap_min: float = 20.0
    gap_max: float = 300.0
    gamma_min: float = 5.0
    gamma_max: float = 300.0
    epsinf_min: float = 0.1
    epsinf_max: float = 20.0
    gain_min: float = 0.05
    gain_max: float = 10.0
    phase_min: float = -math.pi
    phase_max: float = math.pi

    @property
    def dimension(self) -> int:
        return 3 * self.n_oscillators + 3

    def lower(self) -> np.ndarray:
        return np.asarray(
            [self.wt_min] * self.n_oscillators
            + [self.gap_min] * self.n_oscillators
            + [self.gamma_min] * self.n_oscillators
            + [self.epsinf_min, self.gain_min, self.phase_min],
            dtype=np.float64,
        )

    def upper(self) -> np.ndarray:
        return np.asarray(
            [self.wt_max] * self.n_oscillators
            + [self.gap_max] * self.n_oscillators
            + [self.gamma_max] * self.n_oscillators
            + [self.epsinf_max, self.gain_max, self.phase_max],
            dtype=np.float64,
        )

    def names(self) -> list[str]:
        return (
            [f"wt_{i + 1}" for i in range(self.n_oscillators)]
            + [f"gap_{i + 1}" for i in range(self.n_oscillators)]
            + [f"gamma_{i + 1}" for i in range(self.n_oscillators)]
            + ["eps_inf", "gain", "phase_shift"]
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sample_level(dataset_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    npz_path = dataset_dir / "sample_level_spectra.npz"
    csv_path = dataset_dir / "sample_level_summary.csv"
    if not npz_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"Expected sample_level_spectra.npz and sample_level_summary.csv in {dataset_dir}"
        )
    z = np.load(npz_path, allow_pickle=True)
    table = pd.read_csv(csv_path).fillna("")
    if len(table) != len(z["sample_id"]):
        raise ValueError("Sample table and sample-level spectra have different lengths")
    return dict(z), table


def select_window(w: np.ndarray, y: np.ndarray, wmin: float, wmax: float, stride: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (w >= wmin) & (w <= wmax)
    w_sel = np.asarray(w[mask], dtype=np.float64)
    y_sel = np.asarray(y[mask], dtype=np.float64)
    if stride > 1:
        return w_sel[::stride], y_sel[::stride]
    return w_sel, y_sel


def classify_sample(row: pd.Series) -> str:
    specimen = str(row.get("specimen_name", ""))
    sample_id = str(row.get("sample_id", ""))
    if "TA0038190758" in specimen or sample_id.upper().startswith("TA-"):
        return "organoid"
    if "类器官" in specimen or str(row.get("class_label", "")).lower() == "organoid":
        return "organoid"
    if "扁桃体" in specimen or str(row.get("class_label", "")).lower() == "tonsil":
        return "tonsil"
    return str(row.get("class_label", "unknown"))


def extract_features(signal: np.ndarray, feature_dim: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) >= 7:
        window = min(15, len(signal) if len(signal) % 2 else len(signal) - 1)
        window = max(5, window)
        smooth = savgol_filter(signal, window_length=window, polyorder=min(3, window - 2))
    else:
        smooth = signal
    center = np.median(np.abs(smooth)) + EPS
    normalized = (smooth - np.median(smooth)) / center
    x_old = np.linspace(0.0, 1.0, len(normalized))
    x_new = np.linspace(0.0, 1.0, feature_dim)
    return np.interp(x_new, x_old, normalized).astype(np.float32)


def latin_hypercube(bounds: PlatformBounds, n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=bounds.dimension, seed=seed)
    unit = sampler.random(n=n_samples)
    return qmc.scale(unit, bounds.lower(), bounds.upper()).astype(np.float32)


def to_unit(params: np.ndarray, bounds: PlatformBounds) -> np.ndarray:
    return (params - bounds.lower()) / (bounds.upper() - bounds.lower())


def from_unit(unit: torch.Tensor, bounds: PlatformBounds) -> torch.Tensor:
    lower = torch.as_tensor(bounds.lower(), dtype=unit.dtype, device=unit.device)
    upper = torch.as_tensor(bounds.upper(), dtype=unit.dtype, device=unit.device)
    return lower + unit * (upper - lower)


def phase0_row_to_candidate(row: pd.Series, bounds: PlatformBounds) -> np.ndarray:
    values: list[float] = []
    centers = [float(row.get(f"center_{i}", np.nan)) for i in range(1, bounds.n_oscillators + 1)]
    gammas = [float(row.get(f"gamma_{i}", np.nan)) for i in range(1, bounds.n_oscillators + 1)]
    strengths = [float(row.get(f"strength_{i}", np.nan)) for i in range(1, bounds.n_oscillators + 1)]
    for center in centers:
        if np.isfinite(center):
            values.append(float(np.clip(center, bounds.wt_min, bounds.wt_max)))
        else:
            values.append(float((bounds.wt_min + bounds.wt_max) / 2.0))
    for strength in strengths:
        if np.isfinite(strength):
            gap = 20.0 + 14.0 * strength
        else:
            gap = 80.0
        values.append(float(np.clip(gap, bounds.gap_min, bounds.gap_max)))
    for gamma in gammas:
        if np.isfinite(gamma):
            values.append(float(np.clip(gamma, bounds.gamma_min, bounds.gamma_max)))
        else:
            values.append(float((bounds.gamma_min + bounds.gamma_max) / 2.0))
    values.append(float(np.clip(float(row.get("eps_inf", 1.5)), bounds.epsinf_min, bounds.epsinf_max)))
    values.append(float(np.clip(float(row.get("gain", 1.0)), bounds.gain_min, bounds.gain_max)))
    values.append(float(np.clip(float(row.get("phase_shift", 0.0)), bounds.phase_min, bounds.phase_max)))
    return np.asarray(values, dtype=np.float32)


def load_phase0_batch_initials(
    phase0_fit_dir: Path | None, batch_name: str, bounds: PlatformBounds
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if phase0_fit_dir is None:
        return {}, {}
    batch_dir = phase0_fit_dir / batch_name
    csv_path = batch_dir / "sample_fit_results.csv"
    if not csv_path.exists():
        return {}, {}
    df = pd.read_csv(csv_path).fillna("")
    sample_map: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        sample_map[str(row["sample_id"])] = phase0_row_to_candidate(row, bounds)

    group_map: dict[str, np.ndarray] = {}
    group_cols = ["analysis_group", "specimen_name", "tune_wavenumber"]
    numeric_cols = [f"center_{i}" for i in range(1, bounds.n_oscillators + 1)]
    numeric_cols += [f"strength_{i}" for i in range(1, bounds.n_oscillators + 1)]
    numeric_cols += [f"gamma_{i}" for i in range(1, bounds.n_oscillators + 1)]
    numeric_cols += ["eps_inf", "gain", "phase_shift"]
    for analysis_group, group in df.groupby("analysis_group", sort=True):
        base = group.iloc[0].copy()
        for col in numeric_cols:
            base[col] = pd.to_numeric(group[col], errors="coerce").astype(float).mean()
        for col in group_cols:
            base[col] = group[col].iloc[0]
        group_map[str(analysis_group)] = phase0_row_to_candidate(base, bounds)
    return sample_map, group_map


class SeniorStyleSnomModel(nn.Module):
    """Differentiable semi-infinite tip model with senior-style reference normalization."""

    def __init__(self, bounds: PlatformBounds, n_time: int = 65, device: str = "cpu"):
        super().__init__()
        self.bounds = bounds
        self.n_time = n_time
        self.device_name = device
        self.register_buffer("tip_length", torch.tensor(300e-9, dtype=torch.float64))
        self.register_buffer("tip_radius", torch.tensor(33e-9, dtype=torch.float64))
        self.register_buffer("tip_amplitude", torch.tensor(62.944e-9, dtype=torch.float64))
        self.register_buffer("tip_frequency", torch.tensor(256100.066, dtype=torch.float64))
        self.register_buffer("reference_eps", torch.tensor(complex(-20.0, 5.0), dtype=torch.complex128))

    def decode(self, unit: torch.Tensor) -> dict[str, torch.Tensor]:
        params = from_unit(unit, self.bounds)
        n = self.bounds.n_oscillators
        return {
            "wt": params[..., :n],
            "gap": params[..., n : 2 * n],
            "gamma": params[..., 2 * n : 3 * n],
            "eps_inf": params[..., 3 * n],
            "gain": params[..., 3 * n + 1],
            "phase_shift": params[..., 3 * n + 2],
        }

    def encode(self, params: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not isinstance(params, torch.Tensor):
            params = torch.as_tensor(params, dtype=torch.float64)
        lower = torch.as_tensor(self.bounds.lower(), dtype=params.dtype, device=params.device)
        upper = torch.as_tensor(self.bounds.upper(), dtype=params.dtype, device=params.device)
        unit = (params - lower) / (upper - lower)
        unit = torch.clamp(unit, 1e-5, 1.0 - 1e-5)
        return torch.logit(unit)

    def _epsilon(self, w: torch.Tensor, decoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if w.ndim == 1:
            w = w.unsqueeze(0)
        wt = decoded["wt"].to(torch.float64)
        gap = decoded["gap"].to(torch.float64)
        gamma = decoded["gamma"].to(torch.float64)
        eps_inf = decoded["eps_inf"].to(torch.float64)
        wl = wt + gap
        wv = w.unsqueeze(-1)
        denom = wv**2 - wt.unsqueeze(1) ** 2 + 1j * gamma.unsqueeze(1) * wv
        terms = (wt.unsqueeze(1) ** 2 - wl.unsqueeze(1) ** 2) / (denom + 1e-12)
        return eps_inf.unsqueeze(1).to(torch.complex128) * (1.0 + terms.sum(dim=-1))

    def compute_reflectivity(self, eps: torch.Tensor, Au: bool = False) -> torch.Tensor:
        if Au:
            eps = self.reference_eps.to(dtype=torch.complex128, device=eps.device).expand_as(eps)
        root = torch.sqrt(eps)
        return (root - 1.0) / (root + 1.0 + 1e-12)

    def compute_beta(self, eps: torch.Tensor, Au: bool = False) -> torch.Tensor:
        if Au:
            eps = self.reference_eps.to(dtype=torch.complex128, device=eps.device).expand_as(eps)
        return (eps - 1.0) / (eps + 1.0 + 1e-12)

    def integral_sinf(
        self,
        q: torch.Tensor,
        reflectivity: torch.Tensor,
        beta: torch.Tensor,
        gain: torch.Tensor,
        phase_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.tensor(math.e, dtype=torch.float64, device=q.device)
        L = self.tip_length
        R = self.tip_radius
        Z = self.tip_amplitude
        f = self.tip_frequency
        omega = 2 * math.pi * f
        T = 1.0 / f

        t = torch.linspace(-T / 2, T / 2, steps=self.n_time, dtype=torch.float64, device=q.device)
        H = Z * (1.0 + torch.cos(omega * t))
        H = H.unsqueeze(0).unsqueeze(0)

        if q.ndim == 1:
            q = q.unsqueeze(-1)
        reflectivity = reflectivity.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        G = gain.to(torch.float64).unsqueeze(-1).unsqueeze(-1) * torch.exp(
            1j * phase_shift.to(torch.float64).unsqueeze(-1).unsqueeze(-1)
        )

        p1 = (R**2) * L * (2 * L / R + torch.log(R / (4 * e * L))) / torch.log(4 * L / (e**2))
        p2 = 2 + (
            beta
            * (G - (R + H) / L)
            * torch.log(4 * L / (4 * H + 3 * R))
        ) / (
            torch.log(4 * L / R)
            - beta
            * (G - (3 * R + 4 * H) / (4 * L))
            * torch.log(2 * L / (2 * H + R))
        )

        exp_term = torch.exp(-1j * 2 * omega * t).unsqueeze(0).unsqueeze(0)
        integrand = (p1 * p2) * exp_term
        s2 = omega * torch.trapz(integrand, t, dim=-1) / T

        c = torch.exp(-1j * 200 * math.pi * q.squeeze(-1) * Z * math.cos(math.pi / 3))
        signal = s2 * (1 + c * reflectivity.squeeze(-1)) ** 2
        return torch.angle(signal), torch.abs(signal)

    def compute_signal_sinf(self, w: torch.Tensor, decoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self._epsilon(w, decoded)
        r_sample = self.compute_reflectivity(eps, Au=False)
        beta_sample = self.compute_beta(eps, Au=False)

        ref_eps = self.reference_eps.to(dtype=torch.complex128, device=w.device).expand_as(eps)
        r_ref = self.compute_reflectivity(ref_eps, Au=True)
        beta_ref = self.compute_beta(ref_eps, Au=True)

        phase_sample, amp_sample = self.integral_sinf(
            w, r_sample, beta_sample, decoded["gain"], decoded["phase_shift"]
        )
        phase_ref, amp_ref = self.integral_sinf(
            w, r_ref, beta_ref, decoded["gain"], decoded["phase_shift"]
        )
        phase = torch.atan2(torch.sin(phase_sample - phase_ref), torch.cos(phase_sample - phase_ref))
        amp = amp_sample / (amp_ref + EPS)
        return phase, amp

    def forward(self, w: torch.Tensor, unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if unit.ndim == 1:
            unit = unit.unsqueeze(0)
        w = w.to(torch.float64)
        decoded = self.decode(torch.sigmoid(unit))
        return self.compute_signal_sinf(w, decoded)


class ParameterPredictor(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def amplitude_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    scale = torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS
    pred_norm = pred / scale
    true_norm = true / scale
    shape_pred = pred / (torch.median(torch.abs(pred), dim=-1, keepdim=True).values + EPS)
    shape_true = true / (torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS)
    return 0.7 * F.mse_loss(pred_norm, true_norm) + 0.3 * F.mse_loss(shape_pred, shape_true)


def generate_synthetic_training_data(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    bounds: PlatformBounds,
    n_samples: int,
    feature_dim: int,
    seed: int,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = latin_hypercube(bounds, n_samples=n_samples, seed=seed)
    unit = torch.as_tensor(to_unit(params, bounds), dtype=torch.float64)
    features: list[np.ndarray] = []
    units: list[np.ndarray] = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        with torch.no_grad():
            _, amp = model(w, model.encode(torch.as_tensor(params[start:end], dtype=torch.float64)))
        amp_np = amp.detach().cpu().numpy()
        for row in amp_np:
            features.append(extract_features(row, feature_dim))
        units.append(unit[start:end].cpu().numpy())
    return (
        torch.as_tensor(np.asarray(features), dtype=torch.float32),
        torch.as_tensor(np.concatenate(units), dtype=torch.float32),
    )


def pretrain_predictor(
    predictor: ParameterPredictor,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)
    split = max(1, int(0.8 * len(features)))
    train_idx, val_idx = indices[:split], indices[split:]
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        predictor.train()
        optimizer.zero_grad()
        pred = torch.sigmoid(predictor(features[train_idx]))
        loss = F.mse_loss(pred, targets[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        predictor.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(predictor(features[val_idx])) if len(val_idx) else pred
            val_loss = F.mse_loss(val_pred, targets[val_idx]) if len(val_idx) else loss
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "val_loss": float(val_loss.detach())})
    return history


def evaluate_candidates(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    candidate_params: np.ndarray,
) -> tuple[np.ndarray, int]:
    with torch.no_grad():
        logits = model.encode(torch.as_tensor(candidate_params, dtype=torch.float64))
        _, pred = model(w, logits)
        true = amp_true.unsqueeze(0).expand_as(pred)
        losses = torch.mean((pred - true) ** 2, dim=1).cpu().numpy()
    return losses, int(np.argmin(losses))


def refine_single(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    initial_params: np.ndarray,
    steps: int,
    learning_rate: float,
    phase_weight: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw = nn.Parameter(model.encode(initial_params).detach().clone())
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad()
        phase_pred, amp_pred = model(w, raw)
        loss = amplitude_loss(amp_pred, amp_true.unsqueeze(0))
        if phase_weight > 0:
            phase_res = torch.angle(torch.exp(1j * (phase_pred[0] - phase_true)))
            loss = loss + phase_weight * torch.mean(phase_res**2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_raw = raw.detach().clone()
    with torch.no_grad():
        phase_pred, amp_pred = model(w, best_raw)
        phase_res = torch.angle(torch.exp(1j * (phase_pred[0] - phase_true)))
        amp_mse = float(torch.mean((amp_pred[0] - amp_true) ** 2))
        phase_mse = float(torch.mean(phase_res**2))
        final_params = from_unit(torch.sigmoid(best_raw), model.bounds).detach().cpu().numpy()
    return final_params, amp_pred[0].detach().cpu().numpy(), amp_mse, phase_mse


def save_prediction_plot(
    out_path: Path,
    batch_name: str,
    records: list[dict[str, object]],
    w: np.ndarray,
) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(14, max(4.0, 2.6 * n)), sharex=True)
    if n == 1:
        axes = np.asarray([axes])
    for idx, record in enumerate(records):
        ax_amp, ax_phase = axes[idx]
        ax_amp.plot(w, record["amp_true"], color="black", lw=1.1, label="true")
        ax_amp.plot(w, record["amp_pred"], color="tab:blue", lw=1.0, label="physics+ML")
        ax_amp.set_ylabel("amp")
        ax_amp.set_title(f'{record["analysis_group"]} | {record["sample_id"]}', fontsize=9)
        ax_amp.grid(alpha=0.2)
        ax_phase.plot(w, record["phase_true"], color="black", lw=1.1, label="true")
        ax_phase.plot(w, record["phase_pred"], color="tab:orange", lw=1.0, label="physics+ML")
        ax_phase.set_ylabel("phase")
        ax_phase.grid(alpha=0.2)
        if idx == 0:
            ax_amp.legend(fontsize=8)
            ax_phase.legend(fontsize=8)
    axes[-1, 0].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.suptitle(f"{batch_name}: senior-style physics + ML fit", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_batch(
    batch_name: str,
    curves: list[dict[str, object]],
    output_dir: Path,
    w_np: np.ndarray,
    bounds: PlatformBounds,
    args: argparse.Namespace,
    device: torch.device,
    phase0_init_maps: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None,
) -> dict[str, object]:
    batch_dir = output_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    w = torch.as_tensor(w_np, dtype=torch.float64, device=device)
    model = SeniorStyleSnomModel(bounds, n_time=args.n_time).to(device)
    predictor = ParameterPredictor(args.feature_dim, bounds.dimension).to(device)

    synth_features, synth_targets = generate_synthetic_training_data(
        model=model,
        w=w,
        bounds=bounds,
        n_samples=args.synthetic_samples,
        feature_dim=args.feature_dim,
        seed=args.seed,
        batch_size=args.synthetic_batch_size,
    )
    synth_features = synth_features.to(device)
    synth_targets = synth_targets.to(device)
    history = pretrain_predictor(
        predictor,
        synth_features,
        synth_targets,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        seed=args.seed,
    )
    pd.DataFrame(history).to_csv(batch_dir / "ml_pretrain_history.csv", index=False)

    phase0_by_sample: dict[str, np.ndarray] = {}
    phase0_by_group: dict[str, np.ndarray] = {}
    if phase0_init_maps:
        phase0_by_sample, phase0_by_group = phase0_init_maps

    if phase0_by_sample or phase0_by_group:
        real_features: list[np.ndarray] = []
        real_targets: list[np.ndarray] = []
        for curve in curves:
            sample_id = str(curve["sample_id"])
            analysis_group = f'{curve["specimen_name"]}__{curve["tune_wavenumber"] or "na"}'
            phase0_target = phase0_by_sample.get(sample_id)
            if phase0_target is None:
                phase0_target = phase0_by_group.get(analysis_group)
            if phase0_target is None:
                continue
            real_features.append(extract_features(np.asarray(curve["amp"], dtype=np.float64), args.feature_dim))
            real_targets.append(to_unit(phase0_target, bounds))
        if real_features and args.phase0_warmup_epochs > 0:
            warm_features = torch.as_tensor(np.asarray(real_features), dtype=torch.float32).to(device)
            warm_targets = torch.as_tensor(np.asarray(real_targets), dtype=torch.float32).to(device)
            warm_history = pretrain_predictor(
                predictor,
                warm_features,
                warm_targets,
                epochs=args.phase0_warmup_epochs,
                learning_rate=args.phase0_warmup_lr,
                seed=args.seed + 97,
            )
            pd.DataFrame(warm_history).to_csv(batch_dir / "phase0_warmup_history.csv", index=False)

    rows: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []
    for curve in curves:
        amp_true_np = np.asarray(curve["amp"], dtype=np.float64)
        phase_true_np = np.asarray(curve["phase"], dtype=np.float64)
        amp_true = torch.as_tensor(amp_true_np, dtype=torch.float64, device=device)
        phase_true = torch.as_tensor(phase_true_np, dtype=torch.float64, device=device)
        feature = torch.as_tensor(extract_features(amp_true_np, args.feature_dim), dtype=torch.float32, device=device).unsqueeze(0)
        analysis_group = f'{curve["specimen_name"]}__{curve["tune_wavenumber"] or "na"}'
        phase0_candidate = phase0_by_sample.get(str(curve["sample_id"]))
        if phase0_candidate is None:
            phase0_candidate = phase0_by_group.get(analysis_group)
        predictor.eval()
        with torch.no_grad():
            nn_unit = torch.sigmoid(predictor(feature)).cpu().numpy()[0]
        nn_params = from_unit(torch.as_tensor(nn_unit, dtype=torch.float64), bounds).cpu().numpy()
        lhs_params = latin_hypercube(bounds, n_samples=args.init_candidates, seed=args.seed + len(rows) + 1)
        candidate_list = [nn_params]
        if phase0_candidate is not None:
            candidate_list.insert(0, phase0_candidate)
        candidate_list.extend(lhs_params)
        candidates = np.vstack(candidate_list)
        losses, best_idx = evaluate_candidates(model, w, amp_true, candidates)
        initial_params = candidates[best_idx]
        if phase0_candidate is not None:
            initial_source = "phase0" if best_idx == 0 else ("nn" if best_idx == 1 else "lhs")
        else:
            initial_source = "nn" if best_idx == 0 else "lhs"
        final_params, amp_pred_np, amp_mse, phase_mse = refine_single(
            model=model,
            w=w,
            amp_true=amp_true,
            phase_true=phase_true,
            initial_params=initial_params,
            steps=args.refine_steps,
            learning_rate=args.refine_lr,
            phase_weight=args.phase_weight,
        )
        with torch.no_grad():
            final_phase, _ = model(
                w,
                model.encode(torch.as_tensor(final_params, dtype=torch.float64, device=device)),
            )
        final_phase_np = final_phase[0].cpu().numpy()
        analysis_group = f'{curve["specimen_name"]}__{curve["tune_wavenumber"] or "na"}'
        row = {
            "batch": batch_name,
            "sample_id": curve["sample_id"],
            "analysis_group": analysis_group,
            "class_label": curve["class_label"],
            "original_class_label": curve["original_class_label"],
            "specimen_name": curve["specimen_name"],
            "marker": curve["marker"],
            "tune_wavenumber": curve["tune_wavenumber"],
            "n_points": curve["n_points"],
            "initial_source": initial_source,
            "initial_loss": float(losses[best_idx]),
            "amp_mse": amp_mse,
            "phase_circular_mse": phase_mse,
        }
        for name, value in zip(bounds.names(), final_params):
            row[name] = float(value)
        rows.append(row)
        plot_records.append(
            {
                "sample_id": curve["sample_id"],
                "analysis_group": analysis_group,
                "amp_true": amp_true_np,
                "amp_pred": amp_pred_np,
                "phase_true": phase_true_np,
                "phase_pred": final_phase_np,
            }
        )

    detail = pd.DataFrame(rows).sort_values(["analysis_group", "sample_id"]).reset_index(drop=True)
    detail.to_csv(batch_dir / "sample_fit_results.csv", index=False, encoding="utf-8-sig")
    stability_rows: list[dict[str, object]] = []
    for analysis_group, group in detail.groupby("analysis_group", sort=True):
        row = {
            "analysis_group": analysis_group,
            "class_label": group["class_label"].iloc[0],
            "specimen_name": group["specimen_name"].iloc[0],
            "tune_wavenumber": group["tune_wavenumber"].iloc[0],
            "n_samples": len(group),
            "amp_mse_mean": group["amp_mse"].mean(),
            "phase_circular_mse_mean": group["phase_circular_mse"].mean(),
        }
        for name in bounds.names():
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values))
            row[f"{name}_cv"] = float(np.std(values) / (abs(np.mean(values)) + EPS))
        stability_rows.append(row)
    pd.DataFrame(stability_rows).to_csv(batch_dir / "parameter_stability.csv", index=False, encoding="utf-8-sig")
    save_prediction_plot(batch_dir / f"{batch_name}_pred_vs_true.png", batch_name, plot_records, w_np)

    summary = {
        "batch": batch_name,
        "n_samples": len(detail),
        "n_analysis_groups": int(detail["analysis_group"].nunique()),
        "amp_mse_mean": float(detail["amp_mse"].mean()),
        "phase_circular_mse_mean": float(detail["phase_circular_mse"].mean()),
        "synthetic_samples": args.synthetic_samples,
        "pretrain_epochs": args.pretrain_epochs,
        "refine_steps": args.refine_steps,
        "phase_weight": args.phase_weight,
        "model": "senior-style semi-infinite multi-phonon + MLP",
    }
    (batch_dir / "platform_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Senior-style physics + ML SNOM fitting.")
    parser.add_argument("--dataset", required=True, help="Sample-level analysis directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wmin", type=float, default=DEFAULT_WMIN)
    parser.add_argument("--wmax", type=float, default=DEFAULT_WMAX)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--n-oscillators", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--synthetic-samples", type=int, default=256)
    parser.add_argument("--synthetic-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--phase0-fit-dir", default=None)
    parser.add_argument("--phase0-warmup-epochs", type=int, default=20)
    parser.add_argument("--phase0-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--init-candidates", type=int, default=24)
    parser.add_argument("--refine-steps", type=int, default=120)
    parser.add_argument("--refine-lr", type=float, default=3e-2)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--n-time", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    phase0_fit_dir = Path(args.phase0_fit_dir) if args.phase0_fit_dir else None
    z, table = load_sample_level(Path(args.dataset))
    w_full = np.asarray(z["wavenumber"], dtype=np.float64)
    amp_all = np.asarray(z["amp_mean"], dtype=np.float64)
    phase_all = np.asarray(z["phase_mean"], dtype=np.float64)
    w_np, _ = select_window(w_full, amp_all[0], args.wmin, args.wmax, args.stride)
    amp_curves = []
    for idx, row in table.iterrows():
        _, amp = select_window(w_full, amp_all[idx], args.wmin, args.wmax, args.stride)
        _, phase = select_window(w_full, phase_all[idx], args.wmin, args.wmax, args.stride)
        original_label = str(row.get("class_label", ""))
        amp_curves.append(
            {
                "sample_id": str(row["sample_id"]),
                "class_label": classify_sample(row),
                "original_class_label": original_label,
                "specimen_name": str(row.get("specimen_name", "")),
                "marker": str(row.get("marker", "")),
                "tune_wavenumber": str(row.get("tune_wavenumber", "")),
                "n_points": int(row.get("n_points", 0)),
                "amp": amp,
                "phase": phase,
            }
        )
    bounds = PlatformBounds(n_oscillators=args.n_oscillators)
    batches: dict[str, list[dict[str, object]]] = {}
    for curve in amp_curves:
        batch_name = f'group2_raw_reference__{curve["tune_wavenumber"] or "na"}'
        batches.setdefault(batch_name, []).append(curve)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for batch_name, curves in sorted(batches.items()):
        phase0_init_maps = load_phase0_batch_initials(phase0_fit_dir, batch_name, bounds)
        summaries.append(fit_batch(batch_name, curves, output_dir, w_np, bounds, args, device, phase0_init_maps))
    pd.DataFrame(summaries).to_csv(output_dir / "platform_fit_summary.csv", index=False, encoding="utf-8-sig")
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["class_rule"] = "TA0038190758 and TA-* are grouped as organoid"
    (output_dir / "platform_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
