"""Physics-plus-ML SNOM fitting for processed organoid and tonsil spectra.

This is a self-contained inheritance of the senior semi-infinite workflow:

1. Generate synthetic spectra from a semi-infinite S-SNOM forward kernel.
2. Pretrain an MLP to predict bounded physical parameters from amplitude/phase shape.
3. Use Latin-hypercube candidates and the MLP prediction as initial guesses.
4. Refine each experimental sample with gradient-based physical fitting.

The input is the existing sample-level summary. Point spectra are still
aggregated before this stage. The main route fits amplitude and phase together;
amplitude scale and phase offset are kept as nuisance calibration parameters.
"""

from __future__ import annotations

import argparse
import copy
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
    epsinf_min: float = 1.0
    epsinf_max: float = 10.0
    log_amp_scale_min: float = -8.0
    log_amp_scale_max: float = 24.0
    phase_offset_min: float = -math.pi
    phase_offset_max: float = math.pi

    @property
    def dimension(self) -> int:
        return 3 * self.n_oscillators + 3

    def lower(self) -> np.ndarray:
        return np.asarray(
            [self.wt_min] * self.n_oscillators
            + [self.gap_min] * self.n_oscillators
            + [self.gamma_min] * self.n_oscillators
            + [
                self.epsinf_min,
                self.log_amp_scale_min,
                self.phase_offset_min,
            ],
            dtype=np.float64,
        )

    def upper(self) -> np.ndarray:
        return np.asarray(
            [self.wt_max] * self.n_oscillators
            + [self.gap_max] * self.n_oscillators
            + [self.gamma_max] * self.n_oscillators
            + [
                self.epsinf_max,
                self.log_amp_scale_max,
                self.phase_offset_max,
            ],
            dtype=np.float64,
        )

    def names(self) -> list[str]:
        return (
            [f"wT_{i + 1}" for i in range(self.n_oscillators)]
            + [f"gap_{i + 1}" for i in range(self.n_oscillators)]
            + [f"gamma_{i + 1}" for i in range(self.n_oscillators)]
            + ["eps_inf", "log_amp_scale", "phase_offset"]
        )


@dataclass(frozen=True)
class LiteratureBand:
    label: str
    band_min: float
    band_max: float

    @property
    def center(self) -> float:
        return 0.5 * (self.band_min + self.band_max)


LIVER_FTIR_BANDS: tuple[LiteratureBand, ...] = (
    LiteratureBand("glycogen_nucleic_1020_1090", 1020.0, 1090.0),
    LiteratureBand("glycogen_1140_1165", 1140.0, 1165.0),
    LiteratureBand("collagen_amide_1170_1210", 1170.0, 1210.0),
    LiteratureBand("collagen_amide_1220_1285", 1220.0, 1285.0),
    LiteratureBand("collagen_1300_1350", 1300.0, 1350.0),
    LiteratureBand("protein_coo_1380_1410", 1380.0, 1410.0),
    LiteratureBand("amide_II_1535_1565", 1535.0, 1565.0),
    LiteratureBand("amide_I_1635_1670", 1635.0, 1670.0),
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
    if feature_dim <= 0:
        return np.zeros(0, dtype=np.float32)
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


def extract_spectrum_features(amp: np.ndarray, phase: np.ndarray, feature_dim: int) -> np.ndarray:
    amp_dim = max(1, feature_dim // 2)
    phase_dim = max(0, feature_dim - amp_dim)
    sin_dim = phase_dim // 2
    cos_dim = phase_dim - sin_dim
    parts = [extract_features(amp, amp_dim)]
    if sin_dim:
        parts.append(extract_features(np.sin(phase), sin_dim))
    if cos_dim:
        parts.append(extract_features(np.cos(phase), cos_dim))
    return np.concatenate(parts).astype(np.float32)


def latin_hypercube(bounds: PlatformBounds, n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=bounds.dimension, seed=seed)
    unit = sampler.random(n=n_samples)
    params = qmc.scale(unit, bounds.lower(), bounds.upper())
    return canonicalize_oscillator_order(params, bounds).astype(np.float32)


def to_unit(params: np.ndarray, bounds: PlatformBounds) -> np.ndarray:
    return (params - bounds.lower()) / (bounds.upper() - bounds.lower())


def from_unit(unit: torch.Tensor, bounds: PlatformBounds) -> torch.Tensor:
    lower = torch.as_tensor(bounds.lower(), dtype=unit.dtype, device=unit.device)
    upper = torch.as_tensor(bounds.upper(), dtype=unit.dtype, device=unit.device)
    return lower + unit * (upper - lower)


def canonicalize_oscillator_order(params: np.ndarray, bounds: PlatformBounds) -> np.ndarray:
    arr = np.asarray(params, dtype=np.float64).copy()
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    n = bounds.n_oscillators
    for row in arr:
        order = np.argsort(row[:n])
        row[:n] = row[:n][order]
        row[n : 2 * n] = row[n : 2 * n][order]
        row[2 * n : 3 * n] = row[2 * n : 3 * n][order]
    return arr[0] if was_1d else arr


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
    gain = max(float(row.get("gain", 1.0)), EPS)
    values.append(float(np.clip(math.log(gain), bounds.log_amp_scale_min, bounds.log_amp_scale_max)))
    values.append(float(np.clip(float(row.get("phase_shift", 0.0)), bounds.phase_offset_min, bounds.phase_offset_max)))
    return canonicalize_oscillator_order(np.asarray(values, dtype=np.float32), bounds).astype(np.float32)


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

    def __init__(
        self,
        bounds: PlatformBounds,
        n_time: int = 65,
        device: str = "cpu",
        reference_material: str = "sio2",
        fixed_g_factor: float = 0.5,
        fixed_g_phase: float = 0.03,
        matched_bg_amp: np.ndarray | None = None,
        matched_bg_phase: np.ndarray | None = None,
    ):
        super().__init__()
        self.bounds = bounds
        self.n_time = n_time
        self.device_name = device
        self.reference_material = reference_material.lower()
        self.register_buffer("tip_length", torch.tensor(300e-9, dtype=torch.float64))
        self.register_buffer("tip_radius", torch.tensor(33e-9, dtype=torch.float64))
        self.register_buffer("tip_amplitude", torch.tensor(62.944e-9, dtype=torch.float64))
        self.register_buffer("tip_frequency", torch.tensor(256100.066, dtype=torch.float64))
        self.register_buffer("fixed_g_factor", torch.tensor(fixed_g_factor, dtype=torch.float64))
        self.register_buffer("fixed_g_phase", torch.tensor(fixed_g_phase, dtype=torch.float64))
        self.register_buffer("reference_eps_au", torch.tensor(complex(-20.0, 5.0), dtype=torch.complex128))
        self.register_buffer("reference_eps_si", torch.tensor(complex(13.0, 0.0), dtype=torch.complex128))
        self.register_buffer("sio2_eps_inf", torch.tensor(2.0, dtype=torch.float64))
        self.register_buffer("sio2_wt", torch.tensor([450.0, 800.0, 1045.0], dtype=torch.float64))
        self.register_buffer("sio2_wl", torch.tensor([505.0, 830.0, 1240.0], dtype=torch.float64))
        self.register_buffer("sio2_gamma", torch.tensor([51.0, 10.0, 10.0], dtype=torch.float64))
        if self.reference_material == "matched-bg":
            if matched_bg_amp is None or matched_bg_phase is None:
                raise ValueError("reference-material matched-bg requires bg_amp_mean/bg_phase_mean in the dataset")
            matched_bg_amp_arr = np.asarray(matched_bg_amp, dtype=np.float64)
            bg_scale = np.nanmedian(np.abs(matched_bg_amp_arr)) + EPS
            self.register_buffer("matched_bg_amp", torch.as_tensor(matched_bg_amp_arr / bg_scale, dtype=torch.float64))
            self.register_buffer("matched_bg_phase", torch.as_tensor(matched_bg_phase, dtype=torch.float64))
        else:
            self.register_buffer("matched_bg_amp", torch.empty(0, dtype=torch.float64))
            self.register_buffer("matched_bg_phase", torch.empty(0, dtype=torch.float64))

    def decode(self, unit: torch.Tensor) -> dict[str, torch.Tensor]:
        params = from_unit(unit, self.bounds)
        n = self.bounds.n_oscillators
        return {
            "wt": params[..., :n],
            "gap": params[..., n : 2 * n],
            "gamma": params[..., 2 * n : 3 * n],
            "eps_inf": params[..., 3 * n],
            "g_factor": self.fixed_g_factor.expand_as(params[..., 3 * n]),
            "g_phase": self.fixed_g_phase.expand_as(params[..., 3 * n]),
            "log_amp_scale": params[..., 3 * n + 1],
            "amp_scale": torch.exp(params[..., 3 * n + 1]),
            "phase_offset": params[..., 3 * n + 2],
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

    def _sio2_epsilon(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim == 1:
            w = w.unsqueeze(0)
        w = w.to(torch.float64)
        wt = self.sio2_wt.to(device=w.device)
        wl = self.sio2_wl.to(device=w.device)
        gamma = self.sio2_gamma.to(device=w.device)
        eps = self.sio2_eps_inf.to(device=w.device).to(torch.complex128) * torch.ones_like(
            w, dtype=torch.complex128
        )
        for wt_i, wl_i, gamma_i in zip(wt, wl, gamma):
            eps = eps * (w**2 - wl_i**2 + 1j * gamma_i * w) / (w**2 - wt_i**2 + 1j * gamma_i * w + 1e-12)
        return eps

    def reference_epsilon(self, w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        material = self.reference_material
        if material == "au":
            return self.reference_eps_au.to(dtype=torch.complex128, device=like.device).expand_as(like)
        if material == "si":
            return self.reference_eps_si.to(dtype=torch.complex128, device=like.device).expand_as(like)
        if material == "sio2":
            return self._sio2_epsilon(w.to(like.device)).expand_as(like)
        raise ValueError(f"Unsupported reference material: {self.reference_material}")

    def compute_reflectivity(self, eps: torch.Tensor) -> torch.Tensor:
        root = torch.sqrt(eps)
        return (root - 1.0) / (root + 1.0 + 1e-12)

    def compute_beta(self, eps: torch.Tensor) -> torch.Tensor:
        return (eps - 1.0) / (eps + 1.0 + 1e-12)

    def integral_sinf(
        self,
        q: torch.Tensor,
        reflectivity: torch.Tensor,
        beta: torch.Tensor,
        g_factor: torch.Tensor,
        g_phase: torch.Tensor,
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
        G = g_factor.to(torch.float64).unsqueeze(-1).unsqueeze(-1) * torch.exp(
            1j * g_phase.to(torch.float64).unsqueeze(-1).unsqueeze(-1)
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

    def compute_signal_core(self, w: torch.Tensor, decoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self._epsilon(w, decoded)
        r_sample = self.compute_reflectivity(eps)
        beta_sample = self.compute_beta(eps)

        phase_sample, amp_sample = self.integral_sinf(
            w, r_sample, beta_sample, decoded["g_factor"], decoded["g_phase"]
        )
        if self.reference_material == "matched-bg":
            matched_amp = self.matched_bg_amp.to(device=w.device).unsqueeze(0)
            matched_phase = self.matched_bg_phase.to(device=w.device).unsqueeze(0)
            phase_core = phase_sample - matched_phase
            amp_core = amp_sample / (torch.abs(matched_amp) + EPS)
            amp_denom = torch.clamp(
                torch.median(torch.abs(amp_core), dim=1, keepdim=True).values,
                min=torch.finfo(torch.float64).tiny,
            )
            amp_core = amp_core / amp_denom
        else:
            ref_eps = self.reference_epsilon(w, eps)
            r_ref = self.compute_reflectivity(ref_eps)
            beta_ref = self.compute_beta(ref_eps)
            phase_ref, amp_ref = self.integral_sinf(
                w, r_ref, beta_ref, decoded["g_factor"], decoded["g_phase"]
            )
            phase_core = phase_sample - phase_ref
            amp_core = amp_sample / (amp_ref + EPS)
        return phase_core, amp_core

    def compute_signal_sinf(self, w: torch.Tensor, decoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        phase_core, amp_core = self.compute_signal_core(w, decoded)
        phase = torch.atan2(
            torch.sin(phase_core + decoded["phase_offset"].unsqueeze(1)),
            torch.cos(phase_core + decoded["phase_offset"].unsqueeze(1)),
        )
        amp = decoded["amp_scale"].unsqueeze(1) * amp_core
        return phase, amp

    def profile_nuisance(
        self,
        w: torch.Tensor,
        unit: torch.Tensor,
        amp_true: torch.Tensor,
        phase_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if unit.ndim == 1:
            unit = unit.unsqueeze(0)
        if amp_true.ndim == 1:
            amp_true = amp_true.unsqueeze(0).expand(unit.shape[0], -1)
        if phase_true.ndim == 1:
            phase_true = phase_true.unsqueeze(0).expand(unit.shape[0], -1)
        decoded = self.decode(torch.sigmoid(unit))
        phase_core, amp_core = self.compute_signal_core(w.to(torch.float64), decoded)
        if weights is None:
            weight_row = torch.ones_like(amp_core, dtype=torch.float64)
        else:
            weight_row = weights.to(torch.float64).unsqueeze(0).expand_as(amp_core)

        numerator = torch.sum(weight_row * amp_core * amp_true, dim=1)
        denominator = torch.sum(weight_row * amp_core * amp_core, dim=1) + EPS
        amp_scale = torch.clamp(numerator / denominator, min=math.exp(-16), max=math.exp(16))
        amp = amp_scale.unsqueeze(1) * amp_core

        phase_delta = phase_true - phase_core
        sin_sum = torch.sum(weight_row * torch.sin(phase_delta), dim=1)
        cos_sum = torch.sum(weight_row * torch.cos(phase_delta), dim=1)
        phase_offset = torch.atan2(sin_sum, cos_sum)
        phase = torch.atan2(
            torch.sin(phase_core + phase_offset.unsqueeze(1)),
            torch.cos(phase_core + phase_offset.unsqueeze(1)),
        )
        return phase, amp, amp_scale, phase_offset

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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    return torch.sum(values * weights) / (torch.sum(weights) + EPS)


def amplitude_loss(pred: torch.Tensor, true: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    scale = torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS
    pred_norm = pred / scale
    true_norm = true / scale
    shape_pred = pred / (torch.median(torch.abs(pred), dim=-1, keepdim=True).values + EPS)
    shape_true = true / (torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS)
    value_loss = weighted_mean((pred_norm - true_norm) ** 2, weights)
    shape_loss = weighted_mean((shape_pred - shape_true) ** 2, weights)
    return 0.7 * value_loss + 0.3 * shape_loss


def amplitude_shape_loss(pred: torch.Tensor, true: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    shape_pred = pred / (torch.median(torch.abs(pred), dim=-1, keepdim=True).values + EPS)
    shape_true = true / (torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS)
    return weighted_mean((shape_pred - shape_true) ** 2, weights)


def phase_loss(pred: torch.Tensor, true: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_mean(torch.angle(torch.exp(1j * (pred - true))) ** 2, weights)


def spectrum_derivative_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    weights: torch.Tensor | None = None,
    circular: bool = False,
) -> torch.Tensor:
    if pred.shape[-1] < 2 or true.shape[-1] < 2:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if circular:
        pred_delta = torch.angle(torch.exp(1j * (pred[..., 1:] - pred[..., :-1])))
        true_delta = torch.angle(torch.exp(1j * (true[..., 1:] - true[..., :-1])))
    else:
        scale = torch.median(torch.abs(true), dim=-1, keepdim=True).values + EPS
        pred_delta = pred[..., 1:] / scale - pred[..., :-1] / scale
        true_delta = true[..., 1:] / scale - true[..., :-1] / scale
    diff_weights = None if weights is None else weights[..., 1:]
    return weighted_mean((pred_delta - true_delta) ** 2, diff_weights)


def combine_task_losses(
    amp_component: torch.Tensor,
    phase_component: torch.Tensor,
    phase_weight: float,
    task_weight_mode: str = "fixed",
    ema_state: dict[str, float] | None = None,
    ema_decay: float = 0.9,
    update_ema: bool = True,
) -> tuple[torch.Tensor, float]:
    if phase_weight <= 0:
        return amp_component, 0.0
    effective_phase_weight = float(phase_weight)
    if task_weight_mode == "ema":
        if ema_state is None:
            ema_state = {}
        amp_value = float(amp_component.detach().cpu())
        phase_value = float(phase_component.detach().cpu())
        if "amp" not in ema_state:
            ema_state["amp"] = amp_value
            ema_state["phase"] = phase_value
        elif update_ema:
            ema_state["amp"] = ema_decay * ema_state["amp"] + (1.0 - ema_decay) * amp_value
            ema_state["phase"] = ema_decay * ema_state["phase"] + (1.0 - ema_decay) * phase_value
        ratio = ema_state["amp"] / (ema_state["phase"] + EPS)
        effective_phase_weight = float(np.clip(phase_weight * ratio, 0.02, 5.0))
    return amp_component + effective_phase_weight * phase_component, effective_phase_weight


def reconstruction_loss(
    amp_pred: torch.Tensor,
    phase_pred: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    phase_weight: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = amplitude_shape_loss(amp_pred, amp_true, weights)
    if phase_weight > 0:
        loss = loss + phase_weight * phase_loss(phase_pred, phase_true, weights)
    return loss


def joint_spectrum_loss(
    amp_pred: torch.Tensor,
    phase_pred: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    phase_weight: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = amplitude_loss(amp_pred, amp_true, weights)
    if phase_weight > 0:
        loss = loss + phase_weight * phase_loss(phase_pred, phase_true, weights)
    return loss


def loss_weights_for_wavenumber(
    w: torch.Tensor,
    band_min: float,
    band_max: float,
    band_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(w, dtype=torch.float64)
    if band_weight != 1.0:
        in_band = (w >= band_min) & (w <= band_max)
        weights = torch.where(in_band, torch.full_like(weights, band_weight), weights)
    return weights


def active_literature_bands(prior_name: str, wmin: float, wmax: float) -> list[LiteratureBand]:
    if prior_name == "none":
        return []
    if prior_name != "liver-ftir":
        raise ValueError(f"Unsupported literature prior: {prior_name}")
    return [band for band in LIVER_FTIR_BANDS if band.band_max >= wmin and band.band_min <= wmax]


def literature_center_prior_loss(
    params: torch.Tensor,
    bounds: PlatformBounds,
    bands: list[LiteratureBand],
    distance_scale: float,
) -> torch.Tensor:
    if not bands:
        return torch.zeros((), dtype=params.dtype, device=params.device)
    n = bounds.n_oscillators
    centers = params[..., :n]
    penalties = []
    scale = max(float(distance_scale), EPS)
    for band in bands:
        lower = torch.as_tensor(band.band_min, dtype=params.dtype, device=params.device)
        upper = torch.as_tensor(band.band_max, dtype=params.dtype, device=params.device)
        below = torch.relu(lower - centers)
        above = torch.relu(centers - upper)
        penalties.append(((below + above) / scale) ** 2)
    stacked = torch.stack(penalties, dim=0)
    nearest = torch.min(stacked, dim=0).values
    return torch.mean(nearest)


def gamma_width_penalty(
    params: torch.Tensor,
    bounds: PlatformBounds,
    gamma_soft_max: float,
    distance_scale: float,
) -> torch.Tensor:
    n = bounds.n_oscillators
    gamma = params[..., 2 * n : 3 * n]
    scale = max(float(distance_scale), EPS)
    excess = torch.relu(gamma - gamma_soft_max) / scale
    return torch.mean(excess**2)


def literature_regularization_from_params(
    params: torch.Tensor,
    bounds: PlatformBounds,
    bands: list[LiteratureBand],
    center_weight: float,
    gamma_weight: float,
    gamma_soft_max: float,
    center_distance_scale: float,
) -> torch.Tensor:
    if not bands and gamma_weight <= 0:
        return torch.zeros((), dtype=params.dtype, device=params.device)
    loss = torch.zeros((), dtype=params.dtype, device=params.device)
    if bands and center_weight > 0:
        loss = loss + center_weight * literature_center_prior_loss(
            params,
            bounds,
            bands,
            center_distance_scale,
        )
    if gamma_weight > 0:
        gamma_scale = max(bounds.gamma_max - gamma_soft_max, EPS)
        loss = loss + gamma_weight * gamma_width_penalty(
            params,
            bounds,
            gamma_soft_max,
            gamma_scale,
        )
    return loss


def literature_regularization_loss(
    raw: torch.Tensor,
    model: SeniorStyleSnomModel,
    bands: list[LiteratureBand],
    center_weight: float,
    gamma_weight: float,
    gamma_soft_max: float,
    center_distance_scale: float,
) -> torch.Tensor:
    params = from_unit(torch.sigmoid(raw), model.bounds)
    return literature_regularization_from_params(
        params,
        model.bounds,
        bands,
        center_weight,
        gamma_weight,
        gamma_soft_max,
        center_distance_scale,
    )


def literature_band_annotation(center: float, bands: list[LiteratureBand]) -> tuple[bool, str, float]:
    if not bands or not np.isfinite(center):
        return False, "", np.nan
    distances = []
    for band in bands:
        if band.band_min <= center <= band.band_max:
            return True, band.label, 0.0
        distances.append((min(abs(center - band.band_min), abs(center - band.band_max)), band.label))
    distance, label = min(distances, key=lambda item: item[0])
    return False, label, float(distance)


def in_band(value: float, band_min: float, band_max: float) -> bool:
    return bool(np.isfinite(value) and band_min <= value <= band_max)


def point_stability_weights_for_curves(
    curves: list[dict[str, object]],
    floor: float = 0.25,
) -> np.ndarray | None:
    amp_rows = [np.asarray(curve["amp_std"], dtype=np.float64) for curve in curves if curve.get("amp_std") is not None]
    phase_rows = [
        np.asarray(curve["phase_std"], dtype=np.float64) for curve in curves if curve.get("phase_std") is not None
    ]
    if not amp_rows or not phase_rows:
        return None
    amp_noise = np.nanmean(np.asarray(amp_rows, dtype=np.float64), axis=0)
    phase_noise = np.nanmean(np.asarray(phase_rows, dtype=np.float64), axis=0)
    amp_norm = amp_noise / (np.nanmedian(amp_noise) + EPS)
    phase_norm = phase_noise / (np.nanmedian(phase_noise) + EPS)
    noise = 0.5 * (amp_norm + phase_norm)
    weights = 1.0 / np.maximum(noise, EPS)
    weights = np.clip(weights, floor, 1.0 / max(floor, EPS))
    weights = weights / (np.nanmean(weights) + EPS)
    return weights.astype(np.float64)


def region_mse(
    w: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    band_min: float,
    band_max: float,
    circular: bool = False,
) -> dict[str, float]:
    in_band = (w >= band_min) & (w <= band_max)
    out_band = ~in_band
    residual = np.angle(np.exp(1j * (pred - true))) if circular else pred - true
    return {
        "band": float(np.mean(residual[in_band] ** 2)) if np.any(in_band) else np.nan,
        "outside": float(np.mean(residual[out_band] ** 2)) if np.any(out_band) else np.nan,
    }


def generate_synthetic_training_data(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    bounds: PlatformBounds,
    n_samples: int,
    feature_dim: int,
    seed: int,
    batch_size: int = 32,
    amp_noise_std: float = 0.0,
    phase_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    params = latin_hypercube(bounds, n_samples=n_samples, seed=seed)
    unit = torch.as_tensor(to_unit(params, bounds), dtype=torch.float64)
    rng = np.random.default_rng(seed + 13)
    features: list[np.ndarray] = []
    units: list[np.ndarray] = []
    amp_targets: list[np.ndarray] = []
    phase_targets: list[np.ndarray] = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        with torch.no_grad():
            phase, amp = model(w, model.encode(torch.as_tensor(params[start:end], dtype=torch.float64)))
        amp_np = amp.detach().cpu().numpy()
        phase_np = phase.detach().cpu().numpy()
        for amp_row, phase_row in zip(amp_np, phase_np):
            amp_targets.append(amp_row.astype(np.float64, copy=False))
            phase_targets.append(phase_row.astype(np.float64, copy=False))
            if amp_noise_std > 0:
                amp_row = np.clip(amp_row * (1.0 + rng.normal(0.0, amp_noise_std, size=amp_row.shape)), EPS, None)
            if phase_noise_std > 0:
                phase_row = phase_row + rng.normal(0.0, phase_noise_std, size=phase_row.shape)
            features.append(extract_spectrum_features(amp_row, phase_row, feature_dim))
        units.append(unit[start:end].cpu().numpy())
    return (
        torch.as_tensor(np.asarray(features), dtype=torch.float32),
        torch.as_tensor(np.concatenate(units), dtype=torch.float32),
        torch.as_tensor(np.asarray(amp_targets), dtype=torch.float32),
        torch.as_tensor(np.asarray(phase_targets), dtype=torch.float32),
    )


def pretrain_predictor(
    predictor: ParameterPredictor,
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    patience: int = 20,
    spectrum_weight: float = 1.0,
    param_weight: float = 0.2,
    phase_weight: float = 0.2,
    weights: torch.Tensor | None = None,
    batch_size: int = 128,
    task_weight_mode: str = "fixed",
    task_ema_decay: float = 0.9,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)
    split = max(1, int(0.8 * len(features)))
    train_idx, val_idx = indices[:split], indices[split:]
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in predictor.state_dict().items()})
    best_epoch = 0
    stalled = 0
    ema_state: dict[str, float] = {}
    for epoch in range(epochs):
        predictor.train()
        epoch_order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        train_loss_total = 0.0
        train_spec_total = 0.0
        train_param_total = 0.0
        train_seen = 0
        for start in range(0, len(epoch_order), batch_size):
            batch_idx = epoch_order[start : start + batch_size]
            optimizer.zero_grad()
            train_logits = predictor(features[batch_idx])
            train_unit = torch.sigmoid(train_logits)
            train_phase, train_amp = model(w, train_logits)
            param_loss = F.mse_loss(train_unit, targets[batch_idx])
            amp_component = amplitude_shape_loss(train_amp, amp_true[batch_idx], weights)
            phase_component = phase_loss(train_phase, phase_true[batch_idx], weights)
            spec_loss, effective_phase_weight = combine_task_losses(
                amp_component,
                phase_component,
                phase_weight=phase_weight,
                task_weight_mode=task_weight_mode,
                ema_state=ema_state,
                ema_decay=task_ema_decay,
                update_ema=True,
            )
            loss = spectrum_weight * spec_loss + param_weight * param_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            n_batch = int(len(batch_idx))
            train_loss_total += float(loss.detach()) * n_batch
            train_spec_total += float(spec_loss.detach()) * n_batch
            train_param_total += float(param_loss.detach()) * n_batch
            train_seen += n_batch
        predictor.eval()
        with torch.no_grad():
            if len(val_idx):
                val_loss_total = 0.0
                val_spec_total = 0.0
                val_param_total = 0.0
                val_seen = 0
                for start in range(0, len(val_idx), batch_size):
                    batch_idx = val_idx[start : start + batch_size]
                    val_logits = predictor(features[batch_idx])
                    val_unit = torch.sigmoid(val_logits)
                    val_phase, val_amp = model(w, val_logits)
                    val_param_loss = F.mse_loss(val_unit, targets[batch_idx])
                    val_amp_component = amplitude_shape_loss(val_amp, amp_true[batch_idx], weights)
                    val_phase_component = phase_loss(val_phase, phase_true[batch_idx], weights)
                    val_spec_loss, _ = combine_task_losses(
                        val_amp_component,
                        val_phase_component,
                        phase_weight=phase_weight,
                        task_weight_mode=task_weight_mode,
                        ema_state=ema_state,
                        ema_decay=task_ema_decay,
                        update_ema=False,
                    )
                    val_loss = spectrum_weight * val_spec_loss + param_weight * val_param_loss
                    n_batch = int(len(batch_idx))
                    val_loss_total += float(val_loss.detach()) * n_batch
                    val_spec_total += float(val_spec_loss.detach()) * n_batch
                    val_param_total += float(val_param_loss.detach()) * n_batch
                    val_seen += n_batch
                val_loss_value = val_loss_total / max(val_seen, 1)
                val_spec_value = val_spec_total / max(val_seen, 1)
                val_param_value = val_param_total / max(val_seen, 1)
            else:
                val_loss_value = train_loss_total / max(train_seen, 1)
                val_spec_value = train_spec_total / max(train_seen, 1)
                val_param_value = train_param_total / max(train_seen, 1)
        train_loss_value = train_loss_total / max(train_seen, 1)
        train_spec_value = train_spec_total / max(train_seen, 1)
        train_param_value = train_param_total / max(train_seen, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
                "train_spec_loss": train_spec_value,
                "train_param_loss": train_param_value,
                "val_spec_loss": val_spec_value,
                "val_param_loss": val_param_value,
            }
        )
        if val_loss_value + 1e-6 < best_val:
            best_val = val_loss_value
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in predictor.state_dict().items()})
            best_epoch = epoch + 1
            stalled = 0
        else:
            stalled += 1
        if stalled >= patience:
            break
    predictor.load_state_dict(best_state)
    history.append({"epoch": best_epoch, "best_val_loss": best_val, "restored_best": 1.0})
    return history


def calibrate_candidate_amp_scales(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    candidate_params: np.ndarray,
    weights: torch.Tensor | None = None,
) -> np.ndarray:
    calibrated = np.asarray(candidate_params, dtype=np.float64).copy()
    log_scale_idx = 3 * model.bounds.n_oscillators + 1
    with torch.no_grad():
        logits = model.encode(torch.as_tensor(calibrated, dtype=torch.float64, device=w.device))
        _, amp_pred = model(w, logits)
        target = amp_true.unsqueeze(0).expand_as(amp_pred)
        if weights is None:
            numerator = torch.sum(amp_pred * target, dim=1)
            denominator = torch.sum(amp_pred * amp_pred, dim=1) + EPS
        else:
            weight_row = weights.unsqueeze(0).expand_as(amp_pred)
            numerator = torch.sum(weight_row * amp_pred * target, dim=1)
            denominator = torch.sum(weight_row * amp_pred * amp_pred, dim=1) + EPS
        alpha = torch.clamp(numerator / denominator, min=math.exp(-16), max=math.exp(16))
        log_delta = torch.log(alpha).cpu().numpy()
    calibrated[:, log_scale_idx] = np.clip(
        calibrated[:, log_scale_idx] + log_delta,
        model.bounds.log_amp_scale_min,
        model.bounds.log_amp_scale_max,
    )
    return calibrated


def score_candidate_params(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    candidate_params: np.ndarray,
    phase_weight: float,
    weights: torch.Tensor | None = None,
    profile_nuisance: bool = False,
    task_weight_mode: str = "fixed",
    literature_bands: list[LiteratureBand] | None = None,
    literature_center_weight: float = 0.0,
    gamma_width_penalty_weight: float = 0.0,
    gamma_soft_max: float = 120.0,
    literature_distance_scale: float = 25.0,
    derivative_weight: float = 0.0,
    phase_derivative_weight: float = 0.0,
) -> float:
    losses, _ = evaluate_candidates(
        model,
        w,
        amp_true,
        phase_true,
        np.asarray(candidate_params, dtype=np.float64).reshape(1, -1),
        phase_weight,
        weights=weights,
        profile_nuisance=profile_nuisance,
        task_weight_mode=task_weight_mode,
        literature_bands=literature_bands,
        literature_center_weight=literature_center_weight,
        gamma_width_penalty_weight=gamma_width_penalty_weight,
        gamma_soft_max=gamma_soft_max,
        literature_distance_scale=literature_distance_scale,
        derivative_weight=derivative_weight,
        phase_derivative_weight=phase_derivative_weight,
    )
    return float(losses[0])


def differential_evolution_candidate(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    phase_weight: float,
    weights: torch.Tensor | None = None,
    profile_nuisance: bool = False,
    task_weight_mode: str = "fixed",
    literature_bands: list[LiteratureBand] | None = None,
    literature_center_weight: float = 0.0,
    gamma_width_penalty_weight: float = 0.0,
    gamma_soft_max: float = 120.0,
    literature_distance_scale: float = 25.0,
    derivative_weight: float = 0.0,
    phase_derivative_weight: float = 0.0,
    maxiter: int = 12,
    popsize: int = 10,
    polish: bool = False,
    seed: int = 42,
) -> np.ndarray:
    from scipy.optimize import differential_evolution

    bounds = list(zip(model.bounds.lower().tolist(), model.bounds.upper().tolist()))

    def objective(x: np.ndarray) -> float:
        params = canonicalize_oscillator_order(np.asarray(x, dtype=np.float64), model.bounds)
        return score_candidate_params(
            model,
            w,
            amp_true,
            phase_true,
            params,
            phase_weight,
            weights=weights,
            profile_nuisance=profile_nuisance,
            task_weight_mode=task_weight_mode,
            literature_bands=literature_bands,
            literature_center_weight=literature_center_weight,
            gamma_width_penalty_weight=gamma_width_penalty_weight,
            gamma_soft_max=gamma_soft_max,
            literature_distance_scale=literature_distance_scale,
            derivative_weight=derivative_weight,
            phase_derivative_weight=phase_derivative_weight,
        )

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=polish,
        updating="deferred",
        workers=1,
        disp=False,
    )
    return canonicalize_oscillator_order(np.asarray(result.x, dtype=np.float64), model.bounds)


def evaluate_candidates(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    amp_true: torch.Tensor,
    phase_true: torch.Tensor,
    candidate_params: np.ndarray,
    phase_weight: float,
    weights: torch.Tensor | None = None,
    profile_nuisance: bool = False,
    task_weight_mode: str = "fixed",
    literature_bands: list[LiteratureBand] | None = None,
    literature_center_weight: float = 0.0,
    gamma_width_penalty_weight: float = 0.0,
    gamma_soft_max: float = 120.0,
    literature_distance_scale: float = 25.0,
    derivative_weight: float = 0.0,
    phase_derivative_weight: float = 0.0,
) -> tuple[np.ndarray, int]:
    with torch.no_grad():
        params_t = torch.as_tensor(candidate_params, dtype=torch.float64, device=w.device)
        logits = model.encode(params_t)
        if profile_nuisance:
            phase_pred, amp_pred, _, _ = model.profile_nuisance(w, logits, amp_true, phase_true, weights=weights)
        else:
            phase_pred, amp_pred = model(w, logits)
        amp_target = amp_true.unsqueeze(0).expand_as(amp_pred)
        phase_target = phase_true.unsqueeze(0).expand_as(phase_pred)
        amp_scale = torch.median(torch.abs(amp_true)) + EPS
        amp_residual = ((amp_pred - amp_target) / amp_scale) ** 2
        phase_residual = torch.angle(torch.exp(1j * (phase_pred - phase_target))) ** 2
        if weights is None:
            amp_losses = torch.mean(amp_residual, dim=1)
            phase_losses = torch.mean(phase_residual, dim=1)
        else:
            weight_row = weights.unsqueeze(0).expand_as(amp_residual)
            denom = torch.sum(weight_row, dim=1) + EPS
            amp_losses = torch.sum(weight_row * amp_residual, dim=1) / denom
            phase_losses = torch.sum(weight_row * phase_residual, dim=1) / denom
        if derivative_weight > 0 or phase_derivative_weight > 0:
            derivative_losses = []
            for idx in range(amp_pred.shape[0]):
                deriv_loss = torch.zeros((), dtype=amp_pred.dtype, device=amp_pred.device)
                if derivative_weight > 0:
                    deriv_loss = deriv_loss + derivative_weight * spectrum_derivative_loss(
                        amp_pred[idx : idx + 1],
                        amp_true.unsqueeze(0),
                        weights=weights,
                    )
                if phase_derivative_weight > 0:
                    deriv_loss = deriv_loss + phase_derivative_weight * spectrum_derivative_loss(
                        phase_pred[idx : idx + 1],
                        phase_true.unsqueeze(0),
                        weights=weights,
                        circular=True,
                    )
                derivative_losses.append(deriv_loss)
            amp_losses = amp_losses + torch.stack(derivative_losses)
        effective_phase_weight = phase_weight
        if task_weight_mode == "ema" and phase_weight > 0:
            effective_phase_weight = float(
                np.clip(
                    phase_weight
                    * float(torch.mean(amp_losses).detach().cpu())
                    / (float(torch.mean(phase_losses).detach().cpu()) + EPS),
                    0.02,
                    5.0,
                )
            )
        total_losses = amp_losses + effective_phase_weight * phase_losses
        if literature_bands and (literature_center_weight > 0 or gamma_width_penalty_weight > 0):
            prior_losses = []
            for params_row in params_t:
                prior_losses.append(
                    literature_regularization_from_params(
                        params_row,
                        model.bounds,
                        literature_bands,
                        literature_center_weight,
                        gamma_width_penalty_weight,
                        gamma_soft_max,
                        literature_distance_scale,
                    )
                )
            total_losses = total_losses + torch.stack(prior_losses)
        losses = total_losses.cpu().numpy()
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
    weights: torch.Tensor | None = None,
    profile_nuisance: bool = False,
    task_weight_mode: str = "fixed",
    task_ema_decay: float = 0.9,
    literature_bands: list[LiteratureBand] | None = None,
    literature_center_weight: float = 0.0,
    gamma_width_penalty_weight: float = 0.0,
    gamma_soft_max: float = 120.0,
    literature_distance_scale: float = 25.0,
    derivative_weight: float = 0.0,
    phase_derivative_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw = nn.Parameter(model.encode(initial_params).detach().clone())
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    ema_state: dict[str, float] = {}
    for _ in range(steps):
        optimizer.zero_grad()
        if profile_nuisance:
            phase_pred, amp_pred, _, _ = model.profile_nuisance(w, raw, amp_true, phase_true, weights=weights)
        else:
            phase_pred, amp_pred = model(w, raw)
        amp_component = amplitude_loss(amp_pred, amp_true.unsqueeze(0), weights)
        phase_component = phase_loss(phase_pred, phase_true.unsqueeze(0), weights)
        if derivative_weight > 0:
            amp_component = amp_component + derivative_weight * spectrum_derivative_loss(
                amp_pred,
                amp_true.unsqueeze(0),
                weights=weights,
            )
        if phase_derivative_weight > 0:
            phase_component = phase_component + phase_derivative_weight * spectrum_derivative_loss(
                phase_pred,
                phase_true.unsqueeze(0),
                weights=weights,
                circular=True,
            )
        loss, _ = combine_task_losses(
            amp_component,
            phase_component,
            phase_weight=phase_weight,
            task_weight_mode=task_weight_mode,
            ema_state=ema_state,
            ema_decay=task_ema_decay,
            update_ema=True,
        )
        if literature_bands and (literature_center_weight > 0 or gamma_width_penalty_weight > 0):
            loss = loss + literature_regularization_loss(
                raw,
                model,
                literature_bands,
                literature_center_weight,
                gamma_width_penalty_weight,
                gamma_soft_max,
                literature_distance_scale,
            )
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_raw = raw.detach().clone()
    with torch.no_grad():
        if profile_nuisance:
            phase_pred, amp_pred, amp_scale, phase_offset = model.profile_nuisance(
                w, best_raw, amp_true, phase_true, weights=weights
            )
        else:
            phase_pred, amp_pred = model(w, best_raw)
            params_for_nuisance = from_unit(torch.sigmoid(best_raw), model.bounds)
            n = model.bounds.n_oscillators
            amp_scale = torch.exp(params_for_nuisance[..., 3 * n + 1])
            phase_offset = params_for_nuisance[..., 3 * n + 2]
        phase_res = torch.angle(torch.exp(1j * (phase_pred[0] - phase_true)))
        amp_mse = float(torch.mean((amp_pred[0] - amp_true) ** 2))
        phase_mse = float(torch.mean(phase_res**2))
        final_params = canonicalize_oscillator_order(
            from_unit(torch.sigmoid(best_raw), model.bounds).detach().cpu().numpy(),
            model.bounds,
        )
        n = model.bounds.n_oscillators
        amp_scale_value = amp_scale.reshape(-1)[0]
        phase_offset_value = phase_offset.reshape(-1)[0]
        final_params[3 * n + 1] = float(torch.log(amp_scale_value).detach().cpu())
        final_params[3 * n + 2] = float(phase_offset_value.detach().cpu())
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
    matched_bg_amp = None
    matched_bg_phase = None
    if args.reference_material == "matched-bg":
        bg_amp_rows = [np.asarray(curve["bg_amp"], dtype=np.float64) for curve in curves if curve.get("bg_amp") is not None]
        bg_phase_rows = [
            np.asarray(curve["bg_phase"], dtype=np.float64) for curve in curves if curve.get("bg_phase") is not None
        ]
        if not bg_amp_rows or not bg_phase_rows:
            raise ValueError("reference-material matched-bg requires bg_amp_mean/bg_phase_mean in sample_level_spectra.npz")
        matched_bg_amp = np.nanmean(np.asarray(bg_amp_rows, dtype=np.float64), axis=0)
        matched_bg_phase = np.nanmean(np.asarray(bg_phase_rows, dtype=np.float64), axis=0)
    model = SeniorStyleSnomModel(
        bounds,
        n_time=args.n_time,
        reference_material=args.reference_material,
        fixed_g_factor=args.fixed_g_factor,
        fixed_g_phase=args.fixed_g_phase,
        matched_bg_amp=matched_bg_amp,
        matched_bg_phase=matched_bg_phase,
    ).to(device)
    loss_weights = loss_weights_for_wavenumber(
        w,
        args.substrate_band_min,
        args.substrate_band_max,
        args.substrate_band_weight,
    )
    point_weights_np = None
    if args.point_stability_weight:
        point_weights_np = point_stability_weights_for_curves(curves, floor=args.point_stability_floor)
        if point_weights_np is not None:
            point_weights = torch.as_tensor(point_weights_np, dtype=torch.float64, device=device)
            loss_weights = loss_weights * point_weights
            pd.DataFrame({"wavenumber": w_np, "point_weight": point_weights_np}).to_csv(
                batch_dir / "point_stability_weights.csv", index=False, encoding="utf-8-sig"
            )
    literature_bands = active_literature_bands(args.literature_prior, args.wmin, args.wmax)
    if literature_bands:
        pd.DataFrame(
            [
                {"label": band.label, "band_min": band.band_min, "band_max": band.band_max}
                for band in literature_bands
            ]
        ).to_csv(batch_dir / "literature_prior_bands.csv", index=False, encoding="utf-8-sig")
    predictor = ParameterPredictor(args.feature_dim, bounds.dimension).to(device)

    synth_features, synth_targets, synth_amp, synth_phase = generate_synthetic_training_data(
        model=model,
        w=w,
        bounds=bounds,
        n_samples=args.synthetic_samples,
        feature_dim=args.feature_dim,
        seed=args.seed,
        batch_size=args.synthetic_batch_size,
        amp_noise_std=args.synthetic_amp_noise,
        phase_noise_std=args.synthetic_phase_noise,
    )
    synth_features = synth_features.to(device)
    synth_targets = synth_targets.to(device)
    synth_amp = synth_amp.to(device)
    synth_phase = synth_phase.to(device)
    history = pretrain_predictor(
        predictor,
        model,
        w,
        synth_features,
        synth_targets,
        synth_amp,
        synth_phase,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        seed=args.seed,
        patience=args.pretrain_patience,
        spectrum_weight=args.physics_recon_weight,
        param_weight=args.physics_param_weight,
        phase_weight=args.phase_weight,
        weights=loss_weights,
        batch_size=args.pretrain_batch_size,
        task_weight_mode=args.task_weight_mode,
        task_ema_decay=args.task_ema_decay,
    )
    pd.DataFrame(history).to_csv(batch_dir / "ml_pretrain_history.csv", index=False)

    phase0_by_sample: dict[str, np.ndarray] = {}
    phase0_by_group: dict[str, np.ndarray] = {}
    if phase0_init_maps:
        phase0_by_sample, phase0_by_group = phase0_init_maps

    if phase0_by_sample or phase0_by_group:
        real_features: list[np.ndarray] = []
        real_targets: list[np.ndarray] = []
        real_amp_targets: list[np.ndarray] = []
        real_phase_targets: list[np.ndarray] = []
        for curve in curves:
            sample_id = str(curve["sample_id"])
            analysis_group = f'{curve["specimen_name"]}__{curve["tune_wavenumber"] or "na"}'
            phase0_target = phase0_by_sample.get(sample_id)
            if phase0_target is None:
                phase0_target = phase0_by_group.get(analysis_group)
            if phase0_target is None:
                continue
            real_features.append(
                extract_spectrum_features(
                    np.asarray(curve["amp"], dtype=np.float64),
                    np.asarray(curve["phase"], dtype=np.float64),
                    args.feature_dim,
                )
            )
            real_targets.append(to_unit(phase0_target, bounds))
            real_amp_targets.append(np.asarray(curve["amp"], dtype=np.float64))
            real_phase_targets.append(np.asarray(curve["phase"], dtype=np.float64))
        if real_features and args.phase0_warmup_epochs > 0:
            warm_features = torch.as_tensor(np.asarray(real_features), dtype=torch.float32).to(device)
            warm_targets = torch.as_tensor(np.asarray(real_targets), dtype=torch.float32).to(device)
            warm_amp = torch.as_tensor(np.asarray(real_amp_targets), dtype=torch.float32).to(device)
            warm_phase = torch.as_tensor(np.asarray(real_phase_targets), dtype=torch.float32).to(device)
            warm_history = pretrain_predictor(
                predictor,
                model,
                w,
                warm_features,
                warm_targets,
                warm_amp,
                warm_phase,
                epochs=args.phase0_warmup_epochs,
                learning_rate=args.phase0_warmup_lr,
                seed=args.seed + 97,
                patience=max(5, args.phase0_warmup_epochs // 4),
                spectrum_weight=args.physics_recon_weight,
                param_weight=args.physics_param_weight,
                phase_weight=args.phase_weight,
                weights=loss_weights,
                batch_size=args.pretrain_batch_size,
                task_weight_mode=args.task_weight_mode,
                task_ema_decay=args.task_ema_decay,
            )
            pd.DataFrame(warm_history).to_csv(batch_dir / "phase0_warmup_history.csv", index=False)

    rows: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []
    for curve in curves:
        amp_true_np = np.asarray(curve["amp"], dtype=np.float64)
        phase_true_np = np.asarray(curve["phase"], dtype=np.float64)
        amp_true = torch.as_tensor(amp_true_np, dtype=torch.float64, device=device)
        phase_true = torch.as_tensor(phase_true_np, dtype=torch.float64, device=device)
        feature = torch.as_tensor(
            extract_spectrum_features(amp_true_np, phase_true_np, args.feature_dim),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        analysis_group = f'{curve["specimen_name"]}__{curve["tune_wavenumber"] or "na"}'
        phase0_candidate = phase0_by_sample.get(str(curve["sample_id"]))
        if phase0_candidate is None:
            phase0_candidate = phase0_by_group.get(analysis_group)
        predictor.eval()
        with torch.no_grad():
            nn_unit = torch.sigmoid(predictor(feature)).cpu().numpy()[0]
        nn_params = canonicalize_oscillator_order(
            from_unit(torch.as_tensor(nn_unit, dtype=torch.float64), bounds).cpu().numpy(),
            bounds,
        )
        lhs_params = latin_hypercube(bounds, n_samples=args.init_candidates, seed=args.seed + len(rows) + 1)
        candidate_list = [nn_params]
        if phase0_candidate is not None:
            candidate_list.insert(0, phase0_candidate)
        candidate_list.extend(lhs_params)
        if args.global_init == "de":
            de_params = differential_evolution_candidate(
                model,
                w,
                amp_true,
                phase_true,
                args.phase_weight,
                weights=loss_weights,
                profile_nuisance=args.profile_nuisance,
                task_weight_mode=args.task_weight_mode,
                literature_bands=literature_bands,
                literature_center_weight=args.literature_center_weight,
                gamma_width_penalty_weight=args.literature_gamma_weight,
                gamma_soft_max=args.literature_gamma_soft_max,
                literature_distance_scale=args.literature_center_distance_scale,
                derivative_weight=args.derivative_weight,
                phase_derivative_weight=args.phase_derivative_weight,
                maxiter=args.global_init_iters,
                popsize=args.global_init_popsize,
                polish=args.global_init_polish,
                seed=args.seed + 1009 + len(rows),
            )
            candidate_list.append(de_params)
        candidates = calibrate_candidate_amp_scales(model, w, amp_true, np.vstack(candidate_list), weights=loss_weights)
        losses, best_idx = evaluate_candidates(
            model,
            w,
            amp_true,
            phase_true,
            candidates,
            args.phase_weight,
            weights=loss_weights,
            profile_nuisance=args.profile_nuisance,
            task_weight_mode=args.task_weight_mode,
            literature_bands=literature_bands,
            literature_center_weight=args.literature_center_weight,
            gamma_width_penalty_weight=args.literature_gamma_weight,
            gamma_soft_max=args.literature_gamma_soft_max,
            literature_distance_scale=args.literature_center_distance_scale,
            derivative_weight=args.derivative_weight,
            phase_derivative_weight=args.phase_derivative_weight,
        )
        initial_params = candidates[best_idx]
        if phase0_candidate is not None:
            if best_idx == 0:
                initial_source = "phase0"
            elif best_idx == 1:
                initial_source = "nn"
            elif args.global_init == "de" and best_idx == len(candidate_list) - 1:
                initial_source = "de"
            else:
                initial_source = "lhs"
        else:
            if best_idx == 0:
                initial_source = "nn"
            elif args.global_init == "de" and best_idx == len(candidate_list) - 1:
                initial_source = "de"
            else:
                initial_source = "lhs"
        final_params, amp_pred_np, amp_mse, phase_mse = refine_single(
            model=model,
            w=w,
            amp_true=amp_true,
            phase_true=phase_true,
            initial_params=initial_params,
            steps=args.refine_steps,
            learning_rate=args.refine_lr,
            phase_weight=args.phase_weight,
            weights=loss_weights,
            profile_nuisance=args.profile_nuisance,
            task_weight_mode=args.task_weight_mode,
            task_ema_decay=args.task_ema_decay,
            literature_bands=literature_bands,
            literature_center_weight=args.literature_center_weight,
            gamma_width_penalty_weight=args.literature_gamma_weight,
            gamma_soft_max=args.literature_gamma_soft_max,
            literature_distance_scale=args.literature_center_distance_scale,
            derivative_weight=args.derivative_weight,
            phase_derivative_weight=args.phase_derivative_weight,
        )
        with torch.no_grad():
            final_logits = model.encode(torch.as_tensor(final_params, dtype=torch.float64, device=device))
            if args.profile_nuisance:
                final_phase, _, _, _ = model.profile_nuisance(w, final_logits, amp_true, phase_true, weights=loss_weights)
            else:
                final_phase, _ = model(w, final_logits)
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
        amp_region = region_mse(w_np, amp_true_np, amp_pred_np, args.substrate_band_min, args.substrate_band_max)
        phase_region = region_mse(
            w_np,
            phase_true_np,
            final_phase_np,
            args.substrate_band_min,
            args.substrate_band_max,
            circular=True,
        )
        row["amp_mse_substrate_band"] = amp_region["band"]
        row["amp_mse_outside_substrate_band"] = amp_region["outside"]
        row["phase_circular_mse_substrate_band"] = phase_region["band"]
        row["phase_circular_mse_outside_substrate_band"] = phase_region["outside"]
        for name, value in zip(bounds.names(), final_params):
            row[name] = float(value)
        n_osc = bounds.n_oscillators
        row["g_factor"] = float(model.fixed_g_factor.detach().cpu())
        row["g_phase"] = float(model.fixed_g_phase.detach().cpu())
        row["amp_scale"] = float(math.exp(final_params[3 * n_osc + 1]))
        for osc_idx in range(n_osc):
            row[f"wL_{osc_idx + 1}"] = float(final_params[osc_idx] + final_params[n_osc + osc_idx])
            center = float(final_params[osc_idx])
            in_lit, lit_label, lit_distance = literature_band_annotation(center, literature_bands)
            row[f"literature_band_label_{osc_idx + 1}"] = lit_label
            row[f"center_in_literature_band_{osc_idx + 1}"] = bool(in_lit)
            row[f"center_distance_to_literature_{osc_idx + 1}"] = lit_distance
            row[f"center_in_sio2_band_{osc_idx + 1}"] = in_band(
                center, args.substrate_band_min, args.substrate_band_max
            )
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
        for name in bounds.names() + ["g_factor", "g_phase", "amp_scale"] + [
            f"wL_{i + 1}" for i in range(bounds.n_oscillators)
        ] + [
            f"center_in_literature_band_{i + 1}" for i in range(bounds.n_oscillators)
        ] + [
            f"center_in_sio2_band_{i + 1}" for i in range(bounds.n_oscillators)
        ]:
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
        "physics_recon_weight": args.physics_recon_weight,
        "physics_param_weight": args.physics_param_weight,
        "reference_material": args.reference_material,
        "fixed_g_factor": args.fixed_g_factor,
        "fixed_g_phase": args.fixed_g_phase,
        "substrate_band": [args.substrate_band_min, args.substrate_band_max],
        "substrate_band_weight": args.substrate_band_weight,
        "matched_bg_used": args.reference_material == "matched-bg",
        "profile_nuisance": args.profile_nuisance,
        "point_stability_weight": args.point_stability_weight,
        "task_weight_mode": args.task_weight_mode,
        "global_init": args.global_init,
        "global_init_iters": args.global_init_iters,
        "global_init_popsize": args.global_init_popsize,
        "derivative_weight": args.derivative_weight,
        "phase_derivative_weight": args.phase_derivative_weight,
        "literature_prior": args.literature_prior,
        "literature_prior_bands": [
            {"label": band.label, "min": band.band_min, "max": band.band_max} for band in literature_bands
        ],
        "model": "senior-style semi-infinite multi-phonon + phase-aware MLP",
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
    parser.add_argument("--n-oscillators", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--synthetic-samples", type=int, default=1024)
    parser.add_argument("--synthetic-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--pretrain-patience", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--synthetic-amp-noise", type=float, default=0.01)
    parser.add_argument("--synthetic-phase-noise", type=float, default=0.03)
    parser.add_argument("--physics-recon-weight", type=float, default=1.0)
    parser.add_argument("--physics-param-weight", type=float, default=0.2)
    parser.add_argument("--phase0-fit-dir", default=None)
    parser.add_argument("--phase0-warmup-epochs", type=int, default=0)
    parser.add_argument("--phase0-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--init-candidates", type=int, default=24)
    parser.add_argument(
        "--global-init",
        choices=["lhs", "de"],
        default="lhs",
        help="Add an optional global-optimization candidate before local refinement.",
    )
    parser.add_argument("--global-init-iters", type=int, default=8)
    parser.add_argument("--global-init-popsize", type=int, default=8)
    parser.add_argument("--global-init-polish", action="store_true")
    parser.add_argument("--refine-steps", type=int, default=120)
    parser.add_argument("--refine-lr", type=float, default=3e-2)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument(
        "--derivative-weight",
        type=float,
        default=0.0,
        help="Weight for matching first differences of amplitude spectra during candidate scoring/refinement.",
    )
    parser.add_argument(
        "--phase-derivative-weight",
        type=float,
        default=0.0,
        help="Weight for matching circular first differences of phase spectra during candidate scoring/refinement.",
    )
    parser.add_argument("--n-time", type=int, default=65)
    parser.add_argument("--fixed-g-factor", type=float, default=0.5)
    parser.add_argument("--fixed-g-phase", type=float, default=0.03)
    parser.add_argument(
        "--profile-nuisance",
        action="store_true",
        help="Analytically calibrate amp_scale and phase_offset during candidate scoring/refinement.",
    )
    parser.add_argument(
        "--point-stability-weight",
        action="store_true",
        help="Down-weight wavenumbers with high sample-level amp/phase dispersion.",
    )
    parser.add_argument("--point-stability-floor", type=float, default=0.25)
    parser.add_argument(
        "--task-weight-mode",
        choices=["fixed", "ema"],
        default="fixed",
        help="fixed uses phase_weight directly; ema balances phase loss against amp loss by EMA scale.",
    )
    parser.add_argument("--task-ema-decay", type=float, default=0.9)
    parser.add_argument(
        "--reference-material",
        choices=["au", "si", "sio2", "matched-bg"],
        default="sio2",
        help="Reference/substrate response used for normalized S-SNOM signal. matched-bg uses saved raw bg spectra.",
    )
    parser.add_argument("--substrate-band-min", type=float, default=1000.0)
    parser.add_argument("--substrate-band-max", type=float, default=1250.0)
    parser.add_argument(
        "--substrate-band-weight",
        type=float,
        default=1.0,
        help="Loss weight inside the substrate resonance band. Use <1 to down-weight this region.",
    )
    parser.add_argument(
        "--literature-prior",
        choices=["none", "liver-ftir"],
        default="none",
        help="Soft literature-informed prior for experimental refinement. liver-ftir uses reported liver/HCC FTIR bands.",
    )
    parser.add_argument(
        "--literature-center-weight",
        type=float,
        default=0.002,
        help="Weight for penalizing oscillator centers outside literature-supported bands.",
    )
    parser.add_argument(
        "--literature-center-distance-scale",
        type=float,
        default=25.0,
        help="cm^-1 scale used by the soft center-band penalty.",
    )
    parser.add_argument(
        "--literature-gamma-weight",
        type=float,
        default=0.0002,
        help="Weight for the soft penalty on Lorentz gamma broader than --literature-gamma-soft-max.",
    )
    parser.add_argument(
        "--literature-gamma-soft-max",
        type=float,
        default=120.0,
        help="Soft gamma ceiling in cm^-1. The hard optimizer bound remains PlatformBounds.gamma_max.",
    )
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
    amp_std_all = np.asarray(z["amp_std"], dtype=np.float64) if "amp_std" in z else None
    phase_std_all = np.asarray(z["phase_std"], dtype=np.float64) if "phase_std" in z else None
    bg_amp_all = np.asarray(z["bg_amp_mean"], dtype=np.float64) if "bg_amp_mean" in z else None
    bg_phase_all = np.asarray(z["bg_phase_mean"], dtype=np.float64) if "bg_phase_mean" in z else None
    w_np, _ = select_window(w_full, amp_all[0], args.wmin, args.wmax, args.stride)
    amp_curves = []
    for idx, row in table.iterrows():
        _, amp = select_window(w_full, amp_all[idx], args.wmin, args.wmax, args.stride)
        _, phase = select_window(w_full, phase_all[idx], args.wmin, args.wmax, args.stride)
        amp_std = None
        phase_std = None
        if amp_std_all is not None and phase_std_all is not None:
            _, amp_std = select_window(w_full, amp_std_all[idx], args.wmin, args.wmax, args.stride)
            _, phase_std = select_window(w_full, phase_std_all[idx], args.wmin, args.wmax, args.stride)
        bg_amp = None
        bg_phase = None
        if bg_amp_all is not None and bg_phase_all is not None:
            _, bg_amp = select_window(w_full, bg_amp_all[idx], args.wmin, args.wmax, args.stride)
            _, bg_phase = select_window(w_full, bg_phase_all[idx], args.wmin, args.wmax, args.stride)
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
                "amp_std": amp_std,
                "phase_std": phase_std,
                "bg_amp": bg_amp,
                "bg_phase": bg_phase,
            }
        )
    bounds = PlatformBounds(n_oscillators=args.n_oscillators)
    batches: dict[str, list[dict[str, object]]] = {}
    dataset_name = Path(args.dataset).name
    for curve in amp_curves:
        batch_name = f'{dataset_name}__{curve["tune_wavenumber"] or "na"}'
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
    config["class_rule"] = "liver/tumor labels are read from preprocessing metadata; TA/organoid aliases are still supported"
    (output_dir / "platform_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
