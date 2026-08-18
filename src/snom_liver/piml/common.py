"""Shared PIML utilities for synthetic training and experimental inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from snom_liver.physical_fit import normalize_amp_np
from snom_liver.physics import MultiLorentzSnomModel, TipParameters


@dataclass(frozen=True)
class MultiLorentzBounds:
    center_min: float = 700.0
    center_max: float = 1600.0
    strength_min: float = 0.0
    strength_max: float = 4.0
    gamma_min: float = 20.0
    gamma_max: float = 220.0
    eps_inf_min: float = 1.5
    eps_inf_max: float = 12.0


def parameter_names(n_oscillators: int) -> list[str]:
    names: list[str] = []
    for group in ("center", "strength", "gamma"):
        names.extend(f"{group}_{i + 1}" for i in range(n_oscillators))
    names.append("eps_inf")
    return names


def parameter_lows_highs(bounds: MultiLorentzBounds, n_oscillators: int) -> tuple[np.ndarray, np.ndarray]:
    lows = (
        [bounds.center_min] * n_oscillators
        + [bounds.strength_min] * n_oscillators
        + [bounds.gamma_min] * n_oscillators
        + [bounds.eps_inf_min]
    )
    highs = (
        [bounds.center_max] * n_oscillators
        + [bounds.strength_max] * n_oscillators
        + [bounds.gamma_max] * n_oscillators
        + [bounds.eps_inf_max]
    )
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def sample_parameters(
    rng: np.random.Generator,
    bounds: MultiLorentzBounds,
    n_oscillators: int,
) -> dict[str, np.ndarray | float]:
    centers = np.sort(rng.uniform(bounds.center_min, bounds.center_max, size=n_oscillators))
    strengths = rng.uniform(bounds.strength_min, bounds.strength_max, size=n_oscillators)
    gammas = rng.uniform(bounds.gamma_min, bounds.gamma_max, size=n_oscillators)
    eps_inf = float(rng.uniform(bounds.eps_inf_min, bounds.eps_inf_max))
    return {"centers": centers, "strengths": strengths, "gammas": gammas, "eps_inf": eps_inf}


def params_to_vector(params: dict[str, np.ndarray | list[float] | float], n_oscillators: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(params["centers"], dtype=np.float32),
            np.asarray(params["strengths"], dtype=np.float32),
            np.asarray(params["gammas"], dtype=np.float32),
            np.asarray([params["eps_inf"]], dtype=np.float32),
        ]
    )[: 3 * n_oscillators + 1]


def vector_to_params(vector: np.ndarray | torch.Tensor, n_oscillators: int) -> dict[str, list[float] | float]:
    if isinstance(vector, torch.Tensor):
        values = vector.detach().cpu().numpy()
    else:
        values = np.asarray(vector)
    centers = np.asarray(values[:n_oscillators], dtype=np.float64)
    strengths = np.asarray(values[n_oscillators : 2 * n_oscillators], dtype=np.float64)
    gammas = np.asarray(values[2 * n_oscillators : 3 * n_oscillators], dtype=np.float64)
    order = np.argsort(centers)
    return {
        "centers": [float(v) for v in centers[order]],
        "strengths": [float(v) for v in strengths[order]],
        "gammas": [float(v) for v in gammas[order]],
        "eps_inf": float(values[3 * n_oscillators]),
    }


def normalize_param_vector(vector: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return (vector - lows) / (highs - lows + 1e-12)


def denormalize_param_tensor(tensor: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    return lows + tensor * (highs - lows)


def make_multilorentz_model(
    params: dict[str, list[float] | float],
    n_oscillators: int,
    reference_material: str,
    tip: TipParameters,
    wmin: float,
    wmax: float,
) -> MultiLorentzSnomModel:
    return MultiLorentzSnomModel(
        n_oscillators=n_oscillators,
        centers=list(params["centers"]),
        strengths=list(params["strengths"]),
        gammas=list(params["gammas"]),
        eps_inf=float(params["eps_inf"]),
        tip=tip,
        reference_material=reference_material,
        bounds={
            "centers": (wmin, wmax),
            "strengths": (0.0, 30.0),
            "gammas": (10.0, 300.0),
            "eps_inf": (1.0, 30.0),
            "g_factor": (0.1, 1.5),
            "g_phase": (-0.5, 0.5),
        },
    )


def synthesize_spectrum(
    wavenumber: np.ndarray,
    params: dict[str, np.ndarray | float],
    n_oscillators: int,
    reference_material: str,
    tip: TipParameters,
    amp_normalization: str,
    reference_band: tuple[float, float],
    noise_std: float,
    device: torch.device,
) -> np.ndarray:
    clean_params = {
        "centers": [float(v) for v in np.asarray(params["centers"])],
        "strengths": [float(v) for v in np.asarray(params["strengths"])],
        "gammas": [float(v) for v in np.asarray(params["gammas"])],
        "eps_inf": float(params["eps_inf"]),
    }
    model = make_multilorentz_model(
        clean_params,
        n_oscillators,
        reference_material,
        tip,
        float(np.min(wavenumber)),
        float(np.max(wavenumber)),
    ).to(device)
    w = torch.tensor(wavenumber, dtype=torch.float64, device=device)
    with torch.no_grad():
        phase, amp = model(w)
    amp_np = amp.detach().cpu().numpy()
    phase_np = phase.detach().cpu().numpy()
    amp_np = normalize_amp_np(amp_np, wavenumber, amp_normalization, reference_band)
    phase_np = phase_np - np.nanmedian(phase_np)
    spectrum = np.stack([amp_np, phase_np], axis=0).astype(np.float32)
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=spectrum.shape).astype(np.float32)
        spectrum = spectrum + noise
    return spectrum


class SpectraToParamsNet(nn.Module):
    """Small inverse net: normalized amplitude+phase spectrum -> physical parameters."""

    def __init__(self, n_wavenumbers: int, n_params: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 + n_wavenumbers * 2, 256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, n_params),
            nn.Sigmoid(),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv(spectra).flatten(1)
        raw_features = spectra.flatten(1)
        return self.head(torch.cat([conv_features, raw_features], dim=1))
