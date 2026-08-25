"""Batch-level physical fitting for processed SNOM spectra.

This is the second attempt for the organoid / tonsil branch:
- input spectra are already processed spectra;
- no raw background chain is re-derived here;
- a single shared complex gain is fitted per batch;
- each sample point keeps its own Lorentz parameters.

The model is intentionally compact:
    chi(ω) = eps_inf + Σ_j strength_j * ω_j^2 / (ω_j^2 - ω^2 - i gamma_j ω)
    z(ω) = gain * exp(i * phase) * chi(ω)
    amp = |z|
    phase = arg(z)

This is not a strict absolute dielectric inversion. It is a relative,
physics-constrained fit meant for spectrum-shape reproduction and
within-class stability analysis.
Phase is treated as diagnostic by default so amplitude does not get dragged
down by a bad phase channel.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter


DEFAULT_WMIN = 690.0
DEFAULT_WMAX = 1800.0
DEFAULT_STRIDE = 4
DEFAULT_AMP_WEIGHT = 1.0
DEFAULT_PHASE_WEIGHT = 0.0
DEFAULT_PHASE_CALIBRATION_MODE = "none"
DEFAULT_PHASE_CALIBRATION_WINDOWS = ((690.0, 880.0), (1450.0, 1750.0))
EPS = 1e-12


@dataclass(frozen=True)
class SampleCurve:
    dataset_name: str
    sample_id: str
    class_label: str
    specimen_name: str
    marker: str
    tune_wavenumber: str
    wavenumber: np.ndarray
    amp: np.ndarray
    phase: np.ndarray


@dataclass(frozen=True)
class ReferenceSpectrum:
    path: Path
    wavenumber: np.ndarray
    complex_response: np.ndarray


@dataclass(frozen=True)
class PhaseCalibration:
    mode: str
    phi0: float
    phi1: float
    w0: float
    windows: tuple[tuple[float, float], ...]
    n_points: int


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def find_npz(dataset_dir: Path) -> Path:
    for name in ("sample_level_spectra.npz", "spectra_normalized.npz", "spectra_raw_reference.npz"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spectra npz found in {dataset_dir}")


def load_sample_curves(dataset_name: str, dataset_dir: Path) -> list[SampleCurve]:
    npz_path = find_npz(dataset_dir)
    meta_path = dataset_dir / "sample_level_summary.csv"
    if not meta_path.exists():
        meta_path = dataset_dir / "metadata_normalized.csv"
    if not meta_path.exists():
        meta_path = dataset_dir / "metadata.csv"

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path).fillna("")
    wavenumber = np.asarray(z["wavenumber"], dtype=np.float64)
    amp = np.asarray(z["amp_mean"] if "amp_mean" in z.files else z["o2a"], dtype=np.float64)
    phase = np.asarray(z["phase_mean"] if "phase_mean" in z.files else z["o2p"], dtype=np.float64)
    sample_ids = np.asarray(z["sample_id"], dtype=str)
    meta_by_sample = meta.set_index("sample_id", drop=False) if "sample_id" in meta.columns else None

    if len(meta) != len(sample_ids):
        raise ValueError(f"Metadata row count does not match spectra rows in {dataset_dir}")
    if meta_by_sample is not None:
        missing = sorted(set(sample_ids) - set(meta_by_sample.index.astype(str)))
        if missing:
            raise ValueError(f"Metadata missing sample_id rows in {dataset_dir}: {missing[:5]}")

    curves: list[SampleCurve] = []
    for i, sample_id in enumerate(sample_ids):
        row = meta_by_sample.loc[str(sample_id)] if meta_by_sample is not None else meta.iloc[i]
        specimen_name = str(row.get("specimen_name", ""))
        class_label = str(row.get("class_label", ""))
        if specimen_name.startswith("TA0038190758"):
            class_label = "organoid"
        curves.append(
            SampleCurve(
                dataset_name=dataset_name,
                sample_id=str(sample_id),
                class_label=class_label,
                specimen_name=specimen_name,
                marker=str(row.get("marker", "")),
                tune_wavenumber=str(row.get("tune_wavenumber", "")),
                wavenumber=wavenumber,
                amp=amp[i],
                phase=phase[i],
            )
        )
    return curves


def load_reference_spectrum(reference_path: Path, amp_col: str = "O2A", phase_col: str = "O2P") -> ReferenceSpectrum:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference spectrum not found: {reference_path}")

    df = pd.read_csv(reference_path, comment="#", sep="\t", engine="python").fillna("")
    df = df[[col for col in df.columns if not str(col).startswith("Unnamed")]]
    if "Wavenumber" not in df.columns or amp_col not in df.columns or phase_col not in df.columns:
        raise ValueError(f"Reference spectrum missing required columns in {reference_path}")

    w = pd.to_numeric(df["Wavenumber"], errors="coerce").to_numpy(dtype=np.float64)
    amp = pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=np.float64)
    phase = pd.to_numeric(df[phase_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(w) & np.isfinite(amp) & np.isfinite(phase)
    w = w[mask]
    amp = amp[mask]
    phase = phase[mask]
    order = np.argsort(w)
    w = w[order]
    amp = amp[order]
    phase = phase[order]
    complex_response = amp * np.exp(1j * phase)
    return ReferenceSpectrum(path=reference_path, wavenumber=w, complex_response=complex_response)


def reference_response_at(reference: ReferenceSpectrum | None, w: np.ndarray) -> np.ndarray:
    if reference is None:
        return np.ones_like(w, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)
    real = np.interp(w, reference.wavenumber, np.real(reference.complex_response))
    imag = np.interp(w, reference.wavenumber, np.imag(reference.complex_response))
    return real + 1j * imag


def compose_with_reference(
    amp: np.ndarray,
    phase: np.ndarray,
    w: np.ndarray,
    reference: ReferenceSpectrum | None,
) -> tuple[np.ndarray, np.ndarray]:
    ref = reference_response_at(reference, w)
    z = np.asarray(amp, dtype=np.float64) * np.exp(1j * np.asarray(phase, dtype=np.float64))
    z_total = z * ref
    return np.abs(z_total), np.angle(z_total)


def compose_complex_with_reference(
    z: np.ndarray,
    w: np.ndarray,
    reference: ReferenceSpectrum | None,
) -> np.ndarray:
    return np.asarray(z, dtype=np.complex128) * reference_response_at(reference, w)


def group_batches(curves: Iterable[SampleCurve]) -> dict[str, list[SampleCurve]]:
    batches: dict[str, list[SampleCurve]] = {}
    for curve in curves:
        key = f"{curve.dataset_name}__{curve.tune_wavenumber or 'na'}"
        batches.setdefault(key, []).append(curve)
    return batches


def select_window(w: np.ndarray, y: np.ndarray, wmin: float, wmax: float, stride: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (w >= wmin) & (w <= wmax)
    w_sel = w[mask]
    y_sel = y[mask]
    if stride > 1:
        w_sel = w_sel[::stride]
        y_sel = y_sel[::stride]
    return w_sel, y_sel


def wrap_phase(diff: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * diff))


def complex_spectrum_from_amp_phase(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.asarray(amp, dtype=np.float64) * np.exp(1j * np.asarray(phase, dtype=np.float64))


def circular_mean_phase(phases: np.ndarray, axis: int = 0) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float64)
    return np.angle(np.mean(np.exp(1j * phases), axis=axis))


def parse_phase_calib_windows(value: str) -> tuple[tuple[float, float], ...]:
    windows: list[tuple[float, float]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid phase calibration window: {chunk}")
        lo_s, hi_s = chunk.split(":", 1)
        windows.append((float(lo_s), float(hi_s)))
    if not windows:
        raise ValueError("At least one phase calibration window is required.")
    return tuple(windows)


def phase_window_mask(w: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros_like(np.asarray(w, dtype=np.float64), dtype=bool)
    for lo, hi in windows:
        mask |= (w >= lo) & (w <= hi)
    return mask


def phase_calibration_line(w: np.ndarray, calibration: PhaseCalibration | None) -> np.ndarray:
    if calibration is None or calibration.mode == "none":
        return np.zeros_like(np.asarray(w, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64)
    return calibration.phi0 + calibration.phi1 * (w - calibration.w0)


def apply_phase_calibration(phase: np.ndarray, w: np.ndarray, calibration: PhaseCalibration | None) -> np.ndarray:
    return wrap_phase(np.asarray(phase, dtype=np.float64) - phase_calibration_line(w, calibration))


def estimate_phase_calibration(
    w: np.ndarray,
    phase: np.ndarray,
    mode: str,
    windows: tuple[tuple[float, float], ...],
) -> PhaseCalibration:
    w = np.asarray(w, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    if mode == "none":
        return PhaseCalibration(mode="none", phi0=0.0, phi1=0.0, w0=float(np.median(w)), windows=windows, n_points=0)

    mask = phase_window_mask(w, windows)
    if not np.any(mask):
        mask = np.ones_like(w, dtype=bool)
    w_sel = w[mask]
    phase_sel = np.unwrap(phase[mask])
    w0 = float(np.median(w_sel))
    x = w_sel - w0

    if mode == "offset":
        phi0 = float(np.median(phase_sel))
        phi1 = 0.0
    elif mode == "slope":
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, phase_sel, rcond=None)
        phi0 = float(coef[0])
        phi1 = float(coef[1])
    else:
        raise ValueError(f"Unknown phase calibration mode: {mode}")

    return PhaseCalibration(mode=mode, phi0=phi0, phi1=phi1, w0=w0, windows=windows, n_points=int(len(w_sel)))


def calibrate_phase_array(
    phase: np.ndarray,
    w: np.ndarray,
    calibration: PhaseCalibration | None,
) -> np.ndarray:
    return apply_phase_calibration(phase, w, calibration)


def lorentz_chi(w: np.ndarray, eps_inf: float, centers: np.ndarray, gammas: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    chi = np.full_like(w, float(eps_inf), dtype=np.complex128)
    for center, gamma, strength in zip(centers, gammas, strengths):
        num = strength * (center**2)
        den = (center**2 - w**2) - 1j * gamma * w
        chi = chi + num / (den + 1e-18)
    return chi


def forward_spectrum(
    w: np.ndarray,
    eps_inf: float,
    centers: np.ndarray,
    gammas: np.ndarray,
    strengths: np.ndarray,
    gain: float,
    phase_shift: float,
    amp_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    chi = lorentz_chi(w, eps_inf, centers, gammas, strengths)
    chi_scale = np.median(np.abs(chi)) + EPS
    z = gain * np.exp(1j * phase_shift) * (chi / chi_scale)
    amp = np.abs(z) + amp_offset
    phase = np.angle(z)
    return amp, phase


def initial_center_guesses(w: np.ndarray, amp: np.ndarray, n_lorentz: int = 3) -> np.ndarray:
    smooth = savgol_filter(amp, 15 if len(amp) >= 15 else max(5, len(amp) // 2 * 2 + 1), 3)
    peaks, props = find_peaks(smooth, prominence=max(1e-3, 0.03 * np.ptp(smooth)))
    if len(peaks) >= n_lorentz:
        ranked = peaks[np.argsort(smooth[peaks])[-n_lorentz:]]
        centers = np.sort(w[ranked])
    else:
        default = np.array([920.0, 1275.0, 1650.0], dtype=np.float64)
        centers = default[:n_lorentz]
        if len(centers) < n_lorentz:
            extra = np.linspace(max(w.min() + 40.0, 780.0), min(w.max() - 40.0, 1700.0), n_lorentz)
            centers = extra
    return np.asarray(centers, dtype=np.float64)


def sample_loss_vector(
    x: np.ndarray,
    w: np.ndarray,
    amp_true: np.ndarray,
    phase_true: np.ndarray,
    n_lorentz: int = 3,
    use_amp_offset: bool = False,
    amp_weight: float = 1.0,
    phase_weight: float = 0.15,
    reference: ReferenceSpectrum | None = None,
) -> np.ndarray:
    eps_inf = x[0]
    centers = x[1 : 1 + n_lorentz]
    gammas = x[1 + n_lorentz : 1 + 2 * n_lorentz]
    strengths = x[1 + 2 * n_lorentz : 1 + 3 * n_lorentz]
    gain = x[1 + 3 * n_lorentz]
    phase_shift = x[2 + 3 * n_lorentz]
    amp_offset = x[3 + 3 * n_lorentz] if use_amp_offset else 0.0

    amp_pred, phase_pred = forward_spectrum(
        w, eps_inf, centers, gammas, strengths, gain, phase_shift, amp_offset=amp_offset
    )
    amp_pred, phase_pred = compose_with_reference(amp_pred, phase_pred, w, reference)
    amp_scale = np.median(np.abs(amp_true)) + EPS
    amp_res = (amp_pred - amp_true) / amp_scale
    phase_res = wrap_phase(phase_pred - phase_true)
    return np.concatenate([amp_weight * amp_res, phase_weight * phase_res])


def sample_complex_loss_vector(
    x: np.ndarray,
    w: np.ndarray,
    z_true: np.ndarray,
    n_lorentz: int = 3,
    reference: ReferenceSpectrum | None = None,
) -> np.ndarray:
    eps_inf = x[0]
    centers = x[1 : 1 + n_lorentz]
    gammas = x[1 + n_lorentz : 1 + 2 * n_lorentz]
    strengths = x[1 + 2 * n_lorentz : 1 + 3 * n_lorentz]
    gain = x[1 + 3 * n_lorentz]
    phase_shift = x[2 + 3 * n_lorentz]

    amp_pred, phase_pred = forward_spectrum(
        w, eps_inf, centers, gammas, strengths, gain, phase_shift, amp_offset=0.0
    )
    z_pred = compose_complex_with_reference(complex_spectrum_from_amp_phase(amp_pred, phase_pred), w, reference)
    z_scale = np.median(np.abs(z_true)) + EPS
    z_res = (z_pred - z_true) / z_scale
    return np.concatenate([np.real(z_res), np.imag(z_res)])


def fit_single_sample(
    curve: SampleCurve,
    shared: dict[str, float],
    wmin: float,
    wmax: float,
    stride: int,
    n_lorentz: int = 3,
    use_amp_offset: bool = False,
    amp_weight: float = DEFAULT_AMP_WEIGHT,
    phase_weight: float = DEFAULT_PHASE_WEIGHT,
    fit_space: str = "amp_phase",
    reference: ReferenceSpectrum | None = None,
    phase_calibration: PhaseCalibration | None = None,
) -> dict[str, object]:
    w, amp = select_window(curve.wavenumber, curve.amp, wmin, wmax, stride)
    _, phase_raw = select_window(curve.wavenumber, curve.phase, wmin, wmax, stride)
    phase = apply_phase_calibration(phase_raw, w, phase_calibration)
    if len(w) < 30:
        raise ValueError(f"Too few points after windowing for {curve.sample_id}")

    centers0 = initial_center_guesses(w, amp, n_lorentz=n_lorentz)
    if len(centers0) < n_lorentz:
        centers0 = np.pad(centers0, (0, n_lorentz - len(centers0)), mode="edge")
    gamma0 = np.full(n_lorentz, 80.0, dtype=np.float64)
    strength0 = np.full(n_lorentz, 1.0, dtype=np.float64)
    eps_inf0 = float(np.clip(np.median(amp[-max(10, len(amp) // 10) :]), 0.1, 20.0))
    gain0 = float(shared["gain"])
    phase0 = float(shared["phase"])
    amp_offset0 = float(shared.get("amp_offset", 0.0)) if use_amp_offset else None
    if fit_space == "complex" and use_amp_offset:
        raise ValueError("Complex fit space does not support amp_offset.")

    x0 = [eps_inf0, *centers0.tolist(), *gamma0.tolist(), *strength0.tolist(), gain0, phase0]
    bounds_lo = [0.1] + [wmin] * n_lorentz + [5.0] * n_lorentz + [0.0] * n_lorentz + [0.01, -math.pi]
    bounds_hi = [20.0] + [wmax] * n_lorentz + [300.0] * n_lorentz + [20.0] * n_lorentz + [10.0, math.pi]
    if use_amp_offset:
        x0.append(amp_offset0 if amp_offset0 is not None else 0.0)
        bounds_lo.append(-0.5)
        bounds_hi.append(0.5)

    bounds = (np.asarray(bounds_lo, dtype=np.float64), np.asarray(bounds_hi, dtype=np.float64))
    x_start = np.asarray(x0, dtype=np.float64)
    amp_only = least_squares(
        sample_loss_vector,
        x_start,
        bounds=bounds,
        args=(w, amp, phase, n_lorentz, use_amp_offset, amp_weight, 0.0, reference),
        max_nfev=220,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    if fit_space == "complex":
        z_true = complex_spectrum_from_amp_phase(amp, phase)
        res = least_squares(
            sample_complex_loss_vector,
            amp_only.x,
            bounds=bounds,
            args=(w, z_true, n_lorentz, reference),
            max_nfev=280,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
    else:
        res = least_squares(
            sample_loss_vector,
            amp_only.x,
            bounds=bounds,
            args=(w, amp, phase, n_lorentz, use_amp_offset, amp_weight, phase_weight, reference),
            max_nfev=260,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

    x = res.x
    eps_inf = float(x[0])
    centers = x[1 : 1 + n_lorentz].astype(float)
    gammas = x[1 + n_lorentz : 1 + 2 * n_lorentz].astype(float)
    strengths = x[1 + 2 * n_lorentz : 1 + 3 * n_lorentz].astype(float)
    gain = float(x[1 + 3 * n_lorentz])
    phase_shift = float(x[2 + 3 * n_lorentz])
    amp_offset = float(x[3 + 3 * n_lorentz]) if use_amp_offset else 0.0
    amp_pred, phase_pred = forward_spectrum(w, eps_inf, centers, gammas, strengths, gain, phase_shift, amp_offset)
    amp_pred, phase_pred = compose_with_reference(amp_pred, phase_pred, w, reference)
    z_true = complex_spectrum_from_amp_phase(amp, phase)
    z_pred = complex_spectrum_from_amp_phase(amp_pred, phase_pred)
    z_scale = np.median(np.abs(z_true)) + EPS
    z_res = (z_pred - z_true) / z_scale
    amp_scale = np.median(np.abs(amp)) + EPS
    amp_res = (amp_pred - amp) / amp_scale
    phase_res = wrap_phase(phase_pred - phase)

    return {
        "sample_id": curve.sample_id,
        "class_label": curve.class_label,
        "specimen_name": curve.specimen_name,
        "marker": curve.marker,
        "tune_wavenumber": curve.tune_wavenumber,
        "reference_file": str(reference.path) if reference is not None else "",
        "reference_mode": "fixed_multiplicative_layer" if reference is not None else "none",
        "fit_space": fit_space,
        "eps_inf": eps_inf,
        "centers": centers.tolist(),
        "gammas": gammas.tolist(),
        "strengths": strengths.tolist(),
        "gain": gain,
        "phase_shift": phase_shift,
        "amp_offset": amp_offset,
        "amp_mse": float(np.mean((amp_pred - amp) ** 2)),
        "phase_circular_mse": float(np.mean(phase_res**2)),
        "amp_relative_mse": float(np.mean(amp_res**2)),
        "complex_mse": float(np.mean(np.abs(z_res) ** 2)),
        "complex_real_mse": float(np.mean(np.real(z_res) ** 2)),
        "complex_imag_mse": float(np.mean(np.imag(z_res) ** 2)),
        "residual_l2": float(np.mean(np.concatenate([amp_res, phase_res]) ** 2)),
        "analysis_group": f"{curve.specimen_name}__{curve.tune_wavenumber or 'na'}",
        "n_points": int(len(w)),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
    }


def fit_batch(
    batch_name: str,
    curves: list[SampleCurve],
    output_dir: Path,
    wmin: float,
    wmax: float,
    stride: int,
    n_lorentz: int,
    use_amp_offset: bool,
    n_outer_iter: int,
    amp_weight: float,
    phase_weight: float,
    fit_space: str,
    phase_calibration_mode: str,
    phase_calibration_windows: tuple[tuple[float, float], ...],
    reference: ReferenceSpectrum | None,
) -> dict[str, object]:
    batch_output = output_dir / batch_name
    batch_output.mkdir(parents=True, exist_ok=True)

    batch_mean_amp = np.mean([curve.amp for curve in curves], axis=0)
    batch_mean_phase = np.mean([curve.phase for curve in curves], axis=0)
    w_full = curves[0].wavenumber
    w, mean_amp = select_window(w_full, batch_mean_amp, wmin, wmax, stride)
    _, mean_phase_raw = select_window(w_full, batch_mean_phase, wmin, wmax, stride)
    phase_calibration = estimate_phase_calibration(w, mean_phase_raw, phase_calibration_mode, phase_calibration_windows)
    mean_phase = apply_phase_calibration(mean_phase_raw, w, phase_calibration)

    shared = {
        "gain": 0.7,
        "phase": 0.0,
        "amp_offset": 0.0,
    }

    sample_results: list[dict[str, object]] = []
    for _ in range(max(1, n_outer_iter)):
        sample_results = [
            fit_single_sample(
                curve,
                shared=shared,
                wmin=wmin,
                wmax=wmax,
                stride=stride,
                n_lorentz=n_lorentz,
                use_amp_offset=use_amp_offset,
                amp_weight=amp_weight,
                phase_weight=phase_weight,
                fit_space=fit_space,
                reference=reference,
                phase_calibration=phase_calibration,
            )
            for curve in curves
        ]

        def shared_loss(x: np.ndarray) -> np.ndarray:
            gain = float(x[0])
            phase_shift = float(x[1])
            amp_offset = float(x[2]) if use_amp_offset else 0.0
            residuals = []
            for curve, sample in zip(curves, sample_results):
                w_loc, amp_loc = select_window(curve.wavenumber, curve.amp, wmin, wmax, stride)
                _, phase_loc_raw = select_window(curve.wavenumber, curve.phase, wmin, wmax, stride)
                phase_loc = apply_phase_calibration(phase_loc_raw, w_loc, phase_calibration)
                centers = np.asarray(sample["centers"], dtype=np.float64)
                gammas = np.asarray(sample["gammas"], dtype=np.float64)
                strengths = np.asarray(sample["strengths"], dtype=np.float64)
                amp_pred, phase_pred = forward_spectrum(
                    w_loc,
                    float(sample["eps_inf"]),
                    centers,
                    gammas,
                    strengths,
                    gain,
                    phase_shift,
                    amp_offset=amp_offset,
                )
                amp_pred, phase_pred = compose_with_reference(amp_pred, phase_pred, w_loc, reference)
                if fit_space == "complex":
                    z_true = complex_spectrum_from_amp_phase(amp_loc, phase_loc)
                    z_pred = complex_spectrum_from_amp_phase(amp_pred, phase_pred)
                    z_scale = np.median(np.abs(z_true)) + EPS
                    z_res = (z_pred - z_true) / z_scale
                    residuals.append(np.real(z_res))
                    residuals.append(np.imag(z_res))
                else:
                    amp_scale = np.median(np.abs(amp_loc)) + EPS
                    residuals.append(amp_weight * (amp_pred - amp_loc) / amp_scale)
                    residuals.append(phase_weight * wrap_phase(phase_pred - phase_loc))
            return np.concatenate(residuals)

        x0 = np.array([shared["gain"], shared["phase"], shared["amp_offset"]], dtype=np.float64)
        bounds_lo = np.array([0.01, -math.pi, -0.5], dtype=np.float64) if use_amp_offset else np.array([0.01, -math.pi], dtype=np.float64)
        bounds_hi = np.array([10.0, math.pi, 0.5], dtype=np.float64) if use_amp_offset else np.array([10.0, math.pi], dtype=np.float64)
        res = least_squares(
            shared_loss,
            x0 if use_amp_offset else x0[:2],
            bounds=(bounds_lo, bounds_hi),
            max_nfev=200,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        shared["gain"] = float(res.x[0])
        shared["phase"] = float(res.x[1])
        if use_amp_offset:
            shared["amp_offset"] = float(res.x[2])

    rows = []
    for curve, sample in zip(curves, sample_results):
        w_loc, amp_loc = select_window(curve.wavenumber, curve.amp, wmin, wmax, stride)
        _, phase_loc_raw = select_window(curve.wavenumber, curve.phase, wmin, wmax, stride)
        phase_loc = apply_phase_calibration(phase_loc_raw, w_loc, phase_calibration)
        centers = np.asarray(sample["centers"], dtype=np.float64)
        gammas = np.asarray(sample["gammas"], dtype=np.float64)
        strengths = np.asarray(sample["strengths"], dtype=np.float64)
        amp_pred, phase_pred = forward_spectrum(
            w_loc,
            float(sample["eps_inf"]),
            centers,
            gammas,
            strengths,
            shared["gain"],
            shared["phase"],
            amp_offset=shared.get("amp_offset", 0.0),
        )
        amp_pred, phase_pred = compose_with_reference(amp_pred, phase_pred, w_loc, reference)
        amp_res = amp_pred - amp_loc
        phase_res = wrap_phase(phase_pred - phase_loc)
        z_true = complex_spectrum_from_amp_phase(amp_loc, phase_loc)
        z_pred = complex_spectrum_from_amp_phase(amp_pred, phase_pred)
        z_scale = np.median(np.abs(z_true)) + EPS
        z_res = (z_pred - z_true) / z_scale
        row = {
            "batch": batch_name,
            "dataset_name": curve.dataset_name,
            "sample_id": curve.sample_id,
            "class_label": curve.class_label,
            "specimen_name": curve.specimen_name,
            "marker": curve.marker,
            "tune_wavenumber": curve.tune_wavenumber,
            "reference_file": str(reference.path) if reference is not None else "",
            "reference_mode": "fixed_multiplicative_layer" if reference is not None else "none",
            "fit_space": fit_space,
            "phase_calibration_mode": phase_calibration.mode,
            "phase_calibration_phi0": phase_calibration.phi0,
            "phase_calibration_phi1": phase_calibration.phi1,
            "phase_calibration_w0": phase_calibration.w0,
            "analysis_group": f"{curve.specimen_name}__{curve.tune_wavenumber or 'na'}",
            "n_points": int(len(w_loc)),
            "eps_inf": float(sample["eps_inf"]),
            "gain": float(shared["gain"]),
            "phase_shift": float(shared["phase"]),
            "amp_offset": float(shared.get("amp_offset", 0.0)),
            "amp_mse": float(np.mean(amp_res**2)),
            "phase_circular_mse": float(np.mean(phase_res**2)),
            "amp_relative_mse": float(sample["amp_relative_mse"]),
            "complex_mse": float(np.mean(np.abs(z_res) ** 2)),
            "complex_real_mse": float(np.mean(np.real(z_res) ** 2)),
            "complex_imag_mse": float(np.mean(np.imag(z_res) ** 2)),
            "residual_l2": float(sample["residual_l2"]),
            "success": bool(sample["success"]),
        }
        for idx, value in enumerate(sample["centers"]):
            row[f"center_{idx+1}"] = float(value)
        for idx, value in enumerate(sample["gammas"]):
            row[f"gamma_{idx+1}"] = float(value)
        for idx, value in enumerate(sample["strengths"]):
            row[f"strength_{idx+1}"] = float(value)
        rows.append(row)

    detail = pd.DataFrame(rows).sort_values(["analysis_group", "sample_id"]).reset_index(drop=True)
    detail.to_csv(batch_output / "sample_fit_results.csv", index=False, encoding="utf-8-sig")

    agg_spec = {
        "n_samples": ("sample_id", "count"),
        "amp_mse_mean": ("amp_mse", "mean"),
        "phase_circular_mse_mean": ("phase_circular_mse", "mean"),
        "residual_l2_mean": ("residual_l2", "mean"),
    }
    for idx in range(1, n_lorentz + 1):
        agg_spec[f"center_{idx}_mean"] = (f"center_{idx}", "mean")
        agg_spec[f"center_{idx}_std"] = (f"center_{idx}", "std")

    class_summary = detail.groupby("analysis_group", sort=True).agg(**agg_spec).reset_index()
    class_summary.to_csv(batch_output / "class_fit_summary.csv", index=False, encoding="utf-8-sig")

    batch_summary = {
        "batch": batch_name,
        "dataset_name": curves[0].dataset_name,
        "tune_wavenumber": curves[0].tune_wavenumber,
        "n_samples": len(curves),
        "n_classes": int(detail["class_label"].nunique()),
        "wmin": wmin,
        "wmax": wmax,
        "stride": stride,
        "n_lorentz": n_lorentz,
        "shared_gain": shared["gain"],
        "shared_phase": shared["phase"],
        "shared_amp_offset": shared.get("amp_offset", 0.0),
        "reference_file": str(reference.path) if reference is not None else "",
        "reference_mode": "fixed_multiplicative_layer" if reference is not None else "none",
        "fit_space": fit_space,
        "phase_calibration_mode": phase_calibration.mode,
        "phase_calibration_phi0": phase_calibration.phi0,
        "phase_calibration_phi1": phase_calibration.phi1,
        "phase_calibration_w0": phase_calibration.w0,
        "amp_mse_mean": float(detail["amp_mse"].mean()),
        "phase_circular_mse_mean": float(detail["phase_circular_mse"].mean()),
        "complex_mse_mean": float(detail["complex_mse"].mean()),
        "residual_l2_mean": float(detail["residual_l2"].mean()),
        "classes": "|".join(sorted(detail["class_label"].astype(str).unique())),
        "analysis_groups": "|".join(sorted(detail["analysis_group"].astype(str).unique())),
    }
    (batch_output / "batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return batch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-level relative physical fit for processed SNOM spectra.")
    parser.add_argument("--dataset", action="append", required=True, help="Dataset path, optionally as name=path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wmin", type=float, default=DEFAULT_WMIN)
    parser.add_argument("--wmax", type=float, default=DEFAULT_WMAX)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--n-lorentz", type=int, default=3)
    parser.add_argument("--n-outer-iter", type=int, default=2)
    parser.add_argument("--use-amp-offset", action="store_true")
    parser.add_argument("--amp-weight", type=float, default=DEFAULT_AMP_WEIGHT)
    parser.add_argument("--phase-weight", type=float, default=DEFAULT_PHASE_WEIGHT)
    parser.add_argument("--fit-space", choices=("amp_phase", "complex"), default="amp_phase")
    parser.add_argument(
        "--phase-calibration-mode",
        choices=("none", "offset", "slope"),
        default=DEFAULT_PHASE_CALIBRATION_MODE,
    )
    parser.add_argument(
        "--phase-calibration-windows",
        default="690:880,1450:1750",
        help="Comma-separated low-structure windows as lo:hi pairs.",
    )
    parser.add_argument("--class-filter", action="append", default=None, help="Optional class label filters.")
    parser.add_argument("--reference-file", default=None, help="Optional fixed reference spectrum used as a multiplicative background layer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference_spectrum(Path(args.reference_file)) if args.reference_file else None
    phase_calibration_windows = parse_phase_calib_windows(args.phase_calibration_windows)
    if args.fit_space == "complex" and args.use_amp_offset:
        raise ValueError("Complex fit space does not support --use-amp-offset.")

    all_curves: list[SampleCurve] = []
    for dataset_arg in args.dataset:
        dataset_name, dataset_dir = parse_dataset_arg(dataset_arg)
        all_curves.extend(load_sample_curves(dataset_name, dataset_dir))

    if args.class_filter:
        allowed = {item.strip() for item in args.class_filter if item.strip()}
        all_curves = [curve for curve in all_curves if curve.class_label in allowed]

    if not all_curves:
        raise RuntimeError("No sample curves available after filtering.")

    batch_map = group_batches(all_curves)
    batch_rows = []
    for batch_name, curves in sorted(batch_map.items()):
        batch_rows.append(
            fit_batch(
                batch_name=batch_name,
                curves=curves,
                output_dir=output_dir,
                wmin=args.wmin,
                wmax=args.wmax,
                stride=args.stride,
                n_lorentz=args.n_lorentz,
                use_amp_offset=args.use_amp_offset,
                n_outer_iter=args.n_outer_iter,
                amp_weight=args.amp_weight,
                phase_weight=args.phase_weight,
                fit_space=args.fit_space,
                phase_calibration_mode=args.phase_calibration_mode,
                phase_calibration_windows=phase_calibration_windows,
                reference=reference,
            )
        )

    summary = pd.DataFrame(batch_rows)
    summary.to_csv(output_dir / "batch_fit_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
