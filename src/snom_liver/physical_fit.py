"""Fit the clean semi-infinite S-SNOM model to one processed spectrum."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snom_liver.physics import MultiLorentzSnomModel, SemiInfiniteSnomModel, SinglePhononParameters, TipParameters
from snom_liver.preprocess import read_neaspec_txt


DATASET_NPZ_NAMES = ("spectra_normalized.npz", "spectra_raw_reference.npz")
EPS = 1e-12


def find_npz(path: Path) -> Path:
    if path.is_file():
        return path
    for name in DATASET_NPZ_NAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spectra npz found in {path}")


def find_metadata(path: Path, npz_path: Path) -> Path | None:
    folder = path if path.is_dir() else npz_path.parent
    for name in ("metadata_normalized.csv", "metadata.csv"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def parse_header_float(path: Path, label: str) -> float | None:
    if not path or not path.exists():
        return None
    pattern = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if label.lower() not in line.lower():
            continue
        matches = pattern.findall(line)
        if not matches:
            return None
        return float(matches[-1].replace(",", ""))
    return None


def interpolate_bg_spectrum(bg_file: Path, wavenumber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = read_neaspec_txt(bg_file)
    bg_w = bg["Wavenumber"].to_numpy(dtype=np.float64)
    bg_amp = np.interp(wavenumber, bg_w, bg["O2A"].to_numpy(dtype=np.float64))
    bg_phase = np.interp(wavenumber, bg_w, bg["O2P"].to_numpy(dtype=np.float64))
    return bg_amp, bg_phase


def _processing_mode(data) -> str:
    if "processing_mode" not in data.files:
        return ""
    value = data["processing_mode"]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    return str(value)


def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def _amp_center_scale(amp: np.ndarray) -> np.ndarray:
    return amp / (np.nanmedian(np.abs(amp)) + EPS)


def als_baseline_np(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline estimate for a 1D spectrum."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    values = np.asarray(y, dtype=np.float64)
    length = values.size
    if length < 3:
        return np.full_like(values, np.nanmedian(values), dtype=np.float64)
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    weights = np.ones(length, dtype=np.float64)
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        z = weight_matrix + lam * diff.T @ diff
        baseline = spsolve(z, weights * values)
        weights = p * (values > baseline) + (1.0 - p) * (values <= baseline)
    return np.asarray(baseline, dtype=np.float64)


def apply_amp_baseline_np(
    amp: np.ndarray,
    mode: str,
    lam: float,
    p: float,
    niter: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "none":
        return amp, None
    baseline = als_baseline_np(amp, lam=lam, p=p, niter=niter)
    baseline_level = np.nanmedian(baseline)
    if mode == "als-subtract":
        return amp - baseline + baseline_level, baseline
    if mode == "als-divide":
        return amp / (baseline + EPS) * baseline_level, baseline
    raise ValueError(f"Unsupported amp baseline mode: {mode}")


def peak_initial_centers(
    wavenumber: np.ndarray,
    amp: np.ndarray,
    n_centers: int,
    wmin: float,
    wmax: float,
    min_spacing: float = 80.0,
) -> list[float]:
    """Pick data-driven initial Lorentz centers from prominent amplitude deviations."""
    w = np.asarray(wavenumber, dtype=np.float64)
    y = np.asarray(amp, dtype=np.float64)
    if len(w) == 0:
        return np.linspace(wmin, wmax, n_centers + 2)[1:-1].tolist()
    smooth_window = max(3, min(9, len(y) // 5 * 2 + 1))
    if smooth_window >= 3:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        pad = smooth_window // 2
        smooth = np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
    else:
        smooth = y
    score = np.abs(smooth - np.nanmedian(smooth))
    order = np.argsort(score)[::-1]
    chosen: list[float] = []
    for idx in order:
        center = float(w[idx])
        if center < wmin or center > wmax:
            continue
        if all(abs(center - existing) >= min_spacing for existing in chosen):
            chosen.append(center)
        if len(chosen) == n_centers:
            break
    if len(chosen) < n_centers:
        fallback = np.linspace(wmin, wmax, n_centers + 2)[1:-1]
        for center in fallback:
            value = float(center)
            if all(abs(value - existing) >= min_spacing for existing in chosen):
                chosen.append(value)
            if len(chosen) == n_centers:
                break
    return sorted(chosen[:n_centers])


def normalize_amp_np(
    amp: np.ndarray,
    wavenumber: np.ndarray,
    mode: str,
    reference_band: tuple[float, float],
) -> np.ndarray:
    if mode == "none":
        return amp
    if mode == "median":
        return amp / (np.nanmedian(np.abs(amp)) + EPS)
    if mode in {"zscore", "snv"}:
        return (amp - np.nanmean(amp)) / (np.nanstd(amp) + EPS)
    if mode == "reference-band":
        keep = (wavenumber >= reference_band[0]) & (wavenumber <= reference_band[1])
        if not np.any(keep):
            raise ValueError(f"No points in reference band {reference_band}")
        return amp / (np.nanmedian(np.abs(amp[keep])) + EPS)
    raise ValueError(f"Unsupported amp normalization: {mode}")


def normalize_amp_torch(
    amp: torch.Tensor,
    wavenumber: torch.Tensor,
    mode: str,
    reference_band: tuple[float, float],
) -> torch.Tensor:
    if mode == "none":
        return amp
    if mode == "median":
        return amp / (torch.median(torch.abs(amp)) + EPS)
    if mode in {"zscore", "snv"}:
        return (amp - torch.mean(amp)) / (torch.std(amp) + EPS)
    if mode == "reference-band":
        keep = (wavenumber >= reference_band[0]) & (wavenumber <= reference_band[1])
        if not bool(torch.any(keep)):
            raise ValueError(f"No points in reference band {reference_band}")
        return amp / (torch.median(torch.abs(amp[keep])) + EPS)
    raise ValueError(f"Unsupported amp normalization: {mode}")


def load_spectrum(
    dataset_path: Path,
    spectrum_index: int,
    wmin: float,
    wmax: float,
    step: int,
    fit_mode: str,
    amp_normalization: str = "median",
    reference_band: tuple[float, float] = (1500.0, 1600.0),
    amp_baseline: str = "none",
    als_lam: float = 1e5,
    als_p: float = 0.01,
    als_niter: int = 10,
):
    npz_path = find_npz(dataset_path)
    metadata_path = find_metadata(dataset_path, npz_path)
    metadata = pd.read_csv(metadata_path).fillna("") if metadata_path else None
    data = np.load(npz_path, allow_pickle=True)
    wavenumber = np.asarray(data["wavenumber"], dtype=np.float64)
    processing_mode = _processing_mode(data)
    if fit_mode == "auto":
        fit_mode = "measured-bg" if processing_mode == "bg-normalized" else "raw-reference"

    if fit_mode == "measured-bg":
        required = {"o2a_raw", "o2p_raw", "o2a_norm", "o2p_norm"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"measured-bg mode requires {missing} in {npz_path}")
        amp_key = "o2a_norm"
        phase_key = "o2p_norm"
    elif fit_mode == "raw-reference":
        amp_key = "o2a_raw" if "o2a_raw" in data.files else "o2a"
        phase_key = "o2p_raw" if "o2p_raw" in data.files else "o2p"
    elif fit_mode == "processed-direct":
        amp_key = "o2a_norm" if "o2a_norm" in data.files else "o2a"
        phase_key = "o2p_norm" if "o2p_norm" in data.files else "o2p"
    else:
        raise ValueError(f"Unsupported fit mode: {fit_mode}")

    indices = np.where((wavenumber >= wmin) & (wavenumber <= wmax))[0][::step]
    if len(indices) == 0:
        raise ValueError("No wavenumbers remain after filtering")

    amp = np.asarray(data[amp_key][spectrum_index, indices], dtype=np.float64)
    phase = np.asarray(data[phase_key][spectrum_index, indices], dtype=np.float64)
    amp, amp_baseline_values = apply_amp_baseline_np(amp, amp_baseline, als_lam, als_p, als_niter)

    bg_amp = None
    bg_phase = None
    bg_reference_source = ""
    source_file = ""
    sample_id = ""
    point_id = ""
    class_label = ""
    specimen_name = ""
    marker = ""
    tip_frequency_hz = None
    tapping_amplitude_nm = None

    if metadata is not None and spectrum_index < len(metadata):
        row = metadata.iloc[spectrum_index]
        sample_id = str(row.get("sample_id", ""))
        point_id = str(row.get("point_id", ""))
        class_label = str(row.get("class_label", ""))
        specimen_name = str(row.get("specimen_name", ""))
        marker = str(row.get("marker", ""))
        source_file = str(row.get("source_file", ""))
        source_path = Path(source_file) if source_file else None
        if source_path:
            tip_frequency_hz = parse_header_float(source_path, "Tip Frequency")
            tapping_amplitude_nm = parse_header_float(source_path, "Tapping Amplitude")

    if fit_mode == "measured-bg":
        bg_file = ""
        if metadata is not None and spectrum_index < len(metadata):
            bg_file = str(metadata.iloc[spectrum_index].get("background_file", ""))
        if bg_file and Path(bg_file).exists():
            bg_amp, bg_phase = interpolate_bg_spectrum(Path(bg_file), wavenumber[indices])
            bg_reference_source = bg_file
        else:
            raw_amp = np.asarray(data["o2a_raw"][spectrum_index, indices], dtype=np.float64)
            raw_phase = np.asarray(data["o2p_raw"][spectrum_index, indices], dtype=np.float64)
            bg_amp = raw_amp / (amp + EPS)
            bg_phase = raw_phase - phase
            bg_reference_source = "derived_from_raw_over_norm"

    amp = normalize_amp_np(amp, wavenumber[indices], amp_normalization, reference_band)
    phase = phase - np.nanmedian(phase)

    return {
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path) if metadata_path else "",
        "wavenumber": wavenumber[indices],
        "amp": amp,
        "phase": phase,
        "bg_amp": bg_amp,
        "bg_phase": bg_phase,
        "amp_key": amp_key,
        "phase_key": phase_key,
        "fit_mode": fit_mode,
        "processing_mode": processing_mode,
        "bg_reference_source": bg_reference_source,
        "source_file": source_file,
        "sample_id": sample_id,
        "point_id": point_id,
        "class_label": class_label,
        "specimen_name": specimen_name,
        "marker": marker,
        "tip_frequency_hz": tip_frequency_hz,
        "tapping_amplitude_nm": tapping_amplitude_nm,
        "amp_normalization": amp_normalization,
        "reference_band": reference_band,
        "amp_baseline": amp_baseline,
        "als_lam": als_lam,
        "als_p": als_p,
        "als_niter": als_niter,
        "amp_baseline_values": amp_baseline_values,
    }


def apply_experimental_reference(
    pred_phase: torch.Tensor,
    pred_amp: torch.Tensor,
    bg_amp: torch.Tensor | None,
    bg_phase: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bg_amp is None or bg_phase is None:
        return pred_phase, pred_amp
    pred_amp = pred_amp / (bg_amp + EPS)
    pred_phase = _wrap_phase(pred_phase - bg_phase)
    return pred_phase, pred_amp


def apply_amp_correction(
    pred_amp: torch.Tensor,
    amp_scale: torch.Tensor,
    amp_offset: torch.Tensor | None,
) -> torch.Tensor:
    corrected = amp_scale * pred_amp
    if amp_offset is not None:
        corrected = corrected + amp_offset
    return corrected


def least_squares_amp_correction(
    pred_amp: torch.Tensor,
    target_amp: torch.Tensor,
    scale_min: float,
    scale_max: float,
    offset_bound: float,
) -> tuple[float, float]:
    x = pred_amp.detach()
    y = target_amp.detach()
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    denom = torch.sum((x - x_mean) ** 2)
    if float(denom.cpu()) < EPS:
        return 1.0, float(y_mean.cpu())
    scale = torch.sum((x - x_mean) * (y - y_mean)) / denom
    offset = y_mean - scale * x_mean
    return (
        float(torch.clamp(scale, scale_min, scale_max).cpu()),
        float(torch.clamp(offset, -offset_bound, offset_bound).cpu()),
    )


def relative_amp_loss(pred_amp: torch.Tensor, target_amp: torch.Tensor) -> torch.Tensor:
    denom = torch.abs(target_amp) + 0.05 * torch.median(torch.abs(target_amp)) + EPS
    return torch.mean(((pred_amp - target_amp) / denom) ** 2)


def shape_loss(pred_amp: torch.Tensor, target_amp: torch.Tensor) -> torch.Tensor:
    pred_norm = pred_amp / (torch.median(torch.abs(pred_amp)) + EPS)
    target_norm = target_amp / (torch.median(torch.abs(target_amp)) + EPS)
    return torch.mean((torch.diff(pred_norm) - torch.diff(target_norm)) ** 2)


def circular_phase_loss(pred_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
    return torch.mean(1.0 - torch.cos(pred_phase - target_phase))


def combined_loss(
    pred_amp: torch.Tensor,
    target_amp: torch.Tensor,
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    phase_weight: float,
    shape_weight: float,
    amp_normalization: str = "median",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if amp_normalization in {"zscore", "snv"}:
        amp_rel = torch.mean((pred_amp - target_amp) ** 2)
    else:
        amp_rel = relative_amp_loss(pred_amp, target_amp)
    amp_shape = shape_loss(pred_amp, target_amp)
    phase_circular = circular_phase_loss(pred_phase, target_phase)
    total = amp_rel + shape_weight * amp_shape + phase_weight * phase_circular
    return total, {
        "amp_relative_loss": amp_rel,
        "amp_shape_loss": amp_shape,
        "phase_circular_loss": phase_circular,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    loaded = load_spectrum(
        Path(args.dataset),
        args.spectrum_index,
        args.wmin,
        args.wmax,
        args.step,
        args.fit_mode,
        args.amp_normalization,
        (args.reference_band_min, args.reference_band_max),
        args.amp_baseline,
        args.als_lam,
        args.als_p,
        args.als_niter,
    )
    w = torch.tensor(loaded["wavenumber"], dtype=torch.float64, device=device)
    target_amp = torch.tensor(loaded["amp"], dtype=torch.float64, device=device)
    target_phase = torch.tensor(loaded["phase"], dtype=torch.float64, device=device)
    bg_amp = (
        torch.tensor(loaded["bg_amp"], dtype=torch.float64, device=device)
        if loaded["bg_amp"] is not None
        else None
    )
    bg_phase = (
        torch.tensor(loaded["bg_phase"], dtype=torch.float64, device=device)
        if loaded["bg_phase"] is not None
        else None
    )

    tip_frequency_hz = loaded["tip_frequency_hz"] or args.tip_frequency_hz
    tapping_amplitude_nm = loaded["tapping_amplitude_nm"] or args.tapping_amplitude_nm
    tip = TipParameters(
        length_m=args.tip_length_nm * 1e-9,
        radius_m=args.tip_radius_nm * 1e-9,
        tapping_amplitude_m=tapping_amplitude_nm * 1e-9,
        tapping_frequency_hz=tip_frequency_hz,
        sample_thickness_m=args.sample_thickness_nm * 1e-9,
        substrate_material=args.substrate_material,
        use_three_layer_reflectivity=args.use_three_layer_reflectivity,
        reflectivity_backend=args.reflectivity_backend,
        g_factor=args.init_g_factor,
        g_phase=args.init_g_phase,
    )

    if args.model == "single-phonon":
        model = SemiInfiniteSnomModel(
            initial=SinglePhononParameters(
                w_to=args.init_w_to,
                w_lo=args.init_w_lo,
                gamma=args.init_gamma,
                eps_inf=args.init_eps_inf,
            ),
            tip=tip,
            reference_material=args.reference_material,
            bounds={
                "w_to": (args.wmin, args.wmax),
                "w_lo": (args.wmin, args.wmax + 200),
                "gamma": (1.0, 120.0),
                "eps_inf": (1.0, 30.0),
                "g_factor": (0.1, 1.5),
                "g_phase": (-0.5, 0.5),
            },
        ).to(device)
    else:
        init_centers = args.init_centers
        if args.center_init == "peaks":
            init_centers = peak_initial_centers(
                loaded["wavenumber"],
                loaded["amp"],
                args.n_oscillators,
                args.wmin,
                args.wmax,
                args.peak_min_spacing,
            )
        model = MultiLorentzSnomModel(
            n_oscillators=args.n_oscillators,
            centers=init_centers,
            strengths=args.init_strengths,
            gammas=args.init_gammas,
            eps_inf=args.init_eps_inf,
            tip=tip,
            reference_material=args.reference_material,
            bounds={
                "centers": (args.wmin, args.wmax),
                "strengths": (0.0, 30.0),
                "gammas": (10.0, 300.0),
                "eps_inf": (1.0, 30.0),
                "g_factor": (0.1, 1.5),
                "g_phase": (-0.5, 0.5),
            },
        ).to(device)

    amp_scale = (
        torch.nn.Parameter(torch.tensor(args.init_amp_scale, dtype=torch.float64, device=device))
        if args.use_amp_scale
        else torch.tensor(1.0, dtype=torch.float64, device=device)
    )
    amp_offset = (
        torch.nn.Parameter(torch.tensor(args.init_amp_offset, dtype=torch.float64, device=device))
        if args.use_amp_offset
        else None
    )
    optimizer_params = list(model.parameters())
    if isinstance(amp_scale, torch.nn.Parameter):
        optimizer_params.append(amp_scale)
    if amp_offset is not None:
        optimizer_params.append(amp_offset)

    with torch.no_grad():
        pred_phase, pred_amp = model(w)
        pred_phase, pred_amp = apply_experimental_reference(pred_phase, pred_amp, bg_amp, bg_phase)
        if args.auto_init_amp_correction and (args.use_amp_scale or args.use_amp_offset):
            scale0, offset0 = least_squares_amp_correction(
                pred_amp,
                target_amp,
                args.amp_scale_min,
                args.amp_scale_max,
                args.amp_offset_bound,
            )
            if isinstance(amp_scale, torch.nn.Parameter):
                amp_scale.copy_(torch.tensor(scale0, dtype=torch.float64, device=device))
            if amp_offset is not None:
                amp_offset.copy_(torch.tensor(offset0, dtype=torch.float64, device=device))
        pred_amp = apply_amp_correction(pred_amp, amp_scale, amp_offset)
        pred_amp = normalize_amp_torch(
            pred_amp,
            w,
            args.amp_normalization,
            (args.reference_band_min, args.reference_band_max),
        )
        initial_amp_mse = torch.mean((pred_amp - target_amp) ** 2).item()
        initial_phase_mse = torch.mean((pred_phase - target_phase) ** 2).item()
        initial_total_loss, initial_parts = combined_loss(
            pred_amp,
            target_amp,
            pred_phase,
            target_phase,
            args.phase_weight,
            args.shape_weight,
            args.amp_normalization,
        )

    optimizer = torch.optim.Adam(optimizer_params, lr=args.learning_rate)
    history = []
    best_loss = float(initial_total_loss.item())
    best_amp_mse = initial_amp_mse
    best_params = model.parameters_dict() | {
        "amp_scale": float(amp_scale.detach().cpu()),
        "amp_offset": float(amp_offset.detach().cpu()),
    }

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred_phase, pred_amp = model(w)
        pred_phase, pred_amp = apply_experimental_reference(pred_phase, pred_amp, bg_amp, bg_phase)
        pred_amp = apply_amp_correction(pred_amp, amp_scale, amp_offset)
        pred_amp = normalize_amp_torch(
            pred_amp,
            w,
            args.amp_normalization,
            (args.reference_band_min, args.reference_band_max),
        )
        loss, parts = combined_loss(
            pred_amp,
            target_amp,
            pred_phase,
            target_phase,
            args.phase_weight,
            args.shape_weight,
            args.amp_normalization,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
        optimizer.step()
        model.clamp_parameters()
        with torch.no_grad():
            if isinstance(amp_scale, torch.nn.Parameter):
                amp_scale.clamp_(args.amp_scale_min, args.amp_scale_max)
            if amp_offset is not None:
                amp_offset.clamp_(-args.amp_offset_bound, args.amp_offset_bound)

        row = {
            "epoch": epoch,
            "total_loss": float(loss.item()),
            "amp_relative_loss": float(parts["amp_relative_loss"].item()),
            "amp_shape_loss": float(parts["amp_shape_loss"].item()),
            "phase_circular_loss": float(parts["phase_circular_loss"].item()),
            "amp_mse": float(torch.mean((pred_amp - target_amp) ** 2).item()),
            "phase_mse": float(torch.mean((_wrap_phase(pred_phase - target_phase)) ** 2).item()),
            "amp_scale": float(amp_scale.detach().cpu()),
            "amp_offset": float(amp_offset.detach().cpu()) if amp_offset is not None else 0.0,
        }
        history.append(row)
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            best_amp_mse = row["amp_mse"]
            best_params = model.parameters_dict() | {
                "amp_scale": row["amp_scale"],
                "amp_offset": row["amp_offset"],
            }

    return {
        "dataset": str(args.dataset),
        "npz_path": loaded["npz_path"],
        "metadata_path": loaded["metadata_path"],
        "spectrum_index": args.spectrum_index,
        "model": args.model,
        "fit_mode": loaded["fit_mode"],
        "processing_mode": loaded["processing_mode"],
        "reference_material": args.reference_material,
        "amp_normalization": loaded["amp_normalization"],
        "reference_band": loaded["reference_band"],
        "amp_baseline": loaded["amp_baseline"],
        "als_lam": loaded["als_lam"],
        "als_p": loaded["als_p"],
        "als_niter": loaded["als_niter"],
        "use_amp_scale": args.use_amp_scale,
        "use_amp_offset": args.use_amp_offset,
        "bg_reference_source": loaded["bg_reference_source"],
        "source_file": loaded["source_file"],
        "tip_frequency_hz": tip_frequency_hz,
        "tapping_amplitude_nm": tapping_amplitude_nm,
        "tip_radius_nm": args.tip_radius_nm,
        "tip_length_nm": args.tip_length_nm,
        "sample_thickness_nm": args.sample_thickness_nm,
        "substrate_material": args.substrate_material,
        "use_three_layer_reflectivity": args.use_three_layer_reflectivity,
        "reflectivity_backend": args.reflectivity_backend,
        "amp_key": loaded["amp_key"],
        "phase_key": loaded["phase_key"],
        "n_wavenumbers": int(len(loaded["wavenumber"])),
        "wmin": args.wmin,
        "wmax": args.wmax,
        "step": args.step,
        "epochs": args.epochs,
        "initial_amp_mse": initial_amp_mse,
        "initial_phase_mse": initial_phase_mse,
        "initial_total_loss": float(initial_total_loss.item()),
        "initial_loss_parts": {k: float(v.item()) for k, v in initial_parts.items()},
        "best_total_loss": best_loss,
        "best_amp_mse": best_amp_mse,
        "best_params": best_params,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--spectrum-index", type=int, default=0)
    parser.add_argument("--model", choices=["multi-lorentz", "single-phonon"], default="multi-lorentz")
    parser.add_argument(
        "--fit-mode",
        default="auto",
        choices=["auto", "measured-bg", "raw-reference", "processed-direct"],
        help=(
            "auto uses measured-bg for bg-normalized data and raw-reference otherwise. "
            "measured-bg is the main biological-material mode."
        ),
    )
    parser.add_argument("--reference-material", default="si", choices=["si", "au", "air"])
    parser.add_argument("--wmin", type=float, default=700.0)
    parser.add_argument("--wmax", type=float, default=1600.0)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase-weight", type=float, default=0.1)
    parser.add_argument("--shape-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tip-radius-nm", type=float, default=33.0)
    parser.add_argument("--tip-length-nm", type=float, default=300.0)
    parser.add_argument("--tapping-amplitude-nm", type=float, default=80.0)
    parser.add_argument("--tip-frequency-hz", type=float, default=260000.0)
    parser.add_argument("--sample-thickness-nm", type=float, default=1000.0)
    parser.add_argument("--substrate-material", default="si", choices=["si", "au", "air"])
    parser.add_argument("--use-three-layer-reflectivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflectivity-backend", default="fresnel", choices=["fresnel", "berreman"])
    parser.add_argument("--init-g-factor", type=float, default=0.7)
    parser.add_argument("--init-g-phase", type=float, default=0.06)
    parser.add_argument("--init-amp-scale", type=float, default=1.0)
    parser.add_argument("--init-amp-offset", type=float, default=0.0)
    parser.add_argument("--amp-scale-min", type=float, default=0.001)
    parser.add_argument("--amp-scale-max", type=float, default=100.0)
    parser.add_argument("--use-amp-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-offset-bound", type=float, default=0.05)
    parser.add_argument("--use-amp-offset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-init-amp-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-normalization",
        choices=["none", "median", "zscore", "snv", "reference-band"],
        default="median",
    )
    parser.add_argument("--reference-band-min", type=float, default=1500.0)
    parser.add_argument("--reference-band-max", type=float, default=1600.0)
    parser.add_argument(
        "--amp-baseline",
        choices=["none", "als-subtract", "als-divide"],
        default="none",
        help="Optional measured-amplitude baseline correction before normalization.",
    )
    parser.add_argument("--als-lam", type=float, default=1e5)
    parser.add_argument("--als-p", type=float, default=0.01)
    parser.add_argument("--als-niter", type=int, default=10)
    parser.add_argument("--n-oscillators", type=int, default=3)
    parser.add_argument("--center-init", choices=["fixed", "peaks"], default="fixed")
    parser.add_argument("--peak-min-spacing", type=float, default=80.0)
    parser.add_argument("--init-centers", type=float, nargs="+", default=[900.0, 1200.0, 1450.0])
    parser.add_argument("--init-strengths", type=float, nargs="+", default=[1.0, 1.0, 1.0])
    parser.add_argument("--init-gammas", type=float, nargs="+", default=[80.0, 80.0, 80.0])
    parser.add_argument("--init-w-to", type=float, default=900.0)
    parser.add_argument("--init-w-lo", type=float, default=1000.0)
    parser.add_argument("--init-gamma", type=float, default=10.0)
    parser.add_argument("--init-eps-inf", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
