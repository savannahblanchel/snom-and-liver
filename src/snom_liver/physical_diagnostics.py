"""Diagnostics for fitted physical parameters and residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from snom_liver.physical_batch_fit import make_model
from snom_liver.physical_fit import (
    apply_amp_correction,
    apply_experimental_reference,
    circular_phase_loss,
    load_spectrum,
    normalize_amp_torch,
)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def set_material_params(model, params: dict) -> None:
    with torch.no_grad():
        if hasattr(model, "centers") and "centers" in params:
            model.centers.copy_(torch.tensor(params["centers"], dtype=torch.float64, device=model.centers.device))
            model.strengths.copy_(
                torch.tensor(params["strengths"], dtype=torch.float64, device=model.strengths.device)
            )
            model.gammas.copy_(torch.tensor(params["gammas"], dtype=torch.float64, device=model.gammas.device))
            model.eps_inf.copy_(torch.tensor(params["eps_inf"], dtype=torch.float64, device=model.eps_inf.device))
        elif hasattr(model, "w_to") and "w_to" in params:
            model.w_to.copy_(torch.tensor(params["w_to"], dtype=torch.float64, device=model.w_to.device))
            model.w_lo.copy_(torch.tensor(params["w_lo"], dtype=torch.float64, device=model.w_lo.device))
            model.gamma.copy_(torch.tensor(params["gamma"], dtype=torch.float64, device=model.gamma.device))
            model.eps_inf.copy_(torch.tensor(params["eps_inf"], dtype=torch.float64, device=model.eps_inf.device))


def params_to_row(prefix: str, params: dict) -> dict:
    row = {}
    if "centers" in params:
        for i, value in enumerate(params["centers"], start=1):
            row[f"{prefix}center_{i}"] = value
        for i, value in enumerate(params["strengths"], start=1):
            row[f"{prefix}strength_{i}"] = value
        for i, value in enumerate(params["gammas"], start=1):
            row[f"{prefix}gamma_{i}"] = value
        row[f"{prefix}eps_inf"] = params.get("eps_inf", np.nan)
    else:
        for key in ("w_to", "w_lo", "gamma", "eps_inf"):
            row[f"{prefix}{key}"] = params.get(key, np.nan)
    return row


def predict(args: argparse.Namespace, batch: dict, spectrum: dict, device: torch.device):
    shared = batch["shared_params"]
    shared_g_factor = torch.nn.Parameter(torch.tensor(shared["g_factor"], dtype=torch.float64, device=device))
    shared_g_phase = torch.nn.Parameter(torch.tensor(shared["g_phase"], dtype=torch.float64, device=device))

    loaded = load_spectrum(
        Path(batch["dataset"]),
        int(spectrum["spectrum_index"]),
        float(batch["wmin"]),
        float(batch["wmax"]),
        int(batch["step"]),
        batch.get("fit_mode", "auto"),
        batch.get("amp_normalization", args.amp_normalization),
        tuple(batch.get("reference_band", [args.reference_band_min, args.reference_band_max])),
        batch.get("amp_baseline", args.amp_baseline),
        float(batch.get("als_lam", args.als_lam)),
        float(batch.get("als_p", args.als_p)),
        int(batch.get("als_niter", args.als_niter)),
    )
    model_args = argparse.Namespace(
        model=batch.get("model", args.model),
        tip_frequency_hz=args.tip_frequency_hz,
        tapping_amplitude_nm=args.tapping_amplitude_nm,
        tip_length_nm=shared.get("tip_length_nm", args.tip_length_nm),
        tip_radius_nm=shared.get("tip_radius_nm", args.tip_radius_nm),
        sample_thickness_nm=shared.get("sample_thickness_nm", args.sample_thickness_nm),
        substrate_material=shared.get("substrate_material", args.substrate_material),
        use_three_layer_reflectivity=shared.get(
            "use_three_layer_reflectivity", args.use_three_layer_reflectivity
        ),
        reflectivity_backend=shared.get("reflectivity_backend", args.reflectivity_backend),
        init_g_factor=shared["g_factor"],
        init_g_phase=shared["g_phase"],
        reference_material=args.reference_material,
        wmin=float(batch["wmin"]),
        wmax=float(batch["wmax"]),
        center_init=batch.get("center_init", "fixed"),
        peak_min_spacing=args.peak_min_spacing,
        center_jitter=0.0,
        seed=0,
        n_oscillators=args.n_oscillators,
        init_centers=args.init_centers,
        init_strengths=args.init_strengths,
        init_gammas=args.init_gammas,
        init_eps_inf=args.init_eps_inf,
        init_w_to=args.init_w_to,
        init_w_lo=args.init_w_lo,
        init_gamma=args.init_gamma,
    )
    model = make_model(model_args, loaded, shared_g_factor, shared_g_phase).to(device)
    set_material_params(model, spectrum["material_params"])

    w = torch.tensor(loaded["wavenumber"], dtype=torch.float64, device=device)
    target_amp = torch.tensor(loaded["amp"], dtype=torch.float64, device=device)
    target_phase = torch.tensor(loaded["phase"], dtype=torch.float64, device=device)
    bg_amp = (
        torch.tensor(loaded["bg_amp"], dtype=torch.float64, device=device) if loaded["bg_amp"] is not None else None
    )
    bg_phase = (
        torch.tensor(loaded["bg_phase"], dtype=torch.float64, device=device) if loaded["bg_phase"] is not None else None
    )

    amp_scale = torch.tensor(shared.get("amp_scale", 1.0), dtype=torch.float64, device=device)
    amp_offset = (
        torch.tensor(shared.get("amp_offset", 0.0), dtype=torch.float64, device=device)
        if batch.get("use_amp_offset", False)
        else None
    )

    with torch.no_grad():
        pred_phase, pred_amp = model(w)
        pred_phase, pred_amp = apply_experimental_reference(pred_phase, pred_amp, bg_amp, bg_phase)
        pred_amp = apply_amp_correction(pred_amp, amp_scale, amp_offset)
        pred_amp = normalize_amp_torch(
            pred_amp,
            w,
            batch.get("amp_normalization", args.amp_normalization),
            tuple(batch.get("reference_band", [args.reference_band_min, args.reference_band_max])),
        )

    return loaded, {
        "wavenumber": w.detach().cpu().numpy(),
        "target_amp": target_amp.detach().cpu().numpy(),
        "pred_amp": pred_amp.detach().cpu().numpy(),
        "target_phase": target_phase.detach().cpu().numpy(),
        "pred_phase": pred_phase.detach().cpu().numpy(),
        "phase_circular_loss": float(circular_phase_loss(pred_phase, target_phase).detach().cpu()),
    }


def residual_metrics(pred: dict, smooth_window: int) -> dict:
    amp_residual = pred["target_amp"] - pred["pred_amp"]
    phase_residual = np.angle(np.exp(1j * (pred["target_phase"] - pred["pred_phase"])))
    smooth = moving_average(amp_residual, smooth_window)
    residual_var = float(np.var(amp_residual))
    smooth_fraction = float(np.var(smooth) / (residual_var + 1e-12))
    if np.std(pred["target_amp"]) > 1e-12 and np.std(pred["pred_amp"]) > 1e-12:
        amp_corr = float(np.corrcoef(pred["target_amp"], pred["pred_amp"])[0, 1])
    else:
        amp_corr = np.nan
    return {
        "amp_mse": float(np.mean(amp_residual**2)),
        "amp_mae": float(np.mean(np.abs(amp_residual))),
        "amp_corr": amp_corr,
        "amp_residual_mean": float(np.mean(amp_residual)),
        "amp_residual_std": float(np.std(amp_residual)),
        "amp_residual_smooth_fraction": smooth_fraction,
        "phase_circular_loss": pred["phase_circular_loss"],
        "phase_residual_std": float(np.std(phase_residual)),
    }


def plot_residual(output_dir: Path, row: dict, pred: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(pred["wavenumber"], pred["target_amp"], label="target amp", linewidth=1.6)
    axes[0].plot(pred["wavenumber"], pred["pred_amp"], label="pred amp", linewidth=1.4)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[1].plot(pred["wavenumber"], pred["target_amp"] - pred["pred_amp"], label="amp residual")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Wavenumber (cm^-1)")
    axes[1].set_ylabel("Residual")
    axes[1].legend()
    fig.suptitle(f"spectrum {row['spectrum_index']} | {row.get('class_label', '')} | {row.get('sample_id', '')}")
    fig.tight_layout()
    fig.savefig(output_dir / f"residual_spectrum_{row['spectrum_index']}.png", dpi=180)
    plt.close(fig)


def stability_summary(params_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in params_df.columns
        if any(token in col for token in ("center_", "strength_", "gamma_", "eps_inf"))
    ]
    rows = []
    for group, sub in params_df.groupby(group_col, dropna=False):
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            rows.append(
                {
                    group_col: group,
                    "parameter": col,
                    "n": int(values.shape[0]),
                    "mean": mean,
                    "std": std,
                    "cv_abs": abs(std / mean) if abs(mean) > 1e-12 else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = json.loads(Path(args.batch_result).read_text(encoding="utf-8"))
    device = torch.device(args.device)

    param_rows = []
    residual_rows = []
    for plot_i, spectrum in enumerate(batch["spectra"]):
        loaded, pred = predict(args, batch, spectrum, device)
        row = {
            "spectrum_index": int(spectrum["spectrum_index"]),
            "sample_id": loaded.get("sample_id", ""),
            "point_id": loaded.get("point_id", ""),
            "class_label": loaded.get("class_label", ""),
            "specimen_name": loaded.get("specimen_name", ""),
            "marker": loaded.get("marker", ""),
            "source_file": loaded.get("source_file", ""),
            "bg_reference_source": loaded.get("bg_reference_source", ""),
            "tip_frequency_hz": loaded.get("tip_frequency_hz"),
            "tapping_amplitude_nm": loaded.get("tapping_amplitude_nm"),
        }
        row.update(params_to_row("", spectrum["material_params"]))
        param_rows.append(row)

        residual = row.copy()
        residual.update(residual_metrics(pred, args.smooth_window))
        residual_rows.append(residual)
        if plot_i < args.max_plots:
            plot_residual(output_dir, row, pred)

    params_df = pd.DataFrame(param_rows)
    residual_df = pd.DataFrame(residual_rows)
    stability_df = stability_summary(params_df, args.group_col) if args.group_col in params_df.columns else pd.DataFrame()

    params_path = output_dir / "fitted_parameters.csv"
    residual_path = output_dir / "residual_summary.csv"
    stability_path = output_dir / "parameter_stability.csv"
    params_df.to_csv(params_path, index=False, encoding="utf-8-sig")
    residual_df.to_csv(residual_path, index=False, encoding="utf-8-sig")
    stability_df.to_csv(stability_path, index=False, encoding="utf-8-sig")

    summary = {
        "batch_result": str(args.batch_result),
        "output_dir": str(output_dir),
        "n_spectra": int(len(params_df)),
        "mean_amp_mse": float(residual_df["amp_mse"].mean()) if not residual_df.empty else np.nan,
        "mean_amp_corr": float(residual_df["amp_corr"].mean()) if not residual_df.empty else np.nan,
        "mean_smooth_fraction": float(residual_df["amp_residual_smooth_fraction"].mean())
        if not residual_df.empty
        else np.nan,
        "parameters_csv": str(params_path),
        "residual_csv": str(residual_path),
        "stability_csv": str(stability_path),
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--group-col", default="class_label")
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--max-plots", type=int, default=6)
    parser.add_argument("--model", choices=["multi-lorentz", "single-phonon"], default="multi-lorentz")
    parser.add_argument("--reference-material", default="si", choices=["si", "au", "air"])
    parser.add_argument("--amp-normalization", default="reference-band")
    parser.add_argument("--reference-band-min", type=float, default=1500.0)
    parser.add_argument("--reference-band-max", type=float, default=1600.0)
    parser.add_argument("--amp-baseline", choices=["none", "als-subtract", "als-divide"], default="none")
    parser.add_argument("--als-lam", type=float, default=1e5)
    parser.add_argument("--als-p", type=float, default=0.01)
    parser.add_argument("--als-niter", type=int, default=10)
    parser.add_argument("--peak-min-spacing", type=float, default=80.0)
    parser.add_argument("--tip-radius-nm", type=float, default=33.0)
    parser.add_argument("--tip-length-nm", type=float, default=300.0)
    parser.add_argument("--tapping-amplitude-nm", type=float, default=80.0)
    parser.add_argument("--tip-frequency-hz", type=float, default=260000.0)
    parser.add_argument("--sample-thickness-nm", type=float, default=1000.0)
    parser.add_argument("--substrate-material", default="si", choices=["si", "au", "air"])
    parser.add_argument("--use-three-layer-reflectivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflectivity-backend", default="fresnel", choices=["fresnel", "berreman"])
    parser.add_argument("--n-oscillators", type=int, default=3)
    parser.add_argument("--init-centers", type=float, nargs="+", default=[900.0, 1200.0, 1450.0])
    parser.add_argument("--init-strengths", type=float, nargs="+", default=[1.0, 1.0, 1.0])
    parser.add_argument("--init-gammas", type=float, nargs="+", default=[80.0, 80.0, 80.0])
    parser.add_argument("--init-eps-inf", type=float, default=5.0)
    parser.add_argument("--init-w-to", type=float, default=900.0)
    parser.add_argument("--init-w-lo", type=float, default=1000.0)
    parser.add_argument("--init-gamma", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
