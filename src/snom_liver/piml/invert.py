"""Use a trained PIML model to initialize experimental SNOM parameter inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snom_liver.physical_fit import (
    apply_amp_correction,
    apply_experimental_reference,
    combined_loss,
    load_spectrum,
    normalize_amp_np,
    normalize_amp_torch,
)
from snom_liver.physics import MultiLorentzSnomModel, TipParameters
from snom_liver.piml.common import SpectraToParamsNet, denormalize_param_tensor, vector_to_params


def flatten_params(prefix: str, params: dict[str, list[float] | float] | None) -> dict[str, float]:
    if not params:
        return {}
    row: dict[str, float] = {}
    for i, value in enumerate(params.get("centers", []), start=1):
        row[f"{prefix}_center_{i}"] = float(value)
    for i, value in enumerate(params.get("strengths", []), start=1):
        row[f"{prefix}_strength_{i}"] = float(value)
    for i, value in enumerate(params.get("gammas", []), start=1):
        row[f"{prefix}_gamma_{i}"] = float(value)
    if "eps_inf" in params:
        row[f"{prefix}_eps_inf"] = float(params["eps_inf"])
    if "g_factor" in params:
        row[f"{prefix}_g_factor"] = float(params["g_factor"])
    if "g_phase" in params:
        row[f"{prefix}_g_phase"] = float(params["g_phase"])
    return row


def reconstruct_input(loaded: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    amp = np.asarray(loaded["amp"], dtype=np.float64)
    phase = np.asarray(loaded["phase"], dtype=np.float64)
    if args.use_bg_approximation and loaded["bg_amp"] is not None and loaded["bg_phase"] is not None:
        amp = amp * np.asarray(loaded["bg_amp"], dtype=np.float64)
        phase = np.angle(np.exp(1j * (phase + np.asarray(loaded["bg_phase"], dtype=np.float64))))
    amp = normalize_amp_np(amp, loaded["wavenumber"], args.amp_normalization, tuple(loaded["reference_band"]))
    phase = phase - np.nanmedian(phase)
    return amp, phase


def build_model(params: dict[str, list[float] | float], args: argparse.Namespace, loaded: dict) -> MultiLorentzSnomModel:
    tip = TipParameters(
        length_m=args.tip_length_nm * 1e-9,
        radius_m=args.tip_radius_nm * 1e-9,
        tapping_amplitude_m=(loaded["tapping_amplitude_nm"] or args.tapping_amplitude_nm) * 1e-9,
        tapping_frequency_hz=loaded["tip_frequency_hz"] or args.tip_frequency_hz,
        sample_thickness_m=args.sample_thickness_nm * 1e-9,
        substrate_material=args.substrate_material,
        use_three_layer_reflectivity=args.use_three_layer_reflectivity,
        reflectivity_backend=args.reflectivity_backend,
        g_factor=args.init_g_factor,
        g_phase=args.init_g_phase,
    )
    return MultiLorentzSnomModel(
        n_oscillators=args.n_oscillators,
        centers=params["centers"],
        strengths=params["strengths"],
        gammas=params["gammas"],
        eps_inf=params["eps_inf"],
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
    )


def run(args: argparse.Namespace) -> dict:
    ckpt = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    lows = torch.tensor(ckpt["param_lows"], dtype=torch.float32, device=args.device)
    highs = torch.tensor(ckpt["param_highs"], dtype=torch.float32, device=args.device)
    net = SpectraToParamsNet(len(ckpt["wavenumber"]), len(ckpt["param_lows"])).to(args.device)
    state = torch.load(Path(args.model_state), map_location=args.device)
    net.load_state_dict(state["model_state_dict"])
    net.eval()

    rows = []
    for idx in args.indices:
        loaded = load_spectrum(
            Path(args.dataset),
            idx,
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
        amp_np, phase_np = reconstruct_input(loaded, args)
        spectrum = torch.tensor(np.stack([amp_np, phase_np], axis=0)[None, ...], dtype=torch.float32, device=args.device)
        with torch.no_grad():
            pred_norm = net(spectrum)[0]
        pred_vec = denormalize_param_tensor(pred_norm, lows, highs)
        pred_params = vector_to_params(pred_vec, args.n_oscillators)

        model = build_model(pred_params, args, loaded).to(args.device)
        if not args.fit_g:
            model.g_factor.requires_grad_(False)
            model.g_phase.requires_grad_(False)
        w = torch.tensor(loaded["wavenumber"], dtype=torch.float64, device=args.device)
        target_amp = torch.tensor(loaded["amp"], dtype=torch.float64, device=args.device)
        target_phase = torch.tensor(loaded["phase"], dtype=torch.float64, device=args.device)
        bg_amp = torch.tensor(loaded["bg_amp"], dtype=torch.float64, device=args.device) if loaded["bg_amp"] is not None else None
        bg_phase = torch.tensor(loaded["bg_phase"], dtype=torch.float64, device=args.device) if loaded["bg_phase"] is not None else None

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        best_loss = float("inf")
        best_params = None
        for _ in range(args.epochs):
            optimizer.zero_grad()
            pred_phase, pred_amp = model(w)
            pred_phase, pred_amp = apply_experimental_reference(pred_phase, pred_amp, bg_amp, bg_phase)
            pred_amp = normalize_amp_torch(
                pred_amp,
                w,
                args.amp_normalization,
                (args.reference_band_min, args.reference_band_max),
            )
            loss, _ = combined_loss(
                pred_amp,
                target_amp,
                pred_phase,
                target_phase,
                args.phase_weight,
                args.shape_weight,
                args.amp_normalization,
            )
            loss.backward()
            optimizer.step()
            model.clamp_parameters()
            current = float(loss.item())
            if current < best_loss:
                best_loss = current
                best_params = model.parameters_dict()

        row = {
            "spectrum_index": idx,
            "sample_id": loaded.get("sample_id", ""),
            "point_id": loaded.get("point_id", ""),
            "class_label": loaded.get("class_label", ""),
            "specimen_name": loaded.get("specimen_name", ""),
            "marker": loaded.get("marker", ""),
            "source_file": loaded.get("source_file", ""),
            "best_loss": best_loss,
        }
        row.update(flatten_params("init", pred_params))
        row.update(flatten_params("fit", best_params))
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "piml_inversion.csv", index=False, encoding="utf-8-sig")
    summary = {"n_spectra": len(rows), "output_dir": str(output_dir), "csv": str(output_dir / "piml_inversion.csv")}
    (output_dir / "piml_inversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-state", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fit-mode", default="measured-bg", choices=["auto", "measured-bg", "raw-reference", "processed-direct"])
    p.add_argument("--wmin", type=float, default=700.0)
    p.add_argument("--wmax", type=float, default=1600.0)
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--reference-material", default="si", choices=["si", "au", "air"])
    p.add_argument("--amp-normalization", default="reference-band", choices=["none", "median", "zscore", "snv", "reference-band"])
    p.add_argument("--reference-band-min", type=float, default=1500.0)
    p.add_argument("--reference-band-max", type=float, default=1600.0)
    p.add_argument("--amp-baseline", default="none", choices=["none", "als-subtract", "als-divide"])
    p.add_argument("--als-lam", type=float, default=1e5)
    p.add_argument("--als-p", type=float, default=0.01)
    p.add_argument("--als-niter", type=int, default=10)
    p.add_argument("--n-oscillators", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--phase-weight", type=float, default=0.1)
    p.add_argument("--shape-weight", type=float, default=0.25)
    p.add_argument("--fit-g", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--init-g-factor", type=float, default=0.7)
    p.add_argument("--init-g-phase", type=float, default=0.06)
    p.add_argument("--tip-radius-nm", type=float, default=33.0)
    p.add_argument("--tip-length-nm", type=float, default=300.0)
    p.add_argument("--tapping-amplitude-nm", type=float, default=80.0)
    p.add_argument("--tip-frequency-hz", type=float, default=260000.0)
    p.add_argument("--sample-thickness-nm", type=float, default=1000.0)
    p.add_argument("--substrate-material", default="si", choices=["si", "au", "air"])
    p.add_argument("--use-three-layer-reflectivity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reflectivity-backend", default="fresnel", choices=["fresnel", "berreman"])
    p.add_argument("--use-bg-approximation", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
