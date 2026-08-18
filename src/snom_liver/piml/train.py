"""Train a small PIML inverse model on synthetic SNOM spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from snom_liver.piml.common import (
    MultiLorentzBounds,
    SpectraToParamsNet,
    parameter_lows_highs,
    params_to_vector,
    sample_parameters,
    synthesize_spectrum,
)
from snom_liver.physics import TipParameters


def build_dataset(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    wavenumber = np.arange(args.wmin, args.wmax + 1e-9, args.step, dtype=np.float64)
    bounds = MultiLorentzBounds(
        center_min=args.center_min,
        center_max=args.center_max,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        eps_inf_min=args.eps_inf_min,
        eps_inf_max=args.eps_inf_max,
    )
    tip = TipParameters(
        length_m=args.tip_length_nm * 1e-9,
        radius_m=args.tip_radius_nm * 1e-9,
        tapping_amplitude_m=args.tapping_amplitude_nm * 1e-9,
        tapping_frequency_hz=args.tip_frequency_hz,
        sample_thickness_m=args.sample_thickness_nm * 1e-9,
        substrate_material=args.substrate_material,
        use_three_layer_reflectivity=args.use_three_layer_reflectivity,
        reflectivity_backend=args.reflectivity_backend,
        g_factor=args.g_factor,
        g_phase=args.g_phase,
    )
    spectra = []
    targets = []
    for _ in range(args.n_samples):
        params = sample_parameters(rng, bounds, args.n_oscillators)
        spectra.append(
            synthesize_spectrum(
                wavenumber,
                params,
                args.n_oscillators,
                args.reference_material,
                tip,
                args.amp_normalization,
                (args.reference_band_min, args.reference_band_max),
                args.noise_std,
                torch.device(args.device),
            )
        )
        targets.append(params_to_vector(params, args.n_oscillators))
    lows, highs = parameter_lows_highs(bounds, args.n_oscillators)
    return wavenumber, np.asarray(spectra, dtype=np.float32), np.asarray(targets, dtype=np.float32), lows, highs, bounds


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wavenumber, spectra, targets, lows, highs, bounds = build_dataset(args)
    n = len(spectra)
    order = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_fraction)))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train_ds = TensorDataset(
        torch.tensor(spectra[train_idx], dtype=torch.float32),
        torch.tensor(targets[train_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(spectra[val_idx], dtype=torch.float32),
        torch.tensor(targets[val_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SpectraToParamsNet(len(wavenumber), targets.shape[1]).to(args.device)
    lows_t = torch.tensor(lows, dtype=torch.float32, device=args.device)
    highs_t = torch.tensor(highs, dtype=torch.float32, device=args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=12)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for spectra_b, targets_b in train_loader:
            spectra_b = spectra_b.to(args.device)
            targets_b = targets_b.to(args.device)
            optimizer.zero_grad()
            pred = model(spectra_b)
            true = (targets_b - lows_t) / (highs_t - lows_t + 1e-12)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for spectra_b, targets_b in val_loader:
                spectra_b = spectra_b.to(args.device)
                targets_b = targets_b.to(args.device)
                pred = model(spectra_b)
                true = (targets_b - lows_t) / (highs_t - lows_t + 1e-12)
                val_losses.append(float(criterion(pred, true).item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": float(optimizer.param_groups[0]["lr"]) }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

    checkpoint = {
        "model_class": "SpectraToParamsNet",
        "n_oscillators": args.n_oscillators,
        "n_params": targets.shape[1],
        "wavenumber": wavenumber.tolist(),
        "amp_normalization": args.amp_normalization,
        "reference_band": [args.reference_band_min, args.reference_band_max],
        "reference_material": args.reference_material,
        "tip": {
            "length_nm": args.tip_length_nm,
            "radius_nm": args.tip_radius_nm,
            "tapping_amplitude_nm": args.tapping_amplitude_nm,
            "tapping_frequency_hz": args.tip_frequency_hz,
            "sample_thickness_nm": args.sample_thickness_nm,
            "substrate_material": args.substrate_material,
            "use_three_layer_reflectivity": args.use_three_layer_reflectivity,
            "reflectivity_backend": args.reflectivity_backend,
            "g_factor": args.g_factor,
            "g_phase": args.g_phase,
        },
        "bounds": {
            "center_min": bounds.center_min,
            "center_max": bounds.center_max,
            "strength_min": bounds.strength_min,
            "strength_max": bounds.strength_max,
            "gamma_min": bounds.gamma_min,
            "gamma_max": bounds.gamma_max,
            "eps_inf_min": bounds.eps_inf_min,
            "eps_inf_max": bounds.eps_inf_max,
        },
        "param_lows": lows.tolist(),
        "param_highs": highs.tolist(),
        "history": history,
        "best": {
            "epoch": int(best_state["epoch"]) if best_state else None,
            "val_loss": float(best_state["val_loss"]) if best_state else None,
        },
        "best_val_loss": best_val,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "piml_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(best_state, output_dir / "piml_model.pt")
    return checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-oscillators", type=int, default=3)
    p.add_argument("--wmin", type=float, default=700.0)
    p.add_argument("--wmax", type=float, default=1600.0)
    p.add_argument("--step", type=float, default=30.0)
    p.add_argument("--reference-material", default="si", choices=["si", "au", "air"])
    p.add_argument("--amp-normalization", default="reference-band", choices=["none", "median", "zscore", "snv", "reference-band"])
    p.add_argument("--reference-band-min", type=float, default=1500.0)
    p.add_argument("--reference-band-max", type=float, default=1600.0)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--center-min", type=float, default=700.0)
    p.add_argument("--center-max", type=float, default=1600.0)
    p.add_argument("--strength-min", type=float, default=0.0)
    p.add_argument("--strength-max", type=float, default=4.0)
    p.add_argument("--gamma-min", type=float, default=20.0)
    p.add_argument("--gamma-max", type=float, default=220.0)
    p.add_argument("--eps-inf-min", type=float, default=1.5)
    p.add_argument("--eps-inf-max", type=float, default=12.0)
    p.add_argument("--tip-radius-nm", type=float, default=33.0)
    p.add_argument("--tip-length-nm", type=float, default=300.0)
    p.add_argument("--tapping-amplitude-nm", type=float, default=80.0)
    p.add_argument("--tip-frequency-hz", type=float, default=260000.0)
    p.add_argument("--sample-thickness-nm", type=float, default=1000.0)
    p.add_argument("--substrate-material", default="si", choices=["si", "au", "air"])
    p.add_argument("--use-three-layer-reflectivity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reflectivity-backend", default="fresnel", choices=["fresnel", "berreman"])
    p.add_argument("--g-factor", type=float, default=0.7)
    p.add_argument("--g-phase", type=float, default=0.06)
    return p.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({"best_val_loss": summary["best_val_loss"], "output": summary["best"]["epoch"] if summary["best"] else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
