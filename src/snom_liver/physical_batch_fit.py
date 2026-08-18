"""Batch-level physical fitting with shared instrument parameters.

Each spectrum owns its material parameters. The experimental batch shares
`g_factor`, `g_phase`, `amp_scale`, and `amp_offset`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from snom_liver.physical_fit import (
    apply_amp_correction,
    apply_experimental_reference,
    combined_loss,
    least_squares_amp_correction,
    load_spectrum,
    normalize_amp_torch,
    peak_initial_centers,
)
from snom_liver.physics import MultiLorentzSnomModel, SinglePhononParameters, SemiInfiniteSnomModel, TipParameters


def parse_indices(value: str | None, max_spectra: int | None) -> list[int]:
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    n = max_spectra or 4
    return list(range(n))


def initial_centers(args: argparse.Namespace, loaded: dict, spectrum_index: int) -> list[float]:
    if args.center_init == "peaks":
        centers = peak_initial_centers(
            loaded["wavenumber"],
            loaded["amp"],
            args.n_oscillators,
            args.wmin,
            args.wmax,
            args.peak_min_spacing,
        )
    else:
        centers = list(args.init_centers)
    start_id = getattr(args, "start_id", 0)
    if start_id > 0 and args.center_jitter > 0:
        rng = np.random.default_rng(args.seed + 997 * start_id + int(spectrum_index))
        jitter = rng.normal(0.0, args.center_jitter, size=len(centers))
        centers = np.clip(np.asarray(centers, dtype=np.float64) + jitter, args.wmin, args.wmax)
        centers = sorted(float(value) for value in centers)
    return centers


def make_model(args: argparse.Namespace, loaded: dict, shared_g_factor, shared_g_phase):
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
        )
    else:
        centers = initial_centers(args, loaded, loaded.get("spectrum_index", 0))
        model = MultiLorentzSnomModel(
            n_oscillators=args.n_oscillators,
            centers=centers,
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
        )
    model.g_factor = shared_g_factor
    model.g_phase = shared_g_phase
    return model


def model_material_parameters(model):
    return [param for name, param in model.named_parameters() if name not in {"g_factor", "g_phase"}]


def prepare_records(args: argparse.Namespace, device: torch.device, shared_g_factor, shared_g_phase):
    records = []
    for idx in parse_indices(args.indices, args.max_spectra):
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
        loaded["spectrum_index"] = idx
        model = make_model(args, loaded, shared_g_factor, shared_g_phase).to(device)
        record = {
            "index": idx,
            "loaded": loaded,
            "model": model,
            "w": torch.tensor(loaded["wavenumber"], dtype=torch.float64, device=device),
            "target_amp": torch.tensor(loaded["amp"], dtype=torch.float64, device=device),
            "target_phase": torch.tensor(loaded["phase"], dtype=torch.float64, device=device),
            "bg_amp": torch.tensor(loaded["bg_amp"], dtype=torch.float64, device=device)
            if loaded["bg_amp"] is not None
            else None,
            "bg_phase": torch.tensor(loaded["bg_phase"], dtype=torch.float64, device=device)
            if loaded["bg_phase"] is not None
            else None,
        }
        records.append(record)
    return records


def predict_record(record, amp_scale, amp_offset, args):
    pred_phase, pred_amp = record["model"](record["w"])
    pred_phase, pred_amp = apply_experimental_reference(
        pred_phase, pred_amp, record["bg_amp"], record["bg_phase"]
    )
    pred_amp = apply_amp_correction(pred_amp, amp_scale, amp_offset)
    pred_amp = normalize_amp_torch(
        pred_amp,
        record["w"],
        args.amp_normalization,
        (args.reference_band_min, args.reference_band_max),
    )
    return pred_phase, pred_amp


def _run_once(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    shared_g_factor = torch.nn.Parameter(torch.tensor(args.init_g_factor, dtype=torch.float64, device=device))
    shared_g_phase = torch.nn.Parameter(torch.tensor(args.init_g_phase, dtype=torch.float64, device=device))
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

    records = prepare_records(args, device, shared_g_factor, shared_g_phase)

    with torch.no_grad():
        pred_all = []
        target_all = []
        for record in records:
            pred_phase, pred_amp = record["model"](record["w"])
            _, pred_amp = apply_experimental_reference(
                pred_phase, pred_amp, record["bg_amp"], record["bg_phase"]
            )
            pred_all.append(
                normalize_amp_torch(
                    pred_amp,
                    record["w"],
                    args.amp_normalization,
                    (args.reference_band_min, args.reference_band_max),
                )
            )
            target_all.append(record["target_amp"])
        if args.auto_init_amp_correction and (args.use_amp_scale or args.use_amp_offset):
            scale0, offset0 = least_squares_amp_correction(
                torch.cat(pred_all),
                torch.cat(target_all),
                args.amp_scale_min,
                args.amp_scale_max,
                args.amp_offset_bound,
            )
            if isinstance(amp_scale, torch.nn.Parameter):
                amp_scale.copy_(torch.tensor(scale0, dtype=torch.float64, device=device))
            if amp_offset is not None:
                amp_offset.copy_(torch.tensor(offset0, dtype=torch.float64, device=device))

    params = [shared_g_factor, shared_g_phase]
    if isinstance(amp_scale, torch.nn.Parameter):
        params.append(amp_scale)
    if amp_offset is not None:
        params.append(amp_offset)
    for record in records:
        params.extend(model_material_parameters(record["model"]))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        losses = []
        parts_accum = {"amp_relative_loss": 0.0, "amp_shape_loss": 0.0, "phase_circular_loss": 0.0}
        amp_mse = 0.0
        phase_mse = 0.0
        for record in records:
            pred_phase, pred_amp = predict_record(record, amp_scale, amp_offset, args)
            loss, parts = combined_loss(
                pred_amp,
                record["target_amp"],
                pred_phase,
                record["target_phase"],
                args.phase_weight,
                args.shape_weight,
                args.amp_normalization,
            )
            losses.append(loss)
            for key in parts_accum:
                parts_accum[key] += float(parts[key].detach().cpu())
            amp_mse += float(torch.mean((pred_amp - record["target_amp"]) ** 2).detach().cpu())
            phase_mse += float(torch.mean((pred_phase - record["target_phase"]) ** 2).detach().cpu())

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            shared_g_factor.clamp_(0.1, 1.5)
            shared_g_phase.clamp_(-0.5, 0.5)
            if isinstance(amp_scale, torch.nn.Parameter):
                amp_scale.clamp_(args.amp_scale_min, args.amp_scale_max)
            if amp_offset is not None:
                amp_offset.clamp_(-args.amp_offset_bound, args.amp_offset_bound)
            for record in records:
                record["model"].clamp_parameters()

        n = len(records)
        row = {
            "epoch": epoch,
            "total_loss": float(total_loss.detach().cpu()),
            "amp_relative_loss": parts_accum["amp_relative_loss"] / n,
            "amp_shape_loss": parts_accum["amp_shape_loss"] / n,
            "phase_circular_loss": parts_accum["phase_circular_loss"] / n,
            "amp_mse": amp_mse / n,
            "phase_mse": phase_mse / n,
            "g_factor": float(shared_g_factor.detach().cpu()),
            "g_phase": float(shared_g_phase.detach().cpu()),
            "amp_scale": float(amp_scale.detach().cpu()),
            "amp_offset": float(amp_offset.detach().cpu()) if amp_offset is not None else 0.0,
        }
        history.append(row)
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            best_state = row.copy()

    spectra = []
    for record in records:
        loaded = record["loaded"]
        spectra.append(
            {
                "spectrum_index": record["index"],
                "source_file": loaded["source_file"],
                "bg_reference_source": loaded["bg_reference_source"],
                "tip_frequency_hz": loaded["tip_frequency_hz"],
                "tapping_amplitude_nm": loaded["tapping_amplitude_nm"],
                "material_params": record["model"].parameters_dict(),
            }
        )

    return {
        "dataset": str(args.dataset),
        "model": args.model,
        "fit_mode": records[0]["loaded"]["fit_mode"] if records else args.fit_mode,
        "n_spectra": len(records),
        "indices": [record["index"] for record in records],
        "n_wavenumbers": int(len(records[0]["loaded"]["wavenumber"])) if records else 0,
        "wmin": args.wmin,
        "wmax": args.wmax,
        "step": args.step,
        "epochs": args.epochs,
        "start_id": getattr(args, "start_id", 0),
        "center_init": args.center_init,
        "center_jitter": args.center_jitter,
        "seed": args.seed,
        "amp_normalization": args.amp_normalization,
        "reference_band": [args.reference_band_min, args.reference_band_max],
        "amp_baseline": args.amp_baseline,
        "als_lam": args.als_lam,
        "als_p": args.als_p,
        "als_niter": args.als_niter,
        "use_amp_scale": args.use_amp_scale,
        "use_amp_offset": args.use_amp_offset,
        "shared_params": {
            "g_factor": float(shared_g_factor.detach().cpu()),
            "g_phase": float(shared_g_phase.detach().cpu()),
            "amp_scale": float(amp_scale.detach().cpu()),
            "amp_offset": float(amp_offset.detach().cpu()) if amp_offset is not None else 0.0,
            "tip_radius_nm": args.tip_radius_nm,
            "tip_length_nm": args.tip_length_nm,
            "sample_thickness_nm": args.sample_thickness_nm,
            "substrate_material": args.substrate_material,
            "use_three_layer_reflectivity": args.use_three_layer_reflectivity,
            "reflectivity_backend": args.reflectivity_backend,
        },
        "best_epoch": best_state,
        "spectra": spectra,
        "history": history,
    }


def run(args: argparse.Namespace) -> dict:
    start_summaries = []
    best_result = None
    best_loss = float("inf")
    for start_id in range(args.n_starts):
        trial_args = copy.copy(args)
        trial_args.start_id = start_id
        result = _run_once(trial_args)
        loss = float(result["best_epoch"]["total_loss"])
        start_summaries.append(
            {
                "start_id": start_id,
                "best_total_loss": loss,
                "amp_mse": result["best_epoch"]["amp_mse"],
                "phase_mse": result["best_epoch"]["phase_mse"],
                "g_factor": result["shared_params"]["g_factor"],
                "g_phase": result["shared_params"]["g_phase"],
            }
        )
        if loss < best_loss:
            best_loss = loss
            best_result = result
    if best_result is None:
        raise RuntimeError("No batch fit starts were run")
    best_result["n_starts"] = args.n_starts
    best_result["selected_start_id"] = best_result["start_id"]
    best_result["start_summaries"] = start_summaries
    return best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--indices", default=None, help="Comma-separated spectrum indices. Overrides --max-spectra.")
    parser.add_argument("--max-spectra", type=int, default=4)
    parser.add_argument("--model", choices=["multi-lorentz", "single-phonon"], default="multi-lorentz")
    parser.add_argument("--fit-mode", default="auto", choices=["auto", "measured-bg", "raw-reference", "processed-direct"])
    parser.add_argument("--reference-material", default="si", choices=["si", "au", "air"])
    parser.add_argument("--wmin", type=float, default=700.0)
    parser.add_argument("--wmax", type=float, default=1600.0)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
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
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--center-jitter", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=13)
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
    print(json.dumps({k: v for k, v in result.items() if k not in {"history", "spectra"}}, ensure_ascii=False, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
