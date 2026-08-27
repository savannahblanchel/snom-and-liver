"""Scan fixed tapping-coupling G values for the senior-style SNOM platform."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def read_selected_features(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna("")
    if "feature" not in df.columns:
        return []
    return df["feature"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-scan fixed G values for physical_ml_platform.py.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--g-factors", default="0.5,0.7,0.9,1.1")
    parser.add_argument("--g-phases", default="0.03,0.06,0.10")
    parser.add_argument("--tune-wavenumber", default="1280")
    parser.add_argument("--wmin", type=float, default=690.0)
    parser.add_argument("--wmax", type=float, default=1750.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n-oscillators", type=int, default=2)
    parser.add_argument("--synthetic-samples", type=int, default=1024)
    parser.add_argument("--synthetic-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pretrain-patience", type=int, default=20)
    parser.add_argument("--synthetic-amp-noise", type=float, default=0.01)
    parser.add_argument("--synthetic-phase-noise", type=float, default=0.03)
    parser.add_argument("--init-candidates", type=int, default=16)
    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--reference-material", choices=["au", "si", "sio2"], default="sio2")
    parser.add_argument("--substrate-band-min", type=float, default=1000.0)
    parser.add_argument("--substrate-band-max", type=float, default=1250.0)
    parser.add_argument("--substrate-band-weight", type=float, default=1.0)
    parser.add_argument("--n-time", type=int, default=65)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-global", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for g_factor in parse_float_list(args.g_factors):
        for g_phase in parse_float_list(args.g_phases):
            tag = f"g{g_factor:.3f}_p{g_phase:.3f}".replace(".", "p")
            fit_dir = output_dir / tag / "fit"
            feature_dir = output_dir / tag / "features"
            run_command(
                [
                    sys.executable,
                    str(HERE / "physical_ml_platform.py"),
                    "--dataset",
                    args.dataset,
                    "--output-dir",
                    str(fit_dir),
                    "--wmin",
                    str(args.wmin),
                    "--wmax",
                    str(args.wmax),
                    "--stride",
                    str(args.stride),
                    "--n-oscillators",
                    str(args.n_oscillators),
                    "--synthetic-samples",
                    str(args.synthetic_samples),
                    "--synthetic-batch-size",
                    str(args.synthetic_batch_size),
                    "--pretrain-epochs",
                    str(args.pretrain_epochs),
                    "--pretrain-patience",
                    str(args.pretrain_patience),
                    "--synthetic-amp-noise",
                    str(args.synthetic_amp_noise),
                    "--synthetic-phase-noise",
                    str(args.synthetic_phase_noise),
                    "--init-candidates",
                    str(args.init_candidates),
                    "--refine-steps",
                    str(args.refine_steps),
                    "--refine-lr",
                    str(args.refine_lr),
                    "--phase-weight",
                    str(args.phase_weight),
                    "--reference-material",
                    args.reference_material,
                    "--fixed-g-factor",
                    str(g_factor),
                    "--fixed-g-phase",
                    str(g_phase),
                    "--substrate-band-min",
                    str(args.substrate_band_min),
                    "--substrate-band-max",
                    str(args.substrate_band_max),
                    "--substrate-band-weight",
                    str(args.substrate_band_weight),
                    "--n-time",
                    str(args.n_time),
                    "--device",
                    args.device,
                ]
            )
            selector_cmd = [
                sys.executable,
                str(HERE / "physical_feature_selector.py"),
                "--fit-dir",
                str(fit_dir),
                "--output-dir",
                str(feature_dir),
                "--tune-wavenumber",
                args.tune_wavenumber,
                "--n-oscillators",
                str(args.n_oscillators),
                "--substrate-band-min",
                str(args.substrate_band_min),
                "--substrate-band-max",
                str(args.substrate_band_max),
            ]
            if args.include_global:
                selector_cmd.append("--include-global")
            run_command(selector_cmd)

            summary = pd.read_csv(fit_dir / "platform_fit_summary.csv")
            selected = read_selected_features(feature_dir / "selected_physical_features.csv")
            target = summary[summary["batch"].astype(str).str.endswith(f"__{args.tune_wavenumber}")]
            if target.empty:
                target = summary
            row = target.iloc[0].to_dict()
            row.update(
                {
                    "g_factor": g_factor,
                    "g_phase": g_phase,
                    "selected_features": ";".join(selected),
                    "n_selected_features": len(selected),
                    "fit_dir": str(fit_dir),
                    "feature_dir": str(feature_dir),
                }
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "fixed_g_scan_summary.csv", index=False, encoding="utf-8-sig")

    (output_dir / "fixed_g_scan_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).sort_values(["n_selected_features", "amp_mse_mean"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
