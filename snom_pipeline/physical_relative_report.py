"""Visual report for batch-level relative SNOM physical fits.

Produces:
- predicted-vs-true spectrum overlays for each sample;
- residual curves;
- class-wise parameter stability summaries and boxplots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from physical_batch_relative_fit import (  # noqa: E402
    EPS,
    PhaseCalibration,
    ReferenceSpectrum,
    apply_phase_calibration,
    compose_with_reference,
    forward_spectrum,
    load_reference_spectrum,
    select_window,
    wrap_phase,
)


CENTER_CV_THRESHOLD = 0.1
BOUNDARY_ATOL = 1e-9


def load_sample_bundle(dataset_dir: Path) -> dict[str, np.ndarray]:
    npz_path = dataset_dir / "sample_level_spectra.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing sample_level_spectra.npz in {dataset_dir}")
    return dict(np.load(npz_path, allow_pickle=True))


def load_sample_table(dataset_dir: Path) -> pd.DataFrame:
    csv_path = dataset_dir / "sample_level_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sample_level_summary.csv in {dataset_dir}")
    return pd.read_csv(csv_path).fillna("")


def load_fit_table(batch_dir: Path) -> pd.DataFrame:
    csv_path = batch_dir / "sample_fit_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sample_fit_results.csv in {batch_dir}")
    df = pd.read_csv(csv_path).fillna("")
    if "analysis_group" not in df.columns:
        df["analysis_group"] = df["specimen_name"].astype(str) + "__" + df["tune_wavenumber"].astype(str).replace({"": "na"})
    return df


def load_batch_summary(batch_dir: Path) -> dict[str, object]:
    json_path = batch_dir / "batch_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing batch_summary.json in {batch_dir}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def batch_source_dir(batch_name: str, spectra_root: Path) -> Path:
    dataset_name = batch_name.split("__", 1)[0]
    return spectra_root / dataset_name


def _numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    def key(col: str) -> int:
        return int(col.rsplit("_", 1)[-1])

    cols = [col for col in df.columns if col.startswith(f"{prefix}_") and col.rsplit("_", 1)[-1].isdigit()]
    return sorted(cols, key=key)


def parameter_columns(fit_df: pd.DataFrame) -> list[str]:
    return ["eps_inf", *_numbered_columns(fit_df, "center"), *_numbered_columns(fit_df, "gamma"), *_numbered_columns(fit_df, "strength")]


def _param_bounds(param: str, wmin: float, wmax: float) -> tuple[float, float] | None:
    if param.startswith("center_"):
        return wmin, wmax
    if param.startswith("gamma_"):
        return 5.0, 300.0
    if param.startswith("strength_"):
        return 0.0, 20.0
    if param == "eps_inf":
        return 0.1, 20.0
    return None


def _param_stats(values: pd.Series, lower: float | None, upper: float | None) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }
    stats["cv"] = float(stats["std"] / (abs(stats["mean"]) + EPS))
    if lower is not None and upper is not None:
        boundary_hit = np.isclose(arr, lower, atol=BOUNDARY_ATOL) | np.isclose(arr, upper, atol=BOUNDARY_ATOL)
        stats["min"] = float(np.min(arr))
        stats["max"] = float(np.max(arr))
        stats["boundary_hit_rate"] = float(np.mean(boundary_hit))
    return stats


def reconstruct_sample_prediction(
    row: pd.Series,
    w_full: np.ndarray,
    amp_full: np.ndarray,
    phase_full: np.ndarray,
    wmin: float,
    wmax: float,
    stride: int,
    reference: ReferenceSpectrum | None = None,
    phase_calibration: PhaseCalibration | None = None,
) -> dict[str, np.ndarray]:
    w, amp_true = select_window(w_full, amp_full, wmin, wmax, stride)
    _, phase_true = select_window(w_full, phase_full, wmin, wmax, stride)
    phase_true = apply_phase_calibration(phase_true, w, phase_calibration)
    centers = np.array([row[col] for col in _numbered_columns(row.to_frame().T, "center")], dtype=np.float64)
    gammas = np.array([row[col] for col in _numbered_columns(row.to_frame().T, "gamma")], dtype=np.float64)
    strengths = np.array([row[col] for col in _numbered_columns(row.to_frame().T, "strength")], dtype=np.float64)
    amp_pred, phase_pred = forward_spectrum(
        w,
        float(row["eps_inf"]),
        centers,
        gammas,
        strengths,
        float(row["gain"]),
        float(row["phase_shift"]),
        float(row.get("amp_offset", 0.0)),
    )
    amp_pred, phase_pred = compose_with_reference(amp_pred, phase_pred, w, reference)
    return {
        "w": w,
        "amp_true": amp_true,
        "phase_true": phase_true,
        "amp_pred": amp_pred,
        "phase_pred": phase_pred,
        "amp_residual": amp_pred - amp_true,
        "phase_residual": wrap_phase(phase_pred - phase_true),
    }


def plot_batch_overview(
    batch_name: str,
    fit_df: pd.DataFrame,
    spectra_bundle: dict[str, np.ndarray],
    wmin: float,
    wmax: float,
    stride: int,
    output_dir: Path,
    reference: ReferenceSpectrum | None = None,
    phase_calibration: PhaseCalibration | None = None,
) -> Path:
    w_full = np.asarray(spectra_bundle["wavenumber"], dtype=np.float64)
    sample_ids = np.asarray(spectra_bundle["sample_id"], dtype=str)
    amp_all = np.asarray(spectra_bundle["amp_mean"], dtype=np.float64)
    phase_all = np.asarray(spectra_bundle["phase_mean"], dtype=np.float64)
    sample_index = {sid: idx for idx, sid in enumerate(sample_ids)}

    rows = [row for _, row in fit_df.sort_values(["analysis_group", "sample_id"]).iterrows()]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(3.0, 2.0 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        idx = sample_index[str(row["sample_id"])]
        pred = reconstruct_sample_prediction(
            row, w_full, amp_all[idx], phase_all[idx], wmin, wmax, stride, reference, phase_calibration
        )
        title = f'{row["analysis_group"]} | {row["class_label"]}'

        ax = axes[i, 0]
        ax.plot(pred["w"], pred["amp_true"], color="black", lw=1.2, label="true")
        ax.plot(pred["w"], pred["amp_pred"], color="tab:blue", lw=1.0, label="pred")
        ax.set_ylabel("amp")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[i, 1]
        ax.plot(pred["w"], pred["phase_true"], color="black", lw=1.2, label="true")
        ax.plot(pred["w"], pred["phase_pred"], color="tab:orange", lw=1.0, label="pred")
        ax.set_ylabel("phase (corrected)")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.suptitle(f"{batch_name}: predicted vs true spectra", y=0.995, fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{batch_name}_pred_vs_true.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_batch_residuals(
    batch_name: str,
    fit_df: pd.DataFrame,
    spectra_bundle: dict[str, np.ndarray],
    wmin: float,
    wmax: float,
    stride: int,
    output_dir: Path,
    reference: ReferenceSpectrum | None = None,
    phase_calibration: PhaseCalibration | None = None,
) -> Path:
    w_full = np.asarray(spectra_bundle["wavenumber"], dtype=np.float64)
    sample_ids = np.asarray(spectra_bundle["sample_id"], dtype=str)
    amp_all = np.asarray(spectra_bundle["amp_mean"], dtype=np.float64)
    phase_all = np.asarray(spectra_bundle["phase_mean"], dtype=np.float64)
    sample_index = {sid: idx for idx, sid in enumerate(sample_ids)}

    rows = [row for _, row in fit_df.sort_values(["class_label", "sample_id"]).iterrows()]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(3.0, 2.0 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        idx = sample_index[str(row["sample_id"])]
        pred = reconstruct_sample_prediction(
            row, w_full, amp_all[idx], phase_all[idx], wmin, wmax, stride, reference, phase_calibration
        )
        title = f'{row["analysis_group"]} | {row["class_label"]}'

        ax = axes[i, 0]
        ax.plot(pred["w"], pred["amp_residual"], color="tab:red", lw=1.0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("amp res")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)

        ax = axes[i, 1]
        ax.plot(pred["w"], pred["phase_residual"], color="tab:green", lw=1.0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("phase res (corrected)")
        ax.grid(alpha=0.2)

    axes[-1, 0].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.suptitle(f"{batch_name}: residual curves", y=0.995, fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{batch_name}_residuals.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_stability_table(fit_df: pd.DataFrame, wmin: float, wmax: float) -> pd.DataFrame:
    param_cols = parameter_columns(fit_df)
    rows = []
    for analysis_group, group in fit_df.groupby("analysis_group", sort=True):
        row = {
            "analysis_group": analysis_group,
            "class_label": str(group["class_label"].iloc[0]),
            "specimen_name": str(group["specimen_name"].iloc[0]),
            "tune_wavenumber": str(group["tune_wavenumber"].iloc[0]),
            "n_samples": int(len(group)),
        }
        for col in param_cols:
            bounds = _param_bounds(col, wmin, wmax)
            stats = _param_stats(group[col], None if bounds is None else bounds[0], None if bounds is None else bounds[1])
            if not stats:
                continue
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
            row[f"{col}_cv"] = stats["cv"]
            if "min" in stats:
                row[f"{col}_min"] = stats["min"]
                row[f"{col}_max"] = stats["max"]
                row[f"{col}_boundary_hit_rate"] = stats["boundary_hit_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_center_shortlist(stability_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    center_cols = sorted(
        {
            col.removesuffix("_cv")
            for col in stability_df.columns
            if col.startswith("center_") and col.endswith("_cv")
        },
        key=lambda col: int(col.rsplit("_", 1)[-1]),
    )
    for _, row in stability_df.iterrows():
        for center_col in center_cols:
            cv_key = f"{center_col}_cv"
            hit_key = f"{center_col}_boundary_hit_rate"
            if cv_key not in row or hit_key not in row:
                continue
            cv = float(row[cv_key])
            boundary_hit_rate = float(row[hit_key])
            if not np.isfinite(cv):
                continue
            if cv <= CENTER_CV_THRESHOLD and boundary_hit_rate == 0.0:
                rows.append(
                    {
                        "analysis_group": row["analysis_group"],
                        "class_label": row["class_label"],
                        "specimen_name": row["specimen_name"],
                        "tune_wavenumber": row["tune_wavenumber"],
                        "center_param": center_col,
                        "mean": float(row[f"{center_col}_mean"]),
                        "std": float(row[f"{center_col}_std"]),
                        "cv": cv,
                        "boundary_hit_rate": boundary_hit_rate,
                        "min": float(row.get(f"{center_col}_min", np.nan)),
                        "max": float(row.get(f"{center_col}_max", np.nan)),
                        "n_samples": int(row["n_samples"]),
                    }
                )
    shortlist = pd.DataFrame(rows)
    if not shortlist.empty:
        shortlist = shortlist.sort_values(["cv", "analysis_group", "center_param"]).reset_index(drop=True)
    return shortlist


def plot_parameter_boxes(batch_name: str, fit_df: pd.DataFrame, output_dir: Path) -> Path:
    params = [*_numbered_columns(fit_df, "center"), *_numbered_columns(fit_df, "gamma"), *_numbered_columns(fit_df, "strength")]
    classes = list(sorted(fit_df["analysis_group"].astype(str).unique()))
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(4, 3.5 * n_rows)))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for idx, param in enumerate(params):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        grouped = [fit_df.loc[fit_df["analysis_group"].astype(str) == cls, param].astype(float).to_numpy() for cls in classes]
        ax.boxplot(grouped, tick_labels=classes, showmeans=True)
        ax.set_title(param)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelrotation=20)
    for idx in range(len(params), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")
    fig.suptitle(f"{batch_name}: class-wise parameter stability", y=0.995, fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{batch_name}_parameter_stability.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_for_batch(batch_dir: Path, spectra_root: Path, output_root: Path) -> dict[str, object]:
    batch_name = batch_dir.name
    batch_summary = load_batch_summary(batch_dir)
    fit_df = load_fit_table(batch_dir)
    source_dir = batch_source_dir(batch_name, spectra_root)
    bundle = load_sample_bundle(source_dir)
    sample_table = load_sample_table(source_dir)
    reference_file = str(batch_summary.get("reference_file", "") or "")
    reference = load_reference_spectrum(Path(reference_file)) if reference_file else None
    phase_calibration = PhaseCalibration(
        mode=str(batch_summary.get("phase_calibration_mode", "none")),
        phi0=float(batch_summary.get("phase_calibration_phi0", 0.0)),
        phi1=float(batch_summary.get("phase_calibration_phi1", 0.0)),
        w0=float(batch_summary.get("phase_calibration_w0", (float(batch_summary["wmin"]) + float(batch_summary["wmax"])) / 2.0)),
        windows=tuple(),
        n_points=0,
    )

    out_dir = output_root / batch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_plot = plot_batch_overview(
        batch_name=batch_name,
        fit_df=fit_df,
        spectra_bundle=bundle,
        wmin=float(batch_summary["wmin"]),
        wmax=float(batch_summary["wmax"]),
        stride=int(batch_summary["stride"]),
        output_dir=out_dir,
        reference=reference,
        phase_calibration=phase_calibration,
    )
    resid_plot = plot_batch_residuals(
        batch_name=batch_name,
        fit_df=fit_df,
        spectra_bundle=bundle,
        wmin=float(batch_summary["wmin"]),
        wmax=float(batch_summary["wmax"]),
        stride=int(batch_summary["stride"]),
        output_dir=out_dir,
        reference=reference,
        phase_calibration=phase_calibration,
    )
    stab_plot = plot_parameter_boxes(batch_name=batch_name, fit_df=fit_df, output_dir=out_dir)
    stability = compute_stability_table(fit_df, wmin=float(batch_summary["wmin"]), wmax=float(batch_summary["wmax"]))
    stability.to_csv(out_dir / "class_parameter_stability.csv", index=False, encoding="utf-8-sig")
    center_shortlist = build_center_shortlist(stability)
    center_shortlist.to_csv(out_dir / "center_parameter_shortlist.csv", index=False, encoding="utf-8-sig")

    sample_join = sample_table[["sample_id", "n_points", "amp_abs_dev_mean", "phase_abs_dev_mean", "amp_std_mean", "phase_std_mean"]].copy()
    fit_join_cols = ["sample_id", "analysis_group", "amp_mse", "phase_circular_mse", "residual_l2", "eps_inf", "gain", "phase_shift", *_numbered_columns(fit_df, "center")]
    if "fit_space" in fit_df.columns:
        fit_join_cols.insert(2, "fit_space")
    if "complex_mse" in fit_df.columns:
        fit_join_cols.insert(3, "complex_mse")
    if "phase_calibration_mode" in fit_df.columns:
        fit_join_cols.insert(2, "phase_calibration_mode")
    if "phase_calibration_phi0" in fit_df.columns:
        fit_join_cols.insert(3, "phase_calibration_phi0")
    if "phase_calibration_phi1" in fit_df.columns:
        fit_join_cols.insert(4, "phase_calibration_phi1")
    fit_join = fit_df[fit_join_cols].copy()
    merged = pd.merge(sample_join, fit_join, on="sample_id", how="inner")
    merged.to_csv(out_dir / "fit_vs_stability_join.csv", index=False, encoding="utf-8-sig")

    report = {
        "batch": batch_name,
        "source_dir": str(source_dir),
        "pred_plot": str(pred_plot),
        "resid_plot": str(resid_plot),
        "stability_plot": str(stab_plot),
        "n_samples": int(len(fit_df)),
        "n_classes": int(fit_df["class_label"].nunique()),
        "amp_mse_mean": float(fit_df["amp_mse"].mean()),
        "phase_circular_mse_mean": float(fit_df["phase_circular_mse"].mean()),
        "complex_mse_mean": float(fit_df["complex_mse"].mean()) if "complex_mse" in fit_df.columns else np.nan,
        "residual_l2_mean": float(fit_df["residual_l2"].mean()),
        "reference_file": reference_file,
        "reference_mode": str(batch_summary.get("reference_mode", "none")),
        "phase_calibration_mode": str(batch_summary.get("phase_calibration_mode", "none")),
        "phase_calibration_phi0": float(batch_summary.get("phase_calibration_phi0", 0.0)),
        "phase_calibration_phi1": float(batch_summary.get("phase_calibration_phi1", 0.0)),
        "center_shortlist_csv": str(out_dir / "center_parameter_shortlist.csv"),
    }
    (out_dir / "report_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual report for relative SNOM physical fits.")
    parser.add_argument("--fit-dir", required=True, help="Directory containing batch fit subdirectories.")
    parser.add_argument("--spectra-root", required=True, help="Directory containing sample-level spectra folders.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_dir = Path(args.fit_dir)
    spectra_root = Path(args.spectra_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = [path for path in sorted(fit_dir.iterdir()) if path.is_dir() and (path / "sample_fit_results.csv").exists()]
    reports = []
    for batch_dir in batch_dirs:
        reports.append(report_for_batch(batch_dir, spectra_root, output_dir))

    summary = pd.DataFrame(reports)
    summary.to_csv(output_dir / "visual_report_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
