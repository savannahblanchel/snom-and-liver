"""Select stable physical parameters for downstream SNOM classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-8


def load_fit_rows(fit_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(fit_dir.glob("*/sample_fit_results.csv")):
        df = pd.read_csv(path).fillna("")
        df["fit_batch_dir"] = path.parent.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No sample_fit_results.csv found under {fit_dir}")
    return pd.concat(rows, ignore_index=True)


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def add_ordered_oscillator_features(df: pd.DataFrame, n_oscillators: int, sort_by: str = "wT") -> pd.DataFrame:
    out = df.copy()
    for i in range(1, n_oscillators + 1):
        if f"wT_{i}" not in out.columns and f"wt_{i}" in out.columns:
            out[f"wT_{i}"] = out[f"wt_{i}"]
        if f"wL_{i}" not in out.columns and f"wT_{i}" in out.columns and f"gap_{i}" in out.columns:
            out[f"wL_{i}"] = pd.to_numeric(out[f"wT_{i}"], errors="coerce") + pd.to_numeric(
                out[f"gap_{i}"], errors="coerce"
            )

    sort_cols = [f"{sort_by}_{i}" for i in range(1, n_oscillators + 1)]
    needed = sort_cols + [f"wT_{i}" for i in range(1, n_oscillators + 1)]
    needed += [f"wL_{i}" for i in range(1, n_oscillators + 1)]
    needed += [f"gap_{i}" for i in range(1, n_oscillators + 1)]
    needed += [f"gamma_{i}" for i in range(1, n_oscillators + 1)]
    missing = [col for col in needed if col not in out.columns]
    if missing:
        raise ValueError(f"Fit table is missing oscillator columns: {missing}")

    for prefix in ("wT", "wL", "gap", "gamma"):
        values = out[[f"{prefix}_{i}" for i in range(1, n_oscillators + 1)]].apply(pd.to_numeric, errors="coerce")
        sort_values = out[sort_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        source_values = values.to_numpy(dtype=float)
        ordered = np.full_like(source_values, np.nan, dtype=float)
        for row_idx in range(len(out)):
            order = np.argsort(sort_values[row_idx])
            ordered[row_idx] = source_values[row_idx, order]
        for i in range(n_oscillators):
            out[f"ordered_{prefix}_{i + 1}"] = ordered[:, i]
    return out


def feature_bounds(feature: str, n_oscillators: int) -> tuple[float, float] | None:
    if feature.startswith("ordered_wT_"):
        return 650.0, 1650.0
    if feature.startswith("ordered_wL_"):
        return 670.0, 1950.0
    if feature.startswith("ordered_gap_"):
        return 20.0, 300.0
    if feature.startswith("ordered_gamma_"):
        return 5.0, 300.0
    if feature == "eps_inf":
        return 1.0, 10.0
    if feature == "g_factor":
        return 0.7, 0.7
    if feature == "g_phase":
        return 0.06, 0.06
    return None


def feature_family(feature: str) -> str:
    if feature.startswith("ordered_wT_") or feature.startswith("ordered_wL_"):
        return "center"
    if feature.startswith("ordered_gap_"):
        return "gap"
    if feature.startswith("ordered_gamma_"):
        return "gamma"
    if feature in {"eps_inf", "g_factor", "g_phase"}:
        return "global"
    return "other"


def ordered_oscillator_index(feature: str) -> int | None:
    for prefix in ("ordered_wT_", "ordered_wL_", "ordered_gap_", "ordered_gamma_"):
        if feature.startswith(prefix):
            try:
                return int(feature.removeprefix(prefix))
            except ValueError:
                return None
    return None


def cv_threshold_for(feature: str, args: argparse.Namespace) -> float:
    family = feature_family(feature)
    if family == "center":
        return args.center_cv_max
    if family == "gamma":
        return args.gamma_cv_max
    if family == "gap":
        return args.gap_cv_max
    if family == "global":
        return args.global_cv_max
    return args.global_cv_max


def separation_score(values: np.ndarray, labels: np.ndarray) -> float | None:
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    best = 0.0
    for i, left in enumerate(classes):
        for right in classes[i + 1 :]:
            a = values[labels == left]
            b = values[labels == right]
            if len(a) == 0 or len(b) == 0:
                continue
            pooled = np.sqrt((np.var(a, ddof=0) + np.var(b, ddof=0)) / 2.0) + EPS
            best = max(best, float(abs(np.mean(a) - np.mean(b)) / pooled))
    return best


def summarize_feature(df: pd.DataFrame, feature: str, args: argparse.Namespace) -> dict[str, object]:
    values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
    labels = df["class_label"].astype(str).to_numpy()
    finite = np.isfinite(values) & (labels != "")
    values = values[finite]
    labels = labels[finite]
    row: dict[str, object] = {
        "feature": feature,
        "family": feature_family(feature),
        "n": int(len(values)),
        "n_classes": int(len(np.unique(labels))),
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "std": float(np.std(values, ddof=0)) if len(values) else np.nan,
    }
    lower_upper = feature_bounds(feature, args.n_oscillators)
    if lower_upper:
        lower, upper = lower_upper
        boundary = np.isclose(values, lower, atol=args.boundary_atol) | np.isclose(values, upper, atol=args.boundary_atol)
        row["boundary_hit_rate"] = float(np.mean(boundary)) if len(values) else np.nan
        row["lower_bound"] = lower
        row["upper_bound"] = upper
    else:
        row["boundary_hit_rate"] = 0.0
        row["lower_bound"] = np.nan
        row["upper_bound"] = np.nan

    cv_values = []
    for class_label in sorted(np.unique(labels)):
        group = values[labels == class_label]
        mean = float(np.mean(group))
        std = float(np.std(group, ddof=0))
        cv = float(std / (abs(mean) + EPS))
        row[f"{class_label}_mean"] = mean
        row[f"{class_label}_std"] = std
        row[f"{class_label}_cv"] = cv
        row[f"{class_label}_n"] = int(len(group))
        cv_values.append(cv)
    row["within_class_cv_max"] = float(max(cv_values)) if cv_values else np.nan
    row["within_class_cv_mean"] = float(np.mean(cv_values)) if cv_values else np.nan
    sep = separation_score(values, labels)
    row["separation"] = np.nan if sep is None else sep

    cv_limit = cv_threshold_for(feature, args)
    row["cv_limit"] = cv_limit
    center_in_substrate_band = bool(
        feature_family(feature) == "center"
        and len(values)
        and args.substrate_band_min <= float(np.nanmean(values)) <= args.substrate_band_max
    )
    row["in_substrate_band"] = center_in_substrate_band
    osc_idx = ordered_oscillator_index(feature)
    lit_col = f"center_in_literature_band_{osc_idx}" if osc_idx is not None else None
    sio2_col = f"center_in_sio2_band_{osc_idx}" if osc_idx is not None else None
    if lit_col and lit_col in df.columns:
        lit_values = _truthy_series(df.loc[finite, lit_col])
        row["literature_supported_rate"] = float(lit_values.mean()) if len(lit_values) else np.nan
    else:
        row["literature_supported_rate"] = np.nan
    if sio2_col and sio2_col in df.columns:
        sio2_values = _truthy_series(df.loc[finite, sio2_col])
        row["sio2_band_rate"] = float(sio2_values.mean()) if len(sio2_values) else np.nan
    else:
        row["sio2_band_rate"] = np.nan
    row["selected"] = bool(
        len(values) >= args.min_total_n
        and len(np.unique(labels)) >= 2
        and row["boundary_hit_rate"] <= args.max_boundary_hit_rate
        and row["within_class_cv_max"] <= cv_limit
        and (sep is not None and sep >= args.min_separation)
        and (args.allow_substrate_band_centers or not center_in_substrate_band)
    )
    return row


def candidate_features(n_oscillators: int, include_gap: bool, include_global: bool) -> list[str]:
    features = [f"ordered_wT_{i}" for i in range(1, n_oscillators + 1)]
    features += [f"ordered_wL_{i}" for i in range(1, n_oscillators + 1)]
    if include_gap:
        features += [f"ordered_gap_{i}" for i in range(1, n_oscillators + 1)]
    features += [f"ordered_gamma_{i}" for i in range(1, n_oscillators + 1)]
    if include_global:
        features += ["eps_inf"]
    return features


def filter_tune(df: pd.DataFrame, tune: str) -> tuple[pd.DataFrame, str]:
    if tune == "all":
        return df.copy(), "all"
    selected = df[df["tune_wavenumber"].astype(str) == tune].copy()
    if selected.empty:
        raise ValueError(f"No fitted rows for tune_wavenumber={tune}")
    return selected, tune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select stable physical features from platform fit results.")
    parser.add_argument("--fit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-wavenumber", default="1280", help="Use 1280 by default; pass all to pool all fitted rows.")
    parser.add_argument("--n-oscillators", type=int, default=2)
    parser.add_argument("--sort-by", choices=["wT", "wL"], default="wT")
    parser.add_argument("--min-total-n", type=int, default=4)
    parser.add_argument("--center-cv-max", type=float, default=0.12)
    parser.add_argument("--gamma-cv-max", type=float, default=0.35)
    parser.add_argument("--gap-cv-max", type=float, default=0.35)
    parser.add_argument("--global-cv-max", type=float, default=0.25)
    parser.add_argument("--min-separation", type=float, default=0.8)
    parser.add_argument("--max-boundary-hit-rate", type=float, default=0.34)
    parser.add_argument("--boundary-atol", type=float, default=1e-6)
    parser.add_argument("--include-gap", action="store_true")
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--substrate-band-min", type=float, default=1000.0)
    parser.add_argument("--substrate-band-max", type=float, default=1250.0)
    parser.add_argument(
        "--allow-substrate-band-centers",
        action="store_true",
        help="Allow center features inside the SiO2 substrate resonance band to be selected.",
    )
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        default=0,
        help="Optional exploratory fallback when no strict feature passes. Default 0 keeps classification strictly stable-feature only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_ordered_oscillator_features(load_fit_rows(Path(args.fit_dir)), args.n_oscillators, sort_by=args.sort_by)
    train_df, tune = filter_tune(df, args.tune_wavenumber)
    features = candidate_features(args.n_oscillators, args.include_gap, args.include_global)
    summary = pd.DataFrame([summarize_feature(train_df, feature, args) for feature in features])
    summary = summary.sort_values(["selected", "separation", "within_class_cv_max"], ascending=[False, False, True])
    selected = summary[summary["selected"]].copy()
    if selected.empty and args.fallback_top_k > 0:
        fallback = summary[
            (summary["n_classes"] >= 2)
            & (summary["boundary_hit_rate"] <= args.max_boundary_hit_rate)
            & np.isfinite(summary["separation"])
            & (args.allow_substrate_band_centers | ~summary["in_substrate_band"])
        ].copy()
        fallback = fallback.sort_values(["separation", "within_class_cv_max"], ascending=[False, True]).head(
            args.fallback_top_k
        )
        fallback["selected"] = True
        fallback["fallback_selected"] = True
        selected = fallback
    else:
        selected["fallback_selected"] = False

    summary.to_csv(output_dir / "physical_feature_stability.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_physical_features.csv", index=False, encoding="utf-8-sig")
    config = vars(args).copy()
    config["tune_wavenumber_used"] = tune
    config["n_training_rows"] = int(len(train_df))
    config["class_counts"] = {str(k): int(v) for k, v in train_df["class_label"].astype(str).value_counts().items()}
    config["selected_features"] = selected["feature"].astype(str).tolist()
    (output_dir / "feature_selection_summary.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
