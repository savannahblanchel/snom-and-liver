"""Classify a processed SNOM spectrum after physical forward fitting.

The classifier is intentionally downstream of the physics fit:

1. Train a small classifier from existing fitted physical parameters.
2. Fit the unseen amplitude/phase spectrum with the same senior-style kernel.
3. Predict the class from the fitted material-side parameters.

Input spectra should already be processed in the same way as
``sample_level_spectra.npz``. This script does not do raw background matching.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from physical_ml_platform import (  # noqa: E402
    EPS,
    ParameterPredictor,
    PlatformBounds,
    SeniorStyleSnomModel,
    active_literature_bands,
    calibrate_candidate_amp_scales,
    canonicalize_oscillator_order,
    differential_evolution_candidate,
    evaluate_candidates,
    extract_spectrum_features,
    from_unit,
    generate_synthetic_training_data,
    latin_hypercube,
    literature_band_annotation,
    loss_weights_for_wavenumber,
    pretrain_predictor,
    region_mse,
    refine_single,
    seed_everything,
    select_window,
    to_unit,
)
from physical_feature_selector import add_ordered_oscillator_features  # noqa: E402


def load_fit_rows(fit_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(fit_dir.glob("*/sample_fit_results.csv")):
        rows.append(pd.read_csv(path).fillna(""))
    if not rows:
        raise FileNotFoundError(f"No sample_fit_results.csv found under {fit_dir}")
    df = pd.concat(rows, ignore_index=True)
    if "class_label" not in df.columns:
        raise ValueError("Fit table must contain class_label for classifier training")
    return df


def material_feature_columns(df: pd.DataFrame, bounds: PlatformBounds, include_calibration: bool) -> list[str]:
    df = add_ordered_oscillator_features(df, bounds.n_oscillators)
    cols = []
    for i in range(1, bounds.n_oscillators + 1):
        cols.append(f"ordered_wT_{i}")
    cols.extend([f"ordered_wL_{i}" for i in range(1, bounds.n_oscillators + 1)])
    cols.extend([f"ordered_gamma_{i}" for i in range(1, bounds.n_oscillators + 1)])
    cols.extend(["eps_inf"])
    if include_calibration:
        cols.extend(["log_amp_scale", "phase_offset"])
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Fit table is missing classifier feature columns: {missing}")
    return cols


def classifier_training_table(
    fit_df: pd.DataFrame,
    bounds: PlatformBounds,
    include_calibration: bool,
    selected_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = add_ordered_oscillator_features(fit_df, bounds.n_oscillators)
    feature_cols = selected_features or material_feature_columns(table, bounds, include_calibration)
    missing = [col for col in feature_cols if col not in table.columns]
    if missing:
        raise ValueError(f"Fit table is missing selected feature columns: {missing}")
    x = table[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = table["class_label"].astype(str).to_numpy()
    keep = np.isfinite(x).all(axis=1) & (y != "")
    x = x[keep]
    y = y[keep]
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes in the fit results")
    return x, y, feature_cols


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_selected_features(path: Path | None, allow_fallback: bool = False) -> list[str] | None:
    if path is None:
        return None
    df = pd.read_csv(path).fillna("")
    if "feature" not in df.columns:
        raise ValueError(f"Feature selection file {path} must contain a feature column")
    if not allow_fallback and "fallback_selected" in df.columns:
        df = df[~_truthy_series(df["fallback_selected"])].copy()
    features = df["feature"].astype(str).tolist()
    if not features:
        raise ValueError(
            f"Feature selection file {path} does not contain any strict stable features. "
            "Relax the stability thresholds or pass --allow-fallback-features for exploratory classification."
        )
    return features


def infer_tune_wavenumber(path: Path, sample_id: str, fit_df: pd.DataFrame) -> str | None:
    if sample_id:
        match = fit_df[fit_df["sample_id"].astype(str) == sample_id]
        if len(match):
            return str(match["tune_wavenumber"].iloc[0])
    text = f"{path.stem} {sample_id}"
    match = re.search(r"(?<!\d)(1000|1200|1280)(?!\d)", text)
    return match.group(1) if match else None


def filter_training_rows(
    fit_df: pd.DataFrame,
    requested_tune: str,
    inferred_tune: str | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    tune = inferred_tune if requested_tune == "auto" else (None if requested_tune == "all" else requested_tune)
    if not tune:
        return fit_df, None, "No tune wavenumber was inferred; classifier used all fitted samples."
    filtered = fit_df[fit_df["tune_wavenumber"].astype(str) == str(tune)].copy()
    if len(filtered) == 0:
        return fit_df, tune, f"No fitted training rows for tune {tune}; classifier used all fitted samples."
    if filtered["class_label"].astype(str).nunique() < 2:
        return fit_df, tune, f"Tune {tune} has fewer than two classes; classifier used all fitted samples."
    return filtered, tune, None


def make_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42),
    )


def validate_classifier(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    if len(y) < 3 or min(np.bincount(pd.factorize(y)[0])) < 1:
        return {"method": "not_enough_samples", "accuracy": None}
    pred = cross_val_predict(make_classifier(), x, y, cv=LeaveOneOut())
    accuracy = float(accuracy_score(y, pred))
    result = {
        "method": "leave_one_out",
        "accuracy": accuracy,
        "predictions": [
            {"true_label": str(t), "pred_label": str(p), "correct": bool(t == p)}
            for t, p in zip(y, pred)
        ],
    }
    if accuracy < 0.7:
        result["warning"] = "Low leave-one-out accuracy; treat the class prediction as exploratory."
    return result


def load_unknown_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    lower = {col.lower(): col for col in df.columns}
    w_col = lower.get("wavenumber") or lower.get("w") or lower.get("wn")
    amp_col = lower.get("amp") or lower.get("amplitude") or lower.get("o2a")
    phase_col = lower.get("phase") or lower.get("o2p")
    if not w_col or not amp_col or not phase_col:
        raise ValueError("CSV input needs columns wavenumber, amp, phase")
    return (
        pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[phase_col], errors="coerce").to_numpy(dtype=np.float64),
    )


def load_unknown_npz(
    path: Path, sample_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    z = np.load(path, allow_pickle=True)
    w = np.asarray(z["wavenumber"], dtype=np.float64)
    amp_key = "amp_mean" if "amp_mean" in z.files else "amp"
    phase_key = "phase_mean" if "phase_mean" in z.files else "phase"
    if amp_key not in z.files or phase_key not in z.files:
        raise ValueError("NPZ input needs amp_mean/phase_mean or amp/phase arrays")
    amp = np.asarray(z[amp_key], dtype=np.float64)
    phase = np.asarray(z[phase_key], dtype=np.float64)
    if amp.ndim == 2:
        amp = amp[sample_index]
        phase = phase[sample_index]
    bg_amp = None
    bg_phase = None
    if "bg_amp_mean" in z.files and "bg_phase_mean" in z.files:
        bg_amp = np.asarray(z["bg_amp_mean"], dtype=np.float64)
        bg_phase = np.asarray(z["bg_phase_mean"], dtype=np.float64)
        if bg_amp.ndim == 2:
            bg_amp = bg_amp[sample_index]
            bg_phase = bg_phase[sample_index]
    sample_id = ""
    if "sample_id" in z.files:
        sample_id = str(np.asarray(z["sample_id"]).astype(str)[sample_index])
    return w, amp, phase, sample_id, bg_amp, bg_phase


def load_unknown_spectrum(
    path: Path, sample_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_unknown_npz(path, sample_index)
    if suffix == ".csv":
        w, amp, phase = load_unknown_csv(path)
        return w, amp, phase, path.stem, None, None
    raise ValueError("Unknown spectrum must be .csv or .npz")


def train_synthetic_predictor(
    model: SeniorStyleSnomModel,
    w: torch.Tensor,
    bounds: PlatformBounds,
    args: argparse.Namespace,
    device: torch.device,
) -> ParameterPredictor:
    predictor = ParameterPredictor(args.feature_dim, bounds.dimension).to(device)
    features, targets, amp_true, phase_true = generate_synthetic_training_data(
        model=model,
        w=w,
        bounds=bounds,
        n_samples=args.synthetic_samples,
        feature_dim=args.feature_dim,
        seed=args.seed,
        batch_size=args.synthetic_batch_size,
        amp_noise_std=args.synthetic_amp_noise,
        phase_noise_std=args.synthetic_phase_noise,
    )
    features = features.to(device)
    targets = targets.to(device)
    amp_true = amp_true.to(device)
    phase_true = phase_true.to(device)
    pretrain_predictor(
        predictor,
        model,
        w,
        features,
        targets,
        amp_true,
        phase_true,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        seed=args.seed,
        patience=args.pretrain_patience,
        spectrum_weight=args.physics_recon_weight,
        param_weight=args.physics_param_weight,
        phase_weight=args.phase_weight,
        weights=loss_weights_for_wavenumber(
            w,
            args.substrate_band_min,
            args.substrate_band_max,
            args.substrate_band_weight,
        ),
        batch_size=args.pretrain_batch_size,
    )
    return predictor


def seed_candidates_from_fit_table(fit_df: pd.DataFrame, bounds: PlatformBounds, limit: int) -> np.ndarray:
    rows = []
    names = bounds.names()
    table = fit_df.copy()
    for i in range(1, bounds.n_oscillators + 1):
        if f"wT_{i}" not in table.columns and f"wt_{i}" in table.columns:
            table[f"wT_{i}"] = table[f"wt_{i}"]
    for _, row in table.head(max(0, limit)).iterrows():
        if all(name in row.index for name in names):
            values = pd.to_numeric(row[names], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(values).all():
                rows.append(values)
    if not rows:
        return np.empty((0, bounds.dimension), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def fit_unknown_spectrum(
    w_np: np.ndarray,
    amp_np: np.ndarray,
    phase_np: np.ndarray,
    bg_amp_np: np.ndarray | None,
    bg_phase_np: np.ndarray | None,
    fit_df: pd.DataFrame,
    bounds: PlatformBounds,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str]:
    w_t = torch.as_tensor(w_np, dtype=torch.float64, device=device)
    amp_t = torch.as_tensor(amp_np, dtype=torch.float64, device=device)
    phase_t = torch.as_tensor(phase_np, dtype=torch.float64, device=device)
    model = SeniorStyleSnomModel(
        bounds,
        n_time=args.n_time,
        reference_material=args.reference_material,
        fixed_g_factor=args.fixed_g_factor,
        fixed_g_phase=args.fixed_g_phase,
        matched_bg_amp=bg_amp_np,
        matched_bg_phase=bg_phase_np,
    ).to(device)
    loss_weights = loss_weights_for_wavenumber(
        w_t,
        args.substrate_band_min,
        args.substrate_band_max,
        args.substrate_band_weight,
    )
    literature_bands = active_literature_bands(args.literature_prior, args.wmin, args.wmax)
    predictor = train_synthetic_predictor(model, w_t, bounds, args, device)

    feature = torch.as_tensor(
        extract_spectrum_features(amp_np, phase_np, args.feature_dim),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    predictor.eval()
    with torch.no_grad():
        nn_unit = torch.sigmoid(predictor(feature)).cpu().numpy()[0]
    nn_params = canonicalize_oscillator_order(
        from_unit(torch.as_tensor(nn_unit, dtype=torch.float64), bounds).cpu().numpy(),
        bounds,
    )
    table_params = canonicalize_oscillator_order(
        seed_candidates_from_fit_table(fit_df, bounds, args.fit_seed_candidates),
        bounds,
    )
    lhs_params = latin_hypercube(bounds, n_samples=args.init_candidates, seed=args.seed + 11)
    candidate_parts = [nn_params.reshape(1, -1), table_params, lhs_params]
    if args.global_init == "de":
        de_params = differential_evolution_candidate(
            model,
            w_t,
            amp_t,
            phase_t,
            args.phase_weight,
            weights=loss_weights,
            literature_bands=literature_bands,
            literature_center_weight=args.literature_center_weight,
            gamma_width_penalty_weight=args.literature_gamma_weight,
            gamma_soft_max=args.literature_gamma_soft_max,
            literature_distance_scale=args.literature_center_distance_scale,
            derivative_weight=args.derivative_weight,
            phase_derivative_weight=args.phase_derivative_weight,
            maxiter=args.global_init_iters,
            popsize=args.global_init_popsize,
            polish=args.global_init_polish,
            seed=args.seed + 11,
        )
        candidate_parts.append(de_params)
    candidates = np.vstack(candidate_parts)
    candidates = calibrate_candidate_amp_scales(model, w_t, amp_t, candidates, weights=loss_weights)
    losses, best_idx = evaluate_candidates(
        model,
        w_t,
        amp_t,
        phase_t,
        candidates,
        args.phase_weight,
        weights=loss_weights,
        literature_bands=literature_bands,
        literature_center_weight=args.literature_center_weight,
        gamma_width_penalty_weight=args.literature_gamma_weight,
        gamma_soft_max=args.literature_gamma_soft_max,
        literature_distance_scale=args.literature_center_distance_scale,
        derivative_weight=args.derivative_weight,
        phase_derivative_weight=args.phase_derivative_weight,
    )
    table_end = 1 + len(table_params)
    lhs_end = table_end + len(lhs_params)
    if best_idx == 0:
        initial_source = "nn"
    elif best_idx < table_end:
        initial_source = "fit_table"
    elif args.global_init == "de" and best_idx >= lhs_end:
        initial_source = "de"
    else:
        initial_source = "lhs"
    final_params, amp_pred, amp_mse, phase_mse = refine_single(
        model=model,
        w=w_t,
        amp_true=amp_t,
        phase_true=phase_t,
        initial_params=candidates[best_idx],
        steps=args.refine_steps,
        learning_rate=args.refine_lr,
        phase_weight=args.phase_weight,
        weights=loss_weights,
        literature_bands=literature_bands,
        literature_center_weight=args.literature_center_weight,
        gamma_width_penalty_weight=args.literature_gamma_weight,
        gamma_soft_max=args.literature_gamma_soft_max,
        literature_distance_scale=args.literature_center_distance_scale,
        derivative_weight=args.derivative_weight,
        phase_derivative_weight=args.phase_derivative_weight,
    )
    with torch.no_grad():
        phase_pred, _ = model(w_t, model.encode(torch.as_tensor(final_params, dtype=torch.float64, device=device)))
    return final_params, amp_pred, phase_pred[0].cpu().numpy(), amp_mse, phase_mse, initial_source


def params_to_feature_row(
    params: np.ndarray,
    bounds: PlatformBounds,
    feature_cols: list[str],
    fixed_g_factor: float,
    fixed_g_phase: float,
) -> pd.DataFrame:
    values = dict(zip(bounds.names(), params))
    for i in range(1, bounds.n_oscillators + 1):
        values[f"wL_{i}"] = values[f"wT_{i}"] + values[f"gap_{i}"]
    values["g_factor"] = fixed_g_factor
    values["g_phase"] = fixed_g_phase
    values["amp_scale"] = math.exp(values["log_amp_scale"])
    row = add_ordered_oscillator_features(pd.DataFrame([values]), bounds.n_oscillators)
    missing = [col for col in feature_cols if col not in row.columns]
    if missing:
        raise ValueError(f"Unknown fitted spectrum is missing selected feature columns: {missing}")
    return row[feature_cols]


def save_fit_plot(
    path: Path,
    w: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    amp_pred: np.ndarray,
    phase_pred: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(w, amp, color="black", lw=1.2, label="true")
    axes[0].plot(w, amp_pred, color="tab:blue", lw=1.1, label="fit")
    axes[0].set_ylabel("amp")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(w, phase, color="black", lw=1.2, label="true")
    axes[1].plot(w, phase_pred, color="tab:orange", lw=1.1, label="fit")
    axes[1].set_ylabel("phase")
    axes[1].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and classify an unseen processed SNOM spectrum.")
    parser.add_argument("--fit-dir", required=True, help="Existing physical_ml_platform output directory.")
    parser.add_argument("--spectrum", required=True, help="Unknown .csv or .npz spectrum.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0, help="Row index when --spectrum is a multi-sample NPZ.")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--train-tune-wavenumber",
        default="auto",
        help="Use fitted rows from this tune only; choose auto, all, 1000, 1200, or 1280.",
    )
    parser.add_argument("--wmin", type=float, default=690.0)
    parser.add_argument("--wmax", type=float, default=1750.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n-oscillators", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--synthetic-samples", type=int, default=512)
    parser.add_argument("--synthetic-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--pretrain-patience", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--synthetic-amp-noise", type=float, default=0.01)
    parser.add_argument("--synthetic-phase-noise", type=float, default=0.03)
    parser.add_argument("--physics-recon-weight", type=float, default=1.0)
    parser.add_argument("--physics-param-weight", type=float, default=0.2)
    parser.add_argument("--fit-seed-candidates", type=int, default=16)
    parser.add_argument("--init-candidates", type=int, default=24)
    parser.add_argument(
        "--global-init",
        choices=["lhs", "de"],
        default="lhs",
        help="Add an optional differential-evolution candidate before refinement.",
    )
    parser.add_argument("--global-init-iters", type=int, default=6)
    parser.add_argument("--global-init-popsize", type=int, default=6)
    parser.add_argument("--global-init-polish", action="store_true")
    parser.add_argument("--refine-steps", type=int, default=100)
    parser.add_argument("--refine-lr", type=float, default=2e-2)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--derivative-weight", type=float, default=0.0)
    parser.add_argument("--phase-derivative-weight", type=float, default=0.0)
    parser.add_argument("--n-time", type=int, default=65)
    parser.add_argument("--fixed-g-factor", type=float, default=0.5)
    parser.add_argument("--fixed-g-phase", type=float, default=0.03)
    parser.add_argument(
        "--reference-material",
        choices=["au", "si", "sio2", "matched-bg"],
        default="sio2",
        help="Reference/substrate response used for normalized S-SNOM signal. matched-bg requires bg arrays in NPZ input.",
    )
    parser.add_argument("--substrate-band-min", type=float, default=1000.0)
    parser.add_argument("--substrate-band-max", type=float, default=1250.0)
    parser.add_argument("--substrate-band-weight", type=float, default=1.0)
    parser.add_argument(
        "--literature-prior",
        choices=["none", "liver-ftir"],
        default="none",
        help="Soft literature-informed prior for unknown-spectrum refinement.",
    )
    parser.add_argument("--literature-center-weight", type=float, default=0.002)
    parser.add_argument("--literature-center-distance-scale", type=float, default=25.0)
    parser.add_argument("--literature-gamma-weight", type=float, default=0.0002)
    parser.add_argument("--literature-gamma-soft-max", type=float, default=120.0)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--feature-selection", default=None, help="CSV from physical_feature_selector.py.")
    parser.add_argument(
        "--allow-all-material-features",
        action="store_true",
        help="Use all material features when no --feature-selection is provided. Exploratory only.",
    )
    parser.add_argument(
        "--allow-fallback-features",
        action="store_true",
        help="Use fallback-selected features from physical_feature_selector.py. Exploratory only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    bounds = PlatformBounds(n_oscillators=args.n_oscillators)

    w_full, amp_full, phase_full, loaded_id, bg_amp_full, bg_phase_full = load_unknown_spectrum(
        Path(args.spectrum), args.sample_index
    )
    w, amp = select_window(w_full, amp_full, args.wmin, args.wmax, args.stride)
    _, phase = select_window(w_full, phase_full, args.wmin, args.wmax, args.stride)
    bg_amp = None
    bg_phase = None
    if bg_amp_full is not None and bg_phase_full is not None:
        _, bg_amp = select_window(w_full, bg_amp_full, args.wmin, args.wmax, args.stride)
        _, bg_phase = select_window(w_full, bg_phase_full, args.wmin, args.wmax, args.stride)
    sample_id = args.sample_id or loaded_id or Path(args.spectrum).stem

    fit_df = load_fit_rows(Path(args.fit_dir))
    inferred_tune = infer_tune_wavenumber(Path(args.spectrum), sample_id, fit_df)
    train_df, used_tune, filter_warning = filter_training_rows(fit_df, args.train_tune_wavenumber, inferred_tune)
    if args.feature_selection:
        selected_features = load_selected_features(
            Path(args.feature_selection),
            allow_fallback=args.allow_fallback_features,
        )
    elif args.allow_all_material_features:
        selected_features = None
    else:
        raise ValueError(
            "Classification requires --feature-selection so only stable parameters are used. "
            "Pass --allow-all-material-features for exploratory classification."
        )
    x_train, y_train, feature_cols = classifier_training_table(
        train_df,
        bounds,
        args.include_calibration,
        selected_features=selected_features,
    )
    validation = validate_classifier(x_train, y_train)
    classifier = make_classifier()
    classifier.fit(x_train, y_train)

    params, amp_pred, phase_pred, amp_mse, phase_mse, initial_source = fit_unknown_spectrum(
        w,
        amp,
        phase,
        bg_amp,
        bg_phase,
        train_df,
        bounds,
        args,
        device,
    )
    literature_bands = active_literature_bands(args.literature_prior, args.wmin, args.wmax)

    x_unknown = params_to_feature_row(params, bounds, feature_cols, args.fixed_g_factor, args.fixed_g_phase)
    pred_label = str(classifier.predict(x_unknown.to_numpy(dtype=np.float64))[0])
    classes = classifier.classes_.astype(str).tolist()
    probabilities = classifier.predict_proba(x_unknown.to_numpy(dtype=np.float64))[0]
    max_probability = float(np.max(probabilities))
    warnings = [item for item in [filter_warning, validation.get("warning")] if item]
    if max_probability < 0.65:
        warnings.append("Low class probability margin; inspect the fitted parameters and overlay plot.")

    param_values = dict(zip(bounds.names(), map(float, params)))
    for i in range(1, bounds.n_oscillators + 1):
        param_values[f"wL_{i}"] = float(param_values[f"wT_{i}"] + param_values[f"gap_{i}"])
        in_lit, lit_label, lit_distance = literature_band_annotation(
            param_values[f"wT_{i}"], literature_bands
        )
        param_values[f"center_in_literature_band_{i}"] = bool(in_lit)
        param_values[f"literature_band_label_{i}"] = lit_label
        param_values[f"center_distance_to_literature_{i}"] = float(lit_distance)
        param_values[f"center_in_sio2_band_{i}"] = bool(
            args.substrate_band_min <= param_values[f"wT_{i}"] <= args.substrate_band_max
        )
    param_values["g_factor"] = args.fixed_g_factor
    param_values["g_phase"] = args.fixed_g_phase
    param_values["amp_scale"] = float(math.exp(param_values["log_amp_scale"]))
    amp_region = region_mse(w, amp, amp_pred, args.substrate_band_min, args.substrate_band_max)
    phase_region = region_mse(w, phase, phase_pred, args.substrate_band_min, args.substrate_band_max, circular=True)

    result = {
        "sample_id": sample_id,
        "pred_label": pred_label,
        "class_probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)},
        "max_class_probability": max_probability,
        "train_tune_wavenumber": used_tune or "all",
        "training_samples": int(len(y_train)),
        "training_class_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
        "warning": "; ".join(warnings) if warnings else None,
        "warnings": warnings,
        "amp_mse": float(amp_mse),
        "phase_circular_mse": float(phase_mse),
        "amp_mse_substrate_band": amp_region["band"],
        "amp_mse_outside_substrate_band": amp_region["outside"],
        "phase_circular_mse_substrate_band": phase_region["band"],
        "phase_circular_mse_outside_substrate_band": phase_region["outside"],
        "reference_material": args.reference_material,
        "substrate_band": [args.substrate_band_min, args.substrate_band_max],
        "substrate_band_weight": args.substrate_band_weight,
        "literature_prior": args.literature_prior,
        "global_init": args.global_init,
        "global_init_iters": args.global_init_iters,
        "global_init_popsize": args.global_init_popsize,
        "derivative_weight": args.derivative_weight,
        "phase_derivative_weight": args.phase_derivative_weight,
        "literature_prior_bands": [
            {"label": band.label, "min": band.band_min, "max": band.band_max} for band in literature_bands
        ],
        "initial_source": initial_source,
        "classifier_features": feature_cols,
        "classifier_validation": validation,
        "parameters": param_values,
    }
    (output_dir / "prediction_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"sample_id": sample_id, **{f"prob_{cls}": prob for cls, prob in result["class_probabilities"].items()}, **param_values}]).to_csv(
        output_dir / "prediction_result.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if validation.get("predictions"):
        pd.DataFrame(validation["predictions"]).to_csv(output_dir / "classifier_leave_one_out.csv", index=False, encoding="utf-8-sig")
    save_fit_plot(output_dir / "unknown_fit_overlay.png", w, amp, phase, amp_pred, phase_pred, f"{sample_id}: {pred_label}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
