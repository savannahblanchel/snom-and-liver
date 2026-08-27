"""scikit-learn PLS-DA baseline for processed SNOM spectra datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATASET_NPZ_NAMES = ("spectra_normalized.npz", "spectra_raw_reference.npz")


def find_npz(dataset_dir: Path) -> Path:
    for name in DATASET_NPZ_NAMES:
        path = dataset_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No spectra npz found in {dataset_dir}")


def load_dataset(dataset_dir: Path, label_col: str, filters: list[str] | None = None, group_col: str = "sample_id"):
    npz_path = find_npz(dataset_dir)
    metadata_path = dataset_dir / "metadata_normalized.csv"
    if not metadata_path.exists():
        metadata_path = dataset_dir / "metadata.csv"

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(metadata_path).fillna("")
    amp = np.asarray(z["o2a"], dtype=np.float64)
    phase = np.asarray(z["o2p"], dtype=np.float64)
    wavenumber = np.asarray(z["wavenumber"], dtype=np.float64)

    if len(meta) != amp.shape[0]:
        raise ValueError(f"Metadata rows do not match spectra rows in {dataset_dir}")
    if label_col not in meta.columns:
        raise ValueError(f"Missing label column {label_col!r} in {metadata_path}")

    if filters:
        keep = np.ones(len(meta), dtype=bool)
        for item in filters:
            if "=" not in item:
                raise ValueError(f"Filter must be column=value, got {item!r}")
            col, value = item.split("=", 1)
            if col not in meta.columns:
                raise ValueError(f"Filter column {col!r} not present in {metadata_path}")
            keep &= meta[col].astype(str).to_numpy() == value
        meta = meta.loc[keep].reset_index(drop=True)
        amp = amp[keep]
        phase = phase[keep]

    if group_col not in meta.columns:
        raise ValueError(f"Group column {group_col!r} not present in {metadata_path}")

    labels = meta[label_col].astype(str).to_numpy()
    groups = meta[group_col].astype(str).to_numpy()
    return amp, phase, wavenumber, labels, groups, meta


def build_features(amp: np.ndarray, phase: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "amp":
        return amp
    if feature_set == "phase":
        return phase
    if feature_set == "amp_phase":
        return np.concatenate([amp, phase], axis=1)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def one_hot(y_int: np.ndarray) -> np.ndarray:
    encoder = OneHotEncoder(sparse_output=False, categories="auto")
    return encoder.fit_transform(y_int.reshape(-1, 1))


def make_plsda(n_components: int):
    return make_pipeline(
        StandardScaler(),
        PLSRegression(n_components=n_components, scale=False),
    )


def predict_labels(model, x: np.ndarray, label_encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
    scores = model.predict(x)
    pred_int = np.argmax(scores, axis=1)
    return label_encoder.inverse_transform(pred_int), scores


def max_components_for(x: np.ndarray, groups: np.ndarray, requested: int) -> int:
    return max(1, min(requested, x.shape[0] - 2, x.shape[1], len(np.unique(groups)) - 1))


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_accuracy_curve(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for feature_set, df in summary.groupby("feature_set"):
        df = df.sort_values("n_components")
        plt.plot(df["n_components"], df["point_accuracy"], marker="o", label=f"{feature_set} point")
        plt.plot(df["n_components"], df["sample_accuracy"], marker="s", linestyle="--", label=f"{feature_set} sample")
    plt.xlabel("PLS components")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def coefficient_matrix(model, n_features: int) -> np.ndarray:
    coef = model.named_steps["plsregression"].coef_
    coef = np.asarray(coef)
    if coef.shape[0] != n_features and coef.shape[1] == n_features:
        coef = coef.T
    return coef


def save_feature_importance(
    x: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    wavenumber: np.ndarray,
    labels: np.ndarray,
    feature_set: str,
    n_components: int,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    y_int = label_encoder.transform(labels)
    y_oh = one_hot(y_int)
    model = make_plsda(n_components)
    model.fit(x, y_oh)

    coef = coefficient_matrix(model, x.shape[1])
    if coef.shape[1] == 2:
        importance = np.abs(coef[:, 1] - coef[:, 0])
    else:
        importance = np.linalg.norm(coef - coef.mean(axis=1, keepdims=True), axis=1)

    rows = []
    n_w = len(wavenumber)
    if feature_set in {"amp", "phase"}:
        channel = "O2A" if feature_set == "amp" else "O2P"
        for i, wn in enumerate(wavenumber):
            rows.append({"channel": channel, "wavenumber": wn, "importance": importance[i]})
    else:
        for i, wn in enumerate(wavenumber):
            rows.append({"channel": "O2A", "wavenumber": wn, "importance": importance[i]})
        for i, wn in enumerate(wavenumber):
            rows.append({"channel": "O2P", "wavenumber": wn, "importance": importance[n_w + i]})

    imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
    imp.to_csv(output_dir / "plsda_feature_importance.csv", index=False, encoding="utf-8-sig")
    imp.head(30).to_csv(output_dir / "plsda_top_wavenumbers.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    for channel, df in imp.sort_values("wavenumber").groupby("channel"):
        plt.plot(df["wavenumber"], df["importance"], label=channel, linewidth=1.4)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Coefficient importance")
    plt.title(f"PLS-DA feature importance ({feature_set}, {n_components} comps)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plsda_feature_importance.png", dpi=180)
    plt.close()


def evaluate_dataset(
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    label_col: str,
    feature_sets: list[str],
    max_components: int,
    filters: list[str] | None,
    group_col: str,
) -> pd.DataFrame:
    amp, phase, wavenumber, labels, groups, meta = load_dataset(dataset_dir, label_col, filters=filters, group_col=group_col)
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(labels)
    classes = label_encoder.classes_.tolist()
    y_oh = one_hot(y_int)

    if len(classes) < 2:
        raise ValueError(f"{dataset_name} has fewer than two classes: {classes}")

    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_predictions = []
    sample_rows = []

    logo = LeaveOneGroupOut()

    for feature_set in feature_sets:
        x = build_features(amp, phase, feature_set)
        for n_components in range(1, max_components_for(x, groups, max_components) + 1):
            point_true_all = []
            point_pred_all = []
            fold_point_acc = []
            fold_sample_correct = []

            for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x, y_int, groups)):
                if len(np.unique(y_int[train_idx])) < len(classes):
                    continue

                model = make_plsda(n_components)
                model.fit(x[train_idx], y_oh[train_idx])
                pred_labels, scores = predict_labels(model, x[test_idx], label_encoder)

                true_labels = labels[test_idx]
                point_acc = accuracy_score(true_labels, pred_labels)
                fold_point_acc.append(point_acc)

                score_means = scores.mean(axis=0)
                sample_pred = classes[int(np.argmax(score_means))]
                sample_true_values = np.unique(true_labels)
                sample_true = sample_true_values[0] if len(sample_true_values) == 1 else "|".join(sample_true_values)
                sample_correct = sample_pred == sample_true
                fold_sample_correct.append(sample_correct)

                sample_rows.append(
                    {
                        "dataset": dataset_name,
                        "feature_set": feature_set,
                        "n_components": n_components,
                        "fold": fold_idx,
                        "group_col": group_col,
                        "group_value": groups[test_idx][0],
                        "true_label": sample_true,
                        "pred_label": sample_pred,
                        "correct": bool(sample_correct),
                        "point_accuracy": point_acc,
                    }
                )

                point_true_all.extend(true_labels.tolist())
                point_pred_all.extend(pred_labels.tolist())

                for local_i, row_idx in enumerate(test_idx):
                    row = meta.iloc[row_idx].to_dict()
                    row.update(
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_set,
                            "n_components": n_components,
                            "fold": fold_idx,
                            "true_label": true_labels[local_i],
                            "pred_label": pred_labels[local_i],
                            "correct": bool(true_labels[local_i] == pred_labels[local_i]),
                        }
                    )
                    for class_i, class_name in enumerate(classes):
                        row[f"score_{class_name}"] = scores[local_i, class_i]
                    all_predictions.append(row)

            if not point_true_all:
                continue

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set,
                    "n_components": n_components,
                    "classes": "|".join(classes),
                    "n_points": len(labels),
                    "n_groups": len(np.unique(groups)),
                    "point_accuracy": accuracy_score(point_true_all, point_pred_all),
                    "sample_accuracy": float(np.mean(fold_sample_correct)),
                    "mean_fold_point_accuracy": float(np.mean(fold_point_acc)),
                }
            )

    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(all_predictions)
    sample_predictions = pd.DataFrame(sample_rows)

    summary.to_csv(dataset_output / "plsda_sklearn_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(dataset_output / "plsda_sklearn_predictions.csv", index=False, encoding="utf-8-sig")
    sample_predictions.to_csv(dataset_output / "plsda_sklearn_sample_predictions.csv", index=False, encoding="utf-8-sig")

    best = summary.sort_values(["sample_accuracy", "point_accuracy"], ascending=False).iloc[0].to_dict()
    (dataset_output / "plsda_sklearn_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_feature = best["feature_set"]
    best_components = int(best["n_components"])
    best_point = predictions[
        (predictions["feature_set"] == best_feature)
        & (predictions["n_components"] == best_components)
    ]
    best_sample = sample_predictions[
        (sample_predictions["feature_set"] == best_feature)
        & (sample_predictions["n_components"] == best_components)
    ]

    point_cm = confusion_matrix(best_point["true_label"], best_point["pred_label"], labels=classes)
    sample_cm = confusion_matrix(best_sample["true_label"], best_sample["pred_label"], labels=classes)
    plot_confusion(point_cm, classes, f"{dataset_name} point confusion", dataset_output / "point_confusion.png")
    plot_confusion(sample_cm, classes, f"{dataset_name} sample confusion", dataset_output / "sample_confusion.png")
    plot_accuracy_curve(summary, dataset_output / "accuracy_curve.png")

    best_x = build_features(amp, phase, best_feature)
    save_feature_importance(
        x=best_x,
        amp=amp,
        phase=phase,
        wavenumber=wavenumber,
        labels=labels,
        feature_set=best_feature,
        n_components=best_components,
        label_encoder=label_encoder,
        output_dir=dataset_output,
    )

    return summary


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scikit-learn PLS-DA baselines on SNOM datasets.")
    parser.add_argument("--dataset", action="append", required=True, help="name=path or path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="class_label")
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=["amp", "phase", "amp_phase"],
        default=None,
    )
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument(
        "--filter-eq",
        action="append",
        default=None,
        help="Keep only rows matching column=value. Can be supplied multiple times.",
    )
    parser.add_argument("--group-col", default="sample_id", help="Metadata column used as leave-one-group-out unit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = args.feature_set or ["amp_phase"]

    summaries = []
    for dataset_arg in args.dataset:
        name, path = parse_dataset_arg(dataset_arg)
        summaries.append(
            evaluate_dataset(
                dataset_name=name,
                dataset_dir=path,
                output_dir=output_dir,
                label_col=args.label_col,
                feature_sets=feature_sets,
                max_components=args.max_components,
                filters=args.filter_eq,
                group_col=args.group_col,
            )
        )

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(output_dir / "plsda_sklearn_all_summary.csv", index=False, encoding="utf-8-sig")
    print(all_summary.sort_values(["dataset", "feature_set", "n_components"]).to_string(index=False))


if __name__ == "__main__":
    main()
