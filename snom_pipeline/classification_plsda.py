"""PLS-DA baseline classification for processed SNOM spectra datasets.

The implementation is intentionally self-contained and uses only numpy/pandas so
it can run in the current project environment without scikit-learn.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_NPZ_NAMES = ("spectra_normalized.npz", "spectra_raw_reference.npz")


@dataclass
class FoldResult:
    dataset: str
    feature_set: str
    n_components: int
    group: str
    point_accuracy: float
    sample_true: str
    sample_pred: str
    sample_correct: bool


def find_npz(dataset_dir: Path) -> Path:
    for name in DATASET_NPZ_NAMES:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spectra npz found in {dataset_dir}")


def load_dataset(dataset_dir: Path, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    npz_path = find_npz(dataset_dir)
    metadata_path = dataset_dir / "metadata_normalized.csv"
    if not metadata_path.exists():
        metadata_path = dataset_dir / "metadata.csv"

    z = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(metadata_path).fillna("")

    amp = np.asarray(z["o2a"], dtype=np.float64)
    phase = np.asarray(z["o2p"], dtype=np.float64)
    if len(meta) != amp.shape[0]:
        raise ValueError(f"Metadata row count does not match spectra rows in {dataset_dir}")
    if label_col not in meta.columns:
        raise ValueError(f"Label column {label_col!r} not present in {metadata_path}")

    labels = meta[label_col].astype(str).to_numpy()
    groups = meta["sample_id"].astype(str).to_numpy()
    return amp, phase, labels, groups, meta


def build_features(amp: np.ndarray, phase: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "amp":
        return amp
    if feature_set == "phase":
        return phase
    if feature_set == "amp_phase":
        return np.concatenate([amp, phase], axis=1)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        y[row, mapping[label]] = 1.0
    return y


class PLSDA:
    def __init__(self, n_components: int, max_iter: int = 500, tol: float = 1e-9):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: np.ndarray, y_onehot: np.ndarray) -> "PLSDA":
        self.x_mean_ = x.mean(axis=0)
        self.x_std_ = x.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        self.y_mean_ = y_onehot.mean(axis=0)

        x_work = (x - self.x_mean_) / self.x_std_
        y_work = y_onehot - self.y_mean_

        n_samples, n_features = x_work.shape
        n_targets = y_work.shape[1]
        max_components = min(self.n_components, n_samples - 1, n_features)

        weights = []
        loadings = []
        y_loadings = []

        for _ in range(max_components):
            col_norms = np.linalg.norm(y_work, axis=0)
            if np.max(col_norms) < 1e-12:
                break
            u = y_work[:, int(np.argmax(col_norms))].copy()

            for _iter in range(self.max_iter):
                w = x_work.T @ u
                w_norm = np.linalg.norm(w)
                if w_norm < 1e-12:
                    break
                w = w / w_norm

                t = x_work @ w
                t_norm_sq = float(t.T @ t)
                if t_norm_sq < 1e-12:
                    break

                q = y_work.T @ t / t_norm_sq
                q_norm_sq = float(q.T @ q)
                if q_norm_sq < 1e-12:
                    break

                u_next = y_work @ q / q_norm_sq
                if np.linalg.norm(u_next - u) < self.tol:
                    u = u_next
                    break
                u = u_next

            w = x_work.T @ u
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-12:
                break
            w = w / w_norm
            t = x_work @ w
            t_norm_sq = float(t.T @ t)
            if t_norm_sq < 1e-12:
                break

            p = x_work.T @ t / t_norm_sq
            q = y_work.T @ t / t_norm_sq

            x_work = x_work - np.outer(t, p)
            y_work = y_work - np.outer(t, q)

            weights.append(w)
            loadings.append(p)
            y_loadings.append(q)

        if not weights:
            raise RuntimeError("PLS-DA failed to extract any component.")

        self.weights_ = np.column_stack(weights)
        self.loadings_ = np.column_stack(loadings)
        self.y_loadings_ = np.column_stack(y_loadings)
        self.n_components_fit_ = self.weights_.shape[1]

        middle = np.linalg.pinv(self.loadings_.T @ self.weights_)
        self.coef_ = self.weights_ @ middle @ self.y_loadings_.T
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x_scaled = (x - self.x_mean_) / self.x_std_
        return x_scaled @ self.coef_ + self.y_mean_

    def predict(self, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        return classes[np.argmax(scores, axis=1)]


def leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    folds = []
    for group in sorted(np.unique(groups)):
        test_mask = groups == group
        train_mask = ~test_mask
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0], group))
    return folds


def evaluate_dataset(
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    label_col: str,
    feature_sets: list[str],
    max_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    amp, phase, labels, groups, meta = load_dataset(dataset_dir, label_col)
    classes = np.array(sorted(np.unique(labels)))
    if len(classes) < 2:
        raise ValueError(f"Need at least two classes for {dataset_name}; got {classes.tolist()}")

    all_fold_rows = []
    all_prediction_rows = []

    for feature_set in feature_sets:
        x = build_features(amp, phase, feature_set)
        max_allowed = min(max_components, len(np.unique(groups)) - 1, x.shape[0] - 2, x.shape[1])
        component_values = range(1, max(1, max_allowed) + 1)

        for n_components in component_values:
            fold_rows: list[FoldResult] = []
            prediction_rows = []

            for train_idx, test_idx, group in leave_one_group_out(groups):
                train_classes = np.unique(labels[train_idx])
                if len(train_classes) < len(classes):
                    continue

                model = PLSDA(n_components=n_components)
                y_train = one_hot(labels[train_idx], classes)
                model.fit(x[train_idx], y_train)

                scores = model.decision_function(x[test_idx])
                point_pred = classes[np.argmax(scores, axis=1)]
                point_true = labels[test_idx]
                point_accuracy = float(np.mean(point_pred == point_true))

                sample_scores = scores.mean(axis=0)
                sample_pred = str(classes[int(np.argmax(sample_scores))])
                sample_true_values = np.unique(point_true)
                sample_true = str(sample_true_values[0]) if len(sample_true_values) == 1 else "|".join(sample_true_values)
                sample_correct = sample_pred == sample_true

                fold_rows.append(
                    FoldResult(
                        dataset=dataset_name,
                        feature_set=feature_set,
                        n_components=n_components,
                        group=group,
                        point_accuracy=point_accuracy,
                        sample_true=sample_true,
                        sample_pred=sample_pred,
                        sample_correct=sample_correct,
                    )
                )

                for local_idx, row_idx in enumerate(test_idx):
                    row = meta.iloc[row_idx].to_dict()
                    for class_idx, class_name in enumerate(classes):
                        row[f"score_{class_name}"] = scores[local_idx, class_idx]
                    row.update(
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_set,
                            "n_components": n_components,
                            "true_label": labels[row_idx],
                            "pred_label": point_pred[local_idx],
                            "correct": bool(point_pred[local_idx] == labels[row_idx]),
                        }
                    )
                    prediction_rows.append(row)

            if not fold_rows:
                continue

            point_correct = [
                pred_row["correct"]
                for pred_row in prediction_rows
            ]
            summary_row = {
                "dataset": dataset_name,
                "feature_set": feature_set,
                "n_components": n_components,
                "n_classes": len(classes),
                "classes": "|".join(classes),
                "n_points": len(labels),
                "n_groups": len(np.unique(groups)),
                "point_accuracy": float(np.mean(point_correct)),
                "sample_accuracy": float(np.mean([row.sample_correct for row in fold_rows])),
                "mean_fold_point_accuracy": float(np.mean([row.point_accuracy for row in fold_rows])),
            }
            all_fold_rows.append(summary_row)
            all_prediction_rows.extend(prediction_rows)

    summary = pd.DataFrame(all_fold_rows)
    predictions = pd.DataFrame(all_prediction_rows)

    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dataset_output / "plsda_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(dataset_output / "plsda_predictions.csv", index=False, encoding="utf-8-sig")

    if not summary.empty:
        best = summary.sort_values(["sample_accuracy", "point_accuracy"], ascending=False).iloc[0].to_dict()
        (dataset_output / "plsda_best.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary, predictions


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped PLS-DA baselines on SNOM datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset path, optionally as name=path.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="class_label")
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=["amp", "phase", "amp_phase"],
        default=None,
        help="Feature set(s) to evaluate. Defaults to amp_phase.",
    )
    parser.add_argument("--max-components", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = args.feature_set or ["amp_phase"]

    combined = []
    for dataset_arg in args.dataset:
        dataset_name, dataset_dir = parse_dataset_arg(dataset_arg)
        summary, _ = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            label_col=args.label_col,
            feature_sets=feature_sets,
            max_components=args.max_components,
        )
        combined.append(summary)

    if combined:
        combined_summary = pd.concat(combined, ignore_index=True)
        combined_summary.to_csv(output_dir / "plsda_all_summary.csv", index=False, encoding="utf-8-sig")
        print(combined_summary.sort_values(["dataset", "feature_set", "n_components"]).to_string(index=False))


if __name__ == "__main__":
    main()
