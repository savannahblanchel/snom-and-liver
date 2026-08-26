"""Relative spectrum analysis for processed SNOM spectra.

This step treats the available spectra as already processed inputs and focuses on
sample-level aggregation, within-sample stability, and class-level summaries.

It does not attempt absolute dielectric inversion. The goal is to provide a
clean entry point for the next physical-fitting stage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_NPZ_NAMES = ("spectra_normalized.npz", "spectra_raw_reference.npz")


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    path: Path
    metadata: pd.DataFrame
    spectra: dict[str, np.ndarray]


def find_npz(dataset_dir: Path) -> Path:
    for name in DATASET_NPZ_NAMES:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spectra npz found in {dataset_dir}")


def load_bundle(dataset_name: str, dataset_dir: Path) -> DatasetBundle:
    npz_path = find_npz(dataset_dir)
    metadata_path = dataset_dir / "metadata_normalized.csv"
    if not metadata_path.exists():
        metadata_path = dataset_dir / "metadata.csv"

    spectra = dict(np.load(npz_path, allow_pickle=True))
    metadata = pd.read_csv(metadata_path).fillna("")
    amp = np.asarray(spectra["o2a"], dtype=np.float64)
    phase = np.asarray(spectra["o2p"], dtype=np.float64)

    if len(metadata) != amp.shape[0] or amp.shape != phase.shape:
        raise ValueError(f"Metadata/spectra shape mismatch in {dataset_dir}")

    return DatasetBundle(name=dataset_name, path=dataset_dir, metadata=metadata, spectra=spectra)


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def summarise_group(
    meta: pd.DataFrame,
    amp: np.ndarray,
    phase: np.ndarray,
    bg_amp: np.ndarray | None = None,
    bg_phase: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    wavenumber = np.asarray(meta.attrs.get("wavenumber"))
    if wavenumber.size == 0:
        raise ValueError("Missing wavenumber axis")

    rows = []
    sample_ids = []
    sample_labels = []
    class_labels = []
    specimen_names = []
    markers = []
    tune_wavenumbers = []
    n_points_list = []
    amp_mean_list = []
    amp_std_list = []
    amp_median_list = []
    phase_mean_list = []
    phase_std_list = []
    phase_median_list = []
    bg_amp_mean_list = []
    bg_phase_mean_list = []
    amp_abs_dev_list = []
    phase_abs_dev_list = []

    for sample_id, sample_meta in meta.groupby("sample_id", sort=True):
        idx = sample_meta.index.to_numpy()
        amp_block = amp[idx]
        phase_block = phase[idx]
        bg_amp_block = bg_amp[idx] if bg_amp is not None else None
        bg_phase_block = bg_phase[idx] if bg_phase is not None else None
        specimen_name = str(sample_meta["specimen_name"].iloc[0])
        class_label = str(sample_meta["class_label"].iloc[0])
        if specimen_name.startswith("TA0038190758"):
            class_label = "organoid"

        amp_mean = np.mean(amp_block, axis=0)
        phase_mean = np.mean(phase_block, axis=0)
        amp_median = np.median(amp_block, axis=0)
        phase_median = np.median(phase_block, axis=0)
        amp_std = np.std(amp_block, axis=0, ddof=0)
        phase_std = np.std(phase_block, axis=0, ddof=0)
        if bg_amp_block is not None and bg_phase_block is not None:
            bg_amp_mean_list.append(np.nanmean(bg_amp_block, axis=0))
            bg_phase_mean_list.append(np.nanmean(bg_phase_block, axis=0))

        sample_labels.append(class_label)
        class_labels.append(class_label)
        specimen_names.append(specimen_name)
        markers.append(str(sample_meta["marker"].iloc[0]))
        tune_wavenumbers.append(str(sample_meta["tune_wavenumber"].iloc[0]))
        sample_ids.append(str(sample_id))
        n_points_list.append(int(len(sample_meta)))
        amp_mean_list.append(amp_mean)
        amp_std_list.append(amp_std)
        amp_median_list.append(amp_median)
        phase_mean_list.append(phase_mean)
        phase_std_list.append(phase_std)
        phase_median_list.append(phase_median)
        amp_abs_dev_list.append(float(np.mean(np.abs(amp_block - amp_mean))))
        phase_abs_dev_list.append(float(np.mean(np.abs(phase_block - phase_mean))))

        rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "specimen_name": specimen_name,
                "marker": sample_meta["marker"].iloc[0],
                "tune_wavenumber": sample_meta["tune_wavenumber"].iloc[0],
                "n_points": len(sample_meta),
                "amp_abs_dev_mean": amp_abs_dev_list[-1],
                "phase_abs_dev_mean": phase_abs_dev_list[-1],
                "amp_std_mean": float(np.mean(amp_std)),
                "phase_std_mean": float(np.mean(phase_std)),
                "amp_std_median": float(np.median(amp_std)),
                "phase_std_median": float(np.median(phase_std)),
            }
        )

    table = pd.DataFrame(rows).reset_index(drop=True)
    sample_arrays = {
        "wavenumber": wavenumber.astype(np.float32),
        "sample_id": np.asarray(sample_ids),
        "class_label": np.asarray(class_labels),
        "specimen_name": np.asarray(specimen_names),
        "marker": np.asarray(markers),
        "tune_wavenumber": np.asarray(tune_wavenumbers),
        "n_points": np.asarray(n_points_list, dtype=np.int32),
        "amp_mean": np.asarray(amp_mean_list, dtype=np.float32),
        "amp_std": np.asarray(amp_std_list, dtype=np.float32),
        "amp_median": np.asarray(amp_median_list, dtype=np.float32),
        "phase_mean": np.asarray(phase_mean_list, dtype=np.float32),
        "phase_std": np.asarray(phase_std_list, dtype=np.float32),
        "phase_median": np.asarray(phase_median_list, dtype=np.float32),
    }
    if bg_amp_mean_list and bg_phase_mean_list:
        sample_arrays["bg_amp_mean"] = np.asarray(bg_amp_mean_list, dtype=np.float32)
        sample_arrays["bg_phase_mean"] = np.asarray(bg_phase_mean_list, dtype=np.float32)
    return table, sample_arrays


def analyse_dataset(dataset_name: str, dataset_dir: Path, output_dir: Path) -> dict[str, object]:
    bundle = load_bundle(dataset_name, dataset_dir)
    meta = bundle.metadata.copy()
    spectra = bundle.spectra

    meta.attrs["wavenumber"] = np.asarray(spectra["wavenumber"], dtype=np.float64)
    amp = np.asarray(spectra["o2a"], dtype=np.float64)
    phase = np.asarray(spectra["o2p"], dtype=np.float64)
    bg_amp = np.asarray(spectra["o2a_background"], dtype=np.float64) if "o2a_background" in spectra else None
    bg_phase = np.asarray(spectra["o2p_background"], dtype=np.float64) if "o2p_background" in spectra else None
    wavenumber = np.asarray(spectra["wavenumber"], dtype=np.float64)

    sample_table, sample_arrays = summarise_group(meta, amp, phase, bg_amp=bg_amp, bg_phase=bg_phase)

    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    sample_table.to_csv(dataset_output / "sample_level_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(dataset_output / "sample_level_spectra.npz", **sample_arrays)

    class_rows = []
    for class_label, class_df in sample_table.groupby("class_label", sort=True):
        class_rows.append(
            {
                "class_label": class_label,
                "n_samples": len(class_df),
                "total_points": int(class_df["n_points"].sum()),
                "amp_abs_dev_mean": float(class_df["amp_abs_dev_mean"].mean()),
                "phase_abs_dev_mean": float(class_df["phase_abs_dev_mean"].mean()),
                "amp_std_mean": float(class_df["amp_std_mean"].mean()),
                "phase_std_mean": float(class_df["phase_std_mean"].mean()),
            }
        )
    class_table = pd.DataFrame(class_rows).sort_values("class_label").reset_index(drop=True)
    class_table.to_csv(dataset_output / "class_level_summary.csv", index=False, encoding="utf-8-sig")

    overall = {
        "dataset": dataset_name,
        "n_points": int(len(meta)),
        "n_samples": int(sample_table.shape[0]),
        "n_classes": int(sample_table["class_label"].nunique()),
        "amp_abs_dev_mean": float(sample_table["amp_abs_dev_mean"].mean()),
        "phase_abs_dev_mean": float(sample_table["phase_abs_dev_mean"].mean()),
        "amp_std_mean": float(sample_table["amp_std_mean"].mean()),
        "phase_std_mean": float(sample_table["phase_std_mean"].mean()),
        "wavenumber_min": float(wavenumber.min()),
        "wavenumber_max": float(wavenumber.max()),
        "source_dir": str(dataset_dir),
    }
    (dataset_output / "dataset_summary.json").write_text(
        pd.Series(overall).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise processed SNOM spectra at the sample level.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset path, optionally as name=path.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset_arg in args.dataset:
        dataset_name, dataset_dir = parse_dataset_arg(dataset_arg)
        summaries.append(analyse_dataset(dataset_name, dataset_dir, output_dir))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "relative_analysis_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
