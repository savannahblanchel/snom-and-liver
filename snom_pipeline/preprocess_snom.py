"""Build processed SNOM spectra datasets from Neaspec point-spectroscopy txt files.

This module implements the first layer of the project pipeline:
raw O2A/O2P spectra -> background-normalized spectra -> reusable arrays.

Normalization rule:
    O2A_norm(w) = O2A_sample(w) / O2A_background(w)
    O2P_norm(w) = O2P_sample(w) - O2P_background(w)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WAVENUMBER_MIN = 690.0
DEFAULT_WAVENUMBER_MAX = 1800.0
EPS = 1e-12
MODE_BG_NORMALIZED = "bg-normalized"
MODE_RAW_REFERENCE = "raw-reference"


@dataclass(frozen=True)
class SpectrumRecord:
    path: Path
    dataset_name: str
    sample_id: str
    point_id: str
    class_label: str
    specimen_name: str
    marker: str
    tune_wavenumber: str
    is_background: bool


def read_neaspec_txt(path: Path) -> pd.DataFrame:
    """Read a Neaspec point-spectroscopy txt file into a numeric DataFrame."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Row\tColumn"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"No Neaspec data header found in {path}")

    df = pd.read_csv(path, sep="\t", skiprows=start_idx, engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    unnamed = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"Wavenumber", "O2A", "O2P"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Wavenumber"]).sort_values("Wavenumber").reset_index(drop=True)


def clean_stem(path: Path) -> str:
    stem = path.stem
    return re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{6}\s+PH\s+PSP\s+", "", stem, flags=re.I).strip()


def sample_key_from_name(name: str) -> tuple[str, str]:
    match = re.search(r"-(\d+)$", name)
    if match:
        return name[: match.start()], match.group(1)
    return name, ""


def infer_class_label(name: str) -> str:
    lower = name.lower()
    if "gan" in lower:
        return "liver"
    if "zhong" in lower:
        return "tumor"
    if "bian" in lower or re.search(r"\bta\b", lower):
        return "tonsil"
    if "lei" in lower:
        return "organoid"
    return "unknown"


def infer_specimen_name(name: str) -> str:
    lower = name.lower()
    if "bian" in lower:
        return "人扁桃体_14_PD1_1.500"
    if re.search(r"\bta\b", lower):
        return "TA0038190758_PD1_1.500"
    if "lei 2" in lower or "lei2" in lower:
        return "类器官2 ZX Cell 2"
    if "lei 1" in lower or "lei1" in lower:
        return "类器官1 ZX Cell 1"
    if "gan" in lower:
        return "肝SK类原位"
    if "zhong" in lower:
        return "肿瘤SK类皮下"
    return ""


def infer_marker(name: str) -> str:
    lower = name.lower()
    if "ikzf" in lower or "izkf" in lower:
        return "IKZF1"
    if "mpap" in lower or "mfap" in lower:
        return "MFAP4"
    if "tcr" in lower:
        return "TCR"
    if "pd1" in lower or "bian" in lower or re.search(r"\bta\b", lower):
        return "PD1"
    if "ki67" in lower:
        return "KI67"
    if "lei" in lower:
        return "none"
    return "unknown"


def infer_tune_wavenumber(name: str) -> str:
    match = re.search(r"-(1000|1200|1280)(?:-|$)", name)
    return match.group(1) if match else ""


def discover_records(input_dirs: Iterable[Path]) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    for input_dir in input_dirs:
        dataset_name = input_dir.name
        for path in sorted(input_dir.glob("*.txt")):
            cleaned = clean_stem(path)
            sample_id, point_id = sample_key_from_name(cleaned)
            is_background = bool(re.search(r"(^|[-_\s])bg($|[-_\s])", cleaned, flags=re.I))
            records.append(
                SpectrumRecord(
                    path=path,
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    point_id=point_id,
                    class_label=infer_class_label(cleaned),
                    specimen_name=infer_specimen_name(cleaned),
                    marker=infer_marker(cleaned),
                    tune_wavenumber=infer_tune_wavenumber(cleaned),
                    is_background=is_background,
                )
            )
    return records


def choose_background(records: list[SpectrumRecord], sample: SpectrumRecord) -> SpectrumRecord | None:
    backgrounds = [record for record in records if record.is_background and record.dataset_name == sample.dataset_name]
    if not backgrounds:
        return None

    same_class = [record for record in backgrounds if record.class_label == sample.class_label]
    if same_class:
        return same_class[0]

    return backgrounds[0]


def interpolate_to_axis(source_w: np.ndarray, source_y: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    order = np.argsort(source_w)
    return np.interp(target_w, source_w[order], source_y[order])


def build_dataset(
    input_dirs: list[Path],
    output_dir: Path,
    wavenumber_min: float = DEFAULT_WAVENUMBER_MIN,
    wavenumber_max: float = DEFAULT_WAVENUMBER_MAX,
    mode: str = MODE_BG_NORMALIZED,
) -> None:
    if mode not in {MODE_BG_NORMALIZED, MODE_RAW_REFERENCE}:
        raise ValueError(f"Unsupported mode: {mode}")

    records = discover_records(input_dirs)
    samples = [record for record in records if not record.is_background]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    norm_amp = []
    norm_phase = []
    raw_amp = []
    raw_phase = []
    sample_ids = []
    point_ids = []

    canonical_axis: np.ndarray | None = None
    parsed_cache: dict[Path, pd.DataFrame] = {}

    for sample in samples:
        bg = choose_background(records, sample) if mode == MODE_BG_NORMALIZED else None
        status = "ok" if (bg is not None or mode == MODE_RAW_REFERENCE) else "missing_background"
        error = ""

        try:
            sample_df = parsed_cache.setdefault(sample.path, read_neaspec_txt(sample.path))
            mask = (sample_df["Wavenumber"] >= wavenumber_min) & (sample_df["Wavenumber"] <= wavenumber_max)
            sample_df = sample_df.loc[mask].copy()
            if canonical_axis is None:
                canonical_axis = sample_df["Wavenumber"].to_numpy(dtype=float)

            sample_w = sample_df["Wavenumber"].to_numpy(dtype=float)
            amp_raw = interpolate_to_axis(sample_w, sample_df["O2A"].to_numpy(dtype=float), canonical_axis)
            phase_raw = interpolate_to_axis(sample_w, sample_df["O2P"].to_numpy(dtype=float), canonical_axis)

            if mode == MODE_RAW_REFERENCE:
                norm_amp.append(amp_raw.astype(np.float32))
                norm_phase.append(phase_raw.astype(np.float32))
                raw_amp.append(amp_raw.astype(np.float32))
                raw_phase.append(phase_raw.astype(np.float32))
                sample_ids.append(sample.sample_id)
                point_ids.append(sample.point_id)
            elif bg is not None:
                bg_df = parsed_cache.setdefault(bg.path, read_neaspec_txt(bg.path))
                bg_w = bg_df["Wavenumber"].to_numpy(dtype=float)
                bg_amp = interpolate_to_axis(bg_w, bg_df["O2A"].to_numpy(dtype=float), canonical_axis)
                bg_phase = interpolate_to_axis(bg_w, bg_df["O2P"].to_numpy(dtype=float), canonical_axis)

                small_bg = np.abs(bg_amp) < EPS
                if np.any(small_bg):
                    status = "small_background_amplitude"
                amp_norm = amp_raw / np.where(small_bg, np.nan, bg_amp)
                phase_norm = phase_raw - bg_phase

                norm_amp.append(amp_norm.astype(np.float32))
                norm_phase.append(phase_norm.astype(np.float32))
                raw_amp.append(amp_raw.astype(np.float32))
                raw_phase.append(phase_raw.astype(np.float32))
                sample_ids.append(sample.sample_id)
                point_ids.append(sample.point_id)
        except Exception as exc:  # Keep batch processing auditable.
            status = "error"
            error = str(exc)

        metadata_rows.append(
            {
                "dataset_name": sample.dataset_name,
                "sample_id": sample.sample_id,
                "point_id": sample.point_id,
                "class_label": sample.class_label,
                "specimen_name": sample.specimen_name,
                "marker": sample.marker,
                "tune_wavenumber": sample.tune_wavenumber,
                "source_file": str(sample.path),
                "background_file": "" if bg is None else str(bg.path),
                "processing_mode": mode,
                "status": status,
                "error": error,
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    if canonical_axis is None:
        raise RuntimeError("No sample spectra were found.")

    ok_metadata = metadata[metadata["status"].isin(["ok", "small_background_amplitude"])].reset_index(drop=True)
    ok_metadata.to_csv(output_dir / "metadata_normalized.csv", index=False, encoding="utf-8-sig")

    norm_amp_arr = np.asarray(norm_amp, dtype=np.float32)
    norm_phase_arr = np.asarray(norm_phase, dtype=np.float32)
    raw_amp_arr = np.asarray(raw_amp, dtype=np.float32)
    raw_phase_arr = np.asarray(raw_phase, dtype=np.float32)

    dropped_wavenumbers: list[float] = []
    if norm_amp_arr.size:
        finite_columns = np.isfinite(norm_amp_arr).all(axis=0) & np.isfinite(norm_phase_arr).all(axis=0)
        dropped_wavenumbers = canonical_axis[~finite_columns].astype(float).tolist()
        canonical_axis = canonical_axis[finite_columns]
        norm_amp_arr = norm_amp_arr[:, finite_columns]
        norm_phase_arr = norm_phase_arr[:, finite_columns]
        raw_amp_arr = raw_amp_arr[:, finite_columns]
        raw_phase_arr = raw_phase_arr[:, finite_columns]

    npz_name = "spectra_normalized.npz" if mode == MODE_BG_NORMALIZED else "spectra_raw_reference.npz"
    np.savez_compressed(
        output_dir / npz_name,
        wavenumber=canonical_axis.astype(np.float32),
        o2a=norm_amp_arr,
        o2p=norm_phase_arr,
        o2a_processed=norm_amp_arr,
        o2p_processed=norm_phase_arr,
        o2a_norm=norm_amp_arr,
        o2p_norm=norm_phase_arr,
        o2a_raw=raw_amp_arr,
        o2p_raw=raw_phase_arr,
        sample_id=np.asarray(sample_ids),
        point_id=np.asarray(point_ids),
        processing_mode=np.asarray(mode),
    )

    status_counts = metadata["status"].value_counts(dropna=False).to_dict()
    summary = pd.DataFrame(
        [
            {"metric": "input_dirs", "value": ";".join(str(path) for path in input_dirs)},
            {"metric": "processing_mode", "value": mode},
            {"metric": "wavenumber_min", "value": wavenumber_min},
            {"metric": "wavenumber_max", "value": wavenumber_max},
            {"metric": "total_sample_points", "value": len(samples)},
            {"metric": "normalized_points", "value": len(ok_metadata)},
            {"metric": "dropped_wavenumber_count", "value": len(dropped_wavenumbers)},
            {"metric": "dropped_wavenumbers", "value": dropped_wavenumbers},
            {"metric": "status_counts", "value": status_counts},
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build background-normalized SNOM spectra arrays.")
    parser.add_argument("--input-dir", action="append", required=True, help="Directory containing Neaspec txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory for metadata CSV and NPZ output.")
    parser.add_argument("--wavenumber-min", type=float, default=DEFAULT_WAVENUMBER_MIN)
    parser.add_argument("--wavenumber-max", type=float, default=DEFAULT_WAVENUMBER_MAX)
    parser.add_argument(
        "--mode",
        choices=[MODE_BG_NORMALIZED, MODE_RAW_REFERENCE],
        default=MODE_BG_NORMALIZED,
        help="bg-normalized uses sample/bg and phase subtraction; raw-reference keeps exported O2A/O2P directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        input_dirs=[Path(item) for item in args.input_dir],
        output_dir=Path(args.output_dir),
        wavenumber_min=args.wavenumber_min,
        wavenumber_max=args.wavenumber_max,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
