"""Plot liver/tumor sample-level SNOM waveforms for manual valid-band review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"liver": "#2563eb", "tumor": "#dc2626"}
LABELS_CN = {"liver": "liver", "tumor": "tumor"}


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def sample_title(row: pd.Series) -> str:
    label = LABELS_CN.get(str(row["class_label"]), str(row["class_label"]))
    return f"{label} | {row['marker']} | {row['sample_id']} | n={row['n_points']}"


def plot_12_full(
    root: Path,
    table: pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    ystd: np.ndarray,
    ylabel: str,
    title: str,
    out_name: str,
    noise_start: float,
) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(16, 22), sharex=True)
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        row = table.iloc[idx]
        color = COLORS.get(str(row["class_label"]), "#111827")
        ax.plot(w, y[idx], color=color, lw=1.1)
        ax.fill_between(w, y[idx] - ystd[idx], y[idx] + ystd[idx], color=color, alpha=0.13, lw=0)
        ax.axvspan(noise_start, float(np.nanmax(w)), color="#f59e0b", alpha=0.13)
        ax.axvline(noise_start, color="#92400e", ls="--", lw=0.9)
        ax.set_title(sample_title(row), fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(float(np.nanmin(w)), float(np.nanmax(w)))
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[-2].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.suptitle(title, fontsize=15, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(root / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_12_clean(
    root: Path,
    table: pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    ystd: np.ndarray,
    ylabel: str,
    title: str,
    out_name: str,
    wmax: float,
) -> None:
    mask = w <= wmax
    w_clean = w[mask]
    fig, axes = plt.subplots(6, 2, figsize=(16, 22), sharex=True)
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        row = table.iloc[idx]
        color = COLORS.get(str(row["class_label"]), "#111827")
        yy = y[idx, mask]
        ss = ystd[idx, mask]
        ax.plot(w_clean, yy, color=color, lw=1.15)
        ax.fill_between(w_clean, yy - ss, yy + ss, color=color, alpha=0.13, lw=0)
        ax.axvspan(1000, 1250, color="#a3e635", alpha=0.12)
        ax.set_title(sample_title(row), fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(float(np.nanmin(w_clean)), float(np.nanmax(w_clean)))
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[-2].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.suptitle(title, fontsize=15, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(root / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_overlay(
    root: Path,
    table: pd.DataFrame,
    w: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    noise_start: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    labels = table["class_label"].astype(str).to_numpy()
    for class_label, color in COLORS.items():
        idx = np.where(labels == class_label)[0]
        for row_idx in idx:
            axes[0].plot(w, amp[row_idx], color=color, alpha=0.28, lw=0.9)
            axes[1].plot(w, phase[row_idx], color=color, alpha=0.22, lw=0.8)
        axes[0].plot(w, np.nanmean(amp[idx], axis=0), color=color, lw=2.2, label=f"{class_label} mean")
        axes[1].plot(w, np.nanmean(phase[idx], axis=0), color=color, lw=2.2, label=f"{class_label} mean")
    for ax_idx, ax in enumerate(axes):
        ax.axvspan(noise_start, float(np.nanmax(w)), color="#f59e0b", alpha=0.13)
        ax.axvline(noise_start, color="#92400e", ls="--", lw=0.9)
        ax.axvspan(1000, 1250, color="#a3e635", alpha=0.10)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
        if ax_idx == 0:
            ax.set_title("Liver/tumor 12 spectra overlay with class means")
    axes[0].set_ylabel("normalized O2A")
    axes[1].set_ylabel("phase O2P-bg")
    axes[1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(root / "05_class_overlay_full_690_1800.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_roughness_summary(root: Path, w: np.ndarray, amp: np.ndarray, phase: np.ndarray) -> None:
    rows = []
    for lo, hi in [(690, 1000), (1000, 1250), (1250, 1400), (1400, 1550), (1550, 1750)]:
        mask = (w >= lo) & (w < hi)
        rows.append(
            {
                "band": f"{lo}-{hi}",
                "n_wavenumbers": int(mask.sum()),
                "amp_roughness_mean": float(np.nanmean(np.nanstd(np.diff(amp[:, mask], axis=1), axis=1))),
                "amp_std_mean": float(np.nanmean(np.nanstd(amp[:, mask], axis=1))),
                "phase_roughness_mean": float(np.nanmean(np.nanstd(np.diff(phase[:, mask], axis=1), axis=1))),
            }
        )
    rough = pd.DataFrame(rows)
    rough.to_csv(root / "06_band_roughness_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        rough["band"],
        rough["amp_roughness_mean"],
        color=["#60a5fa", "#60a5fa", "#60a5fa", "#f59e0b", "#f97316"],
    )
    ax.set_ylabel("amp roughness mean")
    ax.set_title("Amplitude roughness by wavenumber band")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "06_band_roughness_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot liver/tumor sample-level waveforms.")
    parser.add_argument(
        "--sample-dir",
        default="snom_pipeline/outputs/relative_analysis_liver_tumor_matched_bg/group1_bg_normalized_matched_bg",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-start", type=float, default=1400.0)
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir)
    z = np.load(sample_dir / "sample_level_spectra.npz", allow_pickle=True)
    table = pd.read_csv(sample_dir / "sample_level_summary.csv")
    w = np.asarray(z["wavenumber"], dtype=float)
    amp = np.asarray(z["amp_mean"], dtype=float)
    amp_std = np.asarray(z["amp_std"], dtype=float)
    phase = np.asarray(z["phase_mean"], dtype=float)
    phase_std = np.asarray(z["phase_std"], dtype=float)

    plot_12_full(
        root,
        table,
        w,
        amp,
        amp_std,
        "normalized O2A",
        "12 sample amplitude spectra: 690-1800, orange marks suspected high-wavenumber noise",
        "01_12_samples_amp_full_690_1800_noise_marked.png",
        args.noise_start,
    )
    plot_12_full(
        root,
        table,
        w,
        phase,
        phase_std,
        "phase O2P-bg",
        "12 sample phase spectra: 690-1800, orange marks suspected high-wavenumber noise",
        "02_12_samples_phase_full_690_1800_noise_marked.png",
        args.noise_start,
    )
    clean_cutoff = int(args.noise_start)
    plot_12_clean(
        root,
        table,
        w,
        amp,
        amp_std,
        "normalized O2A",
        f"12 sample amplitude spectra: cleaned view 690-{clean_cutoff}, green marks 1000-1250 sensitive band",
        f"03_12_samples_amp_clean_690_{clean_cutoff}.png",
        args.noise_start,
    )
    plot_12_clean(
        root,
        table,
        w,
        phase,
        phase_std,
        "phase O2P-bg",
        f"12 sample phase spectra: cleaned view 690-{clean_cutoff}, green marks 1000-1250 sensitive band",
        f"04_12_samples_phase_clean_690_{clean_cutoff}.png",
        args.noise_start,
    )
    plot_class_overlay(root, table, w, amp, phase, args.noise_start)
    save_roughness_summary(root, w, amp, phase)

    print(f"saved to {root}")
    for path in sorted(root.glob("*")):
        print(path.name)


if __name__ == "__main__":
    main()
