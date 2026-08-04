from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import signal

from pipeline_utils import POSTURE_NAMES, natural_key, robust_scale, write_json
from pressure_processing import (
    fill_missing_region,
    map_pressure_frames,
    preprocess_pressure_sequence,
)


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "raw": "#8C8C8C",
    "blue": "#2F5D8C",
    "teal": "#2A9D8F",
    "orange": "#E07A3F",
    "red": "#B5443C",
    "purple": "#7B5AA6",
    "gold": "#D5A021",
    "dark": "#263238",
}
POSE_COLORS = ["#355070", "#2A9D8F", "#E07A3F", "#7B5AA6", "#C94C4C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-style preprocessing audit figures.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/preprocessing_visualization"),
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)


def load_processed(root: Path) -> dict[str, np.ndarray]:
    keys = [
        "bcg",
        "ecg_500",
        "ecg_100",
        "ppg",
        "pressure_mean_map",
        "pressure_features",
        "pressure_valid",
        "rpeak_mask_100",
        "hr_ecg",
        "hr_ppg_ir",
        "hr_ppg_red",
        "quality",
        "posture",
        "window_index",
    ]
    storage: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    storage["subject_number"] = []
    files = sorted((root / "subjects").glob("S*.npz"))
    for subject_number, path in enumerate(files, start=1):
        with np.load(path) as data:
            n = data["posture"].shape[0]
            for key in keys:
                storage[key].append(data[key])
            storage["subject_number"].append(
                np.full(n, subject_number, dtype=np.int16)
            )
    return {key: np.concatenate(parts, axis=0) for key, parts in storage.items()}


def raw_subject_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()], key=natural_key)


def load_raw_window(
    subjects: list[Path],
    subject_number: int,
    window_index: int,
) -> dict[str, np.ndarray]:
    period = subjects[subject_number - 1] / "PeriodData"
    return {
        "bcg": np.load(
            period / "Breath" / f"Breath{window_index}.npy", allow_pickle=False
        ),
        "ecg": np.load(period / "ECG" / f"ECG{window_index}.npy", allow_pickle=False),
        "ppg_ir": np.load(
            period / "PPG_IR" / f"PPG_IR{window_index}.npy", allow_pickle=False
        ),
    }


def select_examples(data: dict[str, np.ndarray]) -> tuple[int, int, np.ndarray]:
    q = data["quality"].astype(float)
    columns = [0, 1, 4, 7, 8, 10]
    features = q[:, columns].copy()
    features[:, 0] = np.log10(features[:, 0] + 1e-5)
    features[:, 3] = np.log1p(features[:, 3])
    features[:, 4] = np.log10(features[:, 4] + 1e-5)
    features[:, 5] = np.log1p(np.maximum(features[:, 5], 0))
    valid = data["pressure_valid"].astype(bool) & np.all(np.isfinite(features), axis=1)
    median = np.nanmedian(features[valid], axis=0)
    mad = np.nanmedian(np.abs(features[valid] - median), axis=0)
    z = (features - median) / np.maximum(1.4826 * mad, 1e-5)

    medoid_score = np.sum(np.abs(z), axis=1)
    medoid_score[~valid] = np.inf
    representative = int(np.argmin(medoid_score))

    severity = (
        np.maximum(z[:, 0], 0)
        + np.maximum(z[:, 1], 0)
        + np.maximum(-z[:, 2], 0)
        + np.maximum(z[:, 3], 0)
        + np.maximum(z[:, 4], 0)
        + 0.5 * np.maximum(z[:, 5], 0)
    )
    candidates = np.flatnonzero(valid & np.isfinite(severity))
    target = np.percentile(severity[candidates], 95)
    challenging = int(candidates[np.argmin(np.abs(severity[candidates] - target))])
    return representative, challenging, severity


def subject_window_label(data: dict[str, np.ndarray], index: int) -> str:
    subject = int(data["subject_number"][index])
    window = int(data["window_index"][index])
    posture = POSTURE_NAMES[int(data["posture"][index])]
    return f"S{subject:03d} · window {window} · {posture}"


def plot_time_domain_examples(
    data: dict[str, np.ndarray],
    raw_dirs: list[Path],
    representative: int,
    challenging: int,
    out_dir: Path,
) -> None:
    examples = [
        ("Cohort-representative window", representative),
        ("95th-percentile artifact burden", challenging),
    ]
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(13.5, 10.8),
        sharex="col",
        constrained_layout=True,
    )
    start_100, stop_100 = 1000, 2000
    start_500, stop_500 = 5000, 10000
    t100 = np.arange(stop_100 - start_100) / 100.0
    t500 = np.arange(stop_500 - start_500) / 500.0

    for column, (heading, index) in enumerate(examples):
        raw = load_raw_window(
            raw_dirs,
            int(data["subject_number"][index]),
            int(data["window_index"][index]),
        )
        processed_bcg = data["bcg"][index]
        raw_bcg_z = robust_scale(raw["bcg"])
        raw_ecg_z = robust_scale(raw["ecg"])
        raw_ppg_z = robust_scale(raw["ppg_ir"])

        axis = axes[0, column]
        axis.plot(
            t100,
            raw_bcg_z[start_100:stop_100],
            color=COLORS["raw"],
            linewidth=0.8,
            alpha=0.70,
            label="Raw, robust-scaled",
        )
        axis.plot(
            t100,
            processed_bcg[0, start_100:stop_100],
            color=COLORS["blue"],
            linewidth=1.0,
            label="Detrended",
        )
        axis.set_title(f"{heading}\n{subject_window_label(data, index)}", pad=9)
        axis.legend(loc="upper right", frameon=False, ncol=2)

        axis = axes[1, column]
        axis.plot(
            t100,
            processed_bcg[1, start_100:stop_100],
            color=COLORS["teal"],
            linewidth=1.1,
        )

        axis = axes[2, column]
        axis.plot(
            t100,
            processed_bcg[2, start_100:stop_100],
            color=COLORS["orange"],
            linewidth=0.9,
            label="0.8–8 Hz",
        )
        axis.plot(
            t100,
            processed_bcg[3, start_100:stop_100],
            color=COLORS["purple"],
            linewidth=0.9,
            alpha=0.88,
            label="10–40 Hz envelope",
        )
        axis.legend(loc="upper right", frameon=False, ncol=2)

        axis = axes[3, column]
        axis.plot(
            t500,
            raw_ecg_z[start_500:stop_500],
            color=COLORS["raw"],
            linewidth=0.55,
            alpha=0.52,
            label="Raw",
        )
        axis.plot(
            t500,
            data["ecg_500"][index, start_500:stop_500],
            color=COLORS["red"],
            linewidth=0.75,
            label="0.5–40 Hz",
        )
        peak_100 = np.flatnonzero(data["rpeak_mask_100"][index])
        visible = peak_100[(peak_100 >= start_100) & (peak_100 < stop_100)]
        peak_500 = visible * 5
        axis.scatter(
            (visible - start_100) / 100.0,
            data["ecg_500"][index, peak_500],
            s=18,
            facecolor="white",
            edgecolor=COLORS["dark"],
            linewidth=0.8,
            zorder=4,
            label="R peaks",
        )
        axis.legend(loc="upper right", frameon=False, ncol=3)

        axis = axes[4, column]
        axis.plot(
            t100,
            raw_ppg_z[start_100:stop_100],
            color=COLORS["raw"],
            linewidth=0.7,
            alpha=0.55,
            label="Raw",
        )
        axis.plot(
            t100,
            data["ppg"][index, 0, start_100:stop_100],
            color=COLORS["purple"],
            linewidth=1.0,
            label="0.4–8 Hz",
        )
        axis.legend(loc="upper right", frameon=False, ncol=2)
        axis.set_xlabel("Time (s)")

    row_labels = [
        "Thoracic pressure\n(robust z-score)",
        "Respiratory component\n0.08–0.70 Hz",
        "Cardiac candidates\n(robust z-score)",
        "ECG\n(robust z-score)",
        "PPG-IR\n(robust z-score)",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
        for column in range(2):
            style_axis(axes[row, column])
            axes[row, column].margins(x=0)
            if row == 0:
                add_panel_label(axes[row, column], chr(ord("a") + column))
    fig.suptitle(
        "Time-domain audit: band-limited structure and residual artifacts remain traceable",
        fontsize=14,
        y=1.025,
    )
    save_figure(fig, out_dir, "figure1_time_domain_audit")


def normalized_welch(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    total_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    f, pxx = signal.welch(
        np.asarray(x, dtype=np.float64),
        fs=fs,
        nperseg=min(nperseg, len(x)),
        noverlap=min(nperseg // 2, max(0, len(x) // 2)),
    )
    total_mask = (f >= total_band[0]) & (f <= total_band[1])
    total = np.trapezoid(pxx[total_mask], f[total_mask]) + 1e-20
    normalized = pxx / total
    db = 10.0 * np.log10(normalized + 1e-20)
    return f, db, total


def band_fraction(
    x: np.ndarray,
    fs: float,
    target: tuple[float, float],
    total: tuple[float, float],
) -> float:
    f, pxx = signal.welch(
        np.asarray(x, dtype=np.float64),
        fs=fs,
        nperseg=min(len(x), 2048 if fs <= 100 else 4096),
    )
    total_mask = (f >= total[0]) & (f <= total[1])
    target_mask = (f >= target[0]) & (f <= target[1])
    denominator = np.trapezoid(pxx[total_mask], f[total_mask]) + 1e-20
    return float(np.trapezoid(pxx[target_mask], f[target_mask]) / denominator)


def collect_spectral_audit(
    data: dict[str, np.ndarray],
    raw_dirs: list[Path],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    spectra: dict[str, list[np.ndarray]] = {
        "bcg_raw": [],
        "bcg_resp": [],
        "bcg_card": [],
        "bcg_env": [],
        "ecg_raw": [],
        "ecg_filtered": [],
        "ppg_raw": [],
        "ppg_filtered": [],
    }
    concentrations: dict[str, list[float]] = {
        "resp_raw": [],
        "resp_processed": [],
        "card_raw": [],
        "card_processed": [],
        "ecg_raw": [],
        "ecg_processed": [],
        "ppg_raw": [],
        "ppg_processed": [],
    }
    frequencies: dict[str, np.ndarray] = {}

    for index in range(data["posture"].size):
        raw = load_raw_window(
            raw_dirs,
            int(data["subject_number"][index]),
            int(data["window_index"][index]),
        )
        frequency, value, _ = normalized_welch(raw["bcg"], 100.0, 2048, (0.05, 45))
        frequencies["bcg"] = frequency
        spectra["bcg_raw"].append(value)
        for channel, name in [(1, "bcg_resp"), (2, "bcg_card"), (3, "bcg_env")]:
            _, value, _ = normalized_welch(
                data["bcg"][index, channel], 100.0, 2048, (0.05, 45)
            )
            spectra[name].append(value)

        frequency, value, _ = normalized_welch(raw["ecg"], 500.0, 4096, (0.05, 100))
        frequencies["ecg"] = frequency
        spectra["ecg_raw"].append(value)
        _, value, _ = normalized_welch(
            data["ecg_500"][index], 500.0, 4096, (0.05, 100)
        )
        spectra["ecg_filtered"].append(value)

        frequency, value, _ = normalized_welch(raw["ppg_ir"], 100.0, 2048, (0.05, 45))
        frequencies["ppg"] = frequency
        spectra["ppg_raw"].append(value)
        _, value, _ = normalized_welch(
            data["ppg"][index, 0], 100.0, 2048, (0.05, 45)
        )
        spectra["ppg_filtered"].append(value)

        concentrations["resp_raw"].append(
            band_fraction(raw["bcg"], 100.0, (0.08, 0.70), (0.05, 45))
        )
        concentrations["resp_processed"].append(
            band_fraction(data["bcg"][index, 1], 100.0, (0.08, 0.70), (0.05, 45))
        )
        concentrations["card_raw"].append(
            band_fraction(raw["bcg"], 100.0, (0.80, 8.0), (0.05, 45))
        )
        concentrations["card_processed"].append(
            band_fraction(data["bcg"][index, 2], 100.0, (0.80, 8.0), (0.05, 45))
        )
        concentrations["ecg_raw"].append(
            band_fraction(raw["ecg"], 500.0, (0.50, 40.0), (0.05, 100))
        )
        concentrations["ecg_processed"].append(
            band_fraction(data["ecg_500"][index], 500.0, (0.50, 40.0), (0.05, 100))
        )
        concentrations["ppg_raw"].append(
            band_fraction(raw["ppg_ir"], 100.0, (0.40, 8.0), (0.05, 45))
        )
        concentrations["ppg_processed"].append(
            band_fraction(data["ppg"][index, 0], 100.0, (0.40, 8.0), (0.05, 45))
        )

    spectrum_arrays = {
        name: (frequencies["bcg" if name.startswith("bcg") else "ecg" if name.startswith("ecg") else "ppg"], np.asarray(rows))
        for name, rows in spectra.items()
    }
    return spectrum_arrays, {
        name: np.asarray(values, dtype=float) for name, values in concentrations.items()
    }


def plot_psd_with_interval(
    axis: plt.Axes,
    frequency: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    linewidth: float = 1.35,
    interval: bool = False,
) -> None:
    median = np.nanmedian(values, axis=0)
    axis.plot(frequency, median, color=color, linewidth=linewidth, label=label)
    if interval:
        low, high = np.nanpercentile(values, [25, 75], axis=0)
        axis.fill_between(frequency, low, high, color=color, alpha=0.12, linewidth=0)


def plot_spectral_audit(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    concentrations: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)

    axis = axes[0, 0]
    for key, label, color, interval in [
        ("bcg_raw", "Raw thoracic pressure", COLORS["raw"], False),
        ("bcg_resp", "Respiratory component", COLORS["teal"], True),
        ("bcg_card", "Cardiac candidate", COLORS["orange"], True),
        ("bcg_env", "High-band envelope", COLORS["purple"], False),
    ]:
        frequency, values = spectra[key]
        mask = (frequency >= 0.05) & (frequency <= 45)
        plot_psd_with_interval(
            axis, frequency[mask], values[:, mask], label, color, interval=interval
        )
    axis.axvspan(0.08, 0.70, color=COLORS["teal"], alpha=0.08)
    axis.axvspan(0.80, 8.0, color=COLORS["orange"], alpha=0.06)
    axis.set_xscale("log")
    axis.set_xlim(0.05, 45)
    axis.set_ylim(-90, 35)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized PSD (dB/Hz)")
    axis.set_title("Thoracic-pressure decomposition")
    axis.legend(frameon=False, ncol=2, loc="lower left")
    style_axis(axis)
    add_panel_label(axis, "a")

    axis = axes[0, 1]
    for key, label, color, interval in [
        ("ecg_raw", "Raw ECG", COLORS["raw"], True),
        ("ecg_filtered", "Filtered ECG", COLORS["red"], True),
    ]:
        frequency, values = spectra[key]
        mask = (frequency >= 0.10) & (frequency <= 100)
        plot_psd_with_interval(
            axis, frequency[mask], values[:, mask], label, color, interval=interval
        )
    axis.axvspan(0.50, 40.0, color=COLORS["red"], alpha=0.06)
    axis.set_xscale("log")
    axis.set_xlim(0.10, 100)
    axis.set_ylim(-85, 15)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized PSD (dB/Hz)")
    axis.set_title("ECG: 0.5–40 Hz passband")
    axis.legend(frameon=False)
    style_axis(axis)
    add_panel_label(axis, "b")

    axis = axes[1, 0]
    for key, label, color, interval in [
        ("ppg_raw", "Raw PPG-IR", COLORS["raw"], True),
        ("ppg_filtered", "Filtered PPG-IR", COLORS["purple"], True),
    ]:
        frequency, values = spectra[key]
        mask = (frequency >= 0.05) & (frequency <= 30)
        plot_psd_with_interval(
            axis, frequency[mask], values[:, mask], label, color, interval=interval
        )
    axis.axvspan(0.40, 8.0, color=COLORS["purple"], alpha=0.07)
    axis.set_xscale("log")
    axis.set_xlim(0.05, 30)
    axis.set_ylim(-85, 25)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized PSD (dB/Hz)")
    axis.set_title("PPG-IR: 0.4–8 Hz passband")
    axis.legend(frameon=False)
    style_axis(axis)
    add_panel_label(axis, "c")

    axis = axes[1, 1]
    categories = ["Respiration", "Cardiac\ncandidate", "ECG", "PPG-IR"]
    raw_keys = ["resp_raw", "card_raw", "ecg_raw", "ppg_raw"]
    processed_keys = [
        "resp_processed",
        "card_processed",
        "ecg_processed",
        "ppg_processed",
    ]
    positions = np.arange(len(categories))
    width = 0.34
    raw_median = [np.median(concentrations[key]) for key in raw_keys]
    processed_median = [np.median(concentrations[key]) for key in processed_keys]
    raw_q = np.asarray(
        [np.percentile(concentrations[key], [25, 75]) for key in raw_keys]
    )
    processed_q = np.asarray(
        [np.percentile(concentrations[key], [25, 75]) for key in processed_keys]
    )
    axis.bar(
        positions - width / 2,
        raw_median,
        width,
        color=COLORS["raw"],
        alpha=0.75,
        label="Raw",
    )
    axis.bar(
        positions + width / 2,
        processed_median,
        width,
        color=COLORS["blue"],
        alpha=0.92,
        label="Processed",
    )
    axis.errorbar(
        positions - width / 2,
        raw_median,
        yerr=np.vstack(
            [np.asarray(raw_median) - raw_q[:, 0], raw_q[:, 1] - np.asarray(raw_median)]
        ),
        fmt="none",
        ecolor=COLORS["dark"],
        elinewidth=0.8,
        capsize=2,
    )
    axis.errorbar(
        positions + width / 2,
        processed_median,
        yerr=np.vstack(
            [
                np.asarray(processed_median) - processed_q[:, 0],
                processed_q[:, 1] - np.asarray(processed_median),
            ]
        ),
        fmt="none",
        ecolor=COLORS["dark"],
        elinewidth=0.8,
        capsize=2,
    )
    axis.set_xticks(positions, categories)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Power fraction inside target band")
    axis.set_title("Spectral concentration (median and IQR)")
    axis.legend(frameon=False)
    style_axis(axis)
    add_panel_label(axis, "d")

    fig.suptitle(
        "Frequency-domain audit across all 752 windows",
        fontsize=14,
        y=1.025,
    )
    save_figure(fig, out_dir, "figure2_frequency_domain_audit")


def find_ppg_peaks(x: np.ndarray, fs: int = 100) -> np.ndarray:
    best_score = -np.inf
    best_peaks = np.empty(0, dtype=int)
    for sign_value in (1.0, -1.0):
        y = sign_value * np.asarray(x, dtype=float)
        mad = np.median(np.abs(y - np.median(y))) + 1e-8
        peaks, properties = signal.find_peaks(
            y,
            distance=int(round(0.35 * fs)),
            prominence=0.35 * mad,
        )
        if peaks.size < 2:
            continue
        rr = np.diff(peaks) / fs
        score = float(np.median(properties["prominences"]) / mad) / (
            1.0 + 3.0 * np.std(rr) / (np.mean(rr) + 1e-8)
        )
        if score > best_score:
            best_score = score
            best_peaks = peaks
    return best_peaks


def plot_event_and_ppg_validation(
    data: dict[str, np.ndarray],
    representative: int,
    out_dir: Path,
) -> dict[str, float]:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    start, stop = 1000, 2000
    time = np.arange(stop - start) / 100.0
    ecg = data["ecg_100"][representative]
    ppg = data["ppg"][representative, 0]
    rpeaks = np.flatnonzero(data["rpeak_mask_100"][representative])
    ppg_peaks = find_ppg_peaks(ppg)
    visible_r = rpeaks[(rpeaks >= start) & (rpeaks < stop)]
    visible_p = ppg_peaks[(ppg_peaks >= start) & (ppg_peaks < stop)]

    axis = axes[0, 0]
    ecg_segment = ecg[start:stop]
    ppg_segment = ppg[start:stop]
    ecg_plot = (ecg_segment - np.median(ecg_segment)) / (
        np.percentile(ecg_segment, 95) - np.percentile(ecg_segment, 5) + 1e-8
    )
    ppg_plot = (ppg_segment - np.median(ppg_segment)) / (
        np.percentile(ppg_segment, 95) - np.percentile(ppg_segment, 5) + 1e-8
    )
    axis.plot(time, ecg_plot + 1.25, color=COLORS["red"], linewidth=0.8)
    axis.plot(time, ppg_plot - 0.35, color=COLORS["purple"], linewidth=0.95)
    axis.scatter(
        (visible_r - start) / 100.0,
        ecg_plot[visible_r - start] + 1.25,
        s=23,
        facecolor="white",
        edgecolor=COLORS["red"],
        linewidth=0.9,
        zorder=4,
    )
    axis.scatter(
        (visible_p - start) / 100.0,
        ppg_plot[visible_p - start] - 0.35,
        s=20,
        marker="v",
        facecolor="white",
        edgecolor=COLORS["purple"],
        linewidth=0.9,
        zorder=4,
    )
    axis.text(0.01, 0.88, "ECG + R peaks", color=COLORS["red"], transform=axis.transAxes)
    axis.text(0.01, 0.10, "PPG-IR + pulse peaks", color=COLORS["purple"], transform=axis.transAxes)
    axis.set_xlim(0, 10)
    axis.set_xlabel("Time (s)")
    axis.set_yticks([])
    axis.set_title("Event preservation after filtering")
    style_axis(axis, grid=False)
    add_panel_label(axis, "a")

    ecg_hr = data["hr_ecg"].astype(float)
    ppg_hr = np.nanmean(
        np.stack([data["hr_ppg_ir"], data["hr_ppg_red"]]).astype(float), axis=0
    )
    valid = np.isfinite(ecg_hr) & np.isfinite(ppg_hr)
    axis = axes[0, 1]
    for posture_id, posture_name in enumerate(POSTURE_NAMES):
        mask = valid & (data["posture"] == posture_id)
        axis.scatter(
            ecg_hr[mask],
            ppg_hr[mask],
            s=22,
            alpha=0.58,
            color=POSE_COLORS[posture_id],
            edgecolors="none",
            label=posture_name,
        )
    limits = [45, 130]
    axis.plot(limits, limits, color=COLORS["dark"], linewidth=1.0, linestyle="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("ECG-derived HR (bpm)")
    axis.set_ylabel("PPG-derived HR (bpm)")
    correlation = float(np.corrcoef(ecg_hr[valid], ppg_hr[valid])[0, 1])
    mae = float(np.mean(np.abs(ppg_hr[valid] - ecg_hr[valid])))
    axis.text(
        0.04,
        0.95,
        f"r = {correlation:.2f}\nMAE = {mae:.2f} bpm",
        transform=axis.transAxes,
        va="top",
        ha="left",
    )
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_title("Cross-modal heart-rate agreement")
    style_axis(axis)
    add_panel_label(axis, "b")

    difference = ppg_hr[valid] - ecg_hr[valid]
    average = (ppg_hr[valid] + ecg_hr[valid]) / 2.0
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
    axis = axes[1, 0]
    axis.scatter(
        average,
        difference,
        s=20,
        color=COLORS["blue"],
        alpha=0.48,
        edgecolors="none",
    )
    axis.axhline(bias, color=COLORS["dark"], linewidth=1.1)
    axis.axhline(lower, color=COLORS["red"], linewidth=0.9, linestyle="--")
    axis.axhline(upper, color=COLORS["red"], linewidth=0.9, linestyle="--")
    axis.text(129, bias + 0.7, f"Bias {bias:.2f}", ha="right", va="bottom")
    axis.text(129, upper + 0.7, f"+1.96 SD {upper:.1f}", ha="right", va="bottom")
    axis.text(129, lower - 0.7, f"−1.96 SD {lower:.1f}", ha="right", va="top")
    axis.set_xlim(45, 130)
    axis.set_xlabel("Mean HR (bpm)")
    axis.set_ylabel("PPG − ECG HR (bpm)")
    axis.set_title("Bland–Altman analysis")
    style_axis(axis)
    add_panel_label(axis, "c")

    axis = axes[1, 1]
    errors = np.abs(ppg_hr - ecg_hr)
    grouped = [
        errors[valid & (data["posture"] == posture_id)] for posture_id in range(5)
    ]
    box = axis.boxplot(
        grouped,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.2},
        whiskerprops={"color": COLORS["dark"], "linewidth": 0.8},
        capprops={"color": COLORS["dark"], "linewidth": 0.8},
        boxprops={"linewidth": 0.8, "color": COLORS["dark"]},
    )
    for patch, color in zip(box["boxes"], POSE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.set_xticks(np.arange(1, 6), POSTURE_NAMES, rotation=20, ha="right")
    axis.set_ylabel("|PPG − ECG HR| (bpm)")
    axis.set_ylim(0, max(25, np.percentile(errors[valid], 97)))
    axis.set_title("Agreement across postures")
    style_axis(axis)
    add_panel_label(axis, "d")

    fig.suptitle(
        "ECG pseudo-label and PPG teacher audit",
        fontsize=14,
        y=1.025,
    )
    save_figure(fig, out_dir, "figure3_event_and_teacher_validation")
    return {
        "ppg_ecg_hr_correlation": correlation,
        "ppg_ecg_hr_mae_bpm": mae,
        "bland_altman_bias_bpm": bias,
        "bland_altman_lower_bpm": lower,
        "bland_altman_upper_bpm": upper,
    }


def plot_pressure_atlas(data: dict[str, np.ndarray], out_dir: Path) -> None:
    maps = data["pressure_mean_map"]
    valid = data["pressure_valid"].astype(bool)
    rows, cols = maps.shape[-2:]
    posture_maps = []
    for posture_id in range(5):
        mask = valid & (data["posture"] == posture_id)
        posture_maps.append(np.nanmean(maps[mask], axis=0))
    vmax = float(np.percentile(np.concatenate([value.ravel() for value in posture_maps]), 99))

    fig, axes = plt.subplots(1, 5, figsize=(15, 5.1))
    image = None
    for posture_id, (axis, mean_map) in enumerate(zip(axes, posture_maps)):
        image = axis.imshow(
            mean_map,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        total = mean_map.sum() + 1e-12
        center_x = float((mean_map * x_grid).sum() / total)
        center_y = float((mean_map * y_grid).sum() / total)
        axis.scatter(
            center_x,
            center_y,
            marker="+",
            s=65,
            linewidth=1.3,
            color="white",
        )
        count = int(np.sum(valid & (data["posture"] == posture_id)))
        axis.set_title(f"{POSTURE_NAMES[posture_id]}\n$n={count}$")
        axis.set_xticks([0, (cols - 1) // 2, cols - 1])
        axis.set_yticks(
            [0, (rows - 1) // 2, rows - 1] if posture_id == 0 else []
        )
        axis.set_xlabel("Sensor column")
        if posture_id == 0:
            axis.set_ylabel("Sensor row")
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + posture_id),
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.025)
    colorbar.set_label("Mean normalized pressure density")
    fig.suptitle(
        "Population pressure atlas after calibrated 1056-channel mapping to 77×32",
        fontsize=13.5,
        y=0.975,
    )
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.12, top=0.78, wspace=0.62)
    save_figure(fig, out_dir, "figure4_pressure_posture_atlas")


def load_pressure_example(
    processed_root: Path,
    data: dict[str, np.ndarray],
    index: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    subject = int(data["subject_number"][index])
    window = int(data["window_index"][index])
    subject_file = processed_root / "subjects" / f"S{subject:03d}.npz"
    with np.load(subject_file) as subject_data:
        local_index = np.flatnonzero(subject_data["window_index"] == window)
        if local_index.size != 1:
            raise ValueError(
                f"Could not uniquely resolve S{subject:03d}, window {window}."
            )
        raw = subject_data["pressure_raw"][int(local_index[0])].astype(np.float32)

    mapped_unfilled = map_pressure_frames(raw)
    mapped_filled = fill_missing_region(mapped_unfilled)
    mapped, masked, clean, thresholds = preprocess_pressure_sequence(raw)
    if not np.allclose(mapped_filled, mapped, rtol=1e-6, atol=1e-4):
        raise RuntimeError("Pressure preprocessing stages are internally inconsistent.")
    return raw, mapped_unfilled, mapped, masked, clean, thresholds


def plot_pressure_processing_stages(
    processed_root: Path,
    data: dict[str, np.ndarray],
    representative: int,
    out_dir: Path,
) -> dict[str, float | int | str]:
    raw, mapped_unfilled, mapped, masked, clean, thresholds = load_pressure_example(
        processed_root,
        data,
        representative,
    )
    total_load = np.sum(mapped, axis=(1, 2))
    frame_index = int(np.argmin(np.abs(total_load - np.median(total_load))))
    raw_frame = raw[frame_index].reshape(66, 16)
    unfilled_frame = mapped_unfilled[frame_index]
    mapped_frame = mapped[frame_index]
    masked_frame = masked[frame_index]
    clean_frame = clean[frame_index]
    foreground = masked_frame > 0

    positive_values = mapped_frame[mapped_frame > 0]
    raw_vmax = float(np.percentile(positive_values, 99.5))
    clean_limit = float(np.percentile(np.abs(clean_frame[clean_frame != 0]), 99))
    clean_limit = max(clean_limit, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 9.2))

    image = axes[0, 0].imshow(
        raw_frame,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    axes[0, 0].set_title("Raw acquisition order (66×16)")
    axes[0, 0].set_xlabel("Acquisition column")
    axes[0, 0].set_ylabel("Acquisition row")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.035).set_label(
        "Raw sensor value"
    )

    image = axes[0, 1].imshow(
        unfilled_frame,
        cmap="magma",
        vmin=0,
        vmax=raw_vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0, 1].add_patch(
        mpl.patches.Rectangle(
            (13.5, 37.5),
            4,
            6,
            fill=False,
            edgecolor="cyan",
            linewidth=1.2,
            linestyle="--",
        )
    )
    axes[0, 1].set_title("Hardware mapping + orientation flip")
    axes[0, 1].set_xlabel("Sensor column")
    axes[0, 1].set_ylabel("Sensor row")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.035).set_label(
        "Pressure value"
    )

    image = axes[0, 2].imshow(
        mapped_frame,
        cmap="magma",
        vmin=0,
        vmax=raw_vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0, 2].add_patch(
        mpl.patches.Rectangle(
            (13.5, 37.5),
            4,
            6,
            fill=False,
            edgecolor="cyan",
            linewidth=1.2,
            linestyle="--",
        )
    )
    axes[0, 2].set_title("Fixed-gap linear interpolation")
    axes[0, 2].set_xlabel("Sensor column")
    axes[0, 2].set_ylabel("Sensor row")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.035).set_label(
        "Pressure value"
    )

    image = axes[1, 0].imshow(
        masked_frame,
        cmap="magma",
        vmin=0,
        vmax=raw_vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        f"OTSU + connected components\nthreshold = {thresholds[frame_index]:.1f}"
    )
    axes[1, 0].set_xlabel("Sensor column")
    axes[1, 0].set_ylabel("Sensor row")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.035).set_label(
        "Foreground pressure"
    )

    image = axes[1, 1].imshow(
        clean_frame,
        cmap="RdBu_r",
        vmin=-clean_limit,
        vmax=clean_limit,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1, 1].set_title("Foreground-only z-score")
    axes[1, 1].set_xlabel("Sensor column")
    axes[1, 1].set_ylabel("Sensor row")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.035).set_label(
        "Standardized pressure"
    )

    time_s = np.arange(mapped.shape[0], dtype=float)
    normalized_load = total_load / max(float(np.median(total_load)), 1e-8)
    active_fraction = np.mean(masked > 0, axis=(1, 2))
    axis = axes[1, 2]
    axis.plot(
        time_s,
        normalized_load,
        color=COLORS["blue"],
        linewidth=1.4,
        marker="o",
        markersize=3,
        label="Total load / median",
    )
    axis.axvline(frame_index, color=COLORS["dark"], linewidth=0.9, linestyle="--")
    axis.set_xlabel("Frame time (s)")
    axis.set_ylabel("Relative total load")
    axis.set_title("30-s temporal stability")
    axis.margins(x=0)
    style_axis(axis)
    second_axis = axis.twinx()
    second_axis.plot(
        time_s,
        active_fraction * 100,
        color=COLORS["orange"],
        linewidth=1.2,
        marker="s",
        markersize=2.7,
        label="Active area",
    )
    second_axis.set_ylabel("Active sensor area (%)")
    second_axis.spines["top"].set_visible(False)
    handles = [
        Line2D([0], [0], color=COLORS["blue"], marker="o", markersize=3),
        Line2D([0], [0], color=COLORS["orange"], marker="s", markersize=3),
    ]
    axis.legend(
        handles,
        ["Relative total load", "Active area"],
        frameon=False,
        loc="upper right",
    )

    for panel_index, axis in enumerate(axes.ravel()):
        axis.text(
            0.025,
            0.975,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            color="white" if panel_index < 5 else COLORS["dark"],
            va="top",
            ha="left",
            zorder=10,
            bbox=(
                {
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.45,
                }
                if panel_index < 5
                else None
            ),
        )
    subject = int(data["subject_number"][representative])
    window = int(data["window_index"][representative])
    posture = POSTURE_NAMES[int(data["posture"][representative])]
    fig.suptitle(
        (
            "Pressure preprocessing audit: acquisition vector to spatial foreground\n"
            f"S{subject:03d}, window {window}, {posture}; median-load frame {frame_index}"
        ),
        fontsize=13.5,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.94,
        bottom=0.07,
        top=0.84,
        hspace=0.36,
        wspace=0.40,
    )
    save_figure(fig, out_dir, "figure6_pressure_processing_stages")

    return {
        "subject_id": f"S{subject:03d}",
        "window_index": window,
        "posture": posture,
        "frame_index": frame_index,
        "otsu_threshold_raw_units": float(thresholds[frame_index]),
        "foreground_fraction": float(np.mean(foreground)),
        "mapped_rows": int(mapped.shape[1]),
        "mapped_columns": int(mapped.shape[2]),
    }


def plot_quality_distributions(
    data: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, float]:
    quality = data["quality"].astype(float)
    panels = [
        (
            quality[:, 0] * 100.0,
            "ECG rail fraction (%)",
            5.0,
            "ECG rail < 5%",
            "below",
        ),
        (
            quality[:, 8] * 100.0,
            "BCG spike fraction (%)",
            8.0,
            "BCG spikes < 8%",
            "below",
        ),
        (
            quality[:, 4],
            "PPG IR–RED correlation",
            0.30,
            "Correlation > 0.30",
            "above",
        ),
        (
            quality[:, 7],
            "PPG HR disagreement (bpm)",
            12.0,
            "Disagreement ≤ 12 bpm",
            "below",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    rng = np.random.default_rng(42)
    for panel_index, (
        axis,
        (values, ylabel, threshold, threshold_label, direction),
    ) in enumerate(
        zip(axes.flat, panels)
    ):
        for posture_id in range(5):
            posture_values = values[
                (data["posture"] == posture_id) & np.isfinite(values)
            ]
            x = (
                np.full(posture_values.size, posture_id + 1, dtype=float)
                + rng.uniform(-0.20, 0.20, size=posture_values.size)
            )
            axis.scatter(
                x,
                posture_values,
                s=11,
                alpha=0.28,
                color=POSE_COLORS[posture_id],
                edgecolors="none",
                rasterized=True,
            )
            q25, median, q75 = np.percentile(posture_values, [25, 50, 75])
            axis.vlines(
                posture_id + 1,
                q25,
                q75,
                color=COLORS["dark"],
                linewidth=2.0,
                zorder=4,
            )
            axis.plot(
                [posture_id + 0.77, posture_id + 1.23],
                [median, median],
                color=COLORS["dark"],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=5,
            )
        axis.axhline(
            threshold,
            color=COLORS["red"],
            linewidth=1.0,
            linestyle="--",
            label=threshold_label,
        )
        finite = values[np.isfinite(values)]
        if panel_index in (0, 1, 3):
            upper = max(threshold * 1.22, float(np.percentile(finite, 99.5)))
            axis.set_ylim(0, upper)
        else:
            axis.set_ylim(-0.05, 1.05)
        pass_fraction = (
            np.mean(finite < threshold)
            if direction == "below"
            else np.mean(finite > threshold)
        )
        axis.text(
            0.98,
            0.93,
            f"Pass: {pass_fraction:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=COLORS["dark"],
        )
        axis.set_xticks(np.arange(1, 6), POSTURE_NAMES, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, loc="upper left")
        style_axis(axis)
        add_panel_label(axis, chr(ord("a") + panel_index))
    fig.suptitle(
        "Dataset-wide quality audit by posture (all 752 windows)",
        fontsize=14,
        y=1.025,
    )
    save_figure(fig, out_dir, "figure5_quality_distributions")

    good = (
        (quality[:, 0] < 0.05)
        & (quality[:, 1] < 0.35)
        & (quality[:, 2] >= 18)
        & (quality[:, 2] <= 66)
        & (quality[:, 8] < 0.08)
    )
    teacher_good = good & (quality[:, 4] > 0.30) & (quality[:, 7] <= 12.0)
    return {
        "signal_quality_gate_pass_fraction": float(np.mean(good)),
        "ppg_teacher_gate_pass_fraction": float(np.mean(teacher_good)),
        "pressure_valid_fraction": float(np.mean(data["pressure_valid"])),
    }


def concentration_summary(concentrations: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, values in concentrations.items():
        output[name] = {
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }
    return output


def main() -> None:
    args = parse_args()
    setup_style()
    raw_root = args.raw.resolve()
    processed_root = args.processed.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(processed_root)
    raw_dirs = raw_subject_dirs(raw_root)
    representative, challenging, severity = select_examples(data)

    plot_time_domain_examples(
        data, raw_dirs, representative, challenging, out_dir
    )
    spectra, concentrations = collect_spectral_audit(data, raw_dirs)
    plot_spectral_audit(spectra, concentrations, out_dir)
    agreement = plot_event_and_ppg_validation(data, representative, out_dir)
    plot_pressure_atlas(data, out_dir)
    pressure_audit = plot_pressure_processing_stages(
        processed_root,
        data,
        representative,
        out_dir,
    )
    quality_summary = plot_quality_distributions(data, out_dir)

    metrics = {
        "dataset": {
            "subjects": int(np.unique(data["subject_number"]).size),
            "windows": int(data["posture"].size),
            "window_seconds": 30,
        },
        "objective_example_selection": {
            "representative": {
                "subject_id": f"S{int(data['subject_number'][representative]):03d}",
                "window_index": int(data["window_index"][representative]),
                "posture": POSTURE_NAMES[int(data["posture"][representative])],
            },
            "artifact_burden_p95": {
                "subject_id": f"S{int(data['subject_number'][challenging]):03d}",
                "window_index": int(data["window_index"][challenging]),
                "posture": POSTURE_NAMES[int(data["posture"][challenging])],
                "severity_percentile": float(
                    np.mean(severity[np.isfinite(severity)] <= severity[challenging])
                ),
            },
        },
        "spectral_concentration": concentration_summary(concentrations),
        "event_and_teacher_agreement": agreement,
        "pressure_processing_audit": pressure_audit,
        "quality_gate": quality_summary,
        "notes": [
            "All cohort spectral summaries use all 752 windows.",
            "Representative window is a multivariate quality medoid; it was not hand-picked.",
            "Artifact example is nearest the 95th percentile of a prespecified artifact score.",
            "Pressure uses the supplied hardware mapping from 1056 channels to 77x32.",
        ],
    }
    write_json(out_dir / "visualization_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote figures and metrics to {out_dir}")


if __name__ == "__main__":
    main()
