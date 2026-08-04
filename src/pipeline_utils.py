from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal


POSTURE_TO_ID = {
    "Supine": 0,
    "Right Side": 1,
    "Left Side": 2,
    "Prone": 3,
    "Sit": 4,
}
POSTURE_NAMES = ["Supine", "Right Side", "Left Side", "Prone", "Sit"]


def natural_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def robust_scale(x: np.ndarray, clip: float = 12.0) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    median = np.median(x64)
    mad = np.median(np.abs(x64 - median))
    scale = max(1.4826 * mad, float(np.std(x64)) * 0.05, 1e-8)
    return np.clip((x64 - median) / scale, -clip, clip).astype(np.float32)


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def preprocess_bcg(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    detrended = signal.detrend(np.asarray(x, dtype=np.float64), type="linear")
    respiration = bandpass(detrended, fs, 0.08, 0.70, order=3)
    cardiac_low = bandpass(detrended, fs, 0.80, 8.00, order=4)
    high_band = bandpass(detrended, fs, 10.0, 40.0, order=4)
    high_envelope = lowpass(np.abs(high_band), fs, 5.0, order=3)
    return np.stack(
        [
            robust_scale(detrended),
            robust_scale(respiration),
            robust_scale(cardiac_low),
            robust_scale(high_envelope),
        ],
        axis=0,
    )


def preprocess_ecg(x: np.ndarray, fs: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    filtered = bandpass(x, fs, 0.50, 40.0, order=4)
    ecg_500 = robust_scale(filtered)
    ecg_100 = signal.resample_poly(ecg_500, up=1, down=5).astype(np.float32)
    return ecg_500, ecg_100


def preprocess_ppg(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    return robust_scale(bandpass(x, fs, 0.40, 8.00, order=4))


def detect_rpeaks(ecg_raw: np.ndarray, fs: int = 500) -> tuple[np.ndarray, dict[str, float]]:
    ecg_raw = np.asarray(ecg_raw, dtype=np.float64)
    y = bandpass(ecg_raw, fs, 5.0, 25.0, order=3)
    energy = y * y
    width = max(3, int(round(0.12 * fs)))
    integrated = np.convolve(energy, np.ones(width) / width, mode="same")
    med = np.median(integrated)
    mad = np.median(np.abs(integrated - med)) + 1e-12
    candidates, props = signal.find_peaks(
        integrated,
        distance=int(round(0.30 * fs)),
        prominence=2.5 * mad,
    )

    refined: list[int] = []
    strengths: list[float] = []
    radius = int(round(0.06 * fs))
    for peak in candidates:
        left = max(0, peak - radius)
        right = min(y.size, peak + radius + 1)
        local = left + int(np.argmax(np.abs(y[left:right])))
        strength = float(abs(y[local]))
        if refined and local - refined[-1] < int(round(0.25 * fs)):
            if strength > strengths[-1]:
                refined[-1] = local
                strengths[-1] = strength
        else:
            refined.append(local)
            strengths.append(strength)

    peaks = np.asarray(refined, dtype=np.int32)
    rr = np.diff(peaks) / fs
    rr_cv = float(np.std(rr) / np.mean(rr)) if rr.size >= 2 and np.mean(rr) > 0 else math.nan
    prom = np.asarray(props.get("prominences", []), dtype=float)
    return peaks, {
        "peak_count": float(peaks.size),
        "hr_bpm": float(peaks.size * 60.0 / (ecg_raw.size / fs)),
        "rr_cv": rr_cv,
        "integrated_prominence_median": float(np.median(prom / mad)) if prom.size else 0.0,
    }


def detect_ppg_rate(ppg: np.ndarray, fs: int = 100) -> tuple[float, float]:
    x = np.asarray(ppg, dtype=np.float64)
    best: tuple[float, float] | None = None
    for sign in (1.0, -1.0):
        y = sign * x
        mad = np.median(np.abs(y - np.median(y))) + 1e-8
        peaks, props = signal.find_peaks(
            y,
            distance=int(round(0.35 * fs)),
            prominence=0.35 * mad,
        )
        if not 15 <= peaks.size <= 75:
            continue
        rr = np.diff(peaks) / fs
        if rr.size < 2:
            continue
        rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-8))
        prominence = float(np.median(props["prominences"]) / mad)
        score = prominence / (1.0 + 3.0 * rr_cv)
        candidate = (score, float(peaks.size * 2.0))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return math.nan, 0.0
    return best[1], best[0]


def spectral_rate(x: np.ndarray, fs: float, low: float, high: float) -> float:
    f, pxx = signal.welch(
        np.asarray(x, dtype=np.float64),
        fs=fs,
        nperseg=min(len(x), int(fs * 20)),
        noverlap=min(len(x) // 2, int(fs * 10)),
    )
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return math.nan
    return float(f[mask][np.argmax(pxx[mask])] * 60.0)


def pressure_features(
    pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.ndim != 3:
        raise ValueError(
            f"Expected pressure maps (frames, rows, cols), got {pressure.shape}"
        )
    maps = pressure
    _frames, rows, cols = maps.shape
    mean_map = maps.mean(axis=0)
    total = mean_map.sum() + 1e-8
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    x_norm = (x_grid / max(cols - 1, 1)) * 2.0 - 1.0
    y_norm = (y_grid / max(rows - 1, 1)) * 2.0 - 1.0
    cx = float((mean_map * x_norm).sum() / total)
    cy = float((mean_map * y_norm).sum() / total)
    sx = float(np.sqrt((mean_map * (x_norm - cx) ** 2).sum() / total))
    sy = float(np.sqrt((mean_map * (y_norm - cy) ** 2).sum() / total))
    left = mean_map[:, : cols // 2].sum()
    right = mean_map[:, (cols + 1) // 2 :].sum()
    top = mean_map[: rows // 2].sum()
    bottom = mean_map[(rows + 1) // 2 :].sum()
    frame_load = maps.sum(axis=(1, 2))
    frame_delta = float(np.mean(np.abs(np.diff(maps, axis=0))))
    active_fraction = float(np.mean(mean_map > 0.0))
    features = np.asarray(
        [
            float(mean_map.mean()),
            float(mean_map.std()),
            float(np.max(mean_map)),
            float(np.percentile(mean_map, 95)),
            float(np.mean(mean_map == 0.0)),
            active_fraction,
            cx,
            cy,
            sx,
            sy,
            float((left - right) / total),
            float((top - bottom) / total),
            float(np.mean(frame_load)),
            float(np.std(frame_load) / (np.mean(frame_load) + 1e-8)),
            frame_delta,
            float(frame_delta / (mean_map.mean() + 1e-8)),
        ],
        dtype=np.float32,
    )
    normalized_map = (mean_map / (mean_map.sum() + 1e-8)).astype(np.float32)
    return features, normalized_map


PRESSURE_FEATURE_NAMES = [
    "mean_pressure",
    "std_pressure",
    "max_mean_map",
    "p95_mean_map",
    "zero_fraction",
    "active_fraction",
    "center_x",
    "center_y",
    "spread_x",
    "spread_y",
    "left_right_asymmetry",
    "top_bottom_asymmetry",
    "mean_frame_load",
    "frame_load_cv",
    "mean_abs_frame_delta",
    "relative_frame_delta",
]


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def nanpercentiles(values: Iterable[float], percentiles: list[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    if not np.any(np.isfinite(array)):
        return {str(p): None for p in percentiles}
    result = np.nanpercentile(array, percentiles)
    return {str(p): finite_or_none(v) for p, v in zip(percentiles, result)}
