from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from classical_screen import load_dataset, regression_metrics
from pipeline_utils import POSTURE_NAMES, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCG periodicity estimator screen.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/preliminary/spectral_results.json"),
    )
    return parser.parse_args()


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, pxx = signal.periodogram(
        np.asarray(x, dtype=np.float64),
        fs=100.0,
        window="hann",
        nfft=6000,
        detrend="linear",
    )
    return f, pxx + 1e-12


def local_salience(pxx: np.ndarray) -> np.ndarray:
    log_power = np.log(pxx)
    background = ndimage.median_filter(log_power, size=101, mode="nearest")
    return log_power - background


def at_frequency(f: np.ndarray, value: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.interp(target, f, value, left=-20.0, right=-20.0)


def estimate_periodicity(window: np.ndarray) -> dict[str, float]:
    f, pxx = spectrum(window[2])
    salience = local_salience(pxx)
    candidates = np.arange(0.80, 2.16, 1.0 / 120.0)
    direct = candidates[np.argmax(at_frequency(f, pxx, candidates))]
    harmonic_score = (
        at_frequency(f, salience, candidates)
        + 0.70 * at_frequency(f, salience, 2.0 * candidates)
        + 0.40 * at_frequency(f, salience, 3.0 * candidates)
    )
    harmonic = candidates[np.argmax(harmonic_score)]

    x = window[2].astype(np.float64)
    x -= x.mean()
    ac = signal.fftconvolve(x, x[::-1], mode="full")[x.size - 1 :]
    ac /= ac[0] + 1e-12
    low_lag = int(np.floor(100.0 / 2.16))
    high_lag = int(np.ceil(100.0 / 0.80))
    segment_ac = ac[low_lag : high_lag + 1]
    peaks, _ = signal.find_peaks(segment_ac, prominence=0.01)
    if peaks.size:
        lag = low_lag + peaks[np.argmax(segment_ac[peaks])]
    else:
        lag = low_lag + int(np.argmax(segment_ac))
    autocorrelation = 100.0 / lag

    # Fuse the normalized harmonic spectrum and autocorrelation curve.
    ac_frequency = 100.0 / np.arange(low_lag, high_lag + 1)
    ac_score = np.interp(candidates, ac_frequency[::-1], segment_ac[::-1])
    h_z = (harmonic_score - np.median(harmonic_score)) / (
        np.std(harmonic_score) + 1e-8
    )
    ac_z = (ac_score - np.median(ac_score)) / (np.std(ac_score) + 1e-8)
    fused = candidates[np.argmax(h_z + 0.60 * ac_z)]
    return {
        "direct_periodogram": float(direct * 60.0),
        "local_whitened_harmonic_sum": float(harmonic * 60.0),
        "autocorrelation": float(autocorrelation * 60.0),
        "harmonic_autocorrelation_fusion": float(fused * 60.0),
    }


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data.resolve())
    rows = [estimate_periodicity(window) for window in data["bcg"]]
    methods = list(rows[0])
    results: dict[str, object] = {"all": {}, "by_posture": {}}
    for method in methods:
        prediction = np.asarray([row[method] for row in rows])
        results["all"][method] = regression_metrics(data["hr_ecg"], prediction)
    for posture_id, posture_name in enumerate(POSTURE_NAMES):
        mask = data["posture"] == posture_id
        results["by_posture"][posture_name] = {}
        for method in methods:
            prediction = np.asarray([row[method] for row in rows])
            results["by_posture"][posture_name][method] = regression_metrics(
                data["hr_ecg"][mask],
                prediction[mask],
            )
    write_json(args.out.resolve(), results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
