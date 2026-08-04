from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline

from classical_screen import (
    extract_signal_features,
    load_dataset,
    regression_metrics,
)
from deep_screen import event_metrics, prepare_data, split_indices
from pipeline_utils import POSTURE_NAMES, spectral_rate, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subgroup and leakage screens.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/preliminary/subgroup_results.json"),
    )
    return parser.parse_args()


def spectral_target_percentile(x: np.ndarray, hr: float) -> float:
    f, pxx = signal.welch(x, fs=100.0, nperseg=1024, noverlap=512)
    mask = (f >= 0.8) & (f <= 3.0)
    band_f, band_p = f[mask], pxx[mask]
    target = hr / 60.0
    nearest = int(np.argmin(np.abs(band_f - target)))
    return float(np.mean(band_p <= band_p[nearest]))


def direct_by_posture(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    prediction = np.asarray(
        [spectral_rate(window[2], 100.0, 0.8, 3.0) for window in data["bcg"]]
    )
    percentile = np.asarray(
        [
            spectral_target_percentile(window[2], hr)
            for window, hr in zip(data["bcg"], data["hr_ecg"])
        ]
    )
    output = {}
    for posture_id, name in enumerate(POSTURE_NAMES):
        mask = data["posture"] == posture_id
        metrics = regression_metrics(data["hr_ecg"][mask], prediction[mask])
        metrics["true_hr_psd_percentile_median"] = float(np.median(percentile[mask]))
        metrics["true_hr_psd_top20_fraction"] = float(np.mean(percentile[mask] >= 0.80))
        output[name] = metrics
    output["all"] = regression_metrics(data["hr_ecg"], prediction)
    output["all"]["true_hr_psd_percentile_median"] = float(np.median(percentile))
    output["all"]["true_hr_psd_top20_fraction"] = float(np.mean(percentile >= 0.80))
    return output


def leakage_check(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    features = extract_signal_features(data["bcg"])
    target = data["hr_ecg"].astype(float)
    groups = data["groups"]
    model = make_pipeline(
        SimpleImputer(),
        ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=4,
            max_features=0.75,
            random_state=42,
            n_jobs=-1,
        ),
    )
    schemes = {
        "subject_disjoint_groupkfold": GroupKFold(n_splits=5).split(
            features, target, groups
        ),
        "random_window_kfold_leaky": KFold(
            n_splits=5, shuffle=True, random_state=42
        ).split(features),
    }
    output = {}
    for name, splits in schemes.items():
        prediction = np.full(target.shape, np.nan)
        for train, test in splits:
            fitted = clone(model)
            fitted.fit(features[train], target[train])
            prediction[test] = fitted.predict(features[test])
        output[name] = regression_metrics(target, prediction)

    group_splits = GroupKFold(n_splits=5)
    dummy = np.full(target.shape, np.nan)
    for train, test in group_splits.split(features, target, groups):
        dummy[test] = np.median(target[train])
    output["train_median_dummy_groupkfold"] = regression_metrics(target, dummy)
    return output


def periodic_event_baseline(data_root: Path) -> dict[str, float]:
    windows = load_dataset(data_root)
    deep_windows = {
        "bcg": windows["bcg"],
        "ecg": windows["ecg"],
        "ppg": windows["ppg"],
        "rpeak": np.concatenate(
            [
                np.load(path)["rpeak_mask_100"]
                for path in sorted((data_root / "subjects").glob("S*.npz"))
            ],
            axis=0,
        ),
        "pressure_features": windows["pressure_features"],
        "quality": windows["quality"],
        "groups": windows["groups"],
    }
    segmented = prepare_data(deep_windows)
    train, _val, test = split_indices(segmented["groups"])
    beats_per_minute = float(np.median(windows["hr_ecg"][np.unique(train // 3)]))
    period = 6000.0 / beats_per_minute
    probability = np.zeros_like(segmented["event_binary"][test])
    for row in probability[:, 0]:
        peaks = np.arange(period / 2.0, row.size, period).astype(int)
        row[peaks] = 1.0
    metrics = event_metrics(segmented["event_binary"][test], probability, 0.5)
    metrics["constant_rate_bpm"] = beats_per_minute
    return metrics


def main() -> None:
    args = parse_args()
    root = args.data.resolve()
    data = load_dataset(root)
    results = {
        "bcg_direct_hr_by_posture": direct_by_posture(data),
        "bcg_leakage_check": leakage_check(data),
        "periodic_event_no_signal_baseline": periodic_event_baseline(root),
    }
    write_json(args.out.resolve(), results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
