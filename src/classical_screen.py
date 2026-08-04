from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal, stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from pipeline_utils import POSTURE_NAMES, spectral_rate, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical feasibility screens.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/preliminary/classical_results.json"),
    )
    return parser.parse_args()


def load_dataset(root: Path) -> dict[str, np.ndarray]:
    accum: dict[str, list[np.ndarray]] = {
        "bcg": [],
        "ecg": [],
        "ppg": [],
        "pressure_features": [],
        "pressure_map": [],
        "pressure_valid": [],
        "posture": [],
        "hr_ecg": [],
        "hr_ppg_ir": [],
        "hr_ppg_red": [],
        "quality": [],
        "groups": [],
    }
    files = sorted((root / "subjects").glob("S*.npz"))
    for group_idx, path in enumerate(files):
        with np.load(path) as z:
            n = z["posture"].shape[0]
            accum["bcg"].append(z["bcg"])
            accum["ecg"].append(z["ecg_100"])
            accum["ppg"].append(z["ppg"])
            accum["pressure_features"].append(z["pressure_features"])
            accum["pressure_map"].append(z["pressure_mean_map"])
            accum["pressure_valid"].append(z["pressure_valid"].astype(bool))
            accum["posture"].append(z["posture"])
            accum["hr_ecg"].append(z["hr_ecg"])
            accum["hr_ppg_ir"].append(z["hr_ppg_ir"])
            accum["hr_ppg_red"].append(z["hr_ppg_red"])
            accum["quality"].append(z["quality"])
            accum["groups"].append(np.full(n, group_idx, dtype=np.int16))
    return {key: np.concatenate(value, axis=0) for key, value in accum.items()}


def band_power_features(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    f, pxx = signal.welch(x, fs=fs, nperseg=1024, noverlap=512)
    pxx = pxx + 1e-12
    bands = [
        (0.08, 0.4),
        (0.4, 0.8),
        (0.8, 1.2),
        (1.2, 1.6),
        (1.6, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
        (3.0, 4.0),
        (4.0, 8.0),
        (8.0, 15.0),
        (15.0, 25.0),
        (25.0, 40.0),
    ]
    total = np.trapezoid(pxx[(f >= 0.08) & (f <= 40.0)], f[(f >= 0.08) & (f <= 40.0)])
    features: list[float] = []
    for low, high in bands:
        mask = (f >= low) & (f < high)
        power = np.trapezoid(pxx[mask], f[mask]) if mask.sum() >= 2 else 0.0
        features.extend([math.log(power + 1e-12), float(power / (total + 1e-12))])
    heart_mask = (f >= 0.7) & (f <= 3.0)
    heart_psd = pxx[heart_mask]
    heart_freq = f[heart_mask]
    peak_index = int(np.argmax(heart_psd))
    probability = heart_psd / heart_psd.sum()
    entropy = -float(np.sum(probability * np.log(probability + 1e-12))) / math.log(
        max(probability.size, 2)
    )
    features.extend(
        [
            float(heart_freq[peak_index]),
            float(heart_psd[peak_index] / (np.median(heart_psd) + 1e-12)),
            entropy,
        ]
    )
    return np.asarray(features, dtype=np.float32)


def temporal_features(x: np.ndarray, fs: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.median(x)
    abs_x = np.abs(centered)
    down = signal.resample_poly(centered, up=1, down=2)
    ac = signal.fftconvolve(down, down[::-1], mode="full")[down.size - 1 :]
    ac /= ac[0] + 1e-12
    low_lag = int(round((fs / 2) * 0.33))
    high_lag = int(round((fs / 2) * 1.50))
    segment_ac = ac[low_lag : high_lag + 1]
    ac_index = low_lag + int(np.argmax(segment_ac))
    return np.asarray(
        [
            np.mean(centered),
            np.std(centered),
            np.median(abs_x),
            np.percentile(abs_x, 90),
            np.percentile(abs_x, 99),
            stats.skew(centered),
            stats.kurtosis(centered, fisher=False),
            np.std(np.diff(centered)),
            np.mean(np.abs(centered) >= 6.0),
            ac_index / (fs / 2),
            float(ac[ac_index]),
        ],
        dtype=np.float32,
    )


def extract_signal_features(array: np.ndarray) -> np.ndarray:
    output = []
    for window in array:
        per_channel = []
        for channel in window:
            per_channel.extend(temporal_features(channel))
            per_channel.extend(band_power_features(channel))
        output.append(per_channel)
    return np.asarray(output, dtype=np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.abs(y_true - y_pred)
    return {
        "mae_bpm": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_bpm": float(np.median(error)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "within_3_bpm": float(np.mean(error <= 3.0)),
        "within_5_bpm": float(np.mean(error <= 5.0)),
        "r2": float(r2_score(y_true, y_pred)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n": int(y_true.size),
    }


def grouped_regression(
    name: str,
    model,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[str, dict[str, float]]:
    mask = np.isfinite(y)
    if valid is not None:
        mask &= valid
    x_use, y_use, groups_use = x[mask], y[mask], groups[mask]
    splitter = GroupKFold(n_splits=5)
    pred = np.full(y_use.shape, np.nan, dtype=np.float64)
    for train, test in splitter.split(x_use, y_use, groups_use):
        fitted = clone(model)
        fitted.fit(x_use[train], y_use[train])
        pred[test] = fitted.predict(x_use[test])
    return name, regression_metrics(y_use, pred)


def direct_rate_metrics(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    target = data["hr_ecg"].astype(float)
    estimates = {
        "bcg_cardiac_psd_0.8_3Hz": np.asarray(
            [spectral_rate(x[2], 100.0, 0.8, 3.0) for x in data["bcg"]]
        ),
        "bcg_high_envelope_psd_0.8_3Hz": np.asarray(
            [spectral_rate(x[3], 100.0, 0.8, 3.0) for x in data["bcg"]]
        ),
        "ppg_ir_peak_detector": data["hr_ppg_ir"].astype(float),
        "ppg_red_peak_detector": data["hr_ppg_red"].astype(float),
        "ppg_mean_peak_detector": np.nanmean(
            np.stack([data["hr_ppg_ir"], data["hr_ppg_red"]]), axis=0
        ),
    }
    result: dict[str, dict[str, float]] = {}
    for name, prediction in estimates.items():
        mask = np.isfinite(target) & np.isfinite(prediction)
        result[name] = regression_metrics(target[mask], prediction[mask])
    return result


def posture_experiment(data: dict[str, np.ndarray]) -> dict[str, object]:
    valid = data["pressure_valid"]
    x = data["pressure_map"][valid].reshape(valid.sum(), -1)
    y = data["posture"][valid]
    groups = data["groups"][valid]
    splitter = GroupKFold(n_splits=5)
    pred = np.full(y.shape, -1, dtype=int)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=40, whiten=True, random_state=42),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    fold_scores = []
    for train, test in splitter.split(x, y, groups):
        fitted = clone(model)
        fitted.fit(x[train], y[train])
        pred[test] = fitted.predict(x[test])
        fold_scores.append(balanced_accuracy_score(y[test], pred[test]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fold_balanced_accuracy": [float(value) for value in fold_scores],
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(5)).tolist(),
        "class_order": POSTURE_NAMES,
        "n": int(y.size),
    }


def waveform_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    direct_corr = []
    lag_corr = []
    nrmse = []
    for truth, pred in zip(target, prediction):
        if np.std(truth) < 1e-6 or np.std(pred) < 1e-6:
            continue
        direct_corr.append(float(np.corrcoef(truth, pred)[0, 1]))
        normalized_truth = (truth - truth.mean()) / (truth.std() + 1e-8)
        normalized_pred = (pred - pred.mean()) / (pred.std() + 1e-8)
        nrmse.append(float(np.sqrt(np.mean((truth - pred) ** 2)) / (truth.std() + 1e-8)))
        cross = signal.correlate(normalized_truth, normalized_pred, mode="full") / truth.size
        lags = signal.correlation_lags(truth.size, pred.size, mode="full")
        allowed = np.abs(lags) <= 12  # +/- 0.48 s at 25 Hz
        lag_corr.append(float(np.max(cross[allowed])))
    return {
        "median_direct_correlation": float(np.median(direct_corr)),
        "mean_direct_correlation": float(np.mean(direct_corr)),
        "median_maxlag_correlation": float(np.median(lag_corr)),
        "median_nrmse": float(np.median(nrmse)),
        "n_segments": len(direct_corr),
    }


def make_segments(array: np.ndarray, channels: list[int]) -> np.ndarray:
    selected = array[:, channels]
    downsampled = signal.resample_poly(selected, up=1, down=4, axis=-1)
    n, c, total = downsampled.shape
    segment_length = 125
    return (
        downsampled.reshape(n, c, total // segment_length, segment_length)
        .transpose(0, 2, 1, 3)
        .reshape(-1, c * segment_length)
        .astype(np.float32)
    )


def waveform_experiment(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    target = make_segments(data["ecg"][:, None, :], [0])
    segment_groups = np.repeat(data["groups"], 6)
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train, test = next(split.split(target, groups=segment_groups))
    pressure_context = np.repeat(data["pressure_features"], 6, axis=0)
    pressure_context = np.nan_to_num(
        pressure_context,
        nan=np.nanmedian(pressure_context, axis=0),
    )
    inputs = {
        "bcg_linear": make_segments(data["bcg"], [0, 2, 3]),
        "bcg_plus_pressure_linear": np.concatenate(
            [make_segments(data["bcg"], [0, 2, 3]), pressure_context], axis=1
        ),
        "ppg_linear_upper_bound": make_segments(data["ppg"], [0, 1]),
    }
    result: dict[str, dict[str, float]] = {}
    for name, x in inputs.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(x[train], target[train])
        prediction = model.predict(x[test])
        result[name] = waveform_metrics(target[test], prediction)
    result["split"] = {
        "train_subjects": int(np.unique(segment_groups[train]).size),
        "test_subjects": int(np.unique(segment_groups[test]).size),
        "test_subject_indices": [int(v) for v in np.unique(segment_groups[test])],
    }
    return result


def main() -> None:
    args = parse_args()
    root = args.data.resolve()
    data = load_dataset(root)
    print(f"Loaded {data['posture'].size} windows from {np.unique(data['groups']).size} subjects")
    print("Extracting BCG and PPG features...")
    bcg_features = extract_signal_features(data["bcg"][:, [0, 1, 2, 3]])
    ppg_features = extract_signal_features(data["ppg"])
    posture_onehot = np.eye(5, dtype=np.float32)[data["posture"]]
    pressure_features = data["pressure_features"]

    ridge = make_pipeline(
        SimpleImputer(),
        RobustScaler(quantile_range=(10.0, 90.0)),
        FunctionTransformer(lambda value: np.clip(value, -10.0, 10.0)),
        Ridge(alpha=10.0),
    )
    trees = make_pipeline(
        SimpleImputer(),
        ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=4,
            max_features=0.75,
            random_state=42,
            n_jobs=-1,
        ),
    )
    y = data["hr_ecg"].astype(float)
    groups = data["groups"]
    regression_runs = [
        grouped_regression("bcg_ridge", ridge, bcg_features, y, groups),
        grouped_regression("bcg_extra_trees", trees, bcg_features, y, groups),
        grouped_regression(
            "bcg_plus_posture_extra_trees",
            trees,
            np.concatenate([bcg_features, posture_onehot], axis=1),
            y,
            groups,
        ),
        grouped_regression(
            "bcg_plus_pressure_extra_trees",
            trees,
            np.concatenate([bcg_features, pressure_features], axis=1),
            y,
            groups,
            valid=data["pressure_valid"],
        ),
        grouped_regression(
            "pressure_only_extra_trees",
            trees,
            pressure_features,
            y,
            groups,
            valid=data["pressure_valid"],
        ),
        grouped_regression("ppg_extra_trees", trees, ppg_features, y, groups),
    ]
    results = {
        "protocol": {
            "heart_rate": "5-fold subject-disjoint GroupKFold",
            "posture": "5-fold subject-disjoint GroupKFold",
            "waveform": "fixed 80/20 subject-disjoint split; 5 s at 25 Hz",
            "warning": "Screening results, not final model selection.",
        },
        "direct_heart_rate": direct_rate_metrics(data),
        "learned_heart_rate": dict(regression_runs),
        "pressure_posture": posture_experiment(data),
        "linear_waveform_mapping": waveform_experiment(data),
    }
    write_json(args.out.resolve(), results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
