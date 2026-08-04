from __future__ import annotations

import argparse
import pickle
import pickletools
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from pipeline_utils import (
    POSTURE_TO_ID,
    PRESSURE_FEATURE_NAMES,
    detect_ppg_rate,
    detect_rpeaks,
    finite_or_none,
    nanpercentiles,
    natural_key,
    preprocess_bcg,
    preprocess_ecg,
    preprocess_ppg,
    pressure_features,
    spectral_rate,
    write_json,
)
from pressure_processing import (
    PRESSURE_COLS,
    PRESSURE_ROWS,
    RAW_SENSOR_COUNT,
    preprocess_pressure_sequence,
)


ALLOWED_PICKLE_OPCODES = {
    "PROTO",
    "FRAME",
    "EMPTY_LIST",
    "MARK",
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "APPENDS",
    "APPEND",
    "STOP",
    "MEMOIZE",
    "BINPUT",
    "BINGET",
    "LONG_BINPUT",
    "LONG_BINGET",
}


def safe_load_labels(path: Path) -> list[str]:
    data = path.read_bytes()
    opcodes = {opcode.name for opcode, _arg, _pos in pickletools.genops(data)}
    unexpected = sorted(opcodes - ALLOWED_PICKLE_OPCODES)
    if unexpected:
        raise ValueError(f"Unsafe/unexpected Label.pkl opcodes at {path}: {unexpected}")
    labels = pickle.loads(data)
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"Label.pkl is not a list[str]: {path}")
    unknown = sorted(set(labels) - set(POSTURE_TO_ID))
    if unknown:
        raise ValueError(f"Unknown posture labels at {path}: {unknown}")
    return labels


def load_numbered(folder: Path, prefix: str, index: int) -> np.ndarray | None:
    path = folder / f"{prefix}{index}.npy"
    return np.load(path, allow_pickle=False) if path.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess multimodal mattress dataset.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    subject_out = out_root / "subjects"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_root} is non-empty; pass --overwrite to replace files.")
    subject_out.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        [path for path in raw_root.iterdir() if path.is_dir()],
        key=natural_key,
    )
    if not subject_dirs:
        raise RuntimeError(f"No subject directories found in {raw_root}")

    manifest_subjects: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    global_counts: Counter[str] = Counter()

    for subject_number, raw_subject in enumerate(subject_dirs, start=1):
        subject_id = f"S{subject_number:03d}"
        period = raw_subject / "PeriodData"
        labels = safe_load_labels(period / "Label" / "Label.pkl")
        n_windows = len(labels)

        bcg = np.empty((n_windows, 4, 3000), dtype=np.float32)
        ecg_500 = np.empty((n_windows, 15000), dtype=np.float32)
        ecg_100 = np.empty((n_windows, 3000), dtype=np.float32)
        ppg = np.empty((n_windows, 2, 3000), dtype=np.float32)
        pressure_raw = np.full(
            (n_windows, 30, RAW_SENSOR_COUNT),
            np.nan,
            dtype=np.float32,
        )
        pressure_mapped = np.full(
            (n_windows, 30, PRESSURE_ROWS, PRESSURE_COLS),
            np.nan,
            dtype=np.float32,
        )
        pressure_clean = np.full_like(pressure_mapped, np.nan)
        pressure_mean_map = np.full(
            (n_windows, PRESSURE_ROWS, PRESSURE_COLS),
            np.nan,
            dtype=np.float32,
        )
        pressure_otsu_thresholds = np.full((n_windows, 30), np.nan, dtype=np.float32)
        pressure_feature_matrix = np.full(
            (n_windows, len(PRESSURE_FEATURE_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        rpeak_mask_100 = np.zeros((n_windows, 3000), dtype=np.uint8)
        posture = np.asarray([POSTURE_TO_ID[label] for label in labels], dtype=np.int8)
        pressure_valid = np.zeros(n_windows, dtype=np.uint8)
        hr_ecg = np.full(n_windows, np.nan, dtype=np.float32)
        hr_ppg_ir = np.full(n_windows, np.nan, dtype=np.float32)
        hr_ppg_red = np.full(n_windows, np.nan, dtype=np.float32)
        rr_bcg = np.full(n_windows, np.nan, dtype=np.float32)
        quality = np.full((n_windows, 12), np.nan, dtype=np.float32)

        for zero_idx in range(n_windows):
            file_idx = zero_idx + 1
            raw_bcg = load_numbered(period / "Breath", "Breath", file_idx)
            raw_ecg = load_numbered(period / "ECG", "ECG", file_idx)
            raw_ir = load_numbered(period / "PPG_IR", "PPG_IR", file_idx)
            raw_red = load_numbered(period / "PPG_RED", "PPG_RED", file_idx)
            raw_pressure = load_numbered(period / "Press", "Press", file_idx)
            required = {
                "Breath": raw_bcg,
                "ECG": raw_ecg,
                "PPG_IR": raw_ir,
                "PPG_RED": raw_red,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise FileNotFoundError(
                    f"{subject_id} window {file_idx} missing required modalities: {missing}"
                )
            assert raw_bcg is not None and raw_ecg is not None
            assert raw_ir is not None and raw_red is not None
            expected_shapes = {
                "Breath": (raw_bcg.shape, (3000,)),
                "ECG": (raw_ecg.shape, (15000,)),
                "PPG_IR": (raw_ir.shape, (3000,)),
                "PPG_RED": (raw_red.shape, (3000,)),
            }
            bad_shapes = {
                name: actual
                for name, (actual, expected) in expected_shapes.items()
                if actual != expected
            }
            if bad_shapes:
                raise ValueError(f"{subject_id} window {file_idx} bad shapes: {bad_shapes}")

            bcg[zero_idx] = preprocess_bcg(raw_bcg)
            ecg_500[zero_idx], ecg_100[zero_idx] = preprocess_ecg(raw_ecg)
            ppg[zero_idx, 0] = preprocess_ppg(raw_ir)
            ppg[zero_idx, 1] = preprocess_ppg(raw_red)

            peaks, ecg_metrics = detect_rpeaks(raw_ecg)
            peak_100 = np.unique(np.clip(np.rint(peaks / 5.0).astype(int), 0, 2999))
            rpeak_mask_100[zero_idx, peak_100] = 1
            hr_ecg[zero_idx] = ecg_metrics["hr_bpm"]
            hr_ppg_ir[zero_idx], ir_score = detect_ppg_rate(ppg[zero_idx, 0])
            hr_ppg_red[zero_idx], red_score = detect_ppg_rate(ppg[zero_idx, 1])
            rr_bcg[zero_idx] = spectral_rate(bcg[zero_idx, 1], 100.0, 0.08, 0.60)

            ir_red_corr = float(np.corrcoef(ppg[zero_idx, 0], ppg[zero_idx, 1])[0, 1])
            rail_fraction = float(np.mean((raw_ecg <= 10) | (raw_ecg >= 4085)))
            bcg_spike_fraction = float(np.mean(np.abs(bcg[zero_idx, 0]) >= 6.0))
            bcg_kurtosis = float(stats.kurtosis(bcg[zero_idx, 0], fisher=False))

            pressure_delta_relative = np.nan
            if raw_pressure is not None and raw_pressure.shape == (30, RAW_SENSOR_COUNT):
                mapped, masked, clean, thresholds = preprocess_pressure_sequence(
                    raw_pressure,
                    min_object_size=10,
                    use_zscore=True,
                )
                pressure_raw[zero_idx] = raw_pressure.astype(np.float32)
                pressure_mapped[zero_idx] = mapped
                pressure_clean[zero_idx] = clean
                pressure_otsu_thresholds[zero_idx] = thresholds
                p_features, p_map = pressure_features(masked)
                pressure_feature_matrix[zero_idx] = p_features
                pressure_mean_map[zero_idx] = p_map
                pressure_valid[zero_idx] = 1
                pressure_delta_relative = float(p_features[-1])
            elif raw_pressure is not None:
                global_counts["malformed_pressure_files"] += 1

            quality[zero_idx] = np.asarray(
                [
                    rail_fraction,
                    ecg_metrics["rr_cv"],
                    ecg_metrics["peak_count"],
                    ecg_metrics["integrated_prominence_median"],
                    ir_red_corr,
                    ir_score,
                    red_score,
                    abs(float(hr_ppg_ir[zero_idx] - hr_ppg_red[zero_idx])),
                    bcg_spike_fraction,
                    bcg_kurtosis,
                    pressure_delta_relative,
                    float(pressure_valid[zero_idx]),
                ],
                dtype=np.float32,
            )

            audit_rows.append(
                {
                    "subject_id": subject_id,
                    "window_index": file_idx,
                    "posture": labels[zero_idx],
                    "pressure_valid": bool(pressure_valid[zero_idx]),
                    "ecg_hr_bpm": finite_or_none(hr_ecg[zero_idx]),
                    "ppg_ir_hr_bpm": finite_or_none(hr_ppg_ir[zero_idx]),
                    "ppg_red_hr_bpm": finite_or_none(hr_ppg_red[zero_idx]),
                    "bcg_resp_rate_bpm": finite_or_none(rr_bcg[zero_idx]),
                    "ecg_rail_fraction": rail_fraction,
                    "ecg_rr_cv": finite_or_none(ecg_metrics["rr_cv"]),
                    "ppg_ir_red_correlation": ir_red_corr,
                    "bcg_spike_fraction": bcg_spike_fraction,
                    "pressure_relative_frame_delta": finite_or_none(pressure_delta_relative),
                }
            )
            global_counts[f"posture_{labels[zero_idx]}"] += 1
            global_counts["windows"] += 1
            global_counts["pressure_valid"] += int(pressure_valid[zero_idx])

        destination = subject_out / f"{subject_id}.npz"
        np.savez_compressed(
            destination,
            subject_id=np.asarray(subject_id),
            window_index=np.arange(1, n_windows + 1, dtype=np.int16),
            posture=posture,
            pressure_valid=pressure_valid,
            bcg=bcg,
            ecg_500=ecg_500,
            ecg_100=ecg_100,
            ppg=ppg,
            pressure_raw=pressure_raw,
            pressure_mapped=pressure_mapped,
            pressure_clean=pressure_clean,
            pressure_mean_map=pressure_mean_map,
            pressure_features=pressure_feature_matrix,
            pressure_otsu_thresholds=pressure_otsu_thresholds,
            rpeak_mask_100=rpeak_mask_100,
            hr_ecg=hr_ecg,
            hr_ppg_ir=hr_ppg_ir,
            hr_ppg_red=hr_ppg_red,
            rr_bcg=rr_bcg,
            quality=quality,
        )
        manifest_subjects.append(
            {
                "subject_id": subject_id,
                "file": f"subjects/{subject_id}.npz",
                "window_count": n_windows,
                "posture_counts": dict(Counter(labels)),
                "pressure_valid_windows": int(pressure_valid.sum()),
            }
        )
        print(
            f"{subject_id}: {n_windows} windows, "
            f"{int(pressure_valid.sum())} valid pressure windows"
        )

    quality_names = [
        "ecg_rail_fraction",
        "ecg_rr_cv",
        "ecg_peak_count",
        "ecg_integrated_prominence_median",
        "ppg_ir_red_correlation",
        "ppg_ir_peak_score",
        "ppg_red_peak_score",
        "ppg_hr_disagreement_bpm",
        "bcg_spike_fraction",
        "bcg_kurtosis",
        "pressure_relative_frame_delta",
        "pressure_valid",
    ]
    manifest = {
        "version": "v1.1-pressure-layout",
        "source": "data/raw/*/PeriodData only",
        "privacy": "Processed subject IDs are anonymized and raw directory names are not exported.",
        "window_seconds": 30,
        "sampling_rates_hz": {
            "bcg": 100,
            "ecg_500": 500,
            "ecg_100": 100,
            "ppg": 100,
            "pressure": 1,
        },
        "shapes_per_window": {
            "bcg": [4, 3000],
            "ecg_500": [15000],
            "ecg_100": [3000],
            "ppg": [2, 3000],
            "pressure_raw": [30, RAW_SENSOR_COUNT],
            "pressure_mapped": [30, PRESSURE_ROWS, PRESSURE_COLS],
            "pressure_clean": [30, PRESSURE_ROWS, PRESSURE_COLS],
            "pressure_mean_map": [PRESSURE_ROWS, PRESSURE_COLS],
        },
        "bcg_channel_names": [
            "detrended_robust_z",
            "respiration_0.08_0.70Hz_robust_z",
            "cardiac_0.80_8.00Hz_robust_z",
            "rectified_10_40Hz_envelope_0_5Hz_robust_z",
        ],
        "ppg_channel_names": ["infrared_0.40_8.00Hz", "red_0.40_8.00Hz"],
        "pressure_feature_names": PRESSURE_FEATURE_NAMES,
        "pressure_preprocessing": {
            "hardware_mapping": (
                "Exact sparse linear refactor of the reference 1056-channel "
                "assignment map; output is 77x32."
            ),
            "layout_matrix": {
                "shape": [PRESSURE_ROWS * PRESSURE_COLS, RAW_SENSOR_COUNT],
                "nonzero_coefficients": 2306,
                "coefficient_values": [0.5, 1.0],
            },
            "steps": [
                "hardware channel mapping",
                "horizontal mattress-orientation flip",
                "linear interpolation of rows 38:44 and columns 14:18",
                "OTSU thresholding in original pressure units",
                "4-connected component removal with area >10",
                "foreground-only z-score normalization",
            ],
            "mean_map": (
                "Mean of OTSU-masked nonnegative maps, normalized to unit sum."
            ),
            "legacy_corrections": [
                "Replaced incorrect direct 32x33 reshape.",
                "OTSU threshold is now compared in the same physical value domain; "
                "the reference code mixed 0-255 and raw-pressure units.",
            ],
        },
        "quality_names": quality_names,
        "posture_ids": POSTURE_TO_ID,
        "subjects": manifest_subjects,
        "counts": dict(global_counts),
        "excluded_sources": {
            "WholeData": "Modality names and shapes are internally inconsistent.",
            "parameters.npy": (
                "SBP/DBP/HR/SP/BR alignment and provenance are unverified; BR contains -999."
            ),
            "information.txt": "Age and sex fields are empty.",
        },
        "known_anomalies": {
            "missing_pressure_windows": int(global_counts["windows"] - global_counts["pressure_valid"]),
            "handling": "Filled pressure tensors/features with NaN and pressure_valid=0.",
            "pressure_layout": (
                "1056 raw channels are mapped to 77x32 using the supplied hardware "
                "wiring transform, then horizontally flipped as in the reference pipeline."
            ),
        },
    }
    write_json(out_root / "manifest.json", manifest)
    write_json(out_root / "window_audit.json", audit_rows)

    audit_summary = {
        "window_count": len(audit_rows),
        "subject_count": len(manifest_subjects),
        "valid_pressure_count": int(sum(row["pressure_valid"] for row in audit_rows)),
        "ecg_hr_bpm_percentiles": nanpercentiles(
            [row["ecg_hr_bpm"] for row in audit_rows], [0, 10, 50, 90, 100]
        ),
        "ppg_ir_vs_ecg_abs_error_bpm_percentiles": nanpercentiles(
            [
                abs(row["ppg_ir_hr_bpm"] - row["ecg_hr_bpm"])
                if row["ppg_ir_hr_bpm"] is not None
                else np.nan
                for row in audit_rows
            ],
            [10, 25, 50, 75, 90],
        ),
        "ecg_rail_fraction_percentiles": nanpercentiles(
            [row["ecg_rail_fraction"] for row in audit_rows], [50, 75, 90, 95, 99]
        ),
        "bcg_spike_fraction_percentiles": nanpercentiles(
            [row["bcg_spike_fraction"] for row in audit_rows], [50, 75, 90, 95, 99]
        ),
        "notes": [
            "ECG HR is a detector-derived pseudo-label and must be spot-checked.",
            "All filtering is zero-phase and intended for offline research, not real-time deployment.",
            "Per-window robust scaling removes absolute amplitude; raw pressure is retained.",
        ],
    }
    write_json(out_root / "audit_summary.json", audit_summary)
    print(f"Wrote processed dataset to {out_root}")


if __name__ == "__main__":
    main()
