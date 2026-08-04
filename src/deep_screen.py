from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, signal
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline_utils import write_json


SEED = 42
SEGMENT_LENGTH = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight deep feasibility screens.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/preliminary/deep_results.json"),
    )
    parser.add_argument("--event-epochs", type=int, default=6)
    parser.add_argument("--waveform-epochs", type=int, default=4)
    return parser.parse_args()


def set_deterministic() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))


def load_windows(root: Path) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        "bcg": [],
        "ecg": [],
        "ppg": [],
        "rpeak": [],
        "pressure_features": [],
        "quality": [],
        "groups": [],
    }
    for group, path in enumerate(sorted((root / "subjects").glob("S*.npz"))):
        with np.load(path) as z:
            n = z["posture"].shape[0]
            for key, source in [
                ("bcg", "bcg"),
                ("ecg", "ecg_100"),
                ("ppg", "ppg"),
                ("rpeak", "rpeak_mask_100"),
                ("pressure_features", "pressure_features"),
                ("quality", "quality"),
            ]:
                output[key].append(z[source])
            output["groups"].append(np.full(n, group, dtype=np.int16))
    return {key: np.concatenate(values, axis=0) for key, values in output.items()}


def segment_signals(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, None, :]
    n, channels, length = x.shape
    if length % SEGMENT_LENGTH:
        raise ValueError(f"Length {length} is not divisible by {SEGMENT_LENGTH}")
    count = length // SEGMENT_LENGTH
    return (
        x.reshape(n, channels, count, SEGMENT_LENGTH)
        .transpose(0, 2, 1, 3)
        .reshape(n * count, channels, SEGMENT_LENGTH)
        .astype(np.float32)
    )


def prepare_data(windows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    event_binary = segment_signals(windows["rpeak"]).astype(np.float32)
    heatmap = ndimage.gaussian_filter1d(event_binary, sigma=2.5, axis=-1)
    heatmap *= np.float32(np.sqrt(2.0 * np.pi) * 2.5)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    groups = np.repeat(windows["groups"], 3)
    quality = windows["quality"]
    window_good = (
        (quality[:, 0] < 0.05)
        & (quality[:, 1] < 0.35)
        & (quality[:, 2] >= 18)
        & (quality[:, 2] <= 66)
        & (quality[:, 8] < 0.08)
    )
    teacher_good = (
        window_good
        & (quality[:, 4] > 0.30)
        & (quality[:, 7] <= 12.0)
    )
    return {
        "bcg": segment_signals(windows["bcg"]),
        "ecg": segment_signals(windows["ecg"]),
        "ppg": segment_signals(windows["ppg"]),
        "event": heatmap,
        "event_binary": event_binary,
        "context": np.repeat(windows["pressure_features"], 3, axis=0).astype(np.float32),
        "groups": groups,
        "good": np.repeat(window_good, 3),
        "teacher_good": np.repeat(teacher_good, 3),
    }


def split_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val, test = next(outer.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    train_rel, val_rel = next(
        inner.split(train_val, groups=groups[train_val])
    )
    return train_val[train_rel], train_val[val_rel], test


def scale_context(
    context: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    output = context.astype(np.float64, copy=True)
    median = np.nanmedian(output[train], axis=0)
    missing_row, missing_col = np.where(~np.isfinite(output))
    output[missing_row, missing_col] = median[missing_col]
    q10, q90 = np.percentile(output[train], [10, 90], axis=0)
    output = (output - median) / np.maximum(q90 - q10, 1e-6)
    return np.clip(output, -8.0, 8.0).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        padding = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=7, padding=padding, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(width, width, kernel_size=7, padding=padding, dilation=dilation),
            nn.GroupNorm(4, width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int, context_dim: int = 0, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, width, kernel_size=9, padding=4)
        self.context = (
            nn.Sequential(nn.Linear(context_dim, width * 2), nn.Tanh())
            if context_dim
            else None
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, 1),
            ResidualBlock(width, 2),
            ResidualBlock(width, 4),
        )
        self.head = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        if self.context is not None:
            gamma, beta = self.context(context).chunk(2, dim=1)
            hidden = hidden * (1.0 + 0.25 * gamma[:, :, None]) + beta[:, :, None]
        return self.head(self.blocks(hidden))


def make_loader(
    x: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    shuffle: bool,
    batch_size: int = 64,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices]),
        torch.from_numpy(context[indices]),
        torch.from_numpy(target[indices]),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def task_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    task: str,
) -> torch.Tensor:
    if task == "event":
        positive_weight = torch.tensor(8.0, device=logits.device)
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=positive_weight,
        )
    point = nn.functional.smooth_l1_loss(logits, target, reduction="none")
    weight = 1.0 + 1.5 * torch.clamp(torch.abs(target) / 3.0, 0.0, 2.0)
    derivative = nn.functional.smooth_l1_loss(
        torch.diff(logits, dim=-1),
        torch.diff(target, dim=-1),
    )
    return torch.mean(point * weight) + 0.15 * derivative


def train_model(
    name: str,
    x: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    task: str,
    epochs: int,
    teacher: TinyTCN | None = None,
    teacher_x: np.ndarray | None = None,
) -> tuple[TinyTCN, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    model = TinyTCN(x.shape[1], context.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = make_loader(x, context, target, train, shuffle=True)
    val_loader = make_loader(x, context, target, val, shuffle=False)
    if teacher is not None:
        teacher.eval()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_context, batch_target, batch_indices in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_context)
            loss = task_loss(logits, batch_target, task)
            if teacher is not None and teacher_x is not None:
                with torch.no_grad():
                    teacher_input = torch.from_numpy(
                        teacher_x[batch_indices.numpy()]
                    )
                    empty_context = torch.empty((teacher_input.shape[0], 0))
                    teacher_probability = torch.sigmoid(
                        teacher(teacher_input, empty_context)
                    )
                loss = loss + 0.40 * nn.functional.mse_loss(
                    torch.sigmoid(logits),
                    teacher_probability,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_context, batch_target, _batch_indices in val_loader:
                val_losses.append(
                    float(task_loss(model(batch_x, batch_context), batch_target, task))
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        print(f"{name} epoch {epoch}/{epochs}: {row}")
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def predict(
    model: TinyTCN,
    x: np.ndarray,
    context: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    dummy = np.zeros((x.shape[0], 1, SEGMENT_LENGTH), dtype=np.float32)
    loader = make_loader(x, context, dummy, indices, shuffle=False)
    model.eval()
    values = []
    with torch.no_grad():
        for batch_x, batch_context, _target, _index in loader:
            values.append(model(batch_x, batch_context).numpy())
    return np.concatenate(values, axis=0)


def match_events(
    truth: np.ndarray,
    prediction: np.ndarray,
    tolerance: int = 10,
) -> tuple[int, int, int, list[int]]:
    used: set[int] = set()
    tp = 0
    errors = []
    for predicted in prediction:
        candidates = [
            (abs(int(predicted) - int(actual)), idx)
            for idx, actual in enumerate(truth)
            if idx not in used and abs(int(predicted) - int(actual)) <= tolerance
        ]
        if candidates:
            error, best = min(candidates)
            used.add(best)
            tp += 1
            errors.append(error)
    return tp, len(prediction) - tp, len(truth) - tp, errors


def event_metrics(
    event_binary: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    total_tp = total_fp = total_fn = 0
    timing_errors: list[int] = []
    hr_errors = []
    for truth_mask, probability in zip(event_binary[:, 0], probabilities[:, 0]):
        truth = np.flatnonzero(truth_mask > 0.5)
        predicted, _ = signal.find_peaks(
            probability,
            height=threshold,
            distance=30,
        )
        tp, fp, fn, errors = match_events(truth, predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        timing_errors.extend(errors)
        hr_errors.append(abs(len(predicted) - len(truth)) * 6.0)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return {
        "event_f1_100ms": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "event_precision_100ms": float(precision),
        "event_recall_100ms": float(recall),
        "timing_mae_ms": float(np.mean(timing_errors) * 10.0) if timing_errors else 1000.0,
        "hr_mae_bpm": float(np.mean(hr_errors)),
        "threshold": float(threshold),
        "n_segments": int(event_binary.shape[0]),
    }


def choose_event_threshold(
    event_binary: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.10, 0.80, 15)
    rows = [(float(value), event_metrics(event_binary, probabilities, float(value))) for value in candidates]
    return max(rows, key=lambda item: item[1]["event_f1_100ms"])


def waveform_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlations = []
    maxlag = []
    nrmse = []
    for truth, pred in zip(target[:, 0], prediction[:, 0]):
        if np.std(truth) < 1e-6 or np.std(pred) < 1e-6:
            continue
        correlations.append(float(np.corrcoef(truth, pred)[0, 1]))
        nrmse.append(float(np.sqrt(np.mean((truth - pred) ** 2)) / (np.std(truth) + 1e-8)))
        truth_z = (truth - truth.mean()) / (truth.std() + 1e-8)
        pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
        cross = signal.correlate(truth_z, pred_z, mode="full") / truth.size
        lags = signal.correlation_lags(truth.size, pred.size, mode="full")
        maxlag.append(float(np.max(cross[np.abs(lags) <= 50])))
    return {
        "median_direct_correlation": float(np.median(correlations)),
        "mean_direct_correlation": float(np.mean(correlations)),
        "median_maxlag_correlation": float(np.median(maxlag)),
        "median_nrmse": float(np.median(nrmse)),
        "n_segments": len(correlations),
    }


def main() -> None:
    args = parse_args()
    set_deterministic()
    windows = load_windows(args.data.resolve())
    data = prepare_data(windows)
    train_all, val, test = split_indices(data["groups"])
    train = train_all[data["good"][train_all]]
    teacher_train = train_all[data["teacher_good"][train_all]]
    context = scale_context(data["context"], train)
    empty_context = np.empty((data["groups"].size, 0), dtype=np.float32)
    print(
        f"Segments train/val/test={len(train)}/{len(val)}/{len(test)}, "
        f"subjects={len(np.unique(data['groups'][train]))}/"
        f"{len(np.unique(data['groups'][val]))}/{len(np.unique(data['groups'][test]))}"
    )

    event_models: dict[str, TinyTCN] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    teacher, histories["ppg_teacher"] = train_model(
        "ppg_teacher",
        data["ppg"],
        empty_context,
        data["event"],
        teacher_train,
        val,
        "event",
        args.event_epochs,
    )
    event_models["ppg_teacher"] = teacher
    baseline, histories["bcg_event"] = train_model(
        "bcg_event",
        data["bcg"],
        empty_context,
        data["event"],
        train,
        val,
        "event",
        args.event_epochs,
    )
    event_models["bcg_event"] = baseline
    contextual, histories["bcg_pressure_event"] = train_model(
        "bcg_pressure_event",
        data["bcg"],
        context,
        data["event"],
        train,
        val,
        "event",
        args.event_epochs,
    )
    event_models["bcg_pressure_event"] = contextual
    distilled, histories["bcg_pressure_distilled"] = train_model(
        "bcg_pressure_distilled",
        data["bcg"],
        context,
        data["event"],
        train,
        val,
        "event",
        args.event_epochs,
        teacher=teacher,
        teacher_x=data["ppg"],
    )
    event_models["bcg_pressure_distilled"] = distilled

    event_results: dict[str, dict[str, object]] = {}
    for name, model in event_models.items():
        x = data["ppg"] if name == "ppg_teacher" else data["bcg"]
        model_context = context if "pressure" in name else empty_context
        val_probability = torch.sigmoid(
            torch.from_numpy(predict(model, x, model_context, val))
        ).numpy()
        threshold, val_metrics = choose_event_threshold(
            data["event_binary"][val],
            val_probability,
        )
        test_probability = torch.sigmoid(
            torch.from_numpy(predict(model, x, model_context, test))
        ).numpy()
        clean_mask = data["good"][test]
        event_results[name] = {
            "validation": val_metrics,
            "test_all": event_metrics(
                data["event_binary"][test],
                test_probability,
                threshold,
            ),
            "test_clean": event_metrics(
                data["event_binary"][test][clean_mask],
                test_probability[clean_mask],
                threshold,
            ),
        }

    waveform_results: dict[str, dict[str, object]] = {}
    waveform_specs = [
        ("bcg_waveform", data["bcg"], empty_context),
        ("bcg_pressure_waveform", data["bcg"], context),
        ("ppg_waveform_upper_bound", data["ppg"], empty_context),
    ]
    for name, x, model_context in waveform_specs:
        model, history = train_model(
            name,
            x,
            model_context,
            data["ecg"],
            train,
            val,
            "waveform",
            args.waveform_epochs,
        )
        histories[name] = history
        prediction = predict(model, x, model_context, test)
        clean_mask = data["good"][test]
        waveform_results[name] = {
            "test_all": waveform_metrics(data["ecg"][test], prediction),
            "test_clean": waveform_metrics(
                data["ecg"][test][clean_mask],
                prediction[clean_mask],
            ),
        }

    results = {
        "protocol": {
            "sampling_rate_hz": 100,
            "segment_seconds": 10,
            "model": "Tiny dilated TCN, 16 channels, subject-disjoint split",
            "event_target": "ECG R peaks converted to sigma=25 ms Gaussian heatmaps",
            "event_tolerance_ms": 100,
            "train_quality_gate": (
                "ECG rail<5%, RR-CV<0.35, 18-66 beats/30s, BCG spikes<8%"
            ),
            "warning": "Single-split low-budget screening, not publication-grade evaluation.",
        },
        "split": {
            "train_segments": int(len(train)),
            "validation_segments": int(len(val)),
            "test_segments": int(len(test)),
            "train_subjects": [int(v) for v in np.unique(data["groups"][train])],
            "validation_subjects": [int(v) for v in np.unique(data["groups"][val])],
            "test_subjects": [int(v) for v in np.unique(data["groups"][test])],
            "clean_test_fraction": float(np.mean(data["good"][test])),
        },
        "event_detection": event_results,
        "waveform_reconstruction": waveform_results,
        "training_history": histories,
    }
    write_json(args.out.resolve(), results)
    print(json.dumps({k: v for k, v in results.items() if k != "training_history"}, indent=2))
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
