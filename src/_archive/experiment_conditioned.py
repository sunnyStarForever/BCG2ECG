"""
experiment_conditioned.py — 压力阵列条件化的 BCG→ECG 重建实验

消融对比矩阵
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  实验名               BCG编码器   压力条件化            聚合方式    动态分支
─ ──────────────────── ─────────── ──────────────────── ─────────── ────────
1  bcg_only              TinyTCN    无                    —           —
2  bcg_posture_onehot    TinyTCN    PostureLabelEncoder    —           —
3  bcg_handcrafted       TinyTCN    HandcraftedPressure    —           —
4  bcg_pressure_weighted TinyTCN    PressureEncoder        weighted    ✓
5  bcg_pressure_maxpool  TinyTCN    PressureEncoder        max_pool    ✓
6  bcg_pressure_nodyn    TinyTCN    PressureEncoder        weighted    ✗
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

评估指标（test set）：
  - 中位数 Pearson r (direct)
  - 中位数 max-lag Pearson r (±12 taps at 25 Hz)
  - 中位数 NRMSE
  - 段数量
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline_utils import write_json
from pressure_encoder import (
    PressureEncoder,
    HandcraftedPressureEncoder,
    PostureLabelEncoder,
    make_pressure_encoder,
)

SEED = 42
SEGMENT_LENGTH = 1000  # 10 seconds at 100 Hz


# ── helpers ─────────────────────────────────────────────────────────


def set_deterministic() -> None:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))


def load_windows(root: Path) -> dict[str, np.ndarray]:
    """Load all subjects, adding pressure_clean to the output dict."""
    output: dict[str, list[np.ndarray]] = {
        "bcg": [],
        "ecg": [],
        "ppg": [],
        "rpeak": [],
        "pressure_features": [],
        "pressure_clean": [],
        "quality": [],
        "posture": [],
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
                ("pressure_clean", "pressure_clean"),
                ("quality", "quality"),
                ("posture", "posture"),
            ]:
                output[key].append(z[source])
            output["groups"].append(np.full(n, group, dtype=np.int16))
    return {key: np.concatenate(values, axis=0) for key, values in output.items()}


def segment_signals(x: np.ndarray) -> np.ndarray:
    """Split 30-second windows into 3×10-second segments.

    Input:  (N, ...time) where time % SEGMENT_LENGTH == 0
    Output: (N*3, ...other_dims, SEGMENT_LENGTH)
    """
    if x.ndim == 2:
        x = x[:, None, :]
    n, channels, length = x.shape
    if length % SEGMENT_LENGTH:
        raise ValueError(f"Length {length} not divisible by {SEGMENT_LENGTH}")
    count = length // SEGMENT_LENGTH
    return (
        x.reshape(n, channels, count, SEGMENT_LENGTH)
        .transpose(0, 2, 1, 3)
        .reshape(n * count, channels, SEGMENT_LENGTH)
        .astype(np.float32)
    )


def prepare_data(windows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Segment signals and build pressure context tensors.

    Since pressure_clean (30 frames, 77, 32) is per-window (1 Hz),
    we repeat it 3 times so every 10-second segment gets the full
    30-frame pressure context.  (The pressure barely changes across
    the 30 s window, so this is a safe approximation.)
    """
    bcg = segment_signals(windows["bcg"])       # (N*3, 4, 1000)
    ecg = segment_signals(windows["ecg"])        # (N*3, 1, 1000)
    ppg = segment_signals(windows["ppg"])        # (N*3, 2, 1000)

    # Pressure: each window → 3 identical copies for its 3 segments
    pressure_clean = windows["pressure_clean"]   # (N, 30, 77, 32)
    pressure_clean = pressure_clean.repeat(3, axis=0)  # (N*3, 30, 77, 32)

    # Pressure features (manual) — repeat 3× for segments
    pressure_features = windows["pressure_features"]
    pressure_features = pressure_features.repeat(3, axis=0)

    # Posture labels — repeat 3× for segments
    posture = windows["posture"]
    posture = np.repeat(posture, 3, axis=0)

    quality = windows["quality"]
    groups = np.repeat(windows["groups"], 3)

    # Quality gate (same as deep_screen.py):
    #   ECG rail < 5%, RR-CV < 0.35, 18-66 beats/30s, BCG spikes < 8%
    window_good = (
        (quality[:, 0] < 0.05)
        & (quality[:, 1] < 0.35)
        & (quality[:, 2] >= 18)
        & (quality[:, 2] <= 66)
        & (quality[:, 8] < 0.08)
    )
    good = np.repeat(window_good, 3)

    return {
        "bcg": bcg,
        "ecg": ecg,
        "ppg": ppg,
        "pressure_clean": pressure_clean,
        "pressure_features": pressure_features,
        "posture": posture,
        "good": good,
        "groups": groups,
    }


def split_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-disjoint train/val/test split.

    80% train+val → further 80/20 → final ~64/16/20 split.
    """
    from sklearn.model_selection import GroupShuffleSplit
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val, test = next(outer.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    train_rel, val_rel = next(
        inner.split(train_val, groups=groups[train_val])
    )
    return train_val[train_rel], train_val[val_rel], test


# ── TinyTCN (adapted from deep_screen) ──────────────────────────────


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        padding = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=7,
                       padding=padding, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(width, width, kernel_size=7,
                       padding=padding, dilation=dilation),
            nn.GroupNorm(4, width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TinyTCN(nn.Module):
    """Minimal TCN with optional FiLM-style conditioning.

    Matches the deep_screen.py TinyTCN exactly, but context_dim=0
    skips the FiLM branch entirely (clean ablation control).
    """

    def __init__(self, in_channels: int,
                 context_dim: int = 0,
                 width: int = 16) -> None:
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

    def forward(self, x: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        if self.context is not None:
            gamma, beta = self.context(context).chunk(2, dim=1)
            hidden = hidden * (1.0 + 0.25 * gamma[:, :, None]) + beta[:, :, None]
        return self.head(self.blocks(hidden))


# ── Training ────────────────────────────────────────────────────────


def make_loader(
    x: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    shuffle: bool,
    batch_size: int = 64,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x[indices]),
        torch.from_numpy(context[indices]),
        torch.from_numpy(target[indices]),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def waveform_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 loss with derivative penalty (from deep_screen)."""
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
    epochs: int,
) -> tuple[TinyTCN, list[dict[str, float]]]:
    """Train one TinyTCN with optional conditioning context."""
    torch.manual_seed(SEED)
    context_dim = context.shape[1] if context.ndim == 2 else 0
    model = TinyTCN(x.shape[1], context_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    train_loader = make_loader(x, context, target, train, shuffle=True)
    val_loader = make_loader(x, context, target, val, shuffle=False)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_context, batch_target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_context)
            loss = waveform_loss(logits, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_context, batch_target, _ in val_loader:
                val_losses.append(
                    float(waveform_loss(
                        model(batch_x, batch_context), batch_target))
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        print(f"  {name} epoch {epoch}/{epochs}: {row}")
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
    """Run inference on the given index set."""
    dummy = np.zeros((x.shape[0], 1, SEGMENT_LENGTH), dtype=np.float32)
    loader = make_loader(x, context, dummy, indices, shuffle=False)
    model.eval()
    values = []
    with torch.no_grad():
        for batch_x, batch_context, _, _ in loader:
            values.append(model(batch_x, batch_context).numpy())
    return np.concatenate(values, axis=0)


def waveform_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Compute waveform similarity metrics (from deep_screen)."""
    direct_corr, lag_corr, nrmse = [], [], []
    for truth, pred in zip(target[:, 0], prediction[:, 0]):
        if np.std(truth) < 1e-6 or np.std(pred) < 1e-6:
            continue
        direct_corr.append(float(np.corrcoef(truth, pred)[0, 1]))
        normalized_truth = (truth - truth.mean()) / (truth.std() + 1e-8)
        normalized_pred = (pred - pred.mean()) / (pred.std() + 1e-8)
        nrmse.append(
            float(np.sqrt(np.mean((truth - pred) ** 2)) / (truth.std() + 1e-8))
        )
        cross = signal.correlate(normalized_truth, normalized_pred,
                                 mode="full") / truth.size
        lags = signal.correlation_lags(truth.size, pred.size, mode="full")
        allowed = np.abs(lags) <= 12
        lag_corr.append(float(np.max(cross[allowed])))
    return {
        "median_direct_correlation": float(np.median(direct_corr)),
        "mean_direct_correlation": float(np.mean(direct_corr)),
        "median_maxlag_correlation": float(np.median(lag_corr)),
        "median_nrmse": float(np.median(nrmse)),
        "n_segments": len(direct_corr),
    }


# ── Ablation experiment runner ──────────────────────────────────────


def build_context(
    data: dict[str, np.ndarray],
    encoder_name: str,
) -> tuple[np.ndarray, int]:
    """Build a context vector for each segment.

    Returns (context_array, context_dim).
    """
    if encoder_name == "none":
        # Empty context: TinyTCN skips the FiLM branch entirely
        return np.empty((data["bcg"].shape[0], 0), dtype=np.float32), 0

    # Load data
    pressure_clean = data["pressure_clean"]  # (N_seg, 30, 77, 32)

    if encoder_name == "posture_onehot":
        posture = data["posture"]  # (N_seg,) values 0-4
        posture_onehot = np.eye(5, dtype=np.float32)[posture]
        with torch.no_grad():
            enc = PostureLabelEncoder(out_dim=128)
            ctx = enc(torch.from_numpy(posture_onehot))
        return ctx.numpy(), 128

    elif encoder_name == "handcrafted":
        features = data.get("pressure_features")
        with torch.no_grad():
            enc = HandcraftedPressureEncoder(out_dim=128)
            ctx = enc(torch.from_numpy(features / (np.nanstd(features, axis=0, keepdims=True) + 1e-8)))
        return ctx.numpy(), 128

    elif encoder_name.startswith("pressure_cnn"):
        # Parse aggregation method from suffix
        # "pressure_cnn_weighted", "pressure_cnn_maxpool", "pressure_cnn_nodyn"
        parts = encoder_name.split("_")
        aggr = "weighted"
        use_dyn = True
        if "maxpool" in parts or "max" in parts:
            aggr = "max_pool"
        if "nodyn" in parts or "mean" in parts:
            use_dyn = False

        with torch.no_grad():
            enc = PressureEncoder(
                out_dim=128,
                temporal_aggregation=aggr,
                use_dynamic_branch=use_dyn,
            )
            ctx = enc(torch.from_numpy(pressure_clean))
        return ctx.numpy(), 128

    raise ValueError(f"Unknown encoder: {encoder_name}")


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pressure-conditioned BCG→ECG ablation experiments"
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/preliminary/conditioned_results.json"))
    parser.add_argument("--epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic()
    root = args.data.resolve()

    print("Loading and segmenting data...")
    windows = load_windows(root)
    data = prepare_data(windows)

    train_all, val, test = split_indices(data["groups"])
    print(
        f"Segments: train={len(train_all)}/{len(train_all)} "
        f"val={len(val)} test={len(test)}"
    )

    # Apply quality gate: keep only segments from clean windows
    # Based on deep_screen quality criteria
    good_mask = data["good"]
    train = train_all[good_mask[train_all]]
    print(f"  training segments after quality gate: {len(train)}")

    # Ablation experiments
    experiments = [
        ("bcg_only",              "none"),
        ("bcg_handcrafted",       "handcrafted"),
        ("bcg_posture_onehot",    "posture_onehot"),
        ("bcg_pressure_weighted", "pressure_cnn_weighted"),
        ("bcg_pressure_maxpool",  "pressure_cnn_maxpool"),
        ("bcg_pressure_nodyn",    "pressure_cnn_nodyn"),
    ]

    all_results: dict[str, object] = {}
    all_histories: dict[str, list[dict[str, float]]] = {}
    ablation_table: list[dict[str, object]] = []

    for exp_name, encoder_name in experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name} (encoder={encoder_name})")
        print(f"{'=' * 60}")

        # Build context
        if encoder_name == "none":
            context = np.empty((data["bcg"].shape[0], 0), dtype=np.float32)
            context_dim = 0
            print(f"  context_dim = {context_dim} (no conditioning)")
        else:
            context, context_dim = build_context(data, encoder_name)
            print(f"  context_dim = {context_dim}")

        # Train
        model, history = train_model(
            exp_name,
            data["bcg"],
            context,
            data["ecg"],
            train,
            val,
            args.epochs,
        )
        all_histories[exp_name] = history

        # Evaluate
        test_pred = predict(model, data["bcg"], context, test)
        metrics = waveform_metrics(data["ecg"][test], test_pred)
        print(f"  test_metrics: {json.dumps(metrics, indent=4)}")

        row = {
            "experiment": exp_name,
            "encoder": encoder_name,
            "median_direct_correlation": metrics["median_direct_correlation"],
            "median_maxlag_correlation": metrics["median_maxlag_correlation"],
            "median_nrmse": metrics["median_nrmse"],
            "n_segments": metrics["n_segments"],
            "val_loss_final": history[-1]["val_loss"],
            "val_loss_best": min(item["val_loss"] for item in history),
        }
        ablation_table.append(row)

    # Compile final output
    results = {
        "protocol": {
            "task": "waveform_reconstruction",
            "model": "TinyTCN (width=16, 3×ResidualBlock)",
            "segment_seconds": 10,
            "sampling_rate_hz": 100,
            "split": "subject-disjoint 64/16/20",
            "epochs": args.epochs,
            "note": "Preliminary feasibility screen, not publication-grade.",
        },
        "split_summary": {
            "train_segments": int(len(train_all)),
            "val_segments": int(len(val)),
            "test_segments": int(len(test)),
            "test_subjects": sorted(
                int(s) for s in np.unique(data["groups"][test])
            ),
        },
        "ablation": ablation_table,
        "training_history": all_histories,
    }
    write_json(args.out.resolve(), results)
    print(f"\n{'=' * 60}")
    print(f"Results written to {args.out.resolve()}")
    print("Summary:")
    for row in ablation_table:
        print(
            f"  {row['experiment']:30s}  "
            f"direct_r={row['median_direct_correlation']:.4f}  "
            f"maxlag_r={row['median_maxlag_correlation']:.4f}  "
            f"nrmse={row['median_nrmse']:.3f}"
        )


if __name__ == "__main__":
    main()