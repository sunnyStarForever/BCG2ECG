"""
Pressure array encoder for BCG→ECG conditioning.

The pressure data has these characteristics:
- 30 frames at 1 Hz, each 77×32 (mapped from 1056 hardware channels)
- ~7-8% foreground pixels (body contact); the rest is background (zero)
- Frame-to-frame Pearson r > 0.989 — the body barely moves in 30 seconds
- Clean data is foreground-only z-score normalized; mapped data is raw pressure

Design rationale (why NOT average pooling):

1. Extreme sparsity (~7.7% active): average pooling drowns the contact signal
   in zero-valued background pixels.
2. Frame redundancy (r > 0.989): simple averaging of 30 nearly-identical
   frames is wasted computation — statistically you need ~1 frame.
3. The useful cross-frame signal is the *subtle variation*: breathing-induced
   micro-movements, contact area shifts, pressure redistribution. Averaging
   suppresses precisely what makes the temporal dimension informative.

Architecture (two-branch + per-frame weighted fusion):

  Static branch:        pressure_mean_map [77,32] → 2D CNN → z_static [D_s]
  Dynamic branch:       per-frame diff [30,77,32]  → 2D CNN → z_dyn   [D_d]
  Weighted per-frame:   per-frame maps [30,77,32]  → 2D CNN
                        → 30 scalar weights via small MLP
                        → weighted sum of per-frame features → z_weighted [D_w]
  Output:               concat(z_static, z_dyn, z_weighted) → [D_total]

The two-branch design gives the model both:
- The "which posture / contact geometry" (static mean map)
- The "how stable / is there movement" (frame-to-frame differences)
- The "which frames have higher contact quality" (learned per-frame weights)

Ablation variants are selectable via the `temporal_aggregation` parameter:
- "weighted": learned scalar weight per frame (default, recommended)
- "max_pool": max-pooling over frames (robust to sparsity)
- "attention": lightweight multi-head self-attention over frames
- "mean": simple mean pooling (baseline, expected to be worst)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import torch
from torch import nn


PRESSURE_ROWS = 77
PRESSURE_COLS = 32
N_FRAMES = 30

_SpatialEncoderOutput = tuple[torch.Tensor, torch.Tensor]  # static, dynamic


# ---------------------------------------------------------------------------
# Spatial backbone: shared 2D CNN applied per-frame
# ---------------------------------------------------------------------------

class SpatialBackbone(nn.Module):
    """Lightweight 2D CNN that encodes a single pressure map (77×32).

    Design notes:
    - kernel sizes (7,5) reflect the aspect ratio — more context needed in the
      longer (vertical/head-to-toe) direction.
    - Depthwise-separable style reduces params without losing expressivity
      (77*32=2464 pixels is small enough that a full conv is fine too).
    - Output is a flat feature vector per map.
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 5), padding=(3, 2)),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 2)),  # → 128 × 4 × 2 = 1024
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 77, 32) or (B*T, 1, 77, 32) → (B, out_dim) or (B*T, out_dim)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Temporal aggregation strategies
# ---------------------------------------------------------------------------

class TemporalAggregation(str, Enum):
    weighted = "weighted"
    max_pool = "max"       # avoid shadowing built-in max()
    attention = "attention"
    mean = "mean"


class FrameWeightPredictor(nn.Module):
    """Predict a scalar importance weight for each frame from its features.

    Input: per-frame spatial features (T, D)
    Output: T scalar weights, softmax-normalised.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, D) → weights: (B, T, 1)"""
        raw = self.net(features)  # (B, T, 1)
        return raw.softmax(dim=1)


class FrameAttention(nn.Module):
    """Lightweight 2-head self-attention over T=30 frames.

    Uses learnable positional encoding because frame order matters
    (earlier frames → later frames, breathing phase evolves).
    """

    def __init__(self, dim: int, heads: int = 2) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, N_FRAMES, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → output: (B, D)"""
        x = x + self.pos
        out, _ = self.attn(x, x, x)  # (B, T, D)
        return out.mean(dim=1)  # Attend then average-pool over T


def aggregate_frames(
    features: torch.Tensor,
    method: TemporalAggregation | str,
    weight_predictor: FrameWeightPredictor | None = None,
    attention: FrameAttention | None = None,
) -> torch.Tensor:
    """Aggregate T per-frame feature vectors into a single context vector.

    Args:
        features: (B, T, D) tensor of per-frame spatial encodings.
        method: Aggregation strategy.
        weight_predictor: Required for "weighted".
        attention: Required for "attention".

    Returns:
        (B, D) aggregated context.
    """
    method = TemporalAggregation(method)
    if method == TemporalAggregation.max_pool:
        return features.max(dim=1).values
    elif method == TemporalAggregation.weighted:
        if weight_predictor is None:
            raise ValueError("weight_predictor required for weighted aggregation")
        weights = weight_predictor(features)  # (B, T, 1)
        return (features * weights).sum(dim=1)  # (B, D)
    elif method == TemporalAggregation.attention:
        if attention is None:
            raise ValueError("attention module required for attention aggregation")
        return attention(features)  # (B, D)
    elif method == TemporalAggregation.mean:
        return features.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Full pressure encoder
# ---------------------------------------------------------------------------

class PressureEncoder(nn.Module):
    """Encode a 30-frame pressure array sequence into a conditioning vector.

    The encoder has three parallel paths:

    1. **Static path**: 2D CNN applied to the mean pressure map (77×32).
       This captures the overall contact geometry — which posture, which side,
       how much body surface is touching the mattress.

    2. **Dynamic path**: 2D CNN applied to the frame-to-frame absolute
       difference map, averaged over adjacent frames. This captures breathing,
       micro-movements, and contact instability.

    3. **Per-frame weighted path**: 2D CNN applied to each of the 30 frames,
       followed by a learned per-frame weight predictor. Weighted sum over
       frames produces a single context vector. This captures frame-quality
       aware spatial features.

    The three outputs are concatenated and projected to `out_dim`.

    Parameters
    ----------
    out_dim:
        Output dimension of the conditioning vector.
    spatial_dim:
        Internal dimension of the spatial backbone.
    temporal_aggregation:
        How to pool the 30 per-frame features. See TemporalAggregation.
    use_dynamic_branch:
        Whether to include the frame-to-frame difference path.
    """

    def __init__(
        self,
        out_dim: int = 128,
        spatial_dim: int = 128,
        temporal_aggregation: str = "weighted",
        use_dynamic_branch: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_aggregation = TemporalAggregation(temporal_aggregation)
        self.use_dynamic_branch = use_dynamic_branch

        # Shared spatial backbone (used by all branches)
        self.backbone = SpatialBackbone(spatial_dim)

        # Static branch: mean map → identical spatial architecture
        # Reuses the same backbone via forward pass on different input

        # Dynamic branch: frame diff mean → same backbone
        # (optional, controlled by use_dynamic_branch)

        # Per-frame weighted branch components
        if self.temporal_aggregation == TemporalAggregation.weighted:
            self.weight_predictor = FrameWeightPredictor(spatial_dim)
        else:
            self.weight_predictor = None

        if self.temporal_aggregation == TemporalAggregation.attention:
            self.frame_attention = FrameAttention(spatial_dim)
        else:
            self.frame_attention = None

        # Fusion: concat all branches → project
        n_branches = 3 if use_dynamic_branch else 2
        self.fusion = nn.Sequential(
            nn.Linear(n_branches * spatial_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _encode_spatial_branch(
        self, maps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode static (mean) and dynamic (frame-diff) maps.

        Args:
            maps: (B, 30, 77, 32) tensor — use pressure_clean (foreground-only z-score).

        Returns:
            (z_static, z_dynamic) each (B, spatial_dim).
        """
        # Static: per-window mean map
        mean_map = maps.mean(dim=1, keepdim=True)  # (B, 1, 77, 32)
        z_static = self.backbone(mean_map)          # (B, spatial_dim)

        # Dynamic: mean absolute frame-to-frame difference
        if self.use_dynamic_branch:
            diff = (maps[:, 1:] - maps[:, :-1]).abs()  # (B, 29, 77, 32)
            diff_mean = diff.mean(dim=1, keepdim=True)   # (B, 1, 77, 32)
            z_dynamic = self.backbone(diff_mean)          # (B, spatial_dim)
        else:
            z_dynamic = torch.zeros(
                maps.shape[0], self.backbone.out_dim,
                device=maps.device,
            )

        return z_static, z_dynamic

    def _encode_perframe_branch(self, maps: torch.Tensor) -> torch.Tensor:
        """Encode each frame individually and aggregate.

        Args:
            maps: (B, 30, 77, 32).

        Returns:
            (B, spatial_dim) aggregated context.
        """
        B, T, H, W = maps.shape
        # Flatten B and T for parallel per-frame encoding
        flat = maps.reshape(B * T, 1, H, W)
        features = self.backbone(flat)  # (B*T, D)
        features = features.reshape(B, T, -1)  # (B, T, D)

        return aggregate_frames(
            features,
            method=self.temporal_aggregation,
            weight_predictor=self.weight_predictor,
            attention=self.frame_attention,
        )

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        """Encode a batch of pressure array sequences.

        Args:
            pressure: (B, 30, 77, 32) tensor.
                Use pressure_clean (foreground-only z-score) for training.

        Returns:
            (B, out_dim) conditioning vector, suitable for FiLM or
            concatenation with BCG features.
        """
        z_static, z_dynamic = self._encode_spatial_branch(pressure)
        z_weighted = self._encode_perframe_branch(pressure)

        if self.use_dynamic_branch:
            combined = torch.cat([z_static, z_dynamic, z_weighted], dim=1)
        else:
            combined = torch.cat([z_static, z_weighted], dim=1)

        return self.fusion(combined)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

class HandcraftedPressureEncoder(nn.Module):
    """Baseline: encode handcrafted 16-d pressure features with an MLP.

    This serves as a lower bound — it compresses the existing statistical
    features (mean, std, center_x, center_y, etc.) via a small MLP.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 16) nan-filled → (B, out_dim)"""
        # Replace NaN with 0 (features already imputed during preprocessing)
        features = torch.nan_to_num(features, nan=0.0)
        return self.net(features)


class PostureLabelEncoder(nn.Module):
    """Baseline: encode the one-hot posture label (5 classes).

    This answers: "does the pressure array provide information
    beyond just knowing which posture the subject is in?"
    """

    def __init__(self, n_postures: int = 5, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_postures, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, posture_onehot: torch.Tensor) -> torch.Tensor:
        """posture_onehot: (B, 5) → (B, out_dim)"""
        return self.net(posture_onehot)


# ---------------------------------------------------------------------------
# Registry for easy ablation
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "pressure_cnn_weighted": PressureEncoder,
    "handcrafted": HandcraftedPressureEncoder,
    "posture_onehot": PostureLabelEncoder,
}


def make_pressure_encoder(
    name: str,
    out_dim: int = 128,
    temporal_aggregation: str = "weighted",
    use_dynamic_branch: bool = True,
) -> nn.Module:
    """Factory for ablation experiments.

    Args:
        name: One of:
            - "pressure_cnn_weighted": Full Spatial CNN with weighted temporal fusion
            - "handcrafted": 16-d statistical features through MLP
            - "posture_onehot": 5-class posture one-hot through MLP
        out_dim: Output conditioning vector dimension.
        temporal_aggregation: For PressureEncoder — how to pool frames.
        use_dynamic_branch: For PressureEncoder — include the frame-diff path.

    Example::

        enc = make_pressure_encoder("pressure_cnn_weighted", out_dim=128)
        z = enc(pressure_clean)  # (B, 128)

        enc = make_pressure_encoder("handcrafted", out_dim=128)
        z = enc(pressure_features)  # (B, 128)
    """
    if name == "pressure_cnn_weighted":
        return PressureEncoder(
            out_dim=out_dim,
            temporal_aggregation=temporal_aggregation,
            use_dynamic_branch=use_dynamic_branch,
        )
    elif name == "handcrafted":
        return HandcraftedPressureEncoder(out_dim=out_dim)
    elif name == "posture_onehot":
        return PostureLabelEncoder(out_dim=out_dim)
    else:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {sorted(_ENCODER_REGISTRY)}"
        )