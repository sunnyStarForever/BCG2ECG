"""
ppg_to_ecg.py — 角度C：PPG→ECG 深度波形重建可行性探索

背景：
  诊断阶段已确认 PPG 的能力梯度——
    心率估计 (标量)  r=0.68   ✅ 优秀
    事件对齐 (PTT)   190ms    ✅ 优秀
    波形重建 (逐点)  r=0.08   ❌ 困难
  线性 Ridge 上限:   r=0.012  ❌ 完全失败
  TinyTCN 基线:     r=0.113  (deep_screen.py 的 ppg_waveform_upper_bound)

目标：探索非线性深度模型能否显著突破 r=0.11 的上限。

实验设计 (递进式消融)：
  A. ppg_tcn_baseline     — 复现 TinyTCN 基线 (验证 r≈0.11)
  B. ppg_tcn_wider        — 加宽 TCN (16→64通道) 看是否容量不足
  C. ppg_tcn_dualinput    — IR+RED 双通道分离编码再融合
  D. ppg_tcn_freqloss     — 加入 STFT 频域损失 (BiM-Diff 思路)
  E. ppg_unet             — 1D U-Net 跳连 (多尺度重建)

关键设计决策：
  - 采样率 100 Hz (ECG_100)，10秒片段
  - 受试者分离 64/16/20 划分
  - 质量门控：只用干净窗口训练
  - 指标：direct r / maxlag r (±30样本=±300ms, 容忍 PTT) / NRMSE
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# 子目录脚本需向上插一级 parent 以导入根目录核心库
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent))

import numpy as np
import torch
from scipy import signal as sp
from torch import nn
from pipeline_utils import write_json

SEED = 42
SEG = 1000  # 10s @ 100Hz


# ── 数据 ────────────────────────────────────────────────────────────

def set_deterministic():
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)


def load_all(root: Path):
    out = {k: [] for k in ["ppg", "ecg", "rpeak", "quality", "groups"]}
    for g, p in enumerate(sorted(root.glob("subjects/S*.npz"))):
        z = np.load(p); n = z["posture"].shape[0]
        out["ppg"].append(z["ppg"])
        out["ecg"].append(z["ecg_100"])
        out["rpeak"].append(z["rpeak_mask_100"])
        out["quality"].append(z["quality"])
        out["groups"].append(np.full(n, g, dtype=np.int16))
        z.close()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def seg(x):
    """(N, C, L) → (N*3, C, SEG), L=3000 → 3×1000."""
    if x.ndim == 2:
        x = x[:, None, :]
    n, c, L = x.shape
    k = L // SEG
    return x.reshape(n, c, k, SEG).transpose(0, 2, 1, 3).reshape(n*k, c, SEG).astype(np.float32)


def prepare(w):
    good = ((w["quality"][:, 0] < 0.05)
            & (w["quality"][:, 1] < 0.35)
            & (w["quality"][:, 2] >= 18)
            & (w["quality"][:, 2] <= 66)
            & (w["quality"][:, 8] < 0.08))
    return {
        "ppg": seg(w["ppg"]),       # (N*3, 2, 1000)
        "ecg": seg(w["ecg"]),       # (N*3, 1, 1000)
        "groups": np.repeat(w["groups"], 3),
        "good": np.repeat(good, 3),
    }


def split(groups):
    from sklearn.model_selection import GroupShuffleSplit
    o = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv, te = next(o.split(groups, groups=groups))
    i = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    tr, va = next(i.split(tv, groups=groups[tv]))
    return tv[tr], tv[va], te


# ── 模型 ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        p = 3 * d
        self.net = nn.Sequential(
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w), nn.GELU(), nn.Dropout(0.05),
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class TCN(nn.Module):
    """可调宽度的 TCN，输入 (B, C_in, L) → (B, 1, L)。"""
    def __init__(self, ci, width=16):
        super().__init__()
        self.stem = nn.Conv1d(ci, width, 9, padding=4)
        self.blocks = nn.Sequential(ResBlock(width, 1), ResBlock(width, 2), ResBlock(width, 4))
        self.head = nn.Conv1d(width, 1, 1)
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class DualInputTCN(nn.Module):
    """IR/RED 双通道分别编码再融合 (角度C-C)。"""
    def __init__(self, width=32):
        super().__init__()
        self.enc_ir = nn.Sequential(nn.Conv1d(1, width, 9, padding=4), nn.GELU())
        self.enc_red = nn.Sequential(nn.Conv1d(1, width, 9, padding=4), nn.GELU())
        self.fuse = nn.Conv1d(width * 2, width, 1)
        self.blocks = nn.Sequential(ResBlock(width, 1), ResBlock(width, 2), ResBlock(width, 4))
        self.head = nn.Conv1d(width, 1, 1)
    def forward(self, x):
        # x: (B, 2, L) — 通道0=IR, 通道1=RED
        h = torch.cat([self.enc_ir(x[:, 0:1]), self.enc_red(x[:, 1:2])], dim=1)
        h = self.fuse(h)
        return self.head(self.blocks(h))


class UNet1D(nn.Module):
    """1D U-Net 带跳连 (角度C-E)，多尺度重建。"""
    def __init__(self, ci=2, base=16):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv1d(ci, base, 7, padding=3), nn.GroupNorm(4, base), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv1d(base, base*2, 7, stride=2, padding=3), nn.GroupNorm(4, base*2), nn.GELU())
        self.e3 = nn.Sequential(nn.Conv1d(base*2, base*4, 7, stride=2, padding=3), nn.GroupNorm(8, base*4), nn.GELU())
        # Bottleneck
        self.bot = nn.Sequential(ResBlock(base*4, 1), ResBlock(base*4, 2))
        # Decoder
        self.d3 = nn.ConvTranspose1d(base*4, base*2, 4, stride=2, padding=1)
        self.u3 = nn.Sequential(nn.Conv1d(base*4, base*2, 1), nn.GroupNorm(4, base*2), nn.GELU())
        self.d2 = nn.ConvTranspose1d(base*2, base, 4, stride=2, padding=1)
        self.u2 = nn.Sequential(nn.Conv1d(base*2, base, 1), nn.GroupNorm(4, base), nn.GELU())
        self.out = nn.Conv1d(base, 1, 1)
    def forward(self, x):
        e1 = self.e1(x)          # (B, base, 1000)
        e2 = self.e2(e1)         # (B, base*2, 500)
        e3 = self.e3(e2)         # (B, base*4, 250)
        b = self.bot(e3)         # (B, base*4, 250)
        d3 = self.d3(b)          # (B, base*2, 500)
        d3 = self.u3(torch.cat([d3, e2], dim=1))
        d2 = self.d2(d3)         # (B, base, 1000)
        d2 = self.u2(torch.cat([d2, e1], dim=1))
        return self.out(d2)


# ── 损失 ────────────────────────────────────────────────────────────

def wave_loss(logits, target, use_freq=False):
    """波形损失 + 可选 STFT 频域损失。"""
    pt = nn.functional.smooth_l1_loss(logits, target, reduction="none")
    w = 1.0 + 1.5 * torch.clamp(torch.abs(target) / 3.0, 0.0, 2.0)
    d = nn.functional.smooth_l1_loss(torch.diff(logits, dim=-1), torch.diff(target, dim=-1))
    loss = (pt * w).mean() + 0.15 * d
    if use_freq:
        # STFT 频域损失 (BiM-Diff 思路)
        Ll = torch.abs(torch.fft.rfft(logits, dim=-1))
        Lt = torch.abs(torch.fft.rfft(target, dim=-1))
        loss = loss + 0.1 * nn.functional.l1_loss(Ll, Lt)
    return loss


# ── 训练 ────────────────────────────────────────────────────────────

def train_one(name, model, x, y, tr, va, epochs, use_freq=False):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(x[tr]); Ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(x[va]); Yva = torch.from_numpy(y[va])
    bs = 64
    best_loss = float("inf"); best_sd = copy.deepcopy(model.state_dict())
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(Xtr.shape[0]); tl = []
        for s in range(0, Xtr.shape[0], bs):
            ix = idx[s:s+bs]
            opt.zero_grad(set_to_none=True)
            L = wave_loss(model(Xtr[ix]), Ytr[ix], use_freq); L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl.append(float(L.detach()))
        model.eval(); vl = []
        with torch.no_grad():
            for s in range(0, Xva.shape[0], bs):
                ix = slice(s, s+bs)
                vl.append(float(wave_loss(model(Xva[ix]), Yva[ix], use_freq)))
        r = {"epoch": ep, "train_loss": float(np.mean(tl)), "val_loss": float(np.mean(vl))}
        hist.append(r); print(f"    ep {ep}: {r}")
        if r["val_loss"] < best_loss:
            best_loss = r["val_loss"]; best_sd = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    return model, hist


def infer(model, x, idx):
    model.eval(); out = []
    X = torch.from_numpy(x[idx]); bs = 128
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            out.append(model(X[s:s+bs]).numpy())
    return np.concatenate(out, axis=0)


def metrics(target, pred):
    dc, mc, nr = [], [], []
    for t, p in zip(target[:, 0], pred[:, 0]):
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: continue
        dc.append(float(np.corrcoef(t, p)[0, 1]))
        nr.append(float(np.sqrt(np.mean((t - p) ** 2)) / (t.std() + 1e-8)))
        tz = (t - t.mean()) / (t.std() + 1e-8)
        pz = (p - p.mean()) / (p.std() + 1e-8)
        xc = sp.correlate(tz, pz, mode="full") / t.size
        lg = sp.correlation_lags(t.size, p.size, mode="full")
        mc.append(float(np.max(xc[np.abs(lg) <= 30])))  # ±300ms 容忍 PTT
    return {"median_direct_r": float(np.median(dc)),
            "mean_direct_r": float(np.mean(dc)),
            "median_maxlag_r": float(np.median(mc)),
            "median_nrmse": float(np.median(nr)), "n": len(dc)}


# ── 主流程 ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PPG->ECG deep waveform reconstruction (angle C)")
    ap.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    ap.add_argument("--out", type=Path, default=Path("experiments/preliminary/ppg_to_ecg_results.json"))
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    set_deterministic()
    root = a.data.resolve()

    print("Loading data...")
    w = load_all(root)
    d = prepare(w)
    tr, va, te = split(d["groups"])
    tr = tr[d["good"][tr]]  # 质量门控
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    x_ppg, y_ecg = d["ppg"], d["ecg"]

    experiments = [
        ("A_ppg_tcn_baseline",  TCN(2, width=16),    False),
        ("B_ppg_tcn_wider",     TCN(2, width=64),    False),
        ("C_ppg_dualinput",     DualInputTCN(width=32), False),
        ("D_ppg_tcn_freqloss",  TCN(2, width=32),    True),
        ("E_ppg_unet",          UNet1D(ci=2, base=16), False),
    ]

    results = {}
    for name, model, use_freq in experiments:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        model, hist = train_one(name, model, x_ppg, y_ecg, tr, va, a.epochs, use_freq)
        pred = infer(model, x_ppg, te)
        m_all = metrics(y_ecg[te], pred)
        te_clean = te[d["good"][te]]
        m_clean = metrics(y_ecg[te_clean], pred[d["good"][te]])
        print(f"  test_all:   direct_r={m_all['median_direct_r']:.4f} maxlag_r={m_all['median_maxlag_r']:.4f} nrmse={m_all['median_nrmse']:.3f}")
        print(f"  test_clean: direct_r={m_clean['median_direct_r']:.4f} maxlag_r={m_clean['median_maxlag_r']:.4f}")
        results[name] = {"test_all": m_all, "test_clean": m_clean,
                         "params": n_params, "history": hist}

    res = {
        "protocol": {"task": "PPG->ECG waveform reconstruction",
                     "baseline_reference": "deep_screen ppg_waveform_upper_bound direct_r=0.113",
                     "segment_seconds": 10, "sampling_hz": 100,
                     "split": "subject-disjoint 64/16/20", "epochs": a.epochs},
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "results": results,
    }
    write_json(a.out.resolve(), res)
    print(f"\n{'='*60}\nPPG->ECG Results (baseline ref: direct_r=0.113):\n")
    for name, r in results.items():
        print(f"  {name:24s}  direct_r={r['test_all']['median_direct_r']:.4f}  "
              f"maxlag_r={r['test_all']['median_maxlag_r']:.4f}  params={r['params']:>7,}")
    print(f"\nSaved to {a.out.resolve()}")


if __name__ == "__main__":
    main()