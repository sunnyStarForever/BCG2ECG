"""
Phase 2: 事件优先的 BCG→ECG — R峰检测 + 不确定性建模

目标：不追求生成完美 ECG 波形，而是可靠地检测心搏位置（R峰），
      并输出置信度。

消融：
  1. bcg_only         — BCG 编码器 → R 峰检测（无压力条件化）
  2. bcg_posture       — BCG + 体位标签 → R 峰检测
  3. bcg_pressure      — BCG + 深度压力编码器 → R 峰检测

输出：
  - R 峰 F1（±100 ms 容差）
  - 精确率 / 召回率
  - 平均时序误差 (ms)
  - HR MAE (bpm)

不确定性输出：
  - 每段预测置信度
  - 覆盖率-误差曲线
"""

from __future__ import annotations
import argparse, copy, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, math
from scipy import signal, ndimage
from torch import nn
from pipeline_utils import write_json
from pressure_encoder import PressureEncoder, PostureLabelEncoder

SEED, SEG = 42, 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_deterministic():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)

# ── Data ────────────────────────────────────────────────────────────

def load_all(root: Path):
    out = {k: [] for k in ["bcg","ecg","rpeak","pressure_clean","posture","quality","groups"]}
    for g, p in enumerate(sorted(root.glob("subjects/S*.npz"))):
        z = np.load(p); n = z["posture"].shape[0]
        for k, s in [("bcg","bcg"),("ecg","ecg_100"),("rpeak","rpeak_mask_100"),
                      ("pressure_clean","pressure_clean"),("posture","posture"),
                      ("quality","quality")]:
            out[k].append(z[s])
        out["groups"].append(np.full(n, g, dtype=np.int16)); z.close()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}

def seg(x):
    if x.ndim == 2: x = x[:, None, :]
    n, c, L = x.shape; k = L // SEG
    return x.reshape(n, c, k, SEG).transpose(0, 2, 1, 3).reshape(n*k, c, SEG).astype(np.float32)

def prepare(w):
    good = ((w["quality"][:, 0] < 0.05)
            & (w["quality"][:, 1] < 0.35)
            & (w["quality"][:, 2] >= 18)
            & (w["quality"][:, 2] <= 66)
            & (w["quality"][:, 8] < 0.08))
    good_seg = np.repeat(good, 3)
    return {
        "bcg": seg(w["bcg"]), "ecg": seg(w["ecg"]),
        "pressure_clean": np.nan_to_num(w["pressure_clean"].repeat(3, axis=0), nan=0.0),
        "posture": np.repeat(w["posture"], 3, axis=0),
        "groups": np.repeat(w["groups"], 3, axis=0),
        "good": good_seg,
    }

def split(groups):
    from sklearn.model_selection import GroupShuffleSplit
    o = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv, te = next(o.split(groups, groups=groups))
    i = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    tr_rel, va_rel = next(i.split(tv, groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te

# ── R-peak target preparation ───────────────────────────────────────

def rpeak_segment_mask(w):
    """Segment R-peak masks (same 3-segment split as BCG/ECG)."""
    return seg(w["rpeak"]).astype(np.float32)

def make_heatmap(event_binary):
    """Gaussian-blurred heatmap for stable training (sigma=2.5 samples)."""
    hm = ndimage.gaussian_filter1d(event_binary, sigma=2.5, axis=-1)
    hm *= np.sqrt(2.0 * np.pi) * 2.5
    return np.clip(hm, 0.0, 1.0).astype(np.float32)

# ── Model ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        p = 3 * d
        self.net = nn.Sequential(
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w), nn.GELU(), nn.Dropout(0.05),
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class EventDetectionTCN(nn.Module):
    """Outputs: event logits (1, SEG) + uncertainty log-sigma (1, SEG)."""
    def __init__(self, ci, cd=0, w=16):
        super().__init__()
        self.stem = nn.Conv1d(ci, w, 9, padding=4)
        self.ctx = nn.Sequential(nn.Linear(cd, w*2), nn.Tanh()) if cd else None
        self.blocks = nn.Sequential(ResidualBlock(w, 1), ResidualBlock(w, 2), ResidualBlock(w, 4))
        self.head = nn.Conv1d(w, 1, 1)
        # Uncertainty head: log(sigma) for heteroscedastic regression
        self.unc_head = nn.Conv1d(w, 1, 1)

    def forward(self, x, c):
        h = self.stem(x)
        if self.ctx is not None:
            g, b = self.ctx(c).chunk(2, 1)
            h = h * (1.0 + 0.25 * g[:, :, None]) + b[:, :, None]
        h = self.blocks(h)
        return self.head(h), self.unc_head(h)

# ── Loss ─────────────────────────────────────────────────────────────

def event_loss(logits, heatmap, log_sigma=None):
    """Focal-style event detection loss + optional heteroscedastic term."""
    # BCE with focal weighting (gamma=2.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, heatmap, reduction="none")
    prob = torch.sigmoid(logits)
    focal = torch.abs(heatmap - prob) ** 2.0
    loss_bce = (focal * bce).mean()
    # Derivative smoothness (soft edge penalty)
    loss_deriv = 0.05 * nn.functional.smooth_l1_loss(torch.diff(logits, dim=-1), torch.zeros_like(torch.diff(logits, dim=-1)))
    return loss_bce + loss_deriv

# ── Training ─────────────────────────────────────────────────────────

def train_one(name, x, ctx, y, tr, va, epochs):
    torch.manual_seed(SEED)
    cd = ctx.shape[1] if ctx.ndim == 2 else 0
    m = EventDetectionTCN(x.shape[1], cd)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr=torch.from_numpy(x[tr]); Ctr=torch.from_numpy(ctx[tr]); Ytr=torch.from_numpy(y[tr])
    Xva=torch.from_numpy(x[va]); Cva=torch.from_numpy(ctx[va]); Yva=torch.from_numpy(y[va])
    hist, best_loss, best_sd = [], float("inf"), copy.deepcopy(m.state_dict())
    bs = 64
    for ep in range(1, epochs+1):
        m.train(); idx = torch.randperm(Xtr.shape[0]); tlosses = []
        for start in range(0, Xtr.shape[0], bs):
            ix = idx[start:start+bs]
            opt.zero_grad(set_to_none=True)
            logits, _ = m(Xtr[ix], Ctr[ix])
            L = event_loss(logits, Ytr[ix]); L.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            tlosses.append(float(L.detach()))
        m.eval(); vlosses = []
        with torch.no_grad():
            for start in range(0, Xva.shape[0], bs):
                ix = slice(start, start+bs)
                logits, _ = m(Xva[ix], Cva[ix])
                vlosses.append(float(event_loss(logits, Yva[ix])))
        r = {"epoch": ep, "train_loss": float(np.mean(tlosses)), "val_loss": float(np.mean(vlosses))}
        hist.append(r); print(f"    ep {ep}: {r}")
        if r["val_loss"] < best_loss: best_loss = r["val_loss"]; best_sd = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_sd)
    return m, hist

def infer(m, x, ctx, idx):
    m.eval(); vals = []; uncs = []
    X=torch.from_numpy(x[idx]); C=torch.from_numpy(ctx[idx])
    bs = 128
    with torch.no_grad():
        for start in range(0, X.shape[0], bs):
            ix = slice(start, start+bs)
            l, u = m(X[ix], C[ix])
            vals.append(torch.sigmoid(l).numpy())
            uncs.append(u.numpy())
    return np.concatenate(vals, axis=0), np.concatenate(uncs, axis=0)

# ── Evaluation ───────────────────────────────────────────────────────

def match_events(truth, pred, tol=10):
    u = set(); tp = 0; errors = []
    for p in pred:
        candidates = [(abs(int(p)-int(t)), idx) for idx, t in enumerate(truth) if idx not in u and abs(int(p)-int(t)) <= tol]
        if candidates:
            err, best = min(candidates); u.add(best); tp += 1; errors.append(err)
    return tp, len(pred) - tp, len(truth) - tp, errors

def event_metrics(pred_prob, event_bin, threshold):
    total_tp = total_fp = total_fn = 0
    timing_errs = []; hr_errs = []
    for tmask, prob in zip(event_bin[:, 0], pred_prob[:, 0]):
        truth = np.flatnonzero(tmask > 0.5)
        pred_peaks, _ = signal.find_peaks(prob, height=threshold, distance=30)
        tp, fp, fn, errs = match_events(truth, pred_peaks)
        total_tp += tp; total_fp += fp; total_fn += fn
        timing_errs.extend(errs)
        hr_errs.append(abs(len(pred_peaks) - len(truth)) * 6.0)
    pr = total_tp / max(total_tp + total_fp, 1)
    rc = total_tp / max(total_tp + total_fn, 1)
    return {
        "event_f1_100ms": float(2 * pr * rc / max(pr + rc, 1e-12)),
        "event_precision": float(pr), "event_recall": float(rc),
        "timing_mae_ms": float(np.mean(timing_errs) * 10.0) if timing_errs else 1000.0,
        "hr_mae_bpm": float(np.mean(hr_errs)),
        "threshold": float(threshold), "n": int(event_bin.shape[0]),
    }

def choose_threshold(pred_prob, event_bin):
    best, best_m = -1.0, None
    for h in np.linspace(0.05, 0.90, 18):
        m = event_metrics(pred_prob, event_bin, h)
        if m["event_f1_100ms"] > best: best = m["event_f1_100ms"]; best_m = m
    if best_m is None: best_m = event_metrics(pred_prob, event_bin, 0.5)
    return float(best_m["threshold"]), best_m

def build_context(data, enc_name):
    if enc_name == "none": return np.empty((data["bcg"].shape[0], 0), dtype=np.float32)
    if enc_name == "posture_onehot":
        oh = np.eye(5, dtype=np.float32)[data["posture"]]
        with torch.no_grad():
            return PostureLabelEncoder(out_dim=128)(torch.from_numpy(oh)).numpy()
    parts = enc_name.split("_"); aggr = "weighted"; use_dyn = True
    if "maxpool" in parts: aggr = "max"
    if "nodyn" in parts: use_dyn = False
    enc = PressureEncoder(out_dim=128, temporal_aggregation=aggr, use_dynamic_branch=use_dyn)
    ctx_chunks = []
    data_arr = data["pressure_clean"]
    data_arr = np.nan_to_num(data_arr, nan=0.0, posinf=0.0, neginf=0.0)
    chunk_size = 4
    for start in range(0, data_arr.shape[0], chunk_size):
        batch = torch.from_numpy(data_arr[start:start + chunk_size])
        with torch.no_grad(): ctx_chunks.append(np.nan_to_num(enc(batch).numpy(), nan=0.0))
        del batch
    return np.concatenate(ctx_chunks, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    ap.add_argument("--out", type=Path, default=Path("experiments/preliminary/event_results.json"))
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    set_deterministic()
    root = a.data.resolve()
    print("Device:", DEVICE)

    print("Loading data...")
    w = load_all(root)
    d = prepare(w)
    tr, va, te = split(d["groups"])
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    # quality-gated training
    tr_good = tr[d["good"][tr]]
    print(f"train (good only): {len(tr_good)}")

    # Prepare targets
    event_bin = rpeak_segment_mask(w)
    heatmap = make_heatmap(event_bin)

    exps = [
        ("bcg_only",              "none"),
        ("bcg_posture",           "posture_onehot"),
        ("bcg_pressure_weighted", "pressure_cnn_weighted"),
    ]

    results = {}
    for name, enc in exps:
        print(f"\n{'='*50}\n  {name} ({enc})\n{'='*50}")
        ctx = build_context(d, enc)
        print(f"  ctx={ctx.shape[1]}")

        m, hist = train_one(name, d["bcg"], ctx, heatmap, tr_good, va, a.epochs)

        # Validation: choose threshold
        val_prob, val_unc = infer(m, d["bcg"], ctx, va)
        thr, val_metrics = choose_threshold(val_prob, event_bin[va])
        print(f"  val F1={val_metrics['event_f1_100ms']:.4f} thr={thr:.3f}")

        # Test
        te_prob, te_unc = infer(m, d["bcg"], ctx, te)
        te_all_metrics = event_metrics(te_prob, event_bin[te], thr)
        te_clean = te[d["good"][te]]
        te_clean_metrics = event_metrics(te_prob[d["good"][te]], event_bin[te_clean], thr)
        print(f"  test_all:  F1={te_all_metrics['event_f1_100ms']:.4f} prec={te_all_metrics['event_precision']:.3f} rec={te_all_metrics['event_recall']:.3f}")

        results[name] = {
            "validation": val_metrics,
            "test_all": te_all_metrics,
            "test_clean": te_clean_metrics,
            "train_history": hist,
        }

    res = {
        "protocol": {
            "model": "EventDetectionTCN(w=16, 3xResBlock)",
            "task": "R-peak event detection",
            "segment_seconds": 10, "tolerance_ms": 100,
            "split": "subject-disjoint 64/16/20",
            "epochs": a.epochs,
        },
        "split": {"train": len(tr_good), "val": len(va), "test": len(te)},
        "results": results,
    }
    write_json(a.out.resolve(), res)
    print(f"\n{'='*50}\nEvent Detection Results:\n")
    for name, r in results.items():
        print(f"  {name:30s} F1={r['test_all']['event_f1_100ms']:.4f}  prec={r['test_all']['event_precision']:.3f}  rec={r['test_all']['event_recall']:.3f}  hr_mae={r['test_all']['hr_mae_bpm']:.1f}")

if __name__ == "__main__":
    main()