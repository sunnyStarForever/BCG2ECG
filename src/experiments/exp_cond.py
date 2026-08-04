"""压力条件化 BCG→ECG 重建 — 消融实验"""
from __future__ import annotations

import argparse, copy, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, os, math
os.environ["TORCH_VITAL"] = "0"  # fix torch 2.10 set_vital issue
from scipy import signal
from torch import nn
from pipeline_utils import write_json
from pressure_encoder import PressureEncoder, HandcraftedPressureEncoder, PostureLabelEncoder

SEED, SEG = 42, 1000

def set_deterministic():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)

def load_all(root: Path):
    """Load subjects with pressure_clean."""
    out = {k: [] for k in ["bcg","ecg","ppg","rpeak","pressure_clean","posture","quality","groups"]}
    for g, p in enumerate(sorted(root.glob("subjects/S*.npz"))):
        z = np.load(p)
        n = z["posture"].shape[0]
        out["bcg"].append(z["bcg"])
        out["ecg"].append(z["ecg_100"])
        out["ppg"].append(z["ppg"])
        out["rpeak"].append(z["rpeak_mask_100"])
        out["pressure_clean"].append(z["pressure_clean"])
        out["posture"].append(z["posture"])
        out["quality"].append(z["quality"])
        out["groups"].append(np.full(n, g, dtype=np.int16))
        z.close()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}

def seg(x: np.ndarray):
    if x.ndim == 2: x = x[:, None, :]
    n, c, L = x.shape
    k = L // SEG
    return x.reshape(n, c, k, SEG).transpose(0, 2, 1, 3).reshape(n * k, c, SEG).astype(np.float32)

def prepare(w):
    return {
        "bcg": seg(w["bcg"]),
        "ecg": seg(w["ecg"]),
        "ppg": seg(w["ppg"]),
        "pressure_clean": w["pressure_clean"].repeat(3, axis=0),
        "posture": np.repeat(w["posture"], 3, axis=0),
        "groups": np.repeat(w["groups"], 3, axis=0),
    }

def split(groups):
    from sklearn.model_selection import GroupShuffleSplit
    o = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv, te = next(o.split(groups, groups=groups))
    i = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    tr_rel, va_rel = next(i.split(tv, groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te

# ── Model ──
class ResidualBlock(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        p = 3 * d
        self.net = nn.Sequential(
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w), nn.GELU(), nn.Dropout(0.05),
            nn.Conv1d(w, w, 7, padding=p, dilation=d), nn.GroupNorm(4, w))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class TinyTCN(nn.Module):
    def __init__(self, ci, cd=0, w=16):
        super().__init__()
        self.stem = nn.Conv1d(ci, w, 9, padding=4)
        self.ctx = nn.Sequential(nn.Linear(cd, w*2), nn.Tanh()) if cd else None
        self.blocks = nn.Sequential(ResidualBlock(w, 1), ResidualBlock(w, 2), ResidualBlock(w, 4))
        self.head = nn.Conv1d(w, 1, 1)
    def forward(self, x, c):
        h = self.stem(x)
        if self.ctx is not None:
            g, b = self.ctx(c).chunk(2, 1)
            h = h * (1.0 + 0.25 * g[:, :, None]) + b[:, :, None]
        return self.head(self.blocks(h))

# ── Training ──
def wave_loss(logits, target):
    pt = nn.functional.smooth_l1_loss(logits, target, reduction="none")
    w = 1.0 + 1.5 * torch.clamp(torch.abs(target) / 3.0, 0.0, 2.0)
    d = nn.functional.smooth_l1_loss(torch.diff(logits, dim=-1), torch.diff(target, dim=-1))
    return (pt * w).mean() + 0.15 * d

def train_one(name, x, ctx, y, tr, va, epochs):
    """Manual minibatch training — no DataLoader for torch 2.10 compat."""
    torch.manual_seed(SEED)
    cd = ctx.shape[1] if ctx.ndim == 2 else 0
    m = TinyTCN(x.shape[1], cd)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(x[tr]); Ctr = torch.from_numpy(ctx[tr]); Ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(x[va]); Cva = torch.from_numpy(ctx[va]); Yva = torch.from_numpy(y[va])
    hist, best_loss, best_sd = [], float("inf"), copy.deepcopy(m.state_dict())
    bs = 64
    for ep in range(1, epochs+1):
        m.train()
        idx = torch.randperm(Xtr.shape[0])
        tlosses = []
        for start in range(0, Xtr.shape[0], bs):
            ix = idx[start:start+bs]
            opt.zero_grad(set_to_none=True)
            L = wave_loss(m(Xtr[ix], Ctr[ix]), Ytr[ix]); L.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            tlosses.append(float(L.detach()))
        m.eval()
        vlosses = []
        with torch.no_grad():
            for start in range(0, Xva.shape[0], bs):
                ix = slice(start, start+bs)
                vlosses.append(float(wave_loss(m(Xva[ix], Cva[ix]), Yva[ix])))
        r = {"epoch": ep, "train_loss": float(np.mean(tlosses)), "val_loss": float(np.mean(vlosses))}
        hist.append(r); print(f"    ep {ep}: {r}")
        if r["val_loss"] < best_loss: best_loss = r["val_loss"]; best_sd = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_sd)
    return m, hist

def infer(m, x, ctx, idx):
    m.eval(); vals = []
    X = torch.from_numpy(x[idx]); C = torch.from_numpy(ctx[idx])
    bs = 128
    with torch.no_grad():
        for start in range(0, X.shape[0], bs):
            ix = slice(start, start+bs)
            vals.append(m(X[ix], C[ix]).numpy())
    return np.concatenate(vals, axis=0)

def metrics(target, pred):
    dc, mc, nr = [], [], []
    for t, p in zip(target[:, 0], pred[:, 0]):
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: continue
        dc.append(float(np.corrcoef(t, p)[0, 1]))
        tz = (t - t.mean()) / (t.std() + 1e-8)
        pz = (p - p.mean()) / (p.std() + 1e-8)
        nr.append(float(np.sqrt(np.mean((t - p) ** 2)) / (t.std() + 1e-8)))
        cr = signal.correlate(tz, pz, mode="full") / t.size
        lg = signal.correlation_lags(t.size, p.size, mode="full")
        mc.append(float(np.max(cr[np.abs(lg) <= 12])))
    return {"median_direct_r": float(np.median(dc)), "mean_direct_r": float(np.mean(dc)),
            "median_maxlag_r": float(np.median(mc)), "median_nrmse": float(np.median(nr)),
            "n": len(dc)}

def build_context(data, enc_name):
    if enc_name == "none": return np.empty((data["bcg"].shape[0], 0), dtype=np.float32)
    if enc_name == "posture_onehot":
        oh = np.eye(5, dtype=np.float32)[data["posture"]]
        with torch.no_grad():
            return PostureLabelEncoder(out_dim=128)(torch.from_numpy(oh)).numpy()
    # pressure_cnn variants — process in chunks to avoid OOM
    parts = enc_name.split("_"); aggr = "weighted"; use_dyn = True
    if "maxpool" in parts: aggr = "max"  # TemporalAggregation string value
    if "nodyn" in parts: use_dyn = False
    with torch.no_grad():
        enc = PressureEncoder(out_dim=128, temporal_aggregation=aggr,
                              use_dynamic_branch=use_dyn)
    data_arr = data["pressure_clean"]  # (N, 30, 77, 32)
    # Replace NaN/inf with 0 (windows without valid pressure)
    data_arr = np.nan_to_num(data_arr, nan=0.0, posinf=0.0, neginf=0.0)
    ctx_chunks = []
    chunk_size = 4  # small: 4*30=120 frames at once through the CNN
    for start in range(0, data_arr.shape[0], chunk_size):
        batch = torch.from_numpy(data_arr[start:start + chunk_size])
        with torch.no_grad():
            ctx_chunks.append(np.nan_to_num(enc(batch).numpy(), nan=0.0))
        del batch
    return np.concatenate(ctx_chunks, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/v1"))
    ap.add_argument("--out", type=Path, default=Path("experiments/preliminary/conditioned_results.json"))
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    set_deterministic()
    root = a.data.resolve()

    print("Loading data...")
    w = load_all(root)
    d = prepare(w)
    tr, va, te = split(d["groups"])
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    exps = [
        ("bcg_only",              "none"),
        ("bcg_posture_onehot",    "posture_onehot"),
        ("bcg_pressure_weighted", "pressure_cnn_weighted"),
    ]
    rows = []; histories = {}
    for name, enc in exps:
        print(f"\n{'='*50}\n  {name} ({enc})\n{'='*50}")
        ctx = build_context(d, enc)
        print(f"  ctx dims = {ctx.shape[1]}")
        m, hist = train_one(name, d["bcg"], ctx, d["ecg"], tr, va, a.epochs)
        histories[name] = hist
        pred = infer(m, d["bcg"], ctx, te)
        mets = metrics(d["ecg"][te], pred)
        print(f"  test: dir_r={mets['median_direct_r']:.4f} lag_r={mets['median_maxlag_r']:.4f} nrmse={mets['median_nrmse']:.3f}")
        rows.append({"experiment": name, "encoder": enc, **mets,
                     "val_loss": min(r["val_loss"] for r in hist)})

    res = {"protocol": {"model": "TinyTCN(w=16, 3×Block)", "segment_seconds": 10,
                         "sampling_hz": 100, "split": "subject-disjoint 64/16/20",
                         "epochs": a.epochs},
           "split": {"train": len(tr), "val": len(va), "test": len(te)},
           "ablation": rows, "history": histories}
    write_json(a.out.resolve(), res)
    print(f"\n{'='*50}\nResults:\n")
    for r in rows:
        print(f"  {r['experiment']:30s} dir_r={r['median_direct_r']:.4f}  lag_r={r['median_maxlag_r']:.4f}  nrmse={r['median_nrmse']:.3f}")
    print(f"\nSaved to {a.out.resolve()}")

if __name__ == "__main__":
    main()