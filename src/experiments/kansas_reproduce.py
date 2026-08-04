"""
kansas_reproduce.py — Kansas BCG→ECG 文献结果复现

目标: 在标准 1kHz Kansas 数据上逼近文献指标
  - PCC 0.896 (BiLSTM, Zhang 2024) ~ 0.984 (BiM-Diff, Zeng 2025)
  - RRI MAE 34ms (Morokuma 2025)

相对 kansas_bcg_to_ecg.py 的关键改进:
  1. 10s 短分段 (Morokuma 风格) — BiLSTM 可跑, 样本数 3x
  2. 多 BCG 通道融合 (4 Film + 最佳 LC), 而非单通道
  3. 更宽 TCN (width=64) + 更多 epoch (30)
  4. 延迟容忍评估 (搜索 ±100ms 最优时延, 对标 Med2ECG)
  5. 专门的事件检测评估 (BCG 包络峰 → RRI)

模型:
  A. TCN_wide       — 宽 TCN, 多通道输入
  B. BiLSTM_short   — BiLSTM 在 10s 段上 (对标 Morokuma)
"""
from __future__ import annotations

import argparse, copy, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent))

import numpy as np
import torch
from scipy import signal as sp
from torch import nn
from pipeline_utils import write_json
from kansas_loader import load_kansas, summarize_kansas

SEED = 42
SEG = 10000  # 10s @ 1000Hz (Morokuma 风格短分段)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_deterministic():
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(8)


def bandpass(x, fs, lo, hi, order=4):
    sos = sp.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sp.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def rscale(x):
    med = np.median(x); mad = np.median(np.abs(x - med))
    s = max(1.4826 * mad, 1e-8)
    return np.clip((x - med) / s, -12, 12).astype(np.float32)


def preprocess_record(rec, seg_sec=10.0):
    """多通道预处理 + 10s 切窗。

    BCG 通道 (5维):
      [0-3] 4 个 PVDF Film (0.5-40Hz 带通)
      [4]   最佳 LC_BCG (与 ECG 心率最匹配的称重通道, 0.5-40Hz)
    另加 [5] Film0 的 15-50Hz 整流包络 (Morokuma 心搏增强)
    """
    fs = rec["fs"]
    seg = int(seg_sec * fs)
    ecg_f = bandpass(rec["ecg"], fs, 0.5, 40.0)
    ecg_z = rscale(ecg_f)

    # 4 个 Film 通道
    films = [rscale(bandpass(f, fs, 0.5, 40.0)) for f in rec["bcg_film"]]
    # 选最佳 LC 通道
    f_e, pxx_e = sp.welch(ecg_f, fs=fs, nperseg=min(2048, len(ecg_f)))
    hr_mask = (f_e >= 0.7) & (f_e <= 3.0)
    ecg_hr_f = f_e[hr_mask][np.argmax(pxx_e[hr_mask])] if hr_mask.any() else 1.2
    best_lc, best_s = None, -1
    for lc in rec["bcg_lc"]:
        try:
            lc_f = bandpass(lc, fs, 0.5, 40.0)
            f_l, pxx_l = sp.welch(lc_f, fs=fs, nperseg=min(2048, len(lc_f)))
            lm = (f_l >= 0.7) & (f_l <= 3.0)
            hr_l = f_l[lm][np.argmax(pxx_l[lm])] if lm.any() else 0
            sc = -abs(hr_l - ecg_hr_f)
            if sc > best_s: best_s, best_lc = sc, lc_f
        except Exception:
            continue
    lc_z = rscale(best_lc) if best_lc is not None else films[0]
    # Film0 的整流包络 (心搏增强)
    film0_card = bandpass(rec["bcg_film"][0], fs, 15.0, 50.0)
    film0_env = rscale(bandpass(np.abs(film0_card), fs, 0.5, 10.0))

    # R 峰
    from pipeline_utils import detect_rpeaks
    peaks, _ = detect_rpeaks(rec["ecg"], fs=int(fs))
    rmask = np.zeros(len(ecg_z), dtype=np.uint8)
    rmask[peaks] = 1

    # 切窗
    n = len(ecg_z) // seg
    if n == 0: return {"n_windows": 0}
    def win(arr): return arr[:n*seg].reshape(n, seg)
    bcg = np.stack([win(films[i]) for i in range(4)] + [win(lc_z)] + [win(film0_env)], axis=1)
    return {
        "n_windows": n,
        "bcg": bcg.astype(np.float32),          # (n, 6, seg)
        "ecg": win(ecg_z)[:, None, :].astype(np.float32),  # (n, 1, seg)
        "rpeak_mask": win(rmask),
        "subject_id": rec["subject_id"],
    }


def build_dataset(records):
    B, E, R, G = [], [], [], []
    for g, rec in enumerate(records):
        p = preprocess_record(rec)
        if p["n_windows"] == 0: continue
        B.append(p["bcg"]); E.append(p["ecg"]); R.append(p["rpeak_mask"])
        G.append(np.full(p["n_windows"], g, dtype=np.int16))
    return {"bcg": np.concatenate(B), "ecg": np.concatenate(E),
            "rpeak_mask": np.concatenate(R), "groups": np.concatenate(G)}


# ── 模型 ──
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
    def __init__(self, ci, width=64):
        super().__init__()
        self.stem = nn.Conv1d(ci, width, 9, padding=4)
        self.blocks = nn.Sequential(ResBlock(width,1), ResBlock(width,2),
                                    ResBlock(width,4), ResBlock(width,8))
        self.head = nn.Conv1d(width, 1, 1)
    def forward(self, x): return self.head(self.blocks(self.stem(x)))


class BiLSTM(nn.Module):
    def __init__(self, ci, hidden=128, layers=2):
        super().__init__()
        self.proj = nn.Conv1d(ci, 16, 1)
        self.lstm = nn.LSTM(16, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden*2, 1)
    def forward(self, x):
        h = self.proj(x).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out).transpose(1, 2)


def wave_loss(logits, target):
    pt = nn.functional.smooth_l1_loss(logits, target, reduction="none")
    w = 1.0 + 1.5 * torch.clamp(torch.abs(target)/3.0, 0.0, 2.0)
    d = nn.functional.smooth_l1_loss(torch.diff(logits, dim=-1), torch.diff(target, dim=-1))
    return (pt*w).mean() + 0.15*d


def train(model, x, y, tr, va, epochs, lr=1e-3, bs=64):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr=torch.from_numpy(x[tr]).to(DEVICE); Ytr=torch.from_numpy(y[tr]).to(DEVICE)
    Xva=torch.from_numpy(x[va]).to(DEVICE); Yva=torch.from_numpy(y[va]).to(DEVICE)
    best=float("inf"); best_sd=copy.deepcopy(model.state_dict())
    for ep in range(1, epochs+1):
        model.train(); idx=torch.randperm(Xtr.shape[0]); tl=[]
        for s in range(0, Xtr.shape[0], bs):
            ix=idx[s:s+bs]; opt.zero_grad(set_to_none=True)
            L=wave_loss(model(Xtr[ix]), Ytr[ix]); L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl.append(float(L.detach()))
            if DEVICE=="cuda": torch.cuda.empty_cache()
        model.eval(); vl=[]
        with torch.no_grad():
            for s in range(0, Xva.shape[0], bs):
                ix=slice(s,s+bs); vl.append(float(wave_loss(model(Xva[ix]), Yva[ix])))
        vloss=float(np.mean(vl))
        print(f"    ep{ep}: tr={np.mean(tl):.4f} va={vloss:.4f}", flush=True)
        if vloss<best: best=vloss; best_sd=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); return model


def infer(model, x, idx, bs=128):
    model.eval(); out=[]
    X=torch.from_numpy(x[idx]).to(DEVICE)
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            out.append(model(X[s:s+bs]).cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics_delay_tolerant(target, pred, max_lag_ms=100, fs=1000):
    """延迟容忍 PCC: 对每段搜索 ±max_lag_ms 最优时延后算 PCC (对标 Med2ECG)。"""
    maxlag = int(max_lag_ms * fs / 1000)
    direct, bestlag, nrmse = [], [], []
    for t, p in zip(target[:,0], pred[:,0]):
        if np.std(t)<1e-6 or np.std(p)<1e-6: continue
        direct.append(float(np.corrcoef(t,p)[0,1]))
        tz=(t-t.mean())/(t.std()+1e-8); pz=(p-p.mean())/(p.std()+1e-8)
        nrmse.append(float(np.sqrt(np.mean((t-p)**2))/(t.std()+1e-8)))
        xc=sp.correlate(tz, pz, mode="full")/t.size
        lg=sp.correlation_lags(t.size, p.size, mode="full")
        bestlag.append(float(np.max(xc[np.abs(lg)<=maxlag])))
    return {"median_pcc": float(np.median(direct)), "mean_pcc": float(np.mean(direct)),
            "median_pcc_delay_tolerant": float(np.median(bestlag)),
            "median_nrmse": float(np.median(nrmse)), "n": len(direct)}


def rri_from_pred(pred, fs=1000):
    """从预测波形提取 RRI (用 5-25Hz 带通 + 峰检测, 模拟伪ECG的R峰)。"""
    rris = []
    for p in pred[:,0]:
        bp = sp.filtfilt(*sp.butter(3, [5,25], btype="bandpass", fs=fs), p)
        pk, _ = sp.find_peaks(bp, distance=int(0.3*fs), prominence=np.std(bp)*1.5)
        if len(pk)>=2: rris.append(np.diff(pk)/fs)
        else: rris.append(None)
    return rris


def rri_metrics(target_mask, pred, fs=1000):
    """对标 Morokuma 的 RRI MAE。"""
    true_rris = []
    for tm in target_mask:
        tp = np.flatnonzero(tm>0.5)
        if len(tp)>=2: true_rris.append(np.diff(tp)/fs)
        else: true_rris.append(None)
    pred_rris = rri_from_pred(pred, fs)
    errs = []
    for tr, pr in zip(true_rris, pred_rris):
        if tr is None or pr is None: continue
        n = min(len(tr), len(pr))
        if n<1: continue
        errs.extend(np.abs(tr[:n]-pr[:n])*1000)
    if not errs: return {"rri_mae_ms": 1000.0, "n_beats": 0}
    return {"rri_mae_ms": float(np.mean(errs)), "rri_median_mae_ms": float(np.median(errs)),
            "n_beats": len(errs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/kansas_bcg"))
    ap.add_argument("--out", type=Path, default=Path("experiments/preliminary/kansas_reproduce.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-loso", type=int, default=5)
    ap.add_argument("--models", default="tcn,bilstm")
    a = ap.parse_args()
    set_deterministic()

    summ = summarize_kansas(a.data.resolve())
    if not summ.get("available"):
        print("[!] 数据未就绪"); return
    print("\nLoading Kansas...", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    records = load_kansas(a.data.resolve())
    print("\nPreprocessing (10s segments, multi-channel BCG)...", flush=True)
    data = build_dataset(records)
    print(f"Windows: {data['bcg'].shape[0]}, subjects: {np.unique(data['groups']).size}", flush=True)
    print(f"BCG: {data['bcg'].shape} (6 ch), ECG: {data['ecg'].shape}", flush=True)

    from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
    splits = list(LeaveOneGroupOut().split(np.zeros(len(data['groups'])),
                                            np.zeros(len(data['groups'])),
                                            data['groups']))[:a.max_loso]
    print(f"\nLOSO: first {len(splits)} folds", flush=True)

    models_to_run = a.models.split(",")
    all_results = {m: [] for m in models_to_run}
    for fold, (tr, te) in enumerate(splits):
        ts = int(np.unique(data['groups'][te])[0])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        tr_rel, va_rel = next(gss.split(tr, groups=data['groups'][tr]))
        tr_idx, va_idx = tr[tr_rel], tr[va_rel]
        print(f"\n{'='*55}\nFold {fold+1}/{len(splits)}: test subj {ts} "
              f"(tr={len(tr_idx)} va={len(va_idx)} te={len(te)})\n{'='*55}", flush=True)
        for mname in models_to_run:
            print(f"  -- {mname} --", flush=True)
            model = TCN(6, width=64) if mname=="tcn" else BiLSTM(6, hidden=64, layers=1)
            try:
                # BiLSTM 用小 batch 防 OOM
                bs = 64 if mname == "tcn" else 8
                model = train(model, data['bcg'], data['ecg'], tr_idx, va_idx, a.epochs, bs=bs)
                pred = infer(model, data['bcg'], te)
                wm = metrics_delay_tolerant(data['ecg'][te], pred)
                rm = rri_metrics(data['rpeak_mask'][te], pred)
                print(f"    PCC={wm['median_pcc']:.4f} (delay-tol {wm['median_pcc_delay_tolerant']:.4f}) "
                      f"RRI_MAE={rm['rri_mae_ms']:.1f}ms", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"    [SKIP {mname}] OOM: {str(e)[:60]}", flush=True)
                if DEVICE == "cuda": torch.cuda.empty_cache()
                continue
            all_results[mname].append({"fold": fold, "test_subject": ts, **wm, **rm})

    summary = {}
    for m, fr in all_results.items():
        if not fr: continue
        summary[m] = {
            "median_pcc": float(np.median([r["median_pcc"] for r in fr])),
            "median_pcc_delay_tolerant": float(np.median([r["median_pcc_delay_tolerant"] for r in fr])),
            "median_rri_mae_ms": float(np.median([r["rri_mae_ms"] for r in fr])),
            "n_folds": len(fr), "folds": fr,
        }
    res = {"protocol": {"dataset": "Kansas", "segment_sec": 10, "fs_hz": 1000,
                        "bcg_channels": "4 Film + best LC + Film0 envelope (6)",
                        "references": {"BiLSTM_PCC": 0.896, "BiM-Diff_PCC": 0.984,
                                       "Morokuma_RRI_MAE_ms": 34},
                        "epochs": a.epochs, "split": "LOSO"},
           "results": summary}
    write_json(a.out.resolve(), res)
    print(f"\n{'='*55}\nKansas Reproduce Results (文献对标):\n")
    for m, r in summary.items():
        print(f"  {m:8s}: PCC={r['median_pcc']:.4f} (delay-tol {r['median_pcc_delay_tolerant']:.4f}) "
              f"RRI_MAE={r['median_rri_mae_ms']:.1f}ms")
    print(f"  [ref]   PCC=0.896~0.984, RRI_MAE=34ms")
    print(f"\nSaved to {a.out.resolve()}")


if __name__ == "__main__":
    main()