"""
kansas_bcg_to_ecg.py — Kansas Bed-BCG 数据集上的 BCG→ECG 基线实验

目标：在标准 1kHz BCG 数据上验证方法架构，与文献直接对标。

对标指标:
  - 波形 PCC (对标 BiM-Diff 0.984, BiLSTM 0.896)
  - RRI MAE (对标 Morokuma 34ms 短时 / 46ms 整夜)
  - HR MAE
  - R峰 F1 (±100ms)

实验 (递进):
  A. bcg_tcn_baseline     — TCN 基线 (复现我们用的 TinyTCN)
  B. bcg_bilstm           — BiLSTM (对标 Zhang 2024 / Morokuma 2025)
  C. bcg_tcn_event_first  — 事件优先 (R峰检测头 + 波形头, 对标 Morokuma 思路)

协议:
  - LOSO (留一受试者交叉验证), 与 Morokuma 一致
  - 30秒窗口, 1000Hz 原始采样 (不下采样, 这是与我们的数据的关键区别)
  - 质量门控可选

注: 数据未到位前, 本脚本会在加载阶段给出清晰的提示并退出。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
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


# ── 数据预处理 (Kansas 专用) ────────────────────────────────────────

def set_deterministic():
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)


def bandpass(x, fs, lo, hi, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def preprocess_kansas_record(rec: dict, win_sec: float = 30.0) -> dict:
    """预处理单条 Kansas 记录 → 切窗。

    Kansas 数据 1000Hz, ECG + 4 个 PVDF Film BCG + 4 个 LC_BCG 同步。
    策略: 用 4 个 Film 传感器 (高频心搏敏感) + 1 个最佳 LC_BCG (低频),
          自动选取与 ECG 心率最一致的电影片通道作为主 BCG。

    返回切窗后的数组:
      bcg (N, 2, win)  通道0=Film(最佳), 通道1=15-50Hz整流包络
      ecg (N, 1, win), rpeak_mask (N, win), hr (N,)
    """
    fs = rec["fs"]
    ecg_raw = rec["ecg"]
    win = int(win_sec * fs)

    # 预处理 ECG
    ecg_f = bandpass(ecg_raw, fs, 0.5, 40.0)

    # 选取最佳 Film 通道: 与 ECG 心搏频谱最匹配
    from scipy.signal import welch
    f_e, pxx_e = welch(ecg_f, fs=fs, nperseg=min(2048, len(ecg_f)))
    ecg_hr_freq = f_e[np.argmax(pxx_e[(f_e>=0.7)&(f_e<=3.0)] + 1e-20)] if (f_e>=0.7).any() else 1.2
    best_bcg, best_score = None, -1
    for film in rec["bcg_film"]:
        try:
            bcg_card = bandpass(film, fs, 15.0, 50.0)
            bcg_env = np.abs(bcg_card)
            bcg_env = bandpass(bcg_env, fs, 0.5, 10.0)
            f_b, pxx_b = welch(bcg_env, fs=fs, nperseg=min(2048, len(bcg_env)))
            hr_freq = f_b[np.argmax(pxx_b[(f_b>=0.7)&(f_b<=3.0)] + 1e-20)] if (f_b>=0.7).any() else 0
            score = -abs(hr_freq - ecg_hr_freq)  # 心率频率越接近越好
            if score > best_score:
                best_score, best_bcg = score, film
        except Exception:
            continue
    if best_bcg is None:
        best_bcg = rec["bcg_film"][0]

    bcg_f = bandpass(best_bcg, fs, 0.5, 40.0)
    bcg_card = bandpass(best_bcg, fs, 15.0, 50.0)
    bcg_env = np.abs(bcg_card)
    bcg_env = bandpass(bcg_env, fs, 0.5, 10.0)
    bcg_env = bandpass(bcg_env, fs, 0.5, 10.0)  # 平滑包络

    # Robust scale per-record
    def rscale(x):
        med = np.median(x); mad = np.median(np.abs(x - med))
        s = max(1.4826 * mad, 1e-8)
        return np.clip((x - med) / s, -12, 12).astype(np.float32)

    ecg_z = rscale(ecg_f)
    bcg_ch0 = rscale(bcg_f)
    bcg_ch1 = rscale(bcg_env)

    # R 峰检测 (用 pipeline_utils 的检测器, 但适配 1000Hz)
    from pipeline_utils import detect_rpeaks
    peaks, _ = detect_rpeaks(ecg_raw, fs=int(fs))
    rmask = np.zeros(len(ecg_z), dtype=np.uint8)
    rmask[peaks] = 1

    # 切窗
    n_win = len(ecg_z) // win
    if n_win == 0:
        return {"n_windows": 0}
    ecg_w = ecg_z[:n_win*win].reshape(n_win, win)
    bcg_ch0_w = bcg_ch0[:n_win*win].reshape(n_win, win)
    bcg_ch1_w = bcg_ch1[:n_win*win].reshape(n_win, win)
    rmask_w = rmask[:n_win*win].reshape(n_win, win)
    bcg_w = np.stack([bcg_ch0_w, bcg_ch1_w], axis=1)  # (N, 2, win)

    # 每窗 HR (从 R 峰)
    hr = np.array([rmask_w[i].sum() * 60.0 / win_sec for i in range(n_win)],
                  dtype=np.float32)

    return {
        "n_windows": n_win,
        "bcg": bcg_w.astype(np.float32),       # (N, 2, win)
        "ecg": ecg_w[:, None, :].astype(np.float32),  # (N, 1, win)
        "rpeak_mask": rmask_w,                  # (N, win)
        "hr": hr,
        "subject_id": rec["subject_id"],
    }


def build_dataset(records: list[dict]) -> dict:
    """把所有受试者的切窗数据拼成一个大数组。"""
    all_bcg, all_ecg, all_rmk, all_hr, all_groups = [], [], [], [], []
    for g, rec in enumerate(records):
        proc = preprocess_kansas_record(rec)
        if proc["n_windows"] == 0:
            continue
        all_bcg.append(proc["bcg"])
        all_ecg.append(proc["ecg"])
        all_rmk.append(proc["rpeak_mask"])
        all_hr.append(proc["hr"])
        all_groups.append(np.full(proc["n_windows"], g, dtype=np.int16))
    return {
        "bcg": np.concatenate(all_bcg, axis=0),
        "ecg": np.concatenate(all_ecg, axis=0),
        "rpeak_mask": np.concatenate(all_rmk, axis=0),
        "hr": np.concatenate(all_hr, axis=0),
        "groups": np.concatenate(all_groups, axis=0),
    }


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
    def __init__(self, ci, width=32):
        super().__init__()
        self.stem = nn.Conv1d(ci, width, 9, padding=4)
        self.blocks = nn.Sequential(ResBlock(width, 1), ResBlock(width, 2), ResBlock(width, 4), ResBlock(width, 8))
        self.head = nn.Conv1d(width, 1, 1)
    def forward(self, x): return self.head(self.blocks(self.stem(x)))


class BiLSTM(nn.Module):
    """对标 Zhang 2024 / Morokuma 2025 的 BiLSTM 回归。"""
    def __init__(self, ci=2, hidden=128, layers=2):
        super().__init__()
        self.proj = nn.Conv1d(ci, 16, 1)  # 降维到 16 通道减少 LSTM 计算量
        self.lstm = nn.LSTM(16, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, x):
        # x: (B, C, L) → (B, L, 16)
        h = self.proj(x).transpose(1, 2)
        out, _ = self.lstm(h)  # (B, L, 2*hidden)
        return self.head(out).transpose(1, 2)  # (B, 1, L)


# ── 训练与评估 ──────────────────────────────────────────────────────

def wave_loss(logits, target):
    pt = nn.functional.smooth_l1_loss(logits, target, reduction="none")
    w = 1.0 + 1.5 * torch.clamp(torch.abs(target) / 3.0, 0.0, 2.0)
    d = nn.functional.smooth_l1_loss(torch.diff(logits, dim=-1), torch.diff(target, dim=-1))
    return (pt * w).mean() + 0.15 * d


def train_model(model, x, y, tr_idx, va_idx, epochs, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr = torch.from_numpy(x[tr_idx]); Ytr = torch.from_numpy(y[tr_idx])
    Xva = torch.from_numpy(x[va_idx]); Yva = torch.from_numpy(y[va_idx])
    bs = 8  # 1000Hz×30s=30000 采样点，batch 小一些防 OOM
    best = float("inf"); best_sd = copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(Xtr.shape[0]); tl = []
        for s in range(0, Xtr.shape[0], bs):
            ix = idx[s:s+bs]
            opt.zero_grad(set_to_none=True)
            L = wave_loss(model(Xtr[ix]), Ytr[ix]); L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl.append(float(L.detach()))
        model.eval(); vl = []
        with torch.no_grad():
            for s in range(0, Xva.shape[0], bs):
                ix = slice(s, s+bs)
                vl.append(float(wave_loss(model(Xva[ix]), Yva[ix])))
        vloss = float(np.mean(vl))
        print(f"    ep {ep}: train={np.mean(tl):.4f} val={vloss:.4f}")
        if vloss < best: best = vloss; best_sd = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    return model


def infer(model, x, idx, bs=8):
    model.eval(); out = []
    X = torch.from_numpy(x[idx])
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            out.append(model(X[s:s+bs]).numpy())
    return np.concatenate(out, axis=0)


def waveform_metrics(target, pred):
    dc, mc, nr = [], [], []
    for t, p in zip(target[:, 0], pred[:, 0]):
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: continue
        dc.append(float(np.corrcoef(t, p)[0, 1]))
        nr.append(float(np.sqrt(np.mean((t - p) ** 2)) / (t.std() + 1e-8)))
        tz = (t - t.mean()) / (t.std() + 1e-8)
        pz = (p - p.mean()) / (p.std() + 1e-8)
        xc = sp.correlate(tz, pz, mode="full") / t.size
        lg = sp.correlation_lags(t.size, p.size, mode="full")
        mc.append(float(np.max(xc[np.abs(lg) <= 50])))  # ±50ms @1kHz
    return {"median_pcc": float(np.median(dc)), "mean_pcc": float(np.mean(dc)),
            "median_maxlag_pcc": float(np.median(mc)),
            "median_nrmse": float(np.median(nr)), "n": len(dc)}


def rri_metrics(target_mask, pred_prob, threshold, fs=1000):
    """RRI MAE 评估 (对标 Morokuma 34ms)。"""
    from scipy.signal import find_peaks
    rri_true_all, rri_pred_all = [], []
    for tm, prob in zip(target_mask, pred_prob[:, 0]):
        t_peaks = np.flatnonzero(tm > 0.5)
        p_peaks, _ = find_peaks(prob, height=threshold, distance=int(0.3*fs))
        if len(t_peaks) < 2 or len(p_peaks) < 2:
            continue
        t_rri = np.diff(t_peaks) / fs
        p_rri = np.diff(p_peaks) / fs
        # 取最小长度对齐
        n = min(len(t_rri), len(p_rri))
        rri_true_all.extend(t_rri[:n])
        rri_pred_all.extend(p_rri[:n])
    if not rri_true_all:
        return {"rri_mae_ms": 1000.0, "rri_corr": 0.0, "n_beats": 0}
    rri_true = np.array(rri_true_all); rri_pred = np.array(rri_pred_all)
    err = np.abs(rri_true - rri_pred) * 1000  # to ms
    corr = float(np.corrcoef(rri_true, rri_pred)[0, 1]) if len(rri_true) > 2 else 0.0
    return {"rri_mae_ms": float(np.mean(err)), "rri_median_mae_ms": float(np.median(err)),
            "rri_corr": corr, "n_beats": len(rri_true)}


def loso_splits(groups):
    """生成 LOSO 的 (train_idx, test_idx) 列表。"""
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(groups)), np.zeros(len(groups)), groups))


# ── 主流程 ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Kansas Bed-BCG → ECG baseline experiments")
    ap.add_argument("--data", type=Path, default=Path("data/kansas_bcg"))
    ap.add_argument("--out", type=Path, default=Path("experiments/preliminary/kansas_results.json"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-loso", type=int, default=5,
                    help="LOSO 折数上限 (40 折太慢, 默认前 5 折做初步验证)")
    a = ap.parse_args()
    set_deterministic()

    # 先探测数据集是否到位
    summary = summarize_kansas(a.data.resolve())
    print(f"Kansas dataset status: {summary}")
    if not summary.get("available"):
        print("\n[!] 数据集尚未下载。请按 README 指引下载到 data/kansas_bcg/ 后重运行。")
        print("    下载完成后, 运行: python src/experiments/kansas_bcg_to_ecg.py")
        return

    # 加载
    print("\nLoading Kansas dataset...")
    records = load_kansas(a.data.resolve())
    if not records:
        print("[!] 没有加载到有效受试者。请检查数据格式。")
        return

    print("\nPreprocessing (30s windows @ native fs)...")
    data = build_dataset(records)
    print(f"Total windows: {data['bcg'].shape[0]}, subjects: {np.unique(data['groups']).size}")
    print(f"BCG shape: {data['bcg'].shape}, ECG shape: {data['ecg'].shape}")

    # LOSO (限制折数加速)
    splits = loso_splits(data["groups"])[:a.max_loso]
    print(f"\nRunning LOSO on first {len(splits)} folds (of {np.unique(data['groups']).size} total)")

    results = {"A_tcn": [], "B_bilstm": []}
    for fold, (tr, te) in enumerate(splits):
        test_subj = np.unique(data["groups"][te])[0]
        # 用训练集再切 10% 作验证
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        tr_rel, va_rel = next(gss.split(tr, groups=data["groups"][tr]))
        tr_idx, va_idx = tr[tr_rel], tr[va_rel]
        print(f"\n{'='*50}\nFold {fold+1}/{len(splits)}: test subject {test_subj} "
              f"(train={len(tr_idx)} val={len(va_idx)} test={len(te)})\n{'='*50}")

        for name, model in [("A_tcn", TCN(2, width=32)),
                            ("B_bilstm", BiLSTM(ci=2, hidden=128, layers=2))]:
            print(f"  -- {name} --")
            model = train_model(model, data["bcg"], data["ecg"], tr_idx, va_idx, a.epochs)
            pred = infer(model, data["bcg"], te)
            wm = waveform_metrics(data["ecg"][te], pred)
            # RRI (用预测波形的峰值作为伪 R 峰)
            from scipy.signal import find_peaks
            pred_prob = np.maximum(pred, 0)  # 简化: 用预测幅度
            rm = rri_metrics(data["rpeak_mask"][te], pred_prob, threshold=0.0,
                             fs=records[0]["fs"])
            print(f"    PCC={wm['median_pcc']:.4f} maxlag={wm['median_maxlag_pcc']:.4f} "
                  f"RRI_MAE={rm['rri_mae_ms']:.1f}ms")
            results[name].append({"fold": fold, "test_subject": int(test_subj),
                                  **wm, **rm})

    # 汇总
    summary_results = {}
    for name, fold_res in results.items():
        if not fold_res: continue
        summary_results[name] = {
            "median_pcc": float(np.median([r["median_pcc"] for r in fold_res])),
            "mean_pcc": float(np.mean([r["median_pcc"] for r in fold_res])),
            "median_maxlag_pcc": float(np.median([r["median_maxlag_pcc"] for r in fold_res])),
            "median_rri_mae_ms": float(np.median([r["rri_mae_ms"] for r in fold_res])),
            "n_folds": len(fold_res),
            "folds": fold_res,
        }

    res = {
        "protocol": {"dataset": "Kansas Bed-BCG (Carlson 2020)",
                     "task": "BCG->ECG waveform + RRI",
                     "references": {"BiM-Diff PCC": 0.984, "BiLSTM PCC": 0.896,
                                    "Morokuma RRI MAE": "34ms"},
                     "split": "LOSO (first N folds)", "epochs": a.epochs,
                     "fs_hz": records[0]["fs"], "window_sec": 30},
        "results": summary_results,
    }
    write_json(a.out.resolve(), res)
    print(f"\n{'='*60}\nKansas BCG->ECG Results (文献对标):\n")
    for name, r in summary_results.items():
        print(f"  {name:12s}: PCC={r['median_pcc']:.4f} (ref 0.896-0.984), "
              f"maxlag={r['median_maxlag_pcc']:.4f}, RRI_MAE={r['median_rri_mae_ms']:.1f}ms (ref 34ms)")
    print(f"\nSaved to {a.out.resolve()}")


if __name__ == "__main__":
    main()