"""
classify_kansas_signals.py — 绕过 MATLAB table 解析，直接从 #refs# 提取并归类信号

策略:
  Bed_System_Database.mat (v7.3 HDF5) 的 #refs# 下有 ~800 个长 float64 数据集。
  40 受试者 × 20 信号/人 = 800。绕过 table 元数据，直接按信号特征归类:
    - ECG: 有尖锐 R 峰 (高峭度、窄峰)
    - BCG: 低频能量主导、峭度低
    - PPG: 平滑脉搏波、中等峭度
    - 呼吸/其它: 极低频

  对每个信号计算:
    1. 长度 (推断采样率/时长)
    2. 峭度 (ECG 通常 >5, BCG/PPG <3)
    3. R峰候选数 (带通5-25Hz后峰值检测)
    4. 主频 (Welch)
    5. 高频/低频能量比

  然后聚类，抽样验证。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import h5py
from scipy import signal as sp
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent.parent
MAT = ROOT / "data/kansas_bcg/Dataset Files/Bed_System_Database.mat"


def signal_features(sig, fs_guess=1000):
    """计算用于归类信号的特征。"""
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    n = len(sig)
    if n < 100:
        return None
    # 基本统计
    kurt = float(kurtosis(sig, fisher=True))
    # R峰检测 (ECG 特征): 5-25Hz 带通 + 峰值
    try:
        bp = sp.filtfilt(*sp.butter(3, [5, 25], btype="bandpass", fs=fs_guess), sig)
    except Exception:
        bp = sig
    r_candidates, _ = sp.find_peaks(bp, distance=int(0.3 * fs_guess),
                                     prominence=np.std(bp) * 2)
    r_rate = len(r_candidates) / (n / fs_guess)  # beats per second * 60 = bpm
    # 频谱
    f, pxx = sp.welch(sig, fs=fs_guess, nperseg=min(2048, n))
    dom_f = f[np.argmax(pxx)]
    lf = np.trapezoid(pxx[(f >= 0.1) & (f <= 1)], f[(f >= 0.1) & (f <= 1)])
    hf = np.trapezoid(pxx[(f >= 5) & (f <= 25)], f[(f >= 5) & (f <= 25)])
    hf_lf_ratio = hf / (lf + 1e-20)
    return {
        "n": n, "kurtosis": kurt, "r_count": len(r_candidates),
        "r_rate_bpm": r_rate * 60, "dom_freq": float(dom_f),
        "hf_lf_ratio": float(hf_lf_ratio),
    }


def main():
    print(f"Reading {MAT}")
    f = h5py.File(MAT, "r")
    refs = f["#refs#"]
    # 收集所有长 float64 信号
    signals = []
    for k in refs.keys():
        o = refs[k]
        if isinstance(o, h5py.Dataset) and o.dtype == np.float64 and o.size > 1000:
            signals.append((k, o))
    print(f"Found {len(signals)} long float64 signals")

    # 长度分布
    lengths = [o.shape[-1] for _, o in signals]
    print(f"Length stats: min={min(lengths)} max={max(lengths)} "
          f"median={int(np.median(lengths))}")

    # 对每个信号算特征 (采样部分以加速)
    print("\nComputing features (sampling first 100k points)...")
    feats = []
    for k, o in signals:
        # 读取前 100k 点加速
        n_read = min(o.shape[-1], 100000)
        sig = o[0, :n_read] if o.ndim == 2 else o[:n_read]
        ft = signal_features(sig, fs_guess=1000)
        if ft:
            feats.append((k, ft))
    print(f"Computed features for {len(feats)} signals")

    # 归类启发式
    # ECG: 高峭度 (>5) + R峰多 (r_rate 40-150 bpm) + 高频比高
    # BCG: 低峭度 (<3) + 低频主导 + 有心搏相关峰
    # PPG: 中峭度 + 平滑
    ecg, bcg, ppg, other = [], [], [], []
    for k, ft in feats:
        if ft["kurtosis"] > 5 and 40 < ft["r_rate_bpm"] < 180:
            ecg.append((k, ft))
        elif ft["kurtosis"] < 3 and ft["dom_freq"] < 5:
            bcg.append((k, ft))
        elif 3 <= ft["kurtosis"] <= 8 and ft["dom_freq"] < 10:
            ppg.append((k, ft))
        else:
            other.append((k, ft))
    print(f"\n=== Heuristic classification ===")
    print(f"  ECG-like:  {len(ecg)}")
    print(f"  BCG-like:  {len(bcg)}")
    print(f"  PPG-like:  {len(ppg)}")
    print(f"  Other:     {len(other)}")

    # 显示每类的特征分布
    for label, group in [("ECG", ecg), ("BCG", bcg), ("PPG", ppg)]:
        if not group: continue
        ks = [g[1]["kurtosis"] for g in group]
        rs = [g[1]["r_rate_bpm"] for g in group]
        dfs = [g[1]["dom_freq"] for g in group]
        ns = [g[1]["n"] for g in group]
        print(f"\n  {label}: n={len(group)}")
        print(f"    kurtosis: median={np.median(ks):.2f} IQR=[{np.percentile(ks,25):.2f},{np.percentile(ks,75):.2f}]")
        print(f"    r_rate_bpm: median={np.median(rs):.1f} IQR=[{np.percentile(rs,25):.1f},{np.percentile(rs,75):.1f}]")
        print(f"    dom_freq: median={np.median(dfs):.2f} Hz")
        print(f"    length: median={int(np.median(ns))}")
        print(f"    sample keys: {[g[0] for g in group[:5]]}")

    f.close()
    print("\nDone.")


if __name__ == "__main__":
    main()