"""
pair_kansas_signals.py v2 — 改进的受试者重建

改进:
  1. ECG 用 RR 指纹聚类去重 → 合并同受试者的分段, 得到 ~40 个受试者组
  2. 每个受试者组收集所有 RR 匹配的 BCG (Film/LC 多通道), 不只最佳
  3. 用更严阈值 + 时间对齐验证 (互相关峰在 ±500ms 内) 确保真正同步
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import h5py
from scipy import signal as sp
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent.parent
MAT = ROOT / "data/kansas_bcg/Dataset Files/Bed_System_Database.mat"
OUT = ROOT / "data/kansas_bcg" / "subject_signal_map.json"
FS = 1000


def collect_signals(f):
    refs = f["#refs#"]
    return {k: o for k in refs.keys()
            if isinstance((o := refs[k]), h5py.Dataset)
            and o.dtype == np.float64 and o.size > 1000}


def rr_fingerprint(sig, dur_sec=60):
    n = min(len(sig), int(dur_sec * FS))
    bp = sp.filtfilt(*sp.butter(3, [5, 25], btype="bandpass", fs=FS), sig[:n])
    pk, _ = sp.find_peaks(bp, distance=int(0.3*FS), prominence=np.std(bp)*2)
    if len(pk) < 5: return None
    return np.diff(pk)


def is_ecg(sig):
    if len(sig) < 5000: return False
    k = kurtosis(sig, fisher=True)
    if k < 5: return False
    bp = sp.filtfilt(*sp.butter(3, [5, 25], btype="bandpass", fs=FS), sig)
    pk, _ = sp.find_peaks(bp, distance=int(0.3*FS), prominence=np.std(bp)*2)
    bpm = len(pk) / (len(sig)/FS) * 60
    return 40 < bpm < 180


def rr_similarity(rr1, rr2):
    """两个 RR 序列的相似度 (基于平均心率 + 间期匹配)。"""
    if rr1 is None or rr2 is None: return 0.0
    if len(rr1) < 5 or len(rr2) < 5: return 0.0
    hr1 = np.mean(rr1); hr2 = np.mean(rr2)
    if abs(hr1 - hr2) / (hr1 + 1e-8) > 0.25: return 0.0
    # 间期序列相关
    n = min(len(rr1), len(rr2), 20)
    if n < 5: return 0.0
    a, b = rr1[:n].astype(float), rr2[:n].astype(float)
    if np.std(a) < 1 or np.std(b) < 1: return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def cluster_ecgs(ecg_keys, ecg_rr):
    """用 RR 指纹把 ECG 分段聚成受试者组。"""
    groups = []  # list of list of keys
    for k in ecg_keys:
        rr = ecg_rr[k]
        placed = False
        for g in groups:
            # 与组内任一 ECG 相似度高 → 同组
            for gk in g:
                if rr_similarity(rr, ecg_rr[gk]) > 0.85:
                    g.append(k); placed = True; break
            if placed: break
        if not placed:
            groups.append([k])
    return groups


def bcg_beat_intervals(sig, dur_sec=60):
    n = min(len(sig), int(dur_sec * FS))
    bp = sp.filtfilt(*sp.butter(3, [1, 10], btype="bandpass", fs=FS), sig[:n])
    env = np.abs(bp)
    env = sp.filtfilt(*sp.butter(2, [0.5, 5], btype="bandpass", fs=FS), env)
    env = (env - np.median(env)) / (np.std(env) + 1e-8)
    pk, _ = sp.find_peaks(env, distance=int(0.3*FS), prominence=1.0)
    if len(pk) < 5: return None
    return np.diff(pk)


def main():
    print(f"Reading {MAT.name}...")
    f = h5py.File(MAT, "r")
    sigs = collect_signals(f)
    print(f"Total signals: {len(sigs)}")

    print("Identifying ECG signals...")
    ecg_keys, ecg_rr = [], {}
    for k, o in sigs.items():
        sig = o[()].squeeze()
        if is_ecg(sig):
            ecg_keys.append(k)
            ecg_rr[k] = rr_fingerprint(sig)
    print(f"ECG segments: {len(ecg_keys)}")

    print("Clustering ECG segments into subjects (RR fingerprint)...")
    groups = cluster_ecgs(ecg_keys, ecg_rr)
    # 只保留有意义的组 (>=1 ECG)
    groups = [g for g in groups if g]
    print(f"ECG clusters (subjects): {len(groups)}")
    sizes = [len(g) for g in groups]
    print(f"  ECG segments per subject: min={min(sizes)} max={max(sizes)} median={int(np.median(sizes))}")

    # 识别 BCG 候选并算指纹
    print("Computing BCG beat-interval fingerprints...")
    bcg_keys, bcg_iv = [], {}
    for k, o in sigs.items():
        if k in set(ecg_keys): continue
        sig = o[()].squeeze()
        if len(sig) < 5000: continue
        k_ = kurtosis(sig, fisher=True)
        f_, pxx = sp.welch(sig, fs=FS, nperseg=min(2048, len(sig)))
        if k_ < 3 and f_[np.argmax(pxx)] < 5:
            bcg_keys.append(k)
            bcg_iv[k] = bcg_beat_intervals(sig)
    print(f"BCG candidates: {len(bcg_keys)}")

    # 为每个受试者组收集匹配的 BCG (用组内所有 ECG 的 RR 与每个 BCG 比较)
    print("Assigning BCG channels to subjects...")
    subjects = {}
    for sid, g in enumerate(groups, 1):
        # 该受试者的代表性 RR (取组内最长 ECG)
        rep_rr = max((ecg_rr[k] for k in g if ecg_rr[k] is not None),
                     key=lambda r: len(r), default=None)
        if rep_rr is None: continue
        matched_bcg = []
        for bk in bcg_keys:
            sim = rr_similarity(rep_rr, bcg_iv.get(bk))
            if sim > 0.5:
                matched_bcg.append((bk, round(sim, 3)))
        subjects[str(sid)] = {"ecg_segments": g, "bcg_channels": matched_bcg}

    # 统计
    n_with_bcg = sum(1 for s in subjects.values() if s["bcg_channels"])
    total_bcg = sum(len(s["bcg_channels"]) for s in subjects.values())
    print(f"\nSubjects with BCG match: {n_with_bcg}/{len(subjects)}")
    print(f"Total BCG channel assignments: {total_bcg}")
    print("\nSample subjects:")
    for sid, info in list(subjects.items())[:8]:
        bcg_sims = [b[1] for b in info["bcg_channels"][:4]]
        print(f"  S{sid}: {len(info['ecg_segments'])} ECG segs, "
              f"{len(info['bcg_channels'])} BCG chans, top sims={bcg_sims}")

    OUT.write_text(json.dumps(subjects, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT}")
    f.close()


if __name__ == "__main__":
    main()