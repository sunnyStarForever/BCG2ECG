"""
explore_ppg.py — PPG 信号质量全面评估

目标：在转向 PPG 为核心方向前，彻底摸清 PPG 的可用性上限
  1. 双波长 PPG (IR/RED) 的信噪比与心率估计精度
  2. PPG → ECG 波形重建的直接可行性（线性 + 深度）
  3. PPG → R 峰事件检测
  4. PPG 心搏周期与 ECG R 峰的时序对齐 (PTT/PAT)
  5. 各体位/受试者下 PPG 质量差异
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import signal as sp
from pipeline_utils import robust_scale, bandpass, POSTURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "v1"


def load_all():
    out = {}
    groups = []
    for g, p in enumerate(sorted(DATA.glob("subjects/S*.npz"))):
        z = np.load(p)
        for k in ["ppg", "ecg_100", "rpeak_mask_100", "hr_ecg",
                  "hr_ppg_ir", "hr_ppg_red", "posture", "quality"]:
            out.setdefault(k, []).append(z[k])
        groups.append(np.full(len(z["posture"]), g, dtype=np.int16))
        z.close()
    out = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    out["groups"] = np.concatenate(groups)
    return out


def welch_hr(sig, fs=100, lo=0.5, hi=3.5):
    f, pxx = sp.welch(sig, fs=fs, nperseg=min(1024, len(sig)),
                      noverlap=min(512, len(sig)//2))
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return float('nan')
    return float(f[mask][np.argmax(pxx[mask])] * 60)


def main():
    print("=" * 70)
    print("PPG Signal Quality Assessment")
    print("=" * 70)
    d = load_all()
    n = d["posture"].size
    print(f"Total windows: {n}, subjects: {np.unique(d['groups']).size}")

    # ── 1. PPG vs ECG HR agreement ──
    print(f"\n-- 1. PPG vs ECG heart-rate agreement --")
    hr_ecg = d["hr_ecg"].astype(float)
    hr_ir = d["hr_ppg_ir"].astype(float)
    hr_red = d["hr_ppg_red"].astype(float)
    hr_ppg_mean = np.nanmean(np.stack([hr_ir, hr_red]), axis=0)

    for name, hr_p in [("PPG-IR", hr_ir), ("PPG-RED", hr_red), ("PPG-mean", hr_ppg_mean)]:
        v = np.isfinite(hr_ecg) & np.isfinite(hr_p)
        mae = float(np.mean(np.abs(hr_p[v] - hr_ecg[v])))
        corr = float(np.corrcoef(hr_p[v], hr_ecg[v])[0, 1])
        within3 = float(np.mean(np.abs(hr_p[v] - hr_ecg[v]) <= 3.0))
        within5 = float(np.mean(np.abs(hr_p[v] - hr_ecg[v]) <= 5.0))
        print(f"  {name:10s}: MAE={mae:.2f} bpm, r={corr:.3f}, "
              f"within3bpm={within3:.1%}, within5bpm={within5:.1%}, n={v.sum()}")

    # ── 2. Direct PPG-ECG waveform correlation ──
    print(f"\n-- 2. PPG vs ECG waveform correlation (filtered 0.4-8 Hz) --")
    ppg_ir = d["ppg"][:, 0]
    ppg_red = d["ppg"][:, 1]
    ecg = d["ecg_100"]

    for name, sig in [("PPG-IR", ppg_ir), ("PPG-RED", ppg_red)]:
        corrs = []
        lag_corrs = []
        for i in range(n):
            if np.std(sig[i]) < 1e-6 or np.std(ecg[i]) < 1e-6:
                continue
            corrs.append(float(np.corrcoef(sig[i], ecg[i])[0, 1]))
            # max-lag correlation (search ±30 samples = ±300ms)
            sz = (sig[i] - sig[i].mean()) / (sig[i].std() + 1e-8)
            ez = (ecg[i] - ecg[i].mean()) / (ecg[i].std() + 1e-8)
            xc = sp.correlate(ez, sz, mode="full") / len(ez)
            lags = sp.correlation_lags(len(ez), len(sz), mode="full")
            allowed = np.abs(lags) <= 30
            lag_corrs.append(float(np.max(xc[allowed])))
        corrs = np.asarray(corrs); lag_corrs = np.asarray(lag_corrs)
        print(f"  {name:10s}: direct r median={np.median(corrs):.3f} mean={np.mean(corrs):.3f}")
        print(f"  {'':10s}  maxlag r median={np.median(lag_corrs):.3f} mean={np.mean(lag_corrs):.3f}")
        print(f"  {'':10s}  >0.3: {(corrs>0.3).sum()}, >0.5: {(corrs>0.5).sum()}, >0.7: {(corrs>0.7).sum()}")

    # ── 3. PPG peak detection vs ECG R-peaks ──
    print(f"\n-- 3. PPG pulse-peak vs ECG R-peak alignment --")
    from pipeline_utils import detect_ppg_rate
    # Use existing detect_ppg_rate to get peaks count
    ppg_peak_counts = []
    rpeak_counts = []
    delays = []  # PPG peak - nearest R peak
    for i in range(n):
        rpeaks = np.flatnonzero(d["rpeak_mask_100"][i])
        rpeak_counts.append(len(rpeaks))
        # Find PPG-IR peaks
        x = ppg_ir[i].astype(float)
        for sign in (1.0, -1.0):
            y = sign * x
            mad = np.median(np.abs(y - np.median(y))) + 1e-8
            peaks, _ = sp.find_peaks(y, distance=int(round(0.35*100)),
                                     prominence=0.35*mad)
            if 15 <= len(peaks) <= 75:
                ppg_peak_counts.append(len(peaks))
                # Compute delays
                for pp in peaks:
                    if len(rpeaks):
                        nearest = rpeaks[np.argmin(np.abs(rpeaks - pp))]
                        delays.append(int(pp - nearest))
                break
        else:
            ppg_peak_counts.append(0)
    delays = np.asarray(delays)
    print(f"  ECG R-peaks per window: mean={np.mean(rpeak_counts):.1f}")
    print(f"  PPG peaks per window:   mean={np.mean(ppg_peak_counts):.1f}")
    if delays.size:
        print(f"  PPG-peak minus R-peak delay (samples @100Hz):")
        print(f"    median={np.median(delays):.0f} ({np.median(delays)*10:.0f} ms)")
        print(f"    mean={np.mean(delays):.0f}, std={np.std(delays):.0f}")
        print(f"    IQR: [{np.percentile(delays,25):.0f}, {np.percentile(delays,75):.0f}]")
        # PTT should be positive ~ 200-400 ms typically
        pos = float(np.mean(delays > 0))
        print(f"    fraction positive (PPG after R): {pos:.1%}")

    # ── 4. Per-posture PPG quality ──
    print(f"\n-- 4. PPG quality by posture --")
    for pid, pname in enumerate(POSTURE_NAMES):
        mask = d["posture"] == pid
        if not mask.any():
            continue
        v = np.isfinite(hr_ecg) & np.isfinite(hr_ir) & mask
        mae = float(np.mean(np.abs(hr_ir[v] - hr_ecg[v])))
        corr = float(np.corrcoef(hr_ir[v], hr_ecg[v])[0, 1])
        # waveform correlation
        wc = []
        for i in np.where(mask)[0]:
            if np.std(ppg_ir[i]) < 1e-6 or np.std(ecg[i]) < 1e-6:
                continue
            wc.append(float(np.corrcoef(ppg_ir[i], ecg[i])[0, 1]))
        wc = np.asarray(wc)
        print(f"  {pname:12s}: n={mask.sum():3d}, HR MAE={mae:.2f}, HR r={corr:.3f}, "
              f"waveform r median={np.median(wc):.3f}")

    # ── 5. Linear PPG->ECG upper bound ──
    print(f"\n-- 5. Linear PPG->ECG reconstruction upper bound --")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupShuffleSplit
    # Use PPG-IR + PPG-RED as input, ECG_100 as target
    # Downsample to 25 Hz for tractable linear regression
    X = d["ppg"]  # (N, 2, 3000)
    Y = ecg  # (N, 3000)
    # Downsample 4x
    X_ds = sp.decimate(X, 4, axis=-1, ftype='fir')  # (N, 2, 750)
    Y_ds = sp.decimate(Y, 4, axis=-1, ftype='fir')  # (N, 750)
    X_flat = X_ds.reshape(n, -1)  # (N, 1500)
    Y_flat = Y_ds  # (N, 750)
    groups = d["groups"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X_flat, Y_flat, groups))
    model = Ridge(alpha=100.0)
    model.fit(X_flat[tr], Y_flat[tr])
    pred = model.predict(X_flat[te])
    # Metrics
    corrs = []
    for i in range(len(te)):
        if np.std(Y_flat[i]) < 1e-6 or np.std(pred[i]) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(Y_flat[i], pred[i])[0, 1]))
    corrs = np.asarray(corrs)
    print(f"  Linear Ridge (PPG IR+RED -> ECG @25Hz), subject-disjoint:")
    print(f"    direct r median={np.median(corrs):.3f} mean={np.mean(corrs):.3f}")
    print(f"    >0.3: {(corrs>0.3).sum()}/{len(corrs)}, >0.5: {(corrs>0.5).sum()}")


if __name__ == "__main__":
    main()