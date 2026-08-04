"""
hr_feasibility.py — 直接验证 BCG 是否有心率信息（手工特征 + ML）

流程：
  1. 对 BCG 4 个通道分别提取 16 个手工频域+时域特征
  2. 用 ExtraTrees 回归标量心率，LOSO 验证
  3. 逐通道消融
  4. PPG 作为上限对照 + shuffled baseline
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import signal as sp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "v1"


def _loso_cv(feat, hr, groups):
    """Leave-One-Subject-Out cross-validation with correct index mapping."""
    valid = np.isfinite(hr)
    n = len(hr)
    pred = np.full(n, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    feat_valid = feat[valid]
    logo = LeaveOneGroupOut()
    for train_rel, test_rel in logo.split(feat_valid, hr[valid], groups[valid]):
        # train_rel and test_rel are indices within the valid subset
        test = valid_indices[test_rel]
        m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=4,
                                max_features=0.75, random_state=42)
        m.fit(feat_valid[train_rel], hr[valid][train_rel])
        pred[test] = m.predict(feat_valid[test_rel])
    ok = np.isfinite(pred)
    mae = float(np.mean(np.abs(pred[ok] - hr[ok])))
    corr = float(np.corrcoef(pred[ok], hr[ok])[0, 1])
    return {"mae": mae, "corr": corr, "n": int(ok.sum()), "pred": pred}


def features(sig, fs=100):
    """Extract 16 features from a single signal window.

    These are designed to capture:
    - Spectral heart-rate power ratio
    - Dominant frequency in HR band (0.5-3.5 Hz)
    - Autocorrelation periodicity confidence
    - Temporal statistics (amplitude distribution, stationarity)
    - Segment-level energies (body movement detection)
    """
    f, pxx = sp.welch(sig, fs=fs, nperseg=min(1024, len(sig)),
                      noverlap=min(512, len(sig)//2))
    hr_mask = (f >= 0.5) & (f <= 3.5)
    total_mask = (f >= 0.05) & (f <= 45)

    hr_pxx = pxx[hr_mask]
    hr_f = f[hr_mask]
    hr_power = float(np.trapezoid(hr_pxx, hr_f) / (np.trapezoid(pxx[total_mask], f[total_mask]) + 1e-20))
    peak_idx = int(np.argmax(hr_pxx))
    peak_f = hr_f[peak_idx]
    peak_prom = hr_pxx[peak_idx] / (float(np.median(hr_pxx)) + 1e-20)
    # Spectral entropy
    prob = hr_pxx / (hr_pxx.sum() + 1e-20)
    entropy = float(-np.sum(prob * np.log(prob + 1e-20)))
    # Autocorrelation
    x = sig - np.median(sig)
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    ac /= ac[0] + 1e-20
    low = int(fs / 3.5)
    high = int(np.ceil(fs / 0.5))
    seg = ac[low:high+1]
    ac_peak = low + int(np.argmax(seg))
    ac_conf = float(seg[np.argmax(seg)])
    # Temporal stats
    abs_x = np.abs(x)
    return np.asarray([
        hr_power, float(peak_f * 60), float(peak_prom), entropy,
        float(np.mean(abs_x)), float(np.std(x)),
        float(np.percentile(abs_x, 90)), float(np.percentile(abs_x, 99)),
        float(ac_peak / fs * 60), ac_conf,
        float(np.std(np.diff(x))), float(np.mean(np.abs(np.diff(x)))),
        float(np.max(abs_x[:500])), float(np.max(abs_x[1000:1500])),
        float(np.max(abs_x[2000:2500])),
        float(np.sum(np.abs(np.fft.rfft(sig)[5:30]))),
    ], dtype=np.float32)


def main():
    print("=" * 60)
    print("HR Feasibility: 手工BCG特征 → 心率回归 (LOSO)")
    print("=" * 60)

    # Build data
    hr_list = []; groups_list = []  # dynamic from npz
    for g, p in enumerate(sorted(DATA.glob("subjects/S*.npz"))):
        z = np.load(p); n = len(z["posture"])
        hr_list.append(z["hr_ecg"].astype(float)); groups_list.append(np.full(n, g))
        z.close()
    hr = np.concatenate(hr_list); groups = np.concatenate(groups_list)
    print(f"Total windows: {hr.size}, HR [{np.nanmin(hr):.0f}, {np.nanmax(hr):.0f}] bpm, subjects: {np.unique(groups).size}")

    # Build BCG and PPG arrays
    bcg_list = []; ppg_list = []
    for g, p in enumerate(sorted(DATA.glob("subjects/S*.npz"))):
        z = np.load(p); bcg_list.append(z["bcg"]); ppg_list.append(z["ppg"]); z.close()
    bcg = np.concatenate(bcg_list, axis=0)
    ppg = np.concatenate(ppg_list, axis=0)
    n_total = len(hr)

    ch_names = ["raw_detrended", "resp_0.08-0.70Hz",
                "cardiac_0.8-8Hz", "high_env_10-40Hz"]
    results = {}

    for ch in range(4):
        feat = np.asarray([features(bcg[i, ch]) for i in range(n_total)])
        r = _loso_cv(feat, hr, groups)
        results[ch_names[ch]] = {"mae": r["mae"], "corr": r["corr"], "n": r["n"]}
        print(f"  {ch_names[ch]:30s}  MAE={r['mae']:.2f} bpm,  r={r['corr']:.3f}")

    # All 4 channels
    feat_all = np.asarray([
        np.concatenate([features(bcg[i, ch]) for ch in range(4)])
        for i in range(n_total)
    ])  # (n_total, 64)
    r = _loso_cv(feat_all, hr, groups)
    results["all_4_channels"] = {"mae": r["mae"], "corr": r["corr"], "n": r["n"]}
    print(f"  {'all_4_channels':30s}  MAE={r['mae']:.2f} bpm,  r={r['corr']:.3f}")

    # PPG upper bound
    feat_ppg = np.asarray([
        np.concatenate([features(ppg[i, 0]), features(ppg[i, 1])])
        for i in range(n_total)
    ])  # (n_total, 32)
    r = _loso_cv(feat_ppg, hr, groups)
    results["ppg_upper_bound"] = {"mae": r["mae"], "corr": r["corr"], "n": r["n"]}
    print(f"  {'ppg_upper_bound':30s}  MAE={r['mae']:.2f} bpm,  r={r['corr']:.3f}")

    # Shuffled baseline
    rng = np.random.RandomState(42)
    hr_shuf = hr.copy(); rng.shuffle(hr_shuf)
    mae_shuf = float(np.mean(np.abs(hr_shuf - hr)))
    results["shuffled_baseline"] = {"mae": mae_shuf, "corr": 0.0}
    print(f"  {'shuffled_baseline':30s}  MAE={mae_shuf:.2f} bpm")

    print(f"\n{'─'*60}")
    print("Summary (LOSO HR regression):")
    for k, v in results.items():
        print(f"  {k:30s}  MAE={v['mae']:.1f} bpm  r={v['corr']:.3f}")


if __name__ == "__main__":
    main()