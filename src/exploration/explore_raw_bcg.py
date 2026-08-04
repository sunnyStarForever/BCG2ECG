"""
explore_raw_bcg.py — Direct raw BCG signal investigation.
Check if cardiac information exists in the raw BCG signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import signal as sp
from pipeline_utils import robust_scale

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 70)
    print("Raw BCG Signal Exploration — Does Cardiac Info Exist?")
    print("=" * 70)

    import os
    raw_root = ROOT / "data" / "raw"
    processed_root = ROOT / "data" / "processed" / "v1"
    subj_dirs = sorted([d for d in os.scandir(raw_root) if d.is_dir()],
                       key=lambda d: int(d.name.split("-")[0]))

    z_proc = np.load(processed_root / "subjects" / "S001.npz")

    print(f"\n-- Raw BCG statistics --")
    print(f"Total subjects: {len(subj_dirs)}")

    for win in [0, 1, 2, 3]:
        period = subj_dirs[0].path + "/PeriodData"
        raw_bcg = np.load(f"{period}/Breath/Breath{win+1}.npy", allow_pickle=False)
        rpeaks = np.flatnonzero(z_proc["rpeak_mask_100"][win])
        bcg_z = robust_scale(raw_bcg)

        print(f"\n-- Window {win+1} / Subject 01 --")
        print(f"  dtype: {raw_bcg.dtype}, shape: {raw_bcg.shape}")
        print(f"  range: [{raw_bcg.min()}, {raw_bcg.max()}], p-p: {raw_bcg.max()-raw_bcg.min()}")
        print(f"  mean: {raw_bcg.mean():.1f}, std: {raw_bcg.std():.1f}")
        unique_vals = len(np.unique(raw_bcg))
        print(f"  unique values: {unique_vals} ({100*unique_vals/raw_bcg.size:.1f}% bit efficiency)")
        if rpeaks.size >= 3:
            windows_bcg = []
            for rp in rpeaks:
                s = max(0, rp - 20)
                e = min(len(raw_bcg), rp + 20)
                if e - s == 40:
                    windows_bcg.append(bcg_z[s:e] - bcg_z[s:e].mean())
            if windows_bcg:
                avg = np.mean(np.asarray(windows_bcg), axis=0)
                print(f"  R-peak count: {len(rpeaks)}")
                print(f"  BCG shape near R-peaks (40 samples):")
                print(f"    min={avg.min():.3f} max={avg.max():.3f} p-p={avg.max()-avg.min():.3f}")
                print(f"    first 8: {[f'{x:.2f}' for x in avg[:8]]}")
                pp = avg.max() - avg.min()
                if pp > 0.1:
                    print(f"    -> BCG is clearly visible near R-peaks")
                elif pp > 0.03:
                    print(f"    -> BCG is weakly visible near R-peaks")
                else:
                    print(f"    -> BCG is NOT visible near R-peaks")

    # Try different bandpass filters
    print(f"\n-- Bandpass filter comparison (Subj 01, Win 1) --")
    raw0 = np.load(f"{subj_dirs[0].path}/PeriodData/Breath/Breath1.npy", allow_pickle=False)
    ecg0 = z_proc["ecg_100"][0]
    rpeaks0 = np.flatnonzero(z_proc["rpeak_mask_100"][0])

    for lo, hi, label in [(0.3, 2, "0.3-2Hz"), (0.5, 3, "0.5-3Hz"), (0.8, 8, "0.8-8Hz"),
                          (1, 4, "1-4Hz"), (1.5, 6, "1.5-6Hz"), (3, 10, "3-10Hz")]:
        from pipeline_utils import bandpass
        filt = bandpass(raw0, 100, lo, hi, order=4)
        fz = robust_scale(filt)
        f, pxx = sp.welch(fz, fs=100, nperseg=1024, noverlap=512)
        mask = (f >= 0.5) & (f <= 3.5)
        hr_bcg = f[mask][np.argmax(pxx[mask])] * 60 if mask.any() else float('nan')
        r = float(np.corrcoef(fz, ecg0)[0, 1])
        print(f"  {label:10s}: HR={hr_bcg:5.1f} (true={z_proc['hr_ecg'][0]:.0f}) diff={abs(hr_bcg-z_proc['hr_ecg'][0]):4.1f}  r={r:.4f}")

    # Check differential
    print(f"\n-- Differential analysis --")
    diff1 = np.diff(raw0.astype(float))
    r1 = float(np.corrcoef(robust_scale(diff1), ecg0[:len(diff1)])[0, 1])
    print(f"  first diff vs ECG: r={r1:.4f}")

    # Check BCG polarity at R-peaks
    if len(rpeaks0) >= 3:
        signs = []
        for rp in rpeaks0:
            seg = raw0[max(0, rp-5): min(len(raw0), rp+6)]
            extreme_idx = np.argmax(np.abs(seg - raw0[rp]))
            sign = 1 if seg[extreme_idx] - raw0[rp] > 0 else -1
            signs.append(sign)
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        print(f"  BCG polarity at R-peaks: positive={pos} negative={neg}")
        # Check ECG-BCG cross-correlation for lag
        ecg_seg = ecg0[:len(raw0)]  # both at 100Hz
        xc = sp.correlate(ecg_seg - ecg_seg.mean(), raw0 - raw0.mean(), mode="full")
        lags = sp.correlation_lags(len(ecg_seg), len(raw0), mode="full")
        peak_lag = lags[np.argmax(np.abs(xc))]
        print(f"  ECG-BCG cross-correlation peak lag: {peak_lag} samples ({peak_lag*10:.0f} ms)")
        print(f"  Max correlation: {xc.max()/ecg_seg.std()/raw0.std()/len(ecg_seg):.4f}")

    # Quick summary across all windows (subject 01)
    print(f"\n-- All-window summary for Subject 01 --")
    n_all = len(z_proc["hr_ecg"])
    corrs = []
    for wi in range(n_all):
        raw = np.load(f"{subj_dirs[0].path}/PeriodData/Breath/Breath{wi+1}.npy", allow_pickle=False)
        ecg = z_proc["ecg_100"][wi]
        fz = robust_scale(raw)
        r = float(np.corrcoef(fz, ecg)[0, 1])
        corrs.append(r)
    corrs = np.asarray(corrs)
    print(f"  Raw BCG vs ECG direct correlation (all {n_all} windows):")
    print(f"    median={np.median(corrs):.4f} mean={np.mean(corrs):.4f}")
    print(f"    >0.1: {(corrs>0.1).sum()} windows, >0.2: {(corrs>0.2).sum()} windows")
    print(f"    max={corrs.max():.4f} min={corrs.min():.4f}")

    # Check all subjects for raw BCG-ECG correlation
    print(f"\n-- All subjects raw BCG-ECG correlation --")
    all_corrs = []
    for gi, p in enumerate(sorted((processed_root / "subjects").glob("S*.npz"))):
        z = np.load(p)
        for wi in range(len(z["hr_ecg"])):
            s = f"{(subj_dirs[gi]).path}/PeriodData/Breath/Breath{wi+1}.npy"
            raw = np.load(s, allow_pickle=False)
            fz = robust_scale(raw)
            r = float(np.corrcoef(fz, z["ecg_100"][wi])[0, 1])
            all_corrs.append(r)
        z.close()
    all_corrs = np.asarray(all_corrs)
    print(f"  All subjects, all windows ({len(all_corrs)}):")
    print(f"    median={np.median(all_corrs):.4f} mean={np.mean(all_corrs):.4f}")
    print(f"    >0.1: {(all_corrs>0.1).sum()}, >0.2: {(all_corrs>0.2).sum()}, >0.3: {(all_corrs>0.3).sum()}")
    print(f"    max={all_corrs.max():.4f} min={all_corrs.min():.4f}")

    # Check the preprocessed BCG cardiac channel vs ECG correlation
    print(f"\n-- Processed BCG cardiac channel (0.8-8 Hz) vs ECG --")
    pc_corrs = []
    for gi, p in enumerate(sorted((processed_root / "subjects").glob("S*.npz"))):
        z = np.load(p)
        for wi in range(len(z["hr_ecg"])):
            fz = robust_scale(z["bcg"][wi, 2])
            r = float(np.corrcoef(fz, z["ecg_100"][wi])[0, 1])
            pc_corrs.append(r)
        z.close()
    pc_corrs = np.asarray(pc_corrs)
    print(f"  median={np.median(pc_corrs):.4f} mean={np.mean(pc_corrs):.4f}")
    print(f"  >0.1: {(pc_corrs>0.1).sum()}, >0.2: {(pc_corrs>0.2).sum()}, >0.3: {(pc_corrs>0.3).sum()}")
    print(f"  max={pc_corrs.max():.4f} min={pc_corrs.min():.4f}")


if __name__ == "__main__":
    main()