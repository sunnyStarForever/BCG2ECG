"""
diagnosis_analysis.py — Diagnose why BCG→ECG is failing.

Steps:
  1. Spectral analysis: does BCG cardiac channel contain heart-rate energy?
  2. Temporal alignment: are BCG cardiac peaks near ECG R-peaks? (PEP)
  3. HR feasibility: can we predict scalar HR from BCG features?
  4. Per-channel contribution: which BCG channel carries the most cardiac info?
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import signal as sp_signal
from pipeline_utils import robust_scale, bandpass

# NumPy 2.0 compat: trapz → trapezoid
def _trapz(y, x=None):
    return np.trapezoid(y, x) if x is not None else np.trapezoid(y)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "v1"


# ── Load all subjects ───────────────────────────────────────────────

def load_all(root: Path):
    out = {}
    for k in ["bcg","ecg_100","rpeak_mask_100","hr_ecg","posture","quality"]:
        vals = []
        for p in sorted(root.glob("subjects/S*.npz")):
            z = np.load(p); vals.append(z[k]); z.close()
        out[k] = np.concatenate(vals, axis=0)
    return out


# ── Step 1: Spectral quality of BCG cardiac channel ─────────────────

def diagnose_spectra(data):
    """Check if BCG contains detectable heart-rate power."""
    bcg_card = data["bcg"][:, 2]      # cardiac candidate 0.8-8 Hz
    ecg = data["ecg_100"]

    results = {"n": data["posture"].size, "spectral_quality": []}

    for i in range(min(100, data["posture"].size)):
        f_b, pxx_b = sp_signal.welch(bcg_card[i], fs=100, nperseg=1024, noverlap=512)
        f_e, pxx_e = sp_signal.welch(ecg[i], fs=100, nperseg=1024, noverlap=512)

        # Heart-rate band 0.8-3 Hz (48-180 bpm)
        hr_mask = (f_b >= 0.8) & (f_b <= 3.0)
        total_mask = (f_b >= 0.08) & (f_b <= 40)

        hr_power = _trapz(pxx_b[hr_mask], f_b[hr_mask])
        total_power = _trapz(pxx_b[total_mask], f_b[total_mask])

        hr_ratio = float(hr_power / (total_power + 1e-20))

        # Heart-rate peak prominence
        hr_band = pxx_b[hr_mask]
        peak_idx = np.argmax(hr_band)
        peak_power = float(hr_band[peak_idx])
        median_power = float(np.median(hr_band))
        peak_to_median = peak_power / (median_power + 1e-20)

        results["spectral_quality"].append({
            "window": i,
            "hr_power_ratio": hr_ratio,
            "hr_peak_to_median": peak_to_median,
        })

    return results


# ── Step 2: Temporal alignment (PEP) ────────────────────────────────

def diagnose_alignment(data):
    """Check if BCG cardiac peaks align with ECG R-peaks."""
    # Use a few clean subjects
    results = {"per_subject": []}

    for subj_path in sorted(DATA.glob("subjects/S*.npz")):
        z = np.load(subj_path)
        n = z["posture"].shape[0]
        sid = str(z["subject_id"])
        win_results = []

        for win in range(min(10, n)):
            rpeaks = np.flatnonzero(z["rpeak_mask_100"][win])
            if len(rpeaks) < 3:
                continue

            bcg_card = z["bcg"][win, 2]  # cardiac candidate
            hr = z["hr_ecg"][win]

            # Find BCG peaks near R-peaks (expect J-wave near R)
            bcgs = []
            for rp in rpeaks:
                left = max(0, rp - 10)
                right = min(len(bcg_card), rp + 10)
                seg = bcg_card[left:right]
                if seg.size:
                    bcgs.append((rp, left + int(np.argmax(np.abs(seg)))))

            if len(bcgs) >= 3:
                delays = [bcg - rpeaks[i] for i, (rpeaks, bcg) in enumerate(bcgs)]
                median_delay = float(np.median(delays))
                delay_std = float(np.std(delays))
                corr = float(np.corrcoef([d for d, _ in bcgs], [d for _, d in bcgs])[0,1]) if len(bcgs)>=3 else 0
                win_results.append({
                    "window": win, "n_beats": len(bcgs),
                    "median_delay_samples": median_delay,
                    "delay_std_samples": delay_std,
                })

        if win_results:
            results["per_subject"].append({
                "subject": sid,
                "windows": win_results,
            })
        z.close()

    return results


# ── Step 3: HR regression feasibility ───────────────────────────────

def hr_regression_feasibility(data):
    """Simple test: can BCG spectral features predict scalar HR?"""
    from sklearn.model_selection import GroupKFold, cross_validate
    from sklearn.ensemble import ExtraTreesRegressor

    bcg_card = data["bcg"][:, 2]  # (N, 3000) cardiac candidate
    hr_ecg = data["hr_ecg"].astype(float)
    groups = np.repeat(np.arange(30), 30)  # 30 subjects, 30 windows each

    # Feature: simple Welch-PSD → dominant frequency * 60
    hr_pred = []
    for sig in bcg_card:
        f, pxx = sp_signal.welch(sig, fs=100, nperseg=min(1024, len(sig)),
                                  noverlap=min(512, len(sig)//2))
        mask = (f >= 0.7) & (f <= 3.0)
        hr_pred.append(f[mask][np.argmax(pxx[mask])] * 60)
    hr_pred = np.asarray(hr_pred)

    valid = np.isfinite(hr_ecg) & np.isfinite(hr_pred)
    mae = float(np.mean(np.abs(hr_pred[valid] - hr_ecg[valid])))
    corr = float(np.corrcoef(hr_pred[valid], hr_ecg[valid])[0, 1])

    return {"spectral_hr_mae": mae, "spectral_hr_corr": corr, "n": int(valid.sum())}


# ── Step 4: Per-channel contribution ────────────────────────────────

def per_channel_feasibility(data):
    """Which BCG channel carries the most heart-rate information?"""
    channel_names = ["detrended_raw", "resp_0.08-0.70Hz", "cardiac_0.8-8Hz", "high_env_10-40Hz"]
    hr_ecg = data["hr_ecg"].astype(float)
    results = {}

    for ch, ch_name in enumerate(channel_names):
        hr_pred = []
        for window in range(data["bcg"].shape[0]):
            sig = data["bcg"][window, ch]
            f, pxx = sp_signal.welch(sig, fs=100, nperseg=min(1024, len(sig)),
                                      noverlap=min(512, len(sig)//2))
            mask = (f >= 0.7) & (f <= 3.0)
            hr_pred.append(f[mask][np.argmax(pxx[mask])] * 60)
        hr_pred = np.asarray(hr_pred)
        valid = np.isfinite(hr_ecg) & np.isfinite(hr_pred)
        mae = float(np.mean(np.abs(hr_pred[valid] - hr_ecg[valid])))
        corr = float(np.corrcoef(hr_pred[valid], hr_ecg[valid])[0, 1])
        results[ch_name] = {"mae_bpm": mae, "correlation": corr}

    return results


# ── Step 5: Check raw BCG before preprocessing ──────────────────────

def check_raw_bcg_signal():
    """Load raw BCG and check its cardiac energy directly."""
    from pipeline_utils import preprocess_bcg
    import numpy as np

    subj = sorted(DATA.glob("subjects/S*.npz"))[0]
    z = np.load(subj)
    # First load raw data
    raw_root = DATA.parent.parent / "raw"
    subj_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if subj_dirs:
        period = subj_dirs[0] / "PeriodData"
        raw_bcg = np.load(period / "Breath" / "Breath1.npy", allow_pickle=False)
        print(f"Raw BCG shape: {raw_bcg.shape}, range: [{raw_bcg.min():.1f}, {raw_bcg.max():.1f}]")
        print(f"Raw BCG dtype: {raw_bcg.dtype}")

        # Spectrum of raw vs processed
        f_raw, pxx_raw = sp_signal.welch(raw_bcg, fs=100, nperseg=1024)
        processed_bcg = preprocess_bcg(raw_bcg)
        f_proc, pxx_proc = sp_signal.welch(processed_bcg[2], fs=100, nperseg=1024)

        hr_band = (0.8, 3.0)
        raw_hr_power = _trapz(pxx_raw[(f_raw>=hr_band[0])&(f_raw<=hr_band[1])],
                                f_raw[(f_raw>=hr_band[0])&(f_raw<=hr_band[1])])
        proc_hr_power = _trapz(pxx_proc[(f_proc>=hr_band[0])&(f_proc<=hr_band[1])],
                                 f_proc[(f_proc>=hr_band[0])&(f_proc<=hr_band[1])])
        raw_total = _trapz(pxx_raw, f_raw)
        proc_total = _trapz(pxx_proc, f_proc)
        print(f"Raw BCG HR power ratio: {raw_hr_power/(raw_total+1e-20):.4f}")
        print(f"Processed BCG cardiac channel HR power ratio: {proc_hr_power/(proc_total+1e-20):.4f}")
    z.close()


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DIAGNOSIS: Why is BCG→ECG failing?")
    print("=" * 60)

    # Load data (first 3 subjects for speed)
    data = load_all(DATA)
    print(f"\nLoaded {data['posture'].size} windows from all subjects")
    print(f"BCG shape: {data['bcg'].shape}")
    print(f"ECG shape: {data['ecg_100'].shape}")
    print(f"HR range: [{np.nanmin(data['hr_ecg']):.0f}, {np.nanmax(data['hr_ecg']):.0f}] bpm")
    print(f"HR nan count: {np.isnan(data['hr_ecg']).sum()} / {data['hr_ecg'].size}")

    # Step 1: Spectral quality
    print(f"\n{'─' * 60}")
    print("Step 1: Spectral quality of BCG cardiac channel")
    spectral = diagnose_spectra(data)
    ratios = [s["hr_power_ratio"] for s in spectral["spectral_quality"]]
    peaks = [s["hr_peak_to_median"] for s in spectral["spectral_quality"]]
    print(f"  HR band power / total: median={np.median(ratios):.4f}, q25={np.percentile(ratios,25):.4f}, q75={np.percentile(ratios,75):.4f}")
    print(f"  HR peak/median: median={np.median(peaks):.2f}, q25={np.percentile(peaks,25):.2f}, q75={np.percentile(peaks,75):.2f}")
    print(f"  {sum(1 for p in peaks if p>2)} / {len(peaks)} windows have prominent HR peak (>2x median)")

    # Step 3: HR regression feasibility (quick, no training required)
    print(f"\n{'─' * 60}")
    print("Step 3: HR regression feasibility (direct spectral estimate)")
    hr_feas = hr_regression_feasibility(data)
    print(f"  Direct spectral HR estimate vs ECG HR:")
    print(f"    MAE = {hr_feas['spectral_hr_mae']:.2f} bpm")
    print(f"    Pearson r = {hr_feas['spectral_hr_corr']:.3f}")
    print(f"    n = {hr_feas['n']}")

    # Step 4: Per-channel contribution
    print(f"\n{'─' * 60}")
    print("Step 4: Per-channel cardiac information")
    channels = per_channel_feasibility(data)
    for ch_name, metrics in channels.items():
        print(f"  {ch_name:30s}: HR MAE = {metrics['mae_bpm']:.2f} bpm, r = {metrics['correlation']:.3f}")

    # Step 5: Raw BCG inspection
    print(f"\n{'─' * 60}")
    print("Step 5: Raw BCG signal quality check")
    check_raw_bcg_signal()

    print(f"\n{'─' * 60}")
    print("Diagnosis complete.")

if __name__ == "__main__":
    main()