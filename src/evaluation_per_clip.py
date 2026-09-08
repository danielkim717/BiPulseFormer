"""Auxiliary fixed-length per-clip HR metrics.

Each 5.3s clip (160 frames) → 1 HR estimate via FFT/periodogram peak.
MAE = mean(|pred_hr - gt_hr|) over ALL clips (not subjects).
Comparison with publications requires matching their window and split protocol.
"""
import numpy as np
import scipy.signal
from src.evaluation import _detrend, _next_power_of_2


def _fft_hr_per_clip(signal, fs=30, low_pass=0.75, high_pass=2.5):
    """Single-clip FFT HR via _next_power_of_2 nfft (same as rPPG-Toolbox)."""
    signal = np.asarray(signal, dtype=np.float64)
    if np.std(signal) < 1e-9:
        return 0.0
    sig = np.expand_dims(signal, 0)
    N = _next_power_of_2(sig.shape[1])
    f, p = scipy.signal.periodogram(sig, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f >= low_pass) & (f <= high_pass))
    if len(fmask) == 0:
        return 0.0
    mf = np.take(f, fmask)
    mp = np.take(p, fmask)
    return float(np.take(mf, np.argmax(mp, 0))[0] * 60)


def evaluate_per_clip(preds_array, gts_array, fs=30, diff_flag=True,
                      low_pass=0.75, high_pass=2.5):
    """Per-clip HR metrics for this repository's configured window length.

    preds_array, gts_array: (N_clips, T) — model output and labels per clip
    Returns dict with MAE, RMSE, MAPE, Pearson(HR) over N_clips pairs.
    """
    if (len(preds_array) == 0 or np.shape(preds_array) != np.shape(gts_array)
            or np.ndim(preds_array) != 2):
        raise ValueError('Expected matching nonempty (N,T) prediction and label arrays')
    if not np.isfinite(preds_array).all() or not np.isfinite(gts_array).all():
        raise ValueError('Non-finite evaluation signals')
    pred_hrs, gt_hrs = [], []
    for i in range(len(preds_array)):
        pred = preds_array[i].astype(np.float64)
        gt = gts_array[i].astype(np.float64)
        if diff_flag:
            pred = _detrend(np.cumsum(pred), 100)
            gt = _detrend(np.cumsum(gt), 100)
        else:
            pred = _detrend(pred, 100)
            gt = _detrend(gt, 100)
        b, a = scipy.signal.butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        pred_f = scipy.signal.filtfilt(b, a, pred)
        gt_f = scipy.signal.filtfilt(b, a, gt)
        hr_p = _fft_hr_per_clip(pred_f, fs=fs, low_pass=low_pass, high_pass=high_pass)
        hr_g = _fft_hr_per_clip(gt_f, fs=fs, low_pass=low_pass, high_pass=high_pass)
        pred_hrs.append(hr_p)
        gt_hrs.append(hr_g)
    pred_hrs = np.array(pred_hrs)
    gt_hrs = np.array(gt_hrs)
    abs_err = np.abs(pred_hrs - gt_hrs)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    nz = gt_hrs != 0
    mape = float(np.mean(abs_err[nz] / gt_hrs[nz]) * 100.0) if nz.any() else 0.0
    if len(pred_hrs) >= 2 and pred_hrs.std() > 1e-9 and gt_hrs.std() > 1e-9:
        hr_pearson = float(np.corrcoef(pred_hrs, gt_hrs)[0, 1])
    else:
        hr_pearson = 0.0
    return {
        'MAE_bpm_clip': mae,
        'RMSE_bpm_clip': rmse,
        'MAPE_pct_clip': mape,
        'Pearson_clip': hr_pearson,
        'n_clips': len(pred_hrs),
        'pred_hrs': pred_hrs,
        'gt_hrs': gt_hrs,
    }
