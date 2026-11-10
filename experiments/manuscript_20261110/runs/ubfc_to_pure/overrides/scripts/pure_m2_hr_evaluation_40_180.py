"""User-fixed HR evaluation; unchanged pinned pipeline with 40--180 BPM band.

Both reference and prediction use the same filter and FFT peak-search limits.
Routing descriptor frequency support and training loss are separate settings.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from evaluation import post_process as post


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True,
                               use_bandpass=True, hr_method='FFT'):
    if diff_flag:
        predictions=post._detrend(np.cumsum(predictions),100)
        labels=post._detrend(np.cumsum(labels),100)
    else:
        predictions=post._detrend(predictions,100)
        labels=post._detrend(labels,100)
    low,high=40/60,180/60
    if use_bandpass:
        b,a=butter(1,[low/fs*2,high/fs*2],btype='bandpass')
        predictions=filtfilt(b,a,np.double(predictions))
        labels=filtfilt(b,a,np.double(labels))
    macc=post._compute_macc(predictions,labels)
    if hr_method!='FFT':raise ValueError('The fixed comparison uses FFT HR only')
    hr_pred=post._calculate_fft_hr(predictions,fs=fs,low_pass=low,high_pass=high)
    hr_label=post._calculate_fft_hr(labels,fs=fs,low_pass=low,high_pass=high)
    snr=post._calculate_SNR(predictions,hr_label,fs=fs,low_pass=low,high_pass=high)
    return hr_label,hr_pred,snr,macc
