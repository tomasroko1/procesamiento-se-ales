import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt, hilbert, welch

def bandpass(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def main():
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    session_dir = base_dir / "ec013.29" / "ec013.423"
    eeg_file = session_dir / "ec013.423.eeg"

    n_channels = 65
    fs = 1250

    # Total file size
    file_size = eeg_file.stat().st_size
    total_samples = file_size // (n_channels * 2)
    total_duration_sec = total_samples / fs
    print(f"Total session duration: {total_duration_sec:.1f} seconds ({total_duration_sec/60:.1f} min)")

    # Read channel 0 across 500 seconds (or chunks) to compute theta/delta power profile
    chunk_sec = 600 # 10 minutes
    n_samples = chunk_sec * fs
    with open(eeg_file, "rb") as f:
        raw_bytes = f.read(n_samples * n_channels * 2)
        data = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, n_channels)

    lfp = data[:, 0].astype(np.float64) * 0.30517578125
    time = np.arange(len(lfp)) / fs

    # Bandpass filter
    theta = bandpass(lfp, 6, 10, fs)
    delta = bandpass(lfp, 1, 4, fs)

    # Envelopes
    theta_env = np.abs(hilbert(theta))
    delta_env = np.abs(hilbert(delta))

    # Smooth with 1-second window
    w = int(1.0 * fs)
    kernel = np.ones(w) / w
    theta_smooth = np.convolve(theta_env, kernel, mode='same')
    delta_smooth = np.convolve(delta_env, kernel, mode='same')
    ratio = theta_smooth / (delta_smooth + 1e-6)

    # Find the top 1% highest theta ratio epoch (Active Locomotion)
    # Find the lowest 1% theta ratio epoch (Pure Inactivity / Rest / LIA)
    # Exclude edges
    valid_idx = np.arange(int(10 * fs), len(time) - int(10 * fs))
    
    # Best active: high theta power and high ratio
    score_active = theta_smooth[valid_idx] * ratio[valid_idx]
    best_act_idx = valid_idx[np.argmax(score_active)]
    t_act = time[best_act_idx]

    # Best inactive: lowest theta power and lowest ratio
    score_inact = theta_smooth[valid_idx] / (delta_smooth[valid_idx] + 1e-6)
    # Let's find segments where theta is lowest
    best_inact_idx = valid_idx[np.argmin(score_inact)]
    t_inact = time[best_inact_idx]

    print(f"Best Active Epoch center: t = {t_act:.2f} s (Theta amplitude = {theta_smooth[best_act_idx]:.1f} uV)")
    print(f"Best Inactive Epoch center: t = {t_inact:.2f} s (Theta amplitude = {theta_smooth[best_inact_idx]:.1f} uV)")

    # Print frequency PSD of 4s windows
    win_samples = int(2.5 * fs)
    seg_act = lfp[best_act_idx - win_samples//2 : best_act_idx + win_samples//2]
    seg_inact = lfp[best_inact_idx - win_samples//2 : best_inact_idx + win_samples//2]

    f_act, psd_act = welch(seg_act, fs=fs, nperseg=fs)
    f_inact, psd_inact = welch(seg_inact, fs=fs, nperseg=fs)

    idx_theta = (f_act >= 6) & (f_act <= 10)
    print(f"Active Theta Peak Power: {np.max(psd_act[idx_theta]):.1f}")
    print(f"Inactive Theta Power in 6-10Hz: {np.max(psd_inact[idx_theta]):.1f}")

if __name__ == "__main__":
    main()
