import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, hilbert

def bandpass(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

eeg_file = r'c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3\ec013.29\ec013.423\ec013.423.eeg'
n_channels = 65
fs = 1250

file_size = 355363840
total_samples = file_size // (n_channels * 2)

m = np.memmap(eeg_file, dtype=np.int16, mode='r', shape=(total_samples, n_channels))
lfp = m[:, 0].astype(np.float64) * 0.30517578125

# Filter full trace in theta
theta = bandpass(lfp, 6, 10, fs)
env = np.abs(hilbert(theta))

# Compute 1-second moving average of envelope
w = fs
kernel = np.ones(w) / w
smooth_env = np.convolve(env, kernel, mode='same')

time_sec = np.arange(len(lfp)) / fs

# Find top 10 lowest envelope points (avoiding first/last 10s)
valid_mask = (time_sec >= 10) & (time_sec <= time_sec[-1] - 10)
valid_times = time_sec[valid_mask]
valid_env = smooth_env[valid_mask]

# Sort
sort_idx = np.argsort(valid_env)

print("Top 10 absolute minimum theta amplitude timestamps:")
for i in range(10):
    idx = sort_idx[i]
    t = valid_times[idx]
    # Check PSD at this t
    seg = lfp[int((t-1.25)*fs) : int((t+1.25)*fs)]
    f, p = welch(seg, fs=fs, nperseg=int(fs))
    p_8 = np.max(p[(f>=6)&(f<=10)])
    p_delta = np.max(p[(f>=1)&(f<=4)])
    print(f"#{i+1}: t = {t:6.2f}s | Theta Env = {valid_env[idx]:5.1f} uV | PSD Theta = {p_8:7.1f} | PSD Delta = {p_delta:7.1f} | Ratio = {p_8/p_delta:5.2f}")

# Also top 5 highest
print("\nTop 5 absolute maximum theta amplitude timestamps:")
for i in range(5):
    idx = sort_idx[-(i+1)]
    t = valid_times[idx]
    seg = lfp[int((t-1.25)*fs) : int((t+1.25)*fs)]
    f, p = welch(seg, fs=fs, nperseg=int(fs))
    p_8 = np.max(p[(f>=6)&(f<=10)])
    p_delta = np.max(p[(f>=1)&(f<=4)])
    print(f"#{i+1}: t = {t:6.2f}s | Theta Env = {valid_env[idx]:5.1f} uV | PSD Theta = {p_8:7.1f} | PSD Delta = {p_delta:7.1f} | Ratio = {p_8/p_delta:5.2f}")
