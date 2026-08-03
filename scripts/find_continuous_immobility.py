import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

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

theta = bandpass(lfp, 6, 10, fs)
env = np.abs(hilbert(theta))

# Find intervals where envelope < 60 uV continuously for at least 2.5 seconds
low_mask = env < 75.0
# Find continuous runs
diffs = np.diff(low_mask.astype(int))
starts = np.where(diffs == 1)[0] + 1
ends = np.where(diffs == -1)[0]

if low_mask[0]:
    starts = np.insert(starts, 0, 0)
if low_mask[-1]:
    ends = np.append(ends, len(low_mask))

durations = (ends - starts) / fs
valid = durations >= 2.5

print(f"Found {np.sum(valid)} long non-theta intervals (>= 2.5s):")
for s, e, d in zip(starts[valid], ends[valid], durations[valid]):
    t_start = s / fs
    t_end = e / fs
    seg = lfp[s:e]
    f, p = welch(seg, fs=fs, nperseg=int(fs*1.2))
    p_8 = np.max(p[(f>=6)&(f<=10)])
    p_delta = np.max(p[(f>=1)&(f<=4)])
    print(f"Interval: {t_start:6.2f}s to {t_end:6.2f}s (Dur: {d:.2f}s) | PSD Theta: {p_8:6.1f} | PSD Delta: {p_delta:6.1f} | Ratio: {p_8/p_delta:.2f}")
