import os
import glob
from pathlib import Path
import numpy as np
from scipy.signal import welch

def main():
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    eeg_files = list(base_dir.rglob("*.eeg"))
    print(f"Found {len(eeg_files)} EEG files:")

    for eeg_file in eeg_files:
        size_mb = eeg_file.stat().st_size / (1024 * 1024)
        n_channels = 65
        fs = 1250
        total_samples = int(eeg_file.stat().st_size // (n_channels * 2))
        dur_min = (total_samples / fs) / 60
        print(f"\n--- File: {eeg_file.name} ({dur_min:.1f} min, {size_mb:.1f} MB) ---")
        
        # Read 60s from beginning, middle, and end
        m = np.memmap(eeg_file, dtype=np.int16, mode='r', shape=(total_samples, n_channels))
        
        for t_start in [10, int(dur_min*30), int(dur_min*50)]:
            if t_start + 10 >= dur_min * 60:
                continue
            seg = m[int(t_start*fs) : int((t_start+10)*fs), 0].astype(np.float64) * 0.30517578125
            f, p = welch(seg, fs=fs, nperseg=fs)
            idx_theta = (f >= 6) & (f <= 10)
            idx_delta = (f >= 1) & (f <= 4)
            p_theta = np.max(p[idx_theta])
            p_delta = np.max(p[idx_delta])
            ratio = p_theta / (p_delta + 1e-6)
            print(f"  t = {t_start:4d}s: Theta Peak = {p_theta:8.1f} uV^2/Hz | Delta Peak = {p_delta:8.1f} uV^2/Hz | Theta/Delta = {ratio:6.2f}")

if __name__ == "__main__":
    main()
