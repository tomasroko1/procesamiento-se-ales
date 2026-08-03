import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def main():
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    session_dir = base_dir / "ec013.29" / "ec013.423"
    eeg_file = session_dir / "ec013.423.eeg"

    n_channels = 65
    fs = 1250

    file_size = eeg_file.stat().st_size
    total_samples = file_size // (n_channels * 2)
    total_sec = total_samples / fs
    print(f"Total session duration: {total_sec:.1f} s")

    # Read channel 0 across entire file (using memory map or chunking)
    # Memory map channel 0
    m = np.memmap(eeg_file, dtype=np.int16, mode='r', shape=(total_samples, n_channels))
    lfp = m[:, 0].astype(np.float64) * 0.30517578125

    # Compute PSD for every 2-second non-overlapping window
    win_sec = 2.0
    win_samples = int(win_sec * fs)
    n_windows = len(lfp) // win_samples

    t_centers = []
    theta_powers = []
    delta_powers = []
    total_powers = []
    theta_ratios = []

    for i in range(n_windows):
        seg = lfp[i * win_samples : (i + 1) * win_samples]
        t = (i + 0.5) * win_sec
        f, p = welch(seg, fs=fs, nperseg=fs)
        
        idx_theta = (f >= 6) & (f <= 10)
        idx_delta = (f >= 1) & (f <= 4)
        idx_tot = (f >= 1) & (f <= 50)
        
        p_theta = np.trapz(p[idx_theta], f[idx_theta])
        p_delta = np.trapz(p[idx_delta], f[idx_delta])
        p_tot = np.trapz(p[idx_tot], f[idx_tot])
        
        t_centers.append(t)
        theta_powers.append(p_theta)
        delta_powers.append(p_delta)
        total_powers.append(p_tot)
        theta_ratios.append(p_theta / (p_tot + 1e-6))

    t_centers = np.array(t_centers)
    theta_powers = np.array(theta_powers)
    delta_powers = np.array(delta_powers)
    theta_ratios = np.array(theta_ratios)

    # Sort windows by theta power and theta ratio
    idx_sorted_ratio = np.argsort(theta_ratios)
    
    print("\n--- TOP 5 MINIMUM THETA WINDOWS (Lowest theta ratio) ---")
    for rank, idx in enumerate(idx_sorted_ratio[:5]):
        print(f"Rank {rank+1}: t = {t_centers[idx]:.1f} s | Theta Ratio = {theta_ratios[idx]:.4f} | Theta Power = {theta_powers[idx]:.1f} | Delta Power = {delta_powers[idx]:.1f}")

    print("\n--- TOP 5 MAXIMUM THETA WINDOWS (Highest theta ratio) ---")
    for rank, idx in enumerate(idx_sorted_ratio[-5:][::-1]):
        print(f"Rank {rank+1}: t = {t_centers[idx]:.1f} s | Theta Ratio = {theta_ratios[idx]:.4f} | Theta Power = {theta_powers[idx]:.1f} | Delta Power = {delta_powers[idx]:.1f}")

    # Plot the full session theta ratio profile
    plt.figure(figsize=(12, 4))
    plt.plot(t_centers, theta_ratios, color='#0284c7', lw=0.8)
    plt.title('Theta Ratio (P_theta / P_total) across entire 36-min session')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta / Total Ratio')
    plt.grid(True)
    plt.savefig(base_dir / "hc3-reproducible-portfolio" / "reports" / "ec013.423" / "figures" / "session_theta_profile.png")
    plt.close()

if __name__ == "__main__":
    main()
