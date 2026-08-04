import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

def bandpass(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def main():
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    eeg_file = base_dir / "ec013.29" / "ec013.423" / "ec013.423.eeg"
    out_dir = base_dir / "hc3-reproducible-portfolio" / "reports" / "ec013.423" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_channels = 65
    fs = 1250

    file_size = eeg_file.stat().st_size
    total_samples = file_size // (n_channels * 2)

    m = np.memmap(eeg_file, dtype=np.int16, mode='r', shape=(total_samples, n_channels))

    # Segment from t = 460s to 468s (8 seconds of clean active locomotion)
    t_start = 460.0
    dur = 8.0
    idx = slice(int(t_start * fs), int((t_start + dur) * fs))

    lfp_raw = m[idx, 0].astype(np.float64) * 0.30517578125
    time = np.arange(len(lfp_raw)) / fs

    # Theta bandpass
    lfp_theta = bandpass(lfp_raw, 4, 12, fs)
    theta_env = np.abs(hilbert(lfp_theta))

    # Plot
    plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=220, sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.22})

    # 1. RAW LFP (Unfiltered continuous signal at 1250 Hz)
    ax1.plot(time, lfp_raw, color='#1e293b', lw=0.9, alpha=0.95, label='LFP Crudo / Raw (CA1, 1250 Hz)')
    ax1.set_title('A. Registro de LFP Crudo (Raw Extracellular Voltage en CA1)', 
                  fontsize=11, fontweight='bold', pad=8, color='#0f172a')
    ax1.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax1.set_ylim(-750, 750)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # 2. FILTERED THETA BAND (4-12 Hz) + HILBERT ENVELOPE
    ax2.plot(time, lfp_theta, color='#0284c7', lw=1.2, label='LFP Filtrado en Banda Theta (4–12 Hz)')
    ax2.plot(time, theta_env, color='#f59e0b', lw=1.5, linestyle='--', label='Envolvente Instantánea de Hilbert $A_\\theta(t)$')
    ax2.plot(time, -theta_env, color='#f59e0b', lw=1.5, linestyle='--')
    ax2.set_title('B. Componente Rítmico Aislado y Envolvente Instantánea', 
                  fontsize=11, fontweight='bold', pad=8, color='#0284c7')
    ax2.set_xlabel('Tiempo relativo (segundos)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax2.set_ylim(-750, 750)
    ax2.set_xlim(0, dur)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    out_file = out_dir / "lfp_raw_and_theta.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved raw vs theta figure to {out_file}")

if __name__ == "__main__":
    main()
