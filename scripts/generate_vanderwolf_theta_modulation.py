import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt, hilbert

def bandpass(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def main():
    # Setup paths
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    session_dir = base_dir / "ec013.29" / "ec013.423"
    eeg_file = session_dir / "ec013.423.eeg"
    out_dir = base_dir / "hc3-reproducible-portfolio" / "reports" / "ec013.423" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_channels = 65
    fs = 1250
    start_sec = 10
    duration_sec = 60 # 60 seconds of continuous data
    start_sample = start_sec * fs
    n_samples = duration_sec * fs

    print(f"Reading {duration_sec}s of LFP from channel 0...")
    with open(eeg_file, "rb") as f:
        f.seek(start_sample * n_channels * 2) # 2 bytes per int16
        raw_bytes = f.read(n_samples * n_channels * 2)
        data = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, n_channels)

    # Channel 0 reference in CA1 pyramidal layer
    lfp_uV = data[:, 0].astype(np.float64) * 0.30517578125
    time_sec = np.arange(len(lfp_uV)) / fs + start_sec

    # Filter Theta (4-12 Hz) and Delta (1-4 Hz)
    theta_filt = bandpass(lfp_uV, 4, 12, fs)
    delta_filt = bandpass(lfp_uV, 1, 4, fs)

    # Hilbert Envelopes
    theta_env = np.abs(hilbert(theta_filt))
    delta_env = np.abs(hilbert(delta_filt))

    # Smooth envelopes with 0.5s moving average
    win_len = int(0.5 * fs)
    kernel = np.ones(win_len) / win_len
    theta_env_smooth = np.convolve(theta_env, kernel, mode='same')
    delta_env_smooth = np.convolve(delta_env, kernel, mode='same')
    theta_delta_ratio = theta_env_smooth / (delta_env_smooth + 1e-6)

    # Determine activity vs inactivity based on theta/delta threshold
    # Active locomotion: high theta, high theta/delta ratio
    threshold = np.percentile(theta_delta_ratio, 50)
    is_active = theta_delta_ratio > threshold

    print("Generating comprehensive Vanderwolf 1969 modulation figure...")

    # Set publication style
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.0

    fig = plt.figure(figsize=(13, 10), dpi=200)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.4, 1.0, 1.2], hspace=0.35, wspace=0.25)

    # 1. Raw LFP trace across time (spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_sec, lfp_uV, color='#1e293b', lw=0.6, alpha=0.9, label='LFP Crudo (CA1)')
    ax1.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax1.set_xlim(start_sec, start_sec + duration_sec)
    ax1.set_title('A. Registro Electrofisiológico Continuo de Campo Local (LFP - 1250 Hz)', fontsize=11, fontweight='bold', loc='left', pad=6)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # 2. Spectrogram (STFT)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    f_spec, t_spec, Sxx = spectrogram(lfp_uV, fs=fs, nperseg=int(fs * 1.0), noverlap=int(fs * 0.85))
    t_spec_adj = t_spec + start_sec
    freq_mask = f_spec <= 20 # Focus on 0-20 Hz
    pcm = ax2.pcolormesh(t_spec_adj, f_spec[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-12), 
                         shading='gouraud', cmap='viridis')
    ax2.axhline(8.0, color='#f59e0b', linestyle='--', lw=1.2, alpha=0.85, label='Centro Banda Theta (8 Hz)')
    ax2.set_ylabel('Frecuencia (Hz)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 20)
    ax2.set_title('B. Espectrograma Tiempo-Frecuencia: Transición de Banda Theta (4–12 Hz)', fontsize=11, fontweight='bold', loc='left', pad=6)
    cbar = fig.colorbar(pcm, ax=ax2, pad=0.015, aspect=18)
    cbar.set_label('Potencia (dB)', fontsize=9)

    # 3. Theta Power Envelope & Behavioral Classification
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    ax3.plot(time_sec, theta_env_smooth, color='#0284c7', lw=1.8, label='Envolvente de Potencia Theta (4–12 Hz)')
    ax3.plot(time_sec, delta_env_smooth, color='#94a3b8', lw=1.2, linestyle=':', label='Envolvente Delta (1–4 Hz)')
    
    # Highlight active epochs in green/blue shade
    ax3.fill_between(time_sec, 0, theta_env_smooth, where=is_active, color='#10b981', alpha=0.25, label='Estado Theta (Locomoción / Exploración Activa)')
    ax3.fill_between(time_sec, 0, theta_env_smooth, where=~is_active, color='#f43f5e', alpha=0.15, label='Estado No-Theta (Inmovilidad / Reposo / LIA)')
    
    ax3.set_ylabel('Amplitud μV', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Tiempo de Sesión (segundos)', fontsize=10, fontweight='bold')
    ax3.set_title('C. Modulación Conductual de la Potencia Theta (Vanderwolf, 1969; Buzsáki, 2002)', fontsize=11, fontweight='bold', loc='left', pad=6)
    ax3.legend(loc='upper right', fontsize=8.5, frameon=True, ncol=2)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # 4. Zoom comparison: Active Locomotion (Theta) vs Inactive (Non-Theta / LIA)
    # Find a clean 2.5s segment of active and inactive
    active_indices = np.where(is_active)[0]
    inactive_indices = np.where(~is_active)[0]
    
    t_act_start = time_sec[active_indices[int(len(active_indices) * 0.4)]]
    t_inact_start = time_sec[inactive_indices[int(len(inactive_indices) * 0.4)]]

    # Ensure 2.5s segment
    mask_act = (time_sec >= t_act_start) & (time_sec <= t_act_start + 2.5)
    mask_inact = (time_sec >= t_inact_start) & (time_sec <= t_inact_start + 2.5)

    ax4_left = fig.add_subplot(gs[3, 0])
    ax4_left.plot(time_sec[mask_act] - t_act_start, lfp_uV[mask_act], color='#047857', lw=1.2, label='LFP Crudo')
    ax4_left.plot(time_sec[mask_act] - t_act_start, theta_filt[mask_act], color='#0284c7', lw=1.8, linestyle='--', label='Banda Theta (4-12 Hz)')
    ax4_left.set_title('D1. Zoom en Locomoción Activa (Ritmo Theta Puro)', fontsize=10, fontweight='bold', color='#047857')
    ax4_left.set_xlabel('Tiempo relativo (s)', fontsize=9)
    ax4_left.set_ylabel('Voltaje (μV)', fontsize=9)
    ax4_left.grid(True, linestyle=':', alpha=0.5)
    ax4_left.legend(fontsize=8, loc='upper right')

    ax4_right = fig.add_subplot(gs[3, 1])
    ax4_right.plot(time_sec[mask_inact] - t_inact_start, lfp_uV[mask_inact], color='#b91c1c', lw=1.2, label='LFP Crudo')
    ax4_right.plot(time_sec[mask_inact] - t_inact_start, theta_filt[mask_inact], color='#94a3b8', lw=1.2, linestyle='--', label='Banda Theta (Atenuada)')
    ax4_right.set_title('D2. Zoom en Inmovilidad / Reposo (Actividad Irregular LIA)', fontsize=10, fontweight='bold', color='#b91c1c')
    ax4_right.set_xlabel('Tiempo relativo (s)', fontsize=9)
    ax4_right.set_ylabel('Voltaje (μV)', fontsize=9)
    ax4_right.grid(True, linestyle=':', alpha=0.5)
    ax4_right.legend(fontsize=8, loc='upper right')

    fig_path = out_dir / "vanderwolf_theta_modulation.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    main()
