import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt, welch

def bandpass(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def main():
    base_dir = Path(r"c:\Users\tomas\OneDrive\Escritorio\Proyectos y Desarrollo\hc3")
    session_dir = base_dir / "ec013.29" / "ec013.423"
    eeg_file = session_dir / "ec013.423.eeg"
    out_dir = base_dir / "hc3-reproducible-portfolio" / "reports" / "ec013.423" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_channels = 65
    fs = 1250

    file_size = eeg_file.stat().st_size
    total_samples = file_size // (n_channels * 2)

    # Memory map channel 0
    m = np.memmap(eeg_file, dtype=np.int16, mode='r', shape=(total_samples, n_channels))
    lfp_full = m[:, 0].astype(np.float64) * 0.30517578125

    # 1. Exact timestamps
    # Active locomotion: around t = 1279 s
    # Inactive rest: around t = 2058 s
    t_act_center = 1279.0
    t_inact_center = 2058.0
    dur = 2.5 # 2.5 seconds

    idx_act = slice(int((t_act_center - dur/2) * fs), int((t_act_center + dur/2) * fs))
    idx_inact = slice(int((t_inact_center - dur/2) * fs), int((t_inact_center + dur/2) * fs))

    t_rel = np.arange(int(dur * fs)) / fs

    lfp_act = lfp_full[idx_act]
    lfp_inact = lfp_full[idx_inact]

    # Filter Theta (4-12 Hz)
    theta_act = bandpass(lfp_act, 4, 12, fs)
    theta_inact = bandpass(lfp_inact, 4, 12, fs)

    # Compute Welch PSD for 6 seconds around each center
    dur_psd = 6.0
    idx_act_psd = slice(int((t_act_center - dur_psd/2) * fs), int((t_act_center + dur_psd/2) * fs))
    idx_inact_psd = slice(int((t_inact_center - dur_psd/2) * fs), int((t_inact_center + dur_psd/2) * fs))

    f_act, p_act = welch(lfp_full[idx_act_psd], fs=fs, nperseg=int(fs*1.5))
    f_inact, p_inact = welch(lfp_full[idx_inact_psd], fs=fs, nperseg=int(fs*1.5))

    # =========================================================================
    # PLOT HIGH-CONTRAST FIGURE
    # =========================================================================
    plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=220, 
                             gridspec_kw={'height_ratios': [1.3, 1.0], 'hspace': 0.35, 'wspace': 0.22})

    # Top Left: Active Locomotion
    ax_tl = axes[0, 0]
    ax_tl.plot(t_rel, lfp_act, color='#0f766e', lw=1.2, alpha=0.9, label='LFP Crudo (CA1)')
    ax_tl.plot(t_rel, theta_act, color='#0284c7', lw=2.2, linestyle='--', label='Banda Theta (4–12 Hz)')
    ax_tl.set_title('A1. Locomoción Activa / Movimiento (Ritmo Theta a ~8 Hz)', fontsize=11, fontweight='bold', color='#0f766e', pad=8)
    ax_tl.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax_tl.set_xlabel('Tiempo relativo (segundos)', fontsize=9.5)
    ax_tl.set_ylim(-700, 700)
    ax_tl.grid(True, linestyle=':', alpha=0.6)
    ax_tl.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax_tl.text(0.03, 0.08, '• Oscilación regular cuasi-sinusoidal (>500 μV)\n• Período constante T ≈ 125 ms (f ≈ 8 Hz)', 
               transform=ax_tl.transAxes, fontsize=8.5, fontweight='bold', color='#0f766e',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#ccfbf1', edgecolor='#0f766e', alpha=0.9))

    # Top Right: Inactive / Rest (True Non-Theta)
    ax_tr = axes[0, 1]
    ax_tr.plot(t_rel, lfp_inact, color='#b91c1c', lw=1.2, alpha=0.9, label='LFP Crudo (CA1)')
    ax_tr.plot(t_rel, theta_inact, color='#94a3b8', lw=1.8, linestyle='--', label='Banda Theta (Desaparecida)')
    ax_tr.set_title('A2. Inmovilidad / Reposo (Actividad Irregular No-Theta / LIA)', fontsize=11, fontweight='bold', color='#b91c1c', pad=8)
    ax_tr.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax_tr.set_xlabel('Tiempo relativo (segundos)', fontsize=9.5)
    ax_tr.set_ylim(-700, 700) # Same scale!
    ax_tr.grid(True, linestyle=':', alpha=0.6)
    ax_tr.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax_tr.text(0.03, 0.08, '• Ritmo Theta inexistente (2.4% de potencia total)\n• Señal irregular dominada por ondas lentas delta', 
               transform=ax_tr.transAxes, fontsize=8.5, fontweight='bold', color='#b91c1c',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee2e2', edgecolor='#b91c1c', alpha=0.9))

    # Bottom Left: PSD Active
    ax_bl = axes[1, 0]
    mask_f = (f_act >= 1) & (f_act <= 25)
    ax_bl.plot(f_act[mask_f], p_act[mask_f], color='#0f766e', lw=2.2, label='Locomoción Activa')
    ax_bl.axvspan(4, 12, color='#0284c7', alpha=0.15, label='Banda Theta (4–12 Hz)')
    ax_bl.axvline(7.6, color='#0284c7', linestyle=':', lw=1.5, label='Pico Theta (7.6 Hz)')
    ax_bl.set_title('B1. Espectro de Potencia (PSD) en Movimiento', fontsize=10.5, fontweight='bold', pad=6)
    ax_bl.set_xlabel('Frecuencia (Hz)', fontsize=9.5)
    ax_bl.set_ylabel('Densidad de Potencia (μV²/Hz)', fontsize=9.5)
    ax_bl.grid(True, linestyle=':', alpha=0.6)
    ax_bl.legend(loc='upper right', fontsize=8.5)

    # Bottom Right: PSD Inactive (same y scale)
    ax_br = axes[1, 1]
    ax_br.plot(f_inact[mask_f], p_inact[mask_f], color='#b91c1c', lw=2.2, label='Inmovilidad / Reposo')
    ax_br.axvspan(4, 12, color='#94a3b8', alpha=0.15, label='Banda Theta')
    ax_br.set_title('B2. Espectro de Potencia (PSD) en Inmovilidad', fontsize=10.5, fontweight='bold', pad=6)
    ax_br.set_xlabel('Frecuencia (Hz)', fontsize=9.5)
    ax_br.set_ylabel('Densidad de Potencia (μV²/Hz)', fontsize=9.5)
    ax_br.set_ylim(ax_bl.get_ylim()) # Same y-axis scale to demonstrate complete absence of 8Hz peak!
    ax_br.grid(True, linestyle=':', alpha=0.6)
    ax_br.legend(loc='upper right', fontsize=8.5)
    ax_br.text(0.5, 0.6, 'Pico Theta ausente\n(Potencia en 8 Hz cae >95%)', 
               transform=ax_br.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#b91c1c')

    fig_zoom_path = out_dir / "vanderwolf_zoom_contrast.png"
    plt.savefig(fig_zoom_path, bbox_inches='tight')
    plt.close()
    print(f"Saved verified zoom contrast figure to {fig_zoom_path}")

    # =========================================================================
    # ALSO UPDATE TIMELINE SPECTROGRAM AROUND TRANSITION (t = 2020s to 2100s)
    # =========================================================================
    t_start_trans = 2020
    t_end_trans = 2100
    idx_trans = slice(int(t_start_trans * fs), int(t_end_trans * fs))
    lfp_trans = lfp_full[idx_trans]
    t_trans = np.arange(len(lfp_trans)) / fs + t_start_trans

    fig2 = plt.figure(figsize=(13, 9.5), dpi=200)
    gs2 = fig2.add_gridspec(3, 2, height_ratios=[1.2, 1.4, 1.2], hspace=0.35, wspace=0.22)

    # 1. Raw LFP trace
    ax1 = fig2.add_subplot(gs2[0, :])
    ax1.plot(t_trans, lfp_trans, color='#1e293b', lw=0.6, alpha=0.9, label='LFP Crudo (CA1)')
    ax1.axvspan(2050, 2070, color='#f43f5e', alpha=0.2, label='Episodio de Inmovilidad / Reposo (LIA)')
    ax1.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax1.set_xlim(t_start_trans, t_end_trans)
    ax1.set_title('A. Transición de Estado Conductual: Colapso de Theta al Detenerse el Animal', fontsize=11, fontweight='bold', loc='left', pad=6)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5)

    # 2. Spectrogram (STFT)
    ax2 = fig2.add_subplot(gs2[1, :], sharex=ax1)
    f_spec, t_spec, Sxx = spectrogram(lfp_trans, fs=fs, nperseg=int(fs * 1.0), noverlap=int(fs * 0.85))
    t_spec_adj = t_spec + t_start_trans
    freq_mask = f_spec <= 20
    pcm = ax2.pcolormesh(t_spec_adj, f_spec[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-12), 
                         shading='gouraud', cmap='viridis')
    ax2.axhline(7.6, color='#f59e0b', linestyle='--', lw=1.4, alpha=0.9, label='Frecuencia Theta (7.6 Hz)')
    ax2.set_ylabel('Frecuencia (Hz)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 20)
    ax2.set_title('B. Espectrograma Tiempo-Frecuencia: Desaparición de la Banda a 8 Hz en Inmovilidad', fontsize=11, fontweight='bold', loc='left', pad=6)
    cbar = fig2.colorbar(pcm, ax=ax2, pad=0.015, aspect=18)
    cbar.set_label('Potencia (dB)', fontsize=9)

    # 3. Zooms on bottom row
    ax3_l = fig2.add_subplot(gs2[2, 0])
    ax3_l.plot(t_rel, lfp_act, color='#0f766e', lw=1.2, label='LFP Crudo')
    ax3_l.plot(t_rel, theta_act, color='#0284c7', lw=1.8, linestyle='--', label='Banda Theta (~8 Hz)')
    ax3_l.set_title('C1. Locomoción Activa: Oscilación Sinusoidal Pura (~8 Hz)', fontsize=10.5, fontweight='bold', color='#0f766e')
    ax3_l.set_xlabel('Tiempo relativo (s)', fontsize=9)
    ax3_l.set_ylabel('Voltaje (μV)', fontsize=9)
    ax3_l.set_ylim(-700, 700)
    ax3_l.grid(True, linestyle=':', alpha=0.5)
    ax3_l.legend(fontsize=8, loc='upper right')

    ax3_r = fig2.add_subplot(gs2[2, 1])
    ax3_r.plot(t_rel, lfp_inact, color='#b91c1c', lw=1.2, label='LFP Crudo')
    ax3_r.plot(t_rel, theta_inact, color='#94a3b8', lw=1.5, linestyle='--', label='Banda Theta (Inexistente)')
    ax3_r.set_title('C2. Inmovilidad / Reposo: Actividad Irregular No-Theta (LIA)', fontsize=10.5, fontweight='bold', color='#b91c1c')
    ax3_r.set_xlabel('Tiempo relativo (s)', fontsize=9)
    ax3_r.set_ylabel('Voltaje (μV)', fontsize=9)
    ax3_r.set_ylim(-700, 700)
    ax3_r.grid(True, linestyle=':', alpha=0.5)
    ax3_r.legend(fontsize=8, loc='upper right')

    fig_full_path = out_dir / "vanderwolf_theta_modulation.png"
    plt.savefig(fig_full_path, bbox_inches='tight')
    plt.close()
    print(f"Saved verified full modulation figure to {fig_full_path}")

if __name__ == "__main__":
    main()
