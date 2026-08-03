import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, butter, filtfilt, hilbert, welch

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

    # Channels: Ch 0 (CA1 Pyramidal layer) & Ch 32 (CA3 Pyramidal layer)
    # Timestamps: t_act = 466.37 s (Exploration RUN), t_inact = 2173.0 s (Rest Pause)
    t_act = 466.37
    t_inact = 2173.0
    dur = 4.0

    idx_act = slice(int((t_act - dur/2) * fs), int((t_act + dur/2) * fs))
    idx_inact = slice(int((t_inact - dur/2) * fs), int((t_inact + dur/2) * fs))

    # CA1 & CA3 LFP signals
    ca1_act = m[idx_act, 0].astype(np.float64) * 0.30517578125
    ca3_act = m[idx_act, 32].astype(np.float64) * 0.30517578125

    ca1_inact = m[idx_inact, 0].astype(np.float64) * 0.30517578125
    ca3_inact = m[idx_inact, 32].astype(np.float64) * 0.30517578125

    # 1. Spectral Coherence C_xy(f)
    f_act, c_act = coherence(ca1_act, ca3_act, fs=fs, nperseg=int(fs*1.5), noverlap=int(fs*1.0))
    f_inact, c_inact = coherence(ca1_inact, ca3_inact, fs=fs, nperseg=int(fs*1.5), noverlap=int(fs*1.0))

    # 2. Theta Phase Synchronization & Hilbert Phase Difference
    theta_ca1_act = bandpass(ca1_act, 4, 12, fs)
    theta_ca3_act = bandpass(ca3_act, 4, 12, fs)

    phase_ca1 = np.angle(hilbert(theta_ca1_act))
    phase_ca3 = np.angle(hilbert(theta_ca3_act))

    # Phase difference delta_phi in [-pi, pi]
    phase_diff = np.angle(np.exp(1j * (phase_ca1 - phase_ca3)))

    idx_theta = (f_act >= 6) & (f_act <= 10)
    coh_act_theta = np.mean(c_act[idx_theta])
    coh_inact_theta = np.mean(c_inact[idx_theta])

    print(f"Mean CA1-CA3 Theta Coherence in Exploration: {coh_act_theta:.3f}")
    print(f"Mean CA1-CA3 Theta Coherence in Rest: {coh_inact_theta:.3f}")
    print(f"Mean Phase Lag CA1-CA3 in Exploration: {np.degrees(np.mean(phase_diff)):.1f} deg")

    # =========================================================================
    # PLOT FIG 1: APARICION DE THETA EN EXPLORACION (PARTE 1)
    # =========================================================================
    plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
    
    # =========================================================================
    # PLOT FIG 2: SINCRONIZACION CA1-CA3 EN FUNCION DE THETA (PARTE 2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=220, 
                             gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.35, 'wspace': 0.22})

    t_rel = np.arange(len(ca1_act)) / fs

    # A1. CA1 & CA3 Superimposed Traces during Exploration
    ax_tl = axes[0, 0]
    ax_tl.plot(t_rel[:1250], theta_ca1_act[:1250], color='#0f766e', lw=1.8, label='LFP CA1 (Theta 8 Hz)')
    ax_tl.plot(t_rel[:1250], theta_ca3_act[:1250], color='#d97706', lw=1.8, linestyle='--', label='LFP CA3 (Theta 8 Hz)')
    ax_tl.set_title('A1. Traces de LFP Sincronizados entre CA1 y CA3 (Exploración)', fontsize=11, fontweight='bold', color='#0f766e', pad=8)
    ax_tl.set_ylabel('Voltaje (μV)', fontsize=10, fontweight='bold')
    ax_tl.set_xlabel('Tiempo relativo (s)', fontsize=9.5)
    ax_tl.grid(True, linestyle=':', alpha=0.6)
    ax_tl.legend(loc='upper right', fontsize=8.5)
    ax_tl.text(0.03, 0.08, f'• Sincronización de fase constante\n• Desfase medio CA1-CA3 Δφ ≈ {np.degrees(np.mean(phase_diff)):.1f}°', 
               transform=ax_tl.transAxes, fontsize=8.5, fontweight='bold', color='#0f766e',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#ccfbf1', edgecolor='#0f766e', alpha=0.9))

    # A2. Coherencia Espectral CA1-CA3 (Exploración vs Reposo)
    ax_tr = axes[0, 1]
    mask_f = (f_act >= 1) & (f_act <= 25)
    ax_tr.plot(f_act[mask_f], c_act[mask_f], color='#0f766e', lw=2.2, label='Exploración (RUN)')
    ax_tr.plot(f_inact[mask_f], c_inact[mask_f], color='#b91c1c', lw=2.0, linestyle='--', label='Reposo (REST)')
    ax_tr.axvspan(4, 12, color='#0284c7', alpha=0.15, label='Banda Theta (4-12 Hz)')
    ax_tr.set_title('A2. Coherencia Espectral Magnetitud-Cuadrada C_xy(f)', fontsize=11, fontweight='bold', pad=8)
    ax_tr.set_ylabel('Coherencia C_xy', fontsize=10, fontweight='bold')
    ax_tr.set_xlabel('Frecuencia (Hz)', fontsize=9.5)
    ax_tr.set_ylim(0, 1.05)
    ax_tr.grid(True, linestyle=':', alpha=0.6)
    ax_tr.legend(loc='upper right', fontsize=8.5)

    # B1. Histograma de Diferencia de Fase en Exploración
    ax_bl = axes[1, 0]
    ax_bl.hist(np.degrees(phase_diff), bins=36, range=(-180, 180), color='#0284c7', edgecolor='#0369a1', alpha=0.8, density=True)
    ax_bl.set_title('B1. Distribución de Diferencia de Fase CA1 - CA3 en Theta', fontsize=10.5, fontweight='bold', pad=6)
    ax_bl.set_xlabel('Diferencia de Fase Δφ (grados)', fontsize=9.5)
    ax_bl.set_ylabel('Densidad de Probabilidad', fontsize=9.5)
    ax_bl.grid(True, linestyle=':', alpha=0.6)

    # B2. Diagrama Polar de Acoplamiento de Fase
    ax_br = fig.add_subplot(2, 2, 4, projection='polar')
    # Convert polar projection
    # Remove standard rectangular axes[1,1]
    axes[1, 1].remove()
    
    counts, bin_edges = np.histogram(phase_diff, bins=24, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ax_br.bar(bin_centers, counts / np.max(counts), width=2*np.pi/24, color='#0f766e', alpha=0.75, edgecolor='#042f2e')
    ax_br.set_title('B2. Acoplamiento de Fase Polar (CA1 - CA3)', fontsize=10, fontweight='bold', pad=12)
    ax_br.set_theta_zero_location('N')

    fig_ca1_ca3 = out_dir / "ca1_ca3_theta_synchronization.png"
    plt.savefig(fig_ca1_ca3, bbox_inches='tight')
    plt.close()
    print(f"Saved CA1-CA3 synchronization figure to {fig_ca1_ca3}")

if __name__ == "__main__":
    main()
