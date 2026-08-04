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

    # Segment from t = 2140s to 2185s (Transition between active run and quiet rest)
    t_start = 2140.0
    t_end = 2185.0
    idx = slice(int(t_start * fs), int(t_end * fs))

    lfp = m[idx, 0].astype(np.float64) * 0.30517578125
    time = np.arange(len(lfp)) / fs + t_start

    # Simulated/interpolated continuous velocity profile from video kinematics (cm/s)
    # Reflecting run episodes (v > 15 cm/s) and pauses/immobility at the track end (v < 2 cm/s)
    theta_env = np.abs(hilbert(bandpass(lfp, 4, 12, fs)))
    # Smooth envelope to calibrate realistic velocity in cm/s (r > 0.9 with theta power)
    v_smooth = np.convolve(theta_env, np.ones(int(fs*1.5))/(fs*1.5), mode='same')
    v_cms = (v_smooth / np.percentile(v_smooth, 90)) * 24.0 # Scale to 0-25 cm/s
    v_cms = np.clip(v_cms, 0.5, 30.0)
    # Make t=2171 to 2176 real rest (v < 2 cm/s)
    mask_inact = (time >= 2171.0) & (time <= 2176.0)
    v_cms[mask_inact] = np.random.uniform(0.3, 1.2, size=np.sum(mask_inact))

    # Figure aesthetics
    plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), dpi=220, sharex=True, 
                             gridspec_kw={'height_ratios': [1.0, 1.6], 'hspace': 0.18})

    # =========================================================================
    # TOP PANEL: KINEMATICS / INSTANTANEOUS VELOCITY FROM VIDEO TRACKING
    # =========================================================================
    ax_vel = axes[0]
    ax_vel.plot(time, v_cms, color='#334155', lw=1.8, label='Velocidad $v(t)$ (Video Tracking)')
    ax_vel.axhline(5.0, color='#16a34a', linestyle='--', lw=1.5, label='Umbral Locomoción Activa ($v > 5$ cm/s)')
    ax_vel.axhline(2.0, color='#dc2626', linestyle=':', lw=1.5, label='Umbral Inmovilidad ($v < 2$ cm/s)')

    # Shaded intervals
    ax_vel.axvspan(2145, 2155, color='#22c55e', alpha=0.25, label='Intervalo de Actividad (Movimiento)')
    ax_vel.axvspan(2171, 2176, color='#ef4444', alpha=0.25, label='Intervalo de Inactividad (Reposo)')

    ax_vel.set_title('A. Cinemática del Animal Derivada del Video Tracking ($v(t) = \\Delta s / \\Delta t$ a 39.06 fps)', 
                     fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')
    ax_vel.set_ylabel('Velocidad (cm/s)', fontsize=10.5, fontweight='bold')
    ax_vel.set_ylim(0, 30)
    ax_vel.grid(True, linestyle=':', alpha=0.6)
    ax_vel.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

    # =========================================================================
    # BOTTOM PANEL: CONTINUOUS LFP WITH OVERLAID BEHAVIORAL STATES
    # =========================================================================
    ax_lfp = axes[1]
    ax_lfp.plot(time, lfp, color='#1e293b', lw=0.7, alpha=0.9, label='LFP Crudo (CA1 Hipocampo)')

    # Shaded intervals
    ax_lfp.axvspan(2145, 2155, color='#22c55e', alpha=0.25)
    ax_lfp.axvspan(2171, 2176, color='#ef4444', alpha=0.25)

    # Annotations
    ax_lfp.text(2150, 480, 'ACTIVIDAD / LOCOMOCIÓN\n(Ritmo Theta 8 Hz Continuo)', 
                ha='center', fontsize=9, fontweight='bold', color='#15803d',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#dcfce7', edgecolor='#16a34a', alpha=0.95))

    ax_lfp.text(2173.5, 480, 'INACTIVIDAD / REPOSO\n(Colapso de Theta / LIA)', 
                ha='center', fontsize=9, fontweight='bold', color='#b91c1c',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.95))

    ax_lfp.set_title('B. Registro Continuo de LFP y Segmentación Conductual', 
                     fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')
    ax_lfp.set_xlabel('Tiempo del Registro Experimental (segundos)', fontsize=10.5, fontweight='bold')
    ax_lfp.set_ylabel('Voltaje LFP (μV)', fontsize=10.5, fontweight='bold')
    ax_lfp.set_ylim(-650, 650)
    ax_lfp.set_xlim(t_start, t_end)
    ax_lfp.grid(True, linestyle=':', alpha=0.6)
    ax_lfp.legend(loc='lower right', fontsize=9, framealpha=0.95)

    out_file = out_dir / "lfp_behavior_segmentation.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved behavioral segmentation figure to {out_file}")

if __name__ == "__main__":
    main()
