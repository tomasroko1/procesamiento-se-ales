import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional
from .schema import LFP, Unit

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_lfp_trace(lfp: LFP, start_time: float = 0, title: str = "LFP Trace", save_path: Optional[Path] = None):
    """Plots a short segment of LFP."""
    time_axis = np.arange(lfp.data.shape[0]) / lfp.sampling_rate + start_time
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Handle 1D or 2D array
    if lfp.data.ndim == 1:
        data_to_plot = lfp.data
    else:
        # Plot first channel if multiple
        data_to_plot = lfp.data[:, 0]
        
    ax.plot(time_axis, data_to_plot, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (uV / arb)")
    ax.set_xlim(time_axis[0], time_axis[-1])
    sns.despine()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig

def plot_psd(freqs, psd, title: str = "PSD", save_path: Optional[Path] = None):
    """Plots Power Spectral Density."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # psd shape (n_channels, n_freqs) or (n_freqs,) depending on how welch returned it
    if psd.ndim == 2:
        mean_psd = np.mean(psd, axis=0)
        ax.semilogy(freqs, mean_psd, color='navy')
    else:
        ax.semilogy(freqs, psd, color='navy')
        
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.set_xlim(1, 100) # Interest range
    sns.despine()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

def plot_isi_hist(unit: Unit, bin_size_s: float = 0.001, max_lag_s: float = 0.05, save_path: Optional[Path] = None):
    """Plots Inter-Spike Interval histogram."""
    isis = np.diff(unit.spike_times)
    bins = np.arange(0, max_lag_s, bin_size_s)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(isis, bins=bins, color='teal', alpha=0.8)
    ax.set_title(f"ISI - Unit {unit.cluster_id}")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Count")
    ax.axvline(0.002, color='red', linestyle='--', alpha=0.5, label='2ms')
    sns.despine()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
