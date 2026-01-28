import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from .schema import LFP


def bandpass_filter(lfp: LFP, lowcut: float, highcut: float, order: int = 4) -> LFP:
    """Applies zero-phase bandpass filter."""
    nyquist = 0.5 * lfp.sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    filtered_data = filtfilt(b, a, lfp.data, axis=0)

    return LFP(
        data=filtered_data, sampling_rate=lfp.sampling_rate, channel_ids=lfp.channel_ids
    )


def compute_hilbert_phase(lfp: LFP) -> np.ndarray:
    """Computes instantaneous phase using Hilbert transform."""
    analytic_signal = hilbert(lfp.data, axis=0)
    return np.angle(analytic_signal)


def compute_psd(lfp: LFP, nperseg: int = 1024):
    """Computes Power Spectral Density."""
    from scipy.signal import welch

    if lfp.data.ndim == 1:
        data = lfp.data
    else:
        data = (
            lfp.data.T
        )  # Welch expects (n_channels, n_samples) if calculating for multiple

    freqs, psd = welch(data, lfp.sampling_rate, nperseg=nperseg)

    # Return in shape (n_freqs, n_channels) or (n_freqs,)
    if psd.ndim == 2:
        return freqs, psd.T
    return freqs, psd
