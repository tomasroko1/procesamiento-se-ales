from typing import Tuple

import numpy as np

from .schema import Unit


def compute_spike_phases(
    unit: Unit, lfp_phase: np.ndarray, lfp_timestamps: np.ndarray
) -> np.ndarray:
    """
    Assigns a phase to each spike based on LFP phase.

    Args:
        unit: Unit object
        lfp_phase: array of phases (radians) same length as lfp_timestamps
        lfp_timestamps: array of timestamps for LFP samples

    Returns:
        Array of phases for each spike (interpolated)
    """
    # LFP is usually lower sampling rate. We interpolate phase to spike times.
    # Unwrapping phase is important for linear interpolation, then re-wrap.

    # Check bounds
    valid_spikes = (unit.spike_times >= lfp_timestamps[0]) & (
        unit.spike_times <= lfp_timestamps[-1]
    )
    spike_times_valid = unit.spike_times[valid_spikes]

    unwrapped_phase = np.unwrap(lfp_phase)

    spike_phases_unwrapped = np.interp(
        spike_times_valid, lfp_timestamps, unwrapped_phase
    )
    spike_phases = (spike_phases_unwrapped + np.pi) % (
        2 * np.pi
    ) - np.pi  # Wrap to -pi to pi

    return spike_phases


def compute_ccg(
    train1: np.ndarray,
    train2: np.ndarray,
    bin_size_s: float = 0.001,
    window_s: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Cross-Correlogram between two spike trains.

    Args:
        train1: spike times in seconds
        train2: spike times in seconds

    Returns:
        bins: centered time bins
        counts: ccg counts
    """
    # Very basic O(N^2) implementation, optimized ones use histograms
    # For long recordings, this is slow. Better to use labeled histograms if N is large.
    # Here we assume reasonable spike counts for portfolio demo.

    # Optimization: iterate over the shorter train
    if len(train1) > len(train2):
        train1, train2 = train2, train1

    lags = []

    # This can be slow in pure python.
    # For a real large dataset, use searchsorted or C++ extensions.
    # Simple NumPy vectorization for medium size:

    # Vectorized approach with limit
    sorted_2 = np.sort(train2)

    for t1 in train1:
        # Find window in t2
        start_idx = np.searchsorted(sorted_2, t1 - window_s)
        end_idx = np.searchsorted(sorted_2, t1 + window_s)

        matches = sorted_2[start_idx:end_idx]
        lags.extend(matches - t1)

    # Binning
    bins = np.arange(-window_s, window_s + bin_size_s, bin_size_s)
    counts, _ = np.histogram(lags, bins=bins)

    # Center bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, counts


def vector_strength(phases: np.ndarray) -> Tuple[float, float]:
    """
    Computes mean resultant length (R) and mean angle.
    """
    if len(phases) == 0:
        return 0.0, 0.0

    z = np.mean(np.exp(1j * phases))
    r = np.abs(z)
    mean_angle = np.angle(z)
    return r, mean_angle
