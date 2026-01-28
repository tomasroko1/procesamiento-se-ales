import numpy as np

from hc3.lfp import bandpass_filter, compute_hilbert_phase
from hc3.schema import LFP


def test_bandpass_filter():
    # Create a synthetic signal: 10Hz sine wave + noise
    fs = 1000
    t = np.arange(0, 1, 1 / fs)  # 1 second
    freq = 10
    signal = np.sin(2 * np.pi * freq * t)

    # Add some high freq noise
    noise = 0.5 * np.sin(2 * np.pi * 100 * t)
    data = (signal + noise).reshape(-1, 1)  # (n_samples, 1)

    lfp = LFP(data=data, sampling_rate=fs, channel_ids=[0])

    # Filter for 10Hz (theta range 4-12)
    filtered = bandpass_filter(lfp, 4, 12)

    # Check that high freq noise is reduced
    # RMS of filtered should be close to RMS of pure signal
    rms_pure = np.sqrt(np.mean(signal**2))
    rms_filtered = np.sqrt(np.mean(filtered.data**2))

    # It won't be exact due to edge effects of filtering, but should be close
    assert np.isclose(rms_filtered, rms_pure, rtol=0.2)


def test_hilbert_phase():
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    freq = 5
    # Cosine has phase 0 at t=0
    signal = np.cos(2 * np.pi * freq * t).reshape(-1, 1)

    lfp = LFP(data=signal, sampling_rate=fs, channel_ids=[0])
    phase = compute_hilbert_phase(lfp)

    # Check phase at t=0 (should be ~0)
    # Filter edge effects might distort start, check middle
    mid_idx = 500
    # At t=0.5, cos(2*pi*5*0.5) = cos(5*pi) = -1, phase should be pi or -pi
    phase_mid = phase[mid_idx, 0]

    assert np.isclose(np.abs(phase_mid), np.pi, rtol=0.1)
