import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..schema import LFP, Session, Unit

logger = logging.getLogger(__name__)


def parse_xml(xml_path: Path) -> Dict[str, Any]:
    """
    Parses NeuroScope XML file (.xml) to extract recording parameter.

    Returns:
        Dict with keys: n_channels, sampling_rate, lfp_sampling_rate, shank_map
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize defaults
    params = {
        "n_channels": 0,
        "sampling_rate": 20000.0,
        "lfp_sampling_rate": 1250.0,
        "shank_map": {},
    }

    # Extract Acquisition System parameters
    acquisition = root.find("acquisitionSystem")
    if acquisition is not None:
        params["n_channels"] = int(acquisition.find("nChannels").text)
        params["sampling_rate"] = float(acquisition.find("samplingRate").text)

    # Extract Field Potential parameters (often different sampling rate)
    field_potential = root.find("fieldPotential")
    if field_potential is not None:
        lfp_sampling_rate_node = field_potential.find("lfpSamplingRate")
        if lfp_sampling_rate_node is not None:
            params["lfp_sampling_rate"] = float(lfp_sampling_rate_node.text)

    # Extract Channel Groups (Shank Map)
    anatomical = root.find("anatomicalDescription")
    if anatomical is not None:
        channel_groups = anatomical.find("channelGroups")
        if channel_groups is not None:
            for i, group in enumerate(channel_groups.findall("group")):
                # Assuming group order is shank ID, 1-indexed in file names usually
                # We will use 1-based indexing for shank_id to match file extensions .res.1
                shank_id = i + 1
                channels = []
                for channel in group.findall("channel"):
                    channels.append(int(channel.text))
                params["shank_map"][shank_id] = channels

    return params


def load_lfp(
    session: Session, channel_id: Optional[int] = None, duration: Optional[float] = None
) -> LFP:
    """
    Loads LFP data from .eeg file.

    Args:
        session: Session object
        channel_id: Single channel ID to load. If None, loads all (careful with memory).
        duration: Seconds to load. If None, load all.

    Returns:
        LFP object
    """
    eeg_file = session.path / f"{session.id}.eeg"
    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    # Calculate bytes per sample per channel (int16 = 2 bytes)
    dtype = np.int16
    n_channels = session.n_channels
    np.dtype(dtype).itemsize

    # Open memmap
    # .eeg is interleaved: ch0_t0, ch1_t0, ... chN_t0, ch0_t1...
    # Shape: (n_samples, n_channels)
    data_mmap = np.memmap(eeg_file, dtype=dtype, mode="r", order="C")
    n_total_samples = data_mmap.size // n_channels
    data_reshaped = data_mmap.reshape((n_total_samples, n_channels))

    samples_to_load = n_total_samples
    if duration is not None:
        samples_to_load = int(duration * session.lfp_sampling_rate)
        samples_to_load = min(samples_to_load, n_total_samples)

    if channel_id is not None:
        # Load specific channel
        if channel_id >= n_channels:
            raise ValueError(f"Channel {channel_id} out of bounds (0-{n_channels-1})")
        data = data_reshaped[:samples_to_load, channel_id].copy()
        channels = [channel_id]
    else:
        # Load all channels
        data = data_reshaped[:samples_to_load, :].copy()
        channels = list(range(n_channels))

    return LFP(data=data, sampling_rate=session.lfp_sampling_rate, channel_ids=channels)


def load_spikes(session: Session, min_cluster_id: int = 2) -> List[Unit]:
    """
    Loads spike times and cluster identities.

    Args:
        session: Session object
        min_cluster_id: Minimum cluster ID to keep (default 2 to skip noise/MUA).

    Returns:
        List of Unit objects.
    """
    units = []

    for shank_id in session.shank_map.keys():
        res_file = session.path / f"{session.id}.res.{shank_id}"
        clu_file = session.path / f"{session.id}.clu.{shank_id}"

        if not res_file.exists() or not clu_file.exists():
            # It's possible some shanks have no sorted data
            logger.debug(f"No spike data for shank {shank_id} in {session.id}")
            continue

        # Read .res file (spike times in samples at full sampling rate)
        # Text file, list of integers
        try:
            spike_times_samples = np.loadtxt(res_file, dtype=np.int64)
        except ValueError:
            logger.warning(f"Could not read {res_file}")
            continue

        # Read .clu file (cluster IDs)
        # First line is number of clusters, skip it
        try:
            cluster_ids = np.loadtxt(clu_file, dtype=np.int32, skiprows=1)
        except ValueError:
            logger.warning(f"Could not read {clu_file}")
            continue

        if len(spike_times_samples) != len(cluster_ids):
            logger.error(
                f"Mismatch in {session.id} shank {shank_id}: res {len(spike_times_samples)} vs clu {len(cluster_ids)}"
            )
            continue

        # Group by cluster
        unique_clusters = np.unique(cluster_ids)

        for clu_id in unique_clusters:
            if clu_id < min_cluster_id:
                continue

            mask = cluster_ids == clu_id
            times_samples = spike_times_samples[mask]
            times_seconds = times_samples / session.sampling_rate

            unit_id = f"{session.id}_sh{shank_id}_c{clu_id}"
            units.append(
                Unit(
                    id=unit_id,
                    session_id=session.id,
                    shank_id=shank_id,
                    cluster_id=int(clu_id),
                    spike_times=times_seconds,
                )
            )

    return units


def load_session_metadata(session_path: Path) -> Session:
    """Creates a Session object by parsing XML."""
    session_id = session_path.name
    xml_path = session_path / f"{session_id}.xml"

    parsed = parse_xml(xml_path)

    return Session(
        id=session_id,
        path=session_path,
        n_channels=parsed["n_channels"],
        sampling_rate=parsed["sampling_rate"],
        lfp_sampling_rate=parsed["lfp_sampling_rate"],
        shank_map=parsed["shank_map"],
    )


def scan_sessions(data_root: Path) -> List[Session]:
    """Recursive scan for valid session folders."""
    sessions = []
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return []

    # Heuristic: look for .xml files
    for xml_file in data_root.rglob("*.xml"):
        session_path = xml_file.parent
        # Check if folder name matches XML name (basic validation)
        if xml_file.stem == session_path.name:
            try:
                s = load_session_metadata(session_path)
                sessions.append(s)
            except Exception as e:
                logger.warning(f"Failed to load session at {session_path}: {e}")

    return sorted(sessions, key=lambda s: s.id)


def load_behavior(session: Session) -> pd.DataFrame:
    """
    Parses .whl file for position data.
    .whl contains x,y coordinates for two LEDs (front/back) at 39.0625 Hz.
    -1 means missing data.

    Returns:
        DataFrame with columns [t, x, y, speed]
    """
    whl_file = session.path / f"{session.id}.whl"
    if not whl_file.exists():
        # logger.warning(f"No .whl file found for {session.id}")
        return pd.DataFrame()

    try:
        # Load data
        df = pd.read_csv(
            whl_file, sep=r"\s+", names=["x1", "y1", "x2", "y2"], header=None
        )

        # Sampling rate for behavior is usually file_sampling_rate / 512
        # Standard in hc-3 is often 39.0625 Hz (20000/512)
        fs_beh = session.sampling_rate / 512.0
        df["t"] = df.index / fs_beh

        # Average LEDs to get centroid (handle missing data -1)
        # Simple approach: masked where > 0
        df_masked = df.replace(-1, np.nan)
        df["x"] = df_masked[["x1", "x2"]].mean(axis=1)
        df["y"] = df_masked[["y1", "y2"]].mean(axis=1)

        # Calculate speed (pixels / s)
        dx = df["x"].diff()
        dy = df["y"].diff()
        dist = np.sqrt(dx**2 + dy**2)
        df["speed"] = dist * fs_beh

        # Smooth speed
        df["speed"] = (
            df["speed"].fillna(0).rolling(window=int(fs_beh / 2), center=True).mean()
        )

        return df[["t", "x", "y", "speed"]]

    except Exception as e:
        logger.error(f"Failed to load behavior for {session.id}: {e}")
        return pd.DataFrame()
