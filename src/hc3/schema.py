from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

@dataclass
class Session:
    """Represents a recording session."""
    id: str  # e.g., "ec013.423"
    path: Path  # Path to the session folder
    n_channels: int
    sampling_rate: float
    lfp_sampling_rate: float
    shank_map: Dict[int, List[int]]  # shank_id -> list of channel_ids
    
    def __repr__(self):
        return f"Session(id={self.id}, n_channels={self.n_channels}, shanks={len(self.shank_map)})"

@dataclass
class LFP:
    """Represents Local Field Potential data."""
    data: np.ndarray  # (n_samples, n_channels) or (n_samples,)
    sampling_rate: float
    channel_ids: List[int]
    
@dataclass
class Unit:
    """Represents a single sorted neuron."""
    id: str  # Unique ID, e.g. "ec013.423_shank1_clu5"
    session_id: str
    shank_id: int
    cluster_id: int
    spike_times: np.ndarray  # In seconds (or samples, but prefer seconds for analysis)
    # spike_samples: np.ndarray # Optional: keep raw indices
    
    @property
    def n_spikes(self) -> int:
        return len(self.spike_times)

    def firing_rate(self, duration: float) -> float:
        return self.n_spikes / duration if duration > 0 else 0.0
