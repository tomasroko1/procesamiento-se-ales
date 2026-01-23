from typing import Dict, Any, List
import numpy as np
import logging
from .schema import Session, Unit, LFP
from .io import load_lfp, load_spikes

logger = logging.getLogger(__name__)

class QC:
    @staticmethod
    def compute_session_stats(session: Session) -> Dict[str, Any]:
        """Calculates basic stats for the session."""
        stats = {
            "id": session.id,
            "duration_seconds": 0.0,
            "n_units": 0,
            "n_shanks_with_spikes": 0,
            "lfp_valid": False
        }
        
        # Check LFP
        try:
            # Load just 1 second to verify
            lfp = load_lfp(session, duration=1.0)
            stats["lfp_valid"] = True
            
            # Estimate duration from file size
            eeg_path = session.path / f"{session.id}.eeg"
            n_samples = (eeg_path.stat().st_size / 2) / session.n_channels
            stats["duration_seconds"] = n_samples / session.lfp_sampling_rate
            
        except Exception as e:
            logger.warning(f"LFP check failed for {session.id}: {e}")
            stats["lfp_error"] = str(e)
            
        # Check Spikes
        try:
            units = load_spikes(session)
            stats["n_units"] = len(units)
            if units:
                stats["n_shanks_with_spikes"] = len(set(u.shank_id for u in units))
                
                # Unit stats
                firing_rates = [u.n_spikes / stats["duration_seconds"] for u in units]
                stats["mean_firing_rate"] = np.mean(firing_rates) if firing_rates else 0.0
                stats["max_firing_rate"] = np.max(firing_rates) if firing_rates else 0.0
        except Exception as e:
            logger.warning(f"Spike check failed for {session.id}: {e}")
            stats["spike_error"] = str(e)
            
        return stats

    @staticmethod
    def check_isi_violations(unit: Unit, refractory_period_s: float = 0.002) -> float:
        """
        Computes ISI violations.
        Returns fraction of ISIs < refractory_period. A high value indicates contamination.
        """
        if unit.n_spikes < 2:
            return 0.0
        
        isis = np.diff(unit.spike_times)
        n_violations = np.sum(isis < refractory_period_s)
        return n_violations / len(isis)
