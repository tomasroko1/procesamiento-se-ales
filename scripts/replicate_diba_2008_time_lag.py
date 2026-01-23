import sys
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hc3.io import load_session_metadata, load_spikes, load_behavior
from hc3.spikes import compute_ccg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Replicate Diba 2008 (Time Lag).")
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    
    session_path = data_root / args.session 
    if not session_path.exists():
         session_path = list(data_root.rglob(args.session))[0]

    logger.info(f"Replicating Diba 2008 for {session_path.name}")
    
    session = load_session_metadata(session_path)
    
    # 1. Load Data
    units = load_spikes(session)
    beh = load_behavior(session)
    
    if beh.empty:
        logger.warning("No behavior data (.whl) found. Cannot correlate with position. Running CCG only.")
        
    out_dir = output_root / args.session / "replication_diba"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Pairwise CCGs
    # Pick a few high firing rate units to demo
    units = sorted(units, key=lambda u: u.n_spikes, reverse=True)[:5]
    if len(units) < 2:
        logger.warning("Not enough units for pairwise analysis.")
        return

    logger.info(f"Computing CCGs for {len(units)} top units.")
    
    # Example: Pair 0 vs 1
    u1 = units[0]
    u2 = units[1]
    
    bin_centers, counts = compute_ccg(u1.spike_times, u2.spike_times, bin_size_s=0.005, window_s=0.5)
    
    plt.figure(figsize=(8, 4))
    plt.bar(bin_centers * 1000, counts, width=5, color='gray') # ms
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Time Lag (ms)")
    plt.ylabel("Spike Coincidences")
    plt.title(f"CCG: Unit {u1.cluster_id} vs {u2.cluster_id}")
    plt.savefig(out_dir / f"ccg_{u1.cluster_id}_{u2.cluster_id}.png")
    
    # 3. Place Fields (if behavior exists)
    if not beh.empty:
        # Simplistic 1D place field (linearize position or just use X if linear track)
        # Using X coordinate for demo
        pos_bins = np.linspace(beh['x'].min(), beh['x'].max(), 50)
        
        # Calculate occupancy
        occupancy, _ = np.histogram(beh['x'], bins=pos_bins)
        occupancy = occupancy.astype(float)
        occupancy[occupancy == 0] = np.nan # Avoid div by zero
        
        # Calculate place fields
        plt.figure(figsize=(10, 5))
        
        for u in units:
            # Get spike positions
            # Interpolate position at spike times
            spike_x = np.interp(u.spike_times, beh['t'], beh['x'])
            
            # Histogram spike positions
            spike_hist, _ = np.histogram(spike_x, bins=pos_bins)
            
            rate = spike_hist / occupancy * (1/0.0390625) # approx rate validation needed
            rate = rate / np.nanmax(rate) # Normalize
            
            centers = (pos_bins[:-1] + pos_bins[1:]) / 2
            plt.plot(centers, rate, label=f"U{u.cluster_id}")
            
        plt.xlabel("Position (X)")
        plt.ylabel("Normalized Firing Rate")
        plt.title("Place Fields (1D Projection)")
        plt.legend()
        plt.savefig(out_dir / "place_fields.png")
        
    logger.info(f"Done. Results in {out_dir}")

    # Generate small report snippet
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Processed {len(units)} units.\n")
        f.write(f"Behavior data found: {not beh.empty}\n")

if __name__ == "__main__":
    main()
