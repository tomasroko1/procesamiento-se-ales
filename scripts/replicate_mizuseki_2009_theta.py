import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hc3.io import load_lfp, load_session_metadata, load_spikes
from hc3.lfp import bandpass_filter, compute_hilbert_phase
from hc3.spikes import compute_spike_phases, vector_strength

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Replicate Mizuseki 2009 (Theta Phase)."
    )
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])

    session_path = data_root / args.session  # Assume direct path or resolved by now
    if not session_path.exists():
        session_path = list(data_root.rglob(args.session))[0]

    logger.info(f"Replicating Mizuseki 2009 for {session_path.name}")

    session = load_session_metadata(session_path)

    # 1. Get Reference Theta Phase
    # Ideally per shank, but we'll use channel 0 (or a high power channel) as global ref for simplicity
    lfp = load_lfp(
        session, channel_id=0
    )  # Load all for duration? Default 100s for speed in demo
    # Let's load more for meaningful stats
    lfp = load_lfp(session, channel_id=0, duration=300.0)

    lfp_theta = bandpass_filter(lfp, 4, 12)
    theta_phase = compute_hilbert_phase(lfp_theta)
    lfp_timestamps = np.arange(len(theta_phase)) / lfp.sampling_rate

    # 2. Get Units
    units = load_spikes(session)
    logger.info(f"Found {len(units)} units")

    results = []

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    for unit in units:
        # Compute phases
        phases = compute_spike_phases(unit, theta_phase, lfp_timestamps)

        if len(phases) < 50:
            continue

        r, angle = vector_strength(phases)

        results.append(
            {
                "Unit": unit.id,
                "Cluster": unit.cluster_id,
                "Shank": unit.shank_id,
                "MeanPhase": angle,
                "ResultantLength": r,
                "N_Spikes": len(phases),
            }
        )

        # Plot vector
        ax.plot([0, angle], [0, r], alpha=0.5, label=f"U{unit.cluster_id}")

    df = pd.DataFrame(results)

    # Save Results
    out_dir = output_root / args.session / "replication_theta"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "theta_phase_stats.csv", index=False)

    ax.set_title("Theta Phase Locking (All Units)")
    fig.savefig(out_dir / "polar_vectors.png")

    # Plot histogram of preferred phases
    plt.figure()
    sns.histplot(df["MeanPhase"], bins=20, color="k")
    plt.title("Distribution of Preferred Theta Phases")
    plt.xlabel("Phase (rad)")
    plt.savefig(out_dir / "phase_dist.png")

    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
