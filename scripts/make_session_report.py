import sys
import argparse
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hc3.io import load_session_metadata, load_lfp, load_spikes
from hc3.qc import QC
from hc3.lfp import bandpass_filter, compute_psd
from hc3.viz import plot_lfp_trace, plot_psd, plot_isi_hist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate session report.")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    
    # Locate session
    session_path = data_root / args.session
    if not session_path.exists():
        found = list(data_root.rglob(args.session))
        if found:
            session_path = found[0]
        else:
            logger.error(f"Session {args.session} not found.")
            sys.exit(1)
            
    session_output = output_root / args.session
    session_output.mkdir(parents=True, exist_ok=True)
    figures_dir = session_output / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating report for {session_path.name} in {session_output}")
    
    # 1. Load Data
    session = load_session_metadata(session_path)
    
    # 2. LFP Analysis
    logger.info("Loading LFP...")
    # Load 1st channel for default analysis
    lfp_chan = 0 # Default first channel
    try:
        lfp = load_lfp(session, channel_id=lfp_chan, duration=10.0) # Load 10s for trace
        plot_lfp_trace(lfp, title=f"Raw LFP (Ch {lfp_chan})", save_path=figures_dir / "lfp_raw.png")
        
        # Theta filter
        lfp_theta = bandpass_filter(lfp, 4, 12)
        plot_lfp_trace(lfp_theta, title=f"Theta (4-12Hz) LFP", save_path=figures_dir / "lfp_theta.png")
        
        # PSD (load more data for PSD)
        lfp_long = load_lfp(session, channel_id=lfp_chan, duration=60.0)
        freqs, psd = compute_psd(lfp_long)
        plot_psd(freqs, psd, save_path=figures_dir / "psd.png")
        
    except Exception as e:
        logger.error(f"LFP analysis failed: {e}")

    # 3. Spike Analysis
    logger.info("Loading Spikes...")
    units = load_spikes(session)
    spike_stats = []
    
    for i, unit in enumerate(units):
        isi_path = figures_dir / f"unit_{unit.cluster_id}_isi.png"
        plot_isi_hist(unit, save_path=isi_path)
        
        # Calculate max ISI violation
        isi_viol = QC.check_isi_violations(unit)
        
        spike_stats.append({
            "Unit": unit.id,
            "Cluster": unit.cluster_id,
            "Shank": unit.shank_id,
            "N_Spikes": unit.n_spikes,
            "ISI_Viol": isi_viol
        })
        
        if i >= 4:
            break # Just do first 5 for the report demo to save time/space
            
    df_spikes = pd.DataFrame(spike_stats)
    
    # 4. Generate Markdown
    report_md = f"""# Session Report: {session.id}
    
## General Info
- **Channels**: {session.n_channels}
- **Sampling Rate**: {session.sampling_rate} Hz
- **LFP Sampling Rate**: {session.lfp_sampling_rate} Hz

## LFP Analysis
![Raw LFP](figures/lfp_raw.png)
![Theta LFP](figures/lfp_theta.png)
![PSD](figures/psd.png)

## Spike Sorting Summary
Found {len(units)} units (cluster > 1).

### First 5 Units
{df_spikes.to_markdown() if not df_spikes.empty else "No spikes found."}

### Example ISIs
"""
    for row in spike_stats:
        report_md += f"![Unit {row['Cluster']} ISI](figures/unit_{row['Cluster']}_isi.png)\n"

    with open(session_output / "REPORT.md", "w") as f:
        f.write(report_md)
        
    logger.info("Report generation complete.")

if __name__ == "__main__":
    main()
