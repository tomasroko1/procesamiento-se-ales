import sys
import argparse
import yaml
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hc3.io import load_session_metadata
from hc3.qc import QC

logging.basicConfig(level=logging.ERROR) # Only show errors in logs, rest in rich

def main():
    parser = argparse.ArgumentParser(description="Validate a single hc-3 session.")
    parser.add_argument("--session", type=str, required=True, help="Session ID (e.g., ec013.423) or relative path")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config["data_root"])
    
    # Try to find the session folder
    # 1. Direct path check
    session_path = data_root / args.session
    if not session_path.exists():
        # 2. Search recursively if user just gave the ID
        found = list(data_root.rglob(args.session))
        if found:
            session_path = found[0]
        else:
            print(f"Session {args.session} not found in {data_root}")
            sys.exit(1)
            
    console = Console()
    console.print(f"[bold blue]Validating {session_path.name}...[/bold blue]")
    
    try:
        session = load_session_metadata(session_path)
        qc_stats = QC.compute_session_stats(session)
        
        status_color = "green" if qc_stats["lfp_valid"] and qc_stats["n_units"] > 0 else "yellow"
        
        content = f"""
        ID: {qc_stats['id']}
        Duration: {qc_stats['duration_seconds']:.2f} s
        LFP Valid: {qc_stats['lfp_valid']}
        Units Found: {qc_stats['n_units']}
        Active Shanks: {qc_stats['n_shanks_with_spikes']}
        Avg Firing Rate: {qc_stats.get('mean_firing_rate', 0):.2f} Hz
        """
        
        console.print(Panel(content, title="Session QC", border_style=status_color))
        
        if qc_stats["n_units"] == 0:
             console.print("[red]WARNING: No units found. Check .res/.clu files.[/red]")
             
    except Exception as e:
        console.print(f"[bold red]Validation Failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
