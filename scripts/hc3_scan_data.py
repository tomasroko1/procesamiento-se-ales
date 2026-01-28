import argparse
import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

# Add src to path so we can import hc3
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hc3.io import scan_sessions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Scan for hc-3 sessions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data_root"])
    print(f"Scanning data root: {data_root}")

    sessions = scan_sessions(data_root)

    if not sessions:
        print("No sessions found! Check your data_root in config.yaml.")
        sys.exit(1)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="dim")
    table.add_column("Path")
    table.add_column("Channels")
    table.add_column("Shanks")
    table.add_column("SR (Spike/LFP)")

    for s in sessions:
        table.add_row(
            s.id,
            str(
                s.path.relative_to(data_root.parent)
                if data_root.parent in s.path.parents
                else s.path
            ),
            str(s.n_channels),
            str(len(s.shank_map)),
            f"{s.sampling_rate:.0f} / {s.lfp_sampling_rate:.0f}",
        )

    console.print(table)
    print(f"\nFound {len(sessions)} valid sessions.")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        # Fallback if rich/pyyaml not installed in environment, though they are in pyproject.toml
        print("Missing dependencies. Please run 'pip install -e .'")
