# Portfolio Summary

## Project Goal
To build a reproducible analysis pipeline for the CRCNS hc-3 dataset, demonstrating data engineering, quality control, and scientific replication skills.

## Achievements

1.  **Architecture**: Created a clean Python package (`hc3`) with separation of concerns:
    *   `hc3.io`: Robust parsers for NeuroScope XML, EEG, Spikes, and Behavior.
    *   `hc3.qc`: Automated validation of session integrity and signal quality.
    *   `hc3.lfp` & `hc3.spikes`: Core signal processing (Theta filtering, Hilbert phase, CCG).
    *   `scripts/`: Reproducible entry points for scanning, reporting, and analysis.

2.  **Quality Control**: 
    *   Scan script found 5 valid sessions.
    *   Validation script confirms `ec013.423` has 77 sorted units and valid LFP.
    *   `make_session_report.py` generates automatic LFP traces, PSDs, and spike ISI plots.

3.  **Replication 1: Theta Phase (Mizuseki 2009)**
    *   Successfully implemented theta filtering (4-12Hz) and Hilbert transform.
    *   Computed phase locking for all units in `ec013.423`.
    *   Generated polar plots of phase vectors and phase distribution histograms.

4.  **Replication 2: Time Lag (Diba 2008)**
    *   Implemented pairwise CCG computation.
    *   Implemented place field estimation (ready for sessions with `.whl` files).
    *   Demonstrated CCG lags on top firing units in `ec013.423`.

## Deliverables
- **Code**: Fully functional `src/hc3` package.
- **Reports**: Generated in `reports/ec013.423/` (HTML/Markdown + Figures).
- **Environment**: Strict dependency management via `pyproject.toml`.

## Next Steps
- obtaining sessions with `.whl` files to fully demonstrate place field stability.
- extending the analysis to population decoding.
