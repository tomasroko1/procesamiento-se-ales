# hc3 Reproducible Analysis Portfolio

![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A professional, reproducible analysis pipeline for electrophysiology data, developed as a technical demonstration for computational neuroscience research. 

This repository replicates key analyses from classic literature using the **CRCNS hc-3 dataset** (hippocampus/entorhinal cortex).

## 🎯 Project Goals
1.  **Rigorous Data Engineering**: Robust parsing of complex neurophysiology data formats (`.xml`, `.eeg`, `.res`, `.clu`, `.whl`).
2.  **Reproducibility**: Environment managed via `pyproject.toml`, pure Python implementation, and deterministic output paths.
3.  **Scientific Validation**: Replication of published findings regarding Theta Phase Precession and Time Lags.

## 📊 Key Replications
This pipeline implements analysis modules for:

*   **Mizuseki et al. (2009)**: Theta phase locking and precession in hippocampus/EC.
    *   *Implementation*: `scripts/replicate_mizuseki_2009_theta.py`
    *   *Methods*: Hilbert transform on bandpass filtered LFP (4-12Hz), circular statistics.
*   **Diba & Buzsaki (2008)**: Theta-scale time lags and place field stability.
    *   *Implementation*: `scripts/replicate_diba_2008_time_lag.py`
    *   *Methods*: Pairwise Cross-Correlograms (CCG), Place Field estimation.

## 🚀 Quickstart

**1. Installation**
```bash
pip install -e .[dev]
```

**2. Configuration**
```bash
cp configs/config.example.yaml configs/config.yaml
# Edit config.yaml to point to your data_root
```

**3. Run Analysis**
```bash
# Scan dataset
python scripts/hc3_scan_data.py

# Generate QC Report
python scripts/make_session_report.py --session ec013.423

# Run Top-Level Replications
python scripts/replicate_mizuseki_2009_theta.py --session ec013.423
```

## 📂 Output Example
See [Sample Session Report](reports/ec013.423/REPORT.md) for automatically generated LFP traces, PSDs, and spike metrics.

## 🛠 Tech Stack
*   **Core**: `numpy`, `pandas`, `scipy` (Signal Processing)
*   **Viz**: `matplotlib`, `seaborn`
*   **DevOps**: `pytest`, `ruff`, `pyyaml`
