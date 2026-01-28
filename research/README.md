# Polar CV-KAN: Theoretical Foundations Research

This directory contains experiments and analysis scripts to rigorously investigate the theoretical claims underlying the Polar CV-KAN architecture.

## Research Questions

1. **Polarization Dynamics**: Under what conditions does the aggregate-transform-broadcast operation increase magnitude separation?
2. **Phase Semantics**: What does phase encode after training?
3. **Complex Necessity**: Do complex numbers provide benefits beyond 2D real vectors?
4. **KAN Framing**: Is the Kolmogorov-Arnold connection justified or merely nominal?
5. **Gradient/Stability**: How does training work without normalization?
6. **Robustness**: Does the "Polar" structure confer geometric invariance? (Phase 6 & 7)

## Directory Structure

```
research/
├── README.md                    # This file
├── run_all.py                   # Master runner for all experiments
├── config.py                    # Shared configuration
│
├── phase1_polarization/         # Polarization dynamics
│   ├── README.md
│   ├── measure_polarization.py  # Track P(Z) across layers
│   ├── extract_transforms.py    # Extract learned f,g functions
│   ├── single_block_analysis.py # Theoretical single-block dynamics
│   └── synthetic_polarization.py# Controlled synthetic experiments
│
├── phase2_semantics/            # Phase semantics
│   ├── README.md
│   ├── phase_clustering.py      # Cluster tokens by phase
│   ├── phase_intervention.py    # Ablate phase vs magnitude
│   └── mutual_information.py    # I(phase; label) computation
│
├── phase3_complex_necessity/    # Complex vs real ablations
│   ├── README.md
│   ├── real_polar_model.py      # 2D real with polar structure
│   ├── real_cartesian_model.py  # 2D real without polar structure
│   └── run_ablations.py         # Compare all three
│
├── phase4_kan_framing/          # KAN justification
│   ├── README.md
│   └── literature_notes.md      # Analysis of KAN claims
│
├── phase5_stability/            # Gradient and stability
│   ├── README.md
│   ├── gradient_flow.py         # Track gradient norms
│   ├── magnitude_dynamics.py    # Track magnitude statistics
│   └── loss_landscape.py        # Visualize loss surface
│
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── metrics.py               # Polarization measures
│   ├── visualization.py         # Plotting functions
│   └── checkpoint_utils.py      # Load/inspect checkpoints
│
└── outputs/                     # Results go here
    ├── phase1/
    ├── phase2/
    ├── phase3/
    ├── phase4/
    └── phase5/
```

## Quick Start

```bash
# Run all experiments (will take several hours)
python research/run_all.py

# Run specific phase
python research/run_all.py --phase 1

# Run individual experiment
python research/phase1_polarization/measure_polarization.py --checkpoint outputs/sst2_best.pt
```

## Requirements

Assumes the main polar-cvkan codebase is installed. Additional:
```bash
pip install scikit-learn umap-learn
```

## Outputs

Each phase produces:
- Quantitative results in `outputs/phaseN/results.json`
- Visualizations in `outputs/phaseN/*.png`
- Summary in `outputs/phaseN/summary.md`

Final synthesis in `outputs/synthesis.md`.
