# Phase 2: Phase Semantics

## Goal

Determine what phase values encode after training. Is phase meaningful, or is the complex representation just a convenient 2D parameterization?

## Key Questions

1. Do phases cluster by semantic category?
2. Is there mutual information between phase and class label?
3. What happens if we scramble phases while preserving magnitudes (and vice versa)?

## Scripts

### `phase_clustering.py`
Cluster tokens by phase and analyze cluster composition.

```bash
python phase_clustering.py --checkpoint path/to/model.pt --domain sst2
```

Outputs:
- `outputs/phase2/phase_clusters.png` (polar histogram by cluster)
- `outputs/phase2/cluster_composition.json`

### `phase_intervention.py`
Ablate phase vs magnitude contribution via interventions.

```bash
python phase_intervention.py --checkpoint path/to/model.pt --domain sst2
```

Outputs:
- `outputs/phase2/intervention_results.json`
- `outputs/phase2/intervention_comparison.png`

### `mutual_information.py`
Compute I(phase; label) and I(magnitude; label).

```bash
python mutual_information.py --checkpoint path/to/model.pt --domain sst2
```

Outputs:
- `outputs/phase2/mutual_information.json`

## Interpretation Guide

**If phase is semantically meaningful:**
- Phases will cluster by word type (POS, sentiment)
- I(phase; label) will be significant
- Scrambling phases will hurt performance more than scrambling magnitudes

**If phase is incidental:**
- Phase clusters will be arbitrary
- I(phase; label) ≈ 0
- Scrambling phases won't hurt (or will hurt less than magnitudes)

**If both matter:**
- Both interventions hurt performance
- The architecture genuinely uses both channels
