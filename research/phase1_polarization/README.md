# Phase 1: Polarization Dynamics

## Goal

Prove (or disprove) that the aggregate-transform-broadcast operation increases magnitude separation under specific conditions.

## Key Questions

1. Does polarization (magnitude variance) increase monotonically with depth?
2. What do the learned transform functions look like?
3. Under what conditions does polarization occur vs. not occur?
4. Are there fixed points or saturation?

## Scripts

### `measure_polarization.py`
Runs a trained model on data and tracks polarization metrics at each layer.

```bash
python measure_polarization.py --checkpoint path/to/model.pt --domain sst2
```

Outputs:
- `outputs/phase1/polarization_trajectory.png`
- `outputs/phase1/layer_metrics.json`

### `extract_transforms.py`
Extracts and visualizes the learned magnitude and phase transform functions.

```bash
python extract_transforms.py --checkpoint path/to/model.pt
```

Outputs:
- `outputs/phase1/transform_magnitude_layer{i}.png`
- `outputs/phase1/transform_phase_layer{i}.png`
- `outputs/phase1/transforms.json`

### `single_block_analysis.py`
Theoretical analysis: derive conditions for polarization in a single block.

```bash
python single_block_analysis.py
```

Outputs:
- `outputs/phase1/single_block_analysis.md`
- `outputs/phase1/delta_polarization_heatmap.png`

### `synthetic_polarization.py`
Controlled experiments with synthetic data to test polarization dynamics.

```bash
python synthetic_polarization.py --n_layers 8 --epochs 50
```

Outputs:
- `outputs/phase1/synthetic_results.json`
- `outputs/phase1/synthetic_polarization_curves.png`

## Expected Outcomes

**If polarization is real:**
- Magnitude variance should increase (or at least not decrease) with depth
- Learned transforms should be non-identity (amplifying large, suppressing small)
- Performance should correlate with polarization strength

**If polarization is incidental:**
- Metrics will be noisy/inconsistent across layers
- Transforms will be near-identity
- Ablations removing polarization won't hurt performance
