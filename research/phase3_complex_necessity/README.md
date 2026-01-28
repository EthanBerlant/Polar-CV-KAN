# Phase 3: Complex Necessity

## Goal

Determine whether complex numbers provide benefits beyond being a convenient 2D parameterization.

## Key Questions

1. Does complex arithmetic (native interference) matter, or just having 2 channels?
2. Does polar decomposition matter, or just having magnitude/phase-like features?
3. Can we get the same results with real-valued alternatives?

## Ablation Hierarchy

Three models with decreasing "complex-ness":

1. **Complex-Polar** (original): Complex numbers + polar decomposition
2. **Real-Polar**: 2D real vectors + explicit polar structure (r, θ)
3. **Real-Cartesian**: 2D real vectors + standard real operations

## Scripts

### `real_polar_model.py`
Implements the architecture using 2D real vectors with explicit polar operations.

### `real_cartesian_model.py`
Implements the architecture using 2D real vectors without polar structure.

### `run_ablations.py`
Trains all three variants and compares.

```bash
python run_ablations.py --domain sst2 --epochs 10
```

Outputs:
- `outputs/phase3/ablation_results.json`
- `outputs/phase3/ablation_comparison.png`
- `outputs/phase3/training_curves.png`

## Interpretation

**If Complex > Real-Polar > Real-Cartesian:**
- Complex arithmetic matters (native interference)
- Polar decomposition helps (even in real)
- Both structural choices contribute

**If Complex ≈ Real-Polar > Real-Cartesian:**
- Polar structure matters, complex notation is convenience
- Can simplify implementation

**If Complex ≈ Real-Polar ≈ Real-Cartesian:**
- Neither complex nor polar matters
- It's just about having 2x dimensions

**If Real-Cartesian > others:**
- Complex/polar may be hurting
- Rethink architecture
