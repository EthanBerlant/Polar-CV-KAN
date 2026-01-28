# Phase 5: Gradient and Stability Analysis

## Goal

Understand training dynamics, especially why removing normalization works.

## Key Questions

1. How do gradients flow without normalization?
2. Do magnitudes explode or collapse during training?
3. What does the loss landscape look like?

## Scripts

### `gradient_flow.py`
Track gradient norms per layer during training.

```bash
python gradient_flow.py --domain sst2 --epochs 5
```

### `magnitude_dynamics.py`
Track magnitude statistics (mean, max, min, variance) during training.

```bash
python magnitude_dynamics.py --domain sst2 --epochs 5
```

### `loss_landscape.py`
Visualize loss surface around trained solution.

```bash
python loss_landscape.py --checkpoint path/to/model.pt --domain sst2
```

## Expected Insights

**Why no-norm works (hypotheses):**
1. Polarization provides implicit regularization
2. Log-magnitude transforms prevent explosion
3. Residual connections stabilize gradients
4. Task-specific: some tasks don't need normalized features

**Warning signs to look for:**
- Gradient explosion (norms >> 1)
- Gradient vanishing (norms << 1e-6)
- Magnitude explosion (|z| → ∞)
- Magnitude collapse (|z| → 0)
