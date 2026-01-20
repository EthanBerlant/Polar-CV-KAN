# Polar CV-KAN

A PyTorch implementation of **Polar Complex-Valued Kolmogorov-Arnold Networks (CV-KAN)**.

This architecture introduces a novel attention mechanism where tokens interact through **phase alignment** and **magnitude polarization** in the complex plane, without using explicit softmax attention matrices.

> **📖 See [ARCHITECTURE.md](ARCHITECTURE.md)** for the full philosophy, design decisions, and module taxonomy.

## Key Features

- **PolarizingBlock**: The core unit that aggregates tokens, decomposes them into polar coordinates (log-magnitude and phase), applies learnable 1D transformations (KAN-style), and recomposes them.
- **Multi-Head Approaches**:
  - `EmergentHeads`: Implicit heads via dimension specialization (best performing).
  - `PhaseOffset`: Explicit fixed phase rotations.
  - `FactoredHeads`: Decoupled phase and magnitude processing.
- **Tasks**:
  - Synthetic Signal/Noise Classification (99.1% accuracy).
  - SST-2 Sentiment Analysis (~81.6% accuracy from scratch).

## Installation

```bash
# Core dependencies (with CUDA support)
pip install -r requirements.txt

# Development tools (optional)
pip install -e ".[dev]"
```

## Structure

```
src/
  modules/       # Core layers (PolarizingBlock, multi-head variants, aggregation)
  models/        # CVKAN model definitions
  data/          # Data loaders (Synthetic, SST-2)
  losses/        # Regularization terms
  tracking.py    # MLflow experiment tracking
experiments/
  train_synthetic.py  # Train on signal/noise task
  train_sst2.py       # Train on Sentiment Analysis (with MLflow)
  train_image.py      # Train on CIFAR-10
  train_audio.py      # Train on Speech Commands
tests/
  test_modules.py     # Unit tests
.agent/workflows/     # Development workflows
```

## Usage

### Training on Synthetic Data

```bash
python experiments/train_synthetic.py --head_approach emergent
```

### Training on SST-2

```bash
python experiments/train_sst2.py --d_complex 64 --input_type real
```

## Results

| Task | Approach | Model | Result |
|---|---|---|---|
| Signal/Noise | Emergent Heads | CV-KAN | **99.1% Accuracy** |
| SST-2 | **Implicit (No Norm)** | CV-KAN | **81.6% Accuracy** |
| SST-2 | Explicit Attention | CV-KAN | **81.5% Accuracy** |
| SST-2 | Transformer | Baseline | ~79% Accuracy |

## Key Finding: The Power of No-Norm
We discovered that **layer normalization suppresses the magnitude-based "polarization" signal**.
- Without normalization (`norm_type='none'`), the original implicit CV-KAN achieves state-of-the-art performance (**81.6%**).
- Explicit phase attention also works (**81.5%**), but is not strictly necessary if normalization is removed.
- **Recommendation**: Use `norm_type='none'` for tasks relying on magnitude polarization.

## Visualization

The model learns to "polarize" important tokens by increasing their magnitude.

![Structure](visualizations/sentiment_analysis.png)
*(Example of attention-like magnitude spiking on sentiment words)*
