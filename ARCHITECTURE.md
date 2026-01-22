# Polar CV-KAN Architecture

> A Complex-Valued Kolmogorov-Arnold Network using polar coordinate decomposition for implicit attention via phase alignment and magnitude polarization.

---

## Philosophy & Conceptual Basis

### Why Complex-Valued Neural Networks?

Traditional neural networks operate in purely real-valued spaces. Complex numbers offer a richer representational substrate:

| Property | Real Networks | Complex Networks |
|----------|---------------|------------------|
| **Angle/Phase** | Requires 2+ neurons | Native representation |
| **Rotation** | Learned approximation | Multiplication by unit complex |
| **Interference** | Not natural | Constructive/destructive natively |

The insight: **attention can be viewed as interference**. When tokens "agree," their phases align and magnitudes add constructively. When they "disagree," destructive interference diminishes their combined influence.

### Why Kolmogorov-Arnold Networks (KAN)?

Complex-valued neural networks face a fundamental challenge: **choosing activation functions is difficult**. Every complex activation comes with tradeoffs—Liouville's theorem tells us no bounded holomorphic function is non-constant, so you must sacrifice one desirable property for another.

KAN sidesteps this entirely. The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be decomposed into sums of univariate functions. Instead of hardcoding an activation, we **learn 1D transformations** on magnitude and phase independently:

- No need to choose between CReLU, zReLU, modReLU, or other complex activations
- Transformations are interpretable: "amplify large magnitudes" or "rotate phases toward alignment"
- Simpler primitives may yield better generalization

> [!WARNING]
> **Theoretical gap**: The Kolmogorov-Arnold theorem is proven for real-valued functions only. Whether an analogous representation exists for complex-valued functions is an open question that warrants formal investigation.

### The Polarization Hypothesis

Traditional attention uses softmax to create a probability distribution over tokens. We propose an alternative:

> **Implicit attention emerges from magnitude polarization**: important tokens naturally grow in magnitude while unimportant tokens shrink, without explicit attention weights.

The inspiration comes from **polar error correction**: using simple, repeated operations that naturally polarize information. Each pass through a polarizing block nudges magnitudes apart—signal accumulates while noise diminishes. No explicit "attention scores" are computed; the separation emerges from the dynamics.

### Core Mechanics

The core idea is **polarization**—the iterative amplification/suppression mechanism. Complex-valued representations with phase alignment provide an **elegant realization** of this idea: when tokens "agree," their phases align and magnitudes add constructively; when they "disagree," destructive interference diminishes their combined influence.

```
Token Sequence → Aggregate → Polar Decompose → Transform → Recompose → Broadcast
     Z              A          (log r, θ)      (r', θ')      A'          Z + A'
```

1. **Aggregate**: Combine tokens into a summary (mean, attention, local window)
2. **Decompose**: Split into log-magnitude and phase angle (polar coordinates)
3. **Transform**: Apply learned 1D functions to magnitude and phase independently
4. **Recompose**: Reconstruct complex representation
5. **Broadcast**: Add transformed aggregate back to all tokens (residual)

> [!NOTE]
> "Polar" appears in two senses: **polar coordinates** (the magnitude/phase representation) and **polarization** (the iterative separation mechanism from polar error correction). The naming is convenient but the concepts are distinct.

### Key Finding: Normalization Suppresses Polarization

> [!IMPORTANT]
> **Layer normalization destroys the attention signal.** By normalizing magnitudes, we prevent the "polarization" effect where important tokens grow larger. Best performance achieved with `norm_type='none'`.

This discovery validates the polarization hypothesis: magnitude IS the attention mechanism.

---

## Module Taxonomy

### Stability Markers

| Marker | Meaning |
|--------|---------|
| 🔒 **Stable** | Core abstraction, API unlikely to change |
| 🧪 **Experimental** | Active development, may change significantly |
| ⚠️ **Deprecated** | Kept for reference, avoid in new code |

### Core Modules (`src/modules/`)

| Module | Status | Purpose |
|--------|--------|---------|
| `PolarizingBlock` | 🔒 | Core primitive: aggregate → polar transform → broadcast |
| `GatedPolarization` | 🔒 | Learnable interpolation between identity and polarization |
| `PhaseAttentionBlock` | 🧪 | Explicit phase-based attention (alternative to implicit) |

### Multi-Head Approaches (`src/modules/multi_head.py`)

| Approach | Status | Description |
|----------|--------|-------------|
| `EmergentHeadsPolarizing` | 🔒 | Implicit heads via dimension diversity (STANDARD) |
| `PhaseOffsetPolarizing` | ⚠️ | Explicit heads with fixed phase rotations (DEMO ONLY) |
| `FactoredHeadsPolarizing` | ⚠️ | Decoupled phase/magnitude processing (DEMO ONLY) |

### Aggregation (`src/modules/aggregation.py`)

| Module | Status | Use Case |
|--------|--------|----------|
| `GlobalMeanAggregation` | 🔒 | Sequences (NLP) |
| `LocalWindowAggregation` | 🧪 | Spatial data (images) |
| `CausalAggregation` | 🧪 | Autoregressive tasks |

### Domain Adapters (`src/models/`)

| Model | Status | Domain |
|-------|--------|--------|
| `CVKAN` | 🔒 | Base sequence classifier |
| `CVKANImageClassifier` | 🧪 | Image classification (patch-based) |
| `CVKANAudio` | 🧪 | Audio/speech (STFT frontend) |
| `CVKANTimeSeries` | 🧪 | Forecasting |

---

## Design Decisions

### Why Log-Magnitude?

Magnitudes are multiplicative (scaling), so log-space makes transformations additive:
- `log(a * b) = log(a) + log(b)`
- Residual connections in log-space = multiplication in linear space
- More stable gradients for wide magnitude ranges

### Why Mean Aggregation by Default?

- **Bounded**: Output scale independent of sequence length (unlike sum)
- **Stable**: No exploding activations for long sequences
- **Simple**: Zero additional parameters

### Why Phase as Sin/Cos Instead of Angle?

Computing `atan2` and then `sin`/`cos` introduces unnecessary discontinuities. We work directly with the unit vector `(real/|z|, imag/|z|)` and renormalize after transformation.

---

## Extension Points

### Adding a New Aggregation Strategy

1. Create class in `src/modules/aggregation.py`
2. Inherit from base pattern (take Z, return aggregate A)
3. Add to `__init__.py` exports
4. Update this document with stability marker

### Adding a New Domain Adapter

1. Create model in `src/models/cv_kan_{domain}.py`
2. Handle domain-specific input preprocessing (patches, STFT, etc.)
3. Use shared `PolarizingBlock` stack from base
4. Add domain adapter in `experiments/domains/{domain}.py`

---

## Experiment Structure

### Unified Entry Points

| Script | Purpose |
|--------|---------|
| `experiments/train.py` | Single training entrypoint for all domains |
| `experiments/run_benchmark.py` | Batch benchmarking across domains |

### Domain Adapters (`experiments/domains/`)

Each domain module provides:
- `create_model(config)` — Model factory
- `create_dataloaders(model_config, train_config)` — Data loader factory
- `Trainer` (optional) — Domain-specific trainer subclass

### Analysis Tools (`experiments/analysis/`)

| Script | Purpose |
|--------|---------|
| `visualize.py` | Phase/magnitude visualization from checkpoints |
| `analyze_curves.py` | Training curve analysis |
| `run_diagnostics.py` | Per-sample diagnostic analysis |

### Baselines (`experiments/baselines/`)

| Script | Purpose |
|--------|---------|
| `base_trainer.py` | Shared training infrastructure |
| `*_baseline.py` | Individual baseline model definitions |

---

## Naming Conventions

| Pattern | Example | Use For |
|---------|---------|---------|
| `CVKAN*` | `CVKANImageClassifier` | Full model classes |
| `*Block` | `PolarizingBlock`, `PhaseAttentionBlock` | Layer-level components |
| `*Aggregation` | `LocalWindowAggregation` | Aggregation strategies |
| `Complex*` | `ComplexLayerNorm` | Complex-valued utilities |

---

## References

- **Kolmogorov-Arnold Networks**: [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **Complex-Valued Neural Networks**: Hirose, A. (2012)
- **Phase-Based Representations**: Reichert & Serre (2013)
