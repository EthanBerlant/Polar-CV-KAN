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

The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be decomposed into sums of univariate functions. CV-KAN applies this principle:

- Instead of learning weight matrices, we learn **1D transformations** on magnitude and phase
- These transformations are interpretable: "amplify large magnitudes" or "rotate phases toward alignment"
- Simpler primitives may yield better generalization

### The Polarization Hypothesis

Traditional attention uses softmax to create a probability distribution over tokens. We propose an alternative:

> **Implicit attention emerges from magnitude polarization**: important tokens naturally grow in magnitude while unimportant tokens shrink, without explicit attention weights.

This is analogous to:
- **Political polarization**: opinions cluster toward extremes
- **Optical polarization**: certain orientations are amplified, others filtered
- **Winner-take-all dynamics**: the rich get richer

### Core Mechanics

```
Token Sequence → Aggregate → Polar Decompose → Transform → Recompose → Broadcast
     Z              A          (log r, θ)      (r', θ')      A'          Z + A'
```

1. **Aggregate**: Combine tokens into a summary (mean, attention, local window)
2. **Decompose**: Split into log-magnitude and phase angle
3. **Transform**: Apply learned 1D functions to magnitude and phase independently
4. **Recompose**: Reconstruct complex representation
5. **Broadcast**: Add transformed aggregate back to all tokens (residual)

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
| `EmergentHeadsPolarizing` | 🔒 | Implicit heads via dimension diversity (recommended) |
| `PhaseOffsetPolarizing` | 🧪 | Explicit heads with fixed phase rotations |
| `FactoredHeadsPolarizing` | 🧪 | Decoupled phase/magnitude processing |

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
4. Add training script in `experiments/train_{domain}.py`

---

## Naming Conventions

| Pattern | Example | Use For |
|---------|---------|---------|
| `CVKAN*` | `CVKANImageClassifier` | Full model classes |
| `*Block` | `PolarizingBlock`, `PhaseAttentionBlock` | Layer-level components |
| `*Aggregation` | `LocalWindowAggregation` | Aggregation strategies |
| `Complex*` | `ComplexLayerNorm` | Complex-valued utilities |
| `train_*.py` | `train_sst2.py` | Training scripts |

---

## References

- **Kolmogorov-Arnold Networks**: [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **Complex-Valued Neural Networks**: Hirose, A. (2012)
- **Phase-Based Representations**: Reichert & Serre (2013)
