## Implementation Plan

### Phase 1: Core Primitive

**Goal**: Validate that phase alignment and magnitude polarization emerge meaningfully.

**Architecture**:
```python
class PolarizingBlock(nn.Module):
    def __init__(self, d_complex, kan_hidden=32):
        super().__init__()
        # Learnable 1D functions (small MLPs as KAN approximation for now)
        self.psi_mag = nn.Sequential(
            nn.Linear(1, kan_hidden), nn.GELU(),
            nn.Linear(kan_hidden, 1)
        )
        self.psi_phase = nn.Sequential(
            nn.Linear(2, kan_hidden), nn.GELU(),  # sin/cos input
            nn.Linear(kan_hidden, 2)               # sin/cos output
        )
        self.mag_scale = nn.Parameter(torch.tensor(0.1))  # small init

    def forward(self, Z):  # Z: (batch, n_tokens, d_complex) complex tensor
        # Aggregate
        A = Z.mean(dim=1, keepdim=True)  # mean more stable than sum

        # Decompose
        mag = torch.log(torch.abs(A) + 1e-6)
        phase_vec = torch.stack([A.real, A.imag], dim=-1)
        phase_vec = phase_vec / (torch.abs(A).unsqueeze(-1) + 1e-6)

        # Transform (with residual structure for stability)
        mag_delta = self.psi_mag(mag.unsqueeze(-1)).squeeze(-1)
        mag_out = mag + self.mag_scale * mag_delta

        phase_out_vec = self.psi_phase(phase_vec)
        phase_out_vec = F.normalize(phase_out_vec, dim=-1)  # stay on unit circle

        # Recompose
        r_out = torch.exp(mag_out)
        A_new = r_out * torch.complex(phase_out_vec[..., 0], phase_out_vec[..., 1])

        # Broadcast interaction back to tokens
        return Z + A_new
```

**Stability measures baked in**:
- Mean aggregation (bounded regardless of sequence length)
- Residual connection (limits per-layer change)
- Small initial scale on magnitude transform
- Phase stays normalized (no explosion)
- Log-magnitude (natural for multiplicative dynamics)

**Toy task**: Sequence classification where k of n tokens are "signal" (from one distribution) and n-k are "noise" (from another). Network must identify and aggregate signal tokens. Success metric: do signal tokens develop coherent phase? Do their magnitudes grow relative to noise?

---

### Phase 2: Multi-Head Approaches

Test three approaches in order of simplicity:

#### Approach A: Emergent Heads via Channel Diversity

**Hypothesis**: With d complex dimensions, different dimensions naturally specialize.

**Implementation**: Just use the basic architecture with d > 1. Each complex dimension evolves independently. Heads are implicit.

**Diagnostic**: After training, cluster dimensions by their learned ψ functions. Do natural groupings emerge? Do different dimensions attend to different input patterns?

**Pros**: Zero additional complexity. Maximally elegant.
**Cons**: No guarantee of diversity. Might collapse to redundant representations.

**Diversity encouragement** (soft):
```python
# Add to loss: penalize correlation between dimensions' phase patterns
phase_matrix = torch.angle(Z)  # (batch, n_tokens, d)
phase_corr = corrcoef(phase_matrix.reshape(-1, d).T)
diversity_loss = (phase_corr.triu(1) ** 2).mean()
```

#### Approach B: Explicit Phase Offsets

**Hypothesis**: Initialize different dimensions with different reference phases. Like beamforming.

**Implementation**:
```python
class MultiHeadPolarizing(nn.Module):
    def __init__(self, n_heads, d_per_head):
        super().__init__()
        self.n_heads = n_heads
        # Fixed phase offsets: 0, 2π/H, 4π/H, ...
        offsets = torch.arange(n_heads) * (2 * math.pi / n_heads)
        self.register_buffer('phase_offsets', offsets)

        # Shared polarizing transform (parameter efficient)
        self.polarizer = PolarizingBlock(d_per_head)

    def forward(self, Z):  # Z: (batch, n_tokens, n_heads, d_per_head)
        # Rotate each head by its offset
        Z_rotated = Z * torch.exp(1j * self.phase_offsets[None, None, :, None])

        # Apply shared polarizer per head
        out = self.polarizer(Z_rotated.flatten(2, 3))
        out = out.unflatten(-1, (self.n_heads, -1))

        # Rotate back
        return out * torch.exp(-1j * self.phase_offsets[None, None, :, None])
```

**Intuition**: Each head has a different "zero angle." Tokens aligning with head h's reference constructively interfere in head h. Same underlying dynamics, different frame of reference.

**Pros**: Explicit head structure. Shared parameters across heads (efficient). Guaranteed diversity of reference frames.
**Cons**: Somewhat arbitrary choice of offset structure.

#### Approach C: Factored Magnitude-Phase Heads

**Hypothesis**: Separate the "what to select" (phase) from "how much" (magnitude) across heads.

**Implementation**:
```python
class FactoredHeads(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        # Each head gets its own phase projection, shared magnitude
        self.phase_projections = nn.Parameter(torch.randn(n_heads, d_model, d_model) * 0.02)
        self.shared_mag_transform = PolarizingMagnitude(d_model)

    def forward(self, Z):
        # Project to head-specific phase spaces
        # Z: (batch, n_tokens, d)
        Z_heads = torch.einsum('btn d, h d e -> btn h e', Z, self.phase_projections)

        # Shared magnitude processing
        mags = torch.abs(Z_heads)
        mags_polar = self.shared_mag_transform(mags)

        # Phases just sum naturally per head
        phases = torch.angle(Z_heads)

        return mags_polar * torch.exp(1j * phases)
```

**Pros**: Clean factorization. Heads differ only in what phase relationships they detect.
**Cons**: More parameters than Approach B.

---

### Phase 3: Stability Mechanisms

#### Magnitude Normalization
```python
class ComplexLayerNorm(nn.Module):
    def forward(self, Z):
        # Normalize log-magnitudes to zero mean, unit variance per layer
        log_mag = torch.log(torch.abs(Z) + 1e-6)
        log_mag_norm = (log_mag - log_mag.mean(dim=-1, keepdim=True)) / (log_mag.std(dim=-1, keepdim=True) + 1e-6)

        # Preserve phase
        phase = torch.angle(Z)

        return torch.exp(log_mag_norm) * torch.exp(1j * phase)
```

#### Soft Polarization via Parameterized Strength

Control how aggressive the polarization is:
```python
class GatedPolarization(nn.Module):
    def __init__(self):
        self.polarization_strength = nn.Parameter(torch.tensor(0.0))  # starts at identity

    def forward(self, mag):
        # Interpolate between identity and full polarization
        alpha = torch.sigmoid(self.polarization_strength)
        mag_polarized = self.polarize(mag)  # learned aggressive function
        return (1 - alpha) * mag + alpha * mag_polarized
```

Network learns how much polarization it needs. Starts linear, becomes nonlinear if helpful.

#### Phase Anchoring

Prevent phase drift/chaos by soft attraction to canonical angles:
```python
def phase_anchor_loss(phase, n_anchors=4):
    # Encourage phases to cluster near 0, π/2, π, 3π/2
    anchors = torch.arange(n_anchors) * (2 * math.pi / n_anchors)
    distances = torch.abs(phase.unsqueeze(-1) - anchors)
    min_dist = distances.min(dim=-1).values
    return min_dist.mean()
```

Add to training loss with small weight. Creates attractor landscape without hardcoding it.

---

### Experimental Protocol

**Stage 1: Synthetic validation**
- Task: Identify k signal tokens among n-k noise tokens
- Metric: Phase coherence of signal tokens, magnitude separation
- Compare: Basic architecture vs. random baseline

**Stage 2: Multi-head comparison**
- Task: Multiple signal classes (need to attend to different groups)
- Compare: Approaches A, B, C
- Metric: Do heads specialize? Classification accuracy?

**Stage 3: Real task**
- Task: Text classification (SST-2 or similar, small)
- Architecture: Embedding → stack of PolarizingBlocks → pool → classify
- Compare: Matched-compute Transformer, linear attention

**Stage 4: Diagnostics**
- Visualize: Phase distributions across layers (do they cluster?)
- Visualize: Magnitude distributions (do they polarize?)
- Ablate: Remove polarization → does it hurt?

---

### Recommendation

Start with **Approach A** (emergent heads) plus **diversity loss**. It's the most elegant—if it works, you've discovered that explicit head structure is unnecessary and emerges from the dynamics.

If Approach A shows head collapse (all dimensions learn the same thing), move to **Approach B** (explicit phase offsets) which guarantees diversity with minimal added complexity.

Skip quaternions/Clifford algebras for now. The S¹ phase space is interpretable and sufficient for initial validation. Higher-dimensional phase spaces are an optimization, not a necessity.
