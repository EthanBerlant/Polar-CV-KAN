# Research Log

Qualitative findings, lessons learned, and insights from CV-KAN development.

---

## 2026-01-20 AMP Breaks Complex Number Operations

**Category**: Training
**Status**: Workaround
**Related**: `experiments/train.py`

### Finding
Automatic Mixed Precision (AMP) causes NaN gradients with complex-valued tensors. `torch.amp.autocast` casts intermediates to float16, losing precision in complex magnitude calculations.

### Impact
Cannot use AMP for training speedup.

### Action Items
- [ ] Investigate selective AMP (real-valued layers only)
- [ ] Test bfloat16 instead of float16
- [x] Disable AMP as default workaround

---

## 2026-01-20 Layer Normalization Suppresses Polarization

**Category**: Architecture
**Status**: Resolved
**Related**: `ARCHITECTURE.md`, `docs/results_summary.md`

### Finding
Layer normalization destroys the magnitude-based attention signal. By normalizing magnitudes, we prevent the "polarization" effect where important tokens grow larger.

### Impact
Best performance achieved with `norm_type='none'`. This validates the polarization hypothesis.

### Action Items
- [x] Document in ARCHITECTURE.md
- [x] Set `norm_type='none'` as default for SST-2

---

## 2026-01-20 Attention Pooling Marginally Better for Classification

**Category**: Architecture
**Status**: Resolved
**Related**: `experiments/domains/image.py`, `experiments/domains/audio.py`, MLflow runs in `outputs/pooling_test/`

### Finding
Compared mean vs attention pooling across all three domains (3 epochs each, full datasets):

| Domain | Mean Pooling | Attention Pooling | Delta |
|--------|-------------|-------------------|-------|
| Image (CIFAR-10) | 46.8% | **48.2%** | +1.4% |
| Audio (Speech Commands) | 29.7% | **30.5%** | +0.8% |
| Timeseries (ETTh1) | 0.92 MSE | 0.92 MSE | 0% |

Attention pooling provides consistent (though small) improvements for classification tasks. No difference observed for timeseries forecasting, which outputs sequences rather than pooled representations.

### Impact
- Recommend `--pooling attention` for image and audio domains
- Default pooling (`mean`) is acceptable but suboptimal for classification
- Earlier 0% audio test accuracy was due to insufficient subset size, not a bug

### Action Items
- [x] Verified attention pooling works correctly across domains
- [x] Confirmed audio pipeline is functional (30% after 3 epochs is reasonable for 35 classes)
- [ ] Update domain defaults to use attention pooling for classification tasks

---

## 2026-01-22 Audio Training Performance Optimization

**Category**: Performance
**Status**: Resolved
**Related**: `experiments/domains/audio.py`, `experiments/train.py`

### Finding
Audio training was significantly slower than timeseries due to:
1. **STFT on every forward pass** (FFT computation per batch)
2. **Large frequency dimension** (513 bins vs 7 timeseries features)
3. **Disk I/O** for waveforms + resampling per-sample

Precomputed infrastructure existed (`precompute_audio.py`, `precomputed_audio.py`) but wasn't integrated into main `train.py` workflow.

### Impact
- Integrated precomputed STFT mode into `train.py` (auto-detects `./data/speech_commands_stft/`)
- Added `speech_commands_fast` preset with `n_fft=512` (257 bins vs 513)
- Expected **2-3x speedup** when using precomputed mode

### Action Items
- [x] Integrated precomputed mode into `domains/audio.py`
- [x] Updated `train.py` to pass `use_precomputed` flag
- [x] Added `speech_commands_fast` preset
- [ ] Benchmark actual speedup with precomputed vs raw

---

## Open Research Questions

**Category**: Theory
**Status**: To Investigate

### Depth vs Width

**Observation**: Width seems to give better results than depth for parameter-matched models.

**Questions**:
- [ ] Does each layer amplify magnitude separation, or does it saturate?
- [ ] Is there a theoretical limit to polarization depth?
- [ ] Profile magnitude variance per layer to quantify polarization accumulation

### Aggregation Strategy Paradox

**Observation**: Attention-based aggregation consistently outperforms mean aggregation, but this seems to contradict the model's thesis that attention emerges implicitly from polarization.

**Hypothesis**: Mean aggregation may be "too noisy" — if only 10% of tokens carry signal, the aggregate is 90% noise, forcing polarization to work against it.

**Alternative Aggregation Strategies to Investigate**:

1. **Magnitude-Weighted Mean** (no new parameters):
   - Weight each token's contribution by its magnitude from previous layers
   - Creates feedback loop: polarized tokens → contribute more → polarize further
   - `weights = |Z| / sum(|Z|)`, then `A = weighted_sum(Z)`

2. **Local Window Aggregation**:
   - Each token aggregates from its neighborhood only
   - May be better for spatial/local structure
   - Trade-off: returns (batch, n_tokens, d) instead of (batch, d)

3. **Causal Aggregation**:
   - Token i only sees tokens 0...i (cumulative mean)
   - Required for autoregressive tasks
   - Already implemented as `CausalAggregation` (🧪)

**Key Experiment**: Test if magnitude-weighted aggregation can match attention aggregation performance while remaining parameter-free.

**Questions**:
- [ ] Is explicit aggregation attention doing the "hard work" while polarization just cleans up?
- [ ] Can magnitude-weighted aggregation close the gap with attention aggregation?
- [ ] Does the aggregation attention benefit shrink with more polarization layers?
- [ ] What specifically does aggregation attention learn vs what polarization learns?

### Phase Encoding Semantics

**Priority**: High

**Questions**:
- [ ] What do phase values encode after training?
- [ ] Do phases cluster around certain angles for semantic classes?
- [ ] Is there a relationship between phase and token position?

### Gradient Flow & Stability

**Questions**:
- [ ] Is log-magnitude the right space for transformations?
- [ ] How do gradients flow without normalization?
- [ ] What effect do different aggregation methods have on gradient dynamics?
- [ ] Are there stability issues during long training runs?

---

## 2026-01-22 Hierarchical Polarization Validated

**Category**: Architecture
**Status**: Confirmed
**Related**: `src/modules/hierarchical.py`, `experiments/compare_hierarchical.py`

### Finding
We implemented a recursive multi-scale polarization module inspired by polar error correction. Comparing it against the flat baseline on balanced signal/noise classification:

| Model | Parameters | Val Acc (Ep 3) | Result |
|-------|------------|----------------|--------|
| **Flat (Mean Agg)** | 174k | 99.7% | Strong baseline |
| **Hierarchical (Shared)** | **58k** | 93.0% | 3x param reduction, decent perf |
| **Hierarchical (Per-Level)** | 580k* | **100.0%** | **Perfect accuracy**, faster convergence |
| **Hierarchical (Mag-Weighted)** | 580k* | **100.0%** | **Perfect accuracy**, fastest convergence |
| **Hierarchical (Top-Down)** | 580k* | 99.3% | Slightly worse than bottom-up only |

*\*Note: 580k params due to default 10-level capacity. For this task (seq_len=16), only ~4 levels are active.*

### Key Insights
1. **Recursive > Flat**: Hierarchical structure achieved perfect 100% accuracy vs 99.7%.
2. **Speed**: Hierarchical models converged to >99% in **1 epoch** vs 3 epochs for flat.
3. **Mag-Weighted Aggregation**: Even stronger when combined with hierarchical structure.
4. **Top-Down Unnecessary**: Reconstruction pass didn't improve classification performance.

### Impact
- `HierarchicalPolarization` is a viable replacement for the flat stack, especially for long sequences.
- Solves the "mean aggregation noise" problem by processing locally first.

### Action Items
- [x] Implement `HierarchicalPolarization`
- [x] Verify performance gain
- [ ] Consider making it the default for sequence tasks (NLP)
