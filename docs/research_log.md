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
