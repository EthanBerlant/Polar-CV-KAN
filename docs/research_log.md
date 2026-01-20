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
