# Optimization Wrap-Up and Overnight Benchmark Plan

## 1. Documentation of Results
create `docs/experiments/image_optimization_log.md` covering:
- **Findings**:
  - **Optimizer**: Adam/AdamW >>> SGD (Momentum). SGD failed to converge (<25%).
  - **AMP**: Caused instability/NaNs. Disabling AMP fixed it.
  - **Architecture Pivot (The Breakdown)**:
    - **Width**: `d=512` provided capacity.
    - **Deep Embedding**: 2-layer Conv stem captured low-level features fast.
    - **Attention Pooling**: Learned aggregation broke the spatial information bottleneck.
  - **Performance**:
    - **Baseline CV-KAN**: ~45-55% (Plateaued)
    - **ViT Baseline**: ~63%
    - **New CV-KAN**: **>73%** (Epoch 27). **Surpassed ViT Baseline**.

## 2. Code Cleanup & Standardization
### A. Shared Argument Parsing
Create `experiments/arg_parser.py` to deduplicate `parse_args` in:
- `train_image.py`
- `train_audio.py`
- `train_audio_precomputed.py`
- `train_timeseries.py` (check if exists/needs creation)

### B. Audio Model Upgrade (`src/models/cv_kan_audio.py`)
- Import `AttentionPool2d`.
- Update `__init__` to instantiate it if `pooling='attention'`.
- Override `_pool` (copy logic from Image classifier).
- Note: This aligns Audio architecture with our winning Image architecture.

### C. Timeseries Investigation
- **Issue**: rapid overfitting.
- **Fix**: Add `dropout` argument to `CVKANTimeseries` and `PolarizingBlock`.

## 3. Overnight Benchmark Suite (`experiments/overnight_benchmark.ps1`)

### A. Image (CIFAR-10)
- **Run 1**: `d=512, L=6, attn, deep` (The Champion - to confirm final score)
- **Run 2**: `d=256, L=6, attn, deep` (Ablation: Is 512 needed or is it just the architecture?)

### B. Audio (SpeechCommands)
- **Run 3**: `d=256, L=6, attn` (Using new Audio class features)
- **Run 4**: `d=128, L=4, mean` (Baseline)

### C. Timeseries (ETTh1)
- **Run 5**: `d=256, L=4, dropout=0.1` (Testing regularization)

## Execution Order
1.  [x] Write experiment log (`docs/experiments/image_optimization_log.md`)
2.  [ ] Create `experiments/arg_parser.py` and refactor defined scripts.
3.  [ ] Update `CVKANAudio` (add AttentionPooling).
4.  [ ] Update `run_benchmark.py`.
5.  [ ] Create and launch `overnight_benchmark.ps1`.
