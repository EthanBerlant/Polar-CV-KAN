# CV-KAN Domain Adaptation Benchmarking Plan

## Overview
This document outlines the strategy for benchmarking the newly implemented CV-KAN domain adaptation modules (`CVKANImageClassifier`, `CVKANTimeSeries`, `CVKANAudio`).

The primary goal is to demonstrate that CV-KAN's **phase-based mechanics** and **complex-valued processing** can achieve competitive performance with industry-standard baselines while maintaining comparable parameter counts.

## General Methodology

### Parameter Budget Constraint
All comparisons will be **parameter-matched**. We will tune the `d_complex`, `kan_hidden`, and `n_layers` of the CV-KAN models to match the parameter count of the chosen baselines within ±10%.

### Training Protocol
- **Optimizer**: AdamW for all models.
- **Scheduler**: Cosine Annealing with Warmup.
- **Hardware**: Single GPU (typical consumer/research grade).
- **Repetitions**: 3 runs per configuration (mean ± std dev).

---

## Domain 1: Image Classification

### Dataset
**CIFAR-10** (and optional **FashionMNIST** for quick iteration)
- **Why**: Standard benchmark, manageable size for rapid experimentation.
- **Input**: 32x32 RGB images.

### Baselines
1.  **ViT-Tiny (Vision Transformer)**
    - *Rationale*: Architectural cousin (patches $\to$ sequences). Best direct comparison for patch-based CV-KAN.
    - *Config*: ~5-6M parameters.
2.  **ResNet-18 (Lightweight/Modified)**
    - *Rationale*: Standard feature extraction baseline.

### CV-KAN Configuration
- **Model**: `CVKANImageClassifier`
- **Key Hyperparams**:
    - `patch_size`: 4x4 (for CIFAR's 32x32 size)
    - `pos_encoding`: `Complex2DPositionalEncoding` (vs `Learnable`)
    - `aggregation`: Compare `LocalWindowAggregation` vs `GlobalMeanAggregation`.

### Metrics
- Top-1 Accuracy
- Convergence Speed (Epochs to 90% accuracy)

---

## Domain 2: Time Series Forecasting

### Dataset
**ETTh1** (Electricity Transformer Temperature)
- **Why**: Industry standard for multivariate time series forecasting.
- **Task**: Multivariate prediction (lookback 96, predict 96).

### Baselines
1.  **Vanilla Transformer**
    - *Rationale*: Standard sequence model benchmark.
2.  **LSTM**
    - *Rationale*: Classic recurrent baseline.
3.  **NLinear / DLinear**
    - *Rationale*: Strong, efficient modern MLPs specifically for time series.

### CV-KAN Configuration
- **Model**: `CVKANTimeSeries`
- **Key Hyperparams**:
    - `output_mode`: Compare `real` vs `phase` (hypothesis: phase captures seasonality better).
    - `aggregation`: `CausalAggregation`.

### Metrics
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

---

## Domain 3: Audio/Speech Classification

### Dataset
**Speech Commands V2** (Google)
- **Why**: Standard keyword spotting task ("Yes", "No", "Up", "Down").
- **Input**: 1-second audio clips (16kHz).

### Baselines
1.  **M5 (Simple CNN)**
    - *Rationale*: Common lightweight baseline for direct waveform/spectrogram processing.
2.  **Audio Spectrogram Transformer (AST) - Nano**
    - *Rationale*: Complex attention-based baseline.

### CV-KAN Configuration
- **Model**: `CVKANAudio`
- **Key Hyperparams**:
    - `use_stft_frontend`: True.
    - `pooling`: Compare `mean` vs `attention` pooling.
    - `d_complex`: Match bin count or project closer to baseline width.

### Metrics
- Classification Accuracy
- Parameter Efficiency (Accuracy per Parameter)

---

## Implementation Roadmap

### Phase 1: Data Preparation
- Create `src/data/` loaders for CIFAR-10, ETTh1, and Speech Commands.
- Ensure efficient caching and pre-processing (especially for STFT).

### Phase 2: Adaptation Scripts
Create new training scripts in `experiments/`:
1.  `train_image.py`: Adapts `train_synthetic.py` loop for image dataloaders.
2.  `train_timeseries.py`: Implements sliding window logic for forecasting.
3.  `train_audio.py`: Handles audio IO and on-the-fly spectrograms.

### Phase 3: Execution
1.  **Pilot Run**: Rapid training on small subsets (10%) to verify pipeline.
2.  **Parameter Tuning**: Adjust CV-KAN width/depth to match baseline parameter counts.
3.  **Full Benchmark**: Run full training for all domains.
