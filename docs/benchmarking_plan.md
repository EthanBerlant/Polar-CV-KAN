# CV-KAN Domain Adaptation Benchmarking Plan

## Overview
This document outlines the strategy for benchmarking the newly implemented CV-KAN domain adaptation modules (`CVKANImageClassifier`, `CVKANTimeSeries`, `CVKANAudio`).

The primary goal is to demonstrate that CV-KAN's **phase-based mechanics** and **complex-valued processing** can achieve competitive performance with industry-standard baselines while maintaining comparable parameter counts.

## General Methodology

### Comparison Philosophy

We support two complementary comparison modes:

1. **Parameter-matched**: Tune CV-KAN (`d_complex`, `kan_hidden`, `n_layers`) to match baseline parameter count within ±10%. Answers: *"Given equal capacity, which architecture performs better?"*

2. **Accuracy-matched**: Find the minimum CV-KAN configuration that matches baseline accuracy. Answers: *"How much more/less capacity does CV-KAN need to achieve equivalent performance?"*

Both modes provide meaningful insights—parameter-matching tests raw efficiency, while accuracy-matching reveals architectural overhead.

### Training Protocol
- **Optimizer**: AdamW for all models (SGD fails to converge on complex-valued networks).
- **Scheduler**: Cosine Annealing with Warmup.
- **Precision**: FP32 only (see Known Issues).
- **Hardware**: Single GPU (typical consumer/research grade).
- **Repetitions**: 3 runs per configuration (mean ± std dev).

### Known Issues

> [!WARNING]
> **AMP Not Supported**: Automatic Mixed Precision causes NaN gradients with complex-valued tensors. PyTorch's `ComplexHalf` support appears insufficient for magnitude calculations. This is a known limitation pending future investigation (bfloat16, selective AMP).


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

## Domain 4: NLP (Text Classification)

### Dataset
**SST-2** (Stanford Sentiment Treebank)
- **Why**: Standard binary sentiment classification benchmark.
- **Input**: Variable-length text sequences.

### Baselines
1.  **Bi-LSTM**
    - *Rationale*: Classic sequence model for text.
2.  **Transformer-Tiny**
    - *Rationale*: Modern attention-based baseline.

### CV-KAN Configuration
- **Model**: `CVKANNLP`
- **Key Hyperparams**:
    - `max_seq_len`: 64 (for SST-2)
    - `input_type`: `real` (embeddings are real-valued)
    - `pooling`: Compare `mean` vs `attention`.

### Metrics
- Classification Accuracy

---

## Implementation

### Unified Entry Points

All benchmarking uses two consolidated scripts:

1. **`experiments/train.py`** — Single training entrypoint for all domains
   ```bash
   python experiments/train.py --domain image --preset cifar10
   python experiments/train.py --domain nlp --preset sst2
   python experiments/train.py --domain timeseries --preset etth1
   python experiments/train.py --domain audio --preset speech_commands
   ```

2. **`experiments/run_benchmark.py`** — Batch benchmarking across domains
   ```bash
   python experiments/run_benchmark.py --pilot             # Quick test
   python experiments/run_benchmark.py --full              # Overnight run
   python experiments/run_benchmark.py --domains image nlp # Specific domains
   ```

### Domain Adapters

Each domain in `experiments/domains/` provides:
- `create_model(config)` — Model factory
- `create_dataloaders(model_config, train_config)` — Data loader factory
- `Trainer` (optional) — Domain-specific trainer subclass
