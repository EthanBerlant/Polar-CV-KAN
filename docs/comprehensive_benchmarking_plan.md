# Comprehensive CV-KAN Benchmarking Plan

## Overview
This document outlines the strategy for benchmarking CV-KAN against SOTA efficiency baselines across 4 domains (Image, Audio, Time Series, NLP).

## Comparison Philosophy
For every dataset, we run 3 models:
1.  **CV-KAN**: The proposed complex-valued KAN architecture.
2.  **Baseline 1 (Inductive Bias)**: The classic efficient inductive bias (CNN, LSTM, etc.).
3.  **Baseline 2 (Universal)**: The modern "universal" architecture (ViT, Transformer, Mixer).

All comparisons use **Parameter-Matched** settings (±10% param count) to test efficiency.

## Domains & Datasets

### 1. Image Classification
| Dataset | Characteristics | Baselines |
|---|---|---|
| **CIFAR-10** | 32x32, 10 classes | ResNet-18 (Small), ViT-Tiny |
| **FashionMNIST** | 28x28, 10 classes | ResNet-18 (Small), ViT-Tiny |
| **TinyImageNet** | 64x64, 200 classes | ResNet-18, ViT-Small |

### 2. Audio Classification
| Dataset | Characteristics | Baselines |
|---|---|---|
| **SpeechCommands** | 1s clips, 35 classes | Custom CNN (M5), AST Nano |
| **ESC-50** | 5s clips, 50 classes | Custom CNN, AST Nano |
| **UrbanSound8K** | <4s clips, 10 classes | Custom CNN, AST Nano |

### 3. Time Series Forecasting
| Dataset | Characteristics | Baselines |
|---|---|---|
| **ETTh1** | Electricity, Hourly | LSTM, DLinear |
| **Weather** | 21 features, Climate | LSTM, DLinear |
| **Exchange** | 8 countries, Daily | LSTM, DLinear |

### 4. NLP (Text Classification)
| Dataset | Characteristics | Baselines |
|---|---|---|
| **IMDB** | Long text, Binary | Bi-LSTM, Transformer-Tiny |
| **AG News** | 4 topics | Bi-LSTM, Transformer-Tiny |
| **SST-5** | Short text, 5 classes | Bi-LSTM, Transformer-Tiny |

## Methodology
1. **Auto-Tuning**: Run the Baseline first to get parameter count $P$. Then tune CV-KAN's $d_{complex}$ and $L$ layers to match $P \pm 10\%$.
2. **Repetition**: 3 seeds per experiment.
3. **Metrics**: Accuracy (Classification) or MSE/MAE (Forecasting).
