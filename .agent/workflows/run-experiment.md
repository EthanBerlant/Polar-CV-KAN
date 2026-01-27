---
description: How to run a tracked experiment with MLflow
---

# Running an Experiment

## Prerequisites
- Dependencies installed (`pip install -r requirements.txt`)
- MLflow available

## Quick Start

Run experiments using the universal `train.py` with standard flags:

```powershell
# Image classification (CIFAR-10) using Batch Norm and Mean Aggregation
// turbo
python src/train.py --dataset cifar10 --epochs 100 --normalization batch --aggregation mean

# NLP / Sentiment analysis (SST-2) with Layer Norm and Magnitude Weighted
// turbo
python src/train.py --dataset sst2 --d_complex 64 --normalization layer --aggregation magnitude_weighted

# Audio classification (Speech Commands)
// turbo
python src/train.py --dataset speech_commands --d_complex 128 --normalization batch
```

## View Results in MLflow

```powershell
// turbo
mlflow ui --port 5000
```

Then open http://localhost:5000 in browser.

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--dataset` | Task dataset | cifar10, sst2, speech_commands |
| `--d_complex` | Complex dimension | 32, 64, 128, 256, 512 |
| `--normalization` | Normalization type | batch, layer, none |
| `--aggregation` | Aggregation type | mean, max, polar, magnitude_weighted |
| `--epochs` | Training epochs | 10, 30, 50, 100 |

## Output Locations

- **MLflow runs**: `mlruns/` directory
- **Model checkpoints**: `outputs/`
