---
description: How to run a tracked experiment with MLflow
---

# Running an Experiment

## Prerequisites
- Dependencies installed (`pip install -r requirements.txt`)
- MLflow available

## Quick Start

Run experiments using the unified `train.py` with `--domain` and `--preset` flags:

```powershell
# Image classification (CIFAR-10)
// turbo
python experiments/train.py --domain image --preset cifar10 --epochs 100

# NLP / Sentiment analysis (SST-2)
// turbo
python experiments/train.py --domain nlp --preset sst2 --d_complex 64 --n_layers 2

# Audio classification (Speech Commands)
// turbo
python experiments/train.py --domain audio --preset speech_commands --d_complex 128

# Time series forecasting (ETTh1)
// turbo
python experiments/train.py --domain timeseries --preset etth1
```

## Run Benchmarks

```powershell
# Quick pilot test across all domains
// turbo
python experiments/run_benchmark.py --pilot

# Full overnight benchmark
// turbo
python experiments/run_benchmark.py --full

# Specific domains only
// turbo
python experiments/run_benchmark.py --pilot --domains image nlp
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
| `--domain` | Task domain | image, audio, timeseries, nlp |
| `--preset` | Model/data preset | cifar10, sst2, etth1, speech_commands |
| `--d_complex` | Complex dimension | 32, 64, 128, 256, 512 |
| `--n_layers` | Number of polarizing layers | 2, 4, 6 |
| `--pooling` | Pooling strategy | mean, max, attention |
| `--epochs` | Training epochs | 10, 30, 50, 100 |
| `--subset_size` | Use subset for pilot runs | 100, 1000 |

## Output Locations

- **MLflow runs**: `mlruns/` directory
- **Model checkpoints**: `outputs/{domain}/{run_name}/`
- **Config**: `outputs/{domain}/{run_name}/config.json`
