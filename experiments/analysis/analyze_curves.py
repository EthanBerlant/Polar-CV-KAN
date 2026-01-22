"""Analyze training curves from benchmark results."""

import json
from pathlib import Path

# Get project root from file location
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "outputs" / "benchmark"

print("=" * 90)
print("TIMESERIES CV-KAN - Training Curve Analysis")
print("=" * 90)
print(
    f"{'Config':<20} | {'Epochs':<6} | {'Train MSE':<10} | {'Val MSE':<10} | {'Test MSE':<10} | {'Status':<12}"
)
print("-" * 90)

for run_dir in sorted((results_dir / "timeseries").iterdir()):
    if "cvkan" not in run_dir.name:
        continue
    results_file = run_dir / "results.json"
    if not results_file.exists():
        continue

    with open(results_file) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    hist = data.get("history", [])

    if not hist:
        continue

    epochs = len(hist)
    last = hist[-1]
    train_mse = last.get("train", {}).get("mse", 0)
    val_mse = last.get("val", {}).get("mse", 0)
    test_mse = data.get("test_mse", 0)

    # Determine if early stopped or still improving
    if epochs < 10:
        status = "Early Stop"
    elif len(hist) >= 2:
        prev_val = hist[-2].get("val", {}).get("mse", 999)
        if val_mse < prev_val:
            status = "Improving"
        else:
            status = "Plateau"
    else:
        status = "Unknown"

    config_str = f"d{cfg.get('d_complex', '?')} L{cfg.get('n_layers', '?')} s{cfg.get('seed', '?')}"
    print(
        f"{config_str:<20} | {epochs:<6} | {train_mse:<10.4f} | {val_mse:<10.4f} | {test_mse:<10.4f} | {status:<12}"
    )

print()
print("=" * 90)
print("IMAGE CV-KAN - Training Curve Analysis")
print("=" * 90)
print(
    f"{'Config':<20} | {'Epochs':<6} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Status':<12}"
)
print("-" * 90)

for run_dir in sorted((results_dir / "image").iterdir()):
    if "cvkan" not in run_dir.name:
        continue
    results_file = run_dir / "results.json"
    if not results_file.exists():
        continue

    with open(results_file) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    hist = data.get("history", [])

    if not hist:
        continue

    epochs = len(hist)
    last = hist[-1]
    train_acc = last.get("train", {}).get("accuracy", 0)
    val_acc = last.get("val", {}).get("accuracy", 0)
    test_acc = data.get("test_acc", 0)

    if epochs < 10:
        status = "Early Stop"
    elif len(hist) >= 2:
        prev_val = hist[-2].get("val", {}).get("accuracy", 0)
        if val_acc > prev_val:
            status = "Improving"
        else:
            status = "Plateau"
    else:
        status = "Unknown"

    config_str = f"d{cfg.get('d_complex', '?')} L{cfg.get('n_layers', '?')} s{cfg.get('seed', '?')}"
    print(
        f"{config_str:<20} | {epochs:<6} | {train_acc:<10.2f} | {val_acc:<10.2f} | {test_acc:<10.2f} | {status:<12}"
    )

# Epoch-by-epoch analysis for one run
print()
print("=" * 90)
print("SAMPLE TRAINING CURVE: timeseries_cvkan_d128_L4_s42")
print("=" * 90)

sample_file = results_dir / "timeseries" / "timeseries_cvkan_d128_L4_s42" / "results.json"
if sample_file.exists():
    with open(sample_file) as f:
        data = json.load(f)
    hist = data.get("history", [])
    print(f"{'Epoch':<6} | {'Train MSE':<12} | {'Val MSE':<12} | {'LR':<12}")
    print("-" * 50)
    for h in hist:
        e = h.get("epoch", 0)
        t = h.get("train", {}).get("mse", 0)
        v = h.get("val", {}).get("mse", 0)
        lr = h.get("lr", 0)
        print(f"{e:<6} | {t:<12.6f} | {v:<12.6f} | {lr:<12.8f}")

print()
print("=" * 90)
print("SAMPLE TRAINING CURVE: image_cvkan_d256_L4_s123 (best image run)")
print("=" * 90)

sample_file = results_dir / "image" / "image_cvkan_d256_L4_s123" / "results.json"
if sample_file.exists():
    with open(sample_file) as f:
        data = json.load(f)
    hist = data.get("history", [])
    print(f"{'Epoch':<6} | {'Train Acc':<12} | {'Val Acc':<12} | {'Train Loss':<12}")
    print("-" * 55)
    for h in hist:
        e = h.get("epoch", 0)
        t_acc = h.get("train", {}).get("accuracy", 0)
        v_acc = h.get("val", {}).get("accuracy", 0)
        t_loss = h.get("train", {}).get("loss", 0)
        print(f"{e:<6} | {t_acc:<12.2f} | {v_acc:<12.2f} | {t_loss:<12.4f}")
