"""
Comprehensive Benchmark Runner V2.

Orchestrates experiments across 4 domains and 12 datasets using the unified train.py interface.
Matches dataset parameters (e.g. image sizes) and model complexities (parameter matching).
"""

import argparse
import subprocess
import sys
from datetime import datetime

# Domain Datasets map
DATASETS = {
    "image": ["CIFAR10", "FashionMNIST", "TinyImageNet"],
    "audio": ["SpeechCommands", "ESC50", "UrbanSound8K"],
    "timeseries": ["ETTh1", "ETTm1", "Exchange"],
    "nlp": ["IMDB", "AG_NEWS", "SST5"],
}

# Parameter matching targets (approximate)
# We tune CV-KAN d_complex/n_layers to match these baselines
TARGET_PARAMS = {
    "image": 250000,  # ~ResNet18-tiny
    "audio": 100000,  # ~AudioCNN-tiny
    "timeseries": 50000,  # ~LSTM/DLinear
    "nlp": 500000,  # ~Small Transformer/LSTM
}

# Heuristics for CV-KAN config to match param targets
# (d_complex, n_layers)
CVKAN_CONFIGS = {
    "image": {
        "CIFAR10": {"d_complex": 128, "n_layers": 4},
        "FashionMNIST": {"d_complex": 128, "n_layers": 4},
        "TinyImageNet": {"d_complex": 160, "n_layers": 4},
    },
    "audio": {"default": {"d_complex": 96, "n_layers": 4}},
    "timeseries": {"default": {"d_complex": 64, "n_layers": 2}},
    "nlp": {"default": {"d_complex": 128, "n_layers": 2}},
}


def get_cvkan_args(domain, dataset):
    if domain in CVKAN_CONFIGS and dataset in CVKAN_CONFIGS[domain]:
        return CVKAN_CONFIGS[domain][dataset]
    return CVKAN_CONFIGS[domain]["default"]


def run_experiment(
    domain,
    dataset,
    model_type="cvkan",
    output_dir="outputs/comprehensive_v2",
    epochs=100,
    patience=5,
    dry_run=False,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{domain}_{dataset}_{model_type}_{timestamp}"

    cmd = [
        sys.executable,
        "experiments/train.py",
        "--domain",
        domain,
        "--dataset",
        dataset,
        "--run_name",
        run_name,
        "--output_dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
    ]

    # Add model-specific args
    if model_type == "cvkan":
        cfg = get_cvkan_args(domain, dataset)
        cmd.extend(["--d_complex", str(cfg["d_complex"])])
        cmd.extend(["--n_layers", str(cfg["n_layers"])])
    else:
        # TODO: Add baseline flag handling when baselines are integrated into train.py or separate scripts
        # For now, we assume train.py might support a --model flag or we use specific baseline scripts
        # If baselines are separate scripts, we'd switch 'script' here.
        # CURRENTLY: train.py only runs CV-KAN variants defined in domain.create_model
        # To run baselines, we likely need the baseline scripts from previous comprehensive_benchmark logic
        # OR we update domain.create_model to accept a --model_arch flag.
        # Given constraint, I will stick to CV-KAN for now to verify infrastructure,
        # but mark baselines as TODO or use the existing baseline_scripts strategy from run_benchmark.py
        pass

    print(f"Running: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str, default="all", choices=["all", "image", "audio", "timeseries", "nlp"]
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per experiment")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    domains_to_run = (
        [args.domain] if args.domain != "all" else ["image", "audio", "timeseries", "nlp"]
    )

    for domain in domains_to_run:
        datasets = DATASETS[domain]
        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Validating {domain} / {dataset}")
            print(f"{'='*50}")

            try:
                run_experiment(
                    domain,
                    dataset,
                    epochs=args.epochs,
                    patience=args.patience,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"FAILED {domain}/{dataset}: {e}")


if __name__ == "__main__":
    main()
