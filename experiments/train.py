"""
Unified training script for all CV-KAN domains.

Usage:
    python experiments/train.py --domain image --epochs 100
    python experiments/train.py --domain audio --d_complex 128
    python experiments/train.py --domain timeseries --pred_len 96
    python experiments/train.py --domain sst2 --n_layers 2
    python experiments/train.py --domain synthetic --task token
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from domains import DOMAINS

from src.tracking import ExperimentTracker


def add_common_args(parser):
    """Add arguments shared across all domains."""
    # Model
    parser.add_argument("--d_complex", type=int, default=64, help="Model width")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--kan_hidden", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "attention"])
    parser.add_argument(
        "--pos_encoding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "none"],
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--batch_size", type=int, default=128)

    # Control
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Mixed precision")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--subset_size", type=int, default=None, help="Use subset for pilot runs")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="MLflow experiment name")

    return parser


def parse_args():
    """Parse command line arguments."""
    # First pass: get domain to load domain-specific args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--domain", type=str, required=True, choices=list(DOMAINS.keys()), help="Training domain"
    )
    pre_args, _ = pre_parser.parse_known_args()

    # Get domain config
    domain_config = DOMAINS[pre_args.domain]

    # Full parser
    parser = argparse.ArgumentParser(
        description=f"Train CV-KAN on {pre_args.domain}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", type=str, required=True, choices=list(DOMAINS.keys()))

    # Common args
    parser = add_common_args(parser)

    # Domain-specific args
    parser = domain_config["add_args"](parser)

    # Apply domain defaults
    parser.set_defaults(**domain_config["defaults"])

    args = parser.parse_args()

    # Set metric mode from domain defaults
    args.metric_name = domain_config["defaults"].get("metric_name", "accuracy")
    args.metric_mode = domain_config["defaults"].get("metric_mode", "max")

    return args


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Domain: {args.domain}")
    print(f"Device: {device}")

    # Output directory
    run_name = args.run_name or f"{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / args.domain / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Get domain config
    domain_config = DOMAINS[args.domain]

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, data_info = domain_config["create_dataloaders"](args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Update args with data info (e.g., vocab_size, n_classes)
    for key, value in data_info.items():
        setattr(args, key, value)

    # Create model
    print("Creating model...")
    model = domain_config["create_model"](args).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Trainer
    TrainerClass = domain_config["trainer"]
    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        args=args,
        use_amp=args.amp,
    )

    # MLflow tracking
    experiment_name = args.experiment_name or f"cvkan-{args.domain}"

    with ExperimentTracker(experiment_name, run_name=run_name) as tracker:
        # Log hyperparameters
        tracker.log_params(
            {
                "domain": args.domain,
                "d_complex": args.d_complex,
                "n_layers": args.n_layers,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "pooling": args.pooling,
                "params": n_params,
            }
        )

        # Training
        print("\nStarting training...")
        history, total_time = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            metric_name=args.metric_name,
        )

        # Log epoch metrics
        for record in history:
            epoch = record["epoch"]
            tracker.log_metrics({f"train_{k}": v for k, v in record["train"].items()}, step=epoch)
            tracker.log_metrics({f"val_{k}": v for k, v in record["val"].items()}, step=epoch)

        # Final evaluation
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(test_loader)
        test_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_results.items()])
        print(f"Test results: {test_str}")

        # Log final metrics
        tracker.log_metrics({f"test_{k}": v for k, v in test_results.items()})
        tracker.log_metrics({f"best_val_{args.metric_name}": trainer.best_val_metric})

        # Save model
        tracker.log_model(model, name="best_model")

        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        print(f"Best val {args.metric_name}: {trainer.best_val_metric:.4f}")
        print(f"Results saved to {output_dir}")
        print(f"MLflow run: {tracker.run_id}")

    # Save final results
    results = {
        "domain": args.domain,
        "n_params": n_params,
        f"best_val_{args.metric_name}": trainer.best_val_metric,
        "test_results": test_results,
        "total_time_seconds": total_time,
        "history": history,
        "config": vars(args),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
