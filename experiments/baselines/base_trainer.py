"""
Shared training infrastructure for baseline models.

This module provides common utilities and a base runner for all baseline
model training scripts, eliminating code duplication across 12+ baseline files.

Usage:
    from experiments.baselines.base_trainer import run_baseline, BaselineTrainer

    class MyModel(nn.Module):
        ...

    if __name__ == "__main__":
        run_baseline(
            model_class=MyModel,
            domain="image",
            default_args={"hidden_dim": 64}
        )
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineTrainer:
    """
    Generic trainer for baseline models.

    Handles training loop, evaluation, checkpointing, and result saving.
    Domain-specific behavior is injected via loss_fn and metric_fn.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str,
        output_dir: Path,
        loss_fn: callable,
        metric_fn: callable,
        metric_name: str = "accuracy",
        metric_mode: str = "max",
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self.use_amp = use_amp
        self.scaler = (
            torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None
        )

        self.history = []
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")

    def train_epoch(self, dataloader, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_metric = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast("cuda"):
                    loss, metric = self._forward_batch(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, metric = self._forward_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_metric += metric
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "loss": total_loss / n_batches,
            self.metric_name: total_metric / n_batches,
        }

    def _forward_batch(self, batch) -> tuple[torch.Tensor, float]:
        """Process a single batch. Override for custom batch handling."""
        # Default: assume batch is (inputs, targets) tuple
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        else:
            # Handle dict batches
            inputs = batch.get("input", batch.get("x", batch.get("indices")))
            targets = batch.get("target", batch.get("y", batch.get("label")))

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)

        # Handle dict outputs
        if isinstance(outputs, dict):
            outputs = outputs.get("logits", outputs.get("output"))

        loss = self.loss_fn(outputs, targets)
        metric = self.metric_fn(outputs, targets)

        return loss, metric

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0
        total_metric = 0
        n_batches = 0

        for batch in dataloader:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, metric = self._forward_batch(batch)
            else:
                loss, metric = self._forward_batch(batch)

            total_loss += loss.item()
            total_metric += metric
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            self.metric_name: total_metric / n_batches,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        patience: int = 10,
    ) -> tuple[list, float]:
        """Full training loop with early stopping."""
        start_time = time.time()
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.metric_name])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": current_lr,
                }
            )

            # Check for improvement
            val_score = val_metrics[self.metric_name]
            is_best = False
            if self.metric_mode == "max":
                if val_score > self.best_metric:
                    is_best = True
                    self.best_metric = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
            elif val_score < self.best_metric:
                is_best = True
                self.best_metric = val_score
                patience_counter = 0
            else:
                patience_counter += 1

            # Save best model
            if is_best:
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best.pt",
                )

            # Print progress
            mark = "✓" if is_best else " "
            print(
                f"  E{epoch}/{epochs} | "
                f"train_{self.metric_name}={train_metrics[self.metric_name]:.4f} | "
                f"val_{self.metric_name}={val_score:.4f} | "
                f"best={self.best_metric:.4f} {mark}"
            )

            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        return self.history, total_time


def add_baseline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common baseline training arguments."""
    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--patience", type=int, default=10)

    # Model args (common)
    parser.add_argument("--d_complex", type=int, default=64, help="Matched CV-KAN width")
    parser.add_argument("--n_layers", type=int, default=4, help="Matched CV-KAN depth")

    # Control
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    parser.add_argument("--run_name", type=str, default=None)

    return parser


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer based on args."""
    if args.optimizer == "adam":
        return Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "sgd":
        return SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def save_results(
    output_dir: Path,
    model_name: str,
    args,
    n_params: int,
    history: list,
    test_metrics: dict,
    train_time: float,
):
    """Save experiment results to JSON."""
    results = {
        "model": model_name,
        "n_params": n_params,
        "config": vars(args) if hasattr(args, "__dict__") else dict(args),
        "history": history,
        "test": test_metrics,
        "train_time_seconds": train_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")


def run_baseline(
    model_class: type,
    domain: str,
    create_dataloaders: callable,
    create_model: callable,
    loss_fn: callable,
    metric_fn: callable,
    metric_name: str = "accuracy",
    metric_mode: str = "max",
    default_args: dict = None,
):
    """
    Generic entry point for running a baseline experiment.

    Args:
        model_class: The model class (for naming)
        domain: Domain name (image, audio, timeseries, nlp)
        create_dataloaders: Function(args) -> (train_loader, val_loader, test_loader, metadata)
        create_model: Function(args, metadata) -> nn.Module
        loss_fn: Loss function(outputs, targets) -> Tensor
        metric_fn: Metric function(outputs, targets) -> float
        metric_name: Name of the metric (accuracy, mse, etc.)
        metric_mode: "max" or "min"
        default_args: Default argument overrides
    """
    parser = argparse.ArgumentParser(description=f"{model_class.__name__} Baseline for {domain}")
    parser = add_baseline_args(parser)
    args = parser.parse_args()

    # Apply defaults
    if default_args:
        for k, v in default_args.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup output directory
    run_name = (
        args.run_name
        or f"{domain}_{model_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create data
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(args)

    # Create model
    print("Creating model...")
    model = create_model(args, metadata).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        metric_name=metric_name,
        metric_mode=metric_mode,
    )

    # Train
    print("Starting training...")
    history, train_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True))
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test {metric_name}: {test_metrics[metric_name]:.4f}")

    # Save results
    save_results(
        output_dir=output_dir,
        model_name=model_class.__name__,
        args=args,
        n_params=n_params,
        history=history,
        test_metrics=test_metrics,
        train_time=train_time,
    )

    print(f"\nTraining complete! Total time: {train_time/60:.1f} minutes")

    return test_metrics
