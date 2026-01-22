import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .utils import cleanup_gpu


class EarlyStopper:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_at_best = None

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.val_score_at_best = val_score
        elif self._is_better(val_score, self.best_score):
            self.best_score = val_score
            self.val_score_at_best = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        # mode == 'max'
        return current > (best + self.min_delta)


class BaseTrainer:
    """Base class for all training scripts."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str,
        output_dir: Path,
        args: Any,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.args = args
        self.use_amp = use_amp

        # AMP Scaler
        self.scaler = (
            torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None
        )

        # Logging
        self.history = []
        self.best_val_metric = float("inf") if args.metric_mode == "min" else float("-inf")

    def train_step(self, batch) -> dict[str, float]:
        """Implement domain-specific training logic."""
        raise NotImplementedError

    def validate_step(self, batch) -> dict[str, float]:
        """Implement domain-specific validation logic."""
        raise NotImplementedError

    def train_epoch(self, dataloader, epoch, metric_name="metric") -> dict[str, float]:
        self.model.train()
        total_metrics = {}
        n_batches = 0

        # Format best metric for display
        best_val = self.best_val_metric
        if best_val in [float("inf"), float("-inf")]:
            best_str = "init"
        else:
            best_str = f"{best_val:.2f}"

        pbar = tqdm(
            dataloader,
            desc=f"E{epoch}/{self.args.epochs} best_{metric_name}={best_str}",
            leave=False,
        )
        for batch in pbar:
            # Move batch to device is handled in train_step usually,
            # but let's leave it to implementations or do it here if standard.
            # We'll let implementations handle device movement for flexibility.

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast("cuda"):
                    metrics = self.train_step(batch)
                    loss = metrics["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                metrics = self.train_step(batch)
                loss = metrics["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                val_item = v.item() if torch.is_tensor(v) else v
                total_metrics[k] = total_metrics.get(k, 0) + val_item

            n_batches += 1

            # Update pbar
            # display_metrics = {k: f"{v:.4f}" for k,v in metrics.items()}
            # pbar.set_postfix(display_metrics)
            # Only show loss in progress bar to avoid clutter
            pbar.set_postfix({"loss": f"{metrics['loss'].item():.4f}"})

        # Average metrics
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict[str, float]:
        self.model.eval()
        total_metrics = {}
        n_batches = 0

        for batch in dataloader:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    metrics = self.validate_step(batch)
            else:
                metrics = self.validate_step(batch)

            for k, v in metrics.items():
                val_item = v.item() if torch.is_tensor(v) else v
                total_metrics[k] = total_metrics.get(k, 0) + val_item
            n_batches += 1

        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        return avg_metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        patience: int = 10,
        metric_name: str = "accuracy",
    ):
        early_stopper = EarlyStopper(patience=patience, mode=self.args.metric_mode)
        start_time = time.time()

        # Build config string for line 1
        prog = os.environ.get("BENCHMARK_PROG", "")
        prefix = f"[{prog}] " if prog else ""

        config_str = (
            f"{prefix}[{self.args.domain.upper()}] d={self.args.d_complex} L={self.args.n_layers}"
        )
        if hasattr(self.args, "embedding_type"):
            config_str += f" emb={self.args.embedding_type}"
        config_str += f" seed={self.args.seed}"

        # Initial header print
        print(f"\n{config_str}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train and measure speed
            train_start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch, metric_name)
            train_duration = time.time() - train_start
            train_speed = len(train_loader) / train_duration if train_duration > 0 else 0
            train_loss = train_metrics.get("loss", 0)

            # Format best metric for display
            best_val = self.best_val_metric
            if best_val in [float("inf"), float("-inf")]:
                best_str = "init"
            else:
                best_str = f"{best_val:.2f}"

            # Show validation status with persistent stats
            # Overwrites the completed tqdm bar
            status = f"  E{epoch}/{epochs} best_{metric_name}={best_str} loss={train_loss:.3f} speed={train_speed:.1f}it/s Validating..."
            sys.stdout.write(f"\r{status:<100}")
            sys.stdout.flush()

            val_metrics = self.evaluate(val_loader)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[metric_name])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Track history
            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": current_lr,
                    "epoch_time": epoch_time,
                }
            )

            # Periodic GPU memory cleanup to prevent OOM
            cleanup_gpu()

            # Checkpoint best
            val_score = val_metrics[metric_name]
            is_best = False
            if self.args.metric_mode == "min":
                if val_score < self.best_val_metric:
                    is_best = True
            elif val_score > self.best_val_metric:
                is_best = True

            if is_best:
                self.best_val_metric = val_score
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(self.args),
                    },
                    self.output_dir / "best.pt",
                )

            # Print epoch status (Line 2) - Move speed here, avoid updating Line 1 with ANSI
            train_loss = train_metrics.get("loss", 0)
            best_disp = (
                self.best_val_metric
                if self.best_val_metric not in [float("inf"), float("-inf")]
                else val_score
            )
            mark = "✓" if is_best else " "
            progress = f"  E{epoch}/{epochs} best_{metric_name}={best_disp:.2f} loss={train_loss:.3f} val={val_score:.2f} speed={train_speed:.1f}it/s {mark}"
            sys.stdout.write(f"\r{progress:<100}")
            sys.stdout.flush()

            # Early stopping
            if early_stopper(val_score):
                break

        total_time = time.time() - start_time

        # Final output: print line 1 with final results (overwrites progress line)
        final_best = (
            f"{self.best_val_metric:.2f}"
            if self.best_val_metric not in [float("inf"), float("-inf")]
            else "N/A"
        )
        final_line = f"{config_str} | best_{metric_name}={final_best} ({total_time:.0f}s)"
        sys.stdout.write(f"\r{final_line:<100}\n")
        sys.stdout.flush()

        return self.history, total_time
