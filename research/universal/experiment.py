import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.universal.metrics import (
    compute_gini_coefficient,
    compute_gradient_norms,
    compute_phase_consistency,
)


class UniversalTrainer:
    """Unified training loop for Polar CV-KAN research.

    Automatically computes and logs:
    - Accuracy / Loss
    - Polarization (Gini)
    - Phase Consistency (if applicable)
    - Gradient Stability
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        output_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": [],
            "polarization_gini": [],
            "phase_consistency": [],
            "gradient_norms": [],
        }

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # Gradient monitoring
        grad_norms = []

        for batch in self.train_loader:
            # UniversalDataset returns (x, mask, y)
            # But standard loaders might return (x, y).
            # We handle both via "duck typing" on properites or len

            mask = None
            if isinstance(batch, dict):
                # Use 'is not None' to avoid boolean ambiguity with tensors
                x = batch.get("indices")
                if x is None:
                    x = batch.get("image")
                if x is None:
                    x = batch.get("x")

                y = batch.get("label")
                if y is None:
                    y = batch.get("y")

                mask = batch.get("mask")
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 3:
                    x, mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError("Unknown batch format")
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            if x is None or y is None:
                raise ValueError(
                    f"Could not extract x or y from batch. Keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}"
                )

            x, y = x.to(self.device), y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            # Try passing mask if model supports it
            try:
                if mask is not None:
                    out = self.model(x, mask=mask)
                else:
                    out = self.model(x)
            except TypeError:
                # Fallback if model doesn't accept mask
                out = self.model(x)

            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            loss = self.criterion(logits, y)
            loss.backward()

            # Record gradients before step
            if len(grad_norms) < 5:  # Sample first few batches only for speed
                grad_norms.append(compute_gradient_norms(self.model))

            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        # Avg grads
        avg_grads = {}
        if grad_norms:
            keys = grad_norms[0].keys()
            for k in keys:
                avg_grads[k] = np.mean([g[k] for g in grad_norms])

        return total_loss / total, correct / total, avg_grads

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0

        # Metric accumulators
        ginis = []
        phase_consistencies = []

        with torch.no_grad():
            for batch in self.test_loader:
                mask = None
                if isinstance(batch, dict):
                    x = batch.get("indices")
                    if x is None:
                        x = batch.get("image")
                    if x is None:
                        x = batch.get("x")

                    y = batch.get("label")
                    if y is None:
                        y = batch.get("y")

                    mask = batch.get("mask")
                elif isinstance(batch, (tuple, list)):
                    if len(batch) == 3:
                        x, mask, y = batch
                    elif len(batch) == 2:
                        x, y = batch

                if x is None or y is None:
                    continue  # Skip or error

                x, y = x.to(self.device), y.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                # Forward
                if mask is not None:
                    out = self.model(x, mask=mask)
                else:
                    out = self.model(x)

                if isinstance(out, dict):
                    logits = out["logits"]

                    # Compute theoretical metrics on features/pooled states
                    if "features" in out:
                        # Gini on features (magnitudes)
                        ginis.append(compute_gini_coefficient(out["features"]).item())

                        # Phase consistency (if complex data available)
                        # We often don't pass raw complex out, but if we did...
                        # Or if "pooled" is in out (before magnitude extraction)
                        # Let's check for 'pooled' (complex)
                        if "pooled" in out and torch.is_complex(out["pooled"]):
                            pc = compute_phase_consistency(out["pooled"], y)
                            phase_consistencies.append(pc)

                else:
                    logits = out

                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        metrics = {
            "test_acc": correct / total,
            "gini": np.mean(ginis) if ginis else 0.0,
            "phase_consistency": np.mean(phase_consistencies) if phase_consistencies else 0.0,
        }
        return metrics

    def train(self, epochs: int):
        print(f"Starting Universal Training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss, train_acc, grads = self.train_epoch(epoch)
            eval_metrics = self.evaluate()

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_acc"].append(eval_metrics["test_acc"])
            self.history["polarization_gini"].append(eval_metrics["gini"])
            self.history["phase_consistency"].append(eval_metrics["phase_consistency"])
            self.history["gradient_norms"].append(grads)

            print(
                f"Epoch {epoch+1}: Train={train_acc:.3f}, Test={eval_metrics['test_acc']:.3f}, "
                f"Gini={eval_metrics['gini']:.2f}, Phase={eval_metrics['phase_consistency']:.2f}"
            )

        return self.save_results()

    def save_results(self):
        results = {
            "final_acc": self.history["test_acc"][-1],
            "final_gini": self.history["polarization_gini"][-1],
            "final_phase": self.history["phase_consistency"][-1],
            "history": self.history,
        }

        with open(self.output_dir / "universal_metrics.json", "w") as f:
            # Convert numpy types
            def default(o):
                if isinstance(o, np.float32):
                    return float(o)
                return str(o)

            json.dump(results, f, indent=2, default=default)

        return results
