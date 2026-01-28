"""Controlled synthetic experiments for polarization dynamics.

Train models on synthetic tasks specifically designed to test polarization:
- Task requires distinguishing "signal" tokens from "noise" tokens
- Measure whether polarization correlates with performance
- Ablate polarization mechanism and compare
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import (
    magnitude_variance,
    plot_comparison_bars,
    plot_training_curves,
    save_results_json,
)


class MinimalPolarizingBlock(nn.Module):
    """Minimal implementation for controlled experiments."""

    def __init__(self, d_complex: int, amplification: float = 1.5):
        super().__init__()
        self.d_complex = d_complex
        self.amplification = amplification

        # Simple learnable magnitude transform
        self.mag_scale = nn.Parameter(torch.tensor(amplification))
        self.mag_bias = nn.Parameter(torch.tensor(0.0))

        # Phase transform (optional)
        self.phase_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Aggregate
        A = Z.mean(dim=1, keepdim=True)

        # Decompose
        mag = torch.abs(A) + 1e-8
        phase = torch.angle(A)
        log_mag = torch.log(mag)

        # Transform
        log_mag_out = self.mag_scale * log_mag + self.mag_bias
        phase_out = self.phase_scale * phase

        # Recompose
        mag_out = torch.exp(log_mag_out)
        A_out = mag_out * torch.exp(1j * phase_out)

        # Broadcast
        return Z + A_out


class IdentityBlock(nn.Module):
    """No-op block for ablation."""

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return Z


class SyntheticPolarizationModel(nn.Module):
    """Model for synthetic polarization experiments."""

    def __init__(
        self,
        d_input: int,
        d_complex: int,
        n_layers: int,
        n_classes: int,
        use_polarization: bool = True,
    ):
        super().__init__()

        # Input projection (real to complex)
        self.embed = nn.Linear(d_input, d_complex * 2)
        self.d_complex = d_complex

        # Stack of blocks
        if use_polarization:
            self.blocks = nn.ModuleList(
                [MinimalPolarizingBlock(d_complex) for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.ModuleList([IdentityBlock() for _ in range(n_layers)])

        # Classifier
        self.classifier = nn.Linear(d_complex, n_classes)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        # x: (batch, seq_len, d_input)
        batch, seq_len, _ = x.shape

        # Embed to complex
        h = self.embed(x)
        Z = torch.complex(h[..., : self.d_complex], h[..., self.d_complex :])

        intermediates = [Z.clone()] if return_intermediates else None

        # Apply blocks
        for block in self.blocks:
            Z = block(Z)
            if return_intermediates:
                intermediates.append(Z.clone())

        # Pool (mean magnitude)
        pooled = torch.abs(Z).mean(dim=1)  # (batch, d_complex)

        # Classify
        logits = self.classifier(pooled)

        if return_intermediates:
            return logits, intermediates
        return logits


def create_polarization_task(
    n_samples: int = 10000,
    seq_len: int = 16,
    d_input: int = 8,
    signal_ratio: float = 0.25,
    snr: float = 3.0,
    seed: int = 42,
) -> tuple:
    """Create a task where signal tokens must be distinguished from noise.

    - Class 0: Signal has positive features
    - Class 1: Signal has negative features
    - Noise tokens have random features in both classes

    Success requires identifying and amplifying signal tokens.
    """
    torch.manual_seed(seed)

    n_signal = int(seq_len * signal_ratio)
    n_noise = seq_len - n_signal

    X = []
    y = []

    for i in range(n_samples):
        label = i % 2

        # Signal tokens: consistent with label
        if label == 0:
            signal = torch.randn(n_signal, d_input) + snr
        else:
            signal = torch.randn(n_signal, d_input) - snr

        # Noise tokens: random
        noise = torch.randn(n_noise, d_input)

        # Combine (signal first, then noise) and shuffle
        tokens = torch.cat([signal, noise], dim=0)
        perm = torch.randperm(seq_len)
        tokens = tokens[perm]

        X.append(tokens)
        y.append(label)

    X = torch.stack(X)
    y = torch.tensor(y)

    return X, y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train model and track metrics."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "polarization": [],  # Track P at each epoch
    }

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validate
        model.eval()
        correct = 0
        total = 0
        all_intermediates = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, intermediates = model(X_batch, return_intermediates=True)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

                # Track polarization (last layer)
                all_intermediates.append(intermediates[-1])

        val_acc = correct / total

        # Compute polarization
        Z_final = torch.cat(all_intermediates, dim=0)
        polarization = magnitude_variance(Z_final, dim=1).mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["polarization"].append(polarization)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_acc={val_acc:.4f}, P={polarization:.4f}"
            )

    return history


def run_experiments(args, output_dir: Path):
    """Run comparison experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dataset
    print("Creating synthetic dataset...")
    X_train, y_train = create_polarization_task(
        n_samples=8000,
        seq_len=args.seq_len,
        d_input=args.d_input,
        signal_ratio=args.signal_ratio,
        snr=args.snr,
    )
    X_val, y_val = create_polarization_task(
        n_samples=2000,
        seq_len=args.seq_len,
        d_input=args.d_input,
        signal_ratio=args.signal_ratio,
        snr=args.snr,
        seed=123,
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    results = {}
    histories = {}

    # Experiment 1: With polarization
    print("\n" + "=" * 50)
    print("Training WITH polarization...")
    print("=" * 50)

    model_polar = SyntheticPolarizationModel(
        d_input=args.d_input,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=2,
        use_polarization=True,
    )

    history_polar = train_model(model_polar, train_loader, val_loader, args.epochs, device=device)
    histories["with_polarization"] = history_polar
    results["with_polarization"] = {
        "final_val_acc": history_polar["val_acc"][-1],
        "final_polarization": history_polar["polarization"][-1],
        "max_val_acc": max(history_polar["val_acc"]),
    }

    # Experiment 2: Without polarization
    print("\n" + "=" * 50)
    print("Training WITHOUT polarization...")
    print("=" * 50)

    model_no_polar = SyntheticPolarizationModel(
        d_input=args.d_input,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=2,
        use_polarization=False,
    )

    history_no_polar = train_model(
        model_no_polar, train_loader, val_loader, args.epochs, device=device
    )
    histories["without_polarization"] = history_no_polar
    results["without_polarization"] = {
        "final_val_acc": history_no_polar["val_acc"][-1],
        "final_polarization": history_no_polar["polarization"][-1],
        "max_val_acc": max(history_no_polar["val_acc"]),
    }

    # Experiment 3: Vary number of layers
    print("\n" + "=" * 50)
    print("Testing layer depth...")
    print("=" * 50)

    depth_results = {}
    for n_layers in [1, 2, 4, 8]:
        print(f"  n_layers = {n_layers}")
        model = SyntheticPolarizationModel(
            d_input=args.d_input,
            d_complex=args.d_complex,
            n_layers=n_layers,
            n_classes=2,
            use_polarization=True,
        )
        history = train_model(model, train_loader, val_loader, args.epochs, device=device)
        depth_results[f"{n_layers}_layers"] = {
            "val_acc": max(history["val_acc"]),
            "final_polarization": history["polarization"][-1],
        }

    results["depth_study"] = depth_results

    # Visualize
    print("\n" + "=" * 50)
    print("Generating visualizations...")
    print("=" * 50)

    # Training curves
    plot_training_curves(
        histories,
        "val_acc",
        output_dir / "synthetic_val_acc.png",
        title="Validation Accuracy: Polarization Effect",
    )
    plot_training_curves(
        histories,
        "polarization",
        output_dir / "synthetic_polarization.png",
        title="Polarization (Magnitude Variance) Over Training",
    )

    # Comparison bars
    comparison = {
        "With Polarization": results["with_polarization"]["max_val_acc"],
        "Without Polarization": results["without_polarization"]["max_val_acc"],
    }
    plot_comparison_bars(
        comparison,
        output_dir / "synthetic_comparison.png",
        title="Polarization Ablation",
        ylabel="Max Validation Accuracy",
    )

    # Depth study
    depth_accs = {k: v["val_acc"] for k, v in depth_results.items()}
    plot_comparison_bars(
        depth_accs,
        output_dir / "synthetic_depth.png",
        title="Effect of Depth",
        ylabel="Max Validation Accuracy",
    )

    # Save results
    save_results_json(results, output_dir / "synthetic_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SYNTHETIC EXPERIMENT RESULTS")
    print("=" * 60)

    acc_with = results["with_polarization"]["max_val_acc"]
    acc_without = results["without_polarization"]["max_val_acc"]
    delta = acc_with - acc_without

    print(f"\nWith polarization:    {acc_with:.4f}")
    print(f"Without polarization: {acc_without:.4f}")
    print(f"Delta:                {delta:+.4f}")

    if delta > 0.02:
        print("\n✓ POLARIZATION HELPS: Significant improvement with polarization")
    elif delta < -0.02:
        print("\n✗ POLARIZATION HURTS: Better without polarization")
    else:
        print("\n~ INCONCLUSIVE: No significant difference")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_complex", type=int, default=32)
    parser.add_argument("--d_input", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--signal_ratio", type=float, default=0.25)
    parser.add_argument(
        "--snr", type=float, default=1.0, help="Signal-to-noise ratio. Lower = harder task"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_experiments(args, output_dir)


if __name__ == "__main__":
    main()
