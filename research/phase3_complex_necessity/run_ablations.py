"""Run ablation experiments comparing complex vs real representations.

Uses synthetic data - no external dependencies required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import plot_comparison_bars, plot_training_curves, save_results_json


def create_synthetic_data(n_samples=8000, seq_len=16, d_input=8, snr=1.5, seed=42):
    """Create synthetic classification data."""
    torch.manual_seed(seed)
    n_signal = seq_len // 4
    n_noise = seq_len - n_signal

    X, y = [], []
    for i in range(n_samples):
        label = i % 2
        if label == 0:
            signal = torch.randn(n_signal, d_input) + snr
        else:
            signal = torch.randn(n_signal, d_input) - snr
        noise = torch.randn(n_noise, d_input)
        tokens = torch.cat([signal, noise], dim=0)
        perm = torch.randperm(seq_len)
        tokens = tokens[perm]
        X.append(tokens)
        y.append(label)

    return torch.stack(X), torch.tensor(y)


class ComplexPolarModel(nn.Module):
    """Simple complex model for comparison."""

    def __init__(self, d_input, d_complex, n_layers, n_classes, kan_hidden=32):
        super().__init__()
        self.d_complex = d_complex
        self.embed = nn.Linear(d_input, d_complex * 2)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(ComplexPolarBlock(d_complex, kan_hidden))

        self.classifier = nn.Sequential(
            nn.Linear(d_complex, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, x):
        h = self.embed(x)
        Z = torch.complex(h[..., : self.d_complex], h[..., self.d_complex :])
        for block in self.blocks:
            Z = block(Z)
        pooled = torch.abs(Z).mean(dim=1)
        return self.classifier(pooled)


class ComplexPolarBlock(nn.Module):
    """Single complex polarizing block."""

    def __init__(self, d_complex, kan_hidden=32):
        super().__init__()
        self.mag_transform = nn.Sequential(
            nn.Linear(1, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 1),
        )
        self.phase_transform = nn.Sequential(
            nn.Linear(2, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 2),
        )

    def forward(self, Z):
        A = Z.mean(dim=1, keepdim=True)
        mag = torch.abs(A) + 1e-8
        phase = torch.angle(A)

        log_mag = torch.log(mag).unsqueeze(-1)
        batch, _, d, _ = log_mag.shape
        log_mag_flat = log_mag.view(-1, 1)
        log_mag_out = self.mag_transform(log_mag_flat).view(batch, 1, d)
        mag_out = torch.exp(log_mag_out)

        sin_p, cos_p = torch.sin(phase), torch.cos(phase)
        phase_in = torch.stack([sin_p, cos_p], dim=-1).view(-1, 2)
        phase_out = self.phase_transform(phase_in).view(batch, 1, d, 2)
        norm = torch.sqrt(phase_out[..., 0] ** 2 + phase_out[..., 1] ** 2 + 1e-8)
        theta_out = torch.atan2(phase_out[..., 0] / norm, phase_out[..., 1] / norm)

        A_out = mag_out * torch.exp(1j * theta_out)
        return Z + A_out


class RealPolarModelSimple(nn.Module):
    """Real-valued model with polar structure."""

    def __init__(self, d_input, d_polar, n_layers, n_classes, kan_hidden=32):
        super().__init__()
        self.d_polar = d_polar
        self.embed = nn.Linear(d_input, d_polar * 2)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(RealPolarBlock(d_polar, kan_hidden))

        self.classifier = nn.Sequential(
            nn.Linear(d_polar, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, x):
        h = self.embed(x)
        r = torch.abs(h[..., : self.d_polar]) + 0.1
        theta = h[..., self.d_polar :] * np.pi
        for block in self.blocks:
            r, theta = block(r, theta)
        pooled = r.mean(dim=1)
        return self.classifier(pooled)


class RealPolarBlock(nn.Module):
    """Real-valued polar block."""

    def __init__(self, d_polar, kan_hidden=32):
        super().__init__()
        self.mag_transform = nn.Sequential(
            nn.Linear(1, kan_hidden), nn.GELU(), nn.Linear(kan_hidden, 1)
        )
        self.phase_transform = nn.Sequential(
            nn.Linear(2, kan_hidden), nn.GELU(), nn.Linear(kan_hidden, 2)
        )

    def forward(self, r, theta):
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        x_mean, y_mean = x.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)
        r_agg = torch.sqrt(x_mean**2 + y_mean**2 + 1e-8)
        theta_agg = torch.atan2(y_mean, x_mean)

        batch, _, d = r_agg.shape
        log_r = torch.log(r_agg + 1e-8).view(-1, 1)
        log_r_out = self.mag_transform(log_r).view(batch, 1, d)
        r_trans = torch.exp(log_r_out)

        phase_in = torch.stack([torch.sin(theta_agg), torch.cos(theta_agg)], dim=-1).view(-1, 2)
        phase_out = self.phase_transform(phase_in).view(batch, 1, d, 2)
        norm = torch.sqrt(phase_out[..., 0] ** 2 + phase_out[..., 1] ** 2 + 1e-8)
        theta_trans = torch.atan2(phase_out[..., 0] / norm, phase_out[..., 1] / norm)

        x_trans = r_trans * torch.cos(theta_trans)
        y_trans = r_trans * torch.sin(theta_trans)
        x_out, y_out = x + x_trans, y + y_trans

        r_out = torch.sqrt(x_out**2 + y_out**2 + 1e-8)
        theta_out = torch.atan2(y_out, x_out)
        return r_out, theta_out


class RealCartesianModelSimple(nn.Module):
    """Real-valued model without polar structure."""

    def __init__(self, d_input, d_model, n_layers, n_classes, kan_hidden=32):
        super().__init__()
        self.embed = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, kan_hidden), nn.GELU(), nn.Linear(kan_hidden, d_model)
                )
                for _ in range(n_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, kan_hidden), nn.GELU(), nn.Linear(kan_hidden, n_classes)
        )

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            agg = h.mean(dim=1, keepdim=True)
            h = h + block(agg)
        return self.classifier(h.mean(dim=1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_correct / val_total)

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch+1}: train_acc={correct/total:.4f}, val_acc={val_correct/val_total:.4f}"
            )

    return history


def run_ablations(args, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nCreating synthetic data...")
    X_train, y_train = create_synthetic_data(8000, seed=42)
    X_val, y_val = create_synthetic_data(2000, seed=123)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    d_input, d_complex, n_layers = 8, 32, 4

    results = {}
    histories = {}

    # Complex Polar
    print("\n" + "=" * 50)
    print("Training COMPLEX-POLAR")
    print("=" * 50)
    model = ComplexPolarModel(d_input, d_complex, n_layers, 2)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    results["complex_polar"] = {"max_val_acc": max(history["val_acc"]), "n_params": n_params}
    histories["complex_polar"] = history

    # Real Polar
    print("\n" + "=" * 50)
    print("Training REAL-POLAR")
    print("=" * 50)
    model = RealPolarModelSimple(d_input, d_complex, n_layers, 2)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    results["real_polar"] = {"max_val_acc": max(history["val_acc"]), "n_params": n_params}
    histories["real_polar"] = history

    # Real Cartesian
    print("\n" + "=" * 50)
    print("Training REAL-CARTESIAN")
    print("=" * 50)
    model = RealCartesianModelSimple(d_input, d_complex * 2, n_layers, 2)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    results["real_cartesian"] = {"max_val_acc": max(history["val_acc"]), "n_params": n_params}
    histories["real_cartesian"] = history

    # Visualize
    print("\nGenerating visualizations...")
    comparison = {k: v["max_val_acc"] for k, v in results.items()}
    plot_comparison_bars(
        comparison,
        output_dir / "ablation_comparison.png",
        title="Complex vs Real Ablation",
        ylabel="Max Validation Accuracy",
    )
    plot_training_curves(
        histories,
        "val_acc",
        output_dir / "ablation_training_curves.png",
        title="Validation Accuracy: Complex vs Real",
    )

    save_results_json(results, output_dir / "ablation_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name}: {res['max_val_acc']:.4f} ({res['n_params']:,} params)")

    best = max(results.items(), key=lambda x: x[1]["max_val_acc"])
    print(f"\nBest: {best[0]}")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ablations(args, output_dir)


if __name__ == "__main__":
    main()
