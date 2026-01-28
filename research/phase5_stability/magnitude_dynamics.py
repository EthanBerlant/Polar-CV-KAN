"""Track magnitude statistics during training."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import magnitude_variance, save_figure, save_results_json


def create_synthetic_data(n_samples=4000, seq_len=16, d_input=8, snr=1.5, seed=42):
    torch.manual_seed(seed)
    n_signal = seq_len // 4
    X, y = [], []
    for i in range(n_samples):
        label = i % 2
        signal = torch.randn(n_signal, d_input) + (snr if label == 0 else -snr)
        noise = torch.randn(seq_len - n_signal, d_input)
        tokens = torch.cat([signal, noise], dim=0)[torch.randperm(seq_len)]
        X.append(tokens)
        y.append(label)
    return torch.stack(X), torch.tensor(y)


class TrackedComplexModel(nn.Module):
    def __init__(self, d_input=8, d_complex=32, n_layers=4):
        super().__init__()
        self.d_complex = d_complex
        self.embed = nn.Linear(d_input, d_complex * 2)
        self.blocks = nn.ModuleList([TrackedBlock(d_complex) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_complex, 2)
        self.layer_stats = []

    def forward(self, x):
        h = self.embed(x)
        Z = torch.complex(h[..., : self.d_complex], h[..., self.d_complex :])

        self.layer_stats = []
        for i, block in enumerate(self.blocks):
            Z = block(Z)
            mags = torch.abs(Z)
            self.layer_stats.append(
                {
                    "layer": i,
                    "mean": mags.mean().item(),
                    "max": mags.max().item(),
                    "var": magnitude_variance(Z, dim=1).mean().item(),
                }
            )

        return self.classifier(torch.abs(Z).mean(dim=1))


class TrackedBlock(nn.Module):
    def __init__(self, d_complex):
        super().__init__()
        self.mag_transform = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, Z):
        A = Z.mean(dim=1, keepdim=True)
        mag = torch.abs(A) + 1e-8
        phase = torch.angle(A)
        log_mag = torch.log(mag).unsqueeze(-1)
        b, _, d, _ = log_mag.shape
        log_mag_out = self.mag_transform(log_mag.view(-1, 1)).view(b, 1, d)
        A_out = torch.exp(log_mag_out) * torch.exp(1j * phase)
        return Z + A_out


def train_with_tracking(model, loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_histories = []

    for epoch in range(epochs):
        model.train()
        layer_accum = defaultdict(list)

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            for stat in model.layer_stats:
                layer_accum[stat["layer"]].append(stat)

        epoch_summary = {}
        for layer_idx, stats in layer_accum.items():
            epoch_summary[f"layer_{layer_idx}"] = {
                "mean_mag": float(np.mean([s["mean"] for s in stats])),
                "max_mag": float(np.max([s["max"] for s in stats])),
                "mean_var": float(np.mean([s["var"] for s in stats])),
            }
        epoch_histories.append({"epoch": epoch + 1, "layers": epoch_summary})

        print(f"Epoch {epoch+1}:")
        for name, data in epoch_summary.items():
            print(f"  {name}: mean={data['mean_mag']:.4f}, var={data['mean_var']:.4f}")

    return epoch_histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nCreating synthetic data...")
    X, y = create_synthetic_data()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    print("\nTraining with magnitude tracking...")
    model = TrackedComplexModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = train_with_tracking(model, loader, args.epochs, 1e-3, device)

    # Plot
    n_layers = 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for layer_idx in range(n_layers):
        means = [r["layers"][f"layer_{layer_idx}"]["mean_mag"] for r in results]
        vars = [r["layers"][f"layer_{layer_idx}"]["mean_var"] for r in results]
        epochs = [r["epoch"] for r in results]
        axes[0].plot(epochs, means, "o-", label=f"Layer {layer_idx}")
        axes[1].plot(epochs, vars, "o-", label=f"Layer {layer_idx}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Magnitude")
    axes[0].set_title("Magnitude Over Training")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Magnitude Variance")
    axes[1].set_title("Polarization Over Training")
    axes[1].legend()

    save_figure(fig, output_dir / "magnitude_dynamics.png")
    save_results_json(results, output_dir / "magnitude_results.json")

    print(f"\nResults saved to {output_dir}")

    # Check for issues
    final = results[-1]["layers"]
    for name, data in final.items():
        if data["max_mag"] > 100:
            print(f"[WARN] {name}: High magnitude ({data['max_mag']:.1f})")
        if data["mean_mag"] < 0.01:
            print(f"[WARN] {name}: Low magnitude ({data['mean_mag']:.4f})")


if __name__ == "__main__":
    main()
