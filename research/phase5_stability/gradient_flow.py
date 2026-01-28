"""Track gradient norms during training on synthetic data."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import save_figure, save_results_json


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


class SimpleComplexModel(nn.Module):
    def __init__(self, d_input=8, d_complex=32, n_layers=4):
        super().__init__()
        self.d_complex = d_complex
        self.embed = nn.Linear(d_input, d_complex * 2)
        self.blocks = nn.ModuleList([ComplexBlock(d_complex) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_complex, 2)

    def forward(self, x):
        h = self.embed(x)
        Z = torch.complex(h[..., : self.d_complex], h[..., self.d_complex :])
        for block in self.blocks:
            Z = block(Z)
        return self.classifier(torch.abs(Z).mean(dim=1))


class ComplexBlock(nn.Module):
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

    batch_norms = []
    epoch_stats = []

    for epoch in range(epochs):
        model.train()
        epoch_norms = []

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()

            total_norm = (
                sum(
                    p.grad.abs().pow(2).sum().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            epoch_norms.append(total_norm)
            batch_norms.append(total_norm)

            optimizer.step()

        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "mean": float(np.mean(epoch_norms)),
                "max": float(np.max(epoch_norms)),
                "min": float(np.min(epoch_norms)),
            }
        )
        print(f"Epoch {epoch+1}: mean_grad_norm={np.mean(epoch_norms):.4f}")

    return {"epoch_stats": epoch_stats, "batch_norms": batch_norms}


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

    print("\nTraining with gradient tracking...")
    model = SimpleComplexModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = train_with_tracking(model, loader, args.epochs, 1e-3, device)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(results["batch_norms"], alpha=0.7)
    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_title("Gradient Norm Per Batch")

    epochs = [s["epoch"] for s in results["epoch_stats"]]
    means = [s["mean"] for s in results["epoch_stats"]]
    axes[1].plot(epochs, means, "o-")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Gradient Norm")
    axes[1].set_title("Gradient Norm Per Epoch")

    save_figure(fig, output_dir / "gradient_flow.png")
    save_results_json(results, output_dir / "gradient_results.json")

    print(f"\nResults saved to {output_dir}")
    final = results["epoch_stats"][-1]["mean"]
    if final > 10:
        print("[WARN] Large gradients - potential instability")
    elif final < 1e-6:
        print("[WARN] Vanishing gradients")
    else:
        print("[OK] Gradients appear healthy")


if __name__ == "__main__":
    main()
