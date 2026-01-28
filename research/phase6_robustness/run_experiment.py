import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn, optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.phase6_robustness.rotated_shapes import get_rotated_shapes_loaders
from research.phase7_real_world.real_polar_baseline import RealPolarImageClassifier
from research.universal.experiment import UniversalTrainer
from src.models.cv_kan_image import CVKANImageClassifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RealCNN(nn.Module):
    """Standard CNN baseline with similar parameter count to small CV-KAN."""

    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.flatten(1)
        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}


# Standard trainers removed in favor of UniversalTrainer
# --- Runner ---


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    print("Generating Rotated Shapes data...")
    train_loader, test_loader = get_rotated_shapes_loaders(
        train_size=args.n_samples, test_size=1000, batch_size=64
    )

    # 2. Models
    models = {}

    # Polar CV-KAN
    print("Initializing Polar CV-KAN...")
    cvkan = CVKANImageClassifier(
        img_size=28,
        patch_size=7,
        in_channels=1,
        d_complex=32,
        n_layers=3,
        n_classes=2,
        kan_hidden=32,
        embedding_type="linear",
    ).to(device)
    models["Polar-CV-KAN"] = cvkan

    # Real Polar
    print("Initializing Real Polar...")
    real_polar = RealPolarImageClassifier(
        img_size=28, patch_size=7, in_channels=1, d_polar=32, n_layers=3, n_classes=2, kan_hidden=32
    ).to(device)
    models["Real-Polar"] = real_polar

    # Real CNN
    print("Initializing Real CNN...")
    cnn = RealCNN(n_classes=2).to(device)
    models["Real-CNN"] = cnn

    # 3. Training
    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        trainer = UniversalTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            output_dir=Path(args.output_dir) / name,
        )

        res = trainer.train(args.epochs)
        results[name] = res

    # 4. Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("ROBUSTNESS SUMMARY")
    print("=" * 50)
    print(f"{'Model':<15} | {'Acc':<10} | {'Gini':<10} | {'Phase':<10}")
    print("-" * 50)
    for name, res in results.items():
        print(
            f"{name:<15} | {res['final_acc']:.4f}     | {res['final_gini']:.4f}     | {res['final_phase']:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="research/outputs/phase6")
    args = parser.parse_args()

    run_experiment(args)
