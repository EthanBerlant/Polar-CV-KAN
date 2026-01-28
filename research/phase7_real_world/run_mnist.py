import argparse
import sys
from pathlib import Path

import torch
from torch import nn, optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.phase7_real_world.mnist_loader import create_robustness_loaders
from research.phase7_real_world.real_polar_baseline import RealPolarImageClassifier
from research.universal.experiment import UniversalTrainer
from src.models.cv_kan_image import CVKANImageClassifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- New Components ---
class WeightedAverageHead(nn.Module):
    """Magnitude-weighted average pooling head."""

    def __init__(self, d_complex, n_classes, kan_hidden):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_complex, kan_hidden), nn.GELU(), nn.Linear(kan_hidden, n_classes)
        )

    def forward(self, z, mask=None):
        # z: (B, N, D) complex
        mag = torch.abs(z) + 1e-8
        if mask is not None:
            mag = mag * mask.unsqueeze(-1)
        weights = mag / mag.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (z * weights).sum(dim=1)
        features = torch.abs(pooled)
        return {"logits": self.classifier(features), "pooled": pooled, "features": features}


# --- Baseline Clone ---
class RealCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        features = self.features(x).flatten(1)
        return {"logits": self.classifier(features), "features": features}


# --- Runner ---


def run_mnist_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    print("Loading MNIST data...")
    train_loader, test_loader_up, test_loader_rot = create_robustness_loaders(
        root="./data", batch_size=64, train_subset=args.n_samples
    )

    # 2. Models
    models = {}

    # Polar CV-KAN Sweep
    for n_layers in [3, 4, 5, 6]:
        name_weighted = f"ComplexPolar-L{n_layers}-Weighted"
        print(f"Initializing {name_weighted}...")
        cvkan_weighted = CVKANImageClassifier(
            img_size=28,
            patch_size=7,
            in_channels=1,
            d_complex=32,
            n_layers=n_layers,
            n_classes=10,
            kan_hidden=32,
            embedding_type="linear",
            pooling="mean",
            dropout=0.1,
        ).to(device)
        # Monkey patch head
        cvkan_weighted.head = WeightedAverageHead(32, 10, 32).to(device)
        models[name_weighted] = cvkan_weighted

    # Hierarchical CV-KAN
    print("Initializing Hierarchical CV-KAN...")
    cvkan_hierarchical = CVKANImageClassifier(
        img_size=28,
        patch_size=7,
        in_channels=1,
        d_complex=32,
        n_layers=1,  # Hierarchical carries its own internal layers/levels
        n_classes=10,
        kan_hidden=32,
        embedding_type="linear",
        pooling="mean",
        dropout=0.1,
        block_type="hierarchical",
        hierarchical_sharing="per_level",
    ).to(device)
    cvkan_hierarchical.head = WeightedAverageHead(32, 10, 32).to(device)
    models["ComplexPolar-Hierarchical"] = cvkan_hierarchical

    # Real Polar Sweep
    for n_layers in [3, 4, 6]:  # Reduced set for real polar to save time
        name_real = f"RealPolar-L{n_layers}"
        print(f"Initializing {name_real}...")
        real_polar = RealPolarImageClassifier(
            img_size=28,
            patch_size=7,
            in_channels=1,
            d_polar=32,
            n_layers=n_layers,
            n_classes=10,
            kan_hidden=32,
            # Note: RealPolarImageClassifier implementation should support dropout if we want parity,
            # but user specifically asked for vision (Complex) vs standard.
        ).to(device)
        models[name_real] = real_polar

    # Real CNN
    print("Initializing Real CNN...")
    cnn = RealCNN(n_classes=10).to(device)
    models["RealCNN"] = cnn

    # 3. Training
    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        n_params = count_parameters(model)
        print(f"\nTraining {name} ({n_params:,} params)...")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # We need two trainers?
        # No, one trainer for standard train/test loop.
        # Then manual evaluate for OOD (Rotated).

        trainer = UniversalTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader_up,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            output_dir=Path(args.output_dir) / name,
        )

        res = trainer.train(args.epochs)

        # OOD Evaluation (Rotated)
        print(f"Evaluating {name} on OOD (Rotated)...")
        ood_metrics = trainer.evaluate()  # This will use test_loader_up by default from constructor
        # Wait, I need to swap test_loader to test_loader_rot
        trainer.test_loader = test_loader_rot
        ood_metrics = trainer.evaluate()

        results[name] = {
            "n_params": n_params,
            "final_up": res["final_acc"],
            "final_rot": ood_metrics["test_acc"],
            "final_gini": res["final_gini"],
            "final_phase": res["final_phase"],
            "history": res["history"],
        }

    # 4. Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} | {'Params':<8} | {'Up':<6} | {'Rot':<6} | {'Drop':<6} | {'Gini':<5}")
    print("-" * 80)
    for name, res in results.items():
        drop = res["final_up"] - res["final_rot"]
        print(
            f"{name:<25} | {res['n_params']:<8,} | {res['final_up']:.3f}  | {res['final_rot']:.3f}  | {drop:.3f}  | {res['final_gini']:.2f}"
        )
    print("-" * 80)

    return results

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="research/outputs/phase7_full")
    args = parser.parse_args()
    run_mnist_experiment(args)
