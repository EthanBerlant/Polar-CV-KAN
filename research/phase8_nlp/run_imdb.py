import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.phase8_nlp.real_polar_nlp import RealPolarNLP
from research.universal.experiment import UniversalTrainer
from src.data.nlp_loader import NLPDataLoader
from src.models.cv_kan_nlp import CVKANNLP


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Metrics ---
def compute_metrics(model, device, loader):
    model.eval()
    try:
        # Get one batch
        batch = next(iter(loader))
        indices = batch["indices"].to(device)
        mask = batch["mask"].to(device)

        with torch.no_grad():
            out = model(indices, mask=mask)

        metrics = {}
        if isinstance(out, dict) and "features" in out:
            # Gini of pooled features
            features = out["features"]
            metrics["feature_gini"] = compute_gini_coefficient(features).item()

            # Phase coherence (if pooled is present)
            if "pooled" in out:
                pooled = out["pooled"]
                phase = torch.angle(pooled)
                labels = batch["label"].to(device)
                coherences = []
                for c in range(2):  # Binary sentiment
                    m = labels == c
                    if m.sum() > 0:
                        p_c = phase[m]
                        R = torch.abs(torch.exp(1j * p_c).mean(dim=0))
                        coherences.append(R.mean().item())
                if coherences:
                    metrics["phase_coherence"] = np.mean(coherences)

        return metrics
    except Exception:
        return {}


# --- Custom Models ---
class RealLSTM(nn.Module):
    """Standard LSTM Baseline."""

    def __init__(self, vocab_size, embed_dim=128, content_dim=64, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, content_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim), nn.ReLU(), nn.Linear(content_dim, n_classes)
        )

    def forward(self, x, mask=None, **kwargs):
        # emb: (B, L, E)
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        # h_n shape: (2, B, hidden)
        cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*hidden)
        logits = self.classifier(cat)
        return {"logits": logits, "features": cat}  # Return 'cat' as features for Gini


# --- Custom Models ---
class CVKANWeightedNLP(CVKANNLP):
    """CV-KAN with Magnitude Weighted Pooling."""

    def _pool(self, z, mask=None):
        mag = torch.abs(z) + 1e-8
        if mask is not None:
            mag = mag * mask.unsqueeze(-1).float()
        weights = mag / mag.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (z * weights).sum(dim=1)
        return pooled

    def forward(self, x, mask=None, **kwargs):
        # Override to ensure we return pooled for phase consistency
        z = self._embed(x)
        z = self._apply_layers(z, mask=mask)
        pooled = self._pool(z, mask=mask)
        features = self._extract_features(pooled)
        logits = self.classifier(features)
        return {"logits": logits, "pooled": pooled, "features": features}


# --- Runner ---
# Standard trainers removed in favor of UniversalTrainer
# --- Runner ---


# --- Runner ---
def run_imdb_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    print("Loading IMDB...")
    loader_utils = NLPDataLoader(
        dataset_name="imdb", batch_size=32, max_seq_len=256, subset_size=args.n_samples
    )
    train_loader, test_loader, n_classes, vocab_size = loader_utils.get_dataloaders()

    # 2. Models
    models = {}

    # CV-KAN
    print("Initializing CV-KAN NLP (Weighted)...")
    cvkan = CVKANWeightedNLP(
        vocab_size=vocab_size,
        d_complex=32,
        n_layers=3,
        n_classes=2,
        kan_hidden=32,
        max_seq_len=256,
        pooling="weighted",
    ).to(device)
    models["CVKAN-Weighted"] = cvkan

    # Hierarchical CV-KAN
    print("Initializing Hierarchical CV-KAN NLP...")
    cvkan_hierarchical = CVKANWeightedNLP(
        vocab_size=vocab_size,
        d_complex=32,
        n_layers=1,
        n_classes=2,
        kan_hidden=32,
        max_seq_len=256,
        pooling="weighted",
        block_type="hierarchical",
    ).to(device)
    models["CVKAN-Hierarchical"] = cvkan_hierarchical

    # Real Polar NLP
    print("Initializing Real Polar NLP (Weighted)...")
    real_polar = RealPolarNLP(
        vocab_size=vocab_size,
        d_polar=32,
        n_layers=3,
        n_classes=2,
        kan_hidden=32,
        pooling="weighted",
        max_seq_len=256,
    ).to(device)
    # Ensure RealPolarNLP returns dict for UniversalTrainer
    # RealPolarNLP already returns dict if we look at its implementation?
    # No, it returns pooled_cart, r_pooled, theta_pooled if not dictionary.
    # Let's check RealPolarNLP implementation.
    models["RealPolar-Weighted"] = real_polar

    # LSTM
    print("Initializing Real LSTM...")
    lstm = RealLSTM(vocab_size=vocab_size, embed_dim=64, content_dim=64).to(device)
    models["Real-LSTM"] = lstm

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

    # 4. Save & Summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("NLP VALIDATION (IMDB) SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} | {'Acc':<6} | {'Gini':<6} | {'Phase':<6}")
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<20} | {res['final_acc']:.3f} | {res['final_gini']:.3f} | {res['final_phase']:.3f}"
        )
    print("-" * 60)

    with open(output_dir / "imdb_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="research/outputs/phase8")
    args = parser.parse_args()

    run_imdb_experiment(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Subset size for faster research iteration"
    )
    parser.add_argument("--output_dir", type=str, default="research/outputs/phase8")
    args = parser.parse_args()

    run_imdb_experiment(args)
