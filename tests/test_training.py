import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import SignalNoiseDataset
from src.models import CVKANTokenClassifier
from src.trainer import BaseTrainer


class SimpleTrainer(BaseTrainer):
    def train_step(self, batch):
        return {"loss": torch.tensor(0.0, requires_grad=True)}  # dummy

    def validate_step(self, batch):
        return {"loss": 0.0}


def test_training_convergence_synthetic():
    """
    Verify that the model can learn a simple signal/noise task.
    This ensures gradients flow correctly and the architecture is capable of learning.
    """
    torch.manual_seed(42)

    # 1. Create Data
    # Easy task: signal tokens have high magnitude, noise has low
    dataset = SignalNoiseDataset(
        n_samples=64, n_tokens=16, k_signal=4, d_complex=16, signal_mag_mean=2.0, noise_mag_mean=0.5
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Create Model
    # Use token classifier since we have per-token labels
    model = CVKANTokenClassifier(
        d_input=16,  # Match data d_complex
        d_complex=32,
        n_layers=2,
        n_classes=2,
        input_type="complex",  # Data is already complex
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Train Loop
    model.train()
    initial_loss = None
    final_loss = None

    losses = []

    for _ in range(10):
        epoch_loss = 0
        for batch in dataloader:
            # batch is dict
            x = batch["sequence"]  # (B, L, D)
            y = batch["token_labels"]  # (B, L)

            optimizer.zero_grad()
            output = model(x)
            logits = output["token_logits"]  # (B, L, C)

            # Flatten for CE
            # logits: (B*L, C), y: (B*L)
            loss = criterion(logits.reshape(-1, 2), y.reshape(-1).long())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

    initial_loss = losses[0]
    final_loss = losses[-1]

    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

    # Check convergence
    assert final_loss < initial_loss * 0.5, "Model failed to converge on simple synthetic task"


if __name__ == "__main__":
    test_training_convergence_synthetic()
