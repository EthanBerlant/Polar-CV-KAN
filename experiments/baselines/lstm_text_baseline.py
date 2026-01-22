"""
LSTM Baseline for Text Classification.

Bidirectional LSTM for sequence classification.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime

from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.text import load_sst2, pad_collate

from .base_trainer import (
    BaselineTrainer,
    add_baseline_args,
    count_parameters,
    create_optimizer,
    save_results,
    set_seed,
)


class TextLSTM(nn.Module):
    """Bidirectional LSTM for text classification."""

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pack padded sequences if mask provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(x)

        # Use mean pooling over all timesteps
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

        output = self.dropout(output)
        return self.fc(output)


class TextTrainer(BaselineTrainer):
    """Custom trainer for text classification with dict batches."""

    def _forward_batch(self, batch):
        """Handle dict batch format for text data."""
        indices = batch["indices"].to(self.device)
        labels = batch["label"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)

        outputs = self.model(indices, mask)
        loss = self.loss_fn(outputs, labels)
        metric = self.metric_fn(outputs, labels)

        return loss, metric


def create_dataloaders(args):
    """Create SST-2 dataloaders."""
    train_dataset, val_dataset, vocab = load_sst2()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate
    )
    # Use val as test for SST-2
    test_loader = val_loader

    return train_loader, val_loader, test_loader, {"vocab_size": len(vocab), "n_classes": 2}


def create_model(args, metadata):
    """Create TextLSTM model."""
    return TextLSTM(
        vocab_size=metadata["vocab_size"],
        embed_dim=args.d_complex,
        hidden_dim=args.d_complex,
        num_layers=args.n_layers,
        num_classes=metadata["n_classes"],
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


def run_text_baseline():
    """Run the text LSTM baseline with custom trainer."""
    parser = argparse.ArgumentParser(description="TextLSTM Baseline for nlp")
    parser = add_baseline_args(parser)
    args = parser.parse_args()

    # Apply defaults
    default_args = {
        "d_complex": 128,
        "n_layers": 2,
        "epochs": 20,
        "output_dir": "outputs/baselines/nlp",
    }
    for k, v in default_args.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup output directory
    run_name = args.run_name or f"nlp_TextLSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create data
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(args)

    # Create model
    print("Creating model...")
    model = create_model(args, metadata).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create custom trainer for text
    trainer = TextTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
    )

    # Train
    print("Starting training...")
    history, train_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True))
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    # Save results
    save_results(
        output_dir=output_dir,
        model_name="TextLSTM",
        args=args,
        n_params=n_params,
        history=history,
        test_metrics=test_metrics,
        train_time=train_time,
    )

    print(f"\nTraining complete! Total time: {train_time/60:.1f} minutes")

    return test_metrics


if __name__ == "__main__":
    run_text_baseline()
