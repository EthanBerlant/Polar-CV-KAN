"""
CNN Baseline for Text Classification.

TextCNN with multiple filter sizes for n-gram feature extraction.
Reference: "Convolutional Neural Networks for Sentence Classification" (Kim, 2014)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.text import load_sst2, pad_collate

from .base_trainer import (
    BaselineTrainer,
    add_baseline_args,
    count_parameters,
    create_optimizer,
    save_results,
    set_seed,
)


class TextCNN(nn.Module):
    """TextCNN for text classification."""

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_classes=2,
        filter_sizes=None,
        num_filters=100,
        dropout=0.5,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs // 2)
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            pooled.append(h)

        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)

        return self.fc(cat)


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
    test_loader = val_loader

    return train_loader, val_loader, test_loader, {"vocab_size": len(vocab), "n_classes": 2}


def create_model(args, metadata):
    """Create TextCNN model."""
    return TextCNN(
        vocab_size=metadata["vocab_size"],
        embed_dim=args.d_complex,
        num_classes=metadata["n_classes"],
        num_filters=args.d_complex,
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


def run_text_cnn_baseline():
    """Run the TextCNN baseline with custom trainer."""
    parser = argparse.ArgumentParser(description="TextCNN Baseline for nlp")
    parser = add_baseline_args(parser)
    args = parser.parse_args()

    # Apply defaults
    default_args = {
        "d_complex": 128,
        "n_layers": 1,
        "epochs": 20,
        "output_dir": "outputs/baselines/nlp",
    }
    for k, v in default_args.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_name = args.run_name or f"nlp_TextCNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(args)

    print("Creating model...")
    model = create_model(args, metadata).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = create_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    print("Starting training...")
    history, train_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True))
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    save_results(
        output_dir=output_dir,
        model_name="TextCNN",
        args=args,
        n_params=n_params,
        history=history,
        test_metrics=test_metrics,
        train_time=train_time,
    )

    print(f"\nTraining complete! Total time: {train_time/60:.1f} minutes")

    return test_metrics


if __name__ == "__main__":
    run_text_cnn_baseline()
