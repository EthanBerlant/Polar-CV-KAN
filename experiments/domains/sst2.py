"""
SST-2 sentiment analysis domain.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.text import load_sst2, pad_collate
from src.models import CVKAN
from src.trainer import BaseTrainer

DEFAULTS = {
    "batch_size": 32,
    "d_complex": 64,
    "d_embed": 64,
    "n_layers": 2,
    "epochs": 10,
    "norm_type": "none",
    "metric_name": "accuracy",
    "metric_mode": "max",
}


def add_args(parser):
    """Add SST2-specific arguments."""
    parser.add_argument("--d_embed", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--norm_type", type=str, default="none", choices=["layer", "rms", "none"])
    parser.add_argument(
        "--block_type", type=str, default="polarizing", choices=["polarizing", "attention"]
    )
    parser.add_argument(
        "--head_approach",
        type=str,
        default="emergent",
        choices=["emergent", "phase_offset", "factored"],
    )
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.0)
    return parser


class TextClassifier(nn.Module):
    """Wrapper that adds embedding layer to CVKAN."""

    def __init__(self, vocab_size: int, embed_dim: int, cvkan_args: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)

    def forward(self, indices, mask=None, return_intermediates=False):
        x = self.embedding(indices)
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


class SST2Trainer(BaseTrainer):
    """Trainer for SST-2 sentiment analysis."""

    def train_step(self, batch):
        indices = batch["indices"].to(self.device)
        mask = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.model(indices, mask=mask, return_intermediates=True)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        # Regularization
        if hasattr(self.args, "diversity_weight") and self.args.diversity_weight > 0:
            from src.losses import diversity_loss

            Z_final = outputs.get(
                "Z", outputs["intermediates"][-1] if "intermediates" in outputs else None
            )
            if Z_final is not None:
                loss = loss + self.args.diversity_weight * diversity_loss(Z_final)

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item() * 100

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        indices = batch["indices"].to(self.device)
        mask = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.model(indices, mask=mask)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item() * 100

        return {"loss": loss, "accuracy": accuracy}


def create_model(args):
    """Create text classification model."""
    return TextClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.d_embed,
        cvkan_args={
            "d_complex": args.d_complex,
            "n_layers": args.n_layers,
            "n_classes": 2,
            "kan_hidden": args.kan_hidden,
            "head_approach": args.head_approach,
            "n_heads": getattr(args, "n_heads", 4),
            "pooling": args.pooling,
            "input_type": "real",
            "norm_type": args.norm_type,
            "block_type": args.block_type,
        },
    )


def create_dataloaders(args):
    """Create SST-2 dataloaders."""
    train_ds, val_ds, vocab = load_sst2(max_len=64)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate
    )

    return train_loader, val_loader, val_loader, {"vocab_size": len(vocab)}
