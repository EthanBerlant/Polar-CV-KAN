"""
Synthetic signal/noise classification domain.
"""

import torch.nn.functional as F

from src.data import SignalNoiseDataset
from src.models import CVKAN
from src.models.cv_kan import CVKANTokenClassifier
from src.trainer import BaseTrainer

DEFAULTS = {
    "batch_size": 32,
    "d_complex": 32,
    "n_layers": 2,
    "epochs": 30,
    "n_samples": 10000,
    "n_tokens": 16,
    "k_signal": 4,
    "task": "sequence",
    "metric_name": "accuracy",
    "metric_mode": "max",
}


def add_args(parser):
    """Add synthetic-specific arguments."""
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_tokens", type=int, default=16)
    parser.add_argument("--k_signal", type=int, default=4)
    parser.add_argument("--signal_mag_mean", type=float, default=1.5)
    parser.add_argument("--signal_phase_std", type=float, default=0.3)
    parser.add_argument("--task", type=str, default="sequence", choices=["sequence", "token"])
    parser.add_argument(
        "--head_approach",
        type=str,
        default="emergent",
        choices=["emergent", "phase_offset", "factored"],
    )
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.0)
    return parser


class SyntheticTrainer(BaseTrainer):
    """Trainer for synthetic signal/noise task."""

    def train_step(self, batch):
        sequence = batch["sequence"].to(self.device)

        if self.args.task == "token":
            labels = batch["token_labels"].long().to(self.device)
            outputs = self.model(sequence, return_intermediates=True)
            logits = outputs["token_logits"]
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item() * 100
        else:
            labels = batch["sequence_label"].long().to(self.device)
            outputs = self.model(sequence, return_intermediates=True)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item() * 100

        # Regularization
        if hasattr(self.args, "diversity_weight") and self.args.diversity_weight > 0:
            from src.losses import diversity_loss

            Z_final = outputs.get("intermediates", [None])[-1]
            if Z_final is not None:
                loss = loss + self.args.diversity_weight * diversity_loss(Z_final)

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        sequence = batch["sequence"].to(self.device)

        if self.args.task == "token":
            labels = batch["token_labels"].long().to(self.device)
            outputs = self.model(sequence)
            logits = outputs["token_logits"]
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item() * 100
        else:
            labels = batch["sequence_label"].long().to(self.device)
            outputs = self.model(sequence)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item() * 100

        return {"loss": loss, "accuracy": accuracy}


def create_model(args):
    """Create synthetic task model."""
    model_class = CVKANTokenClassifier if args.task == "token" else CVKAN

    return model_class(
        d_input=args.d_complex,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=2,
        kan_hidden=args.kan_hidden,
        head_approach=args.head_approach,
        pooling=args.pooling,
        input_type="complex",
    )


def create_dataloaders(args):
    """Create synthetic dataloaders."""
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    dataset = SignalNoiseDataset(
        n_samples=args.n_samples,
        n_tokens=args.n_tokens,
        k_signal=args.k_signal,
        d_complex=args.d_complex,
        signal_mag_mean=args.signal_mag_mean,
        signal_phase_std=args.signal_phase_std,
    )

    # Split into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, val_loader, {}
