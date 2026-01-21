"""
General NLP domain (IMDB, AG_NEWS, SST-5).
"""

import torch.nn.functional as F

from src.configs.model import NLPConfig
from src.configs.training import TrainingConfig
from src.data.nlp_loader import NLPDataLoader
from src.models.cv_kan_nlp import CVKANNLP
from src.trainer import BaseTrainer


def create_model(config: NLPConfig):
    """Create NLP classification model."""
    return CVKANNLP(
        vocab_size=config.vocab_size,
        d_complex=config.d_complex,
        n_layers=config.n_layers,
        n_classes=config.n_classes,
        max_seq_len=config.max_seq_len,
        kan_hidden=config.kan_hidden,
        pooling=config.pooling,
        block_type=config.block_type,
        dropout=config.dropout,
        input_type=config.input_type,
    )


def create_dataloaders(model_config: NLPConfig, train_config: TrainingConfig):
    """Create NLP dataloaders."""
    # Initialize loader wrapper
    loader = NLPDataLoader(
        dataset_name=model_config.dataset_name,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        root="./data",  # Could be configurable in TrainingConfig or constant
        subset_size=train_config.subset_size,
    )

    # helper to get loaders
    train_loader, test_loader, n_classes, vocab_size = loader.get_dataloaders()

    # We use test as val for now if no separate val split
    val_loader = test_loader

    return train_loader, val_loader, test_loader, {"n_classes": n_classes, "vocab_size": vocab_size}


class NLPTrainer(BaseTrainer):
    """Trainer for NLP classification."""

    def train_step(self, batch):
        if isinstance(batch, dict):
            indices = batch["indices"]
            labels = batch["label"]
            mask = batch.get("mask", None)
        else:
            indices, labels = batch
            mask = None

        indices = indices.to(self.device).long()
        labels = labels.to(self.device).long()
        if mask is not None:
            mask = mask.to(self.device)

        # CVKANNLP signature
        outputs = self.model(indices, mask=mask)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item() * 100

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        if isinstance(batch, dict):
            indices = batch["indices"]
            labels = batch["label"]
            mask = batch.get("mask", None)
        else:
            indices, labels = batch
            mask = None

        indices = indices.to(self.device).long()
        labels = labels.to(self.device).long()
        if mask is not None:
            mask = mask.to(self.device)

        outputs = self.model(indices, mask=mask)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item() * 100

        return {"loss": loss, "accuracy": accuracy}


Trainer = NLPTrainer
