import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

# Add project root
# We use Path to avoid PTH120
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.configs.model import ExperimentConfig  # noqa: E402
from src.factories import DataFactory, ModelFactory  # noqa: E402
from src.tracking import ExperimentTracker  # noqa: E402
from src.trainer import BaseTrainer  # noqa: E402
from src.utils import cleanup_gpu  # noqa: E402


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load and merge configuration from defaults, YAML, and CLI args.

    Args:
        args: Parsed command line arguments.

    Returns:
        ExperimentConfig: Fully populated configuration object.
    """
    # 1. Defaults
    config = ExperimentConfig()

    # 2. Load YAML if provided
    if args.config:
        with Path(args.config).open() as f:
            yaml.safe_load(f)
            # Recursively update dataclasses (simplified)
            # TODO: Robust dict->dataclass merge

    # 3. CLI Overrides (simplified mapping)
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.trainer.epochs = args.epochs

    # Model overrides
    if args.d_complex:
        config.model.d_complex = args.d_complex
    if args.aggregation:
        config.model.aggregation = args.aggregation
    if args.normalization:
        config.model.normalization = args.normalization

    return config


class UniversalTrainer(BaseTrainer):
    """Concrete trainer implementation for standard classification tasks."""

    def train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            batch: The input batch from the dataloader.

        Returns:
            Dict containing loss and metrics.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        outputs = self.model(x)
        logits = outputs["logits"]
        loss = torch.nn.functional.cross_entropy(logits, y)

        # Simple accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        return {"loss": loss, "accuracy": acc}

    def validate_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single validation step.

        Args:
            batch: The input batch from the dataloader.

        Returns:
            Dict containing loss and metrics.
        """
        return self.train_step(batch)


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Universal Polar-CV-KAN Trainer")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Epochs")

    # Model args
    parser.add_argument("--d_complex", type=int, help="Complex dimension")
    parser.add_argument("--aggregation", type=str, help="Aggregation type")
    parser.add_argument("--normalization", type=str, help="Normalization type")

    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args)
    print(f"Configuration loaded for dataset: {config.data.dataset_name}")
    print(f"Model: {config.model.aggregation} / {config.model.normalization}")

    # 2. Setup Data
    print("Loading data...")
    train_loader, val_loader, test_loader, meta = DataFactory.create_dataloaders(config)
    print(f"Data loaded. Meta: {meta}")

    # 3. Build Model
    print("Building model...")
    cleanup_gpu()
    model = ModelFactory.create(config, meta)

    # 4. Setup Tracker & Trainer
    tracker = ExperimentTracker(
        experiment_name=config.trainer.project_name,
        run_name=config.trainer.run_name
        or f"{config.data.dataset_name}_{config.model.aggregation}",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.trainer.lr, weight_decay=config.trainer.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config.trainer.metric_mode, patience=config.trainer.patience // 2
    )

    combined_args = argparse.Namespace(
        **asdict(config.model), **asdict(config.data), **asdict(config.trainer)
    )
    combined_args.domain = config.data.dataset_name  # Legacy support

    # Use generic trainer
    trainer = UniversalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=Path(config.trainer.output_dir),
        args=combined_args,
    )

    # 5. Train
    print("Starting training...")
    with tracker:
        tracker.log_params(asdict(config.model))

        trainer.fit(
            train_loader,
            val_loader,
            epochs=config.trainer.epochs,
            patience=config.trainer.patience,
            metric_name=config.trainer.metric_name,
        )


if __name__ == "__main__":
    main()
