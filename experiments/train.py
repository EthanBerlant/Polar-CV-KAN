import argparse
import os
import sys
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import TrainingConfig, get_preset
from src.tracking import ExperimentTracker


def load_domain_module(domain_name):
    # Mapping friendly names to module paths
    # Assuming valid domains are in experiments/domains/
    return import_module(f"experiments.domains.{domain_name}")


def main():
    parser = argparse.ArgumentParser(description="Deep Refactor Trainer")
    parser.add_argument(
        "--domain", type=str, required=True, help="Domain name (nlp, image, audio, etc)"
    )
    parser.add_argument("--preset", type=str, help="Preset name (e.g., sst2, cifar10)")
    parser.add_argument("--config", type=str, help="Path to custom config yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")

    # Model config overrides
    parser.add_argument("--d_complex", type=int, help="Override d_complex")
    parser.add_argument("--n_layers", type=int, help="Override n_layers")
    parser.add_argument("--kan_hidden", type=int, help="Override kan_hidden")
    parser.add_argument("--run_name", type=str, help="Override run name")
    parser.add_argument("--pooling", type=str, help="Override pooling")
    parser.add_argument("--dropout", type=float, help="Override dropout")
    parser.add_argument("--embedding_type", type=str, help="Override embedding_type (image only)")

    parser.add_argument("--subset_size", type=int, help="Limit dataset size")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # 1. Load Configuration
    if args.preset:
        model_config = get_preset(args.preset)
        print(f"Loaded preset: {args.preset}")
    else:
        # Default fallback or error
        raise ValueError("Please provide --preset (config file loading not fully impl yet)")

    # Apply overrides
    if args.d_complex:
        model_config.d_complex = args.d_complex
    if args.n_layers:
        model_config.n_layers = args.n_layers
    if args.kan_hidden:
        model_config.kan_hidden = args.kan_hidden
    if args.pooling:
        model_config.pooling = args.pooling
    if args.dropout is not None:
        model_config.dropout = args.dropout
    if args.embedding_type and hasattr(model_config, "embedding_type"):
        model_config.embedding_type = args.embedding_type

    # Training config
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs if args.epochs else 50,
        batch_size=args.batch_size if args.batch_size else 32,
        subset_size=args.subset_size,
        patience=args.patience if args.patience else 10,
        seed=args.seed if args.seed else 42,
    )

    # 2. Load Domain Module
    domain = load_domain_module(args.domain)

    # 3. Create DataLoaders
    # Adapters should accept config and return loaders + updated config (e.g. n_classes)
    # But updated config might mismatch dataclass if we add fields?
    # We expect adapters to modify the passed config object or return metadata.

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, meta = domain.create_dataloaders(
        model_config, train_config
    )

    # Update config from metadata
    if "n_classes" in meta:
        model_config.n_classes = meta["n_classes"]
    if "vocab_size" in meta and hasattr(model_config, "vocab_size"):
        model_config.vocab_size = meta["vocab_size"]
    # Add other dynamic meta if needed

    # 4. Create Model
    print(f"Creating model for {args.domain}...")
    model = domain.create_model(model_config)

    # 5. Setup Trainer
    # Domain module can provide a specific Trainer class or we use generic
    TrainerClass = getattr(domain, "Trainer", None)
    if TrainerClass is None:
        # Fallback to generic Trainer if possible, or import from src.trainer
        from src.trainer import BaseTrainer

        TrainerClass = BaseTrainer

    tracker = ExperimentTracker(
        experiment_name=f"cvkan_{args.domain}",
        run_name=args.run_name or args.preset or "custom_run",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    # create scheduler (simple placeholder)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=train_config.metric_mode, patience=train_config.patience // 2
    )

    # Update args for BaseTrainer (it expects generic args object)
    # We mix training config and model config for logging
    combined_args = argparse.Namespace(**asdict(train_config))
    for k, v in asdict(model_config).items():
        setattr(combined_args, k, v)
    combined_args.domain = args.domain

    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=Path(train_config.output_dir),
        args=combined_args,
        use_amp=train_config.amp,
    )

    # 6. Train
    print("Starting training...")
    with tracker:
        tracker.log_params(asdict(model_config))
        tracker.log_params(asdict(train_config))

        history, train_time = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_config.epochs,
            patience=train_config.patience,
            metric_name=train_config.metric_name,
        )

        tracker.log_metrics({"train_time": train_time})


if __name__ == "__main__":
    main()
