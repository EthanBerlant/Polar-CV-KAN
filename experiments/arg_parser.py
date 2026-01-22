def add_common_args(parser):
    """
    Adds common arguments to the argument parser.
    """
    # Model args
    parser.add_argument("--d_complex", type=int, default=64, help="Model width")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--kan_hidden", type=int, default=32)
    parser.add_argument(
        "--pooling", type=str, default="attention", choices=["mean", "max", "attention"]
    )
    parser.add_argument(
        "--pos_encoding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "none"],
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128 to prevent OOM)"
    )

    # Control
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Use subset for pilot runs")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="MLflow experiment name")

    return parser


def get_defaults():
    """Returns a dictionary of default values for common arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return vars(parser.parse_args([]))
