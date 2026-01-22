import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.train import load_domain_module
from src.configs import get_preset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detailed_param_count(model):
    """Breakdown parameters by component."""
    total = count_parameters(model)

    # Try to identify main components
    embedding_params = 0
    backbone_params = 0
    head_params = 0

    if hasattr(model, "embedding"):
        embedding_params = count_parameters(model.embedding)

    if hasattr(model, "backbone"):
        backbone_params = count_parameters(model.backbone)

    if hasattr(model, "head"):
        head_params = count_parameters(model.head)

    # Check for discrepancies
    summed = embedding_params + backbone_params + head_params
    other = total - summed

    return {
        "Total": total,
        "Embedding": embedding_params,
        "Backbone": backbone_params,
        "Head": head_params,
        "Other": other,
    }


def main():
    parser = argparse.ArgumentParser(description="CVKAN Parameter Counter")
    parser.add_argument("--domain", type=str, required=True, help="Domain name")
    parser.add_argument("--preset", type=str, required=True, help="Preset name")

    # Config overrides
    parser.add_argument("--d_complex", type=int, help="Override d_complex")
    parser.add_argument("--n_layers", type=int, help="Override n_layers")
    parser.add_argument("--kan_hidden", type=int, default=32, help="Override kan_hidden")
    parser.add_argument("--mlp_expansion", type=int, default=4, help="Override mlp expansion")
    parser.add_argument("--skip", action="store_true", help="Enable skip connections")
    parser.add_argument("--pooling", type=str, default="mean", help="Pooling type")

    args = parser.parse_args()

    # Load config
    config = get_preset(args.preset)

    # Apply overrides
    if args.d_complex:
        config.d_complex = args.d_complex
    if args.n_layers:
        config.n_layers = args.n_layers

    config.kan_hidden = args.kan_hidden
    # Note: mlp_expansion is not in config dataclass yet, passed via kwargs usually or fixed in block
    # For now we assume standard block uses config.kan_hidden

    config.skip_connections = args.skip
    config.pooling = args.pooling

    # Load domain
    domain = load_domain_module(args.domain)

    # Initialize model
    print(f"Creating model for {args.domain} with preset {args.preset}")
    print(
        f"Config: d_complex={config.d_complex}, n_layers={config.n_layers}, kan_hidden={config.kan_hidden}"
    )

    try:
        model = domain.create_model(config)

        # Analyze
        counts = detailed_param_count(model)

        print("\nParameter Breakdown:")
        print("-" * 30)
        for k, v in counts.items():
            print(f"{k:<15}: {v:,}")
        print("-" * 30)

        # Verify backbone per layer
        if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            n_layers = len(model.backbone.layers)
            params_per_layer = counts["Backbone"] / n_layers if n_layers > 0 else 0
            print(f"Backbone layers : {n_layers}")
            print(f"Params per layer: {int(params_per_layer):,}")

    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
