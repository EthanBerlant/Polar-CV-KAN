"""Measure polarization metrics across layers of a trained model.

This script loads a trained checkpoint, runs data through the model,
and tracks how polarization metrics evolve at each layer.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import (
    compute_all_metrics,
    get_device,
    load_checkpoint,
    plot_multi_metric_trajectory,
    plot_polarization_trajectory,
    save_results_json,
    summarize_metrics,
)


class LayerMetricsHook:
    """Hook to capture layer outputs and compute metrics."""

    def __init__(self):
        self.layer_outputs = {}
        self.hooks = []

    def register(self, model: nn.Module):
        """Register hooks on all relevant layers."""
        layer_idx = 0
        for name, module in model.named_modules():
            # Capture outputs of PolarizingBlock, EmergentHeadsPolarizing, etc.
            class_name = module.__class__.__name__
            if "Polarizing" in class_name or "Block" in class_name:
                hook = module.register_forward_hook(self._make_hook(layer_idx, name))
                self.hooks.append(hook)
                layer_idx += 1

    def _make_hook(self, idx: int, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.is_complex():
                self.layer_outputs[idx] = {
                    "name": name,
                    "output": output.detach(),
                }

        return hook

    def clear(self):
        self.layer_outputs = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_dataloader(domain: str, batch_size: int = 32, subset_size: int = 1000):
    """Get a dataloader for the specified domain."""
    if domain == "sst2":
        from src.data import SST2Dataset

        dataset = SST2Dataset(split="validation")
    elif domain == "synthetic":
        from src.data import SignalNoiseDataset

        dataset = SignalNoiseDataset(n_samples=subset_size)
    elif domain == "cifar10":
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Subset
    if len(dataset) > subset_size:
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def measure_polarization(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run data through model and measure polarization at each layer.

    Returns:
        {layer_idx: {metric_name: value, ...}, ...}
    """
    model.eval()
    hook_manager = LayerMetricsHook()
    hook_manager.register(model)

    # Accumulate metrics
    all_layer_metrics = {}
    n_batches = 0

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                hook_manager.clear()

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        inputs, _ = batch
                    elif len(batch) == 3:
                        inputs, _, mask = batch
                else:
                    inputs = batch

                inputs = inputs.to(device)

                # Forward pass
                try:
                    _ = model(inputs)
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    continue

                # Compute metrics for each layer
                for layer_idx, layer_data in hook_manager.layer_outputs.items():
                    Z = layer_data["output"]

                    # Skip if not the right shape (batch, tokens, dim)
                    if Z.ndim != 3:
                        continue

                    metrics = compute_all_metrics(Z, dim=1)  # Across tokens
                    summary = summarize_metrics(metrics)

                    if layer_idx not in all_layer_metrics:
                        all_layer_metrics[layer_idx] = {k: 0.0 for k in summary}
                        all_layer_metrics[layer_idx]["name"] = layer_data["name"]

                    for k, v in summary.items():
                        all_layer_metrics[layer_idx][k] += v

                n_batches += 1
    finally:
        hook_manager.remove_hooks()

    # Average
    for layer_idx in all_layer_metrics:
        for k in all_layer_metrics[layer_idx]:
            if k != "name":
                all_layer_metrics[layer_idx][k] /= n_batches

    return all_layer_metrics


def main():
    parser = argparse.ArgumentParser(description="Measure polarization across layers")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--domain", type=str, default="sst2", choices=["sst2", "synthetic", "cifar10"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset_size", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Setup
    device = get_device()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    config = checkpoint.get("config", {})

    # Instantiate model based on domain
    try:
        if args.domain == "sst2":
            from src.models import CVKAN

            model = CVKAN(**config)
        elif args.domain == "cifar10":
            from src.models import CVKANImageClassifier

            model = CVKANImageClassifier(**config)
        else:
            from src.models import CVKAN

            model = CVKAN(**config)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting generic model load...")
        # Fallback: just use the state dict to analyze
        raise

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    # Get data
    print(f"Loading {args.domain} data...")
    dataloader = get_dataloader(args.domain, args.batch_size, args.subset_size)

    # Measure
    print("Measuring polarization...")
    layer_metrics = measure_polarization(model, dataloader, device)

    if not layer_metrics:
        print("No layer metrics captured. Check model structure.")
        return

    # Report
    print("\n" + "=" * 60)
    print("POLARIZATION METRICS BY LAYER")
    print("=" * 60)

    for layer_idx in sorted(layer_metrics.keys()):
        metrics = layer_metrics[layer_idx]
        print(f"\nLayer {layer_idx} ({metrics.get('name', 'unknown')}):")
        for k, v in sorted(metrics.items()):
            if k != "name":
                print(f"  {k}: {v:.4f}")

    # Save results
    save_results_json(layer_metrics, output_dir / "layer_metrics.json")

    # Visualize
    metrics_to_plot = ["magnitude_variance_mean", "max_mean_ratio_mean", "gini_mean"]

    for metric in metrics_to_plot:
        if all(metric in layer_metrics[l] for l in layer_metrics):
            plot_polarization_trajectory(
                layer_metrics,
                metric,
                output_dir / f"trajectory_{metric}.png",
                title=f"{metric} Across Layers ({args.domain})",
            )

    # Multi-metric plot
    available_metrics = [
        m for m in metrics_to_plot if all(m in layer_metrics[l] for l in layer_metrics)
    ]
    if available_metrics:
        plot_multi_metric_trajectory(
            layer_metrics,
            available_metrics,
            output_dir / "trajectory_all_metrics.png",
            title=f"Polarization Metrics ({args.domain})",
        )

    # Summary
    print(f"\nResults saved to {output_dir}")

    # Quick analysis
    layers = sorted(layer_metrics.keys())
    if len(layers) >= 2:
        first_var = layer_metrics[layers[0]].get("magnitude_variance_mean", 0)
        last_var = layer_metrics[layers[-1]].get("magnitude_variance_mean", 0)

        if last_var > first_var:
            print(
                f"\n✓ POLARIZATION CONFIRMED: Variance increased from {first_var:.4f} to {last_var:.4f}"
            )
        else:
            print(f"\n✗ NO POLARIZATION: Variance decreased from {first_var:.4f} to {last_var:.4f}")


if __name__ == "__main__":
    main()
