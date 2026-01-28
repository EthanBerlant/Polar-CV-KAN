"""Extract and visualize learned KAN transform functions.

This script extracts the magnitude and phase transforms from trained models
and visualizes their learned behavior.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import (
    get_device,
    get_polarizing_blocks,
    load_checkpoint,
    plot_transform_function,
    save_results_json,
)


def find_transforms_in_block(block: nn.Module) -> dict:
    """Find magnitude and phase transform networks in a PolarizingBlock.

    The exact naming varies by implementation. We look for common patterns.
    """
    transforms = {}

    for name, module in block.named_modules():
        name_lower = name.lower()

        # Various naming conventions
        if any(
            x in name_lower for x in ["mag_transform", "kan_mag", "magnitude_mlp", "r_transform"]
        ):
            transforms["magnitude"] = module
        elif any(
            x in name_lower
            for x in ["phase_transform", "kan_phase", "theta_transform", "angle_mlp"]
        ):
            transforms["phase"] = module

    # If not found by name, try to infer from structure
    if not transforms:
        children = list(block.named_children())
        for name, child in children:
            if isinstance(child, nn.Sequential) or isinstance(child, nn.Module):
                # Check if it looks like an MLP
                submodules = list(child.modules())
                if any(isinstance(m, nn.Linear) for m in submodules):
                    if "mag" in name or len(transforms) == 0:
                        transforms["magnitude"] = child
                    elif "phase" in name or "magnitude" in transforms:
                        transforms["phase"] = child

    return transforms


def evaluate_transform_function(
    transform: nn.Module,
    input_range: tuple,
    n_points: int = 500,
    device: torch.device = None,
) -> tuple:
    """Evaluate a transform network over a range of inputs.

    Returns:
        (inputs, outputs) as numpy arrays
    """
    if device is None:
        device = next(transform.parameters()).device

    inputs = torch.linspace(input_range[0], input_range[1], n_points, device=device)

    # Handle different input shapes expected by transforms
    # Try (n, 1), (n,), (1, n, 1) etc.
    transform.eval()

    with torch.no_grad():
        for shape in [(n_points, 1), (n_points,), (1, n_points, 1)]:
            try:
                x = inputs.view(shape)
                y = transform(x)
                if y.numel() == n_points:
                    return inputs.cpu().numpy(), y.view(-1).cpu().numpy()
            except Exception:
                continue

    raise ValueError("Could not evaluate transform - incompatible input shape")


def analyze_transform(inputs: np.ndarray, outputs: np.ndarray) -> dict:
    """Analyze properties of a learned transform."""
    analysis = {}

    # Is it approximately identity?
    identity_error = np.mean((outputs - inputs) ** 2)
    analysis["identity_mse"] = float(identity_error)
    analysis["is_near_identity"] = identity_error < 0.01

    # Is it monotonic?
    diffs = np.diff(outputs)
    analysis["is_monotonic_increasing"] = bool(np.all(diffs >= -1e-6))
    analysis["is_monotonic_decreasing"] = bool(np.all(diffs <= 1e-6))
    analysis["is_monotonic"] = (
        analysis["is_monotonic_increasing"] or analysis["is_monotonic_decreasing"]
    )

    # Slope analysis
    if len(inputs) > 1:
        slopes = diffs / (np.diff(inputs) + 1e-8)
        analysis["mean_slope"] = float(np.mean(slopes))
        analysis["min_slope"] = float(np.min(slopes))
        analysis["max_slope"] = float(np.max(slopes))

        # Amplification: slope > 1 means amplifying differences
        analysis["amplifies"] = bool(np.mean(slopes) > 1.0)
        analysis["compresses"] = bool(np.mean(slopes) < 1.0)

    # Nonlinearity measure
    linear_fit = np.polyfit(inputs, outputs, 1)
    linear_pred = np.polyval(linear_fit, inputs)
    nonlinearity = np.mean((outputs - linear_pred) ** 2)
    analysis["nonlinearity"] = float(nonlinearity)

    # Output range
    analysis["output_min"] = float(np.min(outputs))
    analysis["output_max"] = float(np.max(outputs))
    analysis["output_range"] = float(np.max(outputs) - np.min(outputs))

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize learned transforms")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--mag_range",
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        help="Range for magnitude transform (log-magnitude)",
    )
    parser.add_argument(
        "--phase_range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="Range for phase transform (sin/cos components)",
    )
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)

    # Load model architecture
    config = checkpoint.get("config", {})

    # Try different model classes
    model = None
    for model_cls_name in ["CVKAN", "CVKANImageClassifier", "CVKANAudio"]:
        try:
            module = __import__("src.models", fromlist=[model_cls_name])
            model_cls = getattr(module, model_cls_name)
            model = model_cls(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            print(f"Loaded as {model_cls_name}")
            break
        except Exception:
            continue

    if model is None:
        print("Could not load model. Analyzing state dict directly...")
        # Fallback: analyze state dict for transform patterns
        state_dict = checkpoint["model_state_dict"]
        transform_keys = [
            k for k in state_dict.keys() if "transform" in k.lower() or "kan" in k.lower()
        ]
        print(f"Found transform-related keys: {transform_keys}")
        return

    # Find polarizing blocks
    blocks = get_polarizing_blocks(model)
    print(f"Found {len(blocks)} PolarizingBlock modules")

    if not blocks:
        # Try alternative: look for any module with transforms
        print("No PolarizingBlock found. Searching for transform modules...")
        for name, module in model.named_modules():
            if "transform" in name.lower():
                print(f"  Found: {name}")

    all_results = {}

    for i, (block_name, block) in enumerate(blocks):
        print(f"\nAnalyzing block {i}: {block_name}")

        transforms = find_transforms_in_block(block)

        if not transforms:
            print("  No transforms found in block")
            continue

        block_results = {}

        for transform_type, transform_module in transforms.items():
            print(f"  Processing {transform_type} transform...")

            # Determine range
            if transform_type == "magnitude":
                input_range = tuple(args.mag_range)
                xlabel = "Log-Magnitude (input)"
                ylabel = "Log-Magnitude (output)"
            else:
                input_range = tuple(args.phase_range)
                xlabel = "Phase Component (input)"
                ylabel = "Phase Component (output)"

            try:
                inputs, outputs = evaluate_transform_function(
                    transform_module, input_range, device=device
                )
            except Exception as e:
                print(f"    Error evaluating: {e}")
                continue

            # Analyze
            analysis = analyze_transform(inputs, outputs)
            block_results[transform_type] = {
                "inputs": inputs.tolist(),
                "outputs": outputs.tolist(),
                "analysis": analysis,
            }

            # Report
            print(f"    Identity MSE: {analysis['identity_mse']:.6f}")
            print(f"    Monotonic: {analysis['is_monotonic']}")
            print(f"    Mean slope: {analysis['mean_slope']:.4f}")
            print(f"    Amplifies: {analysis['amplifies']}")

            # Plot
            plot_transform_function(
                inputs,
                outputs,
                output_dir / f"transform_{transform_type}_layer{i}.png",
                title=f"{transform_type.title()} Transform (Layer {i})",
                xlabel=xlabel,
                ylabel=ylabel,
            )

        all_results[f"layer_{i}"] = block_results

    # Save
    save_results_json(all_results, output_dir / "transforms.json")

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFORM ANALYSIS SUMMARY")
    print("=" * 60)

    for layer_name, layer_results in all_results.items():
        print(f"\n{layer_name}:")
        for transform_type, data in layer_results.items():
            analysis = data["analysis"]
            status = []
            if analysis.get("amplifies"):
                status.append("AMPLIFYING")
            if analysis.get("compresses"):
                status.append("COMPRESSING")
            if analysis.get("is_near_identity"):
                status.append("~IDENTITY")
            if analysis.get("is_monotonic"):
                status.append("MONOTONIC")

            print(f"  {transform_type}: {', '.join(status) or 'NONLINEAR'}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
