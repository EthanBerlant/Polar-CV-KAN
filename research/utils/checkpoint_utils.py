"""Utilities for loading and inspecting model checkpoints."""

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


def get_device(preferred: str = "cuda") -> torch.device:
    """Get available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device | None = None) -> dict[str, Any]:
    """Load checkpoint from path.

    Args:
        path: Path to checkpoint file
        device: Device to load to (default: CPU for inspection)

    Returns:
        Checkpoint dict containing model_state_dict, config, etc.
    """
    if device is None:
        device = torch.device("cpu")

    if not path.exists():
        # Try common locations
        candidates = [
            path,
            Path("outputs") / path.name,
            Path("outputs/checkpoints") / path.name,
            Path("outputs/sst2") / path.name,
            Path("outputs/cifar10") / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Searched: {[str(c) for c in candidates]}\n"
                f"Please provide full path to a trained model checkpoint."
            )

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_class: type | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class to instantiate (if None, tries to infer)
        device: Device to load to

    Returns:
        (model, config)
    """
    if device is None:
        device = get_device()

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})

    # Try to import and instantiate model
    if model_class is None:
        # Try to infer from config
        model_type = config.get("model_type", "CVKAN")
        try:
            if model_type == "CVKAN":
                from src.models import CVKAN

                model_class = CVKAN
            elif model_type == "CVKANImageClassifier":
                from src.models import CVKANImageClassifier

                model_class = CVKANImageClassifier
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except ImportError as e:
            print(f"Could not import model class: {e}")
            print("Make sure src/ is in your Python path")
            sys.exit(1)

    # Instantiate model
    model = model_class(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def extract_layer_outputs(
    model: nn.Module,
    inputs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract intermediate layer outputs via hooks.

    Args:
        model: The model
        inputs: Input tensor
        mask: Optional mask

    Returns:
        Dict mapping layer names to their outputs
    """
    outputs = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach()
            elif isinstance(output, tuple):
                outputs[name] = output[0].detach()

        return hook

    # Register hooks on relevant layers
    for name, module in model.named_modules():
        if "polarizing" in name.lower() or "block" in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        if mask is not None:
            _ = model(inputs, mask=mask)
        else:
            _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return outputs


def get_polarizing_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Get all PolarizingBlock modules from model.

    Returns:
        List of (name, module) tuples
    """
    blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "PolarizingBlock":
            blocks.append((name, module))
    return blocks


def extract_kan_transforms(block: nn.Module) -> dict[str, nn.Module]:
    """Extract the learned 1D transform networks from a PolarizingBlock.

    Returns:
        Dict with 'magnitude_transform' and 'phase_transform' modules
    """
    transforms = {}

    for name, module in block.named_modules():
        if "mag" in name.lower() and "transform" in name.lower():
            transforms["magnitude_transform"] = module
        elif "phase" in name.lower() and "transform" in name.lower():
            transforms["phase_transform"] = module
        # Also check for kan_magnitude, kan_phase naming
        elif "kan_mag" in name.lower():
            transforms["magnitude_transform"] = module
        elif "kan_phase" in name.lower():
            transforms["phase_transform"] = module

    return transforms


def evaluate_transform(
    transform: nn.Module,
    input_range: tuple[float, float],
    n_points: int = 1000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a 1D transform over a range.

    Args:
        transform: The transform module
        input_range: (min, max) input values
        n_points: Number of evaluation points
        device: Device to use

    Returns:
        (inputs, outputs) tensors
    """
    if device is None:
        device = next(transform.parameters()).device

    inputs = torch.linspace(input_range[0], input_range[1], n_points, device=device)
    inputs = inputs.unsqueeze(-1)  # (n_points, 1) for MLP

    with torch.no_grad():
        outputs = transform(inputs)

    return inputs.squeeze(), outputs.squeeze()


def get_model_statistics(model: nn.Module) -> dict[str, Any]:
    """Get summary statistics about a model.

    Returns:
        Dict with parameter counts, layer info, etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count complex vs real parameters
    complex_params = 0
    real_params = 0
    for p in model.parameters():
        if p.is_complex():
            complex_params += p.numel()
        else:
            real_params += p.numel()

    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        layer_counts[class_name] = layer_counts.get(class_name, 0) + 1

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "complex_params": complex_params,
        "real_params": real_params,
        "layer_counts": layer_counts,
    }


def find_checkpoints(
    directory: Path,
    pattern: str = "*.pt",
) -> list[Path]:
    """Find all checkpoint files in a directory."""
    return sorted(directory.glob(pattern))
