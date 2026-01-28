"""Intervention experiments: scramble phase vs magnitude.

Tests the causal role of phase vs magnitude by:
1. Scrambling phases while preserving magnitudes
2. Scrambling magnitudes while preserving phases
3. Measuring performance drop from each intervention
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import get_device, load_checkpoint, plot_comparison_bars, save_results_json


class PhaseScrambleHook:
    """Hook that scrambles phases while preserving magnitudes."""

    def __init__(self, mode: str = "scramble"):
        self.mode = mode  # "scramble", "zero", "uniform"

    def __call__(self, module, input, output):
        if not output.is_complex():
            return output

        mag = torch.abs(output)

        if self.mode == "scramble":
            # Permute phases randomly within each sample
            phase = torch.angle(output)
            batch, seq, dim = phase.shape
            # Scramble across sequence dimension
            for b in range(batch):
                perm = torch.randperm(seq, device=phase.device)
                phase[b] = phase[b, perm]
        elif self.mode == "zero":
            phase = torch.zeros_like(torch.angle(output))
        elif self.mode == "uniform":
            phase = torch.rand_like(torch.angle(output)) * 2 * np.pi - np.pi

        return mag * torch.exp(1j * phase)


class MagnitudeScrambleHook:
    """Hook that scrambles magnitudes while preserving phases."""

    def __init__(self, mode: str = "scramble"):
        self.mode = mode

    def __call__(self, module, input, output):
        if not output.is_complex():
            return output

        phase = torch.angle(output)

        if self.mode == "scramble":
            mag = torch.abs(output)
            batch, seq, dim = mag.shape
            for b in range(batch):
                perm = torch.randperm(seq, device=mag.device)
                mag[b] = mag[b, perm]
        elif self.mode == "uniform":
            mag = torch.ones_like(torch.abs(output))
        elif self.mode == "mean":
            mag = torch.abs(output)
            mag = mag.mean(dim=1, keepdim=True).expand_as(mag)

        return mag * torch.exp(1j * phase)


def evaluate_with_intervention(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    hook_class=None,
    hook_mode: str = "scramble",
    target_layers: str = "all",  # "all", "last", "first"
) -> dict:
    """Evaluate model with an intervention hook applied.

    Args:
        model: The model
        dataloader: Data loader
        device: Device
        hook_class: Hook class to apply (None for clean evaluation)
        hook_mode: Mode for the hook
        target_layers: Which layers to apply hook to

    Returns:
        {"accuracy": float, "loss": float}
    """
    model.eval()

    # Find layers to hook
    layers_to_hook = []
    for name, module in model.named_modules():
        if "polarizing" in name.lower() or "block" in name.lower():
            layers_to_hook.append((name, module))

    if target_layers == "last" and layers_to_hook:
        layers_to_hook = [layers_to_hook[-1]]
    elif target_layers == "first" and layers_to_hook:
        layers_to_hook = [layers_to_hook[0]]

    # Register hooks
    hooks = []
    if hook_class is not None:
        hook_fn = hook_class(mode=hook_mode)
        for name, module in layers_to_hook:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch["input_ids"]
                    labels = batch["labels"]

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return {
        "accuracy": correct / total if total > 0 else 0,
        "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0,
    }


def run_interventions(args, output_dir: Path):
    """Run all intervention experiments."""
    device = get_device()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    config = checkpoint.get("config", {})

    # Load model
    try:
        if args.domain == "sst2":
            from src.models import CVKAN

            model = CVKAN(**config)
        else:
            from src.models import CVKAN

            model = CVKAN(**config)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get dataloader
    print(f"Loading {args.domain} data...")
    if args.domain == "sst2":
        from src.data import SST2Dataset

        dataset = SST2Dataset(split="validation")
        # Subset for speed
        if len(dataset) > 1000:
            indices = torch.randperm(len(dataset))[:1000]
            dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    else:
        print(f"Domain {args.domain} not yet supported")
        return

    results = {}

    # Baseline (no intervention)
    print("\nEvaluating baseline...")
    baseline = evaluate_with_intervention(model, dataloader, device)
    results["baseline"] = baseline
    print(f"  Accuracy: {baseline['accuracy']:.4f}")

    # Phase scramble
    print("\nPhase scramble interventions:")
    for mode in ["scramble", "zero", "uniform"]:
        print(f"  Mode: {mode}...")
        result = evaluate_with_intervention(
            model,
            dataloader,
            device,
            hook_class=PhaseScrambleHook,
            hook_mode=mode,
        )
        results[f"phase_{mode}"] = result
        drop = baseline["accuracy"] - result["accuracy"]
        print(f"    Accuracy: {result['accuracy']:.4f} (Δ={-drop:+.4f})")

    # Magnitude scramble
    print("\nMagnitude scramble interventions:")
    for mode in ["scramble", "uniform", "mean"]:
        print(f"  Mode: {mode}...")
        result = evaluate_with_intervention(
            model,
            dataloader,
            device,
            hook_class=MagnitudeScrambleHook,
            hook_mode=mode,
        )
        results[f"magnitude_{mode}"] = result
        drop = baseline["accuracy"] - result["accuracy"]
        print(f"    Accuracy: {result['accuracy']:.4f} (Δ={-drop:+.4f})")

    # Layer-specific interventions
    print("\nLayer-specific phase scramble:")
    for layer_target in ["first", "last", "all"]:
        print(f"  Target: {layer_target}...")
        result = evaluate_with_intervention(
            model,
            dataloader,
            device,
            hook_class=PhaseScrambleHook,
            hook_mode="scramble",
            target_layers=layer_target,
        )
        results[f"phase_scramble_{layer_target}_layer"] = result
        drop = baseline["accuracy"] - result["accuracy"]
        print(f"    Accuracy: {result['accuracy']:.4f} (Δ={-drop:+.4f})")

    # Compute summary
    phase_drop = baseline["accuracy"] - results["phase_scramble"]["accuracy"]
    magnitude_drop = baseline["accuracy"] - results["magnitude_scramble"]["accuracy"]

    results["summary"] = {
        "baseline_accuracy": baseline["accuracy"],
        "phase_scramble_drop": phase_drop,
        "magnitude_scramble_drop": magnitude_drop,
        "phase_matters_more": phase_drop > magnitude_drop,
        "magnitude_matters_more": magnitude_drop > phase_drop,
    }

    # Visualize
    print("\nGenerating visualizations...")

    comparison = {
        "Baseline": baseline["accuracy"],
        "Phase Scrambled": results["phase_scramble"]["accuracy"],
        "Magnitude Scrambled": results["magnitude_scramble"]["accuracy"],
        "Phase Zeroed": results["phase_zero"]["accuracy"],
        "Magnitude Uniform": results["magnitude_uniform"]["accuracy"],
    }

    plot_comparison_bars(
        comparison,
        output_dir / "intervention_comparison.png",
        title="Effect of Phase vs Magnitude Interventions",
        ylabel="Accuracy",
    )

    # Save
    save_results_json(results, output_dir / "intervention_results.json")

    # Report
    print("\n" + "=" * 60)
    print("INTERVENTION ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nBaseline accuracy: {baseline['accuracy']:.4f}")
    print(f"\nPhase scramble drop:     {phase_drop:+.4f}")
    print(f"Magnitude scramble drop: {magnitude_drop:+.4f}")

    if phase_drop > magnitude_drop + 0.02:
        print("\n→ PHASE MATTERS MORE: Scrambling phases hurts more than magnitudes")
        print("  This suggests phase encodes important information")
    elif magnitude_drop > phase_drop + 0.02:
        print("\n→ MAGNITUDE MATTERS MORE: Scrambling magnitudes hurts more")
        print("  This supports the polarization hypothesis")
    else:
        print("\n→ BOTH MATTER EQUALLY: No clear winner")
        print("  The architecture uses both channels effectively")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="sst2")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_interventions(args, output_dir)


if __name__ == "__main__":
    main()
