"""Compute mutual information between phase/magnitude and class labels.

I(phase; label) and I(magnitude; label) quantify how much information
each component carries about the classification target.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import (
    extract_layer_outputs,
    get_device,
    load_checkpoint,
    plot_comparison_bars,
    save_results_json,
)


def extract_representations_for_mi(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 2000,
) -> dict:
    """Extract pooled phase and magnitude for each sample."""
    model.eval()

    all_phases = []
    all_magnitudes = []
    all_labels = []

    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_samples >= max_samples:
                break

            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["input_ids"]
                labels = batch["labels"]

            inputs = inputs.to(device)

            layer_outputs = extract_layer_outputs(model, inputs)

            if not layer_outputs:
                continue

            last_layer = max(layer_outputs.keys())
            Z = layer_outputs[last_layer]["output"]

            phases = torch.angle(Z).mean(dim=1).cpu().numpy()
            magnitudes = torch.abs(Z).mean(dim=1).cpu().numpy()

            all_phases.append(phases)
            all_magnitudes.append(magnitudes)
            all_labels.append(labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels))

            n_samples += len(inputs)

    if not all_phases:
        return None

    return {
        "phases": np.vstack(all_phases),
        "magnitudes": np.vstack(all_magnitudes),
        "labels": np.concatenate(all_labels),
    }


def compute_mutual_information(features: np.ndarray, labels: np.ndarray) -> dict:
    """Compute mutual information between features and labels."""
    mi_scores = mutual_info_classif(
        features,
        labels,
        discrete_features=False,
        n_neighbors=5,
        random_state=42,
    )

    return {
        "per_feature": mi_scores.tolist(),
        "mean": float(np.mean(mi_scores)),
        "max": float(np.max(mi_scores)),
        "total": float(np.sum(mi_scores)),
        "std": float(np.std(mi_scores)),
    }


def compute_phase_mi(phases: np.ndarray, labels: np.ndarray) -> dict:
    """Compute MI for phase features (circular data)."""
    sin_phases = np.sin(phases)
    cos_phases = np.cos(phases)
    phase_features = np.hstack([sin_phases, cos_phases])

    mi_result = compute_mutual_information(phase_features, labels)

    d = phases.shape[1]
    per_dim_mi = []
    for i in range(d):
        pair_mi = (mi_result["per_feature"][i] + mi_result["per_feature"][i + d]) / 2
        per_dim_mi.append(pair_mi)

    return {
        "per_dimension": per_dim_mi,
        "mean": float(np.mean(per_dim_mi)),
        "max": float(np.max(per_dim_mi)),
        "total": float(np.sum(per_dim_mi)),
        "std": float(np.std(per_dim_mi)),
    }


def run_mi_analysis(args, output_dir: Path):
    """Run mutual information analysis."""
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
        from torch.utils.data import DataLoader

        from src.data import SST2Dataset

        dataset = SST2Dataset(split="validation")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    else:
        print(f"Domain {args.domain} not yet supported")
        return

    # Extract representations
    print("Extracting representations...")
    data = extract_representations_for_mi(model, dataloader, device, args.max_samples)

    if data is None:
        print("Could not extract representations")
        return

    print(f"Extracted {len(data['labels'])} samples")

    # Compute MI for magnitudes
    print("Computing I(magnitude; label)...")
    mi_magnitude = compute_mutual_information(data["magnitudes"], data["labels"])

    # Compute MI for phases
    print("Computing I(phase; label)...")
    mi_phase = compute_phase_mi(data["phases"], data["labels"])

    # Compute MI for combined
    print("Computing I(combined; label)...")
    combined = np.hstack(
        [
            data["magnitudes"],
            np.sin(data["phases"]),
            np.cos(data["phases"]),
        ]
    )
    mi_combined = compute_mutual_information(combined, data["labels"])

    # Results
    results = {
        "magnitude": mi_magnitude,
        "phase": mi_phase,
        "combined": mi_combined,
        "n_samples": len(data["labels"]),
        "d_complex": data["magnitudes"].shape[1],
    }

    # Visualize
    comparison = {
        "Magnitude": mi_magnitude["mean"],
        "Phase": mi_phase["mean"],
        "Combined": mi_combined["mean"],
    }

    plot_comparison_bars(
        comparison,
        output_dir / "mutual_information_comparison.png",
        title="Mutual Information with Class Label",
        ylabel="Mean MI (bits)",
    )

    # Save
    save_results_json(results, output_dir / "mutual_information.json")

    # Report
    print("\n" + "=" * 60)
    print("MUTUAL INFORMATION ANALYSIS")
    print("=" * 60)

    print("\nI(magnitude; label):")
    print(f"  Mean: {mi_magnitude['mean']:.4f}")
    print(f"  Max:  {mi_magnitude['max']:.4f}")
    print(f"  Sum:  {mi_magnitude['total']:.4f}")

    print("\nI(phase; label):")
    print(f"  Mean: {mi_phase['mean']:.4f}")
    print(f"  Max:  {mi_phase['max']:.4f}")
    print(f"  Sum:  {mi_phase['total']:.4f}")

    print("\nI(combined; label):")
    print(f"  Mean: {mi_combined['mean']:.4f}")

    # Interpretation
    ratio = mi_magnitude["mean"] / (mi_phase["mean"] + 1e-8)

    print("\n" + "-" * 40)
    if ratio > 2:
        print("→ MAGNITUDE DOMINATES: Carries much more label information")
        print("  This supports magnitude-based polarization as attention")
    elif ratio < 0.5:
        print("→ PHASE DOMINATES: Carries much more label information")
        print("  Phase semantics are important for this task")
    else:
        print("→ BOTH CONTRIBUTE: Roughly equal label information")
        print("  The architecture uses both channels")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="sst2")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_mi_analysis(args, output_dir)


if __name__ == "__main__":
    main()
