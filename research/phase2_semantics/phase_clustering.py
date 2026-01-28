"""Cluster tokens by phase and analyze semantic composition.

Extracts final-layer representations, clusters by phase angle,
and examines what semantic properties align with phase clusters.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import (
    circular_mean,
    circular_variance,
    extract_layer_outputs,
    get_device,
    load_checkpoint,
    plot_magnitude_phase_scatter,
    plot_phase_histogram,
    save_results_json,
)


def circular_kmeans(phases: np.ndarray, n_clusters: int, n_init: int = 10) -> np.ndarray:
    """K-means clustering on circular data (phases).

    Converts phases to unit vectors, clusters in 2D, then assigns.
    """
    # Convert to unit vectors
    X = np.stack([np.cos(phases), np.sin(phases)], axis=1)

    # Standard k-means on 2D unit vectors
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X)

    # Get cluster centers as phases
    centers = np.arctan2(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0])

    return labels, centers


def extract_token_representations(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 1000,
) -> dict:
    """Extract final-layer complex representations for all tokens.

    Returns:
        {
            'phases': array of phase angles,
            'magnitudes': array of magnitudes,
            'tokens': list of token strings (if available),
            'labels': array of sample labels,
            'positions': array of position indices,
        }
    """
    model.eval()

    all_phases = []
    all_magnitudes = []
    all_tokens = []
    all_labels = []
    all_positions = []

    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_samples >= max_samples:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                inputs = batch.get("input_ids", batch.get("inputs"))
                labels = batch.get("labels", batch.get("label"))
                tokens = batch.get("tokens", None)
            elif isinstance(batch, (tuple, list)):
                if len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                    tokens = batch[2] if len(batch) > 2 else None
                else:
                    inputs = batch[0]
                    labels = None
                    tokens = None
            else:
                inputs = batch
                labels = None
                tokens = None

            inputs = inputs.to(device)

            # Get layer outputs
            layer_outputs = extract_layer_outputs(model, inputs)

            if not layer_outputs:
                # Fallback: run forward and hope final representation is accessible
                try:
                    # Try to access internal representation
                    _ = model(inputs)
                    # Look for last complex tensor in model
                    for name, module in reversed(list(model.named_modules())):
                        if hasattr(module, "last_output") and module.last_output.is_complex():
                            Z = module.last_output
                            break
                    else:
                        print("Could not extract complex representations")
                        continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            else:
                # Use the last layer output
                last_layer = max(layer_outputs.keys())
                Z = layer_outputs[last_layer]["output"]

            # Z: (batch, seq_len, d_complex)
            batch_size, seq_len, d_complex = Z.shape

            # Flatten and extract
            phases = torch.angle(Z).cpu().numpy()  # (batch, seq, d)
            magnitudes = torch.abs(Z).cpu().numpy()

            # Average over dimensions to get per-token phase/magnitude
            phases_avg = np.arctan2(
                np.sin(phases).mean(axis=-1), np.cos(phases).mean(axis=-1)
            )  # (batch, seq)
            magnitudes_avg = magnitudes.mean(axis=-1)  # (batch, seq)

            for b in range(batch_size):
                if n_samples >= max_samples:
                    break

                for t in range(seq_len):
                    all_phases.append(phases_avg[b, t])
                    all_magnitudes.append(magnitudes_avg[b, t])
                    all_positions.append(t)

                    if labels is not None:
                        all_labels.append(
                            labels[b].item() if torch.is_tensor(labels[b]) else labels[b]
                        )

                    if tokens is not None and hasattr(tokens, "__getitem__"):
                        try:
                            all_tokens.append(str(tokens[b][t]))
                        except:
                            all_tokens.append(f"pos_{t}")

                n_samples += 1

    return {
        "phases": np.array(all_phases),
        "magnitudes": np.array(all_magnitudes),
        "positions": np.array(all_positions),
        "labels": np.array(all_labels) if all_labels else None,
        "tokens": all_tokens if all_tokens else None,
    }


def analyze_clusters(
    phases: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray,
    tokens: list = None,
) -> dict:
    """Analyze the composition of phase clusters."""
    n_clusters = len(np.unique(cluster_labels))
    analysis = {"clusters": {}}

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_data = {
            "size": int(mask.sum()),
            "mean_phase": float(circular_mean(phases[mask])),
            "phase_variance": float(circular_variance(phases[mask])),
        }

        # Label distribution within cluster
        if labels is not None:
            label_counts = Counter(labels[mask])
            cluster_data["label_distribution"] = dict(label_counts)

            # Purity: fraction of most common label
            if label_counts:
                cluster_data["purity"] = max(label_counts.values()) / mask.sum()

        # Token examples
        if tokens:
            cluster_tokens = [tokens[i] for i in np.where(mask)[0][:20]]
            cluster_data["example_tokens"] = cluster_tokens

        analysis["clusters"][f"cluster_{c}"] = cluster_data

    # Overall metrics
    if labels is not None:
        # Average purity
        purities = [c["purity"] for c in analysis["clusters"].values() if "purity" in c]
        analysis["mean_purity"] = float(np.mean(purities)) if purities else None

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="sst2")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    config = checkpoint.get("config", {})

    # Load model
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
        return

    # Get dataloader
    print(f"Loading {args.domain} data...")
    if args.domain == "sst2":
        from torch.utils.data import DataLoader

        from src.data import SST2Dataset

        dataset = SST2Dataset(split="validation")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    else:
        print(f"Domain {args.domain} not yet supported for phase analysis")
        return

    # Extract representations
    print("Extracting token representations...")
    data = extract_token_representations(model, dataloader, device, args.max_samples)

    if len(data["phases"]) == 0:
        print("No representations extracted. Check model structure.")
        return

    print(f"Extracted {len(data['phases'])} token representations")

    # Cluster by phase
    print(f"Clustering into {args.n_clusters} clusters...")
    cluster_labels, cluster_centers = circular_kmeans(data["phases"], args.n_clusters)

    # Compute silhouette score
    X_2d = np.stack([np.cos(data["phases"]), np.sin(data["phases"])], axis=1)
    silhouette = silhouette_score(X_2d, cluster_labels)
    print(f"Silhouette score: {silhouette:.4f}")

    # Analyze clusters
    analysis = analyze_clusters(
        data["phases"],
        data["labels"],
        cluster_labels,
        data["tokens"],
    )
    analysis["silhouette_score"] = silhouette
    analysis["n_clusters"] = args.n_clusters
    analysis["n_tokens"] = len(data["phases"])

    # Visualizations
    print("Generating visualizations...")

    # Phase histogram by cluster
    plot_phase_histogram(
        data["phases"],
        output_dir / "phase_distribution.png",
        title="Phase Distribution (All Tokens)",
    )

    # Phase histogram colored by label
    if data["labels"] is not None:
        plot_phase_histogram(
            data["phases"],
            output_dir / "phase_by_label.png",
            labels=data["labels"],
            title="Phase Distribution by Class Label",
        )

    # Phase-magnitude scatter
    plot_magnitude_phase_scatter(
        data["magnitudes"],
        data["phases"],
        output_dir / "phase_magnitude_scatter.png",
        labels=cluster_labels,
        title="Magnitude vs Phase (colored by cluster)",
    )

    # Save results
    save_results_json(analysis, output_dir / "phase_clustering.json")

    # Report
    print("\n" + "=" * 60)
    print("PHASE CLUSTERING ANALYSIS")
    print("=" * 60)

    print(f"\nSilhouette score: {silhouette:.4f}")
    if silhouette > 0.5:
        print("  → Strong cluster structure in phase space")
    elif silhouette > 0.25:
        print("  → Moderate cluster structure")
    else:
        print("  → Weak/no cluster structure (phases may be unstructured)")

    if analysis.get("mean_purity"):
        print(f"\nMean cluster purity: {analysis['mean_purity']:.4f}")
        if analysis["mean_purity"] > 0.7:
            print("  → Phases correlate with labels (semantically meaningful)")
        elif analysis["mean_purity"] > 0.55:
            print("  → Weak correlation with labels")
        else:
            print("  → No correlation with labels (phases may encode other info)")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
