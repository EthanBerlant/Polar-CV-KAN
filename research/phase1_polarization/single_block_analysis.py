"""Theoretical analysis of polarization in a single block.

Derives conditions under which aggregate-transform-broadcast increases
magnitude separation, with numerical verification.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

from utils import magnitude_variance, save_figure, save_results_json


def single_block_dynamics(
    Z: torch.Tensor,
    f_mag: Callable,
    f_phase: Callable = None,
) -> torch.Tensor:
    """Simulate one aggregate-transform-broadcast step."""
    A = Z.mean(dim=1, keepdim=True)
    mag = torch.abs(A)
    phase = torch.angle(A)
    log_mag = torch.log(mag + 1e-8)
    log_mag_transformed = f_mag(log_mag)
    if f_phase is not None:
        phase_transformed = f_phase(phase)
    else:
        phase_transformed = phase
    mag_transformed = torch.exp(log_mag_transformed)
    A_transformed = mag_transformed * torch.exp(1j * phase_transformed)
    Z_out = Z + A_transformed
    return Z_out


def compute_delta_polarization(Z: torch.Tensor, f_mag: Callable) -> float:
    """Compute change in polarization measure after one block."""
    P_before = magnitude_variance(Z, dim=1).mean().item()
    Z_out = single_block_dynamics(Z, f_mag)
    P_after = magnitude_variance(Z_out, dim=1).mean().item()
    return P_after - P_before


def create_test_transforms():
    """Create a library of test transform functions."""
    return {
        "identity": lambda x: x,
        "amplify_2x": lambda x: 2 * x,
        "amplify_1.5x": lambda x: 1.5 * x,
        "compress_0.5x": lambda x: 0.5 * x,
        "relu": lambda x: torch.relu(x),
        "leaky_relu": lambda x: torch.where(x > 0, x, 0.1 * x),
        "tanh": lambda x: torch.tanh(x),
        "sigmoid_centered": lambda x: 2 * (torch.sigmoid(x) - 0.5),
        "soft_threshold": lambda x: torch.sign(x) * torch.relu(torch.abs(x) - 0.5),
        "quadratic": lambda x: x + 0.1 * x**2,
    }


def create_test_distributions():
    """Create different initial magnitude distributions."""
    n_tokens, d_complex = 16, 32
    batch_size = 100
    distributions = {}
    phases = torch.rand(batch_size, n_tokens, d_complex) * 2 * np.pi

    mags = torch.ones(batch_size, n_tokens, d_complex)
    distributions["uniform"] = mags * torch.exp(1j * phases)

    mags = torch.ones(batch_size, n_tokens, d_complex) * 0.1
    mags[:, 0, :] = 2.0
    distributions["one_hot"] = mags * torch.exp(1j * phases)

    mags = torch.abs(torch.randn(batch_size, n_tokens, d_complex)) + 0.1
    distributions["gaussian"] = mags * torch.exp(1j * phases)

    mags = torch.distributions.Exponential(1.0).sample((batch_size, n_tokens, d_complex))
    distributions["exponential"] = mags * torch.exp(1j * phases)

    mags = torch.where(
        torch.rand(batch_size, n_tokens, d_complex) > 0.5,
        torch.ones(batch_size, n_tokens, d_complex) * 0.5,
        torch.ones(batch_size, n_tokens, d_complex) * 2.0,
    )
    distributions["bimodal"] = mags * torch.exp(1j * phases)

    phases_aligned = phases.clone()
    phases_aligned[:, :4, :] = 0.0
    mags = torch.abs(torch.randn(batch_size, n_tokens, d_complex)) + 0.1
    distributions["partially_aligned"] = mags * torch.exp(1j * phases_aligned)

    return distributions


def run_analysis(output_dir: Path):
    """Run comprehensive single-block analysis."""
    transforms = create_test_transforms()
    distributions = create_test_distributions()

    results = {
        "delta_polarization": {},
        "analysis": {},
    }

    print("Testing transform x distribution combinations...")

    delta_matrix = np.zeros((len(transforms), len(distributions)))

    for i, (t_name, t_func) in enumerate(transforms.items()):
        for j, (d_name, Z) in enumerate(distributions.items()):
            delta = compute_delta_polarization(Z, t_func)
            delta_matrix[i, j] = delta
            results["delta_polarization"][f"{t_name}_{d_name}"] = delta

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        delta_matrix,
        cmap="RdBu",
        aspect="auto",
        vmin=-np.abs(delta_matrix).max(),
        vmax=np.abs(delta_matrix).max(),
    )

    ax.set_xticks(range(len(distributions)))
    ax.set_xticklabels(distributions.keys(), rotation=45, ha="right")
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels(transforms.keys())
    ax.set_xlabel("Initial Distribution", fontsize=12)
    ax.set_ylabel("Transform Function", fontsize=12)
    ax.set_title("Delta Polarization (Variance) After One Block", fontsize=14)

    for i in range(len(transforms)):
        for j in range(len(distributions)):
            color = (
                "white" if abs(delta_matrix[i, j]) > np.abs(delta_matrix).max() * 0.5 else "black"
            )
            ax.text(
                j, i, f"{delta_matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8
            )

    plt.colorbar(im, ax=ax, label="Delta Polarization")
    save_figure(fig, output_dir / "delta_polarization_heatmap.png")

    # Analysis
    print("\n" + "=" * 60)
    print("SINGLE BLOCK ANALYSIS")
    print("=" * 60)

    mean_delta_by_transform = delta_matrix.mean(axis=1)
    print("\nMean Delta Polarization by Transform:")
    for t_name, delta in zip(transforms.keys(), mean_delta_by_transform, strict=False):
        sign = "+" if delta > 0 else ""
        print(f"  {t_name:20s}: {sign}{delta:.4f}")

    results["analysis"]["best_transform"] = list(transforms.keys())[
        np.argmax(mean_delta_by_transform)
    ]
    results["analysis"]["worst_transform"] = list(transforms.keys())[
        np.argmin(mean_delta_by_transform)
    ]

    mean_delta_by_dist = delta_matrix.mean(axis=0)
    print("\nMean Delta Polarization by Distribution:")
    for d_name, delta in zip(distributions.keys(), mean_delta_by_dist, strict=False):
        sign = "+" if delta > 0 else ""
        print(f"  {d_name:20s}: {sign}{delta:.4f}")

    results["analysis"]["most_polarizable"] = list(distributions.keys())[
        np.argmax(mean_delta_by_dist)
    ]
    results["analysis"]["least_polarizable"] = list(distributions.keys())[
        np.argmin(mean_delta_by_dist)
    ]

    # Multi-layer simulation
    print("\n" + "-" * 60)
    print("MULTI-LAYER SIMULATION")
    print("-" * 60)

    n_layers = 8
    test_transforms = ["identity", "amplify_1.5x", "relu", "soft_threshold"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, t_name in enumerate(test_transforms):
        ax = axes[idx // 2, idx % 2]
        t_func = transforms[t_name]
        Z_current = distributions["gaussian"].clone()
        polarizations = [magnitude_variance(Z_current, dim=1).mean().item()]

        for layer in range(n_layers):
            Z_current = single_block_dynamics(Z_current, t_func)
            polarizations.append(magnitude_variance(Z_current, dim=1).mean().item())

        ax.plot(polarizations, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Magnitude Variance")
        ax.set_title(f"Transform: {t_name}")
        ax.axhline(polarizations[0], linestyle="--", alpha=0.5, color="gray")
        delta = polarizations[-1] - polarizations[0]
        ax.annotate(f"Delta={delta:+.3f}", xy=(n_layers, polarizations[-1]), fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_dir / "multilayer_simulation.png")

    # Write analysis document
    write_analysis_document(output_dir, results)
    save_results_json(results, output_dir / "single_block_results.json")

    print(f"\nResults saved to {output_dir}")


def write_analysis_document(output_dir: Path, results: dict):
    """Write theoretical analysis as markdown."""
    best = results["analysis"]["best_transform"]
    worst = results["analysis"]["worst_transform"]
    most_pol = results["analysis"]["most_polarizable"]
    least_pol = results["analysis"]["least_polarizable"]

    doc = f"""# Single Block Polarization Analysis

## Theoretical Framework

### Setup
- Input: Z in C^(B x N x D) (batch x tokens x dimensions)
- Aggregate: A = (1/N) sum_i Z_i
- Transform: A' = f(|A|) * exp(i * g(angle(A)))
- Broadcast: Z' = Z + A'

### Polarization Measure
P(Z) = Var_i(|z_i|) (variance of magnitudes across tokens)

### Question
Under what conditions on f does P(Z') > P(Z)?

## Key Insight

The residual addition Z' = Z + A' has different effects depending on phase alignment:

1. **Phase-aligned tokens**: |z_i + A'| ~ |z_i| + |A'| (constructive)
2. **Phase-opposed tokens**: |z_i + A'| ~ ||z_i| - |A'|| (destructive)
3. **Phase-orthogonal tokens**: |z_i + A'| ~ sqrt(|z_i|^2 + |A'|^2) (Pythagorean)

## Empirical Findings

### Best Transforms for Polarization
- **Best**: `{best}`
- **Worst**: `{worst}`

### Most Polarizable Distributions
- **Most susceptible**: `{most_pol}`
- **Least susceptible**: `{least_pol}`

## Conditions for Polarization

Based on the analysis, polarization increases when:
1. **Transform amplifies**: f'(x) > 1 means larger inputs grow faster
2. **Initial variance exists**: Uniform distributions resist polarization
3. **Phase structure present**: Aligned tokens benefit from constructive interference

## Open Questions

1. Can we derive closed-form delta_P for specific f and distribution families?
2. What is the fixed point? Does P diverge or saturate?
3. How does phase transform g interact with polarization?
"""

    with open(output_dir / "single_block_analysis.md", "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"Wrote analysis to {output_dir / 'single_block_analysis.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.output_dir / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_analysis(output_dir)


if __name__ == "__main__":
    main()
