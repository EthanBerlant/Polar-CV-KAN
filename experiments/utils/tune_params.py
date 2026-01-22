import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.train import load_domain_module
from experiments.utils.param_counter import detailed_param_count, get_preset

# Targets
TARGETS = {
    "image": 696618,
    "audio": 543011,
    "timeseries": 617632,
    "nlp": 660000,  # Approximate target for backbone
}

PRESETS = {
    "image": "cifar10",
    "audio": "speech_commands",
    "timeseries": "etth1",
    "nlp": "sst2",
}


def get_model_params(domain, preset, d_complex, n_layers, kan_hidden=32):
    config = get_preset(preset)
    config.d_complex = d_complex
    config.n_layers = n_layers
    config.kan_hidden = kan_hidden

    module = load_domain_module(domain)
    try:
        model = module.create_model(config)

        # For NLP, we focus on backbone because embedding is fixed huge cost usually
        # But wait, TextLSTM baseline was 2.5M, 660K without embedding.
        # CVKAN embedding is 2 * vocab * d_complex.
        # If we match backbone params, total will differ if d_complex differs from baseline embed_dim.
        # Let's count BACKBONE + HEAD params for NLP matching.

        counts = detailed_param_count(model)
        if domain == "nlp":
            return counts["Backbone"] + counts["Head"]

        return counts["Total"]
    except Exception:
        return 0


def tune(domain, n_layers, start_d=32):
    target = TARGETS[domain]
    preset = PRESETS[domain]

    print(f"Tuning {domain} (n_layers={n_layers}) to target {target}...")

    best_d = start_d
    best_diff = float("inf")
    best_params = 0

    # Simple sweep
    for d in range(start_d, 300, 2):  # Step 2 for even numbers preference
        params = get_model_params(domain, preset, d, n_layers)
        diff = abs(params - target)

        if diff < best_diff:
            best_diff = diff
            best_d = d
            best_params = params
        # If diff getting worse and we have a valid match, stop
        elif params > target:
            break

    print(f"  Result: d={best_d} -> {best_params:,} params (Diff: {best_params - target:+d})")
    return best_d


def main():
    configs = [
        ("image", [2, 4, 8]),
        ("audio", [2, 4, 8]),
        ("timeseries", [2, 4, 8]),
        ("nlp", [1, 2, 4]),
    ]

    results = {}

    for domain, layers_list in configs:
        results[domain] = {}
        for n in layers_list:
            d = tune(domain, n)
            results[domain][n] = d

    print("\noptimized Configs:")
    print(results)


if __name__ == "__main__":
    main()
