"""
Analyze completed experiments and generate commands for remaining ones.
"""

from pathlib import Path

# Expected experiment grid (copied from run_architecture_experiments.py)
DOMAINS = {
    "image": {
        "preset": "cifar10",
        "configs": {
            "wide_shallow": {"d_complex": 186, "n_layers": 2},
            "balanced": {"d_complex": 128, "n_layers": 4},
            "deep_narrow": {"d_complex": 62, "n_layers": 8},
        },
    },
    "audio": {
        "preset": "speech_commands",
        "configs": {
            "wide_shallow": {"d_complex": 212, "n_layers": 2},
            "balanced": {"d_complex": 110, "n_layers": 4},
            "deep_narrow": {"d_complex": 40, "n_layers": 8},
        },
    },
    "timeseries": {
        "preset": "etth1",
        "configs": {
            "wide_shallow": {"d_complex": 384, "n_layers": 2},
            "balanced": {"d_complex": 256, "n_layers": 4},
            "deep_narrow": {"d_complex": 128, "n_layers": 8},
        },
    },
    "nlp": {
        "preset": "sst2",
        "configs": {
            "wide_shallow": {"d_complex": 192, "n_layers": 1},
            "balanced": {"d_complex": 128, "n_layers": 2},
            "deep_narrow": {"d_complex": 64, "n_layers": 4},
        },
    },
}

ABLATIONS = {"skip": [False, True], "pooling": ["mean", "attention"]}


def get_run_name(domain, config_name, skip, pooling):
    return f"{domain}_{config_name}_skip{skip}_pool{pooling}_p1"


def check_completed(outputs_dir):
    """Check which experiments have completed (have config.json)."""
    completed = set()

    for domain in DOMAINS:
        domain_dir = Path(outputs_dir) / domain
        if not domain_dir.exists():
            continue

        for run_dir in domain_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "config.json").exists():
                # Extract base run name (without timestamp)
                name = run_dir.name
                # Pattern: {domain}_{config}_{skip}_{pool}_p1_{timestamp}
                parts = name.rsplit("_", 2)  # Split off timestamp parts
                if len(parts) >= 2:
                    base_name = parts[0]  # Everything before the YYYYMMDD_HHMMSS
                    completed.add(base_name)

    return completed


def generate_remaining_commands(outputs_dir, epochs=10):
    """Generate commands for experiments that haven't completed."""
    completed = check_completed(outputs_dir)

    print("=== EXPERIMENT STATUS ===\n")

    all_expected = []
    missing = []

    for domain, domain_cfg in DOMAINS.items():
        preset = domain_cfg["preset"]
        for config_name, params in domain_cfg["configs"].items():
            for skip in ABLATIONS["skip"]:
                for pool in ABLATIONS["pooling"]:
                    run_name = get_run_name(domain, config_name, skip, pool)
                    all_expected.append(run_name)

                    if run_name not in completed:
                        missing.append(
                            {
                                "run_name": run_name,
                                "domain": domain,
                                "preset": preset,
                                "d_complex": params["d_complex"],
                                "n_layers": params["n_layers"],
                                "skip": skip,
                                "pool": pool,
                            }
                        )

    print(f"Total expected experiments: {len(all_expected)}")
    print(f"Completed experiments: {len(completed)}")
    print(f"Missing experiments: {len(missing)}")

    print(f"\n=== COMPLETED ({len(completed)}) ===")
    for name in sorted(completed):
        print(f"  ✓ {name}")

    print(f"\n=== MISSING ({len(missing)}) ===")
    for exp in missing:
        print(f"  ✗ {exp['run_name']}")

    if missing:
        print("\n=== COMMANDS TO RUN REMAINING ===\n")
        for exp in missing:
            cmd = (
                f"python experiments/train.py "
                f"--domain {exp['domain']} "
                f"--preset {exp['preset']} "
                f"--d_complex {exp['d_complex']} "
                f"--n_layers {exp['n_layers']} "
                f"--pooling {exp['pool']} "
                f"--run_name {exp['run_name']} "
                f"--epochs {epochs}"
            )
            if exp["skip"]:
                cmd += " --skip_connections"
            print(cmd)

    return completed, missing


if __name__ == "__main__":
    completed, missing = generate_remaining_commands("./outputs")

    # Also save missing commands to a file
    if missing:
        with open("experiments/remaining_experiments.txt", "w") as f:
            f.write(f"# {len(missing)} experiments remaining\n")
            f.write(f"# {len(completed)} experiments completed\n\n")
            for exp in missing:
                cmd = (
                    f"python experiments/train.py "
                    f"--domain {exp['domain']} "
                    f"--preset {exp['preset']} "
                    f"--d_complex {exp['d_complex']} "
                    f"--n_layers {exp['n_layers']} "
                    f"--pooling {exp['pool']} "
                    f"--run_name {exp['run_name']} "
                    f"--epochs 10"
                )
                if exp["skip"]:
                    cmd += " --skip_connections"
                f.write(cmd + "\n")
        print("\nSaved commands to experiments/remaining_experiments.txt")
