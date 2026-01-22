import argparse
import subprocess

# Config definitions
# Tuned on 2026-01-21 to match baselines within ~5%
DOMAINS = {
    "image": {
        "preset": "cifar10",
        "baseline_params": "700K",
        "configs": {
            "wide_shallow": {"d_complex": 186, "n_layers": 2},
            "balanced": {"d_complex": 128, "n_layers": 4},
            "deep_narrow": {"d_complex": 62, "n_layers": 8},
        },
    },
    "audio": {
        "preset": "speech_commands",
        "baseline_params": "540K",
        "configs": {
            "wide_shallow": {"d_complex": 212, "n_layers": 2},
            "balanced": {"d_complex": 110, "n_layers": 4},
            "deep_narrow": {"d_complex": 40, "n_layers": 8},
        },
    },
    "timeseries": {
        "preset": "etth1",
        "baseline_params": "620K",
        "configs": {
            "wide_shallow": {"d_complex": 384, "n_layers": 2},
            "balanced": {"d_complex": 256, "n_layers": 4},
            "deep_narrow": {"d_complex": 128, "n_layers": 8},
        },
    },
    "nlp": {
        "preset": "sst2",
        "baseline_params": "660K (backbone)",
        "configs": {
            "wide_shallow": {"d_complex": 192, "n_layers": 1},
            "balanced": {"d_complex": 128, "n_layers": 2},
            "deep_narrow": {"d_complex": 64, "n_layers": 4},
        },
    },
}

# Ablation grid
ABLATIONS = {"skip": [False, True], "pooling": ["mean", "attention"]}


def run_command(cmd, dry_run=False):
    print(f"Running: {cmd}")
    if not dry_run:
        subprocess.check_call(cmd, shell=True)


def get_run_name(domain, config_name, skip, pooling, phase):
    return f"{domain}_{config_name}_skip{skip}_pool{pooling}_{phase}"


def main():
    parser = argparse.ArgumentParser(description="Run CVKAN Architecture Experiments")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["ablation", "extended"],
        help="Experiment phase",
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS.keys(), help="Domains to run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--subset_size", type=int, help="Use subset for faster training")
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size (useful for memory constraints)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision for reduced memory usage",
    )

    args = parser.parse_args()

    epochs = 10 if args.phase == "ablation" else 30

    for domain in args.domains:
        if domain not in DOMAINS:
            print(f"Skipping unknown domain: {domain}")
            continue

        print(f"\n{'='*50}\nStarting {domain} experiments ({args.phase})\n{'='*50}")
        domain_cfg = DOMAINS[domain]
        preset = domain_cfg["preset"]

        for config_name, params in domain_cfg["configs"].items():
            d_complex = params["d_complex"]
            n_layers = params["n_layers"]

            # Phase 1: Grid Search all ablations
            if args.phase == "ablation":
                for skip in ABLATIONS["skip"]:
                    for pool in ABLATIONS["pooling"]:
                        run_name = get_run_name(domain, config_name, skip, pool, "p1")

                        cmd = (
                            f"python experiments/train.py "
                            f"--domain {domain} "
                            f"--preset {preset} "
                            f"--d_complex {d_complex} "
                            f"--n_layers {n_layers} "
                            f"--pooling {pool} "
                            f"--run_name {run_name} "
                            f"--epochs {epochs} "
                        )

                        if skip:
                            cmd += " --skip_connections"

                        if args.subset_size:
                            cmd += f" --subset_size {args.subset_size}"

                        if args.batch_size:
                            cmd += f" --batch_size {args.batch_size}"

                        if args.amp:
                            cmd += " --amp"

                        try:
                            run_command(cmd, args.dry_run)
                        except subprocess.CalledProcessError as e:
                            print(f"Error running {run_name}: {e}")

            # Phase 2: Best only (Placeholder logic)
            elif args.phase == "extended":
                print("Phase 2 requires manual selection of best ablation settings.")
                print("Please run specific configurations manually based on Phase 1 results.")


if __name__ == "__main__":
    main()
