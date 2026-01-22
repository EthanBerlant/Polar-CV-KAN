"""Analyze experiment results from outputs directory."""

import json
from pathlib import Path

results_dir = Path("outputs")
all_results = []

for results_file in results_dir.rglob("results.json"):
    try:
        with open(results_file) as f:
            data = json.load(f)

        # Skip aggregate results files
        if "experiments" in data or "comprehensive" in str(results_file):
            if "domain" not in data:
                continue

        domain = data.get("domain", "unknown")
        config = data.get("config", {})

        # Handle different result structures
        if "best_val_accuracy" in data:
            best_val = data["best_val_accuracy"]
            metric_type = "acc"
        elif "best_val_mse" in data:
            best_val = data["best_val_mse"]
            metric_type = "mse"
        elif "best_metric" in data:
            best_val = data["best_metric"]
            metric_type = "metric"
        else:
            best_val = None
            metric_type = "N/A"

        test_results = data.get("test_results", {})
        test_acc = test_results.get("accuracy", test_results.get("mse", None))

        n_params = data.get("n_params", config.get("n_params", None))
        dataset = config.get("dataset", config.get("dataset_name", "unknown"))
        d_complex = config.get("d_complex", "N/A")
        n_layers = config.get("n_layers", "N/A")
        pooling = config.get("pooling", "N/A")
        epochs_run = len(data.get("history", []))
        total_time = data.get("total_time_seconds", 0)

        rel_path = str(results_file.relative_to(results_dir))

        all_results.append(
            {
                "path": rel_path,
                "domain": domain,
                "dataset": dataset,
                "d_complex": d_complex,
                "n_layers": n_layers,
                "pooling": pooling,
                "n_params": n_params,
                "best_val": best_val,
                "metric_type": metric_type,
                "test_metric": test_acc,
                "epochs": epochs_run,
                "time_s": round(total_time, 1) if total_time else 0,
            }
        )
    except Exception as e:
        print(f"Error reading {results_file}: {e}")

# Sort by domain then dataset
all_results.sort(key=lambda x: (x["domain"], x["dataset"], -(x.get("best_val") or 0)))

# Print summary table
print(f"Total experiments: {len(all_results)}\n")

# Group by domain
domains = {}
for r in all_results:
    if r["domain"] not in domains:
        domains[r["domain"]] = []
    domains[r["domain"]].append(r)

for domain, results in sorted(domains.items()):
    print(f"\n{'='*100}")
    print(f" {domain.upper()} DOMAIN - {len(results)} experiments")
    print(f"{'='*100}")
    print(
        f"{'Dataset':<18} {'d_cplx':<7} {'L':<3} {'Pool':<10} {'Params':<10} {'Val':<12} {'Test':<12} {'Ep':<4} {'Time':<8}"
    )
    print("-" * 100)

    for r in results:
        val_str = f"{r['best_val']:.2f}" if r["best_val"] else "N/A"
        test_str = f"{r['test_metric']:.2f}" if r["test_metric"] else "N/A"
        if r["n_params"]:
            params_str = (
                f"{r['n_params']/1000:.0f}K" if r["n_params"] >= 1000 else str(r["n_params"])
            )
        else:
            params_str = "N/A"
        time_str = f"{r['time_s']/60:.1f}m" if r["time_s"] > 60 else f"{r['time_s']}s"

        print(
            f"{r['dataset']:<18} {str(r['d_complex']):<7} {str(r['n_layers']):<3} {str(r['pooling']):<10} {params_str:<10} {val_str:<12} {test_str:<12} {r['epochs']:<4} {time_str:<8}"
        )

# Summary stats per domain
print(f"\n{'='*100}")
print(" SUMMARY BY DOMAIN")
print(f"{'='*100}")
for domain, results in sorted(domains.items()):
    valid_vals = [r["best_val"] for r in results if r["best_val"] is not None]
    if valid_vals:
        avg_val = sum(valid_vals) / len(valid_vals)
        max_val = max(valid_vals)
        print(f"{domain:<12}: {len(results)} runs, Best: {max_val:.2f}, Avg: {avg_val:.2f}")
