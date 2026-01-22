"""
Summarize MLflow experiment results.
"""

import pandas as pd
from mlflow.tracking import MlflowClient


def summarize_experiments(experiment_names):
    client = MlflowClient()
    all_results = []

    for exp_name in experiment_names:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            print(f"Experiment {exp_name} not found.")
            continue

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
        )

        for run in runs:
            params = run.data.params
            metrics = run.data.metrics

            # Find best metric (accuracy for classification, mse for regression)
            if params.get("domain") == "timeseries":
                metric_key = "val_mse" if "val_mse" in metrics else "mse"
                score = metrics.get(metric_key, 0.0)
            else:
                metric_key = "val_accuracy" if "val_accuracy" in metrics else "accuracy"
                score = metrics.get(metric_key, 0.0)

            res = {
                "experiment": exp_name,
                "run_name": run.info.run_name,
                "domain": params.get("domain", "unknown"),
                "d_complex": params.get("d_complex", "unknown"),
                "n_layers": params.get("n_layers", "unknown"),
                "pooling": params.get("pooling", "unknown"),
                "skip": params.get("skip_connections", "False"),
                "score": score,
                "metric": "mse" if params.get("domain") == "timeseries" else "acc",
                "status": run.info.status,
            }
            all_results.append(res)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    df = summarize_experiments(["cvkan_image", "cvkan_audio", "cvkan_timeseries", "cvkan_nlp"])
    if not df.empty:
        # Group by domain, config and pooling to see patterns
        summary = df.sort_values(["domain", "score"], ascending=[True, False])
        print(summary.to_string(index=False))

        # Save to markdown compatible format
        summary.to_csv("experiments/summary_results.csv", index=False)
    else:
        print("No results found.")
