import os
import sqlite3

import pandas as pd


def get_results():
    if not os.path.exists("mlflow.db"):
        print("No mlflow.db found.")
        return

    conn = sqlite3.connect("mlflow.db")

    # Get all runs
    runs = pd.read_sql("SELECT run_uuid, name, status, start_time FROM runs", conn)

    # Filter for Phase 1 runs
    runs = runs[runs["name"].str.contains("_p1", na=False)]

    if runs.empty:
        print("No Phase 1 runs found.")
        return

    def parse_name(row):
        try:
            parts = row["name"].split("_")
            # image_balanced_skipTrue_poolattention_p1
            # 0: domain (image)
            # 1: config (balanced)
            # 2: skip (skipTrue)
            # 3: pool (poolattention)
            # 4: phase (p1)

            # Handle possible variations if needed
            if len(parts) >= 5:
                row["domain"] = parts[0]
                row["config"] = parts[1]
                row["skip"] = parts[2].replace("skip", "")
                row["pool"] = parts[3].replace("pool", "")
            return row
        except:
            return row

    runs = runs.apply(parse_name, axis=1)

    metrics = pd.read_sql(
        "SELECT run_uuid, key, value FROM metrics WHERE key = 'val_accuracy'", conn
    )

    if metrics.empty:
        print("No val_accuracy metrics found for p1 runs.")
        # Try finding ANY metric
        metrics = pd.read_sql("SELECT run_uuid, key, value FROM metrics", conn)

    # Max metric per run
    metrics_max = metrics.groupby("run_uuid")["value"].max().reset_index()
    metrics_max.columns = ["run_uuid", "val_accuracy"]

    df = runs.merge(metrics_max, on="run_uuid", how="left")

    # Sort
    if "domain" in df.columns:
        df = df.sort_values(by=["domain", "config", "val_accuracy"], ascending=[True, True, False])

        print("\nPhase 1 Results:")
        cols = ["name", "status", "domain", "config", "skip", "pool", "val_accuracy"]
        # Filter existing cols
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string())

        # Best per config
        print("\nBest Per Config:")
        best = df.sort_values("val_accuracy", ascending=False).groupby(["domain", "config"]).head(1)
        print(best[cols].to_string())
    else:
        print(df.to_string())


if __name__ == "__main__":
    get_results()
