"""
Experiment tracking with MLflow.

This module provides a unified interface for tracking experiments,
logging metrics, parameters, and artifacts. It wraps MLflow with
sensible defaults for research workflows.

Usage:
    from src.tracking import ExperimentTracker

    with ExperimentTracker("my-experiment", run_name="baseline_v1") as tracker:
        tracker.log_params({"lr": 0.001, "batch_size": 32})

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, loader)
            tracker.log_metrics({"train_loss": train_loss}, step=epoch)

        tracker.log_model(model, "final_model")
"""

import json
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn


class ExperimentTracker:
    """
    Unified experiment tracking with MLflow backend.

    Features:
    - Automatic experiment creation if doesn't exist
    - Context manager for clean run lifecycle
    - Simplified metric/param logging
    - Model state dict saving

    Example:
        with ExperimentTracker("sst2-experiments", "emergent_d64") as t:
            t.log_params({"d_complex": 64, "n_layers": 2})
            for epoch in range(10):
                loss = train(model)
                t.log_metrics({"loss": loss}, step=epoch)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict | None = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment (creates if needed)
            run_name: Optional name for this specific run
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
            tags: Optional dict of tags to attach to the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._run = None
        self._run_id = None

    def __enter__(self) -> "ExperimentTracker":
        """Start MLflow run."""
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self._run_id = self._run.info.run_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if exc_type is not None:
            # Log that run failed
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        return False  # Don't suppress exceptions

    @property
    def run_id(self) -> str | None:
        """Get current run ID."""
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dict of parameter names to values
        """
        # MLflow has limits on param value length, so convert complex objects
        cleaned = {}
        for k, v in params.items():
            if isinstance(v, list | dict):
                cleaned[k] = json.dumps(v)
            else:
                cleaned[k] = v
        mlflow.log_params(cleaned)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics for the current step.

        Args:
            metrics: Dict of metric names to values
            step: Optional step number (epoch, batch, etc.)
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: nn.Module,
        name: str = "model",
        save_dir: str | Path | None = None,
    ) -> Path:
        """
        Save model state dict and log as artifact.

        Args:
            model: PyTorch model
            name: Name for the model artifact
            save_dir: Optional directory to save to (default: temp)

        Returns:
            Path to saved model file
        """
        if save_dir is None:
            save_dir = Path("outputs") / "models"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{name}_{self._run_id[:8]}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        return model_path

    def log_artifact(self, path: str | Path) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(str(path))

    def log_artifacts(self, dir_path: str | Path) -> None:
        """Log all files in a directory as artifacts."""
        mlflow.log_artifacts(str(dir_path))

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)


def get_or_create_experiment(name: str) -> str:
    """
    Get experiment ID, creating if it doesn't exist.

    Args:
        name: Experiment name

    Returns:
        Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id


def list_runs(experiment_name: str, max_results: int = 100) -> list[dict]:
    """
    List runs for an experiment.

    Args:
        experiment_name: Name of experiment
        max_results: Maximum runs to return

    Returns:
        List of run info dicts
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
    )
    return runs.to_dict("records")
