"""MLflow experiment logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MLflowLogger:
    """Thin wrapper around the MLflow tracking API.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Local directory or remote URI for MLflow tracking.
    """

    def __init__(self, experiment_name: str, tracking_uri: str | Path) -> None:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment_name)
        self._run: mlflow.ActiveRun | None = None

    def start(self, params: dict[str, Any] | None = None) -> None:
        """Start a new MLflow run and optionally log hyper-parameters."""
        self._run = mlflow.start_run()
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a dictionary of scalar metrics at *step*."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Upload a local file or directory as an MLflow artifact."""
        mlflow.log_artifact(str(path))

    def end(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        self._run = None
