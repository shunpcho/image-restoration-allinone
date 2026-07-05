"""MLflow experiment logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torchvision

from image_restoration_allinone.configs.config import LoggingConfig


class MLflowLogger:
    """Thin wrapper around the MLflow tracking API.

    Args:
        log_cfg: Logging configuration.
    """

    def __init__(self, log_cfg: LoggingConfig) -> None:
        mlflow.set_tracking_uri(str(log_cfg.log_dir))
        mlflow.set_experiment(log_cfg.experiment_name)
        self._run: mlflow.ActiveRun | None = None
        self.img_limit = log_cfg.log_img_limit

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

    def log_imgs(self, cleans: torch.Tensor, noisys: torch.Tensor, outputs: torch.Tensor, step: int) -> None:
        """Log images to MLflow. Compare clean, noisy, and output images in a grid format.

        Args:
            cleans: Clean images tensor of shape (B, C, H, W).
            noisys: Noisy images tensor of shape (B, C, H, W).
            outputs: Output images tensor of shape (B, C, H, W).
            step: Current training step.
        """
        clean_noisy_output = torch.cat(
            (cleans[: self.img_limit], noisys[: self.img_limit], outputs[: self.img_limit]), dim=0
        )
        c_n_p_grid = torchvision.utils.make_grid(
            clean_noisy_output, padding=0, nrow=cleans.size(0), normalize=True, scale_each=True
        )

        """Transform the grid image to array and normalize to uint8 for MLflow logging."""
        c_n_p_grid = c_n_p_grid.cpu().numpy()
        c_n_p_grid = np.transpose(c_n_p_grid, (1, 2, 0))  # (H, W, C)

        mlflow.log_image(c_n_p_grid, f"clean_noisy_output_{step}.png")

    def end(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        self._run = None
