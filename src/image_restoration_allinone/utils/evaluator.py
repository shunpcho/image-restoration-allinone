"""Evaluator: run validation and compute loss, MSE, PSNR, SSIM."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_restoration_allinone.utils.metrics import RunningMetrics


class Evaluator:
    """Run model inference on a validation set and aggregate metrics.

    Recorded metrics per validation run:

    * ``val/loss``  - mean total loss (using the same criterion as training)
    * ``val/mse``   - mean pixel-wise MSE
    * ``val/psnr``  - mean PSNR (dB)
    * ``val/ssim``  - mean SSIM

    Args:
        model: The restoration network (in eval mode).
        criterion: Composite loss (returns ``(total, components)``).
        val_loader: Validation DataLoader (batch_size=1 recommended).
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        val_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self._metrics = RunningMetrics(device)

    @torch.inference_mode()
    def run(self) -> dict[str, float]:
        """Evaluate the model and return a metrics dictionary.

        Returns:
            Dict with keys ``val/loss``, ``val/mse``, ``val/psnr``, ``val/ssim``.
        """
        self.model.eval()
        self._metrics.reset()

        total_loss_sum = 0.0
        num_batches = 0

        for batch in self.val_loader:
            degraded: torch.Tensor = batch["degraded"].to(self.device)
            clean: torch.Tensor = batch["clean"].to(self.device)

            restored = self.model(degraded)
            loss, _ = self.criterion(restored, clean)

            total_loss_sum += loss.item()
            num_batches += 1
            self._metrics.update(restored, clean)

        image_metrics = self._metrics.compute()
        mean_loss = total_loss_sum / max(num_batches, 1)

        return {
            "val/loss": mean_loss,
            "val/mse": image_metrics["mse"],
            "val/psnr": image_metrics["psnr"],
            "val/ssim": image_metrics["ssim"],
        }
