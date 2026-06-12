"""PSNR / SSIM / MSE metric utilities."""

from __future__ import annotations

import torch
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def _to_scalar(metric_value: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> float:
    """Convert a metric output to a Python float.

    Some metric APIs may return either a scalar tensor or a tuple where the
    first element is the scalar value and the second element is an auxiliary
    output (for example, a full-image map).
    """
    if isinstance(metric_value, tuple):
        metric_value = metric_value[0]
    return float(metric_value.item())


@torch.inference_mode()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return average PSNR (dB) over a batch.

    Inputs are expected in ``[0, 1]`` with shape ``(B, C, H, W)``.
    """
    metric: PeakSignalNoiseRatio = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return float(metric(pred, target).item())


@torch.inference_mode()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return average SSIM over a batch.

    Inputs are expected in ``[0, 1]`` with shape ``(B, C, H, W)``.
    """
    metric: StructuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    result = metric(pred, target)
    return float(result.item())


@torch.inference_mode()
def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean MSE over a batch."""
    return float(torch.mean((pred - target) ** 2).item())


class RunningMetrics:
    """Accumulate PSNR, SSIM, and MSE across multiple batches.

    Example::

        rm = RunningMetrics(device)
        for pred, target in loader:
            rm.update(pred, target)
        print(rm.compute())
    """

    def __init__(self, device: torch.device) -> None:
        self._psnr: PeakSignalNoiseRatio = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self._ssim: StructuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(
            data_range=1.0, return_full_image=False
        ).to(device)
        self._mse: torchmetrics.MeanSquaredError = torchmetrics.MeanSquaredError().to(device)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add one batch to the running accumulators."""
        self._psnr.update(pred, target)
        self._ssim.update(pred, target)
        self._mse.update(pred.flatten(), target.flatten())

    def compute(self) -> dict[str, float]:
        """Return accumulated metrics and reset internal state.

        ``StructuralSimilarityIndexMeasure.compute()`` is typed as returning
        either ``torch.Tensor`` or ``tuple[torch.Tensor, torch.Tensor]``.
        We normalize that union with ``_to_scalar`` so static type checkers can
        verify this method without ignores.
        """
        results = {
            "psnr": float(self._psnr.compute().item()),
            "ssim": _to_scalar(self._ssim.compute()),
            "mse": float(self._mse.compute().item()),
        }
        self.reset()
        return results

    def reset(self) -> None:
        """Reset all accumulators."""
        self._psnr.reset()
        self._ssim.reset()
        self._mse.reset()
