"""Composite loss function for image restoration.

MSE is the default loss. Additional losses can be combined by specifying
``losses`` in :class:`~image_restoration_allinone.configs.config.LossConfig`.

Supported loss names
--------------------
* ``"mse"``         - Mean Squared Error (default)
* ``"l1"``          - Mean Absolute Error
* ``"charbonnier"`` - Charbonnier (smooth L1 variant)
* ``"ssim"``        - 1 - SSIM (uses pytorch-msssim)
* ``"perceptual"``  - VGG-based perceptual loss (uses lpips)
"""

from __future__ import annotations

import torch
from torch import nn

from image_restoration_allinone.configs.config_class import LossConfig

# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: sqrt((x - y)^2 + eps^2)."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps**2))


class SSIMLoss(nn.Module):
    """1 - SSIM loss using pytorch-msssim."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from pytorch_msssim import ssim

            self._ssim_fn = ssim
        except ImportError as exc:
            raise ImportError("Install pytorch-msssim to use SSIMLoss.") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim_fn(pred, target, data_range=1.0, size_average=True)


class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss (VGG backbone)."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips

            self._lpips = lpips.LPIPS(net="vgg")
        except ImportError as exc:
            raise ImportError("Install lpips to use PerceptualLoss.") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self._lpips(pred_scaled, target_scaled).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_loss_fn(name: str) -> nn.Module:
    match name:
        case "mse":
            return nn.MSELoss()
        case "l1":
            return nn.L1Loss()
        case "charbonnier":
            return CharbonnierLoss()
        case "ssim":
            return SSIMLoss()
        case "perceptual":
            return PerceptualLoss()
        case _:
            raise ValueError(f"Unknown loss name: '{name}'")


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


class LossComposer(nn.Module):
    """Weighted sum of one or more loss functions.

    Example::

        cfg = LossConfig(losses={"mse": 1.0, "ssim": 0.1})
        criterion = LossComposer(cfg)
        total, components = criterion(pred, target)
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self._names: list[str] = list(cfg.losses.keys())
        self._weights: list[float] = list(cfg.losses.values())
        self._fns = nn.ModuleList([_build_loss_fn(name) for name in self._names])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss and per-component values.

        Returns:
            A ``(total_loss, components)`` tuple where ``components`` maps
            each loss name to its scalar tensor.
        """
        components: dict[str, torch.Tensor] = {}
        total: torch.Tensor = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for name, weight, fn in zip(self._names, self._weights, self._fns, strict=True):
            value: torch.Tensor = fn(pred, target)
            components[name] = value
            total += weight * value
        return total, components
