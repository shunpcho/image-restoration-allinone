"""Visualizer: save side-by-side comparison images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


def tensor_to_uint8(t: torch.Tensor) -> npt.NDArray[np.uint8]:
    """Convert a (C, H, W) float tensor in [0, 1] to a (H, W, C) uint8 array."""
    arr = t.detach().cpu().float().permute(1, 2, 0).numpy()
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def save_comparison(
    degraded: torch.Tensor,
    restored: torch.Tensor,
    clean: torch.Tensor,
    save_path: Path,
) -> None:
    """Save a side-by-side PNG: degraded | restored | clean.

    All tensors must be (C, H, W) float in [0, 1].
    """
    imgs = [tensor_to_uint8(t) for t in (degraded, restored, clean)]
    combined = np.concatenate(imgs, axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(save_path)
