"""Dataset utilities for paired image restoration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset

from image_restoration_allinone.data.transforms import build_default_transform


def _load_image_rgb(path: Path) -> npt.NDArray[np.float32]:
    """Load an image from *path* and return a float32 array in [0, 1] (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def discover_pairs_keyword(directory: Path) -> list[tuple[Path, Path]]:
    """Discover ``_real`` / ``_mean`` paired images in *directory*.

    A file named ``foo_real.png`` is matched with ``foo_mean.png``.
    """
    pairs: list[tuple[Path, Path]] = []
    for degraded_path in sorted(directory.iterdir()):
        if "_real" not in degraded_path.stem:
            continue
        clean_stem = degraded_path.stem.replace("_real", "_mean")
        clean_path = degraded_path.with_name(clean_stem + degraded_path.suffix)
        if clean_path.exists():
            pairs.append((degraded_path, clean_path))
    return pairs


def discover_pairs_separate(directory: Path, lq_name: str = "LQ", gt_name: str = "GT") -> list[tuple[Path, Path]]:
    """Discover paired images from *lq_name/* and *gt_name/* sub-directories.

    Args:
        directory: Parent directory that contains the LQ and GT sub-directories.
        lq_name: Name of the sub-directory holding low-quality (degraded) images.
        gt_name: Name of the sub-directory holding ground-truth (clean) images.
    """
    lq_dir = directory / lq_name
    gt_dir = directory / gt_name
    if not lq_dir.is_dir() or not gt_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for lq_path in sorted(lq_dir.iterdir()):
        gt_path = gt_dir / lq_path.name
        if gt_path.exists():
            pairs.append((lq_path, gt_path))
    return pairs


def discover_pairs_category(
    root: Path,
    lq_name: str = "LQ",
    gt_name: str = "GT",
) -> list[tuple[Path, Path]]:
    """Discover pairs from category sub-directories, each containing *lq_name/* and *gt_name/*.

    Supports structures like::

        root/
        ├── Blur/
        │   ├── LQ/
        │   └── GT/
        └── Haze/
            ├── LQ/
            └── GT/
    """
    pairs: list[tuple[Path, Path]] = []
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir():
            pairs.extend(discover_pairs_separate(subdir, lq_name, gt_name))
    return pairs


def discover_pairs(
    root: Path,
    split: str = "train",
    lq_name: str = "LQ",
    gt_name: str = "GT",
) -> list[tuple[Path, Path]]:
    """Return a list of ``(degraded_path, clean_path)`` pairs.

    Supports the following layouts:

    * **Case 1** - flat directory with ``_real`` / ``_mean`` files.
    * **Case 2** - ``train/`` or ``val/`` sub-directory with ``_real`` / ``_mean`` files.
    * **Case 3** - separate *lq_name/* and *gt_name/* sub-directories.
    * **Case 4** - category sub-directories each containing *lq_name/* and *gt_name/*.

    Args:
        root: Dataset root directory.
        split: Data split sub-directory name (e.g. ``"train"`` or ``"val"``).
        lq_name: Sub-directory name for low-quality images (default: ``"LQ"``).
        gt_name: Sub-directory name for ground-truth images (default: ``"GT"``).
    """
    split_dir = root / split
    if split_dir.is_dir():
        pairs = discover_pairs_keyword(split_dir)
        if pairs:
            return pairs
        pairs = discover_pairs_separate(split_dir, lq_name, gt_name)
        if pairs:
            return pairs
        pairs = discover_pairs_category(split_dir, lq_name, gt_name)
        if pairs:
            return pairs

    # Fall back to root-level search
    pairs = discover_pairs_keyword(root)
    if pairs:
        return pairs
    pairs = discover_pairs_separate(root, lq_name, gt_name)
    if pairs:
        return pairs
    return discover_pairs_category(root, lq_name, gt_name)


class PairedRestorationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that returns ``(degraded, clean)`` image pairs.

    Attributes:
        pairs: List of ``(degraded_path, clean_path)`` tuples.
        transform: Optional callable applied jointly to both images.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Callable[..., dict[str, Any]] | None = None,
        lq_dir_name: str = "LQ",
        gt_dir_name: str = "GT",
    ) -> None:
        self.pairs = discover_pairs(root, split, lq_dir_name, gt_dir_name)
        if not self.pairs:
            raise FileNotFoundError(
                f"No paired images found under '{root}' for split='{split}'. "
                "Check docs/data_structure.md for supported layouts."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        degraded_path, clean_path = self.pairs[index]
        degraded = _load_image_rgb(degraded_path)
        clean = _load_image_rgb(clean_path)

        if self.transform is not None:
            result = self.transform(image=degraded, clean=clean)
            degraded = result["image"]
            clean = result["clean"]
        else:
            # Transform numpy arrays to torch tensors if no transform is provided
            transform = build_default_transform()
            result = transform(image=degraded, clean=clean)
            degraded = result["image"]
            clean = result["clean"]

        return {"degraded": degraded, "clean": clean}
