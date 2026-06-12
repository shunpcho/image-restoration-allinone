"""Dataset utilities for paired image restoration."""

from __future__ import annotations

import random
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
        if not subdir.is_dir():
            continue
        lq_dir = subdir / lq_name
        gt_dir = subdir / gt_name
        if not lq_dir.is_dir() or not gt_dir.is_dir():
            continue
        for lq_path in sorted(lq_dir.iterdir()):
            gt_path = gt_dir / lq_path.name
            if gt_path.exists():
                pairs.append((lq_path, gt_path))
    return pairs


def discover_pairs(
    root: Path,
    split: str = "train",
    lq_name: str = "LQ",
    gt_name: str = "GT",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[tuple[Path, Path]]:
    """Return a list of ``(degraded_path, clean_path)`` pairs for the given split.

    Discovers all pairs from category sub-directories (each containing *lq_name/* and
    *gt_name/*), shuffles them with *seed*, then splits into train / val by *val_ratio*.

    Args:
        root: Dataset root directory.
        split: One of ``"train"`` or ``"val"``.
        lq_name: Sub-directory name for low-quality images (default: ``"LQ"``).
        gt_name: Sub-directory name for ground-truth images (default: ``"GT"``).
        val_ratio: Fraction of all pairs reserved for validation (default: ``0.1``).
        seed: Random seed for reproducible shuffling (default: ``42``).
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    all_pairs = list(discover_pairs_category(root, lq_name, gt_name))
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_val = int(len(all_pairs) * val_ratio)
    if split == "val":
        return all_pairs[:n_val]
    return all_pairs[n_val:]


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
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.pairs = discover_pairs(
            root=root,
            split=split,
            lq_name=lq_dir_name,
            gt_name=gt_dir_name,
            val_ratio=val_ratio,
            seed=seed,
        )
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
