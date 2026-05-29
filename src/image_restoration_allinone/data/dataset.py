"""Dataset utilities for paired image restoration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from torch.utils.data import Dataset


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


def discover_pairs_separate(directory: Path) -> list[tuple[Path, Path]]:
    """Discover paired images from ``degre/`` and ``clean/`` sub-directories."""
    degre_dir = directory / "degre"
    clean_dir = directory / "clean"
    if not degre_dir.is_dir() or not clean_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for degraded_path in sorted(degre_dir.iterdir()):
        clean_path = clean_dir / degraded_path.name
        if clean_path.exists():
            pairs.append((degraded_path, clean_path))
    return pairs


def discover_pairs(root: Path, split: str = "train") -> list[tuple[Path, Path]]:
    """Return a list of ``(degraded_path, clean_path)`` pairs.

    Supports the three layouts described in ``docs/data_structure.md``:

    * **Case 1** - flat directory with ``_real`` / ``_mean`` files.
    * **Case 2** - ``train/`` or ``val/`` sub-directory with ``_real`` / ``_mean`` files.
    * **Case 3** - separate ``degre/`` and ``clean/`` sub-directories.
    """
    split_dir = root / split
    if split_dir.is_dir():
        pairs = discover_pairs_keyword(split_dir)
        if pairs:
            return pairs
        pairs = discover_pairs_separate(split_dir)
        if pairs:
            return pairs

    # Fall back to root-level search
    pairs = discover_pairs_keyword(root)
    if pairs:
        return pairs
    return discover_pairs_separate(root)


class PairedRestorationDataset(Dataset[dict[str, npt.NDArray[np.float32]]]):
    """Dataset that returns ``(degraded, clean)`` image pairs.

    Attributes:
        pairs: List of ``(degraded_path, clean_path)`` tuples.
        transform: Optional callable applied jointly to both images.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: object = None,
    ) -> None:
        self.pairs = discover_pairs(root, split)
        if not self.pairs:
            raise FileNotFoundError(
                f"No paired images found under '{root}' for split='{split}'. "
                "Check docs/data_structure.md for supported layouts."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, npt.NDArray[np.float32]]:
        degraded_path, clean_path = self.pairs[index]
        degraded = _load_image_rgb(degraded_path)
        clean = _load_image_rgb(clean_path)

        if self.transform is not None:
            result = self.transform(image=degraded, clean=clean)  # pyright: ignore[reportCallIssue]
            degraded = result["image"]
            clean = result["clean"]

        return {"degraded": degraded, "clean": clean}
