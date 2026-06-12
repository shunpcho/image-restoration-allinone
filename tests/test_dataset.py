"""Unit tests for dataset discovery and PairedRestorationDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_restoration_allinone.data.dataset import (
    discover_pairs,
    discover_pairs_category,
    PairedRestorationDataset,
)


def _write_rgb(path: Path, value: tuple[int, int, int] = (128, 64, 32)) -> None:
    img = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8))
    img.save(path)


def _setup_category(root: Path, n_categories: int = 1, n_images: int = 1) -> None:
    """Create a Case-4 directory tree with *n_categories* categories, each with *n_images* images."""
    for cat_idx in range(n_categories):
        for img_idx in range(n_images):
            (root / f"cat{cat_idx}" / "LQ").mkdir(parents=True, exist_ok=True)
            (root / f"cat{cat_idx}" / "GT").mkdir(parents=True, exist_ok=True)
            _write_rgb(root / f"cat{cat_idx}" / "LQ" / f"img{img_idx}.png")
            _write_rgb(root / f"cat{cat_idx}" / "GT" / f"img{img_idx}.png")


# ---------------------------------------------------------------------------
# discover_pairs_category
# ---------------------------------------------------------------------------


class TestDiscoverPairsCategory:
    def test_finds_pairs_in_category_subdirs(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=2)
        pairs = discover_pairs_category(tmp_path)
        assert len(pairs) == 2

    def test_custom_dir_names(self, tmp_path: Path) -> None:
        (tmp_path / "cat" / "degre").mkdir(parents=True)
        (tmp_path / "cat" / "clean").mkdir(parents=True)
        _write_rgb(tmp_path / "cat" / "degre" / "img.png")
        _write_rgb(tmp_path / "cat" / "clean" / "img.png")
        pairs = discover_pairs_category(tmp_path, lq_name="degre", gt_name="clean")
        assert len(pairs) == 1

    def test_returns_empty_when_no_lq_gt(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        pairs = discover_pairs_category(tmp_path)
        assert len(pairs) == 0

    def test_skips_category_missing_partner(self, tmp_path: Path) -> None:
        # Only LQ dir, no GT dir
        (tmp_path / "cat" / "LQ").mkdir(parents=True)
        _write_rgb(tmp_path / "cat" / "LQ" / "img.png")
        pairs = discover_pairs_category(tmp_path)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# discover_pairs (shuffle + split)
# ---------------------------------------------------------------------------


class TestDiscoverPairs:
    def test_train_val_split_sizes(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=10)
        train = discover_pairs(tmp_path, split="train", val_ratio=0.2, seed=0)
        val = discover_pairs(tmp_path, split="val", val_ratio=0.2, seed=0)
        assert len(train) + len(val) == 10
        assert len(val) == 2

    def test_train_val_no_overlap(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=10)
        train = discover_pairs(tmp_path, split="train", val_ratio=0.2, seed=0)
        val = discover_pairs(tmp_path, split="val", val_ratio=0.2, seed=0)
        assert set(train).isdisjoint(set(val))

    def test_different_seeds_produce_different_order(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=10)
        pairs_a = discover_pairs(tmp_path, split="train", val_ratio=0.0, seed=0)
        pairs_b = discover_pairs(tmp_path, split="train", val_ratio=0.0, seed=1)
        assert pairs_a != pairs_b

    def test_same_seed_is_reproducible(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=10)
        pairs_a = discover_pairs(tmp_path, split="train", val_ratio=0.2, seed=99)
        pairs_b = discover_pairs(tmp_path, split="train", val_ratio=0.2, seed=99)
        assert pairs_a == pairs_b

    def test_val_ratio_zero_puts_all_in_train(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=5)
        train = discover_pairs(tmp_path, split="train", val_ratio=0.0)
        val = discover_pairs(tmp_path, split="val", val_ratio=0.0)
        assert len(train) == 5
        assert len(val) == 0


# ---------------------------------------------------------------------------
# PairedRestorationDataset
# ---------------------------------------------------------------------------


class TestPairedRestorationDataset:
    def test_raises_when_no_pairs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PairedRestorationDataset(tmp_path)

    def test_len_and_shapes(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=5)
        ds = PairedRestorationDataset(tmp_path, split="train", val_ratio=0.0)
        assert len(ds) == 5
        item = ds[0]
        assert item["degraded"].shape == (3, 4, 4)
        assert item["clean"].shape == (3, 4, 4)

    def test_values_in_unit_range(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=1)
        # overwrite with distinct colours
        _write_rgb(tmp_path / "cat0" / "LQ" / "img0.png", (255, 0, 0))
        _write_rgb(tmp_path / "cat0" / "GT" / "img0.png", (0, 255, 0))
        ds = PairedRestorationDataset(tmp_path, split="train", val_ratio=0.0)
        item = ds[0]
        assert float(item["degraded"].max()) <= 1.0
        assert float(item["clean"].max()) <= 1.0

    def test_val_split(self, tmp_path: Path) -> None:
        _setup_category(tmp_path, n_categories=10)
        train_ds = PairedRestorationDataset(tmp_path, split="train", val_ratio=0.2)
        val_ds = PairedRestorationDataset(tmp_path, split="val", val_ratio=0.2)
        assert len(train_ds) + len(val_ds) == 10
