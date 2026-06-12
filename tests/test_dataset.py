"""Unit tests for dataset discovery and PairedRestorationDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_restoration_allinone.data.dataset import (
    discover_pairs,
    discover_pairs_category,
    discover_pairs_keyword,
    discover_pairs_separate,
    PairedRestorationDataset,
)


def _write_rgb(path: Path, value: tuple[int, int, int] = (128, 64, 32)) -> None:
    img = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8))
    img.save(path)


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------


class TestDiscoverPairsKeyword:
    def test_finds_real_mean_pairs(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "img_real.png")
        _write_rgb(tmp_path / "img_mean.png")
        pairs = discover_pairs_keyword(tmp_path)
        assert len(pairs) == 1
        degraded, clean = pairs[0]
        assert "_real" in degraded.name
        assert "_mean" in clean.name

    def test_ignores_missing_partner(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "only_real.png")
        pairs = discover_pairs_keyword(tmp_path)
        assert len(pairs) == 0


class TestDiscoverPairsSeparate:
    def test_finds_lq_gt_pairs(self, tmp_path: Path) -> None:
        (tmp_path / "LQ").mkdir()
        (tmp_path / "GT").mkdir()
        _write_rgb(tmp_path / "LQ" / "img.png")
        _write_rgb(tmp_path / "GT" / "img.png")
        pairs = discover_pairs_separate(tmp_path)
        assert len(pairs) == 1

    def test_custom_dir_names(self, tmp_path: Path) -> None:
        (tmp_path / "degre").mkdir()
        (tmp_path / "clean").mkdir()
        _write_rgb(tmp_path / "degre" / "img.png")
        _write_rgb(tmp_path / "clean" / "img.png")
        pairs = discover_pairs_separate(tmp_path, lq_name="degre", gt_name="clean")
        assert len(pairs) == 1

    def test_returns_empty_without_dirs(self, tmp_path: Path) -> None:
        pairs = discover_pairs_separate(tmp_path)
        assert len(pairs) == 0


class TestDiscoverPairsCategory:
    def test_finds_pairs_in_category_subdirs(self, tmp_path: Path) -> None:
        for category in ("Blur", "Haze"):
            (tmp_path / category / "LQ").mkdir(parents=True)
            (tmp_path / category / "GT").mkdir(parents=True)
            _write_rgb(tmp_path / category / "LQ" / "img.png")
            _write_rgb(tmp_path / category / "GT" / "img.png")
        pairs = discover_pairs_category(tmp_path)
        assert len(pairs) == 2

    def test_returns_empty_when_no_lq_gt(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        pairs = discover_pairs_category(tmp_path)
        assert len(pairs) == 0


class TestDiscoverPairs:
    def test_case1_flat(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "a_real.png")
        _write_rgb(tmp_path / "a_mean.png")
        pairs = discover_pairs(tmp_path)
        assert len(pairs) == 1

    def test_case2_train_split(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _write_rgb(train_dir / "b_real.png")
        _write_rgb(train_dir / "b_mean.png")
        pairs = discover_pairs(tmp_path, split="train")
        assert len(pairs) == 1

    def test_case3_separate_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "LQ").mkdir()
        (tmp_path / "GT").mkdir()
        _write_rgb(tmp_path / "LQ" / "c.png")
        _write_rgb(tmp_path / "GT" / "c.png")
        pairs = discover_pairs(tmp_path)
        assert len(pairs) == 1

    def test_case4_category_subdirs(self, tmp_path: Path) -> None:
        for category in ("Blur", "Haze"):
            (tmp_path / category / "LQ").mkdir(parents=True)
            (tmp_path / category / "GT").mkdir(parents=True)
            _write_rgb(tmp_path / category / "LQ" / "d.png")
            _write_rgb(tmp_path / category / "GT" / "d.png")
        pairs = discover_pairs(tmp_path)
        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# PairedRestorationDataset
# ---------------------------------------------------------------------------


class TestPairedRestorationDataset:
    def test_raises_when_no_pairs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PairedRestorationDataset(tmp_path)

    def test_len_and_shapes(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "img_real.png")
        _write_rgb(tmp_path / "img_mean.png")
        ds = PairedRestorationDataset(tmp_path)
        assert len(ds) == 1
        item = ds[0]
        assert item["degraded"].shape == (3, 4, 4)
        assert item["clean"].shape == (3, 4, 4)

    def test_values_in_unit_range(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "x_real.png", (255, 0, 0))
        _write_rgb(tmp_path / "x_mean.png", (0, 255, 0))
        ds = PairedRestorationDataset(tmp_path)
        item = ds[0]
        assert float(item["degraded"].max()) <= 1.0
        assert float(item["clean"].max()) <= 1.0
