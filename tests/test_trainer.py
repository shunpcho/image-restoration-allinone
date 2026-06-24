"""Unit tests for epoch-based validation scheduling in Trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from image_restoration_allinone.configs.config import TrainConfig
from image_restoration_allinone.utils.trainer import Trainer


class _SingleBatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Single-item dataset for trainer control-flow tests."""

    def __init__(self, sample: dict[str, torch.Tensor]) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return self._sample


def _build_trainer(output_dir: Path, *, epochs: int, val_interval: int) -> Trainer:
    """Build a lightweight trainer instance for control-flow tests."""
    sample = {
        "degraded": torch.zeros(1, 3, 8, 8),
        "clean": torch.zeros(1, 3, 8, 8),
    }
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(_SingleBatchDataset(sample))
    config = TrainConfig(
        output_dir=output_dir,
        epochs=epochs,
        val_interval=val_interval,
        checkpoint_freq=10_000,
    )
    return Trainer(
        model=nn.Conv2d(3, 3, kernel_size=1),
        criterion=nn.MSELoss(),
        train_loader=loader,
        val_loader=loader,
        cfg=config,
        device=torch.device("cpu"),
        logger=None,
    )


def _patch_epoch_methods(
    trainer: Trainer,
    monkeypatch: pytest.MonkeyPatch,
    validated_epochs: list[int],
) -> None:
    """Patch trainer epoch methods to track validation epochs only."""

    def fake_train_epoch() -> tuple[float, dict[str, torch.Tensor]]:
        return 0.0, {}

    def fake_validate_epoch() -> tuple[float, dict[str, torch.Tensor]]:
        validated_epochs.append(trainer.epoch)
        return 0.0, {}

    def fake_save_checkpoint(_epoch: int) -> None:
        pass

    monkeypatch.setattr(trainer, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "_validate_epoch", fake_validate_epoch)
    monkeypatch.setattr(trainer, "_save_checkpoint", fake_save_checkpoint)


def test_validation_interval_with_final_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _build_trainer(tmp_path, epochs=5, val_interval=2)
    validated_epochs: list[int] = []
    _patch_epoch_methods(trainer, monkeypatch, validated_epochs)
    trainer.run()

    assert validated_epochs == [2, 4, 5]


def test_validation_final_epoch_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _build_trainer(tmp_path, epochs=3, val_interval=10)
    validated_epochs: list[int] = []
    _patch_epoch_methods(trainer, monkeypatch, validated_epochs)
    trainer.run()

    assert validated_epochs == [3]
