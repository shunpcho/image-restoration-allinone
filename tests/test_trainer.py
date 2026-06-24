"""Unit tests for the epoch-based Trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from image_restoration_allinone.configs.config import LossConfig, TrainConfig
from image_restoration_allinone.utils.loss import LossComposer
from image_restoration_allinone.utils.trainer import Trainer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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


def _make_loader() -> DataLoader[dict[str, torch.Tensor]]:
    sample = {
        "degraded": torch.rand(3, 8, 8),
        "clean": torch.rand(3, 8, 8),
    }
    return DataLoader(_SingleBatchDataset(sample))


def _build_trainer(
    output_dir: Path,
    *,
    epochs: int = 5,
    val_interval: int = 1,
    checkpoint_freq: int = 10_000,
) -> Trainer:
    """Build a lightweight trainer instance for control-flow tests."""
    loader = _make_loader()
    config = TrainConfig(
        output_dir=output_dir,
        epochs=epochs,
        val_interval=val_interval,
        checkpoint_freq=checkpoint_freq,
    )
    return Trainer(
        model=nn.Conv2d(3, 3, kernel_size=1),
        criterion=LossComposer(LossConfig(losses={"mse": 1.0})),
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


# ---------------------------------------------------------------------------
# Validation interval scheduling
# ---------------------------------------------------------------------------


class TestValidationSchedule:
    def test_interval_with_final_epoch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = _build_trainer(tmp_path, epochs=5, val_interval=2)
        validated_epochs: list[int] = []
        _patch_epoch_methods(trainer, monkeypatch, validated_epochs)
        trainer.run()

        assert validated_epochs == [2, 4, 5]

    def test_final_epoch_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = _build_trainer(tmp_path, epochs=3, val_interval=10)
        validated_epochs: list[int] = []
        _patch_epoch_methods(trainer, monkeypatch, validated_epochs)
        trainer.run()

        assert validated_epochs == [3]

    def test_every_epoch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = _build_trainer(tmp_path, epochs=4, val_interval=1)
        validated_epochs: list[int] = []
        _patch_epoch_methods(trainer, monkeypatch, validated_epochs)
        trainer.run()

        assert validated_epochs == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_checkpoint_at_freq_and_final_epoch(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=6, checkpoint_freq=3)
        trainer.run()

        saved = sorted(tmp_path.glob("checkpoint_epoch_*.pth"))
        saved_names = [p.name for p in saved]
        assert "checkpoint_epoch_0003.pth" in saved_names
        assert "checkpoint_epoch_0006.pth" in saved_names

    def test_checkpoint_always_at_final_epoch(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=4, checkpoint_freq=100)
        trainer.run()

        saved = sorted(tmp_path.glob("checkpoint_epoch_*.pth"))
        assert len(saved) == 1
        assert saved[0].name == "checkpoint_epoch_0004.pth"

    def test_checkpoint_contents(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=2, checkpoint_freq=2)
        trainer.run()

        ckpt = torch.load(tmp_path / "checkpoint_epoch_0002.pth", weights_only=False)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "scaler_state_dict" in ckpt
        assert ckpt["epoch"] == 2


# ---------------------------------------------------------------------------
# Full training loop integration
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def test_run_completes_without_error(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=2, checkpoint_freq=2)
        trainer.run()

    def test_model_parameters_update(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=3, checkpoint_freq=100)
        initial_params = {k: v.clone() for k, v in trainer.model.named_parameters()}
        trainer.run()

        changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                changed = True
                break
        assert changed

    def test_scheduler_advances(self, tmp_path: Path) -> None:
        trainer = _build_trainer(tmp_path, epochs=3, checkpoint_freq=100)
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.run()
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


class TestMLflowLogging:
    def test_logs_training_metrics_per_step(self, tmp_path: Path) -> None:
        mock_logger = MagicMock()
        trainer = _build_trainer(tmp_path, epochs=2, checkpoint_freq=100)
        trainer.logger = mock_logger
        trainer.run()

        calls = mock_logger.log_metrics.call_args_list
        train_calls = [c for c in calls if any("train/" in k for k in c[0][0])]
        assert len(train_calls) == 2  # 2 epochs x 1 step each

    def test_logs_validation_metrics_per_epoch(self, tmp_path: Path) -> None:
        mock_logger = MagicMock()
        trainer = _build_trainer(tmp_path, epochs=3, val_interval=2, checkpoint_freq=100)
        trainer.logger = mock_logger
        trainer.run()

        calls = mock_logger.log_metrics.call_args_list
        val_calls = [c for c in calls if any("val/" in k for k in c[0][0])]
        # Validation at epoch 2 and 3 (final)
        assert len(val_calls) == 2

    def test_logs_artifact_on_checkpoint(self, tmp_path: Path) -> None:
        mock_logger = MagicMock()
        trainer = _build_trainer(tmp_path, epochs=3, checkpoint_freq=3)
        trainer.logger = mock_logger
        trainer.run()

        assert mock_logger.log_artifact.call_count == 1
