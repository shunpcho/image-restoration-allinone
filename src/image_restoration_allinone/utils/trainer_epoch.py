"""Training loop epoch-based."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from image_restoration_allinone.configs.config import TrainConfig
from image_restoration_allinone.utils.logger import MLflowLogger


def _set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Epoch-based trainer for the all-in-one restoration model.

    Args:
        model: The restoration network.
        criterion: Composite loss returning ``(total, components)``.
        train_loader: Infinite-style DataLoader for training batches.
        val_loader: DataLoader used for periodic validation.
        cfg: Training hyper-parameters.
        device: Compute device.
        logger: Optional MLflow logger.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        device: torch.device,
        logger: MLflowLogger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs * len(train_loader),
            eta_min=cfg.lr_min,
        )
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = GradScaler(amp_device, enabled=cfg.amp and device.type == "cuda")

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the full training loop."""
        _set_seed(self.cfg.seed)
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"Epoch {epoch}/{self.cfg.epochs} :")
            self.epoch = epoch

            train_loss, _ = self._train_epoch()
            val_loss, val_components = self._validate_epoch()

            # Log metrics to MLflow if logger is provided.
            self._log_validation_metrics(val_loss, val_components, epoch)
            self._print_epoch_summary(epoch, train_loss, val_loss)

            # Checkpoint
            if epoch % self.cfg.checkpoint_freq == 0 or epoch == self.cfg.epochs:
                self._save_checkpoint(epoch)

    def _loop(self, train: bool = True) -> tuple[float, dict[str, torch.Tensor]]:
        """Run one epoch of training.

        Args:
            train: If True, perform backpropagation and optimization steps. If False, only compute metrics.

        Returns:
            A dictionary of average metrics for the epoch.

        This method handles both training and validation loops.
        In training mode, it performs backpropagation with gradient scaling and clipping for stability.

        Use PyTorch's automatic mixed precision (AMP) for efficient training.
        The GradScaler helps prevent underflow in gradients when using lower precision.
        And we clip gradients to a max norm to stabilize training.
        """
        self.model.train(train)
        dataloader = self.train_loader if train else self.val_loader
        self.step_in_epoch = len(dataloader)
        self.total_steps = self.cfg.epochs * self.step_in_epoch
        total_components: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for step, batch in enumerate(dataloader, start=1):
            degraded = batch["degraded"].to(self.device)
            clean = batch["clean"].to(self.device)

            if train:
                self.optimizer.zero_grad()
            with torch.autocast(self.device.type, enabled=self.cfg.amp):
                restored = self.model(degraded)
                loss, component = self.criterion(restored, clean)
                total_loss += loss.detach()
                for key, value in component.items():
                    total_components[key] = (
                        total_components.get(key, torch.tensor(0.0, device=self.device)) + value.detach()
                    )

            if train:
                # If training, backpropagate with gradient scaling and clipping.
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # Log training metrics to MLflow if logger is provided.
                self._log_train_metrics(self.optimizer, loss, component, step)

            self._print_progress(step, loss, train)

        epoch_loss = total_loss / self.step_in_epoch
        epoch_components = {key: value / self.step_in_epoch for key, value in total_components.items()}
        return float(epoch_loss.item()), epoch_components

    def _train_epoch(self) -> tuple[float, dict[str, torch.Tensor]]:
        """Run training epoch."""
        return self._loop(train=True)

    def _validate_epoch(self) -> tuple[float, dict[str, torch.Tensor]]:
        """Run validation epoch."""
        with torch.inference_mode():
            return self._loop(train=False)

    @staticmethod
    def _print_epoch_summary(epoch: int, train_loss: float, val_loss: float) -> None:
        """Print summary for the current epoch."""
        print(f"Epoch {epoch} : train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    def _log_train_metrics(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, components: dict[str, torch.Tensor], step: int
    ) -> None:
        """Log training metrics to MLflow if logger is provided for each step."""
        if self.logger is not None:
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = {
                "train/loss": loss.item(),
                "train/lr": lr,
                **{f"train/{k}": float(v.item()) for k, v in components.items()},
            }
            self.logger.log_metrics(train_metrics, step=step + (self.epoch - 1) * self.step_in_epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint for the current epoch."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint -> {checkpoint_path}")
        if self.logger is not None:
            self.logger.log_artifact(checkpoint_path)

    def _log_validation_metrics(self, loss: float, components: dict[str, torch.Tensor], epoch: int) -> None:
        """Log validation metrics to MLflow if logger is provided for each epoch."""
        if self.logger is not None:
            val_metrics = {
                "val/loss": loss,
                **{f"val/{k}": float(v.item()) for k, v in components.items()},
            }
            self.logger.log_metrics(val_metrics, step=epoch)

    def _print_progress(
        self,
        step: int,
        loss: torch.Tensor,
        train: bool,
    ) -> None:
        """Print information related to the current step.

        Args:
            step: Current step (within the epoch).
            loss: Loss value for the current step.
            train: Whether this is a training step or validation step.
        """
        pre_str = f"{step} / {self.step_in_epoch} ["

        loss_str = f"] Train Loss: {loss.item():.6f}" if train else "] Val..."

        term_cols = shutil.get_terminal_size(fallback=(156, 38)).columns
        progress_bar_len = max(16, min(term_cols - len(pre_str) - len(loss_str) - 1, 30))
        progress = int(progress_bar_len * (step / self.step_in_epoch))
        progress_bar_str = f"{progress * '='}>{(progress_bar_len - progress) * '.'}"

        full_string = pre_str + progress_bar_str + loss_str
        print(full_string, end=("\r" if step < self.step_in_epoch else "\n"), flush=True)
