"""Training loop (iteration-based) with MLflow logging and AMP."""

from __future__ import annotations

import random
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from image_restoration_allinone.configs.config import TrainConfig
from image_restoration_allinone.utils.evaluator import Evaluator
from image_restoration_allinone.utils.logger import MLflowLogger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Iteration-based trainer for the all-in-one restoration model.

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
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
        cfg: TrainConfig,
        device: torch.device,
        logger: MLflowLogger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.total_iters,
            eta_min=cfg.lr_min,
        )
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        self.scaler: GradScaler = GradScaler(amp_device, enabled=cfg.amp and device.type == "cuda")

        self.evaluator = Evaluator(model, criterion, val_loader, device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training loop."""
        _set_seed(self.cfg.seed)
        data_iter = _infinite_loader(self.train_loader)

        for iteration in range(1, self.cfg.total_iters + 1):
            loss, components = self._train_step(next(data_iter))

            # Log training metrics every iteration
            lr = self.optimizer.param_groups[0]["lr"]
            metrics: dict[str, float] = {
                "train/loss": loss,
                "train/lr": lr,
                **{f"train/{k}": float(v.item()) for k, v in components.items()},
            }
            self._log(metrics, iteration)
            print(f"[{iteration:>8d}/{self.cfg.total_iters}] loss={loss:.6f}  lr={lr:.2e}")

            # Validation on iteration 1 and every val_interval iterations
            if iteration == 1 or iteration % self.cfg.val_interval == 0:
                val_metrics = self.evaluator.run()
                self._log(val_metrics, iteration)
                print(
                    f"  [val] loss={val_metrics['val/loss']:.6f}  "
                    f"mse={val_metrics['val/mse']:.6f}  "
                    f"psnr={val_metrics['val/psnr']:.2f} dB  "
                    f"ssim={val_metrics['val/ssim']:.4f}"
                )
                self.model.train()

            # Checkpoint
            if iteration % self.cfg.save_interval == 0:
                self._save_checkpoint(iteration)

        self._save_checkpoint(self.cfg.total_iters, name="final.pth")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, torch.Tensor]]:
        self.model.train()
        degraded = batch["degraded"].to(self.device)
        clean = batch["clean"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp):
            restored = self.model(degraded)
            total_loss, components = self.criterion(restored, clean)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return float(total_loss.item()), components

    def _save_checkpoint(self, iteration: int, name: str | None = None) -> None:
        fname = name or f"ckpt_{iteration:08d}.pth"
        ckpt_path = self.output_dir / fname
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint → {ckpt_path}")
        if self.logger:
            self.logger.log_artifact(ckpt_path)

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if self.logger:
            self.logger.log_metrics(metrics, step)


def _infinite_loader(
    loader: DataLoader[dict[str, torch.Tensor]],
) -> Generator[dict[str, torch.Tensor]]:
    """Yield batches indefinitely by restarting the DataLoader."""
    while True:
        yield from loader
