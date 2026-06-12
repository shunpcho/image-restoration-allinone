"""DataLoader factory for the restoration project."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_restoration_allinone.configs.config import DataConfig
from image_restoration_allinone.data.dataset import PairedRestorationDataset
from image_restoration_allinone.data.transforms import build_train_transform, build_val_transform


def build_dataloaders(
    cfg: DataConfig,
    batch_size: int = 8,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Build training and validation :class:`DataLoader` objects.

    Args:
        cfg: Data configuration.
        batch_size: Number of samples per training batch.

    Returns:
        ``(train_loader, val_loader)`` tuple.
    """
    root = Path(cfg.data_root)

    train_transform = build_train_transform(cfg.patch_size) if cfg.use_augmentation else build_val_transform()
    val_transform = build_val_transform()

    train_dataset: PairedRestorationDataset = PairedRestorationDataset(
        root,
        split="train",
        transform=train_transform,
        lq_dir_name=cfg.lq_dir_name,
        gt_dir_name=cfg.gt_dir_name,
        val_ratio=cfg.val_ratio,
        seed=cfg.val_split_seed,
    )
    val_dataset: PairedRestorationDataset = PairedRestorationDataset(
        root,
        split="val",
        transform=val_transform,
        lq_dir_name=cfg.lq_dir_name,
        gt_dir_name=cfg.gt_dir_name,
        val_ratio=cfg.val_ratio,
        seed=cfg.val_split_seed,
    )

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader
