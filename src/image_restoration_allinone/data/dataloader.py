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
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Build training and validation :class:`DataLoader` objects.

    Args:
        cfg: Data configuration.

    Returns:
        ``(train_loader, val_loader)`` tuple.
    """
    root = Path(cfg.data_root)

    train_transform = build_train_transform(cfg.patch_size) if cfg.use_augmentation else build_val_transform()
    val_transform = build_val_transform()

    train_dataset: PairedRestorationDataset = PairedRestorationDataset(root, split="train", transform=train_transform)
    val_dataset: PairedRestorationDataset = PairedRestorationDataset(root, split="val", transform=val_transform)

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size if hasattr(cfg, "batch_size") else 8,  # pyright: ignore[reportAttributeAccessIssue]
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
