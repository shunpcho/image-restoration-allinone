"""Pair-consistent image augmentation using albumentations."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(patch_size: int) -> A.Compose:
    """Return an albumentations pipeline for training.

    Both the degraded and the clean image receive **identical** spatial transforms.
    Color/intensity transforms are applied only to the degraded image.
    """
    return A.Compose(
        [
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),  # HWC → CHW, stays float32 in [0, 1]
        ],
        additional_targets={"clean": "image"},
    )


def build_val_transform() -> A.Compose:
    """Return an albumentations pipeline for validation (no random ops)."""
    return A.Compose(
        [ToTensorV2()],
        additional_targets={"clean": "image"},
    )
