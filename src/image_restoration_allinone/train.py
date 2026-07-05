"""Training entry point for the all-in-one image restoration model."""

from __future__ import annotations

import torch

from image_restoration_allinone.configs.config import build_argument_parser, config_from_args
from image_restoration_allinone.data.dataloader import build_dataloaders
from image_restoration_allinone.models.nafnet.network import build_model
from image_restoration_allinone.utils.logger import MLflowLogger
from image_restoration_allinone.utils.loss import LossComposer
from image_restoration_allinone.utils.trainer import Trainer


def main() -> None:
    """Parse CLI arguments and start training."""
    parser = build_argument_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg.data, batch_size=cfg.train.batch_size)

    # Model
    model = build_model(cfg.model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss
    criterion = LossComposer(cfg.loss)

    # Logger
    logger = MLflowLogger(cfg.logging)
    logger.start(
        params={
            "data_root": str(cfg.data.data_root),
            "patch_size": cfg.data.patch_size,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "lr": cfg.train.lr,
            "width": cfg.model.width,
            "losses": str(cfg.loss.losses),
        }
    )

    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.train,
        device=device,
        logger=logger,
    )
    try:
        trainer.run()
    finally:
        logger.end()


if __name__ == "__main__":
    main()
