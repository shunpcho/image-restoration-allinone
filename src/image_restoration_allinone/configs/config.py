"""Configuration dataclasses for the all-in-one image restoration project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict, Unpack

# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class _DataConfigKwargs(TypedDict, total=False):
    data_root: Path
    patch_size: int
    use_augmentation: bool
    num_workers: int
    pin_memory: bool
    lq_dir_name: str
    gt_dir_name: str
    val_ratio: float
    val_split_seed: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_root: Path = Path("data")
    """Root directory of the dataset."""
    patch_size: int = 256
    """Spatial size of training patches (H = W = patch_size)."""
    use_augmentation: bool = True
    """Whether to apply random augmentation during training."""
    num_workers: int = 4
    """Number of DataLoader worker processes."""
    pin_memory: bool = True
    """Whether to pin memory in DataLoader for faster GPU transfer."""
    lq_dir_name: str = "LQ"
    """Sub-directory name for low-quality (degraded) images."""
    gt_dir_name: str = "GT"
    """Sub-directory name for ground-truth (clean) images."""
    val_ratio: float = 0.1
    """Fraction of all pairs reserved for validation split."""
    val_split_seed: int = 42
    """Random seed for reproducible train/val shuffling."""

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_DataConfigKwargs]) -> Self:
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class _ModelConfigKwargs(TypedDict, total=False):
    width: int
    num_blocks: list[int]
    num_enc_blks: list[int]
    middle_blk_num: int
    dropout_rate: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the NAFNet restoration model."""

    width: int = 32
    """Base channel width of the network."""
    num_enc_blks: tuple[int, ...] = (1, 1, 1, 28)
    """Number of NAFBlocks per encoder stage."""
    middle_blk_num: int = 1
    """Number of NAFBlocks in the middle (bottleneck) stage."""
    num_dec_blks: tuple[int, ...] = (1, 1, 1, 1)
    """Number of NAFBlocks per decoder stage."""
    dropout_rate: float = 0.0
    """Dropout rate inside NAFBlocks (0 = disabled)."""

    def __post_init__(self) -> None:
        if len(self.num_enc_blks) != len(self.num_dec_blks):
            raise ValueError("num_enc_blks and num_dec_blks must have the same length")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_ModelConfigKwargs]) -> Self:
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------


class _LossConfigKwargs(TypedDict, total=False):
    losses: dict[str, float]


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Configuration for the composite loss function.

    ``losses`` maps a loss name to its weight. Supported names:
    ``"mse"``, ``"l1"``, ``"charbonnier"``, ``"ssim"``, ``"perceptual"``.
    """

    losses: dict[str, float] = field(default_factory=lambda: {"mse": 1.0})
    """Mapping of loss-function name → scalar weight."""

    def __post_init__(self) -> None:
        supported = {"mse", "l1", "charbonnier", "ssim", "perceptual"}
        for name in self.losses:
            if name not in supported:
                raise ValueError(f"Unsupported loss '{name}'. Choose from {supported}")
        if not self.losses:
            raise ValueError("losses must contain at least one entry")

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_LossConfigKwargs]) -> Self:
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class _TrainConfigKwargs(TypedDict, total=False):
    output_dir: Path
    log_dir: Path
    experiment_name: str
    batch_size: int
    total_iters: int
    val_interval: int
    save_interval: int
    epochs: int
    val_interval_epoch: int
    checkpoint_freq: int
    lr: float
    lr_min: float
    weight_decay: float
    seed: int
    amp: bool


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Top-level configuration for a training run."""

    output_dir: Path = Path("results")
    """Directory where checkpoints are saved."""
    log_dir: Path = Path("mlruns")
    """MLflow tracking URI / local directory."""
    experiment_name: str = "image-restoration-allinone"
    """MLflow experiment name."""
    batch_size: int = 8
    """Number of image pairs per batch."""
    total_iters: int = 200_000
    """Total number of training iterations."""
    val_interval: int = 1_000
    """Run validation every N iterations."""
    save_interval: int = 10_000
    """Save a checkpoint every N iterations."""
    epochs: int = 100
    """Total number of epochs (epoch-based trainer)."""
    val_interval_epoch: int = 1
    """Run validation every N epochs (epoch-based trainer)."""
    checkpoint_freq: int = 10
    """Frequency of saving checkpoints (in epochs)."""
    lr: float = 1e-3
    """Peak learning rate for AdamW."""
    lr_min: float = 1e-7
    """Minimum learning rate for cosine scheduler."""
    weight_decay: float = 1e-3
    """Weight decay for AdamW."""
    seed: int = 42
    """Random seed for reproducibility."""
    amp: bool = True
    """Enable automatic mixed precision (AMP)."""

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.total_iters <= 0:
            raise ValueError(f"total_iters must be positive, got {self.total_iters}")
        if self.val_interval <= 0:
            raise ValueError(f"val_interval must be positive, got {self.val_interval}")
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.val_interval_epoch <= 0:
            raise ValueError(f"val_interval_epoch must be positive, got {self.val_interval_epoch}")
        if self.checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be positive, got {self.checkpoint_freq}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_TrainConfigKwargs]) -> Self:
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Config (root)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """Root configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train an all-in-one image restoration model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root directory.")
    parser.add_argument("--patch-size", type=int, default=None, help="Training patch size.")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--lq-dir-name", type=str, default=None, help="Sub-directory name for LQ images (default: LQ).")
    parser.add_argument("--gt-dir-name", type=str, default=None, help="Sub-directory name for GT images (default: GT).")
    parser.add_argument(
        "--val-ratio", type=float, default=None, help="Fraction of data used for validation (default: 0.1)."
    )
    parser.add_argument(
        "--val-split-seed", type=int, default=None, help="Random seed for train/val split (default: 42)."
    )

    # Model
    parser.add_argument("--width", type=int, default=None, help="NAFNet base channel width.")

    # Loss  (JSON-like: "mse:1.0,ssim:0.1")
    parser.add_argument(
        "--losses",
        type=str,
        default=None,
        help='Comma-separated loss:weight pairs, e.g. "mse:1.0,ssim:0.1".',
    )

    # Train
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--val-interval-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed CLI arguments."""
    losses: dict[str, float] = {"mse": 1.0}  # default
    if args.losses is not None:
        losses = {}
        for pair in args.losses.split(","):
            name, _, weight_str = pair.partition(":")
            losses[name.strip()] = float(weight_str.strip()) if weight_str else 1.0

    data = DataConfig.from_optional_kwargs(
        data_root=args.data_root,
        patch_size=args.patch_size,
        use_augmentation=not args.no_augmentation if args.no_augmentation else True,
        num_workers=args.num_workers,
        lq_dir_name=args.lq_dir_name,
        gt_dir_name=args.gt_dir_name,
        val_ratio=args.val_ratio,
        val_split_seed=args.val_split_seed,
    )
    model = ModelConfig.from_optional_kwargs(
        width=args.width,
    )
    loss = LossConfig.from_optional_kwargs(
        losses=losses,
    )
    train = TrainConfig.from_optional_kwargs(
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        epochs=args.epochs,
        val_interval_epoch=args.val_interval_epoch,
        checkpoint_freq=args.checkpoint_freq,
        lr=args.lr,
        lr_min=args.lr_min,
        seed=args.seed,
        amp=not args.no_amp if args.no_amp else True,
    )
    return Config(data=data, model=model, loss=loss, train=train)
