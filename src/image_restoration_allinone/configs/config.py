"""Configuration dataclasses for the all-in-one image restoration project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict, Unpack

from fvcore.common.config import CfgNode

from image_restoration_allinone.configs.generate_config import dataclass_from_class
from image_restoration_allinone.models.build import MODEL_REGISTRY

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
    arch_name: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the NAFNet restoration model."""

    arch_name: str = "NAFNet"
    """Name of the architecture to use."""

    def __post_init__(self) -> None:
        """Validate that the specified architecture is registered.

        Generates dataclasses for all registered model classes to ensure that their configurations are available.
        The configurations are generated dynamically based on the constructor signature of each model class.


        Raises:
            ValueError: If the specified architecture is not registered.
        """
        if self.arch_name not in MODEL_REGISTRY._obj_map:
            raise ValueError(f"Model '{self.arch_name}' is not registered.")
        model_objects = list(MODEL_REGISTRY._obj_map.values())
        for model_cls in model_objects:
            dataclass_from_class(model_cls)

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
    batch_size: int
    epochs: int
    val_interval: int
    checkpoint_freq: int
    lr: float
    lr_min: float
    weight_decay: float
    seed: int
    amp: bool


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Top-level configuration for a training run."""

    output_dir: Path = Path("results/default")
    """Directory where checkpoints are saved."""
    batch_size: int = 8
    """Number of image pairs per batch."""
    epochs: int = 100
    """Total number of epochs."""
    val_interval: int = 1
    """Run validation every N epochs."""
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
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.val_interval <= 0:
            raise ValueError(f"val_interval must be positive, got {self.val_interval}")
        if self.checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be positive, got {self.checkpoint_freq}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_TrainConfigKwargs]) -> Self:
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class _LoggingConfigKwargs(TypedDict, total=False):
    log_dir: Path
    experiment_name: str
    log_img_limit: int


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    log_dir: str = "mlruns"
    """MLflow tracking URI / local directory."""
    experiment_name: str = "image-restoration-allinone"
    """MLflow experiment name."""
    log_img_limit: int = 4
    """Maximum number of images to log per batch."""

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_LoggingConfigKwargs]) -> Self:
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
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def config_from_args(cfg: CfgNode) -> Config:
    """Build a :class:`Config` from parsed CLI arguments."""
    losses: dict[str, float] = {"mse": 1.0}  # default
    if cfg.loss is not None:
        losses = {}
        for pair in cfg.loss.split(","):
            name, _, weight_str = pair.partition(":")
            losses[name.strip()] = float(weight_str.strip()) if weight_str else 1.0

    data = DataConfig.from_optional_kwargs(
        data_root=cfg.data.data_root,
        patch_size=cfg.data.patch_size,
        use_augmentation=cfg.data.use_augmentation,
        num_workers=cfg.data.num_workers,
        lq_dir_name=cfg.data.lq_dir_name,
        gt_dir_name=cfg.data.gt_dir_name,
        val_ratio=cfg.data.val_ratio,
        val_split_seed=cfg.data.val_split_seed,
    )
    model = ModelConfig.from_optional_kwargs(
        arch_name=cfg.arch_name,
    )
    loss = LossConfig.from_optional_kwargs(
        losses=losses,
    )
    train = TrainConfig.from_optional_kwargs(
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        val_interval=cfg.val_interval,
        checkpoint_freq=cfg.checkpoint_freq,
        lr=cfg.lr,
        lr_min=cfg.lr_min,
        seed=cfg.seed,
        amp=cfg.data.amp,
    )
    logging = LoggingConfig.from_optional_kwargs(
        log_dir=cfg.log_dir,
        experiment_name=cfg.experiment_name,
        log_img_limit=cfg.log_img_limit,
    )
    return Config(data=data, model=model, loss=loss, train=train, logging=logging)
