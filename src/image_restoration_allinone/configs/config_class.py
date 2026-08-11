"""Configuration dataclasses for the all-in-one image restoration project."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Self, TypedDict, Unpack

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


_UNSET = object()
_LOSS_PAIR_LENGTH = 2


class ConfigurationError(ValueError):
    """Raised when an input configuration does not match the application schema."""


def _model_parameters_class(arch_name: str) -> type:
    try:
        model_class = MODEL_REGISTRY.get(arch_name)
    except KeyError as exc:
        raise ConfigurationError(f"Model '{arch_name}' is not registered.") from exc
    return dataclass_from_class(model_class)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a registered restoration model."""

    arch_name: str = "NAFNet"
    """Name of the architecture to use."""
    parameters: object = field(default=_UNSET)
    """Architecture-specific configuration generated from the model constructor."""

    def __post_init__(self) -> None:
        """Create default parameters and validate explicitly supplied parameters.

        Raises:
            ConfigurationError: If the parameters do not match the selected architecture.
        """
        parameter_class = _model_parameters_class(self.arch_name)
        if self.parameters is _UNSET:
            object.__setattr__(self, "parameters", parameter_class())
        elif not isinstance(self.parameters, parameter_class):
            raise ConfigurationError(f"parameters must be an instance of {parameter_class.__name__}")

    @classmethod
    def from_parameter_kwargs(cls, arch_name: str, **kwargs: object) -> Self:
        """Build a model configuration from architecture-specific keyword arguments.

        Raises:
            ConfigurationError: If the keyword arguments do not match the model schema.
        """
        parameter_class = _model_parameters_class(arch_name)
        try:
            parameters = parameter_class(**kwargs)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid configuration at model.{arch_name.lower()}: {exc}") from exc
        return cls(arch_name=arch_name, parameters=parameters)


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


def _cfg_node_to_dict(value: object) -> object:
    if isinstance(value, CfgNode):
        node = cast("Mapping[object, object]", value)
        return {str(key): _cfg_node_to_dict(item) for key, item in node.items()}
    if isinstance(value, list):
        return [_cfg_node_to_dict(item) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(_cfg_node_to_dict(item) for item in cast("tuple[object, ...]", value))
    return value


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Invalid configuration at {path}: expected a mapping.")
    mapping = cast("Mapping[object, object]", value)
    return {str(key): item for key, item in mapping.items()}


def _section(config: Mapping[str, object], name: str) -> dict[str, object]:
    if name not in config:
        return {}
    return _mapping(config[name], name)


def _construct(config_class: type, values: Mapping[str, object], path: str) -> object:
    dataclass_fields = getattr(config_class, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, Mapping):
        raise TypeError(f"{config_class.__name__} must be a dataclass type.")
    allowed = {
        str(name)
        for name, config_field in cast("Mapping[object, object]", dataclass_fields).items()
        if getattr(config_field, "init", False)
    }
    unknown = set(values).difference(allowed)
    if unknown:
        key = min(unknown)
        raise ConfigurationError(f"Unknown configuration key: {path}.{key}")
    try:
        return config_class(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration at {path}: {exc}") from exc


def _loss_weight(value: object, path: str) -> float:
    if not isinstance(value, (float, int, str)):
        raise ConfigurationError(f"Invalid configuration at {path}: expected a numeric weight.")
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid configuration at {path}: expected a numeric weight.") from exc


def _losses(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(name): _loss_weight(weight, f"loss.losses.{name}") for name, weight in mapping.items()}
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError("Invalid configuration at loss.losses: expected a mapping or pairs.")

    losses: dict[str, float] = {}
    values = cast("list[object] | tuple[object, ...]", value)
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)):
            raise ConfigurationError(f"Invalid configuration at loss.losses[{index}]: expected a name and weight pair.")
        pair = cast("list[object] | tuple[object, ...]", item)
        if len(pair) != _LOSS_PAIR_LENGTH:
            raise ConfigurationError(f"Invalid configuration at loss.losses[{index}]: expected a name and weight pair.")
        name, weight = cast("tuple[object, object]", pair)
        if not isinstance(name, str):
            raise ConfigurationError(f"Invalid configuration at loss.losses[{index}][0]: expected a string.")
        losses[name] = _loss_weight(weight, f"loss.losses[{index}][1]")
    return losses


def _path(value: object, path: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise ConfigurationError(f"Invalid configuration at {path}: expected a path string.")


def _data_config(raw_config: Mapping[str, object]) -> DataConfig:
    data_values = _section(raw_config, "data")
    if "data_root" in data_values:
        data_values["data_root"] = _path(data_values["data_root"], "data.data_root")
    return cast("DataConfig", _construct(DataConfig, data_values, "data"))


def _model_config(raw_config: Mapping[str, object]) -> ModelConfig:
    model_values = _section(raw_config, "model")
    arch_name = model_values.pop("arch_name", "NAFNet")
    if not isinstance(arch_name, str):
        raise ConfigurationError("Invalid configuration at model.arch_name: expected a string.")
    parameter_section = arch_name.lower()
    parameter_values = _mapping(model_values.pop(parameter_section, {}), f"model.{parameter_section}")
    if model_values:
        key = min(model_values)
        raise ConfigurationError(f"Unknown configuration key: model.{key}")
    parameter_class = _model_parameters_class(arch_name)
    parameters = _construct(parameter_class, parameter_values, f"model.{parameter_section}")
    return ModelConfig(arch_name=arch_name, parameters=parameters)


def _loss_config(raw_config: Mapping[str, object]) -> LossConfig:
    loss_values = _section(raw_config, "loss")
    if "losses" in loss_values:
        loss_values["losses"] = _losses(loss_values["losses"])
    return cast("LossConfig", _construct(LossConfig, loss_values, "loss"))


def _train_config(raw_config: Mapping[str, object]) -> TrainConfig:
    train_values = _section(raw_config, "train")
    if "output_dir" in train_values:
        train_values["output_dir"] = _path(train_values["output_dir"], "train.output_dir")
    return cast("TrainConfig", _construct(TrainConfig, train_values, "train"))


def config_from_cfg_node(cfg: CfgNode) -> Config:
    """Convert a parsed :class:`CfgNode` into the typed application configuration.

    Raises:
        ConfigurationError: If the input has unknown keys or invalid configuration values.
    """
    raw_config = _mapping(_cfg_node_to_dict(cfg), "root")
    expected_sections = {"data", "model", "loss", "train", "logging"}
    unknown_sections = set(raw_config).difference(expected_sections)
    if unknown_sections:
        section = min(unknown_sections)
        raise ConfigurationError(f"Unknown configuration key: {section}")

    return Config(
        data=_data_config(raw_config),
        model=_model_config(raw_config),
        loss=_loss_config(raw_config),
        train=_train_config(raw_config),
        logging=cast("LoggingConfig", _construct(LoggingConfig, _section(raw_config, "logging"), "logging")),
    )
