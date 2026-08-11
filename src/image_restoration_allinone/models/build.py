from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast, TYPE_CHECKING

from fvcore.common.registry import Registry
from torch import nn

if TYPE_CHECKING:
    from image_restoration_allinone.configs.config import ModelConfig

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg: ModelConfig) -> nn.Module:
    """Builds the image restoration model.

    Args:
        cfg: The configuration for the model.

    Returns:
        An instance of the image restoration model.

    Raises:
        ValueError: If the model parameters are not a dataclass instance.
    """
    model_name = cfg.arch_name
    model_class = MODEL_REGISTRY.get(model_name)
    if isinstance(cfg.parameters, type) or not is_dataclass(cfg.parameters):
        raise ValueError(f"Model '{model_name}' parameters must be a dataclass instance.")
    return model_class(**asdict(cast("Any", cfg.parameters)))
