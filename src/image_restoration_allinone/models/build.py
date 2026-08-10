from fvcore.common.registry import Registry
from torch import nn

from image_restoration_allinone.configs.config import ModelConfig

MODEL_REGISTRY = Registry("MODEL")


@MODEL_REGISTRY.register()
def build_model(cfg: ModelConfig) -> nn.Module:
    """Builds the image restoration model.

    Args:
        cfg: The configuration for the model.

    Returns:
        An instance of the image restoration model.
    """
    model_name = cfg.arch_name
    model_class = MODEL_REGISTRY.get(model_name)
    return model_class(cfg)
