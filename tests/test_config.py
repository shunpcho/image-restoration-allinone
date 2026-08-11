"""Tests for CfgNode-to-dataclass configuration conversion."""

import pytest
from fvcore.common.config import CfgNode

from image_restoration_allinone.configs.config import config_from_cfg_node
from image_restoration_allinone.configs.default import get_default_cfg


def test_cfg_node_conversion_uses_architecture_parameters() -> None:
    config = config_from_cfg_node(get_default_cfg())

    assert config.model.arch_name == "NAFNet"
    assert config.model.parameters.width == 32
    assert config.model.parameters.dropout_rate == pytest.approx(0.0)


def test_model_parameter_section_can_be_omitted() -> None:
    cfg = CfgNode()
    cfg.model = CfgNode()
    cfg.model.arch_name = "NAFNet"

    config = config_from_cfg_node(cfg)

    assert config.model.parameters.width == 32


def test_unknown_model_parameter_includes_its_path() -> None:
    cfg = CfgNode()
    cfg.model = CfgNode()
    cfg.model.arch_name = "NAFNet"
    cfg.model.nafnet = CfgNode()
    cfg.model.nafnet.wdith = 32

    with pytest.raises(ValueError, match=r"model\.nafnet\.wdith"):
        config_from_cfg_node(cfg)


def test_model_parameter_dataclass_is_cached() -> None:
    first = config_from_cfg_node(get_default_cfg())
    second = config_from_cfg_node(get_default_cfg())

    assert type(first.model.parameters) is type(second.model.parameters)
