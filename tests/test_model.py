"""Unit tests for the NAFNet model."""

from __future__ import annotations

import pytest
import torch

from image_restoration_allinone.configs.config import ModelConfig
from image_restoration_allinone.models.build import build_model
from image_restoration_allinone.models.nafnet.network import NAFNet

_DEVICE = torch.device("cpu")


class TestNAFNet:
    @pytest.fixture
    def model(self) -> NAFNet:
        cfg = ModelConfig(arch_name="NAFNet", width=8, num_enc_blks=(1, 1), middle_blk_num=1, num_dec_blks=(1, 1))
        return build_model(cfg).to(_DEVICE)

    def test_output_shape_matches_input(self, model: NAFNet) -> None:
        x = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == x.shape

    def test_output_in_unit_range(self, model: NAFNet) -> None:
        x = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            out = model(x)
        assert float(out.min().item()) >= 0.0
        assert float(out.max().item()) <= 1.0

    def test_handles_non_power_of_two_input(self, model: NAFNet) -> None:
        """Model should handle inputs whose spatial dims are not multiples of 2^stages."""
        x = torch.rand(1, 3, 37, 51)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == x.shape

    def test_mismatched_enc_dec_raises(self) -> None:
        with pytest.raises(ValueError, match="num_enc_blks and num_dec_blks"):
            ModelConfig(num_enc_blks=(1, 1), num_dec_blks=(1,))
