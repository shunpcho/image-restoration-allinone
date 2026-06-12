"""Unit tests for LossComposer and individual loss components."""

from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]
import torch

from image_restoration_allinone.configs.config import LossConfig
from image_restoration_allinone.utils.loss import CharbonnierLoss, LossComposer


class TestCharbonnierLoss:
    def test_returns_eps_for_identical_inputs(self) -> None:
        loss_fn = CharbonnierLoss()
        x = torch.rand(2, 3, 16, 16)
        value = loss_fn(x, x)
        # sqrt(0^2 + eps^2) = eps, so the loss equals the epsilon value
        assert float(value.item()) == pytest.approx(1e-3, abs=1e-5)

    def test_positive_for_different_inputs(self) -> None:
        loss_fn = CharbonnierLoss()
        x = torch.zeros(1, 3, 8, 8)
        y = torch.ones(1, 3, 8, 8)
        assert float(loss_fn(x, y).item()) > 0


class TestLossComposer:
    def test_mse_default(self) -> None:
        cfg = LossConfig()
        composer = LossComposer(cfg)
        x = torch.rand(1, 3, 8, 8)
        y = torch.rand(1, 3, 8, 8)
        total, components = composer(x, y)
        assert "mse" in components
        assert float(total.item()) >= 0

    def test_weighted_sum(self) -> None:
        cfg = LossConfig(losses={"mse": 1.0, "l1": 0.5})
        composer = LossComposer(cfg)
        x = torch.rand(1, 3, 8, 8)
        y = torch.rand(1, 3, 8, 8)
        total, components = composer(x, y)
        expected = components["mse"] + 0.5 * components["l1"]
        assert float(total.item()) == pytest.approx(float(expected.item()), rel=1e-5)

    def test_unsupported_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported loss"):
            LossConfig(losses={"unknown": 1.0})

    def test_zero_for_identical_inputs_mse(self) -> None:
        cfg = LossConfig(losses={"mse": 1.0})
        composer = LossComposer(cfg)
        x = torch.ones(1, 3, 4, 4) * 0.5
        total, _ = composer(x, x)
        assert float(total.item()) == pytest.approx(0.0, abs=1e-6)

    def test_charbonnier_in_composer(self) -> None:
        cfg = LossConfig(losses={"charbonnier": 1.0})
        composer = LossComposer(cfg)
        x = torch.rand(1, 3, 8, 8)
        y = torch.rand(1, 3, 8, 8)
        total, components = composer(x, y)
        assert "charbonnier" in components
        assert float(total.item()) > 0
