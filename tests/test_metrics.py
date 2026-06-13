"""Unit tests for metric utilities."""

from __future__ import annotations

import pytest
import torch

from image_restoration_allinone.utils.metrics import (
    compute_lpips,
    compute_mse,
    compute_psnr,
    compute_ssim,
    RunningMetrics,
)

_DEVICE = torch.device("cpu")


class TestComputeLpips:
    def test_zero_for_identical(self) -> None:
        x = torch.rand(1, 3, 64, 64)
        assert compute_lpips(x, x) == pytest.approx(0.0, abs=1e-4)

    def test_positive_for_different(self) -> None:
        x = torch.zeros(1, 3, 64, 64)
        y = torch.ones(1, 3, 64, 64)
        assert compute_lpips(x, y) > 0.0


class TestComputeMse:
    def test_zero_for_identical(self) -> None:
        x = torch.rand(2, 3, 16, 16)
        assert compute_mse(x, x) == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_different(self) -> None:
        x = torch.zeros(1, 3, 8, 8)
        y = torch.ones(1, 3, 8, 8)
        assert compute_mse(x, y) == pytest.approx(1.0, rel=1e-5)


class TestComputePsnr:
    def test_high_psnr_for_identical(self) -> None:
        x = torch.rand(1, 3, 32, 32)
        psnr = compute_psnr(x, x)
        assert psnr > 80.0  # torchmetrics returns ~100 dB for identical tensors

    def test_lower_psnr_for_noisy(self) -> None:
        x = torch.rand(1, 3, 32, 32)
        y = x + 0.1 * torch.randn_like(x)
        y = y.clamp(0, 1)
        assert compute_psnr(x, y) < compute_psnr(x, x)


class TestComputeSsim:
    def test_one_for_identical(self) -> None:
        x = torch.rand(1, 3, 32, 32)
        assert compute_ssim(x, x) == pytest.approx(1.0, abs=1e-4)

    def test_less_than_one_for_different(self) -> None:
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)
        assert compute_ssim(x, y) < 1.0


class TestRunningMetrics:
    def test_accumulate_and_compute(self) -> None:
        rm = RunningMetrics(_DEVICE)
        x = torch.rand(1, 3, 32, 32)
        rm.update(x, x)
        result = rm.compute()
        assert set(result.keys()) == {"psnr", "ssim", "mse", "psnr_y", "ssim_y", "lpips"}
        assert result["mse"] == pytest.approx(0.0, abs=1e-6)
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)
        assert result["psnr"] > 80.0
        assert result["ssim_y"] == pytest.approx(1.0, abs=1e-4)
        assert result["psnr_y"] > 80.0
        assert result["lpips"] == pytest.approx(0.0, abs=1e-4)

    def test_reset_clears_state(self) -> None:
        rm = RunningMetrics(_DEVICE)
        x = torch.rand(1, 3, 32, 32)
        rm.update(x, x)
        rm.reset()
        # After reset, calling compute again should still work when fed new data
        y = torch.rand(1, 3, 32, 32)
        rm.update(x, y)
        result = rm.compute()
        assert result["mse"] > 0.0
