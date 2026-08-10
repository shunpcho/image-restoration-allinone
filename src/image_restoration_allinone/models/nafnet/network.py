"""All-in-one image restoration network based on NAFNet.

Reference:
    Chen et al., "Simple Baselines for Image Restoration",
    ECCV 2022. https://arxiv.org/abs/2204.04676
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from image_restoration_allinone.configs.config import ModelConfig
from image_restoration_allinone.models.build import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channel-first Layer Normalisation for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → normalise over channel dim
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and apply element-wise gating."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Non-linear Activation Free Block.

    Combines a depth-wise convolution for local mixing and a channel MLP
    with SimpleGate instead of an explicit nonlinearity.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        dw_channels = channels * 2
        ffn_channels = channels * 4

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.sg1 = SimpleGate()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)

        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Spatial / attention branch
        x_norm = self.norm1(x)
        x_conv = self.conv2(self.conv1(x_norm))
        x_gated = self.sg1(x_conv)
        x_scaled = x_gated * self.sca(x_gated)
        x_out = self.dropout1(self.conv3(x_scaled))
        x = inp + x_out * self.beta

        # Feed-forward branch
        x_norm2 = self.norm2(x)
        x_ffn = self.conv4(x_norm2)
        x_ffn_gated = self.sg2(x_ffn)
        x_out2 = self.dropout2(self.conv5(x_ffn_gated))
        x += x_out2 * self.gamma
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------


def _make_stage(channels: int, num_blocks: int, dropout_rate: float) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels, dropout_rate) for _ in range(num_blocks)])


# ---------------------------------------------------------------------------
# NAFNet
# ---------------------------------------------------------------------------


@MODEL_REGISTRY.register()
class NAFNet(nn.Module):
    """U-Net shaped NAFNet for all-in-one image restoration.

    The network takes a degraded RGB image and produces a restored RGB image.
    No explicit degradation-type conditioning is required; the model learns
    to handle blur, low-light, rain, and other corruptions from paired data.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        width = cfg.width
        enc_blks = cfg.num_enc_blks
        mid_blks = cfg.middle_blk_num
        dec_blks = cfg.num_dec_blks
        drop = cfg.dropout_rate
        num_stages = len(enc_blks)

        self.intro = nn.Conv2d(3, width, 3, padding=1)

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for num_blks in enc_blks:
            self.encoders.append(_make_stage(ch, num_blks, drop))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        # ---- Middle ----
        self.middle = _make_stage(ch, mid_blks, drop)

        # ---- Decoder ----
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blks in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(_make_stage(ch, num_blks, drop))

        self.ending = nn.Conv2d(ch, 3, 3, padding=1)
        self._num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore *x* (B, 3, H, W) in [0, 1]."""
        h, w = x.shape[2], x.shape[3]
        # Pad to multiple of 2^num_stages
        factor = 2**self._num_stages
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        inp = self.intro(x)

        # Encoder pass - collect skip connections
        skips: list[torch.Tensor] = []
        feat = inp
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        # Middle
        feat = self.middle(feat)

        # Decoder pass - fuse skip connections
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            feat = up(feat)
            feat += skip
            feat = decoder(feat)

        out = self.ending(feat) + x

        # Remove padding
        out = out[:, :, :h, :w]
        return out.clamp(0.0, 1.0)
