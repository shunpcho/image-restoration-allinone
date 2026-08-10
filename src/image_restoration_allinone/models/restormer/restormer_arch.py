"""Restormer architecture implementation.

This module implements the Restormer architecture from https://github.com/swz30/Restormer/tree/main.
"""

import torch
from einops import rearrange
from torch import nn

from image_restoration_allinone.models.build import MODEL_REGISTRY


##########################################################################
# Layer Norm
def to_3d(x: torch.Tensor) -> torch.Tensor:
    """Reshape spatial dimensions of a 4D tensor to 3D for LayerNorm.

    Args:
        x: Input tensor of shape (B, C, H, W).

    Returns:
        Tensor of shape (B, H*W, C) suitable for LayerNorm.
    """
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reshape a 3D tensor back to 4D after LayerNorm.

    Args:
        x: Input tensor of shape (B, H*W, C).
        h: Height of the original spatial dimensions.
        w: Width of the original spatial dimensions.

    Returns:
        Tensor of shape (B, C, H, W).
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """Bias-Free Layer Normalization.

    Args:
        normalized_shape: Shape of the input tensor to be normalized.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1, "Only 1D normalization is supported."

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Layer Normalization with Bias.

    Args:
        normalized_shape: Shape of the input tensor to be normalized.
        eps: A value added to the denominator for numerical stability.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1, "Only 1D normalization is supported."

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer Normalization wrapper that can be either bias-free or with bias.

    Args:
        dim: Dimension of the input tensor to be normalized.
        layer_norm_type: Type of layer normalization to use ('BiasFree' or 'WithBias').

    Raises:
        ValueError: If an unsupported layer normalization type is provided.
    """

    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.norm = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.norm = WithBiasLayerNorm(dim)
        else:
            raise ValueError(f"Unsupported layer norm type: {layer_norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network.

    Args:
        dim: Dimension of the input tensor.
        ffn_expansion_factor: Expansion factor for the feed-forward network.
        bias: Whether to include a bias term in the convolutional layers.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-Dconv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """Multi-Dconv Head Transposed Self-Attention.

    Args:
        dim: Dimension of the input tensor.
        num_heads: Number of attention Heads.
        bias: Whether to include a bias term in the convolutional layers.
    """

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
# Transformer
class TransformerBlock(nn.Module):
    """Transformer Block consisting of Attention and FeedForward layers.

    Args:
        dim: Dimension of the input tensor.
        num_heads: Number of attention heads.
        ffn_expansion_factor: Expansion factor for the feed-forward network.
        bias: Whether to include a bias term in the convolutional layers.
        layer_norm_type: Type of layer normalization to use ('BiasFree' or 'WithBias').
    """

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool, layer_norm_type: str) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mdta = x + self.attn(self.norm1(x))
        x_gdfn = x_mdta + self.ffn(self.norm2(x_mdta))
        return x_gdfn


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding using a 3x3 convolution.

    Args:
        inp_channels: Number of input channels.
        embed_dim: Dimension of the embedding space. The default is 48.
        bias: Whether to include a bias term in the convolutional layers. The default is False.
    """

    def __init__(self, inp_channels: int, embed_dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(inp_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


##########################################################################
# Resizeing modules
class Downsample(nn.Module):
    """Downsampling module using a 3x3 Conv.

    Args:
        n_feat: Number of feature dimensions.
    """

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module using a 3x3 Conv.

    Args:
        n_feat: Number of feature dimensions.
    """

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


##########################################################################
# Encoder
class Encoder(nn.Module):
    """Hierarchical encoder used by Restormer."""

    def __init__(
        self,
        inp_channels: int,
        dim: int,
        num_blocks: list[int],
        heads: list[int],
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_enc_level1 = self.encoder_level1(self.patch_embed(x))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))
        return out_enc_level1, out_enc_level2, out_enc_level3, latent


##########################################################################
# Decoder
class Decoder(nn.Module):
    """Hierarchical decoder and refinement stage used by Restormer."""

    def __init__(
        self,
        dim: int,
        num_blocks: list[int],
        num_refinement_blocks: int,
        heads: list[int],
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        # No  1x1 conv to reduce channels
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

    def forward(
        self,
        out_enc_level1: torch.Tensor,
        out_enc_level2: torch.Tensor,
        out_enc_level3: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.refinement(out_dec_level1)


##########################################################################
# Restormer
@MODEL_REGISTRY.register()
class Restormer(nn.Module):
    """Restormer architecture for image restoration.

    Args:
        inp_channels: Number of input channels.
        out_channels: Number of output channels.
        dim: Base dimension for the model. The default is 48.
        num_blocks: List of number of transformer blocks at each stage. The default is [4, 6, 6, 8].
        num_refinement_blocks: Number of transformer blocks in the refinement stage. The default is 4.
        heads: List of number of attention heads at each stage. The default is [1, 2, 4, 8].
        ffn_expansion_factor: Expansion factor for the feed-forward network. The default is 2.66.
        bias: Whether to include a bias term in the convolutional layers. The default is False.
        layer_norm_type: Type of layer normalization to use ("BiasFree" or "WithBias"). The default is "WithBias".
        dual_pixel_task: Whether to use dual pixel task. The default is False.
                         True for dual-pixel defocus deblurring only. Also, set inp_channels=6.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_refinement_blocks: int = 4,
        heads: list[int] | None = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.encoder = Encoder(inp_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, layer_norm_type)
        self.decoder = Decoder(
            dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias, layer_norm_type
        )
        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_enc_level1, out_enc_level2, out_enc_level3, latent = self.encoder(x)
        out_refinement = self.decoder(out_enc_level1, out_enc_level2, out_enc_level3, latent)

        # For Dual-Pixel Defocus Deblurring Task.
        if self.dual_pixel_task:
            return self.skip_conv(out_enc_level1) + self.output(out_refinement)
        return x + self.output(out_refinement)
