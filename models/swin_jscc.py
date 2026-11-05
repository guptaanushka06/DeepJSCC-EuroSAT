"""
models/swin_jscc.py
-------------------
Stage 4: End-to-End Swin Transformer DeepJSCC (SwinJSCC).

Architecture:
  Encoder:
    1. PatchEmbed: Conv2d(3 -> embed_dim, 4x4, stride 4) -> (B, 16x16, embed_dim)
    2. Stage 1: SwinTransformerBlock x depth[0], dim = embed_dim, res = (16, 16)
    3. PatchMerging: downsamples (16, 16) -> (8, 8), dim = embed_dim * 2
    4. Stage 2: SwinTransformerBlock x depth[1], dim = embed_dim * 2, res = (8, 8)
    5. PatchMerging: downsamples (8, 8) -> (4, 4), dim = embed_dim * 4
    6. Stage 3: SwinTransformerBlock x depth[2], dim = embed_dim * 4, res = (4, 4)
    7. Bottleneck projection: Conv2d(dim * 4 -> bottleneck_ch, 1x1) on 4x4 grid.
       Default: 256 channels * 4 * 4 = 4,096 transmitted real channel symbols.

  Channel:
    Power-normalised symbol transmission over AWGN or Rayleigh fading channel.

  Decoder:
    1. Bottleneck projection: Conv2d(bottleneck_ch -> dim * 4, 1x1) on 4x4 grid.
    2. Stage 3': SwinTransformerBlock x depth[2], dim = embed_dim * 4, res = (4, 4)
    3. PatchExpand: upsamples (4, 4) -> (8, 8), dim = embed_dim * 2
    4. Stage 2': SwinTransformerBlock x depth[1], dim = embed_dim * 2, res = (8, 8)
    5. PatchExpand: upsamples (8, 8) -> (16, 16), dim = embed_dim
    6. Stage 1': SwinTransformerBlock x depth[0], dim = embed_dim, res = (16, 16)
    7. Reconstruction Head: Conv2d + PixelShuffle(4) + Conv2d + Sigmoid -> (B, 3, 64, 64).
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import power_normalize, awgn, rayleigh, sample_snr
from models.network_swinir import SwinTransformerBlock, PatchMerging


# ─────────────────────────────────────────────────────────────────────────────
# PatchExpand: Symmetric spatial upsampling for Swin Decoder
# ─────────────────────────────────────────────────────────────────────────────
class PatchExpand(nn.Module):
    """
    Expands spatial resolution by factor of 2 while halving channel dimension.
    Exact inverse of PatchMerging.

    Args:
        input_resolution (tuple[int, int]): (H, W) of input tokens
        dim (int): input channel dimension
        norm_layer: Normalization layer (nn.LayerNorm)
    Input:
        (B, H*W, dim)
    Output:
        (B, 2H*2W, dim // 2)
    """
    def __init__(self, input_resolution: tuple, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input token length {L} does not match resolution {H}x{W}"

        x = self.expand(x)  # (B, H*W, 2*C)
        out_c = C // 2
        # Rearrange (B, H, W, 2*C) = (B, H, W, 2, 2, out_c) -> (B, 2H, 2W, out_c)
        x = x.view(B, H, W, 2, 2, out_c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 2 * H, 2 * W, out_c)
        x = x.reshape(B, -1, out_c)
        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# ─────────────────────────────────────────────────────────────────────────────
# PatchEmbed: Image to token sequence
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    Projects raw RGB image into non-overlapping patch embeddings.
    Args:
        img_size (int): Input image resolution (64)
        patch_size (int): Size of patch kernel (4)
        in_chans (int): Input image channels (3)
        embed_dim (int): Number of linear projection channels (48)
    Input:
        (B, 3, 64, 64)
    Output:
        (B, 16*16=256, embed_dim)
    """
    def __init__(self, img_size: int = 64, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 48,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, embed_dim, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, embed_dim)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full End-to-End SwinJSCC Model
# ─────────────────────────────────────────────────────────────────────────────
class SwinJSCC(nn.Module):
    """
    Stage 4: End-to-End Swin Transformer Deep Joint Source-Channel Coding.

    Standard input: (B, 3, 64, 64) in [0, 1]
    Transmitted symbols: 4,096 reals (matching Models 1, 2, 2b-2e, 3)
    Reconstruction: (B, 3, 64, 64) in [0, 1]
    """
    def __init__(self,
                 channel: str = 'awgn',
                 snr_min: float = 0.0,
                 snr_max: float = 20.0,
                 embed_dim: int = 48,
                 depths: tuple = (2, 2, 2),
                 num_heads: tuple = (3, 6, 6),
                 window_sizes: tuple = (8, 8, 4),
                 bottleneck_ch: int = 256,
                 use_shift: bool = True):
        super().__init__()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.use_shift = use_shift

        # ── 1. Patch Embedding (3 -> 48, 64x64 -> 16x16 tokens) ──────────────
        self.patch_embed = PatchEmbed(
            img_size=64, patch_size=4, in_chans=3, embed_dim=embed_dim
        )

        def _get_shift(i, w_size):
            if not use_shift:
                return 0
            return 0 if (i % 2 == 0) else w_size // 2

        # ── 2. Encoder Stages ────────────────────────────────────────────────
        # Stage 1: (16, 16), dim = embed_dim (48)
        res1 = (16, 16)
        self.enc_stage1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, input_resolution=res1,
                num_heads=num_heads[0], window_size=window_sizes[0],
                shift_size=_get_shift(i, window_sizes[0]),
            )
            for i in range(depths[0])
        ])
        self.enc_merge1 = PatchMerging(res1, dim=embed_dim)

        # Stage 2: (8, 8), dim = embed_dim * 2 (96)
        res2 = (8, 8)
        dim2 = embed_dim * 2
        self.enc_stage2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim2, input_resolution=res2,
                num_heads=num_heads[1], window_size=window_sizes[1],
                shift_size=_get_shift(i, window_sizes[1]),
            )
            for i in range(depths[1])
        ])
        self.enc_merge2 = PatchMerging(res2, dim=dim2)

        # Stage 3: (4, 4), dim = embed_dim * 4 (192)
        res3 = (4, 4)
        dim3 = dim2 * 2
        self.enc_stage3 = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim3, input_resolution=res3,
                num_heads=num_heads[2], window_size=window_sizes[2],
                shift_size=_get_shift(i, window_sizes[2]),
            )
            for i in range(depths[2])
        ])

        # ── 3. Bottleneck Channel Projection ─────────────────────────────────
        # 1x1 Conv preserves 4x4 spatial structure while mapping 192 -> 256
        # 256 channels * 4 * 4 spatial = 4,096 reals!
        self.to_channel = nn.Conv2d(dim3, bottleneck_ch, kernel_size=1)

        # ── 4. Decoder Bottleneck Projection ─────────────────────────────────
        self.from_channel = nn.Conv2d(bottleneck_ch, dim3, kernel_size=1)

        # ── 5. Decoder Stages ────────────────────────────────────────────────
        # Stage 3': (4, 4), dim = 192
        self.dec_stage3 = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim3, input_resolution=res3,
                num_heads=num_heads[2], window_size=window_sizes[2],
                shift_size=_get_shift(i, window_sizes[2]),
            )
            for i in range(depths[2])
        ])
        self.dec_expand3 = PatchExpand(res3, dim=dim3)

        # Stage 2': (8, 8), dim = 96
        self.dec_stage2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim2, input_resolution=res2,
                num_heads=num_heads[1], window_size=window_sizes[1],
                shift_size=_get_shift(i, window_sizes[1]),
            )
            for i in range(depths[1])
        ])
        self.dec_expand2 = PatchExpand(res2, dim=dim2)

        # Stage 1': (16, 16), dim = 48
        self.dec_stage1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, input_resolution=res1,
                num_heads=num_heads[0], window_size=window_sizes[0],
                shift_size=_get_shift(i, window_sizes[0]),
            )
            for i in range(depths[0])
        ])

        # ── 6. Reconstruction Head ───────────────────────────────────────────
        # Maps (B, 48, 16, 16) -> PixelShuffle(4) -> (B, 3, 64, 64) in [0, 1]
        self.recon_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(4),   # 48 // (4*4) = 3 channels, 16*4 = 64 spatial
            nn.Conv2d(embed_dim // 16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),         # output strictly in [0, 1]
        )

    def forward(self,
                img: torch.Tensor,
                channel: str = None,
                snr_db: float = None) -> tuple:
        """
        Forward pass.
        Args:
            img: (B, 3, 64, 64) in [0, 1]
            channel: 'awgn' or 'rayleigh'
            snr_db: channel SNR in dB
        Returns:
            recon: (B, 3, 64, 64) reconstruction in [0, 1]
            z_norm: (B, 4096) power-normalized channel symbols
        """
        B = img.shape[0]
        ch = channel if channel is not None else self.default_channel

        # 1. Patch Embed -> (B, 256, 48)
        x = self.patch_embed(img)

        # 2. Encoder Stage 1 -> Merge
        for blk in self.enc_stage1:
            x = blk(x, (16, 16))
        x = self.enc_merge1(x)              # (B, 64, 96)

        # 3. Encoder Stage 2 -> Merge
        for blk in self.enc_stage2:
            x = blk(x, (8, 8))
        x = self.enc_merge2(x)              # (B, 16, 192)

        # 4. Encoder Stage 3
        for blk in self.enc_stage3:
            x = blk(x, (4, 4))             # (B, 16, 192)

        # 5. Bottleneck to Channel: (B, 192, 4, 4) -> (B, 256, 4, 4) = 4,096 reals
        x_spatial = x.transpose(1, 2).contiguous().view(B, 192, 4, 4)
        z = self.to_channel(x_spatial)
        z_flat = z.reshape(B, -1)              # (B, 4096)

        # 6. Power normalise
        z_norm = power_normalize(z_flat)

        # 7. Channel noise
        if snr_db is None:
            snr_db = sample_snr(self.snr_min, self.snr_max)

        if ch == 'awgn':
            z_rx = awgn(z_norm, snr_db)
        elif ch == 'rayleigh':
            z_rx, _ = rayleigh(z_norm, snr_db)
        elif ch in ('none', 'bypass'):
            z_rx = z_norm
        else:
            raise ValueError(f"Unknown channel: {ch!r}. Use 'awgn', 'rayleigh', or 'bypass'.")

        # 8. Decoder Bottleneck: (B, 256, 4, 4) -> (B, 192, 4, 4) -> (B, 16, 192)
        z_rx_spatial = z_rx.reshape(B, 256, 4, 4)
        dec_x = self.from_channel(z_rx_spatial)
        dec_x = dec_x.flatten(2).transpose(1, 2).contiguous()

        # 9. Decoder Stage 3' -> Expand
        for blk in self.dec_stage3:
            dec_x = blk(dec_x, (4, 4))
        dec_x = self.dec_expand3(dec_x)     # (B, 64, 96)

        # 10. Decoder Stage 2' -> Expand
        for blk in self.dec_stage2:
            dec_x = blk(dec_x, (8, 8))
        dec_x = self.dec_expand2(dec_x)     # (B, 256, 48)

        # 11. Decoder Stage 1'
        for blk in self.dec_stage1:
            dec_x = blk(dec_x, (16, 16))    # (B, 256, 48)

        # 12. Reconstruction Head: (B, 48, 16, 16) -> (B, 3, 64, 64)
        dec_spatial = dec_x.transpose(1, 2).contiguous().view(B, 48, 16, 16)
        recon = self.recon_head(dec_spatial)

        return recon, z_norm
