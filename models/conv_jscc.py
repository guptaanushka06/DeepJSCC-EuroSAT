"""
models/conv_jscc.py
-------------------
Stage 4 Ablation A: Non-Swin Convolutional/Residual JSCC Baseline (ConvJSCC).

Architectural counterpart to SwinJSCC:
  - Replaces Swin transformer attention blocks with standard ResNet convolutional blocks.
  - Matches the channel bottleneck: 256 ch x 4 x 4 spatial = 4,096 real symbols.
  - Matches the 3-stage hierarchical downsampling (16x16 -> 8x8 -> 4x4) and upsampling.
  - Parameter count: ~2.59M (closely matched to SwinJSCC's 2.66M, within 2.7%).
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import power_normalize, awgn, rayleigh, sample_snr


class ResBlock(nn.Module):
    """Standard 2-conv residual block with LeakyReLU."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class ConvJSCC(nn.Module):
    """
    Non-Swin convolutional baseline for SwinJSCC ablation.
    Transmits exactly 4,096 real channel uses per image over AWGN / Rayleigh.
    """
    def __init__(self,
                 channel: str = 'awgn',
                 snr_min: float = 0.0,
                 snr_max: float = 20.0,
                 embed_dim: int = 48,
                 bottleneck_ch: int = 256,
                 n_blocks: tuple = (2, 2, 1)):
        super().__init__()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

        # 1. Stem: Projects 64x64x3 -> 16x16x48 (identical spatial reduction as Swin PatchEmbed)
        self.stem = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)

        # 2. Encoder Stages
        # Stage 1: (B, 48, 16, 16)
        self.enc1 = nn.Sequential(*[ResBlock(embed_dim) for _ in range(n_blocks[0])])
        # Downsample 1: 16x16 -> 8x8, dim: 48 -> 96
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 2: (B, 96, 8, 8)
        self.enc2 = nn.Sequential(*[ResBlock(embed_dim * 2) for _ in range(n_blocks[1])])
        # Downsample 2: 8x8 -> 4x4, dim: 96 -> 192
        self.down2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 3: (B, 192, 4, 4)
        self.enc3 = nn.Sequential(*[ResBlock(embed_dim * 4) for _ in range(n_blocks[2])])

        # 3. Channel Bottleneck: 192 -> 256 on 4x4 grid = 4,096 reals!
        self.to_channel = nn.Conv2d(embed_dim * 4, bottleneck_ch, kernel_size=1)
        self.from_channel = nn.Conv2d(bottleneck_ch, embed_dim * 4, kernel_size=1)

        # 4. Decoder Stages
        # Stage 3': (B, 192, 4, 4)
        self.dec3 = nn.Sequential(*[ResBlock(embed_dim * 4) for _ in range(n_blocks[2])])
        # Upsample 2: 4x4 -> 8x8, dim: 192 -> 96
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 2': (B, 96, 8, 8)
        self.dec2 = nn.Sequential(*[ResBlock(embed_dim * 2) for _ in range(n_blocks[1])])
        # Upsample 1: 8x8 -> 16x16, dim: 96 -> 48
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 1': (B, 48, 16, 16)
        self.dec1 = nn.Sequential(*[ResBlock(embed_dim) for _ in range(n_blocks[0])])

        # 5. Reconstruction Head: identical to SwinJSCC (PixelShuffle x4 -> 64x64x3)
        self.recon_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(4),   # 48 // 16 = 3 channels, 16 * 4 = 64 spatial
            nn.Conv2d(embed_dim // 16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None) -> tuple:
        B = img.shape[0]
        ch = channel if channel is not None else self.default_channel

        # Encoder
        x = self.stem(img)        # (B, 48, 16, 16)
        x = self.enc1(x)
        x = self.down1(x)         # (B, 96, 8, 8)
        x = self.enc2(x)
        x = self.down2(x)         # (B, 192, 4, 4)
        x = self.enc3(x)

        # Bottleneck projection
        z = self.to_channel(x)    # (B, 256, 4, 4)
        z_flat = z.reshape(B, -1) # (B, 4096)

        # Power normalization
        z_norm = power_normalize(z_flat)

        # Channel
        if snr_db is None:
            snr_db = sample_snr(self.snr_min, self.snr_max)

        if ch == 'awgn':
            z_rx = awgn(z_norm, snr_db)
        elif ch == 'rayleigh':
            z_rx, _ = rayleigh(z_norm, snr_db)
        elif ch in ('none', 'bypass'):
            z_rx = z_norm
        else:
            raise ValueError(f"Unknown channel: {ch!r}")

        # Decoder
        z_rx_spatial = z_rx.reshape(B, 256, 4, 4)
        dec_x = self.from_channel(z_rx_spatial)  # (B, 192, 4, 4)
        dec_x = self.dec3(dec_x)
        dec_x = self.up2(dec_x)                  # (B, 96, 8, 8)
        dec_x = self.dec2(dec_x)
        dec_x = self.up1(dec_x)                  # (B, 48, 16, 16)
        dec_x = self.dec1(dec_x)

        # Reconstruction
        recon = self.recon_head(dec_x)           # (B, 3, 64, 64)
        return recon, z_norm

