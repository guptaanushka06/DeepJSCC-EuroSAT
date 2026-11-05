"""
models/deepjscc.py
------------------
Model 1: Plain DeepJSCC
  CNN Encoder → power_normalize → Channel → CNN Decoder

Architecture
------------
Encoder: 4 convolutional blocks (stride-2 downsampling)
         → flattens to 4096 reals (= 2048 complex symbols, ρ = 1/6)

Decoder: mirror of encoder using transposed convolutions
         → output (3, 64, 64) in [0, 1]  (sigmoid final activation)

The model is SNR-agnostic by design (Model 3 adds SNR conditioning).
For training, a random SNR is sampled per batch.

Reference: Bourtsoulatze et al., "Deep JSCC for Image Transmission", 2019.
"""

import torch
import torch.nn as nn
from channels import power_normalize, awgn, rayleigh, sample_snr

# ── bandwidth ratio config ───────────────────────────────────────────────────
# Source: 3 × 64 × 64 = 12,288 real values
# Transmitted: 2,048 complex symbols = 4,096 real values
# ρ_comp = 2,048 / 12,288 = 1/6 (complex ratio); ρ_real = 4,096 / 12,288 = 1/3 (real ratio)
LATENT_DIM = 4096      # number of reals in the latent (= 2k, k = 2048 complex)


# ────────────────────────────────────────────────────────────────────────────
# Helper: one conv block (Conv → BatchNorm → PReLU)
# ────────────────────────────────────────────────────────────────────────────
def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                  padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.PReLU(),
    )


def deconv_block(in_ch: int, out_ch: int,
                 stride: int = 1, last: bool = False) -> nn.Sequential:
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride,
                           padding=1, output_padding=stride - 1, bias=False),
    ]
    if not last:
        layers += [nn.BatchNorm2d(out_ch), nn.PReLU()]
    else:
        layers += [nn.Sigmoid()]   # output pixels in [0, 1]
    return nn.Sequential(*layers)


# ────────────────────────────────────────────────────────────────────────────
# Encoder
# ────────────────────────────────────────────────────────────────────────────
class JSCCEncoder(nn.Module):
    """
    CNN encoder: (B, 3, 64, 64) → (B, LATENT_DIM)

    Spatial progression:
        64×64  →  32×32  →  16×16  →  8×8  →  4×4
    Channel counts: 3 → 32 → 64 → 128 → 256
    Flatten 256×4×4 = 4096 reals.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3,   32,  stride=2),   # (B, 32, 32, 32)
            conv_block(32,  64,  stride=2),   # (B, 64, 16, 16)
            conv_block(64,  128, stride=2),   # (B, 128, 8, 8)
            conv_block(128, 256, stride=2),   # (B, 256, 4, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)                    # (B, 256, 4, 4)
        return feat.view(feat.size(0), -1)    # (B, 4096)


# ────────────────────────────────────────────────────────────────────────────
# Decoder
# ────────────────────────────────────────────────────────────────────────────
class JSCCDecoder(nn.Module):
    """
    CNN decoder: (B, LATENT_DIM) → (B, 3, 64, 64)

    Mirror of encoder using transposed convolutions.
    """

    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(1, (256, 4, 4))   # (B,4096) → (B,256,4,4)
        self.net = nn.Sequential(
            deconv_block(256, 128, stride=2),             # (B, 128, 8, 8)
            deconv_block(128, 64,  stride=2),             # (B, 64, 16, 16)
            deconv_block(64,  32,  stride=2),             # (B, 32, 32, 32)
            deconv_block(32,  3,   stride=2, last=True),  # (B, 3, 64, 64) [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(z)
        return self.net(x)


# ────────────────────────────────────────────────────────────────────────────
# Full Model 1: Plain DeepJSCC
# ────────────────────────────────────────────────────────────────────────────
class PlainDeepJSCC(nn.Module):
    """
    Model 1 – Plain DeepJSCC (CNN encoder + channel + CNN decoder).

    Usage:
        model = PlainDeepJSCC()
        # Training (random SNR per call):
        recon, latent = model(img_unit, channel='awgn')
        # Evaluation at fixed SNR:
        recon, latent = model(img_unit, channel='awgn', snr_db=10.0)

    Args:
        img_unit : (B, 3, 64, 64)  tensor in [0, 1]

    Returns:
        recon  : (B, 3, 64, 64)  reconstructed image in [0, 1]
        latent : (B, 4096)       power-normalised transmitted symbols
                                 (useful for monitoring)
    """

    def __init__(self,
                 channel: str = 'awgn',
                 snr_min: float = 0.0,
                 snr_max: float = 20.0):
        super().__init__()
        self.encoder = JSCCEncoder()
        self.decoder = JSCCDecoder()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self,
                img: torch.Tensor,
                channel: str = None,
                snr_db: float = None) -> tuple:
        """
        Forward pass.

        channel : 'awgn' or 'rayleigh'.  Defaults to self.default_channel.
        snr_db  : fixed SNR in dB.  If None, sample randomly during training.
        """
        ch = channel if channel is not None else self.default_channel

        # 1. Encode
        z = self.encoder(img)                  # (B, 4096)

        # 2. Power normalise
        z_norm = power_normalize(z)            # avg power per complex sym = 1

        # 3. Sample SNR if not given
        if snr_db is None:
            snr_db = sample_snr(self.snr_min, self.snr_max)

        # 4. Pass through channel
        if ch == 'awgn':
            z_rx = awgn(z_norm, snr_db)
        elif ch == 'rayleigh':
            z_rx, _ = rayleigh(z_norm, snr_db)
        elif ch in ('none', 'bypass'):
            z_rx = z_norm
        else:
            raise ValueError(f"Unknown channel: {ch!r}. Use 'awgn', 'rayleigh', or 'bypass'.")

        # 5. Decode
        recon = self.decoder(z_rx)             # (B, 3, 64, 64)

        return recon, z_norm


# ────────────────────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PlainDeepJSCC().to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Forward pass
    dummy = torch.rand(4, 3, 64, 64, device=device)
    recon, latent = model(dummy, channel='awgn', snr_db=10.0)
    print(f"Input  shape : {dummy.shape}")
    print(f"Latent shape : {latent.shape}  (expected [4, 4096])")
    print(f"Recon  shape : {recon.shape}   (expected [4, 3, 64, 64])")
    print(f"Recon  range : [{recon.min():.3f}, {recon.max():.3f}]  (expected [0, 1])")
    print("deepjscc.py OK [OK]")
