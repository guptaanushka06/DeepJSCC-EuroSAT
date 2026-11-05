"""
models/resnet_jscc.py
---------------------
Model 2: ResNet-18 + DeepJSCC + CNN Decoder (NO SwinIR)

Architecture (sender → channel → receiver)
------------------------------------------

SENDER
  Input: img_imgnet (B, 3, 64, 64)  — ImageNet-normalised
  → Frozen pretrained ResNet-18, features from Layer-4 (BEFORE avgpool)
    Output: (B, 512, 2, 2)  = 2,048 values
  → Flatten: (B, 2048)
  → JSCC Encoder MLP: Linear(2048, 4096) + PReLU
    Output: (B, 4096)  ← the channel input (2,048 complex symbols, ρ=1/6)
  → power_normalize: avg power per complex symbol = 1

CHANNEL
  AWGN or Rayleigh (same code as Model 1)

RECEIVER
  → JSCC Decoder MLP: Linear(4096, 4096) + PReLU
  → Unflatten: (B, 256, 4, 4)
  → CNN Decoder (SAME as Model 1): 4 transposed conv blocks → (B, 3, 64, 64)
  → Sigmoid → pixels in [0, 1]

Why this design?
  The ResNet-18 Layer-4 features on 64×64 input are only 512×2×2 = 2,048 values.
  The linear projection doubles this to 4,096 to match the ρ=1/6 channel capacity.
  This means the bottleneck is the ResNet feature map, NOT the channel.
  Expected result: PSNR lower than Model 1, which is the intended baseline for
  showing that SwinIR (Model 3) can recover the lost quality.

Note on input
  This model needs img_imgnet (ImageNet-normalised) as encoder input, but
  the loss is computed against img_unit ([0,1]).  The data loader returns both.
  The train.py routing (_needs_imgnet) handles this automatically.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from channels import power_normalize, awgn, rayleigh, sample_snr
from models.deepjscc import deconv_block   # reuse the same decoder blocks

LATENT_DIM = 4096   # 2 × 2048 complex symbols  (ρ = 1/6)
RESNET_FEAT = 2048  # 512 × 2 × 2  (Layer-4 on 64×64 input)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 feature extractor (frozen)
# ─────────────────────────────────────────────────────────────────────────────
class ResNetFeatureExtractor(nn.Module):
    """
    Frozen pretrained ResNet-18, output = Layer-4 spatial features.

    For a 64×64 input the spatial sizes are:
        conv1 (stride 2): 32×32
        maxpool (stride 2): 16×16
        layer1: 16×16
        layer2 (stride 2): 8×8
        layer3 (stride 2): 4×4
        layer4 (stride 2): 2×2  ← we stop here (before avgpool)

    Output: (B, 512, 2, 2) — flattened to (B, 2048) downstream.
    """

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Keep everything up to (and including) layer4, drop avgpool and fc
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Freeze all parameters — this model is never updated
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.features.eval()   # keep BN in eval mode permanently

    def train(self, mode: bool = True):
        # Always stay in eval mode (frozen backbone)
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)           # (B, 512, 2, 2)
        return feat.view(feat.size(0), -1)    # (B, 2048)


# ─────────────────────────────────────────────────────────────────────────────
# JSCC Encoder (projects ResNet features → channel input)
# ─────────────────────────────────────────────────────────────────────────────
class JSCCEncoderMLP(nn.Module):
    """
    Small MLP that maps ResNet features to the channel latent.
    Linear(2048 → 4096) + PReLU.
    """

    def __init__(self, in_dim: int = RESNET_FEAT, out_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 4096)


# ─────────────────────────────────────────────────────────────────────────────
# JSCC Decoder (maps received symbols → feature map)
# ─────────────────────────────────────────────────────────────────────────────
class JSCCDecoderMLP(nn.Module):
    """
    Small MLP that maps received symbols back to a feature map.
    Linear(4096 → 4096) + PReLU.
    """

    def __init__(self, dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.PReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (B, 4096)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Decoder (same as Model 1)
# ─────────────────────────────────────────────────────────────────────────────
class CNNDecoder(nn.Module):
    """
    Upsampling CNN decoder: (B, 4096) → (B, 3, 64, 64).
    Identical to Model 1's JSCCDecoder so results are comparable.
    """

    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(1, (256, 4, 4))
        self.net = nn.Sequential(
            deconv_block(256, 128, stride=2),             # (B, 128,  8,  8)
            deconv_block(128, 64,  stride=2),             # (B,  64, 16, 16)
            deconv_block(64,  32,  stride=2),             # (B,  32, 32, 32)
            deconv_block(32,  3,   stride=2, last=True),  # (B,   3, 64, 64) [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            return self.net(z)
        return self.net(self.unflatten(z))


# ─────────────────────────────────────────────────────────────────────────────
# Full Model 2
# ─────────────────────────────────────────────────────────────────────────────
class ResNetJSCC(nn.Module):
    """
    Model 2: ResNet-18 (frozen) + JSCC Encoder MLP + Channel + JSCC Decoder MLP
             + CNN Decoder.

    Forward signature:
        recon, latent = model(img_imgnet, channel='awgn', snr_db=10.0)

    Args:
        img_imgnet : (B, 3, 64, 64)  ImageNet-normalised
        channel    : 'awgn' or 'rayleigh'
        snr_db     : float or None (sample randomly if None)

    Returns:
        recon  : (B, 3, 64, 64)  in [0, 1]
        latent : (B, 4096)       power-normalised symbols (for monitoring)
    """

    def __init__(self,
                 channel: str  = 'awgn',
                 snr_min: float = 0.0,
                 snr_max: float = 20.0):
        super().__init__()
        self.backbone        = ResNetFeatureExtractor()
        self.jscc_encoder    = JSCCEncoderMLP()
        self.jscc_decoder    = JSCCDecoderMLP()
        self.cnn_decoder     = CNNDecoder()
        self.default_channel = channel
        self.snr_min         = snr_min
        self.snr_max         = snr_max

    # ── Expose parameter groups for train.py (dual LR is not needed here,
    #    but train.py checks for this method) ──────────────────────────────
    # No param_groups() override needed — train.py will use the default
    # Adam over all requires_grad=True parameters (backbone is frozen).

    def forward(self,
                img: torch.Tensor,
                channel: str  = None,
                snr_db: float = None) -> tuple:

        ch = channel if channel is not None else self.default_channel

        # 1. Extract frozen ResNet-18 Layer-4 features
        feat = self.backbone(img)              # (B, 2048)

        # 2. JSCC encode: project to channel dimension
        z = self.jscc_encoder(feat)            # (B, 4096)

        # 3. Power normalise
        z_norm = power_normalize(z)            # avg power per complex sym = 1

        # 4. Sample SNR if not given
        if snr_db is None:
            snr_db = sample_snr(self.snr_min, self.snr_max)

        # 5. Channel
        if ch == 'awgn':
            z_rx = awgn(z_norm, snr_db)
        elif ch == 'rayleigh':
            z_rx, _ = rayleigh(z_norm, snr_db)
        elif ch in ('none', 'bypass'):
            z_rx = z_norm
        else:
            raise ValueError(f"Unknown channel: {ch!r}. Use 'awgn', 'rayleigh', or 'bypass'.")

        # 6. JSCC decode: received symbols → feature map
        z_dec = self.jscc_decoder(z_rx)        # (B, 4096)

        # 7. CNN decode: feature map → image
        recon = self.cnn_decoder(z_dec)        # (B, 3, 64, 64) in [0,1]

        return recon, z_norm


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNetJSCC().to(device)

    # Parameter count (backbone is frozen, so count trainable only)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {n_total:,}")
    print(f"Trainable parameters : {n_trainable:,}  (backbone frozen)")

    # Verify backbone is truly frozen
    backbone_grads = any(p.requires_grad for p in model.backbone.parameters())
    assert not backbone_grads, "Backbone should be frozen!"
    print("Backbone frozen [OK]")

    # Forward pass
    dummy = torch.rand(4, 3, 64, 64, device=device)    # img_imgnet
    recon, latent = model(dummy, channel='awgn', snr_db=10.0)
    print(f"Input  shape : {dummy.shape}")
    print(f"Latent shape : {latent.shape}  (expected [4, 4096])")
    print(f"Recon  shape : {recon.shape}   (expected [4, 3, 64, 64])")
    print(f"Recon  range : [{recon.min():.3f}, {recon.max():.3f}]  (expected [0,1])")

    # Verify latent power = 1
    power = latent.pow(2).mean(dim=1) * 2.0
    print(f"Avg power per complex sym: {power.mean():.6f}  (expected ≈ 1.0)")

    print("resnet_jscc.py OK [OK]")
