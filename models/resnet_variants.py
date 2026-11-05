"""
models/resnet_variants.py
-------------------------
Architectural variants for Model 2 to address the Layer-4 2×2 spatial bottleneck:

1. ResNetJSCC_2c (Primary Fix):
   - Features from Layer-3 (256×4×4 = 4096 reals, matching ρ=1/6 capacity).
   - Conv2d(256→256) projection (no dense 4096×4096 memorization).
   - Transmit: 4096 reals = 2048 complex symbols.
   - Receiver: Conv2d(256→256) + Dropout(0.1) + CNNDecoder.
   - ResNet backbone is frozen (BN in eval mode).

2. ResNetJSCC_2e (Upsampling Fix):
   - Input to ResNet is bilinearly upsampled to 128×128.
   - Target image remains 64×64.
   - Features from Layer-4 on 128×128 (512×4×4 = 8192 reals).
   - Conv2d(512→256) projection to 4096 reals.
   - Transmit: 4096 reals.
   - Receiver: Conv2d(256→256) + Dropout(0.1) + CNNDecoder.
   - ResNet backbone is frozen (BN in eval mode).

3. ResNetJSCC_2d (Representation Fine-tuning):
   - Architecture identical to 2c (Layer-3 features, 4096 reals).
   - Backbone layer3 is unfrozen and trained with lr = 1e-4.
   - Layers conv1, bn1, layer1, layer2 remain frozen.
   - BatchNorm layers strictly kept in eval() mode.

4. ResNetJSCC_2b (Control: Regularized Layer-4):
   - Architecture identical to original Model 2 (Layer-4, 512×2×2 = 2048 reals).
   - Linear(2048→4096) encoder, Linear(4096→4096) decoder.
   - Receiver-side Dropout(0.1) between dense layers.
   - ResNet backbone is frozen (BN in eval mode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from channels import power_normalize, awgn, rayleigh, sample_snr
from models.deepjscc import deconv_block


# ─────────────────────────────────────────────────────────────────────────────
# Reusable CNN Decoder (outputs 3×64×64 in [0, 1])
# ─────────────────────────────────────────────────────────────────────────────
class CNNDecoder(nn.Module):
    """Upsamples (B, 4096) -> (B, 256, 4, 4) -> (B, 3, 64, 64) in [0, 1]."""
    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(1, (256, 4, 4))
        self.net = nn.Sequential(
            deconv_block(256, 128, stride=2),             # (B, 128,  8,  8)
            deconv_block(128, 64,  stride=2),             # (B,  64, 16, 16)
            deconv_block(64,  32,  stride=2),             # (B,  32, 32, 32)
            deconv_block(32,  3,   stride=2, last=True),  # (B,   3, 64, 64) [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            return self.net(z)
        return self.net(self.unflatten(z))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model 2c: Layer-3 features (256×4×4 = 4096 values) + Conv projection
# ─────────────────────────────────────────────────────────────────────────────
class ResNetFeatureExtractorLayer3(nn.Module):
    """
    Pretrained ResNet-18 truncated after Layer-3.
    Input: (B, 3, 64, 64) -> Output: (B, 256, 4, 4) = 4,096 values.
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        if freeze:
            for p in self.features.parameters():
                p.requires_grad_(False)
        self.features.eval()

    def train(self, mode: bool = True):
        # Keep BatchNorm permanently in eval mode
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 256, 4, 4)


class ResNetJSCC_2c(nn.Module):
    """
    Model 2c: ResNet-18 Layer-3 (frozen) + Conv2d JSCC Encoder + Channel
              + Conv2d JSCC Decoder (receiver Dropout 0.1) + CNN Decoder.
    """
    def __init__(self, channel: str = 'awgn', snr_min: float = 0.0, snr_max: float = 20.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractorLayer3(freeze=True)

        # Sender JSCC Encoder: NO dropout before channel (train/test mismatch prevention)
        self.jscc_encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        # Receiver JSCC Decoder: Dropout(0.1) placed here on the receiver side
        self.jscc_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout2d(0.1),
        )

        self.cnn_decoder = CNNDecoder()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None) -> tuple:
        ch = channel if channel is not None else self.default_channel

        # 1. Extract Layer-3 features (frozen)
        with torch.no_grad():
            feat = self.backbone(img)                   # (B, 256, 4, 4)

        # 2. JSCC encode: conv projection (no dropout)
        h = self.jscc_encoder(feat)                     # (B, 256, 4, 4)
        z = h.view(h.size(0), -1)                       # (B, 4096)

        # 3. Power normalise
        z_norm = power_normalize(z)

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
            raise ValueError(f"Unknown channel: {ch!r}")

        # 6. JSCC decode (receiver conv + dropout 0.1)
        z_spatial = z_rx.view(z_rx.size(0), 256, 4, 4)
        z_dec = self.jscc_decoder(z_spatial)            # (B, 256, 4, 4)

        # 7. CNN decode
        recon = self.cnn_decoder(z_dec.view(z_dec.size(0), -1))  # (B, 3, 64, 64)
        return recon, z_norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model 2e: 128×128 Upsampled Input + Layer-4 (512×4×4 = 8192) -> 4096
# ─────────────────────────────────────────────────────────────────────────────
class ResNetJSCC_2e(nn.Module):
    """
    Model 2e: Input upsampled to 128×128 -> ResNet-18 Layer-4 (512×4×4 = 8192)
              -> Conv2d(512→256) projection to 4096 -> Channel -> Conv Decoder -> CNN Decoder.
    """
    def __init__(self, channel: str = 'awgn', snr_min: float = 0.0, snr_max: float = 20.0):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()

        # Sender JSCC Encoder: Conv2d(512 -> 256) compresses 8192 -> 4096 (no dropout)
        self.jscc_encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        # Receiver JSCC Decoder: with receiver Dropout(0.1)
        self.jscc_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout2d(0.1),
        )

        self.cnn_decoder = CNNDecoder()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()   # keep backbone BN permanently in eval mode
        return self

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None) -> tuple:
        ch = channel if channel is not None else self.default_channel

        # 1. Bilinear upsample from 64×64 to 128×128 for the ResNet branch only
        with torch.no_grad():
            img_up = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=False)
            feat = self.features(img_up)                 # (B, 512, 4, 4) = 8,192 values

        # 2. JSCC encode: conv projection 512 -> 256 (no dropout)
        h = self.jscc_encoder(feat)                     # (B, 256, 4, 4)
        z = h.view(h.size(0), -1)                       # (B, 4096)

        # 3. Power normalise
        z_norm = power_normalize(z)

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
            raise ValueError(f"Unknown channel: {ch!r}")

        # 6. JSCC decode (receiver conv + dropout 0.1)
        z_spatial = z_rx.view(z_rx.size(0), 256, 4, 4)
        z_dec = self.jscc_decoder(z_spatial)            # (B, 256, 4, 4)

        # 7. CNN decode (target is 64×64)
        recon = self.cnn_decoder(z_dec.view(z_dec.size(0), -1))  # (B, 3, 64, 64)
        return recon, z_norm


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model 2d: Layer-3 with Unfrozen layer3 (dual learning rate)
# ─────────────────────────────────────────────────────────────────────────────
class ResNetJSCC_2d(nn.Module):
    """
    Model 2d: Like 2c, but unfreezes ResNet layer3 with lr = 1e-4.
    BatchNorm running stats strictly kept in eval mode.
    """
    def __init__(self, channel: str = 'awgn', snr_min: float = 0.0, snr_max: float = 20.0):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features_frozen = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        for p in self.features_frozen.parameters():
            p.requires_grad_(False)

        # Unfrozen layer3
        self.layer3 = backbone.layer3
        for p in self.layer3.parameters():
            p.requires_grad_(True)

        # Keep BN in eval mode
        self.features_frozen.eval()
        self.layer3.eval()

        # Sender JSCC Encoder: NO dropout
        self.jscc_encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        # Receiver JSCC Decoder: with receiver Dropout(0.1)
        self.jscc_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout2d(0.1),
        )

        self.cnn_decoder = CNNDecoder()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

    def train(self, mode: bool = True):
        super().train(mode)
        # Always force backbone BatchNorm layers into eval mode
        self.features_frozen.eval()
        for m in self.layer3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def param_groups(self, lr: float = 1e-3, lr_backbone: float = 1e-4, wd: float = 1e-4):
        """Dual learning rate parameter groups for train.py."""
        backbone_params = [p for p in self.layer3.parameters() if p.requires_grad]
        jscc_params = (
            list(self.jscc_encoder.parameters()) +
            list(self.jscc_decoder.parameters()) +
            list(self.cnn_decoder.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd},
            {"params": jscc_params,     "lr": lr,          "weight_decay": wd},
        ]

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None) -> tuple:
        ch = channel if channel is not None else self.default_channel

        # 1. Feature extraction
        with torch.no_grad():
            f2 = self.features_frozen(img)              # (B, 128, 8, 8)
        feat = self.layer3(f2)                          # (B, 256, 4, 4) - gradients flow here

        # 2. JSCC encode (no dropout)
        h = self.jscc_encoder(feat)                     # (B, 256, 4, 4)
        z = h.view(h.size(0), -1)                       # (B, 4096)

        # 3. Power normalise
        z_norm = power_normalize(z)

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
            raise ValueError(f"Unknown channel: {ch!r}")

        # 6. JSCC decode (receiver conv + dropout 0.1)
        z_spatial = z_rx.view(z_rx.size(0), 256, 4, 4)
        z_dec = self.jscc_decoder(z_spatial)

        # 7. CNN decode
        recon = self.cnn_decoder(z_dec.view(z_dec.size(0), -1))
        return recon, z_norm


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model 2b: Control — Layer-4 with Receiver Dropout(0.1)
# ─────────────────────────────────────────────────────────────────────────────
class JSCCDecoderMLP_Reg(nn.Module):
    """Linear(4096->4096) + Dropout(0.1) + PReLU on receiver side."""
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1),  # receiver-side dropout
            nn.PReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResNetJSCC_2b(nn.Module):
    """
    Model 2b: Original Model 2 (Layer-4, 512×2×2 = 2048) with receiver Dropout(0.1).
    Used as an experimental control to isolate the effect of regularization vs feature riching.
    """
    def __init__(self, channel: str = 'awgn', snr_min: float = 0.0, snr_max: float = 20.0):
        super().__init__()
        from models.resnet_jscc import ResNetFeatureExtractor, JSCCEncoderMLP
        self.backbone = ResNetFeatureExtractor()
        self.jscc_encoder = JSCCEncoderMLP()       # Linear(2048->4096), no dropout
        self.jscc_decoder = JSCCDecoderMLP_Reg()   # Linear(4096->4096) + Dropout(0.1)
        self.cnn_decoder  = CNNDecoder()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None) -> tuple:
        ch = channel if channel is not None else self.default_channel
        feat = self.backbone(img)
        z = self.jscc_encoder(feat)
        z_norm = power_normalize(z)

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

        z_dec = self.jscc_decoder(z_rx)
        recon = self.cnn_decoder(z_dec)
        return recon, z_norm


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Exact Parameter Counter
# ─────────────────────────────────────────────────────────────────────────────
def get_param_counts(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"trainable": trainable, "frozen": frozen, "total": total}
