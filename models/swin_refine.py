"""
models/swin_refine.py
---------------------
Stage 3: Post-channel refinement with SwinIR + SNR-aware FiLM conditioning.

Pipeline (all variants):
    1. Frozen base model (e.g., Model 2d) produces preliminary reconstruction I_prelim.
    2. Refiner (SwinIR or CNN) takes I_prelim, optionally conditioned on SNR,
       and produces a residual correction.
    3. Output = clamp(I_prelim + residual, 0, 1).

Three model classes:
    SwinRefineJSCC    – SwinIR + FiLM (γ, β from SNR embedding). PRIMARY.
    SwinRefineNoSNR   – SwinIR without SNR conditioning. ABLATION.
    CNNRefineJSCC     – Plain CNN refiner, parameter-matched to SwinIR. CONTROL.

FiLM injection points (configurable via film_at_rstb):
    - Always at initial shallow feature (after conv_first).
    - Optionally at the output of each RSTB block.

The base model is loaded from a config + checkpoint at construction time.
Its weights are frozen and BatchNorm is kept in eval() mode.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import power_normalize, awgn, rayleigh, sample_snr
from models.network_swinir import SwinIR


# ─────────────────────────────────────────────────────────────────────────────
# FiLM: Feature-wise Linear Modulation
# ─────────────────────────────────────────────────────────────────────────────
class SNR_FiLM(nn.Module):
    """
    Maps scalar SNR (dB) → (γ, β) for Feature-wise Linear Modulation.

    Input:  snr_db (float or Tensor)  →  normalized to snr_db / 20.0
    Output: γ (B, C, 1, 1), β (B, C, 1, 1)

    Applied as: feature = γ * feature + β
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * embed_dim),   # (γ, β)
        )
        # Initialize γ ≈ 1, β ≈ 0  (identity modulation at start)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Set the γ bias to 1 so initial γ ≈ 1
        with torch.no_grad():
            self.mlp[-1].bias[:embed_dim] = 1.0

    def forward(self, snr_db: torch.Tensor, C: int):
        """
        Args:
            snr_db: (B,) or scalar SNR in dB.
            C:      number of channels (embed_dim).
        Returns:
            gamma: (B, C, 1, 1)
            beta:  (B, C, 1, 1)
        """
        # Normalize SNR
        snr_norm = snr_db.float().view(-1, 1) / 20.0   # (B, 1)
        out = self.mlp(snr_norm)                         # (B, 2*C)
        gamma = out[:, :C].unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = out[:, C:].unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# FiLM-augmented SwinIR
# ─────────────────────────────────────────────────────────────────────────────
class FiLM_SwinIR(nn.Module):
    """
    SwinIR with FiLM modulation injected at:
      1. The initial shallow feature (after conv_first).
      2. (Optional) The output of each RSTB block.

    Uses the SwinIR 'denoising' mode (upsampler='', upscale=1):
        x_first = conv_first(x)
        res     = conv_after_body(forward_features(x_first)) + x_first
        output  = x + conv_last(res)

    We intercept after conv_first and after each RSTB to inject FiLM.
    """
    def __init__(self, embed_dim: int = 60, depths=(4, 4, 4, 4),
                 num_heads=(6, 6, 6, 6), window_size: int = 8,
                 film_at_rstb: bool = False, use_snr: bool = True):
        super().__init__()
        self.use_snr = use_snr
        self.film_at_rstb = film_at_rstb

        # SwinIR in denoising mode (no upsampling, residual learning)
        # Default input resolution is 32 (matching training crop size)
        self.swinir = SwinIR(
            img_size=32,
            patch_size=1,
            in_chans=3,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            upscale=1,
            img_range=1.0,
            upsampler='',           # denoising mode
            resi_connection='1conv',
        )

        # Zero-initialize the final output conv so initial residual is exactly zero (identity init)
        nn.init.zeros_(self.swinir.conv_last.weight)
        nn.init.zeros_(self.swinir.conv_last.bias)

        # FiLM modules (only if use_snr)
        if self.use_snr:
            self.film_initial = SNR_FiLM(embed_dim)
            if self.film_at_rstb:
                n_rstb = len(depths)
                self.film_rstb = nn.ModuleList([
                    SNR_FiLM(embed_dim) for _ in range(n_rstb)
                ])

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor = None):
        """
        Args:
            x:      (B, 3, 64, 64) preliminary reconstruction in [0, 1].
            snr_db: (B,) SNR in dB. Required if use_snr=True.
        Returns:
            refined: (B, 3, 64, 64) in [0, 1] (clamped).
        """
        H, W = x.shape[2:]
        x_in = self.swinir.check_image_size(x)

        self.swinir.mean = self.swinir.mean.type_as(x_in)
        x_norm = (x_in - self.swinir.mean) * self.swinir.img_range

        # 1. Shallow feature extraction
        x_first = self.swinir.conv_first(x_norm)         # (B, embed_dim, H, W)

        # 2. FiLM at initial feature
        if self.use_snr and snr_db is not None:
            B = x_first.size(0)
            if not isinstance(snr_db, torch.Tensor):
                snr_db = torch.tensor([snr_db] * B, device=x_first.device)
            elif snr_db.dim() == 0:
                snr_db = snr_db.unsqueeze(0).expand(B)
            gamma, beta = self.film_initial(snr_db, x_first.size(1))
            x_first = gamma * x_first + beta

        # 3. Deep feature extraction with optional per-RSTB FiLM
        x_size = (x_first.shape[2], x_first.shape[3])
        feat = self.swinir.patch_embed(x_first)
        if self.swinir.ape:
            feat = feat + self.swinir.absolute_pos_embed
        feat = self.swinir.pos_drop(feat)

        for i, layer in enumerate(self.swinir.layers):
            feat = layer(feat, x_size)
            # FiLM after each RSTB
            if self.use_snr and self.film_at_rstb and snr_db is not None:
                # feat is (B, L, C) in token form; reshape to spatial for FiLM
                B_f, L, C = feat.shape
                feat_spatial = feat.transpose(1, 2).view(B_f, C, x_size[0], x_size[1])
                gamma_r, beta_r = self.film_rstb[i](snr_db, C)
                feat_spatial = gamma_r * feat_spatial + beta_r
                feat = feat_spatial.flatten(2).transpose(1, 2)

        feat = self.swinir.norm(feat)
        feat = self.swinir.patch_unembed(feat, x_size)

        # 4. Residual connection
        res = self.swinir.conv_after_body(feat) + x_first
        out = x_norm + self.swinir.conv_last(res)

        # 5. Denormalize
        out = out / self.swinir.img_range + self.swinir.mean
        out = out[:, :, :H, :W]

        return out


# ─────────────────────────────────────────────────────────────────────────────
# CNN Refiner (control, parameter-matched)
# ─────────────────────────────────────────────────────────────────────────────
class CNNRefiner(nn.Module):
    """
    Plain CNN residual refiner. Parameter count tunable via `mid_ch` and `n_blocks`.
    Architecture:
        conv_in(3 -> mid_ch) -> n_blocks * [Conv-BN-PReLU-Conv-BN + skip] -> conv_out(mid_ch -> 3)
    Output: x_in + conv_out(features)  (residual learning, clamped to [0,1])
    """
    def __init__(self, mid_ch: int = 64, n_blocks: int = 9):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.PReLU(),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(self._res_block(mid_ch))
        self.blocks = nn.ModuleList(blocks)

        self.conv_out = nn.Conv2d(mid_ch, 3, 3, 1, 1)

        # Zero-initialize the final output conv so initial residual is exactly zero (identity init)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    @staticmethod
    def _res_block(ch: int):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor, snr_db=None):
        """snr_db is accepted but ignored (control model, no SNR conditioning)."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = h + block(h)   # residual skip
        residual = self.conv_out(h)
        return x + residual    # output clamped by caller


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load and freeze a base model
# ─────────────────────────────────────────────────────────────────────────────
def _load_frozen_base(base_config_path: str, base_checkpoint_path: str, device):
    """
    Load a base JSCC model from config + checkpoint, freeze all parameters,
    and set to eval mode with BatchNorm frozen.
    """
    from train import load_config, build_model

    base_cfg = load_config(base_config_path)
    base_model = build_model(base_cfg, device)

    if os.path.isfile(base_checkpoint_path):
        ckpt = torch.load(base_checkpoint_path, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        print(f"  [Base] Loaded checkpoint: {base_checkpoint_path}")
    else:
        print(f"  [WARNING] Base checkpoint not found: {base_checkpoint_path}")

    # Freeze everything
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()

    return base_model, base_cfg


def _needs_imgnet_base(base_cfg: dict) -> bool:
    """Check if the base model needs ImageNet-normalised input."""
    return base_cfg["model"]["name"] in (
        "ResNetJSCC", "ResNetJSCC_2b", "ResNetJSCC_2c",
        "ResNetJSCC_2d", "ResNetJSCC_2e",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Wrapper: SwinRefineJSCC
# ─────────────────────────────────────────────────────────────────────────────
class SwinRefineJSCC(nn.Module):
    """
    Stage 3 primary model: Frozen base JSCC → SwinIR refiner with FiLM SNR conditioning.

    The base model's forward(img, channel, snr_db) produces a preliminary reconstruction.
    The SwinIR refiner takes this + SNR → refined reconstruction.

    Config keys (in YAML):
        model:
          name: "SwinRefineJSCC"
          base_config: "configs/model2d.yaml"
          base_checkpoint: "checkpoints/model2d/best.pt"
          refiner: "swinir"          # "swinir" | "swinir_nosnr" | "cnn"
          swinir_embed_dim: 60
          swinir_depths: [6, 6, 6, 6]
          swinir_num_heads: [6, 6, 6, 6]
          swinir_window_size: 8
          film_at_rstb: false
          cnn_mid_ch: 64
          cnn_n_blocks: 8
          random_channel: false      # if true, sample channel per batch
    """
    def __init__(self, channel: str = 'awgn', snr_min: float = 0.0, snr_max: float = 20.0,
                 base_config: str = "configs/model2d.yaml",
                 base_checkpoint: str = "checkpoints/model2d/best.pt",
                 refiner: str = "swinir",
                 swinir_embed_dim: int = 60,
                 swinir_depths: tuple = (4, 4, 4, 4),
                 swinir_num_heads: tuple = (6, 6, 6, 6),
                 swinir_window_size: int = 8,
                 film_at_rstb: bool = False,
                 cnn_mid_ch: int = 64,
                 cnn_n_blocks: int = 9,
                 random_channel: bool = False,
                 device=None):
        super().__init__()
        self.default_channel = channel
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.random_channel = random_channel
        self.refiner_type = refiner

        # Load and freeze the base model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model, self.base_cfg = _load_frozen_base(
            base_config, base_checkpoint, device
        )
        self._base_needs_imgnet = _needs_imgnet_base(self.base_cfg)

        # Build the refiner
        if refiner == "swinir":
            self.refiner = FiLM_SwinIR(
                embed_dim=swinir_embed_dim,
                depths=swinir_depths,
                num_heads=swinir_num_heads,
                window_size=swinir_window_size,
                film_at_rstb=film_at_rstb,
                use_snr=True,
            )
        elif refiner == "swinir_nosnr":
            self.refiner = FiLM_SwinIR(
                embed_dim=swinir_embed_dim,
                depths=swinir_depths,
                num_heads=swinir_num_heads,
                window_size=swinir_window_size,
                film_at_rstb=False,
                use_snr=False,
            )
        elif refiner == "cnn":
            self.refiner = CNNRefiner(mid_ch=cnn_mid_ch, n_blocks=cnn_n_blocks)
        else:
            raise ValueError(f"Unknown refiner type: {refiner!r}")

    def train(self, mode: bool = True):
        """Keep the base model frozen and in eval mode."""
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, img: torch.Tensor, channel: str = None, snr_db: float = None, crop_box: tuple = None):
        """
        Args:
            img:      (B, 3, 64, 64) — ImageNet-normalised if base needs it, or [0,1].
            channel:  "awgn" | "rayleigh" | None (uses default).
            snr_db:   float SNR in dB, or None (sampled from [snr_min, snr_max]).
            crop_box: optional (top, left, height, width) for synchronized training patch cropping.

        Returns:
            recon:   (B, 3, H, W) in [0, 1] — cropped if crop_box is provided, else (B, 3, 64, 64).
            z_norm:  power-normalised latent from base model.
        """
        ch = channel if channel is not None else self.default_channel

        # During training with random_channel, flip a coin
        if self.training and self.random_channel and channel is None:
            ch = "rayleigh" if torch.rand(1).item() > 0.5 else "awgn"

        if snr_db is None:
            snr_db = sample_snr(self.snr_min, self.snr_max)

        # 1. Base model produces preliminary reconstruction (no gradients, full 64x64)
        with torch.no_grad():
            i_prelim, z_norm = self.base_model(img, channel=ch, snr_db=snr_db)
            i_prelim = i_prelim.detach()

        # 2. Refiner: residual correction conditioned on SNR
        B = i_prelim.size(0)
        snr_tensor = torch.tensor([snr_db] * B, device=i_prelim.device, dtype=torch.float32)

        if crop_box is not None:
            top, left, h, w = crop_box
            refine_in = i_prelim[:, :, top:top+h, left:left+w]
        else:
            refine_in = i_prelim

        if self.refiner_type in ("swinir", "swinir_nosnr"):
            refined = self.refiner(refine_in, snr_db=snr_tensor)
        else:
            # CNN refiner (ignores snr_db)
            refined = self.refiner(refine_in, snr_db=snr_tensor)

        # 3. Clamp to [0, 1]
        recon = refined.clamp(0.0, 1.0)

        return recon, z_norm

    def param_groups(self, lr: float = 1e-4, swinir_lr: float = None, wd: float = 1e-4):
        """
        Parameter groups for the optimizer.
        Only the refiner parameters are trainable (base is frozen).
        """
        if swinir_lr is None:
            swinir_lr = lr
        refiner_params = [p for p in self.refiner.parameters() if p.requires_grad]
        return [
            {"params": refiner_params, "lr": swinir_lr, "weight_decay": wd},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting utility
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def print_param_table(model: SwinRefineJSCC):
    """Print a detailed parameter breakdown."""
    counts = count_parameters(model)
    base_counts = count_parameters(model.base_model)
    refiner_counts = count_parameters(model.refiner)

    print("\n" + "=" * 65)
    print("  Parameter Breakdown")
    print("=" * 65)
    print(f"  {'Component':<30s} {'Trainable':>12s} {'Frozen':>12s} {'Total':>12s}")
    print("-" * 65)
    print(f"  {'Base model (frozen)':<30s} {base_counts['trainable']:>12,} {base_counts['frozen']:>12,} {base_counts['total']:>12,}")
    print(f"  {'Refiner':<30s} {refiner_counts['trainable']:>12,} {refiner_counts['frozen']:>12,} {refiner_counts['total']:>12,}")
    print("-" * 65)
    print(f"  {'TOTAL':<30s} {counts['trainable']:>12,} {counts['frozen']:>12,} {counts['total']:>12,}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing FiLM_SwinIR parameter count (depths [4,4,4,4])...")
    swinir_film = FiLM_SwinIR(embed_dim=60, depths=(4, 4, 4, 4),
                               num_heads=(6, 6, 6, 6), window_size=8,
                               film_at_rstb=False, use_snr=True)
    swinir_nosnr = FiLM_SwinIR(embed_dim=60, depths=(4, 4, 4, 4),
                                num_heads=(6, 6, 6, 6), window_size=8,
                                film_at_rstb=False, use_snr=False)
    swinir_film_rstb = FiLM_SwinIR(embed_dim=60, depths=(4, 4, 4, 4),
                                    num_heads=(6, 6, 6, 6), window_size=8,
                                    film_at_rstb=True, use_snr=True)
    cnn = CNNRefiner(mid_ch=64, n_blocks=9)

    for name, m in [("SwinIR+FiLM (initial)", swinir_film),
                    ("SwinIR+FiLM (initial+RSTB)", swinir_film_rstb),
                    ("SwinIR (no SNR)", swinir_nosnr),
                    ("CNN Refiner (param-matched)", cnn)]:
        c = count_parameters(m)
        print(f"  {name:<35s}  trainable={c['trainable']:>10,}  total={c['total']:>10,}")

    # Forward pass & identity init test
    print("\nForward pass & identity initialization check:")
    x = torch.randn(2, 3, 64, 64).clamp(0, 1)
    snr = torch.tensor([10.0, 15.0])
    with torch.no_grad():
        out_swin = swinir_film(x, snr_db=snr)
        out_cnn = cnn(x)
        diff_swin = (out_swin - x).abs().max().item()
        diff_cnn = (out_cnn - x).abs().max().item()

    print(f"  SwinIR max diff vs input at init : {diff_swin:.6f} (must be 0.000000)")
    print(f"  CNN    max diff vs input at init : {diff_cnn:.6f} (must be 0.000000)")
    assert diff_swin < 1e-6, "SwinIR failed identity initialization check!"
    assert diff_cnn < 1e-6, "CNN failed identity initialization check!"
    print("swin_refine.py OK - Identity Initialization Verified")
