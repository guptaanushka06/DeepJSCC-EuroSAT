"""
backend/inference.py
--------------------
Unified inference engine for DeepJSCC EuroSAT models.

Provides a clean, modular API for deployment on Streamlit, Gradio, or Hugging Face Spaces:
  - Zero dependencies on training loops or scripts.
  - Supports all baseline, semantic refiner, and SwinJSCC architectures.
  - Caches loaded checkpoints for instant sub-millisecond repeated inferences.
  - Computes PSNR, SSIM, and dual classifier semantic predictions (ResNet-18 & MobileNet-V2).
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Any, Union, Optional, Tuple

# Add root project path to import model architectures and channels
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from channels import power_normalize, awgn, rayleigh
from classifier import build_classifier

# Class labels for EuroSAT 10-class dataset
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVeg", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Standard ImageNet normalization parameters for classifier backbones
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Default model catalog and checkpoints
AVAILABLE_MODELS = {
    "Model 1: PlainDeepJSCC (Fair Baseline)": {
        "type": "plain",
        "ckpt": "checkpoints/model1_fair/best.pt",
        "config": "configs/model1_fair.yaml",
        "description": "Standard convolutional autoencoder trained with pure MSE (30 epochs)."
    },
    "Model 2d: ResNetJSCC (Layer-3 Tap)": {
        "type": "resnet",
        "ckpt": "checkpoints/model2d/best.pt",
        "config": "configs/model2d.yaml",
        "description": "Hybrid ResNet-18 backbone tapping Layer 3 to match the 4,096 channel budget."
    },
    "Model 2d Long: ResNetJSCC (60 Epochs)": {
        "type": "resnet",
        "ckpt": "checkpoints/model2d_long/best.pt",
        "config": "configs/model2d_long.yaml",
        "description": "Extended 60-epoch baseline proving convergence."
    },
    "Model 3: SwinIR Refiner": {
        "type": "swin_refine",
        "ckpt": "checkpoints/model3_swinir/best.pt",
        "config": "configs/model3_swinir.yaml",
        "description": "Frozen Model 2d + SwinIR residual enhancement module."
    },
    "Model 3b: Semantic Refiner (λ=0.0003, Seed 42)": {
        "type": "swin_refine",
        "ckpt": "checkpoints/model3b_lam0.0003/best.pt",
        "config": "configs/model3b_lam0.0003.yaml",
        "description": "Semantic-aware refiner offering the preferred trade-off under evaluated conditions (λ=0.0003)."
    },
    "Model 3b: Semantic Refiner (λ=0.001)": {
        "type": "swin_refine",
        "ckpt": "checkpoints/model3b_lam0.001/best.pt",
        "config": "configs/model3b_lam0.001.yaml",
        "description": "Semantic-aware refiner trained with lambda=0.001."
    },
    "Model 4: Full SwinJSCC (Transformer End-to-End)": {
        "type": "swin_jscc",
        "ckpt": "checkpoints/model4/best.pt",
        "config": "configs/model4.yaml",
        "description": "Stage 4 end-to-end Vision Transformer JSCC with shifted-window self-attention."
    },
    "Model 4-NoShift: SwinJSCC Ablation (W-MSA Only)": {
        "type": "swin_jscc_noshift",
        "ckpt": "checkpoints/model4_noshift/best.pt",
        "config": "configs/model4_noshift.yaml",
        "description": "SwinJSCC ablation isolating the effect of shifted windows."
    },
    "Model 4-Conv: Non-Swin Baseline (ConvJSCC)": {
        "type": "conv_jscc",
        "ckpt": "checkpoints/model4_conv/best.pt",
        "config": "configs/model4_conv.yaml",
        "description": "Parameter-matched (~2.59M) convolutional residual baseline under identical 4,096 bottleneck."
    }
}


def compute_psnr(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """Computes PSNR in dB for tensors in range [0, 1]."""
    mse = torch.mean((orig - recon) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_quick(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """Analytical structural similarity metric for RGB images."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(orig.device)
        return metric(recon, orig).item()
    except Exception:
        # Fallback simple SSIM formulation if torchmetrics is unavailable
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        mu_x = torch.mean(orig)
        mu_y = torch.mean(recon)
        sigma_x = torch.var(orig)
        sigma_y = torch.var(recon)
        sigma_xy = torch.mean((orig - mu_x) * (recon - mu_y))
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return float(ssim.item())


class DeepJSCCInferenceEngine:
    """Unified inference engine for all DeepJSCC models."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.loaded_models: Dict[str, nn.Module] = {}
        self.clf_resnet18: Optional[nn.Module] = None
        self.clf_mobilenet: Optional[nn.Module] = None
        
        # Load classifiers once
        self._init_classifiers()

    def _init_classifiers(self):
        """Loads both frozen evaluation classifiers into memory."""
        # Primary ResNet-18
        p_path = os.path.join(PROJECT_ROOT, "checkpoints", "classifier", "best.pt")
        if os.path.isfile(p_path):
            try:
                clf1 = build_classifier(arch="resnet18")
                state = torch.load(p_path, map_location=self.device)
                clf1.load_state_dict(state["model"])
                clf1.to(self.device).eval()
                for p in clf1.parameters():
                    p.requires_grad_(False)
                self.clf_resnet18 = clf1
            except Exception as e:
                print(f"[Warning] Failed to load ResNet-18 classifier: {e}")

        # Secondary MobileNet-V2
        s_path = os.path.join(PROJECT_ROOT, "checkpoints", "classifier2", "best.pt")
        if os.path.isfile(s_path):
            try:
                clf2 = build_classifier(arch="mobilenet_v2")
                state = torch.load(s_path, map_location=self.device)
                clf2.load_state_dict(state["model"])
                clf2.to(self.device).eval()
                for p in clf2.parameters():
                    p.requires_grad_(False)
                self.clf_mobilenet = clf2
            except Exception as e:
                print(f"[Warning] Failed to load MobileNet-V2 classifier: {e}")

    def load_model(self, model_key: str) -> nn.Module:
        """Loads and caches a JSCC model checkpoint using its canonical config."""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key: {model_key!r}. Available: {list(AVAILABLE_MODELS.keys())}")

        info = AVAILABLE_MODELS[model_key]
        ckpt_path = os.path.join(PROJECT_ROOT, info["ckpt"])
        config_path = os.path.join(PROJECT_ROOT, info["config"])
        
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        import yaml
        from train import build_model
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model = build_model(cfg, self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state["model"])
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.loaded_models[model_key] = model
        return model

    def preprocess_image(self, img_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Loads and preprocesses any image into normalized [0, 1] tensor of shape (1, 3, 64, 64).
        """
        if isinstance(img_input, str):
            if not os.path.isfile(img_input):
                raise FileNotFoundError(f"Image file does not exist: {img_input}")
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                img_input = (np.clip(img_input, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        # Resize to EuroSAT native 64x64 resolution using high-quality Lanczos interpolation
        if pil_img.size != (64, 64):
            pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)

        arr = np.array(pil_img, dtype=np.float32) / 255.0
        # Shape: (1, 3, 64, 64)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def predict_semantics(self, img_tensor: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs ResNet-18 and MobileNet-V2 on the image tensor.
        """
        mean = IMAGENET_MEAN.to(self.device)
        std = IMAGENET_STD.to(self.device)
        norm_img = (img_tensor - mean) / std

        pred1_info = {"class_name": "N/A", "confidence": 0.0, "class_id": -1}
        pred2_info = {"class_name": "N/A", "confidence": 0.0, "class_id": -1}

        with torch.no_grad():
            if self.clf_resnet18 is not None:
                logits1 = self.clf_resnet18(norm_img)
                probs1 = torch.softmax(logits1, dim=1)[0]
                top_id1 = torch.argmax(probs1).item()
                pred1_info = {
                    "class_name": EUROSAT_CLASSES[top_id1],
                    "confidence": float(probs1[top_id1].item()),
                    "class_id": top_id1,
                    "top3": [(EUROSAT_CLASSES[i], float(probs1[i].item())) for i in torch.topk(probs1, 3).indices]
                }

            if self.clf_mobilenet is not None:
                logits2 = self.clf_mobilenet(norm_img)
                probs2 = torch.softmax(logits2, dim=1)[0]
                top_id2 = torch.argmax(probs2).item()
                pred2_info = {
                    "class_name": EUROSAT_CLASSES[top_id2],
                    "confidence": float(probs2[top_id2].item()),
                    "class_id": top_id2,
                    "top3": [(EUROSAT_CLASSES[i], float(probs2[i].item())) for i in torch.topk(probs2, 3).indices]
                }

        return pred1_info, pred2_info

    def transmit_and_reconstruct(self,
                                 img_input: Union[str, Image.Image, np.ndarray],
                                 model_key: str,
                                 channel: str = "awgn",
                                 snr_db: float = 10.0) -> Dict[str, Any]:
        """
        Full end-to-end inference pass.
        
        Args:
            img_input: Path, PIL Image, or numpy array.
            model_key: Key from AVAILABLE_MODELS.
            channel: 'awgn' or 'rayleigh'.
            snr_db: Signal-to-noise ratio in dB.
            
        Returns:
            Dict containing reconstruction, metrics, semantic predictions, and timing.
        """
        channel = channel.lower()
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"Channel must be 'awgn' or 'rayleigh', got {channel!r}")

        # 1. Preprocess
        orig_tensor = self.preprocess_image(img_input)

        # 2. Get original semantic prediction
        orig_clf1, orig_clf2 = self.predict_semantics(orig_tensor)

        # 3. Load model
        model = self.load_model(model_key)

        # 4. Transmit over channel with timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            # Check model input convention
            # ResNet-based models need ImageNet normalization for backbone
            if AVAILABLE_MODELS[model_key]["type"] in ("resnet", "swin_refine"):
                mean = IMAGENET_MEAN.to(self.device)
                std = IMAGENET_STD.to(self.device)
                inp = (orig_tensor - mean) / std
            else:
                inp = orig_tensor

            recon_tensor, symbols = model(inp, channel=channel, snr_db=snr_db)
            recon_tensor = torch.clamp(recon_tensor, 0.0, 1.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 5. Compute metrics
        psnr_val = compute_psnr(orig_tensor, recon_tensor)
        ssim_val = compute_ssim_quick(orig_tensor, recon_tensor)

        # 6. Reconstructed semantic prediction
        recon_clf1, recon_clf2 = self.predict_semantics(recon_tensor)

        # 7. Convert tensors to displayable PIL images
        orig_np = (orig_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        recon_np = (recon_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        orig_pil = Image.fromarray(orig_np)
        recon_pil = Image.fromarray(recon_np)

        return {
            "model_name": model_key,
            "channel": channel.upper(),
            "snr_db": float(snr_db),
            "original_image": orig_pil,
            "reconstructed_image": recon_pil,
            "psnr_db": round(psnr_val, 3),
            "ssim": round(ssim_val, 4),
            "latency_ms": round(t_elapsed_ms, 2),
            "original_resnet18": orig_clf1,
            "original_mobilenet": orig_clf2,
            "reconstructed_resnet18": recon_clf1,
            "reconstructed_mobilenet": recon_clf2,
            "preserved_resnet18": (orig_clf1["class_id"] == recon_clf1["class_id"]),
            "preserved_mobilenet": (orig_clf2["class_id"] == recon_clf2["class_id"]),
            "channel_uses": symbols.shape[-1] if hasattr(symbols, "shape") else 4096,
        }
