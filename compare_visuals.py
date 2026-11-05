"""
compare_visuals.py
------------------
Generates side-by-side visual comparisons of reconstructed EuroSAT images at 10 dB AWGN:
  Row 1: Ground Truth Original
  Row 2: PlainDeepJSCC (Model 1 Fair, 31.07 dB)
  Row 3: ResNetJSCC_2d Long (Model 2d Long, 25.52 dB)
  Row 4: Semantic Refiner (Model 3b lam=0.0003, 25.39 dB)

Runs strictly on CPU to avoid interfering with any ongoing GPU training runs.
"""

import os
import sys
import torch
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_splits, IMAGENET_MEAN, IMAGENET_STD
from train import build_model, load_config, _needs_imgnet

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVeg", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def main():
    device = torch.device("cpu")
    print("Loading test dataset on CPU...")
    _, _, test_ds = get_splits(root="./data", download=False)

    # Pick 6 diverse classes: Forest(1), Industrial(4), Residential(7), Highway(3), PermanentCrop(6), AnnualCrop(0)
    target_classes = [1, 4, 7, 3, 6, 0]
    selected_samples = {}

    for idx in range(len(test_ds)):
        img_imgnet, img_unit, lbl = test_ds[idx]
        if lbl in target_classes and lbl not in selected_samples:
            selected_samples[lbl] = (img_imgnet.unsqueeze(0), img_unit.unsqueeze(0), lbl)
        if len(selected_samples) == len(target_classes):
            break

    ordered_samples = [selected_samples[c] for c in target_classes]

    models_info = [
        ("PlainDeepJSCC", "configs/model1_fair.yaml", "checkpoints/model1_fair/best.pt"),
        ("Model 2d Long", "configs/model2d_long.yaml", "checkpoints/model2d_long/best.pt"),
        ("Semantic Refiner (lam=0.0003)", "configs/model3b_lam0.0003.yaml", "checkpoints/model3b_lam0.0003/best.pt"),
    ]

    reconstructions = {}

    for name, cfg_path, ckpt_path in models_info:
        print(f"Loading {name} on CPU...")
        cfg = load_config(cfg_path)
        model = build_model(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        recons = []
        with torch.no_grad():
            for img_imgnet, img_unit, lbl in ordered_samples:
                inp = img_imgnet if _needs_imgnet(cfg) else img_unit
                recon, _ = model(inp, channel="awgn", snr_db=10.0)
                recons.append(recon.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy())
        reconstructions[name] = recons

    # Plot 4 rows x 6 columns
    n_cols = len(target_classes)
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.2, 8.5))

    row_labels = [
        "Ground Truth",
        "PlainDeepJSCC\n(31.07 dB)",
        "Model 2d Long\n(25.52 dB)",
        "Model 3b (lam=0.0003)\n(25.39 dB)"
    ]

    for col in range(n_cols):
        img_imgnet, img_unit, lbl = ordered_samples[col]
        orig_np = img_unit.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[0, col].imshow(orig_np)
        axes[0, col].set_title(f"{CLASS_NAMES[lbl]}", fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(reconstructions["PlainDeepJSCC"][col])
        axes[1, col].axis("off")

        axes[2, col].imshow(reconstructions["Model 2d Long"][col])
        axes[2, col].axis("off")

        axes[3, col].imshow(reconstructions["Semantic Refiner (lam=0.0003)"][col])
        axes[3, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.35, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=9, fontweight="bold", va="center", ha="right")

    fig.suptitle("Visual Reconstruction Comparison across Models (AWGN @ 10 dB)", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])

    out_path = "results/final/semantic_analysis/visual_reconstructions_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visual comparison figure saved to {out_path}")


if __name__ == "__main__":
    main()

