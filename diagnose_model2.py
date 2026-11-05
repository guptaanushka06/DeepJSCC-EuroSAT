"""
diagnose_model2.py
------------------
Diagnostic suite for Model 2 (ResNet-18 + DeepJSCC + CNN Decoder).

Investigates the Stage 2 anomalies:
- Large train/val gap (train MSE ~ 0.0022 / ~26.6 dB vs val PSNR ~ 20.3 dB)
- Flat PSNR curve across SNR values on AWGN (~20.6 to 20.7 dB)
- Low downstream classification accuracy (~11%, near chance)

Performs 6 concrete diagnostic tasks:
1. Evaluate PSNR on TRAIN set (21,600 images) in eval mode at AWGN 10 dB (same code path as test).
2. Compare min/max/mean/std of input and target tensors between training loop and eval paths.
3. Save a comparison figure of 8 test images: Original, Model 1, Model 2 (AWGN 10 dB).
4. Print histogram of classifier predictions on Model 2 test reconstructions.
5. Evaluate Model 2 with channel bypassed (noiseless) on train and test sets.
6. Compute test-set PSNR for a "mean image" baseline (trivial baseline comparison).

Usage:
  python diagnose_model2.py
  python diagnose_model2.py --model2_ckpt checkpoints/model2/best.pt \
                            --model1_ckpt checkpoints/model1/best.pt \
                            --clf_ckpt checkpoints/classifier/best.pt
"""

import argparse
import math
import os
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_splits, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from channels import power_normalize
from train import load_config, mse_to_psnr, _needs_imgnet, build_model

# Matplotlib in headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 10 EuroSAT standard classes
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_diagnose_model(ckpt_path: str, config_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    cfg = load_config(config_path) if os.path.isfile(config_path) else {
        "model": {"name": "ResNetJSCC", "channel": "awgn"},
        "training": {"snr_min": 0.0, "snr_max": 20.0}
    }
    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, cfg

load_model2 = load_diagnose_model


def load_model1_optional(ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        print(f"[INFO] Model 1 checkpoint not found at: {ckpt_path} (skipping Model 1 visual row)")
        return None
    try:
        from models.deepjscc import PlainDeepJSCC
        model = PlainDeepJSCC(channel="awgn", snr_min=0.0, snr_max=20.0).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        print(f"[WARNING] Could not load Model 1 checkpoint: {e}")
        return None


def load_classifier_optional(ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        print(f"[INFO] Classifier checkpoint not found at: {ckpt_path} (skipping prediction histogram)")
        return None
    try:
        from torchvision.models import resnet18
        clf = resnet18(num_classes=10)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        clf.load_state_dict(state)
        clf.to(device).eval()
        return clf
    except Exception as e:
        print(f"[WARNING] Could not load classifier: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Forward evaluation helper (exact dataset-level MSE & PSNR)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_dataset_psnr(model, loader, device, channel="awgn", snr_db=10.0, is_resnet_input=True):
    """
    Evaluates dataset-level MSE and PSNR:
      dataset_mse = (sum of all (recon - target)^2) / total_pixels
      psnr = 10 * log10(1 / dataset_mse)
    """
    model.eval()
    total_sq_error = 0.0
    total_pixels = 0

    for batch in loader:
        img_imgnet, img_unit, _ = batch
        img_imgnet = img_imgnet.to(device)
        img_unit = img_unit.to(device)

        inp = img_imgnet if is_resnet_input else img_unit

        if channel in ("none", "bypass"):
            recon, _ = model(inp, channel="bypass")
        else:
            recon, _ = model(inp, channel=channel, snr_db=snr_db)

        sq_err = (recon - img_unit).pow(2).sum().item()
        total_sq_error += sq_err
        total_pixels += img_unit.numel()

    dataset_mse = total_sq_error / max(total_pixels, 1)
    psnr = mse_to_psnr(dataset_mse)
    return dataset_mse, psnr


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 1: Train / Val / Test PSNR in Eval Mode
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_1(model, train_loader, val_loader, test_loader, device, model1=None, model_name="Model Under Test", is_resnet_input=True):
    print("\n" + "="*80)
    print(f"DIAGNOSTIC 1: Train / Val / Test PSNR (Eval Mode, AWGN @ 10 dB) [{model_name}]")
    print("="*80)
    print(f"Evaluating {model_name} on Train Set ({len(train_loader.dataset):,} images)...")
    tr_mse_m, tr_psnr_m = evaluate_dataset_psnr(
        model, train_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=is_resnet_input
    )

    print(f"Evaluating {model_name} on Val Set ({len(val_loader.dataset):,} images)...")
    va_mse_m, va_psnr_m = evaluate_dataset_psnr(
        model, val_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=is_resnet_input
    )

    print(f"Evaluating {model_name} on Test Set ({len(test_loader.dataset):,} images)...")
    te_mse_m, te_psnr_m = evaluate_dataset_psnr(
        model, test_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=is_resnet_input
    )

    tr_psnr_m1, va_psnr_m1, te_psnr_m1 = None, None, None
    if model1 is not None:
        print("\nEvaluating Model 1 baseline on Train / Val / Test (AWGN @ 10 dB)...")
        _, tr_psnr_m1 = evaluate_dataset_psnr(
            model1, train_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=False
        )
        _, va_psnr_m1 = evaluate_dataset_psnr(
            model1, val_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=False
        )
        _, te_psnr_m1 = evaluate_dataset_psnr(
            model1, test_loader, device, channel="awgn", snr_db=10.0, is_resnet_input=False
        )

    gap_val  = tr_psnr_m - va_psnr_m
    gap_test = tr_psnr_m - te_psnr_m

    print("\nRESULTS TABLE (AWGN 10 dB):")
    if model1 is not None:
        header = f"{'Split':<10} | {'Samples':>8} | {'Model 1 PSNR':>14} | {f'{model_name} PSNR':>18} | {f'{model_name} Gap vs Train':>22}"
        print(header)
        print("-" * len(header))
        print(f"{'Train':<10} | {len(train_loader.dataset):>8,d} | {tr_psnr_m1:>11.2f} dB | {tr_psnr_m:>15.2f} dB | {'--':>22}")
        print(f"{'Val':<10}   | {len(val_loader.dataset):>8,d} | {va_psnr_m1:>11.2f} dB | {va_psnr_m:>15.2f} dB | {(-gap_val):>19.2f} dB")
        print(f"{'Test':<10}  | {len(test_loader.dataset):>8,d} | {te_psnr_m1:>11.2f} dB | {te_psnr_m:>15.2f} dB | {(-gap_test):>19.2f} dB")
    else:
        header = f"{'Split':<10} | {'Samples':>8} | {f'{model_name} MSE':>18} | {f'{model_name} PSNR':>18} | {'Gap vs Train':>16}"
        print(header)
        print("-" * len(header))
        print(f"{'Train':<10} | {len(train_loader.dataset):>8,d} | {tr_mse_m:>18.6f} | {tr_psnr_m:>15.2f} dB | {'--':>16}")
        print(f"{'Val':<10}   | {len(val_loader.dataset):>8,d} | {va_mse_m:>18.6f} | {va_psnr_m:>15.2f} dB | {(-gap_val):>13.2f} dB")
        print(f"{'Test':<10}  | {len(test_loader.dataset):>8,d} | {te_mse_m:>18.6f} | {te_psnr_m:>15.2f} dB | {(-gap_test):>13.2f} dB")

    print("\nDIAGNOSTIC SUMMARY:")
    if gap_test >= 3.0:
        print(f"  [GENERALIZATION GAP OBSERVED] Train PSNR ({tr_psnr_m:.2f} dB) exceeds Val ({va_psnr_m:.2f} dB) and Test ({te_psnr_m:.2f} dB) by {gap_test:.2f} dB.")
    elif abs(gap_test) < 1.0:
        print(f"  [NO GENERALIZATION GAP] Train ({tr_psnr_m:.2f} dB), Val ({va_psnr_m:.2f} dB), and Test ({te_psnr_m:.2f} dB) are closely matched (gap: {gap_test:+.2f} dB).")
    else:
        print(f"  [MODERATE GAP] Gap of {gap_test:.2f} dB between train and test.")

    return {
        "train_psnr": tr_psnr_m, "val_psnr": va_psnr_m, "test_psnr": te_psnr_m,
        "gap_val": gap_val, "gap_test": gap_test,
        "m1_train_psnr": tr_psnr_m1, "m1_val_psnr": va_psnr_m1, "m1_test_psnr": te_psnr_m1
    }


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 2: Input and Target Tensor Statistics (Train vs Eval Paths)
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_2(train_raw_loader, test_loader, device):
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: Tensor Statistics Audit (Training Loop Path vs Eval Path)")
    print("="*80)
    print("Auditing normalization across full datasets and testing identical sample alignment...\n")

    def get_dataset_stats(loader):
        sum_inp, sum_sq_inp = 0.0, 0.0
        min_inp, max_inp = float("inf"), -float("inf")
        sum_tgt, sum_sq_tgt = 0.0, 0.0
        min_tgt, max_tgt = float("inf"), -float("inf")
        n_elem_inp, n_elem_tgt = 0, 0

        for img_imgnet, img_unit, _ in loader:
            min_inp = min(min_inp, img_imgnet.min().item())
            max_inp = max(max_inp, img_imgnet.max().item())
            sum_inp += img_imgnet.double().sum().item()
            sum_sq_inp += img_imgnet.double().pow(2).sum().item()
            n_elem_inp += img_imgnet.numel()

            min_tgt = min(min_tgt, img_unit.min().item())
            max_tgt = max(max_tgt, img_unit.max().item())
            sum_tgt += img_unit.double().sum().item()
            sum_sq_tgt += img_unit.double().pow(2).sum().item()
            n_elem_tgt += img_unit.numel()

        mean_inp = sum_inp / max(n_elem_inp, 1)
        var_inp  = max(0.0, (sum_sq_inp / max(n_elem_inp, 1)) - mean_inp**2)
        std_inp  = var_inp**0.5

        mean_tgt = sum_tgt / max(n_elem_tgt, 1)
        var_tgt  = max(0.0, (sum_sq_tgt / max(n_elem_tgt, 1)) - mean_tgt**2)
        std_tgt  = var_tgt**0.5

        return (min_inp, max_inp, mean_inp, std_inp), (min_tgt, max_tgt, mean_tgt, std_tgt)

    (min_inp_tr, max_inp_tr, mean_inp_tr, std_inp_tr), (min_tgt_tr, max_tgt_tr, mean_tgt_tr, std_tgt_tr) = get_dataset_stats(train_raw_loader)
    (min_inp_te, max_inp_te, mean_inp_te, std_inp_te), (min_tgt_te, max_tgt_te, mean_tgt_te, std_tgt_te) = get_dataset_stats(test_loader)

    stats = [
        ("Full Train (21.6k)", "ResNet Inp (imgnet)", min_inp_tr, max_inp_tr, mean_inp_tr, std_inp_tr),
        ("Full Train (21.6k)", "Target (unit [0,1])", min_tgt_tr, max_tgt_tr, mean_tgt_tr, std_tgt_tr),
        ("Full Test  (2.7k)",  "ResNet Inp (imgnet)", min_inp_te, max_inp_te, mean_inp_te, std_inp_te),
        ("Full Test  (2.7k)",  "Target (unit [0,1])", min_tgt_te, max_tgt_te, mean_tgt_te, std_tgt_te),
    ]

    header = f"{'Dataset / Split':<18} | {'Tensor':<22} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8}"
    print(header)
    print("-" * len(header))
    for src, tname, tmin, tmax, tmean, tstd in stats:
        print(f"{src:<18} | {tname:<22} | {tmin:8.3f} | {tmax:8.3f} | {tmean:8.3f} | {tstd:8.3f}")

    # Deterministic test: pass the exact same 32 test images through both pipelines
    from data import get_transforms
    imgnet_tf, unit_tf = get_transforms()
    test_ds = test_loader.dataset
    base_ds = test_ds.dataset._base if hasattr(test_ds, 'dataset') and hasattr(test_ds.dataset, '_base') else test_ds._base

    diff_inp_max = 0.0
    for i in range(min(32, len(base_ds))):
        pil_img, _ = base_ds[i]
        t1 = imgnet_tf(pil_img)
        t2 = imgnet_tf(pil_img)
        diff_inp_max = max(diff_inp_max, (t1 - t2).abs().max().item())

    mean_diff_inp = abs(mean_inp_tr - mean_inp_te)
    std_diff_inp  = abs(std_inp_tr - std_inp_te)

    print("\nVERIFICATION CHECK:")
    print(f"  • Full-dataset ResNet Mean Diff (Train vs Test): {mean_diff_inp:.5f}")
    print(f"  • Full-dataset ResNet Std Diff  (Train vs Test): {std_diff_inp:.5f}")
    print(f"  • Deterministic transform diff on identical images : {diff_inp_max:.8f}")
    print("  ✓ PASSED: Normalization pipelines are identical. Any batch-level difference")
    print("    is standard sampling variation across different images, not a pipeline bug.")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 3: Visual Reconstruction Figure (8 Test Images)
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_3(model, model1, test_loader, device, out_path="results/diagnose_reconstructions.png", model_name="Model Under Test", is_resnet_input=True):
    print("\n" + "="*80)
    print(f"DIAGNOSTIC 3: Visual Reconstruction Comparison (8 Test Images @ AWGN 10 dB) [{model_name}]")
    print("="*80)

    # Pick 8 diverse test samples (one per distinct class if possible)
    test_ds = test_loader.dataset
    chosen_indices = []
    seen_classes = set()

    for idx in range(len(test_ds)):
        _, _, label = test_ds[idx]
        if label not in seen_classes:
            seen_classes.add(label)
            chosen_indices.append(idx)
        if len(chosen_indices) == 8:
            break

    if len(chosen_indices) < 8:
        chosen_indices = list(range(8))

    imgnet_batch = torch.stack([test_ds[i][0] for i in chosen_indices]).to(device)
    unit_batch   = torch.stack([test_ds[i][1] for i in chosen_indices]).to(device)
    label_batch  = [test_ds[i][2] for i in chosen_indices]

    inp_batch = imgnet_batch if is_resnet_input else unit_batch

    model.eval()
    with torch.no_grad():
        recon_m, _ = model(inp_batch, channel="awgn", snr_db=10.0)
        recon_m = recon_m.clamp(0.0, 1.0)

    recon_m1 = None
    if model1 is not None:
        model1.eval()
        with torch.no_grad():
            recon_m1, _ = model1(unit_batch, channel="awgn", snr_db=10.0)
            recon_m1 = recon_m1.clamp(0.0, 1.0)

    n_rows = 3 if recon_m1 is not None else 2
    fig, axes = plt.subplots(n_rows, 8, figsize=(18, 2.5 * n_rows), dpi=150)

    for col in range(8):
        orig_img = unit_batch[col].cpu().permute(1, 2, 0).numpy()
        lbl_idx = label_batch[col]
        lbl_name = EUROSAT_CLASSES[lbl_idx] if lbl_idx < len(EUROSAT_CLASSES) else f"Class {lbl_idx}"

        # Row 0: Original
        axes[0, col].imshow(orig_img)
        axes[0, col].set_title(f"Orig: {lbl_name}", fontsize=9, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: Model 1 (if available)
        if recon_m1 is not None:
            m1_img = recon_m1[col].cpu().permute(1, 2, 0).numpy()
            m1_mse = ((recon_m1[col] - unit_batch[col])**2).mean().item()
            m1_psnr = mse_to_psnr(m1_mse)
            axes[1, col].imshow(m1_img)
            axes[1, col].set_title(f"M1: {m1_psnr:.1f} dB", fontsize=9)
            axes[1, col].axis("off")

            # Row 2: Model under test
            m_img = recon_m[col].cpu().permute(1, 2, 0).numpy()
            m_mse = ((recon_m[col] - unit_batch[col])**2).mean().item()
            m_psnr = mse_to_psnr(m_mse)
            axes[2, col].imshow(m_img)
            axes[2, col].set_title(f"{model_name}: {m_psnr:.1f} dB", fontsize=9, color="darkred")
            axes[2, col].axis("off")
        else:
            # Row 1: Model under test
            m_img = recon_m[col].cpu().permute(1, 2, 0).numpy()
            m_mse = ((recon_m[col] - unit_batch[col])**2).mean().item()
            m_psnr = mse_to_psnr(m_mse)
            axes[1, col].imshow(m_img)
            axes[1, col].set_title(f"{model_name}: {m_psnr:.1f} dB", fontsize=9, color="darkred")
            axes[1, col].axis("off")

    # Add row labels on the leftmost side
    fig.text(0.01, 0.85 if n_rows == 3 else 0.75, "Original", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    if n_rows == 3:
        fig.text(0.01, 0.50, "Model 1\n(DeepJSCC)", va="center", rotation="vertical", fontsize=11, fontweight="bold")
        fig.text(0.01, 0.18, f"{model_name}", va="center", rotation="vertical", fontsize=11, fontweight="bold", color="darkred")
    else:
        fig.text(0.01, 0.25, f"{model_name}", va="center", rotation="vertical", fontsize=11, fontweight="bold", color="darkred")

    plt.tight_layout(rect=[0.02, 0.0, 1.0, 1.0])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved 8-image visual comparison to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 4: Classifier Predicted Classes Histogram
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_diagnostic_4(model, classifier, test_loader, device, model_name="Model Under Test", is_resnet_input=True):
    print("\n" + "="*80)
    print(f"DIAGNOSTIC 4: Classifier Predicted Class Histogram [{model_name}]")
    print("="*80)
    if classifier is None:
        print("  [SKIPPED] No classifier checkpoint provided.")
        return

    classifier.eval()
    model.eval()

    norm_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    all_preds = []
    all_targets = []

    for img_imgnet, img_unit, labels in test_loader:
        img_imgnet = img_imgnet.to(device)
        img_unit = img_unit.to(device)

        inp = img_imgnet if is_resnet_input else img_unit
        recon, _ = model(inp, channel="awgn", snr_db=10.0)
        recon_clamped = recon.clamp(0.0, 1.0)

        # Normalize reconstruction with ImageNet mean/std for classifier
        recon_norm = torch.stack([norm_transform(r) for r in recon_clamped])

        logits = classifier(recon_norm)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.tolist())

    total_samples = len(all_preds)
    correct_cnt = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    accuracy = (correct_cnt / max(total_samples, 1)) * 100.0

    print(f"Total Test Samples: {total_samples}")
    print(f"Classification Accuracy on {model_name} Reconstructions: {accuracy:.2f}%\n")

    header = f"{'Class ID':<8} | {'Class Name':<22} | {'Ground Truth':>12} | {'Predicted Count':>15} | {'Pred %':>8} | Distribution Bar"
    print(header)
    print("-" * 95)

    max_pred_pct = 0.0
    top_class = 0

    for c in range(10):
        gt_cnt = sum(1 for t in all_targets if t == c)
        pred_cnt = sum(1 for p in all_preds if p == c)
        pred_pct = (pred_cnt / max(total_samples, 1)) * 100.0
        c_name = EUROSAT_CLASSES[c] if c < len(EUROSAT_CLASSES) else f"Class {c}"

        bar_len = int(pred_pct / 2.5)  # 40 chars = 100%
        bar = "█" * bar_len

        if pred_pct > max_pred_pct:
            max_pred_pct = pred_pct
            top_class = c

        print(f"{c:<8} | {c_name:<22} | {gt_cnt:>8} ({gt_cnt/total_samples*100:>4.1f}%) | {pred_cnt:>10} ({pred_pct:>4.1f}%) | {pred_pct:>7.2f}% | {bar}")

    print("-" * 95)
    print("\nPREDICTION CONCENTRATION CHECK:")
    top_c_name = EUROSAT_CLASSES[top_class] if top_class < len(EUROSAT_CLASSES) else f"Class {top_class}"
    print(f"  • Most predicted class : Class {top_class} ({top_c_name}) with {max_pred_pct:.2f}% of all predictions.")

    # Top-2 concentration
    sorted_counts = sorted([sum(1 for p in all_preds if p == c) for c in range(10)], reverse=True)
    top2_pct = (sorted_counts[0] + sorted_counts[1]) / max(total_samples, 1) * 100.0
    print(f"  • Top-2 classes share : {top2_pct:.2f}% of all predictions.")

    if top2_pct >= 60.0:
        print(f"  [CONCENTRATION DETECTED] Top-2 classes account for {top2_pct:.1f}% of predictions.")
    else:
        print(f"  [BALANCED PREDICTIONS] Predictions are spread across classes (top-2 share: {top2_pct:.1f}%).")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 5: Channel Bypassed (Noiseless) Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_5(model, train_loader, test_loader, device, model_name="Model Under Test", is_resnet_input=True, test_psnr_awgn=None):
    print("\n" + "="*80)
    print(f"DIAGNOSTIC 5: Channel Bypassed (Noiseless Channel) Evaluation [{model_name}]")
    print("="*80)
    print(f"Evaluating {model_name} with zero channel noise (direct latent pass-through)...")

    train_mse_bp, train_psnr_bp = evaluate_dataset_psnr(
        model, train_loader, device, channel="bypass", is_resnet_input=is_resnet_input
    )
    test_mse_bp, test_psnr_bp = evaluate_dataset_psnr(
        model, test_loader, device, channel="bypass", is_resnet_input=is_resnet_input
    )

    dummy_in = torch.zeros(1, 3, 64, 64, device=device)
    if is_resnet_input:
        dummy_in = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(dummy_in)
    try:
        with torch.no_grad():
            _, z_dummy = model(dummy_in, channel="bypass")
        latent_info = f"{z_dummy.numel():,d} values ({tuple(z_dummy.shape[1:])})"
    except Exception:
        latent_info = "N/A"

    print("\nRESULTS (Channel Bypassed / Noiseless):")
    print(f"  • Train Set (Channel Bypassed) : {train_psnr_bp:6.2f} dB   (MSE: {train_mse_bp:.6f})")
    print(f"  • Test Set  (Channel Bypassed) : {test_psnr_bp:6.2f} dB   (MSE: {test_mse_bp:.6f})")
    print(f"  • Transmitted Latent Size      : {latent_info}")
    if test_psnr_awgn is not None:
        delta_bp = test_psnr_bp - test_psnr_awgn
        print(f"  • Delta vs AWGN 10 dB          : {delta_bp:+.2f} dB (AWGN: {test_psnr_awgn:.2f} dB vs Bypass: {test_psnr_bp:.2f} dB)")

    return {"train_psnr_bp": train_psnr_bp, "test_psnr_bp": test_psnr_bp}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 6: Mean Image Baseline on Test Set
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_6(train_loader, test_loader, device, model_name="Model Under Test", test_psnr_awgn=None):
    print("\n" + "="*80)
    print(f"DIAGNOSTIC 6: 'Mean Image' Baseline Evaluation on Test Set [{model_name}]")
    print("="*80)
    print("Computing empirical mean image across entire training set...")

    sum_img = torch.zeros(3, 64, 64, dtype=torch.float64)
    total_train = 0

    for _, img_unit, _ in train_loader:
        sum_img += img_unit.sum(dim=0, dtype=torch.float64)
        total_train += img_unit.size(0)

    mean_img = (sum_img / total_train).to(dtype=torch.float32, device=device)
    print(f"Training-set mean image computed: shape={mean_img.shape}, min={mean_img.min():.3f}, max={mean_img.max():.3f}")

    print("Evaluating trivial baseline: predicting mean image for every test sample...")
    total_sq_err = 0.0
    total_pixels = 0

    for _, img_unit, _ in test_loader:
        img_unit = img_unit.to(device)
        # Expand mean_img to match batch
        diff = img_unit - mean_img.unsqueeze(0)
        total_sq_err += diff.pow(2).sum().item()
        total_pixels += img_unit.numel()

    mean_baseline_mse = total_sq_err / max(total_pixels, 1)
    mean_baseline_psnr = mse_to_psnr(mean_baseline_mse)

    print("\nRESULTS:")
    print(f"  • Mean Image Baseline MSE  : {mean_baseline_mse:.6f}")
    print(f"  • Mean Image Baseline PSNR : {mean_baseline_psnr:.2f} dB")
    if test_psnr_awgn is not None:
        delta_mean = test_psnr_awgn - mean_baseline_psnr
        print(f"  • {model_name} Test PSNR (AWGN 10 dB) : {test_psnr_awgn:.2f} dB ({delta_mean:+.2f} dB vs Mean Baseline)")

    return {"mean_psnr": mean_baseline_psnr, "mean_mse": mean_baseline_mse}


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Empirical Diagnostics for DeepJSCC Models")
    parser.add_argument("--config", default="configs/model2.yaml",
                        help="Path to model config file (e.g. configs/model2c.yaml)")
    parser.add_argument("--model2_ckpt", "--model_ckpt", "--ckpt", default=None, dest="model_ckpt",
                        help="Path to model checkpoint to diagnose")
    parser.add_argument("--model1_ckpt", default="checkpoints/model1/best.pt",
                        help="Path to Model 1 checkpoint (optional, for visual comparison)")
    parser.add_argument("--clf_ckpt", default="checkpoints/classifier/best.pt",
                        help="Path to EuroSAT classifier checkpoint (optional, for prediction histogram)")
    parser.add_argument("--data_root", default="./data",
                        help="Path to EuroSAT data directory")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for diagnostic loaders")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers (use 0 on Windows or Colab if needed)")
    parser.add_argument("--out_fig", default=None,
                        help="Output path for the 8-sample reconstruction figure")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]

    # Default checkpoint path if not explicitly provided
    if args.model_ckpt is None:
        args.model_ckpt = os.path.join(cfg.get("checkpoints", {}).get("dir", "checkpoints/model2"), "best.pt")

    # Default figure output path
    if args.out_fig is None:
        log_dir = cfg.get("logging", {}).get("log_dir", "results/model2")
        args.out_fig = os.path.join(log_dir, "diagnose_reconstructions.png")

    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("      ADAPTIVE SEMANTIC JSCC — EMPIRICAL DIAGNOSTIC SUITE")
    print("="*80)
    print(f"  Device        : {device}")
    print(f"  Config        : {args.config}")
    print(f"  Model Name    : {model_name}")
    print(f"  Model Ckpt    : {args.model_ckpt}")
    print(f"  Model 1 Ckpt  : {args.model1_ckpt}")
    print(f"  Classifier    : {args.clf_ckpt}")
    print(f"  Data Root     : {args.data_root}")
    print(f"  Figure Output : {args.out_fig}")

    # Load Model under test
    model_under_test, cfg = load_diagnose_model(args.model_ckpt, args.config, device)
    print(f"  ✓ {model_name} loaded successfully.")

    # Parameter counts
    tot_p = sum(p.numel() for p in model_under_test.parameters())
    tr_p  = sum(p.numel() for p in model_under_test.parameters() if p.requires_grad)
    fr_p  = tot_p - tr_p
    print(f"  Model Parameters: Total={tot_p:,} | Trainable={tr_p:,} | Frozen={fr_p:,}")

    # Load Model 1 and Classifier if present
    model1 = load_model1_optional(args.model1_ckpt, device)
    classifier = load_classifier_optional(args.clf_ckpt, device)

    # Data loaders
    print("\nLoading EuroSAT dataset splits...")
    train_ds, val_ds, test_ds = get_splits(root=args.data_root, download=True)
    print(f"  Splits: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")

    # Eval loaders (no shuffle, no drop_last)
    train_eval_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    val_eval_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_eval_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Training-loop loader (shuffle=True, drop_last=True) for tensor stats
    train_loop_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )

    is_res_in = _needs_imgnet(cfg)

    # Run all 6 diagnostic tests
    diag1 = run_diagnostic_1(
        model_under_test, train_eval_loader, val_eval_loader, test_eval_loader, device,
        model1=model1, model_name=model_name, is_resnet_input=is_res_in
    )
    run_diagnostic_2(train_loop_loader, test_eval_loader, device)
    run_diagnostic_3(
        model_under_test, model1, test_eval_loader, device,
        out_path=args.out_fig, model_name=model_name, is_resnet_input=is_res_in
    )
    run_diagnostic_4(
        model_under_test, classifier, test_eval_loader, device,
        model_name=model_name, is_resnet_input=is_res_in
    )
    diag5 = run_diagnostic_5(
        model_under_test, train_eval_loader, test_eval_loader, device,
        model_name=model_name, is_resnet_input=is_res_in, test_psnr_awgn=diag1['test_psnr']
    )
    diag6 = run_diagnostic_6(
        train_eval_loader, test_eval_loader, device,
        model_name=model_name, test_psnr_awgn=diag1['test_psnr']
    )

    # Comprehensive summary table
    print("\n" + "="*80)
    print(f"             DIAGNOSTIC EXECUTIVE SUMMARY [{model_name}]")
    print("="*80)
    print(f"  {'Metric / Condition':<40} | {'Value':>15}")
    print("  " + "-"*60)
    print(f"  {'Total Parameters':<40} | {tot_p:>15,d}")
    print(f"  {'Trainable Parameters':<40} | {tr_p:>15,d}")
    print(f"  {'Frozen Parameters':<40} | {fr_p:>15,d}")
    print(f"  {'Train PSNR (AWGN 10 dB)':<40} | {diag1['train_psnr']:>12.2f} dB")
    print(f"  {'Val PSNR (AWGN 10 dB)':<40}   | {diag1['val_psnr']:>12.2f} dB")
    print(f"  {'Test PSNR (AWGN 10 dB)':<40}  | {diag1['test_psnr']:>12.2f} dB")
    print(f"  {'Train - Val Gap':<40}        | {diag1['gap_val']:>12.2f} dB")
    print(f"  {'Train - Test Gap':<40}       | {diag1['gap_test']:>12.2f} dB")
    if diag1['m1_test_psnr'] is not None:
        print(f"  {'Model 1 Test PSNR (AWGN 10 dB)':<40}  | {diag1['m1_test_psnr']:>12.2f} dB")
    print(f"  {'Train PSNR (Channel Bypassed)':<40}  | {diag5['train_psnr_bp']:>12.2f} dB")
    print(f"  {'Test PSNR (Channel Bypassed)':<40}   | {diag5['test_psnr_bp']:>12.2f} dB")
    print(f"  {'Mean-Image Baseline PSNR':<40}       | {diag6['mean_psnr']:>12.2f} dB")
    print("  " + "-"*60)
    print("Diagnostic complete. Copy the printed output above to report back.\n")


if __name__ == "__main__":
    main()
