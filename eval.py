"""
eval.py
-------
Full evaluation script with dual classifier semantic reporting.

Usage
-----
# Evaluate a single model:
python eval.py --config configs/model1.yaml --checkpoint checkpoints/model1/best.pt

# Evaluate with dual classifiers (primary ResNet-18 + secondary MobileNet-V2):
python eval.py --config configs/model3b_lam0.003.yaml --checkpoint checkpoints/model3b_lam0.003/best.pt \
               --clf_ckpt checkpoints/classifier/best.pt --clf2_ckpt checkpoints/classifier2/best.pt

# Evaluate all models and produce comparison tables + plots:
python eval.py --all

What it measures
----------------
For each (model, SNR, channel) triple on the test set:
  - PSNR          : 10 * log10(1 / MSE_dataset)
  - SSIM          : Structural Similarity Index (via torchmetrics)
  - Accuracy (1)  : Classification accuracy with Primary Classifier (ResNet-18, seed 42)
  - Accuracy (2)  : Classification accuracy with Secondary Classifier (MobileNet-V2, seed 1337)
"""

import argparse
import csv
import os
import random
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import yaml

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    _TORCHMETRICS_OK = True
except ImportError:
    _TORCHMETRICS_OK = False
    print("[WARNING] torchmetrics not installed. SSIM will not be computed. "
          "Run: pip install torchmetrics")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    print("[WARNING] matplotlib not installed. Plots will be skipped.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data     import get_loaders, IMAGENET_MEAN, IMAGENET_STD
from channels import sample_snr
from train    import build_model, load_config, mse_to_psnr, _needs_imgnet


# ─────────────────────────────────────────────────────────────────────────────
# Classifier loading
# ─────────────────────────────────────────────────────────────────────────────
def load_classifier(ckpt_path: str, device: torch.device):
    """
    Load a classifier checkpoint (ResNet-18 or MobileNet-V2).
    Returns model in eval mode with gradients disabled, or None if not found.
    """
    if not os.path.isfile(ckpt_path):
        return None
    from classifier import build_classifier
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "resnet18")
    clf = build_classifier(arch=arch)
    clf.load_state_dict(ckpt["model"])
    clf.to(device).eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


def set_eval_seed(seed: int = 42, channel: str = "awgn", snr_db: float = 10.0):
    """Ensure identical channel noise realizations across models for fair, repeatable eval."""
    ch_offset = 10000 if channel.lower() == "rayleigh" else 0
    point_seed = int(seed + ch_offset + round(snr_db * 100))
    random.seed(point_seed)
    torch.manual_seed(point_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(point_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_model(model, test_loader, device, cfg,
               snr_db: float, channel: str,
               ssim_metric=None, classifier=None, classifier2=None,
               seed: int = 42, save_per_image_path: str = None) -> dict:
    """
    Evaluate model at one (snr_db, channel) pair with deterministic seeding.

    PSNR averaging convention:
        MSE_dataset = (total squared error over all pixels) / (N * C * H * W)
        PSNR = 10 * log10(1 / MSE_dataset)

    Also computes per-image PSNR (mean ± std) for error reporting,
    and optionally exports per-image CSV metrics.
    """
    set_eval_seed(seed=seed, channel=channel, snr_db=snr_db)
    model.eval()

    total_sq_error = 0.0
    total_pixels   = 0

    all_recons  = []
    all_targets = []
    all_labels  = []
    per_image_psnrs = []

    for batch in test_loader:
        img_imgnet, img_unit, labels = batch
        img_imgnet = img_imgnet.to(device)
        img_unit   = img_unit.to(device)
        labels     = labels.to(device)

        inp = img_imgnet if _needs_imgnet(cfg) else img_unit
        recon, _ = model(inp, channel=channel, snr_db=snr_db)

        sq_err = (recon - img_unit).pow(2).sum().item()
        total_sq_error += sq_err
        total_pixels   += img_unit.numel()

        # Per-image PSNR
        B = img_unit.size(0)
        per_pixel = img_unit.numel() // B   # C * H * W
        for i in range(B):
            img_mse = (recon[i] - img_unit[i]).pow(2).sum().item() / per_pixel
            img_psnr = mse_to_psnr(img_mse)
            per_image_psnrs.append(img_psnr)

        all_recons.append(recon.cpu())
        all_targets.append(img_unit.cpu())
        all_labels.append(labels.cpu())

    dataset_mse = total_sq_error / total_pixels
    avg_psnr    = mse_to_psnr(dataset_mse)

    # Per-image PSNR stats (useful for confidence intervals)
    import numpy as np
    psnr_arr = np.array(per_image_psnrs)
    psnr_mean = float(psnr_arr.mean())
    psnr_std  = float(psnr_arr.std())

    recons_all  = torch.cat(all_recons,  dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    labels_all  = torch.cat(all_labels,  dim=0)

    # SSIM (both aggregate and per-image)
    avg_ssim = None
    per_image_ssims = []
    if ssim_metric is not None:
        ssim_metric.reset()
        chunk = 256
        for i in range(0, len(recons_all), chunk):
            r = recons_all[i:i+chunk].to(device)
            t = targets_all[i:i+chunk].to(device)
            ssim_metric.update(r, t)
            if _TORCHMETRICS_OK:
                s_chunk = ssim_fn(r, t, data_range=1.0, reduction='none')
                per_image_ssims.extend(s_chunk.cpu().tolist())
        avg_ssim = ssim_metric.compute().item()

    # ImageNet normalization for classification
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    # Primary Classifier Accuracy
    avg_acc1 = None
    all_preds1 = []
    all_correct1 = []
    if classifier is not None:
        correct, total = 0, 0
        chunk = 128
        for i in range(0, len(recons_all), chunk):
            r_raw = recons_all[i:i+chunk].to(device)
            r_norm = (r_raw - mean_t) / std_t
            lbl = labels_all[i:i+chunk].to(device)
            logits = classifier(r_norm)
            preds  = logits.argmax(dim=1)
            corr = (preds == lbl)
            correct += corr.sum().item()
            total   += lbl.size(0)
            all_preds1.extend(preds.cpu().tolist())
            all_correct1.extend(corr.cpu().int().tolist())
        avg_acc1 = correct / total

    # Secondary Independent Classifier Accuracy (Model-agnostic evaluation)
    avg_acc2 = None
    all_preds2 = []
    all_correct2 = []
    if classifier2 is not None:
        correct, total = 0, 0
        chunk = 128
        for i in range(0, len(recons_all), chunk):
            r_raw = recons_all[i:i+chunk].to(device)
            r_norm = (r_raw - mean_t) / std_t
            lbl = labels_all[i:i+chunk].to(device)
            logits = classifier2(r_norm)
            preds  = logits.argmax(dim=1)
            corr = (preds == lbl)
            correct += corr.sum().item()
            total   += lbl.size(0)
            all_preds2.extend(preds.cpu().tolist())
            all_correct2.extend(corr.cpu().int().tolist())
        avg_acc2 = correct / total

    # Save per-image CSV if requested
    if save_per_image_path:
        os.makedirs(os.path.dirname(save_per_image_path), exist_ok=True)
        with open(save_per_image_path, "w", newline="") as pf:
            pwriter = csv.writer(pf)
            pwriter.writerow([
                "image_idx", "psnr", "ssim", "true_label",
                "pred_clf1", "correct_clf1",
                "pred_clf2", "correct_clf2"
            ])
            for idx in range(len(recons_all)):
                psnr_val = per_image_psnrs[idx] if idx < len(per_image_psnrs) else ""
                ssim_val = per_image_ssims[idx] if idx < len(per_image_ssims) else ""
                lbl_val  = labels_all[idx].item()
                p1 = all_preds1[idx] if idx < len(all_preds1) else ""
                c1 = all_correct1[idx] if idx < len(all_correct1) else ""
                p2 = all_preds2[idx] if idx < len(all_preds2) else ""
                c2 = all_correct2[idx] if idx < len(all_correct2) else ""
                pwriter.writerow([idx, psnr_val, ssim_val, lbl_val, p1, c1, p2, c2])

    return {
        "psnr": avg_psnr,
        "psnr_per_image_mean": psnr_mean,
        "psnr_per_image_std": psnr_std,
        "ssim": avg_ssim,
        "accuracy": avg_acc1,
        "accuracy_clf2": avg_acc2,
        "n_samples": len(recons_all),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(all_results: list, plot_dir: str):
    """Generate per-channel comparison plots."""
    if not _MPL_OK:
        return
    os.makedirs(plot_dir, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(all_results)

    models   = df["model_name"].unique()
    channels = df["channel"].unique()
    snrs     = sorted(df["snr_db"].unique())

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for ch in channels:
        sub = df[df["channel"] == ch]
        for metric, ylabel, fname in [
            ("psnr",     "PSNR (dB)",                   f"psnr_vs_snr_{ch}.png"),
            ("ssim",     "SSIM",                        f"ssim_vs_snr_{ch}.png"),
            ("accuracy", "Primary Accuracy (ResNet-18)",f"acc_vs_snr_{ch}.png"),
        ]:
            if sub[metric].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for i, mn in enumerate(models):
                row = sub[sub["model_name"] == mn].sort_values("snr_db")
                ax.plot(row["snr_db"], row[metric],
                        label=mn,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        linewidth=2, markersize=6)
            ax.set_xlabel("SNR (dB)", fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(f"{ylabel} vs SNR  [{ch.upper()} channel]", fontsize=14)
            ax.set_xticks(snrs)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, fname), dpi=150)
            plt.close()
            print(f"  Plot saved: {os.path.join(plot_dir, fname)}")


# ─────────────────────────────────────────────────────────────────────────────
# Run Single Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def save_sample_reconstructions(model, test_loader, device, cfg,
                                snr_db: float, channel: str,
                                save_dir: str, model_name: str,
                                n_samples: int = 8, seed: int = 42):
    """Save side-by-side original vs reconstructed image samples for visual inspection."""
    if not _MPL_OK:
        return
    set_eval_seed(seed=seed, channel=channel, snr_db=snr_db)
    model.eval()

    imgs_orig = []
    imgs_recon = []

    with torch.no_grad():
        for batch in test_loader:
            img_imgnet, img_unit, labels = batch
            img_imgnet = img_imgnet.to(device)
            img_unit   = img_unit.to(device)

            inp = img_imgnet if _needs_imgnet(cfg) else img_unit
            recon, _ = model(inp, channel=channel, snr_db=snr_db)

            for i in range(img_unit.size(0)):
                if len(imgs_orig) >= n_samples:
                    break
                imgs_orig.append(img_unit[i].cpu())
                imgs_recon.append(recon[i].cpu().clamp(0, 1))
            if len(imgs_orig) >= n_samples:
                break

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.5, 5))
    for i in range(n_samples):
        axes[0, i].imshow(imgs_orig[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title("Original", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(imgs_recon[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title("Reconstructed", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle(f"{model_name} -- {channel.upper()} @ {snr_db} dB", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"samples_{model_name}_{channel}_{int(snr_db)}dB.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sample reconstructions saved: {fname}")


def run_single(config_path: str, ckpt_path: str,
               clf_ckpt: str = "checkpoints/classifier/best.pt",
               clf2_ckpt: str = "checkpoints/classifier2/best.pt",
               save_per_image: bool = True):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_loaders(
        root        = cfg["data"]["root"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        download    = cfg["data"]["download"],
    )

    model = build_model(cfg, device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    ssim_metric = SSIM(data_range=1.0).to(device) if _TORCHMETRICS_OK else None

    # Load Primary & Secondary Classifiers
    classifier1 = load_classifier(clf_ckpt, device)
    if classifier1 is not None:
        print(f"  Loaded Primary Classifier from: {clf_ckpt}")
    else:
        print(f"  [NOTE] Primary classifier not found at {clf_ckpt}.")

    classifier2 = load_classifier(clf2_ckpt, device)
    if classifier2 is not None:
        print(f"  Loaded Secondary Classifier from: {clf2_ckpt}")
    else:
        print(f"  [NOTE] Secondary classifier not found at {clf2_ckpt} (optional).")

    eval_snrs     = cfg["logging"]["eval_snrs"]
    eval_channels = cfg["logging"]["eval_channels"]
    model_name    = cfg["model"]["name"]
    log_dir       = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "test_metrics.csv")
    results  = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "channel", "snr_db",
                         "psnr", "psnr_std", "ssim",
                         "accuracy", "accuracy_clf2", "n_samples"])
        for ch in eval_channels:
            for snr in eval_snrs:
                t0 = time.time()
                per_img_path = None
                if save_per_image:
                    per_img_path = os.path.join(log_dir, f"per_image_{ch}_{int(snr)}.csv")
                r = eval_model(model, test_loader, device, cfg,
                               snr_db=float(snr), channel=ch,
                               ssim_metric=ssim_metric,
                               classifier=classifier1,
                               classifier2=classifier2,
                               save_per_image_path=per_img_path)
                elapsed = time.time() - t0
                psnr_str = f"{r['psnr']:.4f}" if r['psnr'] else "N/A"
                psnr_std_str = f" +/- {r['psnr_per_image_std']:.2f}" if r.get('psnr_per_image_std') is not None else ""
                ssim_str = f"{r['ssim']:.4f}" if r['ssim'] is not None else "N/A"
                acc1_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
                acc2_str = f"  Acc2={r['accuracy_clf2']:.4f}" if r['accuracy_clf2'] is not None else ""

                print(f"  {model_name} | {ch.upper():8s} | SNR={snr:2d}dB | "
                      f"PSNR={psnr_str}{psnr_std_str}  SSIM={ssim_str}  Acc={acc1_str}{acc2_str}  "
                      f"({elapsed:.1f}s)")

                writer.writerow([
                    model_name, ch, snr,
                    r["psnr"], r.get("psnr_per_image_std", "N/A"),
                    r["ssim"],
                    r["accuracy"] if r["accuracy"] is not None else "N/A",
                    r["accuracy_clf2"] if r["accuracy_clf2"] is not None else "N/A",
                    r["n_samples"]
                ])
                results.append({
                    "model_name": model_name,
                    "channel":    ch,
                    "snr_db":     snr,
                    **r
                })

    print(f"Test metrics saved: {csv_path}")

    # Save sample reconstruction images at 10 dB AWGN
    try:
        save_sample_reconstructions(
            model, test_loader, device, cfg,
            snr_db=10.0, channel="awgn",
            save_dir=log_dir, model_name=model_name
        )
    except Exception as e:
        print(f"  [WARNING] Could not save sample reconstructions: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepJSCC models.")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clf_ckpt",   default="checkpoints/classifier/best.pt",
                        help="Primary classifier checkpoint.")
    parser.add_argument("--clf2_ckpt",  default="checkpoints/classifier2/best.pt",
                        help="Secondary independent classifier checkpoint.")
    parser.add_argument("--save_per_image", action="store_true", default=True,
                        help="Export per-image CSV metrics for every channel and SNR evaluation.")
    parser.add_argument("--no_per_image", dest="save_per_image", action="store_false",
                        help="Disable exporting per-image CSV metrics.")
    parser.add_argument("--all",        action="store_true",
                        help="Evaluate all models.")
    args = parser.parse_args()

    all_results = []

    if args.all:
        configs = [
            ("configs/model1.yaml",             "checkpoints/model1/best.pt"),
            ("configs/model1b.yaml",            "checkpoints/model1b/best.pt"),
            ("configs/model1_fair.yaml",        "checkpoints/model1_fair/best.pt"),
            ("configs/model2.yaml",             "checkpoints/model2/best.pt"),
            ("configs/model2d.yaml",            "checkpoints/model2d/best.pt"),
            ("configs/model3_cnn.yaml",         "checkpoints/model3_cnn/best.pt"),
            ("configs/model3_swinir.yaml",      "checkpoints/model3_swinir/best.pt"),
            ("configs/model3_nosnr.yaml",       "checkpoints/model3_nosnr/best.pt"),
            ("configs/model3b_lam0.001.yaml",   "checkpoints/model3b_lam0.001/best.pt"),
            ("configs/model3b_lam0.003.yaml",   "checkpoints/model3b_lam0.003/best.pt"),
            ("configs/model3b_lam0.01.yaml",    "checkpoints/model3b_lam0.01/best.pt"),
            ("configs/model4.yaml",             "checkpoints/model4/best.pt"),
        ]
        for cfg_path, ckpt_path in configs:
            if os.path.isfile(cfg_path) and os.path.isfile(ckpt_path):
                print(f"\n{'='*60}\nEvaluating: {cfg_path}")
                r = run_single(cfg_path, ckpt_path, args.clf_ckpt, args.clf2_ckpt, save_per_image=args.save_per_image)
                all_results.extend(r)
            else:
                print(f"[SKIP] Missing config or checkpoint for {cfg_path}")
    elif args.config and args.checkpoint:
        all_results = run_single(args.config, args.checkpoint, args.clf_ckpt, args.clf2_ckpt, save_per_image=args.save_per_image)
    else:
        parser.print_help()
        return

    if all_results:
        make_plots(all_results, plot_dir="results/plots")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
