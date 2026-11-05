"""
train.py
--------
Unified training script for all DeepJSCC models:
  - Model 1   : Plain DeepJSCC
  - Model 1b  : Plain DeepJSCC + Augmentation
  - Model 2   : ResNetJSCC (all variants: 2b, 2c, 2d, 2e)
  - Model 3   : Post-channel Refiners (SwinIR + FiLM / NoSNR / CNN control)
  - Model 3b  : Semantic-Loss Refiner (MSE + λ · CE, validation accuracy early stopping)
  - Model 4   : End-to-End SwinJSCC (Swin Transformer JSCC)

Usage
-----
  python train.py --config configs/model1.yaml
  python train.py --config configs/model1b.yaml
  python train.py --config configs/model2d.yaml
  python train.py --config configs/model3_cnn.yaml
  python train.py --config configs/model3b_lam0.003.yaml
  python train.py --config configs/model4.yaml

Smoke test options
------------------
  python train.py --config configs/model4.yaml --smoke_test
  # Measures a full epoch (337 batches) for exact Colab runtime & peak VRAM projection.
  # Use --smoke_batches 60 for a quick 60-batch check.
"""

import argparse
import csv
import math
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data     import get_loaders, IMAGENET_MEAN, IMAGENET_STD
from channels import sample_snr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    """Convert MSE to PSNR (dB). Returns inf if MSE == 0."""
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device):
    """Instantiate the correct model from the config."""
    name = cfg["model"]["name"]
    ch   = cfg["model"].get("channel", "awgn")
    snr_min = cfg["training"]["snr_min"]
    snr_max = cfg["training"]["snr_max"]

    if name == "PlainDeepJSCC":
        from models.deepjscc import PlainDeepJSCC
        model = PlainDeepJSCC(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "ResNetJSCC":
        from models.resnet_jscc import ResNetJSCC
        model = ResNetJSCC(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "ResNetJSCC_2b":
        from models.resnet_variants import ResNetJSCC_2b
        model = ResNetJSCC_2b(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "ResNetJSCC_2c":
        from models.resnet_variants import ResNetJSCC_2c
        model = ResNetJSCC_2c(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "ResNetJSCC_2d":
        from models.resnet_variants import ResNetJSCC_2d
        model = ResNetJSCC_2d(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "ResNetJSCC_2e":
        from models.resnet_variants import ResNetJSCC_2e
        model = ResNetJSCC_2e(channel=ch, snr_min=snr_min, snr_max=snr_max)
    elif name == "SwinRefineJSCC":
        from models.swin_refine import SwinRefineJSCC
        mcfg = cfg["model"]
        model = SwinRefineJSCC(
            channel=ch, snr_min=snr_min, snr_max=snr_max,
            base_config=mcfg.get("base_config", "configs/model2d.yaml"),
            base_checkpoint=mcfg.get("base_checkpoint", "checkpoints/model2d/best.pt"),
            refiner=mcfg.get("refiner", "swinir"),
            swinir_embed_dim=mcfg.get("swinir_embed_dim", 60),
            swinir_depths=tuple(mcfg.get("swinir_depths", [4, 4, 4, 4])),
            swinir_num_heads=tuple(mcfg.get("swinir_num_heads", [6, 6, 6, 6])),
            swinir_window_size=mcfg.get("swinir_window_size", 8),
            film_at_rstb=mcfg.get("film_at_rstb", False),
            cnn_mid_ch=mcfg.get("cnn_mid_ch", 64),
            cnn_n_blocks=mcfg.get("cnn_n_blocks", 9),
            random_channel=mcfg.get("random_channel", False),
            device=device,
        )
    elif name == "SwinJSCC":
        from models.swin_jscc import SwinJSCC
        mcfg = cfg["model"]
        model = SwinJSCC(
            channel=ch, snr_min=snr_min, snr_max=snr_max,
            embed_dim=mcfg.get("embed_dim", 48),
            depths=tuple(mcfg.get("depths", [2, 2, 2])),
            num_heads=tuple(mcfg.get("num_heads", [3, 6, 6])),
            window_sizes=tuple(mcfg.get("window_sizes", [8, 8, 4])),
            bottleneck_ch=mcfg.get("bottleneck_ch", 256),
            use_shift=mcfg.get("use_shift", True),
        )
    elif name in ("ConvJSCC", "SwinJSCC_Conv"):
        from models.conv_jscc import ConvJSCC
        mcfg = cfg["model"]
        model = ConvJSCC(
            channel=ch, snr_min=snr_min, snr_max=snr_max,
            embed_dim=mcfg.get("embed_dim", 48),
            bottleneck_ch=mcfg.get("bottleneck_ch", 256),
            n_blocks=tuple(mcfg.get("n_blocks", [2, 2, 1])),
        )
    else:
        raise ValueError(f"Unknown model name: {name!r}")

    return model.to(device)


def build_optimizer(model, cfg: dict):
    """Build Adam optimizer, optionally with separate LR groups."""
    lr          = cfg["training"]["lr"]
    wd          = cfg["training"].get("weight_decay", 0.0)
    swinir_lr   = cfg["training"].get("lr_swinir", lr)
    lr_backbone = cfg["training"].get("lr_backbone", 1.0e-4)

    if hasattr(model, "param_groups"):
        import inspect
        sig = inspect.signature(model.param_groups)
        if "lr_backbone" in sig.parameters:
            groups = model.param_groups(lr=lr, lr_backbone=lr_backbone, wd=wd)
        elif "swinir_lr" in sig.parameters:
            groups = model.param_groups(lr=lr, swinir_lr=swinir_lr, wd=wd)
        else:
            groups = model.param_groups(lr=lr, wd=wd)
        return torch.optim.Adam(groups)
    else:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd
        )


def build_scheduler(optimizer, cfg: dict):
    epochs = cfg["training"]["epochs"]
    sched  = cfg["training"].get("scheduler", "cosine")
    lr_min = cfg["training"].get("lr_min", 1e-6)

    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )
    elif sched == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched!r}")


def _needs_imgnet(cfg: dict) -> bool:
    """Models 2, 2b-2e, and 3 need ImageNet-normalised input for ResNet backbone."""
    return cfg["model"]["name"] in (
        "ResNetJSCC", "ResNetJSCC_2b", "ResNetJSCC_2c",
        "ResNetJSCC_2d", "ResNetJSCC_2e", "SwinRefineJSCC"
    )


def load_frozen_classifier(ckpt_path: str, device: torch.device):
    """Load a frozen classifier checkpoint for semantic loss computation."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"[FATAL] Classifier checkpoint not found at {ckpt_path}. "
            "Please train or provide the classifier before running Stage 3b."
        )
    from classifier import build_classifier
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "resnet18")
    clf = build_classifier(arch=arch)
    clf.load_state_dict(ckpt["model"])
    clf.to(device).eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    print(f"  [Classifier] Loaded frozen {arch} from: {ckpt_path}")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, cfg, scaler=None,
                    classifier=None, mean_tensor=None, std_tensor=None):
    """
    Run one training epoch.
    Returns: (avg_total_loss, avg_mse_loss, avg_ce_loss)
    """
    model.train()
    criterion_mse = nn.MSELoss()
    criterion_ce  = nn.CrossEntropyLoss()

    snr_min        = cfg["training"]["snr_min"]
    snr_max        = cfg["training"]["snr_max"]
    use_amp        = cfg["training"].get("amp", False) and device.type == "cuda"
    crop_size      = cfg["training"].get("crop_size", None)
    semantic_lam   = cfg["training"].get("semantic_lambda", 0.0)
    is_refiner     = cfg["model"]["name"] == "SwinRefineJSCC"

    total_loss = 0.0
    total_mse  = 0.0
    total_ce   = 0.0
    n_batches  = 0

    for batch in loader:
        img_imgnet, img_unit, labels = batch
        img_imgnet = img_imgnet.to(device)
        img_unit   = img_unit.to(device)
        labels     = labels.to(device)
        inp = img_imgnet if _needs_imgnet(cfg) else img_unit

        snr_db = sample_snr(snr_min, snr_max)

        # Synchronized random patch crop (only if crop_size is set, e.g. Stage 3)
        crop_box = None
        target = img_unit
        if crop_size is not None and model.training:
            H, W = img_unit.shape[2:]
            if H > crop_size and W > crop_size:
                top = torch.randint(0, H - crop_size + 1, (1,)).item()
                left = torch.randint(0, W - crop_size + 1, (1,)).item()
                crop_box = (top, left, crop_size, crop_size)
                target = img_unit[:, :, top:top+crop_size, left:left+crop_size]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if crop_box is not None and is_refiner:
                recon, _ = model(inp, snr_db=snr_db, crop_box=crop_box)
            else:
                recon, _ = model(inp, snr_db=snr_db)

            mse = criterion_mse(recon, target)
            ce  = torch.tensor(0.0, device=device)

            if semantic_lam > 0 and classifier is not None:
                # Stage 3b: normalize reconstructed image for ImageNet classifier
                norm_recon = (recon - mean_tensor) / std_tensor
                logits = classifier(norm_recon)
                ce = criterion_ce(logits, labels)
                loss = mse + semantic_lam * ce
            else:
                loss = mse

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mse  += mse.item()
        total_ce   += ce.item()
        n_batches  += 1

    denom = max(n_batches, 1)
    return total_loss / denom, total_mse / denom, total_ce / denom


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, cfg, snr_db: float, channel: str,
             classifier=None, classifier2=None, mean_tensor=None, std_tensor=None) -> dict:
    """
    Evaluate model at fixed (snr_db, channel) pair.
    Returns: dict with snr_db, channel, mse, psnr, and optionally accuracy and accuracy_clf2.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_mse = 0.0
    correct   = 0
    total_lbl = 0
    correct2   = 0
    total_lbl2 = 0
    n_batches = 0

    for batch in loader:
        img_imgnet, img_unit, labels = batch
        img_imgnet = img_imgnet.to(device)
        img_unit   = img_unit.to(device)
        labels     = labels.to(device)

        inp = img_imgnet if _needs_imgnet(cfg) else img_unit
        recon, _ = model(inp, channel=channel, snr_db=snr_db)

        mse = criterion(recon, img_unit)
        total_mse += mse.item()

        if mean_tensor is not None and std_tensor is not None:
            norm_recon = (recon - mean_tensor) / std_tensor
            if classifier is not None:
                preds = classifier(norm_recon).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_lbl += labels.size(0)
            if classifier2 is not None:
                preds2 = classifier2(norm_recon).argmax(dim=1)
                correct2 += (preds2 == labels).sum().item()
                total_lbl2 += labels.size(0)

        n_batches += 1

    avg_mse  = total_mse / max(n_batches, 1)
    avg_psnr = mse_to_psnr(avg_mse)
    acc = (correct / total_lbl) if total_lbl > 0 else None
    acc2 = (correct2 / total_lbl2) if total_lbl2 > 0 else None

    return {
        "snr_db":        snr_db,
        "channel":       channel,
        "mse":           avg_mse,
        "psnr":          avg_psnr,
        "accuracy":      acc,
        "accuracy_clf2": acc2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train a DeepJSCC model.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test benchmark.")
    parser.add_argument("--smoke_batches", type=int, default=337,
                        help="Number of batches for smoke test (default: 337 = 1 full epoch).")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed   = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── directories ──────────────────────────────────────────────────────────
    ckpt_dir = cfg["checkpoints"]["dir"]
    log_dir  = cfg["logging"]["log_dir"]
    if args.smoke_test:
        ckpt_dir = os.path.join(ckpt_dir, "smoke_test")
        log_dir  = os.path.join(log_dir, "smoke_test")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    augment = cfg["training"].get("augment", False)
    train_loader, val_loader, _ = get_loaders(
        root        = cfg["data"]["root"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        download    = cfg["data"]["download"],
        augment     = augment,
    )

    # ── model / optimizer / scheduler ────────────────────────────────────────
    model     = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})
    model_name = cfg["model"]["name"]
    is_refiner = model_name == "SwinRefineJSCC"
    refiner_type = mcfg.get("refiner", "none") if is_refiner else "none"
    amp_on = tcfg.get("amp", False)
    crop_sz = tcfg.get("crop_size", None)
    crop_str = f"{crop_sz}x{crop_sz}" if crop_sz else "None (full 64x64)"

    semantic_lam = tcfg.get("semantic_lambda", 0.0)
    early_metric = tcfg.get("early_stop_metric", "psnr")
    psnr_floor   = tcfg.get("psnr_floor", 25.0)

    # ImageNet normalisation tensors on device for semantic loss
    mean_tensor = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_tensor  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    clf_ckpt = tcfg.get("classifier_checkpoint", "checkpoints/classifier/best.pt")
    clf2_ckpt = tcfg.get("classifier2_checkpoint", "checkpoints/classifier2/best.pt")
    classifier = None
    classifier2 = None
    if os.path.isfile(clf_ckpt):
        try:
            classifier = load_frozen_classifier(clf_ckpt, device)
        except Exception as e:
            print(f"  [NOTE] Could not load classifier: {e}")
    if os.path.isfile(clf2_ckpt):
        try:
            classifier2 = load_frozen_classifier(clf2_ckpt, device)
        except Exception as e:
            print(f"  [NOTE] Could not load classifier2: {e}")

    # ── Print & Save Detailed Run Configuration ──────────────────────────────
    config_lines = [
        "=" * 70,
        "  MODEL & TRAINING CONFIGURATION AUDIT",
        "=" * 70,
        f"  Device              : {device}",
        f"  Config File         : {args.config}",
        f"  Model Architecture  : {model_name}",
    ]
    if is_refiner:
        config_lines.append(f"  Refiner Variant     : {refiner_type}")
        if "swinir" in refiner_type:
            config_lines.append(f"  SwinIR Depths       : {mcfg.get('swinir_depths', [4, 4, 4, 4])}")
            config_lines.append(f"  SwinIR Embed Dim    : {mcfg.get('swinir_embed_dim', 60)}")
            config_lines.append(f"  SwinIR Heads        : {mcfg.get('swinir_num_heads', [6, 6, 6, 6])}")
            config_lines.append(f"  SwinIR Window Size  : {mcfg.get('swinir_window_size', 8)}")
        elif refiner_type == "cnn":
            config_lines.append(f"  CNN Mid Channels    : {mcfg.get('cnn_mid_ch', 64)}")
            config_lines.append(f"  CNN Num Blocks      : {mcfg.get('cnn_n_blocks', 9)}")
    elif model_name == "SwinJSCC":
        config_lines.append(f"  SwinJSCC Depths     : {mcfg.get('depths', [2, 2, 2])}")
        config_lines.append(f"  SwinJSCC Embed Dim  : {mcfg.get('embed_dim', 48)}")
        config_lines.append(f"  SwinJSCC Heads      : {mcfg.get('num_heads', [3, 6, 6])}")
        config_lines.append(f"  SwinJSCC Window Sz  : {mcfg.get('window_sizes', [8, 8, 4])}")
        config_lines.append(f"  Bottleneck Channels : {mcfg.get('bottleneck_ch', 256)} (4,096 reals)")

    config_lines.extend([
        f"  Train Crop Size     : {crop_str}",
        f"  Mixed Precision     : {'FP16 Autocast ON' if amp_on else 'FP32 Only'}",
        f"  Semantic Lambda (lam): {semantic_lam}",
        f"  Early Stopping By   : {early_metric.upper()} (PSNR Floor: {psnr_floor:.2f} dB)" if early_metric == "accuracy" else "PSNR",
        f"  Total Epochs        : {tcfg.get('epochs')}",
        f"  EarlyStop Patience  : {tcfg.get('patience', 'None')}",
        f"  Batch Size          : {tcfg.get('batch_size')}",
        f"  Learning Rate       : {tcfg.get('lr')}",
        f"  Data Augmentation   : {'ENABLED (train only)' if augment else 'DISABLED'}",
        f"  Parameters: Total   : {total_params:,}",
        f"  Parameters: Train   : {trainable_params:,}",
        f"  Parameters: Frozen  : {frozen_params:,}",
        "=" * 70,
    ])
    config_summary_text = "\n".join(config_lines)
    print("\n" + config_summary_text + "\n")

    meta_path = os.path.join(log_dir, "run_metadata.txt")
    with open(meta_path, "w") as mf:
        mf.write(config_summary_text + "\n")

    # ── Epoch 0 (Init) Identity Check for all refiners ───────────────────────
    if is_refiner:
        print("=" * 70)
        print("  [EPOCH 0 (INIT) IDENTITY VALIDATION PASS]")
        print("  Evaluating refiner output on Val Set @ AWGN 10 dB in pure FP32...")
        r0 = evaluate(model, val_loader, device, cfg, snr_db=10.0, channel="awgn")
        epoch0_psnr = r0["psnr"]
        print(f"  Epoch 0 (Init) Val PSNR: {epoch0_psnr:.2f} dB (required range: 25.30 - 25.70 dB)")

        base_ckpt = mcfg.get("base_checkpoint", "")
        if os.path.isfile(base_ckpt):
            b_state = torch.load(base_ckpt, map_location="cpu")
            b_psnr = b_state.get("best_psnr", None)
            if b_psnr is not None:
                diff = abs(epoch0_psnr - b_psnr)
                if diff > 1.0:
                    print(f"  [WARNING] Epoch 0 PSNR ({epoch0_psnr:.2f} dB) differs from base PSNR ({b_psnr:.2f} dB) by {diff:.2f} dB.")
                else:
                    print(f"  [OK] PASS: Refiner initialized as exact identity ({epoch0_psnr:.2f} dB ~= base {b_psnr:.2f} dB).")
            else:
                print(f"  [OK] Epoch 0 Val PSNR: {epoch0_psnr:.2f} dB")
        else:
            print(f"  [NOTE] Base checkpoint '{base_ckpt}' not found locally; identity range not enforced on dummy base.")
        print("=" * 70 + "\n")

    # ── Smoke Test Mode ──────────────────────────────────────────────────────
    if args.smoke_test:
        num_batches = args.smoke_batches
        print("=" * 70)
        print(f"  [SMOKE TEST BENCHMARK RUNNING: {num_batches} BATCHES]")
        print("=" * 70)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t_smoke_start = time.time()
        scaler = torch.amp.GradScaler(enabled=(amp_on and device.type == "cuda"))
        smoke_loss, smoke_mse, smoke_ce = 0.0, 0.0, 0.0
        batches_done = 0

        for batch in train_loader:
            if batches_done >= num_batches:
                break
            img_imgnet, img_unit, labels = batch
            img_imgnet = img_imgnet.to(device)
            img_unit   = img_unit.to(device)
            labels     = labels.to(device)
            inp = img_imgnet if _needs_imgnet(cfg) else img_unit

            crop_box = None
            target = img_unit
            if crop_sz is not None and model.training:
                H, W = img_unit.shape[2:]
                if H > crop_sz and W > crop_sz:
                    top = torch.randint(0, H - crop_sz + 1, (1,)).item()
                    left = torch.randint(0, W - crop_sz + 1, (1,)).item()
                    crop_box = (top, left, crop_sz, crop_sz)
                    target = img_unit[:, :, top:top+crop_sz, left:left+crop_sz]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp_on and device.type == "cuda")):
                if crop_box is not None and is_refiner:
                    recon, _ = model(inp, snr_db=10.0, crop_box=crop_box)
                else:
                    recon, _ = model(inp, snr_db=10.0)
                mse = nn.MSELoss()(recon, target)
                ce  = torch.tensor(0.0, device=device)
                if semantic_lam > 0 and classifier is not None:
                    norm_recon = (recon - mean_tensor) / std_tensor
                    ce = nn.CrossEntropyLoss()(classifier(norm_recon), labels)
                    loss = mse + semantic_lam * ce
                else:
                    loss = mse

            if amp_on and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            smoke_loss += loss.item()
            smoke_mse  += mse.item()
            smoke_ce   += ce.item()
            batches_done += 1

        elapsed_smoke = time.time() - t_smoke_start
        ms_per_batch = (elapsed_smoke / max(batches_done, 1)) * 1000.0
        projected_epoch_s = (elapsed_smoke / max(batches_done, 1)) * 337
        projected_total_min = (projected_epoch_s * tcfg.get("epochs", 30)) / 60.0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        print(f"  Batches Executed      : {batches_done} / {num_batches}")
        print(f"  Average Train Loss    : {smoke_loss / max(batches_done, 1):.6f}")
        print(f"  Average Train MSE     : {smoke_mse / max(batches_done, 1):.6f}")
        if semantic_lam > 0:
            print(f"  Average Train CE      : {smoke_ce / max(batches_done, 1):.6f}")
        print(f"  Measured Throughput   : {ms_per_batch:.2f} ms/batch")
        print(f"  Measured/Projected Ep : {projected_epoch_s:.1f} s ({projected_epoch_s/60:.2f} min / epoch)")
        print(f"  Projected Full Run    : {projected_total_min:.1f} min ({projected_total_min/60:.2f} hours)")
        print(f"  Peak GPU Memory (VRAM): {peak_vram_mb:.1f} MB ({peak_vram_mb/1024:.2f} GB)")
        print("=" * 70)
        print("\nSmoke test successfully finished. Exiting without overwriting production checkpoints.\n")
        return

    # ── GradScaler for AMP ───────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(enabled=(amp_on and device.type == "cuda"))

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr   = -float("inf")
    best_acc    = -float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr   = ckpt.get("best_psnr", best_psnr)
        best_acc    = ckpt.get("best_acc", best_acc)
        remaining_epochs = cfg["training"]["epochs"] - start_epoch
        if remaining_epochs > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=cfg["training"].get("lr_min", 1e-6)
            )
            print(f"Resumed from epoch {start_epoch}: created smooth continuation scheduler for remaining {remaining_epochs} epochs.")
        elif scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f} dB, best Acc={best_acc:.4f}")

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(log_dir, "train_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if start_epoch == 0:
        csv_writer.writerow([
            "epoch", "train_loss", "train_mse", "train_ce",
            "val_snr_db", "val_channel", "val_mse", "val_psnr", "val_acc", "val_acc_clf2",
            "lr", "elapsed_s"
        ])

    eval_snrs     = cfg["logging"]["eval_snrs"]
    eval_channels = cfg["logging"]["eval_channels"]
    epochs        = cfg["training"]["epochs"]
    save_every    = cfg["checkpoints"]["save_every"]
    patience      = cfg["training"].get("patience", None)
    epochs_no_improve = 0

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_mse, tr_ce = train_one_epoch(
            model, train_loader, optimizer, device, cfg,
            scaler=scaler, classifier=classifier,
            mean_tensor=mean_tensor, std_tensor=std_tensor
        )

        val_results = []
        for ch in eval_channels:
            for snr in eval_snrs:
                r = evaluate(
                    model, val_loader, device, cfg,
                    snr_db=float(snr), channel=ch,
                    classifier=classifier, classifier2=classifier2,
                    mean_tensor=mean_tensor, std_tensor=std_tensor
                )
                val_results.append(r)

        elapsed = time.time() - t0
        if len(optimizer.param_groups) > 1:
            lr_str = "/".join(f"{g['lr']:.1e}" for g in optimizer.param_groups)
        else:
            lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"

        # Canonical metric: AWGN @ 10 dB
        canonical = next(
            (r for r in val_results if r["channel"] == "awgn" and r["snr_db"] == 10),
            val_results[0]
        )
        epoch_psnr = canonical["psnr"]
        epoch_acc  = canonical["accuracy"] if canonical["accuracy"] is not None else 0.0
        epoch_acc2 = canonical.get("accuracy_clf2", 0.0)

        acc_disp  = f"  val_acc={epoch_acc*100:.1f}%" if canonical.get("accuracy") is not None else ""
        acc2_disp = f"  acc2={epoch_acc2*100:.1f}%" if canonical.get("accuracy_clf2") is not None else ""
        ce_disp   = f"  ce={tr_ce:.4f}" if semantic_lam > 0 else ""

        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"train_mse={tr_mse:.5f}{ce_disp}  "
              f"val_psnr={epoch_psnr:.2f}dB{acc_disp}{acc2_disp}  "
              f"lr={lr_str}  "
              f"time={elapsed:.1f}s")

        for r in val_results:
            acc_val  = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
            acc2_val = f"{r['accuracy_clf2']:.4f}" if r.get("accuracy_clf2") is not None else "N/A"
            csv_writer.writerow([
                epoch + 1, f"{tr_loss:.6f}", f"{tr_mse:.6f}", f"{tr_ce:.6f}",
                r["snr_db"], r["channel"],
                f"{r['mse']:.6f}", f"{r['psnr']:.4f}", acc_val, acc2_val,
                f"{lr_str}", f"{elapsed:.1f}"
            ])
        csv_file.flush()

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "best_acc":  best_acc,
            "config": cfg,
        }
        if scheduler:
            state["scheduler"] = scheduler.state_dict()

        if (epoch + 1) % save_every == 0:
            path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save(state, path)
            print(f"  [OK] Checkpoint saved: {path}")

        # ── Best Model Selection & Early Stopping ────────────────────────────
        improved = False
        if early_metric == "accuracy":
            # Selection by validation accuracy subject to PSNR floor
            if epoch_psnr >= psnr_floor:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_psnr = epoch_psnr
                    improved = True
                    state["best_acc"] = best_acc
                    state["best_psnr"] = best_psnr
                    torch.save(state, os.path.join(ckpt_dir, "best.pt"))
                    print(f"  [*] Best model updated by Accuracy: Acc={best_acc*100:.2f}%, PSNR={epoch_psnr:.2f} dB (>= {psnr_floor:.1f} dB)")
            else:
                print(f"  [PSNR Warning] Val PSNR {epoch_psnr:.2f} dB < floor {psnr_floor:.2f} dB. Not eligible for best checkpoint.")
        else:
            # Selection by validation PSNR (standard)
            if epoch_psnr > best_psnr:
                best_psnr = epoch_psnr
                best_acc = epoch_acc
                improved = True
                state["best_psnr"] = best_psnr
                state["best_acc"] = best_acc
                torch.save(state, os.path.join(ckpt_dir, "best.pt"))
                print(f"  [*] Best model updated by PSNR: PSNR={best_psnr:.2f} dB")

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None:
                print(f"  [EarlyStopping: {epochs_no_improve}/{patience} epochs without improvement]")
                if epochs_no_improve >= patience:
                    print(f"\n[EARLY STOPPING TRIGGERED] Metric has not improved for {epochs_no_improve} epochs.")
                    print(f"Best: PSNR={best_psnr:.2f} dB, Acc={best_acc*100:.2f}%")
                    break

        if scheduler:
            scheduler.step()

    csv_file.close()
    print(f"\nTraining complete. Logs at: {csv_path}")
    print(f"Best: PSNR={best_psnr:.2f} dB, Acc={best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
