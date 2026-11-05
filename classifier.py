"""
classifier.py
-------------
Train classifiers on CLEAN EuroSAT images.
Supports two independent classifiers:
  1. Primary Classifier (ResNet-18, seed=42):
     checkpoints/classifier/best.pt
     Used as the semantic proxy in Stage 3b loss and evaluation.
  2. Secondary Independent Classifier (MobileNet-V2, seed=1337):
     checkpoints/classifier2/best.pt
     Used ONLY for evaluation (never seen during training loss).

Usage
-----
  # Train primary classifier (ResNet-18, seed 42):
  python classifier.py --train

  # Train secondary independent classifier (MobileNet-V2, seed 1337):
  python classifier.py --train_second

  # Or explicit training:
  python classifier.py --train --arch mobilenet_v2 --seed 1337 --save_dir checkpoints/classifier2 --log_dir results/classifier2

  # Evaluate clean test accuracy:
  python classifier.py --eval --checkpoint checkpoints/classifier/best.pt
  python classifier.py --eval --checkpoint checkpoints/classifier2/best.pt
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
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_splits, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
LR_MIN      = 1e-5
NUM_WORKERS = 2


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClassifierSubset(torch.utils.data.Dataset):
    """Returns (img_imgnet, label) -- [0,1] tensor is not needed here."""
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_imgnet, _, label = self.subset[idx]
        return img_imgnet, label


def get_clf_loaders(root="./data", batch_size=64, num_workers=2, download=True):
    train_ds, val_ds, test_ds = get_splits(root=root, download=download)
    pw = True if num_workers > 0 else False
    train_loader = DataLoader(ClassifierSubset(train_ds),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw,
                              drop_last=True)
    val_loader   = DataLoader(ClassifierSubset(val_ds),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
    test_loader  = DataLoader(ClassifierSubset(test_ds),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
    return train_loader, val_loader, test_loader


def build_classifier(arch: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build classifier with pretrained ImageNet weights, fine-tuned head.
    Supported architectures: 'resnet18', 'mobilenet_v2'
    """
    arch = arch.lower()
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch in ("mobilenet_v2", "mobilenet"):
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown classifier architecture: {arch!r}. Supported: 'resnet18', 'mobilenet_v2'")
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda" and scaler is not None)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_clf(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda")
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def train_classifier(arch: str = "resnet18", seed: int = 42,
                     save_dir: str = "checkpoints/classifier",
                     log_dir: str = "results/classifier",
                     patience: int = 5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining Classifier: arch={arch}, seed={seed}, device={device}")
    print(f"Save dir: {save_dir}  |  Log dir: {log_dir}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_loader, val_loader, test_loader = get_clf_loaders(
        root="./data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print(f"Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches  |  Test: {len(test_loader)} batches")

    model     = build_classifier(arch=arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    csv_path = os.path.join(log_dir, "clf_train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc",
             "val_loss", "val_acc", "lr", "elapsed_s"]
        )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        va_loss, va_acc = evaluate_clf(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}  "
              f"lr={lr_now:.2e}  ({elapsed:.1f}s)")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{tr_loss:.5f}", f"{tr_acc:.5f}",
                f"{va_loss:.5f}", f"{va_acc:.5f}",
                f"{lr_now:.2e}", f"{elapsed:.1f}"
            ])

        state = {
            "epoch":    epoch,
            "arch":     arch,
            "model":    model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_val_acc,
        }
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            state["best_acc"] = best_val_acc
            torch.save(state, os.path.join(save_dir, "best.pt"))
            print(f"  [*] Best classifier saved: val_acc={best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 5:
                print(f"  Early stopping triggered at epoch {epoch+1} (patience={patience})")
                break

        scheduler.step()

    best_state = torch.load(os.path.join(save_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_state["model"])
    _, test_acc = evaluate_clf(model, test_loader, criterion, device)
    print(f"\nClean test accuracy ({arch}, seed {seed}): {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Logs saved: {csv_path}")
    print(f"Checkpoint: {os.path.join(save_dir, 'best.pt')}")
    return test_acc


def eval_only(ckpt_path: str):
    """Evaluate accuracy on clean EuroSAT test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_clf_loaders(
        root="./data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    state = torch.load(ckpt_path, map_location=device)
    arch = state.get("arch", "resnet18")
    model = build_classifier(arch=arch).to(device)
    model.load_state_dict(state["model"])
    criterion = nn.CrossEntropyLoss()
    _, test_acc = evaluate_clf(model, test_loader, criterion, device)
    print(f"Clean test accuracy ({arch}): {test_acc:.4f} ({test_acc*100:.1f}%)")
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Semantic Classifier")
    parser.add_argument("--train", action="store_true",
                        help="Train primary classifier (ResNet-18, seed=42).")
    parser.add_argument("--train_second", action="store_true",
                        help="Train secondary independent classifier (MobileNet-V2, seed=1337).")
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "mobilenet_v2"],
                        help="Classifier architecture.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--save_dir", default="checkpoints/classifier",
                        help="Checkpoint directory.")
    parser.add_argument("--log_dir", default="results/classifier",
                        help="Log directory.")
    parser.add_argument("--eval",  action="store_true",
                        help="Evaluate saved checkpoint on clean test set.")
    parser.add_argument("--checkpoint", default="checkpoints/classifier/best.pt",
                        help="Checkpoint path for --eval mode.")
    args = parser.parse_args()

    if args.train_second:
        train_classifier(
            arch="mobilenet_v2", seed=1337,
            save_dir="checkpoints/classifier2",
            log_dir="results/classifier2"
        )
    elif args.train:
        train_classifier(
            arch=args.arch, seed=args.seed,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    elif args.eval:
        eval_only(args.checkpoint)
    else:
        print("Usage:")
        print("  python classifier.py --train          # ResNet-18 (primary, seed 42)")
        print("  python classifier.py --train_second   # MobileNet-V2 (secondary, seed 1337)")
        print("  python classifier.py --eval --checkpoint checkpoints/classifier/best.pt")
