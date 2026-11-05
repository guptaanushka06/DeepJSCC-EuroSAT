"""
data.py
-------
EuroSAT RGB dataset loading and splitting.

Convention
----------
- 'img_net' transform : mean/std normalised to ImageNet stats
  (used as input to the frozen ResNet-18 branch in Models 2-3).
- 'unit' transform    : pixel values in [0, 1]
  (used as the reconstruction TARGET for every model).

Split: 80 / 10 / 10  (train / val / test), fixed seed = 42.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── constants ────────────────────────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 64          # EuroSAT images are already 64×64
NUM_CLASSES = 10

# ImageNet normalisation (for the ResNet-18 encoder branch)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── transforms ───────────────────────────────────────────────────────────────
def get_transforms():
    """
    Returns two transforms:
      - imgnet_tf : ToTensor + ImageNet normalise  → ResNet encoder input
      - unit_tf   : ToTensor only (values in [0,1]) → reconstruction target
    """
    imgnet_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    unit_tf = transforms.Compose([
        transforms.ToTensor(),      # float32 in [0, 1]
    ])
    return imgnet_tf, unit_tf


# ── dual-transform dataset wrapper ───────────────────────────────────────────
class DualTransformEuroSAT(torch.utils.data.Dataset):
    """
    Wraps EuroSAT so each __getitem__ returns:
        (img_imgnet, img_unit, label)

    img_imgnet : ImageNet-normalised tensor (3, 64, 64)
                 → encoder input for ResNet-18 branch (Models 2, 3)
    img_unit   : [0, 1] tensor (3, 64, 64)
                 → reconstruction target for MSE loss (all models)
                 → also the direct input for Model 1 (plain DeepJSCC)
    label      : int class index (0-9)
    """

    def __init__(self, root: str, download: bool = True):
        self.imgnet_tf, self.unit_tf = get_transforms()
        # Store PIL images via base dataset (transform=None keeps PIL)
        self._base = datasets.EuroSAT(root=root, download=download,
                                      transform=None)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        pil_img, label = self._base[idx]
        img_imgnet = self.imgnet_tf(pil_img)   # (3,64,64)  normalised
        img_unit   = self.unit_tf(pil_img)     # (3,64,64)  in [0,1]
        return img_imgnet, img_unit, label


class AugmentedDualTransformSubset(torch.utils.data.Dataset):
    """
    Applies synchronized geometric data augmentations to training images:
      - Random horizontal flip (p=0.5)
      - Random vertical flip (p=0.5)
      - Random 90-degree rotations (0, 90, 180, 270 deg)
    Augmentations are applied to the raw PIL image BEFORE computing
    img_imgnet and img_unit, guaranteeing that input and target undergo
    the exact identical spatial transform.
    """
    def __init__(self, subset: torch.utils.data.Subset):
        self.subset  = subset
        self.base_ds = subset.dataset._base
        self.indices = subset.indices
        self.imgnet_tf, self.unit_tf = get_transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pil_img, label = self.base_ds[real_idx]

        # 1. Random horizontal flip
        if torch.rand(1).item() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random vertical flip
        if torch.rand(1).item() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random 90-degree rotation (0, 90, 180, 270)
        rot_k = int(torch.randint(0, 4, (1,)).item())
        if rot_k == 1:
            pil_img = pil_img.transpose(Image.ROTATE_90)
        elif rot_k == 2:
            pil_img = pil_img.transpose(Image.ROTATE_180)
        elif rot_k == 3:
            pil_img = pil_img.transpose(Image.ROTATE_270)

        img_imgnet = self.imgnet_tf(pil_img)
        img_unit   = self.unit_tf(pil_img)
        return img_imgnet, img_unit, label


# ── split helper ─────────────────────────────────────────────────────────────
def get_splits(root: str = "./data", download: bool = True):
    """
    Returns three non-overlapping Subsets: train, val, test.
    Ratios: 80 / 10 / 10.  Always reproducible (SEED = 42).
    """
    full_ds = DualTransformEuroSAT(root=root, download=download)
    n       = len(full_ds)          # 27,000
    n_train = int(0.80 * n)         # 21,600
    n_val   = int(0.10 * n)         # 2,700
    n_test  = n - n_train - n_val   # 2,700

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )
    return train_ds, val_ds, test_ds


def get_loaders(root: str = "./data",
                batch_size: int = 64,
                num_workers: int = 2,
                download: bool = True,
                augment: bool = False):
    """
    Returns DataLoaders for (train, val, test).
    Pass num_workers=0 when running on Colab to avoid multiprocessing issues.
    If augment=True, applies synchronized flips/rotations to the training set only.
    Validation and test loaders are NEVER augmented.
    """
    train_ds, val_ds, test_ds = get_splits(root=root, download=download)

    train_data = AugmentedDualTransformSubset(train_ds) if augment else train_ds

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader



# ── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading EuroSAT ...")
    tr, va, te = get_loaders(root="./data", batch_size=8, num_workers=0)
    img_net, img_unit, lbl = next(iter(tr))
    print(f"  Train batches : {len(tr)}")
    print(f"  Val   batches : {len(va)}")
    print(f"  Test  batches : {len(te)}")
    print(f"  img_imgnet  shape={img_net.shape}  "
          f"min={img_net.min():.3f}  max={img_net.max():.3f}")
    print(f"  img_unit    shape={img_unit.shape}  "
          f"min={img_unit.min():.3f}  max={img_unit.max():.3f}")
    print(f"  labels      : {lbl.tolist()}")
    print("data.py OK [OK]")
