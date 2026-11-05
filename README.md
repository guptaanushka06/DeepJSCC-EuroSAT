---
title: DeepJSCC EuroSAT Transmission Suite
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.62.0
app_file: app.py
pinned: false
---

# Adaptive Semantic JSCC with SwinIR Refinement
## README

A reproducible PyTorch implementation of "Adaptive Semantic JSCC with SwinIR Refinement for Image Transmission over Noisy Channels".

---

## Project Structure

```
Data Transmission/
├── data.py                # EuroSAT loader, dual transforms, 80/10/10 split
├── channels.py            # AWGN, Rayleigh, power_normalize + unit tests
├── models/
│   ├── deepjscc.py        # Model 1 & 1b: Plain DeepJSCC
│   ├── resnet_jscc.py     # Model 2: ResNet-18 + DeepJSCC
│   ├── resnet_variants.py # Models 2b, 2c, 2d, 2e (Layer-3 unfrozen)
│   ├── swin_refine.py     # Stage 3: SwinIR+FiLM / NoSNR / CNN control
│   ├── swin_jscc.py       # Stage 4: End-to-End Swin Transformer JSCC
│   └── network_swinir.py  # Vendored from JingyunLiang/SwinIR (Apache 2.0)
├── configs/
│   ├── model1.yaml        # Plain DeepJSCC (80 epochs, no augmentation)
│   ├── model1b.yaml       # Plain DeepJSCC + Augmentation (80 epochs)
│   ├── model1_fair.yaml   # Plain DeepJSCC + Augmentation (30 epochs, FAIR baseline)
│   ├── model2.yaml        # Teacher's pipeline (Layer-4 bottleneck)
│   ├── model2d.yaml       # Winning base model (Layer-3 unfrozen)
│   ├── model3_swinir.yaml # Stage 3: SwinIR + FiLM SNR embed
│   ├── model3_nosnr.yaml  # Stage 3: SwinIR ablation (no SNR conditioning)
│   ├── model3_cnn.yaml    # Stage 3: Parameter-matched CNN control
│   ├── model3b_lam0.003.yaml # Stage 3b: Semantic loss refiner (λ=0.003, primary)
│   ├── model3b_lam0.001.yaml # Stage 3b: Semantic loss refiner (λ=0.001)
│   ├── model3b_lam0.01.yaml  # Stage 3b: Semantic loss refiner (λ=0.01)
│   └── model4.yaml        # Stage 4: End-to-End SwinJSCC (4,096 reals)
├── classifier.py          # Primary (ResNet-18) & Secondary (MobileNet-V2) classifiers
├── train.py               # Unified training & benchmarking script
├── eval.py                # Full evaluation: PSNR, SSIM, dual accuracy + plots + samples
├── compare_results.py     # Standalone result aggregator & summary tables
├── diagnose_model2.py     # Diagnostic suite for the Layer-4 bottleneck
├── run_all.py             # Complete pipeline orchestration (train → eval → compare)
├── requirements.txt
└── results/               # CSVs, plots (auto-created)
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Channel Unit Tests
```bash
python channels.py
```

### 3. Full Training → Evaluation Pipeline
```bash
# Run everything in order (recommended):
python run_all.py

# Or run individual stages:
python run_all.py --stage 1    # Model 1 (fair) only
python run_all.py --stage 2    # Model 2d only
python run_all.py --stage 3    # Stage 3 refiners
python run_all.py --stage 3b   # Stage 3b semantic loss
python run_all.py --stage 4    # SwinJSCC
python run_all.py --stage eval # Full evaluation + comparison
```

### 4. Manual Training (any single model)
```bash
python train.py --config configs/model1_fair.yaml
python train.py --config configs/model2d.yaml
python train.py --config configs/model3b_lam0.003.yaml
python train.py --config configs/model4.yaml
```

### 5. Evaluation
```bash
# Evaluate a single model:
python eval.py --config configs/model2d.yaml --checkpoint checkpoints/model2d/best.pt \
               --clf_ckpt checkpoints/classifier/best.pt --clf2_ckpt checkpoints/classifier2/best.pt

# Evaluate ALL trained models:
python eval.py --all

# Aggregate results into comparison tables and plots:
python compare_results.py
python compare_results.py --count_params
```

---

## Dataset

[EuroSAT RGB](https://github.com/phelber/EuroSAT) — 27,000 images, 64×64×3, 10 land-use classes.
Downloaded automatically by `torchvision.datasets.EuroSAT(download=True)`.

**Split (seed=42):** 21,600 train / 2,700 val / 2,700 test (80/10/10).

---

## Channel Convention

Bandwidth ratio **ρ = 1/6** — 2,048 complex symbols per image.
(Latent vector: 4,096 real values = 2×2,048)

| Symbol | Meaning |
|--------|---------|
| k | Number of complex channel uses = 2,048 (4,096 real symbols) |
| n | Source real values = 3×64×64 = 12,288 |
| ρ | k/n = 2048/12288 = **1/6** (complex channel uses / source reals) |
| ρ_real | k_real/n = 4096/12288 = **1/3** (real channel uses / source reals) |

Power normalisation: average power per complex symbol = 1.
Noise: σ² = 1/(2·SNR) per real component.

---

## The 4 Models

| # | Model | Architecture | Purpose |
|---|-------|-------------|---------|
| 1 | Plain DeepJSCC | CNN enc → channel → CNN dec | Baseline |
| 2 | ResNet-18 + DeepJSCC | Frozen ResNet-18 → JSCC → CNN dec | Teacher's pipeline |
| 2d | Layer-3 Unfrozen | Like Model 2 but Layer-3 + fine-tuning | Best base model |
| 3 | Model 2d + SwinIR | Frozen 2d + CNN/SwinIR refiner | Main refinement model |
| 3b | Semantic Refiner | Model 3 + CE classification loss | Semantic preservation |
| 4 | SwinJSCC | End-to-end Swin Transformer | Transformer comparison |

---

## IMPORTANT NOTE ON LABELS & SEMANTIC LOSS

> **Crucial Rule on Semantic Labels:**
> In **Stage 3b**, ground-truth class labels are used **ONLY** in the training loss as a semantic regularization term:
> $$\mathcal{L} = \text{MSE}(\hat{x}, x) + \lambda \cdot \text{CE}(f_{\text{frozen}}(\hat{x}), y)$$
> Class labels are **NEVER** transmitted across the communication channel, are **NEVER** available at the receiver, and are **NEVER** used during inference/testing. Transmission is strictly end-to-end pixel JSCC.

---

## Classifiers for Evaluation

Two separate, independent classifiers are trained on clean EuroSAT:
1. **Primary Classifier:** Pretrained ResNet-18 (`seed=42`). Used in Stage 3b training loss and evaluation.
2. **Secondary Classifier:** Pretrained MobileNet-V2 (`seed=1337`). An independent architecture and random seed, used **ONLY** for evaluation to ensure semantic gains are not an artifact of overfitting to the primary classifier.

```bash
# Train Primary Classifier (ResNet-18, seed 42):
python classifier.py --train

# Train Secondary Independent Classifier (MobileNet-V2, seed 1337):
python classifier.py --train_second
```

---

## Fair Comparison Note

Model 1 was originally trained for 80 epochs without augmentation. For fair architectural comparison, use `model1_fair.yaml` (30 epochs + augmentation), which matches the training recipe of Models 2d, 3, and 4.

---

## Expected Results

- **Model 2** (Layer-4) will have lower PSNR than Model 1, because ResNet Layer-4 on 64×64 input produces only 512×2×2 = 2,048 values (spatial bottleneck).
- **Model 2d** (Layer-3, unfrozen) substantially improves over Model 2 (~+4.68 dB PSNR).
- **Model 3** (SwinIR refiner) should improve Model 2d, primarily in perceptual quality.
- **Model 4** (SwinJSCC) provides an end-to-end transformer comparison.

Report whatever you get honestly. Do not tune to force wins.

---

## Attribution

SwinIR code in `models/network_swinir.py` is from [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) (Apache 2.0 License).
Adapted for `img_size=64, window_size=8` with pinned commit hash intact.
