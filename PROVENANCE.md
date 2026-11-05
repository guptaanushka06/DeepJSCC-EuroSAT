# Checkpoint Provenance & Experimental Audit (PROVENANCE.md)

This document provides a verifiable, immutable record of all primary checkpoints in the DeepJSCC EuroSAT project.
All checkpoints reside strictly in `checkpoints/` and were trained and evaluated on the local workstation (RTX 4050 Laptop GPU, Windows 11).

---

## 1. Checkpoint Registry & Cryptographic Hashes

| Checkpoint Path | Architecture | File Size (Bytes) | Last Modified | SHA256 Hash |
|---|---|---|---|---|
| `checkpoints/classifier/best.pt` | ResNet-18 Head (Primary) | 134,296,425 | 2026-10-24 19:11:18 | `8dc67a2e85953bdaf7ab4dc9db444637a7cb8645504f619a42bf36f0b668bd71` |
| `checkpoints/classifier2/best.pt` | MobileNet-V2 Head (Independent) | 27,179,849 | 2026-10-24 19:23:53 | `9c6544386b52279cb5d4c6e70daa1c0ac523591ce46308be79789b07620e38b5` |
| `checkpoints/model1_fair/best.pt` | PlainDeepJSCC (Fair baseline) | 9,370,816 | 2026-10-24 20:40:52 | `9dc7667cf0d4f9ad5df68ec961b6cf29bd9b7213d969977e1129ea0e680bf9eb` |
| `checkpoints/model2d/best.pt` | ResNetJSCC_2d (Layer-3 dual LR) | 46,845,979 | 2026-10-24 21:49:58 | `70c6f49e5ed0202a134a73f37440386278e76b173b52e5bc57ce33edb1e2b101` |
| `checkpoints/model3_swinir/best.pt` | SwinRefineJSCC (SwinIR Refiner) | 28,411,309 | 2026-10-24 22:33:32 | `8832119404f14cbc738770346aed50e3feb5e00ba59dbcd444a426a691c1a9ec` |
| `checkpoints/model3b_lam0.001/best.pt`| SwinRefineJSCC (Semantic Refiner $\lambda=0.001$) | 25,600,822 | 2026-10-24 23:00:06 | `33a3606d97ed9d4824ef5c380732829fa8c2a73ebe5113251a0bb5478691a947` |
| `checkpoints/model2d_long/best.pt` | ResNetJSCC_2d Long (60 epochs) | 46,846,363 | 2026-11-03 21:04:12 | `3a88583112cd03d95aceaafe66336a55be396f5271b6823314980365a4d41fcb` |
| `checkpoints/model3b_lam0.0003/best.pt` | SwinRefineJSCC ($\lambda=0.0003$, Seed 42) | 25,600,822 | 2026-11-03 19:10:45 | `96bdf172a36d444fe0f1f1ec1f4bafcfb334c0e1f77a4c1f8e9476da75277597` |
| `checkpoints/model3b_lam0.0003_s43/best.pt` | SwinRefineJSCC ($\lambda=0.0003$, Seed 43) | 25,600,822 | 2026-11-04 02:48:19 | `629439adabed5b2e6cb8830822e9ee512c8ae165b1d83638eb50ddc71e339b6d` |
| `checkpoints/model3b_lam0.0003_s44/best.pt` | SwinRefineJSCC ($\lambda=0.0003$, Seed 44) | 25,600,822 | 2026-11-04 03:39:27 | `2326a509bad3c8cdd97145a7034e0e42eea5bc6ba107ebf01a29ab4d9666eb05` |
| `checkpoints/model3b_lam0.003/best.pt` | SwinRefineJSCC ($\lambda=0.003$, Seed 42) | 25,600,822 | 2026-11-04 00:07:33 | `03afca46b443d6bd4b50f5ddce11be941a82292c4fdd725a89ed2d3dffdc6800` |
| `checkpoints/model4/best.pt` | SwinJSCC (Full Stage 4 Transformer) | 32,472,791 | 2026-11-04 16:44:00 | `4d64273e6235cf0bdb55074c6ada1669a07d75bc80c7c66f1ffe4ea63a4a4d9d` |
| `checkpoints/model4_noshift/best.pt` | SwinJSCC-NoShift (Ablation C, W-MSA) | 32,341,281 | 2026-11-04 20:46:00 | `d1ff28c434da765338b9100ce128a1c9074f0b55f074c7fcb09a7d92dd2180f1` |
| `checkpoints/model4_conv/best.pt` | ConvJSCC (Ablation A, Non-Swin Baseline) | 31,165,117 | 2026-11-04 18:28:00 | `5e8523b2df6c087c3d8f12cecc7da4a0b84528a8ccce2f51c42e5bfd7a28cb09` |

---

## 2. Checkpoint Details & Training Conditions

### A. Primary Classifier (`classifier/best.pt`)
- **Architecture**: ResNet-18 (ImageNet pretrained backbone + custom EuroSAT 10-class linear projection head).
- **Purpose**: Semantic evaluation target and supervision source for Model 3b semantic refiner.
- **Top-1 Test Accuracy**: **95.85%** (2,588 / 2,700 images).

### B. Secondary Independent Classifier (`classifier2/best.pt`)
- **Architecture**: MobileNet-V2 (`mobilenet_v2` with fine-tuned classifier head).
- **Purpose**: Zero-leakage evaluator to test semantic feature transfer. Not seen during training of any DeepJSCC model or refiner.
- **Top-1 Test Accuracy**: **96.93%** (2,617 / 2,700 images).

### C. Model 1 Fair (`model1_fair/best.pt`)
- **Configuration**: `configs/model1_fair.yaml`
- **Architecture**: Pure Convolutional DeepJSCC (5-layer Conv encoder, power normalization, AWGN/Rayleigh channel, 5-layer Transposed Conv decoder).
- **Training Recipe**: 30 epochs, batch size 64, Adam optimizer, initial lr $10^{-3}$, cosine annealing to $10^{-5}$, weight decay $10^{-4}$, data augmentation (random horizontal flip), AMP enabled.
- **Key 10 dB AWGN Metrics**:
  - PSNR: **31.07 dB**
  - SSIM: **0.8871**
  - ResNet-18 Acc: **66.37%**
  - MobileNet-V2 Acc: **42.04%**

### D. Model 2d (`model2d/best.pt`)
- **Configuration**: `configs/model2d.yaml`
- **Architecture**: ResNet-18 Feature Transceiver (Layer-3 feature extraction, unfrozen layer3 with dual learning rate: $\text{lr}=10^{-3}$ for JSCC/decoder, $\text{lr}_{\text{backbone}}=10^{-4}$ for layer3).
- **Training Recipe**: 30 epochs, batch size 64, weight decay $10^{-4}$, AMP enabled, patience 8.
- **Key 10 dB AWGN Metrics**:
  - PSNR: **25.44 dB**
  - SSIM: **0.6848**
  - ResNet-18 Acc: **24.59%**
  - MobileNet-V2 Acc: **16.59%**

### E. Model 3 SwinIR (`model3_swinir/best.pt`)
- **Configuration**: `configs/model3_swinir.yaml`
- **Architecture**: Frozen Model 2d base + SwinIR post-reconstruction enhancement module conditioned on channel SNR via FiLM modulation.
- **Training Recipe**: 6 epochs, MSE loss only, lr $10^{-4}$, batch size 64.
- **Key 10 dB AWGN Metrics**:
  - PSNR: **25.46 dB**
  - SSIM: **0.6863**
  - ResNet-18 Acc: **24.41%**
  - MobileNet-V2 Acc: **15.74%**

### F. Model 3b $\lambda=0.001$ (`model3b_lam0.001/best.pt`)
- **Configuration**: `configs/model3b_lam0.001.yaml`
- **Architecture**: Frozen Model 2d base + parameter-matched 9-block residual CNN refiner (669k params) trained with joint fidelity + semantic objective:
  $$\mathcal{L} = \text{MSE}(\hat{x}, x) + \lambda \cdot \mathcal{L}_{\text{CE}}(\text{Classifier}_1(\hat{x}), y)$$
- **Training Recipe**: 6 epochs, $\lambda = 0.001$, lr $10^{-4}$, batch size 64.
- **Key 10 dB AWGN Metrics**:
  - PSNR: **25.23 dB** ($-0.21\text{ dB}$ trade-off vs Model 2d)
  - SSIM: **0.6532** ($-0.0316$ trade-off vs Model 2d)
  - ResNet-18 Acc: **72.33%** ($+47.74$ percentage points over Model 2d)
  - MobileNet-V2 Acc: **41.30%** ($+24.71$ percentage points over Model 2d)

---

## 3. Scientific Finding on Semantic Transfer

The evaluation across both classifiers confirms a critical scientific nuance:
1. Optimizing with $\lambda=0.001$ against ResNet-18 produces a massive $+47.74$ pp gain on the training classifier ($24.59\% \to 72.33\%$).
2. When evaluated on the completely independent MobileNet-V2 classifier (which was never seen by the refiner), the accuracy increases from $16.59\%$ to $41.30\%$ ($+24.71\text{ pp}$).
3. This establishes genuine semantic transfer across distinct neural architectures, though with an expected transfer gap and a measurable reconstruction fidelity penalty ($\Delta\text{PSNR} = -0.21\text{ dB}$, $\Delta\text{SSIM} = -0.0316$).

