# System Audit & Codebase Changes (CHANGES.md)

This document tracks all modifications, architectural refactorings, metric implementations, and backward-compatibility validations across the DeepJSCC EuroSAT project.

---

## 1. Summary of Modifications

### A. Evaluation Engine (`eval.py`)
- **Deterministic Evaluation Seeding (`set_eval_seed`)**:
  - Sets identical seeds across models for each `(channel, snr_db)` evaluation point (`seed + channel_offset + round(snr_db * 100)`).
  - Guarantees that every model encounters the exact same channel noise realization, eliminating evaluation jitter and ensuring strictly fair paired comparisons.
- **Per-Image Metrics Export (`--save_per_image`)**:
  - Generates detailed per-image CSVs (`results/<model>/per_image_<channel>_<snr>.csv`) logging:
    - `image_idx`: Zero-indexed test set image identifier.
    - `psnr`: Individual sample reconstruction PSNR.
    - `ssim`: Individual sample structural similarity index (via `torchmetrics.functional.image.structural_similarity_index_measure` with `reduction='none'`).
    - `true_label`: Ground truth EuroSAT class label.
    - `pred_clf1`, `correct_clf1`: ResNet-18 prediction and binary correctness flag.
    - `pred_clf2`, `correct_clf2`: MobileNet-V2 prediction and binary correctness flag.
- **Secondary Independent Classifier Evaluation**:
  - Integrated `classifier2` (MobileNet-V2) alongside `classifier1` (ResNet-18) to evaluate cross-architecture semantic transfer and guard against feature-space overfitting.
- **Sample Visualizations**:
  - Automated saving of paired original vs. reconstructed images at 10 dB AWGN in `results/<model>/samples_<model>_awgn_10dB.png`.

### B. Statistical Rigor Module (`stats.py`)
- **Wilson Score 95% Confidence Intervals**:
  - Implemented exact Wilson score intervals for binomial classification accuracy, replacing symmetric Gaussian approximations on bounded proportions.
- **Paired Bootstrap Difference Testing**:
  - 10,000 resamples with replacement computing non-parametric 95% confidence intervals for $\Delta\text{PSNR}$ and $\Delta\text{SSIM}$.
- **Exact McNemar Hypothesis Testing**:
  - Continuity-corrected $\chi^2$ and exact two-sided binomial tests on discordant classification pairs ($b$ and $c$) to determine whether semantic accuracy improvements are statistically significant at $\alpha = 0.05$.

### C. Efficiency Profiler (`efficiency.py`)
- **Resource and Computational Benchmarking**:
  - Accurate counting of trainable, frozen, and total parameters.
  - Measurement of on-disk checkpoint file sizes (MB).
  - High-precision latency benchmarking (ms/image) and throughput (images/sec) at batch sizes 1 and 64 across both GPU (CUDA Events) and CPU.
  - Peak inference VRAM tracking via `torch.cuda.max_memory_allocated()`.
  - MACs documented with analytical notation to avoid unverified external dependency bloat (`thop`).

### D. Training Pipeline (`train.py`)
- **Dual Learning Rate Support**:
  - Configurable learning rates for encoder/decoder heads (`lr`) and pretrained backbone feature extractors (`lr_backbone`).
- **Mixed Precision & Modern Regularization**:
  - Native `torch.cuda.amp.autocast` support for high-throughput FP16 training on Tensor Cores.
  - Weight decay ($10^{-4}$) and early stopping patience logic.
  - Synchronized data augmentations with reflection padding.

### E. Classifier Architecture (`classifier.py`)
- Implemented dual-evaluator infrastructure supporting both ResNet-18 (95.85% test accuracy) and MobileNet-V2 (96.93% test accuracy) on EuroSAT ($64 \times 64 \times 3$).

### F. Stage 4 Vision Transformer JSCC (`models/swin_jscc.py`)
- **End-to-End Swin Transformer Architecture**:
  - PatchEmbed: $64 \times 64 \times 3 \to 16 \times 16 \times 48$.
  - 3-stage hierarchical encoder with PatchMerging downsampling ($16 \times 16 \to 8 \times 8 \to 4 \times 4$).
  - 1x1 Conv bottleneck mapping to exactly $k_{\text{real}} = 4,096$ real channel uses ($256 \times 4 \times 4$).
  - 3-stage hierarchical decoder with PatchExpand symmetric spatial upsampling.
  - Memory contiguity refactoring: Enforced `.contiguous()` and `.reshape()` to prevent stride/view mismatches in PyTorch.
  - Ablation toggle `use_shift`: Supports both Shifted-Window Multi-head Self-Attention (SW-MSA) and regular Window Multi-head Self-Attention (W-MSA).

### G. Stage 4 Non-Swin Baseline (`models/conv_jscc.py`)
- **Parameter-Matched Residual Convolutional JSCC**:
  - Implements 2-conv ResBlocks with LeakyReLU activations matching the 3-stage hierarchical spatial reduction ($16 \times 16 \to 8 \times 8 \to 4 \times 4$).
  - Exact 4,096 channel uses bottleneck ($256 \times 4 \times 4$).
  - Parameter count: $2,585,092$ parameters (matched within 2.7% of SwinJSCC's $2,657,008$).

### H. Production Deployment Backend (`backend/`)
- **`backend/inference.py`**:
  - Decoupled `DeepJSCCInferenceEngine` requiring zero training dependencies.
  - Universal image preprocessor: accepts arbitrary PIL images, numpy arrays, or file paths; applies Lanczos resizing to $64 \times 64 \times 3$.
  - In-memory model caching for sub-10 ms repeat inference.
  - Computes PSNR, SSIM, latency, and dual-classifier semantic predictions with top-3 class probabilities.
- **`backend/test_backend.py`**:
  - Automated unit test suite verifying all 9 models on GPU. 100% passed.
- **`backend/README.md`**:
  - Deployment guide and Hugging Face Spaces compatibility audit.

---

## 2. Mathematical & Bandwidth Ratio Corrections

In previous project drafts and notes, the bandwidth compression ratio was occasionally written as $4096 / 12288 = 1/6$. This has been clarified and corrected throughout the documentation:
- **Input Dimension**: $64 \times 64 \times 3 = 12,288$ real values.
- **Latent Bottleneck**: $4,096$ real values, transmitted as $2,048$ complex channel symbols ($I/Q$ pairs).
- **Complex Bandwidth Ratio**:
  $$\rho_{\text{complex}} = \frac{2,048\text{ complex symbols}}{12,288\text{ source pixels}} = \frac{1}{6} \approx 0.1667$$
- **Real Bandwidth Ratio**:
  $$\rho_{\text{real}} = \frac{4,096\text{ real values}}{12,288\text{ real values}} = \frac{1}{3} \approx 0.3333$$
- **Main Setting**: $1/6$ complex symbols per input pixel is maintained as the primary reporting convention in literature alignment.

---

## 3. Backward Compatibility Assurance

All changes are strictly backward compatible:
- Every original configuration file in `configs/` remains functional with default parameters.
- Running `python train.py --config <path>` or `python eval.py --config <path> --checkpoint <path>` continues to execute identically without breaking existing workflows.
- No existing checkpoints, log folders, or benchmark tables in `results/final/` were overwritten or invalidated.

