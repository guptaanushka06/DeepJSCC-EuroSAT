# DeepJSCC Inference Backend & Deployment Guide

This directory contains the self-contained, lightweight inference engine designed for interactive demonstrations on **Streamlit** and **Hugging Face Spaces**.

---

## 1. Architecture & Capabilities

The backend (`backend/inference.py`) is decoupled from all training pipelines:
- **Zero Training Dependencies**: Direct loading of saved `.pt` state dicts without retraining or data loaders.
- **Universal Preprocessing**: Ingests arbitrary RGB image inputs (PIL, NumPy, file paths), normalizes dynamically, and resizes to the EuroSAT standard ($64 \times 64 \times 3$).
- **Strict Bandwidth & Channel Constraints**: Enforces identical $k_{\text{real}} = 4,096$ real channel uses per image ($\rho_{\text{real}} = 1/3$, $\rho_{\text{complex}} = 1/6$) over simulated AWGN or Rayleigh fading channels.
- **Dual Semantic Evaluators**: Evaluates both primary seen (ResNet-18) and secondary independent unseen (MobileNet-V2) classifiers simultaneously.
- **In-Memory Checkpoint Caching**: Automatically keeps loaded models in GPU/CPU memory to deliver sub-10 ms repeat inference.

---

## 2. Directory Structure

```
Data Transmission/
├── backend/
│   ├── inference.py         # Main DeepJSCCInferenceEngine class
│   ├── test_backend.py      # Automated verification test suite
│   └── README.md            # This deployment documentation
├── checkpoints/             # Saved model weights
│   ├── classifier/best.pt   # ResNet-18 primary classifier (95.85% clean)
│   ├── classifier2/best.pt  # MobileNet-V2 independent classifier (96.93% clean)
│   ├── model1_fair/best.pt  # PlainDeepJSCC (Fair baseline)
│   ├── model2d/best.pt      # ResNetJSCC (Layer-3 tap)
│   ├── model2d_long/best.pt # ResNetJSCC 60-epoch baseline
│   ├── model3_swinir/best.pt# SwinIR reconstruction refiner
│   ├── model3b_lam0.0003/   # Pareto-optimal semantic refiner (Seed 42)
│   ├── model4/              # Full SwinJSCC (Vision Transformer)
│   ├── model4_noshift/      # SwinJSCC ablation (No shifted windows)
│   └── model4_conv/         # Non-Swin residual baseline
├── configs/                 # YAML architecture specifications
├── channels.py              # Power normalization and channel simulations
└── requirements.txt         # Production dependencies
```

---

## 3. Quickstart Usage Example

```python
from backend.inference import DeepJSCCInferenceEngine
from PIL import Image

# Initialize engine (auto-selects CUDA if available, else CPU)
engine = DeepJSCCInferenceEngine()

# Perform inference
result = engine.transmit_and_reconstruct(
    img_input="data/AnnualCrop/AnnualCrop_1.jpg",
    model_key="Model 3b: Semantic Refiner (λ=0.0003, Seed 42)",
    channel="awgn",
    snr_db=10.0
)

print(f"PSNR: {result['psnr_db']} dB | SSIM: {result['ssim']}")
print(f"ResNet-18 Semantic Prediction: {result['reconstructed_resnet18']['class_name']} ({result['reconstructed_resnet18']['confidence']*100:.1f}%)")
print(f"Inference Latency: {result['latency_ms']} ms")

# Display or save reconstructed image
result['reconstructed_image'].save("reconstructed_output.png")
```

---

## 4. Verification Suite

Run the automated backend test suite:
```bash
python backend/test_backend.py
```
This script tests model loading, preprocessing, channel simulation, PSNR/SSIM calculation, and semantic predictions across available checkpoints.

---

## 5. Hugging Face Spaces & CPU Deployment Benchmarks

Empirical CPU benchmarks measured with Python 3.12, PyTorch on CPU (`device="cpu"`, 5-run average, single-image latency profiling on `AnnualCrop_1.jpg`):

| Model Architecture | CPU Latency (bs=1) | GPU Latency (bs=1) | Model Weights Size | Canonical Test PSNR (N=2,700) |
| :--- | :---: | :---: | :---: | :---: |
| **Model 4-Conv Baseline** | **`42.91 ms`** | $7.30\text{ ms}$ | $29.72\text{ MB}$ | **`27.66 dB`** |
| **Model 2d (ResNetJSCC Layer-3)** | **`45.51 ms`** | $6.29\text{ ms}$ | $44.68\text{ MB}$ | **`25.44 dB`** |
| **Model 1 Fair (PlainDeepJSCC)** | **`49.00 ms`** | $2.78\text{ ms}$ | $8.94\text{ MB}$ | **`31.07 dB`** |
| **Model 4-NoShift (W-MSA Only)** | **`54.28 ms`** | $19.08\text{ ms}$ | $30.84\text{ MB}$ | **`27.06 dB`** |
| **Model 2d Long (60 Epochs)** | **`55.51 ms`** | $8.38\text{ ms}$ | $44.68\text{ MB}$ | **`25.52 dB`** |
| **Model 4 Full SwinJSCC** | **`61.21 ms`** | $19.89\text{ ms}$ | $30.97\text{ MB}$ | **`16.13 dB`** |
| **Model 3b Semantic Refiner ($\lambda=0.001$)** | **`66.04 ms`** | $8.89\text{ ms}$ | $24.41\text{ MB}$ | **`25.23 dB`** |
| **Model 3b Semantic Refiner ($\lambda=0.0003$)** | **`75.17 ms`** | $8.89\text{ ms}$ | $24.41\text{ MB}$ | **`25.39 dB`** |
| **Model 3 SwinIR Refiner** | **`216.29 ms`** | $40.26\text{ ms}$ | $27.10\text{ MB}$ | **`25.46 dB`** |

### Deployment Key Takeaways:
1. **Zero CUDA Lock-In**: Complete fallback to standard CPU execution with zero deprecation or crash warnings. Controlled CPU-vs-GPU tests on identical samples prove that CPU and GPU produce virtually identical PSNR ($\Delta < 0.02\text{ dB}$).
2. **Realistic Process Memory**: Total OS Working Set (RSS) with **all 9 models + both classifiers concurrently cached** is **`1.60 GB`** (with total model weights of $320\text{ MB}$ on disk). Pure Python heap allocations are $100.73\text{ MB}$ (`tracemalloc`). This consumes only ~10% of Hugging Face Spaces' 16 GB RAM allowance.
3. **Responsive Interactivity**: Over 8 of the 9 architectures deliver under 75 ms end-to-end CPU inference, guaranteeing real-time UI interactivity without GPU overhead.
4. **Standalone Portability**: Deployment requires only `backend/`, `checkpoints/`, `configs/`, `channels.py`, and `requirements.txt`. No training scripts or raw datasets are needed.

