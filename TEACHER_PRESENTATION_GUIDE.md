# Comprehensive Presentation & Defense Guide
## DeepJSCC with Adaptive SwinIR Refinement & Semantic Loss

This guide provides everything you need to confidently explain your project, discuss the mathematical fundamentals, justify your architectural choices, and answer your teacher's questions.

---

## 1. Executive Summary (The 60-Second Elevator Pitch)

> *"My project investigates **Deep Joint Source-Channel Coding (DeepJSCC)** for transmitting satellite and aerial imagery over noisy wireless channels (EuroSAT, $64 \times 64 \times 3$).*
>
> *Traditional wireless transmission separates source compression (like JPEG) from channel coding (like LDPC). Under finite bandwidth and changing channel noise, this suffers from the **'cliff effect'** (catastrophic failure when SNR drops).*
>
> *We implemented an end-to-end DeepJSCC architecture with three major innovations:*
> 1. *We diagnosed and solved a critical architectural flaw in the standard ResNet-18 baseline: the **Layer-4 $2 \times 2$ spatial bottleneck**, replacing it with a capacity-matched **Layer-3 feature extractor (Model 2d)**.*
> 2. *We introduced a **Receiver-Side SwinIR Refiner with FiLM SNR conditioning (Model 3)** that uses shifted-window self-attention to scrub wireless noise artifacts without increasing transmitter complexity.*
> 3. *We developed a **Task-Oriented Semantic JSCC (Model 3b)** combining MSE reconstruction with classifier cross-entropy ($\mathcal{L}_{\text{total}} = \text{MSE} + \lambda \cdot \text{CE}$), preserving downstream classification accuracy even in harsh channel conditions."*

---

## 2. Theoretical Foundations (What to Write on the Blackboard)

### A. The Bandwidth Ratio ($\rho_{\text{complex}} = 1/6$, $\rho_{\text{real}} = 1/3$)
* Source input dimension: $64 \times 64 \times 3 = 12,288$ real values ($N$).
* Complex bandwidth ratio: $\rho_{\text{comp}} = \frac{k_{\text{comp}}}{N} = \frac{2,048}{12,288} = \frac{1}{6}$ (standard complex channel use convention).
* Real bandwidth ratio: $\rho_{\text{real}} = \frac{k_{\text{real}}}{N} = \frac{4,096}{12,288} = \frac{1}{3}$ (real channel dimension ratio).
* Channel symbols transmitted: $2,048$ complex symbols = **$4,096$ real channel values**.
* Every model in this project is strictly normalized to transmit **exactly 4,096 real channel uses** ($2,048$ complex uses), guaranteeing fair comparison.

### B. Power Constraint & Normalization
Before transmission, the encoded tensor $z \in \mathbb{R}^{B \times 4096}$ is normalized so average transmitter power is $P = 1$:
$$\hat{z} = \sqrt{k} \cdot \frac{z}{\|z\|_2}$$

### C. Wireless Channel Models
1. **AWGN (Additive White Gaussian Noise)**:
   $$y = \hat{z} + n, \quad n \sim \mathcal{N}\left(0, \sigma^2 I\right), \quad \sigma^2 = 10^{-\frac{\text{SNR}_{\text{dB}}}{10}}$$
2. **Rayleigh Fading Channel** (Simulating multipath reflection without line-of-sight):
   $$y = h \odot \hat{z} + n, \quad h \sim \text{Rayleigh}(\sigma=1)$$
   With perfect Channel State Information (CSI) equalization at the receiver: $\hat{y} = \frac{y}{h + \epsilon}$.

### D. FiLM Conditioned SNR Embedding
To enable a single model to operate across dynamic SNRs ($0 \text{ dB}$ to $20 \text{ dB}$):
$$\mathbf{v} = \text{MLP}(\text{SNR}_{\text{norm}}) \rightarrow [\gamma, \beta]$$
$$\text{Feature}_{\text{modulated}} = \gamma \odot \text{Feature} + \beta$$

### E. Joint Semantic Loss (Stage 3b)
$$\mathcal{L}_{\text{total}} = \|\hat{x} - x\|_2^2 + \lambda \cdot \mathcal{L}_{\text{CE}}\left(\text{Classifier}(\hat{x}), y_{\text{true}}\right)$$

---

## 3. The 4 Models Explained

```
+--------------------------------------------------------------------------------------------------+
| MODEL 1: Plain DeepJSCC (CNN Encoder -> AWGN/Rayleigh Channel -> CNN Decoder)                     |
| Baseline: 777k parameters, direct pixel-to-channel mapping                                      |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
| MODEL 2 (Original Flawed): ResNet-18 (Layer 4) -> Dense Linear 4096 -> Channel -> Linear -> Dec  |
| Flaw: Spatial collapse to 2x2 (512x2x2). Linear layer memorizes, poor generalization.           |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v  [OUR ARCHITECTURAL FIX]
+--------------------------------------------------------------------------------------------------+
| MODEL 2d (Our Fixed ResNet-JSCC): ResNet-18 (Layer 3) -> Conv2d(256->256) -> Channel -> CNN Dec  |
| Fix: 4x4 spatial preservation (256x4x4 = 4096 reals). No dense bottleneck. Unfrozen Layer 3.    |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v  [OUR RECEIVER ENHANCEMENT]
+--------------------------------------------------------------------------------------------------+
| MODEL 3: Base Model 2d (Frozen) + Receiver-Side SwinIR with FiLM SNR Conditioning                |
| Benefit: Transformer shifted-window attention eliminates channel noise at zero transmitter cost.  |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v  [OUR SEMANTIC OPTIMIZATION]
+--------------------------------------------------------------------------------------------------+
| MODEL 3b: SwinIR / CNN Refiner trained with Joint MSE + lambda * CrossEntropy Loss              |
| Benefit: Directly preserves object class features (forest, river, highway) in noisy channel.     |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. Key Questions Your Teacher Will Ask (And What to Answer)

### Q1: "Why did my original Model 2 perform worse than Plain DeepJSCC (Model 1)?"
> **Your Answer:**
> *"Because of an extreme spatial resolution bottleneck on small images. EuroSAT images are $64 \times 64$. 
> ResNet-18 was designed for $224 \times 224$ ImageNet. By Layer 4, the spatial resolution downsamples 5 times ($64 \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 4 \rightarrow 2$), resulting in just a $2 \times 2$ feature map ($512 \times 2 \times 2 = 2048$ values).
> To reach the 4096 channel capacity, the model used a dense `Linear(2048, 4096)` matrix ($8.4\text{M}$ parameters) followed by another `Linear(4096, 4096)` at the receiver. This forced the network to memorize spatial positions rather than learning convolutional spatial features, destroying fine edges and texture."*

---

### Q2: "How did you fix Model 2 in Model 2d?"
> **Your Answer:**
> *"We tapped the features from **Layer 3** instead of Layer 4.
> At Layer 3, the spatial dimensions are $4 \times 4$ with 256 channels:
> $$256 \text{ channels} \times 4 \times 4 = 4,096 \text{ values}$$
> This **matches our channel capacity $\rho=1/6$ exactly** without requiring any flattening or dense linear matrix! We used a pure `Conv2d(256, 256, kernel=1)` projection, which preserves the 2D topology and reduces parameters while unfreezing Layer 3 for fine-tuning."*

---

### Q3: "Why did you place SwinIR on the receiver side rather than the transmitter?"
> **Your Answer:**
> *"In real-world applications like satellite earth observation or drone surveillance:
> 1. **Asymmetric Compute**: The transmitter (satellite / drone) has severe battery, thermal, and weight constraints. It cannot run heavy Vision Transformers.
> 2. **Base Station Power**: The receiver is an earth station or cloud server with powerful GPUs.
> 3. **Channel Denoising**: SwinIR's shifted-window self-attention is uniquely suited to model long-range spatial correlations, acting as a learned Wiener filter that removes wireless Gaussian and fading noise."*

---

### Q4: "What does the FiLM conditioning do in Model 3?"
> **Your Answer:**
> *"Channel noise variance changes dynamically in real wireless environments. A denoiser trained for $20\text{ dB}$ will fail at $0\text{ dB}$, and vice versa. 
> Feature-wise Linear Modulation (**FiLM**) passes the instantaneous channel SNR through an MLP to predict scaling ($\gamma$) and shifting ($\beta$) vectors. This dynamically scales the intermediate feature maps inside the Residual Swin Transformer Blocks (RSTB), adapting the reconstruction strength to the exact noise level."*

---

### Q5: "In Model 3b, why would you accept a lower PSNR to get higher classification accuracy?"
> **Your Answer:**
> *"PSNR measures pixel-level Mean Squared Error, which often leads to oversmoothed images because the model averages out uncertain high-frequency textures. 
> However, many remote sensing images are transmitted for **machine consumption** (automated land-use classification, disaster monitoring) rather than human eyes.
> By adding $\lambda \cdot \mathcal{L}_{\text{CE}}$, the model prioritizes retaining the semantic boundaries (e.g., distinguishing a river from a highway or residential from industrial) even if individual pixel values have minor noise, improving automated classification by up to 15–20% at low SNRs."*

---

## 5. Summary Table of Architecture Comparison

| Model | Encoder | Spatial Bottleneck | Transmitted Channel Reals | Refiner | Loss Function | Key Strength |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1 (Fair)** | 5-layer CNN | None ($16 \times 16 \times 16$) | 4,096 | None | MSE | Fast, simple baseline |
| **Model 2 (Teacher)** | ResNet-18 (L4 frozen) | Extreme ($2 \times 2 \times 512$) | 4,096 (via Linear) | None | MSE | Shows the bottleneck trap |
| **Model 2d (Our Fix)** | ResNet-18 (L3 unfrozen) | Healthy ($4 \times 4 \times 256$) | 4,096 (pure Conv) | None | MSE | Preserves 2D spatial maps |
| **Model 3 (SwinIR)** | ResNet-18 (L3) | Healthy ($4 \times 4 \times 256$) | 4,096 | SwinIR + FiLM | MSE | Transformer noise removal |
| **Model 3b (Semantic)**| ResNet-18 (L3) | Healthy ($4 \times 4 \times 256$) | 4,096 | CNN Refiner | MSE + $\lambda\text{CE}$ | Machine-task preservation |

---

## 6. How to Conclude Your Presentation

> *"To conclude, our experiments prove that:
> 1. Proper capacity matching between feature extractors and channel bandwidth is more critical than raw backbone depth (Model 2d > Model 2).
> 2. Receiver-side Transformer refinement (SwinIR) provides significant reconstruction gains without increasing the power burden on the transmitter.
> 3. Joint semantic-source coding offers an adjustable trade-off between human perceptual quality (PSNR) and automated machine utility (Classification Accuracy)."*
