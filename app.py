"""
app.py
------
Interactive Streamlit Web Application for DeepJSCC EuroSAT Remote Sensing Transmission.

Features:
  1. Live Image Transmission & Reconstruction across 9 model architectures.
  2. Wireless channel simulation: AWGN & Rayleigh Block Fading (0 - 20 dB SNR).
  3. Dual semantic evaluation: ResNet-18 (Seen) & MobileNet-V2 (Secondary Independent Evaluator).
  4. Error heatmap and visual fidelity inspection (PSNR, SSIM, Latency).
  5. Master Research Benchmarks, SNR sweeps, and Stage 4 Ablation insights.
"""

import os
import sys
import glob
import time
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

# Configure project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import DeepJSCCInferenceEngine, AVAILABLE_MODELS, EUROSAT_CLASSES

# Set page config
st.set_page_config(
    page_title="DeepJSCC EuroSAT Transmission Suite",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sleek research UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-match {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    """Initializes and caches the DeepJSCC inference engine."""
    return DeepJSCCInferenceEngine()


@st.cache_data
def get_curated_samples():
    """Locates high-quality sample EuroSAT images across all 10 classes."""
    samples = {}
    base_dir = os.path.join(PROJECT_ROOT, "data", "eurosat", "2750")
    if not os.path.isdir(base_dir):
        base_dir = os.path.join(PROJECT_ROOT, "data")
    
    for cls in EUROSAT_CLASSES:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            matches = glob.glob(os.path.join(base_dir, f"*{cls}*"))
            if matches and os.path.isdir(matches[0]):
                cls_dir = matches[0]
        
        if os.path.isdir(cls_dir):
            cands = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.tif"))
            if cands:
                samples[cls] = sorted(cands)[0]
    return samples


def make_diff_heatmap(orig: Image.Image, recon: Image.Image) -> Image.Image:
    """Generates an amplified error heat map."""
    arr_o = np.array(orig.convert("RGB"), dtype=np.float32) / 255.0
    arr_r = np.array(recon.convert("RGB"), dtype=np.float32) / 255.0
    diff = np.abs(arr_o - arr_r).mean(axis=-1)
    diff_amp = np.clip(diff * 4.0, 0.0, 1.0) # amplify by 4x
    colored = cm.inferno(diff_amp)[:, :, :3]
    return Image.fromarray((colored * 255.0).astype(np.uint8))


# Initialize engine
engine = get_engine()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
    st.markdown("### **Transmission Parameters**")
    
    # 1. Model Selector
    model_options = list(AVAILABLE_MODELS.keys())
    default_idx = 4 # Model 3b Pareto
    selected_model_key = st.selectbox(
        "🧠 JSCC Architecture",
        model_options,
        index=default_idx,
        help="Select among the 9 benchmarked deep joint source-channel coding architectures."
    )
    
    # Model description
    info = AVAILABLE_MODELS[selected_model_key]
    st.caption(f"ℹ️ *{info['description']}*")
    
    st.markdown("---")
    
    # 2. Wireless Channel
    channel_choice = st.radio(
        "📶 Wireless Channel Model",
        ["AWGN (Additive White Gaussian Noise)", "Rayleigh (Fast Fading + MMSE Equalization)"],
        index=0,
        help="AWGN adds thermal noise. Rayleigh models multipath fading with coherent MMSE equalization."
    )
    channel_type = "awgn" if "AWGN" in channel_choice else "rayleigh"
    
    # 3. Channel SNR Slider
    snr_db = st.slider(
        "⚡ Signal-to-Noise Ratio (SNR dB)",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        help="Lower SNR = harsher wireless interference; Higher SNR = clearer channel."
    )
    
    st.markdown("---")
    st.markdown("### **Bandwidth Constraints**")
    dev_str = str(engine.device).upper()
    st.markdown(f"""
    - **Source Real Values**: $12,288$ ($64 \\times 64 \\times 3$)
    - **Channel Uses**: $k_{{\\text{{real}}}} = 4,096$
    - **Bandwidth Ratio**: $\\rho_{{\\text{{real}}}} = 1/3$ ($\\rho_{{\\text{{comp}}}} = 1/6$)
    - **Power Constraint**: $\\mathbb{{E}}[|x|^2] = 1.0$
    - **Compute Device**: `{dev_str}`
    """)

# --- MAIN APP HEADER ---
st.markdown("<div class='main-title'>📡 DeepJSCC: Semantic Wireless Image Transmission</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Interactive Demonstration & Scientific Benchmark on EuroSAT Remote Sensing Imagery</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Transmission",
    "🧠 Dual Semantic Intelligence",
    "📊 Research Benchmarks",
    "🔬 Ablation & Theory"
])

# ==============================================================================
# TAB 1: LIVE TRANSMISSION & RECONSTRUCTION
# ==============================================================================
with tab1:
    col_input, col_output = st.columns([1, 1.25], gap="large")
    
    with col_input:
        st.subheader("1. Input Image Selection")
        input_mode = st.radio(
            "Choose Input Source:",
            ["Curated EuroSAT Benchmark Samples", "Upload Custom Image"],
            horizontal=True
        )
        
        curated_samples = get_curated_samples()
        input_image = None
        sample_label = "Custom"
        
        if input_mode == "Curated EuroSAT Benchmark Samples":
            if curated_samples:
                selected_class = st.selectbox(
                    "EuroSAT Class Preset:",
                    list(curated_samples.keys()),
                    index=0
                )
                sample_path = curated_samples[selected_class]
                input_image = Image.open(sample_path).convert("RGB")
                sample_label = selected_class
            else:
                st.warning("Sample directory not found; defaulting to synthetic image.")
                input_image = Image.new("RGB", (64, 64), color=(70, 130, 180))
        else:
            uploaded_file = st.file_uploader("Upload Image (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
            else:
                st.info("Upload an image or toggle preset samples above.")
                if curated_samples:
                    input_image = Image.open(list(curated_samples.values())[0]).convert("RGB")
                    sample_label = list(curated_samples.keys())[0]

        if input_image is not None:
            st.image(input_image, caption=f"Original Input ({sample_label}) - 64x64 RGB", width=220)
            
        transmit_btn = st.button("📡 Transmit Over Simulated Channel", type="primary", width="stretch")

    with col_output:
        st.subheader("2. Wireless Channel Output & Reconstruction")
        
        if input_image is not None:
            # Execute inference
            with st.spinner("Transmitting symbols over wireless channel..."):
                result = engine.transmit_and_reconstruct(
                    img_input=input_image,
                    model_key=selected_model_key,
                    channel=channel_type,
                    snr_db=snr_db
                )
                
            recon_img = result["reconstructed_image"]
            orig_img = result["original_image"]
            diff_img = make_diff_heatmap(orig_img, recon_img)
            
            # Display reconstructed vs original vs error map
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(orig_img, caption="Original Input", width="stretch")
            with c2:
                st.image(recon_img, caption="Reconstructed Output", width="stretch")
            with c3:
                st.image(diff_img, caption="4x Error Heatmap", width="stretch")
                
            # Key Fidelity Metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("PSNR (Peak SNR)", f"{result['psnr_db']:.2f} dB")
            with m2:
                st.metric("SSIM (Structural)", f"{result['ssim']:.4f}")
            with m3:
                st.metric("Latency", f"{result['latency_ms']:.1f} ms")
            with m4:
                st.metric("Channel Uses", f"{result['channel_uses']} symbols")

# ==============================================================================
# TAB 2: DUAL SEMANTIC INTELLIGENCE
# ==============================================================================
with tab2:
    st.subheader("Dual Classifier Semantic Preservation Analysis")
    st.markdown("""
    To eliminate evaluator bias, downstream classification is evaluated simultaneously on **two separate pre-trained backbones**:
    - **Primary Classifier (ResNet-18)**: Trained on clean EuroSAT (95.85% test acc). *Seen during semantic-refinement training.*
    - **Secondary Classifier (MobileNet-V2)**: Independently trained on clean EuroSAT (96.93% test acc). *Secondary independent classifier evaluator, not used to train the semantic refiner.*
    """)
    
    if input_image is not None and 'result' in locals():
        r18_orig = result["original_resnet18"]
        r18_recon = result["reconstructed_resnet18"]
        mb_orig = result["original_mobilenet"]
        mb_recon = result["reconstructed_mobilenet"]
        
        col_r18, col_mb = st.columns(2, gap="large")
        
        with col_r18:
            st.markdown("#### 🔬 Primary Evaluator: ResNet-18")
            match_r18 = result["preserved_resnet18"]
            badge_r18 = "<span class='badge-match'>✅ SEMANTIC PRESERVED</span>" if match_r18 else "<span class='badge-fail'>⚠️ SEMANTIC CORRUPTED</span>"
            st.markdown(badge_r18, unsafe_allow_html=True)
            
            st.write(f"**Original Class**: `{r18_orig['class_name']}` ({r18_orig['confidence']*100:.1f}%)")
            st.write(f"**Reconstructed Class**: `{r18_recon['class_name']}` ({r18_recon['confidence']*100:.1f}%)")
            
            # Bar chart of top 3 predictions
            if "top3" in r18_recon:
                top_df = pd.DataFrame(r18_recon["top3"], columns=["Class", "Confidence"]).set_index("Class")
                st.bar_chart(top_df)
                
        with col_mb:
            st.markdown("#### 📱 Secondary Independent Evaluator: MobileNet-V2")
            match_mb = result["preserved_mobilenet"]
            badge_mb = "<span class='badge-match'>✅ SEMANTIC PRESERVED</span>" if match_mb else "<span class='badge-fail'>⚠️ SEMANTIC CORRUPTED</span>"
            st.markdown(badge_mb, unsafe_allow_html=True)
            
            st.write(f"**Original Class**: `{mb_orig['class_name']}` ({mb_orig['confidence']*100:.1f}%)")
            st.write(f"**Reconstructed Class**: `{mb_recon['class_name']}` ({mb_recon['confidence']*100:.1f}%)")
            
            if "top3" in mb_recon:
                top_df2 = pd.DataFrame(mb_recon["top3"], columns=["Class", "Confidence"]).set_index("Class")
                st.bar_chart(top_df2)

# ==============================================================================
# TAB 3: RESEARCH BENCHMARKS & SNR SWEEPS
# ==============================================================================
with tab3:
    st.subheader("Master Benchmark Summary Table (SNR = 10 dB, N = 2,700 Test Images)")
    
    # Load master CSV
    master_csv_path = os.path.join(PROJECT_ROOT, "results", "final", "master_summary_10dB.csv")
    if os.path.isfile(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
        
        chan_view = st.radio("Channel Condition:", ["AWGN Channel @ 10 dB", "Rayleigh Channel @ 10 dB"], horizontal=True)
        
        param_map = {
            "Model 1 Fair (PlainDeepJSCC)": "0.78M",
            "Model 2d (ResNetJSCC 30ep)": "4.35M",
            "Model 2d Long (ResNetJSCC 60ep)": "4.35M",
            "Model 3 (SwinIR Refiner)": "5.02M",
            "Model 3b (Semantic lam=0.0003 S42)": "5.02M",
            "Model 3b (Semantic lam=0.0003 S43)": "5.02M",
            "Model 3b (Semantic lam=0.0003 S44)": "5.02M",
            "Model 3b (Semantic lam=0.001)": "5.02M",
            "Model 3b (Semantic lam=0.003)": "5.02M",
            "Model 4 (Full SwinJSCC)": "2.66M",
            "Ablation C (SwinJSCC-NoShift)": "2.66M",
            "Ablation A (ConvJSCC Baseline)": "2.59M"
        }
        
        if "AWGN" in chan_view:
            display_cols = ["Model", "AWGN_PSNR", "AWGN_SSIM", "AWGN_ResNet18", "AWGN_MobileNet"]
            df_view = master_df[display_cols].copy()
            df_view.insert(1, "Total Params", df_view["Model"].map(param_map).fillna("N/A"))
            df_view.columns = ["Model Architecture", "Total Params", "PSNR (dB)", "SSIM", "ResNet-18 (%)", "MobileNet-V2 (%)"]
        else:
            display_cols = ["Model", "Rayleigh_PSNR", "Rayleigh_SSIM", "Rayleigh_ResNet18", "Rayleigh_MobileNet"]
            df_view = master_df[display_cols].copy()
            df_view.insert(1, "Total Params", df_view["Model"].map(param_map).fillna("N/A"))
            df_view.columns = ["Model Architecture", "Total Params", "PSNR (dB)", "SSIM", "ResNet-18 (%)", "MobileNet-V2 (%)"]
            
        st.dataframe(df_view.style.highlight_max(subset=["PSNR (dB)", "SSIM", "ResNet-18 (%)", "MobileNet-V2 (%)"], color="#D1FAE5"), width="stretch")
    
    st.markdown("---")
    st.subheader("Publication-Quality SNR Sweep Plots (0 dB to 20 dB)")
    
    sweep_type = st.selectbox(
        "Select Performance Metric:",
        ["Reconstruction PSNR vs. SNR", "Structural Similarity (SSIM) vs. SNR", "ResNet-18 Semantic Accuracy vs. SNR", "MobileNet-V2 Secondary Classifier Accuracy vs. SNR"]
    )
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("**AWGN Channel**")
        if "PSNR" in sweep_type:
            p_img = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_psnr_vs_snr_awgn.png")
        elif "SSIM" in sweep_type:
            p_img = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_ssim_vs_snr_awgn.png")
        elif "ResNet" in sweep_type:
            p_img = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_acc_resnet18_awgn.png")
        else:
            p_img = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_acc_mobilenet_awgn.png")
        if os.path.isfile(p_img):
            st.image(p_img, width="stretch")
            
    with c_p2:
        st.markdown("**Rayleigh Fading Channel**")
        if "PSNR" in sweep_type:
            p_img2 = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_psnr_vs_snr_rayleigh.png")
        elif "SSIM" in sweep_type:
            p_img2 = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_ssim_vs_snr_rayleigh.png")
        elif "ResNet" in sweep_type:
            p_img2 = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_acc_resnet18_rayleigh.png")
        else:
            p_img2 = os.path.join(PROJECT_ROOT, "results", "final", "plots", "final_acc_mobilenet_rayleigh.png")
        if os.path.isfile(p_img2):
            st.image(p_img2, width="stretch")

# ==============================================================================
# TAB 4: ABLATION & SCIENTIFIC INSIGHTS
# ==============================================================================
with tab4:
    st.subheader("Stage 4 Vision Transformer Ablation Audit")
    
    st.markdown(r"""
    ### Why did Full SwinJSCC experience posterior collapse on EuroSAT?
    1. **Empirical Fact**: In EuroSAT test set ($N=2,700$), exactly 275 images belong to class 9 (`SeaLake`, $275/2700 = 10.1852\% \approx 10.19\%$). Both classifiers predicted `SeaLake` on 100% of samples because the decoder output collapsed to the marginal dataset color field ($16.13\text{ dB}$).
    2. **Shifted Window Root Cause**: Applying cyclic shifts ($S=2$) on tiny feature grids ($4 \times 4$) with attention masking isolates tokens during random initialization under channel noise.
    3. **Ablation C (NoShift)**: Removing shifted windows restored convergence immediately, boosting PSNR to **$27.06\text{ dB}$** ($+10.93\text{ dB}$) and ResNet accuracy to **$34.81\%$**.
    4. **Convolutional Superiority**: Parameter-matched `ConvJSCC` achieved **$27.66\text{ dB}$** and **$39.33\%$** accuracy, demonstrating that standard 2D convolutions provide crucial inductive biases for small-resolution JSCC.
    """)
    
    ablation_plot = os.path.join(PROJECT_ROOT, "results", "final", "plots", "ablation_stage4_comparison.png")
    if os.path.isfile(ablation_plot):
        st.image(ablation_plot, caption="Stage 4 JSCC Architecture Comparison (AWGN 10 dB)", width=700)
        
    st.markdown("---")
    st.subheader("Semantic Guidance Trade-off Curve (λ Sweep)")
    pareto_plot = os.path.join(PROJECT_ROOT, "results", "final", "semantic_analysis", "pareto_tradeoff.png")
    if os.path.isfile(pareto_plot):
        st.image(pareto_plot, caption="Fidelity vs. Semantic Accuracy Pareto Tradeoff", width=700)
        
    st.markdown(r"""
    - **$\lambda = 0.0003$ (Preferred Trade-off under Evaluated Conditions)**: Yields $+49.4\text{ pp}$ classification gain ($74.11\%$) with only $0.13\text{ dB}$ PSNR drop, and maintains solid performance on the secondary independent MobileNet-V2 classifier ($47.11\%$).
    - **$\lambda = 0.003$ (Classifier Over-Optimization)**: Further pushes seen ResNet-18 accuracy ($75.96\%$), but degrades secondary MobileNet-V2 performance down to $39.96\%$ and reduces PSNR to $25.08\text{ dB}$.
    """)
