# Adaptive Semantic JSCC – Stage 1 Colab Notebook
# =================================================
# Run this notebook on Google Colab (Runtime → Change runtime type → T4 GPU)
#
# What this notebook does
# -----------------------
# 1. Clone the GitHub repo (update REPO_URL below)
# 2. Install requirements
# 3. Run channel unit tests
# 4. Train Model 1 (Plain DeepJSCC) for 80 epochs
# 5. Print PSNR at SNR = 0, 5, 10, 15, 20 dB (AWGN and Rayleigh)
#
# Stage gate: PSNR MUST rise as SNR rises. If not, debug before Stage 2.

# ─── CELL 1: Mount Drive & set repo URL ────────────────────────────────────
REPO_URL   = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # <--- UPDATE THIS
WORK_DIR   = "/content/data_transmission"

# ─── CELL 2: Clone + Install ────────────────────────────────────────────────
import subprocess, os

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout

print("Cloning repo …")
if not os.path.exists(WORK_DIR):
    run(f"git clone {REPO_URL} {WORK_DIR}")
else:
    run(f"cd {WORK_DIR} && git pull")

os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")

print("Installing requirements …")
run("pip install -q -r requirements.txt")
print("✓ Requirements installed")

# ─── CELL 3: Verify GPU ─────────────────────────────────────────────────────
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── CELL 4: Run channel unit tests ─────────────────────────────────────────
print("\n" + "="*60)
print("Running channel unit tests …")
run("python channels.py")
print("✓ Channel tests passed")

# ─── CELL 5: Sanity-check data loading ──────────────────────────────────────
print("\n" + "="*60)
print("Checking data loader …")
run("python data.py")
print("✓ Data loading OK")

# ─── CELL 6: Smoke-test Model 1 architecture ────────────────────────────────
print("\n" + "="*60)
print("Smoke-testing Model 1 architecture …")
run("python models/deepjscc.py")
print("✓ Model 1 architecture OK")

# ─── CELL 7: Train Model 1 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("Training Model 1 (Plain DeepJSCC) …")
print("This will take ~30-60 min on a T4 GPU.")
run("python train.py --config configs/model1.yaml")
print("✓ Training complete")

# ─── CELL 8: Print PSNR at each SNR (Stage gate check) ─────────────────────
print("\n" + "="*60)
print("Evaluating Model 1 on test set …")

import sys
sys.path.insert(0, WORK_DIR)

import torch
from data     import get_loaders
from channels import awgn, rayleigh, power_normalize
from train    import load_config, mse_to_psnr, _needs_imgnet, build_model
import torch.nn as nn

cfg    = load_config("configs/model1.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_loaders(
    root="./data", batch_size=64, num_workers=0, download=True
)

model = build_model(cfg, device)
ckpt  = torch.load("checkpoints/model1/best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

snrs     = [0, 5, 10, 15, 20]
channels = ["awgn", "rayleigh"]

print("\n{'Model 1 — Plain DeepJSCC':^50}")
print(f"{'Channel':<12} {'SNR (dB)':>8} {'PSNR (dB)':>12}")
print("-" * 35)

criterion = nn.MSELoss()
results_table = {}

with torch.no_grad():
    for ch in channels:
        psnrs = []
        for snr_db in snrs:
            total_mse, n = 0.0, 0
            for img_imgnet, img_unit, _ in test_loader:
                img_unit = img_unit.to(device)
                recon, _ = model(img_unit, channel=ch, snr_db=float(snr_db))
                total_mse += criterion(recon, img_unit).item()
                n += 1
            psnr = mse_to_psnr(total_mse / n)
            psnrs.append(psnr)
            print(f"{ch.upper():<12} {snr_db:>8} {psnr:>12.2f}")
        results_table[ch] = psnrs
        print()

# ─── CELL 9: Stage gate – verify PSNR monotonically increases ───────────────
print("="*50)
print("STAGE GATE CHECK: PSNR must rise with SNR")
print("="*50)
gate_passed = True
for ch in channels:
    psnrs = results_table[ch]
    monotone = all(psnrs[i] <= psnrs[i+1] for i in range(len(psnrs)-1))
    status = "✓ PASS" if monotone else "✗ FAIL – debug before Stage 2!"
    print(f"  {ch.upper():8s}: {status}")
    if not monotone:
        gate_passed = False
        print(f"    PSNR values: {[f'{p:.2f}' for p in psnrs]}")

if gate_passed:
    print("\n🎉 Stage 1 COMPLETE. PSNR rises with SNR on both channels.")
    print("Proceed to Stage 2: Model 2 (ResNet-18 + DeepJSCC).")
else:
    print("\n⚠️  Stage gate FAILED. Do NOT proceed to Stage 2.")
    print("Common causes:")
    print("  1. Power normalisation bug (check channels.py test_power_normalize_unit_power)")
    print("  2. Model trained too few epochs (increase epochs in config)")
    print("  3. LR too high/low (try lr=5e-4)")
