"""
backend/test_backend.py
-----------------------
Unit test suite verifying the deployment inference backend.
"""

import os
import sys
import glob
from PIL import Image

# Add root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import DeepJSCCInferenceEngine, AVAILABLE_MODELS


def run_tests():
    print("=" * 60)
    print("  DEEPJSCC DEPLOYMENT BACKEND VERIFICATION")
    print("=" * 60)

    engine = DeepJSCCInferenceEngine()
    print(f"[Device] Using compute device: {engine.device}")

    # Find a test EuroSAT image
    img_candidates = glob.glob(os.path.join(PROJECT_ROOT, "data", "**", "*.jpg"), recursive=True)
    if not img_candidates:
        img_candidates = glob.glob(os.path.join(PROJECT_ROOT, "data", "**", "*.tif"), recursive=True)

    if img_candidates:
        test_img = img_candidates[0]
        print(f"[Input] Using sample test image: {os.path.basename(test_img)}")
    else:
        # Create a synthetic 64x64 test image
        print("[Input] Creating synthetic 64x64 test image...")
        test_img = Image.new("RGB", (64, 64), color=(73, 109, 137))

    # Test existing models that have checkpoints ready
    test_models = list(AVAILABLE_MODELS.keys())

    all_passed = True
    for mkey in test_models:
        print(f"\n--- Testing: {mkey} ---")
        try:
            res = engine.transmit_and_reconstruct(
                img_input=test_img,
                model_key=mkey,
                channel="awgn",
                snr_db=10.0
            )

            print(f"  Latency       : {res['latency_ms']:.2f} ms")
            print(f"  PSNR (AWGN 10): {res['psnr_db']:.2f} dB")
            print(f"  SSIM          : {res['ssim']:.4f}")
            print(f"  Orig ResNet-18: {res['original_resnet18']['class_name']} ({res['original_resnet18']['confidence']*100:.1f}%)")
            print(f"  Recon ResNet18: {res['reconstructed_resnet18']['class_name']} ({res['reconstructed_resnet18']['confidence']*100:.1f}%)")
            print(f"  Semantic Match: {res['preserved_resnet18']}")
            print(f"  Channel Uses  : {res['channel_uses']}")
            print(f"  Status        : [PASSED]")
        except Exception as e:
            print(f"  Status        : [FAILED] -> {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL BACKEND TESTS PASSED SUCCESSFULLY!")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

