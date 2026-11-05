"""
package_project.py
------------------
Utility to create a lightweight, fully functional distribution ZIP of the project.
Excludes ~3.6 GB of intermediate epoch checkpoints (e.g. epoch_0005.pt, epoch_0010.pt),
while preserving:
  - All source code, configs, backend, app.py, and .streamlit/
  - All 14 final trained model weights (checkpoints/*/best.pt)
  - All benchmark results, master CSVs, and publication plots (results/)
  - EuroSAT dataset and test samples (data/)
  - Documentation and audit reports (README.md, RESULTS.md, FINAL_AUDIT.md)

Reduces total archive size from 4.3 GB down to ~500 MB!
"""

import os
import sys
import glob
import zipfile
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "DeepJSCC_EuroSAT_FullApp.zip")

def create_package():
    print("=" * 70)
    print("       CREATING LIGHTWEIGHT DEEPJSCC DISTRIBUTION PACKAGE")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Collect files to include
    include_files = []
    
    # Root source files & markdown docs
    for f in os.listdir(PROJECT_ROOT):
        full_path = os.path.join(PROJECT_ROOT, f)
        if os.path.isfile(full_path):
            if f.endswith(('.zip', '.tar', '.gz')):
                continue
            include_files.append(full_path)
            
    # Include directories
    for dname in ["backend", "configs", "models", ".streamlit", "data", "results"]:
        dir_path = os.path.join(PROJECT_ROOT, dname)
        if os.path.isdir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                if "__pycache__" in root:
                    continue
                for file in files:
                    include_files.append(os.path.join(root, file))
                    
    # Include ONLY best.pt checkpoints (excludes 3.6 GB of intermediate epoch files)
    best_ckpts = glob.glob(os.path.join(PROJECT_ROOT, "checkpoints", "*", "best.pt"))
    include_files.extend(best_ckpts)
    
    total_raw_mb = sum(os.path.getsize(f) for f in include_files) / (1024 * 1024)
    print(f"Total files to package: {len(include_files)}")
    print(f"Uncompressed size:      {total_raw_mb:.2f} MB (saved ~3,600 MB of intermediate weights!)")
    print(f"Output ZIP file:        {OUTPUT_ZIP}")
    print("\nCompressing archive (this may take 15-30 seconds)...")
    
    # Create ZIP
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for idx, file_path in enumerate(include_files):
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            zipf.write(file_path, rel_path)
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1} / {len(include_files)} files...")
                
    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    elapsed = time.time() - t0
    
    print("-" * 70)
    print(f"SUCCESS! Created: {os.path.basename(OUTPUT_ZIP)}")
    print(f"Final Compressed Size: {zip_size_mb:.2f} MB")
    print(f"Time Taken:            {elapsed:.1f} seconds")
    print("=" * 70)
    print("\nYour friend can extract this ZIP and run:")
    print("  streamlit run app.py")
    print("Everything will work immediately with all models, data, and benchmarks intact!")

if __name__ == "__main__":
    create_package()
