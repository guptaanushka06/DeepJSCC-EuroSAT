"""
run_all.py
----------
Complete pipeline orchestration for the DeepJSCC project.
Trains all models, evaluates them, and produces final comparison tables and plots.

Usage
-----
  python run_all.py                # Run EVERYTHING in order
  python run_all.py --stage 1      # Stage 1: Model 1 (fair baseline)
  python run_all.py --stage 2      # Stage 2: Model 2d (best base)
  python run_all.py --stage clf    # Classifiers only
  python run_all.py --stage 3      # Stage 3: SwinIR/CNN refiners
  python run_all.py --stage 3b     # Stage 3b: Semantic loss
  python run_all.py --stage 4      # Stage 4: SwinJSCC
  python run_all.py --stage eval   # Evaluate all + compare
  python run_all.py --dry_run      # Print what would be run, don't execute
"""

import argparse
import os
import subprocess
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_cmd(cmd: list, description: str, dry_run: bool = False):
    """Run a command with a clear header/footer."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)

    if dry_run:
        print("  [DRY RUN] Skipped.")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"\n  [X] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def run_stage(stage: str, dry_run: bool = False):
    """Run a specific stage or all stages."""
    py = sys.executable  # current Python interpreter

    stages_order = ["tests", "clf", "1", "2", "3", "3b", "4", "eval"]
    if stage == "all":
        stages_to_run = stages_order
    elif "," in stage:
        stages_to_run = [s.strip() for s in stage.split(",") if s.strip()]
    else:
        stages_to_run = [stage]

    for s in stages_to_run:
        if s == "tests":
            # Channel unit tests
            ok = run_cmd(
                [py, "channels.py"],
                "Running channel unit tests",
                dry_run
            )
            if not ok and stage != "all":
                print("Channel tests failed. Fix before proceeding.")
                return

        elif s == "clf":
            # Train primary classifier (ResNet-18)
            run_cmd(
                [py, "classifier.py", "--train"],
                "Training Primary Classifier (ResNet-18, seed=42)",
                dry_run
            )
            # Train secondary classifier (MobileNet-V2)
            run_cmd(
                [py, "classifier.py", "--train_second"],
                "Training Secondary Classifier (MobileNet-V2, seed=1337)",
                dry_run
            )

        elif s == "1":
            # Model 1 fair baseline
            run_cmd(
                [py, "train.py", "--config", "configs/model1_fair.yaml"],
                "Stage 1: Training Model 1 Fair (PlainDeepJSCC, 30 epochs + aug)",
                dry_run
            )

        elif s == "2":
            # Model 2d (best base model)
            run_cmd(
                [py, "train.py", "--config", "configs/model2d.yaml"],
                "Stage 2: Training Model 2d (Layer-3 Unfrozen)",
                dry_run
            )

        elif s == "3":
            # Stage 3: CNN control refiner
            run_cmd(
                [py, "train.py", "--config", "configs/model3_cnn.yaml"],
                "Stage 3: Training CNN Refiner (control)",
                dry_run
            )
            # Stage 3: SwinIR + FiLM
            run_cmd(
                [py, "train.py", "--config", "configs/model3_swinir.yaml"],
                "Stage 3: Training SwinIR + FiLM Refiner",
                dry_run
            )
            # Stage 3: SwinIR no-SNR ablation
            run_cmd(
                [py, "train.py", "--config", "configs/model3_nosnr.yaml"],
                "Stage 3: Training SwinIR No-SNR Ablation",
                dry_run
            )

        elif s == "3b":
            # Stage 3b: Semantic loss sweep
            for lam in ["0.001", "0.003", "0.01"]:
                run_cmd(
                    [py, "train.py", "--config", f"configs/model3b_lam{lam}.yaml"],
                    f"Stage 3b: Training Semantic Refiner (lam={lam})",
                    dry_run
                )

        elif s == "4":
            # Stage 4: SwinJSCC -- first do a smoke test
            run_cmd(
                [py, "train.py", "--config", "configs/model4.yaml", "--smoke_test", "--smoke_batches", "10"],
                "Stage 4: SwinJSCC Smoke Test (10 batches)",
                dry_run
            )
            # Full training
            run_cmd(
                [py, "train.py", "--config", "configs/model4.yaml"],
                "Stage 4: Training SwinJSCC (30 epochs)",
                dry_run
            )

        elif s == "core":
            # Core milestone sequence: Model 2d -> Model 3 (SwinIR) -> Model 3b (lam=0.001) -> Eval
            run_cmd(
                [py, "train.py", "--config", "configs/model2d.yaml"],
                "Core Stage 2: Training Model 2d (Layer-3 Unfrozen ResNet-JSCC)",
                dry_run
            )
            run_cmd(
                [py, "train.py", "--config", "configs/model3_swinir.yaml"],
                "Core Stage 3: Training SwinIR + FiLM Receiver Refiner",
                dry_run
            )
            run_cmd(
                [py, "train.py", "--config", "configs/model3b_lam0.001.yaml"],
                "Core Stage 3b: Training Semantic Refiner (lam=0.001)",
                dry_run
            )
            run_cmd(
                [py, "eval.py", "--all"],
                "Core Stage 5: Evaluating All Models (PSNR, SSIM, Accuracy + Samples)",
                dry_run
            )
            run_cmd(
                [py, "compare_results.py", "--count_params"],
                "Core Stage 5: Aggregating Results (Tables + Plots)",
                dry_run
            )

        elif s == "eval":
            # Evaluate all trained models
            run_cmd(
                [py, "eval.py", "--all"],
                "Evaluating ALL Trained Models (PSNR, SSIM, Accuracy + Samples)",
                dry_run
            )
            # Aggregate results
            run_cmd(
                [py, "compare_results.py", "--count_params"],
                "Aggregating Results: Summary Tables + Comparison Plots",
                dry_run
            )

        elif s == "core_finish":
            run_cmd(
                [py, "train.py", "--config", "configs/model3b_lam0.001.yaml"],
                "Core Stage 3b: Training Semantic Refiner (lam=0.001)",
                dry_run
            )
            run_cmd(
                [py, "eval.py", "--all"],
                "Core Stage 5: Evaluating All Models (PSNR, SSIM, Accuracy + Samples)",
                dry_run
            )
            run_cmd(
                [py, "compare_results.py", "--count_params"],
                "Core Stage 5: Aggregating Results (Tables + Plots)",
                dry_run
            )

        else:
            print(f"Unknown stage: {s!r}")
            print("Valid stages: tests, clf, 1, 2, 3, 3b, 4, eval, core, core_finish, all")
            return

    print("\n" + "=" * 70)
    print("  [DONE] PIPELINE COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DeepJSCC training and evaluation pipeline."
    )
    parser.add_argument(
        "--stage", default="all",
        help="Which stage to run: tests, clf, 1, 2, 3, 3b, 4, eval, or 'all' (default)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them."
    )
    args = parser.parse_args()
    run_stage(args.stage, args.dry_run)


if __name__ == "__main__":
    main()
