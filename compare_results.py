"""
compare_results.py
------------------
Aggregate per-model evaluation results into a unified comparison.

Usage
-----
  python compare_results.py                        # scan results/*/test_metrics.csv
  python compare_results.py --results_dir results  # explicit directory
  python compare_results.py --count_params         # also print parameter counts (requires torch)

Output  (all written to results/final/ – never overwrites per-model files)
------
  results/final/summary_10dB.csv          – combined table at SNR = 10 dB
  results/final/psnr_vs_snr_awgn.png      – PSNR vs SNR (AWGN channel)
  results/final/psnr_vs_snr_rayleigh.png  – PSNR vs SNR (Rayleigh channel)
  results/final/ssim_vs_snr_awgn.png      – SSIM  vs SNR (AWGN)
  results/final/ssim_vs_snr_rayleigh.png  – SSIM  vs SNR (Rayleigh)
  results/final/acc_vs_snr_awgn.png       – Accuracy vs SNR (AWGN)
  results/final/acc_vs_snr_rayleigh.png   – Accuracy vs SNR (Rayleigh)
  results/final/param_counts.csv          – trainable / frozen / total params (--count_params only)

NOTE: This script does NOT require torch/torchvision unless --count_params is used.
"""

import argparse
import csv
import glob
import math
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    print("[WARNING] matplotlib not installed -- plots will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Discover and load all test_metrics.csv files
# ─────────────────────────────────────────────────────────────────────────────
def discover_csvs(results_dir: str) -> list:
    """Find all results/<model>/test_metrics.csv files."""
    pattern = os.path.join(results_dir, "*", "test_metrics.csv")
    paths = sorted(glob.glob(pattern))
    return paths


def load_all_csvs(csv_paths: list) -> list:
    """Load all CSVs and return a list of row dicts. Skips missing/corrupt files."""
    rows = []
    for path in csv_paths:
        folder = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    row = {
                        "model_name": r.get("model", folder),
                        "channel":    r.get("channel", "awgn"),
                        "snr_db":     float(r.get("snr_db", r.get("snr", 10))),
                        "psnr":          _safe_float(r.get("psnr")),
                        "ssim":          _safe_float(r.get("ssim")),
                        "accuracy":      _safe_float(r.get("accuracy", r.get("acc"))),
                        "accuracy_clf2": _safe_float(r.get("accuracy_clf2", r.get("acc_clf2"))),
                        "n_samples":     int(r.get("n_samples", 2700)),
                        "source_dir":    folder,
                    }
                    rows.append(row)
        except Exception as e:
            print(f"  [WARNING] Could not read {path}: {e}")
    return rows


def _safe_float(val):
    if val is None or str(val).strip().upper() in ("", "N/A", "NONE", "NAN"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Binomial 95 % confidence interval for accuracy
# ─────────────────────────────────────────────────────────────────────────────
def binomial_ci_95(acc: float, n: int) -> tuple:
    """
    Wilson score interval for proportion p = acc, sample size n.
    Returns (lower, upper) as fractions.
    """
    if acc is None or n == 0:
        return (None, None)
    z = 1.96  # 95 % CI
    p_hat = acc
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parameter counting (only loaded when --count_params is used)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list: (display_name, config_path)
ALL_MODELS = [
    ("Model 1 (PlainDeepJSCC)",      "configs/model1.yaml"),
    ("Model 1b (Plain + aug)",       "configs/model1b.yaml"),
    ("Model 1 Fair (30ep + aug)",    "configs/model1_fair.yaml"),
    ("Model 2 (ResNetJSCC)",         "configs/model2.yaml"),
    ("Model 2b (Reg Layer-4)",       "configs/model2b.yaml"),
    ("Model 2c (Layer-3 Conv)",      "configs/model2c.yaml"),
    ("Model 2d (Layer-3 Unfrozen)",  "configs/model2d.yaml"),
    ("Model 2e (128px Upsample)",    "configs/model2e.yaml"),
    ("Stage 3: SwinIR+FiLM",        "configs/model3_swinir.yaml"),
    ("Stage 3: SwinIR+FiLM+RSTB",   "configs/model3_swinir_rstb.yaml"),
    ("Stage 3: SwinIR no-SNR",      "configs/model3_nosnr.yaml"),
    ("Stage 3: CNN control",        "configs/model3_cnn.yaml"),
    ("Stage 3b: CNN+CE (lam=0.001)",  "configs/model3b_lam0.001.yaml"),
    ("Stage 3b: CNN+CE (lam=0.003)",  "configs/model3b_lam0.003.yaml"),
    ("Stage 3b: CNN+CE (lam=0.01)",   "configs/model3b_lam0.01.yaml"),
    ("Stage 4: SwinJSCC",           "configs/model4.yaml"),
]


def count_params(cfg_path: str) -> dict:
    """
    Build the model from config, count trainable/frozen/total params.
    Requires torch – only called when --count_params is set.
    """
    import torch
    from train import load_config, build_model

    cfg = load_config(cfg_path)
    device = torch.device("cpu")

    try:
        model = build_model(cfg, device)
    except Exception as e:
        print(f"  [SKIP] Could not build model from {cfg_path}: {e}")
        return {"trainable": None, "frozen": None, "total": None}

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen

    return {"trainable": trainable, "frozen": frozen, "total": total}


def run_param_counts(out_dir: str):
    """Count parameters for every model variant and save to CSV + print table."""
    print("\n" + "=" * 70)
    print("  PARAMETER COUNTS")
    print("=" * 70)
    header = f"{'Model':<32s} {'Trainable':>12s} {'Frozen':>12s} {'Total':>12s}"
    print(header)
    print("-" * len(header))

    rows = []
    for display_name, cfg_path in ALL_MODELS:
        if not os.path.isfile(cfg_path):
            print(f"  {display_name:<32s}  [config not found: {cfg_path}]")
            continue
        counts = count_params(cfg_path)
        t = counts["trainable"]
        f = counts["frozen"]
        a = counts["total"]
        t_str = f"{t:>12,}" if t is not None else f"{'N/A':>12s}"
        f_str = f"{f:>12,}" if f is not None else f"{'N/A':>12s}"
        a_str = f"{a:>12,}" if a is not None else f"{'N/A':>12s}"
        print(f"  {display_name:<32s} {t_str} {f_str} {a_str}")
        rows.append({
            "model": display_name,
            "trainable": t,
            "frozen": f,
            "total": a,
        })

    # Save CSV
    csv_path = os.path.join(out_dir, "param_counts.csv")
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["model", "trainable", "frozen", "total"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Summary table at 10 dB
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_table(rows: list, snr_db: float = 10.0) -> list:
    """Filter rows to a given SNR and build a summary table."""
    summary = []
    for r in rows:
        if abs(r["snr_db"] - snr_db) < 0.5:
            ci_lo, ci_hi = binomial_ci_95(r["accuracy"], r["n_samples"])
            summary.append({
                "model":         r["model_name"],
                "channel":       r["channel"],
                "snr_db":        r["snr_db"],
                "psnr":          r["psnr"],
                "ssim":          r["ssim"],
                "accuracy":      r["accuracy"],
                "accuracy_clf2": r.get("accuracy_clf2"),
                "acc_ci_lo":     ci_lo,
                "acc_ci_hi":     ci_hi,
                "n_samples":     r["n_samples"],
                "source_dir":    r.get("source_dir", ""),
            })
    return summary


def print_summary_table(summary: list, out_dir: str, snr_db: float = 10.0):
    """Print and save the 10 dB summary table."""
    if not summary:
        print(f"\n[WARNING] No data at SNR = {snr_db} dB.")
        return

    # Group by channel for cleaner display
    channels = sorted(set(r["channel"] for r in summary))

    print(f"\n{'=' * 95}")
    print(f"  SUMMARY TABLE  --  SNR = {snr_db} dB")
    print(f"{'=' * 95}")

    has_clf2 = any(r.get("accuracy_clf2") is not None for r in summary)

    for ch in channels:
        ch_rows = [r for r in summary if r["channel"] == ch]
        ch_rows.sort(key=lambda r: -(r["psnr"] or 0))

        print(f"\n  Channel: {ch.upper()}")
        if has_clf2:
            hdr = (f"  {'Model':<28s} {'PSNR (dB)':>10s} {'SSIM':>8s} "
                   f"{'Acc(Res18)':>11s} {'Acc(MobNet)':>12s} {'95% CI':>18s}")
        else:
            hdr = (f"  {'Model':<28s} {'PSNR (dB)':>10s} {'SSIM':>8s} "
                   f"{'Accuracy':>10s} {'95% CI':>18s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for r in ch_rows:
            p = f"{r['psnr']:.4f}" if r["psnr"] is not None else "N/A"
            s = f"{r['ssim']:.4f}" if r["ssim"] is not None else "N/A"
            a = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
            a2 = f"{r['accuracy_clf2']*100:.2f}%" if r.get("accuracy_clf2") is not None else "N/A"
            if r["acc_ci_lo"] is not None:
                ci = f"[{r['acc_ci_lo']*100:.2f}, {r['acc_ci_hi']*100:.2f}]%"
            else:
                ci = "N/A"
            if has_clf2:
                print(f"  {r['model']:<28s} {p:>10s} {s:>8s} {a:>11s} {a2:>12s} {ci:>18s}")
            else:
                print(f"  {r['model']:<28s} {p:>10s} {s:>8s} {a:>10s} {ci:>18s}")

    # Save CSV
    csv_path = os.path.join(out_dir, f"summary_{int(snr_db)}dB.csv")
    fieldnames = ["model", "channel", "snr_db", "psnr", "ssim",
                  "accuracy", "accuracy_clf2", "acc_ci_lo", "acc_ci_hi", "n_samples", "source_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n  Saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────────────────────────────────────

# Curated colour palette & markers for up to 12 models
PALETTE = [
    "#e74c3c",  # red
    "#2980b9",  # blue
    "#27ae60",  # green
    "#f39c12",  # orange
    "#8e44ad",  # purple
    "#16a085",  # teal
    "#d35400",  # burnt orange
    "#2c3e50",  # dark navy
    "#c0392b",  # dark red
    "#7f8c8d",  # grey
    "#1abc9c",  # mint
    "#e67e22",  # tangerine
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

# Human-readable display names: map internal model names → short labels
DISPLAY_NAMES = {
    "PlainDeepJSCC":    "Model 1 (Plain)",
    "ResNetJSCC":       "Model 2 (ResNet L4)",
    "ResNetJSCC_2b":    "Model 2b (Reg L4)",
    "ResNetJSCC_2c":    "Model 2c (L3 Conv)",
    "ResNetJSCC_2d":    "Model 2d (L3 Unfrozen)",
    "ResNetJSCC_2e":    "Model 2e (128px Up)",
    "SwinJSCC":         "Model 4 (SwinJSCC)",
}


def _display_name(model_name: str, source_dir: str) -> str:
    """
    Return a readable label. For models that share a class name,
    use the source directory to disambiguate.
    """
    if model_name == "SwinRefineJSCC":
        dir_map = {
            "model3_swinir":     "S3: SwinIR+FiLM",
            "model3_swinir_rstb":"S3: SwinIR+RSTB",
            "model3_nosnr":      "S3: SwinIR noSNR",
            "model3_cnn":        "S3: CNN control",
            "model3b_lam0.001":  "S3b: CNN+CE lam=0.001",
            "model3b_lam0.003":  "S3b: CNN+CE lam=0.003",
            "model3b_lam0.01":   "S3b: CNN+CE lam=0.01",
        }
        return dir_map.get(source_dir, f"S3: {source_dir}")
    # PlainDeepJSCC with augmentation
    if model_name == "PlainDeepJSCC" and source_dir == "model1b":
        return "Model 1b (Plain+aug)"
    if model_name == "PlainDeepJSCC" and source_dir == "model1_fair":
        return "Model 1 Fair (30ep+aug)"
    return DISPLAY_NAMES.get(model_name, model_name)


def make_comparison_plots(rows: list, out_dir: str):
    """Generate PSNR / SSIM / Accuracy vs SNR plots – one per channel."""
    if not _MPL_OK:
        print("[SKIP] matplotlib not available – no plots generated.")
        return

    # Assign display names
    for r in rows:
        r["display"] = _display_name(r["model_name"], r.get("source_dir", ""))

    # Deduplicate model order (preserve first-seen order)
    seen = set()
    model_order = []
    for r in rows:
        if r["display"] not in seen:
            model_order.append(r["display"])
            seen.add(r["display"])

    channels = sorted(set(r["channel"] for r in rows))
    snrs = sorted(set(r["snr_db"] for r in rows))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
    })

    metrics = [
        ("psnr",     "PSNR (dB)",  "psnr_vs_snr"),
        ("ssim",     "SSIM",       "ssim_vs_snr"),
        ("accuracy", "Accuracy",   "acc_vs_snr"),
    ]

    for ch in channels:
        ch_rows = [r for r in rows if r["channel"] == ch]
        for metric_key, ylabel, fname_base in metrics:
            # Skip if all values are None
            vals = [r[metric_key] for r in ch_rows if r[metric_key] is not None]
            if not vals:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.5))

            for i, mname in enumerate(model_order):
                model_rows = [r for r in ch_rows if r["display"] == mname]
                model_rows.sort(key=lambda r: r["snr_db"])
                xs = [r["snr_db"] for r in model_rows if r[metric_key] is not None]
                ys = [r[metric_key] for r in model_rows if r[metric_key] is not None]

                if metric_key == "accuracy":
                    ys = [y * 100 for y in ys]   # convert to percentage

                if not xs:
                    continue

                color = PALETTE[i % len(PALETTE)]
                marker = MARKERS[i % len(MARKERS)]
                ax.plot(xs, ys, label=mname, color=color, marker=marker,
                        linewidth=2.2, markersize=7, markeredgecolor="white",
                        markeredgewidth=0.8)

            actual_ylabel = ylabel if metric_key != "accuracy" else "Accuracy (%)"
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(actual_ylabel)
            ax.set_title(f"{actual_ylabel}  vs  SNR   [{ch.upper()} channel]")
            ax.set_xticks(snrs)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=9, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            fpath = os.path.join(out_dir, f"{fname_base}_{ch}.png")
            fig.savefig(fpath, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare all DeepJSCC model results.")
    parser.add_argument("--results_dir", default="results",
                        help="Root directory containing per-model result folders.")
    parser.add_argument("--count_params", action="store_true",
                        help="Also compute and print parameter counts (requires torch).")
    parser.add_argument("--snr", type=float, default=10.0,
                        help="SNR (dB) for the summary table (default: 10).")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "final")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    csv_paths = discover_csvs(args.results_dir)
    if not csv_paths:
        print(f"[WARNING] No test_metrics.csv files found in {args.results_dir}/*/")
        if not args.count_params:
            print("  Run eval.py first, or check --results_dir.")
            sys.exit(1)
        else:
            print("  Skipping result aggregation. Running --count_params only.\n")
            run_param_counts(out_dir)
            return

    print(f"\nFound {len(csv_paths)} result file(s):")
    for p in csv_paths:
        print(f"  {p}")

    rows = load_all_csvs(csv_paths)
    if not rows:
        print("[WARNING] All CSVs were empty or unreadable.")
    else:
        print(f"Loaded {len(rows)} data points.\n")

        # ── Summary table at SNR = 10 dB ─────────────────────────────────
        summary = build_summary_table(rows, snr_db=args.snr)
        print_summary_table(summary, out_dir, snr_db=args.snr)

        # ── Plots ─────────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("  GENERATING COMPARISON PLOTS")
        print(f"{'=' * 70}")
        make_comparison_plots(rows, out_dir)

    # ── Parameter counts (optional) ───────────────────────────────────────
    if args.count_params:
        run_param_counts(out_dir)

    print(f"\n[OK]  All outputs saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
