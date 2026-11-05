"""
stats.py
--------
Statistical rigor and hypothesis testing module for DeepJSCC evaluations.

Capabilities:
  1. Wilson Score 95% Confidence Intervals for classifier accuracy.
  2. Paired bootstrap 95% Confidence Intervals (10,000 resamples) for Delta PSNR and Delta SSIM.
  3. McNemar's test (with continuity correction and exact binomial p-value) for accuracy comparison.
  4. Per-image CSV loader and batch comparison reporter.

Usage:
  # Pairwise comparison between two models at AWGN 10 dB:
  python stats.py --model_a results/model1_fair/per_image_awgn_10.csv --model_b results/model2d/per_image_awgn_10.csv

  # Pairwise comparison passing model directories:
  python stats.py --model_a results/model3b_lam0.001 --model_b results/model2d --channel awgn --snr 10

  # Single model statistical summary:
  python stats.py --model_a results/model1_fair/per_image_awgn_10.csv
"""

import os
import sys
import math
import argparse
import csv
import numpy as np
from scipy import stats


def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
    """
    Computes Wilson score interval for binomial proportion.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z ** 2

    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n ** 2)))

    ci_lo = max(0.0, centre - margin)
    ci_hi = min(1.0, centre + margin)
    return p, ci_lo, ci_hi


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_bootstraps: int = 10000,
                        confidence: float = 0.95, seed: int = 42):
    """
    Computes paired bootstrap confidence interval for the mean difference (a - b).
    Returns: (diff_mean, ci_lo, ci_hi, diff_std)
    """
    diff = a - b
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(diff[boot_indices], axis=1)

    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff, ddof=1))

    return diff_mean, ci_lo, ci_hi, diff_std


def mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray):
    """
    Performs McNemar's test for paired binary outcomes.
    corr_a, corr_b: boolean or integer (0 or 1) arrays indicating correctness.
    Returns: dict with contingency table, chi2 with continuity correction, p-value, and exact binomial p-value.
    """
    corr_a = np.asarray(corr_a).astype(bool)
    corr_b = np.asarray(corr_b).astype(bool)

    a = int(np.sum(corr_a & corr_b))        # both correct
    b = int(np.sum(corr_a & (~corr_b)))     # A correct, B wrong
    c = int(np.sum((~corr_a) & corr_b))     # A wrong, B correct
    d = int(np.sum((~corr_a) & (~corr_b)))  # both wrong

    total_discordant = b + c
    if total_discordant == 0:
        chi2 = 0.0
        p_val_chi2 = 1.0
        p_val_exact = 1.0
    else:
        # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
        num = max(0, abs(b - c) - 1) ** 2
        chi2 = float(num / total_discordant)
        p_val_chi2 = float(stats.chi2.sf(chi2, df=1))

        # Two-sided exact binomial test
        try:
            # scipy >= 1.7.0
            res = stats.binomtest(min(b, c), total_discordant, p=0.5, alternative='two-sided')
            p_val_exact = float(res.pvalue)
        except AttributeError:
            p_val_exact = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), total_discordant, 0.5)))

    return {
        "contingency_table": {"a (both correct)": a, "b (A right, B wrong)": b,
                              "c (A wrong, B right)": c, "d (both wrong)": d},
        "discordant_pairs": total_discordant,
        "chi2_continuity": chi2,
        "p_val_chi2": p_val_chi2,
        "p_val_exact": p_val_exact,
        "statistically_significant_05": p_val_exact < 0.05
    }


def load_per_image_csv(path: str):
    """
    Loads per_image_<ch>_<snr>.csv and returns dict of numpy arrays.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Per-image CSV not found: {path}")

    indices = []
    psnrs = []
    ssims = []
    labels = []
    corr1 = []
    corr2 = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["image_idx"]))
            psnrs.append(float(row["psnr"]) if row["psnr"] != "" else np.nan)
            ssims.append(float(row["ssim"]) if row["ssim"] != "" else np.nan)
            labels.append(int(row["true_label"]) if row["true_label"] != "" else -1)
            corr1.append(int(row["correct_clf1"]) if row.get("correct_clf1") not in (None, "") else -1)
            corr2.append(int(row["correct_clf2"]) if row.get("correct_clf2") not in (None, "") else -1)

    return {
        "image_idx": np.array(indices),
        "psnr": np.array(psnrs),
        "ssim": np.array(ssims),
        "true_label": np.array(labels),
        "correct_clf1": np.array(corr1),
        "correct_clf2": np.array(corr2)
    }


def resolve_path(arg_path: str, channel: str = "awgn", snr: int = 10):
    """
    If arg_path is a directory, look for per_image_<channel>_<snr>.csv.
    """
    if os.path.isdir(arg_path):
        target = os.path.join(arg_path, f"per_image_{channel}_{snr}.csv")
        if os.path.isfile(target):
            return target
        # Also check under results/<arg_path>
        alt_target = os.path.join("results", arg_path, f"per_image_{channel}_{snr}.csv")
        if os.path.isfile(alt_target):
            return alt_target
        raise FileNotFoundError(f"Could not find per_image CSV in {arg_path} for {channel} @ {snr} dB")
    return arg_path


def main():
    parser = argparse.ArgumentParser(description="DeepJSCC Statistical Rigor & Hypothesis Testing")
    parser.add_argument("--model_a", required=True, help="Path to per-image CSV or directory for Model A")
    parser.add_argument("--model_b", default=None, help="Path to per-image CSV or directory for Model B (optional)")
    parser.add_argument("--channel", default="awgn", help="Channel condition (awgn or rayleigh)")
    parser.add_argument("--snr", type=int, default=10, help="SNR in dB (default 10)")
    parser.add_argument("--n_bootstraps", type=int, default=10000, help="Number of bootstrap resamples (default 10,000)")
    args = parser.parse_args()

    path_a = resolve_path(args.model_a, args.channel, args.snr)
    data_a = load_per_image_csv(path_a)
    n_a = len(data_a["image_idx"])

    print(f"\n========================================================")
    print(f"Statistical Evaluation: {os.path.basename(path_a)}")
    print(f"Sample size: N = {n_a}")
    print(f"========================================================")

    # Model A Single Stats
    psnr_a = data_a["psnr"]
    ssim_a = data_a["ssim"]
    print(f"PSNR: {np.mean(psnr_a):.4f} +/- {np.std(psnr_a, ddof=1):.4f} dB (range: [{np.min(psnr_a):.2f}, {np.max(psnr_a):.2f}])")
    print(f"SSIM: {np.mean(ssim_a):.4f} +/- {np.std(ssim_a, ddof=1):.4f} (range: [{np.min(ssim_a):.3f}, {np.max(ssim_a):.3f}])")

    if np.any(data_a["correct_clf1"] >= 0):
        c1 = data_a["correct_clf1"]
        p1, lo1, hi1 = wilson_score_interval(int(np.sum(c1 == 1)), len(c1))
        print(f"ResNet-18 Accuracy:    {p1 * 100:.2f}%  [95% Wilson CI: {lo1 * 100:.2f}%, {hi1 * 100:.2f}%]")

    if np.any(data_a["correct_clf2"] >= 0):
        c2 = data_a["correct_clf2"]
        p2, lo2, hi2 = wilson_score_interval(int(np.sum(c2 == 1)), len(c2))
        print(f"MobileNet-V2 Accuracy:  {p2 * 100:.2f}%  [95% Wilson CI: {lo2 * 100:.2f}%, {hi2 * 100:.2f}%]")

    if args.model_b:
        path_b = resolve_path(args.model_b, args.channel, args.snr)
        data_b = load_per_image_csv(path_b)
        n_b = len(data_b["image_idx"])
        if n_a != n_b:
            print(f"[ERROR] Sample sizes differ: N_A={n_a}, N_B={n_b}. Cannot perform paired tests.")
            return

        name_a = os.path.basename(os.path.dirname(os.path.abspath(path_a)))
        name_b = os.path.basename(os.path.dirname(os.path.abspath(path_b)))

        print(f"\n========================================================")
        print(f"Paired Hypothesis Testing: {name_a} vs {name_b}")
        print(f"Condition: {args.channel.upper()} @ {args.snr} dB  (N = {n_a})")
        print(f"========================================================")

        # Delta PSNR Paired Bootstrap
        d_psnr, lo_psnr, hi_psnr, s_psnr = paired_bootstrap_ci(data_a["psnr"], data_b["psnr"],
                                                               n_bootstraps=args.n_bootstraps)
        print(f"\n--- 1. Reconstruction Fidelity Differences (Paired Bootstrap, B={args.n_bootstraps:,}) ---")
        print(f"Delta PSNR ({name_a} - {name_b}):")
        print(f"  Mean Difference:  {d_psnr:+.4f} dB  (std: {s_psnr:.4f} dB)")
        print(f"  95% Bootstrap CI: [{lo_psnr:+.4f} dB, {hi_psnr:+.4f} dB]")
        sig_psnr = "YES (p < 0.05)" if (lo_psnr > 0 or hi_psnr < 0) else "NO (spans 0)"
        print(f"  Statistically Significant: {sig_psnr}")

        # Delta SSIM Paired Bootstrap
        d_ssim, lo_ssim, hi_ssim, s_ssim = paired_bootstrap_ci(data_a["ssim"], data_b["ssim"],
                                                               n_bootstraps=args.n_bootstraps)
        print(f"\nDelta SSIM ({name_a} - {name_b}):")
        print(f"  Mean Difference:  {d_ssim:+.4f}  (std: {s_ssim:.4f})")
        print(f"  95% Bootstrap CI: [{lo_ssim:+.4f}, {hi_ssim:+.4f}]")
        sig_ssim = "YES (p < 0.05)" if (lo_ssim > 0 or hi_ssim < 0) else "NO (spans 0)"
        print(f"  Statistically Significant: {sig_ssim}")

        # McNemar Test on Classifier 1 (ResNet-18)
        if np.any(data_a["correct_clf1"] >= 0) and np.any(data_b["correct_clf1"] >= 0):
            print(f"\n--- 2. Semantic Accuracy Significance (Classifier 1: ResNet-18) ---")
            mcn1 = mcnemar_test(data_a["correct_clf1"], data_b["correct_clf1"])
            tbl1 = mcn1["contingency_table"]
            print(f"  Contingency Table: Both Right={tbl1['a (both correct)']}, "
                  f"A Right/B Wrong={tbl1['b (A right, B wrong)']}, "
                  f"A Wrong/B Right={tbl1['c (A wrong, B right)']}, "
                  f"Both Wrong={tbl1['d (both wrong)']}")
            print(f"  Discordant Pairs (b + c): {mcn1['discordant_pairs']}")
            print(f"  McNemar chi2 (continuity-corrected): {mcn1['chi2_continuity']:.4f} (p = {mcn1['p_val_chi2']:.4e})")
            print(f"  Exact Binomial p-value:              {mcn1['p_val_exact']:.4e}")
            print(f"  Statistically Significant at alpha=0.05: {mcn1['statistically_significant_05']}")

        # McNemar Test on Classifier 2 (MobileNet-V2)
        if np.any(data_a["correct_clf2"] >= 0) and np.any(data_b["correct_clf2"] >= 0):
            print(f"\n--- 3. Independent Classifier Transfer Significance (Classifier 2: MobileNet-V2) ---")
            mcn2 = mcnemar_test(data_a["correct_clf2"], data_b["correct_clf2"])
            tbl2 = mcn2["contingency_table"]
            print(f"  Contingency Table: Both Right={tbl2['a (both correct)']}, "
                  f"A Right/B Wrong={tbl2['b (A right, B wrong)']}, "
                  f"A Wrong/B Right={tbl2['c (A wrong, B right)']}, "
                  f"Both Wrong={tbl2['d (both wrong)']}")
            print(f"  Discordant Pairs (b + c): {mcn2['discordant_pairs']}")
            print(f"  McNemar chi2 (continuity-corrected): {mcn2['chi2_continuity']:.4f} (p = {mcn2['p_val_chi2']:.4e})")
            print(f"  Exact Binomial p-value:              {mcn2['p_val_exact']:.4e}")
            print(f"  Statistically Significant at alpha=0.05: {mcn2['statistically_significant_05']}")


if __name__ == "__main__":
    main()

