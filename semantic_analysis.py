"""
semantic_analysis.py
--------------------
Detailed semantic land-use analysis for EuroSAT DeepJSCC research.

Computes:
  1. Per-class classification accuracy across all 10 EuroSAT land-use classes.
  2. Confusion matrices for both ResNet-18 and MobileNet-V2.
  3. Precision, Recall, and Macro-F1 scores.
  4. Cross-model semantic gains (e.g., Model 3b vs Model 2d Long).
  5. Publication-quality confusion matrix heatmaps and Pareto trade-off curves.

Usage:
  python semantic_analysis.py --all
  python semantic_analysis.py --csv results/model3b_lam0.0003/per_image_awgn_10.csv --name "Model 3b (lam=0.0003)"
"""

import os
import sys
import argparse
import csv
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVeg", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def load_per_image(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    y_true = []
    y_pred1 = []
    y_pred2 = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["true_label"]))
            y_pred1.append(int(row["pred_clf1"]) if row.get("pred_clf1") not in (None, "") else -1)
            y_pred2.append(int(row["pred_clf2"]) if row.get("pred_clf2") not in (None, "") else -1)

    return np.array(y_true), np.array(y_pred1), np.array(y_pred2)


def compute_metrics(y_true, y_pred, num_classes=10):
    valid = (y_pred >= 0)
    y_t = y_true[valid]
    y_p = y_pred[valid]

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class_acc = []
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = cm[c, c]
        total_true = np.sum(cm[c, :])
        total_pred = np.sum(cm[:, c])

        acc = (tp / total_true) if total_true > 0 else 0.0
        prec = (tp / total_pred) if total_pred > 0 else 0.0
        rec = acc
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_acc.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    overall_acc = float(np.sum(np.diag(cm)) / max(len(y_t), 1))

    return {
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "macro_f1": macro_f1,
        "overall_acc": overall_acc,
        "total_samples": len(y_t),
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    if not _MPL_OK:
        return
    plt.figure(figsize=(8, 7))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            plt.text(j, i, val_str,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=7,
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel("True Class", fontsize=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pareto_curve(models_data, save_path="results/final/pareto_tradeoff.png"):
    if not _MPL_OK:
        return
    plt.figure(figsize=(9, 6))

    for m in models_data:
        name = m["name"]
        psnr = m["psnr"]
        r18 = m["acc_r18"] * 100.0
        mob = m["acc_mob"] * 100.0
        color = m.get("color", "blue")
        marker = m.get("marker", "o")

        plt.scatter(psnr, r18, color=color, s=120, marker=marker, label=f"{name} (ResNet-18)")
        plt.scatter(psnr, mob, color=color, s=120, marker="^", edgecolors="black", label=f"{name} (MobileNet-V2)" if "Model 3b (lam=0.0003)" in name else "")
        plt.annotate(f"{name}\nR-18: {r18:.1f}%\nMob: {mob:.1f}%",
                     (psnr, r18), textcoords="offset points", xytext=(8, -8), fontsize=8)

    plt.xlabel("Reconstruction PSNR (dB) @ 10 dB AWGN", fontsize=11)
    plt.ylabel("Downstream Classification Accuracy (%)", fontsize=11)
    plt.title("Pareto Trade-Off: Reconstruction Fidelity vs Semantic Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Pareto plot saved to: {save_path}")


def analyze_model(csv_path: str, model_name: str, out_dir: str):
    y_true, y_pred1, y_pred2 = load_per_image(csv_path)

    print(f"\n========================================================")
    print(f"Semantic Evaluation: {model_name}")
    print(f"Source CSV: {csv_path}")
    print(f"========================================================")

    res1 = compute_metrics(y_true, y_pred1)
    res2 = compute_metrics(y_true, y_pred2)

    print(f"ResNet-18 Accuracy:    {res1['overall_acc']*100:.2f}% | Macro-F1: {res1['macro_f1']*100:.2f}%")
    print(f"MobileNet-V2 Accuracy:  {res2['overall_acc']*100:.2f}% | Macro-F1: {res2['macro_f1']*100:.2f}%")

    print("\n--- Per-Class Accuracies ---")
    print(f"{'Class':<22} | {'ResNet-18':<10} | {'MobileNet-V2':<12}")
    print("-" * 50)
    for c, cname in enumerate(CLASS_NAMES):
        acc1 = res1['per_class_acc'][c] * 100
        acc2 = res2['per_class_acc'][c] * 100
        print(f"{cname:<22} | {acc1:6.2f}%    | {acc2:6.2f}%")

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "")
    cm1_path = os.path.join(out_dir, f"cm_resnet18_{safe_name}.png")
    cm2_path = os.path.join(out_dir, f"cm_mobilenet_{safe_name}.png")

    plot_confusion_matrix(res1["confusion_matrix"], CLASS_NAMES,
                          f"{model_name} -- ResNet-18 Confusion Matrix (AWGN 10 dB)", cm1_path)
    plot_confusion_matrix(res2["confusion_matrix"], CLASS_NAMES,
                          f"{model_name} -- MobileNet-V2 Confusion Matrix (AWGN 10 dB)", cm2_path)
    print(f"Saved confusion matrices to {out_dir}")

    return {
        "model": model_name,
        "res1": res1,
        "res2": res2,
    }


def main():
    parser = argparse.ArgumentParser(description="EuroSAT DeepJSCC Semantic Analysis")
    parser.add_argument("--all", action="store_true", help="Analyze all key models")
    parser.add_argument("--csv", default=None, help="Single per-image CSV path")
    parser.add_argument("--name", default="Model", help="Model name for single evaluation")
    parser.add_argument("--out_dir", default="results/final/semantic_analysis", help="Output directory")
    args = parser.parse_args()

    if args.all:
        models = [
            ("Model 2d Long (Base 60ep)", "results/model2d_long/per_image_awgn_10.csv", 25.52, "gray", "s"),
            ("Model 3b (lam=0.0003)", "results/model3b_lam0.0003/per_image_awgn_10.csv", 25.39, "green", "o"),
            ("Model 3b (lam=0.001)", "results/model3b_lam0.001/per_image_awgn_10.csv", 25.23, "blue", "o"),
            ("Model 3b (lam=0.003)", "results/model3b_lam0.003/per_image_awgn_10.csv", 25.08, "red", "o"),
            ("Model 1 Fair (PlainJSCC)", "results/model1_fair/per_image_awgn_10.csv", 31.07, "purple", "^"),
        ]
        pareto_list = []
        for name, p, psnr_val, color, marker in models:
            if os.path.isfile(p):
                info = analyze_model(p, name, args.out_dir)
                pareto_list.append({
                    "name": name,
                    "psnr": psnr_val,
                    "acc_r18": info["res1"]["overall_acc"],
                    "acc_mob": info["res2"]["overall_acc"],
                    "color": color,
                    "marker": marker
                })

        if pareto_list:
            plot_pareto_curve(pareto_list, os.path.join(args.out_dir, "pareto_tradeoff.png"))

    elif args.csv:
        analyze_model(args.csv, args.name, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

