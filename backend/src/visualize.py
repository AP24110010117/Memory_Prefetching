"""
visualize.py — Publication-Quality Visualizations for ML Memory Prefetching.

Generates and saves PNG plots to results/graphs/:
  1. training_validation_loss.png
  2. training_validation_accuracy.png
  3. combined_learning_curves.png
  4. loss_vs_epoch.png
  5. accuracy_vs_epoch.png
  6. confusion_matrix.png
  7. model_comparison.png
  8. per_class_performance.png
  9. feature_importance.png
  10. ablation_comparison.png
  11. confidence_threshold_sweep.png
  12. cache_hit_miss_comparison.png
  13. prefetch_breakdown.png
  14. amat_comparison.png
  15. boundary_validation.png
"""

import os, sys
os.environ.setdefault("MPLCONFIGDIR", os.path.join(__import__("tempfile").gettempdir(), "mpl_pf"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT = config.GRAPHS_DIR
os.makedirs(OUT, exist_ok=True)

STYLE = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
plt.rcParams.update({"figure.dpi": 200, "font.size": 10})


def _save(fig, name, tight=True):
    if tight:
        fig.tight_layout(pad=1.5)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {path}")
    return path


def plot_loss(history: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss"], "b-o", ms=4, lw=2, label="Train Loss (70%)")
        ax.plot(ep, history["val_loss"], "r--s", ms=4, lw=2, label="Val Loss (10%)")
        ax.set_title("Training vs Validation Loss (Compact MLP 32-16)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "training_validation_loss.png")


def plot_accuracy(history: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ep = history["epoch"]
        tr_p = np.array(history["train_acc"]) * 100
        va_p = np.array(history["val_acc"]) * 100
        ax.plot(ep, tr_p, "g-o", ms=4, lw=2, label="Train Acc (70%)")
        ax.plot(ep, va_p, "m--^", ms=4, lw=2, label="Val Acc (10%)")
        ax.axhline(90.0, color="gray", linestyle=":", label="Target Min (90%)")
        ax.axhline(95.0, color="gray", linestyle=":", label="Target Max (95%)")
        ax.set_title("Training vs Validation Accuracy (Compact MLP 32-16)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="lower right")
        ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "training_validation_accuracy.png")


def plot_combined(history: dict):
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ep = history["epoch"]
        ax1.plot(ep, history["train_loss"], "b-", lw=2, label="Train Loss")
        ax1.plot(ep, history["val_loss"], "r--", lw=2, label="Val Loss")
        ax1.set_title("Loss Trajectory", fontweight="bold")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(); ax1.grid(True, ls=":", alpha=0.7)

        tr_p = np.array(history["train_acc"]) * 100
        va_p = np.array(history["val_acc"]) * 100
        ax2.plot(ep, tr_p, "g-", lw=2, label="Train Accuracy")
        ax2.plot(ep, va_p, "m--", lw=2, label="Val Accuracy")
        ax2.axhline(90.0, color="gray", linestyle=":", label="Target Min (90%)")
        ax2.set_title("Accuracy Trajectory", fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        ax2.legend(loc="lower right"); ax2.grid(True, ls=":", alpha=0.7)

        fig.suptitle("Compact Deep MLP (32–16) Learning Curves", fontweight="bold", y=1.02)
    return _save(fig, "combined_learning_curves.png")


def plot_loss_vs_epoch(history: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss"], color="#1f77b4", lw=2.2, label="Training Loss")
        ax.plot(ep, history["val_loss"], color="#d62728", lw=2.2, linestyle="--", label="Validation Loss")
        ax.set_title("Binary/Multi-Class Cross-Entropy Loss vs Epoch", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "loss_vs_epoch.png")


def plot_acc_vs_epoch(history: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ep = history["epoch"]
        ax.plot(ep, [a * 100 for a in history["train_acc"]], color="#2ca02c", lw=2.2, label="Training Accuracy (%)")
        ax.plot(ep, [a * 100 for a in history["val_acc"]], color="#9467bd", lw=2.2, linestyle="--", label="Validation Accuracy (%)")
        ax.axhline(90.0, color="black", linestyle=":", alpha=0.6, label="90% Benchmark")
        ax.set_title("Accuracy vs Epoch (Compact MLP 32-16)", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="lower right"); ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "accuracy_vs_epoch.png")


def plot_confusion_matrix(cm: np.ndarray):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 7.5))
        cm_norm = cm.astype(float) / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                    cbar_kws={"label": "Normalized Accuracy (%)"}, ax=ax)
        ax.set_title("Confusion Matrix on Independent Test Set", fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Stride Class", fontweight="bold")
        ax.set_ylabel("True Stride Class", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, "confusion_matrix.png")


def plot_model_comparison(results: dict):
    with plt.style.context(STYLE):
        names = list(results.keys())
        val_accs = [results[m]["val_acc"] * 100 for m in names]
        test_accs = [results[m]["test_acc"] * 100 for m in names]

        x = np.arange(len(names))
        w = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - w/2, val_accs, w, label="Validation Accuracy (%)", color="#3498db")
        b2 = ax.bar(x + w/2, test_accs, w, label="Test Accuracy (%)", color="#2ecc71")

        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_title("Comprehensive Model Architecture Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontweight="bold")
        ax.set_ylim([0, 105])
        ax.legend(loc="lower right")

        for rect in list(b1) + list(b2):
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    return _save(fig, "model_comparison.png")


def plot_per_class(p: np.ndarray, r: np.ndarray, f: np.ndarray):
    with plt.style.context(STYLE):
        x = np.arange(len(config.CLASS_NAMES))
        w = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, p * 100, w, label="Precision (%)", color="#3498db")
        ax.bar(x,     r * 100, w, label="Recall (%)",    color="#2ecc71")
        ax.bar(x + w, f * 100, w, label="F1-Score (%)",  color="#e67e22")

        ax.set_ylabel("Score (%)", fontweight="bold")
        ax.set_title("Per-Class Precision, Recall, and F1-Score Breakdown", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=9)
        ax.set_ylim([0, 105])
        ax.legend(loc="lower left")
    return _save(fig, "per_class_performance.png")


def plot_feature_importance(df_imp: pd.DataFrame, top_n: int = 15):
    with plt.style.context(STYLE):
        df_top = df_imp.head(top_n).sort_values("importance_mean", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_top["feature"], df_top["importance_mean"] * 100, color="#16a085", edgecolor="none")
        ax.set_xlabel("Mean Accuracy Drop when Shuffled (%)", fontweight="bold")
        ax.set_title(f"Top {top_n} Permutation Feature Importances", fontweight="bold")
        ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "feature_importance.png")


def plot_ablation(df_ab: pd.DataFrame):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df_ab.sort_values("val_accuracy", ascending=True)
        colors = ["#e74c3c" if "No" in c or "Only" in c else "#27ae60" for c in df_sorted["configuration"]]
        bars = ax.barh(df_sorted["configuration"], df_sorted["val_accuracy"], color=colors)
        ax.set_xlabel("Validation Accuracy (%)", fontweight="bold")
        ax.set_xlim([0, 100])
        ax.set_title("Feature Ablation Study (Impact of Context Features)", fontweight="bold")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    return _save(fig, "ablation_comparison.png")


def plot_threshold_sweep(df_sw: pd.DataFrame):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        th = df_sw["threshold"]
        ax.plot(th, df_sw["useful_precision_pct"], "b-o", lw=2, label="Useful Precision (%)")
        ax.plot(th, df_sw["coverage_pct"],         "g-s", lw=2, label="Coverage (<= 100%)")
        ax.plot(th, df_sw["pollution_rate_pct"],    "r-^", lw=2, label="Pollution Rate (%)")

        ax.set_xlabel("Confidence Threshold (tau)", fontweight="bold")
        ax.set_ylabel("Metric (%)", fontweight="bold")
        ax.set_title("Confidence Threshold Sweep on Validation Set", fontweight="bold")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.7)
    return _save(fig, "confidence_threshold_sweep.png")


def plot_cache_comparison(bl_m: dict, ad_m: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        categories = ["L1 Hit Rate", "L2 Hit Rate", "Overall Hit Rate", "Prefetch Coverage"]
        bl_vals = [bl_m["l1_hit_rate_pct"], bl_m["l2_hit_rate_pct"], bl_m["overall_hit_rate_pct"], 0.0]
        ad_vals = [ad_m["l1_hit_rate_pct"], ad_m["l2_hit_rate_pct"], ad_m["overall_hit_rate_pct"], ad_m["coverage_pct"]]

        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w/2, bl_vals, w, label="Demand-Only Baseline", color="#7f8c8d")
        ax.bar(x + w/2, ad_vals, w, label="Adaptive ML Prefetcher", color="#2980b9")

        ax.set_ylabel("Percentage (%)", fontweight="bold")
        ax.set_title("Cache Hit Rate and Standard Prefetch Coverage Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight="bold")
        ax.set_ylim([0, 105])
        ax.legend(loc="upper left")
    return _save(fig, "cache_hit_miss_comparison.png")


def plot_prefetch_breakdown(ad_m: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Timely Useful", "Late Useful", "Unused / Evicted", "Redundant / Filtered"]
        vals = [ad_m["pf_timely"], ad_m["pf_late"], ad_m["pf_unused"], ad_m["pf_redundant"]]
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

        bars = ax.bar(categories, vals, color=colors, edgecolor="none", width=0.55)
        ax.set_ylabel("Number of Prefetches", fontweight="bold")
        ax.set_title("Prefetch Outcome Breakdown (Test Set)", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    return _save(fig, "prefetch_breakdown.png")


def plot_amat(bl_m: dict, ad_m: dict):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["Demand-Only Baseline", "Adaptive ML Prefetcher"]
        amats = [bl_m["amat_cycles"], ad_m["amat_cycles"]]
        colors = ["#c0392b", "#27ae60"]

        bars = ax.bar(labels, amats, color=colors, width=0.45)
        ax.set_ylabel("Average Memory Access Time (Cycles)", fontweight="bold")
        ax.set_title("AMAT Reduction with Adaptive Prefetching", fontweight="bold")
        reduction = bl_m["amat_cycles"] - ad_m["amat_cycles"]
        red_pct = (reduction / bl_m["amat_cycles"]) * 100
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f} cycles", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim([0, max(amats) * 1.2])
        ax.text(0.5, max(amats) * 1.05, f"AMAT Improvement: -{reduction:.1f} cycles (-{red_pct:.1f}%)",
                ha="center", fontsize=11, fontweight="bold", color="#16a085")
    return _save(fig, "amat_comparison.png")


def plot_boundary_validation(df_tc: pd.DataFrame, boundary_stats: dict):
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        # Left: Test cases pass rate
        counts = df_tc["status"].value_counts()
        ax1.bar(counts.index, counts.values, color=["#2ecc71" if s == "PASSED" else "#e74c3c" for s in counts.index], width=0.4)
        ax1.set_ylabel("Test Cases Count", fontweight="bold")
        ax1.set_title("Boundary Safety Test Cases (12 Tests across Segments)", fontweight="bold")
        for i, v in enumerate(counts.values):
            ax1.text(i, v + 0.2, f"{v} Passed", ha="center", fontweight="bold")

        # Right: Actual prefetch blocks in simulation
        labels = ["Page Crossing Blocked", "Segment Limit Blocked"]
        vals = [boundary_stats["page_blocked"], boundary_stats["seg_blocked"]]
        ax2.bar(labels, vals, color=["#3498db", "#e67e22"], width=0.45)
        ax2.set_ylabel("Invalid Prefetches Prevented", fontweight="bold")
        ax2.set_title("Prefetch Hardware Hazard Prevention (Test Set)", fontweight="bold")
        for i, v in enumerate(vals):
            ax2.text(i, v + max(vals)*0.02, f"{v:,} Blocked", ha="center", fontweight="bold")

        fig.suptitle("Memory Paging & Segmentation Boundary Protection", fontweight="bold", y=1.02)
    return _save(fig, "boundary_validation.png")
