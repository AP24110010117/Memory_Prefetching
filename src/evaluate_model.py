"""
evaluate_model.py — Independent Test Set & Out-of-Distribution Workload Evaluator.

Evaluates trained prefetch models on:
1. The 20% chronological held-out test partition.
2. An independent unseen memory workload (new base addresses, new random seed).
Computes:
- Top-1 Accuracy, Precision, Recall, Macro F1-score
- Confusion matrix and per-class breakdown
- Hardware inference latency
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_on_test(model, X_te: np.ndarray, y_te: np.ndarray,
                     model_name: str = "Compact MLP (32-16)",
                     report_filename: str = "classification_report.txt") -> dict:
    """
    Evaluate model on unseen test data without any tuning.
    Returns metrics dictionary and saves evaluation reports.
    """
    print(f"\n{'='*70}")
    print(f"  TEST SET EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"  Test Samples Evaluated: {len(y_te):,}")

    t0 = time.perf_counter()
    probs = model.predict_proba(X_te)
    t1 = time.perf_counter()
    preds = probs.argmax(1)
    confs = probs.max(1)

    lat_total_ms = (t1 - t0) * 1000
    lat_per_us   = (lat_total_ms / len(y_te)) * 1000

    acc = float(accuracy_score(y_te, preds))
    p, r, f, _ = precision_recall_fscore_support(
        y_te, preds, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    p_mac, r_mac, f_mac = float(p.mean()), float(r.mean()), float(f.mean())

    print(f"\n  Test Accuracy (Target 90-95%+): {acc*100:.2f}%")
    print(f"  Macro Precision               : {p_mac*100:.2f}%")
    print(f"  Macro Recall (Sensitivity)    : {r_mac*100:.2f}%")
    print(f"  Macro F1-Score                : {f_mac*100:.2f}%")
    print(f"  Average Confidence            : {confs.mean()*100:.2f}%")
    print(f"  Average Inference Latency     : {lat_per_us:.2f} µs/access")

    print(f"\n  Classification Report:")
    rpt = classification_report(y_te, preds,
                                target_names=config.CLASS_NAMES,
                                digits=4, zero_division=0)
    print(rpt)

    # Save classification report
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.RESULTS_DIR / report_filename, "w", encoding="utf-8") as f_:
        f_.write(f"Model: {model_name}\n")
        f_.write(f"Test Accuracy: {acc*100:.2f}%\n")
        f_.write(f"Macro Precision: {p_mac*100:.2f}%\n")
        f_.write(f"Macro Recall: {r_mac*100:.2f}%\n")
        f_.write(f"Macro F1-Score: {f_mac*100:.2f}%\n\n")
        f_.write(rpt)

    # Save per-class metrics CSV
    df_per_class = pd.DataFrame({
        "class":     config.CLASS_NAMES,
        "precision": p,
        "recall":    r,
        "f1":        f,
    })
    df_per_class.to_csv(config.RESULTS_DIR / "evaluation_metrics.csv", index=False)

    cm = confusion_matrix(y_te, preds, labels=list(range(config.NUM_CLASSES)))

    return {
        "accuracy":        acc,
        "macro_precision": p_mac,
        "macro_recall":    r_mac,
        "macro_f1":        f_mac,
        "per_class_p":     p,
        "per_class_r":     r,
        "per_class_f":     f,
        "confusion_matrix": cm,
        "mean_confidence": float(confs.mean()),
        "lat_per_us":      lat_per_us,
        "preds":           preds,
        "probs":           probs,
    }
