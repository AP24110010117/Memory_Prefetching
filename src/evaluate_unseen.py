"""
evaluate_unseen.py — Out-of-Distribution Unseen Workload Evaluation.

Evaluates the trained Compact MLP (32–16) on a 100% independent workload
generated with an alternate seed (999), shifted base addresses, and
independent execution stream patterns to prove true generalization without memorization.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.generate_dataset import generate_unseen_workload


def evaluate_unseen_workload(model, scaler, feat_cols: list, seed: int = config.UNSEEN_SEED) -> dict:
    """
    Generates and evaluates an independent unseen memory access trace.
    Returns unseen metrics dictionary.
    """
    print(f"\n{'='*70}")
    print(f"  EVALUATING MODEL ON 100% UNSEEN WORKLOAD (Seed = {seed})")
    print(f"{'='*70}")

    df_unseen = generate_unseen_workload(num_accesses=15000, seed=seed, save=True)

    X_unseen_raw = df_unseen[feat_cols].values.astype(float)
    y_unseen     = df_unseen["target_class"].values.astype(int)

    X_unseen_sc = scaler.transform(X_unseen_raw)
    probs = model.predict_proba(X_unseen_sc)
    preds = probs.argmax(1)
    confs = probs.max(1)

    acc = float(accuracy_score(y_unseen, preds))
    p, r, f, _ = precision_recall_fscore_support(
        y_unseen, preds, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    p_mac, r_mac, f_mac = float(p.mean()), float(r.mean()), float(f.mean())

    print(f"  Unseen Samples Evaluated    : {len(y_unseen):,}")
    print(f"  Unseen Test Accuracy        : {acc*100:.2f}%  (Target: 90-95%+)")
    print(f"  Unseen Macro Precision      : {p_mac*100:.2f}%")
    print(f"  Unseen Macro Recall         : {r_mac*100:.2f}%")
    print(f"  Unseen Macro F1-Score       : {f_mac*100:.2f}%")
    print(f"  Unseen Average Confidence   : {confs.mean()*100:.2f}%")

    rpt = classification_report(y_unseen, preds, target_names=config.CLASS_NAMES, digits=4, zero_division=0)
    print(f"\n  Unseen Workload Classification Report:\n{rpt}")

    # Save unseen report
    unseen_report_path = config.RESULTS_DIR / "unseen_workload_report.txt"
    with open(unseen_report_path, "w", encoding="utf-8") as f_:
        f_.write(f"=== UNSEEN WORKLOAD EVALUATION (Seed = {seed}) ===\n")
        f_.write(f"Total Accesses: {len(y_unseen):,}\n")
        f_.write(f"Unseen Accuracy: {acc*100:.2f}%\n")
        f_.write(f"Unseen Macro Precision: {p_mac*100:.2f}%\n")
        f_.write(f"Unseen Macro Recall: {r_mac*100:.2f}%\n")
        f_.write(f"Unseen Macro F1-Score: {f_mac*100:.2f}%\n\n")
        f_.write(rpt)
    print(f"  Saved unseen evaluation report -> {unseen_report_path}")

    return {
        "unseen_accuracy":        acc,
        "unseen_macro_precision": p_mac,
        "unseen_macro_recall":    r_mac,
        "unseen_macro_f1":        f_mac,
        "unseen_samples":         len(y_unseen),
        "unseen_mean_confidence": float(confs.mean()),
        "confusion_matrix":       confusion_matrix(y_unseen, preds, labels=list(range(config.NUM_CLASSES)))
    }
