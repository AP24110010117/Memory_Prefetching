"""
feature_importance.py — Permutation Feature Importance & Feature Ablation Study.

1. Permutation Feature Importance:
   Measures accuracy degradation when each feature column is shuffled on VALIDATION data.
2. Feature Ablation Study:
   Evaluates performance without Page Features, without Segment Features,
   without Cache Features, and without Stride History.
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_permutation_importance(model, X_va: np.ndarray, y_va: np.ndarray,
                                   feat_cols: list, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Computes permutation feature importance on validation set.
    Returns DataFrame sorted by mean importance drop.
    """
    np.random.seed(seed)
    base_acc = float((model.predict(X_va) == y_va).mean())
    n_feats  = X_va.shape[1]
    means    = np.zeros(n_feats)
    stds     = np.zeros(n_feats)

    for fi in range(n_feats):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_va.copy()
            perm = np.random.permutation(len(X_va))
            X_perm[:, fi] = X_va[perm, fi]
            perm_acc = float((model.predict(X_perm) == y_va).mean())
            drops.append(base_acc - perm_acc)
        means[fi] = float(np.mean(drops))
        stds[fi]  = float(np.std(drops))

    df = pd.DataFrame({
        "feature":         feat_cols,
        "importance_mean": means,
        "importance_std":  stds,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def run_ablation(X_tr_raw: np.ndarray, y_tr: np.ndarray,
                 X_va_raw: np.ndarray, y_va: np.ndarray,
                 feat_cols: list) -> pd.DataFrame:
    """
    Trains fast ablation models with different feature subsets to quantify
    the value of Page Awareness, Segment Awareness, Cache State, and Stride History.
    """
    page_feats   = [f for f in config.PAGE_FEATURES    if f in feat_cols]
    seg_feats    = [f for f in config.SEGMENT_FEATURES if f in feat_cols]
    cache_feats  = [f for f in config.CACHE_FEATURES   if f in feat_cols]
    stride_feats = [f for f in config.STRIDE_FEATURES  if f in feat_cols]

    configs_ab = {
        "Full Model":          feat_cols,
        "No Page Features":    [f for f in feat_cols if f not in page_feats],
        "No Segment Features": [f for f in feat_cols if f not in seg_feats],
        "No Cache Features":   [f for f in feat_cols if f not in cache_feats],
        "No Stride History":   [f for f in feat_cols if f not in stride_feats],
        "Page+Seg Only":       page_feats + seg_feats,
        "Page+Cache Only":     page_feats + cache_feats,
        "Seg+Cache Only":      seg_feats  + cache_feats,
    }

    cw_dict = dict(enumerate(compute_class_weight(
        "balanced", classes=np.arange(config.NUM_CLASSES), y=y_tr)))

    records = []
    print("\n[Ablation Study] Evaluating feature group contributions …")

    for name, cols in configs_ab.items():
        if len(cols) == 0:
            continue
        fi = [feat_cols.index(c) for c in cols]
        Xt = X_tr_raw[:, fi]
        Xv = X_va_raw[:, fi]

        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)

        t0 = time.time()
        # Fast, deterministic RF classifier for ablation comparison
        clf = RandomForestClassifier(
            n_estimators=40, max_depth=12, min_samples_leaf=2,
            class_weight=cw_dict, n_jobs=-1, random_state=config.RANDOM_SEED
        )
        clf.fit(Xt, y_tr)
        dur = time.time() - t0

        val_acc = float((clf.predict(Xv) == y_va).mean())
        print(f"  {name:25s}: Val Acc = {val_acc*100:.2f}%  ({len(cols):2d} feats, {dur:.1f}s)")
        records.append({
            "configuration": name,
            "num_features":  len(cols),
            "val_accuracy":  round(val_acc * 100, 2),
            "train_time_s":  round(dur, 2)
        })

    return pd.DataFrame(records).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
