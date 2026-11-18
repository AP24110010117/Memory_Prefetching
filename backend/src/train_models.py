"""
train_models.py — Model Training, Comparison & Focused Compact MLP Tuning.

Models trained and compared:
  1. Decision Tree
  2. Random Forest
  3. Gradient Boosting / LightGBM
  4. Compact Deep MLP (32–16) (Focused tuning with regularization & early stopping)
  5. Next-Line Baseline (heuristic)

Workflow:
  - StandardScaler fitted on TRAIN set only (zero data leakage).
  - Focused hyperparameter tuning for Compact MLP evaluated on VALIDATION set only.
  - Best model selected strictly using VALIDATION accuracy.
  - Models saved to models/ directory.
"""

import os, sys, time, joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.mlp_model import PrefetchMLP


def load_splits():
    """Load train/val/test CSVs. Returns (df_train, df_val, df_test)."""
    df_train = pd.read_csv(config.DATA_DIR / "train.csv")
    df_val   = pd.read_csv(config.DATA_DIR / "validation.csv")
    df_test  = pd.read_csv(config.DATA_DIR / "test.csv")
    return df_train, df_val, df_test


def get_XY(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].values.astype(float)
    y = df["target_class"].values.astype(int)
    return X, y


def _class_weights(y_tr):
    classes = np.arange(config.NUM_CLASSES)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    return dict(enumerate(cw))


class NextLineBaseline:
    """Standard sequential prefetching baseline: always predicts stride +1 (class 0)."""
    def predict(self, X): return np.zeros(len(X), dtype=int)
    def predict_proba(self, X):
        p = np.zeros((len(X), config.NUM_CLASSES))
        p[:, 0] = 1.0
        return p


def train_all(feat_cols: list, verbose: bool = True):
    """
    Train and compare all models with focused tuning on Compact MLP (32–16).
    Returns (results, best_name, scaler, X_tr, y_tr, X_va, y_va, X_te, y_te).
    """
    df_tr, df_val, df_te = load_splits()

    X_tr_raw, y_tr = get_XY(df_tr,  feat_cols)
    X_va_raw, y_va = get_XY(df_val, feat_cols)
    X_te_raw, y_te = get_XY(df_te,  feat_cols)

    # Scaler fitted strictly on training data
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_va   = scaler.transform(X_va_raw)
    X_te   = scaler.transform(X_te_raw)

    joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
    cw = _class_weights(y_tr)
    results = {}

    # ── 1. Decision Tree ──────────────────────────────────────────────────────
    if verbose: print("\n[1/5] Training Decision Tree …")
    t0 = time.time()
    dt = DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=4,
        class_weight=cw, random_state=config.RANDOM_SEED
    )
    dt.fit(X_tr, y_tr)
    tr_s = time.time() - t0

    t0 = time.time()
    va_acc = (dt.predict(X_va) == y_va).mean()
    vt = (time.time() - t0) / len(y_va) * 1000
    te_acc = (dt.predict(X_te) == y_te).mean()
    joblib.dump(dt, config.MODELS_DIR / "decision_tree.pkl")
    results["Decision Tree"] = dict(model=dt, val_acc=va_acc, test_acc=te_acc,
                                    val_time_ms=vt, train_time_s=tr_s)
    if verbose: print(f"  Val Acc={va_acc*100:.2f}%  Test Acc={te_acc*100:.2f}%  Train={tr_s:.1f}s")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    if verbose: print("\n[2/5] Training Random Forest …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=18, min_samples_leaf=2,
        class_weight=cw, n_jobs=-1, random_state=config.RANDOM_SEED
    )
    rf.fit(X_tr, y_tr)
    tr_s = time.time() - t0

    t0 = time.time()
    va_acc = (rf.predict(X_va) == y_va).mean()
    vt = (time.time() - t0) / len(y_va) * 1000
    te_acc = (rf.predict(X_te) == y_te).mean()
    joblib.dump(rf, config.MODELS_DIR / "random_forest.pkl")
    results["Random Forest"] = dict(model=rf, val_acc=va_acc, test_acc=te_acc,
                                    val_time_ms=vt, train_time_s=tr_s)
    if verbose: print(f"  Val Acc={va_acc*100:.2f}%  Test Acc={te_acc*100:.2f}%  Train={tr_s:.1f}s")

    # ── 3. Gradient Boosting ──────────────────────────────────────────────────
    if verbose: print("\n[3/5] Training Gradient Boosting …")
    t0 = time.time()
    try:
        import lightgbm as lgb
        gb = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=8, num_leaves=31,
            class_weight=cw, n_jobs=-1, random_state=config.RANDOM_SEED, verbose=-1
        )
    except ImportError:
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=5, random_state=config.RANDOM_SEED
        )
    gb.fit(X_tr, y_tr)
    tr_s = time.time() - t0

    t0 = time.time()
    va_acc = (gb.predict(X_va) == y_va).mean()
    vt = (time.time() - t0) / len(y_va) * 1000
    te_acc = (gb.predict(X_te) == y_te).mean()
    joblib.dump(gb, config.MODELS_DIR / "gradient_boosting.pkl")
    results["Gradient Boosting"] = dict(model=gb, val_acc=va_acc, test_acc=te_acc,
                                        val_time_ms=vt, train_time_s=tr_s)
    if verbose: print(f"  Val Acc={va_acc*100:.2f}%  Test Acc={te_acc*100:.2f}%  Train={tr_s:.1f}s")

    # ── 4. Compact Deep MLP (32–16) with Focused Tuning ───────────────────────
    if verbose: print("\n[4/5] Training & Tuning Compact Deep MLP (32–16) …")
    cw_arr = np.array([cw.get(i, 1.0) for i in range(config.NUM_CLASSES)])
    cw_arr = np.clip(cw_arr / cw_arr.mean(), 0.70, 1.90)

    # Grid search candidate configurations evaluated strictly on VALIDATION ACCURACY
    tuning_candidates = [
        {"lr": 0.0035, "l2": 2.0e-4, "dropout": 0.06},
        {"lr": 0.0040, "l2": 1.5e-4, "dropout": 0.05},
        {"lr": 0.0030, "l2": 2.5e-4, "dropout": 0.07},
    ]

    best_mlp = None
    best_mlp_val_acc = -1.0
    best_mlp_hist = None
    best_cfg = None

    t_mlp_start = time.time()
    for idx, cfg in enumerate(tuning_candidates, 1):
        if verbose:
            print(f"  Tuning Candidate {idx}/{len(tuning_candidates)}: lr={cfg['lr']}, l2={cfg['l2']}, dropout={cfg['dropout']}")
        candidate_mlp = PrefetchMLP(
            in_dim=X_tr.shape[1],
            n_classes=config.NUM_CLASSES,
            l2=cfg["l2"],
            dropout=cfg["dropout"],
            class_weights=cw_arr,
            seed=config.RANDOM_SEED
        )
        hist = candidate_mlp.fit(
            X_tr, y_tr, X_va, y_va,
            epochs=config.MLP_EPOCHS,
            batch=config.MLP_BATCH_SIZE,
            lr=cfg["lr"],
            patience=config.MLP_PATIENCE,
            verbose=False
        )
        val_acc_candidate = (candidate_mlp.predict(X_va) == y_va).mean()
        if verbose:
            print(f"    Candidate {idx} Validation Accuracy: {val_acc_candidate*100:.2f}%")

        if val_acc_candidate > best_mlp_val_acc:
            best_mlp_val_acc = val_acc_candidate
            best_mlp = candidate_mlp
            best_mlp_hist = hist
            best_cfg = cfg

    tr_s = time.time() - t_mlp_start
    t0 = time.time()
    va_acc = (best_mlp.predict(X_va) == y_va).mean()
    vt = (time.time() - t0) / len(y_va) * 1000
    te_acc = (best_mlp.predict(X_te) == y_te).mean()

    joblib.dump(best_mlp, config.MODELS_DIR / "mlp_model.pkl")
    pd.DataFrame(best_mlp_hist).to_csv(config.DATA_DIR / "mlp_training_history.csv", index=False)

    n_params, size_kb = best_mlp.param_count()
    results["Compact MLP (32-16)"] = dict(
        model=best_mlp, val_acc=va_acc, test_acc=te_acc,
        val_time_ms=vt, train_time_s=tr_s,
        history=best_mlp_hist, n_params=n_params, size_kb=size_kb,
        best_cfg=best_cfg
    )
    if verbose:
        print(f"  [Best Compact MLP Configuration]: {best_cfg}")
        print(f"  Val Acc={va_acc*100:.2f}%  Test Acc={te_acc*100:.2f}%  "
              f"Params={n_params:,}  Size={size_kb:.2f} KB (Hardware-Optimal)")

    # ── 5. Next-Line Baseline ─────────────────────────────────────────────────
    if verbose: print("\n[5/5] Next-Line Baseline (Heuristic Stride +1) …")
    nl = NextLineBaseline()
    va_acc = (nl.predict(X_va) == y_va).mean()
    te_acc = (nl.predict(X_te) == y_te).mean()
    results["Next-Line Baseline"] = dict(model=nl, val_acc=va_acc, test_acc=te_acc,
                                         val_time_ms=0.0, train_time_s=0.0)
    if verbose: print(f"  Val Acc={va_acc*100:.2f}%  Test Acc={te_acc*100:.2f}%")

    # ── Select Best Model strictly by VALIDATION accuracy ─────────────────────
    best_name = max(
        (k for k in results if k != "Next-Line Baseline"),
        key=lambda k: results[k]["val_acc"]
    )
    best_model = results[best_name]["model"]
    joblib.dump(best_model, config.MODELS_DIR / "best_model.pkl")

    if verbose:
        print("\n" + "="*70)
        print(f"  BEST MODEL SELECTED (by Validation Accuracy): {best_name}")
        print(f"  Validation Accuracy: {results[best_name]['val_acc']*100:.2f}%")
        print(f"  Saved to models/best_model.pkl")
        print("="*70)

    # Save model comparison summary table
    rows = []
    for mname, mdata in results.items():
        rows.append({
            "Model": mname,
            "Validation Accuracy (%)": round(mdata["val_acc"] * 100, 2),
            "Test Accuracy (%)": round(mdata["test_acc"] * 100, 2),
            "Inference Latency (ms/sample)": round(mdata["val_time_ms"], 4),
            "Training Time (s)": round(mdata["train_time_s"], 2)
        })
    df_models = pd.DataFrame(rows).sort_values("Validation Accuracy (%)", ascending=False)
    df_models.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)

    return results, best_name, scaler, X_tr, y_tr, X_va, y_va, X_te, y_te
