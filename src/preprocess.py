"""
Preprocessing utilities.
Scaling is fitted on training data only (no data leakage).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def preprocess(X_train, X_val, X_test, save_scaler=True):
    """
    Fit StandardScaler on X_train, transform all three splits.
    Returns (X_tr, X_va, X_te, scaler).
    """
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_va   = scaler.transform(X_val)
    X_te   = scaler.transform(X_test)
    if save_scaler:
        joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
    return X_tr, X_va, X_te, scaler


def handle_missing(df: pd.DataFrame, strategy="mean") -> pd.DataFrame:
    """Fill NaN values using the given strategy."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == "zero":
        df[num_cols] = df[num_cols].fillna(0)
    return df
