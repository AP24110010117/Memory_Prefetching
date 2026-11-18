"""Leakage-safe next-block Random Forest predictor."""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "prev1", "prev2", "prev3", "delta1", "delta2",
    "recent_unique", "recent_mean", "recent_std",
    "page_id", "page_number", "page_size",
    "segment_id", "segment_type_code", "segment_offset"
]


def build_supervised_rows(data, history=3):
    """For each position t, use only blocks/metadata at positions <= t-1."""
    rows = []
    for _, trace in data.groupby("trace_id", sort=False):
        trace = trace.sort_values("step").reset_index(drop=True)
        blocks = trace["block"].astype(int).tolist()
        for t in range(history, len(blocks)):
            past = blocks[max(0, t-history):t]
            current = blocks[t-1]
            # Metadata is associated with the current/last observed access.
            meta = trace.iloc[t-1]
            recent = past[-min(8, len(past)):]
            rows.append({
                "prev1": past[-1], "prev2": past[-2], "prev3": past[-3],
                "delta1": past[-1] - past[-2],
                "delta2": past[-2] - past[-3],
                "recent_unique": len(set(recent)),
                "recent_mean": float(np.mean(recent)),
                "recent_std": float(np.std(recent)),
                "page_id": int(meta.get("page_id", 0)),
                "page_number": int(meta.get("page_number", meta.get("page_id", 0))),
                "page_size": int(meta.get("page_size", 0)),
                "segment_id": int(meta.get("segment_id", 0)),
                "segment_type_code": {"Code": 0, "Data": 1, "Stack": 2}.get(
                    str(meta.get("segment_type", "Unknown")), 3),
                "segment_offset": int(meta.get("segment_offset", 0)),
                "target": blocks[t]
            })
    return pd.DataFrame(rows)


class RandomForestPredictor:
    def __init__(self, n_estimators=160, random_state=42, max_depth=14):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            max_depth=max_depth, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1
        )
        self.classes_seen = None
        self.is_trained = False
        self.history = 3

    def fit(self, train_data):
        supervised = build_supervised_rows(train_data, self.history)
        if supervised.empty or supervised["target"].nunique() < 2:
            raise ValueError("Training data must contain at least two target block classes")
        self.model.fit(supervised[FEATURE_COLUMNS], supervised["target"].astype(int))
        self.classes_seen = self.model.classes_
        self.is_trained = True
        return self

    def features_from_history(self, history, metadata=None):
        if len(history) < 3:
            raise ValueError("At least three past accesses are required")
        h = [int(x) for x in history[-3:]]
        recent = [int(x) for x in history[-8:]]
        metadata = metadata or {}
        return pd.DataFrame([{
            "prev1": h[-1], "prev2": h[-2], "prev3": h[-3],
            "delta1": h[-1] - h[-2], "delta2": h[-2] - h[-3],
            "recent_unique": len(set(recent)),
            "recent_mean": float(np.mean(recent)),
            "recent_std": float(np.std(recent)),
            "page_id": int(metadata.get("page_id", 0)),
            "page_number": int(metadata.get("page_number", metadata.get("page_id", 0))),
            "page_size": int(metadata.get("page_size", 0)),
            "segment_id": int(metadata.get("segment_id", 0)),
            "segment_type_code": {"Code": 0, "Data": 1, "Stack": 2}.get(
                str(metadata.get("segment_type", "Unknown")), 3),
            "segment_offset": int(metadata.get("segment_offset", 0))
        }], columns=FEATURE_COLUMNS)

    def predict_next(self, history, metadata=None):
        if not self.is_trained:
            raise RuntimeError("Call fit() before predict_next()")
        x = self.features_from_history(history, metadata)
        probabilities = self.model.predict_proba(x)[0]
        index = int(np.argmax(probabilities))
        return int(self.model.classes_[index]), float(probabilities[index])

    def evaluate(self, data):
        supervised = build_supervised_rows(data, self.history)
        if supervised.empty:
            return {"accuracy": 0.0, "precision_weighted": 0.0,
                    "recall_weighted": 0.0, "f1_weighted": 0.0, "samples": 0}
        pred = self.model.predict(supervised[FEATURE_COLUMNS])
        y = supervised["target"].astype(int)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "precision_weighted": float(precision_score(y, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "samples": int(len(y))
        }
