"""
Common utility functions.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def ensure_dirs():
    """Create all project output directories."""
    for d in [config.RAW_DIR, config.PROCESSED_DIR,
              config.MODELS_DIR, config.GRAPHS_DIR]:
        os.makedirs(d, exist_ok=True)


def class_distribution(y, n_classes=config.NUM_CLASSES):
    """Return array of counts per class."""
    return np.bincount(y, minlength=n_classes)


def top_k_accuracy(probs, y, k=3):
    """Top-k accuracy."""
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y[i] in top_k[i] for i in range(len(y))])
