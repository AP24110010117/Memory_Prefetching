"""
mlp_model.py — Compact Deep MLP (32–16) for On-Chip Memory Prefetch Prediction.

Architecture:
  Input (D) → Dense(32) → ReLU → Dropout(0.06) →
              Dense(16) → ReLU → Dropout(0.06) →
              Dense(10) → Softmax

Hardware Profile:
  - Ultra-compact: ~1,560 parameters (~6 KB in float32).
  - Designed specifically for on-chip hardware SRAM prefetch engine.
  - Class-weighted cross-entropy loss with L2 weight decay regularisation.
  - Early stopping with validation accuracy checkpointing.
"""

import numpy as np


class _Dense:
    def __init__(self, fan_in: int, fan_out: int, l2: float = 2.0e-4):
        # He (Kaiming) initialisation
        s = np.sqrt(2.0 / fan_in)
        self.W  = np.random.randn(fan_in, fan_out) * s
        self.b  = np.zeros((1, fan_out))
        self.l2 = l2
        self.dW = self.db = None
        # Adam first and second moments
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        m = self._X.shape[0]
        self.dW = self._X.T @ dZ / m + self.l2 * self.W
        self.db = dZ.sum(0, keepdims=True) / m
        return dZ @ self.W.T

    def adam_step(self, lr: float, b1: float, b2: float, eps: float, t: int):
        for attr_w, attr_m, attr_v in [("W", "mW", "vW"), ("b", "mb", "vb")]:
            g  = getattr(self, "d" + attr_w)
            m_ = b1 * getattr(self, attr_m) + (1.0 - b1) * g
            v_ = b2 * getattr(self, attr_v) + (1.0 - b2) * g**2
            setattr(self, attr_m, m_)
            setattr(self, attr_v, v_)
            m_hat = m_ / (1.0 - b1**t)
            v_hat = v_ / (1.0 - b2**t)
            new_w = getattr(self, attr_w) - lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, attr_w, new_w)


class _Dropout:
    def __init__(self, p: float):
        self.p = p
        self.mask = None

    def forward(self, X: np.ndarray, train: bool) -> np.ndarray:
        if train and self.p > 0:
            self.mask = (np.random.rand(*X.shape) >= self.p) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        return dZ * self.mask if self.mask is not None else dZ


class PrefetchMLP:
    """
    Compact MLP (32–16) for memory prefetch stride prediction.
    Pure NumPy implementation for maximum transparency, speed, and reproducibility.
    """

    def __init__(self, in_dim: int, n_classes: int = 10, l2: float = 2.0e-4,
                 dropout: float = 0.06, class_weights: np.ndarray = None, seed: int = 42):
        np.random.seed(seed)
        self.n_classes     = n_classes
        self.class_weights = class_weights if class_weights is not None else np.ones(n_classes)
        self.l2            = l2
        self.dropout       = dropout

        # Compact (32–16) Layer Stack
        self.l1 = _Dense(in_dim, 32, l2)
        self.d1 = _Dropout(dropout)
        self.l2_layer = _Dense(32, 16, l2)
        self.d2 = _Dropout(dropout)
        self.l3 = _Dense(16, n_classes, l2)
        self._layers = [self.l1, self.l2_layer, self.l3]

    @staticmethod
    def _relu(z): return np.maximum(0, z)
    @staticmethod
    def _relu_d(z): return (z > 0).astype(float)
    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def _forward(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        self._z1 = self.l1.forward(X)
        self._a1 = self._relu(self._z1)
        self._a1d = self.d1.forward(self._a1, train)

        self._z2 = self.l2_layer.forward(self._a1d)
        self._a2 = self._relu(self._z2)
        self._a2d = self.d2.forward(self._a2, train)

        self._z3 = self.l3.forward(self._a2d)
        self._probs = self._softmax(self._z3)
        return self._probs

    def _loss(self, probs: np.ndarray, y_oh: np.ndarray) -> float:
        eps = 1e-12
        p   = np.clip(probs, eps, 1.0 - eps)
        w   = y_oh @ self.class_weights
        ce  = -(y_oh * np.log(p)).sum(1) * w
        l2_penalty = 0.5 * sum(np.sum(ly.W**2) * ly.l2 for ly in self._layers)
        return float(ce.mean() + l2_penalty)

    def _backward(self, y_oh: np.ndarray):
        w   = (y_oh @ self.class_weights)[:, None]
        dZ3 = (self._probs - y_oh) * w

        dA2d = self.l3.backward(dZ3)
        dA2  = self.d2.backward(dA2d)
        dZ2  = dA2 * self._relu_d(self._z2)

        dA1d = self.l2_layer.backward(dZ2)
        dA1  = self.d1.backward(dA1d)
        dZ1  = dA1 * self._relu_d(self._z1)

        self.l1.backward(dZ1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X, train=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        m    = len(y)
        y_oh = np.zeros((m, self.n_classes))
        y_oh[np.arange(m), y] = 1
        p    = self.predict_proba(X)
        loss = self._loss(p, y_oh)
        acc  = float((p.argmax(1) == y).mean())
        return loss, acc

    def param_count(self):
        total = sum(ly.W.size + ly.b.size for ly in self._layers)
        return total, total * 4 / 1024  # Size in KB (float32)

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 45, batch: int = 128, lr: float = 0.0035,
            patience: int = 10, verbose: bool = True):
        """
        Mini-batch Adam with early stopping based on Validation Accuracy.
        Returns history dict with metrics per epoch.
        """
        n, k = len(y_tr), self.n_classes
        y_oh_tr = np.zeros((n, k))
        y_oh_tr[np.arange(n), y_tr] = 1

        history = {"epoch": [], "train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": []}

        best_val_acc = 0.0
        best_val_loss = np.inf
        wait = 0
        best_weights = None
        step = 0

        b1, b2, eps = 0.9, 0.999, 1e-8

        for ep in range(1, epochs + 1):
            # Mini-batch shuffle
            perm = np.random.permutation(n)
            X_s, y_s = X_tr[perm], y_oh_tr[perm]

            for i in range(0, n, batch):
                xb = X_s[i:i + batch]
                yb = y_s[i:i + batch]
                step += 1
                self._forward(xb, train=True)
                self._backward(yb)
                for ly in self._layers:
                    ly.adam_step(lr, b1, b2, eps, step)

            # Epoch evaluation
            tr_loss, tr_acc = self.evaluate(X_tr, y_tr)
            va_loss, va_acc = self.evaluate(X_val, y_val)

            history["epoch"].append(ep)
            history["train_loss"].append(round(tr_loss, 4))
            history["val_loss"].append(round(va_loss, 4))
            history["train_acc"].append(round(tr_acc, 4))
            history["val_acc"].append(round(va_acc, 4))

            if verbose and (ep % 5 == 0 or ep == 1 or ep == epochs):
                print(f"  Epoch {ep:2d}/{epochs:2d} | "
                      f"Train Loss={tr_loss:.4f} Acc={tr_acc*100:.2f}% | "
                      f"Val Loss={va_loss:.4f} Acc={va_acc*100:.2f}%")

            # Checkpoint best weights strictly on validation accuracy
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_val_loss = va_loss
                best_weights = [(ly.W.copy(), ly.b.copy()) for ly in self._layers]
                wait = 0
            else:
                wait += 1
                if wait >= patience and ep >= 20:
                    if verbose:
                        print(f"  Early stopping triggered at epoch {ep} (Best Val Acc: {best_val_acc*100:.2f}%)")
                    break

        # Restore best model weights
        if best_weights is not None:
            for ly, (w, b) in zip(self._layers, best_weights):
                ly.W = w.copy()
                ly.b = b.copy()

        return history
