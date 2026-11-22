"""Z-score standardisation  X_std = (X - μ) / σ, with μ, σ estimated on the TRAIN split only.

Fitting on train only means val/test/unknown statistics never influence the transform
(no leakage). Constant features (σ≈0) get σ=1 so they map to 0 instead of dividing by zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ZScoreScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)                 # population σ (matches sklearn StandardScaler)
        self.scale_ = np.where(sd < self.eps, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit() first")
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path: str | Path) -> "ZScoreScaler":
        d = np.load(path)
        return cls(mean_=d["mean"], scale_=d["scale"])
