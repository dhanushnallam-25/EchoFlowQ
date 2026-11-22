"""MFC-3: sample covariance  C = (1/(n-1)) Xcᵀ Xc  on centred TRAIN features, plus norms & correlations.

Links to PCA: C is symmetric positive semi-definite, so it has real eigenvalues λ_i ≥ 0 and an
orthonormal eigenbasis (spectral theorem) — exactly the principal axes used in pca.py. Hence
  tr(C)   = Σ λ_i   (total variance)
  ‖C‖_F   = sqrt(Σ λ_i²)
  ‖C‖_2   = λ_max
  rank(C) ≤ min(d, n-1)   (n samples lose one dof to centring)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation import plot_style as ps


def covariance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (C, mean). Explicit formula, no np.cov."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    return Xc.T @ Xc / (X.shape[0] - 1), mu


def correlation_from_cov(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def analyze(X: np.ndarray, name: str) -> dict:
    n, dim = X.shape
    C, _ = covariance(X)
    lam = np.linalg.eigvalsh(C)[::-1]
    R = correlation_from_cov(C)
    off = np.abs(R[~np.eye(dim, dtype=bool)])
    iu = np.triu_indices(dim, 1)
    top = np.argsort(-np.abs(R[iu]))[:5]
    tol = lam.max() * dim * np.finfo(float).eps

    out = {
        "name": name, "n": n, "d": dim,
        "trace": float(np.trace(C)), "frobenius": float(np.linalg.norm(C, "fro")),
        "spectral": float(np.linalg.norm(C, 2)), "rank": int((lam > tol).sum()),
        "min_eig": float(lam.min()), "condition_number_nonzero": float(lam[0] / lam[lam > tol][-1]),
        "mean_abs_offdiag_corr": float(off.mean()), "frac_abs_corr_gt_0.5": float((off > 0.5).mean()),
        "top_corr_pairs": [(int(iu[0][t]), int(iu[1][t]), float(R[iu][t])) for t in top],
        "X_frobenius": float(np.linalg.norm(X - X.mean(0), "fro")),
        "X_spectral": float(np.linalg.norm(X - X.mean(0), 2)),
    }
    out["checks"] = {
        "matches np.cov": bool(np.allclose(C, np.cov(X, rowvar=False), atol=1e-10)),
        "symmetric": bool(np.allclose(C, C.T)),
        "PSD (min λ ≥ -tol)": bool(lam.min() > -1e-8),
        "tr(C) = Σλ": bool(np.isclose(out["trace"], lam.sum())),
        "‖C‖_F = sqrt(Σλ²)": bool(np.isclose(out["frobenius"], np.sqrt((lam**2).sum()))),
        "‖C‖_2 = λ_max": bool(np.isclose(out["spectral"], lam[0])),
        "rank ≤ n-1": out["rank"] <= n - 1,
    }
    return out | {"_C": C}


def plot_heatmap(C: np.ndarray, path: Path, title: str) -> None:
    plt = ps.plt
    v = np.abs(C).max()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(C, cmap=ps.DIVERGING, vmin=-v, vmax=v, interpolation="nearest")
    ax.grid(False)
    ax.set_title(title, loc="left")
    ax.set_xlabel("feature index"); ax.set_ylabel("feature index")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
    cb.set_label("covariance", color=ps.TEXT_2)
    fig.savefig(path); plt.close(fig)
