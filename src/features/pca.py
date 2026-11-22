"""MFC-3: PCA from first principles via eigendecomposition of the covariance matrix.

  C v_i = λ_i v_i     (np.linalg.eigh — C is symmetric, so eigh gives real λ, orthonormal V)
  W_k  = [v_1 … v_k]  (top-k eigenvectors, λ_1 ≥ λ_2 ≥ …)
  Z    = (X - μ_train) W_k
  explained variance ratio_i = λ_i / Σ_j λ_j

Fit on TRAIN only; val/test/unknown are projected with the train μ and W_k (no leakage).
Sign convention: each eigenvector is flipped so its largest-|.| entry is positive (deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation import plot_style as ps
from src.features.covariance import covariance


@dataclass
class EigPCA:
    n_components: int | None = None
    mean_: np.ndarray = field(default=None, repr=False)
    eigenvalues_: np.ndarray = field(default=None, repr=False)
    eigenvectors_: np.ndarray = field(default=None, repr=False)  # columns = v_i, sorted by λ desc

    def fit(self, X: np.ndarray) -> "EigPCA":
        C, self.mean_ = covariance(X)
        lam, V = np.linalg.eigh(C)                 # ascending
        order = np.argsort(lam)[::-1]
        lam, V = np.clip(lam[order], 0, None), V[:, order]
        idx = np.abs(V).argmax(axis=0)
        V *= np.sign(V[idx, np.arange(V.shape[1])])
        self.eigenvalues_, self.eigenvectors_ = lam, V
        return self

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.eigenvalues_ / self.eigenvalues_.sum()

    def W(self, k: int | None = None) -> np.ndarray:
        return self.eigenvectors_[:, : (k or self.n_components)]

    def transform(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        return ((np.asarray(X, np.float64) - self.mean_) @ self.W(k)).astype(np.float32)


def check_against_sklearn(pca: EigPCA, X: np.ndarray, k: int = 32) -> dict:
    from sklearn.decomposition import PCA

    sk = PCA(n_components=k, svd_solver="full").fit(X)
    cos = np.abs(np.sum(pca.W(k).T * sk.components_, axis=1))  # |<v_i, v_i^sk>| should be 1
    Z_ours, Z_sk = pca.transform(X, k), sk.transform(X)
    return {
        f"eigenvalues = sklearn explained_variance_ (top {k})": bool(np.allclose(pca.eigenvalues_[:k], sk.explained_variance_, rtol=1e-6)),
        f"|<v_i, v_i_sklearn>| = 1 (min {cos.min():.8f})": bool(np.allclose(cos, 1, atol=1e-6)),
        "Z equal up to column sign": bool(np.allclose(np.abs(Z_ours), np.abs(Z_sk), atol=1e-4)),
        "Wᵀ W = I (orthonormal)": bool(np.allclose(pca.W(k).T @ pca.W(k), np.eye(k), atol=1e-10)),
        "Z_train columns uncorrelated, var = λ": bool(np.allclose(np.cov(Z_ours.astype(np.float64), rowvar=False), np.diag(pca.eigenvalues_[:k]), atol=1e-3)),
    }


def plot_explained_variance(pca: EigPCA, ks: list[int], path: Path, rank: int) -> dict[int, float]:
    plt = ps.plt
    cum = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, cum * 100, color=ps.S1, lw=2)
    marks = {k: float(cum[k - 1]) for k in ks}
    for k, v in marks.items():
        ax.plot(k, v * 100, "o", ms=6, color=ps.S1, mec=ps.SURFACE, mew=1.5, zorder=3)
        ax.annotate(f"k={k}: {v*100:.1f}%", (k, v * 100), xytext=(6, -12 if k > 8 else 4),
                    textcoords="offset points", fontsize=8, color=ps.TEXT)
    ax.axvline(rank, color=ps.MUTED, lw=1, ls="--")
    ax.text(rank, 8, f" rank(C)={rank}\n (n_train−1)", fontsize=8, color=ps.TEXT_2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, len(cum)); ax.set_ylim(0, 102)
    ax.set_xlabel("number of principal components k (log₂ scale)")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title("ECAPA-TDNN (standardised, train): cumulative explained variance", loc="left")
    fig.savefig(path); plt.close(fig)
    return marks


def plot_2d_speakers(Z: dict[str, np.ndarray], spk: dict[str, np.ndarray], labels: list[str],
                     evr: np.ndarray, path: Path) -> None:
    """Small multiples: each panel highlights one speaker (train=blue, test=orange) over gray context;
    the last panel highlights held-out unknown speakers (aqua)."""
    plt = ps.plt
    ctx = np.vstack([Z["train"], Z["test"]])
    allpts = np.vstack([ctx, Z["unknown"]])
    pad = 0.05 * (allpts.max(0) - allpts.min(0))
    lo, hi = allpts.min(0) - pad, allpts.max(0) + pad
    panels = labels + ["unknown"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
    for ax, p in zip(axes.flat, panels + [None]):
        if p is None:
            ax.axis("off")
            for c, m, t in ((ps.S1, "o", "train (enrolled)"), (ps.S2, "^", "test (enrolled)"), (ps.S3, "s", "unknown speakers")):
                ax.scatter([], [], c=c, marker=m, s=30, label=t)
            ax.scatter([], [], c=ps.MUTED, s=12, label="all other enrolled points")
            ax.legend(loc="center", fontsize=9)
            continue
        ax.scatter(ctx[:, 0], ctx[:, 1], s=8, c=ps.MUTED, alpha=0.6, lw=0)
        if p == "unknown":
            u = Z["unknown"]
            ax.scatter(u[:, 0], u[:, 1], s=26, c=ps.S3, marker="s", edgecolors=ps.SURFACE, lw=0.8)
        else:
            for split, c, m in (("train", ps.S1, "o"), ("test", ps.S2, "^")):
                pts = Z[split][spk[split] == p]
                ax.scatter(pts[:, 0], pts[:, 1], s=28, c=c, marker=m, edgecolors=ps.SURFACE, lw=0.8)
        ax.set_title(p, loc="left", fontsize=10)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    for ax in axes[-1]:
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    fig.suptitle("ECAPA-TDNN projected on the first two principal components (PCA fit on train only)",
                 x=0.01, ha="left", fontsize=12, color=ps.TEXT)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
