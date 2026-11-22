"""MFC-3: Singular Value Decomposition of centred TRAIN features,  Xc = U Σ Vᵀ.

Relations verified numerically:
  λ_i = σ_i² / (n − 1)                          (covariance eigenvalues from singular values)
  V's columns = PCA eigenvectors (up to sign)    (Xcᵀ Xc = V Σ² Vᵀ)
  ‖Xc − U_k Σ_k V_kᵀ‖_F = sqrt(Σ_{i>k} σ_i²)    (Eckart–Young: best rank-k approximation)
  ‖Xc‖_F = sqrt(Σ σ_i²),  ‖Xc‖_2 = σ_1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation import plot_style as ps


def svd_analysis(X: np.ndarray, eig_lambda: np.ndarray, eig_V: np.ndarray, ks: list[int]) -> dict:
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    tol = S.max() * max(Xc.shape) * np.finfo(float).eps
    rank = int((S > tol).sum())

    recon = {}
    for k in ks:
        err = np.linalg.norm(Xc - (U[:, :k] * S[:k]) @ Vt[:k], "fro")
        recon[k] = {"error_fro": float(err), "eckart_young": float(np.sqrt((S[k:] ** 2).sum())),
                    "relative": float(err / np.linalg.norm(Xc, "fro"))}

    m = min(len(S), 32)
    v_cos = np.abs(np.sum(Vt[:m] * eig_V[:, :m].T, axis=1))
    out = {
        "n_singular_values": len(S), "numerical_rank": rank,
        "sigma_max": float(S[0]), "sigma_min_nonzero": float(S[rank - 1]),
        "reconstruction": recon,
        "checks": {
            "λ_i = σ_i²/(n−1)": bool(np.allclose(S**2 / (n - 1), eig_lambda[: len(S)], atol=1e-8)),
            f"V = PCA eigenvectors up to sign (top {m})": bool(np.allclose(v_cos, 1, atol=1e-6)),
            "U Σ Vᵀ reconstructs Xc": bool(np.allclose((U * S) @ Vt, Xc, atol=1e-8)),
            "UᵀU = I and VᵀV = I": bool(np.allclose(U.T @ U, np.eye(len(S)), atol=1e-8) and np.allclose(Vt @ Vt.T, np.eye(len(S)), atol=1e-8)),
            "Eckart–Young error matches for all k": all(np.isclose(r["error_fro"], r["eckart_young"]) for r in recon.values()),
            "‖Xc‖_F = sqrt(Σσ²), ‖Xc‖_2 = σ_1": bool(np.isclose(np.linalg.norm(Xc, "fro"), np.sqrt((S**2).sum())) and np.isclose(np.linalg.norm(Xc, 2), S[0])),
            "rank(Xc) = n−1": rank == n - 1,
        },
    }
    return out | {"_S": S}


def plot_spectrum(S: np.ndarray, rank: int, path: Path) -> None:
    """Two panels (not a dual axis): singular values (log) and cumulative energy Σσ²."""
    plt = ps.plt
    x = np.arange(1, len(S) + 1)
    energy = np.cumsum(S**2) / (S**2).sum() * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    a1.semilogy(x[:rank], S[:rank], color=ps.S1)
    a1.set_title("Singular values σᵢ of centred train matrix", loc="left")
    a1.set_xlabel("index i"); a1.set_ylabel("σᵢ (log scale)")
    a1.text(0.98, 0.95, f"rank = {rank} (n−1)\nσ₁ = {S[0]:.1f}\nσ_{rank} = {S[rank-1]:.2f}",
            transform=a1.transAxes, ha="right", va="top", fontsize=8, color=ps.TEXT_2)
    a2.plot(x, energy, color=ps.S1)
    a2.set_ylim(0, 102)
    a2.set_title("Cumulative energy Σσᵢ² (= explained variance)", loc="left")
    a2.set_xlabel("rank-k approximation"); a2.set_ylabel("% of ‖Xc‖²_F")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
