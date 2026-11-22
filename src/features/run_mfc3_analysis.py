"""Stage 5 runner: covariance -> PCA (eigendecomposition) -> SVD -> norms, on standardised ECAPA (train).

Saves figures to results/figures/, PCA-reduced datasets to data/processed/features/ecapa_pca_k{k}.npz,
and results/metrics/mfc3_summary.json.

Usage (project root):  .venv\\Scripts\\python.exe -m src.features.run_mfc3_analysis
"""
from __future__ import annotations

import json

import numpy as np

from src.data.hf_utils import PROJECT_ROOT
from src.features.covariance import analyze, plot_heatmap
from src.features.pca import EigPCA, check_against_sklearn, plot_2d_speakers, plot_explained_variance
from src.features.svd import plot_spectrum, svd_analysis

FEAT = PROJECT_ROOT / "data/processed/features"
FIG = PROJECT_ROOT / "results/figures"
MET = PROJECT_ROOT / "results/metrics"
SPLITS = ("train", "val", "test", "unknown")
MARK_KS = [2, 4, 6, 8, 16, 32, 64]
SAVE_KS = [2, 4, 6, 8, 16, 32, 64]


def rel(p):
    return p.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True); MET.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(FEAT / "ecapa_std.npz"))
    Xtr = d["X_train"].astype(np.float64)
    n, dim = Xtr.shape
    print(f"ECAPA standardised train matrix X: {n} × {dim}  (n < d ⇒ covariance is rank-deficient)\n")

    # 1) Covariance --------------------------------------------------------------
    cov = analyze(Xtr, "ecapa_std_train")
    cov_mfcc = analyze(np.load(FEAT / "mfcc_std.npz")["X_train"].astype(np.float64), "mfcc_std_train")
    f_cov = FIG / "ecapa_covariance_heatmap.png"
    plot_heatmap(cov.pop("_C"), f_cov, f"ECAPA-TDNN covariance C = XᵀX/(n−1)  (standardised train, {dim}×{dim})")
    cov_mfcc.pop("_C")

    # 2) PCA ---------------------------------------------------------------------
    pca = EigPCA().fit(Xtr)
    pca_checks = check_against_sklearn(pca, Xtr, k=32)
    f_ev = FIG / "pca_explained_variance.png"
    marks = plot_explained_variance(pca, MARK_KS, f_ev, cov["rank"])
    Z2 = {s: pca.transform(d[f"X_{s}"], 2) for s in SPLITS}
    spk = {s: d[f"spk_{s}"] for s in SPLITS}
    f_2d = FIG / "pca_2d_speakers.png"
    plot_2d_speakers(Z2, spk, list(d["label_names"]), pca.explained_variance_ratio_, f_2d)

    leak_check = np.allclose(pca.mean_, EigPCA().fit(Xtr).mean_) and not np.allclose(
        pca.mean_, d["X_test"].mean(0), atol=1e-3)
    for k in SAVE_KS:
        out = {kk: v for kk, v in d.items() if not kk.startswith("X_") and not kk.startswith("scaler_")}
        out |= {f"X_{s}": pca.transform(d[f"X_{s}"], k) for s in SPLITS}
        out |= {"pca_mean": pca.mean_, "pca_components": pca.W(k).T, "pca_eigenvalues": pca.eigenvalues_[:k],
                "explained_variance_ratio": pca.explained_variance_ratio_[:k], "k": np.array(k)}
        np.savez_compressed(FEAT / f"ecapa_pca_k{k}.npz", **out)
    np.savez(FEAT / "ecapa_pca_model.npz", mean=pca.mean_, eigenvalues=pca.eigenvalues_, eigenvectors=pca.eigenvectors_)

    # 3) SVD ---------------------------------------------------------------------
    svd = svd_analysis(Xtr, pca.eigenvalues_, pca.eigenvectors_, MARK_KS)
    f_sv = FIG / "svd_spectrum.png"
    plot_spectrum(svd.pop("_S"), svd["numerical_rank"], f_sv)

    # ----------------------------------------------------------------------------- report
    print("=== Covariance (ECAPA std, train) ===")
    print(f"  tr(C) = {cov['trace']:.2f} (= total variance; d·n/(n−1) = {dim*n/(n-1):.2f} for z-scored data)")
    print(f"  ‖C‖_F = {cov['frobenius']:.2f}   ‖C‖_2 = λ_max = {cov['spectral']:.2f}   rank(C) = {cov['rank']} / {dim}")
    print(f"  mean |corr| off-diagonal = {cov['mean_abs_offdiag_corr']:.3f}   pairs with |r|>0.5: {cov['frac_abs_corr_gt_0.5']*100:.2f}%")
    print(f"  top correlated feature pairs (i, j, r): {[(i, j, round(r, 3)) for i, j, r in cov['top_corr_pairs'][:3]]}")
    print(f"  (MFCC for contrast: mean |corr| = {cov_mfcc['mean_abs_offdiag_corr']:.3f}, |r|>0.5: {cov_mfcc['frac_abs_corr_gt_0.5']*100:.2f}%, ‖C‖_2 = {cov_mfcc['spectral']:.2f})")
    print("\n=== PCA: variance retained at k ===")
    print("  " + "  ".join(f"k={k}: {v*100:.1f}%" for k, v in marks.items()))
    print("\n=== SVD: rank-k reconstruction error ‖Xc − U_kΣ_kV_kᵀ‖_F / ‖Xc‖_F ===")
    print("  " + "  ".join(f"k={k}: {r['relative']*100:.1f}%" for k, r in svd["reconstruction"].items()))
    print(f"  σ_1 = {svd['sigma_max']:.2f}  σ_min(nonzero) = {svd['sigma_min_nonzero']:.3f}  numerical rank = {svd['numerical_rank']}")

    rows = [
        ("Standardisation", "X_std=(X−μ)/σ", "standardization.py", "train μ≈0, σ=1 (Stage 4)", "—"),
        ("Covariance", "C=XᵀX/(n−1)", "covariance.py", all(cov["checks"].values()), rel(f_cov)),
        ("Matrix norms", "‖C‖_F=√Σλ², ‖C‖_2=λmax", "covariance.py / svd.py",
         cov["checks"]["‖C‖_F = sqrt(Σλ²)"] and cov["checks"]["‖C‖_2 = λ_max"] and svd["checks"]["‖Xc‖_F = sqrt(Σσ²), ‖Xc‖_2 = σ_1"], "—"),
        ("PCA (eig)", "Cvᵢ=λᵢvᵢ, Z=XW_k", "pca.py", all(pca_checks.values()) and leak_check, rel(f_ev) + ", " + rel(f_2d)),
        ("SVD", "X=UΣVᵀ, λᵢ=σᵢ²/(n−1)", "svd.py", all(svd["checks"].values()), rel(f_sv)),
    ]
    print("\n=== MFC-3 summary ===")
    print(f"  {'concept':16s} {'equation':26s} {'implementation':24s} {'verified':9s} figure")
    for c, e, f, ok, fig in rows:
        v = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
        print(f"  {c:16s} {e:26s} {f:24s} {v:9s} {fig}")

    print("\n=== Numerical checks ===")
    all_checks = {**{f"cov: {k}": v for k, v in cov["checks"].items()},
                  **{f"pca: {k}": v for k, v in pca_checks.items()},
                  "pca: fit on train only (μ_pca = μ_train ≠ μ_test)": bool(leak_check),
                  **{f"svd: {k}": v for k, v in svd["checks"].items()}}
    for k, v in all_checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    summary = {
        "data": {"features": "ECAPA-TDNN 192-D, z-scored on train", "n_train": n, "d": dim},
        "covariance": cov, "covariance_mfcc_contrast": cov_mfcc,
        "pca": {"variance_retained": {str(k): v for k, v in marks.items()},
                "eigenvalues_top10": pca.eigenvalues_[:10].tolist(),
                "saved_datasets": [rel(FEAT / f"ecapa_pca_k{k}.npz") for k in SAVE_KS]},
        "svd": svd, "checks": all_checks,
        "table": [dict(zip(["concept", "equation", "file", "verified", "figure"], [*r[:3], r[3] if isinstance(r[3], str) else bool(r[3]), r[4]])) for r in rows],
    }
    with open(MET / "mfc3_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if not all(all_checks.values()):
        raise SystemExit("Stage 5 verification FAILED")
    print(f"\nStage 5 OK -> {rel(MET / 'mfc3_summary.json')}, PCA datasets k={SAVE_KS}")


if __name__ == "__main__":
    main()
