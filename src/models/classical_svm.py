r"""Classical RBF-kernel SVM for closed-set speaker ID, with the convex dual / KKT made explicit.

-----------------------------------------------------------------------------------------------
1. PRIMAL (soft-margin, binary y_i ∈ {−1,+1}, feature map φ):
     min_{w,b,ξ}  ½‖w‖² + C Σ ξ_i      s.t.  y_i(wᵀφ(x_i)+b) ≥ 1 − ξ_i,   ξ_i ≥ 0
   Convex: quadratic objective + affine constraints ⇒ any local optimum is global; Slater's
   condition holds (ξ large makes every constraint strictly feasible) ⇒ strong duality.

2. LAGRANGIAN with multipliers α_i ≥ 0 (margin constraints) and μ_i ≥ 0 (ξ_i ≥ 0):
     L = ½‖w‖² + CΣξ_i − Σα_i[y_i(wᵀφ(x_i)+b) − 1 + ξ_i] − Σμ_iξ_i
   ∂L/∂w = 0 ⇒ w = Σ α_i y_i φ(x_i)       ∂L/∂b = 0 ⇒ Σ α_i y_i = 0
   ∂L/∂ξ_i = 0 ⇒ α_i + μ_i = C  ⇒  0 ≤ α_i ≤ C

3. DUAL (what libsvm/sklearn actually solves, via SMO):
     max_α  Σα_i − ½ ΣΣ α_i α_j y_i y_j K(x_i,x_j)    s.t.  0 ≤ α_i ≤ C,  Σ α_i y_i = 0
   with K(x,x') = φ(x)ᵀφ(x') = exp(−γ‖x−x'‖²). K is positive semi-definite (Mercer), so the
   Hessian Q_ij = y_i y_j K_ij is PSD ⇒ the dual is a concave QP over a convex polytope.
   Decision function: f(x) = Σ_{SV} α_i y_i K(x_i, x) + b.

4. KKT CONDITIONS at the optimum (necessary & sufficient here, by convexity + Slater):
     stationarity            (the three equations in 2)
     primal feasibility      y_i f(x_i) ≥ 1 − ξ_i,  ξ_i ≥ 0
     dual feasibility        0 ≤ α_i ≤ C
     complementary slackness α_i [y_i f(x_i) − 1 + ξ_i] = 0,   (C − α_i) ξ_i = 0
   ⇒ α_i = 0       : y_i f(x_i) ≥ 1   (outside margin; not a support vector)
     0 < α_i < C   : y_i f(x_i) = 1   (free SV, exactly on the margin)
     α_i = C       : y_i f(x_i) ≤ 1   (bounded SV, inside margin or misclassified)
   `kkt_report()` checks all of these numerically on the trained model.

5. MULTI-CLASS: sklearn's SVC trains one-vs-one, i.e. K(K−1)/2 = 45 binary duals for 10 speakers,
   then votes. Each binary problem has its own α's and support-vector count (logged).
-----------------------------------------------------------------------------------------------
Hyper-parameters (C, γ) are selected on VAL only; TEST is never used for selection.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

C_GRID = [0.1, 1, 10, 100, 1000]
GAMMA_MULT = [1 / 16, 1 / 4, 1, 4, 16]  # × γ_scale = 1 / (d · Var(X_train))
SEED = 42


def gamma_scale(X: np.ndarray) -> float:
    return 1.0 / (X.shape[1] * X.var())


def tune(Xtr, ytr, Xva, yva) -> tuple[dict, list[dict]]:
    """Grid search on val macro-F1 (tie-break: val accuracy, then smaller C, then smaller γ)."""
    g0 = gamma_scale(Xtr)
    grid = []
    for C, m in itertools.product(C_GRID, GAMMA_MULT):
        clf = SVC(kernel="rbf", C=C, gamma=g0 * m).fit(Xtr, ytr)
        p = clf.predict(Xva)
        grid.append({"C": C, "gamma": g0 * m, "gamma_mult": m,
                     "val_macro_f1": f1_score(yva, p, average="macro"), "val_acc": accuracy_score(yva, p)})
    best = max(grid, key=lambda r: (round(r["val_macro_f1"], 6), round(r["val_acc"], 6), -r["C"], -r["gamma"]))
    return best, grid


def fit(Xtr, ytr, C: float, gamma: float) -> SVC:
    """Final model on TRAIN only (val stays held out for open-set threshold calibration).
    probability=True adds Platt scaling (internal 5-fold CV on train) for posterior scores."""
    return SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=SEED).fit(Xtr, ytr)


def kkt_report(clf: SVC, Xtr: np.ndarray, ytr: np.ndarray, tol: float = 1e-2, K: np.ndarray | None = None) -> dict:
    """Recover each one-vs-one binary dual from sklearn/libsvm and verify the KKT conditions.
    For kernel='precomputed', pass the train Gram matrix as both `Xtr` and `K`."""
    C = clf.C
    classes = clf.classes_
    starts = np.r_[0, np.cumsum(clf.n_support_)]
    sv_idx = clf.support_
    old = clf.decision_function_shape
    clf.decision_function_shape = "ovo"
    F = clf.decision_function(Xtr)           # (n, n_pairs) in (i,j) order i<j
    clf.decision_function_shape = old

    pairs, n_sv, viol = [], [], {k: 0 for k in ("box", "sum_alpha_y", "free_on_margin", "bounded_inside", "nonsv_outside")}
    checked = {k: 0 for k in viol}
    sign = None
    for p, (i, j) in enumerate(itertools.combinations(range(len(classes)), 2)):
        # libsvm layout: coef of class-i SVs in pair (i,j) live in row j-1; class-j SVs in row i
        ci = clf.dual_coef_[j - 1, starts[i]:starts[i + 1]]
        cj = clf.dual_coef_[i, starts[j]:starts[j + 1]]
        coef = np.r_[ci, cj]                                   # = α_s y_s
        idx = np.r_[sv_idx[starts[i]:starts[i + 1]], sv_idx[starts[j]:starts[j + 1]]]
        y = np.r_[np.ones(len(ci)), -np.ones(len(cj))]        # class i ↦ +1, class j ↦ −1
        alpha = coef * y
        f = F[:, p]
        if sign is None:  # orientation of sklearn's ovo column vs libsvm's (i ↦ +1) convention
            sign = 1.0 if np.median(f[ytr == classes[i]]) > 0 else -1.0
        f = sign * f
        nz = alpha > 1e-8
        n_sv.append(int(nz.sum()))
        pairs.append({"pair": f"{classes[i]}v{classes[j]}", "n_sv": int(nz.sum()),
                      "n_free": int(((alpha > 1e-8) & (alpha < C - 1e-8)).sum()),
                      "n_bounded": int((alpha >= C - 1e-8).sum())})

        checked["box"] += len(alpha); viol["box"] += int(((alpha < -1e-8) | (alpha > C + 1e-6)).sum())
        checked["sum_alpha_y"] += 1; viol["sum_alpha_y"] += int(abs(coef.sum()) > 1e-6 * max(C, 1))
        m = y * f[idx]                                         # y_s f(x_s) for support vectors
        free = (alpha > 1e-8) & (alpha < C - 1e-8)
        checked["free_on_margin"] += int(free.sum()); viol["free_on_margin"] += int((np.abs(m[free] - 1) > tol).sum())
        bnd = alpha >= C - 1e-8
        checked["bounded_inside"] += int(bnd.sum()); viol["bounded_inside"] += int((m[bnd] > 1 + tol).sum())
        # non-SV training points of classes i, j must satisfy y f ≥ 1
        mask = np.isin(ytr, [classes[i], classes[j]]); mask[idx[nz]] = False
        ym = np.where(ytr[mask] == classes[i], 1.0, -1.0) * f[mask]
        checked["nonsv_outside"] += int(mask.sum()); viol["nonsv_outside"] += int((ym < 1 - tol).sum())

    if K is None:
        K = rbf_kernel(Xtr, gamma=clf._gamma)
    eigs = np.linalg.eigvalsh(K)
    eig_min = float(eigs.min())
    psd_tol = 1e-10 * len(K) * float(eigs.max())  # float round-off scale; RBF is PSD in exact arithmetic
    return {
        "n_pairs": len(pairs), "total_unique_SV": int(len(sv_idx)), "n_train": int(len(ytr)),
        "sv_per_pair": {"min": min(n_sv), "median": float(np.median(n_sv)), "max": max(n_sv)},
        "pairs": pairs,
        "kkt_checked": checked, "kkt_violations": viol,
        "kkt_ok": all(v == 0 for v in viol.values()),
        "kernel_train": {"fro_norm": float(np.linalg.norm(K, "fro")), "spectral_norm": float(np.linalg.norm(K, 2)),
                         "min_eig": eig_min, "PSD": eig_min > -psd_tol, "psd_tol": psd_tol, "mean_offdiag": float(K[~np.eye(len(K), dtype=bool)].mean())},
    }
