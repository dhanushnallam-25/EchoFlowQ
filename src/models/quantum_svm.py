r"""Hybrid quantum-classical SVM: quantum fidelity Gram matrix  +  classical convex SVM dual.

The quantum computer (here: exact statevector simulator) only supplies K_Q(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|².
Training is the same convex dual QP as classical_svm.py with K := K_Q:
    max_α Σα_i − ½ ΣΣ α_i α_j y_i y_j K_Q(x_i, x_j)   s.t. 0 ≤ α_i ≤ C, Σ α_i y_i = 0
K_Q is PSD (Gram matrix of density operators) ⇒ the dual stays concave ⇒ global optimum, KKT apply.
Prediction for a new x needs one kernel row K_Q(x, x_sv) per support vector.

Hyper-parameters: α (angle scale / bandwidth) and C, tuned on VAL only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from src.models.classical_svm import C_GRID
from src.quantum.feature_map import AngleScaler, build_feature_map
from src.quantum.quantum_kernel import fidelity_kernel, statevectors

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEED = 42


def tune_precomputed(K_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]], ytr, yva) -> tuple[dict, list[dict]]:
    """K_by_alpha[α] = (K_train_train, K_val_train). Select on val macro-F1, then val acc,
    then smaller C, then smaller α (smoother kernel) — same tie-break philosophy as the RBF grid."""
    grid = []
    for (a, (Ktr, Kva)), C in itertools.product(K_by_alpha.items(), C_GRID):
        p = SVC(kernel="precomputed", C=C).fit(Ktr, ytr).predict(Kva)
        grid.append({"alpha": a, "C": C, "val_macro_f1": f1_score(yva, p, average="macro"), "val_acc": accuracy_score(yva, p)})
    best = max(grid, key=lambda r: (round(r["val_macro_f1"], 6), round(r["val_acc"], 6), -r["C"], -r["alpha"]))
    return best, grid


@dataclass
class QuantumKernelSVM:
    """Self-contained deployable model: PCA-space input -> angles -> statevectors -> K_Q row -> SVC."""
    k: int
    kind: str
    alpha: float
    scaler: AngleScaler
    train_states: np.ndarray = field(repr=False)
    clf: SVC = field(repr=False)
    tau: float | None = None

    def kernel(self, Z: np.ndarray) -> np.ndarray:
        psi = statevectors(build_feature_map(self.k, self.kind), self.scaler.transform(Z, self.alpha))
        return fidelity_kernel(psi, self.train_states)

    def predict(self, Z):
        return self.clf.predict(self.kernel(Z))

    def predict_proba(self, Z):
        return self.clf.predict_proba(self.kernel(Z))

    def predict_open_set(self, Z, unknown_label: int = -1):
        P = self.predict_proba(Z)
        pred = self.clf.classes_[P.argmax(1)]
        return np.where(P.max(1) >= self.tau, pred, unknown_label), P.max(1)
