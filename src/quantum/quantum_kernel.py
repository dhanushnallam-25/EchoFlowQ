r"""Fidelity quantum kernel  K_Q(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²  by exact statevector simulation.

Implementation: simulate each |φ(x)⟩ once with qiskit.quantum_info.Statevector (2^k amplitudes),
stack into Ψ ∈ ℂ^{n×2^k}, then the whole Gram block is one matrix product:  K = |Ψ_A Ψ_Bᴴ|²  (elementwise).
That is n state simulations instead of n² circuit evaluations; verified equal to qiskit-machine-learning's
FidelityStatevectorKernel in `verify_against_qiskit`.

K_Q is a valid (Mercer) kernel: K_ij = ⟨ρ_i, ρ_j⟩_F with ρ = |φ⟩⟨φ|, a Gram matrix of density
operators ⇒ symmetric, PSD, unit diagonal.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def statevectors(circ: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """(n, k) angles -> (n, 2^k) complex amplitudes."""
    c = circ.decompose()
    return np.stack([Statevector(c.assign_parameters(x)).data for x in np.asarray(X, np.float64)])


def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centred kernel-target alignment  ⟨K_c, L_c⟩_F / (‖K_c‖_F ‖L_c‖_F),  L = Y Yᵀ (one-hot ⇒ L_ij = [y_i = y_j]),
    K_c = H K H, H = I − 11ᵀ/n  (Cortes, Mohri & Rostamizadeh, 2012). 1 = perfectly label-aligned."""
    n = len(y)
    H = np.eye(n) - 1.0 / n
    L = (y[:, None] == y[None, :]).astype(float)
    Kc, Lc = H @ K @ H, H @ L @ H
    return float(np.sum(Kc * Lc) / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


def kernel_properties(K: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh(K)
    tol = 1e-10 * len(K) * eig.max()
    off = K[~np.eye(len(K), dtype=bool)]
    return {
        "symmetric": bool(np.allclose(K, K.T, atol=1e-12)),
        "unit_diagonal": bool(np.allclose(np.diag(K), 1.0, atol=1e-10)),
        "PSD": bool(eig.min() > -tol), "min_eig": float(eig.min()), "psd_tol": float(tol),
        "in_[0,1]": bool((K >= -1e-12).all() and (K <= 1 + 1e-10).all()),
        "fro_norm": float(np.linalg.norm(K, "fro")), "spectral_norm": float(eig.max()),
        "offdiag_mean": float(off.mean()), "offdiag_std": float(off.std()),
    }


def verify_against_qiskit(circ: QuantumCircuit, X: np.ndarray) -> dict:
    """Exact-equality + speed check vs qiskit_machine_learning.kernels.FidelityStatevectorKernel."""
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    t = time.perf_counter(); psi = statevectors(circ, X); K_ours = fidelity_kernel(psi, psi); t_ours = time.perf_counter() - t
    t = time.perf_counter(); K_ref = FidelityStatevectorKernel(feature_map=circ).evaluate(x_vec=X); t_ref = time.perf_counter() - t
    return {"n": len(X), "max_abs_diff": float(np.abs(K_ours - K_ref).max()), "equal": bool(np.allclose(K_ours, K_ref, atol=1e-10)),
            "ours_s": t_ours, "qiskit_ml_s": t_ref}
