r"""Quantum feature maps  |φ(x)⟩ = U_Φ(x) |0⟩^{⊗k}  for k-dimensional PCA speaker features.

ZZ feature map (Havlíček et al., 2019), `reps` repetitions of  U_Φ(x) H^{⊗k}  with
    U_Φ(x) = exp( i Σ_j φ_j(x) Z_j  +  i Σ_{j<l} φ_{jl}(x) Z_j Z_l ),
    φ_j(x) = x_j,   φ_{jl}(x) = (π − x_j)(π − x_l)          (Qiskit default data map)
'full' entanglement ⇒ every qubit pair gets a ZZ phase (2 CNOTs + 1 phase gate each).
Z feature map (ablation): no ZZ terms ⇒ product state ⇒ kernel factorises into a product of
per-feature cosines — classically trivial, isolates the contribution of entanglement.

Angle scaling (fit on TRAIN only):  x_j = α · π · tanh(z_j / σ_j)
  σ_j = train std of PCA component j; tanh bounds every angle to (−απ, απ) ⊂ (−π, π), so the
  2π-periodic gates never wrap distant points onto each other. α is the kernel "bandwidth"
  (small α ⇒ smoother, flatter kernel), tuned on val just like γ for the RBF kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map, zz_feature_map

REPS = 2
ENTANGLEMENT = "full"


def build_feature_map(k: int, kind: str = "zz", reps: int = REPS) -> QuantumCircuit:
    if kind == "zz":
        return zz_feature_map(feature_dimension=k, reps=reps, entanglement=ENTANGLEMENT)
    if kind == "z":
        return z_feature_map(feature_dimension=k, reps=reps)
    raise ValueError(kind)


@dataclass
class AngleScaler:
    sigma_: np.ndarray | None = None

    def fit(self, Z_train: np.ndarray) -> "AngleScaler":
        self.sigma_ = np.maximum(np.asarray(Z_train, np.float64).std(axis=0), 1e-8)
        return self

    def transform(self, Z: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * np.pi * np.tanh(np.asarray(Z, np.float64) / self.sigma_)


def circuit_stats(circ: QuantumCircuit) -> dict:
    t = transpile(circ, basis_gates=["h", "p", "cx"], optimization_level=0)
    ops = dict(t.count_ops())
    return {"qubits": circ.num_qubits, "parameters": circ.num_parameters, "depth": t.depth(),
            "cx": int(ops.get("cx", 0)), "two_qubit_depth": t.depth(lambda g: g.operation.num_qubits == 2),
            "ops": {k: int(v) for k, v in ops.items()}}


def draw(circ: QuantumCircuit, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = circ.draw("mpl", fold=-1, style={"backgroundcolor": "#fcfcfb"})  # native H / P / CX gates
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
