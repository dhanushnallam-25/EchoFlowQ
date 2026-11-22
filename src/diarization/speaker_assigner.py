"""Stage 8: route each diarized cluster (L2-normalised centroid embedding) to spk_01..spk_10 or "Unknown".

Scores are RAW (no Platt scaling — Stage 7 showed it distorts quantum-kernel rejection):
  * cosine     : max_s cos(c, μ_s), μ_s = L2-normalised mean of speaker s's TRAIN embeddings
  * rbf_192    : max one-vs-rest SVM decision value, classical RBF on z-scored ECAPA (192-D)
  * rbf_pca6   : same on PCA-6 features
  * quantum_k6 : same with the ZZ-fidelity quantum kernel on PCA-6 (6 qubits)
A cluster is accepted iff score ≥ τ_router; τ is tuned on validation mixtures (never on test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from src.data.hf_utils import PROJECT_ROOT
from src.features.standardization import ZScoreScaler

FEAT = PROJECT_ROOT / "data/processed/features"
MODELS = PROJECT_ROOT / "results/saved_models"
UNKNOWN = "Unknown"


@dataclass
class Router:
    name: str
    labels: list[str]
    score_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]   # (n,192) -> (pred idx, score)
    tau: float | None = None

    def scores(self, E: np.ndarray):
        return self.score_fn(np.atleast_2d(E))

    def assign(self, E: np.ndarray) -> tuple[list[str], np.ndarray]:
        idx, s = self.scores(E)
        return [self.labels[i] if s_ >= self.tau else UNKNOWN for i, s_ in zip(idx, s)], s


def _ecapa_std():
    return ZScoreScaler.load(FEAT / "ecapa_scaler.npz")


def _pca6():
    m = np.load(FEAT / "ecapa_pca_model.npz")
    return m["mean"], m["eigenvectors"][:, :6]


MULTI_BUNDLE = PROJECT_ROOT / "results/models/multilingual/routers.joblib"
LARGESCALE_BUNDLE = PROJECT_ROOT / "results/models/multilingual/echoflow_v2_routers.joblib"
MODEL_SETS = {"largescale": "8-language large-scale (Colab artifacts)", "multilingual": "8-language · 16 speakers",
              "kannada10": "Kannada · 10 speakers"}


class ArtifactError(RuntimeError):
    pass


def available_model_sets(largescale_path=None) -> list[str]:
    """Best first: large-scale Colab bundle (once extracted) → local 16-speaker set → Stage 1–8 Kannada-10."""
    have = {"largescale": Path(largescale_path or LARGESCALE_BUNDLE).exists(), "multilingual": MULTI_BUNDLE.exists(), "kannada10": True}
    return [k for k in MODEL_SETS if have[k]]


def build_routers(model_set: str = "kannada10", largescale_path=None) -> dict[str, Router]:
    """model_set: 'kannada10' (Stages 4–8 models, τ from Stage 8 val mixtures set by the caller),
    'multilingual' (local v2 cross-lingual bundle) or 'largescale' (Colab echoflow_v2_routers.joblib)."""
    if model_set == "largescale":
        return build_largescale_routers(largescale_path or LARGESCALE_BUNDLE)
    return build_multilingual_routers() if model_set == "multilingual" else build_kannada_routers()


def load_largescale_bundle(path=LARGESCALE_BUNDLE) -> dict:
    b = joblib.load(path)                       # pickle: only load artifacts you trained yourself
    if not isinstance(b, dict) or b.get("format") != "echoflow-routers-v2":
        raise ArtifactError(f"{path} is not an echoflow-routers-v2 bundle")
    return b


def build_largescale_routers(path=LARGESCALE_BUNDLE, verify: bool = True) -> dict[str, Router]:
    """Rebuild the Colab-trained routers from plain arrays + sklearn SVCs, mirroring the notebook's `scores()`
    exactly, then re-score the bundle's check vectors; any mismatch (e.g. library-version drift) raises."""
    from src.quantum.feature_map import build_feature_map
    from src.quantum.quantum_kernel import fidelity_kernel, statevectors

    b = load_largescale_bundle(path)
    labels, mean, scale, pmean, W, mu = b["labels"], b["scaler_mean"], b["scaler_scale"], b["pca_mean"], b["pca_W"], b["mu"]
    q = b["quantum"]
    circ = build_feature_map(q["k"], "zz")
    std = lambda E: (np.asarray(E, np.float32) - mean) / scale

    def cosine(E):
        D = np.asarray(E, np.float32) @ mu.T
        return D.argmax(1), D.max(1)

    def svm(clf, tf):
        def f(E):
            D = clf.decision_function(tf(E))
            return D.argmax(1), D.max(1)
        return f

    def quantum(E):
        Zx = (std(E) - pmean) @ W
        psi = statevectors(circ, q["alpha"] * np.pi * np.tanh(np.asarray(Zx, np.float64) / q["sigma"]))
        D = q["svc"].decision_function(fidelity_kernel(psi, q["train_states"]))
        return D.argmax(1), D.max(1)

    rs = {"cosine": Router("cosine", labels, cosine), "rbf_192": Router("rbf_192", labels, svm(b["rbf_192"], std)),
          "rbf_pca6": Router("rbf_pca6", labels, svm(b["rbf_pca6"], lambda E: ((std(E) - pmean) @ W).astype(np.float32))),
          "quantum_k6": Router("quantum_k6", labels, quantum)}
    for k, r in rs.items():
        r.tau = float(b["tau"][k])
    if verify:
        bad = []
        for k, r in rs.items():
            idx, s = r.scores(b["check"]["emb"])
            exp = b["check"][k]
            if not (np.array_equal(np.asarray(idx), np.asarray(exp["idx"])) and np.allclose(s, exp["score"], rtol=1e-5, atol=1e-6)):
                bad.append(f"{k}: max |Δscore| {np.max(np.abs(np.asarray(s) - np.asarray(exp['score']))):.2e}")
        if bad:
            raise ArtifactError("check vectors do not reproduce (library versions differ from the notebook?): " + "; ".join(bad))
    return rs


def build_multilingual_routers(path=MULTI_BUNDLE) -> dict[str, Router]:
    b = joblib.load(path)
    labels, sc, pmean, W, mu = b["labels"], b["scaler"], b["pca_mean"], b["pca_W"], b["mu"]
    pca = lambda E: ((sc.transform(E).astype(np.float64) - pmean) @ W).astype(np.float32)

    def svm(clf, tf):
        def f(E):
            D = clf.decision_function(tf(E))
            return clf.classes_[D.argmax(1)], D.max(1)
        return f

    def cosine(E):
        S = E @ mu.T
        return S.argmax(1), S.max(1)

    q = b["quantum_k6"]

    def quantum(E):
        D = q.clf.decision_function(q.kernel(pca(E)))
        return q.clf.classes_[D.argmax(1)], D.max(1)

    rs = {"cosine": Router("cosine", labels, cosine), "rbf_192": Router("rbf_192", labels, svm(b["rbf_192"], sc.transform)),
          "rbf_pca6": Router("rbf_pca6", labels, svm(b["rbf_pca6"], pca)), "quantum_k6": Router("quantum_k6", labels, quantum)}
    for k, r in rs.items():
        r.tau = b["tau"][k]
    return rs


def speaker_genders(model_set: str) -> dict[str, str]:
    """speaker_label -> 'F'/'M'/'?' from enrollment metadata (used to pick TTS voices)."""
    if model_set == "largescale":
        return load_largescale_bundle(LARGESCALE_BUNDLE)["label_gender"]
    if model_set == "multilingual":
        return joblib.load(MULTI_BUNDLE)["label_gender"]
    import pandas as pd

    d = pd.read_csv(PROJECT_ROOT / "data/metadata/indicvoices_kn_10spk.csv")
    return d.drop_duplicates("speaker_label").set_index("speaker_label").gender.str[0].to_dict()


def build_kannada_routers() -> dict[str, Router]:
    raw = np.load(FEAT / "ecapa_raw.npz")
    labels = [str(x) for x in raw["label_names"]]
    sc = _ecapa_std()
    pmean, W6 = _pca6()

    mu = np.stack([raw["X_train"][raw["y_train"] == i].mean(0) for i in range(len(labels))])
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)

    def cosine(E):
        S = E @ mu.T
        return S.argmax(1), S.max(1)

    def svm_fn(path, transform):
        b = joblib.load(path)
        clf = b["model"]

        def f(E):
            D = clf.decision_function(transform(E))
            return clf.classes_[D.argmax(1)], D.max(1)
        return f

    std = lambda E: sc.transform(E)
    pca6 = lambda E: ((sc.transform(E).astype(np.float64) - pmean) @ W6).astype(np.float32)

    qb = joblib.load(MODELS / "svm_quantum_k6.joblib")["model"]

    def quantum(E):
        D = qb.clf.decision_function(qb.kernel(pca6(E)))
        return qb.clf.classes_[D.argmax(1)], D.max(1)

    return {
        "cosine": Router("cosine", labels, cosine),
        "rbf_192": Router("rbf_192", labels, svm_fn(MODELS / "svm_classical_ecapa.joblib", std)),
        "rbf_pca6": Router("rbf_pca6", labels, svm_fn(MODELS / "svm_classical_pca6.joblib", pca6)),
        "quantum_k6": Router("quantum_k6", labels, quantum),
    }
