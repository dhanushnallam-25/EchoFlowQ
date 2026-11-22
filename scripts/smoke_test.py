"""Stage 1 smoke test: verifies every critical dependency actually works on this machine.

Run from project root:  .venv\\Scripts\\python.exe scripts\\smoke_test.py
Downloads ~80 MB once (ECAPA-TDNN weights) into pretrained_models/ (git-ignored).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        t = time.perf_counter()
        try:
            msg = fn() or ""
            results.append((name, True, f"{msg} ({time.perf_counter() - t:.1f}s)"))
        except Exception as e:  # noqa: BLE001 - we want to report everything
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


@check("torch + torchaudio")
def _():
    import torch, torchaudio
    wav = torch.randn(1, 44100)
    out = torchaudio.functional.resample(wav, 44100, 16000)
    assert out.shape[-1] == 16000
    return f"torch {torch.__version__}, torchaudio {torchaudio.__version__}, CUDA={torch.cuda.is_available()}"


@check("librosa MFCC")
def _():
    import numpy as np, librosa
    y = np.random.randn(16000).astype("float32")
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20)
    return f"librosa {librosa.__version__}, mfcc shape {m.shape}"


@check("Silero VAD")
def _():
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.zeros(16000), model, sampling_rate=16000)
    return f"loaded, {len(ts)} segments on silence (expected 0)"


@check("SpeechBrain ECAPA-TDNN")
def _():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,  # Windows: symlinks need admin/Developer Mode
    )
    emb = enc.encode_batch(torch.randn(1, 32000))
    return f"embedding shape {tuple(emb.shape)} (expect (1,1,192))"


@check("Qiskit fidelity kernel + precomputed SVM")
def _():
    import numpy as np, qiskit, qiskit_machine_learning as qml
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from sklearn.svm import SVC
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4)); y = np.repeat([0, 1, 2], 4)
    kern = FidelityStatevectorKernel(feature_map=zz_feature_map(4, reps=2))
    K = kern.evaluate(X)
    assert np.allclose(np.diag(K), 1.0) and np.allclose(K, K.T)
    SVC(kernel="precomputed").fit(K, y)
    return f"qiskit {qiskit.__version__}, qml {qml.__version__}, K {K.shape} symmetric, diag=1"


@check("datasets / HF hub / dotenv / pyannote.metrics")
def _():
    import datasets, huggingface_hub, dotenv, pyannote.metrics  # noqa: F401
    return f"datasets {datasets.__version__}, hub {huggingface_hub.__version__}"


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} @ {sys.executable}\n")
    for name, ok, msg in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:45s} {msg}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)
