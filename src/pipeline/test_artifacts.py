"""Verifies the Colab → zip → local workflow without touching the real results/ directory.

Usage (project root):
  .venv\\Scripts\\python.exe -m src.pipeline.test_artifacts <echoflow_v2_artifacts.zip>
Checks
  1. install_artifacts() into a temporary results dir: manifest checksums + check-vector self-test
  2. tampered zips are rejected and nothing is written (checksum mismatch, zip-slip path, missing bundle)
  3. graceful fallback: no bundle → model sets = local multilingual / kannada10
  4. EchoFlowPipeline with the Colab bundle: Kannada→English and English→Kannada voice-to-voice
"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import soundfile as sf

from src.data.hf_utils import PROJECT_ROOT
from src.diarization.speaker_assigner import ArtifactError
from src.pipeline.artifacts import BUNDLE_REL, install_artifacts
from src.pipeline.echoflow_pipeline import EchoFlowPipeline

MIX = PROJECT_ROOT / "data/processed/mixtures"


def rezip(src: bytes, edit) -> bytes:
    zin, out = zipfile.ZipFile(io.BytesIO(src)), io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in zin.infolist():
            r = edit(i.filename, zin.read(i))
            if r is not None:
                z.writestr(*r)
    return out.getvalue()


def main(zip_path: str) -> None:
    data = Path(zip_path).read_bytes()
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        res_dir = Path(tmp) / "results"
        info = install_artifacts(data, res_dir)
        bundle = res_dir / BUNDLE_REL
        checks += [(f"install: {len(info['installed'])} files, {info['n_enrolled']} enrolled speakers, check vectors reproduced", bundle.exists()),
                   ("install: metrics + figures extracted", (res_dir / "metrics/largescale_multilingual_results.json").exists()
                    and any((res_dir / "figures").glob("largescale_*.png")))]

        for name, bad in (("checksum mismatch", rezip(data, lambda n, b: (n, b + b"x") if n.endswith(".png") else (n, b))),
                          ("zip-slip path", _with_extra(data, "../evil.txt", b"x")),
                          ("missing model bundle", rezip(data, lambda n, b: None if n == BUNDLE_REL else (n, b)))):
            d2 = Path(tmp) / f"r_{name.replace(' ', '_')}"
            try:
                install_artifacts(bad, d2); rejected = False
            except ArtifactError:
                rejected = True
            checks.append((f"reject {name}; nothing written", rejected and not d2.exists()))

        fallback = EchoFlowPipeline(largescale_path=Path(tmp) / "nope.joblib")
        checks.append((f"fallback without bundle → {fallback.model_sets()}", "largescale" not in fallback.model_sets()))

        pipe = EchoFlowPipeline(largescale_path=bundle)
        checks.append((f"with bundle → model sets {pipe.model_sets()}", pipe.model_sets()[0] == "largescale"))
        for name, wav, tgt, clf in (("kn→en", "test_mix_kn_to_en", "English", "Classical RBF-SVM (192-D)"),
                                    ("en→kn", "test_mix_en_to_kn", "Kannada", "Hybrid Quantum-Classical SVM (6 Qubits)")):
            r = pipe.run(MIX / f"{wav}.wav", tgt, clf, source_language="Auto-Detect", speaker_model="largescale")
            op = PROJECT_ROOT / r["output_audio_path"]
            segs = [s for s in r["segments"] if s.get("translation")]
            ok = op.exists() and sf.info(op).samplerate == 16000 and all(s.get("tts_audio_path") for s in segs) and len(segs) > 0
            spk = sorted({s["speaker_id"] for s in r["segments"]})
            checks.append((f"{name} via Colab bundle: {r['audio_duration_sec']:.0f}s → {r['output_duration_sec']:.0f}s {tgt} voice, "
                           f"{len(segs)} turns, speakers {spk}, clustering th={r['clustering']['threshold']}", ok))

    print("\n=== Artifact workflow verification ===")
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} passed")
    if not all(ok for _, ok in checks):
        raise SystemExit("artifact workflow test FAILED")


def _with_extra(data: bytes, name: str, payload: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as z:
        for i in zin.infolist():
            z.writestr(i, zin.read(i))
        z.writestr(zipfile.ZipInfo(name), payload)
    return out.getvalue()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.pipeline.test_artifacts <echoflow_v2_artifacts.zip>")
    main(sys.argv[1])
