"""Integration test: EchoFlowPipeline v1 (Kannada, text) + v2 (bidirectional voice-to-voice) + Streamlit app.

Usage (project root):  .venv\\Scripts\\python.exe -m src.pipeline.test_end_to_end
  v1  test_mix01 (10 Kannada speakers): classical + quantum routers, text only
  v2  test_mix_kn_to_en  (Kannada + Hindi speakers) → English voice   · classical RBF-192, 8-language speaker set
      test_mix_en_to_kn  (English + Kannada speakers) → Kannada voice · quantum ZZ (6 qubits), 8-language speaker set
  app headless AppTest: render + full Run click with dual audio players
"""
from __future__ import annotations

import json
import re
import time

import soundfile as sf

from src.data.hf_utils import PROJECT_ROOT
from src.pipeline.echoflow_pipeline import EchoFlowPipeline

MIX = PROJECT_ROOT / "data/processed/mixtures"
REQUIRED = {"segment_id", "start_sec", "end_sec", "duration_sec", "speaker_id", "confidence_margin",
            "detected_language", "transcript", "translation"}
KANNADA = re.compile(r"[ಀ-೿]")
LATIN = re.compile(r"[A-Za-z]")


def identity_accuracy(segments, ref, is_unknown) -> float:
    """Fraction of reference speech time whose predicted speaker is right (unenrolled ↔ Unknown-n)."""
    ok = tot = 0.0
    for rs, re_, rl in ref:
        tot += re_ - rs
        for s in segments:
            o = max(0.0, min(re_, s["end_sec"]) - max(rs, s["start_sec"]))
            if o and ((is_unknown(rl) and s["speaker_id"].startswith("Unknown")) or s["speaker_id"] == rl):
                ok += o
    return ok / tot


def base_checks(res: dict, name: str) -> list[tuple[str, bool]]:
    segs = res["segments"]
    starts = [s["start_sec"] for s in segs]
    return [
        (f"{name}: {len(segs)} segments with all required fields", len(segs) > 0 and all(REQUIRED <= set(s) for s in segs)),
        (f"{name}: segments time-ordered within the audio",
         starts == sorted(starts) and all(0 <= s["start_sec"] < s["end_sec"] <= res["audio_duration_sec"] + 0.05 for s in segs)),
        (f"{name}: ≥80% of segments transcribed", sum(bool(s["transcript"]) for s in segs) >= 0.8 * len(segs)),
        (f"{name}: every transcribed segment translated", all(s["translation"] for s in segs if s["transcript"])),
        (f"{name}: talk-time shares sum to 1", abs(sum(s["talk_share"] for s in res["speakers"]) - 1) < 1e-3),
    ]


def voice_checks(res: dict, name: str, script: re.Pattern, expect_langs: set[str]) -> list[tuple[str, bool]]:
    segs = [s for s in res["segments"] if s.get("translation")]
    op = PROJECT_ROOT / res["output_audio_path"] if res.get("output_audio_path") else None
    info = sf.info(op) if op and op.exists() else None
    place = sorted((s["output_start_sec"], s["output_end_sec"]) for s in segs if "output_start_sec" in s)
    voices = {s["speaker_id"]: s.get("tts_voice") for s in segs}
    return [
        (f"{name}: output track exists, 16 kHz mono, {info.duration:.1f}s" if info else f"{name}: output track exists",
         bool(info and info.samplerate == 16000 and info.channels == 1 and info.duration > 1)),
        (f"{name}: every translated turn has its own TTS clip", all(s.get("tts_audio_path") and (PROJECT_ROOT / s["tts_audio_path"]).exists() for s in segs)),
        (f"{name}: translated turns placed in order without overlap", len(place) == len(segs) and all(b[0] >= a[1] - 1e-6 for a, b in zip(place, place[1:]))),
        (f"{name}: no turn starts before its source turn", all(s["output_start_sec"] >= s["start_sec"] - 1e-6 for s in segs)),
        (f"{name}: translations written in the target script", sum(bool(script.search(s["translation"])) for s in segs) >= 0.9 * len(segs)),
        (f"{name}: detected source languages ⊇ {sorted(expect_langs)} (got {res['detected_languages']})", expect_langs <= set(res["detected_languages"])),
        (f"{name}: distinct speakers get distinct voices ({len(set(voices.values()))} voices / {len(voices)} speakers)",
         len(set(voices.values())) == len(voices)),
    ]


def app_test() -> list[tuple[str, bool]]:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"), default_timeout=900)
    at.run()
    checks = [("app: initial render without exceptions", not at.exception)]
    mix = [s for s in at.sidebar.selectbox if s.label == "Meeting"]
    if mix:
        target = next(o for o in mix[0].options if "test_mix_kn_to_en" in str(o))
        mix[0].select(target).run()
    btn = [b for b in at.button if "Run EchoFlow-Q" in b.label]
    checks.append(("app: Run button present", bool(btn)))
    if btn:
        t = time.perf_counter()
        btn[0].click().run()
        md = " ".join(m.value for m in at.markdown)
        checks += [(f"app: full voice-to-voice run without exceptions ({time.perf_counter() - t:.0f}s)", not at.exception),
                   ("app: side-by-side input / output audio rendered", "Input audio" in md and "Output audio" in md),
                   ("app: speaker cards + per-turn transcript rendered", "efq-card" in md and "efq-turn" in md),
                   ("app: translated-audio download button", any("translated audio" in b.label for b in at.get("download_button")))]
        if at.exception:
            print(at.exception)
    return checks


def main() -> None:
    pipe = EchoFlowPipeline()
    checks, results = [], {}

    # ---- v1: Kannada text pipeline ----
    dres = json.loads((PROJECT_ROOT / "results/metrics/diarization_results.json").read_text(encoding="utf-8"))
    mix = next(m for m in dres["trackA"]["mixtures"] if m["name"] == "test_mix01")
    ref1 = [tuple(r) for r in mix["ref"]]
    for name, clf, tgt in (("v1-classical", "Classical RBF-SVM (192-D)", "English"), ("v1-quantum", "Hybrid Quantum-Classical SVM (6 Qubits)", "Hindi")):
        res = pipe.run(MIX / "test_mix01.wav", tgt, clf, source_language="Kannada", speaker_model="kannada10", synthesize_voice=False)
        res["identity_accuracy"] = identity_accuracy(res["segments"], ref1, lambda l: l == mix["unknown"])
        results[name] = res
        checks += base_checks(res, name)
        print(f"{name}: {len(res['segments'])} segments, identity accuracy {res['identity_accuracy'] * 100:.1f}%, {res['latency_s']['total']:.1f}s")

    # ---- v2: bidirectional voice-to-voice ----
    for name, wav, clf, tgt, script, langs in (
            ("kn→en", "test_mix_kn_to_en", "Classical RBF-SVM (192-D)", "English", LATIN, {"Kannada", "Hindi"}),
            ("en→kn", "test_mix_en_to_kn", "Hybrid Quantum-Classical SVM (6 Qubits)", "Kannada", KANNADA, {"English", "Kannada"})):
        meta = json.loads((MIX / f"{wav}.json").read_text(encoding="utf-8"))
        ref = [(t["start"], t["end"], t["speaker"]) for t in meta["turns"]]
        res = pipe.run(MIX / f"{wav}.wav", tgt, clf, source_language="Auto-Detect", speaker_model="multilingual")
        res["identity_accuracy"] = identity_accuracy(res["segments"], ref, lambda l: l.endswith("_imp"))
        results[name] = res
        checks += base_checks(res, name) + voice_checks(res, name, script, langs)
        print(f"\n=== {name} · {clf} · {res['audio_duration_sec']:.0f}s in → {res['output_duration_sec']:.0f}s {tgt} voice · "
              f"identity {res['identity_accuracy'] * 100:.1f}% · TTS {res['tts_status']} · {res['latency_s']['total']:.1f}s ===")
        for s in res["segments"]:
            print(f"  [{s['start_sec']:5.1f}→{s.get('output_start_sec', float('nan')):5.1f}] {s['speaker_id']:10s} {str(s.get('source_language')):8s} "
                  f"| {s['transcript'][:40]:40s} | {(s['translation'] or '')[:55]:55s} | {s.get('tts_voice')}")
        print("  latency: " + ", ".join(f"{k} {v:.1f}s" for k, v in res["latency_s"].items()))

    checks += app_test()
    (PROJECT_ROOT / "results/metrics/end_to_end_test.json").write_text(
        json.dumps({"checks": [{"check": c, "pass": ok} for c, ok in checks], "results": results}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("\n=== Verification ===")
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    if not all(ok for _, ok in checks):
        raise SystemExit("integration test FAILED")


if __name__ == "__main__":
    main()
