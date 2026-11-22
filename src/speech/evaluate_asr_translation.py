"""Stage 9 runner: Gnani smoke test -> ASR WER/CER benchmark -> end-to-end diarized multilingual transcript.

1. Smoke test: 1 test clip through STT, then translation to English + Hindi (status + raw schema logged).
2. ASR benchmark: 15 held-out Kannada TEST clips (1 per enrolled speaker + 5 seeded extras) vs the
   IndicVoices `normalized` transcripts -> WER / CER (jiwer) after punctuation/space normalisation.
3. End-to-end: test_mix01.wav -> VAD + ECAPA windows -> AHC (val-selected config) -> speaker routing
   (val-tuned τ) -> merge turns -> Gnani STT per turn -> translation (en, hi) if the key allows it.

Usage (project root):  .venv\\Scripts\\python.exe -m src.speech.evaluate_asr_translation
"""
from __future__ import annotations

import json
import re
import time

import jiwer
import numpy as np
import pandas as pd

from src.data.hf_utils import PROJECT_ROOT
from src.diarization.clustering import centroids
from src.diarization.evaluate_diarization import UNK_TEST, build_mixtures, diarize
from src.diarization.segment_embedder import embed_recording
from src.diarization.speaker_assigner import build_routers
from src.speech.asr import ASRCache, merge_turns, transcribe_turns
from src.speech.gnani_client import GnaniClient, GnaniError
from src.speech.translator import Translator

META, MET = PROJECT_ROOT / "data/metadata", PROJECT_ROOT / "results/metrics"
PUNCT = re.compile(r"[\.,!?;:\"'“”‘’()\[\]{}।॥\-–—…]")


TAGS = re.compile(r"<[^>]*>|\[[^\]]*\]")   # IndicVoices annotation tags, e.g. <unintelligible>, [sir]


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", PUNCT.sub(" ", TAGS.sub(" ", str(t)))).strip().lower()


def smoke(client: GnaniClient, splits: pd.DataFrame) -> dict:
    row = splits[splits.split == "test"].iloc[0]
    out = {"clip_id": row.clip_id, "reference": row.transcript}
    r = client.transcribe(PROJECT_ROOT / row.filepath, "kn-IN")
    out["stt"] = {"status": 200, "latency_ms": round(r["latency_ms"]), "schema_keys": sorted(r["raw"].keys()),
                  "transcript": r["transcript"], "detected_language": r["detected_language"]}
    print(f"[smoke] STT 200 in {r['latency_ms']:.0f} ms; keys={out['stt']['schema_keys']}")
    print(f"        ref: {row.transcript}\n        hyp: {r['transcript']}")
    for tgt in ("English", "Hindi"):
        try:
            t = client.translate(r["transcript"], "Kannada", tgt)
            out[f"translate_{tgt}"] = {"status": 200, "latency_ms": round(t["latency_ms"]), "text": t["translation"]}
            print(f"[smoke] translate→{tgt}: 200 in {t['latency_ms']:.0f} ms: {t['translation']}")
        except GnaniError as e:
            out[f"translate_{tgt}"] = {"status": e.status, "error": e.body[:300]}
            print(f"[smoke] translate→{tgt}: HTTP {e.status} {e.body[:160]}")
    return out


def asr_benchmark(client: GnaniClient, splits: pd.DataFrame, n_extra: int = 5) -> dict:
    test = splits[splits.split == "test"]
    rng = np.random.default_rng(42)
    picks = [g.iloc[rng.integers(len(g))] for _, g in test.groupby("speaker_label")]
    rest = test[~test.clip_id.isin([p.clip_id for p in picks])]
    picks += [rest.iloc[i] for i in rng.choice(len(rest), n_extra, replace=False)]
    rows = []
    for p in picks:
        r = client.transcribe(PROJECT_ROOT / p.filepath, "kn-IN")
        ref, hyp = norm(p.transcript), norm(r["transcript"])
        rows.append({"clip_id": p.clip_id, "speaker": p.speaker_label, "duration_sec": p.duration_sec, "reference": p.transcript,
                     "hypothesis": r["transcript"], "wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp),
                     "latency_ms": r["latency_ms"], "server_ms": r["server_ms"], "detected_language": r["detected_language"]})
    refs, hyps = [norm(r["reference"]) for r in rows], [norm(r["hypothesis"]) for r in rows]
    agg = {"n_clips": len(rows), "n_speakers": len({r["speaker"] for r in rows}), "audio_sec": float(sum(r["duration_sec"] for r in rows)),
           "WER": jiwer.wer(refs, hyps), "CER": jiwer.cer(refs, hyps),
           "WER_unnormalised": jiwer.wer([r["reference"] for r in rows], [r["hypothesis"] for r in rows]),
           "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
           "mean_server_ms": float(np.mean([r["server_ms"] for r in rows])),
           "rtf": float(sum(r["latency_ms"] for r in rows) / 1000 / sum(r["duration_sec"] for r in rows)),
           "detected_languages": sorted({str(r["detected_language"]) for r in rows})}
    return {"aggregate": agg, "clips": rows}


def end_to_end(client: GnaniClient, translator: Translator, mix_name: str = "test_mix01") -> dict:
    dres = json.loads((MET / "diarization_results.json").read_text(encoding="utf-8"))
    cfg = dres["clustering_selected_on_val"]
    mixes = {m["name"]: m for m in build_mixtures("test", UNK_TEST, seed=4242)}   # deterministic → identical audio
    m = mixes[mix_name]
    wav = m["wav"]
    routers = build_routers()
    for k, r in routers.items():
        r.tau = dres["routers_tau"][k]

    t = time.perf_counter()
    E = embed_recording(wav)
    lab, turns = diarize(E, cfg["method"], cfg["threshold"], None, cfg["min_windows"])
    cents = centroids(E.emb, lab)
    cids = sorted(cents)
    routed = {k: dict(zip(cids, r.assign(np.stack([cents[c] for c in cids]))[0])) for k, r in routers.items()}
    t_diar = time.perf_counter() - t

    spk_turns = merge_turns([(s, e, routed["cosine"][c]) for s, e, c in turns])
    t = time.perf_counter()
    recs = transcribe_turns(wav, spk_turns, client, "kn-IN", cache=ASRCache())
    t_asr = time.perf_counter() - t
    for rec in recs:   # alternative routers' labels for the same span (majority by overlap)
        for k in ("rbf_192", "quantum_k6"):
            ov = {}
            for s, e, c in turns:
                o = max(0.0, min(e, rec["end_sec"]) - max(s, rec["start_sec"]))
                if o > 0:
                    ov[routed[k][c]] = ov.get(routed[k][c], 0) + o
            rec[f"speaker_{k}"] = max(ov, key=ov.get) if ov else None
    t = time.perf_counter()
    translator.translate_records(recs, ("en", "hi"))
    t_tr = time.perf_counter() - t

    # reference: clip transcripts in time order (speaker-agnostic meeting WER)
    seg = pd.concat([pd.read_csv(META / "vad_segments.csv"), pd.read_csv(META / "vad_unknown_segments.csv")])
    tx = pd.concat([pd.read_csv(META / "splits.csv")[["clip_id", "transcript"]], pd.read_csv(META / "indicvoices_kn_unknown.csv")[["clip_id", "transcript"]]])
    path2clip = dict(zip(seg.vad_filepath, seg.clip_id)); clip2tx = dict(zip(tx.clip_id, tx.transcript))
    ref_turns = [{"start": round(c["start"], 2), "end": round(c["end"], 2), "speaker": c["speaker"], "truncated": c["truncated"],
                  "transcript": clip2tx[path2clip[c["vad_filepath"]]]} for c in m["clips"]]
    ref_text = norm(" ".join(r["transcript"] for r in ref_turns))
    hyp_text = norm(" ".join(r["transcript"] for r in recs))
    return {"mixture": mix_name, "duration_sec": m["dur"], "true_speakers": m["speakers"], "unknown_speaker": m["unknown"],
            "clustering": cfg, "segments": recs, "reference_turns": ref_turns,
            "meeting_WER": jiwer.wer(ref_text, hyp_text), "meeting_CER": jiwer.cer(ref_text, hyp_text),
            "n_truncated_ref_turns": int(sum(r["truncated"] for r in ref_turns)),
            "timing_s": {"diarization_routing": t_diar, "asr": t_asr, "translation": t_tr},
            "translation_status": translator.status, "translator_stats": translator.stats}


def main(skip_benchmark: bool = False) -> None:
    client = GnaniClient()
    translator = Translator(client)
    splits = pd.read_csv(META / "splits.csv")
    print(f"Gnani base URL: {client.base}\n")
    prev = json.loads((MET / "asr_translation_results.json").read_text(encoding="utf-8")) if (MET / "asr_translation_results.json").exists() else {}

    if skip_benchmark and "asr_benchmark" in prev:
        sm, bench = prev["smoke_test"], prev["asr_benchmark"]
        print("[skip] reusing saved smoke test + 15-clip ASR benchmark (no STT credits spent)")
    else:
        sm = smoke(client, splits)
        print("\n=== ASR benchmark: 15 held-out Kannada test clips ===")
        bench = asr_benchmark(client, splits)
    for r in bench["clips"]:
        print(f"  {r['speaker']} {r['duration_sec']:5.1f}s  WER {r['wer']:.2f}  CER {r['cer']:.2f}  {r['latency_ms']:5.0f} ms")
    a = bench["aggregate"]
    print(f"  => WER {a['WER']*100:.1f}%  CER {a['CER']*100:.1f}%  (unnormalised WER {a['WER_unnormalised']*100:.1f}%)  "
          f"mean latency {a['mean_latency_ms']:.0f} ms (server {a['mean_server_ms']:.0f} ms), RTF {a['rtf']:.3f}, langs {a['detected_languages']}")

    print("\n=== End-to-end: test_mix01 (diarization → routing → Gnani ASR → translation) ===")
    e2e = end_to_end(client, translator)
    print(f"  true speakers {e2e['true_speakers']} (unknown: {e2e['unknown_speaker']}); {len(e2e['segments'])} merged turns; "
          f"timing diar {e2e['timing_s']['diarization_routing']:.1f}s · ASR {e2e['timing_s']['asr']:.1f}s")
    print(f"  translation: {e2e['translation_status']}\n")
    for r in e2e["segments"]:
        en, hi = r.get("translation_en") or "—", r.get("translation_hi") or "—"
        print(f"  [{r['start_sec']:5.1f}–{r['end_sec']:5.1f}] {r['speaker_id']:8s} | {r['transcript'] or '(' + str(r['error']) + ')'}\n"
              f"  {'':15s}{'':8s} | EN [{r.get('translation_en_backend')}]: {en}\n  {'':15s}{'':8s} | HI: {hi}")
    print("\n  reference (ground-truth turns):")
    for r in e2e["reference_turns"]:
        print(f"  [{r['start']:5.1f}–{r['end']:5.1f}] {r['speaker']:8s} | {r['transcript']}{'  (clip truncated at 8 s in mixture)' if r['truncated'] else ''}")
    print(f"\n  meeting-level WER {e2e['meeting_WER']*100:.1f}%  CER {e2e['meeting_CER']*100:.1f}%  "
          f"({e2e['n_truncated_ref_turns']} reference turns truncated in the mixture → their tails count as deletions)")

    MET.mkdir(parents=True, exist_ok=True)
    with open(MET / "asr_translation_results.json", "w", encoding="utf-8") as fh:
        json.dump({"api_base": client.base, "smoke_test": sm, "asr_benchmark": bench, "end_to_end": e2e}, fh,
                  indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nSaved -> results/metrics/asr_translation_results.json")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-benchmark", action="store_true", help="reuse saved smoke test + 15-clip ASR benchmark")
    main(ap.parse_args().skip_benchmark)
