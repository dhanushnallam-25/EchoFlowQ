"""EchoFlow-Q v2 unified pipeline: speech in one language → speaker-attributed speech in another.

  1. load (any format) -> 16 kHz mono -> DC removal + peak normalisation
  2. Silero VAD + 1.5 s / 0.75 s sliding windows -> batched ECAPA-TDNN (192-D, L2)
  3. AHC (cosine, average linkage; method/threshold selected on validation mixtures in Stage 8)
  4. route each cluster centroid -> enrolled speaker or Unknown-n  (raw score ≥ τ), with either model set:
       'multilingual' (8 languages, 16 speakers)  or  'kannada10' (Stages 4–8)
  5. merge same-speaker turns -> Gnani Prisma STT per turn (explicit language or indic-auto; cached)
  6. translate each turn from its DETECTED language -> target (Gnani Evon, free fallbacks)
  7. synthesise each translated turn with a per-speaker voice (Gnani timbre-v3.0, edge-tts fallback)
     and stitch them into one track that keeps original turn timing without overlaps.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from src.data.hf_utils import PROJECT_ROOT
from src.diarization.clustering import centroids, cluster, windows_to_turns
from src.diarization.segment_embedder import embed_recording
from src.diarization.speaker_assigner import (LARGESCALE_BUNDLE, MODEL_SETS, UNKNOWN, available_model_sets, build_routers,
                                              load_largescale_bundle, speaker_genders)
from src.features.ecapa import load_encoder
from src.preprocessing.audio_cleaning import peak_normalize, remove_dc
from src.preprocessing.audio_loader import load_audio, resample, to_mono
from src.speech.asr import ASRCache, transcribe_turns
from src.speech.gnani_client import GnaniClient, GnaniError
from src.speech.languages import LANGUAGES, asr_code, code_of, name_of
from src.speech.translator import Translator, to_code
from src.speech.tts import Synthesizer, assemble_track

SR = 16_000
OUT_ROOT = PROJECT_ROOT / "data/processed/outputs"
CLASSIFIERS = {
    "Classical RBF-SVM (192-D)": "rbf_192",
    "Classical RBF-SVM (PCA-6)": "rbf_pca6",
    "Hybrid Quantum-Classical SVM (6 Qubits)": "quantum_k6",
    "Cosine similarity (ECAPA centroids)": "cosine",
}
TARGET_LANGUAGES = [name_of(c) for c in LANGUAGES]                 # English … Punjabi (8)
SOURCE_LANGUAGES = ["Auto-Detect (All 8 Languages)"] + TARGET_LANGUAGES
ASR_LANGUAGES = {"Auto-Detect (indic-auto)": "indic-auto", **{name_of(c): LANGUAGES[c]["bcp47"] for c in LANGUAGES}}


class EchoFlowPipeline:
    def __init__(self, use_gnani: bool = True, largescale_path: str | Path | None = None):
        dres = json.loads((PROJECT_ROOT / "results/metrics/diarization_results.json").read_text(encoding="utf-8"))
        self.cluster_cfg = dres["clustering_selected_on_val"]
        self._kannada_tau = dres["routers_tau"]
        self.largescale_path = Path(largescale_path or LARGESCALE_BUNDLE)
        self._routers: dict[str, dict] = {}
        self._genders: dict[str, dict] = {}
        self._cluster: dict[str, dict] = {}
        try:
            self.client = GnaniClient() if use_gnani else None
        except GnaniError:
            self.client = None
        self.translator = Translator(self.client)
        self.asr_cache = ASRCache()
        load_encoder()

    def model_sets(self) -> list[str]:
        """Available speaker-model sets, best first (large-scale Colab bundle once its zip is extracted)."""
        return available_model_sets(self.largescale_path)

    def routers(self, model_set: str) -> dict:
        if model_set not in self._routers:
            if model_set == "largescale":
                b = load_largescale_bundle(self.largescale_path)
                rs = build_routers("largescale", self.largescale_path)            # verifies check vectors
                self._genders[model_set] = b["label_gender"]
                self._cluster[model_set] = b.get("diarization") or self.cluster_cfg
            else:
                rs = build_routers(model_set)
                if model_set == "kannada10":
                    for k, r in rs.items():
                        r.tau = self._kannada_tau[k]
                self._genders[model_set] = speaker_genders(model_set)
            self._routers[model_set] = rs
        return self._routers[model_set]

    # ------------------------------------------------------------------ main entry
    def run(self, audio: str | Path | tuple[np.ndarray, int], target_language: str = "English",
            classifier: str = "Classical RBF-SVM (192-D)", source_language: str = "Auto-Detect",
            speaker_model: str | None = None, synthesize_voice: bool = True, asr_language: str | None = None,
            progress: Callable[[str, float], None] | None = None) -> dict:
        step = progress or (lambda msg, frac: None)
        speaker_model = speaker_model or self.model_sets()[0]
        router_key = CLASSIFIERS.get(classifier, classifier)
        router = self.routers(speaker_model)[router_key]
        tgt = to_code(target_language)
        lang_code = asr_language or asr_code(source_language)
        src_hint = code_of(source_language)
        lat: dict[str, float] = {}

        t = time.perf_counter(); step("Loading & cleaning audio", 0.03)
        if isinstance(audio, tuple):
            wav, sr = audio
            wav, in_path = resample(to_mono(np.asarray(wav, np.float32)), sr), None
        else:
            wav, _ = load_audio(audio)
            in_path = Path(audio)
        wav = peak_normalize(remove_dc(wav)).astype(np.float32)
        dur = len(wav) / SR
        stem = in_path.stem if in_path else "audio"
        out_dir = OUT_ROOT / re.sub(r"[^\w.-]", "_", f"{stem}__{tgt}__{speaker_model}_{router_key}")
        out_dir.mkdir(parents=True, exist_ok=True)
        if in_path is None:
            in_path = out_dir / "input_audio.wav"; sf.write(in_path, wav, SR, subtype="PCM_16")
        lat["load_clean"] = time.perf_counter() - t

        t = time.perf_counter(); step("Silero VAD + ECAPA-TDNN embeddings", 0.10)
        E = embed_recording(wav)
        lat["vad_embedding"] = time.perf_counter() - t
        if len(E.emb) == 0:
            return self._empty(dur, lat, router_key, target_language, in_path)

        t = time.perf_counter(); step("Agglomerative speaker clustering", 0.30)
        c = self._cluster.get(speaker_model, self.cluster_cfg)
        labels = cluster(E.emb, c["method"], c["threshold"], None, c.get("min_windows", 1))
        turns = windows_to_turns(E.windows, E.region, E.regions, labels, E.step)
        lat["clustering"] = time.perf_counter() - t

        t = time.perf_counter(); step(f"Speaker routing ({classifier})", 0.35)
        cents = centroids(E.emb, labels)
        cids = sorted(cents, key=lambda k: min(s for s, _, l in turns if l == k))     # order of first appearance
        idx, scores = router.scores(np.stack([cents[k] for k in cids]))
        names, unk_n = {}, 0
        for k, i, s in zip(cids, idx, scores):
            if s >= router.tau:
                names[k] = router.labels[int(i)]
            else:
                unk_n += 1
                names[k] = f"{UNKNOWN}-{unk_n}"
        margin = {k: float(s - router.tau) for k, s in zip(cids, scores)}
        lat["routing"] = time.perf_counter() - t

        merged = []                                  # merge same-speaker turns; duration-weighted margin
        for s, e, l in sorted(turns):
            spk, wm = names[l], margin[l] * (e - s)
            if merged and merged[-1][2] == spk and s - merged[-1][1] <= 0.5:
                merged[-1][1] = max(merged[-1][1], e); merged[-1][3] += wm; merged[-1][4] += e - s
            else:
                merged.append([s, e, spk, wm, e - s])

        t = time.perf_counter(); step(f"Gnani Prisma speech recognition ({lang_code})", 0.45)
        recs = transcribe_turns(wav, [(s, e, spk) for s, e, spk, *_ in merged], self.client, lang_code, cache=self.asr_cache)
        lat["asr"] = time.perf_counter() - t

        t = time.perf_counter(); step(f"Translation → {name_of(tgt)}", 0.65)
        for rec, m in zip(recs, merged):
            rec["duration_sec"] = round(rec["end_sec"] - rec["start_sec"], 2)
            rec["confidence_margin"] = round(m[3] / max(m[4], 1e-9), 4)
            rec["is_enrolled"] = not rec["speaker_id"].startswith(UNKNOWN)
            src = code_of(rec.get("detected_language")) or src_hint
            rec["source_language"] = name_of(src) if src else None
            out = self.translator.translate(rec["transcript"], tgt, src) if rec["transcript"] else {"text": "", "backend": None, "error": None}
            rec["translation"], rec["translation_backend"] = out["text"], out["backend"]
            if out.get("error"):
                rec["translation_error"] = out["error"]
        lat["translation"] = time.perf_counter() - t

        out_path, tts_status = None, None
        if synthesize_voice:
            t = time.perf_counter(); step(f"Voice synthesis in {name_of(tgt)}", 0.80)
            synth = Synthesizer(self.client, speaker_genders=self._genders.get(speaker_model))
            for rec in recs:
                r = synth.synthesize(rec.get("translation") or "", tgt, rec["speaker_id"], out_dir / f"{rec['segment_id']}.wav") \
                    if rec.get("translation") else {"path": None, "backend": None, "error": "no translation"}
                rec["tts_audio_path"] = self._rel(r.get("path"))
                rec["tts_backend"], rec["tts_voice"] = r.get("backend"), r.get("voice")
                if r.get("error"):
                    rec["tts_error"] = r["error"]
            track, place = assemble_track([{"start_sec": r["start_sec"], "path": PROJECT_ROOT / r["tts_audio_path"]}
                                           for r in recs if r.get("tts_audio_path")])
            it = iter(place)
            for rec in recs:
                if rec.get("tts_audio_path"):
                    p = next(it); rec["output_start_sec"], rec["output_end_sec"] = p["start_sec"], p["end_sec"]
            out_path = out_dir / "output_translated_audio.wav"
            sf.write(out_path, track, SR, subtype="PCM_16")
            tts_status = synth.status
            lat["tts"] = time.perf_counter() - t

        lat["total"] = sum(lat.values())
        step("Done", 1.0)
        detected = sorted({r["source_language"] for r in recs if r.get("source_language")})
        return {
            "input_audio_path": self._rel(in_path), "output_audio_path": self._rel(out_path),
            "audio_duration_sec": round(dur, 2), "output_duration_sec": round(sf.info(out_path).duration, 2) if out_path else None,
            "classifier": classifier, "router": router_key, "speaker_model": speaker_model, "speaker_model_name": MODEL_SETS[speaker_model],
            "tau": router.tau, "source_language": source_language, "asr_language_code": lang_code, "detected_languages": detected,
            "target_language": name_of(tgt), "clustering": {k: c.get(k) for k in ("method", "threshold", "min_windows")},
            "n_windows": int(len(E.emb)), "n_clusters": len(cids), "segments": recs, "speakers": self._analytics(recs),
            "latency_s": {k: round(v, 3) for k, v in lat.items()}, "real_time_factor": round(lat["total"] / max(dur, 1e-9), 3),
            "translation_status": self.translator.status, "tts_status": tts_status, "asr_cache_hits": self.asr_cache.hits,
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rel(p) -> str | None:
        if not p:
            return None
        p = Path(p)
        try:
            return p.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(p)

    @staticmethod
    def _analytics(recs: list[dict]) -> list[dict]:
        spk: dict[str, dict] = {}
        for r in recs:
            d = spk.setdefault(r["speaker_id"], {"speaker_id": r["speaker_id"], "talk_time_sec": 0.0, "turns": 0, "words": 0,
                                                "_m": 0.0, "is_enrolled": r["is_enrolled"], "languages": set()})
            d["talk_time_sec"] += r["duration_sec"]; d["turns"] += 1; d["words"] += len(r["transcript"].split())
            d["_m"] += r["confidence_margin"] * r["duration_sec"]
            if r.get("source_language"):
                d["languages"].add(r["source_language"])
        total = sum(d["talk_time_sec"] for d in spk.values()) or 1.0
        out = []
        for d in sorted(spk.values(), key=lambda x: -x["talk_time_sec"]):
            d["talk_share"] = round(d["talk_time_sec"] / total, 4)
            d["mean_confidence_margin"] = round(d.pop("_m") / max(d["talk_time_sec"], 1e-9), 4)
            d["talk_time_sec"] = round(d["talk_time_sec"], 2)
            d["languages"] = sorted(d["languages"])
            out.append(d)
        return out

    def _empty(self, dur, lat, router_key, target_language, in_path):
        lat["total"] = sum(lat.values())
        return {"input_audio_path": self._rel(in_path), "output_audio_path": None, "audio_duration_sec": round(dur, 2),
                "router": router_key, "target_language": target_language, "segments": [], "speakers": [], "latency_s": lat,
                "n_windows": 0, "n_clusters": 0, "translation_status": self.translator.status, "warning": "no speech detected"}
