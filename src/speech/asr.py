"""Segment-level ASR over diarized speaker turns with Gnani Prisma STT.

Turns are first merged (same speaker label, gap < `merge_gap` s) so the recogniser gets whole
utterances rather than 0.75 s diarization fragments. Turns longer than 28 s (REST limit 30 s) are
split into equal sub-chunks and the transcripts stitched in order.
An optional on-disk cache keyed by sha1(PCM16 chunk) + language + model means re-running the same
audio (e.g. switching classifier or target language in the UI) never spends STT credits twice.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data.hf_utils import PROJECT_ROOT
from src.speech.gnani_client import STT_MODEL, GnaniClient, GnaniError

SR = 16_000
MAX_CHUNK = 28.0
ASR_CACHE = PROJECT_ROOT / "results/cache/asr_cache.json"


class ASRCache:
    def __init__(self, path: Path = ASR_CACHE):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.hits = 0

    @staticmethod
    def key(chunk: np.ndarray, language_code: str, model: str = STT_MODEL) -> str:
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        return f"{hashlib.sha1(pcm).hexdigest()}:{language_code}:{model}"

    def get(self, k):
        v = self.data.get(k)
        self.hits += v is not None
        return v

    def put(self, k, v):
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


def merge_turns(turns: list[tuple], merge_gap: float = 0.5) -> list[tuple]:
    out: list[list] = []
    for s, e, spk, *rest in sorted(turns):
        if out and out[-1][2] == spk and s - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, spk, *rest])
    return [tuple(x) for x in out]


def chunk_bounds(s: float, e: float, max_len: float = MAX_CHUNK) -> list[tuple[float, float]]:
    n = max(1, math.ceil((e - s) / max_len))
    edges = np.linspace(s, e, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def transcribe_turns(wav: np.ndarray, turns, client: GnaniClient | None, language_code: str = "kn-IN",
                     min_dur: float = 0.3, pad: float = 0.1, cache: ASRCache | None = None) -> list[dict]:
    recs = []
    for i, (s, e, spk, *_) in enumerate(turns):
        rec = {"segment_id": f"seg_{i:03d}", "start_sec": round(s, 2), "end_sec": round(e, 2), "speaker_id": spk,
               "detected_language": None, "transcript": "", "asr_latency_ms": 0.0, "n_chunks": 0, "asr_cached": False, "error": None}
        if e - s < min_dur:
            rec["error"] = "too_short"; recs.append(rec); continue
        parts = []
        for cs, ce in chunk_bounds(s, e):
            a, b = max(0, int((cs - pad) * SR)), min(len(wav), int((ce + pad) * SR))
            k = ASRCache.key(wav[a:b], language_code) if cache else None
            r = cache.get(k) if cache else None
            if r is not None:
                rec["asr_cached"] = True
            else:
                if client is None:
                    rec["error"] = "no Gnani client (GNANI_API_KEY missing)"; break
                try:
                    out = client.transcribe(wav[a:b], language_code)
                except GnaniError as err:
                    rec["error"] = str(err)[:300]; break
                r = {"transcript": out["transcript"], "detected_language": out["detected_language"], "latency_ms": out["latency_ms"]}
                if cache:
                    cache.put(k, r)
                rec["asr_latency_ms"] += r["latency_ms"]
            parts.append(r["transcript"]); rec["detected_language"] = r["detected_language"]; rec["n_chunks"] += 1
        rec["transcript"] = " ".join(p for p in parts if p).strip()
        recs.append(rec)
    return recs
