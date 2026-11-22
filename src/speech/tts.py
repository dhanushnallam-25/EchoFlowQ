"""Neural text-to-speech for translated turns, with per-speaker voices and content-hash caching.

Backends (first that works wins):
  1. Gnani timbre-v3.0  POST {GNANI_API_ENDPOINT}/api/v1/tts/inference  (verified 200 with the project key).
     34 voices, each speaks all 10 Indic languages incl. Punjabi. One active request per key ⇒ sequential.
  2. edge-tts (Microsoft Edge neural voices; online, unofficial) — no Punjabi voice exists.
A 401/403 from Gnani trips a circuit breaker for the session; other Gnani errors fall back per call.

Speaker distinction: each speaker_id is mapped once to a (gender, voice index). Gender comes from
enrollment metadata when the speaker is enrolled, otherwise genders alternate; further speakers of the
same gender get the next voice in the language's list (edge-tts: same voice with a pitch offset).
Output: 16 kHz mono float32 WAV cached at data/cache/tts/<sha1>.wav.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.hf_utils import PROJECT_ROOT
from src.preprocessing.audio_loader import resample
from src.speech.gnani_client import TTS_MODEL, GnaniClient, GnaniError
from src.speech.languages import LANGUAGES, code_of, gnani_voices

SR = 16_000
CACHE_DIR = PROJECT_ROOT / "data/cache/tts"
EDGE_PITCH = ["+0Hz", "-12Hz", "+12Hz", "-24Hz"]


class Synthesizer:
    def __init__(self, client: GnaniClient | None, cache_dir: Path = CACHE_DIR, speaker_genders: dict[str, str] | None = None):
        self.client, self.cache_dir = client, cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.known_gender = {k: v[0].upper() for k, v in (speaker_genders or {}).items() if v}
        self.assign: dict[str, tuple[str, int]] = {}
        self.gnani_blocked: str | None = None if client else "no Gnani client"
        self.stats = {"gnani": 0, "edge": 0, "cache_hits": 0, "errors": 0}

    # ---------------------------------------------------------------- voices
    def voice_slot(self, speaker_id: str) -> tuple[str, int]:
        if speaker_id not in self.assign:
            g = self.known_gender.get(speaker_id)
            if g not in ("F", "M"):
                n_unknown = sum(1 for s in self.assign if s not in self.known_gender)
                g = "F" if n_unknown % 2 == 0 else "M"
            idx = sum(1 for gg, _ in self.assign.values() if gg == g)
            self.assign[speaker_id] = (g, idx)
        return self.assign[speaker_id]

    def voice_for(self, speaker_id: str, lang: str, backend: str) -> tuple[str, str | None]:
        g, i = self.voice_slot(speaker_id)
        if backend == "gnani":
            pool = gnani_voices(lang, g)
            return pool[i % len(pool)], None
        v = LANGUAGES[lang]["edge"].get(g) or next(iter(LANGUAGES[lang]["edge"].values()), None)
        return v, EDGE_PITCH[i % len(EDGE_PITCH)]

    # ---------------------------------------------------------------- synthesis
    def synthesize(self, text: str, language: str, speaker_id: str = "spk", out_path: Path | None = None) -> dict:
        lang = code_of(language)
        text = (text or "").strip()
        if not text or lang is None:
            return {"path": None, "backend": None, "error": "empty text" if not text else f"unknown language {language!r}"}
        errors = []
        for backend in ("gnani", "edge"):
            if backend == "gnani" and self.gnani_blocked:
                continue
            voice, pitch = self.voice_for(speaker_id, lang, backend)
            if voice is None:
                errors.append(f"{backend}: no {LANGUAGES[lang]['name']} voice"); continue
            key = hashlib.sha1(json.dumps([backend, TTS_MODEL if backend == "gnani" else "edge", voice, pitch, lang, text],
                                          ensure_ascii=False).encode()).hexdigest()
            cached = self.cache_dir / f"{key}.wav"
            meta = {"backend": backend, "voice": voice + (f" ({pitch})" if pitch and pitch != "+0Hz" else ""), "cached": cached.exists()}
            if cached.exists():
                self.stats["cache_hits"] += 1
            else:
                try:
                    wav = self._gnani(text, voice, lang) if backend == "gnani" else self._edge(text, voice, pitch)
                except GnaniError as e:
                    if e.status in (401, 403):
                        self.gnani_blocked = f"HTTP {e.status}"
                    errors.append(f"gnani: HTTP {e.status} {e.body[:120]}"); continue
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{backend}: {type(e).__name__}: {str(e)[:120]}"); continue
                sf.write(cached, wav, SR, subtype="PCM_16")
                self.stats[backend] += 1
            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(cached.read_bytes())
            info = sf.info(cached)
            return meta | {"path": str(out_path or cached), "duration_sec": info.frames / info.samplerate, "error": None}
        self.stats["errors"] += 1
        return {"path": None, "backend": None, "error": "; ".join(errors)}

    def _gnani(self, text: str, voice: str, lang: str) -> np.ndarray:
        r = self.client.synthesize(text, voice, LANGUAGES[lang]["bcp47"], sample_rate=SR)
        return resample(np.asarray(r["wav"], np.float32), r["sr"], SR)

    @staticmethod
    def _edge(text: str, voice: str, pitch: str) -> np.ndarray:
        import edge_tts

        async def run() -> bytes:
            out = b""
            async for ch in edge_tts.Communicate(text, voice, pitch=pitch).stream():
                if ch["type"] == "audio":
                    out += ch["data"]
            return out

        data = asyncio.run(run())
        if not data:
            raise RuntimeError("edge-tts returned no audio")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        wav = wav if wav.ndim == 1 else wav.mean(1)
        return resample(wav, sr, SR)

    @property
    def status(self) -> str:
        return "gnani" if not self.gnani_blocked else f"edge-tts fallback (Gnani TTS: {self.gnani_blocked})"


def assemble_track(clips: list[dict], min_gap: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    """Stitch translated clips into one track, keeping each turn at its original start time where possible.
    clips: [{start_sec, path}] in time order. A clip never overlaps the previous one: it starts at
    max(original start, previous end + min_gap), so long translations push later turns back instead
    of talking over them. Returns (wav, placements)."""
    pieces, place, cursor = [], [], 0.0
    for c in clips:
        if not c.get("path"):
            continue
        w = sf.read(c["path"], dtype="float32")[0]
        start = max(float(c["start_sec"]), cursor + (min_gap if place else 0.0))
        gap = int(round((start - cursor) * SR))
        if gap > 0:
            pieces.append(np.zeros(gap, np.float32))
        pieces.append(w.astype(np.float32))
        cursor = start + len(w) / SR
        place.append({"start_sec": round(start, 3), "end_sec": round(cursor, 3), "src_start_sec": c["start_sec"]})
    pieces.append(np.zeros(int(0.3 * SR), np.float32))
    out = np.concatenate(pieces) if pieces else np.zeros(SR, np.float32)
    peak = float(np.abs(out).max()) or 1.0
    return (out * min(1.0, 0.95 / peak)).astype(np.float32), place
