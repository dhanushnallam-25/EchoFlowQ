"""Gnani (vachana.ai) REST client: Prisma STT + Evon chat-completions translation.

Config from .env (never hard-coded, never logged):
  GNANI_API_KEY       -> header  X-API-Key-ID
  GNANI_API_ENDPOINT  -> base URL (default https://api.vachana.ai)
Endpoints (as specified by the project owner):
  POST {base}/stt/v3                multipart: audio_file (WAV), model, language_code
  POST {base}/v1/chat/completions   JSON: model, messages, temperature
Retries 429 / 5xx / connection errors with exponential backoff. On failure the raw status and
response body are surfaced so schema nuances are visible immediately.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

from src.data.hf_utils import PROJECT_ROOT  # noqa: F401  (loads .env)

DEFAULT_BASE = "https://api.vachana.ai"
STT_MODEL = "prisma-v3.0"
LLM_MODEL = "evon-v3.5"
SR = 16_000
MAX_STT_SEC = 30.0

from src.speech.languages import LANGUAGES  # noqa: E402

LANG_NAMES = {k: v["name"] for k, v in LANGUAGES.items()}
TTS_MODEL = "timbre-v3.0"


class GnaniError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(f"{msg} (HTTP {status}): {body[:500]}")
        self.status, self.body = status, body


@dataclass
class CallResult:
    data: Any
    status: int
    latency_ms: float
    attempts: int
    raw: dict = field(repr=False, default_factory=dict)


def _wav_bytes(wav_or_path: str | Path | np.ndarray, sr: int = SR) -> tuple[bytes, float]:
    if isinstance(wav_or_path, (str, Path)):
        wav, sr = sf.read(wav_or_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
    else:
        wav = np.asarray(wav_or_path, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), len(wav) / sr


def _find_text(obj: Any, keys=("transcript", "transcription", "text")) -> str | None:
    """Depth-first search for the first non-empty string under a transcript-like key."""
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for v in obj.values():
            r = _find_text(v, keys)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_text(v, keys)
            if r:
                return r
    return None


class GnaniClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0, max_retries: int = 3):
        self.api_key = api_key or os.getenv("GNANI_API_KEY", "").strip()
        if not self.api_key:
            raise GnaniError("GNANI_API_KEY is not set in .env")
        self.base = (base_url or os.getenv("GNANI_API_ENDPOINT", "").strip() or DEFAULT_BASE).rstrip("/")
        self.timeout, self.max_retries = timeout, max_retries
        self.session = requests.Session()
        self.session.headers["X-API-Key-ID"] = self.api_key

    def __repr__(self) -> str:  # never expose the key
        return f"GnaniClient(base={self.base!r}, key=***)"

    def _post(self, path: str, **kw) -> CallResult:
        url = f"{self.base}{path}"
        last = None
        for attempt in range(1, self.max_retries + 1):
            t = time.perf_counter()
            try:
                r = self.session.post(url, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                last = GnaniError(f"POST {path} connection error: {type(e).__name__}", None, str(e))
            else:
                ms = (time.perf_counter() - t) * 1000
                if r.status_code == 200:
                    try:
                        body = r.json()
                    except ValueError:
                        raise GnaniError(f"POST {path} returned non-JSON", r.status_code, r.text)
                    return CallResult(body, r.status_code, ms, attempt, raw=body if isinstance(body, dict) else {"_": body})
                last = GnaniError(f"POST {path} failed", r.status_code, r.text)
                if r.status_code not in (408, 429) and r.status_code < 500:
                    raise last                                  # client error: retrying won't help
            time.sleep(min(2 ** attempt, 10))
        raise last

    # ------------------------------------------------------------------ STT
    def transcribe_raw(self, wav_or_path, language_code: str = "kn-IN", model: str = STT_MODEL) -> CallResult:
        audio, dur = _wav_bytes(wav_or_path)
        if dur > MAX_STT_SEC:
            raise GnaniError(f"clip is {dur:.1f}s > {MAX_STT_SEC}s REST limit; chunk it first (asr.py does this)")
        return self._post("/stt/v3", files={"audio_file": ("audio.wav", audio, "audio/wav")},
                          data={"model": model, "language_code": language_code})

    def transcribe(self, wav_or_path, language_code: str = "kn-IN", model: str = STT_MODEL) -> dict:
        # Verified schema (2026-11-21): {success, request_id, transcript, model, processing_time,
        #                                end_to_end_latency, detected_language_code}
        res = self.transcribe_raw(wav_or_path, language_code, model)
        d = res.data if isinstance(res.data, dict) else {}
        if d.get("success") is False:
            raise GnaniError("STT returned success=false", res.status, str(d))
        text = d.get("transcript") if isinstance(d.get("transcript"), str) else _find_text(res.data)
        return {"transcript": (text or "").strip(), "detected_language": d.get("detected_language_code") or language_code,
                "latency_ms": res.latency_ms, "server_ms": 1000 * float(d.get("processing_time") or 0),
                "request_id": d.get("request_id"), "attempts": res.attempts, "raw": res.data}

    # ------------------------------------------------------------------ TTS
    def synthesize(self, text: str, voice: str, language_code: str, model: str = TTS_MODEL,
                   sample_rate: int = SR, speed: float = 1.0) -> dict:
        """POST /api/v1/tts/inference (docs.gnani.ai/api/TTS/tts-inference): JSON in, raw WAV bytes out.
        Gnani allows only 1 active TTS request per key, so callers must not parallelise."""
        payload = {"text": text, "model": model, "voice": voice, "language": language_code, "speed": speed,
                   "audio_config": {"sample_rate": sample_rate, "num_channels": 1, "sample_width": 2,
                                    "encoding": "linear_pcm", "container": "wav"}}
        url, last = f"{self.base}/api/v1/tts/inference", None
        for attempt in range(1, self.max_retries + 1):
            t = time.perf_counter()
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last = GnaniError(f"TTS connection error: {type(e).__name__}", None, str(e))
            else:
                if r.status_code == 200 and "json" not in (r.headers.get("content-type") or ""):
                    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
                    return {"wav": wav if wav.ndim == 1 else wav.mean(1), "sr": sr, "latency_ms": (time.perf_counter() - t) * 1000}
                last = GnaniError("TTS failed", r.status_code, r.text)
                if r.status_code not in (408, 429) and r.status_code < 500:
                    raise last
            time.sleep(min(2 ** attempt, 10))
        raise last

    # ------------------------------------------------------------------ translation
    @staticmethod
    def system_prompt(source_lang: str, target_lang: str) -> str:
        return (f"You are a professional translator. Translate the user's {source_lang} text into {target_lang}. "
                f"Output ONLY the {target_lang} translation — no explanations, notes, quotes, transliteration or "
                f"commentary. Preserve meaning, names and numbers. If the text is already in {target_lang}, return it unchanged.")

    def translate_raw(self, text: str, source_lang: str = "Kannada", target_lang: str = "English", model: str = LLM_MODEL) -> CallResult:
        payload = {"model": model, "temperature": 0.0,
                   "messages": [{"role": "system", "content": self.system_prompt(source_lang, target_lang)},
                                {"role": "user", "content": text}]}
        return self._post("/v1/chat/completions", json=payload)

    def translate(self, text: str, source_lang: str = "Kannada", target_lang: str = "English", model: str = LLM_MODEL) -> dict:
        target_lang = LANG_NAMES.get(target_lang, target_lang)
        if not text.strip():
            return {"translation": "", "latency_ms": 0.0, "attempts": 0, "raw": None}
        res = self.translate_raw(text, source_lang, target_lang, model)
        out = None
        try:
            out = res.data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            out = _find_text(res.data, ("content", "translation", "text"))
        if out is None:
            raise GnaniError("could not find translation text in response", res.status, str(res.data))
        return {"translation": out.strip(), "latency_ms": res.latency_ms, "attempts": res.attempts, "raw": res.data}
