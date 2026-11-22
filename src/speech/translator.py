"""Cached multilingual translation between any of the 8 EchoFlow-Q languages.

Backend chain (first that works wins, per segment):
  1. Gnani Evon LLM  (POST /v1/chat/completions) — always tried first while the key allows it.
     A 401/403 (e.g. SCOPE_NOT_ALLOWED: key lacks 'LLM' scope) trips a circuit breaker so we
     don't send a failing request per segment; Gnani is re-tried on the next Translator instance.
  2. Google's public JSON endpoint translate.googleapis.com/translate_a/single (client=gtx).
  3. MyMemory free API (via deep-translator; anonymous quota ~5k chars/day; needs a known source).
  4. deep-translator GoogleTranslator (scrapes translate.google.com/m; usually HTTP 429-blocked here).
  A backend that answers 429 is skipped for the rest of the session. Fallbacks 2–4 are free,
  rate-limited third-party services that receive the transcript text — fine for a demo/UI;
  production should use Gnani Evon (LLM scope) or an official translation API.

The source language is given per call (auto-detected turns may be in any language); when it is
unknown, Google is asked to auto-detect. Source == target returns the text unchanged.
Cache: results/cache/translation_cache.json, key = sha1(text):source:target:backend.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from src.data.hf_utils import PROJECT_ROOT
from src.speech.gnani_client import GnaniClient, GnaniError
from src.speech.languages import CODES, LANGUAGES, code_of, name_of

CACHE = PROJECT_ROOT / "results/cache/translation_cache.json"
SUPPORTED = tuple(CODES)
GTX_URL = "https://translate.googleapis.com/translate_a/single"


def to_code(lang: str) -> str:
    code = code_of(lang)
    if code not in SUPPORTED:
        raise ValueError(f"unsupported language {lang!r}; choose from {SUPPORTED}")
    return code


def _google_deep(text: str, src: str | None, tgt: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=src or "auto", target=tgt).translate(text)


def _google_gtx(text: str, src: str | None, tgt: str) -> str:
    r = requests.get(GTX_URL, params={"client": "gtx", "sl": src or "auto", "tl": tgt, "dt": "t", "q": text}, timeout=20)
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def _mymemory(text: str, src: str | None, tgt: str) -> str:
    """MyMemory free API (anonymous quota ~5k chars/day). Needs an explicit source language."""
    from deep_translator import MyMemoryTranslator

    if src is None:
        raise ValueError("MyMemory needs a known source language")
    code = lambda c: "en-GB" if c == "en" else LANGUAGES[c]["bcp47"]
    return MyMemoryTranslator(source=code(src), target=code(tgt)).translate(text)


class Translator:
    def __init__(self, client: GnaniClient | None, cache_path: Path = CACHE, source_lang: str = "kn"):
        self.client, self.cache_path = client, cache_path
        self.default_source = code_of(source_lang)
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.gnani_blocked: str | None = None if client else "no Gnani client"
        self.dead: set[str] = set()                      # fallbacks that failed hard this session
        self.stats = {"gnani": 0, "google_gtx": 0, "mymemory": 0, "google_deep": 0, "cache_hits": 0, "errors": 0, "same_language": 0}

    @staticmethod
    def key(text: str, source: str | None, target: str, backend: str) -> str:
        return f"{hashlib.sha1(text.strip().encode('utf-8')).hexdigest()}:{source or 'auto'}:{target}:{backend}"

    @property
    def status(self) -> str:
        return "gnani" if not self.gnani_blocked else f"fallback (Gnani: {self.gnani_blocked})"

    def translate(self, text: str, target: str, source: str | None = "default") -> dict:
        target = to_code(target)
        src = self.default_source if source == "default" else code_of(source)
        text = text.strip()
        if not text:
            return {"text": "", "backend": None, "cached": False, "error": None}
        if src == target:
            self.stats["same_language"] += 1
            return {"text": text, "backend": "identity", "cached": False, "error": None}

        # 1) Gnani Evon
        k = self.key(text, src, target, "gnani")
        if k in self.cache:
            self.stats["cache_hits"] += 1
            return {"text": self.cache[k], "backend": "gnani", "cached": True, "error": None}
        if not self.gnani_blocked:
            try:
                r = self.client.translate(text, name_of(src) if src else "the source", name_of(target))
                return self._store(k, r["translation"], "gnani", r["latency_ms"])
            except GnaniError as e:
                if e.status in (401, 403):
                    self.gnani_blocked = f"HTTP {e.status} " + ("SCOPE_NOT_ALLOWED" if "SCOPE" in e.body else e.body[:80])
                else:
                    self.stats["errors"] += 1

        # 2) – 4) free fallbacks
        last_err = None
        for name, fn in (("google_gtx", _google_gtx), ("mymemory", _mymemory), ("google_deep", _google_deep)):
            k = self.key(text, src, target, name)
            if k in self.cache:
                self.stats["cache_hits"] += 1
                return {"text": self.cache[k], "backend": name, "cached": True, "error": None}
            if name in self.dead:
                continue
            t = time.perf_counter()
            try:
                out = fn(text, src, target)
                if out and out.strip():
                    return self._store(k, out.strip(), name, (time.perf_counter() - t) * 1000)
            except Exception as e:  # noqa: BLE001 - any failure -> next backend
                last_err = f"{name}: {type(e).__name__}: {str(e)[:120]}"
                if "TooManyRequests" in type(e).__name__ or "429" in str(e):
                    self.dead.add(name)
        self.stats["errors"] += 1
        return {"text": None, "backend": None, "cached": False, "error": last_err or self.gnani_blocked}

    def translate_records(self, records: list[dict], targets=("en", "hi")) -> list[dict]:
        for rec in records:
            src = code_of(rec.get("detected_language")) or self.default_source
            for t in targets:
                out = self.translate(rec.get("transcript", ""), t, src)
                code = to_code(t)
                rec[f"translation_{code}"] = out["text"]
                rec[f"translation_{code}_backend"] = out["backend"]
                if out["error"]:
                    rec[f"translation_{code}_error"] = out["error"]
        return records

    def _store(self, k: str, text: str, backend: str, latency_ms: float) -> dict:
        self.stats[backend] += 1
        self.cache[k] = text
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"text": text, "backend": backend, "cached": False, "error": None, "latency_ms": latency_ms}
