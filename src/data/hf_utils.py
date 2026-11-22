"""Shared Hugging Face helpers: token loading, streaming, and torchcodec-free audio decoding.

We stream with `Audio(decode=False)` and decode raw bytes with soundfile, which avoids the
torchcodec/FFmpeg dependency that `datasets` uses by default (fragile on Windows + Py3.14).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def hf_token() -> str | None:
    tok = os.getenv("HF_TOKEN")
    return tok.strip() if tok and tok.strip() else None


def stream_split(repo: str, config: str | None, split: str):
    """Stream a split with audio left undecoded (raw bytes)."""
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, config, split=split, streaming=True, token=hf_token())
    for col, feat in (ds.features or {}).items():
        if isinstance(feat, Audio):
            ds = ds.cast_column(col, Audio(decode=False))
    return ds


def decode_audio(audio_field: dict) -> tuple[np.ndarray, int]:
    """Decode an undecoded HF audio dict {'bytes':..., 'path':...} -> (float32 mono array, sr)."""
    raw = audio_field.get("bytes")
    src = io.BytesIO(raw) if raw else audio_field["path"]
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


def describe_value(v, maxlen: int = 80) -> str:
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {describe_value(x, 30)}" for k, x in v.items()) + "}"
    if isinstance(v, (bytes, bytearray)):
        return f"<bytes len={len(v)}>"
    if isinstance(v, (list, tuple)):
        return f"<{type(v).__name__} len={len(v)}> " + (repr(v[:3])[:maxlen] if v else "")
    s = repr(v)
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."
