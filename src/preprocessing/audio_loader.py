"""Load any audio file -> mono float32 waveform at 16 kHz.

soundfile (libsndfile) handles wav/flac/ogg/mp3; anything else falls back to librosa/audioread.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def to_mono(wav: np.ndarray) -> np.ndarray:
    """(samples,) or (samples, channels) -> (samples,) by channel averaging."""
    return wav if wav.ndim == 1 else wav.mean(axis=1)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)), orig_sr, target_sr).numpy()


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Return (waveform float32 mono in [-1, 1], target_sr)."""
    path = Path(path)
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        import librosa  # broader codec support (m4a, aac, ...) via audioread/ffmpeg

        wav, sr = librosa.load(path, sr=None, mono=False)
        wav = wav.T if wav.ndim == 2 else wav
    wav = resample(to_mono(np.asarray(wav, dtype=np.float32)), int(sr), target_sr)
    return wav.astype(np.float32, copy=False), target_sr
