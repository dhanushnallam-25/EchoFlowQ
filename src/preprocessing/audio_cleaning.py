"""Waveform cleaning: DC removal, peak / RMS normalisation, pre-emphasis.

All gains are guarded so silent or near-silent input is never blown up into amplified noise.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8
SILENCE_PEAK = 1e-4  # below this peak amplitude (-80 dBFS) a clip is treated as silence


def remove_dc(wav: np.ndarray) -> np.ndarray:
    return wav - wav.mean() if wav.size else wav


def peak_normalize(wav: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak < SILENCE_PEAK:
        return wav  # don't amplify silence
    return wav * (target_peak / (peak + EPS))


def rms_normalize(wav: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 30.0) -> np.ndarray:
    """Scale to a target RMS level; gain capped at `max_gain_db`, output clipped to [-1, 1]."""
    rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
    if rms < SILENCE_PEAK:
        return wav
    gain_db = min(target_dbfs - 20 * np.log10(rms + EPS), max_gain_db)
    return np.clip(wav * 10 ** (gain_db / 20), -1.0, 1.0)


def pre_emphasis(wav: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - coef * x[n-1]  (boosts high frequencies; optional, off by default)."""
    if wav.size == 0:
        return wav
    return np.append(wav[0], wav[1:] - coef * wav[:-1]).astype(wav.dtype)


def clean(wav: np.ndarray, mode: str = "peak", emphasis: bool = False) -> np.ndarray:
    """DC removal -> normalisation ('peak' | 'rms' | 'none') -> optional pre-emphasis."""
    wav = remove_dc(wav.astype(np.float32, copy=False))
    if mode == "peak":
        wav = peak_normalize(wav)
    elif mode == "rms":
        wav = rms_normalize(wav)
    if emphasis:
        wav = pre_emphasis(wav)
    return wav.astype(np.float32, copy=False)
