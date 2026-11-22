"""MFCC baseline: 20 MFCC + Δ + ΔΔ (60 per frame) -> mean & std over time -> 120-D utterance vector.

Pipeline inside librosa.feature.mfcc:
  STFT (25 ms Hann window, 10 ms hop) -> |X|^2 power spectrum -> Mel filterbank (40 bands)
  -> log (dB) energies -> DCT-II -> keep first 20 cepstral coefficients.
"""
from __future__ import annotations

import librosa
import numpy as np

SR = 16_000
N_MFCC = 20
N_FFT = WIN = int(0.025 * SR)  # 400 samples = 25 ms
HOP = int(0.010 * SR)          # 160 samples = 10 ms
N_MELS = 40
DIM = N_MFCC * 3 * 2           # 120


def frame_features(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return (60, n_frames): MFCC, delta, delta-delta stacked."""
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, window="hann", center=True,
    )
    n = mfcc.shape[1]
    width = min(9, n if n % 2 else n - 1)  # delta window must be odd and <= n_frames
    if width < 3:
        raise ValueError(f"Segment too short for deltas ({n} frames)")
    d1 = librosa.feature.delta(mfcc, width=width, order=1)
    d2 = librosa.feature.delta(mfcc, width=width, order=2)
    return np.vstack([mfcc, d1, d2])


def utterance_vector(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return 120-D [mean(60) | std(60)] summary statistics."""
    f = frame_features(wav, sr)
    return np.concatenate([f.mean(axis=1), f.std(axis=1)]).astype(np.float32)


def feature_names() -> list[str]:
    base = [f"mfcc{i}" for i in range(N_MFCC)] + [f"d_mfcc{i}" for i in range(N_MFCC)] + [f"dd_mfcc{i}" for i in range(N_MFCC)]
    return [f"mean_{b}" for b in base] + [f"std_{b}" for b in base]
