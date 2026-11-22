"""Stage 8: VAD -> sliding windows -> batched ECAPA-TDNN embeddings for a multi-speaker recording.

Speech regions from Silero VAD are cut into `win`-second windows with `step` hop; regions shorter
than `win` (but ≥ `min_win`) become a single window. Windows are zero-padded into one tensor per
mini-batch and embedded with `encode_batch(wavs, wav_lens)` (relative lengths ⇒ padding ignored).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.features.ecapa import l2_normalize, load_encoder
from src.preprocessing.audio_cleaning import remove_dc
from src.preprocessing.vad import SR, detect_speech


@dataclass
class WindowedEmbeddings:
    windows: np.ndarray      # (n, 2) start/end seconds
    region: np.ndarray       # (n,) index of the VAD region each window came from
    emb: np.ndarray          # (n, 192) L2-normalised
    regions: list[tuple[float, float]]
    step: float


def speech_windows(wav: np.ndarray, sr: int = SR, win: float = 1.5, step: float = 0.75, min_win: float = 0.5,
                   vad_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    segs = detect_speech(wav, sr, threshold=vad_threshold, min_speech_sec=0.25, min_silence_sec=0.15)
    regions = [(s.start_sec, s.end_sec) for s in segs]
    W, R = [], []
    for ri, (s, e) in enumerate(regions):
        dur = e - s
        if dur < min_win:
            continue
        if dur <= win:
            W.append((s, e)); R.append(ri); continue
        starts = np.arange(s, e - win + 1e-9, step)
        if e - (starts[-1] + win) > step / 2:      # cover the tail
            starts = np.r_[starts, e - win]
        W += [(t, t + win) for t in starts]; R += [ri] * len(starts)
    return np.asarray(W, dtype=np.float64).reshape(-1, 2), np.asarray(R, dtype=int), regions


@torch.inference_mode()
def embed_windows(wav: np.ndarray, windows: np.ndarray, sr: int = SR, batch_size: int = 64) -> np.ndarray:
    enc = load_encoder()
    out = []
    for b in range(0, len(windows), batch_size):
        chunk = [wav[int(s * sr):int(e * sr)] for s, e in windows[b:b + batch_size]]
        T = max(len(c) for c in chunk)
        batch = torch.zeros(len(chunk), T)
        for i, c in enumerate(chunk):
            batch[i, :len(c)] = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))
        lens = torch.tensor([len(c) / T for c in chunk])
        out.append(enc.encode_batch(batch.to(enc.device), lens.to(enc.device)).squeeze(1).cpu().numpy())
    return l2_normalize(np.concatenate(out).astype(np.float32)) if out else np.zeros((0, 192), np.float32)


def embed_recording(wav: np.ndarray, sr: int = SR, win: float = 1.5, step: float = 0.75) -> WindowedEmbeddings:
    wav = remove_dc(np.asarray(wav, np.float32))
    W, R, regions = speech_windows(wav, sr, win, step)
    return WindowedEmbeddings(W, R, embed_windows(wav, W, sr), regions, step)
