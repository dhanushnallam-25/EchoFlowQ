"""Silero VAD wrapper returning (start_sec, end_sec, start_sample, end_sample) speech segments."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

# silero_vad/model.py calls torch.set_num_threads(1) at import time (a process-wide side effect) which
# silently made every later ECAPA call single-threaded (~3x slower). Import it once here and restore.
_n_threads = torch.get_num_threads()
from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: E402

torch.set_num_threads(_n_threads)

SR = 16_000


@dataclass(frozen=True)
class SpeechSegment:
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@lru_cache(maxsize=1)
def _model():
    return load_silero_vad()


def detect_speech(
    wav: np.ndarray,
    sr: int = SR,
    threshold: float = 0.5,
    min_speech_sec: float = 0.25,
    min_silence_sec: float = 0.15,
    speech_pad_sec: float = 0.03,
) -> list[SpeechSegment]:
    """Run Silero VAD on a 16 kHz mono waveform. Returns segments in time order (may be empty)."""
    if sr != SR:
        raise ValueError(f"Silero VAD here expects {SR} Hz audio, got {sr}")


    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
        _model(),
        threshold=threshold,
        sampling_rate=sr,
        min_speech_duration_ms=int(min_speech_sec * 1000),
        min_silence_duration_ms=int(min_silence_sec * 1000),
        speech_pad_ms=int(speech_pad_sec * 1000),
    )
    return [SpeechSegment(t["start"] / sr, t["end"] / sr, int(t["start"]), int(t["end"])) for t in ts]
