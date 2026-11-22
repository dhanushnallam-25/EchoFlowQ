"""Pretrained ECAPA-TDNN speaker embeddings (speechbrain/spkrec-ecapa-voxceleb), 192-D, CPU.

Not trained here — used as a frozen feature extractor. `LocalStrategy.COPY` avoids the symlink
step that fails on Windows without admin / Developer Mode.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from src.data.hf_utils import PROJECT_ROOT

SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAVEDIR = PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"
DIM = 192
SR = 16_000


@lru_cache(maxsize=1)
def load_encoder(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    enc = EncoderClassifier.from_hparams(
        source=SOURCE, savedir=str(SAVEDIR), run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    enc.eval()
    return enc


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


@torch.inference_mode()
def embed(wav: np.ndarray, sr: int = SR, normalize: bool = True) -> np.ndarray:
    """16 kHz mono waveform -> (192,) float32 embedding (L2-normalised by default)."""
    if sr != SR:
        raise ValueError(f"ECAPA expects {SR} Hz, got {sr}")
    enc = load_encoder()
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0).to(enc.device)
    e = enc.encode_batch(x).squeeze().cpu().numpy().astype(np.float32)
    return l2_normalize(e) if normalize else e
