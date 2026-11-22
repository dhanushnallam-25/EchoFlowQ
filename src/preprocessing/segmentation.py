"""Stage 3 batch runner: load -> DC removal -> Silero VAD -> trim lead/trail silence -> peak norm -> save.

One output segment per clip (first speech onset .. last speech offset). If VAD finds nothing
(very short / quiet utterance), the full clip is kept and flagged `vad_fallback=True`, so no clip is dropped.

Usage (project root):  .venv\\Scripts\\python.exe -m src.preprocessing.segmentation
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data.hf_utils import PROJECT_ROOT
from src.preprocessing.audio_cleaning import peak_normalize, remove_dc
from src.preprocessing.audio_loader import load_audio
from src.preprocessing.vad import SR, detect_speech

META = PROJECT_ROOT / "data/metadata"
OUT_DIR = PROJECT_ROOT / "data/processed/vad_segments"
JOBS = [  # (input csv, output csv, split column present?)
    (META / "splits.csv", META / "vad_segments.csv"),
    (META / "indicvoices_kn_unknown.csv", META / "vad_unknown_segments.csv"),
]
MIN_OUT_SEC = 0.5  # if trimmed span is shorter than this, keep full clip instead


def process_clip(row, a) -> dict:
    wav, sr = load_audio(PROJECT_ROOT / row.filepath)
    wav = remove_dc(wav)
    segs = detect_speech(wav, sr, threshold=a.threshold, min_speech_sec=a.min_speech, min_silence_sec=a.min_silence)

    if segs:
        s, e = segs[0].start_sample, segs[-1].end_sample
        speech = sum(x.end_sample - x.start_sample for x in segs)
    fallback = not segs or (e - s) / sr < MIN_OUT_SEC
    if fallback:
        s, e, speech = 0, len(wav), (speech if segs else 0)

    out = peak_normalize(wav[s:e])
    split = getattr(row, "split", "unknown")
    seg_id = f"{row.clip_id}_seg00"
    path = OUT_DIR / split / row.speaker_label / f"{seg_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, out, sr, subtype="PCM_16")
    return dict(
        segment_id=seg_id, clip_id=row.clip_id, speaker_label=row.speaker_label, split=split,
        start_sec=round(s / sr, 3), end_sec=round(e / sr, 3), vad_duration_sec=round((e - s) / sr, 3),
        orig_duration_sec=round(len(wav) / sr, 3), n_speech_regions=len(segs),
        speech_ratio=round(speech / max(e - s, 1), 3), vad_fallback=fallback,
        vad_filepath=path.relative_to(PROJECT_ROOT).as_posix(),
    )


def verify(df: pd.DataFrame, src: pd.DataFrame, name: str) -> bool:
    checks = [
        (f"{len(df)} / {len(src)} clips produced a segment", len(df) == len(src) and set(df.clip_id) == set(src.clip_id)),
        ("all segments non-empty (vad_duration > 0)", bool((df.vad_duration_sec > 0).all())),
        ("all segment files exist & readable, sr=16k", all(sf.info(PROJECT_ROOT / p).samplerate == SR and sf.info(PROJECT_ROOT / p).frames > 0 for p in df.vad_filepath)),
        ("0 <= start < end <= original duration", bool(((df.start_sec >= 0) & (df.start_sec < df.end_sec) & (df.end_sec <= df.orig_duration_sec + 1e-3)).all())),
        ("segment_id unique", df.segment_id.is_unique),
    ]
    if "split" in src:
        merged = df.merge(src[["clip_id", "split"]], on="clip_id", suffixes=("", "_src"))
        checks.append(("split labels carried over unchanged", bool((merged.split == merged.split_src).all())))
    print(f"\n=== Verification: {name} ===")
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return all(ok for _, ok in checks)


def main(a) -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    all_ok = True
    for in_csv, out_csv in JOBS:
        src = pd.read_csv(in_csv, dtype={"source_recording_id": str, "original_speaker_id": str})
        t = time.perf_counter()
        rows = [process_clip(r, a) for r in tqdm(src.itertuples(index=False), total=len(src), desc=in_csv.name)]
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        el = time.perf_counter() - t

        trimmed = df.orig_duration_sec.sum() - df.vad_duration_sec.sum()
        print(f"\n{in_csv.name}: {len(df)} clips in {el:.1f}s ({el/len(df)*1000:.0f} ms/clip) -> {out_csv.relative_to(PROJECT_ROOT)}")
        print(f"  audio {df.orig_duration_sec.sum()/60:.1f} min -> {df.vad_duration_sec.sum()/60:.1f} min after trim "
              f"(removed {trimmed:.1f}s, {100*trimmed/df.orig_duration_sec.sum():.1f}%)")
        print(f"  vad_duration_sec: min={df.vad_duration_sec.min():.2f} median={df.vad_duration_sec.median():.2f} max={df.vad_duration_sec.max():.2f}")
        print(f"  speech regions/clip: median={df.n_speech_regions.median():.0f} max={df.n_speech_regions.max()}  "
              f"speech_ratio median={df.speech_ratio.median():.2f}  fallbacks={int(df.vad_fallback.sum())}")
        if "split" in src:
            print("  segments per split: " + df.split.value_counts().reindex(["train", "val", "test"]).to_string().replace("\n", "  "))
        all_ok &= verify(df, src, in_csv.name)
    if not all_ok:
        raise SystemExit("Stage 3 verification FAILED")
    print("\nStage 3 OK: every clip has a valid, non-empty VAD segment.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech", type=float, default=0.25)
    ap.add_argument("--min-silence", type=float, default=0.15)
    main(ap.parse_args())
