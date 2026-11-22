"""Stage 2B: curate a 10-speaker Kannada subset (+ a small unknown-speaker set) from IndicVoices.

Verified schema (ai4bharat/IndicVoices, config 'kannada', split 'valid', inspected 2026-11-20):
  audio_filepath {bytes, path='<recording_id>_chunk_<n>.flac'} @ 16 kHz mono
  speaker_id 'S4259...'   text / normalized / verbatim (Kannada)   duration (s)   gender   scenario
Each speaker in 'valid' has a single source recording split into non-overlapping chunks.

Two passes over 'valid' (~425 MB; the 39 GB train split is never touched):
  1. metadata-only scan (no audio bytes) -> choose speakers deterministically (seed 42)
  2. audio scan -> keep only chosen speakers' clips, write 16 kHz mono PCM16 WAVs

Usage (project root):  .venv\\Scripts\\python.exe -m src.data.dataset_loader
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data.hf_utils import PROJECT_ROOT, decode_audio, hf_token, stream_split

REPO, CONFIG, SPLIT = "ai4bharat/IndicVoices", "kannada", "valid"
TARGET_SR = 16_000
CHUNK_RE = re.compile(r"^(?P<rec>.+?)_chunk_(?P<idx>\d+)\.\w+$")

OUT_ENROLLED = PROJECT_ROOT / "data/processed/indicvoices_kn_10spk"
OUT_UNKNOWN = PROJECT_ROOT / "data/processed/indicvoices_kn_unknown"
META_DIR = PROJECT_ROOT / "data/metadata"
CSV_ENROLLED = META_DIR / "indicvoices_kn_10spk.csv"
CSV_UNKNOWN = META_DIR / "indicvoices_kn_unknown.csv"
CSV_INDEX = META_DIR / "indicvoices_kn_valid_index.csv"

META_COLS = ["speaker_id", "duration", "gender", "scenario"]


# ----------------------------------------------------------------------------- pass 1
def scan_metadata() -> pd.DataFrame:
    """Metadata-only scan; parquet column projection means no audio is downloaded."""
    if CSV_INDEX.exists():
        print(f"[pass 1] using cached index {CSV_INDEX.relative_to(PROJECT_ROOT)}")
        return pd.read_csv(CSV_INDEX, dtype={"speaker_id": str})
    from datasets import load_dataset

    ds = load_dataset(REPO, CONFIG, split=SPLIT, streaming=True, token=hf_token(), columns=META_COLS)
    rows = [dict(r, row_idx=i) for i, r in enumerate(tqdm(ds, desc="[pass 1] metadata", unit="row"))]
    df = pd.DataFrame(rows)
    META_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_INDEX, index=False)
    return df


def choose_speakers(df: pd.DataFrame, a) -> tuple[list[str], list[str]]:
    ok = df[(df.duration >= a.min_dur) & (df.duration <= a.max_dur)]
    counts = ok.groupby("speaker_id").size()
    gender = df.groupby("speaker_id").gender.first()
    rng = np.random.default_rng(a.seed)

    eligible = counts[counts >= a.min_clips].index
    # Gender-balanced, seeded draw (half female, half male where possible).
    chosen: list[str] = []
    for g, k in (("Female", a.n_speakers // 2), ("Male", a.n_speakers - a.n_speakers // 2)):
        pool = sorted(s for s in eligible if gender[s] == g)
        chosen += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    if len(chosen) < a.n_speakers:  # top up from either gender if one is short
        rest = sorted(set(eligible) - set(chosen))
        chosen += list(rng.choice(rest, size=a.n_speakers - len(chosen), replace=False))
    print(f"[select] {len(eligible)} speakers have >= {a.min_clips} clips of {a.min_dur}-{a.max_dur}s")

    # Unknown (never-enrolled) speakers for open-set rejection tests.
    unk_pool = sorted(s for s in counts[counts >= a.unknown_clips].index if s not in chosen)
    unknown = list(rng.choice(unk_pool, size=min(a.n_unknown, len(unk_pool)), replace=False)) if a.n_unknown else []
    return sorted(chosen), sorted(unknown)


# ----------------------------------------------------------------------------- pass 2
def to_target_sr(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav
    import torch, torchaudio

    return torchaudio.functional.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()


def collect_audio(wanted: set[str], a) -> dict[str, list[dict]]:
    """Stream 'valid' with audio; keep valid clips of wanted speakers (in memory, bytes only)."""
    ds = stream_split(REPO, CONFIG, SPLIT)
    keep: dict[str, list[dict]] = defaultdict(list)
    rejects = defaultdict(int)
    for r in tqdm(ds, desc="[pass 2] audio", unit="row"):
        spk = r["speaker_id"]
        if spk not in wanted or not (a.min_dur <= r["duration"] <= a.max_dur):
            continue
        m = CHUNK_RE.match(r["audio_filepath"]["path"])
        if not m:
            rejects["bad_path"] += 1
            continue
        wav, sr = decode_audio(r["audio_filepath"])
        wav = to_target_sr(wav, sr)
        dur = len(wav) / TARGET_SR
        if wav.size == 0 or np.abs(wav).max() < 1e-3:
            rejects["silent_or_empty"] += 1
            continue
        if abs(dur - r["duration"]) > 0.1:
            rejects["duration_mismatch"] += 1
            continue
        if not str(r["normalized"] or r["text"] or "").strip():
            rejects["empty_transcript"] += 1
            continue
        keep[spk].append(dict(
            wav=wav, recording=m["rec"], chunk_idx=int(m["idx"]), duration=dur,
            transcript=(r["normalized"] or r["text"]).strip(), gender=r["gender"], scenario=r["scenario"],
        ))
    if rejects:
        print(f"[pass 2] rejected clips: {dict(rejects)}")
    return keep


def write_clips(clips_by_spk: dict[str, list[dict]], spk_labels: dict[str, str], out_dir: Path,
                max_clips: int, seed: int) -> pd.DataFrame:
    if out_dir.exists():
        shutil.rmtree(out_dir)  # always regenerate cleanly; this dir is fully owned by this script
    rng = np.random.default_rng(seed)
    records, seen_md5 = [], set()
    for spk, label in spk_labels.items():
        clips = clips_by_spk.get(spk, [])
        if len(clips) > max_clips:  # seeded subsample, then restore temporal order
            idx = sorted(rng.choice(len(clips), size=max_clips, replace=False))
            clips = [clips[i] for i in idx]
        clips.sort(key=lambda c: (c["recording"], c["chunk_idx"]))
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for c in clips:
            pcm = (np.clip(c["wav"], -1, 1) * 32767).astype(np.int16)
            md5 = hashlib.md5(pcm.tobytes()).hexdigest()
            if md5 in seen_md5:  # identical audio twice -> would be leakage
                continue
            seen_md5.add(md5)
            clip_id = f"{c['recording']}_chunk_{c['chunk_idx']}"
            path = out_dir / label / f"{clip_id}.wav"
            sf.write(path, pcm, TARGET_SR, subtype="PCM_16")
            records.append(dict(
                clip_id=clip_id, dataset_source=f"{REPO}:{CONFIG}/{SPLIT}", language="kn",
                original_speaker_id=spk, speaker_label=label,
                filepath=path.relative_to(PROJECT_ROOT).as_posix(),
                duration_sec=round(c["duration"], 3), sampling_rate=TARGET_SR,
                transcript=c["transcript"], source_recording_id=c["recording"],
                chunk_index=c["chunk_idx"], gender=c["gender"], scenario=c["scenario"], audio_md5=md5,
            ))
    return pd.DataFrame(records)


def main(a) -> None:
    df = scan_metadata()
    print(f"[pass 1] {len(df)} rows, {df.speaker_id.nunique()} speakers in {CONFIG}/{SPLIT}")
    enrolled, unknown = choose_speakers(df, a)

    clips = collect_audio(set(enrolled) | set(unknown), a)

    enr_labels = {s: f"spk_{i:02d}" for i, s in enumerate(enrolled, 1)}
    unk_labels = {s: f"unk_{i:02d}" for i, s in enumerate(unknown, 1)}
    enr = write_clips(clips, enr_labels, OUT_ENROLLED, a.max_clips, a.seed)
    unk = write_clips(clips, unk_labels, OUT_UNKNOWN, a.unknown_clips, a.seed)

    short = enr.groupby("speaker_label").size().loc[lambda s: s < a.min_clips]
    if len(short):
        raise SystemExit(f"Speakers fell below {a.min_clips} valid clips after decoding: {short.to_dict()}")

    enr.to_csv(CSV_ENROLLED, index=False, encoding="utf-8")
    unk.to_csv(CSV_UNKNOWN, index=False, encoding="utf-8")

    summary = enr.groupby(["speaker_label", "original_speaker_id", "gender"]).agg(
        clips=("clip_id", "size"), recordings=("source_recording_id", "nunique"),
        total_min=("duration_sec", lambda d: round(d.sum() / 60, 2)),
        mean_s=("duration_sec", lambda d: round(d.mean(), 2)),
    )
    print("\n=== Enrolled speakers ===")
    print(summary.to_string())
    print(f"\nTotal: {len(enr)} clips, {enr.duration_sec.sum()/60:.1f} min -> {CSV_ENROLLED.relative_to(PROJECT_ROOT)}")
    print(f"Unknown set: {unk.speaker_label.nunique()} speakers, {len(unk)} clips -> {CSV_UNKNOWN.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-speakers", type=int, default=10)
    ap.add_argument("--min-clips", type=int, default=20, help="min valid clips per enrolled speaker (>=20 keeps >=3 test clips)")
    ap.add_argument("--max-clips", type=int, default=30)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--n-unknown", type=int, default=5, help="held-out non-enrolled speakers for open-set tests")
    ap.add_argument("--unknown-clips", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
