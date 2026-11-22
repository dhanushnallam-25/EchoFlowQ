"""Stage 2B: leak-free train/val/test split (70/15/15, seed 42) of the curated 10-speaker set.

Strategy (`--strategy auto` picks per the data):
  * recording  - whole source recordings assigned to one split (needs >=3 recordings per speaker).
  * contiguous - per speaker, order clips by (recording, chunk_index) and cut into consecutive
                 70/15/15 blocks. Chunks never overlap in time, so no audio is shared; only the two
                 block boundaries are temporally adjacent. Used when speakers have 1-2 recordings
                 (true for IndicVoices kannada/valid: ~1 recording per speaker).
  * random     - per-speaker stratified random split with the seed.

Usage (project root):  .venv\\Scripts\\python.exe -m src.data.dataset_split [--strategy auto]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.data.hf_utils import PROJECT_ROOT

IN_CSV = PROJECT_ROOT / "data/metadata/indicvoices_kn_10spk.csv"
OUT_CSV = PROJECT_ROOT / "data/metadata/splits.csv"
SPLITS = ("train", "val", "test")


def block_sizes(n: int, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    n_test, n_val = max(1, math.ceil(n * test_frac)), max(1, math.ceil(n * val_frac))
    return n - n_val - n_test, n_val, n_test


def split_contiguous(g: pd.DataFrame, a, rng) -> list[str]:
    n_tr, n_va, n_te = block_sizes(len(g), a.val, a.test)
    return ["train"] * n_tr + ["val"] * n_va + ["test"] * n_te  # g is already time-ordered


def split_random(g: pd.DataFrame, a, rng) -> list[str]:
    n_tr, n_va, n_te = block_sizes(len(g), a.val, a.test)
    labels = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
    return list(rng.permutation(labels))


def split_recording(g: pd.DataFrame, a, rng) -> list[str]:
    recs = list(rng.permutation(sorted(g.source_recording_id.unique())))
    n_tr, n_va, n_te = block_sizes(len(recs), a.val, a.test)
    rec2split = dict(zip(recs, ["train"] * n_tr + ["val"] * n_va + ["test"] * n_te))
    return [rec2split[r] for r in g.source_recording_id]


def main(a) -> None:
    df = pd.read_csv(IN_CSV, dtype={"original_speaker_id": str, "source_recording_id": str})
    df = df.sort_values(["speaker_label", "source_recording_id", "chunk_index"]).reset_index(drop=True)
    recs_per_spk = df.groupby("speaker_label").source_recording_id.nunique()

    strategy = a.strategy
    if strategy == "auto":
        strategy = "recording" if recs_per_spk.min() >= 3 else "contiguous"
    print(f"recordings per speaker: min={recs_per_spk.min()} max={recs_per_spk.max()} -> strategy='{strategy}'")

    fn = {"recording": split_recording, "contiguous": split_contiguous, "random": split_random}[strategy]
    rng = np.random.default_rng(a.seed)
    df["split"] = ""
    for spk, g in df.groupby("speaker_label", sort=True):
        df.loc[g.index, "split"] = fn(g, a, rng)
    df["split_strategy"] = strategy
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    verify(df, strategy)
    print(f"\nSaved -> {OUT_CSV.relative_to(PROJECT_ROOT)}")


def verify(df: pd.DataFrame, strategy: str) -> None:
    print("\n=== Split balance (clips per speaker) ===")
    table = df.pivot_table(index="speaker_label", columns="split", values="clip_id", aggfunc="size", fill_value=0)[list(SPLITS)]
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())
    dur = df.groupby("split").duration_sec.sum().div(60).round(1)[list(SPLITS)]
    pct = (df.split.value_counts(normalize=True) * 100).round(1)[list(SPLITS)]
    print("\nminutes: " + "  ".join(f"{s}={dur[s]}" for s in SPLITS))
    print("share %: " + "  ".join(f"{s}={pct[s]}" for s in SPLITS))

    checks: list[tuple[str, bool]] = []
    sets = {s: df[df.split == s] for s in SPLITS}
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            for col in ("clip_id", "filepath", "audio_md5"):
                ov = set(sets[s1][col]) & set(sets[s2][col])
                checks.append((f"{s1} ∩ {s2} on {col:9s} = {len(ov)}", not ov))
    checks.append(("every row assigned exactly one split", df.split.isin(SPLITS).all() and df.clip_id.is_unique))
    checks.append(("all 10 speakers in every split",
                   all(sets[s].speaker_label.nunique() == df.speaker_label.nunique() for s in SPLITS)))
    checks.append(("all audio files exist on disk", all((PROJECT_ROOT / p).exists() for p in df.filepath)))
    if strategy == "recording":
        for i, s1 in enumerate(SPLITS):
            for s2 in SPLITS[i + 1:]:
                ov = set(sets[s1].source_recording_id) & set(sets[s2].source_recording_id)
                checks.append((f"{s1} ∩ {s2} on recording = {len(ov)}", not ov))
    if strategy == "contiguous":
        # temporal order per speaker+recording must be train < val < test (no interleaving)
        order = {"train": 0, "val": 1, "test": 2}
        mono = all(g.sort_values("chunk_index").split.map(order).is_monotonic_increasing
                   for _, g in df.groupby(["speaker_label", "source_recording_id"]))
        checks.append(("splits are non-interleaved time blocks per recording", mono))

    print("\n=== Leakage verification ===")
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not all(ok for _, ok in checks):
        raise SystemExit("Split verification FAILED")
    print("  => ZERO OVERLAP confirmed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["auto", "recording", "contiguous", "random"], default="auto")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
