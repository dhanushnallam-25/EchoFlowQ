"""Stage 2A: inspect ai4bharat/IndicVoices schema by streaming a few Kannada rows.

Usage (from project root):  .venv\\Scripts\\python.exe -m src.data.inspect_indicvoices [--n 50]
Downloads nothing beyond the first streamed shard chunk.
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.data.hf_utils import decode_audio, describe_value, hf_token, stream_split

REPO = "ai4bharat/IndicVoices"


def main(n: int, config: str, split: str | None) -> None:
    from datasets import get_dataset_config_names, get_dataset_split_names

    print(f"=== {REPO} ===  token={'set' if hf_token() else 'MISSING'}")
    configs = get_dataset_config_names(REPO, token=hf_token())
    print(f"configs ({len(configs)}): {configs}")
    if config not in configs:
        raise SystemExit(f"Config '{config}' not found; pick one of the above.")
    splits = get_dataset_split_names(REPO, config, token=hf_token())
    print(f"splits for '{config}': {splits}")
    split = split or ("valid" if "valid" in splits else splits[0])

    ds = stream_split(REPO, config, split)
    print(f"\n--- features ({config}/{split}) ---")
    for k, f in ds.features.items():
        print(f"  {k:28s} {f}")

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i + 1 >= n:
            break

    print(f"\n--- first row (raw) ---")
    for k, v in rows[0].items():
        print(f"  {k:28s} {describe_value(v)}")

    audio_cols = [k for k, v in rows[0].items() if isinstance(v, dict) and ("bytes" in v or "path" in v)]
    print(f"\n--- decoded audio (first 5 rows, cols={audio_cols}) ---")
    for r in rows[:5]:
        for c in audio_cols:
            wav, sr = decode_audio(r[c])
            print(f"  sr={sr}  samples={wav.shape[0]}  dur={wav.shape[0]/sr:.2f}s  peak={abs(wav).max():.3f}")

    print(f"\n--- value counts over {len(rows)} streamed rows (candidate ID/meta columns) ---")
    for k in rows[0]:
        if k in audio_cols:
            continue
        vals = [r[k] for r in rows if not isinstance(r[k], (dict, list))]
        if not vals:
            continue
        c = Counter(map(str, vals))
        print(f"  {k:28s} unique={len(c):4d}  top3={c.most_common(3)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--config", default="kannada")
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    main(a.n, a.config, a.split)
