"""Stage 2A: inspect sarvamai/indic-diarbench schema by streaming a few Kannada rows.

Usage (from project root):  .venv\\Scripts\\python.exe -m src.data.inspect_diarbench [--n 5]
DiarBench rows are long multi-speaker recordings, so we stream only a handful.
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.data.hf_utils import decode_audio, describe_value, hf_token, stream_split

REPO = "sarvamai/indic-diarbench"


def main(n: int, config: str | None, split: str | None) -> None:
    from datasets import get_dataset_config_names, get_dataset_split_names

    print(f"=== {REPO} ===  token={'set' if hf_token() else 'MISSING'}")
    configs = get_dataset_config_names(REPO, token=hf_token())
    print(f"configs ({len(configs)}): {configs}")
    if config is None:
        kn = [c for c in configs if c.lower() in {"kannada", "kn", "kn-in", "kn_in"}]
        config = kn[0] if kn else configs[0]
    print(f"using config: {config}")
    splits = get_dataset_split_names(REPO, config, token=hf_token())
    print(f"splits: {splits}")
    split = split or ("test" if "test" in splits else splits[0])

    ds = stream_split(REPO, config, split)
    print(f"\n--- features ({config}/{split}) ---")
    for k, f in (ds.features or {}).items():
        print(f"  {k:28s} {f}")

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i + 1 >= n:
            break

    print("\n--- first row (raw) ---")
    for k, v in rows[0].items():
        print(f"  {k:28s} {describe_value(v, 200)}")

    audio_cols = [k for k, v in rows[0].items() if isinstance(v, dict) and ("bytes" in v or "path" in v)]
    print(f"\n--- decoded audio (cols={audio_cols}) ---")
    for r in rows:
        for c in audio_cols:
            wav, sr = decode_audio(r[c])
            print(f"  sr={sr}  samples={wav.shape[0]}  dur={wav.shape[0]/sr:.1f}s")

    # Print nested annotation structures in full for the first row (speaker turns, etc.)
    print("\n--- nested/list fields in first row ---")
    for k, v in rows[0].items():
        if k in audio_cols:
            continue
        if isinstance(v, (list, dict)):
            items = v if isinstance(v, list) else [v]
            print(f"  {k}: {len(items)} item(s); first 5:")
            for it in items[:5]:
                print(f"     {describe_value(it, 200)}")
            if isinstance(v, list) and v and isinstance(v[0], dict):
                for sk in v[0]:
                    vals = [str(x.get(sk)) for x in v if isinstance(x, dict)]
                    if len(set(vals)) < 30:
                        print(f"     [{sk}] counts: {Counter(vals).most_common(10)}")
        elif isinstance(v, str) and len(v) > 200:
            print(f"  {k}: (long string, {len(v)} chars) {v[:400]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    main(a.n, a.config, a.split)
