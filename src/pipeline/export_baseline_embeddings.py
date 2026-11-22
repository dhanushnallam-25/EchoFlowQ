"""Export the existing Kannada-10 speaker embeddings for the Colab large-scale training notebook.

Writes results/embeddings/kannada10_embeddings.npz with the Stage 4 ECAPA-TDNN embeddings (192-D, L2,
same model + preprocessing as the notebook) for spk_01..spk_10 (train/val/test, enrolled) and
unk_01..unk_05 (impostors), plus labels, splits, genders and clip ids. Upload this file in the notebook's
"merge existing speakers" cell.

Usage (project root):  .venv\\Scripts\\python.exe -m src.pipeline.export_baseline_embeddings
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.data.hf_utils import PROJECT_ROOT

FEAT = PROJECT_ROOT / "data/processed/features/ecapa_raw.npz"
OUT = PROJECT_ROOT / "results/embeddings/kannada10_embeddings.npz"


def main() -> None:
    d = np.load(FEAT)
    meta = pd.concat([pd.read_csv(PROJECT_ROOT / "data/metadata/indicvoices_kn_10spk.csv"),
                      pd.read_csv(PROJECT_ROOT / "data/metadata/indicvoices_kn_unknown.csv")])
    gender = meta.drop_duplicates("speaker_label").set_index("speaker_label").gender.str[0].to_dict()
    E, spk, split, ids = [], [], [], []
    for s in ("train", "val", "test", "unknown"):
        E.append(d[f"X_{s}"]); spk += list(d[f"spk_{s}"]); ids += list(d[f"ids_{s}"])
        split += ["impostor_test" if s == "unknown" and str(x) in ("unk_03", "unk_04", "unk_05") else
                  "impostor_cal" if s == "unknown" else s for x in d[f"spk_{s}"]]
    E = np.vstack(E).astype(np.float32)
    spk = np.array(spk, dtype=str)
    out = {"emb": E, "speaker_label": spk, "split": np.array(split, dtype=str), "clip_id": np.array(ids, dtype=str),
           "language": np.array(["kn"] * len(spk)), "role": np.array(["impostor" if s.startswith("unk") else "enrolled" for s in spk]),
           "gender": np.array([gender.get(s, "?") for s in spk]),
           "model": np.array("speechbrain/spkrec-ecapa-voxceleb"), "preprocessing": np.array("dc-removal + silero-vad trim + peak-norm, 16 kHz")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    norms = np.linalg.norm(E, axis=1)
    print(f"{len(E)} embeddings ({E.shape[1]}-D, L2 norm {norms.min():.4f}–{norms.max():.4f}) · "
          f"{len(set(spk[out['role'] == 'enrolled']))} enrolled + {len(set(spk[out['role'] == 'impostor']))} impostors · "
          f"splits {pd.Series(out['split']).value_counts().to_dict()}")
    print(f"-> {OUT.relative_to(PROJECT_ROOT)}  sha1 {hashlib.sha1(OUT.read_bytes()).hexdigest()[:12]}  ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
