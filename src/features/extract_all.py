"""Stage 4 runner: MFCC (120-D) + ECAPA-TDNN (192-D) for every VAD segment, raw and Z-scored.

Outputs in data/processed/features/:
  mfcc_raw.npz  mfcc_std.npz  mfcc_scaler.npz
  ecapa_raw.npz ecapa_std.npz ecapa_scaler.npz
Each feature npz holds X_{train,val,test,unknown}, y_{...} (int; unknown = -1), ids_{...}
(segment_id), spk_{...} (speaker_label strings) and label_names (index -> spk_xx).

Usage (project root):  .venv\\Scripts\\python.exe -m src.features.extract_all
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data.hf_utils import PROJECT_ROOT
from src.features import ecapa, mfcc
from src.features.standardization import ZScoreScaler

META = PROJECT_ROOT / "data/metadata"
OUT = PROJECT_ROOT / "data/processed/features"
SPLITS = ("train", "val", "test", "unknown")


def load_segments() -> pd.DataFrame:
    enr = pd.read_csv(META / "vad_segments.csv")
    unk = pd.read_csv(META / "vad_unknown_segments.csv")
    return pd.concat([enr, unk], ignore_index=True)


def extract(df: pd.DataFrame, name: str, fn) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    X = np.stack([fn(sf.read(PROJECT_ROOT / p, dtype="float32")[0]) for p in tqdm(df.vad_filepath, desc=name)])
    return X, time.perf_counter() - t


def pack(df: pd.DataFrame, X: np.ndarray, label_names: list[str]) -> dict[str, np.ndarray]:
    lab = {n: i for i, n in enumerate(label_names)}
    out = {"label_names": np.array(label_names)}
    for s in SPLITS:
        m = (df.split == s).to_numpy()
        out[f"X_{s}"] = X[m]
        out[f"y_{s}"] = np.array([lab.get(x, -1) for x in df.speaker_label[m]], dtype=np.int64)
        out[f"ids_{s}"] = df.segment_id[m].to_numpy().astype(str)
        out[f"spk_{s}"] = df.speaker_label[m].to_numpy().astype(str)
    return out


def standardize(raw: dict, name: str) -> dict:
    sc = ZScoreScaler().fit(raw["X_train"])  # TRAIN ONLY
    sc.save(OUT / f"{name}_scaler.npz")
    std = dict(raw)
    for s in SPLITS:
        std[f"X_{s}"] = sc.transform(raw[f"X_{s}"])
    std["scaler_mean"], std["scaler_scale"] = sc.mean_, sc.scale_
    return std


def report(name: str, raw: dict, std: dict, secs: float, n: int, dim: int) -> list[tuple[str, bool]]:
    print(f"\n=== {name.upper()} ({dim}-D) — {secs:.1f}s total, {secs / n * 1000:.0f} ms/segment on CPU ===")
    print(f"  {'split':8s} {'X_raw':>12s} {'y':>7s} classes")
    for s in SPLITS:
        print(f"  {s:8s} {str(raw[f'X_{s}'].shape):>12s} {str(raw[f'y_{s}'].shape):>7s} {sorted(set(raw[f'y_{s}'].tolist()))}")
    allX = np.vstack([d[f"X_{s}"] for d in (raw, std) for s in SPLITS])
    tr = std["X_train"]
    checks = [
        (f"{name}: feature dim == {dim}", all(raw[f"X_{s}"].shape[1] == dim for s in SPLITS)),
        (f"{name}: no NaN / Inf (raw + std)", bool(np.isfinite(allX).all())),
        (f"{name}: train std mean≈0 (max |μ|={np.abs(tr.mean(0)).max():.1e})", bool(np.abs(tr.mean(0)).max() < 1e-4)),
        (f"{name}: train std σ≈1 (range {tr.std(0).min():.4f}–{tr.std(0).max():.4f})", bool(np.allclose(tr.std(0), 1, atol=1e-3))),
        (f"{name}: val/test NOT exactly 0/1 (scaler not refit; val |μ|max={np.abs(std['X_val'].mean(0)).max():.2f})",
         bool(np.abs(std["X_val"].mean(0)).max() > 1e-3)),
        (f"{name}: unknown labels all -1", bool((raw["y_unknown"] == -1).all())),
    ]
    if name == "ecapa":
        norms = np.linalg.norm(np.vstack([raw[f"X_{s}"] for s in SPLITS]), axis=1)
        checks.append((f"ecapa: raw embeddings unit L2 norm (range {norms.min():.4f}–{norms.max():.4f})", bool(np.allclose(norms, 1, atol=1e-4))))
        # Sanity: cosine similarity same-speaker vs different-speaker (train)
        E, y = raw["X_train"], raw["y_train"]
        S = E @ E.T
        same = S[(y[:, None] == y[None]) & ~np.eye(len(y), dtype=bool)].mean()
        diff = S[y[:, None] != y[None]].mean()
        print(f"  sanity: mean cosine same-speaker={same:.3f}  different-speaker={diff:.3f}")
        checks.append(("ecapa: same-speaker cosine > different-speaker cosine", bool(same > diff)))
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_segments()
    label_names = sorted(df.loc[df.split != "unknown", "speaker_label"].unique())
    print(f"{len(df)} segments: " + ", ".join(f"{s}={int((df.split == s).sum())}" for s in SPLITS))

    t = time.perf_counter()
    ecapa.load_encoder()
    print(f"ECAPA-TDNN loaded in {time.perf_counter() - t:.1f}s")

    checks = []
    for name, fn, dim in (("mfcc", mfcc.utterance_vector, mfcc.DIM), ("ecapa", ecapa.embed, ecapa.DIM)):
        X, secs = extract(df, name, fn)
        raw = pack(df, X, label_names)
        std = standardize(raw, name)
        np.savez_compressed(OUT / f"{name}_raw.npz", **raw)
        np.savez_compressed(OUT / f"{name}_std.npz", **std)
        checks += report(name, raw, std, secs, len(df), dim)

    # Reproducibility: re-extract one segment and compare to saved row
    p = PROJECT_ROOT / df.vad_filepath[0]
    w = sf.read(p, dtype="float32")[0]
    saved = {n: np.load(OUT / f"{n}_raw.npz")["X_train"][0] for n in ("mfcc", "ecapa")}
    checks.append(("deterministic re-extraction (mfcc & ecapa)",
                   np.allclose(mfcc.utterance_vector(w), saved["mfcc"], atol=1e-5) and np.allclose(ecapa.embed(w), saved["ecapa"], atol=1e-5)))
    checks.append(("ids align with segment CSV order",
                   list(np.load(OUT / "mfcc_raw.npz")["ids_train"]) == df.segment_id[df.split == "train"].tolist()))

    print("\n=== Verification ===")
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not all(ok for _, ok in checks):
        raise SystemExit("Stage 4 verification FAILED")
    print(f"\nStage 4 OK -> {OUT.relative_to(PROJECT_ROOT)}: " + ", ".join(sorted(f.name for f in OUT.glob('*.npz'))))


if __name__ == "__main__":
    main()
