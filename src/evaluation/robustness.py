"""Stage 12: robustness of the saved speaker classifiers to noise and short utterances.

Held-out data only: 47 enrolled TEST clips (closed-set accuracy) + 30 clips of never-seen unknown
speakers unk_03..05 (open-set AUROC, threshold-free, raw max decision score — no Platt scaling).
Conditions (applied identically to known and unknown clips, seeded per clip):
  * SNR  : additive white Gaussian noise at exactly 20 / 10 / 5 / 0 dB relative to each clip's power
  * Dur. : first 1 / 2 / 3 / 5 s of each VAD-trimmed clip (shorter clips are used whole; counted)
Each degraded clip is re-embedded with ECAPA-TDNN and scored by every saved model through the same
transforms used in deployment (z-score -> [PCA-6 -> ZZ fidelity kernel]).

Usage (project root):  .venv\\Scripts\\python.exe -m src.evaluation.robustness
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

from src.data.hf_utils import PROJECT_ROOT
from src.diarization.speaker_assigner import build_routers
from src.evaluation import plot_style as ps
from src.features import ecapa

META, FIG, MET = PROJECT_ROOT / "data/metadata", PROJECT_ROOT / "results/figures", PROJECT_ROOT / "results/metrics"
SR = 16_000
SNRS = [20, 10, 5, 0]
DURS = [1.0, 2.0, 3.0, 5.0]
UNK_TEST = ["unk_03", "unk_04", "unk_05"]
MODELS = {"cosine": "Cosine (ECAPA-192)", "rbf_192": "Classical RBF-SVM (192-D)",
          "rbf_pca6": "Classical RBF-SVM (PCA-6)", "quantum_k6": "Quantum ZZ-SVM (6 qubits)"}
COLORS = {"cosine": "#1baf7a", "rbf_192": "#2a78d6", "rbf_pca6": "#eda100", "quantum_k6": "#eb6834"}
MARKERS = {"cosine": "s", "rbf_192": "o", "rbf_pca6": "D", "quantum_k6": "^"}


def add_noise(w: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    p = float(np.mean(w.astype(np.float64) ** 2))
    n = np.random.default_rng(seed).standard_normal(len(w)) * np.sqrt(p / 10 ** (snr_db / 10))
    return (w + n).astype(np.float32)


def load_sets():
    seg = pd.read_csv(META / "vad_segments.csv")
    unk = pd.read_csv(META / "vad_unknown_segments.csv")
    known = seg[seg.split == "test"].reset_index(drop=True)
    unk = unk[unk.speaker_label.isin(UNK_TEST)].reset_index(drop=True)
    wk = [sf.read(PROJECT_ROOT / p, dtype="float32")[0] for p in known.vad_filepath]
    wu = [sf.read(PROJECT_ROOT / p, dtype="float32")[0] for p in unk.vad_filepath]
    return known, wk, wu


def evaluate(routers, Ek, Eu, y) -> dict:
    out = {}
    for k, r in routers.items():
        ik, sk = r.scores(Ek)
        _, su = r.scores(Eu)
        out[k] = {"accuracy": float(np.mean(np.asarray(ik) == y)),
                  "auroc": float(roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su]))}
    return out


def plot(res: dict, path) -> None:
    plt = ps.plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey="row")
    axes_cfg = [("snr", ["clean"] + [f"{s} dB" for s in SNRS], "additive white noise (SNR)"),
                ("dur", [f"{d:g} s" for d in DURS] + ["full"], "test-utterance duration")]
    for c, (axis, conds, xlabel) in enumerate(axes_cfg):
        for r, (metric, ylab) in enumerate((("accuracy", "closed-set accuracy (10 speakers)"), ("auroc", "open-set AUROC (known vs unknown)"))):
            ax = axes[r, c]
            x = np.arange(len(conds))
            for m, name in MODELS.items():
                v = [res[axis][cond][m][metric] for cond in conds]
                ax.plot(x, v, color=COLORS[m], marker=MARKERS[m], ms=7, lw=2, mec=ps.SURFACE, mew=1, label=name)
            ax.set_xticks(x, conds)
            ax.set_ylim(0.0 if metric == "accuracy" else 0.4, 1.02)
            ax.set_title(f"{ylab.split(' (')[0]} vs {xlabel}", loc="left", fontsize=10)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel(xlabel)
            if metric == "auroc":
                ax.axhline(0.5, color=ps.MUTED, lw=1, ls="--")
                ax.text(len(conds) - 1, 0.51, "chance", ha="right", va="bottom", fontsize=8, color=ps.TEXT_2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Robustness on held-out data (47 test utterances, 10 speakers; 30 utterances from 3 never-seen speakers)",
                 x=0.01, y=0.995, ha="left", fontsize=11)
    fig.legend(h, l, loc="upper left", ncol=4, bbox_to_anchor=(0.01, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path); plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    routers = build_routers()
    known, wk, wu = load_sets()
    lab = routers["cosine"].labels
    y = np.array([lab.index(s) for s in known.speaker_label])
    ecapa.load_encoder()
    emb = lambda ws: np.stack([ecapa.embed(w) for w in ws])

    res: dict = {"snr": {}, "dur": {}}
    clean = evaluate(routers, emb(wk), emb(wu), y)
    res["snr"]["clean"] = res["dur"]["full"] = clean
    print(f"clean: " + "  ".join(f"{k} acc {v['accuracy']:.3f} AUROC {v['auroc']:.3f}" for k, v in clean.items()))

    for s in SNRS:
        Ek = emb([add_noise(w, s, 1000 + i) for i, w in enumerate(wk)])
        Eu = emb([add_noise(w, s, 5000 + i) for i, w in enumerate(wu)])
        res["snr"][f"{s} dB"] = evaluate(routers, Ek, Eu, y)
        print(f"SNR {s:>2} dB: " + "  ".join(f"{k} {v['accuracy']:.3f}/{v['auroc']:.3f}" for k, v in res["snr"][f"{s} dB"].items()))

    full_k = np.array([len(w) / SR for w in wk]); full_u = np.array([len(w) / SR for w in wu])
    short_info = {}
    for d in DURS:
        n = int(d * SR)
        res["dur"][f"{d:g} s"] = evaluate(routers, emb([w[:n] for w in wk]), emb([w[:n] for w in wu]), y)
        short_info[f"{d:g} s"] = {"known_shorter_than_d": int((full_k < d).sum()), "unknown_shorter_than_d": int((full_u < d).sum())}
        print(f"dur {d:g} s: " + "  ".join(f"{k} {v['accuracy']:.3f}/{v['auroc']:.3f}" for k, v in res["dur"][f"{d:g} s"].items()))

    # minimum duration for "reliable" ID: smallest tested duration with accuracy ≥ 0.95 and AUROC ≥ 0.95
    min_dur = {}
    for m in MODELS:
        ok = [d for d in DURS if res["dur"][f"{d:g} s"][m]["accuracy"] >= 0.95 and res["dur"][f"{d:g} s"][m]["auroc"] >= 0.95]
        min_dur[m] = min(ok) if ok else None

    plot(res, FIG / "robustness_curves.png")
    out = {"protocol": __doc__, "n_known_test": len(wk), "n_unknown_test": len(wu), "unknown_speakers": UNK_TEST,
           "results": res, "duration_notes": short_info, "min_reliable_duration_s(acc>=0.95 & AUROC>=0.95)": min_dur,
           "clip_duration_stats_s": {"known_min": float(full_k.min()), "known_median": float(np.median(full_k)),
                                     "unknown_min": float(full_u.min()), "unknown_median": float(np.median(full_u))},
           "runtime_s": time.perf_counter() - t0}
    (MET / "robustness_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Summary (accuracy / AUROC) ===")
    conds = [("snr", c) for c in ["clean"] + [f"{s} dB" for s in SNRS]] + [("dur", f"{d:g} s") for d in DURS]
    rows = [{"condition": c, **{MODELS[m]: f"{res[a][c][m]['accuracy']:.3f} / {res[a][c][m]['auroc']:.3f}" for m in MODELS}} for a, c in conds]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nmin reliable duration (acc ≥ .95 & AUROC ≥ .95): {min_dur}")
    print(f"clips shorter than the cut (used whole): {short_info}")
    print(f"Done in {out['runtime_s']:.0f}s -> results/metrics/robustness_results.json, results/figures/robustness_curves.png")


if __name__ == "__main__":
    main()
