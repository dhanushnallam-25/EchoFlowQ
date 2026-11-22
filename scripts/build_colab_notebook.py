"""Builds notebooks/EchoFlowQ_Colab_LargeScale_Training.ipynb (and a plain-script twin for local smoke tests).

The notebook is self-contained (no dependency on src/). Every cell is plain Python, so the same code runs:
  * on Colab (T4 GPU, full scale), and
  * locally on CPU in SMOKE mode:  EFQ_SMOKE=1 python notebooks/EchoFlowQ_Colab_LargeScale_Training.py
Usage:  .venv\\Scripts\\python.exe scripts\\build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/EchoFlowQ_Colab_LargeScale_Training.ipynb"
PY = ROOT / "notebooks/EchoFlowQ_Colab_LargeScale_Training.py"

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

md(r"""
# EchoFlow-Q v2 — Large-scale 8-language training on Colab (T4 GPU)

**What this notebook does:**
1. Streams about 2,200 utterances across 8 languages (IndicVoices: kn, hi, ta, te, ml, bn, pa; LibriSpeech: en). Audio is held in RAM only.
2. Optionally merges your existing local Kannada speakers (`spk_01`–`spk_10`, plus `unk_01`–`unk_05` as impostors) from `kannada10_embeddings.npz`.
3. Extracts ECAPA-TDNN embeddings on the GPU.
4. Trains Classical RBF-SVM (192-D), RBF-SVM (PCA-6) and a 6-qubit ZZ fidelity-kernel SVM on identical leak-free splits.
5. Tunes the diarization clustering threshold on multilingual validation mixtures and reports DER on test mixtures.
6. Exports **`echoflow_v2_artifacts.zip`**: portable models, metrics, figures, and a manifest with checksums and check vectors.

**Before you run:**
1. Choose *Runtime → Change runtime type → T4 GPU*.
2. Add `HF_TOKEN` under *Secrets* (🔑). The account must have accepted the terms of `ai4bharat/IndicVoices`.
3. Choose *Runtime → Run all*.

The last cell downloads the zip. In the local app, use **Upload `echoflow_v2_artifacts.zip`** in the sidebar, or unzip it into `results/`.
""")

code(r'''
# ── 1. Environment ────────────────────────────────────────────────────────────────────────────
import os, sys, subprocess
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
SMOKE = os.environ.get("EFQ_SMOKE") == "1"          # tiny local CPU run used to verify this notebook

# Pinned to the versions of the local EchoFlow-Q inference environment so pickled models behave identically.
PINS = ["scikit-learn==1.9.1", "qiskit==2.6.2", "speechbrain==1.1.2", "silero-vad==6.2.3",
        "datasets==5.2.0", "pyannote.metrics==4.2.0", "soundfile", "joblib"]
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Colab={IN_COLAB}  SMOKE={SMOKE}  device={DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "  ⚠ no GPU — Runtime → Change runtime type → T4 GPU"))
''')

code(r'''
# ── 2. Configuration ──────────────────────────────────────────────────────────────────────────
from pathlib import Path
LANGS = {  # code: (name, IndicVoices config or None for LibriSpeech)
    "kn": ("Kannada", "kannada"), "hi": ("Hindi", "hindi"), "ta": ("Tamil", "tamil"), "te": ("Telugu", "telugu"),
    "ml": ("Malayalam", "malayalam"), "bn": ("Bengali", "bengali"), "pa": ("Punjabi", "punjabi"), "en": ("English", None)}
N_ENR, ENR_CLIPS = 10, 20          # enrolled speakers per language, clips per enrolled speaker (14 train / 3 val / 3 test)
N_IMP, IMP_CLIPS = 5, 8            # unseen impostor speakers per language, clips each
MIN_DUR, MAX_DUR = 2.5, 12.0       # source-clip duration filter (s)
N_MIX_VAL, N_MIX_TEST = 2, 3       # diarization mixtures per language (val tunes the threshold, test reports DER)
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
C_GRID, GAMMA_MULT = [0.1, 1, 10, 100, 1000], [1/16, 1/4, 1, 4, 16]
THRESHOLDS = [round(x, 2) for x in [0.3 + 0.05 * i for i in range(13)]]
K, SEED, SR = 6, 42, 16_000
if SMOKE:
    LANGS = {k: LANGS[k] for k in ("kn", "en")}
    N_ENR, ENR_CLIPS, N_IMP, IMP_CLIPS, N_MIX_VAL, N_MIX_TEST = 3, 8, 2, 4, 1, 1
    ALPHAS, THRESHOLDS = [0.02, 0.05, 0.1], [0.5, 0.65, 0.75, 0.85]
WORK = Path("/content/efq_work") if IN_COLAB else Path(os.environ.get("EFQ_WORK", "efq_work"))
OUT = WORK / "export"
for p in (WORK, OUT / "models/multilingual", OUT / "metrics", OUT / "figures"):
    p.mkdir(parents=True, exist_ok=True)
print(f"{len(LANGS)} languages · {N_ENR} enrolled × {ENR_CLIPS} clips + {N_IMP} impostors × {IMP_CLIPS} clips per language")
''')

code(r'''
# ── 3. Hugging Face token (Colab Secrets → env → prompt) ──────────────────────────────────────
HF_TOKEN = None
if IN_COLAB:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception as e:  # secret missing or notebook access not granted
        print("Colab secret HF_TOKEN not available:", type(e).__name__)
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
if not HF_TOKEN and IN_COLAB:
    from getpass import getpass
    HF_TOKEN = getpass("Hugging Face read token (input hidden): ").strip()
assert HF_TOKEN, "HF_TOKEN required (IndicVoices is gated)"
print("HF token: set")
''')

code(r'''
# ── 4. Audio helpers (identical preprocessing to the local pipeline) ──────────────────────────
import io, json, math, time, hashlib, zipfile
from collections import defaultdict
import numpy as np, soundfile as sf, torchaudio

_n = torch.get_num_threads()
from silero_vad import load_silero_vad, get_speech_timestamps   # silero sets torch threads to 1 on import
torch.set_num_threads(_n)
VAD = load_silero_vad()

def decode(audio):
    raw = audio.get("bytes")
    w, sr = sf.read(io.BytesIO(raw) if raw else audio["path"], dtype="float32", always_2d=True)
    return w.mean(1), int(sr)

def resample(w, sr):
    return w if sr == SR else torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(w)), sr, SR).numpy()

def vad(w, threshold=0.5):
    ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), VAD, threshold=threshold,
                               sampling_rate=SR, min_speech_duration_ms=250, min_silence_duration_ms=150, speech_pad_ms=30)
    return [(t["start"], t["end"]) for t in ts]

def clean_clip(w, sr):
    """DC removal -> Silero VAD lead/trail trim -> peak normalisation (as src/preprocessing). None if unusable."""
    w = resample(np.asarray(w, np.float32), sr); w = w - w.mean()
    seg = vad(w)
    if seg:
        w = w[seg[0][0]: seg[-1][1]]
    peak = float(np.abs(w).max()) if len(w) else 0.0
    if len(w) < 1.5 * SR or peak < 1e-3:
        return None
    return (w * (0.95 / peak)).astype(np.float32)
print("helpers ready")
''')

code(r'''
# ── 5. Stream speakers: IndicVoices (7 languages) + LibriSpeech (English) ────────────────────
from datasets import load_dataset, Audio
import re
CHUNK = re.compile(r"^(?P<rec>.+?)_chunk_(?P<idx>\d+)\.\w+$")

def stream(repo, cfg, split, columns=None, audio_col=None):
    ds = load_dataset(repo, cfg, split=split, streaming=True, token=HF_TOKEN, columns=columns)
    return ds.cast_column(audio_col, Audio(decode=False)) if audio_col else ds

def pick_indic(cfg, max_scan=150_000):
    """Metadata-only scan (no audio bytes). A speaker is judged when its contiguous run of rows ends.
    Enrolled: ≥ENR_CLIPS eligible clips, gender-balanced; impostors: ≥IMP_CLIPS. Falls back valid → train."""
    chosen, cands = {}, []
    want_g = {"F": N_ENR // 2, "M": N_ENR - N_ENR // 2}
    for split in ("valid", "train"):
        rows, gender, prev = defaultdict(list), {}, None
        def judge(s):
            n, g = len(rows[s]), (gender.get(s) or "?")[:1].upper()
            if s in chosen or n < IMP_CLIPS:
                return
            n_enr_g = sum(c["role"] == "enrolled" and c["gender"] == g for c in chosen.values())
            if n >= ENR_CLIPS and g in want_g and n_enr_g < want_g[g]:
                chosen[s] = {"role": "enrolled", "gender": g, "split": split, "rows": rows[s][: ENR_CLIPS + 4]}
            elif sum(c["role"] == "impostor" for c in chosen.values()) < N_IMP:
                chosen[s] = {"role": "impostor", "gender": g, "split": split, "rows": rows[s][: IMP_CLIPS + 3]}
            elif n >= ENR_CLIPS:
                cands.append((s, g, split, rows[s][: ENR_CLIPS + 4]))
        done = lambda: sum(c["role"] == "enrolled" for c in chosen.values()) >= N_ENR and sum(c["role"] == "impostor" for c in chosen.values()) >= N_IMP
        for i, r in enumerate(stream("ai4bharat/IndicVoices", cfg, split, columns=["speaker_id", "duration", "gender"])):
            s = r["speaker_id"]
            if prev is not None and s != prev:
                judge(prev)
                if done():
                    break
            if MIN_DUR <= r["duration"] <= MAX_DUR:
                rows[s].append(i); gender[s] = r["gender"]
            prev = s
            if i >= max_scan:
                break
        else:
            judge(prev)
        # relax gender balance if one gender is short
        for s, g, sp, rr in cands:
            if sum(c["role"] == "enrolled" for c in chosen.values()) >= N_ENR:
                break
            if s not in chosen:
                chosen[s] = {"role": "enrolled", "gender": g, "split": sp, "rows": rr}
        if done():
            break
    return chosen

def collect_indic(code, cfg):
    chosen = pick_indic(cfg)
    recs = []
    for split in sorted({c["split"] for c in chosen.values()}):
        want = {i: s for s, c in chosen.items() if c["split"] == split for i in c["rows"]}
        need = {s: (ENR_CLIPS if c["role"] == "enrolled" else IMP_CLIPS) for s, c in chosen.items()}
        got, last = defaultdict(int), max(want)
        for i, r in enumerate(stream("ai4bharat/IndicVoices", cfg, split, audio_col="audio_filepath")):
            if i > last:
                break
            s = want.get(i)
            if s is None or got[s] >= need[s]:
                continue
            w = clean_clip(*decode(r["audio_filepath"]))
            if w is None:
                continue
            m = CHUNK.match(r["audio_filepath"]["path"] or "")
            recs.append({"lang": code, "orig": str(s), "role": chosen[s]["role"], "gender": chosen[s]["gender"],
                         "wav": (w * 32767).astype(np.int16), "text": (r.get("normalized") or r.get("text") or "").strip(),
                         "rec": m["rec"] if m else str(s), "idx": int(m["idx"]) if m else i})
            got[s] += 1
    return recs

def collect_english():
    per, order = defaultdict(list), []
    quota = lambda s: ENR_CLIPS if order.index(s) < N_ENR else IMP_CLIPS
    for r in stream("openslr/librispeech_asr", "clean", "validation", audio_col="audio"):
        s = str(r["speaker_id"])
        if s not in order:
            if len(order) >= N_ENR + N_IMP:
                if all(len(per[x]) >= quota(x) for x in order):
                    break
                continue
            order.append(s)
        if len(per[s]) >= quota(s):
            continue
        w, sr = decode(r["audio"])
        if not MIN_DUR <= len(w) / sr <= MAX_DUR:
            continue
        w = clean_clip(w, sr)
        if w is None:
            continue
        _, chap, utt = r["id"].split("-")
        per[s].append({"lang": "en", "orig": s, "role": "enrolled" if order.index(s) < N_ENR else "impostor", "gender": "?",
                       "wav": (w * 32767).astype(np.int16), "text": r["text"].strip().capitalize(), "rec": chap, "idx": int(utt)})
    return [x for s in order for x in per[s]]

RECS = []
for code, (name, cfg) in LANGS.items():
    t = time.time()
    rs = collect_english() if cfg is None else collect_indic(code, cfg)
    by = defaultdict(list)
    for r in rs:
        by[(r["role"], r["orig"])].append(r)
    n_e = sum(1 for (role, _), v in by.items() if role == "enrolled" and len(v) >= max(6, ENR_CLIPS - 4))
    print(f"{code} {name:9s}: {len(rs):4d} clips · {n_e} enrolled · {sum(1 for (role, _) in by if role == 'impostor')} impostors · {time.time() - t:.0f}s")
    RECS += rs
print(f"total streamed utterances: {len(RECS)}  ({sum(len(r['wav']) for r in RECS) / SR / 3600:.2f} h audio held in RAM)")
''')

code(r'''
# ── 6. Labels + leak-free splits ──────────────────────────────────────────────────────────────
import pandas as pd
rows = []
for code in LANGS:
    rs = [r for r in RECS if r["lang"] == code]
    enr = list(dict.fromkeys(r["orig"] for r in rs if r["role"] == "enrolled"))
    imp = list(dict.fromkeys(r["orig"] for r in rs if r["role"] == "impostor"))
    lab = {s: f"{code}_e{j + 1:02d}" for j, s in enumerate(enr)} | {s: f"{code}_i{j + 1:02d}" for j, s in enumerate(imp)}
    for s in enr + imp:
        g = sorted([r for r in rs if r["orig"] == s], key=lambda r: (r["rec"], r["idx"]))
        if s in enr:
            if len(g) < 6:
                continue                                   # too few usable clips after cleaning
            n_te = n_va = max(1, math.ceil(0.15 * len(g)))
            sp = ["train"] * (len(g) - n_va - n_te) + ["val"] * n_va + ["test"] * n_te
        else:
            sp = ["impostor_test" if imp.index(s) % 2 == 0 else "impostor_cal"] * len(g)
        for r, x in zip(g, sp):
            r.update(label=lab[s], split=x)
            rows.append(r)
RECS = rows
META = pd.DataFrame([{k: v for k, v in r.items() if k != "wav"} for r in RECS])
print(META.groupby(["lang", "split"]).size().unstack(fill_value=0))
''')

code(r'''
# ── 7. ECAPA-TDNN embeddings on GPU (batched, length-sorted to minimise padding) ──────────────
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
ENC = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(WORK / "ecapa"),
                                     run_opts={"device": DEVICE}, local_strategy=LocalStrategy.COPY)
ENC.eval()

@torch.inference_mode()
def embed(wavs, bs=64 if DEVICE == "cuda" else 16):
    order = np.argsort([len(w) for w in wavs])
    out = np.zeros((len(wavs), 192), np.float32)
    for b in range(0, len(order), bs):
        idx = order[b: b + bs]
        T = max(len(wavs[i]) for i in idx)
        x = torch.zeros(len(idx), T)
        for j, i in enumerate(idx):
            w = wavs[i].astype(np.float32) / (32767.0 if wavs[i].dtype == np.int16 else 1.0)
            x[j, :len(w)] = torch.from_numpy(w)
        lens = torch.tensor([len(wavs[i]) / T for i in idx])
        e = ENC.encode_batch(x.to(DEVICE), lens.to(DEVICE)).squeeze(1).float().cpu().numpy()
        out[idx] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out

t = time.time()
EMB = embed([r["wav"] for r in RECS])
print(f"{len(EMB)} embeddings in {time.time() - t:.0f}s on {DEVICE}")
''')

code(r'''
# ── 8. Merge existing local Kannada speakers (spk_01–spk_10 + unk_01–unk_05) ────────────────
BASELINE = Path(os.environ.get("EFQ_BASELINE", WORK / "kannada10_embeddings.npz"))
if IN_COLAB and not BASELINE.exists():
    print("Upload results/embeddings/kannada10_embeddings.npz from your local EchoFlow-Q (Cancel to skip):")
    try:
        from google.colab import files
        up = files.upload()
        for name, data in up.items():
            if name.endswith(".npz"):
                BASELINE.write_bytes(data)
    except Exception as e:
        print("upload skipped:", type(e).__name__)
if BASELINE.exists():
    b = np.load(BASELINE)
    base = pd.DataFrame({"lang": "kn", "label": b["speaker_label"], "split": b["split"], "role": b["role"],
                         "gender": b["gender"], "orig": b["speaker_label"], "text": "", "rec": b["clip_id"], "idx": 0})
    META = pd.concat([META, base], ignore_index=True)
    EMB = np.vstack([EMB, b["emb"].astype(np.float32)])
    print(f"merged {len(base)} baseline embeddings: {sorted(set(base.label[base.role == 'enrolled']))}")
else:
    print("no baseline file — training on streamed speakers only")
HAS_AUDIO = np.r_[np.ones(len(RECS), bool), np.zeros(len(META) - len(RECS), bool)]
LABELS = sorted(META.label[META.role == "enrolled"].unique())
LAB = {s: i for i, s in enumerate(LABELS)}
Y = np.array([LAB.get(s, -1) for s in META.label])
M = {s: (META.split == s).to_numpy() for s in ("train", "val", "test", "impostor_cal", "impostor_test")}
print(f"{len(LABELS)} enrolled speakers · {META[META.role == 'impostor'].label.nunique()} impostors · {len(META)} utterances")
''')

code(r'''
# ── 9. Train Classical RBF-SVM (192-D, PCA-6) + Hybrid quantum ZZ fidelity-kernel SVM ────────
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
import itertools

mean, sd = EMB[M["train"]].mean(0), EMB[M["train"]].std(0)
scale = np.where(sd < 1e-8, 1.0, sd)
S = ((EMB - mean) / scale).astype(np.float32)                                     # z-score fit on train only
C = np.cov(S[M["train"]].astype(np.float64), rowvar=False)
lam, V = np.linalg.eigh(C); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
V *= np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])                       # deterministic sign
pca_mean = S[M["train"]].astype(np.float64).mean(0); W = V[:, :K]
Z = ((S - pca_mean) @ W).astype(np.float32)
EVR = float(lam[:K].sum() / lam.sum())
ytr, yva, yte = Y[M["train"]], Y[M["val"]], Y[M["test"]]

def tune_rbf(X):
    g0, grid = 1.0 / (X[M["train"]].shape[1] * X[M["train"]].var()), []
    for Cc, m in itertools.product(C_GRID, GAMMA_MULT):
        p = SVC(kernel="rbf", C=Cc, gamma=g0 * m).fit(X[M["train"]], ytr).predict(X[M["val"]])
        grid.append((round(f1_score(yva, p, average="macro"), 6), round(accuracy_score(yva, p), 6), -Cc, -g0 * m))
    f, a, Cc, g = max(grid)
    return SVC(kernel="rbf", C=-Cc, gamma=-g).fit(X[M["train"]], ytr)

t = time.time(); RBF192 = tune_rbf(S); RBF6 = tune_rbf(Z); t_rbf = time.time() - t

FMAP = zz_feature_map(feature_dimension=K, reps=2, entanglement="full").decompose()
SIGMA = np.maximum(Z[M["train"]].astype(np.float64).std(0), 1e-8)
angles = lambda Zx, a: a * np.pi * np.tanh(np.asarray(Zx, np.float64) / SIGMA)
states = lambda X: np.stack([Statevector(FMAP.assign_parameters(x)).data for x in X])
fid = lambda A, B: np.abs(A.conj() @ B.T) ** 2

t = time.time()
ST = {a: states(angles(Z, a)) for a in ALPHAS}
t_states = time.time() - t
grid = []
for a in ALPHAS:
    Ktr, Kva = fid(ST[a][M["train"]], ST[a][M["train"]]), fid(ST[a][M["val"]], ST[a][M["train"]])
    for Cc in C_GRID:
        p = SVC(kernel="precomputed", C=Cc).fit(Ktr, ytr).predict(Kva)
        grid.append((round(f1_score(yva, p, average="macro"), 6), round(accuracy_score(yva, p), 6), -Cc, -a))
_, _, Cq, aq = max(grid); Cq, ALPHA = -Cq, -aq
KQ_TR = fid(ST[ALPHA][M["train"]], ST[ALPHA][M["train"]])
QSVC = SVC(kernel="precomputed", C=Cq).fit(KQ_TR, ytr)
MU = np.stack([EMB[M["train"]][ytr == i].mean(0) for i in range(len(LABELS))]); MU /= np.linalg.norm(MU, axis=1, keepdims=True)

def scores(key, X_emb):
    """Raw scores for embeddings (same transforms as local inference). Returns (pred index, max score)."""
    Xs = (X_emb - mean) / scale
    if key == "cosine":
        D = X_emb @ MU.T
    elif key == "rbf_192":
        D = RBF192.decision_function(Xs)
    elif key == "rbf_pca6":
        D = RBF6.decision_function(((Xs - pca_mean) @ W).astype(np.float32))
    else:
        Zx = (Xs - pca_mean) @ W
        D = QSVC.decision_function(fid(states(angles(Zx, ALPHA)), ST[ALPHA][M["train"]]))
    return D.argmax(1), D.max(1)

def choose_tau(sk, pk, yk, su):
    c = np.unique(np.r_[sk, su, 0.0])
    v = [0.5 * (np.mean((sk >= t) & (pk == yk)) + np.mean(su < t)) for t in c]
    return float(c[int(np.flatnonzero(np.isclose(v, max(v)))[0])])

def kta(Kx, y):
    n = len(y); H = np.eye(n) - 1.0 / n; L = (y[:, None] == y[None]).astype(float)
    Kc, Lc = H @ Kx @ H, H @ L @ H
    return float((Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))

NAMES = {"cosine": "Cosine (ECAPA-192)", "rbf_192": "Classical RBF-SVM (192-D)", "rbf_pca6": "Classical RBF-SVM (PCA-6)",
         "quantum_k6": "Hybrid Quantum ZZ-SVM (6 qubits)"}
RESULTS, TAU = {}, {}
test_lang = META.lang[M["test"]].to_numpy()
for key in NAMES:
    pv, sv = scores(key, EMB[M["val"]]); _, sc_ = scores(key, EMB[M["impostor_cal"]])
    TAU[key] = tau = choose_tau(sv, pv, yva, sc_)
    pt, st = scores(key, EMB[M["test"]]); _, si = scores(key, EMB[M["impostor_test"]])
    P, R, F, _ = precision_recall_fscore_support(yte, pt, average="macro", zero_division=0)
    wrong = pt != yte
    RESULTS[key] = {"name": NAMES[key], "test_accuracy": float(accuracy_score(yte, pt)), "test_macro_f1": float(F),
        "test_macro_precision": float(P), "test_macro_recall": float(R),
        "per_language_accuracy": {LANGS[l][0]: float(np.mean(pt[test_lang == l] == yte[test_lang == l])) for l in LANGS if (test_lang == l).any()},
        "errors": int(wrong.sum()), "cross_language_errors": int(sum(META.lang[META.label == LABELS[p]].iloc[0] != META.lang[META.label == LABELS[q]].iloc[0] for p, q in zip(pt[wrong], yte[wrong]))),
        "open_set": {"tau": tau, "AUROC_known_vs_unknown": float(roc_auc_score(np.r_[np.ones(len(st)), np.zeros(len(si))], np.r_[st, si])),
                     "CCR": float(np.mean((st >= tau) & (pt == yte))), "FRR": float(np.mean(st < tau)), "TRR": float(np.mean(si < tau)), "FAR": float(np.mean(si >= tau))}}
KTA = {"rbf_192": kta(rbf_kernel(S[M["train"]], gamma=RBF192._gamma), ytr), "rbf_pca6": kta(rbf_kernel(Z[M["train"]], gamma=RBF6._gamma), ytr),
       "quantum_k6": kta(KQ_TR, ytr), "quantum_by_alpha": {str(a): kta(fid(ST[a][M["train"]], ST[a][M["train"]]), ytr) for a in ALPHAS}}
print(f"PCA-6 variance {EVR:.3f} · quantum α={ALPHA} C={Cq} · states {t_states:.0f}s · RBF tuning {t_rbf:.0f}s")
print(pd.DataFrame({r["name"]: {"acc": r["test_accuracy"], "macro-F1": r["test_macro_f1"], "AUROC": r["open_set"]["AUROC_known_vs_unknown"],
      "CCR": r["open_set"]["CCR"], "TRR": r["open_set"]["TRR"], "FAR": r["open_set"]["FAR"], "KTA": KTA.get(k, float("nan"))}
      for k, r in RESULTS.items()}).T.round(3))
''')

code(r'''
# ── 10. Multilingual diarization: tune AHC threshold on val mixtures, DER on test mixtures ───
from sklearn.cluster import AgglomerativeClustering
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
rng = np.random.default_rng(SEED)

def make_mixture(split_known, split_imp, lang):
    m = (META.lang == lang).to_numpy() & HAS_AUDIO
    enr = sorted(META.label[m & (META.split == split_known).to_numpy()].unique())
    imp = sorted(META.label[m & (META.split == split_imp).to_numpy()].unique())
    if len(enr) < 3 or not imp:
        return None
    spk = list(rng.choice(enr, size=min(len(enr), int(rng.integers(3, 5))), replace=False)) + [str(rng.choice(imp))]
    pools = {s: list(rng.permutation(np.flatnonzero(m & (META.label == s).to_numpy())))[:3] for s in spk}
    pieces, ref, t, last = [], [], 0.4, None
    noise = lambda n: (rng.standard_normal(n) * 10 ** (-55 / 20)).astype(np.float32)
    pieces.append(noise(int(0.4 * SR)))
    while t < 45 and any(pools.values()):
        c = [s for s in spk if pools[s] and s != last] or [s for s in spk if pools[s]]
        s = c[int(rng.integers(len(c)))]; i = pools[s].pop()
        w = RECS[i]["wav"].astype(np.float32) / 32767 * float(rng.uniform(0.6, 1.0))
        pieces.append(w + noise(len(w))); ref.append((t, t + len(w) / SR, s)); t += len(w) / SR
        g = int(rng.uniform(0.3, 0.8) * SR); pieces.append(noise(g)); t += g / SR; last = s
    return {"lang": lang, "wav": np.concatenate(pieces), "ref": ref, "speakers": spk}

def windows(w, win=1.5, step=0.75):
    regs = [(a / SR, b / SR) for a, b in vad(w)]
    W, R = [], []
    for ri, (s, e) in enumerate(regs):
        if e - s < 0.5:
            continue
        st = [s] if e - s <= win else list(np.arange(s, e - win + 1e-9, step))
        if e - s > win and e - (st[-1] + win) > step / 2:
            st.append(e - win)
        W += [(x, min(e, x + win)) for x in st]; R += [ri] * len(st)
    return np.array(W).reshape(-1, 2), np.array(R, int), regs

def turns_from(Wn, R, regs, lab, step=0.75):
    sp = [[max(regs[r][0], (s + e) / 2 - step / 2), min(regs[r][1], (s + e) / 2 + step / 2), int(l), r] for (s, e), r, l in zip(Wn, R, lab)]
    for r in set(R.tolist()):
        ix = [i for i, x in enumerate(sp) if x[3] == r]
        sp[ix[0]][0], sp[ix[-1]][1] = regs[r]
        for a, b in zip(ix, ix[1:]):
            sp[a][1] = sp[b][0] = (sp[a][1] + sp[b][0]) / 2
    out = []
    for s, e, l, _ in sorted(sp):
        if out and out[-1][2] == l and s - out[-1][1] < 1e-6:
            out[-1][1] = e
        else:
            out.append([s, e, l])
    return out

def der(ref, hyp, dur, uri):
    a, b = Annotation(uri=uri), Annotation(uri=uri)
    for i, (s, e, l) in enumerate(ref): a[Segment(s, e), i] = l
    for i, (s, e, l) in enumerate(hyp): b[Segment(s, e), i] = l
    d = DiarizationErrorRate(collar=0.25)(a, b, uem=Timeline([Segment(0, dur)]), detailed=True)
    return {k: float(d[k]) for k in ("missed detection", "false alarm", "confusion", "total")}

MIX = {"val": [], "test": []}
for lang in LANGS:
    for split, imp_split, n in (("val", "impostor_cal", N_MIX_VAL), ("test", "impostor_test", N_MIX_TEST)):
        for _ in range(n):
            mx = make_mixture(split, imp_split, lang)
            if mx:
                Wn, R, regs = windows(mx["wav"])
                mx.update(W=Wn, R=R, regs=regs, E=embed([mx["wav"][int(s * SR):int(e * SR)] for s, e in Wn]), dur=len(mx["wav"]) / SR)
                MIX[split].append(mx)

def run_ahc(mx, th):
    lab = np.zeros(len(mx["E"]), int) if len(mx["E"]) < 2 else AgglomerativeClustering(
        n_clusters=None, distance_threshold=th, metric="cosine", linkage="average").fit_predict(mx["E"])
    return turns_from(mx["W"], mx["R"], mx["regs"], lab), len(set(lab.tolist()))

def agg(rows):
    tot = sum(r["total"] for r in rows)
    if not tot:
        return {"DER": float("nan"), "miss": float("nan"), "false_alarm": float("nan"), "confusion": float("nan")}
    return {"DER": sum(r["missed detection"] + r["false alarm"] + r["confusion"] for r in rows) / tot,
            "miss": sum(r["missed detection"] for r in rows) / tot, "false_alarm": sum(r["false alarm"] for r in rows) / tot,
            "confusion": sum(r["confusion"] for r in rows) / tot}

VAL_GRID = {th: agg([der(m["ref"], run_ahc(m, th)[0], m["dur"], f"v{i}") for i, m in enumerate(MIX["val"])])["DER"] for th in THRESHOLDS}
DIAR_TH = min(VAL_GRID, key=lambda th: (round(VAL_GRID[th], 4), th)) if MIX["val"] else 0.75   # 0.75 = local Stage 8 choice
DIAR = {"threshold": DIAR_TH, "val_grid": VAL_GRID, "per_language": {}, "n_val_mixtures": len(MIX["val"]), "n_test_mixtures": len(MIX["test"])}
rows_all = []
for lang in LANGS:
    rows = []
    for i, m in enumerate([m for m in MIX["test"] if m["lang"] == lang]):
        hyp, k = run_ahc(m, DIAR_TH)
        rows.append(der(m["ref"], hyp, m["dur"], f"{lang}{i}"))
    if rows:
        DIAR["per_language"][LANGS[lang][0]] = agg(rows); rows_all += rows
DIAR["overall"] = agg(rows_all)
print(f"AHC threshold (val) = {DIAR_TH} · test DER overall {DIAR['overall']['DER'] * 100:.1f}%")
print(pd.DataFrame(DIAR["per_language"]).T.round(3))
''')

code(r'''
# ── 11. Figures ───────────────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 170})
COL = {"cosine": "#1baf7a", "rbf_192": "#2a78d6", "rbf_pca6": "#eda100", "quantum_k6": "#eb6834"}
FIGS = []

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (metric, lab) in zip(axes, (("test_accuracy", "closed-set accuracy"), ("AUROC_known_vs_unknown", "open-set AUROC"))):
    vals = [RESULTS[k][metric] if metric in RESULTS[k] else RESULTS[k]["open_set"][metric] for k in NAMES]
    ax.bar(range(len(NAMES)), vals, color=[COL[k] for k in NAMES], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(NAMES)), [NAMES[k].replace(" (", "\n(") for k in NAMES], fontsize=8)
    ax.set_ylim(max(0, min(vals) - 0.1), 1.02); ax.set_title(f"{lab} · {len(LABELS)} enrolled speakers, {len(LANGS)} languages", loc="left", fontsize=10)
fig.tight_layout(); p = OUT / "figures/largescale_model_comparison.png"; fig.savefig(p); plt.close(fig); FIGS.append(p)

pl = pd.DataFrame({NAMES[k]: RESULTS[k]["per_language_accuracy"] for k in NAMES})
fig, ax = plt.subplots(figsize=(8, 0.45 * len(pl) + 1.2))
im = ax.imshow(pl.values, cmap="Blues", vmin=max(0, pl.values.min() - 0.05), vmax=1, aspect="auto"); ax.grid(False)
ax.set_xticks(range(pl.shape[1]), [c.replace(" (", "\n(") for c in pl.columns], fontsize=8); ax.set_yticks(range(len(pl)), pl.index)
for i in range(pl.shape[0]):
    for j in range(pl.shape[1]):
        ax.text(j, i, f"{pl.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if pl.values[i, j] > 0.9 else "black")
ax.set_title("Per-language speaker-ID accuracy (test)", loc="left", fontsize=10)
fig.tight_layout(); p = OUT / "figures/largescale_per_language_accuracy.png"; fig.savefig(p); plt.close(fig); FIGS.append(p)

d = pd.DataFrame(DIAR["per_language"]).T if DIAR["per_language"] else pd.DataFrame({"miss": [0.0], "confusion": [0.0], "false_alarm": [0.0]}, index=["n/a"])
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(d.index, d["miss"] * 100, color="#86b6ef", label="missed speech")
ax.bar(d.index, d["confusion"] * 100, bottom=d["miss"] * 100, color="#2a78d6", label="speaker confusion")
ax.bar(d.index, d["false_alarm"] * 100, bottom=(d["miss"] + d["confusion"]) * 100, color="#0d366b", label="false alarm")
ax.set_ylabel("DER (%)"); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Diarization error rate per language (AHC threshold {DIAR_TH}, collar 0.25 s)", loc="left", fontsize=10)
fig.tight_layout(); p = OUT / "figures/largescale_der_per_language.png"; fig.savefig(p); plt.close(fig); FIGS.append(p)

fig, ax = plt.subplots(figsize=(6, 3.4))
xs = [float(a) for a in KTA["quantum_by_alpha"]]
ax.plot(xs, list(KTA["quantum_by_alpha"].values()), "o-", color=COL["quantum_k6"], label="quantum ZZ (6 qubits)")
ax.axhline(KTA["rbf_pca6"], color=COL["rbf_pca6"], ls="--", label="RBF PCA-6"); ax.axhline(KTA["rbf_192"], color=COL["rbf_192"], ls="--", label="RBF 192-D")
ax.set_xscale("log"); ax.set_xlabel("angle scale α"); ax.set_ylabel("centred KTA (train)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Kernel-target alignment vs quantum bandwidth", loc="left", fontsize=10)
fig.tight_layout(); p = OUT / "figures/largescale_kta_vs_alpha.png"; fig.savefig(p); plt.close(fig); FIGS.append(p)
print("figures:", [f.name for f in FIGS])
''')

code(r'''
# ── 12. Export portable models + metrics → echoflow_v2_artifacts.zip ─────────────────────────
import joblib, platform, sklearn, qiskit
chk = np.flatnonzero(M["test"])[:8]
CHECK = {"emb": EMB[chk]} | {k: {"idx": scores(k, EMB[chk])[0], "score": scores(k, EMB[chk])[1]} for k in NAMES}
gender = META[META.role == "enrolled"].drop_duplicates("label").set_index("label")
BUNDLE = {
    "format": "echoflow-routers-v2", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "smoke": SMOKE,
    "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "qiskit": qiskit.__version__,
                 "numpy": np.__version__, "torch": torch.__version__},
    "embedding_model": "speechbrain/spkrec-ecapa-voxceleb", "labels": LABELS,
    "label_language": {s: gender.lang[s] for s in LABELS}, "label_gender": {s: str(gender.gender[s])[:1].upper() for s in LABELS},
    "scaler_mean": mean, "scaler_scale": scale, "pca_mean": pca_mean, "pca_W": W, "mu": MU,
    "rbf_192": RBF192, "rbf_pca6": RBF6,
    "quantum": {"k": K, "reps": 2, "entanglement": "full", "alpha": ALPHA, "sigma": SIGMA,
                "train_states": ST[ALPHA][M["train"]], "svc": QSVC},
    "tau": TAU, "diarization": {"method": "threshold", "threshold": DIAR_TH, "min_windows": 1}, "check": CHECK}
joblib.dump(BUNDLE, OUT / "models/multilingual/echoflow_v2_routers.joblib", compress=3)

METRICS = {"protocol": {"languages": {k: v[0] for k, v in LANGS.items()}, "enrolled_per_language": N_ENR, "clips_per_enrolled": ENR_CLIPS,
                        "impostors_per_language": N_IMP, "clips_per_impostor": IMP_CLIPS, "duration_filter_s": [MIN_DUR, MAX_DUR],
                        "splits": "per enrolled speaker contiguous 70/15/15; impostors alternate impostor_test / impostor_cal by speaker",
                        "tuning": "C/γ and C/α on val; τ on val + impostor_cal; AHC threshold on val mixtures", "smoke": SMOKE},
           "n_utterances": int(len(META)), "n_streamed": len(RECS), "n_enrolled_speakers": len(LABELS),
           "n_impostors": int(META[META.role == "impostor"].label.nunique()), "split_sizes": META.split.value_counts().to_dict(),
           "baseline_merged": bool(BASELINE.exists()), "pca_k6_explained_variance": EVR,
           "hyperparameters": {"rbf_192": {"C": RBF192.C, "gamma": float(RBF192._gamma)}, "rbf_pca6": {"C": RBF6.C, "gamma": float(RBF6._gamma)},
                               "quantum_k6": {"C": Cq, "alpha": ALPHA}},
           "tau": TAU, "KTA_train": KTA, "results": RESULTS, "diarization": DIAR,
           "runtime_s": {"quantum_states": t_states, "rbf_tuning": t_rbf}, "device": DEVICE, "versions": BUNDLE["versions"]}
(OUT / "metrics/largescale_multilingual_results.json").write_text(json.dumps(METRICS, indent=2, default=float), encoding="utf-8")

files_ = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
manifest = {"format": "echoflow-artifacts-v2", "created_utc": BUNDLE["created_utc"], "smoke": SMOKE,
            "files": {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files_},
            "summary": {k: {"accuracy": r["test_accuracy"], "AUROC": r["open_set"]["AUROC_known_vs_unknown"]} for k, r in RESULTS.items()}
                       | {"diarization_test_DER": DIAR["overall"]["DER"], "n_enrolled": len(LABELS)}}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
ZIP = WORK / "echoflow_v2_artifacts.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print(f"{ZIP} · {ZIP.stat().st_size / 1e6:.1f} MB · {len(files_) + 1} files")
if IN_COLAB:
    from google.colab import files
    files.download(str(ZIP))
''')


def build() -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
    NB.parent.mkdir(parents=True, exist_ok=True)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    script = "# Auto-generated from scripts/build_colab_notebook.py — same code as the notebook.\n" + \
             "\n\n".join(src if kind == "code" else "\n".join("# " + l for l in src.splitlines()) for kind, src in CELLS) + "\n"
    PY.write_text(script, encoding="utf-8")
    print(f"wrote {NB.relative_to(ROOT)} ({sum(k == 'code' for k, _ in CELLS)} code cells) and {PY.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
