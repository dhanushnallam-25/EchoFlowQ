"""v2 Upgrade 1: 8-language cross-lingual speaker benchmark + 2 bidirectional test meetings.

Per language: 2 enrolled speakers (13 clips each; 1 female + 1 male where gender is known) and
1 unseen impostor speaker (8 clips) → 34 utterances, 2.5–12 s, 16 kHz mono.
Sources (verified schemas):
  * ai4bharat/IndicVoices  <config>/valid : audio_filepath, speaker_id, gender, duration, normalized/text
    configs: kannada hindi tamil telugu malayalam bengali punjabi
  * openslr/librispeech_asr clean/validation : audio, text, speaker_id, chapter_id, id   (English; no gender)
Download economy: IndicVoices is first scanned metadata-only (parquet column projection) and stops as soon as the
three speakers qualify; the audio pass then stops at the last needed row. LibriSpeech has no duration column,
so it is one audio pass that stops once three speakers qualify. Speakers are the first to qualify in stream
order (deterministic).
Splits (enrolled): per speaker, clips ordered by (recording, chunk) and cut into contiguous blocks
train 9 / val 2 / test 2. Impostors: by language, impostor_cal = {kn, ta, ml, pa} (threshold calibration only),
impostor_test = {hi, te, bn, en} (reporting only).
Clips are processed like Stage 3: DC removal -> Silero VAD lead/trail trim -> peak normalisation.

Usage (project root):  .venv\\Scripts\\python.exe -m src.data.build_multilingual_dataset
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data.hf_utils import PROJECT_ROOT, decode_audio, hf_token, stream_split
from src.preprocessing.audio_cleaning import peak_normalize, remove_dc
from src.preprocessing.audio_loader import resample
from src.preprocessing.vad import detect_speech
from src.speech.languages import LANGUAGES

SR = 16_000
OUT = PROJECT_ROOT / "data/multilingual"
META = PROJECT_ROOT / "data/metadata/multilingual_splits.csv"
MIX_DIR = PROJECT_ROOT / "data/processed/mixtures"
N_ENR, N_IMP = 13, 8
MIN_DUR, MAX_DUR = 2.5, 12.0
IMP_CAL = {"kn", "ta", "ml", "pa"}
CHUNK_RE = re.compile(r"^(?P<rec>.+?)_chunk_(?P<idx>\d+)\.\w+$")


def clean_clip(wav: np.ndarray, sr: int) -> np.ndarray | None:
    wav = remove_dc(resample(np.asarray(wav, np.float32), sr, SR))
    segs = detect_speech(wav, SR)
    if segs:
        wav = wav[segs[0].start_sample: segs[-1].end_sample]
    if len(wav) < 1.5 * SR or np.abs(wav).max() < 1e-3:
        return None
    return peak_normalize(wav)


# ----------------------------------------------------------------------------- IndicVoices
def pick_indic_speakers(config: str) -> dict[str, dict]:
    """Metadata-only scan until 1F + 1M enrolled (≥N_ENR eligible clips) and 1 impostor (≥N_IMP) qualify."""
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/IndicVoices", config, split="valid", streaming=True, token=hf_token(),
                      columns=["speaker_id", "duration", "gender"])
    rows, gender = defaultdict(list), {}
    chosen: dict[str, dict] = {}
    for i, r in enumerate(ds):
        if MIN_DUR <= r["duration"] <= MAX_DUR:
            rows[r["speaker_id"]].append(i)
            gender[r["speaker_id"]] = r["gender"]
        s = r["speaker_id"]
        n = len(rows[s])
        if s in chosen:
            continue
        enr_g = {c["gender"] for c in chosen.values() if c["role"] == "enrolled"}
        g = (gender.get(s) or "?")[0].upper()
        if n >= N_ENR and g in ("F", "M") and g not in enr_g:
            chosen[s] = {"role": "enrolled", "gender": g}
        elif n >= N_IMP and not any(c["role"] == "impostor" for c in chosen.values()) and \
                (sum(c["role"] == "enrolled" for c in chosen.values()) == 2 or n < N_ENR):
            chosen[s] = {"role": "impostor", "gender": g}
        if sum(c["role"] == "enrolled" for c in chosen.values()) == 2 and any(c["role"] == "impostor" for c in chosen.values()):
            break
    for s, c in chosen.items():
        c["rows"] = rows[s][: N_ENR if c["role"] == "enrolled" else N_IMP]
    return chosen


def collect_indic(lang: str) -> list[dict]:
    cfg = LANGUAGES[lang]["indicvoices"]
    chosen = pick_indic_speakers(cfg)
    if sum(c["role"] == "enrolled" for c in chosen.values()) < 2 or not any(c["role"] == "impostor" for c in chosen.values()):
        raise SystemExit(f"{lang}: not enough qualifying speakers in IndicVoices/{cfg}/valid: {chosen.keys()}")
    want = {i: s for s, c in chosen.items() for i in c["rows"]}
    last = max(want)
    recs = []
    for i, r in enumerate(tqdm(stream_split("ai4bharat/IndicVoices", cfg, "valid"), desc=f"{lang} audio", total=last + 1, leave=False)):
        if i > last:
            break
        if i not in want:
            continue
        s = want[i]
        wav, sr = decode_audio(r["audio_filepath"])
        w = clean_clip(wav, sr)
        if w is None:
            continue
        m = CHUNK_RE.match(r["audio_filepath"]["path"])
        rec_id, idx = (m["rec"], int(m["idx"])) if m else (str(s), i)
        recs.append({"original_speaker_id": str(s), "role": chosen[s]["role"], "gender": chosen[s]["gender"], "wav": w,
                     "transcript": (r.get("normalized") or r.get("text") or "").strip(), "source_recording_id": rec_id,
                     "chunk_index": idx, "dataset_source": f"ai4bharat/IndicVoices:{cfg}/valid"})
    return recs


# ----------------------------------------------------------------------------- LibriSpeech
def collect_english() -> list[dict]:
    per = defaultdict(list)
    order: list[str] = []
    for r in tqdm(stream_split("openslr/librispeech_asr", "clean", "validation"), desc="en audio", leave=False):
        s = str(r["speaker_id"])
        need = N_ENR if len(order) < 2 or s in order[:2] else N_IMP
        if s not in order:
            if len(order) == 3:
                if all(len(per[x]) >= (N_ENR if j < 2 else N_IMP) for j, x in enumerate(order)):
                    break
                continue
            order.append(s)
        if len(per[s]) >= need:
            continue
        wav, sr = decode_audio(r["audio"])
        if not (MIN_DUR <= len(wav) / sr <= MAX_DUR):
            continue
        w = clean_clip(wav, sr)
        if w is None:
            continue
        spk, chap, utt = r["id"].split("-")
        per[s].append({"original_speaker_id": s, "role": "enrolled" if order.index(s) < 2 else "impostor", "gender": "?",
                       "wav": w, "transcript": r["text"].strip().capitalize(), "source_recording_id": chap,
                       "chunk_index": int(utt), "dataset_source": "openslr/librispeech_asr:clean/validation"})
    return [x for s in order for x in per[s]]


# ----------------------------------------------------------------------------- write + split
def write_language(lang: str, recs: list[dict]) -> list[dict]:
    rows = []
    enrolled = sorted({r["original_speaker_id"] for r in recs if r["role"] == "enrolled"},
                      key=lambda s: next(i for i, r in enumerate(recs) if r["original_speaker_id"] == s))
    label = {s: f"{lang}_spk{i + 1}" for i, s in enumerate(enrolled)}
    label |= {r["original_speaker_id"]: f"{lang}_imp" for r in recs if r["role"] == "impostor"}
    by_spk = defaultdict(list)
    for r in recs:
        by_spk[r["original_speaker_id"]].append(r)
    for s, rs in by_spk.items():
        rs.sort(key=lambda r: (r["source_recording_id"], r["chunk_index"]))
        n = len(rs)
        if rs[0]["role"] == "enrolled":
            n_te = n_va = max(1, int(np.ceil(0.15 * n)))
            splits = ["train"] * (n - n_va - n_te) + ["val"] * n_va + ["test"] * n_te
        else:
            splits = ["impostor_cal" if lang in IMP_CAL else "impostor_test"] * n
        d = OUT / lang / label[s]
        d.mkdir(parents=True, exist_ok=True)
        for r, sp in zip(rs, splits):
            clip_id = f"{lang}_{r['source_recording_id']}_{r['chunk_index']}"
            p = d / f"{clip_id}.wav"
            sf.write(p, r["wav"], SR, subtype="PCM_16")
            rows.append({"clip_id": clip_id, "language": lang, "language_name": LANGUAGES[lang]["name"],
                         "dataset_source": r["dataset_source"], "original_speaker_id": s, "speaker_label": label[s],
                         "role": r["role"], "gender": r["gender"], "split": sp, "filepath": p.relative_to(PROJECT_ROOT).as_posix(),
                         "duration_sec": round(len(r["wav"]) / SR, 3), "transcript": r["transcript"],
                         "source_recording_id": r["source_recording_id"], "chunk_index": r["chunk_index"]})
    return rows


def build_mixture(df: pd.DataFrame, name: str, speakers: list[str], seed: int) -> dict:
    """Interleave each speaker's TEST (or impostor_test) clips into one conversation; full clips (no truncation)."""
    rng = np.random.default_rng(seed)
    pools = {s: list(rng.permutation(df[(df.speaker_label == s) & df.split.isin(["test", "impostor_test"])].index.tolist()))[:2]
             for s in speakers}
    seq, last = [], None
    while any(pools.values()):
        cands = [s for s in speakers if pools[s] and s != last] or [s for s in speakers if pools[s]]
        s = cands[int(rng.integers(len(cands)))]
        seq.append(pools[s].pop()); last = s
    pieces, turns, t = [], [], 0.4
    noise = lambda n: (rng.standard_normal(n) * 10 ** (-55 / 20)).astype(np.float32)
    pieces.append(noise(int(t * SR)))
    for i in seq:
        r = df.loc[i]
        w = sf.read(PROJECT_ROOT / r.filepath, dtype="float32")[0] * float(rng.uniform(0.6, 1.0))
        turns.append({"start": round(t, 3), "end": round(t + len(w) / SR, 3), "speaker": r.speaker_label, "language": r.language,
                      "role": r.role, "clip_id": r.clip_id, "transcript": r.transcript})
        pieces.append(w + noise(len(w))); t += len(w) / SR
        gap = int(rng.uniform(0.3, 0.8) * SR); pieces.append(noise(gap)); t += gap / SR
    wav = np.concatenate(pieces)
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(MIX_DIR / f"{name}.wav", wav, SR, subtype="PCM_16")
    meta = {"name": name, "duration_sec": round(len(wav) / SR, 2), "speakers": speakers, "turns": turns}
    (MIX_DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    all_rows = []
    for lang in LANGUAGES:
        recs = collect_english() if LANGUAGES[lang].get("librispeech") else collect_indic(lang)
        rows = write_language(lang, recs)
        all_rows += rows
        d = pd.DataFrame(rows)
        print(f"{lang} {LANGUAGES[lang]['name']:9s}: " + ", ".join(
            f"{s}({g.gender.iloc[0]}) {len(g)}" for s, g in d.groupby("speaker_label")) + f"  → {len(d)} clips, {d.duration_sec.sum()/60:.1f} min")
    df = pd.DataFrame(all_rows)
    df.to_csv(META, index=False, encoding="utf-8")

    # verification
    enr = df[df.role == "enrolled"]
    checks = [
        (f"{df.speaker_label.nunique()} speakers = 16 enrolled + 8 impostors",
         enr.speaker_label.nunique() == 16 and df[df.role == "impostor"].speaker_label.nunique() == 8),
        ("every enrolled speaker in train, val and test", all(set(g.split) == {"train", "val", "test"} for _, g in enr.groupby("speaker_label"))),
        ("no clip_id / file in two splits", df.clip_id.is_unique and df.filepath.is_unique),
        ("impostors never in train/val", set(df[df.role == "impostor"].split) <= {"impostor_cal", "impostor_test"}),
        ("splits are contiguous time blocks per speaker",
         all(g.sort_values(["source_recording_id", "chunk_index"]).split.map({"train": 0, "val": 1, "test": 2}).is_monotonic_increasing
             for _, g in enr.groupby("speaker_label"))),
        (f"durations within [{1.5}, {MAX_DUR}] s after VAD trim", bool(df.duration_sec.between(1.5, MAX_DUR + 0.1).all())),
        ("all files exist", all((PROJECT_ROOT / p).exists() for p in df.filepath)),
    ]
    print("\nper split:", df.split.value_counts().to_dict())
    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    m1 = build_mixture(df, "test_mix_kn_to_en", ["kn_spk1", "kn_spk2", "hi_spk1", "hi_spk2", "hi_imp"], seed=101)
    m2 = build_mixture(df, "test_mix_en_to_kn", ["en_spk1", "en_spk2", "kn_spk1", "kn_spk2", "en_imp"], seed=202)
    for m in (m1, m2):
        print(f"{m['name']}: {m['duration_sec']} s, {len(m['turns'])} turns, speakers {m['speakers']}")
    if not all(ok for _, ok in checks):
        raise SystemExit("multilingual dataset verification FAILED")
    print(f"\nSaved {len(df)} clips -> {OUT.relative_to(PROJECT_ROOT)}, {META.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
