"""Stage 8 runner: diarization + enrolled-speaker routing, two evaluation tracks.

Track A — synthetic group discussions (deterministic, seed 42):
  VAL mixtures  : val clips (10 enrolled) + unk_01..02  -> tune AHC method/threshold and router τ's
  TEST mixtures : test clips + unk_03..05                -> report (never used for tuning)
  Each: 3–4 enrolled speakers + 1 unknown, 45–60 s, 0.3–1.0 s pauses, per-speaker gain, −55 dBFS noise floor.
  Metrics: DER (collar 0.25 s; miss / FA / confusion), identification error rate with routed labels,
           cluster-routing accuracy, unknown-cluster rejection, known false rejection — per router.
Track B — sarvamai/indic-diarbench Kannada (test), first N recordings, real group conversations:
  unsupervised DER (overlap scored and overlap-excluded), oracle-k DER, and routing false-acceptance
  (none of these speakers is enrolled, so every cluster should be 'Unknown').

Usage (project root):  .venv\\Scripts\\python.exe -m src.diarization.evaluate_diarization [--n-diarbench 4]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

from src.data.hf_utils import PROJECT_ROOT, decode_audio, stream_split
from src.diarization.clustering import centroids, cluster, windows_to_turns
from src.diarization.segment_embedder import embed_recording
from src.diarization.speaker_assigner import UNKNOWN, build_routers
from src.evaluation import plot_style as ps
from src.evaluation.metrics import choose_threshold

SR = 16_000
META, FIG, MET = PROJECT_ROOT / "data/metadata", PROJECT_ROOT / "results/figures", PROJECT_ROOT / "results/metrics"
MIX_DIR = PROJECT_ROOT / "data/processed/mixtures"
DB_DIR = PROJECT_ROOT / "data/raw/indic_diarbench"
COLLAR = 0.25
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.05), 2)
UNK_CAL, UNK_TEST = ["unk_01", "unk_02"], ["unk_03", "unk_04", "unk_05"]
N_MIX = 5


# --------------------------------------------------------------------------- mixtures
def build_mixtures(split: str, unk_speakers: list[str], seed: int) -> list[dict]:
    seg = pd.read_csv(META / "vad_segments.csv")
    unk = pd.read_csv(META / "vad_unknown_segments.csv")
    known = seg[seg.split == split]
    unk = unk[unk.speaker_label.isin(unk_speakers)]
    rng = np.random.default_rng(seed)
    spk_all = sorted(known.speaker_label.unique())
    order = list(rng.permutation(spk_all))
    mixes = []
    for m in range(N_MIX):
        n_known = 3 if m % 2 == 0 else 4
        spks = [order.pop() if order else str(rng.choice(spk_all)) for _ in range(n_known)]
        if not order:
            order = list(rng.permutation(spk_all))
        spks = list(dict.fromkeys(spks))
        u = unk_speakers[m % len(unk_speakers)]
        pools = {s: list(rng.permutation(known[known.speaker_label == s].vad_filepath.tolist())) for s in spks}
        pools[u] = list(rng.permutation(unk[unk.speaker_label == u].vad_filepath.tolist()))
        gains = {s: float(rng.uniform(0.5, 1.0)) for s in pools}

        pieces, ref, clips, t, last = [], [], [], 0.0, None
        noise = lambda n: (rng.standard_normal(n) * 10 ** (-55 / 20)).astype(np.float32)
        while t < 45.0:
            cands = [s for s in pools if pools[s] and s != last]
            if not cands:
                break
            s = str(rng.choice(cands))
            path = pools[s].pop()
            w = sf.read(PROJECT_ROOT / path, dtype="float32")[0][: int(8.0 * SR)] * gains[s]
            if t + len(w) / SR > 60.0:
                w = w[: int((60.0 - t) * SR)]
                if len(w) < 2 * SR:
                    break
            pause = noise(int(rng.uniform(0.3, 1.0) * SR))
            pieces += [pause, w + noise(len(w))]
            t += len(pause) / SR
            ref.append((t, t + len(w) / SR, s))
            clips.append({"start": t, "end": t + len(w) / SR, "speaker": s, "vad_filepath": path,
                          "truncated": len(w) < sf.info(PROJECT_ROOT / path).frames})
            t += len(w) / SR
            last = s
        pieces.append(noise(int(0.5 * SR)))
        wav = np.concatenate(pieces)
        name = f"{split}_mix{m + 1:02d}"
        MIX_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(MIX_DIR / f"{name}.wav", wav, SR, subtype="PCM_16")
        mixes.append({"name": name, "wav": wav, "ref": ref, "clips": clips, "dur": len(wav) / SR, "speakers": spks + [u], "unknown": u})
    return mixes


# --------------------------------------------------------------------------- annotation helpers
def ann(turns, uri, mapper=lambda x: x) -> Annotation:
    a = Annotation(uri=uri)
    for i, (s, e, lab) in enumerate(turns):
        a[Segment(s, e), i] = mapper(lab)
    return a


def der(ref_turns, hyp_turns, dur, uri, skip_overlap=False) -> dict:
    m = DiarizationErrorRate(collar=COLLAR, skip_overlap=skip_overlap)
    d = m(ann(ref_turns, uri), ann(hyp_turns, uri), uem=Timeline([Segment(0, dur)]), detailed=True)
    return {k: float(d[k]) for k in ("missed detection", "false alarm", "confusion", "total")}


def ier(ref_turns, hyp_turns, dur, uri) -> dict:
    m = IdentificationErrorRate(collar=COLLAR)
    d = m(ann(ref_turns, uri), ann(hyp_turns, uri), uem=Timeline([Segment(0, dur)]), detailed=True)
    return {k: float(d[k]) for k in ("missed detection", "false alarm", "confusion", "total")}


def summarize(rows: list[dict]) -> dict:
    tot = sum(r["total"] for r in rows)
    s = {k: sum(r[k] for r in rows) / tot for k in ("missed detection", "false alarm", "confusion")}
    return {"DER": sum(s.values()), "miss": s["missed detection"], "false_alarm": s["false alarm"], "confusion": s["confusion"]}


def majority_ref(turns_hyp, ref, cid) -> tuple[str | None, float]:
    ov = defaultdict(float)
    for s, e, lab in turns_hyp:
        if lab != cid:
            continue
        for rs, re_, rl in ref:
            ov[rl] += max(0.0, min(e, re_) - max(s, rs))
    if not ov:
        return None, 0.0
    best = max(ov, key=ov.get)
    return best, ov[best]


# --------------------------------------------------------------------------- pipeline
MIN_WINDOWS = [1, 3, 5]   # 1 = no small-cluster merging


def diarize(E, method, threshold, k=None, min_windows=1):
    lab = cluster(E.emb, method, threshold, k, min_windows)
    return lab, windows_to_turns(E.windows, E.region, E.regions, lab, E.step)


def choose_clustering(val_mixes, embs) -> tuple[dict, list[dict]]:
    """Grid over (method, threshold, min_windows) on VAL mixtures only.
    Select by val DER, ties -> lower val speaker-count error -> less merging -> threshold method."""
    grid = []
    for mw in MIN_WINDOWS:
        for method, th in [("threshold", t) for t in THRESHOLDS] + [("silhouette", None)]:
            outs = [diarize(embs[m["name"]], method, th, None, mw) for m in val_mixes]
            rows = [der(m["ref"], o[1], m["dur"], m["name"]) for m, o in zip(val_mixes, outs)]
            k_err = sum(abs(len(set(o[0].tolist())) - len(m["speakers"])) for m, o in zip(val_mixes, outs))
            grid.append({"method": method, "threshold": None if th is None else float(th), "min_windows": mw,
                         "k_err": int(k_err), **summarize(rows)})
    best = min(grid, key=lambda r: (round(r["DER"], 4), r["k_err"], r["min_windows"], r["method"] != "threshold"))
    return best, grid


def best_of(grid, pred) -> dict:
    return min((g for g in grid if pred(g)), key=lambda r: (round(r["DER"], 4), r["k_err"], r["min_windows"], r["method"] != "threshold"))


def cluster_table(mixes, embs, cfg) -> list[dict]:
    """One row per hypothesised cluster: centroid + majority reference speaker."""
    out = []
    for m in mixes:
        E = embs[m["name"]]
        lab, turns = diarize(E, cfg["method"], cfg["threshold"], None, cfg["min_windows"])
        for cid, c in centroids(E.emb, lab).items():
            ref_spk, ov = majority_ref(turns, m["ref"], cid)
            out.append({"mix": m["name"], "cid": cid, "centroid": c, "ref": ref_spk, "is_unknown": ref_spk == m["unknown"] or ref_spk is None,
                        "dur": sum(e - s for s, e, l in turns if l == cid)})
    return out


def tune_router_tau(router, val_clusters):
    E = np.stack([c["centroid"] for c in val_clusters])
    idx, s = router.scores(E)
    known = np.array([not c["is_unknown"] for c in val_clusters])
    y = np.array([router.labels.index(c["ref"]) if not c["is_unknown"] else -1 for c in val_clusters])
    return choose_threshold(s[known], np.asarray(idx)[known], y[known], s[~known])


def route_eval(router, mixes, embs, cfg) -> dict:
    ier_rows, n = [], defaultdict(int)
    for m in mixes:
        E = embs[m["name"]]
        lab, turns = diarize(E, cfg["method"], cfg["threshold"], None, cfg["min_windows"])
        cents = centroids(E.emb, lab)
        names, _ = router.assign(np.stack([cents[c] for c in sorted(cents)]))
        routed = dict(zip(sorted(cents), names))
        for cid in sorted(cents):
            ref_spk, _ = majority_ref(turns, m["ref"], cid)
            truth = UNKNOWN if ref_spk in (None, m["unknown"]) else ref_spk
            ok = routed[cid] == truth
            n["clusters"] += 1; n["correct"] += ok
            if truth == UNKNOWN:
                n["unk"] += 1; n["unk_rejected"] += routed[cid] == UNKNOWN
            else:
                n["known"] += 1; n["known_rejected"] += routed[cid] == UNKNOWN
                n["known_misid"] += routed[cid] not in (UNKNOWN, truth)
        ref_id = [(s, e, UNKNOWN if l == m["unknown"] else l) for s, e, l in m["ref"]]
        hyp_id = [(s, e, routed[l]) for s, e, l in turns]
        ier_rows.append(ier(ref_id, hyp_id, m["dur"], m["name"]))
    tot = sum(r["total"] for r in ier_rows)
    return {"tau": router.tau, "cluster_routing_acc": n["correct"] / n["clusters"], "n_clusters": n["clusters"],
            "unknown_rejection_rate": n["unk_rejected"] / max(n["unk"], 1), "n_unknown_clusters": n["unk"],
            "known_false_rejection": n["known_rejected"] / max(n["known"], 1), "known_misid": n["known_misid"] / max(n["known"], 1),
            "IER": sum(r["missed detection"] + r["false alarm"] + r["confusion"] for r in ier_rows) / tot,
            "IER_confusion": sum(r["confusion"] for r in ier_rows) / tot}


# --------------------------------------------------------------------------- Track B data
def load_diarbench(n: int) -> list[dict]:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    cached = sorted(DB_DIR.glob("*.json"))
    if len(cached) >= n:
        for p in cached[:n]:
            meta = json.loads(p.read_text(encoding="utf-8"))
            meta["wav"] = sf.read(p.with_suffix(".wav"), dtype="float32")[0]
            recs.append(meta)
        return recs
    for i, row in enumerate(stream_split("sarvamai/indic-diarbench", "Kannada", "test")):
        if i >= n:
            break
        wav, sr = decode_audio(row["audio"])
        assert sr == SR, sr
        ref = [(float(t["start_time"]), float(t["end_time"]), str(t["speaker_id"])) for t in row["annotated_transcript"]]
        # name by sample_id: recording_id is NOT unique (two different kannada_nf_001 rows exist)
        meta = {"name": row["sample_id"], "recording_id": row["recording_id"], "dur": len(wav) / SR, "num_speakers": int(row["num_speakers"]),
                "dataset_type": row["dataset_type"], "ref": ref}
        sf.write(DB_DIR / f"{meta['name']}.wav", wav, SR, subtype="PCM_16")
        (DB_DIR / f"{meta['name']}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        recs.append(meta | {"wav": wav})
    return recs


# --------------------------------------------------------------------------- figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8f8e88"


def plot_timeline(m, embs, cfg, routers, path):
    plt = ps.plt
    E = embs[m["name"]]
    lab, turns = diarize(E, cfg["method"], cfg["threshold"], None, cfg["min_windows"])
    cents = centroids(E.emb, lab)
    spk_order = [s for s in m["speakers"] if s != m["unknown"]]
    color = {s: PALETTE[i] for i, s in enumerate(spk_order)} | {m["unknown"]: GRAY, UNKNOWN: GRAY}
    cl2ref = {cid: majority_ref(turns, m["ref"], cid)[0] for cid in cents}
    rows = [("Ground truth", [(s, e, l, l if l != m["unknown"] else f"{l} (not enrolled)") for s, e, l in m["ref"]]),
            ("Diarization clusters", [(s, e, cl2ref[l], f"C{l + 1}") for s, e, l in turns])]
    for key, title in (("rbf_192", "Routed · classical RBF-192"), ("quantum_k6", "Routed · quantum ZZ k=6")):
        names, _ = routers[key].assign(np.stack([cents[c] for c in sorted(cents)]))
        r = dict(zip(sorted(cents), names))
        rows.append((title, [(s, e, r[l], r[l]) for s, e, l in turns]))
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for yi, (title, segs) in enumerate(rows):
        y = len(rows) - 1 - yi
        for s, e, ckey, text in segs:
            ax.barh(y, e - s, left=s, height=0.62, color=color.get(ckey, GRAY), edgecolor=ps.SURFACE, linewidth=1.5)
            if e - s > 2.2:
                ax.text(s + 0.15, y, text if yi else text.replace(" (not enrolled)", "*"), va="center", fontsize=7,
                        color="#ffffff" if ckey not in (m["unknown"], UNKNOWN) else ps.TEXT)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0, m["dur"]); ax.set_xlabel("time (s)"); ax.grid(axis="y", visible=False)
    ax.set_title(f"{m['name']}: who spoke when → who is it   (gray = unknown / rejected; * = not enrolled; "
                 f"clusters coloured by their majority true speaker)", loc="left", fontsize=10)
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- main
def main(a) -> None:
    t0 = time.perf_counter()
    val_mixes = build_mixtures("val", UNK_CAL, seed=42)
    test_mixes = build_mixtures("test", UNK_TEST, seed=4242)
    print(f"Track A mixtures: val {len(val_mixes)} ({sum(m['dur'] for m in val_mixes):.0f}s), "
          f"test {len(test_mixes)} ({sum(m['dur'] for m in test_mixes):.0f}s) -> {MIX_DIR.relative_to(PROJECT_ROOT)}")

    embs, t_emb, n_win, audio_s = {}, 0.0, 0, 0.0
    for m in val_mixes + test_mixes:
        t = time.perf_counter(); embs[m["name"]] = embed_recording(m["wav"]); t_emb += time.perf_counter() - t
        n_win += len(embs[m["name"]].emb); audio_s += m["dur"]
    print(f"  embedded {n_win} windows from {audio_s:.0f}s audio in {t_emb:.1f}s "
          f"({t_emb / n_win * 1000:.0f} ms/window, real-time factor {t_emb / audio_s:.3f})")

    # ---- tune on VAL ----
    cfg, grid = choose_clustering(val_mixes, embs)
    variants = {"selected": cfg,
                "silhouette": best_of(grid, lambda g: g["method"] == "silhouette"),
                "no_merge": best_of(grid, lambda g: g["min_windows"] == 1)}
    for name, c in variants.items():
        print(f"[val] {name:10s}: {c['method']}" + (f" @ cos-dist {c['threshold']}" if c["threshold"] else "")
              + f", min_windows={c['min_windows']}  val DER {c['DER']:.3f}, val |k error| {c['k_err']}")
    routers = build_routers()
    val_cl = cluster_table(val_mixes, embs, cfg)
    for r in routers.values():
        r.tau = tune_router_tau(r, val_cl)
    print("[val] router τ: " + ", ".join(f"{k}={r.tau:.3f}" for k, r in routers.items())
          + f"  (from {len(val_cl)} val clusters, {sum(c['is_unknown'] for c in val_cl)} unknown)")

    # ---- TEST (Track A) ----
    resA = {}
    for name, c in variants.items():
        outs = [diarize(embs[m["name"]], c["method"], c["threshold"], None, c["min_windows"]) for m in test_mixes]
        rows = [der(m["ref"], o[1], m["dur"], m["name"]) for m, o in zip(test_mixes, outs)]
        resA[name] = summarize(rows) | {"config": {k: c[k] for k in ("method", "threshold", "min_windows")},
                                        "est_k": [len(set(o[0].tolist())) for o in outs]}
    oracle = [der(m["ref"], diarize(embs[m["name"]], "oracle", 0, len(m["speakers"]))[1], m["dur"], m["name"]) for m in test_mixes]
    resA["oracle_k"] = summarize(oracle)
    est_k = resA["selected"]["est_k"]
    routing = {k: route_eval(r, test_mixes, embs, cfg) for k, r in routers.items()}

    # ---- Track B ----
    print(f"\nTrack B: loading {a.n_diarbench} sarvamai/indic-diarbench Kannada recordings ...")
    recs = load_diarbench(a.n_diarbench)
    resB, accept_B, t_b, aud_b = [], defaultdict(int), 0.0, 0.0
    for rec in recs:
        t = time.perf_counter(); E = embed_recording(rec["wav"]); t_b += time.perf_counter() - t; aud_b += rec["dur"]
        lab, turns = diarize(E, cfg["method"], cfg["threshold"], None, cfg["min_windows"])
        _, turns_o = diarize(E, "oracle", 0, rec["num_speakers"])
        sv, nm = variants["silhouette"], variants["no_merge"]
        _, turns_s = diarize(E, "silhouette", None, None, sv["min_windows"])
        _, turns_n = diarize(E, nm["method"], nm["threshold"], None, 1)
        row = {"name": rec["name"], "dur": rec["dur"], "type": rec["dataset_type"], "true_k": rec["num_speakers"], "est_k": int(len(set(lab))),
               "der": der(rec["ref"], turns, rec["dur"], rec["name"]),
               "der_no_overlap": der(rec["ref"], turns, rec["dur"], rec["name"], skip_overlap=True),
               "der_oracle_k": der(rec["ref"], turns_o, rec["dur"], rec["name"]),
               "der_silhouette": der(rec["ref"], turns_s, rec["dur"], rec["name"]),
               "der_no_merge": der(rec["ref"], turns_n, rec["dur"], rec["name"])}
        ref_tl = Annotation()
        for s, e, l in rec["ref"]:
            ref_tl[Segment(s, e)] = l
        row["overlap_frac"] = float(ref_tl.get_overlap().duration() / max(ref_tl.get_timeline().support().duration(), 1e-9))
        resB.append(row)
        cents = centroids(E.emb, lab)
        for k, r in routers.items():
            names, _ = r.assign(np.stack(list(cents.values())))
            accept_B[k] += sum(n != UNKNOWN for n in names)
        accept_B["_clusters"] += len(cents)
    sB = {key: summarize([r[key] for r in resB]) for key in ("der", "der_no_overlap", "der_no_merge", "der_silhouette", "der_oracle_k")}

    plot_timeline(test_mixes[0], embs, cfg, routers, FIG / "diarization_timeline.png")

    out = {"protocol": __doc__, "collar": COLLAR, "clustering_selected_on_val": cfg, "clustering_val_grid": grid,
           "routers_tau": {k: r.tau for k, r in routers.items()},
           "trackA": {"mixtures": [{k: v for k, v in m.items() if k != "wav"} for m in test_mixes], "der": resA,
                      "estimated_k": est_k, "true_k": [len(m["speakers"]) for m in test_mixes], "routing": routing},
           "trackB": {"recordings": resB, "summary": sB, "false_acceptance_clusters": {k: v / accept_B["_clusters"] for k, v in accept_B.items() if k != "_clusters"},
                      "n_clusters": accept_B["_clusters"]},
           "runtime": {"trackA_embed_s": t_emb, "trackA_windows": n_win, "trackA_audio_s": audio_s, "trackB_embed_s": t_b, "trackB_audio_s": aud_b,
                       "total_s": time.perf_counter() - t0}}
    with open(MET / "diarization_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---- report ----
    print("\n=== Track A — synthetic group discussions (5 test mixtures, collar 0.25 s) ===")
    print(f"  true speakers/mix {out['trackA']['true_k']}  estimated {est_k}")
    for k, v in resA.items():
        print(f"  DER [{k:10s}] {v['DER']*100:5.1f}%  (miss {v['miss']*100:4.1f} | FA {v['false_alarm']*100:4.1f} | confusion {v['confusion']*100:4.1f})"
              + (f"  est k {v['est_k']}  {v['config']}" if "est_k" in v else ""))
    print("\n  routing on identical diarized turns (τ tuned on val mixtures):")
    print(f"  {'router':11s} {'τ':>7s} {'cluster-acc':>11s} {'unk-reject':>10s} {'known-FRR':>9s} {'mis-ID':>7s} {'IER':>6s}")
    for k, r in routing.items():
        print(f"  {k:11s} {r['tau']:7.3f} {r['cluster_routing_acc']:11.3f} {r['unknown_rejection_rate']:10.3f} "
              f"{r['known_false_rejection']:9.3f} {r['known_misid']:7.3f} {r['IER']:6.3f}")
    print(f"  ({routing['cosine']['n_clusters']} test clusters, {routing['cosine']['n_unknown_clusters']} with an unknown majority speaker)")

    print(f"\n=== Track B — indic-diarbench Kannada ({len(resB)} recordings, {aud_b/60:.1f} min; embed RTF {t_b/aud_b:.3f}) ===")
    for r in resB:
        print(f"  {r['name']:16s} {r['type']:11s} {r['dur']:5.0f}s  k true/est {r['true_k']}/{r['est_k']}  overlap {r['overlap_frac']*100:4.1f}%  "
              f"DER {r['der']['total'] and (r['der']['missed detection']+r['der']['false alarm']+r['der']['confusion'])/r['der']['total']*100:5.1f}%")
    for key, lab in (("der", "selected config"), ("der_no_overlap", "overlap excluded"), ("der_no_merge", "no merging"),
                     ("der_silhouette", "silhouette k"), ("der_oracle_k", "oracle k")):
        v = sB[key]
        print(f"  DER [{lab:16s}] {v['DER']*100:5.1f}%  (miss {v['miss']*100:4.1f} | FA {v['false_alarm']*100:4.1f} | confusion {v['confusion']*100:4.1f})")
    print("  false acceptance of non-enrolled real speakers (clusters routed to spk_xx): " +
          ", ".join(f"{k}={v:.2f}" for k, v in out["trackB"]["false_acceptance_clusters"].items()) + f"  ({accept_B['_clusters']} clusters)")
    print(f"\nStage 8 done in {out['runtime']['total_s']:.0f}s -> results/metrics/diarization_results.json, results/figures/diarization_timeline.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-diarbench", type=int, default=4)
    main(ap.parse_args())
