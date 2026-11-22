"""EchoFlow-Q — Streamlit UI.

Run from the project root:
    .venv\\Scripts\\streamlit run app\\streamlit_app.py
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diarization.speaker_assigner import MODEL_SETS, ArtifactError, available_model_sets  # noqa: E402
from src.pipeline.artifacts import install_artifacts  # noqa: E402
from src.pipeline.echoflow_pipeline import CLASSIFIERS, SOURCE_LANGUAGES, TARGET_LANGUAGES, EchoFlowPipeline  # noqa: E402

MIX_DIR = ROOT / "data/processed/mixtures"
DB_DIR = ROOT / "data/raw/indic_diarbench"
UPLOAD_DIR = ROOT / "data/processed/uploads"
FIG, MET = ROOT / "results/figures", ROOT / "results/metrics"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNKNOWN_COLOR = "#8f8e88"

st.set_page_config(page_title="EchoFlow-Q", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.efq-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .75rem .9rem; height: 100%;}
.efq-card .spk {font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: .45rem;}
.efq-card .dot {width: .7rem; height: .7rem; border-radius: 50%; display: inline-block;}
.efq-card .big {font-size: 1.5rem; font-weight: 600; margin: .2rem 0 .1rem;}
.efq-card .sub {opacity: .7; font-size: .82rem;}
.efq-turn {border-left: 4px solid; border-radius: 6px; padding: .55rem .8rem; margin: .45rem 0; background: rgba(128,128,128,.06);}
.efq-turn .meta {font-size: .8rem; opacity: .75; margin-bottom: .2rem;}
.efq-turn .src {font-size: 1.02rem;}
.efq-turn .tr {font-size: .95rem; opacity: .85; margin-top: .2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ECAPA-TDNN, classifiers and Gnani client …")
def get_pipeline() -> EchoFlowPipeline:
    return EchoFlowPipeline()


def speaker_colors(speakers: list[str]) -> dict[str, str]:
    enrolled = [s for s in speakers if not s.startswith("Unknown")]
    cols = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(enrolled)}
    return cols | {s: UNKNOWN_COLOR for s in speakers if s.startswith("Unknown")}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ============================================================================ sidebar
with st.sidebar:
    st.title("🎙️ EchoFlow-Q")
    st.caption("Hybrid quantum-classical speaker identification + multilingual transcription for group discussions")
    source = st.radio("Audio source", ["Synthetic test meeting", "IndicDiarBench (Kannada)", "Upload audio"])
    audio_path: Path | None = None
    if source == "Synthetic test meeting":
        opts = sorted(MIX_DIR.glob("test_mix*.wav"))
        if opts:
            audio_path = st.selectbox("Meeting", opts, format_func=lambda p: p.stem)
        else:
            st.warning("No mixtures found — run Stage 8 (evaluate_diarization).")
    elif source == "IndicDiarBench (Kannada)":
        opts = sorted(DB_DIR.glob("*.wav"))
        if opts:
            audio_path = st.selectbox("Recording", opts, format_func=lambda p: p.stem)
            st.caption("Real meetings; none of these speakers is enrolled → all should be *Unknown*.")
        else:
            st.warning("No DiarBench recordings cached — run Stage 8.")
    else:
        up = st.file_uploader("WAV / MP3 / FLAC", type=["wav", "mp3", "flac", "ogg", "m4a"])
        if up is not None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = UPLOAD_DIR / Path(up.name).name
            audio_path.write_bytes(up.getvalue())

    st.divider()
    sets = available_model_sets()
    ls_metrics = load_json(MET / "largescale_multilingual_results.json") or {}
    kannada_meeting = audio_path is not None and re.fullmatch(r"test_mix\d+", audio_path.stem) is not None
    # the original Kannada meetings need spk_01–spk_10: large-scale set only if those speakers were merged in Colab
    default_set = ("largescale" if "largescale" in sets and ls_metrics.get("baseline_merged") else "kannada10") if kannada_meeting else sets[0]
    model_set = st.selectbox("Enrolled speaker set", sets, format_func=MODEL_SETS.get, index=sets.index(default_set),
                             help="Large-scale set appears once echoflow_v2_artifacts.zip is installed below")
    classifier = st.selectbox("Speaker classifier", list(CLASSIFIERS), index=0,
                              help="Quantum = ZZ-feature-map fidelity kernel (6 qubits, exact statevector simulation)")
    default_src = {"test_mix_en_to_kn": "English"}.get(audio_path.stem if audio_path else "", SOURCE_LANGUAGES[0])
    source_lang = st.selectbox("Source language", SOURCE_LANGUAGES, index=SOURCE_LANGUAGES.index(default_src))
    default_tgt = "Kannada" if audio_path is not None and audio_path.stem.endswith("_to_kn") else "English"
    target = st.selectbox("Target output voice language", TARGET_LANGUAGES, index=TARGET_LANGUAGES.index(default_tgt))
    voice = st.toggle("Synthesize translated voice", value=True)
    run = st.button("▶ Run EchoFlow-Q", type="primary", width="stretch", disabled=audio_path is None)

    with st.expander("📦 Upload echoflow_v2_artifacts.zip", expanded="largescale" not in sets):
        st.caption("Large-scale models trained in the Colab notebook. Checksums and check vectors are verified before "
                   "anything in results/ is replaced. The bundle is a pickle — only install artifacts you trained yourself.")
        zf = st.file_uploader("echoflow_v2_artifacts.zip", type=["zip"], label_visibility="collapsed")
        if zf is not None and st.button("Install & reload models", width="stretch"):
            try:
                info = install_artifacts(zf.getvalue())
                get_pipeline.clear()
                st.session_state.pop("result", None)
                st.session_state["install_msg"] = (f"Installed {len(info['installed'])} files · {info['n_enrolled']} enrolled speakers"
                                                   + (" · ⚠ SMOKE-test artifacts" if info["smoke"] else ""))
                st.rerun()
            except ArtifactError as e:
                st.error(f"Not installed: {e}")
        if st.session_state.get("install_msg"):
            st.success(st.session_state["install_msg"])

tab_run, tab_models = st.tabs(["Meeting transcript", "Model & MFC-3 analytics"])

# ============================================================================ run
with tab_run:
    if run and audio_path is not None:
        pipe = get_pipeline()
        with st.status("Processing …", expanded=True) as status:
            bar = st.progress(0.0)

            def prog(msg, frac):
                bar.progress(min(frac, 1.0), text=msg)

            try:
                res = pipe.run(audio_path, target, classifier, source_language=source_lang, speaker_model=model_set,
                               synthesize_voice=voice, progress=prog)
                res["audio_file"] = str(audio_path.relative_to(ROOT)) if audio_path.is_relative_to(ROOT) else audio_path.name
                st.session_state["result"] = res
                status.update(label=f"Done in {res['latency_s']['total']:.1f} s", state="complete", expanded=False)
            except Exception as e:  # surface any failure in the UI instead of a blank page
                status.update(label="Failed", state="error")
                st.exception(e)

    res = st.session_state.get("result")
    if not res:
        st.info("Choose an audio source, classifier and target language in the sidebar, then press **Run EchoFlow-Q**.")
    else:
        seg = pd.DataFrame(res["segments"])
        spk = pd.DataFrame(res["speakers"])
        colors = speaker_colors(list(spk.speaker_id)) if len(spk) else {}
        st.subheader(Path(res.get("audio_file", "")).stem or "Result")

        # ---- side-by-side input / output audio ----
        a_in, a_out = st.columns(2, border=True)
        with a_in:
            st.markdown("**🎧 Input audio · original speech**")
            if res.get("input_audio_path") and (ROOT / res["input_audio_path"]).exists():
                st.audio(str(ROOT / res["input_audio_path"]))
            langs = ", ".join(res.get("detected_languages") or []) or "—"
            st.caption(f"Detected source language(s): **{langs}** · ASR `{res.get('asr_language_code', '')}`")
        with a_out:
            st.markdown(f"**🔊 Output audio · {res['target_language']} voice**")
            op = res.get("output_audio_path")
            if op and (ROOT / op).exists():
                st.audio(str(ROOT / op))
                tts = res.get("tts_status") or ""
                st.caption(f"TTS: **{'Gnani timbre-v3.0' if tts == 'gnani' else tts}** · "
                           f"{res.get('output_duration_sec', 0):.0f} s track, turns kept in order without overlap")
                st.download_button("⬇ Download translated audio (.wav)", (ROOT / op).read_bytes(),
                                   f"{Path(res.get('audio_file', 'audio')).stem}_{res['target_language']}.wav", "audio/wav")
            else:
                st.caption("Voice synthesis was switched off for this run.")

        n_known = int(spk.is_enrolled.sum()) if len(spk) else 0
        c = st.columns(5)
        c[0].metric("Duration", f"{res['audio_duration_sec']:.0f} s")
        c[1].metric("Enrolled speakers found", n_known)
        c[2].metric("Unknown voices", len(spk) - n_known)
        c[3].metric("Processing time", f"{res['latency_s']['total']:.1f} s", f"RTF {res.get('real_time_factor', 0):.2f}", delta_color="off")
        c[4].metric("Classifier", res["router"].replace("_", " "))
        if "fallback" in res.get("translation_status", ""):
            st.caption(f"Translation backend: free fallback (Google / MyMemory) — {res['translation_status']}")

        if len(seg):
            # ---- timeline ----
            order = list(dict.fromkeys(seg.sort_values("start_sec").speaker_id))
            chart = alt.Chart(seg).mark_bar(cornerRadius=3, height={"band": 0.7}).encode(
                x=alt.X("start_sec:Q", title="time (s)", scale=alt.Scale(domain=[0, res["audio_duration_sec"]])),
                x2="end_sec:Q",
                y=alt.Y("speaker_id:N", sort=order, title=None),
                color=alt.Color("speaker_id:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
                tooltip=[alt.Tooltip("speaker_id:N", title="speaker"), alt.Tooltip("start_sec:Q", format=".1f"),
                         alt.Tooltip("end_sec:Q", format=".1f"), alt.Tooltip("confidence_margin:Q", format="+.3f"),
                         alt.Tooltip("transcript:N"), alt.Tooltip("translation:N")],
            ).properties(height=max(140, 34 * len(order)), title="Who spoke when")
            st.altair_chart(chart, width="stretch")

            # ---- speaker cards ----
            st.markdown("##### Speakers")
            cols = st.columns(min(4, len(spk)))
            for i, r in spk.iterrows():
                with cols[i % len(cols)]:
                    tag = "enrolled" if r.is_enrolled else "not enrolled / rejected"
                    st.markdown(f"""<div class="efq-card"><div class="spk"><span class="dot" style="background:{colors[r.speaker_id]}"></span>
                        {html.escape(r.speaker_id)}</div><div class="big">{r.talk_share * 100:.0f}%</div>
                        <div class="sub">{r.talk_time_sec:.1f} s · {r.turns} turn(s) · {r.words} words<br>
                        margin {r.mean_confidence_margin:+.3f} · {tag}</div></div>""", unsafe_allow_html=True)

            # ---- transcript ----
            st.markdown(f"##### Conversation · {', '.join(res.get('detected_languages') or ['source'])} → {res['target_language']}")
            for _, r in seg.iterrows():
                col = colors.get(r.speaker_id, UNKNOWN_COLOR)
                src_txt = html.escape(r.transcript or f"({r.get('error') or 'no speech recognised'})")
                tr_txt = html.escape(r.translation) if isinstance(r.get("translation"), str) and r.translation else "—"
                lang = r.get("source_language") if isinstance(r.get("source_language"), str) else "?"
                voice_txt = f" · 🔊 {html.escape(str(r.get('tts_voice')))}" if isinstance(r.get("tts_voice"), str) else ""
                left, right = st.columns([3, 2], vertical_alignment="center")
                left.markdown(f"""<div class="efq-turn" style="border-color:{col}">
                    <div class="meta"><b>{html.escape(r.speaker_id)}</b> · {r.start_sec:.1f}–{r.end_sec:.1f} s · {html.escape(lang)}
                    · margin {r.confidence_margin:+.3f}{voice_txt}</div>
                    <div class="src">{src_txt}</div><div class="tr">{tr_txt}</div></div>""", unsafe_allow_html=True)
                tp = r.get("tts_audio_path")
                if isinstance(tp, str) and (ROOT / tp).exists():
                    right.audio(str(ROOT / tp))

            # ---- exports + latency ----
            e1, e2, _ = st.columns([1, 1, 3])
            stem = Path(res.get("audio_file", "echoflow")).stem
            e1.download_button("⬇ JSON", json.dumps(res, ensure_ascii=False, indent=2, default=str), f"{stem}_echoflowq.json",
                               "application/json", width="stretch")
            cols_out = ["segment_id", "start_sec", "end_sec", "duration_sec", "speaker_id", "confidence_margin",
                        "detected_language", "transcript", "translation", "tts_voice", "output_start_sec", "output_end_sec", "tts_audio_path"]
            buf = io.StringIO(); seg[[c for c in cols_out if c in seg]].to_csv(buf, index=False)
            e2.download_button("⬇ CSV", buf.getvalue().encode("utf-8-sig"), f"{stem}_echoflowq.csv", "text/csv", width="stretch")

            with st.expander("Stage latency breakdown & raw segment table"):
                lat = pd.DataFrame([{"stage": k, "seconds": v} for k, v in res["latency_s"].items() if k != "total"])
                st.altair_chart(alt.Chart(lat).mark_bar(color=PALETTE[0], cornerRadius=3).encode(
                    x=alt.X("seconds:Q"), y=alt.Y("stage:N", sort=None, title=None),
                    tooltip=["stage", alt.Tooltip("seconds:Q", format=".2f")]).properties(height=180), width="stretch")
                st.dataframe(seg[[c for c in cols_out + ["translation_backend", "tts_backend", "asr_cached"] if c in seg]], width="stretch", hide_index=True)
        else:
            st.warning(res.get("warning", "No speech segments found."))

# ============================================================================ analytics
with tab_models:
    st.subheader("Benchmarks on held-out data")
    st.caption("All numbers below are produced by the Stage 5–12 scripts and read from results/metrics; nothing is hard-coded.")

    ls = load_json(MET / "largescale_multilingual_results.json")
    if ls:
        st.markdown(f"##### Large-scale 8-language benchmark (Colab) · {ls['n_enrolled_speakers']} enrolled + {ls['n_impostors']} impostors · "
                    f"{ls['n_utterances']} utterances" + (" · ⚠ smoke-test artifacts" if ls["protocol"].get("smoke") else ""))
        rows = [{"model": r["name"], "accuracy": r["test_accuracy"], "macro-F1": r["test_macro_f1"],
                 "AUROC": r["open_set"]["AUROC_known_vs_unknown"], "CCR": r["open_set"]["CCR"], "TRR": r["open_set"]["TRR"],
                 "FAR": r["open_set"]["FAR"], "KTA": ls["KTA_train"].get(k), "cross-language errors": f"{r['cross_language_errors']}/{r['errors']}"}
                for k, r in ls["results"].items()]
        st.dataframe(pd.DataFrame(rows).style.format(precision=3, na_rep="—"), width="stretch", hide_index=True)
        dz = ls.get("diarization", {})
        if dz.get("per_language"):
            st.caption(f"Diarization: AHC threshold {dz['threshold']} (tuned on {dz['n_val_mixtures']} val mixtures) · "
                       f"test DER {dz['overall']['DER'] * 100:.1f}% over {dz['n_test_mixtures']} mixtures")
            st.dataframe(pd.DataFrame(dz["per_language"]).T.style.format("{:.3f}"), width="stretch")
        f1, f2 = st.columns(2)
        for col, name in ((f1, "largescale_model_comparison.png"), (f2, "largescale_der_per_language.png")):
            if (FIG / name).exists():
                col.image(str(FIG / name), width="stretch")
        g1, g2 = st.columns(2)
        for col, name in ((g1, "largescale_per_language_accuracy.png"), (g2, "largescale_kta_vs_alpha.png")):
            if (FIG / name).exists():
                col.image(str(FIG / name), width="stretch")

    ml = load_json(MET / "multilingual_benchmark.json")
    if ml:
        st.markdown(f"##### 8-language cross-lingual benchmark · {ml['n_enrolled_speakers']} enrolled speakers + "
                    f"{ml['n_impostors']} unseen impostors · {ml['n_clips']} clips")
        rows = [{"model": r["name"], "accuracy": r["test_accuracy"], "macro-F1": r["test_macro_f1"],
                 "AUROC": r["open_set"]["AUROC_known_vs_unknown"], "CCR": r["open_set"]["CCR"], "TRR": r["open_set"]["TRR"],
                 "FAR": r["open_set"]["FAR"], "KTA": ml["KTA_train"].get(k), "cross-language errors": f"{r['cross_language_errors']}/{r['errors']}"}
                for k, r in ml["results"].items()]
        st.dataframe(pd.DataFrame(rows).style.format(precision=3, na_rep="—"), width="stretch", hide_index=True)
        pl = pd.DataFrame({r["name"]: r["per_language_accuracy"] for r in ml["results"].values()})
        st.caption("Per-language test accuracy (small test set: 4 utterances per language)")
        st.dataframe(pl.style.format(precision=2), width="stretch")

    q = MET / "classical_vs_quantum.csv"
    if q.exists():
        df = pd.read_csv(q)
        a = df[df.experiment == "A"][["model", "test_accuracy", "test_macro_f1", "open_CCR", "open_TRR", "open_FAR",
                                       "open_AUROC_known_vs_unknown", "KTA", "gram_train_s", "predict_ms_per_utt"]]
        a.columns = ["model", "accuracy", "macro-F1", "CCR", "TRR", "FAR", "AUROC", "KTA", "Gram (s)", "ms / utt"]
        st.markdown("##### Speaker ID: classical RBF vs quantum fidelity kernel (identical PCA-ECAPA splits)")
        st.dataframe(a.style.format(precision=3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    for col, name, cap in ((c1, "kernel_gram_comparison.png", "Gram matrices (k = 6) and kernel-target alignment"),
                           (c2, "few_shot_classical_vs_quantum.png", "Few-shot stress test (N clips / speaker)")):
        if (FIG / name).exists():
            col.image(str(FIG / name), caption=cap, width="stretch")

    d = load_json(MET / "diarization_results.json")
    s = load_json(MET / "asr_translation_results.json")
    st.markdown("##### Diarization & speech recognition")
    m = st.columns(4)
    if d:
        m[0].metric("DER · synthetic meetings", f"{d['trackA']['der']['selected']['DER'] * 100:.1f}%")
        m[1].metric("DER · DiarBench Kannada", f"{d['trackB']['summary']['der']['DER'] * 100:.1f}%",
                    f"{d['trackB']['summary']['der_no_overlap']['DER'] * 100:.1f}% excl. overlap", delta_color="off")
    if s:
        m[2].metric("ASR WER (15 test clips)", f"{s['asr_benchmark']['aggregate']['WER'] * 100:.1f}%")
        m[3].metric("ASR CER", f"{s['asr_benchmark']['aggregate']['CER'] * 100:.1f}%")
    if d:
        rt = pd.DataFrame(d["trackA"]["routing"]).T.reset_index().rename(columns={"index": "router"})
        st.dataframe(rt[["router", "tau", "cluster_routing_acc", "unknown_rejection_rate", "known_false_rejection", "known_misid", "IER"]]
                     .style.format(precision=3), width="stretch", hide_index=True)
    if (FIG / "diarization_timeline.png").exists():
        st.image(str(FIG / "diarization_timeline.png"), width="stretch")

    st.markdown("##### MFC-3 mathematics (covariance · PCA · SVD)")
    mf = load_json(MET / "mfc3_summary.json")
    if mf:
        st.dataframe(pd.DataFrame(mf["table"]).astype(str), width="stretch", hide_index=True)
    g1, g2 = st.columns(2)
    for col, name in ((g1, "pca_explained_variance.png"), (g2, "svd_spectrum.png")):
        if (FIG / name).exists():
            col.image(str(FIG / name), width="stretch")
    if (FIG / "pca_2d_speakers.png").exists():
        st.image(str(FIG / "pca_2d_speakers.png"), width="stretch")
    with st.expander("Quantum feature-map circuit & covariance heatmap"):
        for name in ("quantum_feature_map_circuit.png", "ecapa_covariance_heatmap.png"):
            if (FIG / name).exists():
                st.image(str(FIG / name), width="stretch")
