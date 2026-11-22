# EchoFlow-Q

**Hybrid quantum-classical speaker identification and multilingual speech processing for group discussions.**

Give EchoFlow-Q a recorded Kannada group discussion and it tells you **who spoke when**, **who each speaker is** (one of 10 enrolled speakers, or *Unknown*), **what they said** (Gnani AI Prisma speech recognition) and **what it means** in English, Hindi, Tamil, Telugu or Kannada. Speaker identity can be decided by a classical RBF-kernel SVM or by a **6-qubit ZZ-feature-map fidelity-kernel SVM** (Qiskit, exact statevector simulation).

📄 Full technical report with all results: [`docs/CAPSTONE_REPORT.md`](docs/CAPSTONE_REPORT.md)

```
audio ─► VAD ─► ECAPA-TDNN windows ─► AHC clustering ─► speaker router ─► Gnani STT ─► translation
                                                     (RBF-SVM │ Quantum-kernel SVM │ cosine)
```

| Headline result (held-out data) | Value |
|---|---|
| Speaker ID accuracy, 10 speakers (ECAPA + RBF-SVM) | 1.000 (single-session test) |
| Unknown-speaker AUROC, RBF vs Quantum ZZ (k = 6) | 0.989 vs 0.876 (0.977 on raw margins) |
| Kernel-target alignment, RBF vs Quantum (k = 6) | 0.968 vs 0.908 |
| DER: synthetic meetings / real Kannada (DiarBench) | 16.7% / 38.3% (16.6% excluding overlap) |
| Gnani ASR on Kannada: WER / CER | 38.2% / 11.3% |
| Accuracy at 0 dB SNR, RBF-192 vs Quantum | 1.000 vs 0.745 |

## Setup (Windows, Python 3.14, CPU)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\smoke_test.py          # verifies torch, SpeechBrain, Silero, Qiskit, …
```

On Google Colab or Linux, `pip install -r requirements.txt` works unchanged; all paths are relative to the project root.

### `.env` (never commit it; it is already in `.gitignore`)

```ini
# Hugging Face read token; accept the terms of ai4bharat/IndicVoices first
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# Gnani / vachana.ai: sent as the X-API-Key-ID header
GNANI_API_KEY=xxxxxxxxxxxxxxxx
GNANI_API_ENDPOINT=https://api.vachana.ai
```

Translation uses Gnani Evon (`/v1/chat/completions`) when the key has the **LLM** scope. If Gnani returns 403 `SCOPE_NOT_ALLOWED`, it falls back automatically to Google Translate (unofficial endpoint, rate-limited).

## Run

```powershell
# Web UI → http://localhost:8501
.venv\Scripts\streamlit run app\streamlit_app.py

# End-to-end integration test (pipeline with classical + quantum routers, plus a headless UI run)
.venv\Scripts\python.exe -m src.pipeline.test_end_to_end
```

### Use the pipeline from Python

```python
from src.pipeline.echoflow_pipeline import EchoFlowPipeline
pipe = EchoFlowPipeline()
res = pipe.run("meeting.wav", target_language="English",
               classifier="Hybrid Quantum-Classical SVM (6 Qubits)")
for s in res["segments"]:
    print(s["start_sec"], s["speaker_id"], s["transcript"], "→", s["translation"])
```

### Reproduce every experiment (from the project root, in order)

| Stage | Command (`.venv\Scripts\python.exe -m …`) | Writes |
|---|---|---|
| 2 Data | `src.data.inspect_indicvoices`, `src.data.dataset_loader`, `src.data.dataset_split` | `data/metadata/*.csv` |
| 3 VAD | `src.preprocessing.segmentation` | `vad_segments.csv` |
| 4 Features | `src.features.extract_all` | `data/processed/features/*.npz` |
| 5 MFC-3 | `src.features.run_mfc3_analysis` | `results/metrics/mfc3_summary.json` |
| 6 Classical | `src.models.train_classical` | `classical_results.*` |
| 7 Quantum | `src.models.train_quantum` | `quantum_results.json`, `classical_vs_quantum.csv` |
| 8 Diarization | `src.diarization.evaluate_diarization` | `diarization_results.json` |
| 9 ASR / MT | `src.speech.evaluate_asr_translation [--skip-benchmark]` | `asr_translation_results.json` |
| 12 Robustness | `src.evaluation.robustness` | `robustness_results.json` |

Stages 2 and 8 stream small slices from Hugging Face; nothing downloads a full dataset. Stages 9 and 10 call the Gnani API, and repeated audio is served from `results/cache/` so it does not spend credits again.

## v2: 8 languages and voice-to-voice translation

- **Speaker benchmark:** 16 enrolled speakers + 8 unseen impostors across Kannada, English, Hindi, Tamil, Telugu, Malayalam, Bengali and Punjabi. Built by `src.data.build_multilingual_dataset` (IndicVoices + LibriSpeech, streamed), trained by `src.models.train_multilingual` → `results/models/multilingual/routers.joblib`, `results/metrics/multilingual_benchmark.json`.
- **Speech in, speech out:**
  - Gnani Prisma `indic-auto` detects the language of each turn.
  - Each turn is translated from its detected language.
  - Gnani **timbre-v3.0** speaks it in the target language, with a distinct voice per speaker; edge-tts is the fallback.
  - The turns are stitched into `output_translated_audio.wav`, keeping the original timing without overlaps.
- **UI:** choose a source language (or auto-detect) and a target voice language, then play the input and translated audio side by side, as a whole or turn by turn.
- **Test meetings:** `data/processed/mixtures/test_mix_kn_to_en.wav` and `test_mix_en_to_kn.wav`, with ground-truth turns in matching `.json` files.

```python
res = EchoFlowPipeline().run("data/processed/mixtures/test_mix_en_to_kn.wav", target_language="Kannada",
                             classifier="Hybrid Quantum-Classical SVM (6 Qubits)", source_language="Auto-Detect",
                             speaker_model="multilingual")
print(res["output_audio_path"])   # data/processed/outputs/.../output_translated_audio.wav
```

## Repository layout

```
src/
  data/           dataset inspection, curation, leak-free splits
  preprocessing/  audio loading, cleaning, Silero VAD, segmentation
  features/       MFCC, ECAPA-TDNN, z-score, covariance, PCA, SVD (MFC-3)
  models/         classical SVM (dual / KKT verification), quantum-kernel SVM, runners
  quantum/        ZZ / Z feature maps, statevector fidelity kernel, KTA
  diarization/    windowed embeddings, AHC clustering, speaker routing, DER evaluation
  speech/         Gnani client, segment ASR, cached translation with fallback
  pipeline/       EchoFlowPipeline + integration test
  evaluation/     metrics, confusion matrices, robustness, plot style
app/streamlit_app.py
results/{metrics,figures,saved_models}
docs/CAPSTONE_REPORT.md
```

## Notes
- **Data protocol:** splits use seed 42; scalers, PCA and angle scalers are fit on train only; hyper-parameters and thresholds are chosen on val only; unknown speakers are split by speaker between calibration and test.
- **Known limitations:** single-session enrollment, no overlap handling in diarization, and quantum results from noiseless simulation. See §13 of the report.
