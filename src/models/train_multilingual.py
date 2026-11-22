"""v2 Upgrade 1: language-agnostic speaker models on the 8-language benchmark.

ECAPA-TDNN (192-D, L2) for every clip in data/metadata/multilingual_splits.csv, then on identical splits:
  * Cosine to enrolled centroids (reference)
  * Classical RBF-SVM (192-D, z-scored)          (C, γ) tuned on val
  * Classical RBF-SVM (PCA-6)                     PCA fit on train
  * Hybrid quantum ZZ fidelity-kernel SVM (6 qubits, angle x = α·π·tanh(z/σ)), (C, α) tuned on val
Scores are RAW max one-vs-rest decision values (no Platt). τ per model is chosen on val (known) +
impostor_cal (kn/ta/ml/pa impostors); AUROC / TRR / FAR are reported on test + impostor_test (hi/te/bn/en).
Outputs: results/models/multilingual/routers.joblib, results/metrics/multilingual_benchmark.json

Usage (project root):  .venv\\Scripts\\python.exe -m src.models.train_multilingual
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from tqdm import tqdm

from src.data.hf_utils import PROJECT_ROOT
from src.evaluation.metrics import choose_threshold, closed_set, open_set
from src.features import ecapa
from src.features.pca import EigPCA
from src.features.standardization import ZScoreScaler
from src.models.classical_svm import kkt_report, tune
from src.models.quantum_svm import ALPHAS, QuantumKernelSVM, tune_precomputed
from src.quantum.feature_map import AngleScaler, build_feature_map
from src.quantum.quantum_kernel import centered_kta, fidelity_kernel, kernel_properties, statevectors
from src.speech.languages import LANGUAGES

META = PROJECT_ROOT / "data/metadata/multilingual_splits.csv"
OUT_DIR = PROJECT_ROOT / "results/models/multilingual"
MET = PROJECT_ROOT / "results/metrics/multilingual_benchmark.json"
K = 6
NAMES = {"cosine": "Cosine (ECAPA-192)", "rbf_192": "Classical RBF-SVM (192-D)", "rbf_pca6": "Classical RBF-SVM (PCA-6)",
         "quantum_k6": "Hybrid Quantum ZZ-SVM (6 qubits)"}


def main() -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(META, dtype={"original_speaker_id": str, "source_recording_id": str})
    ecapa.load_encoder()
    E = np.stack([ecapa.embed(sf.read(PROJECT_ROOT / p, dtype="float32")[0]) for p in tqdm(df.filepath, desc="ECAPA", leave=False)])
    t_emb = time.perf_counter() - t0

    labels = sorted(df[df.role == "enrolled"].speaker_label.unique())
    lab = {s: i for i, s in enumerate(labels)}
    m = {s: (df.split == s).to_numpy() for s in ("train", "val", "test", "impostor_cal", "impostor_test")}
    y = np.array([lab.get(s, -1) for s in df.speaker_label])
    ytr, yva, yte = y[m["train"]], y[m["val"]], y[m["test"]]

    sc = ZScoreScaler().fit(E[m["train"]])
    S = sc.transform(E)
    pca = EigPCA().fit(S[m["train"]])
    Z = pca.transform(S, K)
    evr = float(pca.explained_variance_ratio_[:K].sum())

    # ---- models ----
    mu = np.stack([E[m["train"]][ytr == i].mean(0) for i in range(len(labels))]); mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    rb, _ = tune(S[m["train"]], ytr, S[m["val"]], yva)
    rbf192 = SVC(kernel="rbf", C=rb["C"], gamma=rb["gamma"]).fit(S[m["train"]], ytr)
    rp, _ = tune(Z[m["train"]], ytr, Z[m["val"]], yva)
    rbf6 = SVC(kernel="rbf", C=rp["C"], gamma=rp["gamma"]).fit(Z[m["train"]], ytr)

    ascale = AngleScaler().fit(Z[m["train"]])
    circ = build_feature_map(K, "zz")
    t = time.perf_counter()
    states = {a: statevectors(circ, ascale.transform(Z, a)) for a in ALPHAS}
    t_q = time.perf_counter() - t
    Kab = {a: (fidelity_kernel(states[a][m["train"]], states[a][m["train"]]), fidelity_kernel(states[a][m["val"]], states[a][m["train"]])) for a in ALPHAS}
    bq, _ = tune_precomputed(Kab, ytr, yva)
    a = bq["alpha"]
    Kq_tr = Kab[a][0]
    qsvc = SVC(kernel="precomputed", C=bq["C"]).fit(Kq_tr, ytr)
    qmodel = QuantumKernelSVM(K, "zz", a, ascale, states[a][m["train"]], qsvc)

    def scores(key: str, mask: np.ndarray):
        if key == "cosine":
            s = E[mask] @ mu.T
            return s.argmax(1), s.max(1)
        if key == "quantum_k6":
            D = qsvc.decision_function(fidelity_kernel(states[a][mask], states[a][m["train"]]))
        else:
            clf, X = (rbf192, S) if key == "rbf_192" else (rbf6, Z)
            D = clf.decision_function(X[mask])
        return D.argmax(1), D.max(1)

    results, taus = {}, {}
    test_lang = df.language[m["test"]].to_numpy()
    for key in NAMES:
        pva, sva = scores(key, m["val"]); _, sic = scores(key, m["impostor_cal"])
        tau = choose_threshold(sva, pva, yva, sic); taus[key] = tau
        pte, ste = scores(key, m["test"]); _, sit = scores(key, m["impostor_test"])
        cs = closed_set(yte, pte)
        per_lang = {LANGUAGES[l]["name"]: float(np.mean(pte[test_lang == l] == yte[test_lang == l])) for l in LANGUAGES}
        wrong = pte != yte
        cross = int(np.sum([labels[p].split("_")[0] != labels[t].split("_")[0] for p, t in zip(pte[wrong], yte[wrong])]))
        auroc = float(roc_auc_score(np.r_[np.ones(len(ste)), np.zeros(len(sit))], np.r_[ste, sit]))
        results[key] = {"name": NAMES[key], **{f"test_{k}": v for k, v in cs.items()}, "per_language_accuracy": per_lang,
                        "errors": int(wrong.sum()), "cross_language_errors": cross,
                        "open_set": open_set(ste, pte, yte, sit, tau) | {"AUROC_known_vs_unknown": auroc}}

    Krbf = rbf_kernel(S[m["train"]], gamma=rb["gamma"])
    Krbf6 = rbf_kernel(Z[m["train"]], gamma=rp["gamma"])
    kta = {"rbf_192": centered_kta(Krbf, ytr), "rbf_pca6": centered_kta(Krbf6, ytr), "quantum_k6": centered_kta(Kq_tr, ytr)}
    kkt = {"rbf_192": kkt_report(rbf192, S[m["train"]], ytr)["kkt_ok"], "rbf_pca6": kkt_report(rbf6, Z[m["train"]], ytr)["kkt_ok"],
           "quantum_k6": kkt_report(qsvc, Kq_tr, ytr, K=Kq_tr)["kkt_ok"]}
    qprops = kernel_properties(Kq_tr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enr = df[df.role == "enrolled"].drop_duplicates("speaker_label").set_index("speaker_label")
    joblib.dump({"labels": labels, "label_language": {s: enr.language[s] for s in labels}, "label_gender": {s: enr.gender[s] for s in labels},
                 "scaler": sc, "pca_mean": pca.mean_, "pca_W": pca.W(K), "mu": mu, "rbf_192": rbf192, "rbf_pca6": rbf6,
                 "quantum_k6": qmodel, "tau": taus, "k": K}, OUT_DIR / "routers.joblib")

    split_counts = df.groupby(["language", "split"]).size().unstack(fill_value=0).to_dict("index")
    out = {"protocol": __doc__, "n_clips": len(df), "n_enrolled_speakers": len(labels), "n_impostors": int(df[df.role == "impostor"].speaker_label.nunique()),
           "split_sizes": df.split.value_counts().to_dict(), "per_language_split_counts": split_counts,
           "pca_k6_explained_variance": evr, "hyperparameters": {"rbf_192": {"C": rb["C"], "gamma": rb["gamma"]},
           "rbf_pca6": {"C": rp["C"], "gamma": rp["gamma"]}, "quantum_k6": {"C": bq["C"], "alpha": a}},
           "tau": taus, "KTA_train": kta, "kkt_ok": kkt, "quantum_kernel": {k: qprops[k] for k in ("symmetric", "unit_diagonal", "PSD", "offdiag_mean")},
           "results": results, "runtime_s": {"embedding": t_emb, "quantum_states_all_alphas": t_q, "total": time.perf_counter() - t0}}
    MET.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"{len(df)} clips · {len(labels)} enrolled speakers · 8 impostors · splits {out['split_sizes']} · PCA-6 keeps {evr*100:.1f}% variance")
    print(f"hyper-params: {out['hyperparameters']}")
    print(f"\n{'model':34s} {'acc':>6s} {'mF1':>6s} {'AUROC':>6s} {'CCR':>6s} {'TRR':>6s} {'FAR':>6s} {'KTA':>6s} {'x-lang err':>10s}")
    for k, r in results.items():
        o = r["open_set"]
        print(f"{r['name']:34s} {r['test_accuracy']:6.3f} {r['test_macro_f1']:6.3f} {o['AUROC_known_vs_unknown']:6.3f} {o['CCR']:6.3f} "
              f"{o['TRR']:6.3f} {o['FAR']:6.3f} {kta.get(k, float('nan')):6.3f} {r['cross_language_errors']:>4d}/{r['errors']:<4d}")
    print("\nper-language test accuracy (4 utterances per language):")
    print(pd.DataFrame({r["name"]: r["per_language_accuracy"] for r in results.values()}).round(3).to_string())
    print(f"\nKKT ok: {kkt} · quantum kernel symmetric/unit-diag/PSD: {qprops['symmetric']}/{qprops['unit_diagonal']}/{qprops['PSD']}")
    print(f"Saved -> {OUT_DIR.relative_to(PROJECT_ROOT)}/routers.joblib, {MET.relative_to(PROJECT_ROOT)}  ({out['runtime_s']['total']:.0f}s)")


if __name__ == "__main__":
    main()
