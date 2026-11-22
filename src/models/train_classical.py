"""Stage 6 runner: classical RBF-SVM baselines on identical train/val/test splits.

Experiments: MFCC-120 | ECAPA-192 (z-scored) | PCA-ECAPA k ∈ {2,4,6,8,16,32,64} (PCA fit on train).
  * (C, γ) tuned on val only; final model fit on train only; test touched once at the end.
  * Best PCA k chosen on val macro-F1; k ∈ {4,6,8} also reported as Stage 7 quantum-matched rows.
  * Open-set: unknown speakers split BY SPEAKER — unk_01..02 calibrate τ (with val), unk_03..05 test.

Usage (project root):  .venv\\Scripts\\python.exe -m src.models.train_classical
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.data.hf_utils import PROJECT_ROOT
from src.evaluation.confusion_matrix import plot_confusion
from src.evaluation.metrics import choose_threshold, closed_set, open_set
from src.models.classical_svm import fit, kkt_report, tune

FEAT = PROJECT_ROOT / "data/processed/features"
FIG = PROJECT_ROOT / "results/figures"
MET = PROJECT_ROOT / "results/metrics"
MODELS = PROJECT_ROOT / "results/saved_models"
UNK_CAL = {"unk_01", "unk_02"}          # threshold calibration only
PCA_KS = [2, 4, 6, 8, 16, 32, 64]
QUANTUM_KS = [4, 6, 8]

EXPERIMENTS = [("mfcc", "MFCC (120-D)", "mfcc_std.npz"), ("ecapa", "ECAPA (192-D)", "ecapa_std.npz")] + [
    (f"pca{k}", f"PCA-ECAPA (k={k})", f"ecapa_pca_k{k}.npz") for k in PCA_KS]


def run(key: str, name: str, fname: str) -> dict:
    d = np.load(FEAT / fname)
    Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    Xu, su = d["X_unknown"], d["spk_unknown"]
    cal_mask = np.isin(su, list(UNK_CAL))
    labels = list(d["label_names"])

    t = time.perf_counter()
    best, grid = tune(Xtr, ytr, Xva, yva)
    t_tune = time.perf_counter() - t
    t = time.perf_counter()
    clf = fit(Xtr, ytr, best["C"], best["gamma"])
    t_fit = time.perf_counter() - t

    # open-set threshold on val (known) + calibration unknown speakers
    P_va, P_ucal = clf.predict_proba(Xva), clf.predict_proba(Xu[cal_mask])
    tau = choose_threshold(P_va.max(1), clf.classes_[P_va.argmax(1)], yva, P_ucal.max(1))

    # ---- TEST (touched once) ----
    t = time.perf_counter()
    yhat = clf.predict(Xte)
    t_pred = (time.perf_counter() - t) / len(yte) * 1000
    P_te, P_ute = clf.predict_proba(Xte), clf.predict_proba(Xu[~cal_mask])
    closed = closed_set(yte, yhat)
    opn = open_set(P_te.max(1), clf.classes_[P_te.argmax(1)], yte, P_ute.max(1), tau)
    kkt = kkt_report(clf, Xtr, ytr)

    plot_confusion(yte, yhat, labels, f"Classical RBF-SVM · {name} · test",
                   FIG / f"cm_classical_{key}.png",
                   f"acc {closed['accuracy']:.3f} · macro-F1 {closed['macro_f1']:.3f} · C={best['C']:g}, γ={best['gamma']:.3g}")
    joblib.dump({"model": clf, "tau": tau, "features": fname, "label_names": labels}, MODELS / f"svm_classical_{key}.joblib")

    return {"key": key, "model": f"RBF-SVM | {name}", "dim": int(Xtr.shape[1]), "C": best["C"], "gamma": best["gamma"],
            "val_macro_f1": best["val_macro_f1"], "val_acc": best["val_acc"], **{f"test_{k}": v for k, v in closed.items()},
            **{f"open_{k}": v for k, v in opn.items()}, "n_SV": kkt["total_unique_SV"],
            "sv_per_pair_median": kkt["sv_per_pair"]["median"], "kkt_ok": kkt["kkt_ok"],
            "kernel_PSD": kkt["kernel_train"]["PSD"], "K_fro": kkt["kernel_train"]["fro_norm"],
            "tune_s": t_tune, "fit_s": t_fit, "predict_ms_per_utt": t_pred, "_kkt": kkt, "_grid": grid}


def main() -> None:
    for p in (FIG, MET, MODELS):
        p.mkdir(parents=True, exist_ok=True)
    d0 = np.load(FEAT / "ecapa_std.npz")
    n_ucal = int(np.isin(d0["spk_unknown"], list(UNK_CAL)).sum())
    print(f"splits: train={len(d0['y_train'])} val={len(d0['y_val'])} test={len(d0['y_test'])} | "
          f"unknown: calibration={n_ucal} ({sorted(UNK_CAL)}), test={len(d0['y_unknown']) - n_ucal}\n")

    res = []
    for key, name, fname in EXPERIMENTS:
        r = run(key, name, fname)
        res.append(r)
        print(f"  {r['model']:28s} C={r['C']:<6g} γ={r['gamma']:<9.3g} val-F1={r['val_macro_f1']:.3f}  "
              f"test-acc={r['test_accuracy']:.3f}  #SV={r['n_SV']:3d}  KKT={'ok' if r['kkt_ok'] else 'VIOLATED'}  "
              f"({r['tune_s']:.1f}s tune)")

    pca_rows = [r for r in res if r["key"].startswith("pca")]
    best_k = max(pca_rows, key=lambda r: (round(r["val_macro_f1"], 6), round(r["val_acc"], 6), -r["dim"]))
    for r in res:
        r["role"] = ("baseline" if not r["key"].startswith("pca") else
                     "best-k (val)" if r is best_k else "quantum-matched" if r["dim"] in QUANTUM_KS else "k-sweep")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
    df.to_csv(MET / "classical_results.csv", index=False)
    with open(MET / "classical_results.json", "w", encoding="utf-8") as fh:
        json.dump({"unknown_calibration_speakers": sorted(UNK_CAL), "best_pca_k_by_val": best_k["dim"],
                   "results": [{k: v for k, v in r.items() if k != "_grid"} for r in res],
                   "grids": {r["key"]: r["_grid"] for r in res}}, fh, indent=2, default=float)

    pd.set_option("display.width", 200)
    print("\n=== Closed-set test results (10 enrolled speakers, 47 test utterances) ===")
    show = df[["model", "role", "dim", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
               "test_weighted_precision", "test_weighted_recall", "test_weighted_f1"]].copy()
    show.columns = ["model", "role", "dim", "acc", "macro-P", "macro-R", "macro-F1", "w-P", "w-R", "w-F1"]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== Open-set rejection (τ tuned on val + unk_01..02; reported on test + unk_03..05, 30 utts) ===")
    o = df[["model", "open_tau", "open_CCR", "open_FRR", "open_TRR", "open_FAR", "open_balanced_open_acc", "open_AUROC_known_vs_unknown"]].copy()
    o.columns = ["model", "τ", "CCR", "FRR", "TRR", "FAR", "bal-acc", "AUROC"]
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    k = next(r for r in res if r["key"] == "ecapa")["_kkt"]
    print("\n=== Convex dual / KKT verification (ECAPA-192 model; all models in JSON) ===")
    print(f"  45 one-vs-one binary duals · unique SVs {k['total_unique_SV']}/{k['n_train']} · SV per pair "
          f"min/median/max = {k['sv_per_pair']['min']}/{k['sv_per_pair']['median']:.0f}/{k['sv_per_pair']['max']}")
    for c in k["kkt_checked"]:
        print(f"  [{'PASS' if k['kkt_violations'][c] == 0 else 'FAIL'}] {c:16s} violations {k['kkt_violations'][c]} / {k['kkt_checked'][c]} checked")
    kt = k["kernel_train"]
    print(f"  RBF Gram matrix K (184×184): PSD={kt['PSD']} (λ_min={kt['min_eig']:.2e}), ‖K‖_F={kt['fro_norm']:.2f}, ‖K‖_2={kt['spectral_norm']:.2f}, mean off-diag={kt['mean_offdiag']:.3f}")
    print("  first pairs:", ", ".join(f"{p['pair']}: {p['n_sv']} SV ({p['n_free']} free, {p['n_bounded']} bounded)" for p in k["pairs"][:4]))

    bad = [r["model"] for r in res if not (r["kkt_ok"] and r["kernel_PSD"])]
    if bad:
        raise SystemExit(f"KKT/PSD check failed for: {bad}")
    print(f"\nBest PCA k on val: {best_k['dim']}. Saved: results/metrics/classical_results.{{csv,json}}, "
          f"results/figures/cm_classical_*.png, results/saved_models/svm_classical_*.joblib")


if __name__ == "__main__":
    main()
