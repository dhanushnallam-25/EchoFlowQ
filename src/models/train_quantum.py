"""Stage 7 runner: Quantum-kernel SVM vs classical RBF-SVM on identical PCA-ECAPA splits.

Experiment A — full train set, k ∈ {2,4,6,8}: RBF vs ZZ-quantum vs Z-quantum (entanglement-free ablation).
Experiment B — few-shot stress test, k ∈ {4,6}, N ∈ {3,5,10} clips/speaker × 5 seeds (+ full set):
               RBF vs ZZ-quantum, both re-tuned on val for every subset.
Protocol: PCA + angle scaler are fit on the full train pool (unsupervised); only labelled enrollment
shrinks in B. (C, γ) / (C, α) tuned on val only; test + unk_03..05 touched only for final scoring.

Usage (project root):  .venv\\Scripts\\python.exe -m src.models.train_quantum
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from src.data.hf_utils import PROJECT_ROOT
from src.evaluation import plot_style as ps
from src.evaluation.confusion_matrix import plot_confusion
from src.evaluation.metrics import choose_threshold, closed_set, open_set
from src.models import train_classical
from src.models.classical_svm import kkt_report, tune
from src.models.quantum_svm import ALPHAS, SEED, QuantumKernelSVM, tune_precomputed
from src.quantum.feature_map import AngleScaler, build_feature_map, circuit_stats, draw
from src.quantum.quantum_kernel import centered_kta, fidelity_kernel, kernel_properties, statevectors, verify_against_qiskit

FEAT, FIG, MET, MODELS = (PROJECT_ROOT / p for p in ("data/processed/features", "results/figures", "results/metrics", "results/saved_models"))
SPLITS = ("train", "val", "test", "unknown")
KS = [2, 4, 6, 8]
KINDS = ["zz", "z"]
FS_KS, FS_NS, FS_SEEDS = [4, 6], [3, 5, 10], 5
UNK_CAL = train_classical.UNK_CAL

_data: dict[int, dict] = {}
_states: dict[tuple, dict] = {}   # (kind, k, α) -> {split: Ψ}
_scalers: dict[int, AngleScaler] = {}


def data(k: int) -> dict:
    if k not in _data:
        d = dict(np.load(FEAT / f"ecapa_pca_k{k}.npz"))
        d["cal_mask"] = np.isin(d["spk_unknown"], list(UNK_CAL))
        _data[k] = d
        _scalers[k] = AngleScaler().fit(d["X_train"])
    return _data[k]


def states(kind: str, k: int, a: float) -> dict:
    key = (kind, k, a)
    if key not in _states:
        d, circ = data(k), build_feature_map(k, kind)
        _states[key] = {s: statevectors(circ, _scalers[k].transform(d[f"X_{s}"], a)) for s in SPLITS}
    return _states[key]


# --------------------------------------------------------------------------- Experiment A
def quantum_full(kind: str, k: int) -> dict:
    d = data(k)
    ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    t = time.perf_counter()
    S = {a: states(kind, k, a) for a in ALPHAS}
    t_states_all = time.perf_counter() - t
    Kab = {a: (fidelity_kernel(S[a]["train"], S[a]["train"]), fidelity_kernel(S[a]["val"], S[a]["train"])) for a in ALPHAS}
    best, grid = tune_precomputed(Kab, ytr, yva)
    a, C = best["alpha"], best["C"]
    s = S[a]
    Ktr = Kab[a][0]
    clf = SVC(kernel="precomputed", C=C, probability=True, random_state=SEED).fit(Ktr, ytr)

    P_va = clf.predict_proba(Kab[a][1])
    Ku = fidelity_kernel(s["unknown"], s["train"])
    P_ucal, P_ute = clf.predict_proba(Ku[d["cal_mask"]]), clf.predict_proba(Ku[~d["cal_mask"]])
    tau = choose_threshold(P_va.max(1), clf.classes_[P_va.argmax(1)], yva, P_ucal.max(1))

    # timing: train Gram from scratch, and per-utterance inference (state sim + kernel row + SVC)
    circ = build_feature_map(k, kind)
    t = time.perf_counter(); psi = statevectors(circ, _scalers[k].transform(d["X_train"], a)); fidelity_kernel(psi, psi); t_gram = time.perf_counter() - t
    t = time.perf_counter(); Kte = fidelity_kernel(statevectors(circ, _scalers[k].transform(d["X_test"], a)), psi); yhat = clf.predict(Kte)
    t_inf = (time.perf_counter() - t) / len(yte) * 1000

    P_te = clf.predict_proba(Kte)
    closed = closed_set(yte, yhat)
    opn = open_set(P_te.max(1), clf.classes_[P_te.argmax(1)], yte, P_ute.max(1), tau)
    kkt = kkt_report(clf, Ktr, ytr, K=Ktr)
    props = kernel_properties(Ktr)
    kta_by_alpha = {str(al): centered_kta(Kab[al][0], ytr) for al in ALPHAS}
    name = {"zz": "Quantum ZZ-map", "z": "Quantum Z-map (no entanglement)"}[kind]

    if kind == "zz":
        plot_confusion(yte, yhat, list(d["label_names"]), f"Quantum-kernel SVM (ZZ map, {k} qubits) · test",
                       FIG / f"cm_quantum_k{k}.png", f"acc {closed['accuracy']:.3f} · macro-F1 {closed['macro_f1']:.3f} · C={C:g}, α={a:g}")
        joblib.dump({"model": QuantumKernelSVM(k, kind, a, _scalers[k], s["train"], clf, tau),
                     "features": f"ecapa_pca_k{k}.npz", "label_names": list(d["label_names"])}, MODELS / f"svm_quantum_k{k}.joblib")

    return {"experiment": "A", "k": k, "model": f"{name} | k={k}", "kernel": f"quantum_{kind}", "C": C, "alpha": a, "gamma": None,
            "val_macro_f1": best["val_macro_f1"], **{f"test_{m}": v for m, v in closed.items()}, **{f"open_{m}": v for m, v in opn.items()},
            "n_SV": kkt["total_unique_SV"], "kkt_ok": kkt["kkt_ok"], "kernel_PSD": props["PSD"], "kernel_sym": props["symmetric"],
            "kernel_unit_diag": props["unit_diagonal"], "KTA": kta_by_alpha[str(a)], "K_fro": props["fro_norm"],
            "K_offdiag_mean": props["offdiag_mean"], "gram_train_s": t_gram, "predict_ms_per_utt": t_inf,
            "_kta_by_alpha": kta_by_alpha, "_kkt": kkt, "_grid": grid, "_K": Ktr, "_states_all_s": t_states_all}


def classical_full(k: int) -> dict:
    r = train_classical.run(f"pca{k}", f"PCA-ECAPA (k={k})", f"ecapa_pca_k{k}.npz")
    d = data(k)
    t = time.perf_counter(); K = rbf_kernel(d["X_train"], gamma=r["gamma"]); t_gram = time.perf_counter() - t
    props = kernel_properties(K)
    return {"experiment": "A", "k": k, "model": f"Classical RBF | k={k}", "kernel": "rbf", "C": r["C"], "alpha": None, "gamma": r["gamma"],
            "val_macro_f1": r["val_macro_f1"], **{c: v for c, v in r.items() if c.startswith(("test_", "open_"))},
            "n_SV": r["n_SV"], "kkt_ok": r["kkt_ok"], "kernel_PSD": props["PSD"], "kernel_sym": props["symmetric"],
            "kernel_unit_diag": props["unit_diagonal"], "KTA": centered_kta(K, d["y_train"]), "K_fro": props["fro_norm"],
            "K_offdiag_mean": props["offdiag_mean"], "gram_train_s": t_gram, "predict_ms_per_utt": r["predict_ms_per_utt"], "_K": K}


# --------------------------------------------------------------------------- Experiment B
def few_shot_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(n, int((y == c).sum())), replace=False) for c in np.unique(y)]))


def scores_metrics(clf, Kte_or_X, yte, Ku_or_X) -> dict:
    yhat = clf.predict(Kte_or_X)
    s_known, s_unk = clf.decision_function(Kte_or_X).max(1), clf.decision_function(Ku_or_X).max(1)
    auroc = roc_auc_score(np.r_[np.ones(len(s_known)), np.zeros(len(s_unk))], np.r_[s_known, s_unk])
    return {"acc": accuracy_score(yte, yhat), "macro_f1": f1_score(yte, yhat, average="macro"), "auroc": auroc}


def few_shot_run(k: int, n: int | None, seed: int) -> list[dict]:
    d = data(k)
    ytr_all, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    idx = np.arange(len(ytr_all)) if n is None else few_shot_subset(ytr_all, n, seed)
    Xtr, ytr = d["X_train"][idx], ytr_all[idx]
    Xu = d["X_unknown"][~d["cal_mask"]]
    rows, base = [], {"k": k, "N": "full" if n is None else n, "seed": seed, "n_train": len(idx)}

    best, _ = tune(Xtr, ytr, d["X_val"], yva)
    clf = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"]).fit(Xtr, ytr)
    rows.append(base | {"model": "Classical RBF", **scores_metrics(clf, d["X_test"], yte, Xu)})

    S = {a: states("zz", k, a) for a in ALPHAS}
    Kab = {a: (fidelity_kernel(S[a]["train"][idx], S[a]["train"][idx]), fidelity_kernel(S[a]["val"], S[a]["train"][idx])) for a in ALPHAS}
    bq, _ = tune_precomputed(Kab, ytr, yva)
    s = S[bq["alpha"]]
    qclf = SVC(kernel="precomputed", C=bq["C"]).fit(Kab[bq["alpha"]][0], ytr)
    Kte = fidelity_kernel(s["test"], s["train"][idx])
    Ku = fidelity_kernel(s["unknown"][~d["cal_mask"]], s["train"][idx])
    rows.append(base | {"model": "Quantum ZZ", "alpha": bq["alpha"], **scores_metrics(qclf, Kte, yte, Ku)})
    return rows


# --------------------------------------------------------------------------- figures
def plot_gram_comparison(K_rbf, K_q, y, kta_r, kta_q, gamma, alpha, path):
    plt = ps.plt
    o = np.argsort(y, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, K, t in ((axes[0], K_rbf, f"Classical RBF  (γ={gamma:.3g})\ncentred KTA = {kta_r:.3f}"),
                     (axes[1], K_q, f"Quantum ZZ fidelity  (α={alpha:g}, 6 qubits)\ncentred KTA = {kta_q:.3f}")):
        im = ax.imshow(K[np.ix_(o, o)], cmap=ps.SEQ_BLUE, vmin=0, vmax=1, interpolation="nearest")
        ax.grid(False); ax.set_title(t, loc="left", fontsize=10)
        b = np.flatnonzero(np.diff(y[o])) + 0.5
        for v in b:
            ax.axhline(v, color=ps.SURFACE, lw=0.6); ax.axvline(v, color=ps.SURFACE, lw=0.6)
        ax.set_xlabel("train utterance (sorted by speaker)"); ax.set_ylabel("train utterance")
    cb = fig.colorbar(im, ax=axes, shrink=0.85); cb.outline.set_visible(False); cb.set_label("K(xᵢ, xⱼ)", color=ps.TEXT_2)
    fig.suptitle("Train Gram matrices on identical PCA-ECAPA features (k = 6); bright diagonal blocks = same speaker",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(path); plt.close(fig)


def plot_few_shot(agg: pd.DataFrame, path) -> None:
    plt = ps.plt
    Ns = ["3", "5", "10", "full"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    style = {"Classical RBF": (ps.S1, "o"), "Quantum ZZ": (ps.S2, "^")}
    for c, k in enumerate(FS_KS):
        for r, (m, lab) in enumerate((("macro_f1", "test macro-F1"), ("auroc", "open-set AUROC (known vs unknown)"))):
            ax = axes[r, c]
            for model, (col, mk) in style.items():
                g = agg[(agg.k == k) & (agg.model == model)].set_index("N").reindex(Ns)
                off = -0.06 if model == "Classical RBF" else 0.06
                x = np.arange(len(Ns)) + off
                ax.errorbar(x, g[f"{m}_mean"], yerr=g[f"{m}_std"], color=col, marker=mk, ms=7, lw=2, capsize=3,
                            mec=ps.SURFACE, mew=1, label=model)
                if r == 0 and c == 0:
                    ax.annotate(model, (x[0], g[f"{m}_mean"].iloc[0]), xytext=(8, -14 if model == "Quantum ZZ" else 8),
                                textcoords="offset points", fontsize=8, color=ps.TEXT)
            ax.set_title(f"k = {k} PCA dims / qubits · {lab}", loc="left", fontsize=10)
            ax.set_xticks(range(len(Ns)), ["3", "5", "10", "full\n(14–20)"])
            if r == 1:
                ax.set_xlabel("enrollment clips per speaker (N)")
            ax.set_ylabel(lab.split(" (")[0])
    lo = min(agg["macro_f1_mean"].min() - agg["macro_f1_std"].max(), agg["auroc_mean"].min() - agg["auroc_std"].max())
    for ax in axes.flat:
        ax.set_ylim(max(0, lo - 0.03), 1.01)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Few-shot stress test: Classical RBF vs Quantum ZZ kernel (mean ± std over {FS_SEEDS} seeds; 47 test utts, 30 unknown)",
                 x=0.01, y=0.995, ha="left", fontsize=11)
    fig.legend(h, l, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    pd.set_option("display.width", 220)
    t0 = time.perf_counter()

    print("=== Feature-map circuits (decomposed to H, P, CX; reps=2, full entanglement) ===")
    circ_stats = {}
    for kind in KINDS:
        for k in KS:
            st = circuit_stats(build_feature_map(k, kind)); circ_stats[f"{kind}_k{k}"] = st
            print(f"  {kind.upper():2s}-map k={k}: qubits={st['qubits']} params={st['parameters']} depth={st['depth']:3d} "
                  f"CX={st['cx']:3d} two-qubit depth={st['two_qubit_depth']:3d}  ops={st['ops']}")
    draw(build_feature_map(4, "zz"), FIG / "quantum_feature_map_circuit.png",
         "ZZ feature map, 4 qubits, reps = 2, full entanglement:  |φ(x)⟩ = U_Φ(x) H⊗⁴ U_Φ(x) H⊗⁴ |0⟩⊗⁴")

    d6 = data(6)
    ver = verify_against_qiskit(build_feature_map(6, "zz"), _scalers[6].transform(d6["X_train"], 0.1))
    print(f"\n=== Kernel implementation check (k=6, 184×184) ===\n  ours vs FidelityStatevectorKernel: max|ΔK| = {ver['max_abs_diff']:.1e} "
          f"(equal={ver['equal']}); time ours {ver['ours_s']:.2f}s vs qiskit-ml {ver['qiskit_ml_s']:.2f}s")

    # ---- Experiment A ----
    print("\n=== Experiment A: full training set ===")
    A = []
    for k in KS:
        A.append(classical_full(k))
        for kind in KINDS:
            A.append(quantum_full(kind, k))
        for r in A[-3:]:
            print(f"  {r['model']:44s} val-F1={r['val_macro_f1']:.3f} test-acc={r['test_accuracy']:.3f} KTA={r['KTA']:.3f} "
                  f"{'α=%-5g' % r['alpha'] if r['alpha'] else 'γ=%-7.3g' % r['gamma']} C={r['C']:<5g}")
    kq6 = next(r for r in A if r["k"] == 6 and r["kernel"] == "quantum_zz")
    kc6 = next(r for r in A if r["k"] == 6 and r["kernel"] == "rbf")
    plot_gram_comparison(kc6["_K"], kq6["_K"], d6["y_train"], kc6["KTA"], kq6["KTA"], kc6["gamma"], kq6["alpha"],
                         FIG / "kernel_gram_comparison.png")

    # ---- Experiment B ----
    print(f"\n=== Experiment B: few-shot (k ∈ {FS_KS}, N ∈ {FS_NS}, {FS_SEEDS} seeds) ===")
    B = []
    for k in FS_KS:
        for n in FS_NS:
            for sd in range(FS_SEEDS):
                B += few_shot_run(k, n, sd)
        B += few_shot_run(k, None, 0)
    dfB = pd.DataFrame(B)
    dfB["N"] = dfB["N"].astype(str)
    agg = dfB.groupby(["k", "N", "model"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), runs=("acc", "size")).fillna(0).reset_index()
    plot_few_shot(agg, FIG / "few_shot_classical_vs_quantum.png")

    # paired differences (quantum − classical) per (k, N)
    piv = dfB.pivot_table(index=["k", "N", "seed"], columns="model", values=["macro_f1", "auroc"])
    diff = pd.DataFrame({m: piv[(m, "Quantum ZZ")] - piv[(m, "Classical RBF")] for m in ("macro_f1", "auroc")}).reset_index()
    pdiff = diff.groupby(["k", "N"]).agg(dF1_mean=("macro_f1", "mean"), q_wins_F1=("macro_f1", lambda s: f"{(s > 1e-9).sum()}/{(s < -1e-9).sum()}/{(s.abs() <= 1e-9).sum()}"),
                                          dAUROC_mean=("auroc", "mean"), q_wins_AUROC=("auroc", lambda s: f"{(s > 1e-9).sum()}/{(s < -1e-9).sum()}/{(s.abs() <= 1e-9).sum()}")).reset_index()

    # ---- save ----
    dfA = pd.DataFrame([{c: v for c, v in r.items() if not c.startswith("_")} for r in A])
    out_csv = pd.concat([dfA, agg.assign(experiment="B")], ignore_index=True)
    out_csv.to_csv(MET / "classical_vs_quantum.csv", index=False)
    with open(MET / "quantum_results.json", "w", encoding="utf-8") as fh:
        json.dump({"protocol": __doc__, "alphas": ALPHAS, "circuits": circ_stats, "kernel_verification": ver,
                   "experiment_A": [{c: v for c, v in r.items() if c not in ("_K", "_grid")} for r in A],
                   "experiment_A_grids": {r["model"]: r.get("_grid") for r in A if "_grid" in r},
                   "experiment_B_runs": B, "experiment_B_summary": agg.to_dict("records"),
                   "experiment_B_paired": pdiff.to_dict("records"), "runtime_s": time.perf_counter() - t0},
                  fh, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- report ----
    print("\n=== Experiment A — closed + open set (test: 47 known utts; unknown: 30 utts from unk_03..05) ===")
    show = dfA[["model", "test_accuracy", "test_macro_f1", "open_CCR", "open_TRR", "open_FAR", "open_AUROC_known_vs_unknown",
                "n_SV", "KTA", "K_offdiag_mean", "gram_train_s", "predict_ms_per_utt"]].copy()
    show.columns = ["model", "acc", "macro-F1", "CCR", "TRR", "FAR", "AUROC", "#SV", "KTA", "K̄offdiag", "Gram(s)", "ms/utt"]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  kernel checks (all models): symmetric={} unit-diag(quantum)={} PSD={} KKT={}".format(
        dfA.kernel_sym.all(), dfA[dfA.kernel != "rbf"].kernel_unit_diag.all(), dfA.kernel_PSD.all(), dfA.kkt_ok.all()))
    print("  KTA vs α (ZZ, k=6, train): " + "  ".join(f"α={a}: {v:.3f}" for a, v in kq6["_kta_by_alpha"].items()))

    print("\n=== Experiment B — few-shot (mean ± std over seeds) ===")
    for k in FS_KS:
        for n in ["3", "5", "10", "full"]:
            row = agg[(agg.k == k) & (agg.N == n)].set_index("model")
            c, q = row.loc["Classical RBF"], row.loc["Quantum ZZ"]
            pdr = pdiff[(pdiff.k == k) & (pdiff.N == n)].iloc[0]
            print(f"  k={k} N={n:>4s} | F1  RBF {c.macro_f1_mean:.3f}±{c.macro_f1_std:.3f}  Q {q.macro_f1_mean:.3f}±{q.macro_f1_std:.3f}  "
                  f"Δ={pdr.dF1_mean:+.3f} (Q win/lose/tie {pdr.q_wins_F1}) | AUROC  RBF {c.auroc_mean:.3f}  Q {q.auroc_mean:.3f}  "
                  f"Δ={pdr.dAUROC_mean:+.3f} ({pdr.q_wins_AUROC})")

    bad = dfA[~(dfA.kernel_sym & dfA.kernel_PSD & dfA.kkt_ok)]
    if len(bad) or not ver["equal"]:
        raise SystemExit(f"Stage 7 verification FAILED: {bad.model.tolist()}")
    print(f"\nStage 7 OK in {time.perf_counter() - t0:.0f}s -> results/metrics/{{quantum_results.json, classical_vs_quantum.csv}}")


if __name__ == "__main__":
    main()
