"""Closed-set and open-set speaker-identification metrics, plus threshold selection.

Open-set protocol (score s = max class posterior; accept if s ≥ τ, else 'unknown'):
  known utterances:   CCR = accepted AND correctly identified / n_known
                      FRR = rejected / n_known
  unknown utterances: TRR = rejected / n_unknown       (true rejection rate)
                      FAR = accepted / n_unknown = 1 − TRR  (false acceptance rate)
  τ is chosen on a CALIBRATION set (val known + calibration-only unknown speakers) to maximise
  balanced open-set accuracy = ½(CCR + TRR); the test unknown speakers are never seen in tuning.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def closed_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out |= {f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f}
    return {k: float(v) for k, v in out.items()}


def open_set(score_known, pred_known, y_known, score_unknown, tau: float) -> dict[str, float]:
    acc_k = score_known >= tau
    ccr = float(np.mean(acc_k & (pred_known == y_known)))
    trr = float(np.mean(score_unknown < tau))
    auroc = roc_auc_score(np.r_[np.ones(len(score_known)), np.zeros(len(score_unknown))],
                          np.r_[score_known, score_unknown])
    return {"tau": float(tau), "CCR": ccr, "FRR": float(np.mean(~acc_k)),
            "TRR": trr, "FAR": 1 - trr, "balanced_open_acc": 0.5 * (ccr + trr), "AUROC_known_vs_unknown": float(auroc)}


def choose_threshold(score_known, pred_known, y_known, score_unknown) -> float:
    """Pick τ on calibration data maximising ½(CCR + TRR); ties -> smallest τ (fewest false rejections)."""
    cands = np.unique(np.r_[score_known, score_unknown, 0.0])
    vals = np.array([0.5 * (np.mean((score_known >= t) & (pred_known == y_known)) + np.mean(score_unknown < t)) for t in cands])
    return float(cands[np.flatnonzero(np.isclose(vals, vals.max()))[0]])
