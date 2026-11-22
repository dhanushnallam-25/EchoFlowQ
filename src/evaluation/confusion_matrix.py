"""Confusion-matrix figure: cell colour = row-normalised recall (sequential blue), text = raw count."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.evaluation import plot_style as ps


def plot_confusion(y_true, y_pred, labels: list[str], title: str, path: Path, subtitle: str = "") -> np.ndarray:
    plt = ps.plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(norm, cmap=ps.SEQ_BLUE, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.55 else ps.TEXT)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted speaker"); ax.set_ylabel("true speaker")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False); cb.set_label("row-normalised (recall)", color=ps.TEXT_2)
    fig.savefig(path); plt.close(fig)
    return cm
