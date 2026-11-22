"""Stage 8: unsupervised speaker clustering of window embeddings + conversion to speaker turns.

AHC with cosine distance and average linkage. Speaker count is either
  * 'threshold'  — cut the dendrogram at a cosine-distance threshold (tuned on validation mixtures), or
  * 'silhouette' — try k = 2..k_max clusters and keep the k with the highest cosine silhouette, or
  * 'oracle'     — given k (upper-bound analysis only).
Turns: each window owns the span around its centre (± step/2, clipped to its VAD region);
consecutive spans with the same cluster are merged.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def ahc(emb: np.ndarray, threshold: float | None = None, n_clusters: int | None = None) -> np.ndarray:
    if len(emb) < 2:
        return np.zeros(len(emb), int)
    if n_clusters is not None:
        n_clusters = min(n_clusters, len(emb))
    return AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None if n_clusters else threshold,
                                   metric="cosine", linkage="average").fit_predict(emb)


def ahc_silhouette(emb: np.ndarray, k_max: int = 8) -> tuple[np.ndarray, int]:
    best, best_s = np.zeros(len(emb), int), -1.0
    for k in range(2, min(k_max, len(emb) - 1) + 1):
        lab = ahc(emb, n_clusters=k)
        s = silhouette_score(emb, lab, metric="cosine")
        if s > best_s:
            best, best_s = lab, s
    return best, int(best.max() + 1)


def merge_small(emb: np.ndarray, labels: np.ndarray, min_windows: int) -> np.ndarray:
    """Reassign clusters with < min_windows windows to the nearest (cosine) larger cluster centroid;
    tiny clusters are usually fragments of a real speaker whose centroids are too noisy to route."""
    labels = labels.copy()
    sizes = np.bincount(labels)
    big = [c for c in range(len(sizes)) if sizes[c] >= min_windows]
    if min_windows <= 1 or not big or len(big) == len(sizes):
        return labels
    C = np.stack([emb[labels == c].mean(0) for c in big])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    small = ~np.isin(labels, big)
    labels[small] = np.asarray(big)[(emb[small] @ C.T).argmax(1)]
    return np.unique(labels, return_inverse=True)[1]


def cluster(emb: np.ndarray, method: str, threshold: float = 0.6, k: int | None = None, min_windows: int = 1) -> np.ndarray:
    if method == "threshold":
        lab = ahc(emb, threshold=threshold)
    elif method == "silhouette":
        lab = ahc_silhouette(emb)[0]
    elif method == "oracle":
        return ahc(emb, n_clusters=k)
    else:
        raise ValueError(method)
    return merge_small(emb, lab, min_windows)


def windows_to_turns(windows: np.ndarray, region: np.ndarray, regions: list, labels: np.ndarray, step: float) -> list[tuple[float, float, int]]:
    spans = []
    for (s, e), r, lab in zip(windows, region, labels):
        c = (s + e) / 2
        rs, re_ = regions[r]
        spans.append([max(rs, c - step / 2), min(re_, c + step / 2), int(lab), r])
    # extend first/last span of each region to the region boundary
    for r in set(region.tolist()):
        idx = [i for i, sp in enumerate(spans) if sp[3] == r]
        spans[idx[0]][0] = regions[r][0]; spans[idx[-1]][1] = regions[r][1]
        for a, b in zip(idx, idx[1:]):                       # no gaps / overlaps inside a region
            mid = (spans[a][1] + spans[b][0]) / 2
            spans[a][1] = spans[b][0] = mid
    turns: list[list] = []
    for s, e, lab, _ in sorted(spans):
        if turns and turns[-1][2] == lab and s - turns[-1][1] < 1e-6:
            turns[-1][1] = e
        else:
            turns.append([s, e, lab])
    return [(float(s), float(e), int(l)) for s, e, l in turns if e - s > 1e-3]


def centroids(emb: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for c in np.unique(labels):
        m = emb[labels == c].mean(0)
        out[int(c)] = m / max(np.linalg.norm(m), 1e-10)
    return out
