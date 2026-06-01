"""Task 1 — anomaly scoring and detection metrics (numpy / sklearn only).

The anomaly score for a window is its embedding distance to that file's own
leading-normal windows (a per-file k-NN). Scores are mapped to per-timestep, smoothed,
and thresholded by a high quantile (UCL) of the file's leading-normal scores.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from src.config import Config


def smooth(s: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(s) < k:
        return s
    c = np.cumsum(np.insert(s, 0, 0))
    o = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, o[0]), o])


def knn_score(E: np.ndarray, R: np.ndarray, k: int) -> np.ndarray:
    """1 - mean cosine similarity to the k nearest references (per row of E)."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    sim = En @ Rn.T
    k = min(k, Rn.shape[0])
    return 1 - np.sort(sim, axis=1)[:, -k:].mean(1)


def windows_to_timesteps(starts, wscore, T, cfg: Config):
    ssum = np.zeros(T); cnt = np.zeros(T)
    for s, v in zip(starts, wscore):
        ssum[s:s + cfg.W] += v
        cnt[s:s + cfg.W] += 1
    cnt[cnt == 0] = 1
    return smooth(ssum / cnt, cfg.smooth_k)


def score_file(starts, E, recerr, L, T, cfg: Config):
    """Per-timestep embedding-kNN score (and reconstruction-error score if available)."""
    starts_arr = np.array(starts)
    ref = E[(starts_arr + cfg.W) <= L]
    if len(ref) < cfg.knn:
        ref = E[:max(cfg.knn, 1)]
    emb_w = knn_score(E, ref, cfg.knn)
    emb_s = windows_to_timesteps(starts, emb_w, T, cfg)
    rec_s = windows_to_timesteps(starts, recerr, T, cfg) if recerr is not None else None
    return emb_s, rec_s


def detection_metrics(y, score):
    """Threshold-free metrics for an anomaly score."""
    return dict(auroc=roc_auc_score(y, score), ap=average_precision_score(y, score))


def f1_at_threshold(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(precision=p, recall=r, f1=f1)
