"""Feature generation: windowing and per-sensor patch tokenization.

A window x in R[W, C] is reshaped into C*Np tokens, each token = a length-P patch of
ONE sensor (sensor outer, time-patch inner). This is the tokenization the encoder
consumes.
"""
import numpy as np

from src.config import Config
from src.data import standardize


def win_starts(T: int, cfg: Config):
    """Window start indices covering [0, T), including the tail."""
    s = list(range(0, T - cfg.W + 1, cfg.stride))
    if not s:
        return []
    if s[-1] != T - cfg.W:
        s.append(T - cfg.W)
    return s


def patchify(Xw: np.ndarray, cfg: Config, C: int) -> np.ndarray:
    """[B, W, C] -> [B, C*Np, P], token order = sensor outer, patch inner."""
    B = Xw.shape[0]
    return Xw.transpose(0, 2, 1).reshape(B, C, cfg.Np, cfg.P).reshape(B, C * cfg.Np, cfg.P)


def token_ids(cfg: Config, C: int):
    """Per-token sensor and position ids (aligned with patchify ordering)."""
    sensor_ids = np.repeat(np.arange(C), cfg.Np)
    pos_ids = np.tile(np.arange(cfg.Np), C)
    return sensor_ids, pos_ids


def build_normal_pool(files, cfg: Config, C: int):
    """Pool normal training windows (per-file standardized) and split into train/val.

    Sources: the anomaly-free file (all windows) + each anomalous file's leading-normal
    windows. Returns (Ptr, Pva) patch tensors [N, C*Np, P].
    """
    windows = []
    for f in files:
        Z = standardize(f)
        if f["fault"] == "anomaly-free":
            for s in win_starts(len(Z), cfg):
                windows.append(Z[s:s + cfg.W])
        else:
            if f["L"] >= cfg.W:
                for s in win_starts(f["L"], cfg):       # only the leading-normal part
                    windows.append(Z[s:s + cfg.W])
    P_all = patchify(np.stack(windows), cfg, C)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(P_all))
    nv = max(1, int(0.15 * len(P_all)))
    return P_all[idx[nv:]], P_all[idx[:nv]]             # train, val


def collect_anomalous_windows(files, fid_set, cfg: Config, C: int):
    """Anomalous-window patches + (fault id, file id) for the supervised metric stage."""
    fault2id = {ft: i for i, ft in enumerate(cfg.faults)}
    Pt, lab, fil = [], [], []
    for f in files:
        if f["fid"] not in fid_set or f["fault"] not in fault2id:
            continue
        Z = standardize(f)
        for s in win_starts(len(Z), cfg):
            if f["anom"][s:s + cfg.W].max() > 0:
                Pt.append(patchify(Z[s:s + cfg.W][None], cfg, C)[0])
                lab.append(fault2id[f["fault"]])
                fil.append(f["fid"])
    if not Pt:
        return np.empty((0, C * cfg.Np, cfg.P), "float32"), np.array([]), np.array([])
    return np.stack(Pt), np.array(lab), np.array(fil)
