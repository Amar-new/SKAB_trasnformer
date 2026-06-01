"""Preprocessing: load SKAB (or a synthetic fallback) and fit per-file scalers.

SKAB is a set of semicolon-delimited CSVs. Each file is a separate recording with
its own operating point, so we standardize EACH file by a StandardScaler fit on its
own *leading-normal* rows (the anomaly-free file uses its first 80% of rows). This
per-file standardization is essential — a single global scaler makes other files'
normal segments look anomalous (see README / report).
"""
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import Config

_META = {"datetime", "anomaly", "changepoint", "Unnamed: 0"}
_FAULT_FOLDERS = ["valve1", "valve2", "other", "anomaly-free"]


def fault_of(path: str) -> str:
    p = path.replace("\\", "/").lower()
    for f in _FAULT_FOLDERS:
        if f in p:
            return f
    return "other"


def load_skab(data_dir: str):
    """Return (files, sensor_cols). `files` is a list of dicts with keys
    vals [T, C] float32, anom [T] int, fault str, name str, fid int."""
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required to read SKAB CSVs") from e

    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    files, cols0 = [], None
    for path in paths:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            continue
        cols = [c for c in df.columns if c not in _META]
        if len(cols) < 4:
            continue
        cols0 = cols0 or cols
        if not set(cols0).issubset(df.columns):
            continue
        vals = df[cols0].astype("float32").values
        anom = (df["anomaly"].fillna(0).astype(int).values
                if "anomaly" in df.columns else np.zeros(len(df), int))
        files.append(dict(vals=vals, anom=anom, fault=fault_of(path),
                          name=os.path.basename(path), fid=len(files)))
    return files, cols0


def _gen_series(T, offset, fault, C, rng):
    t = np.arange(T)
    lat = np.sin(2 * np.pi * t / 97) + 0.5 * np.sin(2 * np.pi * t / 53)
    x = np.outer(lat, rng.normal(0, 1, C)) + rng.normal(0, 0.15, (T, C)) + offset
    a = np.zeros(T, int)
    if fault != "anomaly-free":
        s = rng.integers(T // 3, 2 * T // 3)
        L = rng.integers(80, 160)
        ch = rng.choice(C, size=2, replace=False)
        if fault == "valve1":
            x[s:s + L][:, ch] += 2.5                                  # level up
        elif fault == "valve2":
            x[s:s + L][:, ch] -= 2.5                                  # level down
        else:
            x[s:s + L][:, ch] += 1.5 * np.sin(2 * np.pi * np.arange(L) / 7)[:, None]  # oscillation
        a[s:s + L] = 1
    return x.astype("float32"), a


def make_synthetic(cfg: Config, C: int = 8, files_per_fault: int = 8):
    """A 3-fault synthetic dataset so the pipeline runs without SKAB."""
    rng = np.random.default_rng(cfg.seed)
    files = []
    vals, anom = _gen_series(9000, np.zeros(C), "anomaly-free", C, rng)
    files.append(dict(vals=vals, anom=anom, fault="anomaly-free", name="synthetic_normal", fid=0))
    for ft in cfg.faults:
        for _ in range(files_per_fault):
            vals, anom = _gen_series(2000, rng.normal(0, 3, C), ft, C, rng)
            files.append(dict(vals=vals, anom=anom, fault=ft, name=f"synthetic_{ft}", fid=len(files)))
    cols = [f"sensor_{i}" for i in range(C)]
    return files, cols


def get_data(cfg: Config):
    """Single entry point. Returns (files, sensor_cols, C, using_skab)."""
    files, cols = load_skab(cfg.data_dir)
    using_skab = len(files) > 0 and any(f["fault"] != "anomaly-free" for f in files)
    if not using_skab:
        print(f"[data] SKAB not found at '{cfg.data_dir}' -> using synthetic fallback.")
        files, cols = make_synthetic(cfg)
    else:
        counts = {ft: sum(1 for f in files if f["fault"] == ft) for ft in _FAULT_FOLDERS}
        print(f"[data] Loaded SKAB: {len(files)} files {counts}")
    return files, cols, len(cols), using_skab


def lead_len(anom: np.ndarray, cfg: Config) -> int:
    """Number of leading rows treated as that file's normal reference."""
    w = np.where(anom == 1)[0]
    fa = int(w[0]) if len(w) else len(anom)
    return fa if fa >= 200 else min(cfg.lead_n, len(anom))


def fit_scalers(files, cfg: Config):
    """Attach a per-file StandardScaler ('scaler') and leading length ('L')."""
    for f in files:
        if f["fault"] == "anomaly-free":
            n = len(f["vals"])
            f["scaler"] = StandardScaler().fit(f["vals"][:int(0.8 * n)])
            f["L"] = n
        else:
            L = lead_len(f["anom"], cfg)
            f["L"] = L
            f["scaler"] = StandardScaler().fit(f["vals"][:max(L, 50)])
    return files


def standardize(f) -> np.ndarray:
    return f["scaler"].transform(f["vals"]).astype("float32")
