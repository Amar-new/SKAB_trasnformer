"""Task 2 — retrieval, indexing, episode roll-up, and IR metrics (numpy / sklearn only).

Cosine retrieval over a flat normalized matrix (brute-force exact; swap in FAISS for
fleet scale). Same-file neighbours are excluded so we retrieve genuinely similar
episodes elsewhere. Relevance = same fault type, different file.
"""
import numpy as np
from collections import defaultdict

from src.config import Config


# ---------- IR metrics ----------
def average_precision(rel: np.ndarray) -> float:
    if rel.sum() == 0:
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    return float((hits / ranks * rel).sum() / rel.sum())


def ndcg_at(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    ideal = np.sort(rel)[::-1]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------- retrieval ----------
def retrieve(G, q, fid, k, exclude_same_file=True):
    """Return (top-k indices, top-k cosine scores) for query index q."""
    sims = G @ G[q]
    valid = np.ones(len(G), bool); valid[q] = False
    if exclude_same_file:
        valid &= (fid != fid[q])
    sims = np.where(valid, sims, -1e9)
    order = np.argsort(-sims)
    order = order[valid[order]]
    return order[:k], sims[order[:k]]


def evaluate_retrieval(G, fid, fault, cfg: Config):
    """Average precision@k/recall@k, mAP, nDCG, top-1 same-fault + per-fault P@5."""
    k_list = list(cfg.k_list)
    Pm = {k: [] for k in k_list}; Rm = {k: [] for k in k_list}
    APs, ND, T1 = [], [], []
    perf = defaultdict(list)
    for q in range(len(G)):
        valid = (fid != fid[q])
        sims = np.where(valid, G @ G[q], -1e9)
        order = np.argsort(-sims); order = order[valid[order]]
        rel = (fault[order] == fault[q]).astype(int)
        nrel = rel.sum()
        if nrel == 0:
            continue
        for k in k_list:
            Pm[k].append(rel[:k].mean())
            Rm[k].append(rel[:k].sum() / nrel)
        APs.append(average_precision(rel))
        ND.append(ndcg_at(rel, max(k_list)))
        T1.append(rel[0])
        perf[fault[q]].append(rel[:5].mean())
    out = {f"P@{k}": float(np.mean(Pm[k])) for k in k_list}
    out.update({f"R@{k}": float(np.mean(Rm[k])) for k in k_list})
    out["mAP"] = float(np.mean(APs)) if APs else 0.0
    out[f"nDCG@{max(k_list)}"] = float(np.mean(ND)) if ND else 0.0
    out["top1_same_fault"] = float(np.mean(T1)) if T1 else 0.0
    out["per_fault_P@5"] = {ft: float(np.mean(v)) for ft, v in perf.items()}
    out["n_queries"] = len(APs)
    return out


# ---------- episodes ----------
def runs(a: np.ndarray):
    """Contiguous runs of 1s -> list of (start, end) (end exclusive)."""
    out, i, n = [], 0, len(a)
    while i < n:
        if a[i] == 1:
            j = i
            while j < n and a[j] == 1:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def build_episode_gallery(files, per_file_embed, cfg: Config, fid_filter=None):
    """Fault episodes = contiguous anomalous runs; embedding = centroid of its windows.

    `per_file_embed` maps fid -> (starts, E). Returns (EP, EF, EI) numpy arrays.
    """
    EP, EF, EI = [], [], []
    for f in files:
        if fid_filter is not None and f["fid"] not in fid_filter:
            continue
        if f["fid"] not in per_file_embed:
            continue
        starts, E = per_file_embed[f["fid"]]
        starts = np.array(starts)
        for (s, e) in runs(f["anom"]):
            m = (starts < e) & (starts + cfg.W > s)
            if m.sum() == 0:
                continue
            v = E[m].mean(0)
            v = v / (np.linalg.norm(v) + 1e-9)
            EP.append(v); EF.append(f["fault"]); EI.append(f["fid"])
    return np.asarray(EP, "float32"), np.array(EF), np.array(EI)


# ---------- file-disjoint split ----------
def split_files_by_fault(files, cfg: Config):
    """Split files per fault into (metric-train fids, eval fids), disjoint."""
    byf = defaultdict(list)
    for f in files:
        if f["fault"] in cfg.faults:
            byf[f["fault"]].append(f["fid"])
    rng = np.random.default_rng(cfg.seed)
    train_fids, eval_fids = set(), set()
    for ft, fids in byf.items():
        fids = list(fids); rng.shuffle(fids)
        h = max(1, len(fids) // 2)
        train_fids |= set(fids[:h]); eval_fids |= set(fids[h:])
    return train_fids, eval_fids
