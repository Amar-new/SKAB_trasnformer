"""Embedding extraction.

Turns standardized telemetry into window embeddings using the trained encoder
(optionally through the learned-metric head). This is the boundary between the torch
model and the numpy-based anomaly/retrieval modules.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.features import win_starts, patchify
from src.data import standardize


@torch.no_grad()
def embed_patches(model, patches, device, head=None, batch=256):
    """[N, n, P] -> [N, d] embeddings (metric head applied if given)."""
    model.eval()
    if head is not None:
        head.eval()
    out = []
    for i in range(0, len(patches), batch):
        b = torch.tensor(patches[i:i + batch]).to(device)
        if head is not None:
            z = F.normalize(head(model.encode(b)), dim=-1)
        else:
            z = model.encode(b)
        out.append(z.cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.recon.in_features), "float32")


@torch.no_grad()
def embed_file(model, f, cfg: Config, C, device, head=None, want_recon=False):
    """Embed every window of one file. Returns (starts, E[, recon_err])."""
    Z = standardize(f)
    starts = win_starts(len(Z), cfg)
    if not starts:
        return None
    Pt = patchify(np.stack([Z[s:s + cfg.W] for s in starts]), cfg, C)
    model.eval()
    if want_recon and head is None:
        E, R = [], []
        for i in range(0, len(Pt), 256):
            b = torch.tensor(Pt[i:i + 256]).to(device)
            rec, z = model(b)
            E.append(z.cpu().numpy())
            R.append(((rec - b) ** 2).mean(-1).mean(-1).cpu().numpy())
        return starts, np.concatenate(E), np.concatenate(R)
    return starts, embed_patches(model, Pt, device, head=head), None


def build_window_index(model, files, cfg: Config, C, device, head=None,
                       fid_filter=None, anomalous_only=False):
    """Embed all (filtered) windows into a flat index with metadata.

    Returns dict: G [N, d], fid, fault, anom, start (all aligned numpy arrays).
    """
    G, FID, FAULT, ANOM, START = [], [], [], [], []
    for f in files:
        if fid_filter is not None and f["fid"] not in fid_filter:
            continue
        res = embed_file(model, f, cfg, C, device, head=head)
        if res is None:
            continue
        starts, E, _ = res
        for s, e in zip(starts, E):
            is_anom = int(f["anom"][s:s + cfg.W].max() > 0)
            if anomalous_only and not is_anom:
                continue
            G.append(e); FID.append(f["fid"]); FAULT.append(f["fault"])
            ANOM.append(is_anom); START.append(s)
    return dict(G=np.asarray(G, "float32"), fid=np.array(FID), fault=np.array(FAULT),
                anom=np.array(ANOM), start=np.array(START))
