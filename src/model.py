"""Model: the Task 1 bottleneck Transformer autoencoder, plus a metric head for Task 2.

Encoder: per-sensor patch tokens -> 1 Transformer layer (full self-attention) ->
attention pooling to a single L2-normalized window embedding z (the bottleneck).
Decoder: rebuild all patches from z alone (reconstruction objective).
`encode(x)` returns z; this same z is used for anomaly scoring and retrieval.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class TinyTSAutoencoder(nn.Module):
    def __init__(self, C: int, cfg: Config):
        super().__init__()
        d, Np, P = cfg.d_model, cfg.Np, cfg.P
        self.proj = nn.Linear(P, d)
        self.pos_emb = nn.Embedding(Np, d)
        self.sensor_emb = nn.Embedding(C, d)
        enc = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ff, 0.0, "gelu",
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, 1)
        self.pool_q = nn.Parameter(torch.randn(d))
        dec = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ff, 0.0, "gelu",
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dec, 1)
        self.recon = nn.Linear(d, P)
        self.register_buffer("sensor_ids", torch.arange(C).repeat_interleave(Np))
        self.register_buffer("pos_ids", torch.arange(Np).repeat(C))

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """patches [B, n, P] -> L2-normalized window embedding [B, d_model]."""
        e = self.proj(patches) + self.pos_emb(self.pos_ids) + self.sensor_emb(self.sensor_ids)
        h = self.encoder(e)
        a = torch.softmax((h @ self.pool_q) / math.sqrt(h.size(-1)), dim=1)
        return F.normalize((a.unsqueeze(-1) * h).sum(1), dim=-1)

    def forward(self, patches: torch.Tensor):
        z = self.encode(patches)
        tok = z.unsqueeze(1) + self.pos_emb(self.pos_ids) + self.sensor_emb(self.sensor_ids)
        recon = self.recon(self.decoder(tok))
        return recon, z


def make_metric_head(cfg: Config) -> nn.Sequential:
    """Projection head trained with supervised-contrastive loss (Task 2 learned metric)."""
    d = cfg.d_model
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))


def metric_embed(model: TinyTSAutoencoder, head: nn.Sequential, patches: torch.Tensor) -> torch.Tensor:
    return F.normalize(head(model.encode(patches)), dim=-1)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
