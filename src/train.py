"""Model training / inference.

- train_reconstruction: trains the encoder on normal windows (MSE), early stopping.
- train_supcon: supervised-contrastive fine-tune using fault labels (Task 2 metric).
- save_model / load_model: checkpoint I/O.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.config import Config
from src.model import TinyTSAutoencoder, make_metric_head, metric_embed


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_reconstruction(model: TinyTSAutoencoder, Ptr, Pva, cfg: Config, device, verbose=True):
    """Self-supervised reconstruction on normal windows. Returns best val MSE."""
    torch.manual_seed(cfg.seed)
    tr = torch.tensor(Ptr)
    va = torch.tensor(Pva).to(device)
    loader = DataLoader(TensorDataset(tr), batch_size=cfg.batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def mse(x):
        rec, _ = model(x)
        return ((rec - x) ** 2).mean()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = mse(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        tl = tot / max(nb, 1)
        model.eval()
        with torch.no_grad():
            vl = mse(va).item()
        if vl < best - 1e-4:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (ep % 25 == 0 or ep == cfg.epochs - 1):
            print(f"[recon] epoch {ep:3d} | train MSE {tl:.4f} | val MSE {vl:.4f}")
        if bad >= cfg.patience:
            if verbose:
                print(f"[recon] early stop @ {ep} (best val {best:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def supcon_loss(z, labels, fileids, tau):
    """Supervised contrastive loss; positives = same fault from a DIFFERENT file."""
    B = z.size(0)
    sim = z @ z.t() / tau
    sim = sim - sim.max(1, keepdim=True).values.detach()
    self_mask = torch.eye(B, device=z.device).bool()
    same = labels[:, None] == labels[None, :]
    difff = fileids[:, None] != fileids[None, :]
    pos = same & difff & ~self_mask
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    pc = pos.sum(1)
    valid = pc > 0
    if not valid.any():
        return None
    loss = -(log_prob * pos).sum(1)[valid] / pc[valid]
    return loss.mean()


def train_supcon(model: TinyTSAutoencoder, head, sup_Pt, sup_lab, sup_fil, cfg: Config, device, verbose=True):
    """Fine-tune encoder + head with supervised-contrastive loss on fault labels."""
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=cfg.sup_lr)
    sp = torch.tensor(sup_Pt)
    sl = torch.tensor(sup_lab)
    sf = torch.tensor(sup_fil)
    for ep in range(cfg.sup_epochs):
        perm = torch.randperm(len(sp))
        model.train(); head.train()
        tot, nb = 0.0, 0
        for i in range(0, len(sp), cfg.sup_batch):
            b = perm[i:i + cfg.sup_batch]
            if len(b) < 8:
                continue
            z = metric_embed(model, head, sp[b].to(device))
            loss = supcon_loss(z, sl[b].to(device), sf[b].to(device), cfg.tau)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose and (ep % 20 == 0 or ep == cfg.sup_epochs - 1):
            print(f"[supcon] epoch {ep:3d} | loss {tot / max(nb, 1):.4f}")


def save_model(model, head, path):
    torch.save({"model": model.state_dict(),
                "head": (head.state_dict() if head is not None else None)}, path)


def load_model(model, head, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if head is not None and ckpt.get("head") is not None:
        head.load_state_dict(ckpt["head"])
    return model, head
