"""End-to-end Task 2: train encoder -> supervised-contrastive metric -> index -> retrieve.

Reports baseline (Task 1 embedding) vs learned-metric retrieval, at window and
fault-episode level, on a file-disjoint split.

Usage:
    python scripts/run_task2.py
    python scripts/run_task2.py --data-dir data/SKAB --epochs 200 --sup-epochs 80
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import get_data, fit_scalers
from src.features import build_normal_pool, collect_anomalous_windows
from src.model import TinyTSAutoencoder, make_metric_head, count_params
from src.train import train_reconstruction, train_supcon, save_model, load_model, get_device
from src.embed import build_window_index, embed_file
from src.retrieval import evaluate_retrieval, build_episode_gallery, split_files_by_fault


def parse_args():
    c = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=c.data_dir)
    ap.add_argument("--epochs", type=int, default=c.epochs)
    ap.add_argument("--sup-epochs", type=int, default=c.sup_epochs)
    ap.add_argument("--seed", type=int, default=c.seed)
    ap.add_argument("--model-path", default=c.checkpoint_path)
    return ap.parse_args()


def show(title, m):
    pf = ", ".join(f"{k}={v:.3f}" for k, v in m["per_fault_P@5"].items())
    print(f"\n--- {title} ({m['n_queries']} queries) ---")
    print(f"  P@1={m['P@1']:.3f}  P@5={m['P@5']:.3f}  P@10={m['P@10']:.3f}")
    print(f"  mAP={m['mAP']:.3f}  nDCG@10={m['nDCG@10']:.3f}  top-1 same-fault={m['top1_same_fault']:.3f}")
    print(f"  per-fault P@5: {pf}")


def main():
    a = parse_args()
    cfg = Config(data_dir=a.data_dir, epochs=a.epochs, sup_epochs=a.sup_epochs, seed=a.seed,
                 checkpoint_path=a.model_path)
    device = get_device()
    print(f"[task2] device={device}")

    files, cols, C, _ = get_data(cfg)
    fit_scalers(files, cfg)
    Ptr, Pva = build_normal_pool(files, cfg, C)

    model = TinyTSAutoencoder(C, cfg).to(device)
    print(f"[task2] parameters: {count_params(model):,}")
    if os.path.exists(cfg.checkpoint_path):
        load_model(model, None, cfg.checkpoint_path, device)
        print(f"[task2] loaded Task 1 model from {cfg.checkpoint_path} (skipping reconstruction training)")
    else:
        print(f"[task2] no checkpoint at '{cfg.checkpoint_path}'; training reconstruction from scratch")
        train_reconstruction(model, Ptr, Pva, cfg, device)
        save_model(model, None, cfg.checkpoint_path)
        print(f"[task2] model saved -> {cfg.checkpoint_path}")

    # file-disjoint split + supervised-contrastive metric on TRAIN files
    train_fids, eval_fids = split_files_by_fault(files, cfg)
    print(f"[task2] metric-train files {len(train_fids)} | eval files {len(eval_fids)}")
    sup_Pt, sup_lab, sup_fil = collect_anomalous_windows(files, train_fids, cfg, C)
    print(f"[task2] supervised anomalous windows: {sup_Pt.shape}")
    head = make_metric_head(cfg).to(device)
    if len(sup_Pt) >= 8:
        train_supcon(model, head, sup_Pt, sup_lab, sup_fil, cfg, device)
    else:
        print("[task2] not enough supervised windows; skipping metric stage")

    # ---- window-level retrieval on EVAL files (anomalous-only gallery) ----
    idx_base = build_window_index(model, files, cfg, C, device, head=None,
                                  fid_filter=eval_fids, anomalous_only=True)
    idx_learn = build_window_index(model, files, cfg, C, device, head=head,
                                   fid_filter=eval_fids, anomalous_only=True)
    print("\n================ Task 2 — window-level retrieval ================")
    show("baseline (Task 1 embedding)", evaluate_retrieval(idx_base["G"], idx_base["fid"], idx_base["fault"], cfg))
    show("learned metric (SupCon)", evaluate_retrieval(idx_learn["G"], idx_learn["fid"], idx_learn["fault"], cfg))

    # ---- fault-episode-level retrieval ----
    pfe_base, pfe_learn = {}, {}
    for f in files:
        if f["fid"] not in eval_fids:
            continue
        rb = embed_file(model, f, cfg, C, device, head=None)
        rl = embed_file(model, f, cfg, C, device, head=head)
        if rb is not None:
            pfe_base[f["fid"]] = (rb[0], rb[1])
        if rl is not None:
            pfe_learn[f["fid"]] = (rl[0], rl[1])
    EPb, EFb, EIb = build_episode_gallery(files, pfe_base, cfg, fid_filter=eval_fids)
    EPl, EFl, EIl = build_episode_gallery(files, pfe_learn, cfg, fid_filter=eval_fids)
    print("\n================ Task 2 — fault-episode-level retrieval ================")
    if len(EPb):
        show("baseline", evaluate_retrieval(EPb, EIb, EFb, cfg))
        show("learned metric", evaluate_retrieval(EPl, EIl, EFl, cfg))
    else:
        print("not enough episodes to evaluate")


if __name__ == "__main__":
    main()
