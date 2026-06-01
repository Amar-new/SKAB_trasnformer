"""End-to-end Task 1: preprocess -> features -> train encoder -> embed -> anomaly detection.

Usage:
    python scripts/run_task1.py                      # synthetic if no SKAB
    python scripts/run_task1.py --data-dir data/SKAB --epochs 200
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import get_data, fit_scalers
from src.features import build_normal_pool
from src.model import TinyTSAutoencoder, count_params
from src.train import train_reconstruction, get_device
from src.embed import embed_file
from src.anomaly import score_file, detection_metrics, f1_at_threshold


def parse_args():
    c = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=c.data_dir)
    ap.add_argument("--epochs", type=int, default=c.epochs)
    ap.add_argument("--seed", type=int, default=c.seed)
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(data_dir=a.data_dir, epochs=a.epochs, seed=a.seed)
    device = get_device()
    print(f"[task1] device={device}  W={cfg.W} P={cfg.P} dim={cfg.d_model}")

    files, cols, C, _ = get_data(cfg)
    fit_scalers(files, cfg)
    Ptr, Pva = build_normal_pool(files, cfg, C)
    print(f"[task1] normal windows: train {len(Ptr)}, val {len(Pva)}")

    model = TinyTSAutoencoder(C, cfg).to(device)
    print(f"[task1] parameters: {count_params(model):,}")
    train_reconstruction(model, Ptr, Pva, cfg, device)

    # per-file embedding-kNN scoring
    Y, EMB, REC, PRED = [], [], [], []
    for f in files:
        if f["fault"] == "anomaly-free":
            continue
        res = embed_file(model, f, cfg, C, device, want_recon=True)
        if res is None:
            continue
        starts, E, R = res
        T = len(f["vals"])
        emb_s, rec_s = score_file(starts, E, R, f["L"], T, cfg)
        ucl = np.quantile(emb_s[:max(f["L"], 1)], cfg.q_thresh) * 4 / 3
        Y.append(f["anom"][:len(emb_s)]); EMB.append(emb_s); REC.append(rec_s)
        PRED.append((emb_s > ucl).astype(int))

    Y = np.concatenate(Y); EMB = np.concatenate(EMB)
    REC = np.concatenate(REC); PRED = np.concatenate(PRED)
    base = Y.mean()

    me = detection_metrics(Y, EMB)
    mr = detection_metrics(Y, REC)
    f1 = f1_at_threshold(Y, PRED)
    print("\n================ Task 1 — anomaly detection ================")
    print(f"timesteps {len(Y)} | base rate {base:.3f} | trivial-allpos F1 {2*base/(1+base):.3f}")
    print(f"embedding-distance : AUROC {me['auroc']:.3f}  AP {me['ap']:.3f}")
    print(f"reconstruction err : AUROC {mr['auroc']:.3f}  AP {mr['ap']:.3f}")
    print(f"embedding F1 @ per-file UCL : {f1['f1']:.3f}  (P={f1['precision']:.2f}, R={f1['recall']:.2f})")
    print("note: AUROC/AP are threshold-free and the honest model-quality read; F1 is the deployed view.")


if __name__ == "__main__":
    main()
