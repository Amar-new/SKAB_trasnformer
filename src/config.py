"""Central configuration for the SKAB telemetry modeling + retrieval pipeline.

All hyper-parameters live here so the scripts and modules stay clean. Scripts can
override any field from the command line (see scripts/run_task1.py).
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    seed: int = 0
    data_dir: str = "data"        # point at the SKAB folder; if absent -> synthetic fallback

    # --- windowing / tokenization ---
    W: int = 128                       # window length (timesteps, ~seconds at 1 Hz)
    P: int = 16                        # patch length -> one token = P steps of ONE sensor
    stride: int = 32                   # window stride
    lead_n: int = 400                  # per-file leading rows treated as that file's normal reference

    # --- model ---
    d_model: int = 32                  # encoder width = window-embedding dimension
    nhead: int = 2
    ff: int = 64

    # --- reconstruction training (Task 1 encoder) ---
    epochs: int = 200
    batch: int = 64
    lr: float = 1e-3
    patience: int = 20

    # --- supervised-contrastive metric (Task 2) ---
    sup_epochs: int = 80
    sup_batch: int = 128
    sup_lr: float = 5e-4
    tau: float = 0.2

    # --- anomaly scoring (Task 1) ---
    knn: int = 5                       # neighbours for embedding-distance score
    smooth_k: int = 30                 # rolling-mean window on the anomaly score
    q_thresh: float = 0.995            # quantile of leading-normal score for the per-file threshold (UCL)

    # --- retrieval (Task 2) ---
    k_list: Tuple[int, ...] = (1, 5, 10)
    faults: Tuple[str, ...] = ("valve1", "valve2", "other")

    # --- checkpoint ---
    checkpoint_path: str = "task1_model.pt"

    @property
    def Np(self) -> int:               # patches per sensor
        return self.W // self.P
