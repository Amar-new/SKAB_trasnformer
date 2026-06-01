# SKAB Telemetry Modeling & Episode Retrieval

Transformer-based time-series modeling (Task 1) and retrieval of similar historical
episodes (Task 2) for industrial telemetry, evaluated on the
[Skoltech Anomaly Benchmark (SKAB)](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab).

A small Transformer **autoencoder** encodes per-sensor patch tokens into a single
window embedding. Trained only on normal data, the embedding (a) detects anomalies by
distance to a file's own normal behaviour (Task 1) and (b) retrieves similar fault
episodes (Task 2), optionally sharpened by a supervised-contrastive **learned metric**.

> Runs out of the box on a built-in **synthetic** dataset (no download needed). Point
> it at SKAB to use the real data.

## Repository structure

```
skab-telemetry-retrieval/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # all hyper-parameters (Config dataclass)
│   ├── data.py          # PREPROCESSING: load SKAB / synthetic, per-file standardization
│   ├── features.py      # FEATURE GENERATION: windowing + per-sensor patch tokenization
│   ├── model.py         # MODEL: bottleneck Transformer autoencoder + metric head
│   ├── train.py         # TRAINING/INFERENCE: reconstruction + supervised-contrastive
│   ├── embed.py         # EMBEDDING EXTRACTION: windows/files -> vectors, build index
│   ├── anomaly.py       # Task 1 scoring (per-file kNN) + detection metrics
│   └── retrieval.py     # Task 2 RETRIEVAL/INDEXING + IR metrics + episode roll-up
└── scripts/
    ├── run_task1.py     # end-to-end anomaly detection (AUROC / AP / F1)
    └── run_task2.py     # end-to-end retrieval (baseline vs learned metric)
```

Each of the five requested components maps to a module: **preprocessing** → `data.py`,
**feature generation** → `features.py`, **model training/inference** → `train.py`
(+`model.py`), **embedding extraction** → `embed.py`, **retrieval/indexing** →
`retrieval.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Python 3.9+; CPU is fine (the model is ~19k parameters). A GPU is used automatically if
available.

## Data (SKAB)

1. Download SKAB (Kaggle link above, or the [waico/SKAB](https://github.com/waico/SKAB)
   repo).
2. Place the CSVs under `data/SKAB/` keeping the folder layout:

   ```
   data/SKAB/
   ├── anomaly-free/   anomaly-free.csv
   ├── valve1/         *.csv
   ├── valve2/         *.csv
   └── other/          *.csv
   ```

The loader reads semicolon-delimited CSVs recursively, uses the folder name as the
fault label, and treats `anomaly` as the per-timestep label. **If `data/SKAB/` is absent
or empty, a synthetic 3-fault dataset is generated automatically** so every command
still runs.

## Run

From the repository root:

```bash
# Task 1 — anomaly detection
python scripts/run_task1.py --data-dir data

# Task 2 — retrieval of similar historical episodes
python scripts/run_task2.py --data-dir data
```

Common flags: `--epochs`, `--sup-epochs`, `--seed`, `--data-dir`. Omit `--data-dir`
(or leave the default) to run on synthetic data.

## What you get

**Task 1** prints, on the natural full-file base rate:
- `embedding-distance` and `reconstruction-error` **AUROC / AP** (threshold-free model
  quality), and
- `F1` at a per-file threshold (the deployed operating point).

AUROC/AP are the honest model-quality metrics; F1 depends on the chosen threshold and
the base rate, so it is reported as the deployment view, not the headline.

**Task 2** prints **baseline vs learned-metric** retrieval at **window** and
**fault-episode** level: `precision@k`, `recall@k`, `mAP`, `nDCG`, `top-1 same-fault`,
and a per-fault breakdown. Evaluation uses a **file-disjoint split** (the learned metric
is trained on some files, evaluated only on held-out files) and excludes same-file
neighbours, so the numbers reflect generalization.

## Method (one paragraph)

Telemetry is standardized **per file** (each file scaled by its own leading-normal rows,
which is essential — a global scaler makes other files' normal data look anomalous).
Each 128-step window of 8 sensors is split into per-sensor 16-step patch tokens; a
1-layer Transformer encoder with attention pooling compresses the window to one
L2-normalized vector `z`. Trained by reconstruction on normal windows only, `z` lands
anomalies away from a file's normal cluster (Task 1, scored by per-file k-NN), and
clusters windows by behaviour for retrieval (Task 2). A light supervised-contrastive
stage uses the fault labels to make same-fault episodes nearer (Task 2 learned metric).

## Scaling notes

Retrieval uses exact brute-force cosine search (fine at SKAB scale). For a fleet,
uncomment `faiss-cpu` in `requirements.txt` and back the index with FAISS
HNSW/IVF-PQ — the interface (normalized matrix + metadata) is unchanged.
