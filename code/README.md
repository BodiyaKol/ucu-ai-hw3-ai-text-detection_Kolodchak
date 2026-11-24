# Code — HW3 (human / ai / fusion text classification)

All commands are run from `code/`. Seed 42 everywhere (`configs/protocol.yaml`).

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest scripts/tests -q

## Quick check (CPU, ~5 min): `notebooks/demo_pipeline.ipynb`
Runs every step on a mini sample (data prep, 3 models, ensemble) and shows the full-scale results.
Works outside the repo too (e.g. Colab, badge in the root README): it clones the repo and downloads the data
with your own `kaggle.json` (the data cannot be redistributed).

## Full pipeline (Kaggle kernels clone this repo; push code first)
    kaggle kernels push -p notebooks/00_data                                    # format check + mini sample
    kaggle kernels push -p notebooks/01_annotations                             # per-text stats, token counts
    kaggle kernels push -p notebooks/02_similarity                              # near-duplicate pairs
    python scripts/make_split.py                                                # groups + fixed split (committed: splits/)
    python scripts/run_eda.py                                                   # EDA figures -> results/figures
    kaggle kernels push -p notebooks/03_tfidf                                   # baseline (grid, ablation, random-split control)
    kaggle kernels push -p notebooks/04_encoder --accelerator NvidiaTeslaT4     # XLM-R m3
    kaggle kernels push -p notebooks/05_tfidf_pool                              # TF-IDF on 150k texts
    kaggle kernels push -p notebooks/06_xlmr_v2 --accelerator NvidiaTeslaT4     # XLM-R v2 (100k, 3 ep, random chunks)
    kaggle kernels push -p notebooks/07_mdeberta_hf --accelerator NvidiaTeslaT4 # mDeBERTa human-vs-fusion specialist
    # download each kernel: kaggle kernels output bohdankolodchak/<kernel> -p outputs/<NN_name>
    python scripts/ensemble_final.py                                            # -> outputs/m4_final/sub_4.csv (final)

## Layout
- `configs/` — protocol + model configs (`base:` inheritance)
- `scripts/src/` — data (io, features, similarity, groups, chunking), models (encoder loop), eval (metric, post-processing)
- `scripts/*.py` — entry points; `scripts/tests/` — unit tests
- `notebooks/NN_*` — Kaggle kernels; `notebooks/demo_pipeline.ipynb` — demo
- `splits/split.csv.gz` — fixed grouped split; `results/` — metrics, logs, figures; `experiments.md` — log
