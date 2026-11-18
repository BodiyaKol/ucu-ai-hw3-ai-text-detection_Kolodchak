# Code — HW3

All commands are run from the `code/` directory.

## Setup (local, CPU)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest scripts/tests -q

## Data (not committed, course-only license)
    kaggle competitions download -c ai-generated-text-detection-hw3-ucu-ai-2026 -p data/raw
    unzip -q data/raw/ai-generated-text-detection-hw3-ucu-ai-2026.zip -d data/raw
    python scripts/check_data.py          # full data sanity check
    python scripts/make_mini.py           # small local sample in data/mini
    python scripts/check_data.py --mini

## Layout
- `configs/` — experiment configs (YAML, `base:` inheritance)
- `scripts/src/` — reusable code (config, data, features, models, eval)
- `scripts/*.py` — entry points
- `notebooks/` — Kaggle kernels and the final demo notebook
- `results/` — small metric files and figures used in the report
- `experiments.md` — experiment log

## Kaggle kernels
Kernels clone this public repo, so push code before `kaggle kernels push`.

    kaggle kernels push -p notebooks/00_data
    kaggle kernels status bohdankolodchak/hw3-00-data
    kaggle kernels output bohdankolodchak/hw3-00-data -p outputs/00_data
