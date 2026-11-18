"""Project-wide helpers: paths, YAML config loading with inheritance, seeding."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

CODE_DIR = Path(__file__).resolve().parents[2]  # .../code
KAGGLE_INPUT = Path("/kaggle/input")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    """Load YAML; a `base:` key points to a parent YAML (relative to this file)."""
    path = Path(path)
    if not path.is_absolute():
        path = CODE_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_data_root(mini: bool = False) -> Path:
    """Folder with train.csv / test.csv / sample_submission.csv."""
    if mini:
        return CODE_DIR / "data" / "mini"
    if is_kaggle():
        # Search a few depths only (a recursive glob over /kaggle/input can be slow).
        for pattern in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv"):
            hits = sorted(KAGGLE_INPUT.glob(pattern))
            if hits:
                return hits[0].parent
        raise FileNotFoundError("train.csv not found under /kaggle/input")
    return CODE_DIR / "data" / "raw"


def get_output_dir(name: str) -> Path:
    root = Path("/kaggle/working") if is_kaggle() else CODE_DIR / "outputs"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    return out
