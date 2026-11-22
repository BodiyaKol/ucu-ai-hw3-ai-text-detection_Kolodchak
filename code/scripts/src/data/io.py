"""Reading competition CSVs and simple text statistics."""
import pandas as pd

LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def iter_csv(path, chunksize: int = 20000, usecols=None):
    """Yield DataFrame chunks (train.csv is large; laptop RAM is limited)."""
    yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols)


def count_words(texts: pd.Series) -> pd.Series:
    """Whitespace word count; NaN text counts as 0 words."""
    return texts.fillna("").astype(str).str.split().str.len().astype("int32")


def load_train_val(root, split_file):
    """train.csv joined with the fixed split -> (train subset, val subset, all train rows)."""
    split = pd.read_csv(split_file, usecols=["id", "group", "fold", "subset"])
    full = pd.read_csv(root / "train.csv").merge(split, on="id", how="inner")
    tr = full[full["subset"] == "train"].reset_index(drop=True)
    va = full[full["subset"] == "val"].reset_index(drop=True)
    return tr, va, full


def train_pool(full: pd.DataFrame, val_fold: int) -> pd.DataFrame:
    """All train rows outside the validation fold (val and its relatives are never inside)."""
    return full[full["fold"] != val_fold].reset_index(drop=True)
