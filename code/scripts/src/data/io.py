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
