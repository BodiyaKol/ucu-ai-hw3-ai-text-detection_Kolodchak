"""Write and validate the Kaggle submission (id,label; sample order; lowercase class names)."""
import pandas as pd

from src.data.io import LABELS


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == ["id", "label"], f"bad columns {list(sub.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub["id"].is_unique, "duplicate ids"
    assert sub["id"].tolist() == sample["id"].tolist(), "id order differs from sample_submission"
    assert sub["label"].isin(LABELS).all(), "unknown or missing labels"


def write_submission(ids, labels, sample_path, out_path) -> pd.DataFrame:
    pred = pd.DataFrame({"id": list(ids), "label": list(labels)})
    sample = pd.read_csv(sample_path)
    sub = sample[["id"]].merge(pred, on="id", how="left")  # align to sample order
    check_submission(sub, sample)
    sub.to_csv(out_path, index=False)
    return sub
