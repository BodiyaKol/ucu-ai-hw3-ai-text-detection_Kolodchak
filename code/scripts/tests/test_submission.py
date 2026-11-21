import pandas as pd
import pytest

from src.submission import write_submission


def test_write_submission_aligns_order(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "label": ["ai"] * 3}).to_csv(tmp_path / "sample.csv", index=False)
    sub = write_submission(["c", "a", "b"], ["fusion", "human", "ai"], tmp_path / "sample.csv", tmp_path / "sub.csv")
    assert sub["id"].tolist() == ["a", "b", "c"]
    assert sub["label"].tolist() == ["human", "ai", "fusion"]
    assert list(pd.read_csv(tmp_path / "sub.csv").columns) == ["id", "label"]


def test_write_submission_rejects_bad_label(tmp_path):
    pd.DataFrame({"id": ["a"], "label": ["ai"]}).to_csv(tmp_path / "sample.csv", index=False)
    with pytest.raises(AssertionError):
        write_submission(["a"], ["AI"], tmp_path / "sample.csv", tmp_path / "sub.csv")
