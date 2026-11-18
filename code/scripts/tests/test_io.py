import pandas as pd

from src.data.io import LABEL2ID, LABELS, count_words


def test_labels():
    assert LABELS == ["human", "ai", "fusion"]
    assert LABEL2ID["fusion"] == 2


def test_count_words():
    s = pd.Series(["one two  three", "", None, "Ein Satz."])
    assert count_words(s).tolist() == [3, 0, 0, 2]
