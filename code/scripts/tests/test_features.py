import pandas as pd

from src.data.features import text_stats


def test_text_stats_basic():
    s = pd.Series(["Hello, world. This is — a test!\n\n**Bold**", ""])
    st = text_stats(s)
    assert st.loc[0, "n_words"] == 8
    assert st.loc[0, "n_sentences"] == 2
    assert st.loc[0, "n_paragraphs"] == 2
    assert st.loc[0, "n_emdash"] == 1
    assert st.loc[0, "n_markdown"] == 2
    # empty text: no crash, zeros
    assert st.loc[1, "n_words"] == 0
    assert st.loc[1, "ttr"] == 0.0
