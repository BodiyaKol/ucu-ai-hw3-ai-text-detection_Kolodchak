"""Vectorized per-text length and style statistics (no model needed)."""
import re

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(0.0).astype("float32")


def text_stats(texts: pd.Series) -> pd.DataFrame:
    """Length, structure and style statistics for each text (index preserved)."""
    t = texts.fillna("").astype(str)
    words = t.str.split()
    n_words = words.str.len().astype("int32")
    n_chars = t.str.len().astype("int32")
    n_sent = t.str.count(r"[.!?]+(?:\s|$)").clip(lower=1)

    out = pd.DataFrame({
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": n_sent.astype("int32"),
        "n_lines": (t.str.count("\n") + 1).astype("int32"),
        "n_paragraphs": (t.str.count(r"\n\s*\n") + 1).astype("int32"),
    }, index=t.index)
    out["words_per_sentence"] = _safe_div(n_words, n_sent)
    out["avg_word_len"] = _safe_div(t.str.count(r"\w"), n_words)
    # Type-token ratio (lexical diversity); note: decreases with text length.
    out["ttr"] = words.map(lambda w: len({x.lower() for x in w}) / max(len(w), 1)).astype("float32")
    out["punct_ratio"] = _safe_div(t.str.count(r"[^\w\s]"), n_chars)
    out["comma_per_word"] = _safe_div(t.str.count(","), n_words)
    out["digit_ratio"] = _safe_div(t.str.count(r"\d"), n_chars)
    out["upper_ratio"] = _safe_div(t.str.count(r"[A-ZÄÖÜ]"), n_chars)
    # Typical LLM formatting markers.
    out["n_emdash"] = t.str.count("—").astype("int32")
    out["n_curly_quotes"] = t.str.count("[“”„‘’]").astype("int32")
    out["n_markdown"] = t.str.count(r"\*\*|^#+\s", flags=re.M).astype("int32")
    out["n_bullets"] = t.str.count(r"^\s*(?:[-*•]|\d+\.)\s", flags=re.M).astype("int32")
    return out


def token_lengths(texts: list, tokenizer, batch_size: int = 1000) -> np.ndarray:
    """Number of tokens per text (with special tokens, no truncation)."""
    lens = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=True, truncation=False)
        lens.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lens, dtype="int32")
