"""Token windows for texts longer than the model's maximum length."""
import numpy as np


def tokenize_no_special(tokenizer, texts, batch_size: int = 1000) -> list:
    """Token ids of each full text (no special tokens, no truncation)."""
    ids = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), add_special_tokens=False, truncation=False)
        ids.extend(enc["input_ids"])
    return ids


def windows_for(ids: list, body_len: int, mode: str, head_tokens: int = 128,
                max_chunks: int = 8, min_tail: int = 32) -> list:
    """Windows (without special tokens) of one text; body_len = max_len - number of special tokens."""
    if len(ids) <= body_len:
        return [ids]
    if mode == "head":
        return [ids[:body_len]]
    if mode == "head_tail":
        assert head_tokens < body_len
        return [ids[:head_tokens] + ids[-(body_len - head_tokens):]]
    if mode == "chunks":
        chunks = [ids[s:s + body_len] for s in range(0, len(ids), body_len)]
        if len(chunks) > 1 and len(chunks[-1]) < min_tail:
            chunks = chunks[:-1]
        return chunks[:max_chunks]
    raise ValueError(mode)


def special_wrap(tokenizer) -> tuple:
    """(prefix, suffix) special-token ids the tokenizer adds around a single sequence.

    Found from an example (transformers 5 removed build_inputs_with_special_tokens),
    e.g. XLM-R: ([<s>], [</s>]).
    """
    plain = tokenizer("hello world", add_special_tokens=False)["input_ids"]
    full = tokenizer("hello world", add_special_tokens=True)["input_ids"]
    for s in range(len(full) - len(plain) + 1):
        if full[s:s + len(plain)] == plain:
            return full[:s], full[s + len(plain):]
    raise ValueError("cannot locate the text inside the tokenizer's special-token template")


def build_windows(all_ids: list, tokenizer, max_len: int, mode: str, head_tokens: int = 128,
                  max_chunks: int = 8, min_tail: int = 32):
    """All windows with special tokens + owner (text index) of each window."""
    prefix, suffix = special_wrap(tokenizer)
    body = max_len - len(prefix) - len(suffix)
    windows, owner = [], []
    for i, ids in enumerate(all_ids):
        for w in windows_for(ids, body, mode, head_tokens, max_chunks, min_tail):
            windows.append(prefix + list(w) + suffix)
            owner.append(i)
    return windows, np.asarray(owner, dtype=np.int64)
