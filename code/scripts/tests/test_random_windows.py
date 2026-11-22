import numpy as np

from src.data.chunking import random_windows


class FakeTok:
    """Word length as token id; [101] ... [102] as special tokens."""

    def __call__(self, text, add_special_tokens=True):
        ids = [len(w) for w in text.split()]
        return {"input_ids": [101] + ids + [102] if add_special_tokens else ids}


def test_random_windows_shape_and_specials():
    rng = np.random.default_rng(0)
    ids = [list(range(1000)), [7, 8, 9]]
    w = random_windows(ids, FakeTok(), max_len=102, rng=rng)
    assert len(w) == 2
    assert w[0][0] == 101 and w[0][-1] == 102 and len(w[0]) == 102
    assert w[0][1] % 100 == 0                      # a chunk starts at a multiple of the body length
    assert w[1] == [101, 7, 8, 9, 102]             # short text unchanged


def test_random_windows_vary_between_draws():
    rng = np.random.default_rng(0)
    ids = [list(range(800))]
    starts = {random_windows(ids, FakeTok(), 102, rng)[0][1] for _ in range(20)}
    assert len(starts) > 1
