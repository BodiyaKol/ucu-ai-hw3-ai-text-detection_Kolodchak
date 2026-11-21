from src.data.chunking import build_windows, windows_for


def test_short_text_unchanged():
    assert windows_for([1, 2, 3], 10, "chunks") == [[1, 2, 3]]
    assert windows_for([1, 2, 3], 10, "head_tail") == [[1, 2, 3]]


def test_head_and_head_tail():
    ids = list(range(100))
    assert windows_for(ids, 10, "head") == [list(range(10))]
    assert windows_for(ids, 10, "head_tail", head_tokens=4) == [[0, 1, 2, 3] + list(range(94, 100))]


def test_chunks_cap_and_tiny_tail():
    w = windows_for(list(range(1000)), 100, "chunks", max_chunks=8)
    assert len(w) == 8 and w[0] == list(range(100))
    assert len(windows_for(list(range(210)), 100, "chunks", min_tail=32)) == 2   # 10-token tail dropped
    assert len(windows_for(list(range(250)), 100, "chunks", min_tail=32)) == 3   # 50-token tail kept


class FakeTokenizer:
    """CLS=0 ... SEP=2 template, like XLM-R; words map to ids 10, 11, ..."""

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + i for i, _ in enumerate(text.split())]
        return {"input_ids": [0] + ids + [2] if add_special_tokens else ids}


def test_build_windows_adds_special_tokens_within_max_len():
    windows, owner = build_windows([list(range(100, 120)), [7]], FakeTokenizer(), max_len=10, mode="chunks")
    assert all(len(w) <= 10 and w[0] == 0 and w[-1] == 2 for w in windows)
    assert windows[0] == [0] + list(range(100, 108)) + [2]
    assert owner.tolist() == [0, 0, 1]   # 20 tokens -> 8 + 8 (+4-token tail dropped, < min_tail), short text -> 1
