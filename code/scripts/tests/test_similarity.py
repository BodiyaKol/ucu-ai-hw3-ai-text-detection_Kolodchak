import numpy as np
from sklearn.preprocessing import normalize

from src.data.similarity import jaccard, shingle_hashes, topk_cosine


def test_jaccard():
    a = shingle_hashes("the cat sat on the mat")
    assert jaccard(a, a) == 1.0
    b = shingle_hashes("completely different words are here now")
    assert jaccard(a, b) == 0.0
    assert jaccard(a, np.empty(0, dtype=np.int64)) == 0.0


def test_topk_cosine_excludes_self():
    z = normalize(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32))
    idx, sim = topk_cosine(z, z, k=1, exclude_self=True)
    assert idx[:, 0].tolist() == [1, 0, 1]
    assert sim.shape == (3, 1)
