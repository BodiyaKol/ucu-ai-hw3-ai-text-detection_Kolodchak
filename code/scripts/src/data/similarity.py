"""Near-duplicate search: LSA cosine candidates + word 3-gram Jaccard verification."""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def shingle_hashes(text: str, n: int = 3) -> np.ndarray:
    """Sorted unique hashes of lowercase word n-grams (hash is stable within one process)."""
    w = text.lower().split()
    if not w:
        return np.empty(0, dtype=np.int64)
    if len(w) < n:
        return np.array([hash(" ".join(w))], dtype=np.int64)
    grams = (hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1))
    return np.unique(np.fromiter(grams, dtype=np.int64))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


def lsa_embed(train_texts, other_texts, n_components: int = 128, seed: int = 42):
    """TF-IDF -> TruncatedSVD fitted on train_texts only; L2-normalized float32 embeddings."""
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5,
                          max_features=100_000, dtype=np.float32)
    xt = vec.fit_transform(train_texts)
    k = max(1, min(n_components, xt.shape[1] - 1, xt.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    zt = normalize(svd.fit_transform(xt)).astype(np.float32)
    if len(other_texts):
        zo = normalize(svd.transform(vec.transform(other_texts))).astype(np.float32)
    else:
        zo = np.empty((0, k), dtype=np.float32)
    return zt, zo


def topk_cosine(queries: np.ndarray, base: np.ndarray, k: int,
                exclude_self: bool = False, batch: int = 1024):
    """Top-k most similar base rows for each query row (rows are L2-normalized).

    exclude_self=True assumes queries == base (row i is not its own neighbour).
    Returns (indices, similarities), sorted by similarity descending.
    """
    k = min(k, base.shape[0] - int(exclude_self))
    idx = np.empty((len(queries), k), dtype=np.int64)
    sim = np.empty((len(queries), k), dtype=np.float32)
    for s in range(0, len(queries), batch):
        scores = queries[s:s + batch] @ base.T
        if exclude_self:
            r = np.arange(scores.shape[0])
            scores[r, s + r] = -np.inf
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
        sim[s:s + batch] = np.take_along_axis(part_scores, order, axis=1)
    return idx, sim
