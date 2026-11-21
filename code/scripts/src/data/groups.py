"""Pseudo-groups of related texts and grouped, stratified folds."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold


def build_groups(ids: pd.Series, edges: pd.DataFrame) -> np.ndarray:
    """Connected components over edges (columns id, cand_id). Returns a group index per id."""
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    a = pos.reindex(edges["id"].to_numpy()).to_numpy()
    b = pos.reindex(edges["cand_id"].to_numpy()).to_numpy()
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok].astype(np.int64), b[ok].astype(np.int64)
    n = len(ids)
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def grouped_folds(strata: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per row; a group never spans two folds; strata proportions kept."""
    folds = np.full(len(strata), -1, dtype=np.int8)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(sgkf.split(np.zeros(len(strata)), strata, groups)):
        folds[va] = f
    return folds
