import numpy as np
import pandas as pd

from src.data.groups import build_groups, grouped_folds


def test_build_groups_chain():
    ids = pd.Series(["a", "b", "c", "d"])
    edges = pd.DataFrame({"id": ["a", "b"], "cand_id": ["b", "c"]})
    g = build_groups(ids, edges)
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]


def test_grouped_folds_keep_groups_together():
    groups = np.repeat(np.arange(20), 3)          # 20 groups x 3 rows
    strata = np.tile(["x", "y", "z"], 20)
    folds = grouped_folds(strata, groups, n_folds=4, seed=0)
    for g in range(20):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == {0, 1, 2, 3}
