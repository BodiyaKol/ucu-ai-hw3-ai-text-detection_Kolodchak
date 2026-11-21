import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.eval.postproc import blend, fast_macro_f1, predict, tune_bias, tune_weight


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    assert fast_macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_predict_weights_and_blend():
    p = np.array([[0.5, 0.3, 0.2]])
    assert predict(p)[0] == 0
    assert predict(p, np.array([1.0, 1.0, 3.0]))[0] == 2
    assert np.allclose(blend(np.ones((1, 3)), np.zeros((1, 3)), 0.25), 0.25)


def test_tuning_never_worse_than_default():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    pa, pb = rng.dirichlet(np.ones(3), 300), rng.dirichlet(np.ones(3), 300)
    _, s_w = tune_weight(pa, pb, y)
    assert s_w >= fast_macro_f1(y, predict(blend(pa, pb, 1.0)))
    _, s_b = tune_bias(pa, y)
    assert s_b >= fast_macro_f1(y, predict(pa))
