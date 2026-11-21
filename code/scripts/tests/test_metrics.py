import pytest

from src.eval.metrics import evaluate, macro_f1


def test_macro_f1_hand_computed():
    # human: P=1/2 R=1/2 F1=0.5 ; ai: P=2/3 R=1 F1=0.8 ; fusion: P=1 R=1/2 F1=2/3
    y_true = ["human", "human", "ai", "ai", "fusion", "fusion"]
    y_pred = ["human", "ai", "ai", "ai", "fusion", "human"]
    assert macro_f1(y_true, y_pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_perfect_and_missing_class():
    y = ["human", "ai", "fusion"]
    assert macro_f1(y, y) == 1.0
    # always "ai": F1(ai) = 0.5, other classes 0 -> 1/6
    assert macro_f1(y, ["ai", "ai", "ai"]) == pytest.approx(0.5 / 3)


def test_evaluate_by_domain():
    rep = evaluate(["ai", "human"], ["ai", "ai"], domains=["d1", "d2"])
    assert set(rep["f1_by_domain"]) == {"d1", "d2"}
    assert rep["confusion"][1][1] == 1   # true ai -> pred ai
