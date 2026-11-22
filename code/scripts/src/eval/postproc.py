"""Post-hoc tools on class probabilities: blending, class weights (thresholds), fast macro-F1."""
import numpy as np

W_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)              # ensemble weight of model A
B_GRID = np.round(np.exp(np.linspace(-1.0, 1.0, 21)), 3)      # multiplicative class weights 0.37..2.72


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fast_macro_f1(y: np.ndarray, p: np.ndarray, k: int = 3) -> float:
    """Macro-F1 on integer labels (same as sklearn average='macro', much faster for grids)."""
    f = 0.0
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f += 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / k


def blend(pa: np.ndarray, pb: np.ndarray, w: float) -> np.ndarray:
    return w * pa + (1.0 - w) * pb


def predict(p: np.ndarray, weights=None) -> np.ndarray:
    return (p if weights is None else p * weights).argmax(1)


def tune_weight(pa, pb, y):
    """Best ensemble weight of pa on a grid -> (w, macro-F1)."""
    scores = [fast_macro_f1(y, predict(blend(pa, pb, w))) for w in W_GRID]
    i = int(np.argmax(scores))
    return float(W_GRID[i]), float(scores[i])


def tune_bias(p, y, fixed: int = 1):
    """Best multiplicative class weights (class `fixed` keeps weight 1) -> (weights, macro-F1)."""
    others = [c for c in range(p.shape[1]) if c != fixed]
    best_w, best_s = np.ones(p.shape[1]), fast_macro_f1(y, predict(p))
    for b0 in B_GRID:
        for b1 in B_GRID:
            w = np.ones(p.shape[1])
            w[others[0]], w[others[1]] = b0, b1
            s = fast_macro_f1(y, predict(p, w))
            if s > best_s:
                best_w, best_s = w, s
    return best_w, float(best_s)


def simplex(k: int, step: float = 0.1):
    """All weight vectors of length k on a grid with the given step that sum to 1."""
    import itertools
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k):
        if sum(c) == n:
            yield np.array(c, dtype=float) / n


def hierarchical(p: np.ndarray, s: np.ndarray, a: float) -> np.ndarray:
    """Keep P(ai) from p (order human, ai, fusion); split the rest into human/fusion by
    a * specialist P(human | human-or-fusion) + (1 - a) * p's own split."""
    rest = p[:, 0] + p[:, 2]
    own = p[:, 0] / np.maximum(rest, 1e-9)
    share_h = a * s[:, 0] + (1.0 - a) * own
    out = np.empty_like(p)
    out[:, 1] = p[:, 1]
    out[:, 0] = rest * share_h
    out[:, 2] = rest * (1.0 - share_h)
    return out
