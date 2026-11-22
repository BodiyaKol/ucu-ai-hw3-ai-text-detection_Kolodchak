"""Competition metric (macro-F1, exactly as on Kaggle) + per-class / per-domain breakdown."""
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.data.io import LABELS


def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    """sklearn f1_score(average='macro') over `labels`; a never-predicted class gets F1 = 0."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def evaluate(y_true, y_pred, domains=None, labels=LABELS) -> dict:
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "f1": {c: float(v) for c, v in zip(labels, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),  # rows = true, cols = pred
    }
    if domains is not None:
        df = pd.DataFrame({"t": list(y_true), "p": list(y_pred), "d": list(domains)})
        out["f1_by_domain"] = {d: macro_f1(g["t"], g["p"], labels) for d, g in df.groupby("d")}
    return out
