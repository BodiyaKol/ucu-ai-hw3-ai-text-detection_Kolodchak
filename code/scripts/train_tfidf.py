"""M2 baseline: TF-IDF (word + char n-grams) + linear models on the fixed grouped split.

Steps: majority-class floor -> small grid -> feature ablation -> full report of the best
-> val/test scores (for later ensembling) -> submission.csv -> optional random-split control.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

from src.config import CODE_DIR, _deep_merge, get_data_root, get_output_dir, load_config, set_seed
from src.data.io import LABELS, load_train_val
from src.eval.metrics import evaluate, macro_f1
from src.submission import write_submission


def make_model(name: str, C: float, seed: int):
    if name == "logreg":
        return LogisticRegression(C=C, max_iter=300, random_state=seed)
    if name == "linsvc":
        return LinearSVC(C=C, random_state=seed)
    raise ValueError(name)


def vectorize(feat_cfg: dict, train_texts, others: list) -> dict:
    """Fit each vectorizer on train texts only. Returns name -> (X_train, [X_other, ...])."""
    mats = {}
    for name, params in feat_cfg.items():
        t0 = time.time()
        p = dict(params, ngram_range=tuple(params["ngram_range"]))
        vec = TfidfVectorizer(dtype=np.float32, **p)
        xtr = vec.fit_transform(train_texts)
        mats[name] = (xtr, [vec.transform(t) for t in others])
        print(f"  {name}: vocab={len(vec.vocabulary_)} nnz/doc={xtr.nnz / xtr.shape[0]:.0f} "
              f"{time.time() - t0:.0f}s", flush=True)
    return mats


def stack(mats: dict, names: list, i=None):
    """hstack feature blocks; i=None -> train matrices, i=k -> k-th 'other' matrices."""
    return sp.hstack([mats[n][0] if i is None else mats[n][1][i] for n in names]).tocsr()


def save_scores(model, X, ids, path):
    """Decision scores per class (inputs for thresholds / ensembling in M4)."""
    df = pd.DataFrame(model.decision_function(X), columns=[f"score_{c}" for c in model.classes_])
    df.insert(0, "id", list(ids))
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tfidf/base.yaml")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--control", action="store_true", help="also train/eval on a RANDOM split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mini:
        cfg = _deep_merge(cfg, cfg.get("mini", {}))
    seed = cfg["seed"]
    set_seed(seed)
    root = get_data_root(args.mini)
    split_file = CODE_DIR / ("outputs/split/split_mini.csv.gz" if args.mini else cfg["split"]["file"])
    tr, va, full = load_train_val(root, split_file)
    test = pd.read_csv(root / "test.csv")
    out = get_output_dir(cfg["exp"] + ("_mini" if args.mini else ""))
    print(f"train={len(tr)} val={len(va)} test={len(test)} -> {out}")
    res = {"exp": cfg["exp"], "n_train": len(tr), "n_val": len(va)}

    # 0. floor: always predict the most frequent class
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr), 1)), tr["label"])
    res["dummy_macro_f1"] = macro_f1(va["label"], dummy.predict(np.zeros((len(va), 1))))
    print(f"dummy (most frequent) val macro-F1 = {res['dummy_macro_f1']:.4f}")

    names = list(cfg["features"])
    mats = vectorize(cfg["features"], tr["text"], [va["text"], test["text"]])
    Xtr, Xva, Xte = stack(mats, names), stack(mats, names, 0), stack(mats, names, 1)

    # 1. small grid (all features)
    res["grid"] = []
    for g in cfg["grid"]:
        t0 = time.time()
        m = make_model(g["model"], g["C"], seed).fit(Xtr, tr["label"])
        f1 = macro_f1(va["label"], m.predict(Xva))
        res["grid"].append({**g, "val_macro_f1": f1, "sec": round(time.time() - t0)})
        print(f"  grid {g}: val macro-F1 = {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
    best = max(res["grid"], key=lambda r: r["val_macro_f1"])
    print("best:", best)

    # 2. feature ablation with the best model
    res["ablation"] = {"+".join(names): best["val_macro_f1"]}
    for n in names:
        m = make_model(best["model"], best["C"], seed).fit(stack(mats, [n]), tr["label"])
        res["ablation"][n] = macro_f1(va["label"], m.predict(stack(mats, [n], 0)))
    print("ablation (val macro-F1):", {k: round(v, 4) for k, v in res["ablation"].items()})

    # 3. full report for the best model
    model = make_model(best["model"], best["C"], seed).fit(Xtr, tr["label"])
    pred = model.predict(Xva)
    rep = evaluate(va["label"], pred, va["domain"])
    res["best"] = {**best, **rep}
    print(f"\n== best {best['model']} C={best['C']}: val macro-F1 = {rep['macro_f1']:.4f}")
    print("per-class F1:", {k: round(v, 4) for k, v in rep["f1"].items()})
    print("per-domain macro-F1:", {k: round(v, 4) for k, v in rep["f1_by_domain"].items()})
    print("confusion (rows=true, cols=pred, order human/ai/fusion):", rep["confusion"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(va["label"], pred, labels=LABELS, normalize="true",
                                            values_format=".2f", ax=ax, colorbar=False)
    ax.set_title(f"{cfg['exp']} val (row-normalized)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)

    save_scores(model, Xva, va["id"], out / "val_scores.csv")
    save_scores(model, Xte, test["id"], out / "test_scores.csv")
    sub = write_submission(test["id"], model.predict(Xte), root / "sample_submission.csv", out / "submission.csv")
    res["test_pred_share"] = sub["label"].value_counts(normalize=True).round(3).to_dict()
    print("submission OK, predicted label share on test:", res["test_pred_share"])

    # 4. control: same pipeline on a RANDOM (non-grouped) split of the same size
    if args.control:
        rnd = full.sample(n=min(len(full), len(tr) + len(va)), random_state=seed)
        rva, rtr = rnd.iloc[:len(va)], rnd.iloc[len(va):]
        mats_r = vectorize(cfg["features"], rtr["text"], [rva["text"]])
        m = make_model(best["model"], best["C"], seed).fit(stack(mats_r, names), rtr["label"])
        res["random_split"] = evaluate(rva["label"], m.predict(stack(mats_r, names, 0)))
        print(f"\n== control: RANDOM split val macro-F1 = {res['random_split']['macro_f1']:.4f} "
              f"(grouped: {rep['macro_f1']:.4f}); per-class:",
              {k: round(v, 4) for k, v in res["random_split"]["f1"].items()})

    with open(out / "metrics.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
