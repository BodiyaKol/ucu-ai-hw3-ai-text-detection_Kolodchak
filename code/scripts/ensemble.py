"""M4.1: post-hoc experiments on saved probabilities (no retraining).

E1 ensemble:      w * XLM-R + (1 - w) * TF-IDF
E2 class weights: argmax(p * t), t = (t_human, 1, t_fusion), tuned for macro-F1
Honest estimate: parameters tuned on one grouped half of val, evaluated on the other (2-fold cross-fit).
Final parameters for test are tuned on the full val.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import CODE_DIR, get_output_dir, load_config
from src.data.io import LABEL2ID, LABELS
from src.eval.metrics import evaluate
from src.eval.postproc import blend, predict, softmax, tune_bias, tune_weight
from src.submission import write_submission


def load_probs(path) -> tuple:
    """(ids, probabilities in LABELS order). TF-IDF decision scores are softmax-ed (= LogReg proba)."""
    df = pd.read_csv(path)
    if "score_ai" in df.columns:
        p = softmax(df[[f"score_{c}" for c in LABELS]].to_numpy())
    else:
        p = df[[f"prob_{c}" for c in LABELS]].to_numpy()
    return df["id"].to_numpy(), p


def aligned(ids_ref, ids, p):
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return p[pos.loc[ids_ref].to_numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf-dir", default="outputs/03_tfidf/m2_tfidf")
    ap.add_argument("--xlmr-dir", default="outputs/04_encoder/m3_xlmr")
    ap.add_argument("--mode", default="chunks")
    args = ap.parse_args()

    cfg = load_config("configs/protocol.yaml")
    tdir, xdir = CODE_DIR / args.tfidf_dir, CODE_DIR / args.xlmr_dir
    ids_v, px = load_probs(xdir / f"val_probs_{args.mode}.csv")
    pt = aligned(ids_v, *load_probs(tdir / "val_scores.csv"))
    ids_te, px_te = load_probs(xdir / f"test_probs_{args.mode}.csv")
    pt_te = aligned(ids_te, *load_probs(tdir / "test_scores.csv"))

    ann = pd.read_csv(CODE_DIR / "data/annotations/annotations.csv.gz",
                      usecols=["id", "label", "domain"], dtype={"label": "string"})
    split = pd.read_csv(CODE_DIR / cfg["split"]["file"], usecols=["id", "group"])
    val = pd.DataFrame({"id": ids_v}).merge(ann, on="id").merge(split, on="id")
    assert len(val) == len(ids_v) and (val["id"].to_numpy() == ids_v).all()
    y = val["label"].map(LABEL2ID).to_numpy()
    halves = val["group"].to_numpy() % 2          # grouped halves: relatives never split
    print(f"val={len(y)} halves={np.bincount(halves).tolist()} test={len(ids_te)}")

    rows, reports = [], {}

    def report(name, pred, note=""):
        rep = evaluate(val["label"], [LABELS[i] for i in pred], val["domain"])
        reports[name] = rep
        rows.append({"experiment": name, "macro_f1": round(rep["macro_f1"], 4),
                     **{f"f1_{c}": round(v, 4) for c, v in rep["f1"].items()}, "note": note})

    def fit(mask, use_w, use_b):
        w = tune_weight(px[mask], pt[mask], y[mask])[0] if use_w else 1.0
        b = tune_bias(blend(px[mask], pt[mask], w), y[mask])[0] if use_b else None
        return w, b

    def fmt(w, b):
        return f"w={w}" + ("" if b is None else f" t_human={b[0]:.3f} t_fusion={b[2]:.3f}")

    # references
    report("ref TF-IDF (M2)", predict(pt))
    report("ref XLM-R chunks (M3)", predict(px))
    report("E1 ensemble w=0.5 (no tuning)", predict(blend(px, pt, 0.5)))

    all_mask = np.ones(len(y), dtype=bool)
    experiments = {"E1_ensemble": (True, False), "E2_weights_xlmr": (False, True),
                   "E1E2_ensemble_weights": (True, True)}
    finals = {}
    for key, (use_w, use_b) in experiments.items():
        oof, notes = np.zeros(len(y), dtype=int), []
        for h in (0, 1):
            tune, ev = halves != h, halves == h
            w, b = fit(tune, use_w, use_b)
            oof[ev] = predict(blend(px[ev], pt[ev], w), b)
            notes.append(fmt(w, b))
        report(f"{key} (cross-fit, honest)", oof, " | ".join(notes))
        w, b = fit(all_mask, use_w, use_b)
        report(f"{key} (tuned on full val, optimistic)", predict(blend(px, pt, w), b), fmt(w, b))
        finals[key] = (w, b)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    honest = [r for r in rows if "honest" in r["experiment"]]
    best = max(honest, key=lambda r: r["macro_f1"])
    print(f"\nbest honest: {best['experiment']}  macro-F1={best['macro_f1']}")
    rb = reports[best["experiment"]]
    print("per-domain macro-F1:", {k: round(v, 4) for k, v in rb["f1_by_domain"].items()})
    print("confusion (rows=true, cols=pred, order human/ai/fusion):", rb["confusion"])

    out = get_output_dir("m4_ensemble")
    shares = {}
    for key, (w, b) in finals.items():
        pred = predict(blend(px_te, pt_te, w), b)
        sub = pd.DataFrame({"id": ids_te, "label": [LABELS[i] for i in pred]})
        sub.to_csv(out / f"sub_{key}.csv", index=False)
        shares[key] = sub["label"].value_counts(normalize=True).round(3).to_dict()
        print(f"test pred share {key}: {shares[key]}  params: {fmt(w, b)}")
    table.to_csv(out / "experiments.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({"table": rows, "finals": {k: fmt(*v) for k, v in finals.items()},
                   "test_pred_share": shares, "best_honest": best["experiment"],
                   "best_by_domain": rb["f1_by_domain"], "best_confusion": rb["confusion"]},
                  f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
