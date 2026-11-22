"""M4.3: final ensemble of all available models + human-vs-fusion specialist (hierarchical).

Blends: weights on a simplex grid (step 0.1) over the chosen 3-class models.
Specialist: P(ai) from the blend; the human/fusion split mixed with the specialist (a in 0..1).
Honest estimate: tuned on one grouped half of val, evaluated on the other (2-fold cross-fit).
Test submissions use parameters tuned on the full val.
"""
import json

import numpy as np
import pandas as pd

from src.config import CODE_DIR, get_output_dir, load_config
from src.data.io import LABEL2ID, LABELS
from src.eval.metrics import evaluate
from src.eval.postproc import fast_macro_f1, hierarchical, simplex, softmax
from src.submission import write_submission

MODELS = {  # name -> (folder, file suffix)
    "tfidf": ("outputs/05_tfidf_pool/m4_tfidf_pool", "scores"),
    "xlmr_m3": ("outputs/04_encoder/m3_xlmr", "probs_chunks"),
    "xlmr_v2": ("outputs/06_xlmr_v2/m4_xlmr_v2", "probs_chunks"),
}
SPEC = ("outputs/07_mdeberta_hf/m4_mdeberta_hf", "probs_chunks")
CONFIGS = [  # (name, models, use specialist)
    ("tfidf+xlmr_m3 (previous best)", ["tfidf", "xlmr_m3"], False),
    ("tfidf+xlmr_v2", ["tfidf", "xlmr_v2"], False),
    ("tfidf+xlmr_m3+xlmr_v2", ["tfidf", "xlmr_m3", "xlmr_v2"], False),
    ("tfidf+xlmr_v2 + specialist", ["tfidf", "xlmr_v2"], True),
    ("tfidf+xlmr_m3+xlmr_v2 + specialist", ["tfidf", "xlmr_m3", "xlmr_v2"], True),
]
A_GRID = np.round(np.linspace(0, 1, 11), 2)


def load(folder, suffix, split, labels):
    path = CODE_DIR / folder / f"{split}_{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if suffix == "scores":
        p = softmax(df[[f"score_{c}" for c in labels]].to_numpy())
    else:
        p = df[[f"prob_{c}" for c in labels]].to_numpy()
    return pd.DataFrame(p, index=df["id"].to_numpy())


def main():
    cfg = load_config("configs/protocol.yaml")
    probs = {"val": {}, "test": {}}
    for split in probs:
        for name, (folder, suffix) in MODELS.items():
            d = load(folder, suffix, split, LABELS)
            if d is not None:
                probs[split][name] = d
        s = load(*SPEC, split, ["human", "fusion"])
        if s is not None:
            probs[split]["spec"] = s
    print("available:", sorted(probs["val"]))

    ids_v = probs["val"]["tfidf"].index.to_numpy()
    ids_te = probs["test"]["tfidf"].index.to_numpy()
    P = {k: v.loc[ids_v].to_numpy() for k, v in probs["val"].items()}
    PT = {k: v.loc[ids_te].to_numpy() for k, v in probs["test"].items()}

    ann = pd.read_csv(CODE_DIR / "data/annotations/annotations.csv.gz",
                      usecols=["id", "label", "domain"], dtype={"label": "string"})
    split = pd.read_csv(CODE_DIR / cfg["split"]["file"], usecols=["id", "group"])
    val = pd.DataFrame({"id": ids_v}).merge(ann, on="id").merge(split, on="id")
    assert (val["id"].to_numpy() == ids_v).all()
    y = val["label"].map(LABEL2ID).to_numpy()
    halves = val["group"].to_numpy() % 2

    def combine(src, names, w, a):
        p = sum(wi * src[n] for wi, n in zip(w, names))
        return p if a is None else hierarchical(p, src["spec"], a)

    def fit(mask, names, use_spec):
        best = (-1.0, None, None)
        sub = {k: v[mask] for k, v in P.items()}
        for w in simplex(len(names)):
            for a in (A_GRID if use_spec else [None]):
                s = fast_macro_f1(y[mask], combine(sub, names, w, a).argmax(1))
                if s > best[0]:
                    best = (s, w, a)
        return best[1], best[2]

    def fmt(names, w, a):
        txt = " ".join(f"{n}={wi:.1f}" for n, wi in zip(names, w))
        return txt + ("" if a is None else f" spec_a={a}")

    rows, reports, finals = [], {}, {}

    def report(name, pred, note=""):
        rep = evaluate(val["label"], [LABELS[i] for i in pred], val["domain"])
        reports[name] = rep
        rows.append({"experiment": name, "macro_f1": round(rep["macro_f1"], 4),
                     **{f"f1_{c}": round(v, 4) for c, v in rep["f1"].items()}, "note": note})

    for n in MODELS:
        if n in P:
            report(f"single {n}", P[n].argmax(1))
    for name, names, use_spec in CONFIGS:
        if any(n not in P for n in names) or (use_spec and "spec" not in P):
            print(f"skip {name}: missing model")
            continue
        oof, notes = np.zeros(len(y), dtype=int), []
        for h in (0, 1):
            tune, ev = halves != h, halves == h
            w, a = fit(tune, names, use_spec)
            oof[ev] = combine({k: v[ev] for k, v in P.items()}, names, w, a).argmax(1)
            notes.append(fmt(names, w, a))
        report(f"{name} [honest]", oof, " | ".join(notes))
        w, a = fit(np.ones(len(y), dtype=bool), names, use_spec)
        finals[name] = (names, w, a)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    honest = [r for r in rows if "[honest]" in r["experiment"]]
    best = max(honest, key=lambda r: r["macro_f1"])
    rb = reports[best["experiment"]]
    print(f"\nbest honest: {best['experiment']}  macro-F1={best['macro_f1']}")
    print("per-domain macro-F1:", {k: round(v, 4) for k, v in rb["f1_by_domain"].items()})
    print("confusion (rows=true, cols=pred, order human/ai/fusion):", rb["confusion"])

    out = get_output_dir("m4_final")
    sample = CODE_DIR / "outputs/03_tfidf/m2_tfidf/submission.csv"   # validated full-order id list
    shares = {}
    for i, (name, (names, w, a)) in enumerate(finals.items()):
        pred = combine(PT, names, w, a).argmax(1)
        sub = write_submission(ids_te, [LABELS[j] for j in pred], sample, out / f"sub_{i}.csv")
        shares[name] = sub["label"].value_counts(normalize=True).round(3).to_dict()
        print(f"sub_{i}.csv  {name}  params: {fmt(names, w, a)}  test share: {shares[name]}")
    table.to_csv(out / "experiments.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({"table": rows, "finals": {k: fmt(*v) for k, v in finals.items()},
                   "test_pred_share": shares, "best_honest": best["experiment"],
                   "best_by_domain": rb["f1_by_domain"], "best_confusion": rb["confusion"]},
                  f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
