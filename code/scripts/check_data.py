"""M0: sanity check of competition files (rows, ids, labels, lengths, submission format)."""
import argparse
import json

import pandas as pd

from src.config import CODE_DIR, get_data_root
from src.data.io import LABELS, count_words, iter_csv


def scan(path, has_label: bool):
    """Read in chunks, keep only metadata + lengths (no full texts in RAM)."""
    cols = ["id", "lang", "domain"] + (["label"] if has_label else [])
    parts, n_null = [], 0
    for chunk in iter_csv(path):
        n_null += int(chunk["text"].isna().sum())
        part = chunk[cols].copy()
        part["n_words"] = count_words(chunk["text"])
        part["n_chars"] = chunk["text"].fillna("").astype(str).str.len().astype("int32")
        parts.append(part)
    return pd.concat(parts, ignore_index=True), n_null


def summarize(df: pd.DataFrame, n_null: int, has_label: bool) -> dict:
    q = df["n_words"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    s = {
        "rows": len(df),
        "unique_ids": int(df["id"].nunique()),
        "null_text": n_null,
        "lang": df["lang"].value_counts().to_dict(),
        "domain": df["domain"].value_counts().to_dict(),
        "words_quantiles": {str(k): float(v) for k, v in q.items()},
        "share_words_gt_400": float((df["n_words"] > 400).mean()),
    }
    if has_label:
        s["label"] = df["label"].value_counts().to_dict()
        s["label_x_lang"] = pd.crosstab(df["label"], df["lang"]).to_dict()
        s["domain_x_label"] = pd.crosstab(df["domain"], df["label"]).to_dict(orient="index")
        s["median_words_by_label"] = df.groupby("label")["n_words"].median().to_dict()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true", help="use data/mini")
    args = ap.parse_args()

    root = get_data_root(args.mini)
    print(f"data root: {root}")
    train, null_tr = scan(root / "train.csv", has_label=True)
    test, null_te = scan(root / "test.csv", has_label=False)
    sub = pd.read_csv(root / "sample_submission.csv")

    report = {
        "train": summarize(train, null_tr, True),
        "test": summarize(test, null_te, False),
        "checks": {
            "sample_columns": list(sub.columns),
            "sample_rows": len(sub),
            "sample_order_matches_test": bool(sub["id"].tolist() == test["id"].tolist()),
            "train_test_id_overlap": len(set(train["id"]) & set(test["id"])),
            "unknown_labels": sorted(set(train["label"]) - set(LABELS)),
        },
    }

    for name in ("train", "test"):
        r = report[name]
        print(f"\n== {name}: rows={r['rows']} unique_ids={r['unique_ids']} null_text={r['null_text']}")
        print("lang:", r["lang"])
        print("domain:", r["domain"])
        print("words quantiles:", r["words_quantiles"])
        print(f"share >400 words: {r['share_words_gt_400']:.3f}")
    print("\n== train label x lang")
    print(pd.crosstab(train["label"], train["lang"], margins=True))
    print("\n== train domain x label")
    print(pd.crosstab(train["domain"], train["label"], margins=True))
    print("\nmedian words by label:", report["train"]["median_words_by_label"])
    print("\n== checks:", report["checks"])

    out = CODE_DIR / "results" / f"m0_data_check{'_mini' if args.mini else ''}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
