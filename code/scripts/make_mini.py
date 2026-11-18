"""M0: create a small local sample (data/mini) for fast CPU smoke tests."""
import argparse

import pandas as pd

from src.config import CODE_DIR, get_data_root
from src.data.io import iter_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = get_data_root()
    out = CODE_DIR / "data" / "mini"
    out.mkdir(parents=True, exist_ok=True)

    # Uniform random sample of train, chunk by chunk (keeps RAM low).
    total = sum(len(c) for c in iter_csv(root / "train.csv", usecols=["id"]))
    frac = args.n_train / total
    parts = [c.sample(frac=frac, random_state=args.seed + i)
             for i, c in enumerate(iter_csv(root / "train.csv"))]
    train = pd.concat(parts, ignore_index=True)

    test = pd.read_csv(root / "test.csv", nrows=args.n_test)
    sub = pd.read_csv(root / "sample_submission.csv")
    sub = sub[sub["id"].isin(test["id"])]

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    sub.to_csv(out / "sample_submission.csv", index=False)
    print(f"mini: train={len(train)} test={len(test)} sub={len(sub)} -> {out}")
    print("train labels:", train["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
