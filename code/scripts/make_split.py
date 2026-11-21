"""M1.3: pseudo-groups (near-duplicate components) + fixed grouped train/val split.

Inputs (local): annotations (M1.1) and similarity pairs (M1.2).
Output: splits/split.csv.gz with columns id, group, fold, subset ("train" / "val" / "").
"""
import argparse

import numpy as np
import pandas as pd

from src.config import CODE_DIR, get_output_dir, load_config
from src.data.groups import build_groups, grouped_folds


def dist(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts(normalize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    args = ap.parse_args()

    cfg = load_config("configs/protocol.yaml")
    sp, seed = cfg["split"], cfg["seed"]
    if args.mini:
        ann_path = CODE_DIR / "outputs/annotations/annotations_mini.csv.gz"
        pairs_path = CODE_DIR / "outputs/similarity/pairs_mini.csv.gz"
        out = get_output_dir("split") / "split_mini.csv.gz"
    else:
        ann_path = CODE_DIR / "data/annotations/annotations.csv.gz"
        pairs_path = CODE_DIR / "data/similarity/pairs.csv.gz"
        out = CODE_DIR / sp["file"]
        out.parent.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(ann_path, usecols=["id", "split", "lang", "domain", "label"], dtype={"label": "string"})
    test = ann[ann["split"] == "test"]
    tr = ann[ann["split"] == "train"].reset_index(drop=True)
    pairs = pd.read_csv(pairs_path, usecols=["split", "id", "cand_id", "jac"])
    edges = pairs[(pairs["split"] == "train") & (pairs["jac"] > sp["jaccard_threshold"])]

    # 1. groups
    tr["group"] = build_groups(tr["id"], edges)
    sizes = tr["group"].value_counts()
    print(f"train rows={len(tr)} edges={len(edges)} groups={len(sizes)} "
          f"singletons={(sizes == 1).sum()}")
    print("group size counts:", sizes.value_counts().sort_index().head(12).to_dict())
    big = sizes.head(5)
    print("largest groups (size, domains):",
          [(int(n), tr.loc[tr["group"] == g, "domain"].unique().tolist()) for g, n in big.items()])
    comp = tr.groupby("group")["label"].agg(lambda s: "+".join(sorted(set(s))))
    print("group label composition:", comp.value_counts().head(8).to_dict())
    multi = tr["group"].map(sizes) > 1
    print("share of rows in multi-text groups by label:",
          tr.assign(multi=multi).groupby("label")["multi"].mean().round(3).to_dict())

    # 2. grouped stratified folds
    strata = (tr["label"] + "|" + tr["lang"] + "|" + tr["domain"]).to_numpy()
    tr["fold"] = grouped_folds(strata, tr["group"].to_numpy(), sp["n_folds"], seed)
    print("fold sizes:", tr["fold"].value_counts().sort_index().to_dict())

    # 3. fixed samples
    is_val = tr["fold"] == sp["val_fold"]
    val_idx = tr[is_val].sample(n=min(sp["val_size"], int(is_val.sum())), random_state=seed).index
    trn_idx = tr[~is_val].sample(n=min(sp["train_sample"], int((~is_val).sum())), random_state=seed).index
    tr["subset"] = ""
    tr.loc[trn_idx, "subset"] = "train"
    tr.loc[val_idx, "subset"] = "val"

    # 4. checks
    g_tr = set(tr.loc[tr["subset"] == "train", "group"])
    g_va = set(tr.loc[tr["subset"] == "val", "group"])
    g_pool_tr = set(tr.loc[~is_val, "group"])
    print(f"\ngroup overlap: train-sample/val-sample={len(g_tr & g_va)} "
          f"train-pool/val-pool={len(g_pool_tr & set(tr.loc[is_val, 'group']))}")

    # what a RANDOM split would leak: val rows whose group also appears in the rest of train
    rnd_val = tr.sample(n=len(val_idx), random_state=seed)
    rest_groups = set(tr.drop(rnd_val.index)["group"])
    leak = rnd_val["group"].isin(rest_groups)
    print(f"random split: share of val rows with a related text in train = {leak.mean():.3f}",
          rnd_val.assign(leak=leak).groupby("label")["leak"].mean().round(3).to_dict())

    sub_tr, sub_va = tr[tr["subset"] == "train"], tr[tr["subset"] == "val"]
    for col in ("label", "lang", "domain"):
        frames = {"train": dist(sub_tr, col), "val": dist(sub_va, col)}
        if col != "label":
            frames["test"] = dist(test, col)
        print(f"\n== {col} distribution")
        print(pd.concat(frames, axis=1).round(3).to_string())

    tr[["id", "group", "fold", "subset"]].to_csv(out, index=False, compression="gzip")
    print(f"\nsaved {out}  train={len(sub_tr)} val={len(sub_va)}")


if __name__ == "__main__":
    main()
