"""M1.2: nearest-neighbour similarity between texts, within (lang, domain) blocks.

Candidates: top-k cosine neighbours in a TF-IDF -> SVD (LSA) space fitted on TRAIN rows.
Verification: word 3-gram Jaccard on the texts (rewrite / copy similarity).
Train rows search train (self excluded); test rows search train
(checks the organisers' claim that related texts never cross the train/test split).
Output: pairs.csv.gz (split, id, cand_id, rank, cos, jac, label, cand_label).
"""
import argparse
import time

import pandas as pd

from src.config import get_data_root, get_output_dir
from src.data.similarity import jaccard, lsa_embed, shingle_hashes, topk_cosine

MIN_BLOCK = 20  # skip tiny blocks (only happens on the mini sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    root = get_data_root(args.mini)
    cols = ["id", "text", "lang", "domain"]
    train = pd.read_csv(root / "train.csv", usecols=cols + ["label"])
    test = pd.read_csv(root / "test.csv", usecols=cols)

    rows, t_all = [], time.time()
    for (lang, dom), tr in train.groupby(["lang", "domain"]):
        te = test[(test["lang"] == lang) & (test["domain"] == dom)]
        if len(tr) < MIN_BLOCK:
            print(f"skip {lang} {dom}: train={len(tr)}")
            continue
        t0 = time.time()
        zt, zs = lsa_embed(tr["text"].tolist(), te["text"].tolist(), args.dim)
        tr_sh = [shingle_hashes(t) for t in tr["text"]]
        te_sh = [shingle_hashes(t) for t in te["text"]]
        tr_ids = tr["id"].to_numpy()

        for split, q_ids, z, sh, excl in (("train", tr_ids, zt, tr_sh, True),
                                          ("test", te["id"].to_numpy(), zs, te_sh, False)):
            if len(q_ids) == 0:
                continue
            idx, sim = topk_cosine(z, zt, args.k, exclude_self=excl)
            for qi in range(len(q_ids)):
                for r in range(idx.shape[1]):
                    j = idx[qi, r]
                    rows.append((split, q_ids[qi], tr_ids[j], r, float(sim[qi, r]),
                                 jaccard(sh[qi], tr_sh[j])))
        print(f"{lang} {dom}: train={len(tr)} test={len(te)} {time.time() - t0:.0f}s", flush=True)

    pairs = pd.DataFrame(rows, columns=["split", "id", "cand_id", "rank", "cos", "jac"])
    lab = train.set_index("id")["label"]
    pairs["label"] = pairs["id"].map(lab)          # NaN for test queries
    pairs["cand_label"] = pairs["cand_id"].map(lab)

    out = get_output_dir("similarity") / ("pairs_mini.csv.gz" if args.mini else "pairs.csv.gz")
    pairs.to_csv(out, index=False, compression="gzip")
    print(f"\nsaved {out} rows={len(pairs)} in {time.time() - t_all:.0f}s")

    # ---- compact summary for the log ----
    best = pairs.sort_values("jac", ascending=False).drop_duplicates(["split", "id"])
    tb = best[best["split"] == "train"]
    print("\n== train: best-Jaccard neighbour, counts (row = query label, col = neighbour label)")
    print(pd.crosstab(tb["label"], tb["cand_label"]))
    print("\n== train: median Jaccard of best neighbour")
    print(tb.pivot_table(index="label", columns="cand_label", values="jac", aggfunc="median").round(3))
    print("\n== best-neighbour Jaccard quantiles by split / query label")
    q = best.groupby(["split", "label"], dropna=False)["jac"].quantile([0.5, 0.9, 0.99])
    print(q.unstack().round(3))
    print("\n== top-1 cosine quantiles by split / query label")
    top1 = pairs[pairs["rank"] == 0]
    print(top1.groupby(["split", "label"], dropna=False)["cos"].quantile([0.5, 0.9]).unstack().round(3))
    for split, b in best.groupby("split"):
        shares = {thr: round(float((b["jac"] > thr).mean()), 3) for thr in (0.05, 0.1, 0.2, 0.3, 0.5)}
        print(f"{split}: share of queries with best Jaccard > thr: {shares}")


if __name__ == "__main__":
    main()
