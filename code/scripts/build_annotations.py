"""M1.1: annotation table for the whole dataset (train + test), WITHOUT text.

One row per text: id, split, lang, domain, label, length/style stats, token counts.
All EDA plots are built locally from this table.
"""
import argparse
import time

import pandas as pd

from src.config import get_data_root, get_output_dir
from src.data.features import text_stats, token_lengths
from src.data.io import iter_csv

TOKENIZERS = {"xlmr": "FacebookAI/xlm-roberta-base", "mdeberta": "microsoft/mdeberta-v3-base"}
MARKERS = ["n_emdash", "n_curly_quotes", "n_markdown", "n_bullets"]


def load_tokenizers(names):
    toks = {}
    if not names:
        return toks
    from transformers import AutoTokenizer
    for short in names:
        tok = AutoTokenizer.from_pretrained(TOKENIZERS[short])
        tok.model_max_length = int(1e9)  # we only count tokens: silence length warnings
        toks[short] = tok
    return toks


def annotate(path, split, toks):
    parts = []
    for chunk in iter_csv(path):
        meta = ["id", "lang", "domain"] + (["label"] if "label" in chunk else [])
        part = chunk[meta].copy()
        part.insert(1, "split", split)
        stats = text_stats(chunk["text"])
        texts = chunk["text"].fillna("").astype(str).tolist()
        for short, tok in toks.items():
            stats[f"n_tok_{short}"] = token_lengths(texts, tok)
        parts.append(pd.concat([part, stats], axis=1))
        print(f"  {split}: {sum(len(p) for p in parts)} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--tokenizers", nargs="*", default=list(TOKENIZERS),
                    help="subset of: xlmr mdeberta (pass the flag with no values to skip)")
    args = ap.parse_args()

    root = get_data_root(args.mini)
    toks = load_tokenizers(args.tokenizers)
    t0 = time.time()
    df = pd.concat([annotate(root / "train.csv", "train", toks),
                    annotate(root / "test.csv", "test", toks)], ignore_index=True)

    out = get_output_dir("annotations") / ("annotations_mini.csv.gz" if args.mini else "annotations.csv.gz")
    df.to_csv(out, index=False, compression="gzip")
    print(f"\nsaved {out} rows={len(df)} cols={df.shape[1]} in {time.time() - t0:.0f}s")

    # Compact summary for the log.
    num = df.select_dtypes("number").columns
    train = df[df["split"] == "train"]
    print("\n== train medians by label")
    print(train.groupby("label")[num].median().T.round(3).to_string())
    print("\n== train means by label (LLM formatting markers)")
    print(train.groupby("label")[MARKERS].mean().T.round(3).to_string())
    for col in [c for c in num if c.startswith("n_tok_")]:
        print(f"{col}: median={df[col].median():.0f} "
              f"share>512={(df[col] > 512).mean():.3f} share>1024={(df[col] > 1024).mean():.3f}")


if __name__ == "__main__":
    main()
