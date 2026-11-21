"""M1.4: EDA figures + tables from local derived files.

Inputs: data/annotations/annotations.csv.gz (M1.1), data/similarity/pairs.csv.gz (M1.2),
data/mini (texts for examples), outputs/similarity/pairs_mini.csv.gz (example pair).
Outputs: results/figures/eda_*.png + compact tables in stdout.
"""
import difflib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CODE_DIR, get_data_root
from src.data.io import LABELS

FIG = CODE_DIR / "results" / "figures"
COLORS = {"human": "#4C72B0", "fusion": "#DD8452", "ai": "#55A868", "test": "#8C8C8C"}
STYLE = ["avg_word_len", "ttr", "comma_per_word", "digit_ratio"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print(f"saved figure {name}")


def plot_balance(ann):
    tr, te = ann[ann["split"] == "train"], ann[ann["split"] == "test"]
    share = pd.crosstab(tr["domain"], tr["label"], normalize="index")[LABELS]
    dom = pd.concat({"train": tr["domain"].value_counts(normalize=True),
                     "test": te["domain"].value_counts(normalize=True)}, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    share.plot.barh(stacked=True, ax=ax[0], color=[COLORS[c] for c in LABELS])
    ax[0].set_title("Label share per domain (train)")
    ax[0].set_xlabel("share")
    dom.plot.barh(ax=ax[1], color=[COLORS["human"], COLORS["test"]])
    ax[1].set_title("Domain share: train vs test")
    save(fig, "eda_balance.png")
    print("\n== label share per language (train)")
    print(pd.crosstab(tr["lang"], tr["label"], normalize="index")[LABELS].round(3).to_string())


def plot_lengths(ann):
    tr = ann[ann["split"] == "train"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for col, c in (("n_tok_xlmr", COLORS["human"]), ("n_tok_mdeberta", COLORS["fusion"])):
        ax[0].hist(ann[col].clip(upper=4000), bins=80, alpha=0.5, label=col, color=c)
    for x in (512, 1024):
        ax[0].axvline(x, color="k", ls="--", lw=1)
    ax[0].set_title("Tokens per text (clipped at 4000), lines at 512 / 1024")
    ax[0].legend()
    ax[1].boxplot([tr.loc[tr["label"] == c, "n_words"] for c in LABELS],
                  tick_labels=LABELS, showfliers=False)
    ax[1].set_title("Words per text by label (train, no outliers)")
    save(fig, "eda_lengths.png")
    print("\n== share of texts > 512 tokens (XLM-R) by domain")
    print(ann.groupby("domain")["n_tok_xlmr"].apply(lambda s: (s > 512).mean()).round(3).to_string())


def plot_style(ann):
    tr = ann[ann["split"] == "train"]
    med = tr.groupby(["domain", "label"])[STYLE].median()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, f in zip(axes.ravel(), STYLE):
        med[f].unstack()[LABELS].plot.bar(ax=ax, color=[COLORS[c] for c in LABELS],
                                          legend=(f == STYLE[0]))
        ax.set_title(f"{f}: median per domain")
        ax.set_xlabel("")
    save(fig, "eda_style_by_domain.png")
    print("\n== style medians within domain (is the label order the same in every domain?)")
    for f in STYLE:
        m = med[f].unstack()[LABELS]
        print(f"\n{f}\n{m.round(4).to_string()}")
        print(f"  domains with ai > human: {(m['ai'] > m['human']).sum()}/{len(m)}, "
              f"ai > fusion > human: {((m['ai'] > m['fusion']) & (m['fusion'] > m['human'])).sum()}/{len(m)}")
    print("\n== avg_word_len median by language x label")
    print(tr.groupby(["lang", "label"])["avg_word_len"].median().unstack()[LABELS].round(3).to_string())


def plot_jaccard(pairs):
    best = pairs.sort_values("jac", ascending=False).drop_duplicates(["split", "id"])
    tr = best["split"] == "train"
    groups = [("train human", tr & (best["label"] == "human"), COLORS["human"]),
              ("train fusion", tr & (best["label"] == "fusion"), COLORS["fusion"]),
              ("train ai", tr & (best["label"] == "ai"), COLORS["ai"]),
              ("test -> train", best["split"] == "test", COLORS["test"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 51)
    for name, sel, c in groups:
        ax.hist(best.loc[sel, "jac"], bins=bins, histtype="step", density=True, lw=2, label=name, color=c)
    ax.axvline(0.2, color="k", ls="--", lw=1, label="group threshold 0.2")
    ax.set_yscale("log")
    ax.set_xlabel("best-neighbour word 3-gram Jaccard")
    ax.set_title("How similar is each text to its closest train text?")
    ax.legend()
    save(fig, "eda_jaccard.png")
    f = best[tr & (best["label"] == "fusion")]
    print(f"\nfusion: best Jaccard >= 0.95 (almost unchanged): {(f['jac'] >= 0.95).mean():.3f}; "
          f"< 0.2 (no visible source in train): {(f['jac'] < 0.2).mean():.3f}")


def word_diff(a: str, b: str, n_words: int) -> str:
    """Word-level diff of the first n_words: [-removed-] {+added+}."""
    a, b = a.split()[:n_words], b.split()[:n_words]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out += a[i1:i2]
            continue
        if i2 > i1:
            out.append("[-" + " ".join(a[i1:i2]) + "-]")
        if j2 > j1:
            out.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(out)


def show_examples(seed=42):
    mini = pd.read_csv(get_data_root(mini=True) / "train.csv")
    print("\n== one example per class (mini sample, first 300 chars)")
    for c in LABELS:
        r = mini[mini["label"] == c].sample(1, random_state=seed).iloc[0]
        print(f"\n[{c} | {r['lang']} | {r['domain']} | {len(r['text'].split())} words]")
        print(textwrap.fill(r["text"][:300], 100))

    pm = pd.read_csv(CODE_DIR / "outputs/similarity/pairs_mini.csv.gz")
    pm = pm[(pm["split"] == "train") & (pm[["label", "cand_label"]].apply(set, axis=1) == {"human", "fusion"})]
    if pm.empty:
        print("\n(no human-fusion pair in the mini sample)")
        return
    p = pm.sort_values("jac", ascending=False).iloc[0]
    human_id, fusion_id = (p["id"], p["cand_id"]) if p["label"] == "human" else (p["cand_id"], p["id"])
    text = mini.set_index("id")["text"]
    print(f"\n== human -> fusion (Jaccard {p['jac']:.2f}), first 80 words: [-removed-] {{+added+}}")
    print(textwrap.fill(word_diff(text[human_id], text[fusion_id], 80), 100))


def main():
    ann = pd.read_csv(CODE_DIR / "data/annotations/annotations.csv.gz", dtype={"label": "string"})
    pairs = pd.read_csv(CODE_DIR / "data/similarity/pairs.csv.gz",
                        usecols=["split", "id", "jac", "label"], dtype={"label": "string"})
    plot_balance(ann)
    plot_lengths(ann)
    plot_style(ann)
    plot_jaccard(pairs)
    show_examples()


if __name__ == "__main__":
    main()
