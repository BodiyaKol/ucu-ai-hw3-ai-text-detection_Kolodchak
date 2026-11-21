"""M3: fine-tune a multilingual encoder (XLM-R) on the fixed grouped split.

Train on one window per text (train_mode), evaluate the same model with several
long-text strategies (eval_modes: head / head_tail / chunks), save probabilities
for val/test, submission from the best val mode, metrics.json.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import CODE_DIR, _deep_merge, get_data_root, get_output_dir, load_config, set_seed
from src.data.chunking import build_windows, tokenize_no_special
from src.data.io import LABEL2ID, LABELS, load_train_val
from src.eval.metrics import evaluate, macro_f1
from src.models.encoder import aggregate, make_loader, predict_logits, softmax, train
from src.submission import write_submission


def load_model(cfg: dict):
    kw = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    if cfg.get("tiny_random"):  # local smoke test only: 1-layer random model, real tokenizer vocab
        conf = AutoConfig.from_pretrained(cfg["model_name"], num_hidden_layers=1, hidden_size=32,
                                          intermediate_size=64, num_attention_heads=2, **kw)
        return AutoModelForSequenceClassification.from_config(conf)
    return AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], **kw)


def texts_of(df: pd.DataFrame, prefix_meta: bool) -> list:
    if prefix_meta:
        return ("[" + df["domain"] + "] [" + df["lang"] + "] " + df["text"]).tolist()
    return df["text"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/encoder/xlmr.yaml")
    ap.add_argument("--mini", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mini:
        cfg = _deep_merge(cfg, cfg.get("mini", {}))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = bool(cfg["fp16"]) and device == "cuda"
    root = get_data_root(args.mini)
    split_file = CODE_DIR / ("outputs/split/split_mini.csv.gz" if args.mini else cfg["split"]["file"])
    tr, va, _ = load_train_val(root, split_file)
    test = pd.read_csv(root / "test.csv")
    tr = tr.head(cfg.get("n_train") or len(tr))
    va = va.head(cfg.get("n_val") or len(va))
    test = test.head(cfg.get("n_test") or len(test))
    out = get_output_dir(cfg["exp"] + ("_mini" if args.mini else ""))
    print(f"device={device} fp16={fp16} train={len(tr)} val={len(va)} test={len(test)} -> {out}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ids_tr = tokenize_no_special(tok, texts_of(tr, cfg["prefix_meta"]))
    ids_va = tokenize_no_special(tok, texts_of(va, cfg["prefix_meta"]))
    ids_te = tokenize_no_special(tok, texts_of(test, cfg["prefix_meta"]))
    print(f"tokenized in {time.time() - t0:.0f}s", flush=True)

    wkw = dict(head_tokens=cfg["head_tokens"], max_chunks=cfg["max_chunks"], min_tail=cfg["min_tail"])
    L, pad = cfg["max_len"], tok.pad_token_id
    win_tr, own_tr = build_windows(ids_tr, tok, L, cfg["train_mode"], **wkw)
    y_tr = tr["label"].map(LABEL2ID).to_numpy()[own_tr]
    win_va_head, own_va_head = build_windows(ids_va, tok, L, "head", **wkw)

    model = load_model(cfg).to(device)
    loader = make_loader(win_tr, y_tr, cfg["batch_size"], pad, shuffle=True, num_workers=cfg["num_workers"])
    res = {"exp": cfg["exp"], "model": cfg["model_name"], "n_train": len(tr), "n_val": len(va),
           "epoch_val_macro_f1": []}

    def on_epoch_end(ep):
        logits = aggregate(predict_logits(model, win_va_head, cfg["eval_batch_size"], pad, device, fp16),
                           own_va_head, len(va))
        f1 = macro_f1(va["label"], [LABELS[i] for i in logits.argmax(1)])
        res["epoch_val_macro_f1"].append(f1)
        print(f"== epoch {ep}: val macro-F1 (head) = {f1:.4f}", flush=True)

    t0 = time.time()
    train(model, loader, cfg, device, fp16, on_epoch_end)
    res["train_sec"] = round(time.time() - t0)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")

    # evaluate every long-text strategy with the same model
    res["modes"], test_probs = {}, {}
    for mode in cfg["eval_modes"]:
        t0 = time.time()
        for split, df, ids in (("val", va, ids_va), ("test", test, ids_te)):
            w, own = build_windows(ids, tok, L, mode, **wkw)
            probs = softmax(aggregate(predict_logits(model, w, cfg["eval_batch_size"], pad, device, fp16),
                                      own, len(df)))
            pdf = pd.DataFrame(probs, columns=[f"prob_{c}" for c in LABELS])
            pdf.insert(0, "id", df["id"].to_numpy())
            pdf.to_csv(out / f"{split}_probs_{mode}.csv", index=False)
            if split == "val":
                pred = [LABELS[i] for i in probs.argmax(1)]
                res["modes"][mode] = {**evaluate(va["label"], pred, va["domain"]), "n_windows": len(w)}
            else:
                test_probs[mode] = probs
        r = res["modes"][mode]
        print(f"== mode {mode}: val macro-F1 = {r['macro_f1']:.4f} windows={r['n_windows']} "
              f"F1={ {k: round(v, 4) for k, v in r['f1'].items()} } ({time.time() - t0:.0f}s)", flush=True)

    best = max(res["modes"], key=lambda m: res["modes"][m]["macro_f1"])
    res["best_mode"] = best
    rb = res["modes"][best]
    print(f"\nbest mode: {best}  val macro-F1 = {rb['macro_f1']:.4f}")
    print("per-domain macro-F1:", {k: round(v, 4) for k, v in rb["f1_by_domain"].items()})
    print("confusion (rows=true, cols=pred, order human/ai/fusion):", rb["confusion"])

    cm = np.array(rb["confusion"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm / np.maximum(cm.sum(1, keepdims=True), 1), display_labels=LABELS).plot(
        ax=ax, values_format=".2f", colorbar=False)
    ax.set_title(f"{cfg['exp']} val, mode={best} (row-normalized)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)

    sample_path = root / "sample_submission.csv"
    if len(test) == len(pd.read_csv(sample_path)):
        sub = write_submission(test["id"], [LABELS[i] for i in test_probs[best].argmax(1)],
                               sample_path, out / "submission.csv")
        res["test_pred_share"] = sub["label"].value_counts(normalize=True).round(3).to_dict()
        print("submission OK, predicted label share on test:", res["test_pred_share"])
    else:
        print("test subset (mini) -> submission skipped")

    with open(out / "metrics.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
