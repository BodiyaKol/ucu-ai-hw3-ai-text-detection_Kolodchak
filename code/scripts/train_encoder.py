"""M3/M4: fine-tune a multilingual encoder on the fixed grouped split.

Options (config): train_pool + n_train (sample from folds != val_fold), train_mode head | head_tail |
random (one random chunk per text, new draw each epoch), train_labels (specialist on a subset of
classes, e.g. [human, fusion]). Evaluates eval_modes on val, saves val/test probabilities,
submission (3-class models only), metrics.json.
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
from src.data.chunking import build_windows, random_windows, tokenize_no_special
from src.data.io import LABELS, load_train_val, train_pool
from src.eval.metrics import evaluate, macro_f1
from src.models.encoder import aggregate, make_loader, predict_logits, softmax, train
from src.submission import write_submission


def load_model(cfg: dict, labels: list):
    kw = dict(num_labels=len(labels), id2label=dict(enumerate(labels)),
              label2id={c: i for i, c in enumerate(labels)})
    if cfg.get("tiny_random"):  # local smoke test only: 1-layer random model, real tokenizer vocab
        conf = AutoConfig.from_pretrained(cfg["model_name"], num_hidden_layers=1, hidden_size=32,
                                          intermediate_size=64, num_attention_heads=2,
                                          pooler_hidden_size=32, **kw)
        return AutoModelForSequenceClassification.from_config(conf)
    return AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], **kw)


def texts_of(df: pd.DataFrame, prefix_meta: bool) -> list:
    if prefix_meta:
        return ("[" + df["domain"] + "] [" + df["lang"] + "] " + df["text"]).tolist()
    return df["text"].tolist()


def select_train(tr, full, cfg) -> pd.DataFrame:
    if cfg.get("train_pool"):
        tr = train_pool(full, cfg["split"]["val_fold"])
    if cfg.get("train_labels"):
        tr = tr[tr["label"].isin(cfg["train_labels"])]
    n = cfg.get("n_train")
    if n and n < len(tr):
        tr = tr.sample(n=n, random_state=cfg["seed"]) if cfg.get("train_pool") else tr.head(n)
    return tr.reset_index(drop=True)


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
    labels = cfg.get("train_labels") or LABELS
    lab2id = {c: i for i, c in enumerate(labels)}
    root = get_data_root(args.mini)
    split_file = CODE_DIR / ("outputs/split/split_mini.csv.gz" if args.mini else cfg["split"]["file"])
    tr, va, full = load_train_val(root, split_file)
    tr = select_train(tr, full, cfg)
    del full
    test = pd.read_csv(root / "test.csv")
    va = va.head(cfg.get("n_val") or len(va)).reset_index(drop=True)
    test = test.head(cfg.get("n_test") or len(test))
    va_mask = va["label"].isin(labels).to_numpy()      # val rows the model is evaluated on
    out = get_output_dir(cfg["exp"] + ("_mini" if args.mini else ""))
    print(f"device={device} fp16={fp16} model={cfg['model_name']} labels={labels} "
          f"train={len(tr)} {tr['label'].value_counts().to_dict()} val={len(va)} (eval rows {va_mask.sum()}) "
          f"test={len(test)} -> {out}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ids_tr = tokenize_no_special(tok, texts_of(tr, cfg["prefix_meta"]))
    ids_va = tokenize_no_special(tok, texts_of(va, cfg["prefix_meta"]))
    ids_te = tokenize_no_special(tok, texts_of(test, cfg["prefix_meta"]))
    print(f"tokenized in {time.time() - t0:.0f}s", flush=True)

    wkw = dict(head_tokens=cfg["head_tokens"], max_chunks=cfg["max_chunks"], min_tail=cfg["min_tail"])
    L, pad, bs, nw = cfg["max_len"], tok.pad_token_id, cfg["batch_size"], cfg["num_workers"]
    y_text = tr["label"].map(lab2id).to_numpy()
    if cfg["train_mode"] == "random":
        rng = np.random.default_rng(cfg["seed"])

        def loader(ep):
            w = random_windows(ids_tr, tok, L, rng, cfg["max_chunks"], cfg["min_tail"])
            return make_loader(w, y_text, bs, pad, shuffle=True, num_workers=nw)
    else:
        win_tr, own_tr = build_windows(ids_tr, tok, L, cfg["train_mode"], **wkw)
        loader = make_loader(win_tr, y_text[own_tr], bs, pad, shuffle=True, num_workers=nw)

    ids_va_eval = [ids for ids, m in zip(ids_va, va_mask) if m]
    va_eval = va[va_mask].reset_index(drop=True)
    win_va_head, own_va_head = build_windows(ids_va_eval, tok, L, "head", **wkw)

    model = load_model(cfg, labels).float().to(device)  # mDeBERTa checkpoint loads as fp16; AMP needs fp32 weights
    res = {"exp": cfg["exp"], "model": cfg["model_name"], "labels": labels, "n_train": len(tr),
           "train_mode": cfg["train_mode"], "epochs": cfg["epochs"], "epoch_val_macro_f1": []}

    def on_epoch_end(ep):
        logits = aggregate(predict_logits(model, win_va_head, cfg["eval_batch_size"], pad, device, fp16),
                           own_va_head, len(va_eval))
        f1 = macro_f1(va_eval["label"], [labels[i] for i in logits.argmax(1)], labels)
        res["epoch_val_macro_f1"].append(f1)
        print(f"== epoch {ep}: val macro-F1 (head, {len(labels)} classes) = {f1:.4f}", flush=True)

    t0 = time.time()
    train(model, loader, cfg, device, fp16, on_epoch_end)
    res["train_sec"] = round(time.time() - t0)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")

    # evaluate long-text strategies; probabilities are saved for ALL val/test rows (for ensembling)
    res["modes"], test_probs = {}, {}
    for mode in cfg["eval_modes"]:
        t0 = time.time()
        for split, df, ids in (("val", va, ids_va), ("test", test, ids_te)):
            w, own = build_windows(ids, tok, L, mode, **wkw)
            probs = softmax(aggregate(predict_logits(model, w, cfg["eval_batch_size"], pad, device, fp16),
                                      own, len(df)))
            pdf = pd.DataFrame(probs, columns=[f"prob_{c}" for c in labels])
            pdf.insert(0, "id", df["id"].to_numpy())
            pdf.to_csv(out / f"{split}_probs_{mode}.csv", index=False)
            if split == "val":
                pred = np.array([labels[i] for i in probs.argmax(1)])
                res["modes"][mode] = {**evaluate(va["label"][va_mask], pred[va_mask],
                                                 va["domain"][va_mask], labels), "n_windows": len(w)}
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
    print(f"confusion (rows=true, cols=pred, order {labels}):", rb["confusion"])

    cm = np.array(rb["confusion"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm / np.maximum(cm.sum(1, keepdims=True), 1), display_labels=labels).plot(
        ax=ax, values_format=".2f", colorbar=False)
    ax.set_title(f"{cfg['exp']} val, mode={best} (row-normalized)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)

    sample_path = root / "sample_submission.csv"
    if labels == LABELS and len(test) == len(pd.read_csv(sample_path)):
        sub = write_submission(test["id"], [LABELS[i] for i in test_probs[best].argmax(1)],
                               sample_path, out / "submission.csv")
        res["test_pred_share"] = sub["label"].value_counts(normalize=True).round(3).to_dict()
        print("submission OK, predicted label share on test:", res["test_pred_share"])
    else:
        print("specialist or test subset -> submission skipped (used in the ensemble)")

    with open(out / "metrics.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
