"""Minimal fine-tuning / inference loop for a sequence-classification transformer."""
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader


def collate(batch, pad_id: int):
    """Dynamic padding to the longest window in the batch."""
    max_len = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return ids, mask, labels


def make_loader(windows, labels, batch_size: int, pad_id: int, shuffle: bool, num_workers: int = 0):
    data = list(zip(windows, labels))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=partial(collate, pad_id=pad_id))


def train(model, loader, cfg: dict, device: str, fp16: bool, on_epoch_end=None):
    """AdamW + linear warmup/decay, fp16 autocast, gradient clipping."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(loader)
    warm = max(1, int(cfg["warmup_ratio"] * total))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total - step) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    step, t0 = 0, time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        run_loss = 0.0
        for ids, mask, y in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                             labels=y.to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % cfg["log_every"] == 0:
                print(f"  ep {ep} step {step}/{total} loss {run_loss / cfg['log_every']:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
        if on_epoch_end is not None:
            on_epoch_end(ep)


@torch.no_grad()
def predict_logits(model, windows, batch_size: int, pad_id: int, device: str, fp16: bool) -> np.ndarray:
    """Logits per window; windows sorted by length for fast padded batches."""
    model.eval()
    order = np.argsort([len(w) for w in windows])
    out = np.zeros((len(windows), model.config.num_labels), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, _ = collate([(windows[i], 0) for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def aggregate(logits: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    """Mean of window logits per text."""
    agg = np.zeros((n, logits.shape[1]), dtype=np.float32)
    np.add.at(agg, owner, logits)
    cnt = np.bincount(owner, minlength=n)[:, None]
    return agg / np.maximum(cnt, 1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
