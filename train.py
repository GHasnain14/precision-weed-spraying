"""Fine-tune a pretrained CNN on DeepWeeds (one official fold)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import NUM_CLASSES, DeepWeeds, split
from models import build, eval_transform, train_transform


def run_epoch(model, loader, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
            loss = loss_fn(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--no-pretrained", action="store_true", help="for smoke tests only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = DeepWeeds(a.root, split(a.root, "train", a.fold), train_transform())
    va = DeepWeeds(a.root, split(a.root, "val", a.fold), eval_transform())
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=len(tr) > a.batch)
    vl = DataLoader(va, a.batch, num_workers=a.workers, pin_memory=True)

    model = build(a.arch, NUM_CLASSES, pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{a.arch}_fold{a.fold}.pt"
    best, history = float("inf"), []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        tl_loss, tl_acc, n = 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                o = model(x)
                loss = loss_fn(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl_loss += loss.item() * len(y)
            tl_acc += (o.argmax(1) == y).sum().item()
            n += len(y)
        v_loss, v_acc = run_epoch(model, vl, device)
        rec = dict(epoch=ep, train_loss=tl_loss / n, train_acc=tl_acc / n, val_loss=v_loss, val_acc=v_acc,
                   seconds=round(time.time() - t0, 1))
        history.append(rec)
        print(json.dumps(rec))
        if v_loss < best:
            best = v_loss
            torch.save({"arch": a.arch, "fold": a.fold, "state_dict": model.state_dict()}, ckpt)
    (out / f"{a.arch}_fold{a.fold}_history.json").write_text(json.dumps(history, indent=1))
    print(f"best val loss {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
