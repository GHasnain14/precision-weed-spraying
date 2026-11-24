"""Predict on the validation and test sets and save class probabilities.

Validation predictions are used to choose the spray threshold; test predictions
are only used to report the result. This keeps the test set honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import CLASSES, NUM_CLASSES, DeepWeeds, split
from models import build, eval_transform


@torch.no_grad()
def predict(model, root, df, device, batch=128, workers=2) -> pd.DataFrame:
    loader = DataLoader(DeepWeeds(root, df, eval_transform()), batch, num_workers=workers)
    probs = []
    model.eval()
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).float().cpu().numpy())
    p = np.concatenate(probs)
    out = df[["Filename", "Label"]].copy()
    out["Pred"] = p.argmax(1)
    for k in range(NUM_CLASSES):
        out[f"p{k}"] = p[:, k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/mobilenet_v3_large_fold0.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    model = build(ck["arch"], NUM_CLASSES, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"arch": ck["arch"], "fold": ck["fold"]}
    for part in ("val", "test"):
        pred = predict(model, a.root, split(a.root, part, ck["fold"]), device, workers=a.workers)
        pred.to_csv(out / f"{part}_predictions.csv", index=False)
        y, yh = pred["Label"].values, pred["Pred"].values
        metrics[part] = {
            "n": int(len(y)),
            "accuracy": round(float(accuracy_score(y, yh)), 4),
            "macro_f1": round(float(f1_score(y, yh, average="macro")), 4),
            "per_class_recall": {
                CLASSES[k]: round(float((yh[y == k] == k).mean()), 4) for k in range(NUM_CLASSES) if (y == k).any()
            },
        }
        if part == "test":
            cm = confusion_matrix(y, yh, labels=list(range(NUM_CLASSES)))
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "test_confusion_matrix.csv")
    (out / "classification_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in ("val", "test")}, indent=1))


if __name__ == "__main__":
    main()
