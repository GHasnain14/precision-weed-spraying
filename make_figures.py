"""Figures for the README: confusion matrix and the herbicide vs missed-weeds trade-off."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spray import herbicide_use


def confusion(r: Path):
    cm = pd.read_csv(r / "test_confusion_matrix.csv", index_col=0)
    norm = cm.div(cm.sum(axis=1), axis=0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm)), cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(r / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def tradeoff(r: Path):
    s = pd.read_csv(r / "test_threshold_sweep.csv")
    rep = json.loads((r / "spray_report.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in (0.05, 0.10, 0.25, 0.48):
        use = herbicide_use(s["tpr"], s["fpr"], d) * 100
        ax.plot((1 - s["tpr"]) * 100, use, label=f"weed density {d:.0%}")
    for key, v in rep.items():
        if key.startswith("recall_target"):
            miss = v["test"]["weeds_missed"] * 100
            ax.axvline(miss, color="grey", lw=0.8, ls=":")
            ax.text(miss, 101, f"target {v['target_recall_on_val']:.0%}", rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel("Weeds missed (%)")
    ax.set_ylabel("Herbicide used vs blanket spraying (%)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("Spot spraying trade-off on the test set")
    fig.tight_layout()
    fig.savefig(r / "spray_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    r = Path(ap.parse_args().results)
    confusion(r)
    tradeoff(r)
    print("figures written to", r)
