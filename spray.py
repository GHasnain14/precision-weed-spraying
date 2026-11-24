"""Spray decision layer: turn classifier output into spray / no-spray commands.

Model of the sprayer
--------------------
A boom carries nozzles, each with a camera looking at the ground patch in front
of it. Each DeepWeeds image is treated as one such patch. The nozzle opens if
the model's weed probability for that patch is at or above a threshold t:

    p_weed = 1 - P(Negative)        spray  <=>  p_weed >= t

Two numbers matter to a farmer:
  * missed weeds  = weed patches not sprayed / all weed patches  (1 - recall)
  * herbicide use = patches sprayed / all patches  (blanket spraying = 100 %)

Herbicide use depends on how weedy the field is. With weed density d (share of
patches that contain a weed), true positive rate TPR and false positive rate FPR:

    herbicide use = d * TPR + (1 - d) * FPR

DeepWeeds is about 48 % weed images, far weedier than most crops, so results are
reported at the dataset density and at lower, more typical densities.

The threshold is chosen on the validation set (the smallest herbicide use that
still sprays at least `target_recall` of weeds) and then applied unchanged to
the test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NEGATIVE = 8
WEED_NAMES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
              "Rubber vine", "Siam weed", "Snake weed"]


def p_weed(df: pd.DataFrame) -> np.ndarray:
    return 1.0 - df[f"p{NEGATIVE}"].to_numpy()


def is_weed(df: pd.DataFrame) -> np.ndarray:
    return df["Label"].to_numpy() != NEGATIVE


def rates(score: np.ndarray, weed: np.ndarray, t: float) -> tuple[float, float]:
    """True and false positive rates of spraying at threshold t."""
    spray = score >= t
    tpr = spray[weed].mean() if weed.any() else 0.0
    fpr = spray[~weed].mean() if (~weed).any() else 0.0
    return float(tpr), float(fpr)


def herbicide_use(tpr: float, fpr: float, density: float) -> float:
    return density * tpr + (1.0 - density) * fpr


def sweep(score, weed, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], score]))
    rows = [(t, *rates(score, weed, t)) for t in thresholds]
    return pd.DataFrame(rows, columns=["threshold", "tpr", "fpr"])


def choose_threshold(score, weed, target_recall: float) -> float:
    """Highest threshold (least spraying) that still sprays >= target_recall of weeds."""
    s = sweep(score, weed)
    ok = s[s["tpr"] >= target_recall]
    return float(ok["threshold"].max())


def max_speed_kmh(footprint_m: float, latency_s: float) -> float:
    """Fastest travel speed at which one camera/nozzle pair can still decide on
    every patch: the vehicle must not cover more than one footprint per decision."""
    return footprint_m / latency_s * 3.6


def report(val: pd.DataFrame, test: pd.DataFrame, target_recall=0.95, densities=(0.05, 0.10, 0.25)) -> dict:
    t = choose_threshold(p_weed(val), is_weed(val), target_recall)
    sw, ww = p_weed(test), is_weed(test)
    tpr, fpr = rates(sw, ww, t)
    d_data = float(ww.mean())
    species_miss = {}
    lab = test["Label"].to_numpy()
    for k, name in enumerate(WEED_NAMES):
        m = lab == k
        if m.any():
            species_miss[name] = round(float((sw[m] < t).mean()), 4)
    out = {
        "target_recall_on_val": target_recall,
        "threshold": round(t, 4),
        "test": {
            "weeds_sprayed (recall)": round(tpr, 4),
            "weeds_missed": round(1 - tpr, 4),
            "clean_patches_sprayed (FPR)": round(fpr, 4),
            "missed_by_species": species_miss,
            "herbicide_use_vs_blanket": {
                f"dataset density {d_data:.2f}": round(herbicide_use(tpr, fpr, d_data), 4),
                **{f"density {d:.2f}": round(herbicide_use(tpr, fpr, d), 4) for d in densities},
            },
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--targets", default="0.90,0.95,0.98")
    ap.add_argument("--footprint", type=float, default=0.45,
                    help="ground length one image covers along the travel direction, metres (set to your camera)")
    a = ap.parse_args()
    r = Path(a.results)
    val, test = pd.read_csv(r / "val_predictions.csv"), pd.read_csv(r / "test_predictions.csv")
    out = {f"recall_target_{x}": report(val, test, float(x)) for x in a.targets.split(",")}

    lat_file = r / "latency.json"
    if lat_file.exists():
        lat = json.loads(lat_file.read_text())
        out["speed_limit_kmh"] = {
            dev: round(max_speed_kmh(a.footprint, v["ms_per_image"] / 1000), 1) for dev, v in lat.items()
        }
        out["footprint_m_assumed"] = a.footprint
    (r / "spray_report.json").write_text(json.dumps(out, indent=1))
    sweep(p_weed(test), is_weed(test), np.linspace(0, 1, 201)).to_csv(r / "test_threshold_sweep.csv", index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
