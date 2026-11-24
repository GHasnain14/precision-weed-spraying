import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spray import (NEGATIVE, choose_threshold, herbicide_use, max_speed_kmh, rates,  # noqa: E402
                   report, sweep)


def fake_preds(labels, p_neg):
    df = pd.DataFrame({"Filename": [f"{i}.jpg" for i in range(len(labels))], "Label": labels})
    for k in range(9):
        df[f"p{k}"] = 0.0
    df[f"p{NEGATIVE}"] = p_neg
    df["p0"] = 1 - np.asarray(p_neg)
    return df


def test_perfect_classifier_uses_herbicide_only_on_weeds():
    weed = np.array([True] * 30 + [False] * 70)
    score = weed.astype(float)
    tpr, fpr = rates(score, weed, 0.5)
    assert (tpr, fpr) == (1.0, 0.0)
    assert herbicide_use(tpr, fpr, 0.3) == pytest.approx(0.3)


def test_threshold_zero_is_blanket_spraying():
    rng = np.random.default_rng(0)
    score, weed = rng.random(200), rng.random(200) < 0.4
    tpr, fpr = rates(score, weed, 0.0)
    assert herbicide_use(tpr, fpr, 0.1) == pytest.approx(1.0)


def test_rates_fall_as_threshold_rises():
    rng = np.random.default_rng(1)
    score, weed = rng.random(500), rng.random(500) < 0.5
    s = sweep(score, weed, np.linspace(0, 1, 51))
    assert (np.diff(s["tpr"]) <= 1e-12).all()
    assert (np.diff(s["fpr"]) <= 1e-12).all()


@pytest.mark.parametrize("target", [0.8, 0.9, 0.95, 1.0])
def test_chosen_threshold_meets_recall_target(target):
    rng = np.random.default_rng(2)
    weed = rng.random(1000) < 0.5
    score = np.clip(weed * 0.6 + rng.normal(0.2, 0.2, 1000), 0, 1)
    t = choose_threshold(score, weed, target)
    tpr, _ = rates(score, weed, t)
    assert tpr >= target
    # one step stricter must break the target (otherwise t was not the least spraying choice)
    higher = score[score > t]
    if higher.size:
        assert rates(score, weed, higher.min())[0] < target


def test_report_uses_val_threshold_not_test():
    # on val, threshold 0.7 sprays every weed; on test one weed scores 0.4 and must be missed
    val = fake_preds([0, 0, NEGATIVE, NEGATIVE], [0.2, 0.3, 0.9, 0.8])
    test = fake_preds([0, 0, NEGATIVE], [0.2, 0.6, 0.9])
    r = report(val, test, target_recall=1.0)
    assert r["test"]["weeds_missed"] == pytest.approx(0.5)


def test_speed_limit():
    assert max_speed_kmh(0.45, 0.045) == pytest.approx(36.0)
