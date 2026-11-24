"""Measure single-image inference time, the number that limits sprayer speed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data import NUM_CLASSES
from models import IMG_SIZE, build


@torch.no_grad()
def ms_per_image(model, device, runs=100, warmup=10, threads=None) -> float:
    if threads and device.type == "cpu":
        torch.set_num_threads(threads)
    model.eval().to(device)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu-threads", type=int, default=2, help="2 threads ~ a small embedded board")
    a = ap.parse_args()
    model = build(a.arch, NUM_CLASSES, pretrained=False)  # weights do not change timing
    res = {"cpu": {"ms_per_image": round(ms_per_image(model, torch.device("cpu"), threads=a.cpu_threads), 2),
                   "threads": a.cpu_threads}}
    if torch.cuda.is_available():
        res["gpu"] = {"ms_per_image": round(ms_per_image(model, torch.device("cuda")), 2),
                      "name": torch.cuda.get_device_name(0)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
