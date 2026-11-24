"""DeepWeeds data: download, official splits and a PyTorch Dataset.

DeepWeeds (Olsen et al., Scientific Reports 2019): 17,509 field images,
8 weed species (labels 0-7) and a "Negative" class (label 8) with no target weed.
Images and labels are CC BY 4.0.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

LABELS_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv"
IMAGES_DRIVE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"  # images.zip, 468 MB, from the DeepWeeds README

CLASSES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
    "Rubber vine", "Siam weed", "Snake weed", "Negative",
]
NEGATIVE = 8
NUM_CLASSES = len(CLASSES)


def download(root: Path) -> None:
    """Download the label files and images into root/labels and root/images."""
    root = Path(root)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    names = ["labels"] + [f"{s}_subset{i}" for s in ("train", "val", "test") for i in range(5)]
    for name in names:
        dst = root / "labels" / f"{name}.csv"
        if not dst.exists():
            pd.read_csv(LABELS_URL.format(name=name)).to_csv(dst, index=False)

    img_dir = root / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) >= 17509:
        return
    import gdown  # only needed for the download step

    zip_path = root / "images.zip"
    if not zip_path.exists():
        gdown.download(id=IMAGES_DRIVE_ID, output=str(zip_path), quiet=False)
    img_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(img_dir)
    # some archives contain a nested folder; flatten it
    for p in img_dir.rglob("*.jpg"):
        if p.parent != img_dir:
            p.rename(img_dir / p.name)


def split(root: Path, part: str, fold: int = 0) -> pd.DataFrame:
    """Official DeepWeeds split (part = train / val / test, fold 0-4)."""
    return pd.read_csv(Path(root) / "labels" / f"{part}_subset{fold}.csv")


class DeepWeeds(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, transform=None):
        self.img_dir = Path(root) / "images"
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download DeepWeeds")
    ap.add_argument("--root", default="data")
    download(Path(ap.parse_args().root))
    print("done")
