"""Shared pieces for the trained real-vs-AI detector."""

import csv
import io
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MANIFEST = DATA / "manifest.csv"

SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def read_manifest(split: str | None = None) -> list[dict]:
    rows = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split is None or row["split"] == split:
                row["label_ai"] = int(row["label_ai"])
                rows.append(row)
    return rows


def recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JpegAugment:
    """Save and reopen as JPEG so training sees compression, not only clean pixels."""

    def __init__(self, low: int = 50, high: int = 95):
        self.low = low
        self.high = high

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.low, self.high)
        return recompress(img, quality)


def _normalize():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def train_transform(run: str):
    if run == "jpeg":
        ops = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            JpegAugment(50, 95),
            transforms.ToTensor(),
            _normalize(),
        ]
    elif run == "clean":
        ops = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(),
        ]
    else:
        raise ValueError(f"unknown run: {run}")
    return transforms.Compose(ops)


def eval_transform(jpeg_quality: int | None = None):
    ops = []
    if jpeg_quality is not None:
        ops.append(JpegAugmentFixed(jpeg_quality))
    ops.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            _normalize(),
        ]
    )
    return transforms.Compose(ops)


class JpegAugmentFixed:
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        return recompress(img, self.quality)


class ImageRows(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img = Image.open(ROOT / row["filepath"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row["label_ai"], dtype=torch.float32)
        return x, y


def build_model() -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def best_threshold(labels, scores) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    best_t = 0.5
    best_acc = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        acc = float(((scores >= threshold).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(threshold)
    return best_t


def binary_metrics(labels, scores, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    total = max(len(labels), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": int(len(labels)),
        "auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else None,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }
