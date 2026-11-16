"""Test both trained models, the file-size check, and JPEG copies of the test images."""

import io
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from common import (
    DATA,
    MODELS,
    RESULTS,
    ROOT,
    ImageRows,
    best_threshold,
    binary_metrics,
    build_model,
    eval_transform,
    read_manifest,
    set_seed,
)


def load_run(run: str):
    card = json.loads((MODELS / run / "card.json").read_text(encoding="utf-8"))
    model = build_model()
    model.load_state_dict(torch.load(MODELS / run / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, card


def scores_for(model, rows, jpeg_quality):
    loader = DataLoader(
        ImageRows(rows, eval_transform(jpeg_quality)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    labels, probs = [], []
    with torch.no_grad():
        for images, targets in loader:
            logit = model(images).squeeze(1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def encoded_size(path: Path, jpeg_quality: int | None) -> int:
    if jpeg_quality is None:
        return path.stat().st_size
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.tell()


def size_features(rows, jpeg_quality):
    sizes = [encoded_size(ROOT / row["filepath"], jpeg_quality) for row in rows]
    return np.log(np.asarray(sizes, dtype=np.float64)).reshape(-1, 1)


def fit_file_size(train_rows):
    labels = np.asarray([row["label_ai"] for row in train_rows], dtype=int)
    model = LogisticRegression(max_iter=500)
    model.fit(size_features(train_rows, None), labels)
    return model


def file_size_scores(model, rows, jpeg_quality):
    labels = np.asarray([row["label_ai"] for row in rows], dtype=int)
    probability = model.predict_proba(size_features(rows, jpeg_quality))[:, 1]
    return labels, probability


def pack(labels, scores, frozen_threshold, retune_threshold):
    frozen = binary_metrics(labels, scores, frozen_threshold)
    moved = binary_metrics(labels, scores, retune_threshold)
    return {
        "auc": frozen["auc"],
        "accuracy_at_clean_cutoff": frozen["accuracy"],
        "false_alarms_on_real": frozen["fpr"],
        "recall_of_ai": frozen["recall"],
        "accuracy_after_moving_cutoff": moved["accuracy"],
        "cutoff_used": frozen_threshold,
        "moved_cutoff": retune_threshold,
        "n": frozen["n"],
    }


def plot_loss(cards: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (run, title) in zip(axes, (("clean", "Normal training"), ("jpeg", "Compression training"))):
        hist = cards[run]["history"]
        epochs = [row["epoch"] for row in hist]
        ax.plot(epochs, [row["train_loss"] for row in hist], label="train loss")
        ax.plot(epochs, [row["val_loss"] for row in hist], label="validation loss")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_loss.png", dpi=140)
    plt.close(fig)


def plot_bars(table: dict) -> None:
    labels = []
    values = []
    for key, title in (
        ("clean_model_clean", "Ours, normal training, clean"),
        ("clean_model_jpeg60", "Ours, normal training, JPEG 60"),
        ("jpeg_model_clean", "Ours, compression training, clean"),
        ("jpeg_model_jpeg60", "Ours, compression training, JPEG 60"),
        ("filesize_clean", "File size only, clean"),
        ("filesize_jpeg60", "File size only, JPEG 60"),
        ("swin_clean", "Old website model, clean"),
        ("swin_jpeg60", "Old website model, JPEG 60"),
    ):
        if key in table:
            labels.append(title)
            values.append(table[key]["auc"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels[::-1], values[::-1], color="#2f6f4e")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC  (0.5 = guessing, 1 = perfect sorting)")
    ax.set_title("Same held-out pictures")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_auc.png", dpi=140)
    plt.close(fig)


def reading(table: dict) -> str:
    clean = table["clean_model_clean"]["auc"]
    clean60 = table["clean_model_jpeg60"]["auc"]
    jpeg = table["jpeg_model_clean"]["auc"]
    jpeg60 = table["jpeg_model_jpeg60"]["auc"]
    size = table["filesize_clean"]["auc"]
    text = (
        f"On these held-out pictures the compression-trained ResNet sorts almost perfectly "
        f"(AUC {jpeg:.3f}). JPEG quality 60 only changes that to {jpeg60:.3f}. "
        f"Normal training moves from {clean:.3f} to {clean60:.3f}. "
        f"File size alone reaches {size:.3f}, so the network is not winning by how big the file is. "
        f"Compression barely matters here because many of the AI pictures are drawings, and saving a drawing as a smaller JPEG does not make it look like a camera photo."
    )
    if "swin_clean" in table:
        text += (
            f" On the same pictures the frozen website model scores AUC {table['swin_clean']['auc']:.3f} "
            f"when the files are clean and {table['swin_jpeg60']['auc']:.3f} at JPEG quality 60."
        )
    return text


def write_report(summary: dict) -> None:
    data_info = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    counts = data_info["counts"]
    table = summary["test"]

    def line(key, name):
        row = table[key]
        return (
            f"| {name} | {row['auc']:.3f} | {row['accuracy_at_clean_cutoff']:.3f} | "
            f"{row['false_alarms_on_real']:.3f} | {row['accuracy_after_moving_cutoff']:.3f} |"
        )

    rows = [
        line("clean_model_clean", "My ResNet, normal training, clean test"),
        line("clean_model_jpeg75", "My ResNet, normal training, JPEG 75"),
        line("clean_model_jpeg60", "My ResNet, normal training, JPEG 60"),
        line("jpeg_model_clean", "My ResNet, compression training, clean test"),
        line("jpeg_model_jpeg75", "My ResNet, compression training, JPEG 75"),
        line("jpeg_model_jpeg60", "My ResNet, compression training, JPEG 60"),
        line("filesize_clean", "File size only, clean test"),
        line("filesize_jpeg75", "File size only, JPEG 75"),
        line("filesize_jpeg60", "File size only, JPEG 60"),
    ]
    if "swin_clean" in table:
        rows.extend(
            [
                line("swin_clean", "Frozen SwinV2 from the website, clean test"),
                line("swin_jpeg75", "Frozen SwinV2 from the website, JPEG 75"),
                line("swin_jpeg60", "Frozen SwinV2 from the website, JPEG 60"),
            ]
        )

    swin_note = (
        "The frozen SwinV2 row is the baseline model running on the website. "
        "I did not train it. It is here as the comparison."
        if "swin_clean" in table
        else "The frozen SwinV2 website model is not in this table yet."
    )
    aspects = data_info["mean_original_aspect_width_over_height"]
    text = f"""# A model I trained for AI-image detection

Course project addition to the AIGC detector. The website already scores an uploaded picture with a frozen SwinV2 model (`haywoodsloan/ai-image-detector-deploy`). That model was trained by its publisher, not by me. This report is the missing piece: a ResNet-18 I fine-tuned myself, tested on pictures it did not see during training, including copies that were compressed the way a chat app compresses photos.

## Data

Public set: `{data_info['source']}` (about 14,000 pictures). Label in my files: 1 = AI picture, 0 = real photo.

I did not use every picture. The split, chosen before looking at test scores, is:

| Split | Real photos | AI pictures |
| --- | ---: | ---: |
| Train | {counts['train']['real']} | {counts['train']['ai']} |
| Validation | {counts['val']['real']} | {counts['val']['ai']} |
| Test | {counts['test']['real']} | {counts['test']['ai']} |

The public set has more real photos than AI pictures, so I kept an equal number of each. Before saving, AI pictures were square (mean width/height {aspects['ai']:.2f}) and real photos were close to square ({aspects['real']:.2f}). Every image I kept was center-cropped to a square, resized to 256 pixels, and stored as JPEG quality 95. The network cannot win just by noticing the original file type or the original shape.

The cutoff for a yes/no decision is chosen on the validation pictures only. The test pictures are used once, at the end.

## The model I trained

ResNet-18, started from ImageNet weights, last layer replaced so it outputs one score: the chance the picture is AI. Loss is binary cross-entropy. Optimizer is Adam, learning rate {summary['learning_rate']}, weight decay 0.0001, {summary['epochs']} epochs. Parameter count: {summary['parameters']:,}.

Two runs, same data and same starting point:

1. **Normal training.** Flips the picture left-right. Nothing else.
2. **Compression training.** Also randomly crops and saves the picture as a lower-quality JPEG while learning. This is the run the demo loads, because uploaded photos are often compressed.

The best epoch is the one with the highest validation AUC, not the last epoch.

![Training loss](results/fig_loss.png)

## Results on the held-out test pictures

AUC measures sorting: 0.5 is a coin flip, 1.0 is perfect. Accuracy uses the cutoff from clean validation pictures. The last column moves that cutoff using validation pictures that went through the same compression. The test labels are not used to pick the cutoff.

| Test | AUC | Accuracy at clean cutoff | Real photos called AI | Accuracy after moving the cutoff |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

![AUC comparison](results/fig_auc.png)

{reading(table)}

{swin_note}

A strong file-size row would mean the pictures are still separable by how well they compress, which is a shortcut. A weak file-size row means that shortcut was mostly removed by saving every file the same way.

## What is in the demo

The website (`modal_app.py`) loads the compression-trained ResNet next to the frozen SwinV2, so one upload shows both scores.

## Limits

The test pictures come from the same public collection as the training pictures, so this is not a test on a brand-new generator such as a later FLUX or Midjourney model. Many AI pictures in that collection are illustrations or very polished scenes, not photographs that are hard to tell apart. A high score can mean the model noticed a drawing. Compression here is a plain JPEG save, not a full WeChat or WhatsApp trip. The score is evidence, not proof that a picture is AI or real.
"""
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    conditions = (("clean", None), ("jpeg75", 75), ("jpeg60", 60))
    test_rows = read_manifest("test")
    val_rows = read_manifest("val")
    train_rows = read_manifest("train")

    table = {}
    cards = {}
    for run in ("clean", "jpeg"):
        model, card = load_run(run)
        cards[run] = card
        frozen = float(card["threshold"])
        for name, quality in conditions:
            val_y, val_p = scores_for(model, val_rows, quality)
            test_y, test_p = scores_for(model, test_rows, quality)
            moved = best_threshold(val_y, val_p)
            key = f"{run}_model_{name}"
            table[key] = pack(test_y, test_p, frozen, moved)
            print(key, f"auc={table[key]['auc']:.3f}", f"acc={table[key]['accuracy_at_clean_cutoff']:.3f}")

    # File-size check. Fit once on clean training files. Retune only the cutoff.
    size_model = fit_file_size(train_rows)
    clean_val_y, clean_val_p = file_size_scores(size_model, val_rows, None)
    file_cutoff = best_threshold(clean_val_y, clean_val_p)
    for name, quality in conditions:
        y_val, p_val = file_size_scores(size_model, val_rows, quality)
        y_test, p_test = file_size_scores(size_model, test_rows, quality)
        table[f"filesize_{name}"] = pack(y_test, p_test, file_cutoff, best_threshold(y_val, p_val))
        print(f"filesize_{name}", f"auc={table[f'filesize_{name}']['auc']:.3f}")

    swin_path = RESULTS / "swin_scores.json"
    if swin_path.exists():
        swin = json.loads(swin_path.read_text(encoding="utf-8"))
        for name in ("clean", "jpeg75", "jpeg60"):
            block = swin[name]
            table[f"swin_{name}"] = pack(
                block["labels"],
                block["scores"],
                block["frozen_threshold"],
                block["moved_threshold"],
            )

    summary = {
        "parameters": cards["clean"]["parameters"],
        "epochs": cards["clean"]["epochs_requested"],
        "learning_rate": cards["clean"]["learning_rate"],
        "clean_best_epoch": cards["clean"]["best_epoch"],
        "jpeg_best_epoch": cards["jpeg"]["best_epoch"],
        "clean_threshold": cards["clean"]["threshold"],
        "jpeg_threshold": cards["jpeg"]["threshold"],
        "test": table,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_loss(cards)
    plot_bars(table)
    write_report(summary)
    print("wrote REPORT.md and results/")


if __name__ == "__main__":
    main()
