"""Score the same test and validation pictures with the frozen website model."""

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import RESULTS, ROOT, best_threshold, read_manifest, recompress

REPO = "haywoodsloan/ai-image-detector-deploy"


def load():
    processor = AutoImageProcessor.from_pretrained(REPO)
    model = AutoModelForImageClassification.from_pretrained(REPO)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake_idx = next(
        (
            i
            for i, label in id2label.items()
            if label.startswith(("artificial", "ai", "fake", "generated"))
        ),
        0,
    )
    return processor, model, fake_idx


def score_rows(processor, model, fake_idx, rows, jpeg_quality, batch_size=8):
    labels, scores = [], []
    batch, batch_y = [], []

    def flush():
        if not batch:
            return
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[:, fake_idx].cpu().numpy()
        scores.extend(float(p) for p in prob)
        labels.extend(batch_y)
        batch.clear()
        batch_y.clear()

    for i, row in enumerate(rows, start=1):
        img = Image.open(ROOT / row["filepath"]).convert("RGB")
        if jpeg_quality is not None:
            img = recompress(img, jpeg_quality)
        batch.append(img)
        batch_y.append(int(row["label_ai"]))
        if len(batch) == batch_size:
            flush()
            print(f"q={jpeg_quality} {i}/{len(rows)}", flush=True)
    flush()
    return labels, scores


def main():
    processor, model, fake_idx = load()
    print("fake class index", fake_idx, flush=True)
    out = {}
    val_rows = read_manifest("val")
    test_rows = read_manifest("test")
    for name, quality in (("clean", None), ("jpeg75", 75), ("jpeg60", 60)):
        val_y, val_p = score_rows(processor, model, fake_idx, val_rows, quality)
        test_y, test_p = score_rows(processor, model, fake_idx, test_rows, quality)
        frozen = best_threshold(val_y, val_p) if name == "clean" else out["clean"]["frozen_threshold"]
        # The website ships a high bar. For the table we also keep the validation cutoff
        # of this same model, so the comparison uses one rule.
        if name == "clean":
            frozen = best_threshold(val_y, val_p)
        out[name] = {
            "labels": test_y,
            "scores": test_p,
            "frozen_threshold": frozen,
            "moved_threshold": best_threshold(val_y, val_p),
        }
        print(name, "done", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "swin_scores.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote results/swin_scores.json")


if __name__ == "__main__":
    main()
