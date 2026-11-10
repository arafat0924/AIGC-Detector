"""Download a public real-vs-AI set and save one shared JPEG format.

The public set mixes camera photos with AI pictures that are often square and
stored differently. Every image kept here is a center square, resized to 256,
and saved as JPEG quality 95. Shape and file type then cannot be the answer.
"""

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

from common import DATA, SEED

SOURCE = "Thilak5434/AI-vs-Real"
SAVE_SIZE = 256
JPEG_QUALITY = 95
# 0 in the source means AI, 1 means a real photograph.
AI_SOURCE_LABEL = 0


def square_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def parquet_paths() -> list[str]:
    """Read the parquet shards directly. The dataset card's row count is stale."""
    cache = DATA.parent / ".cache" / "hf"
    found = sorted(str(path) for path in cache.glob("**/data/*.parquet"))
    if found:
        return found
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(SOURCE, repo_type="dataset", cache_dir=str(cache)))
    found = sorted(str(path) for path in local.glob("**/*.parquet"))
    if not found:
        raise SystemExit("Could not download the image files.")
    return found


def main() -> None:
    os.environ.setdefault("HF_HOME", str(DATA.parent / ".cache" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from datasets import load_dataset

    files = parquet_paths()
    print(f"Reading {len(files)} image files.")
    dataset = load_dataset("parquet", data_files={"train": files}, split="train")
    labels = [int(value) for value in dataset["binary_label"]]
    ai_idx = [i for i, label in enumerate(labels) if label == AI_SOURCE_LABEL]
    real_idx = [i for i, label in enumerate(labels) if label != AI_SOURCE_LABEL]
    print(f"source images: {len(labels)}  ai: {len(ai_idx)}  real: {len(real_idx)}")

    plan = {
        "train": (2000, 2000),
        "val": (400, 400),
        "test": (400, 400),
    }
    # Take a disjoint slice from each class, in train / val / test order.
    chosen = {}
    for name, pool in (("ai", ai_idx), ("real", real_idx)):
        rng_class = np.random.default_rng(SEED + (1 if name == "ai" else 2))
        order = pool.copy()
        rng_class.shuffle(order)
        cursor = 0
        chosen[name] = {}
        for split, (n_ai, n_real) in plan.items():
            need = n_ai if name == "ai" else n_real
            chosen[name][split] = order[cursor : cursor + need]
            cursor += need
            if len(chosen[name][split]) < need:
                raise SystemExit(f"Not enough {name} images for {split}")

    image_root = DATA / "images"
    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True)

    rows = []
    aspects = {"ai": [], "real": []}
    jobs = []
    for split in ("train", "val", "test"):
        for name, label_ai in (("ai", 1), ("real", 0)):
            for source_index in chosen[name][split]:
                jobs.append((split, name, label_ai, int(source_index)))

    jobs.sort(key=lambda job: job[3])
    for n, (split, name, label_ai, source_index) in enumerate(jobs, start=1):
        image = dataset[source_index]["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        width, height = image.size
        aspects[name].append(width / max(height, 1))
        folder = image_root / split / name
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{n:05d}.jpg"
        path = folder / filename
        square_resize(image, SAVE_SIZE).save(path, format="JPEG", quality=JPEG_QUALITY)
        rows.append(
            {
                "filepath": path.relative_to(DATA.parent).as_posix(),
                "split": split,
                "label_ai": label_ai,
                "class_name": name,
                "source_index": source_index,
                "orig_w": width,
                "orig_h": height,
            }
        )
        if n % 250 == 0 or n == len(jobs):
            print(f"saved {n}/{len(jobs)}")

    import csv

    with (DATA / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": SOURCE,
        "seed": SEED,
        "save_size": SAVE_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "label": "label_ai 1 = AI picture, 0 = real photo",
        "counts": {
            split: {
                "ai": sum(1 for r in rows if r["split"] == split and r["label_ai"] == 1),
                "real": sum(1 for r in rows if r["split"] == split and r["label_ai"] == 0),
            }
            for split in ("train", "val", "test")
        },
        "mean_original_aspect_width_over_height": {
            name: float(np.mean(values)) for name, values in aspects.items()
        },
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print("original mean aspect ai/real:", summary["mean_original_aspect_width_over_height"])


if __name__ == "__main__":
    main()
