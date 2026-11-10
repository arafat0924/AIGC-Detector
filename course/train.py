"""Fine-tune ResNet-18 to tell real photos from AI pictures.

--run clean   learn from the saved pictures only
--run jpeg    same model, but training also compresses and crops the pictures
"""

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    MODELS,
    RESULTS,
    ImageRows,
    best_threshold,
    binary_metrics,
    build_model,
    eval_transform,
    parameter_count,
    read_manifest,
    set_seed,
    train_transform,
)


def predict(model, loader):
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for images, targets in loader:
            logit = model(images).squeeze(1)
            logits.append(logit.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("clean", "jpeg"), required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed()
    train_rows = read_manifest("train")
    val_rows = read_manifest("val")
    train_loader = DataLoader(
        ImageRows(train_rows, train_transform(args.run)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ImageRows(val_rows, eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = MODELS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    history = []
    best_auc = -1.0
    best_epoch = 0
    n_params = parameter_count(model)
    print(f"run={args.run}  images={len(train_rows)} train / {len(val_rows)} val  params={n_params}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logit = model(images).squeeze(1)
            loss = criterion(logit, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            seen += len(targets)
        train_loss = loss_sum / max(seen, 1)

        val_labels, val_logits = predict(model, val_loader)
        val_loss = criterion(torch.tensor(val_logits), torch.tensor(val_labels)).item()
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))
        val_stats = binary_metrics(val_labels, val_scores, 0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "val_auc": val_stats["auc"],
                "val_accuracy_at_0.5": val_stats["accuracy"],
            }
        )
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_stats['auc']:.4f}"
        )
        if val_stats["auc"] > best_auc:
            best_auc = val_stats["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True))
    val_labels, val_logits = predict(model, val_loader)
    threshold = best_threshold(val_labels, 1.0 / (1.0 + np.exp(-val_logits)))
    card = {
        "run": args.run,
        "architecture": "ResNet-18, ImageNet start, final layer replaced with one score",
        "parameters": n_params,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "loss": "binary cross-entropy",
        "optimizer": "Adam",
        "best_val_auc": best_auc,
        "threshold": threshold,
        "threshold_rule": "cutoff chosen on clean validation images to maximize accuracy",
        "history": history,
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'best.pt'}  best_epoch={best_epoch}  threshold={threshold:.2f}")


if __name__ == "__main__":
    main()
