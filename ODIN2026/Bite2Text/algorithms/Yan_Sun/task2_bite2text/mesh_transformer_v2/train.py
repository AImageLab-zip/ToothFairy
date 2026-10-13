#!/usr/bin/env python3
"""Train the contact-aware surface-sampled point token Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import IGNORE_INDEX, SurfacePairDataset, read_manifest, set_seed
from model import ContactPointTransformer


def class_weights(records: list[dict[str, Any]], head_vocabs: dict[str, list[str]]) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for head, vocab in head_vocabs.items():
        counts = np.zeros(len(vocab), dtype=np.float32)
        for record in records:
            target = int(record["targets"][head])
            if target >= 0:
                counts[target] += 1
        weights = np.ones_like(counts)
        observed = counts > 0
        weights[observed] = 1.0 / np.sqrt(counts[observed])
        if observed.any():
            weights[observed] /= weights[observed].mean()
        output[head] = torch.tensor(weights, dtype=torch.float32)
    return output


def metrics_from_counts(counts: dict[str, np.ndarray]) -> dict[str, Any]:
    per_head: dict[str, Any] = {}
    macro_scores: list[float] = []
    for head, matrix in counts.items():
        support = matrix.sum(axis=1)
        total = int(support.sum())
        accuracy = float(matrix.trace() / total) if total else None
        scores = []
        for cls, cls_support in enumerate(support):
            if cls_support == 0:
                continue
            tp = float(matrix[cls, cls])
            fp = float(matrix[:, cls].sum() - tp)
            fn = float(matrix[cls, :].sum() - tp)
            scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        macro_f1 = float(np.mean(scores)) if scores else None
        if macro_f1 is not None:
            macro_scores.append(macro_f1)
        per_head[head] = {"samples": total, "accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix": matrix.tolist()}
    return {"per_head": per_head, "mean_macro_f1": float(np.mean(macro_scores)) if macro_scores else None}


def run_epoch(
    model: ContactPointTransformer,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    head_vocabs: dict[str, list[str]],
    weights: dict[str, Tensor],
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    matrices = {head: np.zeros((len(vocab), len(vocab)), dtype=np.int64) for head, vocab in head_vocabs.items()}
    total_loss, steps = 0.0, 0
    for batch in loader:
        inputs = {key: batch[key].to(device, non_blocking=True) for key in ("upper_xyz", "upper_normals", "lower_xyz", "lower_normals")}
        targets = {head: values.to(device, non_blocking=True) for head, values in batch["targets"].items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs)
            losses = [
                F.cross_entropy(prediction, targets[head], weight=weights[head].to(device), ignore_index=IGNORE_INDEX)
                for head, prediction in logits.items() if (targets[head] != IGNORE_INDEX).any()
            ]
            if not losses:
                continue
            loss = torch.stack(losses).mean()
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach().cpu())
        steps += 1
        for head, prediction in logits.items():
            valid = targets[head] != IGNORE_INDEX
            if valid.any():
                truth = targets[head][valid].detach().cpu().numpy()
                guessed = prediction[valid].argmax(dim=1).detach().cpu().numpy()
                np.add.at(matrices[head], (truth, guessed), 1)
    result = metrics_from_counts(matrices)
    result.update({"loss": total_loss / max(steps, 1), "steps": steps})
    return result


def save_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--tokens-per-jaw", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_manifest(args.manifest)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    splits = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
    if args.max_samples:
        splits = {split: rows[:args.max_samples] for split, rows in splits.items()}
    if not splits["train"] or not splits["val"]:
        raise SystemExit("Manifest requires non-empty train and val splits")
    datasets = {
        "train": SurfacePairDataset(splits["train"], args.data_root, args.num_points, args.seed, augment=True),
        "val": SurfacePairDataset(splits["val"], args.data_root, args.num_points, args.seed + 100000, augment=False),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }
    model = ContactPointTransformer(head_vocabs, tokens_per_jaw=args.tokens_per_jaw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    weights = class_weights(splits["train"], head_vocabs)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"head_vocabs": head_vocabs, "device": str(device), "split_sizes": {name: len(rows) for name, rows in splits.items()}})
    save_json(args.output_dir / "config.json", config)
    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, device, head_vocabs, weights)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], None, scaler, device, head_vocabs, weights)
        scheduler.step()
        score = val_metrics["mean_macro_f1"] if val_metrics["mean_macro_f1"] is not None else -float("inf")
        result = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(result)
        print(json.dumps({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "val_mean_macro_f1": score}), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model": model.state_dict(), "head_vocabs": head_vocabs, "config": config, "val": val_metrics}, args.output_dir / "best.pt")
    save_json(args.output_dir / "history.json", history)
    save_json(args.output_dir / "summary.json", {"best_val_mean_macro_f1": best_score, "epochs": args.epochs, "device": str(device)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
