#!/usr/bin/env python3
"""Train the frozen multi-view 12-head configuration on all 867 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_multiview_12head import (
    MultiViewClassifier,
    MultiViewDataset,
    class_weights,
    make_transforms,
    masked_loss,
    read_rows,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--view-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    head_names = list(head_vocabs)
    class_counts = [len(head_vocabs[name]) for name in head_names]
    fold1_train, fold1_validation = read_rows(
        args.cache_manifest,
        args.labels,
        args.fold_assignments,
        1,
        head_names,
    )
    train_rows = sorted(
        fold1_train + fold1_validation, key=lambda row: row["patient_id"]
    )
    if len(train_rows) != 867:
        raise RuntimeError(f"Expected 867 full-training cases, found {len(train_rows)}")
    train_transform, _eval_transform = make_transforms(args.image_size)
    loader = DataLoader(
        MultiViewDataset(train_rows, train_transform, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewClassifier(class_counts, args.view_checkpoint).to(device)
    criteria = [
        nn.CrossEntropyLoss(weight=value.to(device), label_smoothing=0.02)
        for value in class_weights(train_rows, class_counts)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": args.backbone_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("backbone.")
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, view_mask, labels, _patient_ids in loader:
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images, view_mask)
                loss, _observed = masked_loss(logits, labels, criteria)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            batches += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "head_vocabs": head_vocabs,
        "slot_names": ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"],
        "training_cases": len(train_rows),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "seed": args.seed,
        "view_checkpoint": str(args.view_checkpoint) if args.view_checkpoint else None,
    }
    torch.save(checkpoint, args.output_dir / "model_final.pt")
    summary = {
        "training_cases": len(train_rows),
        "epochs": args.epochs,
        "seed": args.seed,
        "history": history,
        "head_names": head_names,
        "class_counts": dict(zip(head_names, class_counts)),
        "checkpoint": str(args.output_dir / "model_final.pt"),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": summary["checkpoint"], "cases": len(train_rows)}))


if __name__ == "__main__":
    main()
