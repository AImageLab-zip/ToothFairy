#!/usr/bin/env python3
"""Train one fold of a multi-view intraoral-photo 12-head classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SLOT_NAMES = ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(
    cache_manifest: Path,
    labels_path: Path,
    folds_path: Path,
    fold: int,
    head_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    photos_by_patient: dict[str, dict[str, str]] = {}
    with cache_manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            photos_by_patient.setdefault(row["patient_id"], {})[row["view"]] = row[
                "cached_path"
            ]
    labels_by_patient: dict[str, list[int]] = {}
    with labels_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels_by_patient[row["patient_id"]] = [
                int(row[f"label_{index}"]) for index in range(len(head_names))
            ]
    patient_fold: dict[str, int] = {}
    with folds_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_fold[row["patient_id"]] = int(row["fold"])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for patient_id in sorted(patient_fold):
        if patient_id not in labels_by_patient or patient_id not in photos_by_patient:
            continue
        item = {
            "patient_id": patient_id,
            "paths": [photos_by_patient[patient_id].get(slot, "") for slot in SLOT_NAMES],
            "labels": labels_by_patient[patient_id],
        }
        (val_rows if patient_fold[patient_id] == fold else train_rows).append(item)
    return train_rows, val_rows


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.78, 1.28)),
            transforms.RandomRotation(4),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


class MultiViewDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: transforms.Compose, image_size: int):
        self.rows = rows
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        images: list[torch.Tensor] = []
        mask: list[float] = []
        for path_value in row["paths"]:
            path = Path(path_value) if path_value else None
            if path is None or not path.is_file():
                images.append(torch.zeros(3, self.image_size, self.image_size))
                mask.append(0.0)
                continue
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                images.append(self.transform(image))
            mask.append(1.0)
        return (
            torch.stack(images),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(row["labels"], dtype=torch.long),
            row["patient_id"],
        )


class MultiViewClassifier(nn.Module):
    def __init__(
        self,
        class_counts: list[int],
        view_checkpoint: Path | None,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if view_checkpoint is not None:
            checkpoint = torch.load(view_checkpoint, map_location="cpu", weights_only=False)
            state = checkpoint["model_state_dict"]
            compatible = {
                key: value
                for key, value in state.items()
                if key in backbone_model.state_dict()
                and backbone_model.state_dict()[key].shape == value.shape
                and not key.startswith("fc.")
            }
            backbone_model.load_state_dict(compatible, strict=False)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim)
        )
        self.slot_embeddings = nn.Parameter(torch.zeros(len(SLOT_NAMES), projection_dim))
        nn.init.normal_(self.slot_embeddings, std=0.02)
        fused_dim = len(SLOT_NAMES) * projection_dim + 2 * projection_dim + len(SLOT_NAMES)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, count) for count in class_counts)

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> list[torch.Tensor]:
        batch_size, views, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch_size * views, channels, height, width))
        features = features.flatten(1).reshape(batch_size, views, -1)
        projected = self.projection(features) + self.slot_embeddings.unsqueeze(0)
        mask = view_mask.unsqueeze(-1)
        masked = projected * mask
        denominator = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = masked.sum(dim=1) / denominator
        max_input = projected.masked_fill(mask == 0, -1e4)
        max_pool = max_input.max(dim=1).values
        fused = torch.cat([masked.flatten(1), mean_pool, max_pool, view_mask], dim=1)
        hidden = self.fusion(fused)
        return [head(hidden) for head in self.heads]


def class_weights(train_rows: list[dict[str, Any]], class_counts: list[int]) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    for head_index, count in enumerate(class_counts):
        frequencies = Counter(
            row["labels"][head_index]
            for row in train_rows
            if row["labels"][head_index] >= 0
        )
        maximum = max(frequencies.values()) if frequencies else 1
        values = [np.sqrt(maximum / max(frequencies.get(index, 1), 1)) for index in range(count)]
        array = np.asarray(values, dtype=np.float32)
        array /= array.mean()
        weights.append(torch.from_numpy(array))
    return weights


def masked_loss(
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    criteria: list[nn.Module],
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    observed = 0
    for head_index, head_logits in enumerate(logits):
        mask = labels[:, head_index] >= 0
        if not mask.any():
            continue
        losses.append(criteria[head_index](head_logits[mask], labels[mask, head_index]))
        observed += int(mask.sum().item())
    if not losses:
        return sum(head_logits.sum() * 0.0 for head_logits in logits), observed
    return torch.stack(losses).mean(), observed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criteria: list[nn.Module],
    device: torch.device,
    use_amp: bool,
    head_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    patient_ids: list[str] = []
    targets: list[list[int]] = []
    logits_by_head: list[list[np.ndarray]] = [[] for _ in head_names]
    for images, view_mask, labels, batch_patient_ids in loader:
        images = images.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images, view_mask)
            loss, _observed = masked_loss(logits, labels_device, criteria)
        total_loss += float(loss.item())
        total_batches += 1
        patient_ids.extend(batch_patient_ids)
        targets.extend(labels.tolist())
        for head_index, head_logits in enumerate(logits):
            logits_by_head[head_index].append(head_logits.float().cpu().numpy())

    target_array = np.asarray(targets, dtype=np.int64)
    output_logits = [np.concatenate(values, axis=0) for values in logits_by_head]
    head_metrics: dict[str, Any] = {}
    macro_f1_values: list[float] = []
    for head_index, head_name in enumerate(head_names):
        mask = target_array[:, head_index] >= 0
        predictions = output_logits[head_index].argmax(axis=1)
        if mask.any():
            macro_f1 = float(
                f1_score(
                    target_array[mask, head_index], predictions[mask], average="macro", zero_division=0
                )
            )
            accuracy = float(accuracy_score(target_array[mask, head_index], predictions[mask]))
            macro_f1_values.append(macro_f1)
        else:
            macro_f1 = float("nan")
            accuracy = float("nan")
        head_metrics[head_name] = {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "observed": int(mask.sum()),
        }
    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "heads": head_metrics,
    }
    predictions_payload = {
        "patient_ids": np.asarray(patient_ids),
        "targets": target_array,
        "logits": output_logits,
    }
    return metrics, predictions_payload


def save_predictions(
    path: Path, payload: dict[str, Any], head_names: list[str]
) -> None:
    arrays: dict[str, Any] = {
        "patient_ids": payload["patient_ids"],
        "targets": payload["targets"],
        "head_names": np.asarray(head_names),
    }
    for head_name, logits in zip(head_names, payload["logits"]):
        arrays[f"logits_{head_name}"] = logits
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--view-checkpoint", type=Path)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    seed = args.seed + args.fold
    seed_everything(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    head_names = list(head_vocabs)
    class_counts = [len(head_vocabs[name]) for name in head_names]
    train_rows, val_rows = read_rows(
        args.cache_manifest,
        args.labels,
        args.fold_assignments,
        args.fold,
        head_names,
    )
    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = DataLoader(
        MultiViewDataset(train_rows, train_transform, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        MultiViewDataset(val_rows, eval_transform, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewClassifier(class_counts, args.view_checkpoint).to(device)
    weights = [value.to(device) for value in class_weights(train_rows, class_counts)]
    criteria = [nn.CrossEntropyLoss(weight=value, label_smoothing=0.02) for value in weights]
    backbone_parameters = list(model.backbone.parameters())
    other_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": other_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, view_mask, labels, _patient_ids in train_loader:
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
            running_loss += float(loss.item())
            batches += 1
        val_metrics, val_predictions = evaluate(
            model, val_loader, criteria, device, use_amp, head_names
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "val_loss": val_metrics["loss"],
            "val_mean_macro_f1": val_metrics["mean_macro_f1"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if val_metrics["mean_macro_f1"] > best_f1:
            best_f1 = val_metrics["mean_macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "head_vocabs": head_vocabs,
                    "slot_names": SLOT_NAMES,
                    "fold": args.fold,
                    "epoch": epoch,
                    "val_mean_macro_f1": best_f1,
                    "image_size": args.image_size,
                    "seed": seed,
                },
                args.output_dir / "best.pt",
            )
            save_predictions(args.output_dir / "val_predictions_best.npz", val_predictions, head_names)
            (args.output_dir / "val_metrics_best.json").write_text(
                json.dumps(val_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stop_epoch": epoch, "patience": args.patience}), flush=True)
            break

    summary = {
        "fold": args.fold,
        "seed": seed,
        "train_patients": len(train_rows),
        "validation_patients": len(val_rows),
        "best_epoch": best_epoch,
        "best_mean_macro_f1": best_f1,
        "history": history,
        "head_names": head_names,
        "class_counts": dict(zip(head_names, class_counts)),
        "view_checkpoint": str(args.view_checkpoint) if args.view_checkpoint else None,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "best_mean_macro_f1": best_f1}))


if __name__ == "__main__":
    main()
