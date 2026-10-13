#!/usr/bin/env python3
"""Train a five-class intraoral photograph view classifier.

Weak labels are taken only from high-confidence cases that contain exactly the
canonical files ``intraoral_1`` through ``intraoral_5``.  The mapping was
verified visually on representative cases:

1. frontal bite
2. right buccal bite
3. left buccal bite
4. lower occlusal
5. upper occlusal

Horizontal flips are intentionally not used because they swap left and right.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = [
    "frontal",
    "right_buccal",
    "left_buccal",
    "lower_occlusal",
    "upper_occlusal",
]
INDEX_TO_CLASS = {index + 1: class_name for index, class_name in enumerate(CLASS_NAMES)}
STRUCTURAL_CLASS_NAMES = ["frontal", "lateral", "occlusal"]
STRUCTURAL_INDEX_TO_CLASS = {
    1: "frontal",
    2: "lateral",
    3: "lateral",
    4: "occlusal",
    5: "occlusal",
}
CANONICAL_RE = re.compile(r"^intraoral_([1-5])\.(?:jpe?g|png|heic|heif)$", re.IGNORECASE)


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_training_rows(
    manifest_path: Path,
    class_names: list[str],
    index_to_class: dict[int, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_patient: dict[str, list[dict[str, str]]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_patient.setdefault(row["patient_id"], []).append(row)

    examples: list[dict[str, Any]] = []
    selected_patients: list[str] = []
    for patient_id, rows in sorted(by_patient.items()):
        if len(rows) != 5 or any(row["valid"].lower() != "true" for row in rows):
            continue
        indexed_rows: dict[int, dict[str, str]] = {}
        for row in rows:
            match = CANONICAL_RE.match(row["filename"])
            if match:
                indexed_rows[int(match.group(1))] = row
        if set(indexed_rows) != set(index_to_class):
            continue
        selected_patients.append(patient_id)
        for index in sorted(indexed_rows):
            examples.append(
                {
                    "patient_id": patient_id,
                    "path": indexed_rows[index]["source_path"],
                    "label": class_names.index(index_to_class[index]),
                    "class_name": index_to_class[index],
                }
            )
    return examples, selected_patients


class PhotoDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: transforms.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"]), row["patient_id"]


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.80, 1.25)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
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


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels, _patient_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        prediction = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        correct += int((prediction == labels).sum().item())
        targets.extend(labels.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--label-scheme", choices=["semantic5", "structural3"], default="semantic5"
    )
    args = parser.parse_args()

    register_heif()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.label_scheme == "semantic5":
        class_names = CLASS_NAMES
        index_to_class = INDEX_TO_CLASS
    else:
        class_names = STRUCTURAL_CLASS_NAMES
        index_to_class = STRUCTURAL_INDEX_TO_CLASS
    rows, patients = read_training_rows(args.manifest, class_names, index_to_class)
    if not rows:
        raise RuntimeError("No high-confidence canonical five-view cases found")

    rng = random.Random(args.seed)
    shuffled_patients = patients.copy()
    rng.shuffle(shuffled_patients)
    n_val = max(1, round(len(shuffled_patients) * args.val_fraction))
    val_patients = set(shuffled_patients[:n_val])
    train_rows = [row for row in rows if row["patient_id"] not in val_patients]
    val_rows = [row for row in rows if row["patient_id"] in val_patients]

    split_rows = [
        {
            "patient_id": row["patient_id"],
            "path": row["path"],
            "label": row["label"],
            "class_name": row["class_name"],
            "split": "val" if row["patient_id"] in val_patients else "train",
        }
        for row in rows
    ]
    with (args.output_dir / "view_classifier_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = DataLoader(
        PhotoDataset(train_rows, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        PhotoDataset(val_rows, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for images, labels, _patient_ids in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            train_loss += float(loss.item()) * batch_size
            train_total += batch_size
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
        val_loss, val_accuracy, targets, predictions = evaluate(
            model, val_loader, device, criterion
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "label_scheme": args.label_scheme,
                    "image_size": args.image_size,
                    "architecture": "resnet18",
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "seed": args.seed,
                },
                args.output_dir / "view_classifier_best.pt",
            )
            (args.output_dir / "val_predictions_best.json").write_text(
                json.dumps(
                    {"targets": targets, "predictions": predictions}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        scheduler.step()

    checkpoint = torch.load(
        args.output_dir / "view_classifier_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_accuracy, targets, predictions = evaluate(model, val_loader, device, criterion)
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "architecture": "resnet18",
        "pretrained_weights": "ResNet18_Weights.DEFAULT",
        "seed": args.seed,
        "image_size": args.image_size,
        "training_patients": len(set(row["patient_id"] for row in train_rows)),
        "validation_patients": len(val_patients),
        "training_images": len(train_rows),
        "validation_images": len(val_rows),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "final_best_val_loss": val_loss,
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(class_names)))
        ).tolist(),
        "label_scheme": args.label_scheme,
        "classification_report": report,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "best_val_accuracy": best_accuracy}))


if __name__ == "__main__":
    main()
