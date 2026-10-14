
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.functional import hflip

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["right", "front", "left", "upper", "lower"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
LEFT_INDEX = CLASS_TO_INDEX["left"]
RIGHT_INDEX = CLASS_TO_INDEX["right"]
SEED = 20260812
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_SPECS = {
    "efficientnet_b0": {"image_size": 288, "batch_size": 64, "dropout": 0.35, "lr_backbone": 1.5e-4, "lr_head": 5e-4},
    "efficientnet_b3": {"image_size": 320, "batch_size": 32, "dropout": 0.45, "lr_backbone": 1.2e-4, "lr_head": 4e-4},
    "convnext_tiny": {"image_size": 320, "batch_size": 32, "dropout": 0.40, "lr_backbone": 8e-5, "lr_head": 3e-4},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/datasets/aihub/views"))
    parser.add_argument("--metadata", type=Path, default=Path("/datasets/aihub/metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/outputs/view"))
    parser.add_argument("--models", nargs="+", default=["efficientnet_b0", "efficientnet_b3", "convnext_tiny"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--target", type=float, default=0.995)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_records(data_root: Path, metadata_path: Path):
    metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8-sig"))["info"]
    complexity_by_name = {Path(row["image_filepath"]).name: int(row["complexity"]) for row in metadata_rows}
    records = []
    skipped = []
    for label_name in CLASSES:
        folder = data_root / label_name
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel_parts = path.relative_to(folder).parts
            if any(part.startswith(".") for part in rel_parts) or "checkpoint" in path.name.lower():
                skipped.append(str(path))
                continue
            if path.name not in complexity_by_name:
                raise KeyError(f"Missing metadata for {path}")
            records.append({
                "path": str(path),
                "label_name": label_name,
                "label": CLASS_TO_INDEX[label_name],
                "complexity": complexity_by_name[path.name],
                "stratum": f"{label_name}_c{complexity_by_name[path.name]}",
            })
    print(f"Loaded {len(records)} clean images; skipped hidden/checkpoint={len(skipped)}", flush=True)
    print("Class counts", Counter(row["label_name"] for row in records), flush=True)
    return records


def split_records(records):
    indices = np.arange(len(records))
    strata = np.array([row["stratum"] for row in records])
    train_idx, holdout_idx = train_test_split(indices, test_size=0.20, random_state=SEED, shuffle=True, stratify=strata)
    holdout_strata = strata[holdout_idx]
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.50, random_state=SEED + 1, shuffle=True, stratify=holdout_strata)
    split = {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }
    paths = [r["path"] for rows in split.values() for r in rows]
    assert len(paths) == len(set(paths)) == len(records)
    return split


class ViewDataset(Dataset):
    def __init__(self, records, transform, training=False, hflip_swap_p=0.25):
        self.records = list(records)
        self.transform = transform
        self.training = training
        self.hflip_swap_p = hflip_swap_p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        label = int(row["label"])
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            # Horizontal flip changes left <-> right but preserves front/upper/lower.
            if self.training and random.random() < self.hflip_swap_p:
                image = hflip(image)
                if label == LEFT_INDEX:
                    label = RIGHT_INDEX
                elif label == RIGHT_INDEX:
                    label = LEFT_INDEX
            image = self.transform(image)
        return image, label


def make_transforms(image_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = v2.Compose([
        v2.RandomResizedCrop((image_size, image_size), scale=(0.72, 1.0), ratio=(0.82, 1.22)),
        v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.04)], p=0.75),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.18),
        v2.RandomRotation(degrees=8),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        v2.RandomErasing(p=0.18, scale=(0.015, 0.08), ratio=(0.3, 3.3), value="random"),
    ])
    eval_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def worker_seed(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(split, image_size, batch_size, workers):
    train_tf, eval_tf = make_transforms(image_size)
    train_ds = ViewDataset(split["train"], train_tf, training=True)
    val_ds = ViewDataset(split["val"], eval_tf)
    test_ds = ViewDataset(split["test"], eval_tf)
    train_counts = Counter(row["label"] for row in split["train"])
    sample_weights = [1.0 / train_counts[row["label"]] for row in split["train"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(SEED))
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, worker_init_fn=worker_seed)
    return (
        DataLoader(train_ds, sampler=sampler, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_model(name, dropout):
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(in_features, len(CLASSES)))
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(in_features, len(CLASSES)))
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(CLASSES))
        # classifier already has LayerNorm/Flatten; add stochastic-ish dropout by replacing last two? keep official head stable.
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    else:
        raise ValueError(name)
    return model, list(backbone_params), list(head_params)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    acc = float(np.trace(matrix) / max(matrix.sum(), 1))
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0)
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_precision": {CLASSES[i]: float(prec[i]) for i in range(len(CLASSES))},
        "per_class_recall": {CLASSES[i]: float(rec[i]) for i in range(len(CLASSES))},
        "per_class_f1": {CLASSES[i]: float(f1[i]) for i in range(len(CLASSES))},
        "confusion_matrix": matrix.tolist(),
        "labels": y_true,
        "predictions": y_pred,
    }


def train_one(model_name, split, args, device):
    spec = MODEL_SPECS[model_name]
    out = args.output_dir / model_name
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(split, spec["image_size"], spec["batch_size"], args.workers)
    model, backbone_params, head_params = build_model(model_name, spec["dropout"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=spec["lr_head"], weight_decay=args.weight_decay)
    scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = (-1.0, -1.0)
    best_epoch = -1
    stale = 0
    history = []
    ckpt_path = out / "best.pt"
    print(f"\n===== TRAIN {model_name} image_size={spec['image_size']} batch={spec['batch_size']} =====", flush=True)
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": spec["lr_backbone"]},
                {"params": head_params, "lr": spec["lr_head"]},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_epochs, 1), eta_min=1e-6)
        model.train()
        start = time.time()
        running = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        if scheduler is not None:
            scheduler.step()
        val = evaluate(model, val_loader, criterion, device)
        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "seconds": time.time() - start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = (val["macro_f1"], val["accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "classes": CLASSES,
                "architecture": model_name,
                "model_name": model_name,
                "image_size": spec["image_size"],
                "seed": SEED,
                "best_epoch": best_epoch,
                "val_metrics": {k: v for k, v in val.items() if k not in {"labels", "predictions"}},
                "augmentation": "RandomResizedCrop, ColorJitter, GaussianBlur, Rotation, RandomErasing, hflip left-right label swap",
            }, ckpt_path)
        else:
            stale += 1
        if stale >= args.patience and epoch >= max(10, args.freeze_epochs + 2):
            print(f"Early stopping {model_name} at epoch {epoch}", flush=True)
            break
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    val = evaluate(model, val_loader, criterion, device)
    test = evaluate(model, test_loader, criterion, device)
    report = classification_report(test["labels"], test["predictions"], labels=list(range(len(CLASSES))), target_names=CLASSES, digits=6, zero_division=0)
    (out / "test_classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out / "test_confusion_matrix.csv", np.asarray(test["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
    metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "image_size": spec["image_size"],
        "batch_size": spec["batch_size"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation": {k: v for k, v in val.items() if k not in {"labels", "predictions"}},
        "test": {k: v for k, v in test.items() if k not in {"labels", "predictions"}},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps({"model": model_name, "best_epoch": best_epoch, "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"], "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"]}), flush=True)
    print(report, flush=True)
    return metrics


def main():
    args = parse_args()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    records = load_records(args.data_root, args.metadata)
    split = split_records(records)
    print("Split counts", flush=True)
    for name, rows in split.items():
        print(name, len(rows), Counter(r["label_name"] for r in rows), flush=True)
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "label_name", "label", "complexity", "stratum"])
        writer.writeheader()
        for split_name, rows in split.items():
            for row in rows:
                writer.writerow({"split": split_name, **row})
    (args.output_dir / "class_mapping.json").write_text(json.dumps(CLASS_TO_INDEX, indent=2), encoding="utf-8")
    all_metrics = {}
    for model_name in args.models:
        all_metrics[model_name] = train_one(model_name, split, args, device)
    ranking = sorted(all_metrics.values(), key=lambda m: (m["test"]["macro_f1"], m["test"]["accuracy"]), reverse=True)
    summary = {"seed": SEED, "classes": CLASSES, "models": all_metrics, "ranking": [{"model": m["model"], "test_acc": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"], "val_acc": m["validation"]["accuracy"], "val_macro_f1": m["validation"]["macro_f1"]} for m in ranking]}
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary["ranking"]), flush=True)


if __name__ == "__main__":
    main()
