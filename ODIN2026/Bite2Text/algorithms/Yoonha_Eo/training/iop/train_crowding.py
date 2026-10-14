#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2}
INDEX_TO_LABEL = {0: "none", 1: "mild", 2: "moderate"}
DEFAULT_MODEL_SPECS = {
    "efficientnet_v2_s": {"image_size": 384, "batch_size": 32, "lr": 2.0e-4, "dropout": 0.25},
    "convnext_tiny": {"image_size": 384, "batch_size": 28, "lr": 1.5e-4, "dropout": 0.25},
    "swin_t": {"image_size": 384, "batch_size": 24, "lr": 1.2e-4, "dropout": 0.20},
    "regnet_y_8gf": {"image_size": 384, "batch_size": 24, "lr": 1.5e-4, "dropout": 0.25},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def norm_key(path_like):
    p = str(path_like).replace("\\", "/")
    return Path(p).name


def load_labels(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("info", raw if isinstance(raw, list) else [])
    labels = {}
    dup = 0
    for row in items:
        if "image_filepath" not in row or "complexity" not in row:
            continue
        try:
            lab = int(row["complexity"])
        except Exception:
            continue
        if lab not in LABEL_TO_INDEX:
            continue
        k = norm_key(row["image_filepath"])
        if k in labels and labels[k] != lab:
            dup += 1
        labels[k] = lab
    print(f"[metadata] labels={len(labels):,} conflicting_duplicates={dup}", flush=True)
    return labels


def collect_view_samples(data_root, metadata_path, view):
    labels = load_labels(metadata_path)
    view_dir = Path(data_root) / view
    if not view_dir.exists():
        raise FileNotFoundError(view_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    missing = 0
    for path in sorted(view_dir.rglob("*")):
        if path.suffix.lower() not in exts:
            continue
        lab = labels.get(path.name)
        if lab is None:
            missing += 1
            continue
        samples.append((str(path), LABEL_TO_INDEX[lab]))
    counts = Counter(y for _, y in samples)
    print(
        f"[collect] view={view} samples={len(samples):,} missing_label={missing:,} "
        f"counts={{none:{counts[0]}, mild:{counts[1]}, moderate:{counts[2]}}}",
        flush=True,
    )
    if len(samples) < 100:
        raise RuntimeError(f"too few samples for {view}: {len(samples)}")
    return samples


def make_split(samples, seed):
    y = [s[1] for s in samples]
    train, temp = train_test_split(samples, test_size=0.2, random_state=seed, stratify=y)
    y_temp = [s[1] for s in temp]
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=y_temp)
    return {"train": train, "val": val, "test": test}


def save_split(split, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label_index", "label_name"])
        for split_name, rows in split.items():
            for p, y in rows:
                w.writerow([split_name, p, y, INDEX_TO_LABEL[y]])


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def build_transforms(image_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.82, 1.22)),
            transforms.RandomApply([transforms.ColorJitter(0.18, 0.18, 0.12, 0.04)], p=0.75),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.RandomRotation(degrees=7, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.015, 0.08), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.08))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def build_model(name, num_classes, dropout):
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        head_module = model.classifier
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1], nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        head_module = model.classifier
    elif name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        head_module = model.head
    elif name == "regnet_y_8gf":
        weights = models.RegNet_Y_8GF_Weights.DEFAULT
        model = models.regnet_y_8gf(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        head_module = model.fc
    else:
        raise ValueError(f"unknown model: {name}")
    head_ids = {id(p) for p in head_module.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    return model, backbone_params, head_params


def make_loaders(split, image_size, batch_size, workers):
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = ImageDataset(split["train"], train_tf)
    val_ds = ImageDataset(split["val"], eval_tf)
    test_ds = ImageDataset(split["test"], eval_tf)
    counts = Counter(y for _, y in split["train"])
    weights = [1.0 / counts[y] for _, y in split["train"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses, ys, preds, probs_all = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        losses.append(float(loss.item()) * x.size(0))
        ys.extend(y.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        probs_all.extend(prob.cpu().numpy().tolist())
    avg_loss = sum(losses) / max(1, len(ys))
    acc = accuracy_score(ys, preds)
    macro_f1 = f1_score(ys, preds, average="macro")
    return {
        "loss": avg_loss,
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": ys,
        "y_pred": preds,
        "probs": probs_all,
    }


def train_one(view, model_name, split, out_dir, args):
    spec = DEFAULT_MODEL_SPECS[model_name]
    image_size = spec["image_size"]
    batch_size = spec["batch_size"]
    lr = spec["lr"]
    dropout = spec["dropout"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_out = Path(out_dir) / view / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    save_split(split, model_out / "split_manifest.csv")
    train_loader, val_loader, test_loader = make_loaders(split, image_size, batch_size, args.workers)
    model, backbone_params, head_params = build_model(model_name, num_classes=3, dropout=dropout)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.35},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=lr * 0.03)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    print(
        f"[train-start] view={view} model={model_name} size={image_size} batch={batch_size} "
        f"train={len(split['train']):,} val={len(split['val']):,} test={len(split['test']):,}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)
        val = evaluate(model, val_loader, device)
        train_loss = total_loss / max(1, seen)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[epoch] {view}/{model_name} ep={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )
        if val["macro_f1"] > best_f1 + 1e-5:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_name": model_name,
                    "view": view,
                    "num_classes": 3,
                    "class_names": [INDEX_TO_LABEL[i] for i in range(3)],
                    "image_size": image_size,
                    "state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                    "label_mapping": {"1": "none", "2": "mild", "3": "moderate"},
                },
                model_out / "best.pt",
            )
        else:
            no_improve += 1
        with open(model_out / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "seconds"])
            w.writeheader()
            w.writerows(history)
        if no_improve >= args.patience:
            print(f"[early-stop] {view}/{model_name} best_epoch={best_epoch} best_val_f1={best_f1:.4f}", flush=True)
            break
    ckpt = torch.load(model_out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, device)
    report = classification_report(
        test["y_true"],
        test["y_pred"],
        target_names=[INDEX_TO_LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=[0, 1, 2]).tolist()
    metrics = {
        "view": view,
        "model_name": model_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(model_out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"[test] view={view} model={model_name} acc={test['acc']:.4f} macro_f1={test['macro_f1']:.4f} "
        f"best_epoch={best_epoch}",
        flush=True,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata", default="/datasets/aihub/metadata.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["efficientnet_v2_s", "convnext_tiny", "swin_t", "regnet_y_8gf"])
    parser.add_argument("--views", nargs="+", default=["upper", "lower"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-decay", type=float, default=6e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    args = parser.parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for view in args.views:
        samples = collect_view_samples(args.data_root, args.metadata, view)
        split = make_split(samples, args.seed)
        for model_name in args.models:
            try:
                metrics = train_one(view, model_name, split, out_dir, args)
                all_metrics.append(metrics)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[oom] view={view} model={model_name}: {e}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                raise
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print("[summary]", flush=True)
    for m in sorted(all_metrics, key=lambda x: (x["view"], -x["test_macro_f1"])):
        print(
            f"{m['view']}\t{m['model_name']}\ttest_acc={m['test_acc']:.4f}\t"
            f"test_macro_f1={m['test_macro_f1']:.4f}\tbest_epoch={m['best_epoch']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
