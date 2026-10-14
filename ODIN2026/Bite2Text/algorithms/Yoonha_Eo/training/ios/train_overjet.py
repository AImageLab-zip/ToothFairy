#!/usr/bin/env python3
"""Train five reduced-label Official-Base PTv3 single-task classifiers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pointcept.datasets import collate_fn
from pointcept.datasets.transform import Compose
from pointcept.models import build_model


TASKS = (
    ("dentition", 0, ("Deciduous", "Mixed", "Permanent")),
    ("crowding", 1, ("None", "Crowding_below", "Crowding_above")),
    ("malocclusion", 2, ("ClassI", "ClassII", "ClassIII")),
    (
        "overbite",
        3,
        ("Normal", "Deep", "Negative", "Open"),
    ),
    ("overjet", 4, ("Normal", "Deep", "Negative")),
)
SELECTIONS = ("best_macro_f1", "best_accuracy", "last")
BACKBONE_SPEC = {
    "type": "PT-v3m1",
    "depths": (2, 2, 2, 6, 2),
    "channels": (32, 64, 128, 256, 512),
    "heads": (2, 4, 8, 16, 32),
    "patch_size": (48, 48, 48, 48, 48),
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_cuda(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_cuda(item) for item in value)
    return value


def normalize_pair(coord: np.ndarray):
    coord = coord - coord.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(coord, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Invalid STL point-cloud radius")
    return coord / radius


def augment_pair(coord: np.ndarray, normal: np.ndarray, arch: np.ndarray):
    scale = np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
    coord = coord * scale
    normal = normal / scale
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-12)
    angle = np.random.uniform(-0.1, 0.1) * np.pi
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
    coord = coord @ rotation.T
    normal = normal @ rotation.T
    coord += np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
    coord += np.clip(np.random.normal(0, 0.002, size=coord.shape), -0.01, 0.01).astype(np.float32)
    if random.random() < 0.5:
        keep = np.random.choice(len(coord), size=max(1, len(coord) // 2), replace=False)
        coord, normal, arch = coord[keep], normal[keep], arch[keep]
    return coord, normal, arch


def augment_landmarks(coord: np.ndarray, point_label_onehot: np.ndarray):
    scale = np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
    coord = coord * scale
    angle = np.random.uniform(-0.1, 0.1) * np.pi
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
    coord = coord @ rotation.T
    coord += np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
    coord += np.clip(np.random.normal(0, 0.002, size=coord.shape), -0.01, 0.01).astype(np.float32)
    if random.random() < 0.5:
        keep_count = min(len(coord), max(8, len(coord) // 2))
        keep = np.random.choice(len(coord), size=keep_count, replace=False)
        coord = coord[keep]
        point_label_onehot = point_label_onehot[keep]
    return coord, point_label_onehot


class RawSTLPairDataset(Dataset):
    def __init__(
        self,
        prepared_root: Path,
        split: str,
        task_index: int,
        training: bool,
        input_mode: str,
    ):
        self.prepared_root = prepared_root
        self.task_index = task_index
        self.training = training
        self.input_mode = input_mode
        frame = pd.read_csv(prepared_root / "manifest.csv", dtype={"case_id": str})
        frame["case_id"] = frame["case_id"].str.zfill(4)
        task_name, _, class_names = TASKS[task_index]
        self.frame = frame[
            frame["split"].eq(split)
            & frame[f"{task_name}_index"].between(0, len(class_names) - 1)
        ].reset_index(drop=True)
        transforms = [
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
        ]
        if training:
            transforms.append(dict(type="ShufflePoint"))
        transforms.extend(
            [
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "case_id", "label_0", "label_1", "label_2", "label_3", "label_4"),
                    feat_keys=(
                        ["coord", "normal", "arch_onehot"]
                        if input_mode == "raw_stl"
                        else ["coord", "point_label_onehot"]
                    ),
                ),
            ]
        )
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        case_id = row["case_id"]
        with np.load(self.prepared_root / "points" / f"{case_id}.npz") as payload:
            coord = payload["coord"].astype(np.float32, copy=True)
            if self.input_mode == "raw_stl":
                normal = payload["normal"].astype(np.float32, copy=True)
                arch = payload["arch"].astype(np.float32, copy=True)
            else:
                point_label_onehot = payload["point_label_onehot"].astype(np.float32, copy=True)
        coord = normalize_pair(coord)
        if self.input_mode == "raw_stl":
            if self.training:
                coord, normal, arch = augment_pair(coord, normal, arch)
            data = {
                "coord": coord,
                "normal": normal,
                "arch_onehot": arch,
                "case_id": case_id,
                "index_valid_keys": ["coord", "normal", "arch_onehot"],
            }
        else:
            if self.training:
                coord, point_label_onehot = augment_landmarks(coord, point_label_onehot)
            data = {
                "coord": coord,
                "point_label_onehot": point_label_onehot,
                "case_id": case_id,
                "index_valid_keys": ["coord", "point_label_onehot"],
            }
        for task_name, task_index, _ in TASKS:
            data[f"label_{task_index}"] = np.asarray([int(row[f"{task_name}_index"])], dtype=np.int64)
        return self.transform(data)


class ExactClassBalancedSampler(Sampler):
    """Expose every class exactly `max(class_count)` times per epoch.

    Repeated indices are reloaded through the training dataset, so every copy
    receives a fresh joint upper/lower 3D augmentation.
    """

    def __init__(self, labels, seed: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seed = int(seed)
        self.epoch = 0
        self.classes = np.unique(self.labels)
        self.class_indices = {
            int(class_index): np.flatnonzero(self.labels == class_index)
            for class_index in self.classes
        }
        self.target_per_class = max(len(indices) for indices in self.class_indices.values())
        self.num_samples = self.target_per_class * len(self.classes)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        balanced = []
        for class_index in self.classes:
            indices = self.class_indices[int(class_index)]
            if len(indices) < self.target_per_class:
                extra = rng.choice(
                    indices,
                    size=self.target_per_class - len(indices),
                    replace=True,
                )
                indices = np.concatenate([indices, extra])
            else:
                indices = indices.copy()
            balanced.append(indices)
        epoch_indices = np.concatenate(balanced)
        rng.shuffle(epoch_indices)
        return iter(epoch_indices.tolist())


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    sampler=None,
):
    generator = torch.Generator()
    generator.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=3)
    return DataLoader(**kwargs)


def model_config(task_index: int, num_classes: int, class_weight, in_channels: int):
    return {
        "type": "SingleTaskClassifier",
        "backbone": {
            "type": "PT-v3m1",
            "in_channels": in_channels,
            "order": ("z", "z-trans"),
            "stride": (2, 2, 2, 2),
            "enc_depths": BACKBONE_SPEC["depths"],
            "enc_channels": BACKBONE_SPEC["channels"],
            "enc_num_head": BACKBONE_SPEC["heads"],
            "enc_patch_size": BACKBONE_SPEC["patch_size"],
            "mlp_ratio": 4,
            "qkv_bias": True,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "drop_path": 0.3,
            "pre_norm": True,
            "shuffle_orders": True,
            "enable_rpe": False,
            "enable_flash": False,
            "cls_mode": True,
        },
        "backbone_embed_dim": 512,
        "task_index": task_index,
        "num_classes": num_classes,
        "class_weight": class_weight,
        "label_smoothing": 0.0,
    }


def compute_class_weights(dataset: RawSTLPairDataset, task_name: str, num_classes: int):
    labels = dataset.frame[f"{task_name}_index"].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"Empty training class for {task_name}: {counts.tolist()}")
    weights = len(labels) / (num_classes * counts)
    return weights.tolist(), counts.astype(int).tolist()


def classification_metrics(y_true, y_pred, num_classes: int):
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        confusion[actual, guessed] += 1
    precision, recall, f1, support = [], [], [], []
    for class_index in range(num_classes):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        support.append(int(confusion[class_index, :].sum()))
    return {
        "cases": int(len(true)),
        "accuracy": float(np.mean(true == pred)),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "balanced_accuracy": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, task_index: int, num_classes: int, seed: int):
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    previous_shuffle = getattr(model.backbone, "shuffle_orders", None)
    if previous_shuffle is not None:
        model.backbone.shuffle_orders = False
    model.eval()
    y_true, y_pred, case_ids = [], [], []
    loss_sum = 0.0
    try:
        for batch in loader:
            case_ids.extend(batch["case_id"])
            batch = move_to_cuda(batch)
            output = model(batch)
            target = batch[f"label_{task_index}"].view(-1)
            prediction = output["logits"].argmax(dim=1)
            y_true.extend(target.cpu().tolist())
            y_pred.extend(prediction.cpu().tolist())
            loss_sum += float(output["loss"].item()) * len(target)
    finally:
        if previous_shuffle is not None:
            model.backbone.shuffle_orders = previous_shuffle
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = loss_sum / max(1, len(y_true))
    predictions = [
        {"case_id": case_id, "target": int(target), "prediction": int(prediction)}
        for case_id, target, prediction in zip(case_ids, y_true, y_pred)
    ]
    return metrics, predictions


def save_checkpoint(path: Path, model, epoch: int, validation, metadata):
    torch.save(
        {"epoch": epoch, "state_dict": model.state_dict(), "validation": validation, **metadata},
        path,
    )


def train_task(args, task):
    task_name, task_index, class_names = task
    output = args.output_root / task_name
    result_path = output / "result.json"
    if result_path.is_file():
        print(f"SKIP completed task: {task_name}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    run_seed = args.seed * 100 + task_index
    seed_all(run_seed)

    train_dataset = RawSTLPairDataset(
        args.prepared_root, "train", task_index, training=True, input_mode=args.input_mode
    )
    val_dataset = RawSTLPairDataset(
        args.prepared_root, "val", task_index, training=False, input_mode=args.input_mode
    )
    inverse_class_weight, train_support = compute_class_weights(
        train_dataset, task_name, len(class_names)
    )
    balanced_sampler = None
    if args.balanced_augmentation:
        labels = train_dataset.frame[f"{task_name}_index"].to_numpy(dtype=np.int64)
        balanced_sampler = ExactClassBalancedSampler(labels, seed=run_seed)
        class_weight = [1.0] * len(class_names)
        epoch_class_exposure = [balanced_sampler.target_per_class] * len(class_names)
    else:
        class_weight = inverse_class_weight
        epoch_class_exposure = train_support
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        balanced_sampler is None,
        True,
        sampler=balanced_sampler,
    )
    val_loader = make_loader(val_dataset, args.eval_batch_size, 0, False, False)
    in_channels = 8 if args.input_mode == "raw_stl" else 9
    model = build_model(
        model_config(task_index, len(class_names), class_weight, in_channels)
    ).cuda()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    metadata = {
        "task": task_name,
        "task_index": task_index,
        "classes": list(class_names),
        "architecture": "one independent SingleTaskClassifier with one scratch PT-v3m1 Official Base backbone",
        "backbone_spec": BACKBONE_SPEC,
        "input_mode": args.input_mode,
        "input_channels": (
            ["RAS_x", "RAS_y", "RAS_z", "normal_x", "normal_y", "normal_z", "upper_flag", "lower_flag"]
            if args.input_mode == "raw_stl"
            else ["RAS_x", "RAS_y", "RAS_z", "Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint"]
        ),
        "uses_landmarks": args.input_mode == "landmark_only",
        "uses_fdi_as_model_feature": False,
        "uses_tooth_instance_labels_as_model_feature": False,
        "uses_fdi_for_ras_preprocessing": True,
        "excluded_original_labels": {
            "malocclusion": ["Special"],
            "overbite": ["Edge-to-edge_bite"],
            "overjet": ["Edge-to-edge"],
        }.get(task_name, []),
        "ras_alignment": "one common FDI-centroid-derived RAS rigid transform applied to the already occlusally registered upper/lower STL pair",
        "pretrained": False,
        "class_weight": class_weight,
        "train_support": train_support,
        "balanced_augmentation": args.balanced_augmentation,
        "epoch_class_exposure": epoch_class_exposure,
        "epoch_training_samples": int(sum(epoch_class_exposure)),
        "sampling": (
            "exact class-balanced replacement sampling to the majority-class count; "
            "each repeated index receives a fresh joint upper/lower augmentation"
            if args.balanced_augmentation
            else "natural class distribution with inverse-frequency cross-entropy"
        ),
        "parameters": parameter_count,
        "split": "official train split into multilabel-stratified train/val; official test held out until final evaluation",
        "augmentation": "joint RAS-frame unit sphere; anisotropic scale 0.95-1.05; shift +/-0.02; z rotation +/-18deg; jitter; 50% point dropout p=0.5; grid 0.01",
    }
    history_path = output / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_f1 = best_accuracy = -math.inf
    best_f1_epoch = best_accuracy_epoch = 0
    patience_count = 0
    last_validation = None
    stopped_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            batch = move_to_cuda(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output_dict = model(batch)
                loss = output_dict["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_cases = len(batch[f"label_{task_index}"])
            loss_sum += float(loss.item()) * batch_cases
            sample_count += batch_cases
        scheduler.step()
        last_validation, _ = evaluate(model, val_loader, task_index, len(class_names), args.seed + 1000 + task_index)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, sample_count),
            "lr": optimizer.param_groups[0]["lr"],
            "validation": last_validation,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        current_f1 = last_validation["f1"]
        current_accuracy = last_validation["accuracy"]
        if current_f1 > best_f1 + 1e-12:
            best_f1 = current_f1
            best_f1_epoch = epoch
            patience_count = 0
            save_checkpoint(output / "model_best_macro_f1.pth", model, epoch, last_validation, metadata)
        else:
            patience_count += 1
        if current_accuracy > best_accuracy + 1e-12:
            best_accuracy = current_accuracy
            best_accuracy_epoch = epoch
            save_checkpoint(output / "model_best_accuracy.pth", model, epoch, last_validation, metadata)
        print(
            f"[{task_name}] {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.4f} "
            f"val_acc={current_accuracy:.4f} val_macro_f1={current_f1:.4f} "
            f"patience={patience_count}/{args.patience}",
            flush=True,
        )
        if patience_count >= args.patience:
            stopped_epoch = epoch
            break

    save_checkpoint(output / "model_last.pth", model, stopped_epoch, last_validation, metadata)

    # The official test split is instantiated only after all training and validation selection is complete.
    test_dataset = RawSTLPairDataset(
        args.prepared_root, "test", task_index, training=False, input_mode=args.input_mode
    )
    test_loader = make_loader(test_dataset, args.eval_batch_size, 0, False, False)
    evaluations = {}
    checkpoint_epochs = {}
    for selection in SELECTIONS:
        payload = torch.load(output / f"model_{selection}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        validation_metrics, _ = evaluate(
            model, val_loader, task_index, len(class_names), args.seed + 1000 + task_index
        )
        test_metrics, test_predictions = evaluate(
            model, test_loader, task_index, len(class_names), args.seed + 2000 + task_index
        )
        checkpoint_epochs[selection] = int(payload["epoch"])
        evaluations[selection] = {"validation": validation_metrics, "test": test_metrics}
        pd.DataFrame(test_predictions).to_csv(output / f"test_predictions_{selection}.csv", index=False)

    result = {
        **metadata,
        "train_cases": len(train_dataset),
        "val_cases": len(val_dataset),
        "test_cases": len(test_dataset),
        "max_epochs": args.epochs,
        "patience": args.patience,
        "stopped_epoch": stopped_epoch,
        "best_macro_f1_epoch": best_f1_epoch,
        "best_accuracy_epoch": best_accuracy_epoch,
        "checkpoint_epochs": checkpoint_epochs,
        "evaluations": evaluations,
        "test_access_policy": "test was not instantiated or evaluated until training and validation checkpoint selection finished",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, optimizer, scheduler, scaler, train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate(results, output_root: Path):
    rows = []
    for result in results:
        for selection in SELECTIONS:
            metrics = result["evaluations"][selection]["test"]
            rows.append(
                {
                    "task": result["task"],
                    "checkpoint": selection,
                    "epoch": result["checkpoint_epochs"][selection],
                    **{key: metrics[key] for key in ("accuracy", "precision", "recall", "balanced_accuracy", "f1")},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "summary.csv", index=False)
    summary = {
        "design": f"five independent reduced-label scratch PT-v3m1 Official Base classifiers from {results[0]['input_mode'] if results else 'unknown'} inputs",
        "results": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Reduced-label Official Base PTv3: five independent tasks",
        "",
        "Official test metrics. Checkpoints were selected using validation only.",
        "",
        "| Task | Checkpoint | Epoch | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['checkpoint']} | {row['epoch']} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(["", "## Mean across five tasks", "", "| Checkpoint | Accuracy | Macro F1 |", "|---|---:|---:|"])
    for selection in SELECTIONS:
        subset = frame[frame["checkpoint"].eq(selection)]
        if len(subset):
            lines.append(f"| {selection} | {subset['accuracy'].mean():.4f} | {subset['f1'].mean():.4f} |")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("raw_stl", "landmark_only"), required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balanced-augmentation",
        action="store_true",
        help=(
            "Balance every class to the majority count each epoch through replacement "
            "sampling with fresh online augmentation; use uniform CE weights."
        ),
    )
    parser.add_argument("--tasks", nargs="+", choices=[task[0] for task in TASKS], default=[task[0] for task in TASKS])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected_tasks = [task for task in TASKS if task[0] in args.tasks]
    results = []
    for index, task in enumerate(selected_tasks, start=1):
        print(f"=== TASK {index}/{len(selected_tasks)}: {task[0]} ===", flush=True)
        results.append(train_task(args, task))
        aggregate(results, args.output_root)
    aggregate(results, args.output_root)
    print(f"Completed {len(results)} independent Official Base models", flush=True)


if __name__ == "__main__":
    main()
