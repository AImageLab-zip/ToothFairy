#!/usr/bin/env python3
"""Train independent PTv3 classifiers for curve of Spee and Wilson.

Input points are only vertices selected by 3DTeethLand tooth masks. Raw gingiva,
palate, scan borders, and landmark coordinates are excluded. Upper and lower
tooth masks are transformed into one shared RAS frame before per-tooth sampling.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pointcept.datasets import collate_fn
from pointcept.datasets.transform import Compose
from pointcept.models import build_model


CACHE_VERSION = "toothmask_v1_ras_balanced_by_fdi"
FEATURE_DIM = 20  # coord 3 + normal 3 + arch 2 + tooth-position 8 + quadrant 4
TASKS = {
    "spee": {
        "label_col": "Curve of Spee",
        "classes": ["Normal", "Increased"],
    },
    "wilson": {
        "label_col": "Curve of Wilson Degree",
        "classes": ["Normal", "Increased"],
    },
}


def configure_split_profile(profile: str) -> None:
    """Public reconstruction of the class-filter difference recorded in final2 results.

    Wilson's original run retained Reversed Spee cases for joint stratification.
    This switch reconstructs that documented protocol; it is not an archived
    byte-identical copy of the earlier Wilson training script.
    """
    TASKS["spee"]["classes"] = ["Normal", "Increased"]
    if profile == "wilson-final2":
        TASKS["spee"]["classes"].append("Reversed")
    elif profile != "spee-final2":
        raise ValueError(profile)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def move_to_cuda(batch: dict) -> dict:
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_compacted_mesh(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Match the pymeshlab vertex ordering used by 3DTeethLand inference."""
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(stl_path))
    mesh_set.meshing_remove_duplicate_vertices()
    mesh = mesh_set.current_mesh()
    mesh.compact()
    coord = np.asarray(mesh.vertex_matrix(), dtype=np.float32)
    normal = np.asarray(mesh.vertex_normal_matrix(), dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
        raise ValueError(f"Invalid compacted mesh: {stl_path}")
    if normal.shape != coord.shape:
        raise ValueError(f"Normal shape mismatch: {stl_path}: {normal.shape} vs {coord.shape}")
    return coord, normal


def load_segmentation(seg_path: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(seg_path.read_text(encoding="utf-8"))
    labels = np.asarray(payload.get("labels", []), dtype=np.int64).reshape(-1)
    instances = np.asarray(payload.get("instances", []), dtype=np.int64).reshape(-1)
    if len(labels) != vertex_count or len(instances) != vertex_count:
        raise ValueError(
            f"3DTeethLand/mesh vertex mismatch for {seg_path}: "
            f"mesh={vertex_count}, labels={len(labels)}, instances={len(instances)}"
        )
    return labels, instances


def load_original_to_ras(pointset_path: Path) -> np.ndarray:
    payload = json.loads(pointset_path.read_text(encoding="utf-8"))
    matrix = np.asarray(
        payload["transform"]["homogeneous_original_to_ras"], dtype=np.float64
    )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid original_to_ras transform: {pointset_path}")
    return matrix


def transform_geometry(
    coord: np.ndarray, normal: np.ndarray, original_to_ras: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        [coord.astype(np.float64), np.ones((len(coord), 1), dtype=np.float64)], axis=1
    )
    transformed_coord = (homogeneous @ original_to_ras.T)[:, :3].astype(np.float32)
    rotation = original_to_ras[:3, :3]
    transformed_normal = (normal.astype(np.float64) @ rotation.T).astype(np.float32)
    norm = np.linalg.norm(transformed_normal, axis=1, keepdims=True)
    transformed_normal = transformed_normal / np.maximum(norm, 1e-8)
    return transformed_coord, transformed_normal


def valid_fdi_mask(labels: np.ndarray, instances: np.ndarray, arch: str) -> np.ndarray:
    quadrants = (1, 2) if arch == "upper" else (3, 4)
    quadrant = labels // 10
    position = labels % 10
    return (
        (instances >= 0)
        & np.isin(quadrant, quadrants)
        & (position >= 1)
        & (position <= 8)
    )


def sample_by_tooth(
    coord: np.ndarray,
    normal: np.ndarray,
    labels: np.ndarray,
    instances: np.ndarray,
    arch: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    valid = valid_fdi_mask(labels, instances, arch)
    teeth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fdi in sorted(int(value) for value in np.unique(labels[valid])):
        index = np.flatnonzero(valid & (labels == fdi))
        if len(index) >= 8:
            teeth[fdi] = (coord[index], normal[index])
    return teeth


def build_case_cache(
    case_id: str,
    dataset_root: Path,
    segmentation_root: Path,
    pointset_root: Path,
    cache_dir: Path,
    sample_points: int,
    seed: int,
    overwrite: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_id}_n{sample_points}.npz"
    if cache_path.exists() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as cached:
            version = str(cached["cache_version"].item())
            if version == CACHE_VERSION and cached["coord"].shape == (sample_points, 3):
                return cache_path

    pointset_path = pointset_root / f"{case_id}.json"
    if not pointset_path.exists():
        raise FileNotFoundError(f"Missing shared-RAS transform: {pointset_path}")
    original_to_ras = load_original_to_ras(pointset_path)

    all_teeth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    alignment = {}
    for arch in ("upper", "lower"):
        stl_path = dataset_root / case_id / "ios" / f"ios_{arch}.stl"
        seg_path = segmentation_root / f"{case_id}_{arch}.json"
        if not stl_path.exists() or not seg_path.exists():
            raise FileNotFoundError(f"Missing {arch} STL or tooth mask: {stl_path}, {seg_path}")
        coord, normal = load_compacted_mesh(stl_path)
        labels, instances = load_segmentation(seg_path, len(coord))
        coord, normal = transform_geometry(coord, normal, original_to_ras)
        arch_teeth = sample_by_tooth(coord, normal, labels, instances, arch)
        all_teeth.update(arch_teeth)
        alignment[arch] = {
            "mesh_vertices": int(len(coord)),
            "mask_vertices": int(valid_fdi_mask(labels, instances, arch).sum()),
            "teeth": sorted(arch_teeth),
        }

    if len(all_teeth) < 8:
        raise ValueError(f"Too few valid teeth for {case_id}: {sorted(all_teeth)}")

    rng = np.random.default_rng(stable_seed(case_id, seed))
    fdis = sorted(all_teeth)
    base, remainder = divmod(sample_points, len(fdis))
    sampled_coord, sampled_normal, sampled_fdi = [], [], []
    for order, fdi in enumerate(fdis):
        tooth_coord, tooth_normal = all_teeth[fdi]
        count = base + int(order < remainder)
        choice = rng.choice(len(tooth_coord), size=count, replace=len(tooth_coord) < count)
        sampled_coord.append(tooth_coord[choice])
        sampled_normal.append(tooth_normal[choice])
        sampled_fdi.append(np.full(count, fdi, dtype=np.int16))

    coord = np.concatenate(sampled_coord, axis=0).astype(np.float32)
    normal = np.concatenate(sampled_normal, axis=0).astype(np.float32)
    fdi = np.concatenate(sampled_fdi, axis=0)
    permutation = rng.permutation(len(coord))
    coord, normal, fdi = coord[permutation], normal[permutation], fdi[permutation]

    # Preserve metric scale and the upper/lower relationship; only remove global translation.
    coord -= np.median(coord, axis=0, keepdims=True).astype(np.float32)
    quadrant = fdi // 10
    position = fdi % 10
    arch_onehot = np.zeros((sample_points, 2), dtype=np.float32)
    arch_onehot[:, 0] = np.isin(quadrant, (1, 2))
    arch_onehot[:, 1] = np.isin(quadrant, (3, 4))
    position_onehot = np.eye(8, dtype=np.float32)[position - 1]
    quadrant_onehot = np.eye(4, dtype=np.float32)[quadrant - 1]

    np.savez_compressed(
        cache_path,
        coord=coord,
        normal=normal,
        arch_onehot=arch_onehot,
        position_onehot=position_onehot,
        quadrant_onehot=quadrant_onehot,
        fdi=fdi,
        case_id=np.asarray(case_id),
        cache_version=np.asarray(CACHE_VERSION),
        alignment_json=np.asarray(json.dumps(alignment, sort_keys=True)),
    )
    return cache_path


def prepare_samples(args) -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(args.labels_csv, dtype=str)
    required = ["Patient"] + [TASKS[task]["label_col"] for task in TASKS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing label columns: {missing}")

    mappings = {
        task: {name.lower(): index for index, name in enumerate(info["classes"])}
        for task, info in TASKS.items()
    }
    samples, excluded = [], []
    for row in frame.to_dict("records"):
        case_id = str(row["Patient"]).strip()
        labels = {}
        label_names = {}
        invalid = []
        for task, info in TASKS.items():
            text = str(row[info["label_col"]]).strip()
            index = mappings[task].get(text.lower(), -1)
            if index < 0:
                invalid.append(f"{task}={text}")
            labels[task] = index
            label_names[task] = text
        if invalid:
            reason = "excluded_class" if any("spee=Reversed" in item for item in invalid) else "unknown_label"
            excluded.append({"Patient": case_id, "reason": reason, "detail": invalid})
            continue
        try:
            cache_path = build_case_cache(
                case_id=case_id,
                dataset_root=args.dataset_root,
                segmentation_root=args.segmentation_root,
                pointset_root=args.pointset_root,
                cache_dir=args.cache_dir,
                sample_points=args.sample_points,
                seed=args.seed,
                overwrite=args.overwrite_cache,
            )
        except Exception as exc:
            excluded.append(
                {
                    "Patient": case_id,
                    "reason": "cache_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        samples.append(
            {
                "Patient": case_id,
                "path": str(cache_path),
                "spee_label": labels["spee"],
                "spee_name": TASKS["spee"]["classes"][labels["spee"]],
                "wilson_label": labels["wilson"],
                "wilson_name": TASKS["wilson"]["classes"][labels["wilson"]],
            }
        )

    args.split_dir.mkdir(parents=True, exist_ok=True)
    (args.split_dir / "excluded.json").write_text(
        json.dumps(excluded, indent=2), encoding="utf-8"
    )
    return samples, excluded


def make_shared_splits(samples: list[dict], seed: int, split_dir: Path):
    indices = np.arange(len(samples))
    joint = np.asarray(
        [sample["spee_label"] * len(TASKS["wilson"]["classes"]) + sample["wilson_label"] for sample in samples],
        dtype=np.int64,
    )
    train_index, heldout_index = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=joint
    )
    val_index, test_index = train_test_split(
        heldout_index,
        test_size=0.5,
        random_state=seed,
        stratify=joint[heldout_index],
    )
    index_by_split = {
        "train": sorted(train_index.tolist()),
        "val": sorted(val_index.tolist()),
        "test": sorted(test_index.tolist()),
    }
    split_samples = {
        split: [samples[index] for index in split_indices]
        for split, split_indices in index_by_split.items()
    }

    rows = []
    for split, split_rows in split_samples.items():
        for sample in split_rows:
            rows.append({"split": split, **sample})
    pd.DataFrame(rows).sort_values(["split", "Patient"]).to_csv(
        split_dir / "split_manifest.csv", index=False
    )
    summary = {
        "seed": seed,
        "strategy": "joint stratified 80/10/10 on Spee x Wilson; common split for both tasks",
        "sizes": {split: len(rows) for split, rows in split_samples.items()},
        "class_counts": {
            split: {
                task: dict(Counter(row[f"{task}_name"] for row in split_rows))
                for task in TASKS
            }
            for split, split_rows in split_samples.items()
        },
        "joint_counts": {
            split: dict(
                Counter(f"{row['spee_name']}|{row['wilson_name']}" for row in split_rows)
            )
            for split, split_rows in split_samples.items()
        },
    }
    (split_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return split_samples, summary


class ToothMaskDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        task: str,
        training: bool,
        seed: int,
        grid_size: float,
    ):
        self.samples = samples
        self.task = task
        self.training = training
        self.seed = seed
        self.transform = Compose(
            [
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "label_0"),
                    feat_keys=[
                        "coord",
                        "normal",
                        "arch_onehot",
                        "position_onehot",
                        "quadrant_onehot",
                    ],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with np.load(sample["path"], allow_pickle=False) as cache:
            coord = cache["coord"].astype(np.float32).copy()
            normal = cache["normal"].astype(np.float32).copy()
            arch = cache["arch_onehot"].astype(np.float32).copy()
            position = cache["position_onehot"].astype(np.float32).copy()
            quadrant = cache["quadrant_onehot"].astype(np.float32).copy()

        if self.training:
            rng = np.random.default_rng()
            scale = float(rng.uniform(0.98, 1.02))
            angle = float(rng.uniform(-0.08, 0.08))
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            coord = (coord @ rotation.T) * scale
            coord += rng.uniform(-0.5, 0.5, size=(1, 3)).astype(np.float32)
            normal = normal @ rotation.T
            if rng.random() < 0.5:
                keep = rng.random(len(coord)) >= 0.10
                if int(keep.sum()) >= len(coord) // 2:
                    coord, normal = coord[keep], normal[keep]
                    arch, position, quadrant = arch[keep], position[keep], quadrant[keep]
        else:
            np.random.seed(self.seed + index)

        return self.transform(
            {
                "coord": coord,
                "normal": normal,
                "arch_onehot": arch,
                "position_onehot": position,
                "quadrant_onehot": quadrant,
                "label_0": np.asarray([sample[f"{self.task}_label"]], dtype=np.int64),
                "index_valid_keys": [
                    "coord",
                    "normal",
                    "arch_onehot",
                    "position_onehot",
                    "quadrant_onehot",
                ],
            }
        )


def model_config(num_classes: int, class_weights: list[float]) -> dict:
    return dict(
        type="MultiTaskClassifier",
        backbone_embed_dim=512,
        num_classes_list=[num_classes],
        stl_task=0,
        class_weights=[class_weights],
        loss_type="ce",
        label_smoothing=0.03,
        backbone=dict(
            type="PT-v3m1",
            in_channels=FEATURE_DIM,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=True,
        ),
    )


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, drop_last=False):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=3)
    return DataLoader(**kwargs)


def classification_metrics(targets, predictions, classes):
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": int(confusion[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "cases": len(targets),
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "macro_precision": float(np.mean([row["precision"] for row in per_class.values()])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class.values()])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, samples, task, classes, prediction_path=None):
    model.eval()
    targets, predictions, all_probabilities = [], [], []
    total_loss, seen = 0.0, 0
    for batch in loader:
        batch = move_to_cuda(batch)
        output = model(batch)
        logits = output["logits"][0]
        target = batch["label_0"].view(-1)
        probabilities = F.softmax(logits, dim=1)
        prediction = probabilities.argmax(dim=1)
        targets.extend(target.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
        total_loss += float(output["loss_0"].item()) * int(target.numel())
        seen += int(target.numel())
    result = classification_metrics(targets, predictions, classes)
    result["loss"] = total_loss / max(1, seen)
    if prediction_path is not None:
        rows = []
        for sample, target, prediction, probabilities in zip(
            samples, targets, predictions, all_probabilities
        ):
            row = {
                "Patient": sample["Patient"],
                "ground_truth_index": target,
                "ground_truth": classes[target],
                "prediction_index": prediction,
                "prediction": classes[prediction],
            }
            row.update(
                {
                    f"prob_{name.lower().replace(' ', '_')}": float(probabilities[index])
                    for index, name in enumerate(classes)
                }
            )
            rows.append(row)
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def save_checkpoint(path, model, epoch, validation, metadata):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "validation": validation,
            **metadata,
        },
        path,
    )


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = OrderedDict(
        (key.removeprefix("module."), value) for key, value in payload["state_dict"].items()
    )
    model.load_state_dict(state, strict=True)


def train_task(args, task: str, split_samples: dict, split_summary: dict, excluded_count: int):
    output_dir = args.output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.overwrite_results:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                print(f"[{task}] valid result.json exists; skipping", flush=True)
                return result
        except Exception:
            pass

    seed_all(args.seed + (0 if task == "spee" else 10_000))
    info = TASKS[task]
    classes = info["classes"]
    datasets = {
        split: ToothMaskDataset(
            rows,
            task=task,
            training=split == "train",
            seed=args.seed + 1000 * order,
            grid_size=args.grid_size,
        )
        for order, (split, rows) in enumerate(split_samples.items())
    }
    loaders = {
        "train": make_loader(
            datasets["train"], args.batch_size, args.workers, True, drop_last=True
        ),
        "val": make_loader(datasets["val"], args.eval_batch_size, 0, False),
        "test": make_loader(datasets["test"], args.eval_batch_size, 0, False),
    }
    train_counts = np.bincount(
        [row[f"{task}_label"] for row in split_samples["train"]],
        minlength=len(classes),
    ).astype(np.float64)
    if np.any(train_counts == 0):
        raise ValueError(f"Empty training class for {task}: {train_counts.tolist()}")
    class_weights = (
        len(split_samples["train"]) / (len(classes) * train_counts)
    ).tolist()
    model = build_model(model_config(len(classes), class_weights)).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    metadata = {
        "task": task,
        "label_col": info["label_col"],
        "classes": classes,
        "input": (
            "3DTeethLand tooth-mask vertices only; shared-RAS coordinate + normal + "
            "arch one-hot + FDI position one-hot + quadrant one-hot"
        ),
        "feature_dim": FEATURE_DIM,
        "sample_points": args.sample_points,
        "grid_size_mm": args.grid_size,
        "backbone": "Pointcept PT-v3m1 Official Base size, scratch",
        "class_weights": class_weights,
        "seed": args.seed,
        "excluded_count": excluded_count,
    }

    best_f1 = best_accuracy = -1.0
    best_f1_epoch = best_accuracy_epoch = 0
    stale = 0
    stopped_epoch = args.epochs
    last_validation = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            batch = move_to_cuda(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch)
                loss = output["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            cases = int(batch["label_0"].numel())
            total_loss += float(loss.item()) * cases
            seen += cases
        scheduler.step()
        validation = evaluate(
            model, loaders["val"], split_samples["val"], task, classes
        )
        validation.update(
            epoch=epoch,
            train_loss=total_loss / max(1, seen),
            lr=optimizer.param_groups[0]["lr"],
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation) + "\n")

        macro_f1 = validation["macro_f1"]
        accuracy = validation["accuracy"]
        if macro_f1 > best_f1 + args.min_delta:
            best_f1 = macro_f1
            best_f1_epoch = epoch
            stale = 0
            save_checkpoint(
                output_dir / "model_best_macro_f1.pth",
                model,
                epoch,
                validation,
                metadata,
            )
        else:
            stale += 1
        if accuracy > best_accuracy + args.min_delta:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            save_checkpoint(
                output_dir / "model_best_accuracy.pth",
                model,
                epoch,
                validation,
                metadata,
            )
        print(
            f"[toothmask_{task}][{epoch:03d}/{args.epochs}] "
            f"train_loss={total_loss / max(1, seen):.4f} "
            f"val_loss={validation['loss']:.4f} val_acc={accuracy:.4f} "
            f"val_f1={macro_f1:.4f} patience={stale}/{args.patience}",
            flush=True,
        )
        last_validation = validation
        if stale >= args.patience:
            stopped_epoch = epoch
            break

    save_checkpoint(
        output_dir / "model_last.pth",
        model,
        stopped_epoch,
        last_validation,
        metadata,
    )
    evaluations = {}
    for selection in ("best_macro_f1", "best_accuracy", "last"):
        load_checkpoint(output_dir / f"model_{selection}.pth", model)
        evaluations[selection] = {
            "val": evaluate(
                model,
                loaders["val"],
                split_samples["val"],
                task,
                classes,
                output_dir / f"val_predictions_{selection}.csv",
            ),
            "test": evaluate(
                model,
                loaders["test"],
                split_samples["test"],
                task,
                classes,
                output_dir / f"test_predictions_{selection}.csv",
            ),
        }

    result = {
        "status": "complete",
        "task": task,
        "label_col": info["label_col"],
        "classes": classes,
        "split": split_summary,
        "max_epochs": args.epochs,
        "stopped_epoch": stopped_epoch,
        "patience": args.patience,
        "best_macro_f1_epoch": best_f1_epoch,
        "best_accuracy_epoch": best_accuracy_epoch,
        "batch_size": args.batch_size,
        "sample_points": args.sample_points,
        "grid_size_mm": args.grid_size,
        "feature_dim": FEATURE_DIM,
        "class_weights": class_weights,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluations": evaluations,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    del model, optimizer, scheduler, scaler, datasets, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["spee", "wilson", "all"], default="all")
    parser.add_argument("--split-profile", choices=["spee-final2", "wilson-final2"],
                        default="spee-final2")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/datasets/bite2text")
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(
            "/datasets/bite2text/curve_labels.csv"
        ),
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=Path(
            "/datasets/bite2text/segmentations"
        ),
    )
    parser.add_argument(
        "--pointset-root",
        type=Path,
        default=Path(
            "/datasets/bite2text/ras_pointsets"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=12288)
    parser.add_argument("--grid-size", type=float, default=0.4)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split_profile == "wilson-final2" and args.task != "wilson":
        raise ValueError("wilson-final2 is a Wilson-only cohort profile; use --task wilson")
    configure_split_profile(args.split_profile)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(
        f"Preparing tooth-mask cache: sample_points={args.sample_points}, "
        f"grid_size={args.grid_size} mm",
        flush=True,
    )
    samples, excluded = prepare_samples(args)
    if len(samples) < 800:
        raise RuntimeError(
            f"Only {len(samples)} valid cases were prepared; expected at least 800. "
            f"See {args.split_dir / 'excluded.json'}"
        )
    split_samples, split_summary = make_shared_splits(samples, args.seed, args.split_dir)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    results = {
        task: train_task(args, task, split_samples, split_summary, len(excluded))
        for task in tasks
    }
    summary = {
        "status": "complete",
        "profile": "3DTeethLand tooth-mask points -> scratch Pointcept PT-v3m1 Official Base",
        "tasks": tasks,
        "seed": args.seed,
        "valid_cases": len(samples),
        "excluded_cases": len(excluded),
        "split": split_summary,
        "best_macro_f1_test": {
            task: result["evaluations"]["best_macro_f1"]["test"]
            for task, result in results.items()
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_root / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "tasks": tasks,
                "result_json_count": len(list(args.output_root.glob("*/result.json"))),
                "checkpoint_count": len(list(args.output_root.glob("*/model_*.pth"))),
                "prediction_csv_count": len(list(args.output_root.glob("*/*_predictions_*.csv"))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
