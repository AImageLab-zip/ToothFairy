#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
原始STL多视角视觉备用模型。

输入：
    render_stl_multiview.py 生成的 <case>.npz
    shape = [views, 5, height, width]

模型：
    共享ResNet18编码每个视角
    视角注意力池化
    Upper/Lower嵌入
    五分类输出

训练权重：
    full_catboost              1.0
    segmentation_fallback      1.5
    visual_fallback            3.0

分别训练upper和lower模型，验证时报告：
    overall
    full_catboost route
    segmentation_fallback route
    visual_fallback route
    per-class recall

命令：
    train
    predict
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


CLASS_NAMES = ["Spacing", "None", "Mild", "Moderate", "Severe"]
CLASS_TO_INDEX = {
    name.lower(): index
    for index, name in enumerate(CLASS_NAMES)
}
CASE_PATTERN = re.compile(
    r"(?P<patient>[A-Za-z0-9][A-Za-z0-9_.-]*)_ios_(?P<arch>upper|lower)",
    flags=re.IGNORECASE,
)
UNASSESSABLE = {
    "",
    "not assessable",
    "not mentioned",
    "unknown",
    "none provided",
    "nan",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("\\", "/").replace("-", "/").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.replace(" / ", "/")


def parse_label(value: Any) -> Optional[dict[str, Any]]:
    text = normalize_text(value)

    if text in UNASSESSABLE:
        return None

    aliases = {
        "spacing": "spacing",
        "none": "none",
        "none or minimal": "none",
        "absent": "none",
        "mild": "mild",
        "moderate": "moderate",
        "severe": "severe",
    }

    if text in aliases:
        index = CLASS_TO_INDEX[aliases[text]]
        return {
            "raw": str(value),
            "allowed": [index],
            "ambiguous": False,
        }

    parts = [
        aliases.get(part.strip(), part.strip())
        for part in text.split("/")
        if part.strip()
    ]

    if (
        len(parts) == 2
        and all(part in CLASS_TO_INDEX for part in parts)
    ):
        indices = sorted(CLASS_TO_INDEX[part] for part in parts)

        if indices[1] - indices[0] == 1:
            return {
                "raw": str(value),
                "allowed": indices,
                "ambiguous": True,
            }

    return None


def load_ground_truth(
    labels_json: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    records = json.loads(
        labels_json.read_text(encoding="utf-8")
    )
    output = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        patient_id = str(
            record.get("patient_id", "")
        ).strip()
        section = (
            record.get("ios_report", {})
            .get("4_alignment_and_arch", {})
        )

        if not patient_id or not isinstance(section, dict):
            continue

        for arch in ("upper", "lower"):
            parsed = parse_label(
                section.get(f"{arch}_crowding")
            )

            if parsed is not None:
                output[(patient_id, arch)] = parsed

    return output


def load_routing(
    path: Optional[Path],
) -> dict[str, str]:
    if path is None:
        return {}

    frame = pd.read_csv(
        path.resolve(),
        dtype={"case": str},
        keep_default_na=False,
    )
    return {
        str(row["case"]): str(row["route"])
        for _, row in frame.iterrows()
    }


def collect_records(
    tensor_dir: Path,
    labels_json: Optional[Path],
    routing_table: Optional[Path],
) -> pd.DataFrame:
    labels = (
        load_ground_truth(labels_json)
        if labels_json is not None
        else {}
    )
    routing = load_routing(routing_table)

    rows = []

    for tensor_path in sorted(tensor_dir.glob("*_ios_*.npz")):
        case = tensor_path.stem
        match = CASE_PATTERN.fullmatch(case)

        if match is None:
            continue

        patient_id = match.group("patient")
        arch = match.group("arch").lower()
        target = labels.get((patient_id, arch))

        rows.append(
            {
                "case": case,
                "patient_id": patient_id,
                "arch": arch,
                "tensor_path": str(tensor_path),
                "route": routing.get(case, ""),
                "target_raw": (
                    target["raw"]
                    if target is not None
                    else ""
                ),
                "target_allowed": (
                    ",".join(
                        str(index)
                        for index in target["allowed"]
                    )
                    if target is not None
                    else ""
                ),
                "target_ambiguous": (
                    int(target["ambiguous"])
                    if target is not None
                    else 0
                ),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No tensors matching '<case>_ios_<upper|lower>.npz' in {tensor_dir}"
        )
    return pd.DataFrame(rows)


def allowed_indices(value: Any) -> list[int]:
    return [
        int(item)
        for item in str(value).split(",")
        if str(item).strip()
    ]


def route_weight(
    route: str,
    full_weight: float,
    segmentation_weight: float,
    visual_weight: float,
) -> float:
    mapping = {
        "full_catboost": full_weight,
        "segmentation_fallback": segmentation_weight,
        "visual_fallback": visual_weight,
    }
    return float(mapping.get(route, 1.0))


class MultiViewTensorDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        training: bool,
        full_weight: float,
        segmentation_weight: float,
        visual_weight: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.full_weight = full_weight
        self.segmentation_weight = segmentation_weight
        self.visual_weight = visual_weight

    def __len__(self) -> int:
        return len(self.frame)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[-1])

        if random.random() < 0.25:
            view_index = random.randrange(tensor.shape[0])
            tensor = tensor.clone()
            tensor[view_index] = 0.0

        if random.random() < 0.5:
            noise = torch.randn_like(tensor[:, :4]) * 0.015
            tensor = tensor.clone()
            tensor[:, :4] = torch.clamp(
                tensor[:, :4] + noise,
                -1.0,
                1.0,
            )

        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]

        with np.load(
            row["tensor_path"],
            allow_pickle=False,
        ) as data:
            array = np.asarray(
                data["tensor"],
                dtype=np.float32,
            )

        tensor = torch.from_numpy(array)

        if self.training:
            tensor = self._augment(tensor)

        allowed = allowed_indices(row["target_allowed"])
        target = torch.zeros(
            len(CLASS_NAMES),
            dtype=torch.float32,
        )

        if allowed:
            target[allowed] = 1.0 / len(allowed)

        sample_weight = route_weight(
            str(row["route"]),
            self.full_weight,
            self.segmentation_weight,
            self.visual_weight,
        )

        return {
            "image": tensor,
            "target": target,
            "sample_weight": torch.tensor(
                sample_weight,
                dtype=torch.float32,
            ),
            "case": str(row["case"]),
            "patient_id": str(row["patient_id"]),
            "arch": str(row["arch"]),
            "route": str(row["route"]),
            "target_raw": str(row["target_raw"]),
            "target_allowed": str(row["target_allowed"]),
        }


class MultiViewResNet(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            print(
                "警告：预训练权重不可用，改用随机初始化："
                f"{type(exc).__name__}: {exc}"
            )
            backbone = resnet18(weights=None)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            5,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if weights is not None:
                mean_weight = old_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                )
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:4] = mean_weight
                new_conv.weight[:, 4:5] = mean_weight
            else:
                nn.init.kaiming_normal_(
                    new_conv.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

        backbone.conv1 = new_conv
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.view_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, channels, height, width = images.shape
        flattened = images.reshape(
            batch * views,
            channels,
            height,
            width,
        )
        features = self.backbone(flattened)
        features = features.reshape(
            batch,
            views,
            -1,
        )

        attention_logits = self.view_attention(
            features
        ).squeeze(-1)
        attention = torch.softmax(
            attention_logits,
            dim=1,
        )
        pooled = torch.sum(
            features * attention.unsqueeze(-1),
            dim=1,
        )
        logits = self.classifier(pooled)
        return logits, attention


def soft_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(
        logits,
        dim=1,
    )
    weighted_targets = targets * class_weights.unsqueeze(0)
    normalization = torch.clamp(
        weighted_targets.sum(dim=1),
        min=1e-8,
    )
    losses = -(
        weighted_targets * log_probabilities
    ).sum(dim=1) / normalization
    return (
        losses * sample_weights
    ).sum() / torch.clamp(sample_weights.sum(), min=1e-8)


def class_weights_from_frame(
    frame: pd.DataFrame,
) -> torch.Tensor:
    counts = np.zeros(
        len(CLASS_NAMES),
        dtype=np.float64,
    )

    for value in frame["target_allowed"]:
        allowed = allowed_indices(value)

        if not allowed:
            continue

        weight = 1.0 / len(allowed)

        for index in allowed:
            counts[index] += weight

    nonzero = counts[counts > 0]
    maximum = float(nonzero.max()) if len(nonzero) else 1.0
    weights = np.ones_like(counts)

    for index, count in enumerate(counts):
        if count > 0:
            weights[index] = np.clip(
                math.sqrt(maximum / count),
                1.0,
                2.5,
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def load_or_create_split(
    frame: pd.DataFrame,
    split_path: Path,
    validation_fraction: float,
    random_state: int,
) -> tuple[set[str], set[str]]:
    if split_path.is_file():
        saved = joblib.load(split_path)

        if isinstance(saved, dict):
            train = (
                saved.get("train_patients")
                or saved.get("train")
            )
            validation = (
                saved.get("validation_patients")
                or saved.get("val_patients")
                or saved.get("validation")
                or saved.get("val")
            )

            if train is not None and validation is not None:
                return set(map(str, train)), set(map(str, validation))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    indices = np.arange(len(frame))
    groups = frame["patient_id"].astype(str).to_numpy()
    train_indices, validation_indices = next(
        splitter.split(indices, groups=groups)
    )

    train = set(
        frame.iloc[train_indices]["patient_id"].astype(str)
    )
    validation = set(
        frame.iloc[validation_indices]["patient_id"].astype(str)
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "train_patients": sorted(train),
            "validation_patients": sorted(validation),
        },
        split_path,
    )
    return train, validation


def metrics_from_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    allowed = [
        allowed_indices(value)
        for value in frame["target_allowed"]
    ]
    clear = np.asarray(
        [len(values) == 1 for values in allowed],
        dtype=bool,
    )
    acceptable = np.asarray(
        [
            int(prediction in values)
            for prediction, values in zip(predictions, allowed)
        ],
        dtype=np.int64,
    )

    output: dict[str, Any] = {
        "all_count": int(len(frame)),
        "clear_count": int(clear.sum()),
        "acceptable_accuracy": (
            float(acceptable.mean())
            if len(acceptable)
            else float("nan")
        ),
    }

    if clear.any():
        truth = np.asarray(
            [values[0] for values in allowed],
            dtype=np.int64,
        )
        output.update(
            {
                "accuracy_clear": float(
                    accuracy_score(
                        truth[clear],
                        predictions[clear],
                    )
                ),
                "balanced_accuracy_clear": float(
                    balanced_accuracy_score(
                        truth[clear],
                        predictions[clear],
                    )
                ),
                "macro_f1_clear": float(
                    f1_score(
                        truth[clear],
                        predictions[clear],
                        labels=np.arange(5),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "per_class_recall": {
                    CLASS_NAMES[index]: float(value)
                    for index, value in enumerate(
                        recall_score(
                            truth[clear],
                            predictions[clear],
                            labels=np.arange(5),
                            average=None,
                            zero_division=0,
                        )
                    )
                },
                "confusion_matrix": confusion_matrix(
                    truth[clear],
                    predictions[clear],
                    labels=np.arange(5),
                ).tolist(),
            }
        )
    else:
        output.update(
            {
                "accuracy_clear": float("nan"),
                "balanced_accuracy_clear": float("nan"),
                "macro_f1_clear": float("nan"),
                "per_class_recall": {},
                "confusion_matrix": [],
            }
        )

    return output


@torch.no_grad()
def infer_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    model.eval()
    rows = []
    all_probabilities = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        logits, attention = model(images)
        probabilities = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()
        attention_array = attention.cpu().numpy()

        all_probabilities.append(probabilities)

        for index in range(len(probabilities)):
            probability = probabilities[index]
            clipped = np.clip(
                probability,
                1e-12,
                1.0,
            )
            row = {
                "case": batch["case"][index],
                "patient_id": batch["patient_id"][index],
                "arch": batch["arch"][index],
                "route": batch["route"][index],
                "target_raw": batch["target_raw"][index],
                "target_allowed": batch["target_allowed"][index],
                "prediction": CLASS_NAMES[int(probability.argmax())],
                "prediction_index": int(probability.argmax()),
                "model_confidence": float(probability.max()),
                "prediction_entropy": float(
                    -np.sum(clipped * np.log(clipped))
                    / np.log(len(CLASS_NAMES))
                ),
                "view_attention": json.dumps(
                    attention_array[index].tolist()
                ),
            }

            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"prob_{class_name.lower()}"] = float(
                    probability[class_index]
                )

            rows.append(row)

    probability_matrix = (
        np.concatenate(all_probabilities, axis=0)
        if all_probabilities
        else np.empty((0, len(CLASS_NAMES)))
    )
    return pd.DataFrame(rows), probability_matrix


def train_one_arch(
    arch: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_dataset = MultiViewTensorDataset(
        train_frame,
        training=True,
        full_weight=args.full_route_weight,
        segmentation_weight=args.segmentation_route_weight,
        visual_weight=args.visual_route_weight,
    )
    validation_dataset = MultiViewTensorDataset(
        validation_frame,
        training=False,
        full_weight=1.0,
        segmentation_weight=1.0,
        visual_weight=1.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = MultiViewResNet(
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            args.amp
            and device.type == "cuda"
        )
    )
    class_weights = class_weights_from_frame(
        train_frame
    ).to(device)

    best_score = -float("inf")
    patience = 0
    best_path = output_dir / f"{arch}_visual_fallback.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0

        for batch in train_loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            targets = batch["target"].to(
                device,
                non_blocking=True,
            )
            sample_weights = batch["sample_weight"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(
                    args.amp
                    and device.type == "cuda"
                )
            ):
                logits, _ = model(images)
                loss = soft_target_loss(
                    logits,
                    targets,
                    sample_weights,
                    class_weights,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            running_loss += float(loss.item()) * batch_size
            examples += batch_size

        scheduler.step()

        validation_predictions, validation_probabilities = infer_loader(
            model,
            validation_loader,
            device,
        )
        overall_metrics = metrics_from_predictions(
            validation_frame.reset_index(drop=True),
            validation_probabilities,
        )
        score = (
            0.60 * overall_metrics["macro_f1_clear"]
            + 0.40 * overall_metrics["balanced_accuracy_clear"]
        )

        route_metrics = {}
        for route_name in (
            "full_catboost",
            "segmentation_fallback",
            "visual_fallback",
        ):
            indices = np.flatnonzero(
                validation_frame["route"].astype(str).to_numpy()
                == route_name
            )
            if len(indices) == 0:
                continue
            route_metrics[route_name] = metrics_from_predictions(
                validation_frame.iloc[indices].reset_index(drop=True),
                validation_probabilities[indices],
            )

        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(examples, 1),
            "selection_score": score,
            "overall": overall_metrics,
            "by_route": route_metrics,
        }
        history.append(record)

        print(
            f"[{arch}] epoch={epoch:03d} "
            f"loss={record['train_loss']:.4f} "
            f"acc={overall_metrics['accuracy_clear']:.4f} "
            f"bal={overall_metrics['balanced_accuracy_clear']:.4f} "
            f"macroF1={overall_metrics['macro_f1_clear']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score + args.minimum_improvement:
            best_score = score
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": arch,
                    "class_names": CLASS_NAMES,
                    "pretrained": bool(args.pretrained),
                    "dropout": float(args.dropout),
                    "epoch": epoch,
                    "selection_score": score,
                },
                best_path,
            )
        else:
            patience += 1

            if patience >= args.early_stopping_patience:
                print(
                    f"[{arch}] early stop at epoch {epoch}"
                )
                break

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])

    predictions, probabilities = infer_loader(
        model,
        validation_loader,
        device,
    )
    final_metrics = metrics_from_predictions(
        validation_frame.reset_index(drop=True),
        probabilities,
    )
    final_metrics["by_route"] = {}

    for route_name in (
        "full_catboost",
        "segmentation_fallback",
        "visual_fallback",
    ):
        indices = np.flatnonzero(
            validation_frame["route"].astype(str).to_numpy()
            == route_name
        )
        if len(indices) == 0:
            continue
        final_metrics["by_route"][route_name] = metrics_from_predictions(
            validation_frame.iloc[indices].reset_index(drop=True),
            probabilities[indices],
        )

    (output_dir / f"{arch}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_metrics, predictions


def train(args: argparse.Namespace) -> int:
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tensor_dir = args.tensor_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(
        tensor_dir=tensor_dir,
        labels_json=args.labels_json.resolve(),
        routing_table=args.routing_table.resolve()
        if args.routing_table
        else None,
    )
    records = records[
        records["target_allowed"].astype(str).str.strip() != ""
    ].reset_index(drop=True)

    split_path = (
        args.split_joblib.resolve()
        if args.split_joblib
        else output_dir / "patient_split.joblib"
    )
    train_patients, validation_patients = load_or_create_split(
        records,
        split_path,
        args.validation_fraction,
        args.random_state,
    )

    training = records[
        records["patient_id"].astype(str).isin(train_patients)
    ].copy()
    validation = records[
        records["patient_id"].astype(str).isin(validation_patients)
    ].copy()

    summary = {}
    prediction_frames = []

    for arch in ("upper", "lower"):
        arch_train = training[
            training["arch"] == arch
        ].reset_index(drop=True)
        arch_validation = validation[
            validation["arch"] == arch
        ].reset_index(drop=True)

        print("=" * 72)
        print(f"Train visual fallback: {arch}")
        print("=" * 72)
        print(f"Train arches: {len(arch_train)}")
        print(f"Validation arches: {len(arch_validation)}")
        print(
            "Train routes: "
            f"{arch_train['route'].value_counts().to_dict()}"
        )
        print(f"Device: {device}")

        metrics, predictions = train_one_arch(
            arch,
            arch_train,
            arch_validation,
            output_dir,
            args,
            device,
        )
        summary[arch] = metrics
        prediction_frames.append(predictions)

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Visual fallback training complete")
    print("=" * 72)

    for arch, metrics in summary.items():
        print(
            f"{arch}: "
            f"accuracy={metrics['accuracy_clear']:.4f}, "
            f"balanced={metrics['balanced_accuracy_clear']:.4f}, "
            f"macroF1={metrics['macro_f1_clear']:.4f}"
        )

    print(f"Output: {output_dir}")
    return 0


def predict(args: argparse.Namespace) -> int:
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records = collect_records(
        tensor_dir=args.tensor_dir.resolve(),
        labels_json=None,
        routing_table=args.routing_table.resolve()
        if args.routing_table
        else None,
    )
    model_dir = args.model_dir.resolve()
    output_frames = []

    for arch in ("upper", "lower"):
        arch_frame = records[
            records["arch"] == arch
        ].reset_index(drop=True)

        if len(arch_frame) == 0:
            continue

        dataset = MultiViewTensorDataset(
            arch_frame,
            training=False,
            full_weight=1.0,
            segmentation_weight=1.0,
            visual_weight=1.0,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        checkpoint = torch.load(
            model_dir / f"{arch}_visual_fallback.pt",
            map_location=device,
        )
        model = MultiViewResNet(
            pretrained=False,
            dropout=float(checkpoint["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        predictions, _ = infer_loader(
            model,
            loader,
            device,
        )
        output_frames.append(predictions)

    if not output_frames:
        raise RuntimeError(f"No visual fallback predictions produced from {args.tensor_dir}")
    output = pd.concat(output_frames, ignore_index=True)
    output_path = args.output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Predictions: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--tensor-dir", type=Path, required=True)
    train_parser.add_argument("--labels-json", type=Path, required=True)
    train_parser.add_argument("--routing-table", type=Path)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--split-joblib", type=Path)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.35)
    train_parser.add_argument("--early-stopping-patience", type=int, default=10)
    train_parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    train_parser.add_argument("--full-route-weight", type=float, default=1.0)
    train_parser.add_argument(
        "--segmentation-route-weight",
        type=float,
        default=1.5,
    )
    train_parser.add_argument("--visual-route-weight", type=float, default=3.0)
    train_parser.add_argument("--device", type=str)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--no-amp", dest="amp", action="store_false")
    train_parser.set_defaults(handler=train, amp=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--tensor-dir", type=Path, required=True)
    predict_parser.add_argument("--routing-table", type=Path)
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--output-csv", type=Path, required=True)
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--workers", type=int, default=4)
    predict_parser.add_argument("--device", type=str)
    predict_parser.set_defaults(handler=predict)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
