#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拥挤度分类 v2：几何结果 + 牙位 mask 结构特征 + CatBoost
====================================================

为什么不用上一版端到端 EfficientNet：
- 当前日志中最佳验证 Macro-F1 约 0.31，低于简单多数类基线；
- 解冻图像骨干后训练损失下降，但验证指标继续恶化；
- 对约 1900 个牙弓样本，继续微调整个图像网络很容易过拟合；
- 拥挤度本质更接近“结构化几何判断”，应优先使用牙位、牙宽、
  牙弓长度、拟合残差和质量信息。

本脚本读取：
1. work/task_evaluation/case_results/<patient>/<arch>_crowding.json
2. work/rendered_ios_occlusal_mask/Fxxxx_ios_<arch>_label.png
3. cleaned_orthodontic_data.json

自动提取：
- crowding 计算 JSON 中的数值、布尔和短字符串特征；
- 每颗牙在 mask 中的面积、重心、包围盒、方向、离心率；
- 全局牙弓 mask 特征和分割质量；
- quality/status 等类别特征。

分别训练 upper 和 lower 两个 CatBoost 多分类器。

输出：
- upper_model.cbm / lower_model.cbm
- feature_schema.json
- validation_predictions.csv
- validation_summary.json
- predictions.csv（predict 模式）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


CLASS_NAMES = ["Spacing", "None", "Mild", "Moderate", "Severe"]
CLASS_TO_INDEX = {
    name.lower(): index
    for index, name in enumerate(CLASS_NAMES)
}
NUM_CLASSES = len(CLASS_NAMES)

UNASSESSABLE = {
    "",
    "not assessable",
    "not mentioned",
    "unknown",
    "none provided",
    "nan",
}

IGNORE_JSON_KEY_PARTS = {
    "path",
    "file",
    "vertex_indices",
    "vertices",
    "faces",
    "points_xyz",
    "curve_points",
    "visualization",
}


# ============================================================
# 标签
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("\\", "/")
    text = text.replace("-", "/")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" / ", "/")
    return text


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
        indices = sorted(
            [CLASS_TO_INDEX[parts[0]], CLASS_TO_INDEX[parts[1]]]
        )
        if indices[1] - indices[0] == 1:
            return {
                "raw": str(value),
                "allowed": indices,
                "ambiguous": True,
            }

    return None


def load_ground_truth(
    path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("标准 JSON 顶层必须是病例数组")

    output = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        patient_id = str(
            record.get("patient_id", "")
        ).strip()

        if not patient_id:
            continue

        ios_report = record.get("ios_report", {})
        if not isinstance(ios_report, dict):
            continue

        section = ios_report.get(
            "4_alignment_and_arch",
            {},
        )
        if not isinstance(section, dict):
            continue

        for arch in ("upper", "lower"):
            parsed = parse_label(
                section.get(f"{arch}_crowding")
            )
            if parsed is not None:
                output[(patient_id, arch)] = parsed

    return output


# ============================================================
# JSON 特征
# ============================================================

def should_ignore_key(key_path: str) -> bool:
    lower = key_path.lower()

    return any(
        part in lower
        for part in IGNORE_JSON_KEY_PARTS
    )


def flatten_json_features(
    value: Any,
    prefix: str = "geom",
    output: Optional[dict[str, Any]] = None,
    max_list_items: int = 64,
) -> dict[str, Any]:
    """
    将 crowding case JSON 展平成 CatBoost 可用特征。

    - 数值保留为 float；
    - bool 保留为字符串类别；
    - 短字符串保留为类别；
    - 长数组最多展开前 max_list_items 项；
    - 路径、顶点索引和大型坐标数组跳过。
    """
    if output is None:
        output = {}

    if should_ignore_key(prefix):
        return output

    if isinstance(value, dict):
        for key, child in value.items():
            key_text = re.sub(
                r"[^a-zA-Z0-9_]+",
                "_",
                str(key),
            )
            flatten_json_features(
                child,
                prefix=f"{prefix}__{key_text}",
                output=output,
                max_list_items=max_list_items,
            )

    elif isinstance(value, list):
        if len(value) <= max_list_items:
            for index, child in enumerate(value):
                flatten_json_features(
                    child,
                    prefix=f"{prefix}__{index}",
                    output=output,
                    max_list_items=max_list_items,
                )
        else:
            numeric = [
                float(item)
                for item in value
                if isinstance(item, (int, float))
                and np.isfinite(item)
            ]
            if numeric:
                array = np.asarray(
                    numeric,
                    dtype=np.float64,
                )
                output[f"{prefix}__count"] = float(
                    len(array)
                )
                output[f"{prefix}__mean"] = float(
                    array.mean()
                )
                output[f"{prefix}__std"] = float(
                    array.std()
                )
                output[f"{prefix}__min"] = float(
                    array.min()
                )
                output[f"{prefix}__max"] = float(
                    array.max()
                )

    elif isinstance(value, bool):
        output[prefix] = (
            "true" if value else "false"
        )

    elif isinstance(value, (int, float)):
        numeric = float(value)
        output[prefix] = (
            numeric
            if np.isfinite(numeric)
            else np.nan
        )

    elif isinstance(value, str):
        text = value.strip()

        if (
            text
            and len(text) <= 100
            and not should_ignore_key(prefix)
        ):
            output[prefix] = text

    return output


def read_geometry_json(
    path: Path,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    features = flatten_json_features(data)

    features["geom_json_present"] = "true"
    features["geom_json_size_kb"] = (
        path.stat().st_size / 1024.0
    )

    return features


# ============================================================
# Mask 特征
# ============================================================

def safe_moments(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[float, float, float]:
    """
    返回：
        sin(2θ), cos(2θ), eccentricity
    """
    if len(xs) < 3:
        return 0.0, 1.0, 0.0

    coords = np.column_stack(
        [xs, ys]
    ).astype(np.float64)

    coords -= coords.mean(axis=0)

    covariance = np.cov(
        coords,
        rowvar=False,
    )

    if (
        covariance.shape != (2, 2)
        or np.any(~np.isfinite(covariance))
    ):
        return 0.0, 1.0, 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(
        eigenvalues[order],
        1e-8,
    )
    principal = eigenvectors[:, order[0]]

    theta = math.atan2(
        principal[1],
        principal[0],
    )

    eccentricity = math.sqrt(
        max(
            0.0,
            1.0
            - eigenvalues[1] / eigenvalues[0],
        )
    )

    return (
        math.sin(2.0 * theta),
        math.cos(2.0 * theta),
        eccentricity,
    )


def extract_mask_features(
    mask_path: Path,
) -> dict[str, Any]:
    mask = np.asarray(
        Image.open(mask_path).convert("L"),
        dtype=np.uint8,
    )

    if mask.ndim != 2:
        raise ValueError(
            f"mask 必须二维，当前 {mask.shape}"
        )

    if mask.min() < 0 or mask.max() > 16:
        raise ValueError(
            f"mask 像素必须在 0~16，当前 "
            f"{mask.min()}~{mask.max()}"
        )

    height, width = mask.shape
    total = float(mask.size)

    features: dict[str, Any] = {
        "mask_height": float(height),
        "mask_width": float(width),
    }

    counts = np.bincount(
        mask.reshape(-1),
        minlength=17,
    )

    tooth_mask = mask > 0
    tooth_count = int(tooth_mask.sum())

    features["mask_tooth_area_ratio"] = (
        tooth_count / total
    )

    valid_label_count = 0
    tiny_label_count = 0
    present_centroids = []

    # 1024 图下约 100 像素以上认为是有效牙位。
    size_scale = total / (1024.0 * 1024.0)
    valid_threshold = max(
        10,
        int(round(100 * size_scale)),
    )
    tiny_threshold = max(
        3,
        int(round(30 * size_scale)),
    )

    for label in range(1, 17):
        ys, xs = np.where(
            mask == label
        )
        count = len(xs)
        area_ratio = count / total

        prefix = f"mask_tooth_{label:02d}"

        features[f"{prefix}__present"] = (
            "true" if count > 0 else "false"
        )
        features[f"{prefix}__area_ratio"] = (
            float(area_ratio)
        )

        if count == 0:
            features[f"{prefix}__cx"] = np.nan
            features[f"{prefix}__cy"] = np.nan
            features[f"{prefix}__bbox_w"] = np.nan
            features[f"{prefix}__bbox_h"] = np.nan
            features[f"{prefix}__sin2theta"] = np.nan
            features[f"{prefix}__cos2theta"] = np.nan
            features[f"{prefix}__eccentricity"] = np.nan
            continue

        cx = float(xs.mean() / max(width - 1, 1))
        cy = float(ys.mean() / max(height - 1, 1))

        bbox_w = float(
            (xs.max() - xs.min() + 1) / width
        )
        bbox_h = float(
            (ys.max() - ys.min() + 1) / height
        )

        sin2theta, cos2theta, eccentricity = (
            safe_moments(xs, ys)
        )

        features[f"{prefix}__cx"] = cx
        features[f"{prefix}__cy"] = cy
        features[f"{prefix}__bbox_w"] = bbox_w
        features[f"{prefix}__bbox_h"] = bbox_h
        features[f"{prefix}__sin2theta"] = sin2theta
        features[f"{prefix}__cos2theta"] = cos2theta
        features[f"{prefix}__eccentricity"] = eccentricity

        if count >= valid_threshold:
            valid_label_count += 1
            present_centroids.append(
                (label, cx, cy)
            )

        if 0 < count < tiny_threshold:
            tiny_label_count += 1

    features["mask_valid_label_count"] = float(
        valid_label_count
    )
    features["mask_tiny_label_count"] = float(
        tiny_label_count
    )

    if tooth_count > 0:
        ys, xs = np.where(tooth_mask)

        features["mask_arch_bbox_w"] = float(
            (xs.max() - xs.min() + 1) / width
        )
        features["mask_arch_bbox_h"] = float(
            (ys.max() - ys.min() + 1) / height
        )
        features["mask_arch_cx"] = float(
            xs.mean() / max(width - 1, 1)
        )
        features["mask_arch_cy"] = float(
            ys.mean() / max(height - 1, 1)
        )

        sin2theta, cos2theta, eccentricity = (
            safe_moments(xs, ys)
        )

        features["mask_arch_sin2theta"] = sin2theta
        features["mask_arch_cos2theta"] = cos2theta
        features["mask_arch_eccentricity"] = eccentricity
    else:
        for key in (
            "mask_arch_bbox_w",
            "mask_arch_bbox_h",
            "mask_arch_cx",
            "mask_arch_cy",
            "mask_arch_sin2theta",
            "mask_arch_cos2theta",
            "mask_arch_eccentricity",
        ):
            features[key] = np.nan

    # 简单的牙位中心间距离统计。
    if len(present_centroids) >= 2:
        centroid_array = np.asarray(
            [
                [cx, cy]
                for _, cx, cy in present_centroids
            ],
            dtype=np.float64,
        )

        distances = np.linalg.norm(
            centroid_array[:, None, :]
            - centroid_array[None, :, :],
            axis=2,
        )
        distances[
            distances == 0
        ] = np.nan

        nearest = np.nanmin(
            distances,
            axis=1,
        )

        features[
            "mask_nearest_centroid_mean"
        ] = float(
            np.nanmean(nearest)
        )
        features[
            "mask_nearest_centroid_min"
        ] = float(
            np.nanmin(nearest)
        )
        features[
            "mask_nearest_centroid_max"
        ] = float(
            np.nanmax(nearest)
        )
    else:
        features["mask_nearest_centroid_mean"] = np.nan
        features["mask_nearest_centroid_min"] = np.nan
        features["mask_nearest_centroid_max"] = np.nan

    # 质量分数仅作为特征和重试依据，不直接决定类别。
    score = 0.0
    score += np.clip(
        (valid_label_count - 5) / 9.0,
        0.0,
        1.0,
    ) * 0.55

    if 0.025 <= tooth_count / total <= 0.45:
        score += 0.25
    elif 0.015 <= tooth_count / total <= 0.60:
        score += 0.12

    if tiny_label_count == 0:
        score += 0.20
    elif tiny_label_count <= 2:
        score += 0.10

    features["mask_quality"] = float(
        np.clip(score, 0.0, 1.0)
    )

    return features


# ============================================================
# 数据构建
# ============================================================

def collect_feature_rows(
    case_results_dir: Path,
    mask_dir: Path,
    labels_json: Optional[Path],
    require_ground_truth: bool,
) -> pd.DataFrame:
    ground_truth = (
        load_ground_truth(labels_json)
        if labels_json is not None
        else {}
    )

    records = []

    patient_dirs = sorted(
        [
            path
            for path in case_results_dir.iterdir()
            if path.is_dir()
        ]
    )

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        for arch in ("upper", "lower"):
            geometry_path = (
                patient_dir
                / f"{arch}_crowding.json"
            )
            mask_path = (
                mask_dir
                / f"{patient_id}_ios_{arch}_label.png"
            )

            if (
                not geometry_path.is_file()
                or not mask_path.is_file()
            ):
                continue

            label_info = ground_truth.get(
                (patient_id, arch)
            )

            if require_ground_truth and label_info is None:
                continue

            try:
                features = {
                    "patient_id": patient_id,
                    "arch": arch,
                }
                features.update(
                    read_geometry_json(
                        geometry_path
                    )
                )
                features.update(
                    extract_mask_features(
                        mask_path
                    )
                )

                features["geometry_json_path"] = str(
                    geometry_path
                )
                features["mask_path"] = str(
                    mask_path
                )

                if label_info is not None:
                    features["target_raw"] = (
                        label_info["raw"]
                    )
                    features["target_allowed"] = ",".join(
                        str(index)
                        for index
                        in label_info["allowed"]
                    )
                    features["target_ambiguous"] = bool(
                        label_info["ambiguous"]
                    )
                else:
                    features["target_raw"] = ""
                    features["target_allowed"] = ""
                    features["target_ambiguous"] = False

                records.append(features)

            except Exception as exc:
                print(
                    f"⚠️ 跳过 {patient_id} {arch}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    return pd.DataFrame(records)


def prepare_feature_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    matrix = dataframe.reindex(
        columns=feature_columns
    ).copy()

    for column in feature_columns:
        if column in categorical_columns:
            matrix[column] = (
                matrix[column]
                .fillna("__MISSING__")
                .astype(str)
            )
        else:
            matrix[column] = pd.to_numeric(
                matrix[column],
                errors="coerce",
            )

    return matrix


def infer_feature_schema(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    excluded = {
        "patient_id",
        "arch",
        "target_raw",
        "target_allowed",
        "target_ambiguous",
        "geometry_json_path",
        "mask_path",
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    categorical_columns = []

    for column in feature_columns:
        series = dataframe[column]

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical_columns.append(
                column
            )

    return feature_columns, categorical_columns


def expand_training_targets(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    复合标签例如 Mild/Moderate：
    复制为两行，每行权重 0.5。
    """
    rows = []
    targets = []
    weights = []

    for _, row in dataframe.iterrows():
        allowed = [
            int(value)
            for value
            in str(row["target_allowed"]).split(",")
            if str(value).strip()
        ]

        if not allowed:
            continue

        weight = 1.0 / len(allowed)

        for target in allowed:
            rows.append(row.copy())
            targets.append(target)
            weights.append(weight)

    return (
        pd.DataFrame(rows).reset_index(drop=True),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def sqrt_class_weights(
    targets: np.ndarray,
    sample_weights: np.ndarray,
) -> list[float]:
    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for target, weight in zip(
        targets,
        sample_weights,
    ):
        counts[target] += weight

    counts = np.maximum(
        counts,
        1.0,
    )

    maximum = counts.max()

    weights = np.sqrt(
        maximum / counts
    )

    weights = np.clip(
        weights,
        1.0,
        3.0,
    )

    return weights.tolist()


# ============================================================
# 评估
# ============================================================

def evaluate_predictions(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(
        axis=1
    )

    allowed_lists = [
        [
            int(value)
            for value
            in str(text).split(",")
            if str(value).strip()
        ]
        for text in dataframe[
            "target_allowed"
        ]
    ]

    acceptable_correct = [
        int(prediction in allowed)
        for prediction, allowed
        in zip(predictions, allowed_lists)
    ]

    clear_indices = [
        index
        for index, allowed
        in enumerate(allowed_lists)
        if len(allowed) == 1
    ]

    if clear_indices:
        y_true = np.asarray(
            [
                allowed_lists[index][0]
                for index in clear_indices
            ],
            dtype=np.int64,
        )
        y_pred = predictions[
            clear_indices
        ]

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )
        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        confusion = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        )
    else:
        accuracy = np.nan
        balanced_accuracy = np.nan
        macro_f1 = np.nan
        confusion = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )

    return {
        "samples": int(len(dataframe)),
        "clear_samples": int(len(clear_indices)),
        "accuracy_clear": float(accuracy),
        "balanced_accuracy_clear": float(
            balanced_accuracy
        ),
        "macro_f1_clear": float(macro_f1),
        "acceptable_accuracy_all": float(
            np.mean(acceptable_correct)
        ),
        "confusion_matrix_clear": (
            confusion.tolist()
        ),
        "prediction_distribution": {
            CLASS_NAMES[index]: int(
                np.sum(predictions == index)
            )
            for index in range(NUM_CLASSES)
        },
    }


# ============================================================
# Train
# ============================================================

def train_command(
    args: argparse.Namespace,
) -> None:
    case_results_dir = Path(
        args.case_results_dir
    ).resolve()
    mask_dir = Path(
        args.mask_dir
    ).resolve()
    labels_json = Path(
        args.labels_json
    ).resolve()
    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = collect_feature_rows(
        case_results_dir=case_results_dir,
        mask_dir=mask_dir,
        labels_json=labels_json,
        require_ground_truth=True,
    )

    if len(dataframe) < 100:
        raise RuntimeError(
            f"只收集到 {len(dataframe)} 个样本，"
            "请检查 case_results 和 mask 路径"
        )

    # 只过滤极端坏 mask；review 病例保留，并让模型看到质量特征。
    dataframe = dataframe[
        dataframe["mask_quality"]
        >= args.minimum_mask_quality
    ].reset_index(drop=True)

    patients = dataframe[
        "patient_id"
    ].astype(str).values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_ratio,
        random_state=args.seed,
    )

    train_indices, validation_indices = next(
        splitter.split(
            np.arange(len(dataframe)),
            groups=patients,
        )
    )

    train_df = dataframe.iloc[
        train_indices
    ].reset_index(drop=True)
    validation_df = dataframe.iloc[
        validation_indices
    ].reset_index(drop=True)

    feature_columns, categorical_columns = (
        infer_feature_schema(
            train_df
        )
    )

    schema = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "class_names": CLASS_NAMES,
        "minimum_mask_quality": (
            args.minimum_mask_quality
        ),
    }

    with (
        output_dir
        / "feature_schema.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            schema,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    validation_rows = []
    summary = {}

    for arch in ("upper", "lower"):
        arch_train = train_df[
            train_df["arch"] == arch
        ].reset_index(drop=True)
        arch_validation = validation_df[
            validation_df["arch"] == arch
        ].reset_index(drop=True)

        expanded_train, y_train, sample_weights = (
            expand_training_targets(
                arch_train
            )
        )

        x_train = prepare_feature_matrix(
            expanded_train,
            feature_columns,
            categorical_columns,
        )
        x_validation = prepare_feature_matrix(
            arch_validation,
            feature_columns,
            categorical_columns,
        )

        cat_feature_indices = [
            feature_columns.index(column)
            for column in categorical_columns
        ]

        class_weights = sqrt_class_weights(
            y_train,
            sample_weights,
        )

        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            class_weights=class_weights,
            l2_leaf_reg=args.l2_leaf_reg,
            random_strength=args.random_strength,
            random_seed=args.seed,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
            allow_writing_files=False,
        )

        print()
        print("=" * 72)
        print(f"训练 {arch} CatBoost")
        print("=" * 72)
        print(f"训练牙弓: {len(arch_train)}")
        print(f"展开后训练行: {len(expanded_train)}")
        print(f"验证牙弓: {len(arch_validation)}")
        print(f"特征数量: {len(feature_columns)}")
        print(f"类别特征: {len(categorical_columns)}")
        print(f"class_weights: {class_weights}")

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            cat_features=cat_feature_indices,
            eval_set=(
                x_validation,
                np.asarray(
                    [
                        int(
                            str(value).split(",")[0]
                        )
                        for value in arch_validation[
                            "target_allowed"
                        ]
                    ],
                    dtype=np.int64,
                ),
            ),
            early_stopping_rounds=(
                args.early_stopping_rounds
            ),
            use_best_model=True,
        )

        model_path = (
            output_dir
            / f"{arch}_model.cbm"
        )
        model.save_model(
            str(model_path)
        )

        probabilities = model.predict_proba(
            x_validation
        )

        metrics = evaluate_predictions(
            arch_validation,
            probabilities,
        )
        summary[arch] = metrics

        predictions = probabilities.argmax(
            axis=1
        )

        for row_index, (
            _,
            row,
        ) in enumerate(
            arch_validation.iterrows()
        ):
            output_row = {
                "patient_id": row["patient_id"],
                "arch": arch,
                "ground_truth_raw": row[
                    "target_raw"
                ],
                "ground_truth_allowed": row[
                    "target_allowed"
                ],
                "prediction": CLASS_NAMES[
                    int(predictions[row_index])
                ],
                "prediction_index": int(
                    predictions[row_index]
                ),
                "confidence": float(
                    probabilities[
                        row_index
                    ].max()
                ),
                "mask_quality": float(
                    row["mask_quality"]
                ),
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                output_row[
                    f"prob_{class_name.lower()}"
                ] = float(
                    probabilities[
                        row_index,
                        class_index,
                    ]
                )

            validation_rows.append(
                output_row
            )

        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": (
                    model.get_feature_importance()
                ),
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        importance.to_csv(
            output_dir
            / f"{arch}_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(
        validation_rows
    ).to_csv(
        output_dir
        / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        output_dir
        / "validation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        {
            "train_patients": sorted(
                set(
                    train_df[
                        "patient_id"
                    ].astype(str)
                )
            ),
            "validation_patients": sorted(
                set(
                    validation_df[
                        "patient_id"
                    ].astype(str)
                )
            ),
        },
        output_dir
        / "patient_split.joblib",
    )

    print()
    print("=" * 72)
    print("训练完成")
    print("=" * 72)

    for arch in ("upper", "lower"):
        metrics = summary[arch]
        print(
            f"{arch}: "
            f"accuracy={metrics['accuracy_clear']:.4f}, "
            f"balanced={metrics['balanced_accuracy_clear']:.4f}, "
            f"macroF1={metrics['macro_f1_clear']:.4f}, "
            f"acceptable={metrics['acceptable_accuracy_all']:.4f}"
        )

    print(
        f"输出目录: {output_dir}"
    )


# ============================================================
# Predict
# ============================================================

def probability_entropy(
    probabilities: np.ndarray,
) -> float:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    return float(
        -np.sum(
            probabilities
            * np.log(
                np.clip(
                    probabilities,
                    1e-12,
                    1.0,
                )
            )
        )
    )


def find_geometry_outlier(
    row: pd.Series,
) -> list[str]:
    reasons = []

    for column, value in row.items():
        if (
            "crowding_mm" in str(column).lower()
            and isinstance(value, (int, float, np.number))
            and np.isfinite(value)
            and abs(float(value)) > 30.0
        ):
            reasons.append(
                f"extreme_{column}"
            )

    return reasons


def predict_command(
    args: argparse.Namespace,
) -> None:
    case_results_dir = Path(
        args.case_results_dir
    ).resolve()
    mask_dir = Path(
        args.mask_dir
    ).resolve()
    model_dir = Path(
        args.model_dir
    ).resolve()
    output_csv = Path(
        args.output_csv
    ).resolve()

    with (
        model_dir
        / "feature_schema.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    feature_columns = schema[
        "feature_columns"
    ]
    categorical_columns = schema[
        "categorical_columns"
    ]

    if not case_results_dir.is_dir():
        raise FileNotFoundError(
            f"几何结果目录不存在：{case_results_dir}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"mask目录不存在：{mask_dir}\n"
            "请先从选中分割结果重新生成mask，例如：\n"
            "python render_occlusal_label_masks_compat.py "
            "--segmentation-dir work/output2_selected "
            "--output-dir work/rendered_ios_occlusal_mask_selected "
            "--overwrite"
        )

    geometry_files = sorted(
        case_results_dir.glob("*/*_crowding.json")
    )
    mask_files = sorted(
        mask_dir.glob("*_label.png")
    )

    dataframe = collect_feature_rows(
        case_results_dir=case_results_dir,
        mask_dir=mask_dir,
        labels_json=None,
        require_ground_truth=False,
    )

    if dataframe.empty or "arch" not in dataframe.columns:
        geometry_examples = [
            str(path.relative_to(case_results_dir))
            for path in geometry_files[:3]
        ]
        mask_examples = [
            path.name
            for path in mask_files[:3]
        ]

        raise RuntimeError(
            "没有找到任何可用于Full CatBoost推理的“几何JSON + mask”配对。\n"
            f"几何JSON数量：{len(geometry_files)}\n"
            f"mask数量：{len(mask_files)}\n"
            "脚本要求同一病例同时存在：\n"
            "  case_results/<patient>/upper_crowding.json\n"
            "  rendered_masks/<patient>_ios_upper_label.png\n"
            "下颌同理。\n"
            f"几何文件示例：{geometry_examples}\n"
            f"mask文件示例：{mask_examples}\n"
            "最常见原因是 selected mask 尚未生成，或mask文件名不符合"
            " Fxxxx_ios_upper_label.png / Fxxxx_ios_lower_label.png。"
        )

    print(
        "Full CatBoost输入配对："
        f"{len(dataframe)} 个牙弓；"
        f"几何JSON={len(geometry_files)}，"
        f"mask={len(mask_files)}"
    )

    rows = []

    for arch in ("upper", "lower"):
        arch_df = dataframe[
            dataframe["arch"] == arch
        ].reset_index(drop=True)

        if len(arch_df) == 0:
            continue

        matrix = prepare_feature_matrix(
            arch_df,
            feature_columns,
            categorical_columns,
        )

        model = CatBoostClassifier()
        model.load_model(
            str(
                model_dir
                / f"{arch}_model.cbm"
            )
        )

        probabilities = model.predict_proba(
            matrix
        )
        predictions = probabilities.argmax(
            axis=1
        )

        for index, (
            _,
            source_row,
        ) in enumerate(
            arch_df.iterrows()
        ):
            confidence = float(
                probabilities[index].max()
            )
            entropy = probability_entropy(
                probabilities[index]
            )
            mask_quality = float(
                source_row.get(
                    "mask_quality",
                    0.0,
                )
            )

            retry_reasons = []

            if mask_quality < args.retry_mask_quality:
                retry_reasons.append(
                    "low_mask_quality"
                )
            if confidence < args.retry_confidence:
                retry_reasons.append(
                    "low_model_confidence"
                )
            if entropy > args.retry_entropy:
                retry_reasons.append(
                    "high_prediction_entropy"
                )

            retry_reasons.extend(
                find_geometry_outlier(
                    source_row
                )
            )

            output_row = {
                "patient_id": source_row[
                    "patient_id"
                ],
                "arch": arch,
                "prediction": CLASS_NAMES[
                    int(predictions[index])
                ],
                "prediction_index": int(
                    predictions[index]
                ),
                "model_confidence": confidence,
                "prediction_entropy": entropy,
                "mask_quality": mask_quality,
                "recommend_retry_segmentation": int(
                    len(retry_reasons) > 0
                ),
                "retry_reasons": ",".join(
                    retry_reasons
                ),
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                output_row[
                    f"prob_{class_name.lower()}"
                ] = float(
                    probabilities[
                        index,
                        class_index,
                    ]
                )

            rows.append(
                output_row
            )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"推理完成：{output_csv}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "几何 JSON + 牙位 mask 特征的 "
            "CatBoost 拥挤度分类"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train"
    )
    train_parser.add_argument(
        "--case-results-dir",
        default="work/task_evaluation/case_results",
    )
    train_parser.add_argument(
        "--mask-dir",
        default="work/rendered_ios_occlusal_mask",
    )
    train_parser.add_argument(
        "--labels-json",
        default=(
            "data/"
            "cleaned_orthodontic_data.json"
        ),
    )
    train_parser.add_argument(
        "--output-dir",
        default="work/crowding_catboost_v2",
    )
    train_parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )
    train_parser.add_argument(
        "--minimum-mask-quality",
        type=float,
        default=0.15,
        help=(
            "仅过滤极端乱码；一般 review 样本仍保留"
        ),
    )
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=1600,
    )
    train_parser.add_argument(
        "--depth",
        type=int,
        default=6,
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )
    train_parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=8.0,
    )
    train_parser.add_argument(
        "--random-strength",
        type=float,
        default=0.5,
    )
    train_parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=150,
    )
    train_parser.add_argument(
        "--task-type",
        choices=("CPU", "GPU"),
        default="CPU",
    )
    train_parser.add_argument(
        "--devices",
        default="0",
    )
    train_parser.add_argument(
        "--verbose",
        type=int,
        default=50,
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    predict_parser = subparsers.add_parser(
        "predict"
    )
    predict_parser.add_argument(
        "--case-results-dir",
        default="work/task_evaluation/case_results",
    )
    predict_parser.add_argument(
        "--mask-dir",
        default="work/rendered_ios_occlusal_mask",
    )
    predict_parser.add_argument(
        "--model-dir",
        default="work/crowding_catboost_v2",
    )
    predict_parser.add_argument(
        "--output-csv",
        default=(
            "work/crowding_catboost_v2/"
            "predictions.csv"
        ),
    )
    predict_parser.add_argument(
        "--retry-mask-quality",
        type=float,
        default=0.30,
    )
    predict_parser.add_argument(
        "--retry-confidence",
        type=float,
        default=0.45,
    )
    predict_parser.add_argument(
        "--retry-entropy",
        type=float,
        default=1.35,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"致命错误：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1)