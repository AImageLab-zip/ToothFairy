#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
不依赖 crowding 几何公式的拥挤度备用分类器。

输入仅来自分割 QC：
- 每颗牙是否出现
- 顶点数、面数、连通分量
- 最大连通块比例
- 牙位三维中心与包围盒
- 分割置信度
- 相邻牙中心距离
- 全牙弓 QC 指标

分别训练 upper / lower CatBoost 五分类器。

命令：
- train
- predict
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
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
UNASSESSABLE = {
    "",
    "not assessable",
    "not mentioned",
    "unknown",
    "none provided",
    "nan",
}

ARCH_CORE_ORDER = {
    "upper": [26, 25, 24, 23, 22, 21, 11, 12, 13, 14, 15, 16],
    "lower": [36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46],
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
        return {"raw": str(value), "allowed": [index], "ambiguous": False}

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


def load_ground_truth(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    output = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        patient_id = str(record.get("patient_id", "")).strip()
        section = (
            record.get("ios_report", {})
            .get("4_alignment_and_arch", {})
        )
        if not patient_id or not isinstance(section, dict):
            continue

        for arch in ("upper", "lower"):
            parsed = parse_label(section.get(f"{arch}_crowding"))
            if parsed is not None:
                output[(patient_id, arch)] = parsed

    return output


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (
        isinstance(value, float)
        and np.isnan(value)
    ):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def list_count(value: Any) -> float:
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return 0.0
    return float(len([item for item in text.split(",") if item.strip()]))


def build_case_features(
    summary_row: pd.Series,
    tooth_rows: pd.DataFrame,
) -> dict[str, Any]:
    case = str(summary_row["case"])
    patient_id = str(summary_row["patient_id"])
    arch = str(summary_row["arch"]).lower()

    features: dict[str, Any] = {
        "case": case,
        "patient_id": patient_id,
        "arch": arch,
        "qc_status": str(summary_row.get("status_v2", "")),
        "qc_selection_score": finite(
            summary_row.get("selection_score_v2")
        ),
        "qc_retry": finite(
            summary_row.get("retry_recommended_v2"), 1.0
        ),
        "qc_geometry_available": finite(
            summary_row.get("geometry_available_v2"), 0.0
        ),
        "qc_hard_failure_count": finite(
            summary_row.get("hard_failure_count"), 0.0
        ),
        "qc_critical_reason_count": finite(
            summary_row.get("critical_reason_count"), 0.0
        ),
        "qc_soft_reason_count": finite(
            summary_row.get("soft_reason_count"), 0.0
        ),
        "qc_mean_core_confidence": finite(
            summary_row.get("mean_core_confidence")
        ),
    }

    for column in (
        "missing_core_teeth",
        "low_vertex_core_teeth",
        "no_face_core_teeth",
        "fragmented_review_teeth",
        "fragmented_severe_teeth",
        "oversized_review_teeth",
        "combined_severe_teeth",
        "low_confidence_core_teeth",
        "missing_second_molars",
        "missing_wisdom_teeth",
    ):
        features[f"qc_{column}_count"] = list_count(
            summary_row.get(column, "")
        )

    current = tooth_rows[tooth_rows["case"].astype(str) == case].copy()
    if len(current) == 0:
        return features

    indexed = {
        int(row["tooth"]): row
        for _, row in current.iterrows()
    }

    present_centroids = []
    for tooth, row in indexed.items():
        if as_bool(row.get("present", False)):
            centroid = np.asarray(
                [
                    finite(row.get("centroid_x")),
                    finite(row.get("centroid_y")),
                    finite(row.get("centroid_z")),
                ],
                dtype=np.float64,
            )
            if np.all(np.isfinite(centroid)):
                present_centroids.append(centroid)

    if present_centroids:
        centroid_array = np.vstack(present_centroids)
        case_center = np.median(centroid_array, axis=0)
        radial = np.linalg.norm(
            centroid_array - case_center,
            axis=1,
        )
        case_scale = max(float(np.quantile(radial, 0.90)), 1.0)
    else:
        case_center = np.zeros(3, dtype=np.float64)
        case_scale = 1.0

    features["seg_present_tooth_count"] = float(
        sum(
            as_bool(row.get("present", False))
            for row in indexed.values()
        )
    )
    features["seg_case_scale_mm"] = case_scale

    for tooth in sorted(indexed):
        row = indexed[tooth]
        prefix = f"tooth_{tooth}"
        present = as_bool(row.get("present", False))
        features[f"{prefix}__present"] = "true" if present else "false"

        numeric_mapping = {
            "vertex_count": "vertex_count",
            "vertex_ratio": "vertex_ratio",
            "complete_face_count": "complete_face_count",
            "component_count": "component_count",
            "largest_component_fraction": "largest_component_fraction",
            "bbox_diag_mm": "bbox_diag_mm",
            "mean_confidence": "mean_confidence",
            "median_confidence": "median_confidence",
            "p10_confidence": "p10_confidence",
        }
        for source, target in numeric_mapping.items():
            features[f"{prefix}__{target}"] = finite(row.get(source))

        centroid = np.asarray(
            [
                finite(row.get("centroid_x")),
                finite(row.get("centroid_y")),
                finite(row.get("centroid_z")),
            ],
            dtype=np.float64,
        )
        if present and np.all(np.isfinite(centroid)):
            normalized = (centroid - case_center) / case_scale
            features[f"{prefix}__cx_norm"] = float(normalized[0])
            features[f"{prefix}__cy_norm"] = float(normalized[1])
            features[f"{prefix}__cz_norm"] = float(normalized[2])
        else:
            features[f"{prefix}__cx_norm"] = np.nan
            features[f"{prefix}__cy_norm"] = np.nan
            features[f"{prefix}__cz_norm"] = np.nan

    core_order = ARCH_CORE_ORDER[arch]
    adjacent_distances = []

    for left, right in zip(core_order[:-1], core_order[1:]):
        left_row = indexed.get(left)
        right_row = indexed.get(right)
        key = f"adjacent_{left}_{right}__distance_norm"

        if (
            left_row is None
            or right_row is None
            or not as_bool(left_row.get("present", False))
            or not as_bool(right_row.get("present", False))
        ):
            features[key] = np.nan
            continue

        left_center = np.asarray(
            [
                finite(left_row.get("centroid_x")),
                finite(left_row.get("centroid_y")),
                finite(left_row.get("centroid_z")),
            ]
        )
        right_center = np.asarray(
            [
                finite(right_row.get("centroid_x")),
                finite(right_row.get("centroid_y")),
                finite(right_row.get("centroid_z")),
            ]
        )

        if not (
            np.all(np.isfinite(left_center))
            and np.all(np.isfinite(right_center))
        ):
            features[key] = np.nan
            continue

        distance = float(
            np.linalg.norm(left_center - right_center) / case_scale
        )
        features[key] = distance
        adjacent_distances.append(distance)

    if adjacent_distances:
        array = np.asarray(adjacent_distances)
        features["adjacent_distance_mean"] = float(array.mean())
        features["adjacent_distance_std"] = float(array.std())
        features["adjacent_distance_min"] = float(array.min())
        features["adjacent_distance_max"] = float(array.max())
    else:
        features["adjacent_distance_mean"] = np.nan
        features["adjacent_distance_std"] = np.nan
        features["adjacent_distance_min"] = np.nan
        features["adjacent_distance_max"] = np.nan

    return features


def collect_rows(
    summary_path: Path,
    tooth_metrics_path: Path,
    labels_json: Optional[Path],
    require_target: bool,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path, dtype={"case": str})
    teeth = pd.read_csv(
        tooth_metrics_path,
        dtype={"case": str, "tooth": int},
    )
    ground_truth = (
        load_ground_truth(labels_json)
        if labels_json is not None
        else {}
    )

    records = []

    for _, summary_row in summary.iterrows():
        features = build_case_features(summary_row, teeth)
        key = (features["patient_id"], features["arch"])
        label = ground_truth.get(key)

        if require_target and label is None:
            continue

        if label is not None:
            features["target_raw"] = label["raw"]
            features["target_allowed"] = ",".join(
                str(index) for index in label["allowed"]
            )
            features["target_ambiguous"] = bool(label["ambiguous"])
        else:
            features["target_raw"] = ""
            features["target_allowed"] = ""
            features["target_ambiguous"] = False

        records.append(features)

    return pd.DataFrame(records)


def infer_schema(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {
        "case",
        "patient_id",
        "arch",
        "target_raw",
        "target_allowed",
        "target_ambiguous",
    }
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]
    categorical = [
        column
        for column in feature_columns
        if dataframe[column].dtype == object
    ]
    return feature_columns, categorical


def prepare_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    matrix = dataframe.reindex(columns=feature_columns).copy()
    categorical_set = set(categorical_columns)

    for column in feature_columns:
        if column in categorical_set:
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


def allowed_indices(text: str) -> list[int]:
    return [
        int(item)
        for item in str(text).split(",")
        if str(item).strip()
    ]


def expand_training_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    labels = []
    weights = []

    for _, row in dataframe.iterrows():
        allowed = allowed_indices(row["target_allowed"])
        if not allowed:
            continue
        weight = 1.0 / len(allowed)

        for target in allowed:
            rows.append(row)
            labels.append(target)
            weights.append(weight)

    return (
        pd.DataFrame(rows).reset_index(drop=True),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def class_weights_from_labels(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    nonzero = counts[counts > 0]
    reference = float(np.max(nonzero)) if len(nonzero) else 1.0
    output = []
    for count in counts:
        if count <= 0:
            output.append(1.0)
        else:
            output.append(float(np.clip(math.sqrt(reference / count), 1.0, 3.0)))
    return output


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def evaluate(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    allowed = [
        allowed_indices(value)
        for value in dataframe["target_allowed"]
    ]
    acceptable = np.asarray(
        [
            int(prediction in target)
            for prediction, target in zip(predictions, allowed)
        ],
        dtype=np.int64,
    )

    clear_mask = np.asarray(
        [len(target) == 1 for target in allowed],
        dtype=bool,
    )
    y_true = np.asarray(
        [target[0] if target else -1 for target in allowed],
        dtype=np.int64,
    )

    if clear_mask.any():
        exact_accuracy = accuracy_score(
            y_true[clear_mask],
            predictions[clear_mask],
        )
        balanced = balanced_accuracy_score(
            y_true[clear_mask],
            predictions[clear_mask],
        )
        macro_f1 = f1_score(
            y_true[clear_mask],
            predictions[clear_mask],
            average="macro",
            labels=np.arange(len(CLASS_NAMES)),
            zero_division=0,
        )
        matrix = confusion_matrix(
            y_true[clear_mask],
            predictions[clear_mask],
            labels=np.arange(len(CLASS_NAMES)),
        ).tolist()
    else:
        exact_accuracy = balanced = macro_f1 = float("nan")
        matrix = []

    return {
        "clear_count": int(clear_mask.sum()),
        "all_count": int(len(dataframe)),
        "accuracy_clear": float(exact_accuracy),
        "balanced_accuracy_clear": float(balanced),
        "macro_f1_clear": float(macro_f1),
        "acceptable_accuracy": float(acceptable.mean()),
        "confusion_matrix": matrix,
    }


def load_or_create_split(
    dataframe: pd.DataFrame,
    split_path: Path,
    validation_fraction: float,
    random_state: int,
) -> tuple[set[str], set[str]]:
    if split_path.is_file():
        saved = joblib.load(split_path)
        if isinstance(saved, dict):
            train = saved.get("train_patients") or saved.get("train")
            val = (
                saved.get("validation_patients")
                or saved.get("val_patients")
                or saved.get("validation")
                or saved.get("val")
            )
            if train is not None and val is not None:
                return set(map(str, train)), set(map(str, val))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    groups = dataframe["patient_id"].astype(str).to_numpy()
    indices = np.arange(len(dataframe))
    train_indices, val_indices = next(
        splitter.split(indices, groups=groups)
    )
    train = set(dataframe.iloc[train_indices]["patient_id"].astype(str))
    val = set(dataframe.iloc[val_indices]["patient_id"].astype(str))

    split_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "train_patients": sorted(train),
            "validation_patients": sorted(val),
        },
        split_path,
    )
    return train, val


def train(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = collect_rows(
        args.selected_qc.resolve(),
        args.selected_tooth_metrics.resolve(),
        args.labels_json.resolve(),
        require_target=True,
    )
    split_path = (
        args.split_joblib.resolve()
        if args.split_joblib
        else output_dir / "patient_split.joblib"
    )
    train_patients, val_patients = load_or_create_split(
        dataframe,
        split_path,
        args.validation_fraction,
        args.random_state,
    )

    train_df = dataframe[
        dataframe["patient_id"].astype(str).isin(train_patients)
    ].copy()
    val_df = dataframe[
        dataframe["patient_id"].astype(str).isin(val_patients)
    ].copy()

    feature_columns, categorical_columns = infer_schema(train_df)
    schema = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validation_rows = []
    validation_summary = {}

    for arch in ("upper", "lower"):
        arch_train = train_df[train_df["arch"] == arch].reset_index(drop=True)
        arch_val = val_df[val_df["arch"] == arch].reset_index(drop=True)

        expanded, y_train, sample_weight = expand_training_rows(arch_train)
        x_train = prepare_matrix(
            expanded,
            feature_columns,
            categorical_columns,
        )
        x_val = prepare_matrix(
            arch_val,
            feature_columns,
            categorical_columns,
        )

        clear_val_mask = (
            arch_val["target_allowed"]
            .astype(str)
            .map(lambda value: len(allowed_indices(value)) == 1)
            .to_numpy()
        )
        if clear_val_mask.any():
            x_eval = x_val.loc[clear_val_mask]
            y_eval = np.asarray(
                [
                    allowed_indices(value)[0]
                    for value in arch_val.loc[
                        clear_val_mask,
                        "target_allowed",
                    ]
                ],
                dtype=np.int64,
            )
            eval_set = (x_eval, y_eval)
        else:
            eval_set = None

        weights = class_weights_from_labels(y_train)
        parameters = {
            "loss_function": "MultiClass",
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "random_seed": args.random_state,
            "class_weights": weights,
            "verbose": 50,
            "allow_writing_files": False,
            "task_type": args.task_type,
            "od_type": "Iter",
            "od_wait": args.early_stopping_rounds,
        }
        if args.task_type == "GPU":
            parameters["devices"] = args.devices

        model = CatBoostClassifier(**parameters)
        cat_indices = [
            feature_columns.index(column)
            for column in categorical_columns
        ]

        fit_kwargs = {
            "X": x_train,
            "y": y_train,
            "sample_weight": sample_weight,
            "cat_features": cat_indices,
        }
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True

        print("=" * 72)
        print(f"训练 {arch} segmentation fallback")
        print("=" * 72)
        print(f"训练牙弓: {len(arch_train)}")
        print(f"展开行: {len(expanded)}")
        print(f"验证牙弓: {len(arch_val)}")
        print(f"特征数: {len(feature_columns)}")
        print(f"class_weights: {weights}")

        model.fit(**fit_kwargs)
        model.save_model(str(output_dir / f"{arch}_model.cbm"))

        probabilities = model.predict_proba(x_val)
        metrics = evaluate(arch_val, probabilities)
        validation_summary[arch] = metrics

        predictions = probabilities.argmax(axis=1)
        for index, (_, row) in enumerate(arch_val.iterrows()):
            output = {
                "case": row["case"],
                "patient_id": row["patient_id"],
                "arch": arch,
                "target_raw": row["target_raw"],
                "target_allowed": row["target_allowed"],
                "prediction": CLASS_NAMES[int(predictions[index])],
                "prediction_index": int(predictions[index]),
                "model_confidence": float(probabilities[index].max()),
                "prediction_entropy": entropy(probabilities[index]),
                "qc_status": row.get("qc_status", ""),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                output[f"prob_{class_name.lower()}"] = float(
                    probabilities[index, class_index]
                )
            validation_rows.append(output)

        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(
            output_dir / f"{arch}_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(validation_rows).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Segmentation fallback training complete")
    print("=" * 72)
    for arch, metrics in validation_summary.items():
        print(
            f"{arch}: accuracy={metrics['accuracy_clear']:.4f}, "
            f"balanced={metrics['balanced_accuracy_clear']:.4f}, "
            f"macroF1={metrics['macro_f1_clear']:.4f}, "
            f"acceptable={metrics['acceptable_accuracy']:.4f}"
        )
    print(f"Output: {output_dir}")
    return 0


def predict(args: argparse.Namespace) -> int:
    dataframe = collect_rows(
        args.selected_qc.resolve(),
        args.selected_tooth_metrics.resolve(),
        labels_json=None,
        require_target=False,
    )
    model_dir = args.model_dir.resolve()
    schema = json.loads(
        (model_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    feature_columns = schema["feature_columns"]
    categorical_columns = schema["categorical_columns"]

    rows = []

    for arch in ("upper", "lower"):
        arch_df = dataframe[dataframe["arch"] == arch].reset_index(drop=True)
        if len(arch_df) == 0:
            continue

        matrix = prepare_matrix(
            arch_df,
            feature_columns,
            categorical_columns,
        )
        model = CatBoostClassifier()
        model.load_model(str(model_dir / f"{arch}_model.cbm"))
        probabilities = model.predict_proba(matrix)
        predictions = probabilities.argmax(axis=1)

        for index, (_, row) in enumerate(arch_df.iterrows()):
            output = {
                "case": row["case"],
                "patient_id": row["patient_id"],
                "arch": arch,
                "prediction": CLASS_NAMES[int(predictions[index])],
                "prediction_index": int(predictions[index]),
                "model_confidence": float(probabilities[index].max()),
                "prediction_entropy": entropy(probabilities[index]),
                "qc_status": row.get("qc_status", ""),
                "qc_selection_score": row.get("qc_selection_score", np.nan),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                output[f"prob_{class_name.lower()}"] = float(
                    probabilities[index, class_index]
                )
            rows.append(output)

    output_path = args.output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
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
    train_parser.add_argument("--selected-qc", type=Path, required=True)
    train_parser.add_argument(
        "--selected-tooth-metrics",
        type=Path,
        required=True,
    )
    train_parser.add_argument("--labels-json", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--split-joblib", type=Path)
    train_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--iterations", type=int, default=1800)
    train_parser.add_argument("--depth", type=int, default=6)
    train_parser.add_argument("--learning-rate", type=float, default=0.03)
    train_parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
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
    train_parser.add_argument("--devices", type=str, default="0")
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--selected-qc", type=Path, required=True)
    predict_parser.add_argument(
        "--selected-tooth-metrics",
        type=Path,
        required=True,
    )
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--output-csv", type=Path, required=True)
    predict_parser.set_defaults(handler=predict)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
