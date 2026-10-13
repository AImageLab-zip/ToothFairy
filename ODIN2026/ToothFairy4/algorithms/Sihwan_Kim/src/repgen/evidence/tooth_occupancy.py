"""Dependency-free R218H tooth-slot occupancy and exact-absence guard."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODEL_PATH = Path("/opt/ml/model/r233_tooth_slot_model.json")
RAW_PRESENT_MIN_VOXELS = 100
LOCAL_HALF_WIDTH = 24
MIN_JAWBONE_VOXELS = 128
THIRD_MOLARS = {"18", "28", "38", "48"}
BASE_RELATIONS_PER_CASE = 2
MAXIMUM_PER_CASE = 3
EXTRA_RAW_PRESENCE_THRESHOLD = 0.005


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "r233_r218_histgb_json_v1":
        raise ValueError("unsupported R233 tooth-slot model schema")
    return payload


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def predict_present_probability(
    features: Sequence[float], payload: Mapping[str, Any]
) -> float:
    raw = float(payload["baseline_prediction"])
    for nodes in payload["trees"]:
        index = 0
        while not nodes[index]["is_leaf"]:
            node = nodes[index]
            value = float(features[int(node["feature_idx"])])
            if math.isnan(value):
                go_left = bool(node["missing_go_to_left"])
            else:
                go_left = value <= float(node["threshold"])
            index = int(node["left"] if go_left else node["right"])
        raw += float(nodes[index]["value"])
    return _sigmoid(raw)


def _cube(mask: np.ndarray, point: np.ndarray | None, half_width: int) -> np.ndarray:
    if point is None:
        return mask[0:0, 0:0, 0:0]
    center = np.rint(point).astype(int)
    slices = tuple(
        slice(
            max(0, int(value) - half_width),
            min(mask.shape[axis], int(value) + half_width + 1),
        )
        for axis, value in enumerate(center)
    )
    return mask[slices]


def _expected_point(
    slot: str,
    centroids: Mapping[str, np.ndarray],
    arch_order: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray | None, str, int]:
    arch = "upper" if slot[0] in "12" else "lower"
    order = arch_order[arch]
    target = order.index(slot)
    observed = sorted(order.index(fdi) for fdi in centroids if fdi in order)
    if target in observed:
        return np.asarray(centroids[slot], dtype=np.float64), "observed", 0
    left = [index for index in observed if index < target]
    right = [index for index in observed if index > target]
    if left and right:
        lo, hi = max(left), min(right)
        weight = (target - lo) / (hi - lo)
        return (
            centroids[order[lo]] * (1.0 - weight) + centroids[order[hi]] * weight,
            "interior",
            hi - lo,
        )
    if len(left) >= 2 and target - max(left) <= 2:
        lo, hi = left[-2], left[-1]
        weight = (target - lo) / (hi - lo)
        return (
            centroids[order[lo]] * (1.0 - weight) + centroids[order[hi]] * weight,
            "terminal",
            target - hi,
        )
    if len(right) >= 2 and min(right) - target <= 2:
        lo, hi = right[0], right[1]
        weight = (target - lo) / (hi - lo)
        return (
            centroids[order[lo]] * (1.0 - weight) + centroids[order[hi]] * weight,
            "terminal",
            lo - target,
        )
    return None, "unavailable", 99


def slot_feature_rows(mask: np.ndarray, base: Any) -> list[dict[str, Any]]:
    label_slices = base._find_label_slices(mask)
    statistics: dict[str, dict[str, Any]] = {}
    centroids: dict[str, np.ndarray] = {}
    for fdi in base.FDI_SLOTS:
        label = int(base.FDI_TO_CLASS[fdi])
        label_slice = (
            label_slices[label - 1] if label - 1 < len(label_slices) else None
        )
        if label_slice is None:
            statistics[fdi] = {
                "voxels": 0,
                "components": 0,
                "largest_fraction": 0.0,
            }
            continue
        local = mask[label_slice] == label
        coords = np.argwhere(local)
        voxels = int(coords.shape[0])
        _, largest, components = base._largest_component(local)
        statistics[fdi] = {
            "voxels": voxels,
            "components": components,
            "largest_fraction": float(largest / voxels) if voxels else 0.0,
        }
        if voxels:
            offset = np.asarray(
                [axis.start for axis in label_slice], dtype=np.float64
            )
            centroids[fdi] = coords.mean(axis=0) + offset

    rows: list[dict[str, Any]] = []
    tooth_labels = tuple(int(value) for value in base.CLASS_TO_FDI)
    for slot in base.FDI_SLOTS:
        stats = statistics[slot]
        point, source, gap = _expected_point(slot, centroids, base.ARCH_SLOT_ORDER)
        arch = "upper" if slot[0] in "12" else "lower"
        jaw_label = 2 if arch == "upper" else 1
        local = _cube(mask, point, LOCAL_HALF_WIDTH)
        jaw_voxels = int(np.count_nonzero(local == jaw_label))
        local_tooth_voxels = int(np.count_nonzero(np.isin(local, tooth_labels)))
        order = base.ARCH_SLOT_ORDER[arch]
        index = order.index(slot)
        raw_present = int(stats["voxels"] >= RAW_PRESENT_MIN_VOXELS)
        observable = int(
            raw_present
            or (point is not None and jaw_voxels >= MIN_JAWBONE_VOXELS)
        )
        normalized = (
            point
            / np.maximum(np.asarray(mask.shape, dtype=np.float64) - 1.0, 1.0)
            if point is not None
            else np.asarray([-1.0, -1.0, -1.0])
        )
        rows.append(
            {
                "fdi": slot,
                "observable": observable,
                "raw_present": raw_present,
                "raw_voxels_log": math.log1p(int(stats["voxels"])),
                "component_count": int(stats["components"]),
                "largest_component_fraction": float(stats["largest_fraction"]),
                "expected_point_available": int(point is not None),
                "expected_point_source_observed": int(source == "observed"),
                "expected_point_source_interior": int(source == "interior"),
                "expected_point_source_terminal": int(source == "terminal"),
                "expected_gap": gap,
                "local_jaw_voxels_log": math.log1p(jaw_voxels),
                "local_any_tooth_voxels_log": math.log1p(local_tooth_voxels),
                "previous_slot_present": int(
                    index > 0 and order[index - 1] in centroids
                ),
                "next_slot_present": int(
                    index + 1 < len(order) and order[index + 1] in centroids
                ),
                "same_arch_present_count": sum(
                    fdi in centroids for fdi in order
                ),
                "same_quadrant_present_count": sum(
                    fdi in centroids for fdi in base.FDI_SLOTS if fdi[0] == slot[0]
                ),
                "quadrant": int(slot[0]),
                "position": int(slot[1]),
                "center_z": float(normalized[0]),
                "center_y": float(normalized[1]),
                "center_x": float(normalized[2]),
            }
        )
    return rows


def score(
    mask: np.ndarray,
    base: Any,
    model_path: str = str(MODEL_PATH),
) -> tuple[list[str], dict[str, Any]]:
    payload = load_model(model_path)
    feature_order = list(payload["feature_order"])
    residual_threshold = float(payload["residual_presence_threshold"])
    scored: list[dict[str, Any]] = []
    for row in slot_feature_rows(mask, base):
        if not row["observable"]:
            continue
        features = np.asarray(
            [float(row[name]) for name in feature_order], dtype=np.float32
        ).tolist()
        probability = predict_present_probability(features, payload)
        raw_absence = not bool(row["raw_present"])
        residual_absence = probability < residual_threshold
        eligible = (
            row["fdi"] not in THIRD_MOLARS
            and (raw_absence or residual_absence)
        )
        scored.append(
            {
                "fdi": row["fdi"],
                "present_probability": probability,
                "raw_absence": raw_absence,
                "residual_absence": residual_absence,
                "eligible": eligible,
                "observable": True,
            }
        )
    ranked = sorted(
        (row for row in scored if row["eligible"]),
        key=lambda row: (
            not bool(row["raw_absence"]),
            row["present_probability"],
            int(row["fdi"]),
        ),
    )
    eligible = ranked[:BASE_RELATIONS_PER_CASE]
    for row in ranked[BASE_RELATIONS_PER_CASE:]:
        if len(eligible) >= MAXIMUM_PER_CASE:
            break
        contiguous = any(
            row["fdi"][0] == selected_row["fdi"][0]
            and abs(int(row["fdi"]) - int(selected_row["fdi"])) == 1
            for selected_row in eligible
        )
        if (
            row["raw_absence"]
            and row["present_probability"] <= EXTRA_RAW_PRESENCE_THRESHOLD
            and contiguous
        ):
            eligible.append(row)
    selected = sorted((row["fdi"] for row in eligible), key=int)
    policy = dict(payload["absence_policy"])
    policy.update(
        {
            "base_relations_per_case": BASE_RELATIONS_PER_CASE,
            "maximum_per_case": MAXIMUM_PER_CASE,
            "extra_relation_policy": "contiguous_raw_absence_below_0p005",
        }
    )
    return selected, {
        "available": True,
        "source": "R218H_TF3_OOF_residual_occupancy_R234_adaptive_guard",
        "selected_fdis": selected,
        "selected_count": len(selected),
        "policy": policy,
        "candidate_priority": "raw_absence_before_residual_only_then_probability",
        "independent_tf3_oof_precision": 0.957983193277311,
        "independent_tf3_oof_support": 119,
        "scores": scored,
    }
