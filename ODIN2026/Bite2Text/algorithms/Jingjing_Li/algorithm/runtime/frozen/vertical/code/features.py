#!/usr/bin/env python3
"""Stable feature contracts for frozen Vertical occlusion classifiers."""
from __future__ import annotations

from typing import Any

import numpy as np

FRAME_KEYS = (
    "pair_count",
    "pair_dispersion_deg",
    "front_back_distance_mm",
    "vertical_separation_mm",
    "roll_correction_deg",
    "basis_determinant",
)
LANDMARK_SCALAR_KEYS = (
    "projected_crown_height_mm",
    "source_point_count",
    "sampled_point_count",
    "incisal_point_count",
    "labial_point_count",
    "incisal_si_iqr_mm",
    "labial_ap_iqr_mm",
)
SIDE_KEYS = (
    "overjet_tip_mm",
    "overjet_surface_mm",
    "overjet_surface_matched_count",
    "overjet_surface_iqr_mm",
    "overbite_mm",
    "lower_projected_crown_height_mm",
    "overbite_projected_crown_ratio",
)
SUMMARY_KEYS = (
    "overjet_tip_mean_mm",
    "overjet_tip_min_mm",
    "overjet_tip_max_mm",
    "overjet_tip_asymmetry_mm",
    "overjet_surface_mean_mm",
    "overjet_surface_min_mm",
    "overjet_surface_max_mm",
    "overjet_surface_asymmetry_mm",
    "overbite_mean_mm",
    "overbite_min_mm",
    "overbite_max_mm",
    "overbite_asymmetry_mm",
    "overbite_projected_crown_mean_ratio",
    "overbite_projected_crown_min_ratio",
    "overbite_projected_crown_max_ratio",
    "overbite_projected_crown_asymmetry_ratio",
)
STATUS_VALUES = ("ok", "review", "not_assessable", "error")
REVIEW_REASONS = (
    "central_incisor_not_anterior_11",
    "central_incisor_not_anterior_21",
    "central_incisor_not_anterior_31",
    "central_incisor_not_anterior_41",
    "front_back_distance_implausible",
    "landmark_unavailable_11",
    "landmark_unavailable_21",
    "left_right_order_invalid_41_31",
    "overbite_implausible_left",
    "overbite_implausible_right",
    "overbite_side_asymmetry_large",
    "overjet_side_asymmetry_large",
    "overjet_tip_implausible_left",
    "overjet_tip_implausible_right",
    "pair_dispersion_too_large",
    "selected_segmentation_hard_fail",
    "tooth_11_crown_height_implausible",
    "tooth_21_crown_height_implausible",
    "tooth_31_crown_height_implausible",
    "tooth_41_crown_height_implausible",
)
WARNINGS = (
    "selected_segmentation_needs_review",
    "surface_overjet_unavailable_left",
    "surface_overjet_unavailable_right",
)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def feature_names() -> list[str]:
    names = [f"frame.{key}" for key in FRAME_KEYS]
    names.extend(("mapping.upper_vertices", "mapping.lower_vertices"))
    for tooth in ("11", "21", "31", "41"):
        names.extend(f"landmark.{tooth}.point_{axis}" for axis in "xyz")
        names.extend(f"landmark.{tooth}.{key}" for key in LANDMARK_SCALAR_KEYS)
        names.append(f"landmark.{tooth}.available")
    for side in ("right", "left"):
        names.extend(f"side.{side}.{key}" for key in SIDE_KEYS)
        names.append(f"side.{side}.available")
    names.extend(f"summary.{key}" for key in SUMMARY_KEYS)
    names.extend(f"status.{value}" for value in STATUS_VALUES)
    names.extend(f"review_reason.{value}" for value in REVIEW_REASONS)
    names.extend(f"warning.{value}" for value in WARNINGS)
    return names


def geometry_features(payload: dict[str, Any] | None) -> np.ndarray:
    """Flatten a live geometry result to one fixed 111-dimensional row."""
    result = payload or {}
    values: list[float] = []

    frame = result.get("frame", {})
    values.extend(safe_float(frame.get(key)) for key in FRAME_KEYS)
    mapping = result.get("mapping", {})
    values.extend(
        safe_float(mapping.get(key))
        for key in ("upper_vertices", "lower_vertices")
    )

    landmarks = result.get("landmarks", {})
    for tooth in ("11", "21", "31", "41"):
        landmark = landmarks.get(tooth, {})
        point = landmark.get("point_local_xyz_mm") or [np.nan, np.nan, np.nan]
        point = list(point)[:3] + [np.nan] * max(0, 3 - len(point))
        values.extend(safe_float(value) for value in point[:3])
        values.extend(
            safe_float(landmark.get(key))
            for key in LANDMARK_SCALAR_KEYS
        )
        values.append(float(bool(landmark)))

    sides = result.get("sides", {})
    for side_name in ("right", "left"):
        side = sides.get(side_name, {})
        values.extend(safe_float(side.get(key)) for key in SIDE_KEYS)
        values.append(float(bool(side)))

    summary = result.get("summary", {})
    values.extend(safe_float(summary.get(key)) for key in SUMMARY_KEYS)
    status = str(result.get("status", "missing"))
    values.extend(float(status == value) for value in STATUS_VALUES)

    quality = result.get("quality", {})
    reasons = set(quality.get("review_reasons") or [])
    warnings = set(quality.get("warnings") or [])
    values.extend(float(reason in reasons) for reason in REVIEW_REASONS)
    values.extend(float(warning in warnings) for warning in WARNINGS)

    row = np.asarray(values, dtype=np.float32)
    expected = len(feature_names())
    if row.shape != (expected,):
        raise RuntimeError(f"Geometry feature shape {row.shape}, expected {(expected,)}")
    return row
