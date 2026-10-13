#!/usr/bin/env python3
"""Task-specific feature contracts used by the frozen Vertical ensemble."""
from __future__ import annotations

from typing import Any

import numpy as np

from features import safe_float

COMMON_KEYS = (
    "pair_count",
    "pair_dispersion_deg",
    "front_back_distance_mm",
    "vertical_separation_mm",
    "roll_correction_deg",
)
OVERBITE_KEYS = (
    "right_overbite_mm",
    "right_overbite_projected_crown_ratio",
    "left_overbite_mm",
    "left_overbite_projected_crown_ratio",
    "overbite_mean_mm",
    "overbite_min_mm",
    "overbite_max_mm",
    "overbite_asymmetry_mm",
    "overbite_projected_crown_mean_ratio",
)


def overbite_geometry_features(result: dict[str, Any] | None) -> np.ndarray:
    payload = result or {}
    frame = payload.get("frame", {})
    sides = payload.get("sides", {})
    summary = payload.get("summary", {})
    values = [safe_float(frame.get(key)) for key in COMMON_KEYS]
    values.extend(
        [
            safe_float(sides.get("right", {}).get("overbite_mm")),
            safe_float(sides.get("right", {}).get("overbite_projected_crown_ratio")),
            safe_float(sides.get("left", {}).get("overbite_mm")),
            safe_float(sides.get("left", {}).get("overbite_projected_crown_ratio")),
        ]
    )
    values.extend(
        safe_float(summary.get(key))
        for key in (
            "overbite_mean_mm",
            "overbite_min_mm",
            "overbite_max_mm",
            "overbite_asymmetry_mm",
            "overbite_projected_crown_mean_ratio",
        )
    )
    numeric = np.asarray(values, dtype=np.float32)
    missing = np.isnan(numeric).astype(np.float32)
    status = str(payload.get("status", "missing"))
    statuses = np.asarray(
        [float(status == value) for value in ("ok", "review", "not_assessable", "error")],
        dtype=np.float32,
    )
    selected_status = str(payload.get("source", {}).get("selected_segmentation_status", "missing"))
    segmentation = np.asarray(
        [float(selected_status == value) for value in ("acceptable", "retry", "hard_fail")],
        dtype=np.float32,
    )
    row = np.concatenate([numeric, missing, statuses, segmentation])
    if row.shape != (35,):
        raise RuntimeError(f"Overbite geometry feature shape {row.shape}, expected (35,)")
    return row


def segmentation_features(
    upper_labels: np.ndarray,
    lower_labels: np.ndarray,
) -> np.ndarray:
    """Match shared_features_v1 internal C17 count/fraction/presence contract."""
    counts_by_arch = []
    for labels in (upper_labels, lower_labels):
        values = np.asarray(labels, dtype=np.int64).reshape(-1)
        counts_by_arch.append(
            np.bincount(values.clip(min=0), minlength=17)[1:17].astype(np.float32)
        )
    counts = np.concatenate(counts_by_arch)
    fractions = counts.copy()
    fractions[:16] /= max(float(counts[:16].sum()), 1.0)
    fractions[16:] /= max(float(counts[16:].sum()), 1.0)
    row = np.concatenate(
        [
            np.log1p(counts),
            fractions * 100.0,
            (counts >= 50).astype(np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    if row.shape != (98,):
        raise RuntimeError(f"Segmentation feature shape {row.shape}, expected (98,)")
    return row
