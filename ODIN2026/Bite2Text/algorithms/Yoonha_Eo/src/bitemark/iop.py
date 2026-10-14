"""Public IOP routing and placeholder interpretation rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import Finding

REQUIRED_VIEWS = ("left", "right", "upper", "front", "lower")


def assign_required_views(
    image_paths: Sequence[Path], probability_matrix: np.ndarray
) -> dict[str, Path]:
    """Select one unique image per required view by maximum total probability."""
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.shape != (len(image_paths), len(REQUIRED_VIEWS)):
        raise ValueError("Expected an N x 5 view-probability matrix")
    if len(image_paths) < len(REQUIRED_VIEWS):
        raise ValueError("At least five IOP images are required")
    rows, columns = linear_sum_assignment(-probabilities)
    if len(columns) != len(REQUIRED_VIEWS):
        raise RuntimeError("Could not assign every required view")
    selected = {column: Path(image_paths[row]) for row, column in zip(rows, columns)}
    return {view: selected[column] for column, view in enumerate(REQUIRED_VIEWS)}


def interpret_midline(displacement: float, config: Mapping[str, object]) -> str:
    centered = float(config["midline_centered_tooth_width"])
    deviated = float(config["midline_deviated_tooth_width"])
    magnitude = abs(float(displacement))
    if magnitude <= centered:
        return "Centered"
    if magnitude <= deviated:
        return "Slightly deviated"
    return "Deviated"


def findings_from_synthetic_iop(
    features: Mapping[str, object], config: Mapping[str, object]
) -> list[Finding]:
    return [
        Finding("crowding", str(features["upper_crowding"]), "IOP", side="upper"),
        Finding("crowding", str(features["lower_crowding"]), "IOP", side="lower"),
        Finding(
            "midline",
            interpret_midline(float(features["midline_displacement_tooth_width"]), config),
            "IOP",
        ),
    ]
