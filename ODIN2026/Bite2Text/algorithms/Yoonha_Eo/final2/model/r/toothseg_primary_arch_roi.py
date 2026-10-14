from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    index: int
    fdi: int
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray


def _full_mask(npz, index: int, image_shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, npz["crop_boxes"][index])
    crop = np.asarray(npz[f"mask_{index}"], dtype=bool)
    full = np.zeros(image_shape, dtype=bool)
    full[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return full


def load_detections(npz_path: Path) -> tuple[list[Detection], tuple[int, int]]:
    with np.load(npz_path) as npz:
        image_shape = tuple(map(int, npz["image_shape"]))
        detections = []
        for index, (fdi, confidence, box) in enumerate(
            zip(npz["fdis"], npz["confidences"], npz["crop_boxes"])
        ):
            detections.append(
                Detection(
                    index=index,
                    fdi=int(fdi),
                    confidence=float(confidence),
                    box=tuple(map(int, box)),
                    mask=_full_mask(npz, index, image_shape),
                )
            )
    return detections, image_shape


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(left, right).sum()
    return float(intersection / max(1, union))


def deduplicate_detections(
    detections: list[Detection], iou_threshold: float = 0.65
) -> list[Detection]:
    kept = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(mask_iou(candidate.mask, existing.mask) < iou_threshold for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.index)


def fdi_matches_view(fdi: int, view: str) -> bool:
    if view == "upper":
        return 11 <= fdi <= 28
    if view == "lower":
        return 31 <= fdi <= 48
    raise ValueError(f"ROI selection supports upper/lower views, received: {view}")


def _padded_box(
    mask: np.ndarray,
    view: str,
    horizontal_margin: float,
    dental_margin: float,
    mirror_margin: float,
) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot form an ROI from an empty tooth mask")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    span_x = max(1, x2 - x1)
    span_y = max(1, y2 - y1)
    margin_x = int(round(span_x * horizontal_margin))
    if view == "upper":
        margin_top = int(round(span_y * mirror_margin))
        margin_bottom = int(round(span_y * dental_margin))
    else:
        margin_top = int(round(span_y * dental_margin))
        margin_bottom = int(round(span_y * mirror_margin))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_top),
        min(width, x2 + margin_x),
        min(height, y2 + margin_bottom),
    )


def select_primary_arch_roi(
    npz_path: Path,
    view: str,
    minimum_teeth: int = 4,
    horizontal_margin: float = 0.12,
    dental_margin: float = 0.34,
    mirror_margin: float = 0.08,
) -> dict:
    detections, image_shape = load_detections(npz_path)
    unique = deduplicate_detections(detections)
    expected = [item for item in unique if fdi_matches_view(item.fdi, view)]
    if len(expected) < minimum_teeth:
        raise RuntimeError(
            f"Only {len(expected)} unique {view}-arch tooth masks remained; "
            f"at least {minimum_teeth} are required for a reliable ROI"
        )
    selected_mask = np.logical_or.reduce([item.mask for item in expected])
    crop_box = _padded_box(
        selected_mask,
        view,
        horizontal_margin=horizontal_margin,
        dental_margin=dental_margin,
        mirror_margin=mirror_margin,
    )
    x1, y1, x2, y2 = crop_box
    crop_fraction = float((x2 - x1) * (y2 - y1) / np.prod(image_shape))
    return {
        "crop_box": crop_box,
        "selected_mask": selected_mask,
        "all_mask": np.logical_or.reduce([item.mask for item in unique]),
        "selected_fdis": [item.fdi for item in expected],
        "selected_confidences": [item.confidence for item in expected],
        "raw_detection_count": len(detections),
        "unique_detection_count": len(unique),
        "selected_detection_count": len(expected),
        "crop_fraction": crop_fraction,
        "image_shape": image_shape,
    }

