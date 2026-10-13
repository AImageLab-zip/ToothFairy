"""Self-contained R212 Bone Atrophy inference adapter.

This module mirrors the frozen R212 crop contract. It is intentionally not
called by the submission runtime until the predicted-ROI and writer gates pass.
"""

from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
import torch
from torch import nn


REGIONS = tuple(
    f"{jaw}_{side}_{section}"
    for jaw in ("upper", "lower")
    for side in ("left", "right")
    for section in ("anterior", "posterior")
)
REGION_ID = {region: index for index, region in enumerate(REGIONS)}
FDI_SEQUENCE = (
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
)
FDI_TO_LABEL = {fdi: label for label, fdi in enumerate(FDI_SEQUENCE, start=11)}
LOWER_JAW_LABEL = 1
UPPER_JAW_LABEL = 2
MODEL_SHAPE = (128, 128, 128)
SOURCE_SHAPE = (160, 160, 160)  # 48 mm at the runtime's fixed 0.3 mm spacing.
HYBRID_MIN_OBSERVED = 2


def ecdf_percentile(score: float, reference: Sequence[float]) -> float:
    """Map a score to the midpoint ECDF used by the R212Q offline audit."""
    if not reference:
        raise ValueError("empty ECDF reference")
    ordered = sorted(float(value) for value in reference)
    left = bisect.bisect_left(ordered, float(score))
    right = bisect.bisect_right(ordered, float(score))
    return (0.5 * (left + right) + 0.5) / (len(ordered) + 1.0)


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class CandidateEncoder3D(nn.Module):
    def __init__(self, embedding_dim: int, base_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, base_channels, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels * 2, 2),
            ResidualBlock3D(base_channels * 2, base_channels * 4, 2),
            ResidualBlock3D(base_channels * 4, base_channels * 8, 2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.blocks(self.stem(x))))


class BoneAtrophyEvidenceModel(nn.Module):
    def __init__(self, embedding_dim: int, base_channels: int, region_dim: int) -> None:
        super().__init__()
        self.encoder = CandidateEncoder3D(embedding_dim, base_channels)
        self.region_embedding = nn.Embedding(len(REGIONS), region_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + region_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.instance_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        fused = self.fusion(
            torch.cat([embedding, self.region_embedding(region_id)], dim=1)
        )
        return self.instance_head(fused).squeeze(1)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    return coords.mean(axis=0) if coords.size else None


def _tooth_centroids(mask: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for fdi, label in FDI_TO_LABEL.items():
        coords = np.argwhere(mask == label)
        if coords.shape[0] >= 16:
            output[fdi] = coords.mean(axis=0)
    return output


def _all_tooth_centroids(mask: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for fdi, label in FDI_TO_LABEL.items():
        coords = np.argwhere(mask == label)
        if coords.size:
            output[fdi] = coords.mean(axis=0)
    return output


def _geometry_cache(mask: np.ndarray) -> dict[str, Any]:
    """Collect tooth and jaw coordinates once without changing centroid math."""
    tooth_coords: dict[str, np.ndarray] = {}
    foreground = np.argwhere(
        np.logical_and(mask >= min(FDI_TO_LABEL.values()), mask <= max(FDI_TO_LABEL.values()))
    )
    foreground_labels = (
        mask[tuple(foreground.T)] if foreground.size else np.empty(0, dtype=mask.dtype)
    )
    for fdi, label in FDI_TO_LABEL.items():
        tooth_coords[fdi] = foreground[foreground_labels == label]
    return {
        "tooth_coords": tooth_coords,
        "all_tooth_centers": {
            fdi: coords.mean(axis=0) for fdi, coords in tooth_coords.items() if coords.size
        },
        "split_tooth_centers": {
            fdi: coords.mean(axis=0)
            for fdi, coords in tooth_coords.items()
            if coords.shape[0] >= 16
        },
        "jaw_coords": {
            "lower": np.argwhere(mask == LOWER_JAW_LABEL),
            "upper": np.argwhere(mask == UPPER_JAW_LABEL),
        },
    }


def _midpoint(
    centers: Mapping[str, np.ndarray],
    jaw: str,
    first_values: set[str],
    second_values: set[str],
    fdi_axis: int,
    axis: int,
) -> tuple[float | None, bool | None]:
    first, second = [], []
    for fdi, center in centers.items():
        if jaw == "upper" and fdi[0] not in {"1", "2"}:
            continue
        if jaw == "lower" and fdi[0] not in {"3", "4"}:
            continue
        if fdi[fdi_axis] in first_values:
            first.append(float(center[axis]))
        elif fdi[fdi_axis] in second_values:
            second.append(float(center[axis]))
    if not first or not second:
        return None, None
    first_mean, second_mean = float(np.mean(first)), float(np.mean(second))
    return (first_mean + second_mean) / 2.0, first_mean >= second_mean


def jawbone_subregion(
    mask: np.ndarray,
    region: str,
    geometry_cache: Mapping[str, Any] | None = None,
) -> np.ndarray:
    jaw, side, section = region.split("_")
    base = mask == (UPPER_JAW_LABEL if jaw == "upper" else LOWER_JAW_LABEL)
    coords = (
        np.asarray(geometry_cache["jaw_coords"][jaw])
        if geometry_cache is not None
        else np.argwhere(base)
    )
    if not coords.size:
        return base
    keep = np.ones(coords.shape[0], dtype=bool)
    centers = (
        geometry_cache["split_tooth_centers"]
        if geometry_cache is not None
        else _tooth_centroids(mask)
    )
    midpoint, left_is_high = _midpoint(
        centers, jaw, {"2", "3"}, {"1", "4"}, fdi_axis=0, axis=2
    )
    if midpoint is None or left_is_high is None:
        midpoint, left_is_high = float(np.median(coords[:, 2])), True
    want_high = left_is_high if side == "left" else not left_is_high
    keep &= coords[:, 2] >= midpoint if want_high else coords[:, 2] <= midpoint
    active = coords[keep]
    midpoint, anterior_is_high = _midpoint(
        centers,
        jaw,
        {"1", "2", "3"},
        {"4", "5", "6", "7", "8"},
        fdi_axis=1,
        axis=1,
    )
    if midpoint is None or anterior_is_high is None:
        midpoint = float(np.median(active[:, 1])) if active.size else 0.0
        anterior_is_high = False
    want_high = anterior_is_high if section == "anterior" else not anterior_is_high
    keep &= coords[:, 1] >= midpoint if want_high else coords[:, 1] <= midpoint
    output = np.zeros(mask.shape, dtype=bool)
    selected = coords[keep]
    if selected.size:
        output[tuple(selected.T)] = True
    return output


def fdis_for_region(region: str) -> list[str]:
    jaw, side, section = region.split("_")
    quadrant = {
        ("upper", "right"): "1",
        ("upper", "left"): "2",
        ("lower", "left"): "3",
        ("lower", "right"): "4",
    }[(jaw, side)]
    positions = range(1, 4) if section == "anterior" else range(4, 9)
    return [f"{quadrant}{position}" for position in positions]


def _clamp_to_bbox(
    point: np.ndarray, region: np.ndarray, margin: int = 8
) -> np.ndarray:
    coords = np.argwhere(region)
    if not coords.size:
        return np.minimum(np.maximum(point, 0), np.asarray(region.shape) - 1)
    low = coords.min(axis=0).astype(float) + margin
    high = coords.max(axis=0).astype(float) - margin
    high = np.maximum(high, low)
    return np.minimum(np.maximum(point, low), high)


def _estimate_slot(
    fdi: str, centers: Mapping[str, np.ndarray], region: np.ndarray
) -> np.ndarray | None:
    target = int(fdi[1])
    quadrant = fdi[0]
    local = {int(key[1]): value for key, value in centers.items() if key[0] == quadrant}
    if target - 1 in local and target + 1 in local:
        return (local[target - 1] + local[target + 1]) / 2.0
    if target - 1 in local and target - 2 in local:
        return _clamp_to_bbox(local[target - 1] * 2.0 - local[target - 2], region)
    if target + 1 in local and target + 2 in local:
        return _clamp_to_bbox(local[target + 1] * 2.0 - local[target + 2], region)
    present = sorted(local)
    if len(present) >= 2:
        first, second = sorted(present, key=lambda position: abs(position - target))[:2]
        first, second = sorted((first, second))
        alpha = (target - first) / max(1, second - first)
        return _clamp_to_bbox(
            (1.0 - alpha) * local[first] + alpha * local[second], region
        )
    region_center = _centroid(region)
    if present and region_center is not None:
        nearest = min(present, key=lambda position: abs(position - target))
        return _clamp_to_bbox(0.55 * local[nearest] + 0.45 * region_center, region)
    return region_center


def _add_ellipsoid(mask: np.ndarray, center: np.ndarray) -> None:
    radius = np.asarray((30.0, 10.0 / 0.3, 10.0 / 0.3))
    low = np.maximum(np.floor(center - radius).astype(int), 0)
    high = np.minimum(np.ceil(center + radius).astype(int), np.asarray(mask.shape) - 1)
    slices = tuple(slice(int(low[axis]), int(high[axis]) + 1) for axis in range(3))
    grids = np.ogrid[slices]
    distance = np.zeros(tuple((high - low + 1).astype(int)), dtype=np.float32)
    for axis, grid in enumerate(grids):
        distance += ((grid.astype(np.float32) - float(center[axis])) / radius[axis]) ** 2
    mask[slices] |= distance <= 1.0


def candidate_geometry(
    segmentation: np.ndarray,
    region: str,
    geometry_cache: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int, int, str]:
    region_mask = jawbone_subregion(segmentation, region, geometry_cache)
    context_centers = (
        geometry_cache["all_tooth_centers"]
        if geometry_cache is not None
        else _all_tooth_centroids(segmentation)
    )
    centers, observed, estimated = [], 0, 0
    for fdi in fdis_for_region(region):
        tooth_coords = (
            np.asarray(geometry_cache["tooth_coords"][fdi])
            if geometry_cache is not None
            else np.argwhere(segmentation == FDI_TO_LABEL[fdi])
        )
        center = tooth_coords.mean(axis=0) if tooth_coords.shape[0] >= 32 else None
        if center is not None:
            observed += 1
        else:
            center = _estimate_slot(fdi, context_centers, region_mask)
            estimated += int(center is not None)
        if center is not None:
            centers.append(center)
    if not centers:
        return np.zeros(segmentation.shape, dtype=bool), None, 0, 0, "unavailable"
    slot_center = np.median(np.stack(centers), axis=0)
    support = np.zeros(segmentation.shape, dtype=bool)
    for center in centers:
        _add_ellipsoid(support, center)
    if np.any(region_mask):
        anatomy = np.logical_and(region_mask, support)
        if not np.any(anatomy):
            anatomy = region_mask
    else:
        anatomy = support
    use_slot = observed >= HYBRID_MIN_OBSERVED
    center = slot_center if use_slot else _centroid(anatomy)
    return anatomy, center, observed, estimated, "slot_median" if use_slot else "anatomy_centroid"


def _crop_with_padding(
    array: np.ndarray, center: np.ndarray, shape: Sequence[int], fill: float
) -> np.ndarray:
    output = np.full(tuple(shape), fill, dtype=array.dtype)
    starts = [int(round(float(center[axis]) - shape[axis] / 2.0)) for axis in range(3)]
    source, target = [], []
    for axis, start in enumerate(starts):
        end = start + int(shape[axis])
        source_start, source_end = max(0, start), min(array.shape[axis], end)
        if source_end <= source_start:
            return output
        target_start = source_start - start
        source.append(slice(source_start, source_end))
        target.append(slice(target_start, target_start + source_end - source_start))
    output[tuple(target)] = array[tuple(source)]
    return output


def _normalize(crop: np.ndarray) -> np.ndarray:
    crop_f = crop.astype(np.float32, copy=False)
    finite = crop_f[np.isfinite(crop_f)]
    if not finite.size:
        return np.zeros(crop_f.shape, dtype=np.float16)
    low, high = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    crop_f = np.clip(crop_f, low, high)
    crop_f = (crop_f - low) / (high - low)
    crop_f = crop_f * 2.0 - 1.0
    return crop_f.astype(np.float16)


def _resize(array: np.ndarray, order: int) -> np.ndarray:
    factors = tuple(target / source for target, source in zip(MODEL_SHAPE, array.shape))
    return ndimage.zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def fixed_pointer() -> np.ndarray:
    pointer = np.zeros(MODEL_SHAPE, dtype=bool)
    center = (np.asarray(MODEL_SHAPE, dtype=float) - 1.0) / 2.0
    radius = np.asarray((24.0, 27.0, 27.0))
    low = np.maximum(np.floor(center - radius).astype(int), 0)
    high = np.minimum(np.ceil(center + radius).astype(int), np.asarray(MODEL_SHAPE) - 1)
    slices = tuple(slice(int(low[axis]), int(high[axis]) + 1) for axis in range(3))
    grids = np.ogrid[slices]
    distance = np.zeros(tuple((high - low + 1).astype(int)), dtype=np.float32)
    for axis, grid in enumerate(grids):
        distance += ((grid.astype(np.float32) - float(center[axis])) / radius[axis]) ** 2
    pointer[slices] = distance <= 1.0
    return pointer.astype(np.float32)


def build_candidates(
    image: np.ndarray, segmentation: np.ndarray
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    pointer = fixed_pointer()
    geometry_cache = _geometry_cache(segmentation)
    tensors, regions, audit = [], [], []
    for region in REGIONS:
        anatomy, center, observed, estimated, center_policy = candidate_geometry(
            segmentation, region, geometry_cache
        )
        if center is None:
            audit.append({"region": region, "available": False})
            continue
        image_crop = _crop_with_padding(
            image.astype(np.float32, copy=False), center, SOURCE_SHAPE, float(np.median(image))
        )
        image_model = _normalize(_resize(image_crop, order=1)).astype(np.float32)
        tensors.append(np.stack([image_model, pointer], axis=0))
        regions.append(region)
        audit.append(
            {
                "region": region,
                "available": True,
                "observed_slots": observed,
                "estimated_slots": estimated,
                "center_policy": center_policy,
                "center_zyx": [float(value) for value in center],
                "anatomy_voxels": int(anatomy.sum()),
            }
        )
    if not tensors:
        return np.empty((0, 2, *MODEL_SHAPE), dtype=np.float32), [], audit
    return np.stack(tensors), regions, audit


def load_model(checkpoint_path: Path, device: torch.device) -> BoneAtrophyEvidenceModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    base_channels = int(state["encoder.stem.0.weight"].shape[0])
    embedding_dim = int(state["encoder.projection.1.weight"].shape[0])
    region_dim = int(state["region_embedding.weight"].shape[1])
    model = BoneAtrophyEvidenceModel(embedding_dim, base_channels, region_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def score_ensemble(
    image: np.ndarray,
    segmentation: np.ndarray,
    checkpoints: Sequence[Path],
    device: str | torch.device = "cuda",
    candidate_batch_size: int = 2,
    selection_score_references: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    tensors, regions, audit = build_candidates(image, segmentation)
    if not regions:
        return {"available": False, "case_score": None, "regions": [], "geometry": audit}
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    target = torch.device(device)
    x = torch.from_numpy(tensors)
    region_id = torch.tensor([REGION_ID[region] for region in regions])
    scores = []
    for checkpoint in checkpoints:
        model = load_model(Path(checkpoint), target)
        current = []
        for start in range(0, len(regions), candidate_batch_size):
            stop = min(start + candidate_batch_size, len(regions))
            with torch.amp.autocast(
                device_type=target.type,
                dtype=torch.bfloat16,
                enabled=target.type == "cuda",
            ):
                logits = model(
                    x[start:stop].to(target, non_blocking=True),
                    region_id[start:stop].to(target, non_blocking=True),
                )
            current.append(torch.sigmoid(logits.float()).cpu().numpy())
        scores.append(np.concatenate(current))
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    matrix = np.stack(scores)
    instance = matrix.mean(axis=0)
    case_by_model = np.mean(np.sort(matrix, axis=1)[:, -min(2, len(regions)):], axis=1)
    normalized_case_by_model = None
    calibrated_case_score = None
    if selection_score_references is not None:
        if len(selection_score_references) != len(checkpoints):
            raise ValueError(
                "selection_score_references must match the checkpoint count"
            )
        normalized_case_by_model = [
            ecdf_percentile(float(score), reference)
            for score, reference in zip(case_by_model, selection_score_references)
        ]
        calibrated_case_score = max(normalized_case_by_model)
    order = np.argsort(-instance)
    return {
        "available": True,
        "case_score": float(case_by_model.mean()),
        "case_score_std": float(case_by_model.std()),
        "case_score_by_model": [float(value) for value in case_by_model],
        "normalized_case_score_by_model": normalized_case_by_model,
        "calibrated_case_score": calibrated_case_score,
        "case_score_normalization": (
            "selection_split_ecdf_max_across_models"
            if normalized_case_by_model is not None
            else None
        ),
        "regions": [
            {
                "region": regions[index],
                "score": float(instance[index]),
                "score_std": float(matrix[:, index].std()),
            }
            for index in order
        ],
        "geometry": audit,
    }
