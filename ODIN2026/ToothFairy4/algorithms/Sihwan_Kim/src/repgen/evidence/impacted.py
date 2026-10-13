"""R225 label-blind impacted-tooth consensus runtime.

The branch proposes twelve high-risk FDI locations from the predicted 47-class
segmentation, scores them with the frozen R198 five-fold MIL ensemble, and only
emits an exact-FDI finding when the independent R134D2 position-8 guard agrees.
Report text and reference labels are never inputs to this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from . import models as weak_runtime


MODEL_ROOT = weak_runtime.MODEL_ROOT
CONTRACT_PATH = Path(__file__).resolve().parents[3] / "configs/impacted_consensus.json"
HIGH_RISK_FDIS = tuple(
    f"{quadrant}{slot}" for quadrant in "1234" for slot in (3, 7, 8)
)
LATERAL_MIRROR = {"18": "28", "28": "18", "38": "48", "48": "38"}
LOCAL_SHAPE = (160, 160, 160)
CONTEXT_SHAPE = (224, 224, 224)
CONTEXT_DOWNSAMPLED_SHAPE = (112, 112, 112)
OUTPUT_SIZE = 64
POSITION_DIM = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "r225_impacted_runtime_contract_v1":
        raise ValueError("unsupported R225 impacted runtime contract")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("R225 impacted contract must contain five folds")
    return payload


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask.astype(bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def _center(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("cannot compute center of an empty ROI")
    return points.mean(axis=0)


def _crop_with_padding(
    array: np.ndarray,
    center: Sequence[float],
    shape: Sequence[int],
    fill: float,
) -> np.ndarray:
    shape_t = tuple(int(value) for value in shape)
    output = np.full(shape_t, fill, dtype=array.dtype)
    starts = [
        int(round(float(center[axis]) - shape_t[axis] / 2.0))
        for axis in range(3)
    ]
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis, start in enumerate(starts):
        end = start + shape_t[axis]
        source_start = max(0, start)
        source_end = min(array.shape[axis], end)
        if source_end <= source_start:
            return output
        target_start = source_start - start
        target_end = target_start + source_end - source_start
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    output[tuple(target_slices)] = array[tuple(source_slices)]
    return output


def _normalize_crop(image: np.ndarray) -> np.ndarray:
    return weak_runtime.normalize_crop(image).astype(np.float32, copy=False)


def _rotation_angle(target: np.ndarray, anchor: np.ndarray) -> float:
    delta = _center(target) - _center(anchor)
    return math.degrees(math.atan2(float(delta[1]), float(delta[2])))


def _rotate(array: np.ndarray, angle: float, image: bool) -> np.ndarray:
    if abs(angle) < 1e-8:
        return array.copy()
    return ndimage.rotate(
        array.astype(np.float32, copy=False),
        angle=angle,
        axes=(1, 2),
        reshape=False,
        order=1 if image else 0,
        mode="constant",
        cval=-1.0 if image else 0.0,
        prefilter=image,
    )


def _canonicalize(
    arrays: list[np.ndarray],
    *,
    target_index: int,
    anchor_index: int,
    angle: float,
    image_indices: set[int],
    upper_jaw: bool,
) -> list[np.ndarray]:
    rotated = [
        _rotate(array, angle, index in image_indices)
        for index, array in enumerate(arrays)
    ]
    target = rotated[target_index] > 0
    anchor = rotated[anchor_index] > 0
    horizontal_flip = bool(
        np.any(target)
        and np.any(anchor)
        and float(_center(target)[2]) <= float(_center(anchor)[2])
    )
    if horizontal_flip:
        rotated = [np.flip(array, axis=2).copy() for array in rotated]
    if upper_jaw:
        rotated = [np.flip(array, axis=0).copy() for array in rotated]
    return rotated


def _downsample_context(array: np.ndarray, image: bool) -> np.ndarray:
    if array.shape != CONTEXT_SHAPE:
        raise ValueError(f"unexpected R225 context shape: {array.shape}")
    blocks = array.reshape(112, 2, 112, 2, 112, 2)
    return blocks.mean(axis=(1, 3, 5)) if image else blocks.max(axis=(1, 3, 5))


def _resize(array: np.ndarray, *, binary: bool = False) -> np.ndarray:
    tensor = torch.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32)
    )[None, None]
    resized = F.interpolate(
        tensor,
        size=(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE),
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    output = resized.numpy()
    return (output > 0.5).astype(np.float32) if binary else output.astype(np.float32)


def _candidate_position(fdi: str) -> np.ndarray:
    features = np.zeros(POSITION_DIM, dtype=np.float32)
    quadrant, slot = int(fdi[0]), int(fdi[1])
    features[quadrant - 1] = 1.0
    features[4] = (slot - 1) / 7.0
    features[5] = float(slot >= 4)
    features[6] = float(slot == 8)
    features[7] = float(quadrant <= 2)
    return features


def _tooth_components(mask: np.ndarray, quadrant: str) -> dict[int, np.ndarray]:
    components: dict[int, np.ndarray] = {}
    for slot in range(1, 9):
        fdi = f"{quadrant}{slot}"
        component = _largest_component(mask == weak_runtime.FDI_TO_LABEL[fdi])
        if np.any(component):
            components[slot] = component
    return components


def _component_route_state(
    component: np.ndarray | None,
    fdi: str,
    volume_priors: Mapping[str, float],
) -> str:
    voxels = 0 if component is None else int(component.sum())
    guard = max(100, int(math.ceil(float(volume_priors[fdi]) * 0.05)))
    subguard = max(100, int(math.ceil(guard * 0.50)))
    if voxels >= guard:
        return "exact_pass"
    if voxels >= subguard:
        return "exact_subguard_anchor"
    return "geometry_fallback"


def _estimated_center(
    components: Mapping[int, np.ndarray],
    target_slot: int,
    jawbone: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if target_slot in components:
        return _center(components[target_slot]), "predicted_target_component"
    slots = sorted(components)
    if len(slots) >= 2:
        lower = [slot for slot in slots if slot < target_slot]
        upper = [slot for slot in slots if slot > target_slot]
        if lower and upper:
            anterior, posterior = max(lower), min(upper)
        elif len(lower) >= 2:
            anterior, posterior = lower[-2], lower[-1]
        elif len(upper) >= 2:
            anterior, posterior = upper[0], upper[1]
        else:
            anterior, posterior = sorted(
                sorted(slots, key=lambda slot: (abs(slot - target_slot), slot))[:2]
            )
        anterior_point = _center(components[anterior])
        posterior_point = _center(components[posterior])
        raw_step = (posterior_point - anterior_point) / float(posterior - anterior)
        raw_step_mm = float(np.linalg.norm(raw_step) * 0.3)
        if raw_step_mm > 0.0:
            bounded_step_mm = float(np.clip(raw_step_mm, 4.0, 16.0))
            step = raw_step * (bounded_step_mm / raw_step_mm)
            return (
                posterior_point + float(target_slot - posterior) * step,
                "two_tooth_arch_geometry",
            )
    if len(slots) == 1 and np.any(jawbone):
        slot = slots[0]
        point = _center(components[slot])
        direction = point - _center(jawbone)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            step_voxels = 10.0 / 0.3
            return (
                point + float(target_slot - slot) * direction / norm * step_voxels,
                "single_tooth_jaw_radial",
            )
    return None, "unresolved_no_arch_anchor"


def _search_sphere(shape: Sequence[int], center: np.ndarray) -> np.ndarray:
    radius = 8.0 / 0.3
    low = np.maximum(np.floor(center - radius).astype(int), 0)
    high = np.minimum(np.ceil(center + radius).astype(int) + 1, shape)
    if np.any(low >= high):
        return np.zeros(tuple(shape), dtype=bool)
    grid = np.ogrid[
        tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    ]
    distance = sum(
        (axis.astype(np.float64) - float(value)) ** 2
        for axis, value in zip(grid, center)
    )
    output = np.zeros(tuple(shape), dtype=bool)
    output[
        tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    ] = distance <= radius**2
    return output


def _route_payload(
    image: np.ndarray,
    jawbone: np.ndarray,
    target: np.ndarray,
    fdi: str,
) -> dict[str, np.ndarray]:
    center = _center(target)
    fill = float(np.median(image))
    local_image = _normalize_crop(
        _crop_with_padding(image, center, LOCAL_SHAPE, fill)
    )
    local_target = _crop_with_padding(
        target.astype(np.uint8), center, LOCAL_SHAPE, 0
    )
    local_target = ndimage.binary_dilation(local_target > 0, iterations=12)
    context_image = _normalize_crop(
        _crop_with_padding(image, center, CONTEXT_SHAPE, fill)
    )
    context_target = _crop_with_padding(
        target.astype(np.uint8), center, CONTEXT_SHAPE, 0
    )
    context_jaw = _crop_with_padding(
        jawbone.astype(np.uint8), center, CONTEXT_SHAPE, 0
    )
    jaw_anchor = _largest_component(context_jaw)
    if int(jaw_anchor.sum()) >= 500:
        marker = np.zeros(CONTEXT_SHAPE, dtype=np.uint8)
        marker_center = np.rint(_center(jaw_anchor)).astype(int)
        marker_slices = tuple(
            slice(max(0, int(value) - 2), min(CONTEXT_SHAPE[axis], int(value) + 3))
            for axis, value in enumerate(marker_center)
        )
        marker[marker_slices] = 1
        angle = _rotation_angle(context_target, marker)
    else:
        marker = np.zeros(CONTEXT_SHAPE, dtype=np.uint8)
        angle = 0.0
    (
        local_image,
        local_target,
        context_image,
        context_target,
        context_jaw,
        _,
    ) = _canonicalize(
        [local_image, local_target, context_image, context_target, context_jaw, marker],
        target_index=3,
        anchor_index=5,
        angle=angle,
        image_indices={0, 2},
        upper_jaw=fdi[0] in {"1", "2"},
    )
    context_image_112 = _downsample_context(context_image, image=True)
    context_target_112 = _downsample_context(context_target, image=False) > 0
    context_jaw_112 = _downsample_context(context_jaw, image=False) > 0
    context_route_112 = np.logical_or(context_target_112, context_jaw_112)
    return {
        "context_image": _resize(context_image_112),
        "local_image": _resize(local_image),
        "context_mask": _resize(context_route_112),
        "local_mask": _resize(local_target, binary=True),
        "target_context_mask": _resize(context_target_112),
        "target_local_mask": _resize(local_target, binary=True),
    }


def build_candidates(
    image: np.ndarray,
    segmentation: np.ndarray,
    volume_priors: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jawbone = np.isin(segmentation, weak_runtime.JAW_LABELS)
    components = {
        quadrant: _tooth_components(segmentation, quadrant)
        for quadrant in "1234"
    }
    candidates: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for fdi in HIGH_RISK_FDIS:
        exact = components[fdi[0]].get(int(fdi[1]))
        route_state = _component_route_state(exact, fdi, volume_priors)
        if exact is not None and route_state == "geometry_fallback":
            center, source = _center(exact), "predicted_target_component_geometry"
        else:
            center, source = _estimated_center(
                components[fdi[0]], int(fdi[1]), jawbone
            )
        if center is None:
            audit.append({"tooth_fdi": fdi, "available": False, "route_source": source})
            continue
        target = exact if route_state != "geometry_fallback" else None
        if target is None:
            target = _search_sphere(image.shape, center)
        if not np.any(target):
            audit.append({"tooth_fdi": fdi, "available": False, "route_source": source})
            continue
        try:
            arrays = _route_payload(image, jawbone, target, fdi)
        except Exception as error:
            audit.append(
                {
                    "tooth_fdi": fdi,
                    "available": False,
                    "route_source": source,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        candidates.append({"tooth_fdi": fdi, "arrays": arrays})
        audit.append(
            {
                "tooth_fdi": fdi,
                "available": True,
                "route_source": source,
                "route_state": route_state,
                "target_voxels": int(np.count_nonzero(target)),
            }
        )
    return candidates, audit


class Small3DEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4)
        layers: list[nn.Module] = []
        current = in_channels
        for width in widths:
            groups = next(group for group in (8, 4, 2, 1) if width % group == 0)
            layers.extend(
                [
                    nn.Conv3d(current, width, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(inplace=True),
                ]
            )
            current = width
        layers.append(nn.AdaptiveAvgPool3d(1))
        self.net = nn.Sequential(*layers)
        self.output_dim = widths[-1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).flatten(1)


class DualScaleROIMIL(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        image_channels = int(config.get("image_base_channels", 8))
        mask_channels = int(config.get("mask_base_channels", 4))
        embedding_dim = int(config.get("embedding_dim", 128))
        attention_dim = int(config.get("attention_dim", 64))
        self.image_encoder = Small3DEncoder(1, image_channels)
        self.mask_encoder = Small3DEncoder(1, mask_channels)
        feature_dim = self.image_encoder.output_dim * 4 + self.mask_encoder.output_dim * 4
        position_dim = int(config.get("position_embedding_dim", 16))
        self.position_encoder = nn.Sequential(
            nn.Linear(POSITION_DIM, position_dim),
            nn.LayerNorm(position_dim),
            nn.SiLU(inplace=True),
        )
        feature_dim += position_dim
        self.project = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(config.get("dropout", 0.25))),
        )
        self.attention_v = nn.Linear(embedding_dim, attention_dim)
        self.attention_u = nn.Linear(embedding_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.instance_head = nn.Linear(embedding_dim, 1)
        self.bag_head = nn.Sequential(
            nn.Dropout(float(config.get("dropout", 0.25))),
            nn.Linear(embedding_dim, 1),
        )
        self.candidate_evidence_blend = float(config.get("candidate_evidence_blend", 0.25))
        self.candidate_pool_temperature = float(config.get("candidate_pool_temperature", 0.5))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branches = [
            self.image_encoder(batch["context_image"]),
            self.image_encoder(batch["local_image"]),
            self.mask_encoder(batch["context_mask"]),
            self.mask_encoder(batch["local_mask"]),
            self.image_encoder(batch["context_image"] * batch["target_context_mask"]),
            self.image_encoder(batch["local_image"] * batch["target_local_mask"]),
            self.mask_encoder(batch["target_context_mask"]),
            self.mask_encoder(batch["target_local_mask"]),
            self.position_encoder(batch["candidate_position"]),
        ]
        embeddings = self.project(torch.cat(branches, dim=1))
        instance_logits = self.instance_head(embeddings).flatten()
        raw_attention = self.attention_w(
            torch.tanh(self.attention_v(embeddings))
            * torch.sigmoid(self.attention_u(embeddings))
        ).flatten()
        attention = torch.softmax(raw_attention.float(), dim=0)
        pooled = torch.sum(attention.to(embeddings.dtype).unsqueeze(1) * embeddings, dim=0)
        context_logit = self.bag_head(pooled).flatten()[0]
        temperature = self.candidate_pool_temperature
        evidence_logit = temperature * (
            torch.logsumexp(instance_logits.float() / temperature, dim=0)
            - math.log(instance_logits.numel())
        )
        bag_logit = (
            (1.0 - self.candidate_evidence_blend) * context_logit.float()
            + self.candidate_evidence_blend * evidence_logit
        )
        return bag_logit, instance_logits, attention


def _batch(candidates: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "context_image",
        "local_image",
        "context_mask",
        "local_mask",
        "target_context_mask",
        "target_local_mask",
    )
    batch = {
        key: torch.from_numpy(
            np.stack([candidate["arrays"][key] for candidate in candidates])
        ).float()[:, None].to(device)
        for key in keys
    }
    batch["candidate_position"] = torch.from_numpy(
        np.stack([_candidate_position(str(candidate["tooth_fdi"])) for candidate in candidates])
    ).to(device)
    return batch


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


@torch.inference_mode()
def score_r198_ensemble(
    candidates: Sequence[Mapping[str, Any]],
    device: str | torch.device = "cuda",
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    if len(candidates) < 2:
        return {"available": False, "reason": "fewer_than_two_candidates"}
    contract = _load_contract(contract_path)
    resolved_device = torch.device(device)
    batch = _batch(candidates, resolved_device)
    fold_rows: list[dict[str, Any]] = []
    attention_by_fdi = {str(candidate["tooth_fdi"]): [] for candidate in candidates}
    instance_by_fdi = {str(candidate["tooth_fdi"]): [] for candidate in candidates}
    top_votes = {str(candidate["tooth_fdi"]): 0 for candidate in candidates}
    for fold_contract in contract["folds"]:
        checkpoint_path = MODEL_ROOT / str(fold_contract["path"])
        if _sha256(checkpoint_path) != str(fold_contract["sha256"]):
            raise ValueError(f"R198 fold hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=False,
        )
        config = dict(checkpoint.get("config") or {})
        model = DualScaleROIMIL(config).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        bag_logit, instance_logits, attention = model(batch)
        bag_score = float(torch.sigmoid(bag_logit).cpu())
        fold_positive = int(bag_score >= float(fold_contract["inner_threshold"]))
        center = float(fold_contract["score_normalizer_center"])
        scale = max(float(fold_contract["score_normalizer_scale"]), 1e-6)
        top_index = int(torch.argmax(attention).cpu())
        top_fdi = str(candidates[top_index]["tooth_fdi"])
        top_votes[top_fdi] += 1
        instance_scores = torch.sigmoid(instance_logits).cpu().numpy()
        attention_values = attention.cpu().numpy()
        for index, candidate in enumerate(candidates):
            fdi = str(candidate["tooth_fdi"])
            attention_by_fdi[fdi].append(float(attention_values[index]))
            instance_by_fdi[fdi].append(float(instance_scores[index]))
        fold_rows.append(
            {
                "fold": int(fold_contract["fold"]),
                "bag_score": bag_score,
                "bag_prediction": fold_positive,
                "inner_normalized_margin": (_logit(bag_score) - center) / scale,
                "top_fdi": top_fdi,
            }
        )
        del model
    ranked = sorted(
        attention_by_fdi,
        key=lambda fdi: (
            -top_votes[fdi],
            -float(np.mean(attention_by_fdi[fdi])),
            -float(np.mean(instance_by_fdi[fdi])),
            int(fdi),
        ),
    )
    top_fdi, second_fdi = ranked[:2]
    margin = float(np.mean(attention_by_fdi[top_fdi])) - float(
        np.mean(attention_by_fdi[second_fdi])
    )
    positive_votes = sum(int(row["bag_prediction"]) for row in fold_rows)
    policy = contract["policy"]
    exact_eligible = bool(
        positive_votes >= int(policy["minimum_positive_votes"])
        and top_votes[top_fdi] >= int(policy["minimum_fdi_votes"])
        and margin >= float(policy["minimum_attention_margin"])
    )
    return {
        "available": True,
        "candidate_count": len(candidates),
        "positive_fold_votes": positive_votes,
        "mean_threshold_normalized_margin": float(
            np.mean([row["inner_normalized_margin"] for row in fold_rows])
        ),
        "top_fdi": top_fdi,
        "top_fdi_vote_count": top_votes[top_fdi],
        "second_fdi": second_fdi,
        "attention_margin": margin,
        "r198_exact_guard": exact_eligible,
        "folds": fold_rows,
    }


def _legacy_guard(
    legacy_rows: Sequence[Mapping[str, Any]],
    online_threshold: float,
) -> dict[str, Any]:
    eligible = [
        row
        for row in legacy_rows
        if row.get("label") == "impacted" and str(row.get("tooth_fdi")) in LATERAL_MIRROR
    ]
    if not eligible:
        return {"available": False, "guard": False, "top_fdi": ""}
    top = max(
        eligible,
        key=lambda row: (float(row.get("score") or 0.0), int(str(row["tooth_fdi"]))),
    )
    return {
        "available": True,
        "guard": bool(
            float(top.get("score") or 0.0) >= online_threshold
            and top.get("writer_candidate_eligible")
        ),
        "top_fdi": str(top["tooth_fdi"]),
        "score": float(top.get("score") or 0.0),
        "online_threshold": online_threshold,
        "route_guard": bool(top.get("writer_candidate_eligible")),
    }


def score_consensus(
    image: np.ndarray,
    segmentation: np.ndarray,
    legacy_rows: Sequence[Mapping[str, Any]],
    volume_priors: Mapping[str, float],
    device: str | torch.device = "cuda",
) -> tuple[list[str], dict[str, Any]]:
    try:
        contract = _load_contract()
        candidates, candidate_audit = build_candidates(
            image, segmentation, volume_priors
        )
        r198 = score_r198_ensemble(candidates, device=device)
        legacy = _legacy_guard(
            legacy_rows,
            float(contract["policy"]["legacy_online_score_threshold"]),
        )
        new_fdi = str(r198.get("top_fdi", ""))
        old_fdi = str(legacy.get("top_fdi", ""))
        agreement = bool(
            new_fdi
            and old_fdi
            and new_fdi in {old_fdi, LATERAL_MIRROR.get(old_fdi, old_fdi)}
        )
        selected = bool(
            r198.get("r198_exact_guard")
            and legacy.get("guard")
            and agreement
        )
        return ([new_fdi] if selected else []), {
            "schema_version": "r225_impacted_cross_model_consensus_audit_v1",
            "available": True,
            "selected": selected,
            "selected_fdis": [new_fdi] if selected else [],
            "cross_model_location_agreement": agreement,
            "r198": r198,
            "r134d2": legacy,
            "candidate_routes": candidate_audit,
            "report_label_used": False,
        }
    except Exception as error:
        return [], {
            "schema_version": "r225_impacted_cross_model_consensus_audit_v1",
            "available": False,
            "selected": False,
            "selected_fdis": [],
            "error_type": type(error).__name__,
            "error": str(error),
            "report_label_used": False,
        }
