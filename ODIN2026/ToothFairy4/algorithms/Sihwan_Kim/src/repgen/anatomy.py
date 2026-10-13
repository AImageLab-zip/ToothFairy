"""Memory-bounded ODIN2026 Task 1 inference and fail-closed report generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ROOT = Path("/opt/ml/model")
NNUNET_RESULTS = MODEL_ROOT / "nnUNet_results"
DATASET_NAME = "Dataset317_ToothFairy3R142Unique"
TRAINER_NAME = "nnUNetTrainer_R142B_FPGuard_500ep"
PLANS_NAME = "nnUNetResEncUNetLPlans_torchres"
CONFIGURATION = "3d_fullres_torchres_mambabot2_ps128x256x256_bs1"
MODEL_FOLDER_NAME = f"{TRAINER_NAME}__{PLANS_NAME}__{CONFIGURATION}"
MODEL_FOLDER = NNUNET_RESULTS / DATASET_NAME / MODEL_FOLDER_NAME
CHECKPOINT_NAME = "checkpoint_best.pth"
CHECKPOINT = MODEL_FOLDER / "fold_0" / CHECKPOINT_NAME
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
VOLUME_PRIORS = MODEL_ROOT / "r117_fdi_priors.json"
TARGET_SPACING_XYZ = (0.3, 0.3, 0.3)
EXPECTED_CHECKPOINT_SHA256 = "c07af3d856d283d2583e5b35c31b3a3c4e2e4d104520bc3b24f33e82c6c16232"

FDI_SLOTS = tuple(f"{quadrant}{position}" for quadrant in "1234" for position in "12345678")
CLASS_TO_FDI = {
    **{label: str(label) for label in range(11, 19)},
    **{label: str(label + 2) for label in range(19, 27)},
    **{label: str(label + 4) for label in range(27, 35)},
    **{label: str(label + 6) for label in range(35, 43)},
}
FDI_TO_CLASS = {fdi: label for label, fdi in CLASS_TO_FDI.items()}
THIRD_MOLARS = {"18", "28", "38", "48"}
ARCH_SLOT_ORDER = {
    "upper": tuple([f"1{position}" for position in range(8, 0, -1)] + [f"2{position}" for position in range(1, 9)]),
    "lower": tuple([f"3{position}" for position in range(8, 0, -1)] + [f"4{position}" for position in range(1, 9)]),
}
LABEL_GROUPS = {
    "jawbone": (1, 2),
    "canal": (3, 4, 43, 44, 45),
    "sinus": (5, 6),
    "bridge": (8,),
    "crown": (9,),
    "implant": (10,),
}
DIRECT_OBJECT_POLICY = {
    "implant": {
        "min_component_voxels": 53,
        "calibrated_threshold_voxels": 52.6,
        "limit_per_component": 1,
        "source": "R164P_F_P_multiseed_positive_envelope_factor_0p10",
    },
    "crown": {
        "min_component_voxels": 24,
        "calibrated_threshold_voxels": 24.0,
        "limit_per_component": 1,
        "source": "R153_legacy_emission_retained_after_R164P_writer_recall_failure",
    },
    "bridge": {
        "min_component_voxels": 48,
        "calibrated_threshold_voxels": 48.0,
        "limit_per_component": 3,
        "source": "R153_legacy_emission_retained_after_R154B_recall_failure",
    },
}
PROSTHETIC_RELATION_POLICY = {
    "implant": {
        "promoted": False,
        "kind": "slot_hungarian",
        "observed_slot_penalty_mm": 0.0,
    },
    "crown": {
        "promoted": True,
        "promotion_tier": "engineering_promoted",
        "kind": "surface_covers",
        "surface_threshold_mm": 1.0,
        "max_relations": 3,
        "min_contact_voxels": 1,
        "min_relative_near_2mm": 0.0,
        "centroid_fallback": False,
        "source": "R165_TF3_FP97_geometry_plus_frozen_A20_noninferiority",
    },
    "bridge": {
        "promoted": False,
        "kind": "centroid_top3",
    },
}
SAFE_EMPTY_REPORT = "CBCT: no report-ready tooth-numbered finding is available from the structured evidence."


def _sitk():
    import SimpleITK

    return SimpleITK


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_runtime_dirs(tmp_root: Path) -> None:
    for relative in ("home", "cache", "triton", "nnunet_raw", "nnunet_preprocessed", "input", "prediction"):
        (tmp_root / relative).mkdir(parents=True, exist_ok=True)


def verify_model_bundle(verify_hash: bool = False) -> dict[str, Any]:
    required = (CHECKPOINT, MODEL_FOLDER / "dataset.json", MODEL_FOLDER / "plans.json", VOLUME_PRIORS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model bundle files: " + ", ".join(missing))
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8")) if MODEL_MANIFEST.is_file() else {}
    expected = str(manifest.get("checkpoint_sha256") or EXPECTED_CHECKPOINT_SHA256)
    if expected != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Unexpected checkpoint declared by model bundle: {expected}")
    if verify_hash:
        actual = sha256(CHECKPOINT)
        if actual != expected:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual} != {expected}")
    return manifest


def resample_cbct(source: Path, destination: Path) -> dict[str, Any]:
    sitk = _sitk()
    image = sitk.ReadImage(str(source))
    source_size = tuple(int(value) for value in image.GetSize())
    source_spacing = tuple(float(value) for value in image.GetSpacing())
    target_size = tuple(
        max(1, int(round(size * spacing / target_spacing)))
        for size, spacing, target_spacing in zip(source_size, source_spacing, TARGET_SPACING_XYZ)
    )
    resampled = sitk.Resample(
        image,
        target_size,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        TARGET_SPACING_XYZ,
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(resampled, str(destination), True)
    del resampled
    del image
    return {
        "source_size_xyz": source_size,
        "source_spacing_xyz": source_spacing,
        "target_size_xyz": target_size,
        "target_spacing_xyz": TARGET_SPACING_XYZ,
    }


def _cuda_preflight() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R142B U-Mamba2 segmentation model")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError(
            f"GPU compute capability {capability[0]}.{capability[1]} is unsupported by the frozen Mamba2 runtime; "
            "select the Grand Challenge A10G instance"
        )
    return {
        "device": device,
        "name": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def run_segmentation(input_file: Path, output_dir: Path) -> dict[str, Any]:
    cuda = _cuda_preflight()
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": "/tmp/nnunet_raw",
            "nnUNet_preprocessed": "/tmp/nnunet_preprocessed",
            "nnUNet_results": str(NNUNET_RESULTS),
        }
    )
    command = [
        sys.executable,
        "/opt/app/nnunet_predict.py",
        "-i",
        str(input_file.parent),
        "-o",
        str(output_dir),
        "-d",
        "317",
        "-c",
        CONFIGURATION,
        "-p",
        PLANS_NAME,
        "-tr",
        TRAINER_NAME,
        "-f",
        "0",
        "-chk",
        CHECKPOINT_NAME,
        "-npp",
        "1",
        "-nps",
        "1",
        "-step_size",
        "0.7",
        "-device",
        "cuda",
        "--disable_progress_bar",
        "--disable_tta",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"nnU-Net prediction failed with exit code {completed.returncode}")
    return {"cuda": cuda, "wall_seconds": round(time.perf_counter() - started, 4), "command": command}


def _find_label_slices(mask: np.ndarray) -> list[tuple[slice, ...] | None]:
    from scipy import ndimage

    return list(ndimage.find_objects(mask, max_label=46))


def _largest_component(local_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled, component_count = ndimage.label(local_mask, structure=structure)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return np.zeros_like(local_mask, dtype=bool), 0, 0
    largest_label = int(np.argmax(counts[1:]) + 1)
    return labeled == largest_label, int(counts[largest_label]), int(component_count)


def load_volume_priors(path: Path = VOLUME_PRIORS) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    priors = {str(fdi): float(values["median"]) for fdi, values in payload.get("volume_prior", {}).items()}
    missing = sorted(set(FDI_SLOTS) - set(priors), key=int)
    if missing:
        raise ValueError(f"Missing R117 volume priors: {missing}")
    return priors


def size_guard_threshold(fdi: str, median_voxels: float) -> float:
    if fdi in THIRD_MOLARS:
        return max(900.0, 0.20 * median_voxels)
    return max(600.0, 0.05 * median_voxels)


def apply_r117_guard(mask: np.ndarray, priors: dict[str, float]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    guarded = mask.copy()
    label_slices = _find_label_slices(mask)
    audits: list[dict[str, Any]] = []
    for fdi in sorted(FDI_SLOTS, key=int):
        label_id = FDI_TO_CLASS[fdi]
        label_slice = label_slices[label_id - 1] if label_id - 1 < len(label_slices) else None
        threshold = size_guard_threshold(fdi, priors[fdi])
        if label_slice is None:
            audits.append(
                {
                    "fdi": fdi,
                    "raw_voxels": 0,
                    "largest_component_voxels": 0,
                    "component_count": 0,
                    "threshold_voxels": threshold,
                    "guard_pass": 0,
                }
            )
            continue
        local_raw = mask[label_slice] == label_id
        local_kept, largest_voxels, component_count = _largest_component(local_raw)
        guard_pass = largest_voxels >= threshold
        local_guarded = guarded[label_slice]
        local_guarded[local_raw] = 0
        if guard_pass:
            local_guarded[local_kept] = label_id
        audits.append(
            {
                "fdi": fdi,
                "raw_voxels": int(local_raw.sum()),
                "largest_component_voxels": largest_voxels,
                "component_count": component_count,
                "threshold_voxels": threshold,
                "guard_pass": int(guard_pass),
            }
        )
    return guarded, audits


def tooth_centroids(mask: np.ndarray) -> dict[str, np.ndarray]:
    label_slices = _find_label_slices(mask)
    output: dict[str, np.ndarray] = {}
    for fdi in sorted(FDI_SLOTS, key=int):
        label_id = FDI_TO_CLASS[fdi]
        label_slice = label_slices[label_id - 1] if label_id - 1 < len(label_slices) else None
        if label_slice is None:
            continue
        local_coords = np.argwhere(mask[label_slice] == label_id)
        if local_coords.shape[0] < 16:
            continue
        offset = np.asarray([axis.start for axis in label_slice], dtype=np.float64)
        output[fdi] = local_coords.mean(axis=0) + offset
    return output


def expected_slot_centroids(
    centroids: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, str]]:
    expected = {
        fdi: (centroid, "observed_tooth_centroid") for fdi, centroid in centroids.items()
    }
    for order in ARCH_SLOT_ORDER.values():
        present_indices = [index for index, fdi in enumerate(order) if fdi in centroids]
        for index, fdi in enumerate(order):
            if fdi in expected:
                continue
            left = max((value for value in present_indices if value < index), default=None)
            right = min((value for value in present_indices if value > index), default=None)
            if left is None or right is None:
                continue
            weight = float(index - left) / float(right - left)
            point = centroids[order[left]] * (1.0 - weight) + centroids[order[right]] * weight
            expected[fdi] = (point, f"interpolated:{order[left]}-{order[right]}")
    return expected


def _component_records(
    mask: np.ndarray,
    label_ids: tuple[int, ...],
    min_voxels: int,
    connectivity: int = 6,
) -> list[dict[str, Any]]:
    from scipy import ndimage

    coordinates = np.argwhere(np.isin(mask, label_ids))
    if coordinates.shape[0] < min_voxels:
        return []
    lo = coordinates.min(axis=0)
    hi = coordinates.max(axis=0) + 1
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi))
    local = np.isin(mask[slices], label_ids)
    structure = (
        np.ones((3, 3, 3), dtype=np.uint8)
        if connectivity == 26
        else ndimage.generate_binary_structure(rank=3, connectivity=1)
    )
    labeled, count = ndimage.label(local, structure=structure)
    components: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        local_coords = np.argwhere(labeled == component_id)
        if local_coords.shape[0] >= min_voxels:
            component_lo = local_coords.min(axis=0)
            component_hi = local_coords.max(axis=0) + 1
            component_slices = tuple(
                slice(int(lo[axis] + component_lo[axis]), int(lo[axis] + component_hi[axis]))
                for axis in range(3)
            )
            local_slices = tuple(
                slice(int(component_lo[axis]), int(component_hi[axis])) for axis in range(3)
            )
            components.append(
                {
                    "voxels": int(local_coords.shape[0]),
                    "centroid": local_coords.mean(axis=0) + lo,
                    "slices": component_slices,
                    "local_mask": labeled[local_slices] == component_id,
                }
            )
    components.sort(key=lambda item: int(item["voxels"]), reverse=True)
    return components[:16]


def _component_statistics(mask: np.ndarray, label_ids: tuple[int, ...], min_voxels: int) -> list[tuple[int, np.ndarray]]:
    return [
        (int(component["voxels"]), np.asarray(component["centroid"], dtype=np.float64))
        for component in _component_records(mask, label_ids, min_voxels)
    ]


def _component_surface_candidates(
    mask: np.ndarray,
    component: dict[str, Any],
    fdis: Iterable[str],
    margin_mm: float,
) -> list[dict[str, Any]]:
    from scipy import ndimage

    pad = max(1, int(np.ceil(margin_mm / TARGET_SPACING_XYZ[0])))
    component_slices = component["slices"]
    crop_slices = tuple(
        slice(max(0, axis_slice.start - pad), min(mask.shape[axis], axis_slice.stop + pad))
        for axis, axis_slice in enumerate(component_slices)
    )
    crop_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in crop_slices)
    object_crop = np.zeros(crop_shape, dtype=bool)
    insert = tuple(
        slice(
            component_slices[axis].start - crop_slices[axis].start,
            component_slices[axis].stop - crop_slices[axis].start,
        )
        for axis in range(3)
    )
    object_crop[insert] = component["local_mask"]
    distance = ndimage.distance_transform_edt(
        ~object_crop, sampling=tuple(reversed(TARGET_SPACING_XYZ))
    )
    segmentation_crop = mask[crop_slices]
    candidates: list[dict[str, Any]] = []
    for fdi in fdis:
        tooth = segmentation_crop == FDI_TO_CLASS[fdi]
        tooth_voxels = int(tooth.sum())
        if tooth_voxels == 0:
            continue
        values = distance[tooth]
        candidates.append(
            {
                "fdi": fdi,
                "surface_distance_mm": float(values.min()),
                "near_1mm_voxels": int(np.count_nonzero(values <= 1.0)),
                "near_2mm_voxels": int(np.count_nonzero(values <= 2.0)),
            }
        )
    return candidates


def _component_arch(component: dict[str, Any], slots: dict[str, tuple[np.ndarray, str]]) -> str:
    centroid = np.asarray(component["centroid"], dtype=np.float64)
    nearest = min(
        slots,
        key=lambda fdi: float(np.linalg.norm(centroid - slots[fdi][0])),
        default="",
    )
    return arch_for_fdi(nearest) if nearest else ""


def assign_object_teeth(
    mask: np.ndarray,
    label_ids: tuple[int, ...],
    centroids: dict[str, np.ndarray],
    limit_per_component: int,
    min_voxels: int,
) -> list[str]:
    return _assign_components_to_teeth(
        _component_statistics(mask, label_ids, min_voxels), centroids, limit_per_component
    )


def _assign_components_to_teeth(
    components: list[tuple[int, np.ndarray]],
    centroids: dict[str, np.ndarray],
    limit_per_component: int,
) -> list[str]:
    assigned: set[str] = set()
    for _, centroid in components:
        distances = sorted(
            ((float(np.linalg.norm(centroid - tooth_centroid)), int(fdi), fdi) for fdi, tooth_centroid in centroids.items()),
            key=lambda item: (item[0], item[1]),
        )
        assigned.update(fdi for _, _, fdi in distances[:limit_per_component])
    return sorted(assigned, key=int)


def _assign_implant_relations(
    components: list[dict[str, Any]],
    centroids: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not components:
        return [], []
    if not policy.get("promoted"):
        legacy = [(int(row["voxels"]), np.asarray(row["centroid"])) for row in components]
        return _assign_components_to_teeth(legacy, centroids, 1), []
    from scipy.optimize import linear_sum_assignment

    slots = expected_slot_centroids(centroids)
    if not slots:
        return [], []
    slot_fdis = sorted(slots, key=int)
    costs = np.full((len(components), len(slot_fdis)), 1e12, dtype=np.float64)
    for component_index, component in enumerate(components):
        arch = _component_arch(component, slots)
        centroid = np.asarray(component["centroid"], dtype=np.float64)
        for slot_index, fdi in enumerate(slot_fdis):
            if arch_for_fdi(fdi) != arch:
                continue
            point, source = slots[fdi]
            distance_mm = float(np.linalg.norm(centroid - point) * TARGET_SPACING_XYZ[0])
            observed = not str(source).startswith("interpolated:")
            costs[component_index, slot_index] = distance_mm + (
                float(policy["observed_slot_penalty_mm"]) if observed else 0.0
            )
    component_indices, slot_indices = linear_sum_assignment(costs)
    assigned: list[str] = []
    audits: list[dict[str, Any]] = []
    for component_index, slot_index in zip(component_indices, slot_indices):
        if costs[int(component_index), int(slot_index)] >= 1e11:
            continue
        fdi = slot_fdis[int(slot_index)]
        assigned.append(fdi)
        audits.append(
            {
                "component_rank": int(component_index) + 1,
                "fdi": fdi,
                "score_cost": round(float(costs[int(component_index), int(slot_index)]), 4),
                "slot_source": slots[fdi][1],
                "relation": "IMPLANT_AT",
            }
        )
    return sorted(set(assigned), key=int), audits


def _assign_crown_relations(
    mask: np.ndarray,
    components: list[dict[str, Any]],
    centroids: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not components:
        return [], []
    if not policy.get("promoted"):
        legacy = [(int(row["voxels"]), np.asarray(row["centroid"])) for row in components]
        return _assign_components_to_teeth(legacy, centroids, 1), []
    slots = expected_slot_centroids(centroids)
    assigned: set[str] = set()
    audits: list[dict[str, Any]] = []
    for component_index, component in enumerate(components, start=1):
        arch = _component_arch(component, slots)
        candidates = _component_surface_candidates(
            mask,
            component,
            (fdi for fdi in centroids if arch_for_fdi(fdi) == arch),
            6.0,
        )
        candidates = [
            row
            for row in candidates
            if row["surface_distance_mm"] <= float(policy["surface_threshold_mm"])
            and row["near_1mm_voxels"] >= int(policy["min_contact_voxels"])
        ]
        maximum_near_2mm = max(
            (int(row["near_2mm_voxels"]) for row in candidates), default=0
        )
        if maximum_near_2mm > 0:
            candidates = [
                row
                for row in candidates
                if int(row["near_2mm_voxels"])
                >= float(policy["min_relative_near_2mm"]) * maximum_near_2mm
            ]
        candidates.sort(
            key=lambda row: (
                row["surface_distance_mm"],
                -row["near_2mm_voxels"],
                int(row["fdi"]),
            )
        )
        selected = candidates[: int(policy["max_relations"])]
        if not selected and policy.get("centroid_fallback"):
            legacy = _assign_components_to_teeth(
                [(int(component["voxels"]), np.asarray(component["centroid"]))], centroids, 1
            )
            selected = [{"fdi": fdi, "surface_distance_mm": None, "near_1mm_voxels": 0} for fdi in legacy]
        for row in selected:
            assigned.add(str(row["fdi"]))
            audits.append(
                {
                    "component_rank": component_index,
                    "fdi": str(row["fdi"]),
                    "surface_distance_mm": row["surface_distance_mm"],
                    "near_1mm_voxels": row["near_1mm_voxels"],
                    "relation": "COVERS",
                }
            )
    return sorted(assigned, key=int), audits


def _assign_bridge_relations(
    mask: np.ndarray,
    components: list[dict[str, Any]],
    centroids: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not components:
        return [], []
    if not policy.get("promoted"):
        legacy = [(int(row["voxels"]), np.asarray(row["centroid"])) for row in components]
        return _assign_components_to_teeth(legacy, centroids, 3), []
    slots = expected_slot_centroids(centroids)
    assigned: set[str] = set()
    audits: list[dict[str, Any]] = []
    for component_index, component in enumerate(components, start=1):
        arch = _component_arch(component, slots)
        candidates = _component_surface_candidates(
            mask,
            component,
            (fdi for fdi in centroids if arch_for_fdi(fdi) == arch),
            6.0,
        )
        candidates = [
            row
            for row in candidates
            if row["surface_distance_mm"] <= float(policy["surface_threshold_mm"])
            and row["near_2mm_voxels"] >= 4
        ]
        candidates.sort(
            key=lambda row: (
                row["surface_distance_mm"],
                -row["near_2mm_voxels"],
                int(row["fdi"]),
            )
        )
        selected = candidates[: int(policy["max_contacts"])]
        if len(selected) < 2 and policy.get("centroid_fallback"):
            nearest = _assign_components_to_teeth(
                [(int(component["voxels"]), np.asarray(component["centroid"]))], centroids, 2
            )
            selected = [{"fdi": fdi, "surface_distance_mm": None} for fdi in nearest]
        anchors = {str(row["fdi"]) for row in selected}
        span = ordered_bridge_span(anchors)
        if not span:
            continue
        assigned.update(span)
        audits.append(
            {
                "component_rank": component_index,
                "abutment_teeth": order_fdis(anchors),
                "span_teeth": span,
                "pontic_candidates": [fdi for fdi in span if fdi not in anchors],
                "relation": "ABUTS_PONTIC_AT_SPANS",
            }
        )
    return sorted(assigned, key=int), audits


def extract_direct_objects(
    mask: np.ndarray, centroids: dict[str, np.ndarray]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    assignments: dict[str, list[str]] = {}
    object_audits: dict[str, Any] = {}
    for object_name in ("implant", "crown", "bridge"):
        policy = DIRECT_OBJECT_POLICY[object_name]
        relation_policy = PROSTHETIC_RELATION_POLICY[object_name]
        components = _component_records(
            mask,
            LABEL_GROUPS[object_name],
            1,
            connectivity=26 if relation_policy.get("promoted") else 6,
        )
        emitted = [
            component
            for component in components
            if int(component["voxels"]) >= int(policy["min_component_voxels"])
        ]
        if object_name == "implant":
            assignment, relation_audit = _assign_implant_relations(emitted, centroids, relation_policy)
        elif object_name == "crown":
            assignment, relation_audit = _assign_crown_relations(mask, emitted, centroids, relation_policy)
        else:
            assignment, relation_audit = _assign_bridge_relations(mask, emitted, centroids, relation_policy)
        assignments[object_name] = assignment
        object_audits[object_name] = {
            "source": policy["source"],
            "calibrated_threshold_voxels": policy["calibrated_threshold_voxels"],
            "effective_min_component_voxels": policy["min_component_voxels"],
            "relation_policy": relation_policy,
            "raw_component_count": len(components),
            "emitted_component_count": len(emitted),
            "relations": relation_audit,
            "components": [
                {
                    "rank": rank,
                    "voxels": int(component["voxels"]),
                    "centroid_zyx": [
                        round(float(value), 4) for value in component["centroid"]
                    ],
                    "guard_pass": int(
                        int(component["voxels"]) >= int(policy["min_component_voxels"])
                    ),
                }
                for rank, component in enumerate(components, start=1)
            ],
        }
    return assignments, {
        "schema_version": "r165_geometry_calibrated_prosthetic_relation_v1",
        "component_connectivity": {
            label: 26 if policy.get("promoted") else 6
            for label, policy in PROSTHETIC_RELATION_POLICY.items()
        },
        "objects": object_audits,
    }


def segmentation_slot_states(present: set[str]) -> dict[str, list[str]]:
    not_detected: set[str] = set()
    by_quadrant = {quadrant: {fdi for fdi in present if fdi[0] == quadrant} for quadrant in "1234"}
    for quadrant, quadrant_present in by_quadrant.items():
        if len(quadrant_present) >= 2:
            not_detected.update({f"{quadrant}{position}" for position in "12345678"} - quadrant_present)
    visible = present | not_detected
    return {
        "visible_slots": sorted(visible, key=int),
        "outside_fov_slots": [],
        "unknown_visibility_slots": sorted(set(FDI_SLOTS) - visible, key=int),
        "tooth_seen_slots": sorted(present, key=int),
        "no_tooth_seen_slots": sorted(not_detected, key=int),
        "missing_candidates": sorted(not_detected, key=int),
    }


def extract_direct_evidence(mask: np.ndarray, priors: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    guarded, guard_audit = apply_r117_guard(mask, priors)
    centroids = tooth_centroids(guarded)
    present = set(centroids)
    slot_states = segmentation_slot_states(present)
    unique_labels = {int(value) for value in np.unique(guarded)}
    case_labels: set[str] = set()
    if unique_labels.intersection(LABEL_GROUPS["canal"]):
        case_labels.update({"mandibular_canal", "canal_regular"})
    if unique_labels.intersection(LABEL_GROUPS["sinus"]):
        case_labels.add("sinus")
    if unique_labels.intersection(LABEL_GROUPS["jawbone"]):
        case_labels.add("fov_included")
    if not unique_labels.difference({0}):
        case_labels.add("fov_not_included")

    object_assignments, direct_object_audit = extract_direct_objects(guarded, centroids)
    implants = object_assignments["implant"]
    crowns = object_assignments["crown"]
    bridge_members = object_assignments["bridge"]
    if implants:
        case_labels.add("implant")
    if crowns:
        case_labels.add("crown")
    if bridge_members:
        case_labels.add("bridge")
    direct = {
        "teeth_present": sorted(present, key=int),
        "segmentation_tooth_absence": slot_states["missing_candidates"],
        "segmentation_visible_slots": slot_states["visible_slots"],
        "segmentation_outside_fov_slots": [],
        "segmentation_unknown_visibility_slots": slot_states["unknown_visibility_slots"],
        "segmentation_implant": implants,
        "segmentation_crown": crowns,
        "segmentation_bridge": bridge_members,
        "direct_case_labels": sorted(case_labels),
        "slot_state_candidates": slot_states,
        "direct_object_component_audit": direct_object_audit,
    }
    del guarded
    return direct, guard_audit


def _values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(";") if item.strip()}
    if isinstance(value, Iterable):
        return {str(item) for item in value if str(item)}
    return set()


def arch_for_fdi(fdi: str) -> str:
    return "upper" if fdi[0] in "12" else "lower"


def ordered_bridge_span(members: set[str]) -> list[str]:
    if len(members) < 2 or len({arch_for_fdi(fdi) for fdi in members}) != 1:
        return []
    arch = arch_for_fdi(next(iter(members)))
    order = ARCH_SLOT_ORDER[arch]
    indices = [order.index(fdi) for fdi in members]
    return list(order[min(indices) : max(indices) + 1])


def order_fdis(fdis: set[str]) -> list[str]:
    if not fdis:
        return []
    if len({arch_for_fdi(fdi) for fdi in fdis}) == 1:
        return [fdi for fdi in ARCH_SLOT_ORDER[arch_for_fdi(next(iter(fdis)))] if fdi in fdis]
    return sorted(fdis, key=int)


def build_writer_evidence(direct: dict[str, Any], case_id: str = "hidden") -> dict[str, Any]:
    present = _values(direct.get("teeth_present"))
    not_detected = _values(direct.get("segmentation_tooth_absence"))
    implants = _values(direct.get("segmentation_implant"))
    crowns = _values(direct.get("segmentation_crown"))
    bridge_members = _values(direct.get("segmentation_bridge"))
    case_labels = _values(direct.get("direct_case_labels"))
    coverage_guard = present.isdisjoint(not_detected) and present | not_detected == set(FDI_SLOTS)
    fov_guard = "fov_included" in case_labels and "fov_not_included" not in case_labels

    absence = sorted(not_detected, key=int) if coverage_guard and fov_guard else []
    eligible_crowns = sorted(crowns & (present | implants), key=int)
    span_fdis = ordered_bridge_span(bridge_members)
    abutments = order_fdis(bridge_members & (present | implants))
    pontics = order_fdis(bridge_members & not_detected)
    unresolved = order_fdis(bridge_members - present - implants - not_detected)
    bridge_eligible = bool(span_fdis and len(abutments) >= 2 and not unresolved)

    direct_case_labels: list[str] = []
    for label in ("mandibular_canal", "canal_regular", "sinus", "fov_included", "fov_not_included"):
        if label not in case_labels:
            continue
        if label == "canal_regular" and "mandibular_canal" not in case_labels:
            continue
        if label == "sinus" and not fov_guard:
            continue
        direct_case_labels.append(label)
    direct_case_labels.sort()

    ready: dict[str, Any] = {
        "tooth_absence": absence,
        "implant": sorted(implants, key=int),
        "crown_oracle": eligible_crowns,
    }
    if bridge_eligible:
        ready["bridge"] = {
            "involved_teeth": span_fdis,
            "abutment_teeth": abutments,
            "pontic_teeth": pontics,
            "weak_span_candidates": [
                {
                    "source": "R138_object_identity_graph",
                    "span_teeth": span_fdis,
                    "pontic_or_missing_candidates": pontics,
                }
            ],
        }
    positive_labels = set(direct_case_labels)
    for label, values in (("tooth_absence", absence), ("implant", implants), ("crown", eligible_crowns)):
        if values:
            positive_labels.add(label)
    if bridge_eligible:
        positive_labels.add("bridge")
    return {
        "case_id": case_id,
        "positive_case_labels": sorted(positive_labels),
        "report_ready_evidence": ready,
        "image_predicted_case_evidence": {
            "present_teeth": sorted(present, key=int),
            "direct_case_labels": direct_case_labels,
        },
        "emission_policy": "calibrated_fail_closed_writer_projection",
        "writer_ready_fact_count": (
            len(absence)
            + len(implants)
            + len(eligible_crowns)
            + (len(span_fdis) if bridge_eligible else 0)
            + len(direct_case_labels)
        ),
    }


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def sorted_teeth(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=int)


def format_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def sentence_teeth(prefix: str, teeth: Iterable[str], suffix: str = "") -> str:
    ordered = sorted_teeth(teeth)
    if not ordered:
        return ""
    word = "tooth" if len(ordered) == 1 else "teeth"
    tail = f" {suffix.strip()}" if suffix.strip() else ""
    return clean_sentence(f"{prefix} {word} {format_list(ordered)}{tail}")


def _jaw_of(fdi: str) -> str:
    return "maxilla" if fdi[0] in "12" else "mandible"


def _bridge_sentences(ready: dict[str, Any]) -> list[str]:
    bridge = ready.get("bridge")
    if not isinstance(bridge, dict):
        return []
    output: list[str] = []
    involved = sorted_teeth(bridge.get("involved_teeth", []))
    if involved:
        output.append(sentence_teeth("A fixed prosthetic bridge involving", involved))
    for candidate in bridge.get("weak_span_candidates", []):
        span = sorted_teeth(candidate.get("span_teeth", [])) if isinstance(candidate, dict) else []
        if len(span) > 1:
            output.append(clean_sentence(f"Bridge span from {format_list(span)}"))
    return output


def _jaw_sentences(pack: dict[str, Any], jaw: str) -> list[str]:
    ready = dict(pack.get("report_ready_evidence") or {})
    selected = lambda key: [fdi for fdi in sorted_teeth(ready.get(key, [])) if _jaw_of(fdi) == jaw]
    output: list[str] = []
    absence = selected("tooth_absence")
    if absence:
        if len(absence) >= 2:
            region = "mandibular" if jaw == "mandible" else "maxillary"
            output.append(clean_sentence(f"Partial edentulism is present in the {region} arch"))
        output.append(sentence_teeth("Absence of", absence))
    implants = selected("implant")
    if implants:
        output.append(sentence_teeth("Presence of an endosseous implant in position", implants))
    crowns = selected("crown_oracle")
    if crowns:
        output.append(sentence_teeth("Prosthetic crown is noted on", crowns))
    for sentence in _bridge_sentences(ready):
        fdis = re.findall(r"(?<!\d)([1-4][1-8])(?!\d)", sentence)
        if fdis and all(_jaw_of(fdi) == jaw for fdi in fdis):
            output.append(sentence)
    return output


def generate_report(pack: dict[str, Any]) -> str:
    labels = set(pack.get("positive_case_labels", []))
    mandible = _jaw_sentences(pack, "mandible")
    maxilla = _jaw_sentences(pack, "maxilla")
    lines: list[str] = []
    if mandible:
        lines.append("Mandible: " + " ".join(mandible))
    if "mandibular_canal" in labels:
        if "canal_regular" in labels:
            lines.append("Mandibular canal: The course and foraminal emergence of the inferior alveolar canals are regular.")
        else:
            lines.append("Mandibular canal: The mandibular canal is described in the report.")
    if maxilla:
        lines.append("Maxilla: " + " ".join(maxilla))
    if "sinus" in labels:
        lines.append(
            "Maxillary sinus: The maxillary sinuses are included as far as can be assessed from the available scan."
        )
    if "fov_not_included" in labels:
        lines.append(
            "Scan volume: The maxilla or mandibular condyles may be partially included or not included in the scan volume. "
            "Mandibular condyles are not included in the acquisition field when outside the scan volume."
        )
    elif "fov_included" in labels:
        lines.append("Scan volume: The examined volume provides assessable coverage for the reported dental and osseous findings.")
    return "\n".join(lines).strip() or SAFE_EMPTY_REPORT


def execute_case(input_path: Path, case_id: str, tmp_root: Path) -> tuple[str, dict[str, Any]]:
    prepare_runtime_dirs(tmp_root)
    verify_model_bundle(verify_hash=True)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    resampled_path = tmp_root / "input" / f"{case_id}_0000.nii.gz"
    resample_started = time.perf_counter()
    geometry = resample_cbct(input_path, resampled_path)
    timings["resample_seconds"] = round(time.perf_counter() - resample_started, 4)

    segmentation = run_segmentation(resampled_path, tmp_root / "prediction")
    timings["segmentation_seconds"] = segmentation["wall_seconds"]
    prediction_path = tmp_root / "prediction" / f"{case_id}.nii.gz"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"nnU-Net did not produce {prediction_path}")

    evidence_started = time.perf_counter()
    sitk = _sitk()
    prediction_image = sitk.ReadImage(str(prediction_path))
    mask = sitk.GetArrayFromImage(prediction_image).astype(np.int16, copy=False)
    direct, guard_audit = extract_direct_evidence(mask, load_volume_priors())
    pack = build_writer_evidence(direct, case_id)
    report = generate_report(pack)
    timings["evidence_writer_seconds"] = round(time.perf_counter() - evidence_started, 4)
    timings["total_seconds"] = round(time.perf_counter() - started, 4)
    audit = {
        "case_id": case_id,
        "geometry": geometry,
        "cuda": segmentation["cuda"],
        "timings": timings,
        "direct_evidence": direct,
        "writer_evidence": pack,
        "r117_guard_pass_count": sum(int(row["guard_pass"]) for row in guard_audit),
        "report_characters": len(report),
    }
    del mask
    shutil.rmtree(tmp_root / "prediction", ignore_errors=True)
    return report, audit
