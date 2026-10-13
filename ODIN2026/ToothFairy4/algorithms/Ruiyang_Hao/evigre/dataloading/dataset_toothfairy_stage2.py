"""Shared schema and tensor utilities for the released Stage 2 branch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import blosc2
import numpy as np
import torch
import torch.nn.functional as F


GLOBAL_LABELS = [
    "maxilla_partially_included",
    "maxilla_not_included",
    "maxillary_sinuses_not_included",
    "mandibular_condyles_not_included",
    "periodontitis_present",
    "atrophy_maxilla",
    "atrophy_mandible",
    "atrophy_severe",
    "osteolytic_lesion_present",
    "osteosclerotic_lesion_present",
    "sinus_pathology_present",
    "poor_quality_or_artifact",
]

TOOTH_LABELS = [
    "missing",
    "implant",
    "crown",
    "restoration",
    "endodontic_treatment",
    "prosthesis_related",
    "impacted_or_disodontiasis",
    "caries",
    "root_remnant",
    "root_fracture",
    "lesion_related",
]

RELATION_LABELS = [
    "38_close_to_left_IAC",
    "48_close_to_right_IAC",
    "any_tooth_close_to_IAC",
    "lesion_close_to_IAC",
    "left_IAC_predominantly_lingual",
    "left_IAC_predominantly_buccal",
    "right_IAC_predominantly_lingual",
    "right_IAC_predominantly_buccal",
]

FDI_TEETH = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]


def fdi_to_compact_label(tooth_id: str) -> int | None:
    value = int(tooth_id)
    if 11 <= value <= 18:
        return value
    if 21 <= value <= 28:
        return value - 2
    if 31 <= value <= 38:
        return value - 4
    if 41 <= value <= 48:
        return value - 6
    return None


FDI_TO_COMPACT = {tooth_id: fdi_to_compact_label(tooth_id) for tooth_id in FDI_TEETH}
TOOTH_COMPACT_LABELS = np.asarray(
    [value for value in FDI_TO_COMPACT.values() if value is not None], dtype=np.uint8
)
LEFT_LOWER_TOOTH_LABELS = np.asarray(
    [FDI_TO_COMPACT[tooth] for tooth in FDI_TEETH if tooth.startswith("3")], dtype=np.uint8
)
RIGHT_LOWER_TOOTH_LABELS = np.asarray(
    [FDI_TO_COMPACT[tooth] for tooth in FDI_TEETH if tooth.startswith("4")], dtype=np.uint8
)
MANDIBULAR_CANAL_LABELS = np.asarray([3, 4], dtype=np.uint8)
MAXILLARY_SINUS_LABELS = np.asarray([5, 6], dtype=np.uint8)
BRIDGE_LABEL = 8
CROWN_LABEL = 9
IMPLANT_LABEL = 10


def parse_size(value: Sequence[int] | str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if value is None:
        return default
    parts = [int(item) for item in value.split(",")] if isinstance(value, str) else [int(item) for item in value]
    if len(parts) != 3:
        raise ValueError(f"Expected a 3D size, got {value!r}")
    return parts[0], parts[1], parts[2]


def read_blosc2(path: str | Path) -> np.ndarray:
    return np.asarray(blosc2.open(str(path), mmap_mode="r")).copy()


def path_exists(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null"} and Path(text).exists())


def normalize_cbct(array: np.ndarray) -> np.ndarray:
    array = np.clip(array.astype(np.float32, copy=False), -997.0, 3558.0)
    return (array - 809.5) / 1014.369


def metal_window(array: np.ndarray) -> np.ndarray:
    """Retain continuous variation in the high-density material range."""
    array = np.clip(array.astype(np.float32, copy=False), 3558.0, 16000.0)
    return (array - 3558.0) / (16000.0 - 3558.0)


def resize_3d_tensor(
    tensor: torch.Tensor,
    size: Tuple[int, int, int],
    mode: str = "trilinear",
) -> torch.Tensor:
    if tuple(tensor.shape[-3:]) == tuple(size):
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor.unsqueeze(0), size=size, mode="nearest")[0]
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, align_corners=False)[0]


def load_npz_crop(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(path)) as archive:
        ct = archive["ct"] if "ct" in archive else archive[archive.files[0]]
        mask = archive["mask"] if "mask" in archive else np.zeros_like(ct, dtype=np.uint8)
    return ct, mask


def semantic_mask_channels(
    mask: np.ndarray,
    target_tooth_id: str | None = None,
    *,
    include_prosthetic_channels: bool = False,
) -> torch.Tensor:
    mask_u8 = mask.astype(np.uint8, copy=False)
    teeth = np.isin(mask_u8, TOOTH_COMPACT_LABELS)
    selected = np.zeros(mask_u8.shape, dtype=bool)
    if target_tooth_id == "lower_left_teeth":
        selected = np.isin(mask_u8, LEFT_LOWER_TOOTH_LABELS)
    elif target_tooth_id == "lower_right_teeth":
        selected = np.isin(mask_u8, RIGHT_LOWER_TOOTH_LABELS)
    elif target_tooth_id:
        compact = FDI_TO_COMPACT.get(str(target_tooth_id))
        if compact is not None:
            selected = mask_u8 == compact

    canal = np.isin(mask_u8, MANDIBULAR_CANAL_LABELS)
    sinus = np.isin(mask_u8, MAXILLARY_SINUS_LABELS)
    if include_prosthetic_channels:
        implant = mask_u8 == IMPLANT_LABEL
        crown = mask_u8 == CROWN_LABEL
        bridge = mask_u8 == BRIDGE_LABEL
        known = teeth | canal | sinus | implant | crown | bridge
        channels = [teeth, selected, canal, sinus, implant, crown, bridge, (mask_u8 > 0) & ~known]
    else:
        known = teeth | canal | sinus
        channels = [teeth, selected, canal, sinus, (mask_u8 > 0) & ~known]
    return torch.from_numpy(np.stack(channels, axis=0).astype(np.float32)).float()


def mask_tensor(
    mask: np.ndarray,
    size: Tuple[int, int, int],
    *,
    mask_mode: str = "semantic",
    target_tooth_id: str | None = None,
    include_prosthetic_channels: bool = False,
) -> torch.Tensor:
    if mask_mode == "binary":
        channels = torch.from_numpy((mask.astype(np.float32) > 0)[None].astype(np.float32))
    elif mask_mode == "semantic":
        channels = semantic_mask_channels(
            mask,
            target_tooth_id,
            include_prosthetic_channels=include_prosthetic_channels,
        )
    else:
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}")
    return resize_3d_tensor(channels.float(), size, mode="nearest")


__all__ = [
    "FDI_TEETH",
    "FDI_TO_COMPACT",
    "GLOBAL_LABELS",
    "RELATION_LABELS",
    "TOOTH_LABELS",
    "load_npz_crop",
    "mask_tensor",
    "metal_window",
    "normalize_cbct",
    "parse_size",
    "path_exists",
    "read_blosc2",
    "resize_3d_tensor",
    "semantic_mask_channels",
]
