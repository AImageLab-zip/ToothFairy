#!/usr/bin/env python3
"""Prepare the exact Stage-2 tensors used by the released single-case inference path.

This module contains inference geometry only. It does not construct supervision,
supervision construction, dataset splitting, or experiment statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import blosc2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_closing, generate_binary_structure, label as ndi_label, map_coordinates

GLOBAL_LABELS = [
    "maxilla_partially_included", "maxilla_not_included", "maxillary_sinuses_not_included",
    "mandibular_condyles_not_included", "periodontitis_present", "atrophy_maxilla",
    "atrophy_mandible", "atrophy_severe", "osteolytic_lesion_present",
    "osteosclerotic_lesion_present", "sinus_pathology_present", "poor_quality_or_artifact",
]
TOOTH_LABELS = [
    "missing", "implant", "crown", "restoration", "endodontic_treatment",
    "prosthesis_related", "impacted_or_disodontiasis", "caries", "root_remnant",
    "root_fracture", "lesion_related",
]
RELATION_LABELS = [
    "38_close_to_left_IAC", "48_close_to_right_IAC", "any_tooth_close_to_IAC",
    "lesion_close_to_IAC", "left_IAC_predominantly_lingual", "left_IAC_predominantly_buccal",
    "right_IAC_predominantly_lingual", "right_IAC_predominantly_buccal",
]
RELATION_CROPS = ["left_lower_jawbone_region", "right_lower_jawbone_region"]
FDI_TEETH = [
    "11", "12", "13", "14", "15", "16", "17", "18", "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38", "41", "42", "43", "44", "45", "46", "47", "48",
]

def _fdi_to_compact(tooth_id: str) -> Optional[int]:
    t = int(tooth_id)
    if 11 <= t <= 18: return t
    if 21 <= t <= 28: return t - 2
    if 31 <= t <= 38: return t - 4
    if 41 <= t <= 48: return t - 6
    return None

FDI_TO_COMPACT = {t: _fdi_to_compact(t) for t in FDI_TEETH}
COMPACT_TO_FDI = {v: k for k, v in FDI_TO_COMPACT.items() if v is not None}
TOOTH_COMPACT_LABELS = sorted(COMPACT_TO_FDI)
LOWER_JAWBONE_LABEL = 1
LEFT_IAC_LABEL = 3
RIGHT_IAC_LABEL = 4
LEFT_LOWER_TOOTH_LABELS = [FDI_TO_COMPACT[t] for t in FDI_TEETH if t.startswith("3") and FDI_TO_COMPACT[t] is not None]
RIGHT_LOWER_TOOTH_LABELS = [FDI_TO_COMPACT[t] for t in FDI_TEETH if t.startswith("4") and FDI_TO_COMPACT[t] is not None]
LEFT_POSTERIOR_RELATION_LABELS = [FDI_TO_COMPACT[t] for t in ("36", "37", "38") if FDI_TO_COMPACT[t] is not None]
RIGHT_POSTERIOR_RELATION_LABELS = [FDI_TO_COMPACT[t] for t in ("46", "47", "48") if FDI_TO_COMPACT[t] is not None]
ORIGINAL_TO_COMPACT_LABEL = {i: i for i in range(0, 19)}
ORIGINAL_TO_COMPACT_LABEL.update({i: i - 2 for i in range(21, 29)})
ORIGINAL_TO_COMPACT_LABEL.update({i: i - 4 for i in range(31, 39)})
ORIGINAL_TO_COMPACT_LABEL.update({i: i - 6 for i in range(41, 49)})
ORIGINAL_TO_COMPACT_LABEL.update({50: 46, 51: 43, 52: 44, 53: 45})
ORIGINAL_ONLY_LABELS = {47, 48, 50, 51, 52, 53}
TOOTH_CROP_SCHEMA = "stage2_tooth_oriented_pca_margin_rawresized_v3_longx_64x64x96_margin6mm"
RELATION_CROP_SCHEMA = "stage2_relation_pred_iac_posterior_physical_fov_rawresized_v8_80x128x128_fov80x96x64"


def parse_size(text: str) -> Tuple[int, int, int]:
    vals = [int(v) for v in str(text).split(',')]
    if len(vals) != 3:
        raise ValueError(f"Expected D,H,W size, got: {text}")
    return vals[0], vals[1], vals[2]

def parse_float_triplet(text: str) -> Tuple[float, float, float]:
    vals = [float(v) for v in str(text).split(',')]
    if len(vals) != 3:
        raise ValueError(f"Expected z,y,x triplet, got: {text}")
    return vals[0], vals[1], vals[2]

def nifti_spacing_zyx(path: Path) -> Tuple[float, float, float]:
    img = sitk.ReadImage(str(path))
    spacing_xyz = img.GetSpacing()
    return float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])

def voxels_from_mm(mm_zyx: Tuple[float, float, float], spacing_zyx: Tuple[float, float, float], *, min_vox: int = 1) -> Tuple[int, int, int]:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    mm = np.asarray(mm_zyx, dtype=np.float64)
    if np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing_zyx: {spacing_zyx}")
    vals = np.ceil(mm / spacing).astype(np.int64)
    vals = np.maximum(vals, int(min_vox))
    return int(vals[0]), int(vals[1]), int(vals[2])

def write_blosc2(path: Path, array: np.ndarray, clevel: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    chunks = tuple(array.shape)
    blocks = tuple(min(int(s), 64) for s in array.shape)
    cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=clevel, typesize=array.dtype.itemsize)
    blosc2.asarray(np.ascontiguousarray(array), urlpath=str(path), chunks=chunks, blocks=blocks, cparams=cparams, mode='w')

def read_blosc2(path: Path) -> np.ndarray:
    return np.asarray(blosc2.open(str(path), mmap_mode='r')).copy()

def cbct_to_int16_checked(array: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(array)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{context}: CBCT array contains non-finite values')
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    info = np.iinfo(np.int16)
    if lo < info.min or hi > info.max:
        raise ValueError(f'{context}: CBCT intensity range [{lo}, {hi}] cannot be represented as int16 without overflow')
    return arr.astype(np.int16, copy=False)

def load_nifti_zyx(path: Path, dtype=np.float32) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(dtype, copy=False)

def load_stage1_mask_zyx(path: Path) -> np.ndarray:
    """Load a Stage-1 mask from either EviDent-CBCT Blosc2 or nnU-Net raw-space NIfTI."""
    if path.name.endswith('.b2nd'):
        return read_blosc2(path).astype(np.uint8, copy=False)
    if path.name.endswith('.nii.gz') or path.suffix in {'.nii', '.mha', '.mhd', '.nrrd'}:
        return load_nifti_zyx(path, np.uint8)
    raise ValueError(f"Unsupported Stage1 mask format: {path}")

def align_mask_to_image(mask: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(image_shape):
        return mask
    transposed = np.transpose(mask, (2, 1, 0))
    if tuple(transposed.shape) == tuple(image_shape):
        return transposed
    raise ValueError(f"Mask shape {mask.shape} cannot be aligned to image shape {image_shape}")

def normalize_stage1_mask_label_space(mask: np.ndarray, label_space: str, case_id: str = '') -> np.ndarray:
    """Normalize Stage1 masks to the internal compact 0..46 FDI-mapped space.

    Stage2 targets are indexed by FDI tooth IDs. The segmentation mask is used only
    to localize each FDI tooth slot and derive conservative missing/relation labels.
    """
    label_space = str(label_space).lower()
    if label_space not in {'compact', 'original', 'auto'}:
        raise ValueError(f"Invalid mask label space: {label_space}. Use compact, original, or auto.")

    unique = {int(v) for v in np.unique(mask)}
    if label_space == 'auto':
        label_space = 'original' if unique & ORIGINAL_ONLY_LABELS else 'compact'

    if label_space == 'compact':
        unknown = sorted(v for v in unique if v < 0 or v > 46)
        if unknown:
            raise ValueError(f"{case_id}: compact Stage1 mask has labels outside 0..46: {unknown[:20]}")
        return mask.astype(np.uint8, copy=False)

    out = np.zeros(mask.shape, dtype=np.uint8)
    for src, dst in ORIGINAL_TO_COMPACT_LABEL.items():
        out[mask == src] = dst
    unknown = sorted(unique - set(ORIGINAL_TO_COMPACT_LABEL.keys()))
    if unknown:
        raise ValueError(f"{case_id}: original Stage1 mask has labels not covered by original->compact mapping: {unknown[:20]}")
    return out

def cleanup_tooth_mask_components(mask: np.ndarray, closing_iterations: int = 1, connectivity: int = 1) -> np.ndarray:
    """Clean every FDI tooth label independently before Stage2 crop generation.

    The Stage1 Task1 postprocess removes tiny label components by physical volume,
    but it does not force each individual tooth to be a single connected object.
    Stage2 tooth crops are indexed by FDI tooth ID, so we close small gaps and keep
    the largest connected component per tooth label. Non-tooth labels are left
    untouched except that closed tooth voxels are not allowed to overwrite them.
    """
    closing_iterations = int(closing_iterations)
    connectivity = int(connectivity)
    if closing_iterations < 0:
        raise ValueError(f"closing_iterations must be >= 0, got {closing_iterations}")
    if connectivity not in {1, 2, 3}:
        raise ValueError(f"connectivity must be 1, 2, or 3, got {connectivity}")
    out = mask.astype(np.uint8, copy=True)
    structure = generate_binary_structure(3, connectivity)
    for label_id in TOOTH_COMPACT_LABELS:
        tooth = mask == int(label_id)
        if not tooth.any():
            continue
        protected = (out != 0) & (out != int(label_id))
        if closing_iterations > 0:
            tooth = binary_closing(tooth, structure=structure, iterations=closing_iterations)
        cc, num = ndi_label(tooth, structure=structure)
        if num > 1:
            counts = np.bincount(cc.ravel())
            counts[0] = 0
            tooth = cc == int(counts.argmax())
        out[mask == int(label_id)] = 0
        out[tooth & ~protected] = int(label_id)
    return out.astype(np.uint8, copy=False)

def tooth_crop_schema_for(size_zyx: Tuple[int, int, int], margin_mm_zyx: Tuple[float, float, float]) -> str:
    if tuple(size_zyx) == (64, 64, 96) and all(abs(float(v) - 6.0) < 1e-6 for v in margin_mm_zyx):
        return TOOTH_CROP_SCHEMA
    return f"stage2_tooth_oriented_pca_margin_rawresized_v3_longx_{_format_crop_size(size_zyx)}_margin{_format_margin_tag(margin_mm_zyx)}"

def tooth_crop_dir_name_for(size_zyx: Tuple[int, int, int], margin_mm_zyx: Tuple[float, float, float]) -> str:
    return f"tooth_crops_oriented_pca_{_format_crop_size(size_zyx)}_margin{_format_margin_tag(margin_mm_zyx)}"

def crop_fixed(arr: np.ndarray, center_zyx: np.ndarray, size_zyx: Tuple[int, int, int], pad_value: float = 0) -> np.ndarray:
    center = np.asarray(center_zyx, dtype=np.float64)
    size = np.asarray(size_zyx, dtype=np.int64)
    lbs = np.floor(center - size / 2.0).astype(int)
    ubs = lbs + size
    out = np.full(tuple(size), pad_value, dtype=arr.dtype)
    src_lbs = np.maximum(lbs, 0)
    src_ubs = np.minimum(ubs, np.asarray(arr.shape))
    dst_lbs = src_lbs - lbs
    dst_ubs = dst_lbs + (src_ubs - src_lbs)
    if np.all(src_ubs > src_lbs):
        out[dst_lbs[0]:dst_ubs[0], dst_lbs[1]:dst_ubs[1], dst_lbs[2]:dst_ubs[2]] = arr[src_lbs[0]:src_ubs[0], src_lbs[1]:src_ubs[1], src_lbs[2]:src_ubs[2]]
    return out

def crop_bounds(arr: np.ndarray, lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
    lbs = np.maximum(np.asarray(lbs, dtype=np.int64), 0)
    ubs = np.minimum(np.asarray(ubs, dtype=np.int64), np.asarray(arr.shape, dtype=np.int64))
    if np.any(ubs <= lbs):
        raise ValueError(f'Invalid crop bounds: lbs={lbs.tolist()}, ubs={ubs.tolist()}, shape={arr.shape}')
    return arr[lbs[0]:ubs[0], lbs[1]:ubs[1], lbs[2]:ubs[2]]

def bbox_bounds(mask_bool: np.ndarray, margin_zyx: Tuple[int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        return None
    margin = np.asarray(margin_zyx, dtype=np.int64)
    lbs = coords.min(axis=0).astype(np.int64) - margin
    ubs = coords.max(axis=0).astype(np.int64) + 1 + margin
    return lbs, ubs

def crop_npz_matches_schema(path: str | Path, expected_schema: str) -> bool:
    try:
        with np.load(str(path), allow_pickle=False) as z:
            if 'crop_schema' not in z:
                return False
            value = z['crop_schema']
            return str(value.item() if getattr(value, 'shape', ()) == () else value.tolist()) == expected_schema
    except Exception:
        return False

def mask_centers(mask_zyx: np.ndarray) -> Dict[str, np.ndarray]:
    centers = {}
    for label_id, tooth_id in COMPACT_TO_FDI.items():
        coords = np.argwhere(mask_zyx == label_id)
        if coords.size:
            centers[tooth_id] = coords.mean(axis=0)
    return centers

def tooth_arch_neighbors(tooth_id: str) -> Tuple[Optional[str], Optional[str]]:
    upper_arch = ["18", "17", "16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26", "27", "28"]
    lower_arch = ["48", "47", "46", "45", "44", "43", "42", "41", "31", "32", "33", "34", "35", "36", "37", "38"]
    arch = upper_arch if tooth_id in upper_arch else lower_arch if tooth_id in lower_arch else []
    if not arch:
        return None, None
    i = arch.index(str(tooth_id))
    prev_id = arch[i - 1] if i > 0 else None
    next_id = arch[i + 1] if i < len(arch) - 1 else None
    return prev_id, next_id

def crop_oriented_pca_labelmap(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    tooth_mask: np.ndarray,
    tooth_id: str,
    tooth_margin_mm: Tuple[float, float, float],
    spacing_zyx: Tuple[float, float, float],
    centers: Dict[str, np.ndarray],
    output_size_zyx: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    coords_zyx = np.argwhere(tooth_mask)
    if coords_zyx.shape[0] < 8:
        raise ValueError('too_few_tooth_voxels_for_pca')

    pts_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)
    center_xyz = pts_xyz.mean(axis=0)
    pts0 = pts_xyz - center_xyz[None, :]

    cov = np.cov(pts0, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    v1 = evecs[:, order[0]].astype(np.float32)
    v2_pca = evecs[:, order[1]].astype(np.float32)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    if float(v1[2]) < 0.0:
        v1 = -v1

    v2_raw = v2_pca.copy()
    c_zyx = centers.get(str(tooth_id))
    prev_id, next_id = tooth_arch_neighbors(str(tooth_id))
    c_prev = centers.get(str(prev_id)) if prev_id is not None else None
    c_next = centers.get(str(next_id)) if next_id is not None else None
    if c_zyx is not None:
        c_xyz = c_zyx[[2, 1, 0]].astype(np.float32)
        if c_prev is not None and c_next is not None:
            v2_raw = c_next[[2, 1, 0]].astype(np.float32) - c_prev[[2, 1, 0]].astype(np.float32)
        elif c_prev is not None:
            v2_raw = c_prev[[2, 1, 0]].astype(np.float32) - c_xyz
        elif c_next is not None:
            v2_raw = c_next[[2, 1, 0]].astype(np.float32) - c_xyz

    v2 = v2_raw - float(np.dot(v2_raw, v1)) * v1
    if np.linalg.norm(v2) < 1e-6:
        tmp = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tmp, v1))) > 0.9:
            tmp = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        v2 = tmp - float(np.dot(tmp, v1)) * v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    if centers:
        mouth_xyz = np.stack([v[[2, 1, 0]] for v in centers.values()], axis=0).astype(np.float32).mean(axis=0)
        if float(np.linalg.norm(center_xyz + v2 * 10.0 - mouth_xyz)) > float(np.linalg.norm(center_xyz - v2 * 10.0 - mouth_xyz)):
            v2 = -v2

    v3 = np.cross(v1, v2)
    v3 = v3 / (np.linalg.norm(v3) + 1e-8)
    basis = np.stack([v1, v2, v3], axis=1).astype(np.float32)

    local = pts0 @ basis
    margin_vox = np.asarray(voxels_from_mm(tooth_margin_mm, spacing_zyx, min_vox=1), dtype=np.float32)
    margin_local = np.full((3,), float(np.mean(margin_vox)), dtype=np.float32)
    lo = local.min(axis=0).astype(np.float32) - margin_local
    hi = local.max(axis=0).astype(np.float32) + margin_local

    def sample_local_crop(size_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        axes = []
        for i, n in enumerate(size_xyz.astype(np.int64)):
            n_int = max(1, int(n))
            if n_int <= 1:
                axes.append(np.asarray([(float(lo[i]) + float(hi[i])) * 0.5], dtype=np.float32))
            else:
                axes.append(np.linspace(float(lo[i]), float(hi[i]), n_int, dtype=np.float32))
        aa, bb, cc = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        local_grid = np.stack([aa, bb, cc], axis=-1)
        xyz = center_xyz[None, None, None, :] + np.einsum('...k,jk->...j', local_grid, basis)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        ct_xyz = map_coordinates(
            image_zyx.astype(np.float32, copy=False),
            [z, y, x],
            order=1,
            mode='constant',
            cval=-1024.0,
        ).astype(np.float32, copy=False)
        mask_xyz = map_coordinates(
            mask_zyx.astype(np.float32, copy=False),
            [z, y, x],
            order=0,
            mode='constant',
            cval=0.0,
        ).astype(np.uint8, copy=False)
        # Stored arrays are z/y/x. Local anatomical axes are v3/v2/v1.
        return np.transpose(ct_xyz, (2, 1, 0)), np.transpose(mask_xyz, (2, 1, 0))

    # The resized tensor is stored in z/y/x order, but its local anatomical axes
    # are local-v3/local-v2/local-v1. v1 is the tooth long axis and is assigned
    # to the output x dimension, so a 64x64x96 crop yields center slices
    # 64x96, 64x96, and 64x64.
    out_size_xyz = np.asarray([output_size_zyx[2], output_size_zyx[1], output_size_zyx[0]], dtype=np.int64)
    raw_size_xyz = np.maximum(1, np.ceil(hi - lo).astype(np.int64))
    ct_raw, mask_raw = sample_local_crop(raw_size_xyz)
    ct_crop, mask_crop = sample_local_crop(out_size_xyz)

    lbs = np.floor(coords_zyx.min(axis=0) - margin_vox).astype(np.int32)
    ubs = np.ceil(coords_zyx.max(axis=0) + 1 + margin_vox).astype(np.int32)
    center_zyx = center_xyz[[2, 1, 0]].astype(np.float32)
    roi_meta = {
        'crop_mode': 'tooth_mask_oriented_pca_long_axis_x_margin_rawresized_v3',
        'basis_xyz': basis.tolist(),
        'basis_columns': ['local_v1_tooth_long_axis_output_x', 'local_v2_arch_axis_output_y', 'local_v3_cross_axis_output_z'],
        'output_axis_order_zyx': ['local_v3_cross_axis', 'local_v2_arch_axis', 'local_v1_tooth_long_axis'],
        'raw_axis_order_zyx': ['local_v3_cross_axis', 'local_v2_arch_axis', 'local_v1_tooth_long_axis'],
        'output_size_zyx': [int(v) for v in output_size_zyx],
        'raw_size_zyx': [int(v) for v in ct_raw.shape],
        'center_xyz': center_xyz.astype(np.float32).tolist(),
        'local_lo_xyz': lo.astype(np.float32).tolist(),
        'local_hi_xyz': hi.astype(np.float32).tolist(),
        'margin_vox_mean': float(np.mean(margin_vox)),
    }
    return ct_crop, mask_crop, ct_raw, mask_raw, center_zyx, lbs, ubs, roi_meta

def estimate_tooth_center(tooth_id: str, centers: Dict[str, np.ndarray], mask_zyx: np.ndarray) -> np.ndarray:
    if tooth_id in centers:
        return centers[tooth_id]
    quadrant = tooth_id[0]
    q_teeth = [t for t in FDI_TEETH if t[0] == quadrant]
    xs = []
    ys = []
    for idx, t in enumerate(q_teeth):
        if t in centers:
            xs.append(idx)
            ys.append(centers[t])
    if len(xs) >= 2:
        target_idx = q_teeth.index(tooth_id)
        xs_np = np.asarray(xs, dtype=np.float64)
        ys_np = np.stack(ys, axis=0)
        pred = []
        for dim in range(3):
            order = np.argsort(xs_np)
            x_sorted = xs_np[order]
            y_sorted = ys_np[order, dim]
            if target_idx < x_sorted[0]:
                slope = (y_sorted[1] - y_sorted[0]) / max(1.0, x_sorted[1] - x_sorted[0])
                val = y_sorted[0] + slope * (target_idx - x_sorted[0])
            elif target_idx > x_sorted[-1]:
                slope = (y_sorted[-1] - y_sorted[-2]) / max(1.0, x_sorted[-1] - x_sorted[-2])
                val = y_sorted[-1] + slope * (target_idx - x_sorted[-1])
            else:
                val = np.interp(target_idx, x_sorted, y_sorted)
            pred.append(val)
        return np.asarray(pred, dtype=np.float64)
    if len(xs) == 1:
        return np.asarray(ys[0], dtype=np.float64)
    tooth_union = np.isin(mask_zyx, TOOTH_COMPACT_LABELS)
    coords = np.argwhere(tooth_union)
    if coords.size:
        return coords.mean(axis=0)
    return np.asarray(mask_zyx.shape, dtype=np.float64) / 2.0

def _side_anchor_mask(mask_zyx: np.ndarray, side: str) -> np.ndarray:
    if side == 'left':
        labels = [LEFT_IAC_LABEL, *LEFT_LOWER_TOOTH_LABELS]
    elif side == 'right':
        labels = [RIGHT_IAC_LABEL, *RIGHT_LOWER_TOOTH_LABELS]
    else:
        raise ValueError(f'Invalid side: {side}')
    return np.isin(mask_zyx, labels)

def _infer_lower_jaw_split(mask_zyx: np.ndarray) -> Tuple[int, float, bool]:
    left_anchor = _side_anchor_mask(mask_zyx, 'left')
    right_anchor = _side_anchor_mask(mask_zyx, 'right')
    if left_anchor.any() and right_anchor.any():
        left_center = np.argwhere(left_anchor).mean(axis=0)
        right_center = np.argwhere(right_anchor).mean(axis=0)
        axis = int(np.argmax(np.abs(left_center - right_center)))
        split = float((left_center[axis] + right_center[axis]) / 2.0)
        left_is_lower = bool(left_center[axis] <= right_center[axis])
        return axis, split, left_is_lower

    axis = mask_zyx.ndim - 1
    split = float(mask_zyx.shape[axis] / 2.0)
    if left_anchor.any():
        left_center = float(np.argwhere(left_anchor).mean(axis=0)[axis])
        return axis, split, bool(left_center <= split)
    if right_anchor.any():
        right_center = float(np.argwhere(right_anchor).mean(axis=0)[axis])
        return axis, split, bool(right_center > split)
    # Most CBCT arrays arrive as z,y,x; when side anchors are absent, x is the
    # safest last-resort left/right split. This path should be rare.
    return axis, split, True

def lower_jawbone_side_region_mask(mask_zyx: np.ndarray, side: str) -> Tuple[np.ndarray, int, float]:
    lower_jaw = mask_zyx == LOWER_JAWBONE_LABEL
    anchor = _side_anchor_mask(mask_zyx, side)
    axis, split, left_is_lower = _infer_lower_jaw_split(mask_zyx)

    coord_shape = [1] * mask_zyx.ndim
    coord_shape[axis] = mask_zyx.shape[axis]
    coords = np.arange(mask_zyx.shape[axis], dtype=np.float32).reshape(coord_shape)
    lower_side_selector = coords <= split if left_is_lower else coords >= split
    if side == 'right':
        lower_side_selector = ~lower_side_selector

    region = (lower_jaw & lower_side_selector) | anchor
    if not region.any():
        region = anchor
    if not region.any():
        region = lower_jaw
    return region, axis, split

def save_case_crops(
    case_id: str,
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    relation_mask_zyx: np.ndarray,
    out_dir: Path,
    tooth_size: Tuple[int, int, int],
    relation_size: Tuple[int, int, int],
    tooth_margin_mm: Tuple[float, float, float],
    relation_margin_mm: Tuple[float, float, float],
    spacing_zyx: Tuple[float, float, float],
    relation_fov_mm: Tuple[float, float, float],
    tooth_crop_schema: str = TOOTH_CROP_SCHEMA,
    tooth_crop_dir_name: str = "tooth_crops_oriented_pca",
    resume: bool = True,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create the released oriented-tooth and fixed-FOV relation crops."""
    tooth_margin = voxels_from_mm(tooth_margin_mm, spacing_zyx, min_vox=1)
    relation_margin = voxels_from_mm(relation_margin_mm, spacing_zyx, min_vox=1)
    relation_fov_vox = voxels_from_mm(relation_fov_mm, spacing_zyx, min_vox=8)
    centers = mask_centers(mask_zyx)

    tooth_paths: Dict[str, str] = {}
    case_tooth_dir = out_dir / tooth_crop_dir_name / case_id
    case_tooth_dir.mkdir(parents=True, exist_ok=True)
    for tooth_id in FDI_TEETH:
        output = case_tooth_dir / f"{tooth_id}.npz"
        tooth_paths[tooth_id] = str(output)
        if resume and output.exists() and crop_npz_matches_schema(output, tooth_crop_schema):
            continue

        compact_label = FDI_TO_COMPACT.get(tooth_id)
        tooth_mask = (
            mask_zyx == compact_label
            if compact_label is not None
            else np.zeros(mask_zyx.shape, dtype=bool)
        )
        bounds = bbox_bounds(tooth_mask, tooth_margin)
        is_present = bounds is not None
        if bounds is not None:
            try:
                ct_crop_f, mask_crop, ct_raw_f, mask_raw, center, lbs, ubs, roi_meta = (
                    crop_oriented_pca_labelmap(
                        image_zyx,
                        mask_zyx,
                        tooth_mask,
                        tooth_id,
                        tooth_margin_mm,
                        spacing_zyx,
                        centers,
                        tooth_size,
                    )
                )
                ct_raw = ct_raw_f.astype(np.int16)
                mask_raw = mask_raw.astype(np.uint8, copy=False)
                ct_crop = ct_crop_f.astype(np.int16)
                crop_mode = str(
                    roi_meta.get("crop_mode", "tooth_mask_oriented_pca_margin_resized")
                )
            except (ValueError, np.linalg.LinAlgError):
                lbs, ubs = bounds
                ct_raw = crop_bounds(image_zyx, lbs, ubs).astype(np.float32, copy=False)
                mask_raw = crop_bounds(mask_zyx, lbs, ubs).astype(np.float32, copy=False)
                ct_crop = resize_channels_cpu(ct_raw[None], tooth_size, mode="trilinear")[0].astype(np.int16)
                mask_crop = resize_channels_cpu(mask_raw[None], tooth_size, mode="nearest")[0].astype(np.uint8)
                center = np.argwhere(tooth_mask).mean(axis=0).astype(np.float32)
                crop_mode = "tooth_mask_oriented_pca_fallback_bbox_margin_resized"
                roi_meta = {"crop_mode": crop_mode}
        else:
            center = estimate_tooth_center(tooth_id, centers, mask_zyx).astype(np.float32)
            ct_raw = crop_fixed(image_zyx, center, tooth_size, pad_value=-1024).astype(np.int16)
            mask_raw = crop_fixed(mask_zyx, center, tooth_size, pad_value=0).astype(np.uint8)
            ct_crop = ct_raw.copy()
            mask_crop = mask_raw.copy()
            lbs = np.asarray(np.floor(center - np.asarray(tooth_size) / 2.0), dtype=np.int32)
            ubs = lbs + np.asarray(tooth_size, dtype=np.int32)
            crop_mode = "fallback_missing_or_unsegmented_no_loss"
            roi_meta = {"crop_mode": crop_mode}

        np.savez_compressed(
            output,
            ct=ct_crop,
            mask=mask_crop,
            ct_raw=ct_raw.astype(np.int16, copy=False),
            mask_raw=mask_raw.astype(np.uint8, copy=False),
            tooth_id=tooth_id,
            center_zyx=np.asarray(center, dtype=np.float32),
            bbox_lbs_zyx=np.asarray(lbs, dtype=np.int32),
            bbox_ubs_zyx=np.asarray(ubs, dtype=np.int32),
            is_present=np.asarray(is_present, dtype=np.bool_),
            crop_mode=np.asarray(crop_mode),
            crop_schema=np.asarray(tooth_crop_schema),
            raw_shape_zyx=np.asarray(ct_raw.shape, dtype=np.int32),
            resized_shape_zyx=np.asarray(ct_crop.shape, dtype=np.int32),
            spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32),
            margin_mm_zyx=np.asarray(tooth_margin_mm, dtype=np.float32),
            margin_vox_zyx=np.asarray(tooth_margin, dtype=np.int32),
            roi_meta=np.asarray(json.dumps(roi_meta, ensure_ascii=False)),
        )

    relation_paths: Dict[str, str] = {}
    relation_dir = out_dir / "relation_crops" / case_id
    relation_dir.mkdir(parents=True, exist_ok=True)
    relation_specs = {
        "left_lower_jawbone_region": {
            "side": "left",
            "source_labels": [LOWER_JAWBONE_LABEL, LEFT_IAC_LABEL, *LEFT_LOWER_TOOTH_LABELS],
            "anchor_labels": [LEFT_IAC_LABEL, *LEFT_POSTERIOR_RELATION_LABELS],
        },
        "right_lower_jawbone_region": {
            "side": "right",
            "source_labels": [LOWER_JAWBONE_LABEL, RIGHT_IAC_LABEL, *RIGHT_LOWER_TOOTH_LABELS],
            "anchor_labels": [RIGHT_IAC_LABEL, *RIGHT_POSTERIOR_RELATION_LABELS],
        },
    }
    fallback_labels = [
        LOWER_JAWBONE_LABEL,
        LEFT_IAC_LABEL,
        RIGHT_IAC_LABEL,
        *LEFT_LOWER_TOOTH_LABELS,
        *RIGHT_LOWER_TOOTH_LABELS,
    ]
    fallback_center = _union_center(
        relation_mask_zyx,
        fallback_labels,
        fallback=np.asarray(relation_mask_zyx.shape) / 2.0,
    )

    for name, specification in relation_specs.items():
        side = str(specification["side"])
        source_labels = [int(value) for value in specification["source_labels"]]
        anchor_labels = [int(value) for value in specification["anchor_labels"]]
        output = relation_dir / f"{name}.npz"
        relation_paths[name] = str(output)
        if resume and output.exists() and crop_npz_matches_schema(output, RELATION_CROP_SCHEMA):
            continue

        region_mask, split_axis, split_coord = lower_jawbone_side_region_mask(
            relation_mask_zyx, side
        )
        anchor_mask = np.isin(relation_mask_zyx, anchor_labels)
        crop_anchor = anchor_mask if anchor_mask.any() else region_mask
        bounds = bbox_bounds(crop_anchor, relation_margin)
        center = (
            np.argwhere(crop_anchor).mean(axis=0).astype(np.float32)
            if bounds is not None
            else np.asarray(fallback_center, dtype=np.float32)
        )
        ct_raw = crop_fixed(image_zyx, center, relation_fov_vox, pad_value=-1024).astype(
            np.float32, copy=False
        )
        mask_raw = crop_fixed(
            relation_mask_zyx, center, relation_fov_vox, pad_value=0
        ).astype(np.float32, copy=False)
        lbs = np.asarray(
            np.floor(center - np.asarray(relation_fov_vox) / 2.0), dtype=np.int32
        )
        ubs = lbs + np.asarray(relation_fov_vox, dtype=np.int32)
        if bounds is None:
            crop_mode = "fallback_no_lower_jawbone_or_side_anchor_fixed_physical_fov"
        elif anchor_mask.any():
            crop_mode = "pred_iac_posterior_anchor_fixed_physical_fov_resized"
        else:
            crop_mode = "pred_lower_jawbone_side_fixed_physical_fov_resized"
        ct_crop = resize_channels_cpu(ct_raw[None], relation_size, mode="trilinear")[0].astype(np.int16)
        mask_crop = resize_channels_cpu(mask_raw[None], relation_size, mode="nearest")[0].astype(np.uint8)

        np.savez_compressed(
            output,
            ct=ct_crop,
            mask=mask_crop,
            ct_raw=ct_raw,
            mask_raw=mask_raw,
            relation=name,
            side=np.asarray(side),
            source_labels=np.asarray(source_labels, dtype=np.uint8),
            anchor_labels=np.asarray(anchor_labels, dtype=np.uint8),
            used_anchor_mask=np.asarray(bool(anchor_mask.any()), dtype=np.bool_),
            split_axis=np.asarray(split_axis, dtype=np.int32),
            split_coord=np.asarray(split_coord, dtype=np.float32),
            center_zyx=np.asarray(center, dtype=np.float32),
            bbox_lbs_zyx=np.asarray(lbs, dtype=np.int32),
            bbox_ubs_zyx=np.asarray(ubs, dtype=np.int32),
            crop_mode=np.asarray(crop_mode),
            crop_schema=np.asarray(RELATION_CROP_SCHEMA),
            raw_shape_zyx=np.asarray(ct_raw.shape, dtype=np.int32),
            resized_shape_zyx=np.asarray(ct_crop.shape, dtype=np.int32),
            spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32),
            margin_mm_zyx=np.asarray(relation_margin_mm, dtype=np.float32),
            margin_vox_zyx=np.asarray(relation_margin, dtype=np.int32),
            relation_fov_mm_zyx=np.asarray(relation_fov_mm, dtype=np.float32),
            relation_fov_vox_zyx=np.asarray(relation_fov_vox, dtype=np.int32),
        )
    return tooth_paths, relation_paths

def _union_center(mask_zyx: np.ndarray, labels: List[int], fallback: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.isin(mask_zyx, labels))
    if coords.size:
        return coords.mean(axis=0)
    return np.asarray(fallback, dtype=np.float64)

def normalize_cbct_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, -997.0, 3558.0)
    return (x - 809.5) / 1014.369

def resize_channels_cpu(x: np.ndarray, size_zyx: Tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
    if tuple(tensor.shape[-3:]) == tuple(size_zyx):
        out = tensor[0]
    elif mode == 'nearest':
        out = F.interpolate(tensor, size=size_zyx, mode='nearest')[0]
    else:
        out = F.interpolate(tensor, size=size_zyx, mode='trilinear', align_corners=False)[0]
    return out.cpu().numpy()

def resize_mask_channel(mask_bool: np.ndarray, size_zyx: Tuple[int, int, int]) -> np.ndarray:
    if tuple(mask_bool.shape) == tuple(size_zyx):
        return mask_bool.astype(np.float32, copy=False)
    return resize_channels_cpu(mask_bool.astype(np.float32, copy=False)[None], size_zyx, mode='nearest')[0]

def resized_global_semantic_channels(mask: np.ndarray, size_zyx: Tuple[int, int, int]) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False)
    teeth = np.isin(mask_u8, TOOTH_COMPACT_LABELS)
    canal = np.isin(mask_u8, [3, 4])
    sinus = np.isin(mask_u8, [5, 6])
    other = (mask_u8 > 0) & ~(teeth | canal | sinus)
    selected_empty = np.zeros(size_zyx, dtype=np.float32)
    return np.stack([
        resize_mask_channel(teeth, size_zyx),
        selected_empty,
        resize_mask_channel(canal, size_zyx),
        resize_mask_channel(sinus, size_zyx),
        resize_mask_channel(other, size_zyx),
    ], axis=0).astype(np.float32, copy=False)

def build_global_input_tensor(image_path: Path, mask_path: Path, size_zyx: Tuple[int, int, int]) -> np.ndarray:
    image = normalize_cbct_array(read_blosc2(image_path))
    mask = read_blosc2(mask_path).astype(np.uint8, copy=False)
    image_small = resize_channels_cpu(image[None], size_zyx, mode='trilinear')
    mask_small = resized_global_semantic_channels(mask, size_zyx)
    return np.concatenate([image_small, mask_small], axis=0).astype(np.float16, copy=False)



def prepare_case(
    case_id: str,
    image_path: Path,
    stage1_mask_path: Path,
    visual_tokens_path: Path,
    out_dir: Path,
    *,
    tooth_size: Tuple[int, int, int] = (64, 64, 96),
    relation_size: Tuple[int, int, int] = (80, 128, 128),
    tooth_margin_mm: Tuple[float, float, float] = (6.0, 6.0, 6.0),
    relation_margin_mm: Tuple[float, float, float] = (4.8, 9.6, 9.6),
    relation_fov_mm: Tuple[float, float, float] = (80.0, 96.0, 64.0),
    global_size: Tuple[int, int, int] = (96, 192, 192),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_nifti_zyx(image_path, np.float32)
    spacing_zyx = nifti_spacing_zyx(image_path)
    image_b2 = out_dir / "imagesGlobal" / f"{case_id}_0000.b2nd"
    write_blosc2(image_b2, cbct_to_int16_checked(image, f"{case_id} image"))

    mask = align_mask_to_image(load_stage1_mask_zyx(stage1_mask_path), tuple(image.shape))
    mask = normalize_stage1_mask_label_space(mask, "compact", case_id)
    mask = cleanup_tooth_mask_components(mask, closing_iterations=1, connectivity=1)
    mask_b2 = out_dir / "stage1_masks" / f"{case_id}.b2nd"
    write_blosc2(mask_b2, mask.astype(np.uint8, copy=False))

    tooth_schema = tooth_crop_schema_for(tooth_size, tooth_margin_mm)
    tooth_dir_name = tooth_crop_dir_name_for(tooth_size, tooth_margin_mm)
    tooth_paths, relation_paths = save_case_crops(
        case_id,
        image,
        mask,
        mask,
        out_dir,
        tooth_size,
        relation_size,
        tooth_margin_mm,
        relation_margin_mm,
        spacing_zyx,
        relation_fov_mm,
        tooth_crop_schema=tooth_schema,
        tooth_crop_dir_name=tooth_dir_name,
        resume=False,
    )

    present = {tooth for tooth in FDI_TEETH if np.any(mask == int(FDI_TO_COMPACT[tooth]))}
    missing = [tooth for tooth in FDI_TEETH if tooth not in present]
    tooth_mask = np.zeros((len(FDI_TEETH), len(TOOTH_LABELS)), dtype=np.float32)
    for i, tooth in enumerate(FDI_TEETH):
        if tooth in present:
            tooth_mask[i, :] = 1.0

    label_dir = out_dir / "labels_inference"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{case_id}.json"
    label_obj = {
        "report_id": case_id,
        "case_id": case_id,
        "global_target": [0.0] * len(GLOBAL_LABELS),
        "global_loss_mask": [0.0] * len(GLOBAL_LABELS),
        "tooth_order": FDI_TEETH,
        "tooth_labels": TOOTH_LABELS,
        "tooth_target": np.zeros((len(FDI_TEETH), len(TOOTH_LABELS)), dtype=np.float32).tolist(),
        "tooth_loss_mask": tooth_mask.tolist(),
        "tooth_crop_paths": [tooth_paths[t] for t in FDI_TEETH],
        "relation_labels": RELATION_LABELS,
        "relation_target": [0.0] * len(RELATION_LABELS),
        "relation_loss_mask": [0.0] * len(RELATION_LABELS),
        "relation_crops": RELATION_CROPS,
        "relation_crop_paths": [relation_paths[name] for name in RELATION_CROPS],
        "missing_teeth_from_mask": missing,
        "source_missing_teeth": [],
    }
    label_path.write_text(json.dumps(label_obj, indent=2) + "\n")

    global_dir = out_dir / f"global_tensors_{global_size[0]}x{global_size[1]}x{global_size[2]}"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_path = global_dir / f"{case_id}.b2nd"
    tensor = build_global_input_tensor(image_b2, mask_b2, global_size)
    write_blosc2(global_path, tensor, clevel=5)

    if not visual_tokens_path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 visual-token artifact: {visual_tokens_path}")

    manifest = out_dir / "validation_case_merged.csv"
    pd.DataFrame([{
        "report_id": case_id,
        "case_id": case_id,
        "series_id": case_id,
        "image_blosc2_path": str(image_b2),
        "mask_blosc2_path": str(mask_b2),
        "stage2_label_json": str(label_path),
        "global_tensor_path": str(global_path),
        "visual_structure_tokens_npz_path": str(visual_tokens_path),
        "source_report_ids": "",
    }]).to_csv(manifest, index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--stage1-mask", type=Path, required=True)
    parser.add_argument("--visual-tokens", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_case(args.case_id, args.image, args.stage1_mask, args.visual_tokens, args.out_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
