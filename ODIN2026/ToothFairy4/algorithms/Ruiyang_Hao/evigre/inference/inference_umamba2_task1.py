#!/usr/bin/env python3
"""Run original U-Mamba2 ToothFairy3 Task1 inference from EviDent-CBCT.

This wrapper is intentionally narrow: it loads the original
``documentation/competitions/Toothfairy3/task1_inference.py`` predictor and
uses its checkpoint restore, preprocessing, and sliding-window TTA. EviDent-CBCT
adds a spacing-aware physical-volume version of the original ToothFairy3
postprocessing before optionally converting the final Grand Challenge labels
back to the compact 0..46 Stage-1 label space expected by ToothFairy4 Stage 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from tqdm import tqdm


DEFAULT_TASK1_INFERENCE = None
DEFAULT_CHECKPOINT = None


def detect_id_column(df: pd.DataFrame) -> str:
    for col in ("identifier", "series_id", "case_id", "report_id"):
        if col in df.columns:
            return col
    raise ValueError(f"CSV missing identifier column. Current columns: {list(df.columns)}")


def detect_image_column(df: pd.DataFrame) -> str:
    # Prefer raw NIfTI paths. The original U-Mamba2 predictor must see the raw
    # image and checkpoint preprocessor, not EviDent-CBCT's Stage-1 Blosc2 tensors.
    for col in ("image_path", "raw_image_path", "nifti_path"):
        if col in df.columns:
            return col
    raise ValueError(
        "CSV missing raw image path column. Need one of: image_path, raw_image_path, nifti_path. "
        f"Current columns: {list(df.columns)}"
    )


def load_original_task1_module(task1_inference_py: Path):
    if not task1_inference_py.exists():
        raise FileNotFoundError(f"Original U-Mamba2 task1_inference.py not found: {task1_inference_py}")
    spec = importlib.util.spec_from_file_location("evigre_original_umamba2_task1_inference", task1_inference_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import original task1_inference.py: {task1_inference_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_lr_mapping() -> List[Tuple[int, int]]:
    mapping: List[Tuple[int, int]] = [(3, 4), (5, 6), (43, 44)]
    mapping.extend(zip(range(19, 27), range(11, 19)))
    mapping.extend(zip(range(27, 35), range(35, 43)))
    return [(int(left), int(right)) for left, right in mapping]


def resolve_checkpoint(args) -> Tuple[Path, str, str]:
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        fold_dir = checkpoint_path.parent
        if not fold_dir.name.startswith("fold_"):
            raise ValueError(f"Checkpoint must live under a fold_* directory: {checkpoint_path}")
        model_folder = fold_dir.parent
        fold = fold_dir.name[len("fold_"):]
        return model_folder, fold, checkpoint_path.name

    if not args.model_folder:
        raise ValueError("Provide --checkpoint-path or --model-folder")
    model_folder = Path(args.model_folder)
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")
    return model_folder, str(args.fold), args.checkpoint_name


def original_to_compact_mapping() -> np.ndarray:
    mapping = np.zeros(54, dtype=np.uint8)
    for label in range(0, 19):
        mapping[label] = label
    for label in range(21, 29):
        mapping[label] = label - 2
    for label in range(31, 39):
        mapping[label] = label - 4
    for label in range(41, 49):
        mapping[label] = label - 6
    mapping[50] = 46
    mapping[51] = 43
    mapping[52] = 44
    mapping[53] = 45
    return mapping


def convert_original_to_compact(seg: np.ndarray, identifier: str) -> np.ndarray:
    unique = np.unique(seg)
    unsupported = sorted(int(v) for v in unique if v >= 54 or original_to_compact_mapping()[int(v)] == 0 and int(v) != 0)
    if unsupported:
        raise ValueError(f"{identifier}: original Task1 prediction has unsupported labels: {unsupported[:20]}")
    return original_to_compact_mapping()[seg.astype(np.int16, copy=False)]



def parse_spacing(text: str) -> Tuple[float, float, float]:
    vals = [float(v) for v in str(text).split(',')]
    if len(vals) != 3:
        raise ValueError(f"Expected x,y,z spacing triplet, got: {text}")
    return vals[0], vals[1], vals[2]


def scaled_class_volume_thresholds(base_thresholds: Dict[int, float], spacing_xyz: Tuple[float, float, float],
                                   reference_spacing_xyz: Tuple[float, float, float]) -> Dict[int, float]:
    """Convert U-Mamba2's fixed voxel-count cleanup thresholds to physical-volume thresholds.

    The original ToothFairy3 model was planned around 0.3 mm isotropic data. Subset A is often
    much finer (around 0.15-0.18 mm), so using the same voxel-count threshold would keep much
    smaller false-positive components. We preserve the original physical volume represented by
    each threshold and convert it back to voxels for the current image spacing.
    """
    current_voxel_mm3 = float(np.prod(np.asarray(spacing_xyz, dtype=np.float64)))
    reference_voxel_mm3 = float(np.prod(np.asarray(reference_spacing_xyz, dtype=np.float64)))
    if current_voxel_mm3 <= 0 or reference_voxel_mm3 <= 0:
        raise ValueError(f"Invalid spacing for physical-volume postprocessing: current={spacing_xyz}, reference={reference_spacing_xyz}")
    scale = reference_voxel_mm3 / current_voxel_mm3
    return {int(k): float(v) * scale for k, v in base_thresholds.items()}


COMPACT_TOOTH_LABELS = list(range(11, 19)) + list(range(19, 27)) + list(range(27, 35)) + list(range(35, 43))
FDI_TEETH = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
FDI_TO_COMPACT = {}
for _t in FDI_TEETH:
    _ti = int(_t)
    if 11 <= _ti <= 18:
        FDI_TO_COMPACT[_t] = _ti
    elif 21 <= _ti <= 28:
        FDI_TO_COMPACT[_t] = _ti - 2
    elif 31 <= _ti <= 38:
        FDI_TO_COMPACT[_t] = _ti - 4
    elif 41 <= _ti <= 48:
        FDI_TO_COMPACT[_t] = _ti - 6
LEFT_LOWER_TOOTH_LABELS = [FDI_TO_COMPACT[t] for t in FDI_TEETH if t.startswith('3')]
RIGHT_LOWER_TOOTH_LABELS = [FDI_TO_COMPACT[t] for t in FDI_TEETH if t.startswith('4')]
VISUAL_TOKEN_SCHEMA = 'cbct_evigre_stage1_inference_encoder_structure_tokens_v1'


def parse_int_list(text: str) -> List[int]:
    vals = [int(v) for v in str(text).split(',') if str(v).strip()]
    if not vals:
        raise ValueError(f'Expected comma-separated integer list, got {text!r}')
    return vals


def resize_label_nearest(mask: np.ndarray, shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape_zyx):
        return mask.astype(np.uint8, copy=False)
    t = torch.from_numpy(mask.astype(np.float32, copy=False)[None, None])
    out = F.interpolate(t, size=tuple(shape_zyx), mode='nearest')[0, 0].cpu().numpy()
    return out.astype(np.uint8, copy=False)


def visual_global_specs() -> List[Tuple[str, List[int]]]:
    specs = [
        ('all_teeth', COMPACT_TOOTH_LABELS),
        ('lower_jawbone', [1]),
        ('upper_jawbone', [2]),
        ('left_iac', [3]),
        ('right_iac', [4]),
        ('maxillary_sinuses', [5, 6]),
    ]
    specs.extend((f'tooth:{t}', [int(FDI_TO_COMPACT[t])]) for t in FDI_TEETH)
    return specs


def visual_tooth_specs() -> List[Tuple[str, List[int]]]:
    return [(f'tooth:{t}', [int(FDI_TO_COMPACT[t])]) for t in FDI_TEETH]


def visual_relation_names() -> List[str]:
    return ['relation:left_lower_jawbone_region', 'relation:right_lower_jawbone_region']


def side_anchor_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == 'left':
        labels = [3, *LEFT_LOWER_TOOTH_LABELS]
    elif side == 'right':
        labels = [4, *RIGHT_LOWER_TOOTH_LABELS]
    else:
        raise ValueError(side)
    return np.isin(mask, labels)


def lower_jaw_side_masks(mask: np.ndarray) -> List[np.ndarray]:
    lower = mask == 1
    left_anchor = side_anchor_mask(mask, 'left')
    right_anchor = side_anchor_mask(mask, 'right')
    axis = mask.ndim - 1
    split = float(mask.shape[axis] / 2.0)
    left_is_lower = True
    if left_anchor.any() and right_anchor.any():
        lc = np.argwhere(left_anchor).mean(axis=0)
        rc = np.argwhere(right_anchor).mean(axis=0)
        axis = int(np.argmax(np.abs(lc - rc)))
        split = float((lc[axis] + rc[axis]) / 2.0)
        left_is_lower = bool(lc[axis] <= rc[axis])
    grid = np.indices(mask.shape)[axis]
    if left_is_lower:
        left_lower = lower & (grid <= split)
        right_lower = lower & (grid > split)
    else:
        left_lower = lower & (grid > split)
        right_lower = lower & (grid <= split)
    left = left_lower | left_anchor
    right = right_lower | right_anchor
    return [left, right]


def downsample_mask_to_feature(mask_patch: np.ndarray, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
    return resize_label_nearest(mask_patch, spatial_shape)


def pool_feature_specs(feat: torch.Tensor, mask_small: np.ndarray, specs: List[Tuple[str, List[int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = [np.isin(mask_small, np.asarray(labels, dtype=np.uint8)) for _name, labels in specs]
    return pool_feature_bool_masks(feat, masks)


def pool_feature_bool_masks(feat: torch.Tensor, masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = int(feat.shape[0])
    flat = feat.reshape(c, -1)
    sums = np.zeros((len(masks), c), dtype=np.float64)
    maxs = np.full((len(masks), c), -np.inf, dtype=np.float32)
    counts = np.zeros(len(masks), dtype=np.int64)
    for i, m in enumerate(masks):
        sel_np = m.reshape(-1).astype(bool, copy=False)
        n = int(sel_np.sum())
        counts[i] = n
        if n > 0:
            sel = torch.from_numpy(sel_np)
            vals = flat[:, sel]
            sums[i] = vals.sum(dim=1).cpu().numpy().astype(np.float64, copy=False)
            maxs[i] = vals.max(dim=1).values.cpu().numpy().astype(np.float32, copy=False)
    return sums, maxs, counts


def init_pool_state(n_tokens: int, channels: int) -> Dict[str, np.ndarray]:
    return {
        'sum': np.zeros((n_tokens, channels), dtype=np.float64),
        'max': np.full((n_tokens, channels), -np.inf, dtype=np.float32),
        'count': np.zeros(n_tokens, dtype=np.int64),
    }


def update_pool_state(state: Dict[str, np.ndarray], sums: np.ndarray, maxs: np.ndarray, counts: np.ndarray) -> None:
    state['sum'] += sums
    state['max'] = np.maximum(state['max'], maxs)
    state['count'] += counts


def finalize_pool_state(state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = state['count'].astype(np.float32)
    mean = np.zeros_like(state['sum'], dtype=np.float32)
    nz = count > 0
    if np.any(nz):
        mean[nz] = (state['sum'][nz] / count[nz, None]).astype(np.float32)
    maxv = state['max'].astype(np.float32)
    maxv[~np.isfinite(maxv)] = 0.0
    return mean, maxv, state['count'].astype(np.int32)


def save_stage1_inference_visual_tokens(identifier: str, image: np.ndarray, props: Dict[str, Any], pred_compact: np.ndarray,
                                        predictor, output_dir: Path, args) -> Optional[str]:
    if not args.save_visual_tokens:
        return None
    token_dir = output_dir / 'visual_structure_tokens'
    maybe_mkdir_p(str(token_dir))
    out = token_dir / f'{identifier}.npz'
    if args.skip_existing and out.exists():
        return str(out)

    levels = parse_int_list(args.visual_token_levels)
    with torch.no_grad():
        data, _, props_pre = predictor.preprocessor.run_case_npy(
            image, None, dict(props), predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json,
        )
    data = torch.from_numpy(data).float()
    mask_pre = resize_label_nearest(pred_compact, tuple(int(v) for v in data.shape[1:]))
    data_padded, slicer_revert_padding = pad_nd_image(
        data, predictor.configuration_manager.patch_size, 'constant', {'value': 0}, True, None,
    )
    mask_padded, _ = pad_nd_image(
        mask_pre, predictor.configuration_manager.patch_size, 'constant', {'constant_values': 0}, True, None,
    )
    if isinstance(mask_padded, torch.Tensor):
        mask_padded = mask_padded.cpu().numpy()
    mask_padded = mask_padded.astype(np.uint8, copy=False)
    slicers = predictor._internal_get_sliding_window_slicers(data_padded.shape[1:])
    device = predictor.device
    global_specs = visual_global_specs()
    tooth_specs = visual_tooth_specs()
    relation_names = visual_relation_names()
    states: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    feature_shapes: Dict[int, Tuple[int, int, int, int]] = {}

    with torch.inference_mode(), torch.autocast(device.type, enabled=(device.type == 'cuda')):
        for sl in slicers:
            patch = data_padded[sl][None].to(device=device, non_blocking=True)
            skips = list(predictor.network.encoder(patch))
            if hasattr(predictor.network, 'mamba_layer'):
                skips[-1] = predictor.network.mamba_layer(skips[-1])
            mask_patch = mask_padded[sl[1:]]
            for level in levels:
                if level < 0 or level >= len(skips):
                    raise ValueError(f'{identifier}: requested visual token level {level}, got {len(skips)} encoder levels')
                feat = skips[level].detach().float().cpu()[0]
                if level not in states:
                    c = int(feat.shape[0])
                    states[level] = {
                        'global': init_pool_state(len(global_specs), c),
                        'tooth': init_pool_state(len(tooth_specs), c),
                        'relation': init_pool_state(len(relation_names), c),
                    }
                    feature_shapes[level] = tuple(int(v) for v in feat.shape)
                mask_small = downsample_mask_to_feature(mask_patch, tuple(int(v) for v in feat.shape[-3:]))
                g_sum, g_max, g_count = pool_feature_specs(feat, mask_small, global_specs)
                t_sum, t_max, t_count = pool_feature_specs(feat, mask_small, tooth_specs)
                r_sum, r_max, r_count = pool_feature_bool_masks(feat, lower_jaw_side_masks(mask_small))
                update_pool_state(states[level]['global'], g_sum, g_max, g_count)
                update_pool_state(states[level]['tooth'], t_sum, t_max, t_count)
                update_pool_state(states[level]['relation'], r_sum, r_max, r_count)
            del patch, skips

    arrays: Dict[str, np.ndarray] = {
        'schema': np.asarray(VISUAL_TOKEN_SCHEMA),
        'identifier': np.asarray(identifier),
        'source': np.asarray('original_umamba2_task1_inference_sliding_window_encoder'),
        'levels': np.asarray(levels, dtype=np.int16),
        'patch_size_zyx': np.asarray(predictor.configuration_manager.patch_size, dtype=np.int32),
        'preprocessed_shape_zyx': np.asarray(data.shape[1:], dtype=np.int32),
        'padded_shape_zyx': np.asarray(data_padded.shape[1:], dtype=np.int32),
        'raw_prediction_shape_zyx': np.asarray(pred_compact.shape, dtype=np.int32),
        'global_names': np.asarray([name for name, _labels in global_specs]),
        'global_label_ids': np.asarray(['+'.join(map(str, labels)) for _name, labels in global_specs]),
        'tooth_order': np.asarray(FDI_TEETH),
        'tooth_compact_labels': np.asarray([FDI_TO_COMPACT[t] for t in FDI_TEETH], dtype=np.int16),
        'relation_names': np.asarray(relation_names),
        'pooling': np.asarray('sliding_window_encoder_mask_mean_and_max_after_final_postprocess_mask_resized_to_preprocessed_grid'),
    }
    for level in levels:
        arrays[f'level{level}_feature_shape_czyx'] = np.asarray(feature_shapes[level], dtype=np.int32)
        for group in ['global', 'tooth', 'relation']:
            mean, maxv, count = finalize_pool_state(states[level][group])
            arrays[f'level{level}_{group}_mean'] = mean.astype(np.float16, copy=False)
            arrays[f'level{level}_{group}_max'] = maxv.astype(np.float16, copy=False)
            arrays[f'level{level}_{group}_counts'] = count
    np.savez_compressed(out, **arrays)
    return str(out)


def compact_mask_quality_stats(seg_compact: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Dict[str, Any]:
    voxel_mm3 = float(np.prod(np.asarray(spacing_xyz, dtype=np.float64)))
    counts = {int(label): int(count) for label, count in zip(*np.unique(seg_compact, return_counts=True))}
    def vox(label: int) -> int:
        return int(counts.get(int(label), 0))
    tooth_vox = int(sum(vox(label) for label in COMPACT_TOOTH_LABELS))
    lower_labels = list(range(27, 43))
    upper_labels = list(range(11, 27))
    return {
        "spacing_xyz": [float(v) for v in spacing_xyz],
        "voxel_volume_mm3": voxel_mm3,
        "shape_zyx": [int(v) for v in seg_compact.shape],
        "lower_jaw_voxels": vox(1),
        "upper_jaw_voxels": vox(2),
        "left_iac_voxels": vox(3),
        "right_iac_voxels": vox(4),
        "sinus_voxels": vox(5) + vox(6),
        "tooth_voxels": tooth_vox,
        "lower_jaw_mm3": vox(1) * voxel_mm3,
        "upper_jaw_mm3": vox(2) * voxel_mm3,
        "left_iac_mm3": vox(3) * voxel_mm3,
        "right_iac_mm3": vox(4) * voxel_mm3,
        "sinus_mm3": (vox(5) + vox(6)) * voxel_mm3,
        "tooth_mm3": tooth_vox * voxel_mm3,
        "present_teeth_count": int(sum(vox(label) > 0 for label in COMPACT_TOOTH_LABELS)),
        "present_lower_teeth_count": int(sum(vox(label) > 0 for label in lower_labels)),
        "present_upper_teeth_count": int(sum(vox(label) > 0 for label in upper_labels)),
        "has_left_iac": bool(vox(3) > 0),
        "has_right_iac": bool(vox(4) > 0),
    }


def _classwise_resampled_argmax(
    logits: torch.Tensor,
    target_shape: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Resample one class at a time and retain only the running argmax."""
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(int(v) for v in logits.shape[1:]) == target_shape:
        return logits.argmax(dim=0).cpu()

    logits = logits.cpu()
    result_dtype = torch.uint8 if int(logits.shape[0]) < 256 else torch.int16
    labels = torch.zeros(target_shape, dtype=result_dtype, device=device)
    best_scores = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
    for class_index in range(int(logits.shape[0])):
        class_logits = logits[class_index : class_index + 1].to(device=device)
        scores = F.interpolate(
            class_logits[None].float(),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
            antialias=False,
        )[0, 0]
        if best_scores is None:
            best_scores = scores
        else:
            better = scores > best_scores
            torch.maximum(best_scores, scores, out=best_scores)
            labels[better] = class_index
            del better, scores
        del class_logits

    result = labels.cpu()
    del labels, best_scores
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _memory_bounded_predictor_class(task1_module):
    class MemoryBoundedBasePredictor(task1_module.BasePredictor):
        @torch.inference_mode(mode=True)
        def predict_preprocessed_image(self, image):
            data, slicer_revert_padding = pad_nd_image(
                image,
                self.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            del image
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            data = data.to(
                device=self.device,
                dtype=torch.float16 if self.explicit_half else None,
                non_blocking=True,
            )
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.float16,
                device="cpu",
            )
            gaussian = compute_gaussian(
                tuple(self.configuration_manager.patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                device=self.device,
                dtype=torch.float16,
            )

            if not self.allow_tqdm and self.verbose:
                print(f"running prediction: {len(slicers)} steps")

            with torch.autocast(self.device.type, enabled=not self.explicit_half):
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    patch = data[sl][None].to(device=self.device, non_blocking=True)
                    pred = self._internal_maybe_mirror_and_predict(patch)[0].to(dtype=torch.float16)
                    pred /= pred.max() / 100
                    weighted = pred * gaussian
                    accumulated = predicted_logits[sl].to(device=self.device)
                    accumulated.add_(weighted)
                    predicted_logits[sl].copy_(accumulated.cpu())
                    del patch, pred, weighted, accumulated

            return predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
            config = self.configuration_manager.configuration
            kwargs = config.get("resampling_fn_probabilities_kwargs", {})
            if (
                config.get("resampling_fn_probabilities") != "resample_torch_fornnunet"
                or kwargs.get("is_seg") is not False
                or kwargs.get("force_separate_z") is not False
            ):
                raise RuntimeError(
                    "Memory-bounded Task1 conversion requires the fixed "
                    "resample_torch_fornnunet trilinear probability plan."
                )

            target_shape = tuple(int(v) for v in props["shape_after_cropping_and_before_resampling"])
            segmentation = _classwise_resampled_argmax(predicted_logits, target_shape, self.device).numpy()
            del predicted_logits

            dtype = np.uint8 if len(self.label_manager.foreground_labels) < 255 else np.uint16
            segmentation_reverted_cropping = np.zeros(props["shape_before_cropping"], dtype=dtype)
            slicer = task1_module.bounding_box_to_slice(props["bbox_used_for_cropping"])
            segmentation_reverted_cropping[slicer] = segmentation
            del segmentation
            return segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)

    return MemoryBoundedBasePredictor


def build_predictor(task1_module, model_folder: Path, fold: str, checkpoint_name: str, args):
    lr_mapping = None if args.disable_left_right_mirroring else build_lr_mapping()
    predictor_class = (
        task1_module.BasePredictor
        if args.accumulation_device == "cuda"
        else _memory_bounded_predictor_class(task1_module)
    )
    predictor = predictor_class(
        tile_step_size=args.tile_step_size,
        use_mirroring=not args.disable_mirroring,
        use_gaussian=True,
        perform_everything_on_device=args.accumulation_device == "cuda",
        allow_tqdm=not args.disable_tqdm,
        tta_batch_size=args.tta_batch_size,
        lr_mapping=lr_mapping,
        n_class=47,
        verbose=args.verbose,
    )
    predictor._evigre_original_cls_vol = dict(predictor.cls_vol)
    if args.device:
        predictor.device = torch.device(args.device)
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=[fold],
        checkpoint_name=checkpoint_name,
    )
    return predictor


def run_inference(args) -> Dict[str, Any]:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    model_folder, fold, checkpoint_name = resolve_checkpoint(args)
    task1_module = load_original_task1_module(Path(args.task1_inference_py))
    predictor = build_predictor(task1_module, model_folder, fold, checkpoint_name, args)

    output_dir = Path(args.output)
    pred_dir = output_dir / "predictions"
    maybe_mkdir_p(str(pred_dir))

    df = pd.read_csv(csv_path)
    id_col = detect_id_column(df)
    image_col = detect_image_column(df)
    if args.max_cases > 0:
        df = df.head(args.max_cases).copy()

    rw = SimpleITKIO()
    label_space = "compact_0_46" if args.output_label_space == "compact" else "original_task1_gc"
    results = []
    qa_rows = []
    reference_spacing_xyz = parse_spacing(args.reference_spacing)

    print("Using original U-Mamba2 ToothFairy3 Task1 inference backend")
    print(f"  task1_inference.py: {args.task1_inference_py}")
    print(f"  checkpoint: {model_folder / ('fold_' + fold) / checkpoint_name}")
    print(f"  output_label_space: {label_space}")
    print(f"  tile_step_size: {args.tile_step_size}")
    print(f"  accumulation_device: {args.accumulation_device}")
    print(f"  use_mirroring: {not args.disable_mirroring}")
    print(f"  left_right_mirroring: {not args.disable_left_right_mirroring}")
    print(f"  physical_volume_postprocess: {args.physical_volume_postprocess}")
    print(f"  reference_spacing_xyz: {args.reference_spacing}")
    print(f"  cases: {len(df)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="U-Mamba2 Task1 inference"):
        identifier = str(row[id_col])
        image_path = Path(str(row[image_col]))
        pred_path = pred_dir / f"{identifier}.nii.gz"

        if args.skip_existing and pred_path.exists():
            visual_token_path = ""
            token_path = output_dir / "visual_structure_tokens" / f"{identifier}.npz"
            if args.save_visual_tokens:
                if token_path.exists():
                    visual_token_path = str(token_path)
                else:
                    if not image_path.exists():
                        raise FileNotFoundError(f"{identifier}: image path does not exist: {image_path}")
                    image_existing, props_existing = rw.read_images([image_path])
                    pred_existing, _ = rw.read_seg(str(pred_path))
                    pred_existing = pred_existing[0].astype(np.uint8, copy=False)
                    pred_compact_existing = pred_existing if args.output_label_space == "compact" else convert_original_to_compact(pred_existing, identifier)
                    visual_token_path = save_stage1_inference_visual_tokens(
                        identifier, image_existing, props_existing, pred_compact_existing, predictor, output_dir, args,
                    ) or ""
                    del image_existing, props_existing, pred_existing, pred_compact_existing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            results.append({
                "identifier": identifier,
                "pred_path": str(pred_path),
                "image_path": str(image_path),
                "inference_time": 0.0,
                "skipped_existing": True,
                "label_space": label_space,
                "visual_structure_tokens_npz_path": visual_token_path,
            })
            continue

        if not image_path.exists():
            raise FileNotFoundError(f"{identifier}: image path does not exist: {image_path}")

        image, props = rw.read_images([image_path])
        spacing_xyz = tuple(float(v) for v in props.get("spacing", (1.0, 1.0, 1.0)))
        threshold_scale = 1.0
        if args.physical_volume_postprocess:
            predictor.cls_vol = scaled_class_volume_thresholds(
                predictor._evigre_original_cls_vol,
                spacing_xyz=spacing_xyz,
                reference_spacing_xyz=reference_spacing_xyz,
            )
            threshold_scale = float(np.prod(np.asarray(reference_spacing_xyz, dtype=np.float64)) / np.prod(np.asarray(spacing_xyz, dtype=np.float64)))
        else:
            predictor.cls_vol = dict(predictor._evigre_original_cls_vol)
        start = time.time()
        with torch.no_grad():
            pred = task1_module.predict_semseg(image, props, predictor)
        elapsed = time.time() - start

        pred_for_qa = convert_original_to_compact(pred, identifier)
        visual_token_path = save_stage1_inference_visual_tokens(
            identifier, image, props, pred_for_qa, predictor, output_dir, args,
        )
        if args.output_label_space == "compact":
            pred = pred_for_qa

        rw.write_seg(pred.astype(np.uint8, copy=False), pred_path, props)
        qa = compact_mask_quality_stats(pred_for_qa, spacing_xyz)
        qa.update({
            "identifier": identifier,
            "pred_path": str(pred_path),
            "image_path": str(image_path),
            "inference_time": float(elapsed),
            "label_space": label_space,
            "physical_volume_postprocess": bool(args.physical_volume_postprocess),
            "postprocess_threshold_scale": threshold_scale,
            "skipped_existing": False,
            "visual_structure_tokens_npz_path": visual_token_path or "",
        })
        qa_rows.append(qa)
        results.append({
            "identifier": identifier,
            "pred_path": str(pred_path),
            "image_path": str(image_path),
            "inference_time": float(elapsed),
            "shape": list(pred.shape),
            "label_space": label_space,
            "spacing_xyz": [float(v) for v in spacing_xyz],
            "physical_volume_postprocess": bool(args.physical_volume_postprocess),
            "postprocess_threshold_scale": threshold_scale,
            "visual_structure_tokens_npz_path": visual_token_path or "",
        })

        del image, props, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = float(sum(item["inference_time"] for item in results))
    summary: Dict[str, Any] = {
        "backend": "original_umamba2_task1",
        "model_folder": str(model_folder),
        "fold": fold,
        "checkpoint_name": checkpoint_name,
        "task1_inference_py": str(args.task1_inference_py),
        "output_label_space": label_space,
        "tile_step_size": float(args.tile_step_size),
        "accumulation_device": args.accumulation_device,
        "use_mirroring": not args.disable_mirroring,
        "left_right_mirroring": not args.disable_left_right_mirroring,
        "physical_volume_postprocess": bool(args.physical_volume_postprocess),
        "reference_spacing_xyz": [float(v) for v in reference_spacing_xyz],
        "save_visual_tokens": bool(args.save_visual_tokens),
        "visual_token_levels": parse_int_list(args.visual_token_levels) if args.save_visual_tokens else [],
        "visual_token_schema": VISUAL_TOKEN_SCHEMA if args.save_visual_tokens else "",
        "total_cases": len(results),
        "total_time": total_time,
        "avg_time_per_case": total_time / len(results) if results else 0.0,
    }
    if torch.cuda.is_available():
        summary["peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)

    with (output_dir / "predictions.json").open("w") as f:
        json.dump({"summary": summary, "per_case_results": results}, f, indent=2)
    if qa_rows:
        pd.DataFrame(qa_rows).to_csv(output_dir / "stage1_mask_quality.csv", index=False)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Original U-Mamba2 ToothFairy3 Task1 inference wrapper for EviDent-CBCT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", required=True, help="CSV with raw image_path and identifier/case_id columns")
    parser.add_argument("--output", required=True, help="Output directory; predictions are written under predictions/")
    parser.add_argument("--task1-inference-py", required=True,
                        help="Path to original U-Mamba2 Toothfairy3/task1_inference.py")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Direct path to fold_*/checkpoint_*.pth. Overrides --model-folder/--fold/--checkpoint-name")
    parser.add_argument("--model-folder", default=None,
                        help="Model folder containing fold_* directories")
    parser.add_argument("--fold", default="all", help="Fold name without fold_ prefix, for example 0 or all")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth", help="Checkpoint filename")
    parser.add_argument("--output-label-space", choices=("compact", "original"), default="compact",
                        help="compact gives Stage2-ready 0..46 labels; original exports Grand Challenge Task1 IDs")
    parser.add_argument("--tile-step-size", type=float, default=0.95,
                        help="Tile step size used by original task1_inference.py")
    parser.add_argument("--tta-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accumulation-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Keep full-volume logits on CUDA or use bounded CPU accumulation and classwise resampling.",
    )
    parser.add_argument("--disable-mirroring", action="store_true", default=False)
    parser.add_argument("--disable-left-right-mirroring", action="store_true", default=False,
                        help="Keep mirroring enabled but do not add the left/right axis with label swapping")
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--physical-volume-postprocess", action="store_true", default=True,
                        help="Scale U-Mamba2 connected-component thresholds by voxel volume so they represent the same physical volume across A/P/F/S domains.")
    parser.add_argument("--no-physical-volume-postprocess", dest="physical_volume_postprocess", action="store_false")
    parser.add_argument("--reference-spacing", default="0.3,0.3,0.3",
                        help="x,y,z mm spacing used as the reference for original U-Mamba2 voxel-count thresholds.")
    parser.add_argument("--save-visual-tokens", action="store_true", default=False,
                        help="During Stage1 inference, run a second encoder-only sliding-window pass and save structure-pooled visual tokens aligned to the final postprocessed mask.")
    parser.add_argument("--visual-token-levels", default="2,3,4",
                        help="0-based Stage1 encoder levels to pool into visual structure tokens.")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--max-cases", type=int, default=-1)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
