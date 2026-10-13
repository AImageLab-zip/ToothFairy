"""R178 strict online weak-label scoring for ToothFairy4 Task 1.

This module is deliberately self-contained: it embeds the frozen model
architectures, crop policies, thresholds, and artifact hashes needed to score
weak evidence from an already 0.3 mm-resampled CBCT numpy volume plus a guarded
47-class segmentation mask. Missing or invalid branch artifacts fail closed and
are reported in the returned audit payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F


MODEL_ROOT = Path(os.environ.get("R178_MODEL_ROOT", "/opt/ml/model"))
FIXED_SPACING_XYZ_MM = (0.3, 0.3, 0.3)
R176_FIXED_SIZE = 128

THRESHOLDS = {
    "bone_atrophy": 0.9955627918243408,
    "endodontic": 0.11609826982021332,
    "impacted": 0.8916303515434265,
    "periapical": 0.43828174876843556,
    "periodontal": 0.9417994618415833,
    "sinus_mucosal": 0.48380459907154244,
}

FDI_SEQUENCE = (
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
)
FDI_TO_LABEL = {fdi: label for label, fdi in enumerate(FDI_SEQUENCE, start=11)}
LABEL_TO_FDI = {label: fdi for fdi, label in FDI_TO_LABEL.items()}
FDI_TO_DATASET_NAME = {
    "11": "upper_right_central_incisor",
    "12": "upper_right_lateral_incisor",
    "13": "upper_right_canine",
    "14": "upper_right_first_premolar",
    "15": "upper_right_second_premolar",
    "16": "upper_right_first_molar",
    "17": "upper_right_second_molar",
    "18": "upper_right_third_molar_wisdom_tooth",
    "21": "upper_left_central_incisor",
    "22": "upper_left_lateral_incisor",
    "23": "upper_left_canine",
    "24": "upper_left_first_premolar",
    "25": "upper_left_second_premolar",
    "26": "upper_left_first_molar",
    "27": "upper_left_second_molar",
    "28": "upper_left_third_molar_wisdom_tooth",
    "31": "lower_left_central_incisor",
    "32": "lower_left_lateral_incisor",
    "33": "lower_left_canine",
    "34": "lower_left_first_premolar",
    "35": "lower_left_second_premolar",
    "36": "lower_left_first_molar",
    "37": "lower_left_second_molar",
    "38": "lower_left_third_molar_wisdom_tooth",
    "41": "lower_right_central_incisor",
    "42": "lower_right_lateral_incisor",
    "43": "lower_right_canine",
    "44": "lower_right_first_premolar",
    "45": "lower_right_second_premolar",
    "46": "lower_right_first_molar",
    "47": "lower_right_second_molar",
    "48": "lower_right_third_molar_wisdom_tooth",
}
THIRD_MOLARS = {"18", "28", "38", "48"}
JAW_LABELS = (1, 2)
LOWER_JAW_LABEL = 1
UPPER_JAW_LABEL = 2
LEFT_SINUS_LABEL = 5
RIGHT_SINUS_LABEL = 6
MERGED_PULP_LABEL = 46
SINUS_VISIBILITY_THRESHOLDS = {
    "min_sinus_voxels": 5197.0,
    "min_surface_voxels": 2680.0,
    "min_bbox_z": 8.0,
    "min_bbox_y": 40.0,
    "min_bbox_x": 33.0,
    "ref_sinus_voxels": 23891.0,
    "ref_surface_voxels": 5130.0,
    "ref_bbox_z": 20.0,
    "ref_bbox_y": 73.0,
    "ref_bbox_x": 48.0,
    "max_edge_faces_for_assessable": 1,
    "max_edge_fraction": 0.40045624,
    "min_coverage_score": 0.55,
}

WEAK_ARTIFACTS = {
    "r101_bone_atrophy": {
        "path": "weak_heads/r101/bone_atrophy.pt",
        "sha256": "73626a7f210417d0a60f61391a836bb84a44cc42750e4e40721de48ab364f5d1",
    },
    "r101_endodontic": {
        "path": "weak_heads/r101/endodontic.pt",
        "sha256": "3ec6e758a28aad8751498b6d99f08a232523db9fddab680bb5eeac52dd4a4b11",
    },
    "r101_periodontal": {
        "path": "weak_heads/r101/periodontal.pt",
        "sha256": "5056ccf5fb972bdad75ea3b398f9b19b7bd44026e9286385f888705d9c1e3309",
    },
    "r134d2_impacted": {
        "path": "weak_heads/r134d2/impacted.pt",
        "sha256": "c63727750082ec64eafe9e4ba7b815fa9beea495b6cdc170e57ebfd64a030e84",
    },
    "r176_dolchid_coarse": {
        "path": "weak_heads/r176/r176_dolchid_coarse_proposal.pth",
        "sha256": "b667020d9a6ed13cfe59ba3eaefe582c2bf36a6082fa8fcf602da8ed8f1df4d7",
    },
    "r176_dolchid_refiner": {
        "path": "weak_heads/r176/r176_dolchid_predicted_roi_refiner.pth",
        "sha256": "cfca91e82dafae260c9f8e6c47781a46892d34a4d62311a328caa2e42844261f",
    },
    "r134e_sinus_seed134_fold1": {
        "path": "weak_heads/r134e_sinus/seed_134/fold_1.pt",
        "sha256": "2dd2b4c3b8be5c01f295016d774d2376e3b84df6e803c06a0180a5f2fc291e5b",
    },
    "r134e_sinus_seed134_fold2": {
        "path": "weak_heads/r134e_sinus/seed_134/fold_2.pt",
        "sha256": "f169b80b492b03a175ced28790b123620f26db0fcb83587a8f6a444f232a4b1a",
    },
    "r134e_sinus_seed134_fold3": {
        "path": "weak_heads/r134e_sinus/seed_134/fold_3.pt",
        "sha256": "40852bbbf433dff8c69abf850d0893a58f692dddc7ab8be6deefe6ea498dacea",
    },
    "r134e_sinus_seed134_fold4": {
        "path": "weak_heads/r134e_sinus/seed_134/fold_4.pt",
        "sha256": "fc0ae235ec2bd6c9df17e8addf2d931cd02befc64b380ee6c60dda122fcdae46",
    },
    "r134e_sinus_seed135_fold1": {
        "path": "weak_heads/r134e_sinus/seed_135/fold_1.pt",
        "sha256": "763bda88a3e60f46bc0896c276ee47a5dc61d86dd8721af9b56593d96a421340",
    },
    "r134e_sinus_seed135_fold2": {
        "path": "weak_heads/r134e_sinus/seed_135/fold_2.pt",
        "sha256": "cbadd9c792dc860ece1a181a31a0d4723d9e83b82b821ca31de4e05039853a70",
    },
    "r134e_sinus_seed135_fold3": {
        "path": "weak_heads/r134e_sinus/seed_135/fold_3.pt",
        "sha256": "03901c6867e9af59ad30ba4872083047a06fbe990c3361cb66244d1750652422",
    },
    "r134e_sinus_seed135_fold4": {
        "path": "weak_heads/r134e_sinus/seed_135/fold_4.pt",
        "sha256": "d63f6b4c5e2bac8efe492000495d3428ea48ebe5f256679ec03735a32d792636",
    },
    "r134e_sinus_seed136_fold1": {
        "path": "weak_heads/r134e_sinus/seed_136/fold_1.pt",
        "sha256": "563a9231b798874581d95aa80a156e5246002c275aa26a08dd768bf0326ebfe9",
    },
    "r134e_sinus_seed136_fold2": {
        "path": "weak_heads/r134e_sinus/seed_136/fold_2.pt",
        "sha256": "eb8caa7c09667a4f3981657956415716a87083690f60e0092c83510a9375941f",
    },
    "r134e_sinus_seed136_fold3": {
        "path": "weak_heads/r134e_sinus/seed_136/fold_3.pt",
        "sha256": "4360e9d9255b9f7620e8d2e25944c451ab3caa178a59c6082aaf067725b95061",
    },
    "r134e_sinus_seed136_fold4": {
        "path": "weak_heads/r134e_sinus/seed_136/fold_4.pt",
        "sha256": "72a947f8babfea4c486d45eef34742ac3b2ffb6a1adcd503de9529c1bfec78b5",
    },
}

SINUS_ARTIFACT_IDS = tuple(key for key in WEAK_ARTIFACTS if key.startswith("r134e_sinus_"))

ROI_POLICIES = {
    "bone_atrophy": ("jawbone_region_160_d8", (160, 160, 160), 8),
    "endodontic": ("tooth_pulp_96_d2", (96, 96, 96), 2),
    "periodontal": ("tooth_bone_128_d10", (128, 128, 128), 10),
    "impacted": ("tooth_impaction_context_160_d12", (160, 160, 160), 12),
    "sinus_mucosal": ("sinus_wall_128x160x160_d4", (128, 160, 160), 4),
}
JAWBONE_CANDIDATES = (
    "jawbone_upper_left_anterior",
    "jawbone_upper_right_anterior",
    "jawbone_upper_left_posterior",
    "jawbone_upper_right_posterior",
    "jawbone_lower_left_anterior",
    "jawbone_lower_right_anterior",
    "jawbone_lower_left_posterior",
    "jawbone_lower_right_posterior",
)

SEGMENTATION_LABEL_CONTRACT = {
    "lower_jawbone": LOWER_JAW_LABEL,
    "upper_jawbone": UPPER_JAW_LABEL,
    "left_maxillary_sinus": LEFT_SINUS_LABEL,
    "right_maxillary_sinus": RIGHT_SINUS_LABEL,
    "pulp": MERGED_PULP_LABEL,
    **{
        FDI_TO_DATASET_NAME[fdi]: label
        for fdi, label in FDI_TO_LABEL.items()
    },
}


class Small3DCropHead(nn.Module):
    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, affine: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=affine),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=affine),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class TinyUNet3D(nn.Module):
    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, affine=True)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, affine=True)
        self.bottom = ConvBlock(base_channels * 2, base_channels * 4, affine=True)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2, affine=True)
        self.dec1 = ConvBlock(base_channels * 3, base_channels, affine=True)
        self.head = nn.Conv3d(base_channels, 1, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first = self.enc1(tensor)
        second = self.enc2(F.max_pool3d(first, 2))
        bottom = self.bottom(F.max_pool3d(second, 2))
        up_second = F.interpolate(bottom, size=second.shape[-3:], mode="trilinear", align_corners=False)
        dec_second = self.dec2(torch.cat([up_second, second], dim=1))
        up_first = F.interpolate(dec_second, size=first.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.dec1(torch.cat([up_first, first], dim=1)))


class CompactUNet3D(nn.Module):
    def __init__(self, base: int = 6) -> None:
        super().__init__()
        self.encoder1 = RefinerBlock(1, base)
        self.encoder2 = RefinerBlock(base, base * 2)
        self.bottleneck = RefinerBlock(base * 2, base * 4)
        self.decoder2 = RefinerBlock(base * 4 + base * 2, base * 2)
        self.decoder1 = RefinerBlock(base * 2 + base, base)
        self.output = nn.Conv3d(base, 1, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(tensor)
        second = self.encoder2(F.max_pool3d(first, 2))
        bottleneck = self.bottleneck(F.max_pool3d(second, 2))
        up_second = F.interpolate(bottleneck, size=second.shape[-3:], mode="trilinear", align_corners=False)
        decoded_second = self.decoder2(torch.cat([up_second, second], dim=1))
        up_first = F.interpolate(decoded_second, size=first.shape[-3:], mode="trilinear", align_corners=False)
        return self.output(self.decoder1(torch.cat([up_first, first], dim=1)))


class RefinerBlock(nn.Module):
    """Exact R176 refiner block; its convolutions retain train-time biases."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


def _masked_pool_features(feature: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    resized_roi = F.interpolate(roi, size=feature.shape[-3:], mode="nearest")
    denominator = resized_roi.sum(dim=(2, 3, 4)).clamp_min(1.0)
    mean = (feature * resized_roi).sum(dim=(2, 3, 4)) / denominator
    masked = feature.masked_fill(resized_roi <= 0, torch.finfo(feature.dtype).min)
    maximum = masked.amax(dim=(2, 3, 4))
    empty = resized_roi.sum(dim=(2, 3, 4)) == 0
    maximum = torch.where(empty.expand_as(maximum), torch.zeros_like(maximum), maximum)
    return torch.cat([mean, maximum], dim=1)


def _masked_raw_image_statistics(image: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    denominator = roi.sum(dim=(2, 3, 4)).clamp_min(1.0)
    mean = (image * roi).sum(dim=(2, 3, 4)) / denominator
    variance = (((image - mean[:, :, None, None, None]) ** 2) * roi).sum(dim=(2, 3, 4)) / denominator
    std = torch.sqrt(variance.clamp_min(1e-8))
    soft_air = torch.sigmoid((-0.5 - image) * 8.0)
    soft_air_fraction = (soft_air * roi).sum(dim=(2, 3, 4)) / denominator
    empty = roi.sum(dim=(2, 3, 4)) == 0
    output = torch.cat([mean, std, soft_air_fraction], dim=1)
    return torch.where(empty.expand_as(output), torch.zeros_like(output), output)


class RoiGatedMultiScaleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (16, 32, 64, 96)
        stages = []
        in_channels = 2
        for out_channels in channels:
            stages.append(_R134EConvStage(in_channels, out_channels))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(
            nn.Linear(2 * sum(channels) + 2 * channels[-1] + 3, 128),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError("R134E models expect [B,2,D,H,W] image+roi_mask input")
        image = x[:, :1]
        roi = x[:, 1:2]
        feature = x
        pooled = []
        for stage in self.stages:
            feature = stage(feature)
            pooled.append(_masked_pool_features(feature, roi))
        global_context = torch.cat(
            [
                F.adaptive_avg_pool3d(feature, 1).flatten(1),
                F.adaptive_max_pool3d(feature, 1).flatten(1),
            ],
            dim=1,
        )
        features = torch.cat([*pooled, global_context, _masked_raw_image_statistics(image, roi)], dim=1)
        return self.head(features)


class _R134EConvStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels >= 32 else 4
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True)
class CropBounds:
    start: tuple[int, int, int]
    end: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.end))  # type: ignore[return-value]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifacts(required_ids: Sequence[str], model_root: Path = MODEL_ROOT) -> dict[str, Any]:
    rows = []
    ok = True
    for artifact_id in required_ids:
        spec = WEAK_ARTIFACTS[artifact_id]
        path = model_root / str(spec["path"])
        row = {"id": artifact_id, "path": str(path), "expected_sha256": spec["sha256"]}
        if not path.is_file():
            row.update({"ok": False, "error": "missing"})
            ok = False
        else:
            actual = _sha256_file(path)
            row.update({"actual_sha256": actual, "ok": actual == spec["sha256"]})
            ok = ok and actual == spec["sha256"]
        rows.append(row)
    return {"ok": ok, "rows": rows}


def verify_segmentation_label_contract(
    model_root: Path = MODEL_ROOT,
) -> dict[str, Any]:
    candidates = sorted(
        (model_root / "nnUNet_results").glob("Dataset317*/**/dataset.json")
    )
    if len(candidates) != 1:
        return {
            "ok": False,
            "error": f"expected one Dataset317 dataset.json, found {len(candidates)}",
            "paths": [str(path) for path in candidates],
        }
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, Mapping):
        return {"ok": False, "error": "dataset.json has no labels mapping", "path": str(path)}
    observed = {str(name): int(value) for name, value in labels.items()}
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in SEGMENTATION_LABEL_CONTRACT.items()
        if observed.get(name) != expected
    }
    unexpected_ids = sorted(
        {
            int(value)
            for value in observed.values()
            if int(value) < 0 or int(value) > 46
        }
    )
    return {
        "ok": not mismatches and not unexpected_ids,
        "path": str(path),
        "mismatches": mismatches,
        "unexpected_ids": unexpected_ids,
        "label_count": len(observed),
    }


def _safe_device(device: str | torch.device) -> torch.device:
    value = torch.device(device)
    if value.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return value


def _state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        state = payload.get("model", payload.get("state_dict", payload))
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint payload does not contain a state dict")
    return state  # type: ignore[return-value]


def _load_model(model: nn.Module, path: Path, device: torch.device) -> tuple[nn.Module, Mapping[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(_state_dict_from_payload(payload), strict=True)
    model.to(device).eval()
    return model, payload if isinstance(payload, Mapping) else {}


def _sigmoid_score(model: nn.Module, x: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        tensor = torch.from_numpy(x[None].astype(np.float32, copy=False)).to(device)
        score = torch.sigmoid(model(tensor)).detach().cpu().reshape(-1)[0].item()
    if not math.isfinite(score):
        raise ValueError(f"invalid score: {score!r}")
    return float(score)


def _validate_inputs(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image)
    mask = np.asarray(mask)
    if image.ndim != 3 or mask.ndim != 3 or image.shape != mask.shape:
        raise ValueError(f"image and guarded_mask must be aligned 3D arrays, got {image.shape} and {mask.shape}")
    if not np.isfinite(image).any():
        raise ValueError("image has no finite voxels")
    if int(mask.min(initial=0)) < 0 or int(mask.max(initial=0)) > 46:
        raise ValueError("guarded_mask contains labels outside the 47-class range 0..46")
    return image.astype(np.float32, copy=False), mask.astype(np.int16, copy=False)


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    crop_f = crop.astype(np.float32, copy=False)
    finite = crop_f[np.isfinite(crop_f)]
    if finite.size == 0:
        return np.zeros(crop_f.shape, dtype=np.float16)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    crop_f = np.clip(crop_f, lo, hi)
    return ((crop_f - lo) / (hi - lo) * 2.0 - 1.0).astype(
        np.float16, copy=False
    )


def normalize_volume(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("image has no finite voxels")
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    scale = max(float(hi - lo), 1e-6)
    return np.clip((image - float(lo)) / scale, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_volume_r151(image: np.ndarray) -> np.ndarray:
    """Reproduce the volume normalization used by the frozen R151 R101 path."""
    image_f = image.astype(np.float32, copy=False)
    finite = image_f[np.isfinite(image_f)]
    if finite.size == 0:
        return np.zeros_like(image_f, dtype=np.float16)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(image_f, dtype=np.float16)
    normalized = np.clip(image_f, lo, hi)
    normalized = (normalized - float(lo)) / float(hi - lo)
    return (normalized * 2.0 - 1.0).astype(np.float16, copy=False)


def tensor_signature(value: np.ndarray) -> dict[str, Any]:
    """Return a compact deterministic signature for parity diagnostics."""
    array = np.ascontiguousarray(value.astype(np.float32, copy=False))
    channels = array if array.ndim == 4 else array[None]
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "channels": [
            {
                "minimum": float(np.min(channel)),
                "maximum": float(np.max(channel)),
                "mean": float(np.mean(channel, dtype=np.float64)),
                "standard_deviation": float(
                    np.std(channel, dtype=np.float64)
                ),
                "nonzero_voxels": int(np.count_nonzero(channel)),
            }
            for channel in channels
        ],
    }


def resize_volume(volume: np.ndarray, size: int, *, order: int) -> np.ndarray:
    return ndimage.zoom(volume, [size / axis for axis in volume.shape], order=order)


def bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(axis=0), coords.max(axis=0)


def crop_bounds_for_center(center: Sequence[float], volume_shape: Sequence[int], crop_shape: Sequence[int]) -> CropBounds:
    start = []
    end = []
    for c, dim, size in zip(center, volume_shape, crop_shape):
        lo = int(round(float(c) - float(size) / 2.0))
        lo = max(0, min(lo, int(dim) - int(size))) if dim >= size else 0
        hi = min(int(dim), lo + int(size))
        start.append(lo)
        end.append(hi)
    return CropBounds(tuple(start), tuple(end))


def crop_with_padding(array: np.ndarray, center: Sequence[float], shape: Sequence[int], fill: float) -> np.ndarray:
    shape_t = tuple(int(v) for v in shape)
    out = np.full(shape_t, fill, dtype=array.dtype)
    starts = [int(round(float(center[axis]) - shape_t[axis] / 2.0)) for axis in range(3)]
    src_slices = []
    dst_slices = []
    for axis, start in enumerate(starts):
        end = start + shape_t[axis]
        src_start = max(0, start)
        src_end = min(array.shape[axis], end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        if src_end <= src_start:
            return out
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = array[tuple(src_slices)]
    return out


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or not np.any(mask):
        return mask.astype(bool, copy=False)
    return ndimage.binary_dilation(mask.astype(bool), iterations=iterations)


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or not np.any(mask):
        return mask.astype(bool, copy=False)
    return ndimage.binary_erosion(mask.astype(bool), iterations=iterations)


def _centered_input(image: np.ndarray, roi: np.ndarray, shape: Sequence[int], dilation: int) -> tuple[np.ndarray, int, int]:
    bounds = bbox_from_mask(roi)
    if bounds is None:
        raise ValueError("empty ROI")
    lo, hi = bounds
    center = (lo + hi) / 2.0
    image_crop = crop_with_padding(image, center, shape, fill=float(np.median(image)))
    roi_crop = crop_with_padding(roi.astype(np.uint8), center, shape, fill=0) > 0
    roi_crop = dilate(roi_crop, dilation)
    x = np.stack([normalize_crop(image_crop), roi_crop.astype(np.float32)], axis=0)
    return x.astype(np.float32, copy=False), int(roi.sum()), int(roi_crop.sum())


def _tooth_mask(mask: np.ndarray, fdi: str) -> np.ndarray:
    # Dataset317 has one merged pulp class, whereas the frozen R101/R134D2
    # builders only attached tooth-specific pulp labels. Their online input is
    # therefore the exact FDI tooth mask.
    return mask == FDI_TO_LABEL[fdi]


def _tooth_context(mask: np.ndarray, fdi: str, label: str) -> np.ndarray:
    tooth = _tooth_mask(mask, fdi)
    if label in {"endodontic", "periodontal", "impacted"}:
        return tooth
    raise ValueError(f"unsupported tooth context label: {label}")


def _tooth_centroids(mask: np.ndarray) -> dict[str, np.ndarray]:
    centers = {}
    for fdi, label in FDI_TO_LABEL.items():
        coords = np.argwhere(mask == label)
        if coords.shape[0] >= 16:
            centers[fdi] = coords.mean(axis=0)
    return centers


def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    coordinates = np.argwhere(mask)
    return coordinates.mean(axis=0) if coordinates.size else None


def _clamp_to_mask_bbox(
    point: np.ndarray, region_mask: np.ndarray, margin: int = 8
) -> np.ndarray:
    bounds = bbox_from_mask(region_mask)
    if bounds is None:
        return np.minimum(
            np.maximum(point, 0.0),
            np.asarray(region_mask.shape, dtype=np.float64) - 1.0,
        )
    lo, hi = bounds
    lo_f = lo.astype(np.float64) + margin
    hi_f = np.maximum(hi.astype(np.float64) - margin, lo_f)
    return np.minimum(np.maximum(point, lo_f), hi_f)


def _same_quadrant_centroids(
    mask: np.ndarray, fdi: str
) -> dict[int, np.ndarray]:
    quadrant = fdi[0]
    return {
        int(candidate[1]): center
        for candidate, center in _tooth_centroids(mask).items()
        if candidate[0] == quadrant
    }


def _estimate_third_molar_center(
    mask: np.ndarray, fdi: str, region_mask: np.ndarray
) -> tuple[np.ndarray | None, str]:
    target = 8
    centroids = _same_quadrant_centroids(mask, fdi)
    if 7 in centroids and 6 in centroids:
        center = centroids[7] + (centroids[7] - centroids[6])
        return _clamp_to_mask_bbox(center, region_mask), "extrapolate_from_mesial_7_6"
    present = sorted(centroids)
    if len(present) >= 2:
        first, second = present[-2:]
        delta = (centroids[second] - centroids[first]) / max(1, second - first)
        center = centroids[second] + (target - second) * delta
        return (
            _clamp_to_mask_bbox(center, region_mask),
            f"linear_from_positions_{first}_{second}",
        )
    region_center = _mask_centroid(region_mask)
    if present and region_center is not None:
        position = present[-1]
        center = 0.55 * centroids[position] + 0.45 * region_center
        return (
            _clamp_to_mask_bbox(center, region_mask),
            f"blend_tooth_{position}_with_region",
        )
    if region_center is not None:
        return region_center, "region_centroid"
    return None, "no_context"


def _ellipsoid_mask(
    shape: Sequence[int],
    center: np.ndarray,
    radius_zyx: Sequence[float] = (38.0, 48.0, 48.0),
) -> np.ndarray:
    shape_i = np.asarray(shape, dtype=np.int64)
    radius = np.asarray(radius_zyx, dtype=np.float64)
    lo = np.maximum(np.floor(center - radius).astype(np.int64), 0)
    hi = np.minimum(np.ceil(center + radius).astype(np.int64), shape_i - 1)
    slices = tuple(
        slice(int(lo[axis]), int(hi[axis]) + 1) for axis in range(3)
    )
    grids = np.ogrid[slices]
    local_shape = tuple(int(hi[axis] - lo[axis] + 1) for axis in range(3))
    distance = np.zeros(local_shape, dtype=np.float32)
    for axis, grid in enumerate(grids):
        distance += (
            (grid.astype(np.float32) - float(center[axis])) / float(radius[axis])
        ) ** 2
    output = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    output[slices] = distance <= 1.0
    return output


def estimated_third_molar_slot_roi(
    mask: np.ndarray, fdi: str
) -> tuple[np.ndarray, dict[str, Any]]:
    quadrant_regions = {
        "1": "jawbone_upper_right_posterior",
        "2": "jawbone_upper_left_posterior",
        "3": "jawbone_lower_left_posterior",
        "4": "jawbone_lower_right_posterior",
    }
    region_id = quadrant_regions[fdi[0]]
    region = jawbone_subregion(mask, region_id)
    center, strategy = _estimate_third_molar_center(mask, fdi, region)
    if center is None:
        return np.zeros(mask.shape, dtype=bool), {
            "route_source": "missing_arch_recovery_context",
            "center_strategy": strategy,
            "arch_recovery_guard_pass": False,
        }
    slot = _ellipsoid_mask(mask.shape, center)
    roi = np.logical_and(slot, region) if np.any(region) else slot
    if not np.any(roi):
        roi = slot
    return roi, {
        "route_source": "estimated_posterior_third_molar_slot_roi",
        "center_strategy": strategy,
        "arch_recovery_guard_pass": bool(np.any(roi)),
        "center_zyx": [round(float(value), 3) for value in center],
    }


def _split_half(mask: np.ndarray, axis: int, threshold: float, want_high: bool) -> np.ndarray:
    coords = np.argwhere(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if coords.size:
        keep = coords[:, axis] >= threshold if want_high else coords[:, axis] <= threshold
        if np.any(keep):
            out[tuple(coords[keep].T)] = True
    return out


def _midpoint_from_teeth(
    centers: Mapping[str, np.ndarray],
    jaw: str,
    first_selector: Any,
    second_selector: Any,
    axis: int,
) -> tuple[float | None, bool | None]:
    first = []
    second = []
    for fdi, center in centers.items():
        if jaw == "upper" and fdi[0] not in {"1", "2"}:
            continue
        if jaw == "lower" and fdi[0] not in {"3", "4"}:
            continue
        if first_selector(fdi):
            first.append(float(center[axis]))
        elif second_selector(fdi):
            second.append(float(center[axis]))
    if not first or not second:
        return None, None
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    return (first_mean + second_mean) / 2.0, first_mean >= second_mean


def jawbone_subregion(mask: np.ndarray, roi_id: str) -> np.ndarray:
    region = roi_id.removeprefix("jawbone_")
    parts = set(region.split("_"))
    jaw = "upper" if "upper" in parts else "lower"
    base = mask == (UPPER_JAW_LABEL if jaw == "upper" else LOWER_JAW_LABEL)
    if not np.any(base):
        return base
    out = base
    centers = _tooth_centroids(mask)
    if "left" in parts or "right" in parts:
        side = "left" if "left" in parts else "right"
        midpoint, left_is_high = _midpoint_from_teeth(
            centers, jaw, lambda fdi: fdi[0] in {"2", "3"}, lambda fdi: fdi[0] in {"1", "4"}, axis=2
        )
        coords = np.argwhere(out)
        if midpoint is None or left_is_high is None:
            midpoint = float(np.median(coords[:, 2])) if coords.size else 0.0
            left_is_high = True
        out = _split_half(out, 2, midpoint, left_is_high if side == "left" else not left_is_high)
    if "anterior" in parts or "posterior" in parts:
        ap = "anterior" if "anterior" in parts else "posterior"
        midpoint, anterior_is_high = _midpoint_from_teeth(
            centers, jaw, lambda fdi: fdi[1] in {"1", "2", "3"}, lambda fdi: fdi[1] in {"4", "5", "6", "7", "8"}, axis=1
        )
        coords = np.argwhere(out)
        if midpoint is None or anterior_is_high is None:
            midpoint = float(np.median(coords[:, 1])) if coords.size else 0.0
            anterior_is_high = False
        out = _split_half(out, 1, midpoint, anterior_is_high if ap == "anterior" else not anterior_is_high)
    return out


def sinus_shell(mask: np.ndarray, side: str, erosion: int = 8) -> np.ndarray:
    sinus_label = LEFT_SINUS_LABEL if side == "left" else RIGHT_SINUS_LABEL
    sinus = mask == sinus_label
    if not np.any(sinus):
        return sinus
    inner = erode(sinus, erosion)
    shell = np.logical_and(sinus, np.logical_not(inner))
    return shell if np.any(shell) else sinus


def _one_voxel_surface(binary: np.ndarray) -> int:
    if not np.any(binary):
        return 0
    interior = erode(binary, 1)
    return int(np.count_nonzero(np.logical_and(binary, np.logical_not(interior))))


def sinus_visibility(sinus: np.ndarray, edge_margin: int = 2) -> dict[str, Any]:
    coords = np.argwhere(sinus)
    if coords.size == 0:
        return {
            "visible": False,
            "state": "unknown_no_sinus_mask",
            "coverage_score": 0.0,
            "reasons": ["no_sinus_mask"],
        }
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    extent = hi - lo + 1
    edge_flags: list[bool] = []
    near_edge = np.zeros(coords.shape[0], dtype=bool)
    for axis in range(3):
        low = coords[:, axis] < edge_margin
        high = coords[:, axis] >= sinus.shape[axis] - edge_margin
        edge_flags.extend([bool(np.any(low)), bool(np.any(high))])
        near_edge |= low | high
    row = {
        "sinus_voxels": int(coords.shape[0]),
        "surface_voxels": _one_voxel_surface(sinus),
        "bbox_z": int(extent[0]),
        "bbox_y": int(extent[1]),
        "bbox_x": int(extent[2]),
        "edge_face_count": int(sum(edge_flags)),
        "edge_voxel_fraction": float(np.count_nonzero(near_edge) / coords.shape[0]),
    }
    thresholds = SINUS_VISIBILITY_THRESHOLDS
    reasons = [
        f"low_{field}"
        for field in ("sinus_voxels", "surface_voxels", "bbox_z", "bbox_y", "bbox_x")
        if float(row[field]) < float(thresholds[f"min_{field}"])
    ]
    volume_score = min(
        1.0, row["sinus_voxels"] / float(thresholds["ref_sinus_voxels"])
    )
    surface_score = min(
        1.0, row["surface_voxels"] / float(thresholds["ref_surface_voxels"])
    )
    extent_score = min(
        min(1.0, row[field] / float(thresholds[f"ref_{field}"]))
        for field in ("bbox_z", "bbox_y", "bbox_x")
    )
    edge_faces = int(row["edge_face_count"])
    edge_penalty = 1.0 if edge_faces == 0 else 0.85 if edge_faces == 1 else 0.55
    coverage = round(
        float(
            (0.40 * volume_score + 0.25 * surface_score + 0.35 * extent_score)
            * edge_penalty
        ),
        6,
    )
    if reasons:
        state = "unknown_small_support"
    elif edge_faces > int(thresholds["max_edge_faces_for_assessable"]):
        state, reasons = "unknown_truncated", ["multiple_volume_edges"]
    elif float(row["edge_voxel_fraction"]) > float(
        thresholds["max_edge_fraction"]
    ):
        state, reasons = "unknown_truncated", ["high_edge_voxel_fraction"]
    elif coverage < float(thresholds["min_coverage_score"]):
        state, reasons = "unknown_low_coverage_score", ["low_coverage_score"]
    elif edge_faces == 1:
        state, reasons = "visible_partial_assessable", ["single_volume_edge"]
    else:
        state, reasons = "visible", []
    return {
        **row,
        "visible": state in {"visible", "visible_partial_assessable"},
        "state": state,
        "coverage_score": coverage,
        "reasons": reasons,
    }


def sinus_model_input(
    image: np.ndarray,
    sinus: np.ndarray,
    crop_shape: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    bounds = bbox_from_mask(sinus)
    if bounds is None:
        raise ValueError("empty sinus ROI")
    lo, hi = bounds
    center = (lo + hi) / 2.0
    image_crop = crop_with_padding(
        image, center, crop_shape, fill=float(np.median(image))
    )
    sinus_crop = crop_with_padding(
        sinus.astype(np.uint8), center, crop_shape, fill=0
    ).astype(bool)
    shell = dilate(
        sinus_shell_from_binary(sinus, erosion=8),
        iterations=4,
    )
    shell_crop = crop_with_padding(
        shell.astype(np.uint8), center, crop_shape, fill=0
    ).astype(bool)
    full_sinus = int(sinus.sum())
    full_shell = int(shell.sum())
    retained_sinus = int(sinus_crop.sum()) / max(1, full_sinus)
    retained_shell = int(shell_crop.sum()) / max(1, full_shell)
    x = np.stack(
        [
            normalize_crop(image_crop).astype(np.float32),
            shell_crop.astype(np.float32),
        ],
        axis=0,
    )
    return x, {
        "full_sinus_voxels": full_sinus,
        "full_mucosal_shell_voxels": full_shell,
        "sinus_crop_voxels": int(sinus_crop.sum()),
        "mucosal_shell_crop_voxels": int(shell_crop.sum()),
        "retained_sinus_fraction": retained_sinus,
        "retained_shell_fraction": retained_shell,
        "crop_retention_guard_pass": bool(
            retained_sinus >= 0.99 and retained_shell >= 0.99
        ),
    }


def sinus_shell_from_binary(sinus: np.ndarray, erosion: int = 8) -> np.ndarray:
    if not np.any(sinus):
        return sinus.astype(bool, copy=False)
    inner = erode(sinus, erosion)
    shell = np.logical_and(sinus, np.logical_not(inner))
    return shell if np.any(shell) else sinus.astype(bool, copy=False)


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    labels, count = ndimage.label(binary.astype(bool))
    sizes = ndimage.sum(binary.astype(np.int32), labels, index=np.arange(1, count + 1))
    return labels, int(count), [int(v) for v in sizes]


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels, count, sizes = _connected_components(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool), {
            "component_count": 0,
            "main_component_voxel_count": 0,
            "largest_component_fraction": 0.0,
        }
    largest_id = int(np.argmax(sizes)) + 1
    largest = labels == largest_id
    return largest, {
        "component_count": count,
        "main_component_voxel_count": int(sizes[largest_id - 1]),
        "largest_component_fraction": float(sizes[largest_id - 1] / max(1, int(mask.sum()))),
    }


def impacted_mask_guard(mask: np.ndarray, fdi: str) -> tuple[bool, np.ndarray, dict[str, Any]]:
    raw = mask == FDI_TO_LABEL[fdi]
    largest, audit = _largest_component(raw)
    raw_bbox = bbox_from_mask(raw)
    main_bbox = bbox_from_mask(largest)
    if raw_bbox is None or main_bbox is None:
        audit.update({"mask_integrity_state": "empty", "raw_to_main_bbox_volume_ratio": math.inf})
        return False, largest, audit
    raw_volume = int(np.prod(raw_bbox[1] - raw_bbox[0] + 1))
    main_volume = int(np.prod(main_bbox[1] - main_bbox[0] + 1))
    ratio = float(raw_volume / max(1, main_volume))
    ok = (
        audit["main_component_voxel_count"] >= 16
        and audit["largest_component_fraction"] >= 0.95
        and ratio <= 2.0
    )
    audit.update(
        {
            "mask_integrity_state": "pass" if ok else "fail",
            "raw_to_main_bbox_volume_ratio": ratio,
            "raw_bbox_expansion_flag": ratio > 2.0,
            "raw_bbox_expansion_guard_pass": ratio <= 2.0,
        }
    )
    return ok, largest, audit


def impacted_candidate_roi(
    mask: np.ndarray, fdi: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Route an impacted candidate without promoting an anatomy-free slot guess."""
    exact_present = bool(np.any(mask == FDI_TO_LABEL[fdi]))
    if exact_present:
        guard_ok, roi, exact_audit = impacted_mask_guard(mask, fdi)
        if guard_ok:
            return roi, {
                **exact_audit,
                "route_source": "exact_semantic_roi",
                "exact_semantic_present": True,
                "arch_recovery_guard_pass": False,
                "writer_candidate_eligible": True,
            }

        fallback_roi, fallback_audit = estimated_third_molar_slot_roi(mask, fdi)
        return fallback_roi, {
            **fallback_audit,
            "route_source": "estimated_posterior_third_molar_slot_roi_after_exact_guard_fail",
            "exact_semantic_present": True,
            "exact_mask_integrity_state": exact_audit.get("mask_integrity_state"),
            "exact_raw_bbox_expansion_guard_pass": exact_audit.get(
                "raw_bbox_expansion_guard_pass"
            ),
            "writer_candidate_eligible": bool(
                fallback_audit.get("arch_recovery_guard_pass")
            ),
        }

    roi, audit = estimated_third_molar_slot_roi(mask, fdi)
    return roi, {
        **audit,
        "exact_semantic_present": False,
        "writer_candidate_eligible": False,
    }


def _ranked_components(probability: np.ndarray, threshold: float, maximum: int = 5, minimum_voxels: int = 8) -> list[dict[str, Any]]:
    labels, count, sizes = _connected_components(probability >= threshold)
    rows = []
    for component_id in range(1, count + 1):
        size = sizes[component_id - 1]
        if size < minimum_voxels:
            continue
        mask = labels == component_id
        rows.append(
            {
                "mask": mask,
                "voxels": size,
                "coarse_mean": float(probability[mask].mean()),
                "coarse_max": float(probability[mask].max()),
            }
        )
    rows.sort(key=lambda row: (row["coarse_mean"], row["voxels"]), reverse=True)
    return rows[:maximum]


def _fixed_index_to_physical(
    index_zyx: Sequence[float],
    geometry: Mapping[str, Any],
) -> np.ndarray:
    shape_zyx = np.asarray(geometry["shape_zyx"], dtype=np.float64)
    source_index_zyx = (
        (np.asarray(index_zyx, dtype=np.float64) + 0.5)
        * shape_zyx
        / float(R176_FIXED_SIZE)
        - 0.5
    )
    index_xyz = source_index_zyx[::-1]
    spacing = np.asarray(geometry["spacing_xyz"], dtype=np.float64)
    direction = np.asarray(geometry["direction"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(geometry["origin_xyz"], dtype=np.float64)
    return origin + direction @ (index_xyz * spacing)


def _tooth_apex_points(
    mask: np.ndarray,
    geometry: Mapping[str, Any] | None = None,
    cap_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    points = {}
    for label, fdi in LABEL_TO_FDI.items():
        coords = np.argwhere(mask == label)
        if coords.size == 0:
            continue
        if geometry is None:
            physical = coords.astype(np.float64)
            physical_z = physical[:, 0]
        else:
            physical = np.stack(
                [_fixed_index_to_physical(index, geometry) for index in coords],
                axis=0,
            )
            physical_z = physical[:, 2]
        if fdi[0] in {"1", "2"}:
            cap = physical[
                physical_z >= np.quantile(physical_z, 1.0 - cap_fraction)
            ]
        else:
            cap = physical[physical_z <= np.quantile(physical_z, cap_fraction)]
        if cap.size:
            points[fdi] = cap.mean(axis=0)
    return points


def _nearest_apex(
    candidate_mask: np.ndarray,
    apex_points: Mapping[str, np.ndarray],
    original_shape: Sequence[int],
    geometry: Mapping[str, Any] | None = None,
) -> tuple[str, float]:
    coords = np.argwhere(candidate_mask)
    if coords.size == 0 or not apex_points:
        return "", float("inf")
    fixed_center = coords.mean(axis=0)
    if geometry is None:
        scale = np.asarray(original_shape, dtype=np.float64) / float(
            R176_FIXED_SIZE
        )
        center = fixed_center
    else:
        center = _fixed_index_to_physical(fixed_center, geometry)
    best_fdi = ""
    best_dist = float("inf")
    for fdi, point in apex_points.items():
        delta = center - point
        if geometry is None:
            delta = delta * scale * FIXED_SPACING_XYZ_MM[0]
        dist = float(np.linalg.norm(delta))
        if dist < best_dist:
            best_fdi, best_dist = fdi, dist
    return best_fdi, best_dist


def _score_crop_head_branch(
    label: str,
    artifact_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    device: torch.device,
    audit: dict[str, Any],
    score_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    model_root: Path,
) -> None:
    verify = verify_artifacts([artifact_id], model_root)
    audit["artifact_checks"].extend(verify["rows"])
    if not verify["ok"]:
        audit["branches"][label] = {"ok": False, "error": "artifact verification failed"}
        return
    profile, crop_shape, dilation = ROI_POLICIES[label]
    model, _ = _load_model(Small3DCropHead(in_channels=2), model_root / WEAK_ARTIFACTS[artifact_id]["path"], device)
    threshold = THRESHOLDS[label]
    capture_signatures = os.environ.get(
        "R179_CAPTURE_TENSOR_SIGNATURES", ""
    ).strip().lower() in {"1", "true", "yes"}
    scored = 0
    if label == "bone_atrophy":
        selected[label] = False
        for roi_id in JAWBONE_CANDIDATES:
            roi = jawbone_subregion(mask, roi_id)
            if not np.any(roi):
                continue
            x, roi_voxels, roi_crop_voxels = _centered_input(image, roi, crop_shape, dilation)
            score = _sigmoid_score(model, x, device)
            pred = score >= threshold
            selected[label] = bool(selected[label] or pred)
            score_rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "roi_id": roi_id,
                    "scope": "regional_case",
                    "score": score,
                    "threshold": threshold,
                    "selected": pred,
                    "crop_profile": profile,
                    "roi_voxels": roi_voxels,
                    "roi_crop_voxels": roi_crop_voxels,
                    **(
                        {"tensor_signature": tensor_signature(x)}
                        if capture_signatures
                        else {}
                    ),
                }
            )
            scored += 1
    else:
        if label == "periodontal":
            selected[label] = False
        else:
            selected[label] = []
        teeth = [
            fdi for fdi in FDI_SEQUENCE if np.any(mask == FDI_TO_LABEL[fdi])
        ]
        if label == "impacted":
            teeth = sorted(THIRD_MOLARS, key=int)
        for fdi in teeth:
            route_audit: dict[str, Any] = {}
            if label == "impacted":
                roi, route_audit = impacted_candidate_roi(mask, fdi)
                if not route_audit["arch_recovery_guard_pass"] and not np.any(roi):
                    score_rows.append(
                        {
                            "case_id": case_id,
                            "label": label,
                            "tooth_fdi": fdi,
                            "score": 0.0,
                            "threshold": threshold,
                            "selected": False,
                            "mask_integrity_guard_pass": None,
                            "raw_bbox_expansion_guard_pass": None,
                            **route_audit,
                        }
                    )
                    continue
            else:
                roi = _tooth_context(mask, fdi, label)
            if not np.any(roi):
                continue
            x, roi_voxels, roi_crop_voxels = _centered_input(image, roi, crop_shape, dilation)
            score = _sigmoid_score(model, x, device)
            pred = score >= threshold
            if label == "impacted":
                pred = pred and bool(route_audit["writer_candidate_eligible"])
            if pred:
                if label == "periodontal":
                    selected[label] = True
                else:
                    selected[label].append(fdi)
            row = {
                "case_id": case_id,
                "label": label,
                "tooth_fdi": fdi,
                "scope": "case_only" if label == "periodontal" else "tooth_fdi",
                "score": score,
                "threshold": threshold,
                "selected": pred,
                "crop_profile": profile,
                "roi_voxels": roi_voxels,
                "roi_crop_voxels": roi_crop_voxels,
            }
            if capture_signatures:
                row["tensor_signature"] = tensor_signature(x)
            if label == "impacted":
                row.update(route_audit)
            score_rows.append(row)
            scored += 1
    audit["branches"][label] = {"ok": True, "crops_scored": scored}


def _score_r176_periapical(
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    geometry: Mapping[str, Any] | None,
    device: torch.device,
    audit: dict[str, Any],
    score_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    model_root: Path,
) -> None:
    label = "periapical"
    selected[label] = False
    verify = verify_artifacts(["r176_dolchid_coarse", "r176_dolchid_refiner"], model_root)
    audit["artifact_checks"].extend(verify["rows"])
    if not verify["ok"]:
        audit["branches"][label] = {"ok": False, "error": "artifact verification failed"}
        return
    coarse_payload = torch.load(model_root / WEAK_ARTIFACTS["r176_dolchid_coarse"]["path"], map_location=device, weights_only=False)
    coarse_base = int(coarse_payload.get("args", {}).get("base_channels", 8)) if isinstance(coarse_payload, Mapping) else 8
    coarse = TinyUNet3D(base_channels=coarse_base)
    coarse.load_state_dict(_state_dict_from_payload(coarse_payload), strict=True)
    coarse.to(device).eval()
    coarse_threshold = float(coarse_payload.get("threshold", 0.5)) if isinstance(coarse_payload, Mapping) else 0.5

    refiner_payload = torch.load(model_root / WEAK_ARTIFACTS["r176_dolchid_refiner"]["path"], map_location=device, weights_only=False)
    refiner_base = int(refiner_payload.get("args", {}).get("refiner_base_channels", 6)) if isinstance(refiner_payload, Mapping) else 6
    refiner = CompactUNet3D(base=refiner_base)
    refiner.load_state_dict(_state_dict_from_payload(refiner_payload), strict=True)
    refiner.to(device).eval()

    fixed_image = resize_volume(normalize_volume(image), R176_FIXED_SIZE, order=1).astype(np.float32)
    fixed_mask = resize_volume(mask, R176_FIXED_SIZE, order=0).astype(np.int16)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        tensor = torch.from_numpy(fixed_image[None, None]).to(device)
        coarse_prob = torch.sigmoid(coarse(tensor))[0, 0].float().cpu().numpy()
    components = _ranked_components(coarse_prob, coarse_threshold, maximum=5, minimum_voxels=8)
    fallback_used = False
    proposal_threshold = coarse_threshold
    if not components:
        proposal_threshold = min(coarse_threshold, max(0.1, float(np.quantile(coarse_prob, 0.999))))
        components = _ranked_components(coarse_prob, proposal_threshold, maximum=5, minimum_voxels=8)
        fallback_used = proposal_threshold < coarse_threshold
    apex_points = _tooth_apex_points(fixed_mask, geometry)
    threshold = THRESHOLDS[label]
    for rank, component in enumerate(components, start=1):
        center = np.argwhere(component["mask"]).mean(axis=0)
        bounds = crop_bounds_for_center(
            center, fixed_image.shape, (96, 96, 96)
        )
        crop = fixed_image[bounds.slices].astype(np.float32, copy=False)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            refined_prob = torch.sigmoid(
                refiner(torch.from_numpy(crop[None, None]).to(device))
            )[0, 0].float().cpu().numpy()
        refined_mask = refined_prob >= 0.5
        refined_full = np.zeros(fixed_image.shape, dtype=bool)
        refined_full[bounds.slices] = refined_mask
        route_mask = refined_full if np.any(refined_full) else component["mask"]
        refined_score = float(np.quantile(refined_prob, 0.99))
        refined_branch_score = math.sqrt(
            max(0.0, float(component["coarse_mean"]))
            * max(0.0, refined_score)
        )
        jaw = ndimage.binary_dilation(np.isin(fixed_mask, JAW_LABELS), iterations=3)
        jaw_overlap = float(
            np.logical_and(route_mask, jaw).sum()
            / max(1, int(route_mask.sum()))
        )
        fdi, dist = _nearest_apex(
            route_mask, apex_points, image.shape, geometry
        )
        distance_weight = math.exp(-dist / 8.0) if math.isfinite(dist) else 0.0
        final_score = refined_branch_score * distance_weight
        route_ok = jaw_overlap > 0.05 and dist <= 8.0
        selected_row = bool(route_ok and final_score >= threshold)
        selected[label] = bool(selected[label] or selected_row)
        score_rows.append(
            {
                "case_id": case_id,
                "label": label,
                "scope": "case_only",
                "candidate_id": f"{case_id}_lesion_{rank:02d}",
                "score": final_score,
                "threshold": threshold,
                "selected": selected_row,
                "coarse_score": float(component["coarse_mean"]),
                "coarse_max": float(component["coarse_max"]),
                "refined_score": refined_score,
                "refined_branch_score": refined_branch_score,
                "distance_weight": distance_weight,
                "route_mask_source": (
                    "refined" if np.any(refined_full) else "coarse_fallback"
                ),
                "component_voxels": int(component["voxels"]),
                "refined_voxels": int(refined_full.sum()),
                "proposal_threshold": proposal_threshold,
                "adaptive_fallback_used": fallback_used,
                "jaw_overlap": jaw_overlap,
                "nearest_fdi": fdi,
                "distance_mm": dist if math.isfinite(dist) else "",
                "root_apex_or_jawbone_route_pass": route_ok,
            }
        )
    audit["branches"][label] = {"ok": True, "components_scored": len(components), "adaptive_fallback_used": fallback_used}


def _score_sinus(
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    device: torch.device,
    audit: dict[str, Any],
    score_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    model_root: Path,
) -> None:
    label = "sinus_mucosal"
    selected[label] = False
    verify = verify_artifacts(SINUS_ARTIFACT_IDS, model_root)
    audit["artifact_checks"].extend(verify["rows"])
    if not verify["ok"]:
        audit["branches"][label] = {"ok": False, "error": "artifact verification failed"}
        return
    threshold = THRESHOLDS[label]
    _, crop_shape, dilation = ROI_POLICIES[label]
    side_scores = []
    for side in ("left", "right"):
        sinus_label = LEFT_SINUS_LABEL if side == "left" else RIGHT_SINUS_LABEL
        sinus = mask == sinus_label
        visibility = sinus_visibility(sinus)
        if not visibility["visible"]:
            score_rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "side": side,
                    "score": 0.0,
                    "threshold": threshold,
                    "selected": False,
                    "sinus_fov_visible": False,
                    "image_guard_pass": False,
                    "crop_retention_guard_pass": False,
                    "visibility_state": visibility["state"],
                    "coverage_score": visibility["coverage_score"],
                    "visibility_reasons": visibility["reasons"],
                }
            )
            continue
        x, retention = sinus_model_input(image, sinus, crop_shape)
        if not retention["crop_retention_guard_pass"]:
            score_rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "side": side,
                    "score": 0.0,
                    "threshold": threshold,
                    "selected": False,
                    "sinus_fov_visible": True,
                    "image_guard_pass": True,
                    "crop_retention_guard_pass": False,
                    "visibility_state": visibility["state"],
                    "coverage_score": visibility["coverage_score"],
                    **retention,
                }
            )
            continue
        scores = []
        for artifact_id in SINUS_ARTIFACT_IDS:
            model, _ = _load_model(RoiGatedMultiScaleNet(), model_root / WEAK_ARTIFACTS[artifact_id]["path"], device)
            scores.append(_sigmoid_score(model, x, device))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        score = float(np.mean(scores))
        pred = score >= threshold
        selected[label] = bool(selected[label] or pred)
        side_scores.append(score)
        score_rows.append(
            {
                "case_id": case_id,
                "label": label,
                "side": side,
                "scope": "regional_case",
                "score": score,
                "threshold": threshold,
                "selected": pred,
                "sinus_fov_visible": True,
                "image_guard_pass": True,
                "crop_retention_guard_pass": True,
                "visibility_state": visibility["state"],
                "coverage_score": visibility["coverage_score"],
                "checkpoint_count": len(scores),
                **retention,
            }
        )
    audit["branches"][label] = {"ok": True, "sides_scored": len(side_scores)}


def score_weak_evidence(
    image: np.ndarray,
    guarded_mask: np.ndarray,
    case_id: str,
    device: str | torch.device = "cuda",
    geometry: Mapping[str, Any] | None = None,
    raw_mask: np.ndarray | None = None,
    crop_head_preprocess: str = "online_raw",
    labels: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return selected weak facts, raw score rows, and an audit dictionary.

    The selected facts schema is:
    ``bone_atrophy``/``periodontal``/``periapical``/``sinus_mucosal`` booleans
    and ``endodontic``/``impacted`` sorted FDI tooth arrays.
    """
    selected: dict[str, Any] = {
        "bone_atrophy": False,
        "endodontic": [],
        "impacted": [],
        "periapical": False,
        "periodontal": False,
        "sinus_mucosal": False,
    }
    score_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "schema_version": "r178_weak_runtime_audit_v1",
        "case_id": case_id,
        "model_root": str(MODEL_ROOT),
        "thresholds": dict(THRESHOLDS),
        "crop_head_preprocess": crop_head_preprocess,
        "requested_labels": sorted(labels) if labels is not None else None,
        "artifact_checks": [],
        "branches": {},
        "errors": [],
    }
    label_contract = verify_segmentation_label_contract(MODEL_ROOT)
    audit["segmentation_label_contract"] = label_contract
    if not label_contract["ok"]:
        audit["errors"].append(
            {"branch": "segmentation_label_contract", "error": label_contract}
        )
        return selected, score_rows, audit
    try:
        image_f, mask_i = _validate_inputs(image, guarded_mask)
        raw_mask_i = (
            _validate_inputs(image, raw_mask)[1]
            if raw_mask is not None
            else mask_i
        )
    except Exception as exc:
        audit["errors"].append({"branch": "input", "error": repr(exc)})
        return selected, score_rows, audit

    if crop_head_preprocess == "online_raw":
        crop_head_image = image_f
    elif crop_head_preprocess == "r151_volume_normalized":
        crop_head_image = normalize_volume_r151(image_f)
    else:
        audit["errors"].append(
            {
                "branch": "crop_head_preprocess",
                "error": f"unsupported mode: {crop_head_preprocess}",
            }
        )
        return selected, score_rows, audit

    dev = _safe_device(device)
    branches = (
        ("bone_atrophy", lambda: _score_crop_head_branch("bone_atrophy", "r101_bone_atrophy", crop_head_image, mask_i, case_id, dev, audit, score_rows, selected, MODEL_ROOT)),
        ("endodontic", lambda: _score_crop_head_branch("endodontic", "r101_endodontic", crop_head_image, mask_i, case_id, dev, audit, score_rows, selected, MODEL_ROOT)),
        ("impacted", lambda: _score_crop_head_branch("impacted", "r134d2_impacted", image_f, raw_mask_i, case_id, dev, audit, score_rows, selected, MODEL_ROOT)),
        ("periapical", lambda: _score_r176_periapical(image_f, raw_mask_i, case_id, geometry, dev, audit, score_rows, selected, MODEL_ROOT)),
        ("periodontal", lambda: _score_crop_head_branch("periodontal", "r101_periodontal", crop_head_image, mask_i, case_id, dev, audit, score_rows, selected, MODEL_ROOT)),
        ("sinus_mucosal", lambda: _score_sinus(image_f, raw_mask_i, case_id, dev, audit, score_rows, selected, MODEL_ROOT)),
    )
    available_labels = {branch for branch, _ in branches}
    requested_labels = available_labels if labels is None else set(labels)
    unsupported_labels = sorted(requested_labels - available_labels)
    if unsupported_labels:
        audit["errors"].append(
            {
                "branch": "requested_labels",
                "error": f"unsupported labels: {unsupported_labels}",
            }
        )
        return selected, score_rows, audit
    for branch, run_branch in branches:
        if branch not in requested_labels:
            audit["branches"][branch] = {
                "ok": True,
                "skipped": True,
                "reason": "not_requested",
            }
            continue
        try:
            run_branch()
        except Exception as exc:
            selected[branch] = [] if branch in {"endodontic", "impacted"} else False
            audit["branches"][branch] = {"ok": False, "error": repr(exc)}
            audit["errors"].append({"branch": branch, "error": repr(exc)})
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    selected["endodontic"] = sorted(set(selected["endodontic"]), key=lambda fdi: (int(fdi[0]), int(fdi[1])))
    selected["impacted"] = sorted(set(selected["impacted"]), key=lambda fdi: (int(fdi[0]), int(fdi[1])))
    return selected, score_rows, audit
