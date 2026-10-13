"""Inference-only Stage-2 datasets used to load the released branch checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from evigre.dataloading.dataset_toothfairy_stage2 import (
    FDI_TEETH, parse_size, read_blosc2, normalize_cbct, metal_window,
    resize_3d_tensor, load_npz_crop, path_exists, mask_tensor, semantic_mask_channels,
)

TEN_REGION_NAMES = [
    "upper_right_molars", "upper_right_premolars", "upper_anterior", "upper_left_premolars",
    "upper_left_molars", "lower_left_molars", "lower_left_premolars", "lower_anterior",
    "lower_right_premolars", "lower_right_molars",
]

def tooth_region_id(tooth_id: str, scheme: str = "ten_region") -> int:
    t = str(tooth_id)
    if len(t) != 2 or not t.isdigit(): return 0
    q, p = int(t[0]), int(t[1])
    if str(scheme).lower() not in {"ten", "ten_region", "10", "10_region"}:
        raise ValueError(f"Unsupported tooth_region_scheme={scheme!r}")
    if q == 1 and p >= 6: return 0
    if q == 1 and 4 <= p <= 5: return 1
    if q in {1, 2} and p <= 3: return 2
    if q == 2 and 4 <= p <= 5: return 3
    if q == 2 and p >= 6: return 4
    if q == 3 and p >= 6: return 5
    if q == 3 and 4 <= p <= 5: return 6
    if q in {3, 4} and p <= 3: return 7
    if q == 4 and 4 <= p <= 5: return 8
    if q == 4 and p >= 6: return 9
    return 0


def _load_csvs(csv_paths: str | Sequence[str]) -> pd.DataFrame:
    paths = [str(csv_paths)] if isinstance(csv_paths, (str, Path)) else [str(p) for p in csv_paths]
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    required = ["report_id", "case_id", "image_blosc2_path", "mask_blosc2_path", "stage2_label_json"]
    missing = [key for key in required if key not in frame.columns]
    if missing: raise ValueError(f"Missing required columns {missing}")
    return frame


def _parse_int_list(value, default=(2, 3, 4)) -> List[int]:
    if value is None: return [int(v) for v in default]
    return [int(v) for v in (value.split(",") if isinstance(value, str) else value)]


def _parse_stat_list(value, default=("mean", "max")) -> List[str]:
    if value is None: return [str(v).lower() for v in default]
    return [str(v).strip().lower() for v in (value.split(",") if isinstance(value, str) else value)]


def _visual_path(row: pd.Series) -> str:
    value = row.get("visual_structure_tokens_npz_path", "")
    if value is None or (isinstance(value, float) and np.isnan(value)): return ""
    return str(value).strip()


class _CropLoadingMixin:
    def _init_visual_tokens(self, use_visual_tokens=False, visual_token_levels=None, visual_token_stats=None):
        self.use_visual_tokens = bool(use_visual_tokens)
        self.visual_token_levels = _parse_int_list(visual_token_levels)
        self.visual_token_stats = _parse_stat_list(visual_token_stats)
        self._visual_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if self.use_visual_tokens:
            for row in self.df.itertuples(index=False):
                path = str(getattr(row, "visual_structure_tokens_npz_path", "") or "").strip()
                if not path or not Path(path).is_file():
                    raise FileNotFoundError(f"Missing Stage-1 visual-token artifact for {getattr(row, 'report_id', 'case')}")

    def _load_visual_groups(self, path: str) -> Dict[str, np.ndarray]:
        if path in self._visual_cache: return self._visual_cache[path]
        with np.load(path, allow_pickle=False) as z:
            out = {}
            for group in ("global", "tooth", "relation"):
                chunks = []
                for level in self.visual_token_levels:
                    for stat in self.visual_token_stats:
                        key = f"level{int(level)}_{group}_{stat}"
                        if key not in z: raise KeyError(f"{path}: missing visual-token key {key}")
                        chunks.append(np.asarray(z[key], dtype=np.float32))
                out[group] = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
        self._visual_cache[path] = out
        return out

    def _visual_group_tensor(self, path: str, group: str) -> torch.Tensor:
        return torch.from_numpy(self._load_visual_groups(path)[group]).float()

    def _visual_tooth_tensor(self, path: str, tooth_idx: int) -> torch.Tensor:
        return torch.from_numpy(self._load_visual_groups(path)["tooth"][int(tooth_idx)]).float()

    def _mask_tensor(self, mask, size, target_tooth_id=None):
        return mask_tensor(mask, size, mask_mode=self.mask_mode, target_tooth_id=target_tooth_id,
                           include_prosthetic_channels=getattr(self, "include_prosthetic_channels", False))

    def _load_crop_tensor(self, path, size, target_tooth_id=None):
        ct, mask = load_npz_crop(path)
        ct_np = np.stack([normalize_cbct(ct), metal_window(ct)], axis=0) if getattr(self, "metal_channel", False) else normalize_cbct(ct)[None]
        ct_tensor = resize_3d_tensor(torch.from_numpy(ct_np).float(), size, mode="trilinear")
        return torch.cat([ct_tensor, self._mask_tensor(mask, size, target_tooth_id)], dim=0)

    def _load_global(self, image_path, mask_path, global_tensor_path=None):
        if path_exists(global_tensor_path):
            cached = torch.from_numpy(read_blosc2(global_tensor_path)).float()
            if tuple(cached.shape[-3:]) != tuple(self.global_size):
                cached = resize_3d_tensor(cached, self.global_size, mode="trilinear")
            return cached if self.include_mask_channel else cached[:1]
        image = resize_3d_tensor(torch.from_numpy(normalize_cbct(read_blosc2(image_path))[None]).float(), self.global_size, mode="trilinear")
        if not self.include_mask_channel: return image
        return torch.cat([image, self._mask_tensor(read_blosc2(mask_path), self.global_size)], dim=0)


class ToothFairyStage2ToothInstanceDataset(_CropLoadingMixin, Dataset):
    def __init__(self, csv_paths, tooth_crop_size=(64, 96, 96), include_mask_channel=True,
                 mask_mode="semantic", tooth_region_scheme="ten_region", use_visual_tokens=False,
                 visual_token_levels=None, visual_token_stats=None, include_prosthetic_channels=False,
                 metal_channel=False, **_ignored):
        self.df = _load_csvs(csv_paths)
        self._init_visual_tokens(use_visual_tokens, visual_token_levels, visual_token_stats)
        self.tooth_crop_size = parse_size(tooth_crop_size, (64, 96, 96))
        self.include_mask_channel = bool(include_mask_channel)
        self.include_prosthetic_channels = bool(include_prosthetic_channels)
        self.metal_channel = bool(metal_channel)
        self.mask_mode = str(mask_mode)
        self.tooth_region_scheme = str(tooth_region_scheme)
        self.instances: List[Dict[str, Any]] = []
        for row_idx, row in self.df.iterrows():
            obj = json.loads(Path(row["stage2_label_json"]).read_text())
            mask = np.asarray(obj["tooth_loss_mask"], dtype=np.float32)
            target = np.asarray(obj["tooth_target"], dtype=np.float32)
            order = [str(t) for t in obj.get("tooth_order", FDI_TEETH)]
            paths = obj["tooth_crop_paths"]
            for tooth_idx, tooth_id in enumerate(order):
                base_mask = mask[tooth_idx].astype(np.float32)
                if float(base_mask.sum()) <= 0: continue
                self.instances.append({
                    "row_idx": int(row_idx), "case_id": str(row["case_id"]), "report_id": str(row["report_id"]),
                    "tooth_idx": int(tooth_idx), "tooth_id": tooth_id, "region_id": tooth_region_id(tooth_id, self.tooth_region_scheme),
                    "crop_path": str(paths[tooth_idx]), "visual_token_path": _visual_path(row),
                    "target": target[tooth_idx].astype(np.float32), "base_mask": base_mask,
                })
        self.active_indices = list(range(len(self.instances)))
        self.active_masks = [inst["base_mask"].copy() for inst in self.instances]

    def __len__(self): return len(self.active_indices)

    def __getitem__(self, idx):
        inst_idx = self.active_indices[int(idx)]
        inst = self.instances[inst_idx]
        item = {
            "tooth_crop": self._load_crop_tensor(inst["crop_path"], self.tooth_crop_size, inst["tooth_id"]),
            "tooth_region": torch.tensor(inst["region_id"], dtype=torch.long),
            "tooth_target": torch.from_numpy(inst["target"]).float(),
            "tooth_loss_mask": torch.from_numpy(self.active_masks[int(idx)]).float(),
            "case_id": inst["case_id"], "report_id": inst["report_id"], "tooth_id": inst["tooth_id"],
        }
        if self.use_visual_tokens:
            item["tooth_visual_token"] = self._visual_tooth_tensor(inst["visual_token_path"], inst["tooth_idx"])
        return item


class ToothFairyStage2GlobalDataset(_CropLoadingMixin, Dataset):
    def __init__(self, csv_paths, global_size=(160, 288, 288), include_mask_channel=True, mask_mode="semantic",
                 use_visual_tokens=False, visual_token_levels=None, visual_token_stats=None):
        self.df = _load_csvs(csv_paths)
        self._init_visual_tokens(use_visual_tokens, visual_token_levels, visual_token_stats)
        self.global_size = parse_size(global_size, (160, 288, 288))
        self.include_mask_channel = bool(include_mask_channel)
        self.mask_mode = str(mask_mode)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        obj = json.loads(Path(row["stage2_label_json"]).read_text())
        item = {
            "global_image": self._load_global(row["image_blosc2_path"], row["mask_blosc2_path"], row.get("global_tensor_path", None)),
            "global_target": torch.tensor(obj["global_target"], dtype=torch.float32),
            "global_loss_mask": torch.tensor(obj["global_loss_mask"], dtype=torch.float32),
            "case_id": str(row["case_id"]), "report_id": str(row["report_id"]),
        }
        if self.use_visual_tokens: item["global_visual_tokens"] = self._visual_group_tensor(_visual_path(row), "global")
        return item


class ToothFairyStage2RelationDataset(_CropLoadingMixin, Dataset):
    def __init__(self, csv_paths, relation_crop_size=(80, 128, 128), include_mask_channel=True, mask_mode="semantic",
                 use_visual_tokens=False, visual_token_levels=None, visual_token_stats=None):
        self.df = _load_csvs(csv_paths)
        self._init_visual_tokens(use_visual_tokens, visual_token_levels, visual_token_stats)
        self.relation_crop_size = parse_size(relation_crop_size, (80, 128, 128))
        self.include_mask_channel = bool(include_mask_channel)
        self.mask_mode = str(mask_mode)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        obj = json.loads(Path(row["stage2_label_json"]).read_text())
        crops = []
        for i, path in enumerate(obj["relation_crop_paths"]):
            name = str(obj.get("relation_crops", [])[i])
            target = "lower_left_teeth" if name.startswith("left_lower_jawbone") else "lower_right_teeth" if name.startswith("right_lower_jawbone") else None
            crops.append(self._load_crop_tensor(path, self.relation_crop_size, target))
        item = {
            "relation_crops": torch.stack(crops, dim=0),
            "relation_target": torch.tensor(obj["relation_target"], dtype=torch.float32),
            "relation_loss_mask": torch.tensor(obj["relation_loss_mask"], dtype=torch.float32),
            "case_id": str(row["case_id"]), "report_id": str(row["report_id"]),
        }
        if self.use_visual_tokens: item["relation_visual_tokens"] = self._visual_group_tensor(_visual_path(row), "relation")
        return item
