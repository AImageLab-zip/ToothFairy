"""
Dental Ferrara Dataset
Author: Lorenzo Borghi (lorenzobrg@pm.me)
"""

import os
import json
import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import pointops
from torch.utils.data import Dataset
from copy import deepcopy

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


# CONSTANTS

POINT_CLASSES = [
    "Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint"
]
POINT_CLASS_TO_IDX = {c: i for i, c in enumerate(POINT_CLASSES)}
FDI_TEETH = [*range(11, 19), *range(21, 29), *range(31, 39), *range(41, 49)]
FDI_TO_IDX = {fdi: i for i, fdi in enumerate(FDI_TEETH)}

RIGHT_LEFT_CLASSES = ["prima classe", "seconda classe", "terza classe"]
ANTERIOR_CLASSES   = ["normale", "profondo", "aperto", "inverso"]
TRANSVERSE_CLASSES = ["normale", "scissor", "cross"]
MIDLINE_CLASSES    = ["centrata", "deviata"]

STRING2IDX = {
    "right":      {c: i for i, c in enumerate(RIGHT_LEFT_CLASSES)},
    "left":       {c: i for i, c in enumerate(RIGHT_LEFT_CLASSES)},
    "anterior":   {c: i for i, c in enumerate(ANTERIOR_CLASSES)},
    "transverse": {c: i for i, c in enumerate(TRANSVERSE_CLASSES)},
    "midline":    {c: i for i, c in enumerate(MIDLINE_CLASSES)},
}

def _map_label(value: str, category: str) -> int:
    try:
        return STRING2IDX[category][value]
    except KeyError:
        print("DataLoader: found invalid label to ignore")
        return -1

@DATASETS.register_module()
class DentalDataset(Dataset):
    r""".
    
    Folder layout
    -------------
    data_root/
        ├── train/
        │     ├── dental_123.json
        │     └── ...
        ├── test/
        │     └── ...
        └── labels.csv            ← six clinical columns

    The CSV columns are:
        id, right, left, anterior, transverse, midline
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str | os.PathLike = "data/dental",
        class_names: Sequence[str] | None = None,      # kept only for compatibility
        transform: Sequence[dict] | None = None,
        num_points: int | None = 8192,
        uniform_sampling: bool = True,
        save_record: bool = True,
        test_mode: bool = False,
        test_cfg=None,
        loop: int = 1,
        label_csv: str | os.PathLike | None = None,
        debug=False
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.json_dir = self.data_root / self.split
        if label_csv is None:
            label_csv = self.data_root / "labels.csv"
        self.labels_df = pd.read_csv(label_csv, dtype=str)
        self.debug = debug

        self.num_points = num_points
        self.uniform_sampling = uniform_sampling
        if not callable(transform):
            self.transform = Compose(transform)
        else:
            self.transform = transform
        self.loop = loop if not test_mode else 1

        self.test_mode = test_mode
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]

        self.data_list = self._build_data_list()
        logger = get_root_logger()
        logger.info(f"Totally {len(self.data_list)} x {self.loop} samples in {split} set.")

        record_name = f"dental_{self.split}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = self.data_root / f"{record_name}.pth"

        if record_path.is_file():
            logger.info(f"Loading record: {record_name} …")
            self.data_cache = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} …")
            self.data_cache = {}
            for idx, data_name in enumerate(self.data_list):
                logger.info(f"Parsing [{idx + 1}/{len(self.data_list)}] {data_name}")
                self.data_cache[data_name] = self._load_sample(idx)
            if save_record:
                torch.save(self.data_cache, record_path)

    def _build_data_list(self) -> list[str]:
        """Return base filenames (w/out .json) for the chosen split."""
        assert self.json_dir.is_dir(), f"{self.json_dir} not found"
        return sorted(p.stem for p in self.json_dir.glob("dental_*.json"))

    def _load_sample(self, idx: int) -> dict:
        """Load one sample from disk, apply FPS if needed, build data_dict."""
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]                 # e.g. dental_0123
        patient_id = data_name.split("_")[1]                 # '0123'
        patient_id = int(patient_id)
        patient_id = str(patient_id)

        # geometry (coords + one-hot)
        json_path = self.json_dir / f"{data_name}.json"
        with open(json_path, "r") as fp:
            obj = json.load(fp)

        coords = np.asarray([o["coord"] for o in obj["objects"]], dtype=np.float32)

        one_hot_list = []
        for o in obj["objects"]:
            if o["class"] == "Mesh":
                one_hot = np.zeros(len(POINT_CLASSES), dtype=np.float32)
            else:
                idx = POINT_CLASS_TO_IDX[o["class"]]
                one_hot = np.zeros(len(POINT_CLASSES), dtype=np.float32)
                one_hot[idx] = 1.0
            one_hot_list.append(one_hot)

        one_hot = np.stack(one_hot_list, axis=0)  # (N, 6)

        anatomy_list = []
        for o, landmark_one_hot in zip(obj["objects"], one_hot):
            tooth_one_hot = np.zeros(len(FDI_TEETH), dtype=np.float32)
            fdi = int(o.get("fdi", 0) or 0)
            if fdi in FDI_TO_IDX:
                tooth_one_hot[FDI_TO_IDX[fdi]] = 1.0
            arch_one_hot = np.zeros(2, dtype=np.float32)
            if o.get("arch") == "upper":
                arch_one_hot[0] = 1.0
            elif o.get("arch") == "lower":
                arch_one_hot[1] = 1.0
            confidence = np.array([float(o.get("confidence", 0.0))], dtype=np.float32)
            modality = np.array([float(o.get("modality", o["class"] == "Mesh"))], dtype=np.float32)
            anatomy_list.append(
                np.concatenate([landmark_one_hot, tooth_one_hot, arch_one_hot, confidence, modality])
            )
        anatomy_feat = np.stack(anatomy_list, axis=0)  # (N, 42)
        # labels from CSV 
        row = self.labels_df[self.labels_df.iloc[:, 0] == patient_id]
        if row.empty:
            raise KeyError(f"Patient {patient_id} not found in labels.csv")
        row = row.iloc[0]

        lbl_right  = _map_label(row[1], "right")
        lbl_left   = _map_label(row[2], "left")
        lbl_ant    = _map_label(row[3], "anterior")
        lbl_trans  = _map_label(row[4], "transverse")
        lbl_mid    = _map_label(row[5], "midline")

        # create dict

        data_dict = {
            "coord": coords,                        # (N,3) float32
            "point_label_onehot": one_hot,          # (N,6) float32
            "anatomy_feat": anatomy_feat,           # (N,42) float32
            "preserve_landmarks": bool(np.any(anatomy_feat[:, -1] > 0.5)),

            # Five clinical targets, each shape (1,) int64
            "label_0":      np.array([lbl_right],  dtype=np.int64),
            "label_1":       np.array([lbl_left],   dtype=np.int64),
            "label_2":   np.array([lbl_ant],    dtype=np.int64),
            "label_3": np.array([lbl_trans],  dtype=np.int64),
            "label_4":    np.array([lbl_mid],    dtype=np.int64),
        }
        return data_dict

    def __len__(self) -> int:
        return len(self.data_list) * self.loop

    def get_data_name(self, idx: int) -> str:
        return self.data_list[idx % len(self.data_list)]

    def __getitem__(self, idx: int):
        if self.test_mode:
            return self._prepare_test_item(idx)
        return self._prepare_train_item(idx)

    def _prepare_train_item(self, idx: int) -> dict:
        item = copy.deepcopy(self.data_cache[self.get_data_name(idx)])
        return self.transform(item)

    '''
    def _prepare_test_item(self, idx: int) -> dict:
        assert idx < len(self.data_list)
        base = copy.deepcopy(self.data_cache[self.get_data_name(idx)])
        category = base.pop("category")

        base = self.transform(base)
        augged = [aug(deepcopy(base)) for aug in self.aug_transform]
        augged = [self.post_transform(x) for x in augged]

        return dict(
            voting_list=augged,
            category=category,
            name=self.get_data_name(idx),
        )
    '''
    
    def _prepare_test_item(self, idx: int) -> dict:
        item = copy.deepcopy(self.data_cache[self.get_data_name(idx)])
        item = self.transform(item)
        if "coord" not in item or item["coord"].shape[0] == 0:
            raise ValueError(f"Empty coord array for sample {self.get_data_name(idx)}")

        num_points = int(item["coord"].shape[0])
        item["offset"] = torch.tensor([0, num_points], dtype=torch.long)
        return item
