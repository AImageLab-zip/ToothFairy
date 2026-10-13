"""Pointcept dataset for pre-sampled Bite2Text IOS pairs."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class Bite2TextDataset(Dataset):
    """Load paired-jaw point samples and an arbitrary number of report heads."""

    def __init__(
        self,
        split: str = "train",
        data_root: str | os.PathLike = "data/bite2text_ptv3",
        transform: Sequence[dict] | None = None,
        save_record: bool = True,
        test_mode: bool = False,
        test_cfg=None,
        loop: int = 1,
        label_csv: str | os.PathLike | None = None,
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.sample_dir = self.data_root / self.split
        self.labels_df = pd.read_csv(label_csv or self.data_root / "labels.csv", dtype=str)
        self.labels_df = self.labels_df.set_index("patient_id", drop=False)
        self.label_columns = sorted(
            (column for column in self.labels_df.columns if column.startswith("label_")),
            key=lambda column: int(column.split("_")[1]),
        )
        if not self.label_columns:
            raise ValueError("labels.csv contains no label_<index> columns")
        self.test_mode = test_mode
        self.transform = transform if callable(transform) else Compose(transform)
        self.loop = loop if not test_mode else 1
        self.data_list = sorted(path.stem for path in self.sample_dir.glob("dental_*.npz"))
        if max_samples:
            self.data_list = self.data_list[:max_samples]
        if not self.data_list:
            raise FileNotFoundError(f"No dental_*.npz samples found in {self.sample_dir}")

        logger = get_root_logger()
        logger.info(f"Totally {len(self.data_list)} x {self.loop} Bite2Text samples in {self.split} set.")
        record_suffix = f"_{max_samples}" if max_samples else ""
        record_path = self.data_root / f"bite2text_{self.split}{record_suffix}.pth"
        if record_path.is_file():
            logger.info(f"Loading record: {record_path.name} ...")
            self.data_cache = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_path.name} ...")
            self.data_cache = {}
            for index, data_name in enumerate(self.data_list):
                logger.info(f"Parsing [{index + 1}/{len(self.data_list)}] {data_name}")
                self.data_cache[data_name] = self._load_sample(data_name)
            if save_record:
                torch.save(self.data_cache, record_path)

    def _load_sample(self, data_name: str) -> dict:
        patient_id = data_name[len("dental_") :]
        if patient_id not in self.labels_df.index:
            raise KeyError(f"Patient {patient_id} not found in labels.csv")
        with np.load(self.sample_dir / f"{data_name}.npz", allow_pickle=False) as payload:
            coord = np.asarray(payload["coord"], dtype=np.float32)
        if coord.ndim != 2 or coord.shape[1] != 3 or not np.isfinite(coord).all():
            raise ValueError(f"Invalid coordinates for {patient_id}: {coord.shape}")
        row = self.labels_df.loc[patient_id]
        output = {"name": data_name, "coord": coord}
        for index, column in enumerate(self.label_columns):
            value = int(row[column])
            output[f"label_{index}"] = np.asarray([value if value >= 0 else -1], dtype=np.int64)
        return output

    def __len__(self) -> int:
        return len(self.data_list) * self.loop

    def get_data_name(self, index: int) -> str:
        return self.data_list[index % len(self.data_list)]

    def __getitem__(self, index: int) -> dict:
        item = copy.deepcopy(self.data_cache[self.get_data_name(index)])
        # Bits2Bites mesh pretraining used a 9-D input: xyz plus six landmark
        # one-hot channels. Bite2Text has no landmarks, so the six channels are
        # intentionally zero while retaining exact encoder compatibility.
        item["point_label_onehot"] = np.zeros((len(item["coord"]), 6), dtype=np.float32)
        item["index_valid_keys"] = ["coord", "point_label_onehot"]
        return self.transform(item)
