import json
import os
from pathlib import Path
import re
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import DataLoader, Dataset


class TeethSegDataModule(pl.LightningDataModule):
    """Implements data module that loads meshes of the 3DTeethSeg challenge."""

    def __init__(
        self,
        root: Union[str, Path],
        regex_filter: str,
        extensions: str,
        clean: bool,
        val_size: float,
        include_val_as_train: bool,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
        **kwargs,
    ):        
        super().__init__()
        
        self.root = Path(root)
        self.filter = f'/(?!{os.sep}\.)[^{os.sep}\.]*({regex_filter})'
        self.extensions = extensions
        self.clean = clean
        self.val_size = val_size
        self.include_val_as_train = include_val_as_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed =  seed

    def _files(
        self,
        stage: str,
        exclude: List[str]=[
            # braces
            '017U3R3T_upper',
            '01A91JH6_lower',
            '01A91JH6_upper',
            '01HMF5HV_lower',
            '01HMF5HV_upper',
            'E23G704K_lower',
            'E23G704K_upper',
            # chaos
            '016FSM14_lower',
            '01FPTYH2_lower',
        ],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        mesh_files = []
        for ext in self.extensions:
            mesh_files.extend(sorted(self.root.glob(f'**/*.{ext}')))
        scan_ids = [f.stem for f in mesh_files]
        _, unique_idxs = np.unique(scan_ids, return_index=True)
        mesh_files = [mesh_files[i] for i in unique_idxs]
        mesh_files = [f for f in mesh_files if re.search(self.filter, str(f.relative_to(self.root.parent)))]
        mesh_files = [f for f in mesh_files if f.stem not in exclude]
        mesh_files = [f.relative_to(self.root) for f in mesh_files]

        ann_files = sorted(self.root.glob('**/*.json'))
        if stage == 'predict' or not ann_files or (
            hasattr(self, 'batch') and self.batch is not None
        ):
            return mesh_files

        scan_ids = [f.stem for f in ann_files]
        _, unique_idxs = np.unique(scan_ids, return_index=True)
        ann_files = [ann_files[i] for i in unique_idxs]
        ann_files = [f for f in ann_files if re.search(self.filter, str(f.relative_to(self.root.parent)))]
        ann_files = [f for f in ann_files if f.stem not in exclude]
        ann_files = [f.relative_to(self.root) for f in ann_files]
        
        return list(zip(mesh_files, ann_files))

    def _split(
        self,
        files: List[Tuple[Path, Path, '...']],
    ) -> Tuple[List[Tuple[Path, Path, '...']]]:
        if len(files) <= 1:
            return files, []

        # determine classes and mesh and annotation files of each subject
        subject_idxs = {}
        subject_files = [[] for _ in files]
        subject_labels = np.zeros((len(files), 86))
        for case_files in files:
            with open(self.root / case_files[1], 'rb') as f:
                annotation = json.load(f)

            subject = re.split('_|-', case_files[0].stem)[0]
            labels = np.unique(annotation['labels'])

            subject_idx = subject_idxs.setdefault(subject, len(subject_idxs))
            subject_files[subject_idx].append(case_files)
            subject_labels[subject_idx, labels] = 1

        # get available files and classes
        subjects = len(subject_idxs)
        subject_files = subject_files[:subjects][::-1]
        subject_labels = subject_labels[:subjects][::-1]

        # split subjects with stratification on multi-labels
        stratifier = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[self.val_size, 1 - self.val_size],
        )
        train_idxs, val_idxs = next(stratifier.split(
            X=subject_files,
            y=subject_labels,
        ))
        train_files = [f for i in train_idxs for f in subject_files[i]]
        val_files = [f for i in val_idxs for f in subject_files[i]]

        if self.include_val_as_train:
            train_files += val_files

        return train_files, val_files

    def _dataloader(
        self,
        dataset: Dataset,
        shuffle: bool=False,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.pred_dataset)
