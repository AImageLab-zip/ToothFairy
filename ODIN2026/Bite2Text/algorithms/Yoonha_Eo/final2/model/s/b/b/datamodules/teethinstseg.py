from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from pytorch_lightning.trainer.states import RunningStage
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethseg import TeethSegDataModule
from teethland.data.datasets import TeethSegDataset
import teethland.data.transforms as T
from teethland.visualization import draw_point_clouds


class TeethInstSegDataModule(TeethSegDataModule):
    """Data module to load intraoral scans with teeth instances."""

    def __init__(
        self,
        batch: Optional[Tuple[int, int]],
        uniform_density_voxel_size: int,
        distinguish_upper_lower: bool,
        boundary_aware: Dict[str, Union[bool, float]],
        **dm_cfg,
    ):
        super().__init__(**dm_cfg)

        self.default_transforms = T.Compose(
            T.UniformDensityDownsample(uniform_density_voxel_size[0]),
            T.XYZAsFeatures(),
            T.NormalAsFeatures(),
            T.ToTensor(),
        )
        
        self.batch = None if batch is None else slice(*batch)
        self.use_boundary_aware = boundary_aware.pop('use')
        self.boundary_aware_cfg = boundary_aware
        self.distinguish_upper_lower = distinguish_upper_lower
        self.is_lower = None

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)
        self.default_transforms = T.Compose(
            (
                T.BoundaryAwareDownsample(**self.boundary_aware_cfg, rng=rng)
                if self.use_boundary_aware else dict
            ),
            self.default_transforms,
        )

        if stage is None or stage == 'fit':
            files = self._files('fit')
            print('Total number of files:', len(files))
            train_files, val_files = self._split(files)

            train_transforms = T.Compose(
                T.RandomAxisFlip(rng=rng),
                T.RandomScale(rng=rng),
                T.RandomZAxisRotate(rng=rng),
                self.default_transforms,
            )

            self.train_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                clean=self.clean,
                transform=train_transforms,
            )
            self.val_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                clean=self.clean,
                transform=self.default_transforms,
            )

        if stage is None or stage == 'predict':
            files = self._files('predict', exclude=[])
            self.pred_dataset = TeethSegDataset(
                stage='predict',
                root=self.root,
                files=files if self.batch is None else files[self.batch],
                clean=self.clean,
                transform=self.default_transforms,
            )

    @property
    def num_channels(self) -> int:
        return 6

    @property
    def num_classes(self) -> int:
        return 7 if self.filter or not self.distinguish_upper_lower else 14

    def teeth_labels_to_classes(
        self,
        labels: Union[
            NDArray[np.int64],
            TensorType['N', torch.int64],
        ]
    ) -> Union[
        NDArray[np.int64],
        TensorType['N', torch.int64],
    ]:
        if isinstance(labels, np.ndarray):
            classes = labels.copy()
        elif isinstance(labels, torch.Tensor):
            classes = labels.clone()
        else:
            raise ValueError(
                f'Expected np.ndarray or torch.Tensor, got {type(labels)}.',
            )

        classes[(11 <= labels) & (labels <= 17)] -= 11
        classes[labels == 18] = 6
        classes[(21 <= labels) & (labels <= 27)] -= 21
        classes[labels == 28] = 6

        if self.filter == 'lower' or not self.distinguish_upper_lower:
            classes[(31 <= labels) & (labels <= 37)] -= 31
            classes[labels == 38] = 6
            classes[(41 <= labels) & (labels <= 47)] -= 41
            classes[labels == 48] = 6
        else:
            classes[(31 <= labels) & (labels <= 37)] -= 24
            classes[labels == 38] = 13
            classes[(41 <= labels) & (labels <= 47)] -= 34
            classes[labels == 48] = 13
        
        return classes
    
    def determine_rot_matrix(
        self,
        centroids: PointTensor,
        verbose: bool=False,
    ):
        X, Y, _ = centroids.C.T.cpu().numpy()
        _, _, degrees = cv2.fitEllipse(np.column_stack((X, Y)))
        if degrees > 90:
            degrees = degrees - 180
        angle = -degrees / 180 * np.pi
        cosval, sinval = np.cos(angle), np.sin(angle)

        R = np.array([
            [cosval, -sinval, 0, 0],
            [sinval, cosval,  0, 0],
            [0,       0,      1, 0],
            [0,       0,      0, 1],
        ])
        R = torch.from_numpy(R).to(centroids.C)        

        if verbose:
            import matplotlib.pyplot as plt
            new_coords = centroids.C @ R.T
            new_coords = new_coords.cpu().numpy()
            plt.scatter(X, Y, label='Original')
            plt.scatter(new_coords[:, 0], new_coords[:, 1], label='Rotated')
            # plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
            # plt.scatter(xs, ys)
            plt.legend()
            plt.show(block=True)


        return R

    def reorder_teeth(
        self,
        classes: PointTensor,
        side_mask: TensorType['N', torch.bool],
        dist_thresh_nonmolar: float=0.35,
        dist_thresh_molar: float=0.5,
    ) -> PointTensor:
        # fix teeth labels iteratively on one side
        side_idxs = torch.nonzero(side_mask)[:, 0]
        side_classes = classes[side_idxs]
        sort_cols = torch.column_stack((side_classes.F, side_classes.C[:, 1]))
        _, argsort1 = torch.sort(sort_cols[:, 1], stable=True)
        _, argsort0 = torch.sort(sort_cols[argsort1, 0], stable=True)
        argsort = argsort1[argsort0]
        batch_idxs = side_classes.batch_indices[argsort]
        coords = side_classes.C[argsort]
        for i in range(side_idxs.shape[0] - 1):
            if batch_idxs[i] != batch_idxs[i + 1]:
                continue

            i_ = side_idxs[argsort[i]]
            j_ = side_idxs[argsort[i + 1]]
            if classes.F[i_] < 2:
                continue
            
            dist_thresh = dist_thresh_nonmolar if classes.F[j_] < 5 else dist_thresh_molar
            dist = torch.sqrt(torch.sum((coords[i, :2] - coords[i + 1, :2])**2))
            dist_offset = classes.F[j_] - classes.F[i_]
            if dist / dist_thresh < 1 + dist_offset:
                continue
            
            classes.F[j_] = classes.F[i_] + (dist / dist_thresh).long()

        # add third molar if three molars are present on one side
        if (classes.F[side_idxs] >= 5).sum() == 3:
            m3_idx = classes.C[classes.F >= 5, 1].argmax()
            classes.F[torch.nonzero(classes.F >= 5)[m3_idx, 0]] = 7        

        return classes

    def teeth_classes_to_labels(
        self,
        classes: PointTensor,
    ) -> PointTensor:

        # determine tooth instances on the right side of the arch
        right_mask = torch.zeros(0, dtype=torch.bool, device=classes.C.device)
        for batch_idx in range(classes.batch_size):
            is_incisor = (classes.batch_indices == batch_idx) & (classes.F <= 1)
            is_incisor = torch.nonzero(is_incisor)[:, 0]
            is_incisor = is_incisor[torch.argsort(classes.C[is_incisor, 1])[:4]]

            if is_incisor.shape[0] <= 2:
                zero_x = 0.0
            else:
                weights = 1 - classes.F[is_incisor] / 2
                zero_x = torch.sum(weights * classes.C[is_incisor, 0]) / torch.sum(weights)

            right = classes.batch(batch_idx).C[:, 0] < zero_x
            right_mask = torch.cat((right_mask, right))

        # make room for third molars
        labels = classes.clone()
        labels.F[labels.F >= 7] += 1

        # fix duplicate classes on one side, introducing third molars
        labels = self.reorder_teeth(labels, right_mask)
        labels = self.reorder_teeth(labels, ~right_mask)

        # translate index label to FDI label
        if self.filter == 'lower':
            labels.F += 31 + 10 * right_mask
        else:
            if self.distinguish_upper_lower:
                labels.F += 12 * (labels.F >= 8)
            else:
                labels.F = torch.clip(labels.F, 0, 7)
                labels.F += 20 * self.is_lower[classes.batch_indices]
                
            labels.F += 11 + 10 * right_mask
        
        return labels

    def collate_downsample(
        self,
        point_counts: List[TensorType[torch.int64]],
        downsample_idxs: List[TensorType['m', torch.int64]],
        downsample_counts: List[TensorType[torch.int64]],
    ) -> TensorType['M', torch.int64]:
        point_counts = torch.stack(point_counts)
        downsample_idxs = torch.cat(downsample_idxs)
        downsample_counts = torch.stack(downsample_counts)
        batch_offsets = point_counts.cumsum(dim=-1) - point_counts
        batch_offsets = batch_offsets.repeat_interleave(downsample_counts)
        downsample_idxs += batch_offsets
        
        return downsample_idxs

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', torch.bool],
        Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        Union[
            PointTensor,
            Tuple[PointTensor, PointTensor],
        ],        
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        is_lower = torch.stack(batch_dict['is_lower'])

        # collate input points and features
        point_counts = torch.stack(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            features=torch.cat(batch_dict['features']),
            batch_counts=point_counts,
        )

        x.cache['cp_downsample_idxs'] = self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs'],
            batch_dict['ud_downsample_count'],
        )
        x.cache['ts_downsample_idxs'] = self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ba_downsample_idxs'],
            batch_dict['ba_downsample_count'],
        ) if self.use_boundary_aware else x.cache['cp_downsample_idxs']

        # collate output points
        points = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            batch_counts=point_counts,
        )
        if self.trainer.state.stage == RunningStage.PREDICTING:
            return scan_file, is_lower, x, points

        # collate tooth instance centroids and classes
        instance_centroids = [ic[1:] for ic in batch_dict['instance_centroids']]
        instance_labels = [il[1:] for il in batch_dict['instance_labels']]
        instance_counts = torch.stack(batch_dict['instance_count']) - 1
        instances = PointTensor(
            coordinates=torch.cat(instance_centroids),
            features=self.teeth_labels_to_classes(torch.cat(instance_labels)),
            batch_counts=instance_counts,
        )
        
        # determine gingiva (-1) or tooth instance index (>= 0) for each point
        points.F = torch.cat(batch_dict['instances']) - 1
        instance_offsets = instance_counts.cumsum(dim=0) - instance_counts
        instance_offsets = instance_offsets.repeat_interleave(point_counts)
        points.F[points.F >= 0] += instance_offsets[points.F >= 0]

        # take subsample and remove instances not present in subsample
        points = points[x.cache['ts_downsample_idxs']]
        unique, inverse_idxs = torch.unique(points.F, return_inverse=True)

        instances = instances[unique[unique >= 0]]
        points.F = inverse_idxs - 1
        
        return scan_file, is_lower, x, (instances, points)

    def _transfer_fit_batch_to_device(
        self,
        x: Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        y: Tuple[PointTensor, PointTensor],
        device: torch.device,
    ) -> Tuple[PointTensor, Tuple[PointTensor, PointTensor]]:
        instances, points = y

        return x.to(device), (instances.to(device), points.to(device))

    def _transfer_predict_batch_to_device(
        self,
        x: Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        y: PointTensor,
        device: torch.device,
    ) -> Tuple[PointTensor, PointTensor]:
        return x.to(device), y.to(device)

    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Union[
        Tuple[PointTensor, PointTensor],
        Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
    ]:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)

        x, y = batch[2:]

        if isinstance(y, PointTensor):
            return self._transfer_predict_batch_to_device(x, y, device)
        else:
            return self._transfer_fit_batch_to_device(x, y, device)
