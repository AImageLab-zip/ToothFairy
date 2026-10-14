from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethseg import TeethSegDataModule
from teethland.data.datasets import TeethLandDataset
import teethland.data.transforms as T


class TeethLandDataModule(TeethSegDataModule):
    """Implements data module that loads meshes and landmarks of the 3DTeethLand challenge."""

    def __init__(
        self,
        landmarks_root: Path,
        uniform_density_voxel_size: int,
        proposal_points: int,
        max_proposals: int,
        include_cusps: bool,
        to_left_right: bool,
        separate_front_posterior: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.default_transforms = T.Compose(
            T.XYZAsFeatures(),
            T.NormalAsFeatures(),
            T.CentroidOffsetsAsFeatures(),
            T.ToTensor(),
        )

        self.landmarks_root = Path(landmarks_root)
        self.uniform_density_voxel_size = uniform_density_voxel_size[1]
        self.proposal_points = proposal_points
        self.max_proposals = max_proposals
        self.include_cusps = include_cusps
        self.to_left_right = to_left_right
        self.separate_front_posterior = separate_front_posterior

    def _files(
        self,
        stage: str,
        exclude: List[str]=[
            # missing tooth segmentation
            'K4BAII5F_upper',
            '87N5YSES_upper',
            '67PV9M7X_lower',
            # missing more than one landmark
            'S0AON6PZ_lower',
        ],
    ):
        seg_files = super()._files(stage, exclude=exclude)
        seg_stems = set([fs[0].stem for fs in seg_files])
        
        landmark_files = sorted(self.landmarks_root.glob('**/*.json'))
        landmark_files = [f.relative_to(self.landmarks_root) for f in landmark_files]
        landmark_stems = set([f.name.split('__')[0] for f in landmark_files])

        ann_stems = seg_stems & landmark_stems
        files = []
        for stem in sorted(ann_stems):
            seg_fs = [fs for fs in seg_files if fs[0].stem == stem][0]
            landmark_f = [f for f in landmark_files if f.name.split('__')[0] == stem][0]
            files.append((*seg_fs, landmark_f))

        return files    

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)
        self.default_transforms = T.Compose(
            T.UniformDensityDownsample(self.uniform_density_voxel_size, inplace=True),
            T.GenerateProposals(self.proposal_points, self.max_proposals, rng=rng),
            self.default_transforms,
        )

        if stage is None or stage == 'fit':
            files = self._files('fit')
            print('Total number of files:', len(files))
            train_files, val_files = self._split(files)

            landmark_flip_idxs = np.arange(10)
            m, d = [TeethLandDataset.landmark_classes[k] for k in ['Mesial', 'Distal']]
            if self.to_left_right:
                landmark_flip_idxs[[m, d]] = landmark_flip_idxs[[d, m]]
            if self.separate_front_posterior:
                landmark_flip_idxs[[5, 6]] = 6, 5
                                      
            train_transforms = T.Compose(
                T.RandomAxisFlip(rng=rng, landmark_flip_idxs=landmark_flip_idxs),
                T.RandomScale(rng=rng),
                T.RandomZAxisRotate(rng=rng),
                self.default_transforms,
            )

            self.train_dataset = TeethLandDataset(
                stage='fit',
                seg_root=self.root,
                landmarks_root=self.landmarks_root,
                include_cusps=self.include_cusps,
                to_left_right=self.to_left_right,
                separate_front_posterior=self.separate_front_posterior,
                files=train_files,
                clean=self.clean,
                transform=train_transforms,
            )
            self.val_dataset = TeethLandDataset(
                stage='fit',
                seg_root=self.root,
                landmarks_root=self.landmarks_root,
                include_cusps=self.include_cusps,
                to_left_right=self.to_left_right,
                separate_front_posterior=self.separate_front_posterior,
                files=val_files,
                clean=self.clean,
                transform=self.default_transforms,
            )
    
    @property
    def num_channels(self) -> int:
        return 9
    
    @property
    def num_classes(self) -> int:
        return 5 + 2 * self.separate_front_posterior + self.include_cusps

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        Path,
        TensorType['B', torch.bool],
        PointTensor,
        Tuple[PointTensor, PointTensor, PointTensor],       
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        is_lower = torch.stack(batch_dict['is_lower'])

        # collate input points and features
        point_counts = torch.cat(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']).reshape(-1, 3),
            features=torch.cat(batch_dict['features']).reshape(-1, self.num_channels),
            batch_counts=point_counts,
        )

        # collate tooth instance centroids and classes
        instance_counts = torch.stack(batch_dict['instance_count'])
        if 'instance_landmarks' in batch_dict:
            instances = PointTensor(
                coordinates=torch.zeros(instance_counts.sum(), 3),
                features=torch.cat(batch_dict['instance_landmarks']),
                batch_counts=instance_counts,
            )
        else:
            instances = None
        
        if 'landmarks' in batch_dict:
            landmark_counts = torch.cat([
                torch.bincount(lands[:, 4].long(), minlength=points.shape[0])
                for lands, points in zip(batch_dict['landmarks'], batch_dict['points'])
            ])
            landmarks = PointTensor(
                coordinates=torch.cat(batch_dict['landmarks'])[:, :3],
                features=torch.cat(batch_dict['landmarks'])[:, 3].long(),
                batch_counts=landmark_counts,
            )
        else:
            landmarks = None

        points = x.new_tensor(features=torch.cat(batch_dict['labels']).flatten() - 1)
        instance_offsets = torch.arange(instance_counts.sum())
        instance_offsets = instance_offsets.repeat_interleave(point_counts)
        points.F[points.F >= 0] += instance_offsets[points.F >= 0]

        return scan_file, is_lower, x, (instances, landmarks, points)
    
    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Tuple[PointTensor, Tuple[PointTensor, PointTensor, PointTensor]]:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)

        x, (instances, landmarks, points) = batch[2:]
        x = x.to(device)
        instances = instances.to(device) if instances else instances
        landmarks = landmarks.to(device) if landmarks else landmarks
        points = points.to(device)

        return x, (instances, landmarks, points)
