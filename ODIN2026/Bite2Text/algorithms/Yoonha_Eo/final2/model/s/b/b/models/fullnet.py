import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch_scatter import scatter_mean
from torchtyping import TensorType

from sklearn.decomposition import PCA
import teethland
from teethland import PointTensor
import teethland.data.transforms as T
from teethland.cluster import learned_region_cluster
import teethland.nn as nn


class FullNet(pl.LightningModule):

    def __init__(
        self,
        instseg: Dict[str, Any],
        landmarks: Dict[str, Any],
        proposal_points: int,
        dbscan_cfg: dict[str, Any],
        out_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # instance segmentation stage
        ckpt = instseg.pop('checkpoint_path')
        self.instances_model = nn.StratifiedTransformer(
            in_channels=kwargs['in_channels'],
            out_channels=[6, 1, None],
            **instseg,
        )
        self.load_ckpt(self.instances_model, ckpt)

        self.identify_model = nn.MaskedAveragePooling(
            num_features=self.instances_model.out_channels[-1],
            out_channels=kwargs['num_classes'],
        )
        self.load_ckpt(self.identify_model, ckpt)
        
        # landmark prediction stage
        ckpt = landmarks.pop('checkpoint_path')
        self.landmarks_model = nn.StratifiedTransformer(
            in_channels=9,
            out_channels=[1, 4, 4, 4, 4, 4],
            **landmarks,
        )
        self.load_ckpt(self.landmarks_model, ckpt)

        self.gen_proposals = T.GenerateProposals(proposal_points, max_proposals=32)
        self.dbscan_cfg = dbscan_cfg
        self.only_dentalnet = out_dir.name == 'dentalnetPr'
        self.out_dir = out_dir
        from teethland.data.datasets import TeethLandDataset
        self.landmark_class2label = {
            v: k for k, v in TeethLandDataset.landmark_classes.items()
        }

    def load_ckpt(self, model, ckpt: str):
        state_dict = model.state_dict()
        if ckpt:
            ckpt = torch.load(ckpt, weights_only=False)['state_dict']
            ckpt = {k.split('.', 1)[1]: v for k, v in ckpt.items()}
            ckpt = {k: v for k, v in ckpt.items() if k in state_dict}
            model.load_state_dict(ckpt)
            model.requires_grad_(False)

    def instances_stage(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, PointTensor]:
        # downsample
        x_down = x[x.cache['instseg_downsample_idxs']]
        
        # forward pass
        _, (spatial_embeds, seeds, _) = self.instances_model(x_down)

        # cluster
        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])
        clusters = learned_region_cluster(
            offsets, sigmas, seeds,
        )

        # determine instance centroids
        centroids = scatter_mean(
            src=clusters.C[clusters.F >= 0],
            index=clusters.F[clusters.F >= 0],
            dim=0,
        )
        centroids = PointTensor(coordinates=centroids)
        
        # rotate intra-oral scan
        affine = self.trainer.datamodule.determine_rot_matrix(centroids)
        coords_rot = x.C @ affine[:3, :3].T
        normals_rot = x.F[:, 3:] @ affine[:3, :3].T
        x_rot = x.new_tensor(
            coordinates=coords_rot,
            features=torch.column_stack((coords_rot, normals_rot)),
        )
        x_rot.cache = x.cache
        x_rot_down = x_rot[x.cache['instseg_downsample_idxs']]

        # second pass
        _, (spatial_embeds, seeds, features) = self.instances_model(x_rot_down)

        # cluster
        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])
        clusters = learned_region_cluster(
            offsets, sigmas, seeds,
        )

        # determine FDI number on rotated scan
        _, classes = self.identify_model(features, clusters)
        classes = classes.new_tensor(features=classes.F.argmax(-1))
        labels = self.trainer.datamodule.teeth_classes_to_labels(classes)
        labels.cache['rot_matrix'] = affine

        # interpolate clusters back to original scan
        instances = clusters.interpolate(x_rot)

        return x_rot, instances, labels
    
    def landmarks_stage(
        self,
        x: PointTensor,
        instances: PointTensor,
        labels: PointTensor,
    ) -> Tuple[PointTensor, PointTensor]:
        instances = instances[x.cache['landmarks_downsample_idxs']]
        x_down = x[x.cache['landmarks_downsample_idxs']]
        
        # generate proposals based on predicted instances
        data_dict = {
            'points': x_down.C.cpu().numpy(),
            'instances': instances.F.cpu().numpy(),
            'instance_centroids': labels.C.cpu().numpy(),
            'normals': x_down.F[:, 3:].cpu().numpy(),
        }
        data_dict = self.gen_proposals(**data_dict)
        points = torch.from_numpy(data_dict['points']).to(x.C)
        normals = torch.from_numpy(data_dict['normals']).to(x.C)
        centroids = torch.from_numpy(data_dict['instance_centroids']).to(x.C)
        proposals = PointTensor(
            coordinates=points.reshape(-1, 3),
            features=torch.column_stack((
                points.reshape(-1, 3),
                normals.reshape(-1, 3),
                (points - centroids[:, None]).reshape(-1, 3),
            )),
            batch_counts=torch.tensor([points.shape[1]]*points.shape[0]).to(x.C.device),
        )

        # run the proposals through the models
        _, preds = self.landmarks_model(proposals)
        seg = preds[0]
        points_offsets = preds[1:]

        # interpolate segmentations to original points
        instances = torch.full_like(x.F[:, 0], -1).long()
        max_probs = torch.zeros_like(instances).float()
        probs = seg.new_tensor(features=torch.sigmoid(seg.F[:, 0]))
        for b in range(probs.batch_size):
            interp = probs.batch(b).interpolate(x, dist_thresh=0.03).F
            instances = torch.where((interp >= 0.5) & (interp > max_probs), b, instances)
            max_probs = torch.maximum(max_probs, interp)
        instances = x.new_tensor(features=instances)
        instances = self.trainer.datamodule.process_instances(instances)

        # process point-level landmarks
        landmarks_list = []
        for i, offsets in enumerate(points_offsets):
            kpt_mask = offsets.F[:, 0] < 0.15  # 2.5 mm
            coords = proposals.C + offsets.F[:, 1:]
            dists = torch.clip(offsets.F[:, 0], 0, 0.15)
            weights = (0.15 - dists) / 0.15
            landmarks = PointTensor(
                coordinates=coords[kpt_mask],
                features=weights[kpt_mask],
                batch_counts=torch.bincount(
                    input=proposals.batch_indices[kpt_mask],
                    minlength=proposals.batch_size,
                ),
            )
            landmarks = landmarks.cluster(**self.dbscan_cfg)
            landmarks = landmarks.new_tensor(features=torch.column_stack((landmarks.F, 
                torch.full((landmarks.C.shape[0],), i).to(coords.device),
            )))
            landmarks_list.append(landmarks)
        landmarks = teethland.cat(landmarks_list)

        landmarks = self.trainer.datamodule.process_landmarks(labels, landmarks)

        return instances, landmarks

    def forward(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor]:
        # stage 1
        x, instances, labels = self.instances_stage(x)
        if torch.all(instances.F == -1) or self.only_dentalnet:
            return instances, labels, None
        
        # stage 2
        instances, landmarks = self.landmarks_stage(x, instances, labels)

        return instances, labels, landmarks
    
    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        instances, classes, landmarks = self(batch)

        self.save_segmentation(instances, classes)
        self.save_landmarks(landmarks)

    def save_segmentation(self, instances: PointTensor, labels: PointTensor):
        labels = instances.new_tensor(
            features=torch.where(instances.F >= 0, labels.F[instances.F], 0),
        )

        out_dict = {
            'instances': instances.F.cpu().tolist(),
            'labels': labels.F.cpu().tolist(),
        }
        
        out_name = Path(self.trainer.datamodule.scan_file).with_suffix('.json')
        if self.out_dir.name:
            out_file = self.out_dir / out_name.name
        else:
            out_file = self.trainer.datamodule.root / out_name
        with open(out_file, 'w') as f:
            json.dump(out_dict, f, indent=2)

    def save_landmarks(self, landmarks: Optional[PointTensor]):
        if landmarks is None:
            return
        
        template = {
            'version': '1.1',
            'description': 'landmarks',
            'key': self.trainer.datamodule.scan_file,
            'objects': [],
        }
        for i, (coords, (score, cls), instance) in enumerate(zip(
            landmarks.C, landmarks.F, landmarks.batch_indices,
        )):
            landmark = {
                'key': f'uuid_{i}',
                'score': score.cpu().item(),
                'class': self.landmark_class2label[cls.cpu().item()],
                'coord': coords.cpu().tolist(),
                'instance_id': instance.cpu().item()
            }
            template['objects'].append(landmark)

        out_name = Path(template['key'][:-4] + '__kpt.json')
        if self.out_dir.name:
            out_file = self.out_dir / out_name.name
        else:
            out_file = self.trainer.datamodule.root / out_name
        with open(out_file, 'w') as f:
            json.dump(template, f, indent=2)
