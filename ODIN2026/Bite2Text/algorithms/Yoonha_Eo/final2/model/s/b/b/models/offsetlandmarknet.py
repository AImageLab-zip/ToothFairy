from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
)
from torchmetrics.regression import MeanSquaredError
from torchtyping import TensorType

import teethland
from teethland import PointTensor
from teethland.metrics import (
    LandmarkF1Score,
    LandmarkMeanAveragePrecision,
)
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class OffsetLandmarkNet(pl.LightningModule):
    """
    Implements network for tooth instance landmark detection.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        dbscan_cfg: dict,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.model = nn.StratifiedTransformer(
            out_channels=[1, 4, 4, 4, 4, 4],
            **model_args,
        )

        self.landmark_criterion = nn.OffsetLandmarkLoss()
        self.seg_criterion = nn.BCELoss()

        self.dice = BinaryF1Score()
        self.iou = BinaryJaccardIndex()
        self.mse = MeanSquaredError()
        self.landmark_f1 = LandmarkF1Score()
        self.landmark_map = LandmarkMeanAveragePrecision()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.dbscan_cfg = dbscan_cfg

    def forward(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, PointTensor, PointTensor, PointTensor]:
        _, (seg, mesial_distal, facial, outer, inner, cusps) = self.model(x)

        return seg, mesial_distal, facial, outer, inner, cusps

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (instances, landmarks, labels) = batch

        seg, mesial_distal, facial, outer, inner, cusps = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        mesial_distal_loss = self.landmark_criterion(mesial_distal, landmarks, [0, 1])
        facial_loss = self.landmark_criterion(facial, landmarks, [2])
        outer_loss = self.landmark_criterion(outer, landmarks, [3])
        inner_loss = self.landmark_criterion(inner, landmarks, [4])
        cusps_loss = self.landmark_criterion(cusps, landmarks, [5])

        loss = {
            'loss/train_seg': seg_loss,
            'loss/train_mesial_distal': mesial_distal_loss,
            'loss/train_facial': facial_loss,
            'loss/train_outer': outer_loss,
            'loss/train_inner': inner_loss,
            'loss/train_cusps': cusps_loss,
        }
        
        # self.mse(coords.F[instances.F[:, -1] == 1], instances.F[instances.F[:, -1] == 1, :-1])
        self.dice((seg.F[:, 0] >= 0).long(), (labels.F >= 0).long())

        log_dict = {
            **loss,
            'loss/train': sum([v for v in loss.values()]),
            'dice/train': self.dice,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return sum([v for v in loss.values()])

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> Tuple[PointTensor, PointTensor]:
        x, (instances, landmarks, labels) = batch

        seg, mesial_distal, facial, outer, inner, cusps = self(x)

        seg_loss = self.seg_criterion(seg, labels)
        mesial_distal_loss = self.landmark_criterion(mesial_distal, landmarks, [0, 1])
        facial_loss = self.landmark_criterion(facial, landmarks, [2])
        outer_loss = self.landmark_criterion(outer, landmarks, [3])
        inner_loss = self.landmark_criterion(inner, landmarks, [4])
        cusps_loss = self.landmark_criterion(cusps, landmarks, [5])

        loss = {
            'loss/val_seg': seg_loss,
            'loss/val_mesial_distal': mesial_distal_loss,
            'loss/val_facial': facial_loss,
            'loss/val_outer': outer_loss,
            'loss/val_inner': inner_loss,
            'loss/val_cusps': cusps_loss,
        }
        
        # self.mse(coords.F[instances.F[:, -1] == 1], instances.F[instances.F[:, -1] == 1, :-1])
        self.dice((seg.F[:, 0] >= 0).long(), (labels.F >= 0).long())


        # process point-level landmarks
        landmarks_list = []
        for i, offsets in enumerate([mesial_distal, facial, outer, inner, cusps]):
            kpt_mask = offsets.F[:, 0] < 0.08
            coords = x.C + offsets.F[:, 1:]
            pred_landmarks = PointTensor(
                coordinates=coords[kpt_mask],
                batch_counts=torch.bincount(
                    x.batch_indices[kpt_mask],
                    minlength=x.batch_size,
                ),
            )
            pred_landmarks = pred_landmarks.cluster(**self.dbscan_cfg)
            pred_landmarks = pred_landmarks.new_tensor(
                features=torch.full((pred_landmarks.C.shape[0],), i).to(coords.device),
            )
            landmarks_list.append(pred_landmarks)
        pred_landmarks = teethland.cat(landmarks_list)

        landmarks = landmarks.new_tensor(features=torch.clip(landmarks.F - 1, 0, 4))
        self.landmark_map.update(pred_landmarks, landmarks)

        log_dict = {
            **loss,
            'loss/val': sum([v for v in loss.values()]),
            'dice/val': self.dice,
            'map/val': self.landmark_map,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=self.model.param_groups(self.lr),
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
