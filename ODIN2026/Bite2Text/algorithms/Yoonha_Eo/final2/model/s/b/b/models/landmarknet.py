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

from teethland import PointTensor
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class LandmarkNet(pl.LightningModule):
    """
    Implements network for tooth instance landmark detection.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        num_classes: int,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.backbone = nn.StratifiedTransformer(
            out_channels=[1, None, 1 + 3],
            **model_args,
        )
        self.map_mlp = nn.MaskedAveragePooling(
            num_features=self.backbone.out_channels[1],
            out_channels=num_classes * 4,
        )

        self.landmark_criterion = nn.LandmarkLoss()
        self.seg_criterion = nn.BCELoss()

        self.dice = BinaryF1Score()
        self.iou = BinaryJaccardIndex()
        self.mse = MeanSquaredError()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        x: PointTensor,
        labels: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, PointTensor]:
        _, (seg, features, point_offsets) = self.backbone(x)
        prototypes, instance_offsets = self.map_mlp(features, labels)

        return seg, prototypes, instance_offsets, point_offsets

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (instances, landmarks, labels) = batch

        seg, prototypes, instance_offsets, point_offsets = self(x, labels)

        seg_loss = self.seg_criterion(seg, labels)
        instances_loss, points_loss = self.landmark_criterion(
            prototypes, instance_offsets, point_offsets, instances, landmarks,
        )

        loss = seg_loss + instances_loss + points_loss

        log_dict = {
            'loss/train_seg': seg_loss,
            'loss/train_instances': instances_loss,
            'loss/train_points': points_loss,
            'loss/train': loss,
        }
        
        # self.mse(coords.F[instances.F[:, -1] == 1], instances.F[instances.F[:, -1] == 1, :-1])
        self.dice((seg.F[:, 0] >= 0).long(), (labels.F >= 0).long())
        log_dict.update({'dice/train': self.dice})

        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor, PointTensor]],
        batch_idx: int,
    ):
        x, (instances, landmarks, labels) = batch

        seg, prototypes, instance_offsets, point_offsets = self(x, labels)

        from teethland.visualization import draw_landmarks
        # draw_landmarks(x.batch(0).F[:, -3:].cpu().numpy(), instance_offsets.F.reshape(-1, 5, 4)[0, :, :3].cpu().numpy())
        # points_mask = torch.argsort(point_offsets.batch(0).F[:, 0])[:200]
        # landmarks.batch(0).F
        # draw_landmarks(x.batch(0).C.cpu().numpy(), (x.batch(0).C + point_offsets.batch(0).F[:, 1:])[points_mask].cpu().numpy())

        seg_loss = self.seg_criterion(seg, labels)
        instances_loss, points_loss = self.landmark_criterion(
            prototypes, instance_offsets, point_offsets, instances, landmarks,
        )

        loss = seg_loss + instances_loss + points_loss

        log_dict = {
            'loss/val_seg': seg_loss,
            'loss/val_instances': instances_loss,
            'loss/val_points': points_loss,
            'loss/val': loss,
        }
        
        # self.mse(coords.F[instances.F[:, -1] == 1], instances.F[instances.F[:, -1] == 1, :-1])
        self.dice((seg.F[:, 0] >= 0).long(), (labels.F >= 0).long())
        log_dict.update({'dice/val': self.dice})

        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=[
                *self.backbone.param_groups(self.lr),
                *self.map_mlp.param_groups(self.lr),
            ],
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
