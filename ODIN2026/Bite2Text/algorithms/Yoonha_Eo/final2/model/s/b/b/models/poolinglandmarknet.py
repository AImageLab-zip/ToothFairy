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


class PoolingLandmarkNet(pl.LightningModule):
    """
    Implements network for tooth instance landmark detection.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.backbone = nn.StratifiedTransformer(
            out_channels=[1, None],
            **model_args,
        )
        self.map_mlp = nn.MaskedAveragePooling(
            num_features=self.backbone.out_channels[-1],
            out_channels=10*4,
        )

        self.landmark_criterion = nn.PoolingLandmarkLoss()
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
    ) -> Tuple[PointTensor, PointTensor, PointTensor]:
        _, (seg, features) = self.backbone(x)

        prototypes, landmarks = self.map_mlp(features, labels)

        return seg, prototypes, landmarks

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (instances, labels) = batch

        seg, prototypes, landmarks = self(x, labels)

        seg_loss = self.seg_criterion(seg, labels)
        landmark_loss = self.landmark_criterion(prototypes, landmarks, instances)

        loss = seg_loss + landmark_loss
        
        # self.mse(coords.F[instances.F[:, -1] == 1], instances.F[instances.F[:, -1] == 1, :-1])
        self.dice((seg.F[:, 0] >= 0).long(), (labels.F >= 0).long())

        log_dict = {
            'loss/train_seg': seg_loss,
            'loss/train_landmark': landmark_loss,
            'loss/train': loss,
            'dice/train': self.dice
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> Tuple[PointTensor, PointTensor]:
        x, (instances, labels) = batch

        seg, prototypes, landmarks = self(x, labels)

        seg_loss = self.seg_criterion(seg, labels)
        landmark_loss = self.landmark_criterion(prototypes, landmarks, instances)

        loss = seg_loss + landmark_loss

        log_dict = {
            'loss/val_seg': seg_loss,
            'loss/val_landmark': landmark_loss,
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
