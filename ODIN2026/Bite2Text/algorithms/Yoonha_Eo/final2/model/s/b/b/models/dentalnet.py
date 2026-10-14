from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
    MulticlassF1Score,
)
from torchtyping import TensorType

from teethland import PointTensor
from teethland.cluster import learned_region_cluster
from teethland.metrics import ToothF1Score
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class DentalNet(pl.LightningModule):
    """
    Implements DentalNet for point cloud instance segmentation.
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
        self.instance_model = nn.StratifiedTransformer(
            out_channels=[6, 1, None],
            **model_args,
        )
        with torch.no_grad():
            output_layer = self.instance_model.heads[0].linear
            # offsets
            output_layer.weight[:, :3].fill_(0)
            output_layer.bias[:3].fill_(0)
            # sigmas
            output_layer.weight[:, 3:].fill_(0)
            output_layer.bias[3:].fill_(1)

        self.identify_model = nn.MaskedAveragePooling(
            num_features=self.instance_model.out_channels[-1],
            out_channels=model_args['num_classes'],
        )

        self.instance_criterion = nn.SpatialEmbeddingLoss()
        self.identify_criterion = nn.IdentificationLoss()

        self.dice = BinaryF1Score()
        self.iou = BinaryJaccardIndex()
        self.fdi_f1_batch = MulticlassF1Score(num_classes=model_args['num_classes'])
        self.fdi_f1_epoch = MulticlassF1Score(num_classes=model_args['num_classes'])
        self.tooth_f1 = ToothF1Score()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        x: PointTensor,
        labels: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, PointTensor, torch.Tensor]:
        x = x[x.cache['cp_downsample_idxs']]
        
        _, (spatial_embeds, seeds, features) = self.instance_model(x)
        prototypes, classes = self.identify_model(features, labels)

        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])

        return offsets, sigmas, seeds, prototypes, classes

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (instances, labels) = batch

        offsets, sigmas, seeds, prototypes, classes = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        identify_loss = self.identify_criterion(prototypes, classes, instances)

        if torch.isnan(identify_loss):
            print('NaN loss:', self.trainer.datamodule.scan_file)

        loss = instance_loss + identify_loss
        self.log_dict({
            'loss/train_instance': instance_loss,
            'loss/train_identify': identify_loss,
            'loss/train': loss,
        }, batch_size=x.batch_size)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> Tuple[PointTensor, PointTensor]:
        x, (instances, labels) = batch

        offsets, sigmas, seeds, prototypes, classes = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        identify_loss = self.identify_criterion(prototypes, classes, instances)

        loss = instance_loss + identify_loss

        log_dict = {
            'loss/val_instance': instance_loss,
            'loss/val_identify': identify_loss,
            'loss/val': loss,
        }

        if self.trainer.state.fn == 'validate' or self.current_epoch >= 1:
            pred_labels = learned_region_cluster(
                offsets, sigmas, seeds,
            )
            self.dice((pred_labels.F >= 0).long(), (labels.F >= 0).long())
            self.iou((pred_labels.F >= 0).long(), (labels.F >= 0).long())
            
            metric_dict = self.tooth_f1(pred_labels, classes, labels, instances)
            metric_dict = {f'{k}/val': v for k, v in metric_dict.items()}

            log_dict.update({
                'dice/val': self.dice,
                'iou/val': self.iou,
                **metric_dict,
            })
        
        self.fdi_f1_batch(classes.F, instances.F)
        self.fdi_f1_epoch(classes.F, instances.F)
        log_dict.update({'fdi_f1/val_batch': self.fdi_f1_batch.compute()})
        log_dict.update({'fdi_f1/val_epoch': self.fdi_f1_epoch})

        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=[
                *self.instance_model.param_groups(self.lr),
                *self.identify_model.param_groups(self.lr),
            ],
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
