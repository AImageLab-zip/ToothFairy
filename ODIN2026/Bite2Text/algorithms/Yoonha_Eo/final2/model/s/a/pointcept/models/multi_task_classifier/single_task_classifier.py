import torch
import torch.nn as nn
from torch_scatter import segment_csr

from ..builder import MODELS, build_model
from pointcept.models.utils.structure import Point


@MODELS.register_module()
class SingleTaskClassifier(nn.Module):
    """One PTv3 backbone and one task-specific MLP classifier."""

    def __init__(
        self,
        backbone,
        num_classes,
        backbone_embed_dim,
        task_index,
        class_weight=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.task_index = int(task_index)
        self.head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        weight = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )

    def forward(self, input_dict):
        point = self.backbone(Point(input_dict))
        if isinstance(point, Point):
            feat = segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
        else:
            feat = point
        logits = self.head(feat)
        output = {"logits": logits}
        label_key = f"label_{self.task_index}"
        if label_key in input_dict:
            target = input_dict[label_key].view(-1)
            output["loss"] = self.criterion(logits, target)
        return output
