import torch
from torchmetrics import Metric
from torchtyping import TensorType

from teethland import PointTensor


class ToothF1Score(Metric):

    full_state_update = False

    def __init__(self, iou_thresh: float=0.5, fdi: bool=False):
        super().__init__()

        self.iou_thresh = iou_thresh
        self.fdi = fdi
        
        self.add_state('tooth_tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tooth_fdi_tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tooth_fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tooth_fdi_fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tooth_fn', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tooth_fdi_fn', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        pred_instances: PointTensor,
        pred_classes: PointTensor,
        instances: PointTensor,
        classes: PointTensor,
    ) -> None:
        num_pred = pred_instances.F.amax() + 1
        num_true = instances.F.amax() + 1

        if num_pred == 0 or num_true == 0:
            self.tooth_fp += num_pred
            self.tooth_fdi_fp += num_pred
            self.tooth_fn += num_true
            self.tooth_fdi_fn += num_true
            return

        ious = torch.zeros((num_pred, num_true)).to(instances.F.device, torch.float32)
        tp_idxs = set()
        for i in range(num_pred):
            for j in range(num_true):
                if j in tp_idxs:
                    continue

                pred_pos = pred_instances.F == i
                target_pos = instances.F == j
                tp = (pred_pos & target_pos).sum()
                fp = (pred_pos & ~target_pos).sum()
                fn = (~pred_pos & target_pos).sum()
                
                iou = tp / (tp + fp + fn)
                ious[i, j] = iou

                if iou >= self.iou_thresh:
                    tp_idxs.add(j)
                    break                    
        
        fp = (ious.amax(1) < self.iou_thresh).sum()
        fn = (ious.amax(0) < self.iou_thresh).sum()
        tp = (num_pred + num_true - fp - fn) // 2
        self.tooth_fp += fp
        self.tooth_fn += fn
        self.tooth_tp += tp

        if not self.fdi:
            return

        idxs = torch.nonzero(ious >= self.iou_thresh)
        match_classes = torch.column_stack((
            pred_classes.F.argmax(-1)[idxs[:, 0]],
            classes.F[idxs[:, 1]],
        ))
        fp += (match_classes[:, 0] != match_classes[:, 1]).sum()
        fn += (match_classes[:, 0] != match_classes[:, 1]).sum()
        tp -= 2 * (match_classes[:, 0] != match_classes[:, 1]).sum()      
        self.tooth_fdi_fp += fp
        self.tooth_fdi_fn += fn
        self.tooth_fdi_tp += tp

    def compute(self) -> TensorType[torch.float32]:
        out_dict = {
            'tooth_precision': self.tooth_tp / (self.tooth_tp + self.tooth_fp),
            'tooth_sensitivity': self.tooth_tp / (self.tooth_tp + self.tooth_fn),
            'tooth_f1': 2 * self.tooth_tp / (2 * self.tooth_tp + self.tooth_fp + self.tooth_fn),
        }
        if not self.fdi:
            return out_dict
        
        out_dict.update({ 
            'tooth_fdi_precision': self.tooth_fdi_tp / (self.tooth_fdi_tp + self.tooth_fdi_fp),
            'tooth_fdi_sensitivity': self.tooth_fdi_tp / (self.tooth_fdi_tp + self.tooth_fdi_fn),
            'tooth_fdi_f1': 2 * self.tooth_fdi_tp / (2 * self.tooth_fdi_tp + self.tooth_fdi_fp + self.tooth_fdi_fn),
        })
        
        return out_dict
