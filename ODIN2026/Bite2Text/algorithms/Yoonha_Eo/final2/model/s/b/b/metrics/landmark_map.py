from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchtyping import TensorType

from teethland import PointTensor


class LandmarkMeanAveragePrecision(Metric):

    full_state_update = False

    def __init__(
        self,
        dist_threshs: List[float]=[
            0.5 / 17.3281,
            1 / 17.3281,
            2 / 17.3281,
            3 / 17.3281,
        ],
    ):
        super().__init__()

        self.dist_threshs = dist_threshs
        
        self.add_state('pred_coords', default=[], dist_reduce_fx='cat')
        self.add_state('pred_classes', default=[], dist_reduce_fx='cat')
        self.add_state('pred_batch_counts', default=[], dist_reduce_fx='cat')
        
        self.add_state('gt_coords', default=[], dist_reduce_fx='cat')
        self.add_state('gt_classes', default=[], dist_reduce_fx='cat')
        self.add_state('gt_batch_counts', default=[], dist_reduce_fx='cat')

    def update(
        self,
        pred_landmarks: PointTensor,
        landmarks: PointTensor,
    ) -> None:
        self.pred_coords.append(pred_landmarks.C)
        self.pred_classes.append(pred_landmarks.F)
        self.pred_batch_counts.append(torch.tensor(pred_landmarks.C.shape[0]).to(pred_landmarks.F))

        self.gt_coords.append(landmarks.C)
        self.gt_classes.append(landmarks.F)
        self.gt_batch_counts.append(torch.tensor(landmarks.C.shape[0]).to(landmarks.F))

    def voc_ap(self, rec, prec):
        rec = rec.cpu().numpy()
        prec = prec.cpu().numpy()

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return torch.tensor(ap)

    def eval_ap(
        self,
        pred_coords,
        pred_batch_idxs,
        gt_coords,
        gt_batch_idxs,
        dist_thresh,
    ):
        batch_idxs = torch.unique(torch.cat((pred_batch_idxs, gt_batch_idxs)))

        # construct gt objects
        class_recs = {}  # {mesh idx: {'kp': kp list, 'det': matched list}}
        npos = 0
        for i in batch_idxs:
            keypoints = gt_coords[gt_batch_idxs == i]
            det = [False] * keypoints.shape[0]
            npos += keypoints.shape[0]
            class_recs[i.item()] = {'kp': keypoints, 'det': det}

        # construct dets
        mesh_idxs = []
        confidence = []
        KP = []
        for i in batch_idxs:
            for kp in pred_coords[pred_batch_idxs == i]:
                mesh_idxs.append(i)
                confidence.append(1.0)
                KP.append(kp)
        confidence = torch.tensor(confidence).to(pred_coords)
        KP = torch.stack(KP)

        # go down dets and mark TPs and FPs
        nd = pred_coords.shape[0]
        tp = torch.zeros(nd)
        fp = torch.zeros(nd)
        for d in range(nd):
            R = class_recs[pred_batch_idxs[d].item()]
            kp = KP[d]
            dmin = torch.inf
            KPGT = R['kp']

            if KPGT.numel() > 0:
                distance = torch.linalg.norm(kp.reshape(-1, 3) - KPGT, axis=1)
                dmin = min(distance)
                jmin = torch.argmin(distance)

            # print dmin
            if dmin < dist_thresh:
                if not R['det'][jmin]:
                    tp[d] = 1.
                    R['det'][jmin] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = torch.cumsum(fp, dim=0)
        tp = torch.cumsum(tp, dim=0)
        rec = tp / float(npos)
        
        prec = tp / torch.maximum(tp + fp, torch.tensor(1e-9))
        ap = self.voc_ap(rec, prec)

        return ap

    def eval_map(
        self,
        dist_thresh: float,
    ):
        aps = []
        for cls in torch.unique(self.gt_classes):
            pred_coords = self.pred_coords[self.pred_classes == cls]            
            pred_batch_idxs = torch.arange(self.pred_batch_counts.shape[0]).to(self.pred_classes)
            pred_batch_idxs = pred_batch_idxs.repeat_interleave(self.pred_batch_counts)
            pred_batch_idxs = pred_batch_idxs[self.pred_classes == cls]

            gt_coords = self.gt_coords[self.gt_classes == cls]            
            gt_batch_idxs = torch.arange(self.gt_batch_counts.shape[0]).to(self.gt_classes)
            gt_batch_idxs = gt_batch_idxs.repeat_interleave(self.gt_batch_counts)
            gt_batch_idxs = gt_batch_idxs[self.gt_classes == cls]

            ap = self.eval_ap(
                pred_coords, pred_batch_idxs, gt_coords, gt_batch_idxs, dist_thresh,
            )
            aps.append(ap)

        return torch.stack(aps)

    def compute(self) -> Dict[str, TensorType[torch.float32]]:
        self.pred_coords = dim_zero_cat(self.pred_coords)
        self.pred_classes = dim_zero_cat(self.pred_classes)
        self.pred_batch_counts = dim_zero_cat(self.pred_batch_counts)
        self.gt_coords = dim_zero_cat(self.gt_coords)
        self.gt_classes = dim_zero_cat(self.gt_classes)
        self.gt_batch_counts = dim_zero_cat(self.gt_batch_counts)

        out_dict = {}
        for dist_thresh in self.dist_threshs:
            aps = self.eval_map(dist_thresh)
            print(dist_thresh, ':', aps.mean(), aps)
            out_dict[f'mAP_{dist_thresh:.2f}'] = aps.mean()
        
        
        return torch.tensor(sum(out_dict.values()) / len(out_dict))
 