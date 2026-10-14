from scipy.optimize import linear_sum_assignment
import torch
from torchmetrics import Metric
from torchtyping import TensorType

from teethland import PointTensor


class LandmarkF1Score(Metric):

    full_state_update = False

    def __init__(self, dist_thresh: float=17.3281):
        super().__init__()

        self.dist_thresh = dist_thresh
        
        self.add_state('landmark_tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('landmark_fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('landmark_fn', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        pred_landmarks: PointTensor,
        landmarks: PointTensor,
    ) -> None:
        num_pred = pred_landmarks.C.shape[0]
        num_true = landmarks.C.shape[0]

        if num_pred == 0 or num_true == 0:
            self.landmark_fp += num_pred
            self.landmark_fn += num_true
            return
        
        dists = torch.linalg.norm(
            pred_landmarks.C[:, None] - landmarks.C[None], dim=-1,
        )
        pred_idxs, gt_idxs = linear_sum_assignment(dists.cpu().numpy())
        pred_idxs = torch.from_numpy(pred_idxs).to(landmarks.F)
        gt_idxs = torch.from_numpy(gt_idxs).to(landmarks.F)

        pair_dists = dists[pred_idxs, gt_idxs]
        pred_idxs = pred_idxs[pair_dists < self.dist_thresh]
        gt_idxs = gt_idxs[pair_dists < self.dist_thresh]
        
        fp = num_pred - pred_idxs.shape[0]
        fn = num_true - gt_idxs.shape[0]
        tp = (num_pred + num_true - fp - fn) // 2
        self.landmark_fp += fp
        self.landmark_fn += fn
        self.landmark_tp += tp

    def compute(self) -> TensorType[torch.float32]:
        out_dict = {
            'landmark_precision': self.landmark_tp / (self.landmark_tp + self.landmark_fp),
            'landmark_sensitivity': self.landmark_tp / (self.landmark_tp + self.landmark_fn),
            'landmark_f1': 2 * self.landmark_tp / (2 * self.landmark_tp + self.landmark_fp + self.landmark_fn),
        }
        
        return out_dict
    