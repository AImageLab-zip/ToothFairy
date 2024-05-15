import torch
from torch import nn
from typing import Callable

try:
    from jitfields.distance import euclidean_distance_transform
except (ImportError, ModuleNotFoundError) as edt_e:
    euclidean_distance_transform = None

from nnunetv2.utilities.helpers import softmax_helper_dim1


class HausdorffDTLoss(nn.Module):
    """
    This is a multiclass implementation of the HausdorffDTLoss as implemented in 
    https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
    with some minor modifications
    """

    def __init__(self, apply_nonlin: Callable = softmax_helper_dim1, alpha=2.0, max_dt=None) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.max_dt = max_dt

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        if euclidean_distance_transform is None:
            raise edt_e
        fg_mask = (img > 0.5).float()
        if not fg_mask.any():
            return (torch.zeros_like(img),)*2
        fg_dist = euclidean_distance_transform(fg_mask, ndim=img.ndim-2)
        bg_dist = euclidean_distance_transform(1-fg_mask, ndim=img.ndim-2)
        return fg_dist, bg_dist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one multiple channels: 0 - bg, 1..c-1 - fg
        pred: (b, c, x, y(, z))
        target: (b, 1, x, y(, z)) or (b, c, x, y(, z))
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        shp_pred = pred.shape
        shp_target = target.shape

        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(pred)

        with torch.no_grad():
            if len(shp_pred) != len(shp_target):
                target = target.view((shp_target[0], 1, *shp_target[1:]))

            if all([i == j for i, j in zip(shp_pred, shp_target)]):
                # if this is the case then gt is probably already a one hot encoding
                target_onehot = target
            else:
                gt = target.long()
                target_onehot = torch.zeros(shp_pred, device=pred.device, dtype=pred.dtype)
                target_onehot.scatter_(1, gt, 1)

        # we don't care about background
        pred = pred[:, 1:]
        target_onehot = target_onehot[:, 1:]

        pred_dt_fg, pred_dt_bg = self.distance_field(pred)
        target_dt_fg, target_dt_bg = self.distance_field(target_onehot)

        if self.max_dt is not None:
            torch.clamp_max_(torch.nan_to_num_(pred_dt_fg, 0, 0, 0), self.max_dt)
            torch.clamp_max_(torch.nan_to_num_(pred_dt_bg, 0, 0, 0), self.max_dt)
            torch.clamp_max_(torch.nan_to_num_(target_dt_fg, 0, 0, 0), self.max_dt)
            torch.clamp_max_(torch.nan_to_num_(target_dt_bg, 0, 0, 0), self.max_dt)

        pred_error = (pred - target_onehot) ** 2
        distance_fg = pred_dt_fg ** self.alpha + target_dt_bg ** self.alpha
        distance_bg = pred_dt_bg ** self.alpha + target_dt_fg ** self.alpha

        dt_field_fg = pred_error * distance_fg
        dt_field_bg = pred_error * distance_bg
        
        loss = 0.5 * (dt_field_fg[distance_fg!=0].mean() + dt_field_bg[distance_bg!=0].mean())

        # loss can be nan if distance_fg==0 or distance_bg==0 everywhere
        return torch.nan_to_num_(loss)