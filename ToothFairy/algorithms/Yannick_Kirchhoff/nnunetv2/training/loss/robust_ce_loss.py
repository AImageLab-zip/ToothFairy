import torch
from torch import nn, Tensor
import numpy as np
from typing import Union, Callable, List

from nnunetv2.utilities.helpers import softmax_helper_dim1


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class FocalLoss(nn.CrossEntropyLoss):
    """
    This is an implementation of Focal Loss (https://arxiv.org/abs/1708.02002)
    similar to https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py
    :param apply_nonlin: non linearity applied to raw logits
    :param alpha: class weighting
    :param gamma: scaling factor to stronger penalize worse predictions, gamma=0 is basically cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param smooth: smoothing factor (similar to label_smoothing in pytorch's CrossEntropyLoss)
    :param eps: additional smoothing if smooth=0. prevents infs in pt.log()
    :param reduction: reduction method, can either be 'mean' or 'sum'
    """

    def __init__(self, apply_nonlin: Union[None, Callable]=softmax_helper_dim1, alpha: Union[None, float, List, np.ndarray]=None, gamma: float=2, 
                 balance_index: int=0, smooth: Union[None, float]=1e-5, eps: Union[None, float]=1e-8, reduction: str="mean"):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth if smooth is not None else 0
        self.eps = smooth if smooth>0 or eps is None else eps
        self.reduction = reduction

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, loss_mask=None):
        logit_shp = logit.shape
        target_shp = target.shape
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit_shp[1]

        if self.alpha is None:
            alpha = torch.ones(*logit_shp, device=logit.device)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == num_class
            alpha = torch.tensor(self.alpha).view(1, num_class, *(1,)*len(logit_shp[2:])).to(device=logit.device)
            alpha = alpha / alpha.sum()
            alpha = alpha.expand(*logit_shp)
        elif isinstance(self.alpha, float):
            alpha = torch.ones(*logit_shp, device=logit.device)
            alpha = alpha * (1 - self.alpha)
            alpha[:,self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        with torch.no_grad():
            if len(logit_shp)!=len(target_shp):
                target = target.view((target_shp[0], 1, *target_shp[1:]))

            if all([i == j for i, j in zip(logit_shp, target_shp)]):
                gt = target.argmax(1, keepdim=True)
                target_onehot = target
            else:
                gt = target.long()
                target_onehot = torch.zeros(logit_shp, device=logit.device, dtype=torch.bool)
                target_onehot.scatter_(1, gt, 1)

        target_onehot = torch.clamp(target_onehot, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (target_onehot * logit).sum(1, keepdim=True) + self.eps
        logpt = pt.log()

        loss = -1 * alpha.gather(1, gt) * torch.pow((1 - pt), self.gamma) * logpt

        if loss_mask is not None:
            loss = loss[loss_mask]

        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        else:
            raise NotImplementedError(f"it seems that the reduction method {self.reduction} is not implemented")