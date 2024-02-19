### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
from enum import Enum

### External Imports ###
import torch as tc
import torch.nn.functional as F
from monai import losses

### Internal Imports ###

########################


### Volumetric Losses ###

def dice_loss(prediction : tc.Tensor, target : tc.Tensor) -> tc.Tensor:
    """
    Dice as PyTorch cost function.
    """
    smooth = 1
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = tc.sum(prediction * target)
    return 1 - ((2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))

def dice_loss_multichannel(prediction : tc.Tensor, target : tc.Tensor) -> tc.Tensor:
    """
    Dice loss for multichannel masks (equally averaged)
    """
    no_channels = prediction.size(1)
    for i in range(no_channels):
        if i == 0:
            loss = dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
        else:
            loss += dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
    loss = loss / no_channels
    return loss

########################


### MONAI Volumetric Losses ###

def dice_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Dice loss based on MONAI implementation.
    """
    return losses.DiceLoss(reduction='mean', **kwargs)(prediction, target)

def dice_ce_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Cross Entropy losses based on MONAI implementation.
    """
    return losses.DiceCELoss(reduction='mean', **kwargs)(prediction, target)

def dice_ce_loss_monai_2(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Cross Entropy losses based on MONAI implementation.
    """
    return losses.DiceCELoss(reduction='mean', sigmoid=True, **kwargs)(prediction, target)

def dice_focal_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Focal losses based on MONAI implementation.
    """
    return losses.DiceFocalLoss(reduction='mean', **kwargs)(prediction, target)

def generalized_dice_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Generalized Dice loss based on MONAI implementation.
    """
    return losses.GeneralizedDiceLoss(reduction='mean', **kwargs)(prediction, target)

def generalized_dice_focal_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Generalized Dice and Focal losses based on MONAI implementation.
    """
    return losses.GeneralizedDiceFocalLoss(reduction='mean', **kwargs)(prediction, target)

########################
