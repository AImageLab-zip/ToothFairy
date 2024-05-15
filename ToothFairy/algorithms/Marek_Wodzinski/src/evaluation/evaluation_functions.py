### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
from scipy.ndimage import morphology

### Internal Imports ###
from evaluation import np_metrics as npm
from evaluation import tc_metrics as tcm

########################

def dc(
    input : Union[np.ndarray, tc.Tensor],
    ground_truth : Union[np.ndarray, tc.Tensor]) -> float:
    """
    Computes the Dice Coefficient between the Input and Grount-Truth.
    """
    return dice_coefficient(input, ground_truth)

def dice_coefficient(
    input : Union[np.ndarray, tc.Tensor],
    ground_truth : Union[np.ndarray, tc.Tensor]) -> float:
    """
    Computes the Dice Coefficient between the Input and Ground-Truth.
    """
    if all(isinstance(var, np.ndarray) for var in [input, ground_truth]):
        return npm.dc(input, ground_truth)
    elif all(isinstance(var, tc.Tensor) for var in [input, ground_truth]):
        raise NotImplementedError() # TODO - Dice Coefficient for Tensors
    else:
        raise ValueError("Input and Grount-Truth must be of the same type.")
    

def multiclass_dice_coefficient(
    input : Union[np.ndarray, tc.Tensor],
    ground_truth : Union[np.ndarray, tc.Tensor]) -> float:
    """
    Computes the Dice Coefficient between the Input and Ground-Truth.
    """
    if all(isinstance(var, np.ndarray) for var in [input, ground_truth]):
        dice = 0.0
        for i in range(ground_truth.shape[0]):
            dice += npm.dc(input[i], ground_truth[i])
        return dice / ground_truth.shape[0]
    elif all(isinstance(var, tc.Tensor) for var in [input, ground_truth]):
        raise NotImplementedError() # TODO - Dice Coefficient for Tensors
    else:
        raise ValueError("Input and Grount-Truth must be of the same type.")

def hausdorff_distance_95(
    input : Union[np.ndarray, tc.Tensor],
    ground_truth : Union[np.ndarray, tc.Tensor],
    voxelspacing : tuple=None,
    connectivity : int=1) -> float:
    """
    Computes the 95th Percentile of the Hausdorff Distance between the Input and Ground-Truth.
    """
    if all(isinstance(var, np.ndarray) for var in [input, ground_truth]):    
        return npm.hd95(input, ground_truth, voxelspacing, connectivity)
    elif all(isinstance(var, tc.Tensor) for var in [input, ground_truth]):
        raise NotImplementedError() # TODO - HD95 for Tensors
    else:
        raise ValueError("Input and Grount-Truth must be of the same type.")
    

def msd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(bool))
    input_2 = np.atleast_1d(input2.astype(bool))
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    return sds.mean()