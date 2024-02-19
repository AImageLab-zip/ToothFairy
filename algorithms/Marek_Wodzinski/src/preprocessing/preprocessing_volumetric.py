### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import numpy as np
import scipy.ndimage as nd
import torch as tc
import torch.nn.functional as F

### Internal Imports ###
from augmentation import volumetric as aug_vol
from augmentation import aug


########################


def resample_to_shape_np(image : np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Resamples image to a given shape.
    """
    shape = image.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[0]), np.arange(new_shape[2]))
    grid_x = grid_x * (shape[1] / new_shape[1])
    grid_y = grid_y * (shape[0] / new_shape[0])
    grid_z = grid_z * (shape[2] / new_shape[2])
    resampled_image = nd.map_coordinates(image, [grid_y, grid_x, grid_z], order=0, cval=0)
    return resampled_image

def generate_grid(tensor_size: tc.Tensor, device=None):
    """
    Generates the identity grid for a given tensor size.
    Parameters
    ----------
    tensor_size : tc.Tensor or tc.Size
        The tensor size used to generate the regular grid
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    
    Returns
    ----------
    grid : tc.Tensor
        The regular grid (relative for warp_tensor with align_corners=False)
    """
    identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def resample_tensor(tensor: tc.Tensor, new_size: tc.Tensor, device: str=None, mode: str='bilinear'):
    """
    Resamples the input tensor to a given, new size (may be used both for down and upsampling).
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".
    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be resampled (BxYxXxZxD)
    new_size : tc.Tensor (or list, or tuple)
        The resampled tensor size
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    Returns
    ----------
    resampled_tensor : tc.Tensor
        The resampled tensor (Bxnew_sizexD)
    """
    device = device if device is not None else tensor.device
    sampling_grid = generate_grid(new_size, device=device)
    resampled_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return resampled_tensor

def resample_to_spacing_np(image : np.ndarray, old_spacing : tuple, new_spacing : tuple) -> np.ndarray:
    """
    Resamples image to a given spacing.
    """
    shape = image.shape
    multiplier = (np.array(old_spacing, dtype=np.float32) / np.array(new_spacing, dtype=np.float32))
    multiplier[0], multiplier[1] = multiplier[1], multiplier[0]
    new_shape = shape * multiplier
    new_shape = np.ceil(new_shape).astype(int)
    resampled_image = resample_to_shape_np(image, new_shape)
    return resampled_image

def resample_to_spacing_tc(tensor : tc.Tensor, old_spacing : Union[tc.Tensor, tuple], new_spacing : Union[tc.Tensor, tuple], mode : str="linear") -> tc.Tensor:
    """
    Resamples input tensor to a given spacing.
    """
    shape = tensor.shape[2:]
    multiplier = (np.array(old_spacing, dtype=np.float32) / np.array(new_spacing, dtype=np.float32))
    multiplier[0], multiplier[1] = multiplier[1], multiplier[0]
    new_shape = shape * multiplier
    new_shape = np.ceil(new_shape).astype(int)
    new_shape = (tensor.shape[0], tensor.shape[1]) + tuple(new_shape)
    resampled_image = resample_tensor(tensor, new_shape, mode=mode)
    return resampled_image

def resample_to_spacing(volume : Union[np.ndarray, tc.Tensor], old_spacing : Union[tc.Tensor, tuple], new_spacing : Union[tc.Tensor, tuple]) -> Union[np.ndarray, tc.Tensor]:
    if isinstance(volume, tc.Tensor):
        return resample_to_spacing_tc(volume, old_spacing, new_spacing)
    elif isinstance(volume, np.ndarray):
        return resample_to_spacing_np(volume, old_spacing, new_spacing)
    else:
        raise ValueError("Unsupported type.")


### Preprocessing Generators ###

def generate_resampling_to_spacing_function(new_spacing : tuple[float, float, float]) -> Callable[[tc.Tensor, dict], tc.Tensor]:
    """
    Generates transform to resample a given tensor to new_spacing.
    """
    def inner(tensor : tc.Tensor, **kwargs : dict) -> tc.Tensor:
        try:
            old_spacing = kwargs['old_spacing']
        except:
            raise ValueError("Original spacing must be provided.")
        transform = aug_vol.generate_resampling_transform(old_spacing, new_spacing)
        resampled_tensor, _ = aug.apply_transform(tensor, transform=transform)
        return resampled_tensor[0].unsqueeze(0)
    return inner

def generate_resampling_to_shape_function(new_shape : tuple[int, int, int]) -> Callable[[tc.Tensor, dict], tc.Tensor]:
    """
    Generates transform to resample a given tensor to new shape.
    """
    def inner(tensor, **kwargs) -> tc.Tensor:
        old_shape = tensor.shape[2:]
        transform = aug_vol.generate_resampling_to_shape_transform(old_shape, new_shape)
        output_volume, _ = aug.apply_transform(tensor, transform=transform)
        return output_volume[0]
    return inner