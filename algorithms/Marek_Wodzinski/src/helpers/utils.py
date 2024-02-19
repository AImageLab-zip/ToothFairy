### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union
from enum import Enum

### External Imports ###
import numpy as np
import scipy.ndimage as nd
import torch as tc
import SimpleITK as sitk

### Internal Imports ###



########################

def normalize(tensor : tc.Tensor):
    return (tensor - tc.min(tensor)) / (tc.max(tensor) - tc.min(tensor))

def normalize_to_window(tensor : tc.Tensor, min_value : float, max_value : float):
    return normalize(tensor) * (max_value - min_value) + min_value

def load_volume_sitk(
    input_path : Union[str, pathlib.Path],
    load_origin : bool=False,
    load_direction : bool=False) -> tuple[np.ndarray, tuple, dict]:
    """
    Utility function to load 3-D volume using SimpleITK.
    """
    image = sitk.ReadImage(str(input_path))
    spacing = image.GetSpacing()
    volume = sitk.GetArrayFromImage(image).swapaxes(0, 1).swapaxes(1, 2).astype(np.float32)
    metadata = dict()
    if load_origin:
        origin = image.GetOrigin()
        metadata['origin'] = origin
    if load_direction:
        direction = image.GetDirection()
        metadata['direction'] = direction
    return volume, spacing, metadata

def save_volume_sitk(
    volume : np.ndarray,
    spacing : tuple,
    save_path : Union[str, pathlib.Path],
    use_compression : bool=True,
    origin : tuple=None,
    direction : tuple=None) -> None:
    """
    Utility function to save 3-D volume using SimpleITK.
    """
    image  = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0).astype(np.uint8))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(save_path), useCompression=use_compression)


########################

def image_warping(
    image: np.ndarray,
    displacement_field: np.ndarray,
    order: int=1,
    cval: float=0.0) -> np.ndarray:
    """
    Warps the given image using the provided displacement field.
    """
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
    transformed_image = nd.map_coordinates(image, [grid_y + displacement_field[1], grid_x + displacement_field[0], grid_z + displacement_field[2]], order=order, cval=cval)
    return transformed_image