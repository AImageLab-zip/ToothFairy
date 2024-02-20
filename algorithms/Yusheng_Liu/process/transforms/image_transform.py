import numpy as np
from typing import Optional
import skimage.exposure as sk_exposure


__all__ = [
    'voxel_coord_2_world',
    'world_2_voxel_coord',
    'clip_image',
    'normalize_mean_std',
    'normalize_min_max_and_clip',
    'clip_and_normalize_mean_std']


def voxel_coord_2_world(voxel_coord: list, origin: list, spacing: list, directions: Optional[list] = None):
    if directions is None:
        directions = np.array([1]*len(voxel_coord))
    stretched_voxel_coord = np.array(voxel_coord) * np.array(spacing).astype(float)
    world_coord = np.array(origin) + stretched_voxel_coord * np.array(directions)

    return world_coord


def world_2_voxel_coord(world_coord: list, origin: list, spacing: list):
    stretched_voxel_coord = np.array(world_coord) - np.array(origin)
    voxel_coord = np.absolute(stretched_voxel_coord / np.array(spacing))

    return voxel_coord


def clip_image(image: np.ndarray, min_window: float = -1200.0, max_window: float = 600.0):
    return np.clip(image, min_window, max_window)


def normalize_min_max_and_clip(image: np.ndarray, min_window: float = -1200.0, max_window: float = 600.0):
    """
    Normalize image HU value to [-1, 1] using window of [min_window, max_window].
    """
    image = (image - min_window) / (max_window - min_window)
    image = image * 2 - 1.0
    image = image.clip(-1, 1)
    return image


def normalize_mean_std(image: np.ndarray, global_mean: Optional[float] = None, global_std: Optional[float] = None):
    """
    Normalize image by (voxel - mean) / std, the operate should be local or global normalization.
    """
    if not global_mean or not global_std:
        mean = np.mean(image)
        std = np.std(image)
    else:
        mean, std = global_mean, global_std

    image = (image - mean) / (std + 1e-5)
    return image


def clip_and_normalize_mean_std(image: np.ndarray, min_window: float = -1200, max_window: float = 600):
    """
    Clip image in a range of [min_window, max_window] in HU values.
    """
    image = np.clip(image, min_window, max_window)
    mean = np.mean(image)
    std = np.std(image)

    image = (image - mean) / (std + 1e-5)

    return image


def equalize_adapthist(array: np.ndarray):
    num_slices = array.shape[0]
    out_image = []
    for i in range(num_slices):
        image = array[i]
        if image.any() != 0:
            image = (image - image.min()) / (image.max() - image.min())
            image = sk_exposure.equalize_adapthist(image)
        out_image.append(image)
    out_image = np.stack(out_image, axis=0)

    return out_image


def equalize_adapthist_and_normalize(array: np.ndarray):
    num_slices = array.shape[0]
    out_image = []
    for i in range(num_slices):
        image = array[i]
        if image.any() != 0:
            image = (image - image.min()) / (image.max() - image.min())
            image = sk_exposure.equalize_adapthist(image)
        out_image.append(image)
    out_image = np.stack(out_image, axis=0)
    out_image = (out_image - np.mean(out_image)) / (np.std(out_image) + 1e-5)

    return out_image