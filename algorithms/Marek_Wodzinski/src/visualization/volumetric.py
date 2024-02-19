### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import io

### External Imports ###
import numpy as np
import torch as tc
import matplotlib
import matplotlib.pyplot as plt

### Internal Imports ###


########################


def show_volume_2d(
    volume : Union[np.ndarray, tc.Tensor],
    spacing : Union[np.ndarray, tc.Tensor, tuple],
    x_slice : int=None,
    y_slice : int=None,
    z_slice : int=None,
    show : bool=True,
    return_buffer : bool=True,
    suptitle : str=None,
    dpi : int=200,
    font_size : int=8):
    """
    Utility function to show 3-D volume as 2-D projections.
    """
    if isinstance(volume, tc.Tensor):
        volume = volume.detach().cpu().numpy()[0, :, :, :]
    y_size, x_size, z_size = volume.shape
    x_slice = int((x_size - 1) / 2) if x_slice is None else x_slice
    y_slice = int((y_size - 1) / 2) if y_slice is None else y_slice
    z_slice = int((z_size - 1) / 2) if z_slice is None else z_slice

    # TODO - set with params
    fig = plt.figure(dpi=dpi)
    font = {'size' : font_size}
    matplotlib.rc('font', **font)
    rows = 1
    cols = 3

    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(np.flip(volume[:, :, z_slice], axis=1), cmap='gray')
    ax.axis('off')
    ax.set_aspect(spacing[0] / spacing[1])
    ax.set_title(f"Z Slice: {z_slice}")
    ax = fig.add_subplot(rows, cols, 2)
    ax.imshow(np.flip(volume[:, x_slice, :].T), cmap='gray')
    ax.set_aspect(spacing[2] / spacing[0])
    ax.axis('off')
    ax.set_title(f"X Slice: {x_slice}")
    ax = fig.add_subplot(rows, cols, 3)
    ax.imshow(np.flip(volume[y_slice, :, :].T), cmap='gray')
    ax.set_aspect(spacing[2] / spacing[1])
    ax.axis('off')
    ax.set_title(f"Y Slice: {y_slice}")
    plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle)
    
    if show:
        plt.show()

    if return_buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf

def show_volumes_2d(
    *volumes : Iterable[Union[np.ndarray, tc.Tensor]],
    spacing : Union[np.ndarray, tc.Tensor, tuple] = (1.0, 1.0, 1.0),
    x_slice : int=None,
    y_slice : int=None,
    z_slice : int=None,
    names: list=None,
    show : bool=True,
    return_buffer : bool=True,
    suptitle : str=None,
    dpi : int=200,
    font_size : int=8):
    """
    Utility function to show several 3-D volumes as 2-D projections.
    """
    number_of_volumes = len(volumes)
    if number_of_volumes == 0:
        raise ValueError("Number of volumes must be >= 1")

    fig = plt.figure(dpi=dpi)
    font = {'size' : font_size}
    matplotlib.rc('font', **font)
    rows = number_of_volumes
    cols = 3

    if names is None:
        names = [""] * number_of_volumes

    for i in range(number_of_volumes):
        volume =  volumes[i]
        if isinstance(volume, tc.Tensor):
            volume = volume.detach().cpu().numpy()[0, :, :, :]
        y_size, x_size, z_size = volume.shape
        x_slice = int((x_size - 1) / 2) if x_slice is None else x_slice
        y_slice = int((y_size - 1) / 2) if y_slice is None else y_slice
        z_slice = int((z_size - 1) / 2) if z_slice is None else z_slice

        ax = fig.add_subplot(rows, cols, i*3 + 1)
        ax.imshow(np.flip(volume[:, :, z_slice], axis=1), cmap='gray')
        ax.axis('off')
        ax.set_aspect(spacing[0] / spacing[1])
        ax.set_title(f"{names[i]} Z Slice: {z_slice}")
        ax = fig.add_subplot(rows, cols, i*3 + 2)
        ax.imshow(np.flip(volume[:, x_slice, :].T), cmap='gray')
        ax.set_aspect(spacing[2] / spacing[0])
        ax.axis('off')
        ax.set_title(f"{names[i]} X Slice: {x_slice}")
        ax = fig.add_subplot(rows, cols, i*3 + 3)
        ax.imshow(np.flip(volume[y_slice, :, :].T), cmap='gray')
        ax.set_aspect(spacing[2] / spacing[1])
        ax.axis('off')
        ax.set_title(f"{names[i]} Y Slice: {y_slice}")
        plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle)
    
    if show:
        plt.show()

    if return_buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf