import torch
import math
import numpy as np
from typing import Callable, List, Optional, Sequence, Union
from torch.nn.modules.loss import _Loss
from scipy.ndimage import distance_transform_edt as distance
from monai.utils import LossReduction
from monai.networks import one_hot


from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import Tensor

from packaging import version


from typing import List

import torch
import torch.nn.functional as F


import math
from math import sqrt
from typing import List, Optional, Tuple

import torch


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def gaussian_discrete_erf(window_size: int, sigma) -> torch.Tensor:
    r"""Discrete Gaussian by interpolating the error function. Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    device = sigma.device if isinstance(sigma, torch.Tensor) else None
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=device)
    x = torch.arange(window_size).float() - window_size // 2
    t = 0.70710678 / torch.abs(sigma)
    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss = gauss.clamp(min=0)
    return gauss / gauss.sum()


def _modified_bessel_0(x: torch.Tensor) -> torch.Tensor:
    r"""Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        return 1.0 + y * (
            3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))
        )
    ax = torch.abs(x)
    y = 3.75 / ax
    ans = 0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)))
    coef = 0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans)))
    return (torch.exp(ax) / torch.sqrt(ax)) * coef


def _modified_bessel_1(x: torch.Tensor) -> torch.Tensor:
    r"""adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))
        return torch.abs(x) * (0.5 + y * (0.87890594 + y * ans))
    ax = torch.abs(x)
    y = 3.75 / ax
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
    ans = ans * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n: int, x: torch.Tensor) -> torch.Tensor:
    r"""adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if n < 2:
        raise ValueError("n must be greater than 1.")
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / torch.abs(x)
    ans = torch.tensor(0.0, device=device)
    bip = torch.tensor(0.0, device=device)
    bi = torch.tensor(1.0, device=device)
    m = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans


def gaussian_discrete(window_size, sigma) -> torch.Tensor:
    r"""Discrete Gaussian kernel based on the modified Bessel functions. Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    device = sigma.device if isinstance(sigma, torch.Tensor) else None
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=device)
    sigma2 = sigma * sigma
    tail = int(window_size // 2)
    out_pos: List[Optional[torch.Tensor]] = [None] * (tail + 1)
    out_pos[0] = _modified_bessel_0(sigma2)
    out_pos[1] = _modified_bessel_1(sigma2)
    for k in range(2, len(out_pos)):
        out_pos[k] = _modified_bessel_i(k, sigma2)
    out = out_pos[:0:-1]
    out.extend(out_pos)
    out = torch.stack(out) * torch.exp(sigma2)  # type: ignore
    return out / out.sum()  # type: ignore


def laplacian_1d(window_size) -> torch.Tensor:
    r"""One could also use the Laplacian of Gaussian formula
    to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d: torch.Tensor = filter_1d
    return laplacian_1d


def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.0) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ]
    )


def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_diff_kernel3d(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel: torch.Tensor = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel: torch.Tensor = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


def get_spatial_gradient_kernel3d(mode: str, order: int, device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == 'sobel':
        raise NotImplementedError("Sobel kernel for 3d gradient is not implemented yet")
    if mode == 'diff' and order == 1:
        kernel = get_diff_kernel3d(device, dtype)
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)
    else:
        raise NotImplementedError("")
    return kernel


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_discrete_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients
    based on the modified Bessel functions. Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_discrete_kernel1d(3, 2.5)
        tensor([0.3235, 0.3531, 0.3235])

        >>> get_gaussian_discrete_kernel1d(5, 1.5)
        tensor([0.1096, 0.2323, 0.3161, 0.2323, 0.1096])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d = gaussian_discrete(kernel_size, sigma)
    return window_1d


def get_gaussian_erf_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients by interpolating the error function,
    adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_erf_kernel1d(3, 2.5)
        tensor([0.3245, 0.3511, 0.3245])

        >>> get_gaussian_erf_kernel1d(5, 1.5)
        tensor([0.1226, 0.2331, 0.2887, 0.2331, 0.1226])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d = gaussian_discrete_erf(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
    kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool = False
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma: gaussian standard deviation in the x and y
         direction.
        force_even: overrides requirement for odd kernel size.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def get_laplacian_kernel1d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size: filter size. It should be odd and positive.

    Returns:
        1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples:
        >>> get_laplacian_kernel1d(3)
        tensor([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
        tensor([ 1.,  1., -4.,  1.,  1.])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(f"ksize must be an odd positive integer. Got {kernel_size}")
    window_1d: torch.Tensor = laplacian_1d(kernel_size)
    return window_1d


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter size should be odd.

    Returns:
        2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(f"ksize must be an odd positive integer. Got {kernel_size}")

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


def get_pascal_kernel_2d(kernel_size: int, norm: bool = True) -> torch.Tensor:
    """Generate pascal filter kernel by kernel size.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: True.

    Returns:
        kernel shaped as :math:`(kernel_size, kernel_size)`

    Examples:
    >>> get_pascal_kernel_2d(1)
    tensor([[1.]])
    >>> get_pascal_kernel_2d(4)
    tensor([[0.0156, 0.0469, 0.0469, 0.0156],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0156, 0.0469, 0.0469, 0.0156]])
    >>> get_pascal_kernel_2d(4, norm=False)
    tensor([[1., 3., 3., 1.],
            [3., 9., 9., 3.],
            [3., 9., 9., 3.],
            [1., 3., 3., 1.]])
    """
    a = get_pascal_kernel_1d(kernel_size)

    filt = a[:, None] * a[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt


def get_pascal_kernel_1d(kernel_size: int, norm: bool = False) -> torch.Tensor:
    """Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: False.

    Returns:
        kernel shaped as :math:`(kernel_size,)`

    Examples:
    >>> get_pascal_kernel_1d(1)
    tensor([1.])
    >>> get_pascal_kernel_1d(2)
    tensor([1., 1.])
    >>> get_pascal_kernel_1d(3)
    tensor([1., 2., 1.])
    >>> get_pascal_kernel_1d(4)
    tensor([1., 3., 3., 1.])
    >>> get_pascal_kernel_1d(5)
    tensor([1., 4., 6., 4., 1.])
    >>> get_pascal_kernel_1d(6)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])
    """
    pre: List[float] = []
    cur: List[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.as_tensor(cur)
    if norm:
        out = out / torch.sum(out)
    return out


def get_canny_nms_kernel(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_hysteresis_kernel(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_hanning_kernel1d(kernel_size: int, device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    r"""Returns Hanning (also known as Hann) kernel, used in signal processing and KCF tracker.

    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    See further in numpy docs https://numpy.org/doc/stable/reference/generated/numpy.hanning.html

    Args:
        kernel_size: The size the of the kernel. It should be positive.

    Returns:
        1D tensor with Hanning filter coefficients.
            .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples:
        >>> get_hanning_kernel1d(4)
        tensor([0.0000, 0.7500, 0.7500, 0.0000])
    """
    if not isinstance(kernel_size, int) or kernel_size <= 2:
        raise TypeError(f"ksize must be an positive integer > 2. Got {kernel_size}")

    x: torch.Tensor = torch.arange(kernel_size, device=device, dtype=dtype)
    x = 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / float(kernel_size - 1))
    return x


def get_hanning_kernel2d(kernel_size: Tuple[int, int],
                         device=torch.device('cpu'),
                         dtype=torch.float) -> torch.Tensor:
    r"""Returns 2d Hanning kernel, used in signal processing and KCF tracker.

    Args:
        kernel_size: The size of the kernel for the filter. It should be positive.

    Returns:
        2D tensor with Hanning filter coefficients.
            .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)

    Shape:
        - Output: math:`(\text{kernel_size[0], kernel_size[1]})`

    """
    if kernel_size[0] <= 2 or kernel_size[1] <= 2:
        raise TypeError(f"ksize must be an tuple of positive integers > 2. Got {kernel_size}")
    ky: torch.Tensor = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx: torch.Tensor = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    kernel2d = ky @ kx
    return kernel2d

def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with two 1d kernels, in x and y directions.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel_x: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kW)` or :math:`(B, kW)`.
        kernel_y: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH)` or :math:`(B, kH)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)

        >>> filter2d_separable(input, kernel, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    out_x = filter2d(input, kernel_x.unsqueeze(0), border_type, normalized, padding)
    out = filter2d(out_x, kernel_y.unsqueeze(-1), border_type, normalized, padding)
    return out


def filter3d(
    input: torch.Tensor, kernel: torch.Tensor, border_type: str = 'replicate', normalized: bool = False
) -> torch.Tensor:
    r"""Convolve a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.

    Return:
        the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input border_type is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input border_type is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(kernel)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")

    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError(f"Invalid kernel shape, we expect 1xDxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: List[int] = _compute_padding([depth, height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, d, h, w)

def torch_version() -> str:
    """Parse the `torch.__version__` variable and removes +cu*/cpu."""
    return torch.__version__.split('+')[0]


def torch_version_geq(major, minor) -> bool:
    _version = version.parse(torch_version())
    return _version >= version.parse(f"{major}.{minor}")


if version.parse(torch_version()) > version.parse("1.7.1"):
    # TODO: remove the type: ignore once Python 3.6 is deprecated.
    # It turns out that Pytorch has no attribute `torch.linalg` for
    # Python 3.6 / PyTorch 1.7.0, 1.7.1
    from torch.linalg import solve  # type: ignore
else:
    from torch import solve as _solve

    # NOTE: in previous versions `torch.solve` accepted arguments in another order.
    def solve(A: Tensor, B: Tensor) -> Tensor:
        return _solve(B, A).solution


if version.parse(torch_version()) > version.parse("1.7.1"):
    # TODO: remove the type: ignore once Python 3.6 is deprecated.
    # It turns out that Pytorch has no attribute `torch.linalg` for
    # Python 3.6 / PyTorch 1.7.0, 1.7.1
    from torch.linalg import qr as linalg_qr  # type: ignore
else:
    from torch import qr as linalg_qr  # type: ignore # noqa: F401


# TODO: remove after deprecating < 1.9.1
if version.parse(torch_version()) > version.parse("1.9.1"):

    if not TYPE_CHECKING:

        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors, indexing=indexing)

else:

    if TYPE_CHECKING:

        def torch_meshgrid(tensors: List[Tensor], indexing: Optional[str] = None) -> Tuple[Tensor, ...]:
            return torch.meshgrid(tensors)

    else:

        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors)

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: Tensor = torch.stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    zs: Tensor = torch.linspace(0, depth - 1, depth, device=device, dtype=dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (zs / (depth - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch_meshgrid([zs, xs, ys], indexing="ij"), dim=-1)  # DxWxHx3
    return base_grid.permute(0, 2, 1, 3).unsqueeze(0)  # 1xDxHxWx3

def distance_transform(
    image: torch.Tensor,
    kernel_size: int = 3,
    h: float = 0.35
) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = kornia.contrib.distance_transform(tensor)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))
    grid = create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False,
                           device=image.device, dtype=image.dtype)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type='replicate')
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        #cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        out += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    return out


class DistanceTransform(torch.nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    """
    def __init__(
        self,
        kernel_size: int = 3,
        h: float = 0.35
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # If images have multiple channels, view the channels in the batch dimension to match kernel shape.
        if image.shape[1] > 1:
            image_in = image.view(-1, 1, image.shape[-2], image.shape[-1])
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)

def compute_dtm01(img_gt, out_shape, device=None):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """
    
    normalized_dtm = np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        # ignore background ops choosed before
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)
    normalized_dtm = torch.Tensor(normalized_dtm).float().cuda(device)
    
    #from copy import deepcopy
    #normalized_dtm_npy = deepcopy(normalized_dtm)
    #normal_fuc = lambda x: x/np.max(x)
    #normalized_dtm = np.array([[normal_fuc( distance(posmask.astype(np.bool)) ) \
    #                             if posmask.astype(np.bool).any() \
    #                             else np.zeros(posmask.shape) \
    #                            for posmask in img_gt_b ] \
    #                            for img_gt_b in img_gt])
    """
    def _gen_one_map(ind):
        posmask = img_gt[int(ind/out_shape[1])][ind%out_shape[1]]
        posmask = posmask.astype(np.bool)
        if posmask.any():
            posdis = distance(posmask)
            out = posdis/np.max(posdis)
        else:
            out = np.zeros(posmask.shape)
        return out
    
    normalized_dtm =  Parallel(n_jobs=30, backend='threading')(delayed(_gen_one_map)(x) for x in range(out_shape[0]*out_shape[1]))
    normalized_dtm = torch.Tensor(normalized_dtm).float().cuda(device)
    normalized_dtm = normalized_dtm.view(out_shape)
    """
    #assert(normalized_dtm_npy == normalized_dtm).all()
    return normalized_dtm

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        #ignore background ops choosed before
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def compute_dtm_gpu(img_gt, out_shape, kernel_size = 5):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """
    if len(out_shape) == 5: # B,C,H,W,D
        fg_dtm = torch.cat([distance_transform(1-img_gt[b].float(), kernel_size = kernel_size).unsqueeze(0)\
                            for b in range(out_shape[0])], axis=0)
    else:
        fg_dtm = distance_transform(1-img_gt.float(), kernel_size = kernel_size)
    
    fg_dtm[~torch.isfinite(fg_dtm)] = kernel_size
    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft - gt.float()) ** 2
    s_dtm = seg_dtm ** 2
    g_dtm = gt_dtm ** 2
    dtm = s_dtm + g_dtm
    if len(delta_s.shape) == 5: # B,C,H,W,D
        multipled = torch.einsum('bcxyz, bcxyz->bcxyz', delta_s, dtm)
    elif len(delta_s.shape) == 4: # B,C,H,W
        multipled = torch.einsum('bcxy, bcxy->bcxy', delta_s, dtm)
    else:
        raise RuntimeError("Got Error dim in HD Loss {}".format(delta_s.shape))
    #multipled = multipled.mean()

    return multipled
    

    
class HDLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

               # - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                pass
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                pass
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                pass
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        with torch.no_grad():
            # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
            #gt_dtm = compute_dtm01(target.cpu().numpy(), input.shape, input.device.index)
            gt_dtm = compute_dtm_gpu(target>0.5, input.shape)
            #seg_dtm = compute_dtm01(input.cpu().numpy()>0.5, input.shape, input.device.index)
            seg_dtm = compute_dtm_gpu(input>0.5, input.shape)
        loss_hd = hd_loss(input, target, seg_dtm, gt_dtm)
        if self.reduction == LossReduction.MEAN.value:
            loss_hd = torch.mean(loss_hd)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            loss_hd = torch.sum(loss_hd)  # sum over the batch and channel dims
        #elif self.reduction == LossReduction.NONE.value:
        #    # If we are not computing voxelwise loss components at least
        #    # make sure a none reduction maintains a broadcastable shape
        #    broadcast_shape = list(loss_hd.shape[0:2]) + [1] * (len(input.shape) - 2)
        #    loss_hd = loss_hd.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return loss_hd