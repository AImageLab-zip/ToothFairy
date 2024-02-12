
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Common.transforms.mask_one_hot import one_hot


__all__ = ['ItkResample', 'ScipyResample', 'TensorResample']


class TensorResample(object):
    """Resample image or mask  in tensor to target size"""
    def __init__(self):
        super(TensorResample, self).__init__()

    @staticmethod
    def resample_image(image: torch.Tensor, out_size: List[int], dim: int = 3) -> torch.Tensor:
        """
        Resample image to target size.

        Args:
        :param image: image tensor
        :param out_size: target image size
        :param dim: 2 or 3
        :return:

        Examples:
        a = torch.randint(0, 3, (8, 8, 8))
        print(a.size())
        print(a[0])
        b = TensorResample.resample_image(a, [16, 16, 16], dim=3)
        print(b.size())
        print(b[0])
        """
        if dim not in (2, 3):
            raise TypeError(f'The dim of resample image must be 2 or 3, while the input {dim} dim!')

        assert (dim == 2 and len(image) <= 4) or \
               dim == 3 and len(image) <= 5

        image = image.float()
        length = image.dim()
        expand_time = 4 - length if dim == 2 else 5 - length
        for _ in range(expand_time):
            image = image.unsqueeze(0)
        if dim == 2:
            image = F.interpolate(image, size=out_size, mode='bilinear', align_corners=True)
        else:
            image = F.interpolate(image, size=out_size, mode='trilinear', align_corners=True)
        for _ in range(expand_time):
            image = image.squeeze(0)

        return image

    @staticmethod
    def resample_mask(mask: torch.Tensor, out_size: List[int], dim: int = 3, mode: str = 'nearest') -> torch.Tensor:
        """
        Resample mask to target size

        :param mask: mask tensor
        :param out_size: target size
        :param dim: 2 or 3
        :param mode: 'nearest', 'bilinear'(2D), 'trilinear'(3D)
        :return:

        Examples:
        a = torch.randint(0, 3, (8, 8, 8))
        print(a.size())
        print(a[0])
        b = TensorResample.resample_image(a, [16, 16, 16], dim=3)
        print(b.size())
        print(b[0])
        """
        if mode not in ['nearest', 'bilinear', 'trilinear']:
            raise TypeError(f'The mode of resample mask must be nearest/bilinear/trilinear!')
        if dim not in [2, 3]:
            raise TypeError(f'The dim of resample mask must be 2 or 3, while the input {dim} dim!')

        assert (dim == 2 and len(mask) <= 4 and mode != 'trilinear') or \
               dim == 3 and len(mask) <= 5 and mode != 'bilinear'

        mask = mask.float()
        length = mask.dim()
        expand_time = 4 - length if dim == 2 else 5 - length
        for _ in range(expand_time):
            mask = mask.unsqueeze(0)
        if mode == 'bilinear' or mode == 'trilinear':
            mask = one_hot(mask, max(2, int(torch.max(mask))+1), dim=1)
            mask = F.interpolate(mask, size=out_size, mode=mode, align_corners=True)
            mask = torch.where(mask >= 0.5, torch.tensor(1).to(mask.device), torch.tensor(0).to(mask.device))
            mask = torch.argmax(mask, dim=1)
            expand_time -= 1
        else:
            mask = F.interpolate(mask, size=out_size, mode=mode)
        for _ in range(expand_time):
            mask = mask.squeeze(0)

        return mask


class ItkResample(object):
    def __init__(self):
        super(ItkResample, self).__init__()

    def resample_to_spacing(self, itk_image,
                            target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.0):
        source_spacing = itk_image.GetSpacing()
        source_size = itk_image.GetSize()

        zoom_factor = np.divide(source_spacing, target_spacing)
        offset = self._calculate_origin_offset(target_spacing, source_spacing)

        target_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, source_size), decimals=5)), dtype=np.int16)
        reference_image = self._sitk_new_blank_image(size=target_size, spacing=target_spacing,
                                                     direction=itk_image.GetDirection(),
                                                     origin=itk_image.GetOrigin() + offset,
                                                     default_value=default_value)
        return self._sitk_resample_to_image(itk_image, reference_image,
                                            interpolator=interpolator, default_value=default_value)

    def resample_to_size(self, itk_image, target_size, interpolator=sitk.sitkLinear, default_value=0.0):
        source_spacing = itk_image.GetSpacing()
        source_size = itk_image.GetSize()

        target_spacing = np.empty(3)
        for i in range(3):
            target_spacing[i] = source_size[i] * 1.0 * source_spacing[i] / target_size[i]

        blank_image = self._sitk_new_blank_image(size=target_size, spacing=target_spacing,
                                                 direction=itk_image.GetDirection(),
                                                 origin=itk_image.GetOrigin(),
                                                 default_value=default_value)
        return self._sitk_resample_to_image(itk_image, blank_image,
                                            interpolator=interpolator, default_value=default_value)

    def _calculate_origin_offset(self, target_spacing, source_spacing):
        return np.subtract(target_spacing, source_spacing) / 2.0

    def _sitk_new_blank_image(self, size, spacing, direction, origin, default_value=0.0):
        itk_image = sitk.GetImageFromArray(np.ones(size, dtype=np.float32).T * default_value)
        itk_image.SetSpacing(spacing)
        itk_image.SetDirection(direction)
        itk_image.SetOrigin(origin)
        return itk_image

    def _sitk_resample_to_image(self, itk_image, reference_image, interpolator=sitk.sitkLinear, default_value=0.,
                                transform=None, output_pixel_type=None):
        if transform is None:
            transform = sitk.Transform()
            transform.SetIdentity()
        if output_pixel_type is None:
            output_pixel_type = itk_image.GetPixelID()
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetTransform(transform)
        resample_filter.SetOutputPixelType(output_pixel_type)
        resample_filter.SetDefaultPixelValue(default_value)
        resample_filter.SetReferenceImage(reference_image)
        return resample_filter.Execute(itk_image)


class ScipyResample(object):
    def __init__(self):
        super(ScipyResample, self).__init__()

    @staticmethod
    def resample_to_spacing(npy_image, source_spacing, target_spacing, order=1):
        scale = np.array(source_spacing) / np.array(target_spacing)
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_npy_image = zoom(npy_image, scale, order=order)
        return target_npy_image, zoom_factor, scale

    @staticmethod
    def resample_to_size(npy_image, target_size, order=1):
        source_size = npy_image.shape
        scale = np.array(target_size) / source_size
        zoom_factor = source_size / np.array(target_size)
        target_npy_image = zoom(npy_image, scale, order=order)
        return target_npy_image, zoom_factor

    @staticmethod
    def resample_mask_to_spacing(npy_mask, source_spacing, target_spacing, num_label, order=1):
        scale = np.array(source_spacing) / np.array(target_spacing)
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        for i in range(1, num_label + 1):
            current_mask = npy_mask.copy()

            current_mask[current_mask != i] = 0
            current_mask[current_mask == i] = 1

            current_mask = zoom(current_mask, scale, order=order)
            current_mask = (current_mask > 0.5).astype(np.uint8)
            target_npy_mask[current_mask != 0] = i
        return target_npy_mask, zoom_factor, scale

    @staticmethod
    def nearest_resample_mask(npy_mask, target_size):
        source_size = npy_mask.shape
        scale = np.array(target_size) / source_size
        target_npy_mask = zoom(npy_mask, scale, order=0)

        return target_npy_mask

    @staticmethod
    def resample_mask_to_size(npy_mask, target_size, num_label, order=1):
        source_size = npy_mask.shape
        scale = np.array(target_size) / source_size
        zoom_factor = source_size / np.array(target_size)
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        for i in range(1, num_label + 1):
            current_mask = npy_mask.copy()

            current_mask[current_mask != i] = 0
            current_mask[current_mask == i] = 1

            current_mask = zoom(current_mask, scale, order=order)
            current_mask = (current_mask > 0.5).astype(np.uint8)
            target_npy_mask[current_mask != 0] = i
        return target_npy_mask, zoom_factor


if __name__ == '__main__':
    t_tensor = torch.randint(0, 3, (4, 4, 4))
    print(t_tensor[0])
    print(t_tensor.size())
    resample = TensorResample()
    b_tensor = resample.resample_mask(t_tensor, [8, 8, 8], dim=3)
    print(b_tensor[0])
    print(b_tensor.size())
    resample = ScipyResample()
    c_array = resample.nearest_resample_mask(t_tensor.numpy(), [8, 8, 8])
    print(c_array[0])
