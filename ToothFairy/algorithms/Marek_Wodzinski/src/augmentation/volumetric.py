### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import torch as tc
import torchio

### Internal Imports ###
from preprocessing import preprocessing_volumetric as pre_vol


########################


def generate_resampling_transform(old_spacing : tuple, new_spacing : tuple, mode : str) -> Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]:
    """
    Generates transform that resamples the input binary 3-D volume to given spacing.
    TODO
    """
    def inner(tensor : tc.Tensor, **kwargs : dict) -> tuple[tc.Tensor, dict]:
        try:
            olds = kwargs['old_spacing']
            news = kwargs['new_spacing']
            int_mode = kwargs['mode']
        except:
            olds = old_spacing
            news = new_spacing
            int_mode = mode
        return pre_vol.resample_to_spacing_tc(tensor.unsqueeze(0), olds, news, mode=int_mode)[0], kwargs
    return inner

def generate_resampling_to_shape_transform(old_shape : tuple, new_shape : tuple) -> Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]:
    """
    Generates transform that resamples the input binary 3-D volume to given shape.
    TODO
    """
    def inner(tensor : tc.Tensor, **kwargs : dict) -> tuple[tc.Tensor, dict]:
        try:
            olds = kwargs['old_shape']
            news = kwargs['new_shape']
        except:
            olds = old_shape
            news = new_shape
        return pre_vol.resample_tensor(tensor.unsqueeze(0), (tensor.shape[0], tensor.shape[1]) + tuple(news), mode='nearest')[0], kwargs
    return inner

def generate_flip_transform(axes : Union[int, tuple[int]]= 0, probability : float = 0.5) -> Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]:
    """
    Generates random flip transform for 3-D volumes.
    """
    transforms = torchio.RandomFlip(axes=axes, flip_probability=probability)
    def inner(tensor : tc.Tensor, **kwargs : dict) -> tuple[tc.Tensor, dict]:
        try:
            transform = kwargs['flip_transform']
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=tensor))
            result = transform(subject)
            result = result['tensor'].data
        except:
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=tensor))
            result = transforms(subject)
            to_reproduce = result.get_composed_history()
            result = result['tensor'].data
            kwargs['flip_transform'] = to_reproduce
        return result, kwargs
    return inner

def generate_affine_transform(scales : tuple=(0.98, 1.02), degrees : tuple=(-5, 5), translation : tuple=(-3, 3), probability : float = 0.5) -> Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]:
    """
    Generates random affine transform for binary 3-D volumes.
    TODO
    """
    transforms = torchio.RandomAffine(scales=scales, degrees=degrees, translation=translation, image_interpolation='nearest', p=probability)
    def inner(tensor : tc.Tensor, **kwargs : dict) -> tuple[tc.Tensor, dict]:
        try:
            transform = kwargs['affine_transform']
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=tensor))
            result = transform(subject)
            result = result['tensor'].data
        except:
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=tensor))
            result = transforms(subject)
            to_reproduce = result.get_composed_history()
            result = result['tensor'].data
            kwargs['affine_transform'] = to_reproduce
        return result, kwargs
    return inner