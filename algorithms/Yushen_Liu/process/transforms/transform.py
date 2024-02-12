import abc
from warnings import warn
from typing import List, Optional

import torch
import random
import numpy as np
from torchvision import transforms

from transforms.functional import random_contrast, random_brightness_multiplicative, random_gamma, \
    random_gaussian_noise, random_gaussian_blur, in_painting, out_painting


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"

        return ret_str


class ColorJitter(AbstractTransform):
    """Randomly change the brightness, contrast, and gamma.
    Args:

    """
    def __init__(self, brightness=(0.8, 1.2), contrast=(0.7, 1.3),
                 gamma=(0.5, 1.5), p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.gamma = self._check_input(gamma, 'gamma')
        self.p = p

    def _check_input(self, parameters, name):
        if parameters is None:
            parameters = (0.5, 2)
        elif len(parameters) != 2:
            raise ValueError(f'the length of {name} parameter must be two!')

        return parameters

    @staticmethod
    def get_params(parameters):
        if np.random.random() < 0.5 and parameters[0] < 1:
            para = np.random.uniform(parameters[0], 1)
        else:
            para = np.random.uniform(max(parameters[0], 1), parameters[1])

        return para

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample, None
        fn_idx = torch.randperm(3)
        for fn_id in fn_idx:
            if fn_id == 0:
                image = random_contrast(image, contrast_range=self.contrast, preserve_range=True, p=1.0)
            elif fn_id == 1:
                image = random_brightness_multiplicative(image, multiplier_range=self.brightness, p=1.0)
            elif fn_id == 2:
                image = random_gamma(image, gamma_range=self.gamma, p=1.0)

        if isinstance(sample, dict):
            return {'image': image, 'label': label}
        else:
            return image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', gamma={0}'.format(self.gamma)

        return format_string


class NoiseJitter(AbstractTransform):
    """Randomly change the gaussian noise and blur.
    Args:

    """
    def __init__(self, noise_sigma=(0, 0.2), blur_sigma=(0.5, 1.0), p=0.5):
        self.noise_sigma = noise_sigma
        self.blur_sigma = blur_sigma
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample, None

        if np.random.random() < 0.5:
            image = random_gaussian_noise(image, self.noise_sigma, p=1.0)
        else:
            image = random_gaussian_blur(image, self.blur_sigma, p=1.0)

        if isinstance(sample, dict):
            return {'image': image, 'label': label}
        else:
            return image


class PaintingJitter(AbstractTransform):
    def __init__(self, cutout_range=(0.1, 0.2), retain_range=(0.8, 0.9), cnt=3, p=0.5):
        self.cutout_range = cutout_range
        self.retain_range = retain_range
        self.cnt = cnt
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample, None

        if np.random.random() < 0.5:
            image = in_painting(image, cutout_range=self.cutout_range, cnt=self.cnt, p=1.0)
        else:
            image = out_painting(image, retain_range=self.retain_range, cnt=self.cnt, p=1.0)

        if isinstance(sample, dict):
            return {'image': image, 'label': label}
        else:
            return image