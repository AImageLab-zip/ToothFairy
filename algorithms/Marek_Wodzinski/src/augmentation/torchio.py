### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###
from augmentation import aug
from preprocessing import preprocessing_volumetric as pre_vol
from helpers import utils as u

########################



def toothfairy_final_transforms_a():
    random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=5, translation=15, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=10, translation=25, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.5), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.03), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_motion : 1,
        random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transform_6 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5, transform_6])
    return transforms

def toothfairy_final_transforms_b():
    random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=5, translation=15, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=10, translation=20, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.5), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.03), p=0.5)
    transforms = tio.Compose([random_flip, random_motion, random_gamma, random_affine, random_anisotropy, random_noise])
    return transforms




def toothfairy_final_transforms_c():
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=4, translation=5, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=4, translation=10, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.0), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.01), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_motion : 1,
        random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transform_6 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5, transform_6])
    return transforms

def toothfairy_final_transforms_d():
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=4, translation=5, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=4, translation=10, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.0), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.01), p=0.5)
    transforms = tio.Compose([random_flip, random_motion, random_gamma, random_affine, random_anisotropy, random_noise])
    return transforms



def toothfairy_final_transforms_e():
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=4, translation=5, p=0.5)
    # random_gamma = tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=4, translation=10, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.0), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.01), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_motion : 1,
        # random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5])
    return transforms

def toothfairy_final_transforms_f():
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=4, translation=5, p=0.5)
    # random_gamma = tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.9, 1.25), degrees=4, translation=10, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.0), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.01), p=0.5)
    transforms = tio.Compose([random_flip, random_motion, random_affine, random_anisotropy, random_noise])
    return transforms


def toothfairy_final_transforms():
    transform_1 = toothfairy_final_transforms_a()
    transform_2 = toothfairy_final_transforms_b()
    transform_dict = {
        transform_1 : 0.5,
        transform_2 : 0.5,
    }
    transforms = tio.OneOf(transform_dict)
    return transforms




def toothfairy_final_transforms_2():
    transform_1 = toothfairy_final_transforms_c()
    transform_2 = toothfairy_final_transforms_d()
    transform_dict = {
        transform_1 : 0.5,
        transform_2 : 0.5,
    }
    transforms = tio.OneOf(transform_dict)
    return transforms


def toothfairy_final_transforms_3():
    transform_1 = toothfairy_final_transforms_e()
    transform_2 = toothfairy_final_transforms_f()
    transform_dict = {
        transform_1 : 0.5,
        transform_2 : 0.5,
    }
    transforms = tio.OneOf(transform_dict)
    return transforms























