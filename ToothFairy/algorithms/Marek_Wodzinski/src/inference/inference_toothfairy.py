### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib
import math

### External Imports ###
import numpy as np
import torch as tc
import torch.nn.functional as F
import SimpleITK as sitk
import skimage.measure as measure
import torchio as tio
import scipy.ndimage as nd
from skimage.filters import threshold_otsu

### Internal Imports ###
from paths import paths as p
from preprocessing import preprocessing_volumetric as pre_vol
from networks import runet
from helpers import utils as u

########################


def default_single_step_inference_params(checkpoint_path):
    config = {}
    device = "cuda:0"
    network_config = runet.default_config()
    model = runet.RUNet(**network_config).to(device)
    checkpoint = tc.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)
    echo = True
    threshold = 0.5
    config['device'] = device
    config['output_size'] =  (256, 256, 256)
    config['model'] = model
    config['echo'] = echo
    config['postprocess'] = True
    config['threshold'] = threshold
    return config


def single_step_inference(volume : np.ndarray, **params):
    device = params['device']
    model = params['model']
    output_size = params['output_size']
    echo = params['echo']
    threshold = params['threshold']
    with tc.set_grad_enabled(False):
        volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        volume_tc = u.normalize(volume_tc)
        print(f"Input shape: {volume_tc.shape}") if echo else None
        original_shape = volume_tc.shape
        volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
        print(f"Resampled input shape: {volume_tc.shape}") if echo else None
        output_tc = model(volume_tc)
        print(f"Output shape: {output_tc.shape}") if echo else None
        output_tc = pre_vol.resample_tensor(output_tc, original_shape, mode='bilinear')
        print(f"Resampled output shape: {output_tc.shape}") if echo else None
        output = (tc.sigmoid(output_tc[0, 0, :, :, :]) > threshold).detach().cpu().numpy()
    return output



def run_inference(input_path, inference_method, inference_method_params, ground_truth_path=None, output_path=None):
    threshold = inference_method_params['threshold']
    echo = inference_method_params['echo']
    volume = sitk.ReadImage(input_path)
    spacing = volume.GetSpacing()
    direction = volume.GetDirection()
    origin = volume.GetOrigin()
    volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
    if ground_truth_path is not None:
        ground_truth = sitk.ReadImage(ground_truth_path)
        ground_truth = sitk.GetArrayFromImage(ground_truth).swapaxes(0, 1).swapaxes(1, 2)
    output = inference_method(volume, spacing=spacing, **inference_method_params)
    if threshold is None:
        threshold = threshold_otsu(output)
        output = output > threshold
    if inference_method_params['postprocess']:
        labels = measure.label(output)
        unique, counts = np.unique(labels, return_counts=True)
        threshold = 50
        unique_reduced = unique[counts < threshold]
        for unique_val in unique_reduced:
            output[labels == unique_val] = 0.0
    if output_path is not None:
        to_save = sitk.GetImageFromArray(output.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(spacing)
        to_save.SetDirection(direction)
        to_save.SetOrigin(origin)
        sitk.WriteImage(to_save, str(output_path), useCompression=True)
    return output


def run_inference_direct(volume, inference_method, inference_method_params):
    threshold = inference_method_params['threshold']
    output = inference_method(volume, **inference_method_params)
    if threshold is None:
        threshold = threshold_otsu(output)
        output = output > threshold
    if inference_method_params['postprocess']:
        labels = measure.label(output)
        unique, counts = np.unique(labels, return_counts=True)
        threshold = 50
        unique_reduced = unique[counts < threshold]
        for unique_val in unique_reduced:
            output[labels == unique_val] = 0.0
    return output