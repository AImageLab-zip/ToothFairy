### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import torch as tc

### Internal Imports ###
from helpers import utils as u
from input_output import utils_io as uio

default_volumetric_pytorch_load_params = {
'backend': uio.InputOutputBackend.SITK,
'load_as': uio.Representation.PYTORCH,
'dtype': tc.float32,
'load_origin': True,
'load_direction': True
}

default_volumetric_numpy_load_params = {
'backend': uio.InputOutputBackend.SITK,
'load_as': uio.Representation.NUMPY,
'dtype': tc.float32,
'load_origin': True,
'load_direction': True   
}

default_volumetric_save_params = {
'backend': uio.InputOutputBackend.SITK,
'use_compression': True,
'origin': None,
'direction': None,
}

class VolumetricLoader():
    def __init__(self, **loading_params : dict):
        """
        TODO
        """
        self.loading_params = loading_params
        self.backend = loading_params['backend']
        self.load_as = loading_params['load_as']
        self.dtype = loading_params['dtype']

    def load(self, input_path : Union[str, pathlib.Path]):
        match self.backend:
            case uio.InputOutputBackend.SITK:
                self.volume, self.spacing, self.metadata = u.load_volume_sitk(input_path, self.loading_params['load_origin'], self.loading_params['load_direction'])
                match self.load_as:
                    case uio.Representation.NUMPY:
                        self.volume = self.volume.astype(self.dtype)
                        self.spacing = np.array(self.spacing)
                    case uio.Representation.PYTORCH:
                        self.volume = tc.from_numpy(self.volume).type(self.dtype).unsqueeze(0)
                        self.spacing = tc.Tensor(self.spacing)
                    case _:
                        raise ValueError("Unsupported volume representation.")
            case uio.InputOutputBackend.NUMPY:
                self.volume = np.load(input_path).swapaxes(0, 1).swapaxes(1, 2)
                self.spacing = (1.0, 1.0, 1.0)
                self.metadata = {}
                match self.load_as:
                    case uio.Representation.NUMPY:
                        self.volume = self.volume.astype(self.dtype)
                        self.spacing = np.array(self.spacing)
                    case uio.Representation.PYTORCH:
                        self.volume = tc.from_numpy(self.volume).type(self.dtype).unsqueeze(0)
                        self.spacing = tc.Tensor(self.spacing)
                    case _:
                        raise ValueError("Unsupported volume representation.")
            case _:
                raise ValueError("Unsupported backend.")
        return self

class VolumetricSaver():
    """
    TODO
    """
    def __init__(self, **saving_params : dict):
        self.saving_params = saving_params
        self.backend = saving_params['backend']

    def save(self, volume, spacing, save_path):
        match self.backend:
            case uio.InputOutputBackend.SITK:
                if isinstance(volume, tc.Tensor):
                    volume = volume.detach().cpu().numpy()[0, :, :, :]
                if isinstance(spacing, tc.Tensor):
                    spacing = list(spacing.numpy().astype(np.float64))
                u.save_volume_sitk(volume, spacing, save_path,
                use_compression = self.saving_params['use_compression'],
                origin = self.saving_params['origin'],
                direction = self.saving_params['direction'])
            case _:
                raise ValueError("Unsupported backend.")
        return self