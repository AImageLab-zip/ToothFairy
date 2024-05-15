### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable, Callable, Any

### External Imports ###
import torch as tc

### Internal Imports ###

########################

def compose_transforms(*transforms : Iterable[Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]]) -> Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]:
    """
    Composes the input transforms into a single Callable.
    TODO
    """
    def composed_transforms(tensor : tc.Tensor, **kwargs : dict) -> tuple[tc.Tensor, dict]:
        for transform in transforms:
            tensor, kwargs = transform(tensor, **kwargs)
        return tensor, kwargs
    return composed_transforms

def compose_transform_parameters(*transform_parameters : Iterable[dict]) -> dict:
    """
    Composes the dictionaries of transform parameters into a single one (warning: assumes different parameter names for each transform, otherwise overwrites).
    TODO
    """
    composed_parameters = {}
    for i in range(len(transform_parameters)):
        composed_parameters = composed_parameters | transform_parameters[i]
    return composed_parameters

def apply_transform(*args : Iterable[tc.Tensor], transform : Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]]=None, **kwargs : dict) -> tuple[Iterable[tc.Tensor], dict]:
    """
    Applies the input transform into an iterable of tensors.
    TODO
    """
    metadata = dict()
    try:
        metadata['spacing'] = kwargs['new_spacing']
    except:
        pass

    output = [None] * len(args)
    for (i, item) in enumerate(args):
        output[i], kwargs = transform(item, **kwargs)
    return output, metadata