"""Checkpoint reconstruction for released EviDent-CBCT Stage 2 branches."""

from __future__ import annotations

import pydoc
from pathlib import Path

import torch


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Reconstruct a released branch model from its nnU-Net checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    plans = checkpoint["plans"]
    configuration = checkpoint.get("configuration", "3d_fullres")
    architecture = plans["configurations"][configuration]["architecture"]
    kwargs = dict(architecture["arch_kwargs"])
    for key in architecture.get("_kw_requires_import", []):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = pydoc.locate(kwargs[key])

    model_class = pydoc.locate(architecture["network_class_name"])
    if model_class is None:
        raise ImportError(architecture["network_class_name"])
    valid = getattr(model_class, "allowed_init_kwargs", set(kwargs))
    model = model_class(**{key: value for key, value in kwargs.items() if key in valid}).to(device)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    model.eval()
    return model, kwargs
