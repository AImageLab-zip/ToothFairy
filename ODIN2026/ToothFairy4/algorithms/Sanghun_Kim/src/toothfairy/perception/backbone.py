"""nnU-Net segmentation backbone loader + feature hooks + pseudo-GT mask.

Perception components: loading the backbone (`build_predictor` — the only loader in this
codebase), the [encoder-stem; decoder-penultimate] feature hooks, and pseudo-GT mask generation.
The backbone spec (model_dir, fold, checkpoint) is passed in as arguments: the dental network
(6 channels) is the default, and the tf2 network (33 channels) has the same nnU-Net structure,
so the same hooks read it.

Feature definitions:
  encoder-stem        = forward hook OUTPUT of encoder.stages[0]         = full-res 32ch
  decoder-penultimate = forward_pre_hook INPUT of decoder.seg_layers[-1] = 32ch right before the
                        final 1x1 seg conv
  → concatenated to 64ch per voxel.
"""
from __future__ import annotations

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def build_predictor(model_dir: str, fold: int = 0,
                    checkpoint_name: str = "checkpoint_final.pth",
                    tile_step_size: float = 0.5,
                    device: str = "cuda") -> nnUNetPredictor:
    """Load a pretrained nnU-Net model folder as a predictor (mirroring OFF for speed)."""
    pred = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(fold,),
                                              checkpoint_name=checkpoint_name)
    # load the single fold's weights into the network once and keep them resident
    net = pred.network
    if hasattr(net, "_orig_mod"):
        net._orig_mod.load_state_dict(pred.list_of_parameters[0])
    else:
        net.load_state_dict(pred.list_of_parameters[0])
    pred.network = net.to(pred.device).eval()
    return pred


def encoder_stem_module(net: torch.nn.Module) -> torch.nn.Module:
    """Encoder stem (stages[0]); its forward OUTPUT is the encoder-stem feature (32ch)."""
    return net.encoder.stages[0]


def decoder_penultimate_module(net: torch.nn.Module) -> torch.nn.Module:
    """Final 1x1 seg conv; its forward_pre_hook INPUT is the decoder-penultimate feature (32ch)."""
    return net.decoder.seg_layers[-1]






