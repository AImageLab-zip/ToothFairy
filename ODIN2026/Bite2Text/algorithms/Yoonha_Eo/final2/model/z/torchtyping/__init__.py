"""Inference-only torchtyping compatibility."""
import torch
class TensorType:
    def __class_getitem__(cls, _parameters):
        return torch.Tensor
