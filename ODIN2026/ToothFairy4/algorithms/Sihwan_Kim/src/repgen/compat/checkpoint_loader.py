"""Inference-only compatibility alias for the submitted checkpoint loader."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_R142B_FPGuard_500ep(nnUNetTrainer):
    """Use the standard plans-driven network builder during inference."""
