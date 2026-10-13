"""Loss functions for the report generation pipeline.

Parts:
  findings    — focal CE/BCE + per-tooth and non-teeth categorical claim loss
  contrastive — region-text SigLIP sigmoid contrastive alignment
"""
from .findings import (  # noqa: F401
    focal_ce, focal_bce, teeth_gate_loss, region_axes_loss, gate_loss, axis_soft_probs,
)
from .contrastive import FrozenTextEncoder, ContrastiveHead  # noqa: F401
