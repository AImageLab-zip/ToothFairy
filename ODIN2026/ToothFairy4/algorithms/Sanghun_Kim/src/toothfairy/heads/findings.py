"""FindingsHead — contextual tokens -> per-element categorical claim logits.

Per-tooth 10-axis claims (aux_head, ctx[FDI_POS] -> (B,32,AUX_DIM)) plus the non-tooth region
heads (maxilla, mandible, nerve [shared by right/left]). Categorical claim supervision and report
generation use this same head (when the findings head is enabled).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..schema.claims import region_aux_dim
from ..encoder.region_encoder import FDI_POS_IN_STACK


class FindingsHead(nn.Module):
    """Per-tooth + non-teeth categorical claim heads on the [38,d_model] token stack.

    There is no arch head: perio_stage moved onto the maxilla/mandible axes, so the two arch
    summary tokens (slots 20 and 37) are not claim targets. The CLS queries themselves stay in
    ToothQueryCrossAttention and keep serving as arch context for the cross-label self-attention.
    The input stack is therefore still 38 anatomical tokens while only 36 carry claims.
    """

    def __init__(self, d_model: int):
        super().__init__()
        from ..schema.claims import AXES, MAXILLA_AXES, MANDIBLE_AXES, NERVE_AXES
        self.aux_head = nn.Linear(d_model, sum(a.logit_dim for a in AXES))       # 29
        self.maxilla_head = nn.Linear(d_model, region_aux_dim(MAXILLA_AXES))     # 22
        self.mandible_head = nn.Linear(d_model, region_aux_dim(MANDIBLE_AXES))   # 27
        self.nerve_head = nn.Linear(d_model, region_aux_dim(NERVE_AXES))         # 7  (R/L shared)

    def teeth_logits(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx (B,N,d) -> (B,32,AUX_DIM) per-FDI claim logits."""
        return self.aux_head(ctx[:, FDI_POS_IN_STACK, :])

    def nonteeth_logits(self, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        """ctx (B,N,d) -> {maxilla (B,·), mandible (B,·), nerve (B,2,·)}.
        nerve slots 2/3 (R/L) share nerve_head. The arch summary slots carry no claims."""
        return {
            "maxilla": self.maxilla_head(ctx[:, 0]),
            "mandible": self.mandible_head(ctx[:, 1]),
            "nerve": torch.stack([self.nerve_head(ctx[:, 2]),
                                  self.nerve_head(ctx[:, 3])], dim=1),          # (B,2,7)
        }
