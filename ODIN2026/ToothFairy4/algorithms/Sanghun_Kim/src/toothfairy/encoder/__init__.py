"""Encoder — per-patch region bags -> contextual anatomical tokens.

Parts:
  tokens         — sinusoidal_pe, build_tokens, token_in_dim (patch-token helpers)
  gated_attention — GatedAttention (MIL pooler for the non-tooth regions)
  tooth_query    — ToothQueryCrossAttention (teeth: 16 FDI queries + CLS, arch-separated)
  cross_label    — CrossLabelEncoder (token-id/presence + self-attention)
  region_encoder — RegionEncoder (assembles the above -> contextual tokens + presence),
                   FDI_POS_IN_STACK
"""
from .tokens import sinusoidal_pe, build_tokens, token_in_dim  # noqa: F401
from .gated_attention import GatedAttention  # noqa: F401
from .tooth_query import ToothQueryCrossAttention  # noqa: F401
from .cross_label import CrossLabelEncoder  # noqa: F401
from .region_encoder import (  # noqa: F401
    RegionEncoder, FDI_POS_IN_STACK,
    NONTEETH_SLOT, UPCLS_SLOT, LOCLS_SLOT, V1_TEETH_SLOTS, FDI_TO_SLOT, FDI_TO_ROW,
)
