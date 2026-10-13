"""Gated attention MIL pooler for the non-teeth regions (maxilla/mandible/canal).

token x_p=[pooled;PE;log m] -> value z_p=GELU(Linear(x_p)) ∈ R^d_model;
gate scores z_p (V/SiLU/w); v=LayerNorm(Σ a_p·z_p). Value is the nonlinear z_p (not
raw pooled) so PE/log-mass reach the output and the sinusoidal-PE cancellation a linear
value would suffer is avoided. Concentrates onto the lesion patch instead of the
FOV-dependent mean that washes texture out (seg-probe gap 0.125). Every pooler emits
d_model so all anatomic tokens share one space for the cross-label self-attention; absent
class -> zero vector + presence=0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tokens import build_tokens, token_in_dim


class GatedAttention(nn.Module):
    """SiLU-gated attention MIL pooler, shared across the non-teeth classes.

    Returns (v, present): v (B,d_model), present (B,) = any valid patch.
    """

    def __init__(self, feat_dim: int, pe_k: int, d_model: int, gate_hidden: int,
                 dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.pe_k = pe_k
        self.d_model = d_model
        in_dim = token_in_dim(feat_dim, pe_k)
        self.value = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU())
        self.V = nn.Linear(d_model, gate_hidden)
        self.w = nn.Linear(gate_hidden, 1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, pooled: torch.Tensor, center: torch.Tensor, mass: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pooled (B,P,feat), mask (B,P) True=valid
        x = build_tokens(pooled, center, mass, self.pe_k)     # (B,P,in_dim)
        z = self.value(x)                                     # (B,P,d_model)
        e = self.w(self.drop(torch.nn.functional.silu(self.V(z)))).squeeze(-1)  # (B,P)
        # large-negative (NOT -inf) so an all-masked row's softmax backward isn't nan
        e = e.masked_fill(~mask, -1e9)
        a = torch.softmax(e, dim=1)                           # (B,P)
        v = torch.einsum("bp,bpf->bf", a, z)                 # (B,d_model)
        present = mask.any(dim=1)
        # LayerNorm first, then mask: absent row -> norm(0)*0 = 0 (presence flag stays 0)
        v = self.norm(v) * present.unsqueeze(-1).float()     # absent -> zero
        return v, present
