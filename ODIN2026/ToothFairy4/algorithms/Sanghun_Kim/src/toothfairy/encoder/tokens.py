"""Patch-token construction helpers for the encoder.

Each patch of a class's RegionBag becomes a token [pooled ; PE(centre) ; log mass]:
pooled feature carries appearance, the 3D sinusoidal PE carries position, log-mass
carries size. Shared by GatedAttention and ToothQueryCrossAttention.
"""
from __future__ import annotations

import torch


def sinusoidal_pe(centers: torch.Tensor, k: int) -> torch.Tensor:
    """(..., 3) normalised coords -> (..., 3*2*k) sinusoidal PE (no grad needed)."""
    if centers.numel() == 0:
        return centers.new_zeros((*centers.shape[:-1], 3 * 2 * k))
    freqs = (2.0 ** torch.arange(k, device=centers.device, dtype=centers.dtype)) * torch.pi
    out = []
    for ax in range(3):
        ang = centers[..., ax:ax + 1] * freqs                 # (...,k)
        out.append(torch.sin(ang))
        out.append(torch.cos(ang))
    return torch.cat(out, dim=-1)


def build_tokens(pooled: torch.Tensor, center: torch.Tensor, mass: torch.Tensor,
                 pe_k: int) -> torch.Tensor:
    """(B,P,feat),(B,P,3),(B,P) -> (B,P, feat + 3*2*pe_k + 1) = [pooled ; PE ; log mass]."""
    pe = sinusoidal_pe(center, pe_k)                          # (B,P,3*2*k)
    logm = torch.log(mass.clamp_min(1e-3)).unsqueeze(-1)      # (B,P,1)
    return torch.cat([pooled, pe, logm], dim=-1)


def token_in_dim(feat_dim: int, pe_k: int) -> int:
    return feat_dim + 3 * 2 * pe_k + 1
