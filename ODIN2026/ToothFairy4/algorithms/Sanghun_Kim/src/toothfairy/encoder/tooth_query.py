"""Tooth-query cross-attention for teeth.

16 shared FDI queries + 1 CLS query, + arch embedding, W_q/W_k/W_v +
queries SHARED across arches, run SEPARATELY per arch -> 32 FDI + 2 CLS context vectors
in R^d_model. Output stays d_model (no dimension reduction — the cross-label 1-layer
fuse handles it); the classification head on top is the findings head (heads/findings.py).

hybrid-dental-tf2 (use_tf2=True) extends the teeth K/V: K/V = [T dental patches] union
[the 16 tf2 per-tooth vectors of that arch]. Masking makes **FDI query i attend to the
T dental patches plus its own tf2 vector only (= T+1)**, while the **CLS query attends to
the T dental patches only**. When tf2 fails to detect a tooth (present=False), a
**learnable absent vector** takes that slot, so the model can learn "tf2 did not see it"
apart from "the tooth is not there". tf2 vectors get their **own in_proj**, separate from
dental, because the two backbones have different feature distributions. The dental patch
bag is left untouched, so it recovers teeth that tf2 missed as a fallback.
With use_tf2=False this block is **structurally and numerically identical** to the
dental-only variant (the tf2 modules are not created; the None path is bit-identical).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tokens import build_tokens, token_in_dim

N_TEETH_PER_ARCH = 16    # tf2 per-tooth vectors per arch (= number of FDI queries)


class ToothQueryCrossAttention(nn.Module):
    """16 shared FDI queries + 1 CLS, arch embedding, arch-separated cross-attention.

    With use_tf2=True the 16 tf2 per-tooth vectors are concatenated onto the per-arch K/V
    and a per-query mask is applied.
    """

    def __init__(self, feat_dim: int, pe_k: int, d_model: int, n_heads: int,
                 dropout: float = 0.1, use_tf2: bool = False, use_tf2_density: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.pe_k = pe_k
        self.d = d_model
        self.h = n_heads
        self.use_tf2 = use_tf2
        self.use_tf2_density = use_tf2_density and use_tf2
        in_dim = token_in_dim(feat_dim, pe_k)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.query = nn.Parameter(torch.randn(17, d_model) * 0.02)    # 16 FDI + 1 CLS
        self.arch_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)  # upper / lower
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # no dimension reduction: teeth tokens stay d_model (cross-label 1-layer fuse handles it)
        if use_tf2:
            # tf2 per-tooth in_proj (separate from the dental in_proj; the two backbones have
            # different feature distributions) + learnable absent vector (d_model space, i.e.
            # after in_proj; inserted wherever present=False).
            self.tf2_in_proj = nn.Linear(in_dim, d_model)   # shared by whole + lo/hi/sat
            self.tf2_absent = nn.Parameter(torch.randn(d_model) * 0.02)
            if self.use_tf2_density:
                # Inject the density sub-pools (lo/hi/sat) as extra diagonal K/V tokens, exactly
                # as whole is injected. One learnable absent per band (an empty band has mass 0).
                self.lo_absent = nn.Parameter(torch.randn(d_model) * 0.02)
                self.hi_absent = nn.Parameter(torch.randn(d_model) * 0.02)
                self.sat_absent = nn.Parameter(torch.randn(d_model) * 0.02)

    def _kv_mask(self, dental_mask: torch.Tensor, T: int, n_blocks: int = 1) -> torch.Tensor:
        """(B,17,T+16*n_blocks) attention mask. Dental columns: every query attends as
        dental_mask says. Within each tf2 block, FDI query i attends to column i of that block
        only (the CLS row, 16, never attends to tf2).
        n_blocks = 1 (whole only) or 4 (whole + lo/hi/sat density sub-pools)."""
        B, dev = dental_mask.shape[0], dental_mask.device
        dental_part = dental_mask[:, None, :].expand(B, 17, T)           # (B,17,T)
        pat = torch.zeros(17, N_TEETH_PER_ARCH, dtype=torch.bool, device=dev)
        idx = torch.arange(N_TEETH_PER_ARCH, device=dev)
        pat[idx, idx] = True                                            # FDI query i -> block col i
        block = pat[None].expand(B, -1, -1)                            # (B,17,16) diagonal
        return torch.cat([dental_part] + [block] * n_blocks, dim=2)     # (B,17,T+16*n_blocks)

    def _attend(self, pooled, center, mass, mask, arch_id: int, tf2: dict | None = None):
        """-> ctx (B,17,d_model), present (B,). present follows the dental mask (unchanged)."""
        B, T = pooled.shape[0], pooled.shape[1]
        tok = build_tokens(pooled, center, mass, self.pe_k)          # (B,T,in_dim)
        kv = self.in_proj(tok)                                       # (B,T,d) dental
        q = self.query[None] + self.arch_emb[arch_id][None, None]    # (1,17,d)
        q = q.expand(B, -1, -1)
        Q = self.Wq(q).view(B, 17, self.h, self.d // self.h).transpose(1, 2)
        if tf2 is None:                                              # dental-only: bit-identical
            K = self.Wk(kv).view(B, T, self.h, self.d // self.h).transpose(1, 2)
            V = self.Wv(kv).view(B, T, self.h, self.d // self.h).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / (self.d // self.h) ** 0.5   # (B,h,17,T)
            scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        else:                                                        # hybrid: tf2 blocks on K/V
            def _block(pooled, center, mass, present, absent):
                """One tf2 sub-pool -> (B,16,d) K/V block. Shared tf2_in_proj; empty slots take
                the learnable absent vector."""
                tok = build_tokens(pooled, center, mass, self.pe_k)  # (B,16,in_dim)
                kvb = self.tf2_in_proj(tok)                          # (B,16,d) shared in_proj
                return torch.where(present.unsqueeze(-1), kvb, absent.to(kvb.dtype))
            blocks = [_block(tf2["pooled"], tf2["center"], tf2["mass"],
                             tf2["present"], self.tf2_absent)]        # whole (present=detected)
            if self.use_tf2_density:                                 # + density (mass 0 -> absent)
                blocks.append(_block(tf2["lo_pooled"], tf2["lo_center"], tf2["lo_mass"],
                                     tf2["lo_mass"] > 0, self.lo_absent))
                blocks.append(_block(tf2["hi_pooled"], tf2["hi_center"], tf2["hi_mass"],
                                     tf2["hi_mass"] > 0, self.hi_absent))
                blocks.append(_block(tf2["sat_pooled"], tf2["sat_center"], tf2["sat_mass"],
                                     tf2["sat_mass"] > 0, self.sat_absent))
            kv = torch.cat([kv] + blocks, dim=1)                    # (B, T+16*n_blocks, d)
            n_blocks = len(blocks)
            C = kv.shape[1]
            K = self.Wk(kv).view(B, C, self.h, self.d // self.h).transpose(1, 2)
            V = self.Wv(kv).view(B, C, self.h, self.d // self.h).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / (self.d // self.h) ** 0.5   # (B,h,17,C)
            scores = scores.masked_fill(~self._kv_mask(mask, T, n_blocks)[:, None], -1e9)
        w = self.drop(torch.softmax(scores, dim=-1))
        ctx = (w @ V).transpose(1, 2).reshape(B, 17, self.d)         # (B,17,d)
        present = mask.any(dim=1)
        ctx = self.norm(self.out(ctx)) * present.view(B, 1, 1).float()
        return ctx, present                                         # (B,17,d_model)

    def forward(self, up, lo, tf2_up: dict | None = None, tf2_lo: dict | None = None):
        """up/lo = dict(pooled,center,mass,mask). tf2_up/tf2_lo = dict(pooled,center,mass,present)
        (16 teeth per arch; only when use_tf2). Returns:
        fdi_tokens (B,32,d_model), cls_tokens (B,2,d_model), present (B,2) upper/lower."""
        cu, pu = self._attend(up["pooled"], up["center"], up["mass"], up["mask"], 0, tf2_up)
        cl, pl = self._attend(lo["pooled"], lo["center"], lo["mass"], lo["mask"], 1, tf2_lo)
        fdi = torch.cat([cu[:, :16], cl[:, :16]], dim=1)             # (B,32,d_model)
        cls = torch.cat([cu[:, 16:17], cl[:, 16:17]], dim=1)         # (B,2,d_model)
        present = torch.stack([pu, pl], dim=1)                       # (B,2)
        return fdi, cls, present
