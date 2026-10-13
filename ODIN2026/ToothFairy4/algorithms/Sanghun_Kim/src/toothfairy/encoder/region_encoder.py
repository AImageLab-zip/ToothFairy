"""RegionEncoder — cached region bags -> contextual anatomical tokens.

Assembles the canonical [38, d_model] anatomical token stack and contextualises it:
  non-teeth (maxilla·mandible·nerve_R·nerve_L) -> GatedAttention  (4 tokens)
  teeth (upper·lower)                          -> ToothQueryCrossAttention (32 FDI + 2 CLS)
  -> CrossLabelEncoder (token-id + presence + self-attention) -> ctx [B,38,d_model].

Pure component: it takes explicit hyperparameters rather than a config object (the assembler in
models/report_model unpacks the config and passes them in).

Batch contract (from data collate):
  regions[c] = {pooled (B,Pc,64), mass (B,Pc), center (B,Pc,3), mask (B,Pc)}  c∈{1,2,5}
  teeth[arch]= {pooled, mass, center, mask}                                   arch∈{upper,lower}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..schema.claims import FDI_ALL
from .gated_attention import GatedAttention
from .tooth_query import ToothQueryCrossAttention
from .cross_label import CrossLabelEncoder

# canonical [38,d_model] token order: 4 non-teeth (maxilla·mandible·nerve_R·nerve_L),
# then upper(16 FDI + CLS), lower(16 FDI + CLS). The mandibular canal is split left/right.
# 32 FDI positions in the stack (skip the 2 CLS tokens at 20 and 37).
FDI_POS_IN_STACK = list(range(4, 20)) + list(range(21, 37))

# --- slot mapping constants (semantic positions in the 38-token stack; shared by the findings
# head and the generator) --- #
# The canal is split left/right across the 38 slots: nerve_right = slot 2, nerve_left = slot 3
NONTEETH_SLOT = {"maxilla": 0, "mandible": 1, "nerve_right": 2, "nerve_left": 3}  # one token each
UPCLS_SLOT, LOCLS_SLOT = 20, 37                               # arch-level CLS slots
V1_TEETH_SLOTS = list(range(4, 38))                          # all 34 teeth-side tokens (slot 4..37)
# FDI_ALL[j] (row j of the findings head)  ->  stack slot  /  head row
FDI_TO_SLOT = {f: FDI_POS_IN_STACK[j] for j, f in enumerate(FDI_ALL)}
FDI_TO_ROW = {f: j for j, f in enumerate(FDI_ALL)}


class RegionEncoder(nn.Module):
    def __init__(self, feat_dim: int, pe_k: int, d_model: int, gate_hidden: int,
                 n_heads: int, dropout: float, n_tokens: int,
                 cl_layers: int, cl_heads: int, cl_mlp_hidden: int, cl_dropout: float,
                 use_presence_emb: bool = True, use_in_proj: bool = True,
                 use_tf2: bool = False, use_tf2_density: bool = False,
                 drop_arch_cls: bool = False):
        super().__init__()
        self.use_tf2 = use_tf2
        self.drop_arch_cls = drop_arch_cls
        self.gated = GatedAttention(feat_dim, pe_k, d_model, gate_hidden, dropout)
        self.teeth = ToothQueryCrossAttention(feat_dim, pe_k, d_model, n_heads, dropout,
                                              use_tf2=use_tf2, use_tf2_density=use_tf2_density)
        self.cross_label = CrossLabelEncoder(
            d_model=d_model, n_tokens_max=n_tokens, n_layers=cl_layers,
            n_heads=cl_heads, mlp_hidden=cl_mlp_hidden, dropout=cl_dropout,
            use_presence_emb=use_presence_emb, use_in_proj=use_in_proj)

    def aggregate(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """non-teeth gated attention + teeth tooth-query cross-attention -> [B,38,d_model]."""
        regions = batch["regions"]
        teeth = batch["teeth"]
        nt_tok, nt_pres = [], []
        for c in (1, 2):                                     # maxilla, mandible (one token each)
            r = regions[c]
            v, p = self.gated(r["pooled"], r["center"], r["mass"], r["mask"])
            nt_tok.append(v); nt_pres.append(p)
        # Split the canal (class 5) left/right -> nerve_R (slot 2), nerve_L (slot 3).
        # **In plans space, axis 2 is the left/right axis** — axis 0 is anterior/posterior, and an
        # earlier version incorrectly split on it. Measured: for the two canal clusters the
        # difference between the cluster centres is dominated by axis 2 in 242/242 patients
        # (|d ax2| = 0.38 vs ax0 = 0.02, ax1 = 0.01). The sign was pinned down by agreement with
        # the laterality of the missing-tooth ground truth (missing on the right correlates with a
        # mass deficit at ax2 < 0.5, corr -0.23), so **ax2 < 0.5 is the right side** — which is why
        # (cx < 0.5) comes first and maps to nerve_right (slot 2). (Axis 1 is superior/inferior:
        # maxilla > mandible.)
        r5 = regions[5]
        cx = r5["center"][..., 2]                            # (B,P) normalised L/R coordinate
        for side in (cx < 0.5, cx >= 0.5):                   # right (slot 2), then left (slot 3)
            m = r5["mask"] & side
            v, p = self.gated(r5["pooled"], r5["center"], r5["mass"], m)
            nt_tok.append(v); nt_pres.append(p)
        tf2 = batch.get("teeth_tf2") if self.use_tf2 else None
        if tf2 is not None:                                  # inject the tf2 per-tooth K/V as well
            fdi, cls, arch_pres = self.teeth(teeth["upper"], teeth["lower"],
                                             tf2_up=tf2["upper"], tf2_lo=tf2["lower"])
        else:
            fdi, cls, arch_pres = self.teeth(teeth["upper"], teeth["lower"])  # (B,32,f)(B,2,f)(B,2)
        # assemble canonical 38: [nt4, up16, upCLS, lo16, loCLS]
        up_pres = arch_pres[:, 0:1].expand(-1, 16)
        lo_pres = arch_pres[:, 1:2].expand(-1, 16)
        tokens = torch.cat([torch.stack(nt_tok, dim=1),       # (B,4,f)
                            fdi[:, :16], cls[:, 0:1],         # up
                            fdi[:, 16:], cls[:, 1:2]], dim=1)  # lo  -> (B,38,f)
        presence = torch.cat([torch.stack(nt_pres, dim=1),
                              up_pres, arch_pres[:, 0:1],
                              lo_pres, arch_pres[:, 1:2]], dim=1)  # (B,38)
        return tokens, presence

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """-> ctx (B,38,d_model), presence (B,38)."""
        tokens, presence = self.aggregate(batch)              # (B,38,d_model),(B,38)
        B, N, _ = tokens.shape
        token_ids = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        ctx = self.cross_label(tokens, token_ids, presence.long(),
                               hide=self._hide_mask(N, tokens.device))   # (B,38,d_model)
        return ctx, presence

    def _hide_mask(self, n: int, device):
        """With drop_arch_cls=True the two arch CLS slots (20 and 37) are removed from the keys of
        the cross-label attention. With False (the default) it returns None, so the numbers stay
        **bit-identical** to a model built without the option."""
        if not self.drop_arch_cls:
            return None
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[UPCLS_SLOT] = True
        m[LOCLS_SLOT] = True
        return m
