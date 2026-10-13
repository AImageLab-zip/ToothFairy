"""Cross-label self-attention — anatomical tokens -> contextualised tokens.

[N,d_model] aggregated tokens (gated attention for non-teeth + tooth-query cross-attention
for teeth, both emit d_model) + a token-id embedding (so the model knows token i is FDI-17
vs maxilla vs the upper CLS) + a presence embedding. token-id/presence are ADDED (BERT-style
segment embedding, not concatenated — all three live in the same d_model space), then a
1-layer fusion (Linear+GELU) -> self-attention (1~2 layers, lets findings reference each
other, e.g. periodontitis "affecting both arches" spans teeth+bone) -> contextual tokens
[N,d_model] that the generator projects to the LLM visual prefix.

Absent tokens carry a learned presence(0) embedding so the LLM can distinguish
"structure absent / out of FOV" from "structure present, no finding".
The two ablation flags (use_presence_emb, use_in_proj) are kept as constructor args.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossLabelEncoder(nn.Module):
    def __init__(self, d_model: int, n_tokens_max: int,
                 n_layers: int = 2, n_heads: int = 4, mlp_hidden: int = 256,
                 dropout: float = 0.1, use_presence_emb: bool = True,
                 use_in_proj: bool = True):
        super().__init__()
        self.token_id_emb = nn.Embedding(n_tokens_max, d_model)
        # presence embedding: absent tokens carry presence(0) so the LLM can tell
        # "absent/out-of-FOV" from "present, no finding". False removes it (ablation
        # checking whether it is redundant).
        self.use_presence_emb = use_presence_emb
        if use_presence_emb:
            self.presence_emb = nn.Embedding(2, d_model)         # 0 absent, 1 present
        # token + token_id + presence are ADDED (all d_model), then a single Linear+GELU
        # fuses them before self-attention. With use_in_proj=False this MLP is dropped and the
        # sum goes straight into self-attention.
        self.use_in_proj = use_in_proj
        if use_in_proj:
            self.in_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=mlp_hidden,
            dropout=dropout, batch_first=True, norm_first=True)
        self.self_attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, token_ids: torch.Tensor,
                presence: torch.Tensor, hide: torch.Tensor | None = None) -> torch.Tensor:
        """tokens (B,N,d_model), token_ids (B,N) long, presence (B,N) {0,1} ->
        (B,N,d_model). All N tokens are always present as sequence positions
        (absent ones are zero+flag), so no attention padding mask is needed.

        `hide` (N,) or (B,N) bool — positions that are True are **removed from the keys**.
        This is how the two arch-summary slots are taken out of the model: the indices and the
        weight shapes stay as they are and only attention changes, so the constants of the
        38-token stack are untouched (slot misassignment is a known failure mode of this
        codebase). Hidden positions are still computed, but nothing reads their output. With
        36 keys left, no row is fully masked, so no NaN can arise from that.
        """
        x = tokens + self.token_id_emb(token_ids)
        if self.use_presence_emb:
            x = x + self.presence_emb(presence.long())
        if self.use_in_proj:
            x = self.in_proj(x)                                 # (B,N,d) 1-layer fuse
        pad = None
        if hide is not None:
            pad = hide if hide.dim() == 2 else hide.unsqueeze(0).expand(x.shape[0], -1)
            pad = pad.to(x.device)
        x = self.self_attn(x, src_key_padding_mask=pad)         # (B,N,d)
        return self.norm(x)
