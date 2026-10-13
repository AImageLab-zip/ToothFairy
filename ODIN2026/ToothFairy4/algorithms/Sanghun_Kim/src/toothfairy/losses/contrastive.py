"""Region-text contrastive alignment (SigLIP).

Aligns the visual prefix (projector output, in LLM hidden space) with the region GT text
(frozen PubMedBERT) through the SigLIP sigmoid pairwise loss, which strengthens the
conditioning on vision. The SigLIP projections / logit parameters and the pairwise logic live
in a self-contained module (ContrastiveHead), the text target in FrozenTextEncoder. Collecting
the vision and text embeddings (which needs access to realizer._region_prefix and the slot
tokens) is the job of the assembling model (report_model); this file only computes the
alignment loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenTextEncoder(nn.Module):
    """Frozen mean-pooled sentence embeddings from PubMedBERT (or any HF encoder), with a cache.

    A frozen anchor that only pulls the vision side (reusing the Qwen embeddings is forbidden:
    it would make the target depend on the model being trained). Loaded directly with the
    transformers AutoModel plus mean pooling, which avoids a sentence-transformers dependency.
    The GT text is frozen, so a (text -> embedding) cache removes re-encoding; the cache is
    keyed by the text itself, so it stays valid when the GT report drawn for a patient changes
    between epochs."""

    def __init__(self, name: str = "neuml/pubmedbert-base-embeddings",
                 device=None, max_len: int = 128, cache: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        self.max_len = max_len
        self._cache: dict[str, torch.Tensor] | None = {} if cache else None
        if device is not None:
            self.model.to(device)

    def train(self, mode: bool = True):
        """The text encoder is **always** in eval() — it is the fixed anchor of the alignment
        and must not move.

        `model.text_encoder = FrozenTextEncoder(...)` is an nn.Module attribute assignment, so
        it registers as a submodule and the assembling model's `model.train()` recursively puts
        the PubMedBERT inside back into training mode. The `self.model.eval()` in the
        constructor runs once and cannot prevent that. PubMedBERT has hidden and attention
        dropout of 0.1, so in training mode **the same sentence gives a different vector every
        time** (measured: cos 0.83-0.87 between two draws, 0.91-0.93 against the eval vector,
        while the mean cos between different sentences is 0.494) and the cache in `encode`
        stores that first draw permanently. Since the seed differs per rank, one SigLIP matrix
        would mix different positives of the same sentence, and `siglip_hard_neg_mask` would
        assume an identity that no longer holds.
        """
        super().train(mode)
        self.model.eval()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """list[str] -> (N, dim) mean-pooled embeddings (frozen; uses the cache). The tensor is
        returned on model.device. Empty strings are encoded too (minimal tokens, so that no
        empty vector is produced)."""
        dev = self.device
        if self._cache is None:
            return self._encode_batch(texts, dev)
        miss = [t for t in texts if t not in self._cache]
        if miss:
            uniq = list(dict.fromkeys(miss))                 # encode each unique string once
            emb = self._encode_batch(uniq, dev)
            for t, e in zip(uniq, emb):
                self._cache[t] = e.detach()
        return torch.stack([self._cache[t].to(dev) for t in texts], dim=0)

    def _encode_batch(self, texts: list[str], dev) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(dev)
        out = self.model(**enc).last_hidden_state             # (N,L,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()    # (N,L,1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-6)
        return summed / counts                                # (N,H) mean-pool


class ContrastiveHead(nn.Module):
    """SigLIP sigmoid pairwise contrastive between region visual prefix and region GT text.

    vision (M,hidden) mean-pooled visual prefix, text (M,text_dim) frozen encoder embedding →
    projected/normalised → sigmoid pairwise loss (+1 diagonal positive, -1 off-diagonal). Params:
    vision/text proj + learnable logit tau (log10 init) / bias (SigLIP init).
    """

    def __init__(self, hidden: int, text_dim: int, proj_dim: int):
        super().__init__()
        self.vision_proj = nn.Linear(hidden, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.302585))   # log 10 → τ≈10 (SigLIP init)
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, vision: torch.Tensor, text: torch.Tensor, texts: list[str],
                hard_neg_mask: bool) -> torch.Tensor:
        """vision (M,hidden), text (M,text_dim), texts list[str] (used for the hard-negative
        test) -> scalar. The assembling model does not call this when M < 2 (no pairs)."""
        zv = F.normalize(self.vision_proj(vision), dim=-1)      # (M,dp)
        ztp = F.normalize(self.text_proj(text), dim=-1)        # (M,dp)
        logits = self.logit_scale.exp() * (zv @ ztp.t()) + self.logit_bias   # (M,M)
        M = logits.shape[0]
        eye = torch.eye(M, device=logits.device, dtype=torch.bool)
        y = 2.0 * eye.float() - 1.0                            # +1 on the diagonal, -1 elsewhere
        lmat = -F.logsigmoid(y * logits)                       # (M,M) SigLIP sigmoid pairwise
        if hard_neg_mask:
            # Hard-negative masking (avoids a clinical false negative): off-diagonal pairs
            # whose GT text is exactly identical (i.e. the same finding clinically, e.g.
            # "46 absent" for two patients) are dropped from the negatives. Pushing an
            # identical target apart is misalignment by definition. The diagonal (positive) is
            # kept. With no same-text pair keep=all, which is the plain mean again.
            same = torch.tensor([[a == b for b in texts] for a in texts],
                                device=logits.device)          # (M,M) identical GT text
            keep = (eye | ~same).float()                       # diagonal + off-diag with other text
            return (lmat * keep).sum() / keep.sum().clamp_min(1.0)
        return lmat.mean()
