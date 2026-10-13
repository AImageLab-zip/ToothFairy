"""FindingsModel — perception features -> encoder -> findings head (no LLM).

This is the subset of `ReportModel` with the realizer (the LLM) removed: the encoder, the
findings head and the loss are the very same components, so a classification-only run stays
comparable with the full pipeline and nothing is implemented twice.

  encoder(batch) -> ctx [B,38,d], presence
  findings_head -> teeth_logits [B,32,AUX_DIM] + non-teeth logits
  loss = L_findings (10 tooth axes + maxilla, mandible, mandibular canal; effective-number
         class weights) [+ lambda_siglip * L_contrastive]

**SigLIP option**: categorical claim supervision only teaches the 21 predefined classes, so
wording outside those axes (`pericoronal bone resorption of 1-2 mm` and the like) has nowhere to
be learned. SigLIP fills that gap by aligning a region's anatomical token with the **frozen
embedding of that region's reference sentence**. The two objectives were measured to teach
different things (SigLIP: global AUC +0.071, within-patient +0.010; categorical supervision:
+0.006 and +0.029). lambda sets the mix, and at lambda=0 the model is **bitwise identical** to
the categorical-only one.

WARNING: the vision side differs from `ReportModel._siglip_loss` — there the **LLM input** after
the projector is used, but there is no LLM here, so **ctx (the anatomical tokens) itself** is
aligned. That is exactly what the perception stage trains, so it is the right target, and
staying off the LLM keeps this a cheap stage.

`_encode` has the same signature as `ReportModel._encode`, so the findings-F1 evaluation code can
be shared.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ReportModelConfig
from ..schema.claims import (logit_slices, region_logit_slices, N_AXES,
                            MAXILLA_AXES, MANDIBLE_AXES, NERVE_AXES)
from ..heads import FindingsHead
from ..losses import gate_loss
from ..losses.contrastive import ContrastiveHead
from .report_model import build_encoder, ReportModel


class FindingsModel(nn.Module):
    """Encoder + findings head only. Takes ReportModelConfig as is; decode keys are ignored."""

    def __init__(self, cfg: ReportModelConfig, device=None):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        # Built from the **same components** as ReportModel — the comparison only holds if
        # the encoder is the same one.
        self.encoder = build_encoder(cfg, device)
        self.findings_head = FindingsHead(d_model)
        self._logit_slices = logit_slices()
        self._nt_slices = {"maxilla": region_logit_slices(MAXILLA_AXES),
                           "mandible": region_logit_slices(MANDIBLE_AXES),
                           "nerve": region_logit_slices(NERVE_AXES)}
        # Per-axis effective-number class weights, injected by the trainer from the training
        # ground-truth frequencies. A dict is not moved by model.to, so the loss moves it to
        # logits.device. None = unweighted.
        self.class_weights: dict[str, torch.Tensor] | None = None
        # Region-text contrastive (optional). The text encoder is **injected by the trainer** so
        # that runs without SigLIP do not pay for the heavy HF load (same convention as
        # ReportModel).
        self.use_siglip = bool(cfg.train.use_siglip)
        self.text_encoder = None
        self.contrastive = (ContrastiveHead(d_model, cfg.train.siglip_text_dim,
                                            cfg.train.siglip_proj_dim)
                            if self.use_siglip else None)
        aw = torch.ones(N_AXES)
        aw[0] = float(cfg.train.gate_state_weight)          # weight of the state axis
        self.register_buffer("axis_weights", aw)

    def _encode(self, batch):
        """aggregate + cross-label -> ctx (B,38,d); teeth aux logits (B,32,AUX_DIM)."""
        ctx, _presence = self.encoder(batch)
        return ctx, self.findings_head.teeth_logits(ctx)

    def _siglip_loss(self, ctx: torch.Tensor, batch) -> torch.Tensor:
        """SigLIP between a region's anatomical token and that region's reference sentence
        (frozen text encoder).

        Normal regions (empty target) are excluded: with no sentence there is nothing to match,
        and including them would collapse every empty-string embedding onto the same point and
        destroy the diagonal. With fewer than two pairs the pairwise loss is undefined, so zero
        is returned (gradient-safe).
        """
        vz, texts = [], []
        for i, targets in enumerate(batch.get("region_targets_text") or []):
            if not targets:
                continue
            for rid, txt in targets.items():
                if not txt or not txt.strip():
                    continue
                # Use the **same slot table** as ReportModel._slot_tokens: a mismatch here aligns
                # the vector of the wrong region with the sentence, and the loss still looks fine.
                tok = ReportModel._slot_tokens(self, ctx[i], rid, None, False)[0]
                vz.append(tok.mean(dim=0))
                texts.append(txt)
        if len(vz) < 2:
            return ctx.new_zeros(())
        V = torch.stack(vz)                                      # (M,d_model)
        with torch.no_grad():
            zt = self.text_encoder.encode(texts).to(V.device).float()
        return self.contrastive(V, zt, texts, self.cfg.train.siglip_hard_neg_mask)

    def forward(self, batch) -> dict:
        ctx, aux_logits = self._encode(batch)
        gamma = float(self.cfg.train.gate_focal_gamma or 0.0)
        nt = batch.get("nonteeth_labels")
        nonteeth_logits = self.findings_head.nonteeth_logits(ctx) if nt is not None else None
        l_gate = gate_loss(aux_logits, nonteeth_logits, batch["fdi_labels"], nt,
                           self._logit_slices, self._nt_slices, self.class_weights,
                           self.axis_weights, gamma)
        out = {"loss": l_gate, "l_gate": l_gate}
        if self.contrastive is not None and self.text_encoder is not None:
            l_sig = self._siglip_loss(ctx, batch)
            out["l_siglip"] = l_sig.detach()
            out["loss"] = l_gate + float(self.cfg.train.lambda_siglip) * l_sig
        return out
