"""Findings classification loss — per-tooth 10-axis + non-teeth region categorical claim.

Written as pure functions: everything the loss needs is passed in as an argument. CE/BCE carry the
focal modulation term (1-p_t)^gamma, so gamma=0 reduces exactly to F.cross_entropy /
BCEWithLogits(mean). Class weighting (effective number, schema.compute_class_weights) and the focal
term are orthogonal — both are applied multiplicatively.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..schema.claims import AXES


def focal_ce(logits: torch.Tensor, target: torch.Tensor,
             weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """Focal CE for a softmax axis. logits (M,C) · target (M,) int64.
    Normalised as a weighted mean, following the F.cross_entropy weight convention:
    sum(alpha_t · loss) / sum(alpha_t)."""
    if gamma == 0.0:
        return F.cross_entropy(logits, target, weight=weight, reduction="mean")
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)       # (M,) log p_true
    p_t = logp_t.exp()
    focal = (1.0 - p_t).clamp_min(0.0).pow(gamma) * (-logp_t)     # (M,)
    if weight is not None:
        alpha = weight[target]                                    # (M,) class weight per elem
        return (alpha * focal).sum() / alpha.sum().clamp_min(1e-8)
    return focal.mean()


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              pos_weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """Focal BCE for a sigmoid axis. logits (M,) · target (M,) float (0/1). pos_weight = alpha for
    the positive class. Reduction is an element mean; with gamma=0 this is identical to
    BCEWithLogits(reduction='mean')."""
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")  # (M,)
    if gamma == 0.0:
        return bce.mean()
    p = torch.sigmoid(logits)
    p_t = torch.where(target > 0.5, p, 1.0 - p)                   # probability of the true class
    return ((1.0 - p_t).clamp_min(0.0).pow(gamma) * bce).mean()


def teeth_gate_loss(teeth_logits: torch.Tensor, labels: torch.Tensor,
                    logit_slices: list[slice], class_weights: dict | None,
                    axis_weights: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """(B,32,AUX_DIM) + (B,32,N_AXES) int64 → scalar. Per-tooth axes, each a (class-weighted)
    focal CE/BCE, combined as a weighted mean over axis_weights (which scales the state axis).
    class_weights None = unweighted. gamma=0 reduces exactly to CE/BCE."""
    B, T, _ = teeth_logits.shape
    cw = class_weights
    losses = []
    for j, ax in enumerate(AXES):
        lg = teeth_logits[..., logit_slices[j]]          # (B,32,logit_dim)
        tg = labels[..., j]                               # (B,32) int64
        if ax.head == "softmax":
            w = cw.get(ax.name) if cw else None           # (n_cat,) or None
            if w is not None:
                w = w.to(lg.device)
            l = focal_ce(lg.reshape(B * T, -1), tg.reshape(B * T), w, gamma)
        else:                                             # sigmoid: logit (B,32,1)
            pw = cw.get(ax.name) if cw else None          # scalar tensor or None
            if pw is not None:
                pw = pw.to(lg.device)
            l = focal_bce(lg.squeeze(-1), tg.float(), pw, gamma)
        losses.append(l)
    L = torch.stack(losses)                           # (N_AXES,) per-axis loss
    aw = axis_weights.to(L.device)                    # multiplier on the state axis (idx 0)
    return (L * aw).sum() / aw.sum()                  # weighted mean (= mean when uniform)


def region_axes_loss(logits: torch.Tensor, labels: torch.Tensor, axes,
                     slices: list[slice], gamma: float = 0.0) -> torch.Tensor:
    """Non-teeth region: logits(...,region_aux_dim) + labels(...,n_axes) int64 → uniform mean over
    the per-axis focal CE (softmax) / BCE (sigmoid). No class weighting is applied here.
    gamma=0 reduces exactly to CE/BCE."""
    flat_lg = logits.reshape(-1, logits.shape[-1])
    flat_tg = labels.reshape(-1, labels.shape[-1])
    losses = []
    for j, ax in enumerate(axes):
        lg = flat_lg[:, slices[j]]                       # (N, logit_dim)
        tg = flat_tg[:, j]                               # (N,)
        if ax.head == "softmax":
            losses.append(focal_ce(lg, tg, None, gamma))
        else:
            losses.append(focal_bce(lg.squeeze(-1), tg.float(), None, gamma))
    return torch.stack(losses).mean()


def gate_loss(teeth_logits: torch.Tensor, nonteeth_logits: dict | None,
              fdi_labels: torch.Tensor, nonteeth_labels: dict | None,
              logit_slices: list[slice], nt_slices: dict, class_weights: dict | None,
              axis_weights: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """L_findings = per-tooth axes + non-teeth (maxilla, mandible, right/left mandibular canal)
    claim loss. Without non-teeth labels only the teeth term is used. The focal term
    (gate_focal_gamma) applies to every axis. The non-teeth head outputs are computed by the caller
    (FindingsHead.nonteeth_logits) and passed in."""
    l_teeth = teeth_gate_loss(teeth_logits, fdi_labels, logit_slices,
                              class_weights, axis_weights, gamma)
    if nonteeth_labels is None or nonteeth_logits is None:
        return l_teeth
    parts = [l_teeth,
             region_axes_loss(nonteeth_logits["maxilla"], nonteeth_labels["maxilla"],
                              MAXILLA_AXES, nt_slices["maxilla"], gamma),
             region_axes_loss(nonteeth_logits["mandible"], nonteeth_labels["mandible"],
                              MANDIBLE_AXES, nt_slices["mandible"], gamma),
             region_axes_loss(nonteeth_logits["nerve"], nonteeth_labels["nerve"],
                              NERVE_AXES, nt_slices["nerve"], gamma),
             ]                                        # no arch term: staging moved to the regions
    return torch.stack(parts).mean()                  # uniform mean over teeth + 3 regions


def axis_soft_probs(aux_logits_i: torch.Tensor, logit_slices: list[slice]) -> torch.Tensor:
    """(32, AUX_DIM) → (32, SOFT_DIM). For a softmax axis the abnormal probabilities (normal/none
    at index 0 dropped), for a sigmoid axis the positive probability, concatenated in AXES order."""
    parts = []
    for j, ax in enumerate(AXES):
        lg = aux_logits_i[..., logit_slices[j]]
        if ax.head == "softmax":
            parts.append(torch.softmax(lg, dim=-1)[..., 1:])   # drop normal/none
        else:
            parts.append(torch.sigmoid(lg))                    # positive prob (·,1)
    return torch.cat(parts, dim=-1)
