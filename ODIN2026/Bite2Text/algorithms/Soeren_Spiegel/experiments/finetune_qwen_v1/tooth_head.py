"""Auxiliary per-tooth head: 32 FDI slots predicted from the pooled vision features.

The judge trace showed the model
had learned the population prior per finding type rather than the case (claims tooth 36 absent in
45 of 62 val cases against a reference count of 13). The cause is the target: an unremarkable tooth
has no entry in the canonical record, so "36 is present and unremarkable" never produces a
gradient. This head supervises that negative class densely, per tooth, off the vision features —
where prose glue cannot dilute it and where accuracy is measurable without the judge.

Two pieces live here so ``train.py`` stays a training loop:

  * ``VisionPooler`` — a forward hook on the vision merger, i.e. the module whose output is what
    actually enters the LM. The module path is FOUND BY INSPECTION of the loaded model (it moved
    to ``model.visual.merger`` in transformers 5.x, and PEFT prefixes it again), and the captured
    tensor is shape-asserted against ``image_grid_thw``/``video_grid_thw`` before any training
    uses it — a silently wrong capture would train the head on the wrong tensor and still show a
    falling loss.
  * ``ToothHead`` — LayerNorm -> MLP -> ``32 x (4 status logits + 6 attribute logits)``, plus the
    masked losses. Class weighting is inverse-frequency on the *actual shipped* label
    distribution (computed from the records at startup, never hardcoded), because the weak
    PRESENT label outnumbers ABSENT 2.3:1 and outnumbers ROOT_REMNANT 44:1; without it the head's
    optimum is "PRESENT everywhere", which is precisely the failure it exists to detect.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionPooler:
    """Capture the vision-tower merger output (the features that enter the LM) and mean-pool it.

    Capture is **opt-in**: the hook is inert unless inside ``with pooler.capturing():``. That is
    not fastidiousness. The hook is attached to a module every forward passes through, including
    the end-of-epoch val loop and anything that calls ``generate``. An always-on hook accumulated
    all 62 val forwards and the next training step concatenated them — 313,004 tokens against the
    4,470 its grids implied — killing fiveviews_n1_v1 at the start of epoch 2 (and holding ~2 GB
    of captures alive through the val loop). Scoping capture to the step that consumes it makes
    that class of bug unrepresentable rather than merely detected."""

    def __init__(self, model, spatial_merge_size: int):
        self.merge_area = spatial_merge_size * spatial_merge_size
        matches = [(n, m) for n, m in model.named_modules() if n.endswith("visual.merger")]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one '*visual.merger' module, found {len(matches)}: "
                f"{[n for n, _ in matches]}. The vision tower has been renamed; find the module "
                "whose output enters the LM and update this matcher.")
        self.path, module = matches[0]
        self.dim = module.mlp[-1].out_features
        self._outs = []
        self._enabled = False
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        if not self._enabled:
            return
        # The tower is invoked ONCE PER MODALITY — Qwen2.5-VL calls get_image_features and
        # get_video_features separately — so a single forward fires this hook twice and keeping
        # only the last capture silently drops the stills (1344 of 3299 tokens on a 24-frame
        # case). Accumulate; take() concatenates and checks the total against the grids.
        self._outs.append(out)

    @contextlib.contextmanager
    def capturing(self):
        """Arm the hook for exactly one forward. Entering clears any stale capture, and leaving
        disarms even on an exception, so a failed step cannot poison the next one."""
        self._outs = []
        self._enabled = True
        try:
            yield self
        finally:
            self._enabled = False

    def expected_tokens(self, batch) -> int:
        n = 0
        for key in ("image_grid_thw", "video_grid_thw"):
            g = batch.get(key)
            if g is not None:
                n += int(g.prod(dim=-1).sum().item()) // self.merge_area
        return n

    def take(self, batch, assert_shape: bool = True) -> torch.Tensor:
        """Mean-pooled vision features, [1, dim]. Consumes the capture (so a missing forward is
        an error rather than a stale reuse). ``--batch 1`` only: Qwen flattens the vision tensors
        across the batch, so per-sample pooling at batch>1 needs the grids split explicitly."""
        outs, self._outs = self._outs, []
        if not outs:
            raise RuntimeError(
                f"no vision features captured at {self.path}. Either the forward did not reach "
                "the merger (a text-only batch), or it ran outside `with pooler.capturing():`.")
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        if assert_shape:
            want = self.expected_tokens(batch)
            if out.shape[0] != want or out.shape[-1] != self.dim:
                raise RuntimeError(
                    f"vision capture {tuple(out.shape)} disagrees with the processor grids "
                    f"(expected [{want}, {self.dim}] from image/video_grid_thw / "
                    f"merge_area={self.merge_area}). Refusing to train the head on it.")
        return out.float().mean(dim=0, keepdim=True)


class ToothHead(nn.Module):
    """Pooled vision features -> per-slot status logits [B,32,4] and attribute logits [B,32,6]."""

    def __init__(self, d_in: int, n_slots: int, n_status: int, n_attr: int, hidden: int = 512):
        super().__init__()
        self.n_slots, self.n_status, self.n_attr = n_slots, n_status, n_attr
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, n_slots * (n_status + n_attr)),
        )

    def forward(self, pooled: torch.Tensor):
        z = self.net(pooled).view(-1, self.n_slots, self.n_status + self.n_attr)
        return z[..., :self.n_status], z[..., self.n_status:]


def label_stats(records, tooth_state, n_slots: int, n_status: int, n_attr: int):
    """Class frequencies over the training records, for the loss weights.

    Computed from the records the run will actually train on and through the same
    ``tooth_state`` the dataset calls, so the weights can never drift from the labels.
    ``superior_truncated`` is unknown here (it needs a render); False is the permissive
    setting, which over-counts upper-arch slots slightly and is the conservative direction
    for a frequency estimate."""
    status = torch.zeros(n_status)
    attr_pos = torch.zeros(n_attr)
    attr_n = torch.zeros(n_attr)
    for r in records:
        rec = r.get("record")
        if rec is None:
            continue
        labels, mask = tooth_state(rec)
        s = torch.from_numpy(labels["status"])
        s = s[s >= 0]
        status += torch.bincount(s, minlength=n_status).float()
        m = torch.from_numpy(mask["attr"]).float()
        attr_pos += (torch.from_numpy(labels["attr"]) * m).sum(0)
        attr_n += m.sum(0)
    return status, attr_pos, attr_n


def loss_weights(status_counts, attr_pos, attr_n, power: float = 1.0,
                 max_ratio: float = 50.0, max_pos_weight: float = 20.0):
    """Inverse-frequency status weights and per-attribute BCE ``pos_weight``.

    ``w_c = (N / (K * n_c)) ** power``, which at ``power=1`` is exact inverse frequency
    normalised so ``sum_c n_c * w_c == N`` — i.e. the weighting rebalances the classes without
    rescaling the loss, so ``aux_loss`` stays comparable across runs. On the shipped train
    distribution that is roughly present 0.33 / absent 1.3 / impacted 6.0 / root_remnant 24.

    ``max_ratio`` caps max/min. It is not cosmetic: on a reduced ``--max-cases`` smoke a class
    can have 2 examples, and uncapped inverse frequency then hands it a four-figure weight.
    ``max_pos_weight`` does the same job for the attributes — implants are 4.3% of supervised
    slots, and an uncapped neg/pos ratio there is a 22:1 incentive to guess positive."""
    n = status_counts.clamp(min=1)
    w = (n.sum() / (len(n) * n)) ** power
    w = w.clamp(max=float(w.min()) * max_ratio)
    pos_w = ((attr_n - attr_pos) / attr_pos.clamp(min=1)).clamp(max=max_pos_weight)
    return w, pos_w


def aux_loss(status_logits, attr_logits, batch, status_weight, pos_weight, ignore_index: int):
    """Masked CE over the 4 status classes + masked BCE over the 6 attributes.

    Returns ``(loss, parts)``; ``parts`` carries the two terms and the per-class hit counts the
    caller needs to log per-class recall — a head that predicts PRESENT everywhere must be
    visible in the logs, not inferable from an aggregate."""
    tgt = batch["tooth_status"]                                  # [B,32] long, -100 = ignore
    n_status = status_logits.shape[-1]
    sup = tgt != ignore_index
    if sup.any():
        ce = F.cross_entropy(status_logits.reshape(-1, n_status).float(), tgt.reshape(-1),
                             weight=status_weight, ignore_index=ignore_index)
    else:
        # Every slot masked. F.cross_entropy computes 0/0 and returns NaN here (its gradient is
        # a clean zero, so weights survive, but the NaN poisons the logged loss for the whole
        # accumulation window — 8 of 40 steps on the first smoke run). 17 of 498 train cases hit
        # this: the fully-gated edentulous ones, where the label builder is correctly refusing to
        # emit anything. Same trap NaN-guard fixed for the LM loss.
        ce = status_logits.sum() * 0.0

    m = batch["tooth_attr_mask"]                                 # [B,32,6] bool
    n_attr_sup = m.sum()
    if n_attr_sup > 0:
        bce_all = F.binary_cross_entropy_with_logits(
            attr_logits.float(), batch["tooth_attr"].float(),
            pos_weight=pos_weight, reduction="none")
        bce = (bce_all * m).sum() / n_attr_sup
    else:
        bce = attr_logits.sum() * 0.0

    with torch.no_grad():
        sup = tgt != ignore_index
        pred = status_logits.argmax(-1)
        hit = torch.zeros(n_status, device=tgt.device)
        tot = torch.zeros(n_status, device=tgt.device)
        pred_n = torch.zeros(n_status, device=tgt.device)
        for c in range(n_status):
            is_c = sup & (tgt == c)
            tot[c] = is_c.sum()
            hit[c] = (is_c & (pred == c)).sum()
            pred_n[c] = (sup & (pred == c)).sum()

    loss = ce + bce
    parts = {"ce": ce.detach(), "bce": bce.detach(),
             "hit": hit, "tot": tot, "pred": pred_n}
    return loss, parts
