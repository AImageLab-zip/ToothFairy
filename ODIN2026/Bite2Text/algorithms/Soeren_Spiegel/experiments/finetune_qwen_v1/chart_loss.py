"""Token-level chart/report segmentation and the optional weighted CE.

The default chart-QA run needs **none** of this: chart and report are both supervised at weight
1.0 by the existing completion-only mask, which is plain LM cross-entropy and leaves the
training path bit-comparable to combined_v1. Everything here exists for two things:

  * ``--chart-loss-weight`` — a knob the plan asks for and pins at 1.0. The aux-head lesson about
    choosing an auxiliary weight from the *gradient ratio* does not apply: this is the same
    head, the same CE, the same output cardinality, so the loss values are commensurable and
    there is no tuning round to run.
  * ``--chart-class-weights`` — **the one contingency lever**, armed only if the smoke
    finds the chart collapsing to the all-``present`` solution (aux-head unweighted smoke escaped
    exactly that). It up-weights the tokens that spell a rare class inside the chart segment by
    inverse class frequency, which is the token-space analogue of the head's class weighting.

**Segmenting the target.** The chart/report boundary has to be found in *token* space, and the
tokenization of the delimiter depends on its context, so it is not safe to encode the chart
alone and count. Instead each supervised token is decoded individually and the pieces are
concatenated: Qwen2.5's tokenizer is byte-level BPE, so for an ASCII target that reconstruction
is exact, and the cumulative character offsets give a token -> character map. The reconstruction
is *checked* against the delimiter rather than assumed; a target that does not contain it falls
back to unweighted and is counted, because a silent fallback here would report a weighted run
that was never weighted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from toothfairy.chart_parse import CHART_CLASSES, DELIMITER


class TokenTexts:
    """``id -> decoded text`` with a cache. One decode per distinct token id, then free."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._cache: dict[int, str] = {}

    def __call__(self, ids) -> list[str]:
        out = []
        for i in ids:
            i = int(i)
            s = self._cache.get(i)
            if s is None:
                s = self._cache[i] = self.tok.decode([i])
            out.append(s)
        return out


def segment_offsets(texts: list[str]) -> tuple[str, list[int]]:
    """Concatenated text of a token run plus each token's start offset in it."""
    starts, n = [], 0
    for t in texts:
        starts.append(n)
        n += len(t)
    return "".join(texts), starts


def chart_span(texts: list[str]) -> tuple[int, int] | None:
    """``(chart_start, chart_end)`` as token indices into ``texts``, or ``None`` when the frozen
    delimiter is not present. ``chart_end`` is exclusive and stops **before** the delimiter, so
    the delimiter's own tokens are neither chart nor report — they are format, and weighting
    them would weight a constant."""
    text, starts = segment_offsets(texts)
    at = text.find(DELIMITER)
    if at == -1:
        return None
    end = 0
    while end < len(starts) and starts[end] < at:
        end += 1
    return 0, end


def class_occurrences(texts: list[str], span: tuple[int, int]
                      ) -> list[tuple[int, int, int]]:
    """``[(first_token, last_token_exclusive, class_index), ...]`` for every class name spelled
    inside the chart segment.

    ``first_token`` is where the decision actually happens: with the FDI code already decoded,
    the next token is the model choosing between ``present`` / ``absent`` / ``implant`` / …, so
    teacher-forced accuracy at that one position is per-class recall of the chart — the direct
    analogue of the auxiliary head's ``aux/recall_*``, and the smoke gate against the all-present
    collapse. The rest of the range exists for the weighting, which wants the whole word."""
    start, end = span
    text, starts = segment_offsets(texts)
    chart_end_char = starts[end] if end < len(starts) else len(text)
    chart_text = text[:chart_end_char]
    out: list[tuple[int, int, int]] = []
    # Longest class name first. No class name is a prefix of another, but "root remnant"
    # contains no other name either, so a longest-first scan is unambiguous.
    for cls in sorted(CHART_CLASSES, key=len, reverse=True):
        ci = CHART_CLASSES.index(cls)
        at = chart_text.find(cls)
        while at != -1:
            toks = [j for j in range(start, end)
                    if starts[j] < at + len(cls) and starts[j] + len(texts[j]) > at]
            if toks:
                out.append((toks[0], toks[-1] + 1, ci))
            at = chart_text.find(cls, at + len(cls))
    return sorted(out)


def class_weights(counts: dict[str, int], power: float = 1.0) -> dict[str, float]:
    """Inverse-frequency weight per chart class, normalised so the *mean weight over charted
    slots* is 1.0 — i.e. the total chart gradient is unchanged and only its distribution over
    classes moves. Without that normalisation, turning the lever on would also silently scale
    the chart segment's contribution against the report's."""
    total = sum(counts.values())
    present = {c: max(counts.get(c, 0), 1) for c in CHART_CLASSES}
    raw = {c: (total / n) ** power for c, n in present.items()}
    mean = sum(raw[c] * counts.get(c, 0) for c in CHART_CLASSES) / max(total, 1)
    return {c: raw[c] / mean for c in CHART_CLASSES}


def chart_class_counts(records, chart_state) -> dict[str, int]:
    """Class frequencies over the training split's charted slots (the un-augmented reading;
    the mirror is quadrant-symmetric so it does not move class frequencies)."""
    counts = {c: 0 for c in CHART_CLASSES}
    for r in records:
        rec = r.get("record")
        if rec is None:
            continue
        for cls in chart_state(rec).values():
            counts[cls] += 1
    return counts


def token_weights(texts: list[str], span: tuple[int, int], occ: list[tuple[int, int, int]],
                  chart_weight: float, cls_w: dict[str, float] | None) -> torch.Tensor:
    """Per-token loss weight over one supervised run.

    Report tokens weigh 1.0. Chart tokens weigh ``chart_weight``, times the class weight when
    ``cls_w`` is given — applied only to the tokens that actually spell the class name, not to
    the FDI code or the separators, so the up-weighting lands on the decision and not on the
    scaffolding around it."""
    w = torch.ones(len(texts), dtype=torch.float32)
    start, end = span
    w[start:end] = chart_weight
    if cls_w is None:
        return w
    for lo, hi, ci in occ:
        w[lo:hi] = chart_weight * cls_w[CHART_CLASSES[ci]]
    return w


def class_id_tensor(n_tokens: int, occ: list[tuple[int, int, int]]) -> torch.Tensor:
    """int64[n_tokens]: the class index at each occurrence's FIRST token, -1 everywhere else.
    Rides in the batch so the trainer can log per-class chart recall without re-tokenizing."""
    ids = torch.full((n_tokens,), -1, dtype=torch.long)
    for lo, _hi, ci in occ:
        ids[lo] = ci
    return ids


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy with a per-position weight, matching HF's shift and ignore_index.

    Only supervised positions are gathered before the CE, so the float32 log-softmax is over a
    few hundred rows rather than the full sequence — the whole reason this can run inside the
    same VRAM budget as the unweighted path."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_w = weights[:, 1:]
    sel = shift_labels != -100
    if not sel.any():
        return logits.sum() * 0.0
    lg = shift_logits[sel].float()
    lb = shift_labels[sel]
    ww = shift_w[sel]
    loss = F.cross_entropy(lg, lb, reduction="none")
    return (loss * ww).sum() / ww.sum()
