"""Multilabel inverse-frequency token weighting for the chart-v2 target.

Kept in **token space**, not a sigmoid head: the token
path is what produced chart<->report consistency 1.000, it is already built, and swapping in a
head would be a second change in a round that already has one.

**What is different from `chart_loss.py` (v1), and why this is a new file.**

  * v1's target was ``chart + delimiter + prose`` and its whole first job was finding that
    boundary so the two halves could be weighted separately. **The v2 target is the chart
    alone**, so there is no boundary and ``--chart-loss-weight`` has nothing to trade off.
  * v1 had one class per slot. v2 has a **set**, so one slot contributes to several counts and a
    slot's tokens can carry several different weights.
  * The class vocabulary is the 8 multilabel classes **plus `healthy` and `n/a`**, which are not
    decoration — they are the two commonest surface forms in the target and the ones a naive
    unweighted run would learn to emit unconditionally.

v1 is left untouched so `fiveviews_chartqa_v1` stays reproducible.

**The imbalance this exists to fix**, counted over the real targets (622 cases):

    n/a 9446   healthy 5124   absent 2675   filling 1113   endo 887
    crown 608  lesion 453     impacted 363  implant 280    root remnant 76

`n/a` outnumbers `root remnant` by **124x**. Unweighted, the cheapest policy is to emit `n/a`
and `healthy` everywhere — which is precisely the all-`present` collapse aux-head smoke escaped in
v1, in a new costume.

**Normalisation is load-bearing.** Weights are scaled so the *mean weight over class-token
occurrences is 1.0*: the total gradient the chart contributes is unchanged and only its
distribution over classes moves. Without it, turning the lever on would also silently rescale
the chart against everything else in the batch and no A/B would be interpretable.

⚠ **`n/a` is a mask marker, not a finding.** It is weighted like any other class here because
the model must still learn to emit it, but a per-class recall of `n/a` is a statement about
FOV reasoning, never about perception. Do not read it as a tooth metric.

────────────────────────────────────────────────────────────────────────────────────────────
**The four NON-TOOTH lines are weighted too.**

Until now `SURFACE` was the 8 tooth labels + `n/a` and deliberately stopped there, so a
whole-chart scan could not mis-weight a FOV value. Correct as a safety property, and it meant
`FOV:` / `IAN:` / `Sinuses:` / `Bone:` trained **unweighted** beside tooth findings carrying up
to 7.38 — and they collapsed to their majority value. Measured on `chartv2_pano_v1` ep7's test
charts: `Sinuses mucosal_thickening` gold 6 -> predicted **0**, `FOV.maxillary_sinuses included`
gold 7 -> predicted **0**. That is aux-head all-`present` collapse in a fourth costume.

**The class identity is `(field, value)`, never the value alone.** `included` is 83% of
`FOV.mandible` and a minority of `FOV.maxillary_sinuses`; pooling them would under-weight it
exactly where it is rare. The scan is therefore **per line, per item, by the item's own
character span** — no substring search, so cross-field theft is impossible by construction
rather than by a collision assertion. (The tooth scan is now restricted to the `Q` lines for the
same reason; it was already collision-free, and the self-test asserts the restriction changed
nothing.)

**Two groups, two normalisations.** Tooth and non-tooth occurrences are each normalised to mean
weight 1.0, and the non-tooth half is normalised **per field**, which implies the group law (a
mean of per-field means, each exactly 1.0). Without the split, arming this lever would silently
re-balance teeth against non-teeth and no A/B against ep7 would be interpretable.
`--chart-v2-weight-power` stays the tooth knob; `--nontooth-weight-power` is the new one.

⚠ **A cap, which the tooth side did not need.** `FOV.mandible excluded` occurs in **1** train
case: plain inverse-sqrt would hand it ~15x the mean gradient of the whole non-tooth block.
`NONTOOTH_WEIGHT_CAP` bounds the ratio before normalisation, so no non-tooth class carries more
than that multiple of the mean — set to 8.0, just above the tooth side's realised 7.38, so the
two groups keep the same dynamic range.

⚠⚠ **The IAN field is NOT a harmonised enum, and this lever will chase that noise. Measured
2026-08-14 over the 367 multi-report cases (`python data_canonical/chart_v2.py --nontooth`):**

  * annotators agree on `IAN.left` only **37%** of the time (`right` 39%), against 65-93% on
    every other field. The largest cell in the whole matrix is `lingual|regular` (**180
    pairs**) — larger than `regular|regular` (140) or `lingual|lingual` (105).
  * the cause is in the vocabulary, not the annotators: `CANAL_COURSE` mixes **two axes**.
    `regular` describes the canal's *course*, `lingual`/`buccal`/`apico_lingual` its
    *bucco-lingual position*, and the two are not exclusive — the notes under `course=lingual`
    read "**regular course**, predominantly lingual".
  * `apico_lingual` is a refinement of `lingual`, used interchangeably: 39 cross-pairs against
    **8** `apico_lingual|apico_lingual` self-pairs.

No weight fixes that, because the target itself is ambiguous — fixing it means changing
`CANAL_COURSE`, i.e. the format, the parser, the prose template and the metric, which is a
vocabulary round and not this one. **Read any IAN movement in non-tooth weighting against a 37% ceiling on
annotator agreement**, and prefer `FOV.maxilla` / `Bone.atrophy` (65% / 85% agreement, both
near-balanced) as the fields where a real signal is legible.

⚠ **`not_mentioned` is a reporting-behaviour class, not a perception class** on three fields:
`FOV.coronoid_processes` disagrees `excluded|not_mentioned` on **93** pairs against 3
`excluded|excluded` self-pairs, `FOV.maxillary_sinuses` on 81. Up-weighting its complement
teaches the model to guess which annotator wrote the report. It is weighted here because it is
in the target and the pre-registered metric scores it — which is a reason to read the per-field
numbers, not to trust their mean.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from toothfairy.chart_fmt_v2 import (
    BONE_FLAGS, CANAL_COURSE, FOV_REGIONS, FOV_STATUS, FOV_SURFACE, LABELS, MASKED,
    SINUS_STATUS, enum_surface,
)

# Surface spellings, longest first so a scan cannot match a prefix of a longer name. These are
# what actually appear in the target text; `LABELS` holds the internal keys.
#
# ⚠ `healthy` was removed 2026-08-13. The target now spells the empty label set `n/a`, the same
# as an unsupervised slot, so nothing in a target says "healthy" and a class kept here would
# count zero forever while still taking a share of the normalisation. `n/a` absorbs both:
# 9288 masked + 5124 healthy = ~14.4k of 19904 slots, so it is now ~73% of the vocabulary and
# its weight drops accordingly — which is the whole reason this vector is recomputed from the
# rendered corpus at launch rather than hardcoded.
SURFACE: tuple[str, ...] = tuple(c.replace("_", " ") for c in LABELS) + (MASKED,)
SURFACE_INDEX: dict[str, int] = {c: i for i, c in enumerate(SURFACE)}
N_SURFACE = len(SURFACE)

# ── the non-tooth vocabulary ─────────────────────────────────────────────
# A field is one statement the chart makes about something other than a tooth, and a class is a
# (field, value) pair. Ten fields — the same ten pre-registers a majority
# floor for — except that `Bone` is ONE field here rather than two booleans, because at the
# token level that is what it is: the line spells `atrophy`, `periodontitis`, both, or `none`.
#
# ⚠ The negative of a bone flag has NO TOKEN. "Bone: atrophy" says nothing about periodontitis;
# a weight can only land on characters that exist, so `Bone.periodontitis == False` is
# unweightable by construction. `none` (neither flag) is the one negative that is spelled, and
# it is weighted. Say this in the write-up rather than reporting a bone weight that looks like
# it covers both directions.
BONE_NONE = "none"

NONTOOTH_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    tuple((f"FOV.{r}", FOV_STATUS) for r in FOV_REGIONS)
    + (("IAN.left", CANAL_COURSE), ("IAN.right", CANAL_COURSE),
       ("Sinuses", SINUS_STATUS), ("Bone", BONE_FLAGS + (BONE_NONE,)))
)
NONTOOTH_FIELD_VOCAB: dict[str, tuple[str, ...]] = dict(NONTOOTH_FIELDS)

# Four-character labels for the trainer's log line. Written out rather than truncated, because
# every mechanical abbreviation of these nine names collides somewhere (`FOV.maxilla` with
# `FOV.maxillary_sinuses`, `FOV.mandible` with `FOV.mandibular_condyles`) and a collided label
# silently averages two fields into one number.
NONTOOTH_FIELD_ABBR: dict[str, str] = {
    "FOV.mandible": "mand", "FOV.maxilla": "maxl", "FOV.maxillary_sinuses": "msin",
    "FOV.mandibular_condyles": "cond", "FOV.coronoid_processes": "coro",
    "IAN.left": "ianL", "IAN.right": "ianR", "Sinuses": "sinu", "Bone": "bone",
}


def nt_name(field: str, value: str) -> str:
    """The stable name of one non-tooth class. Used as a dict key, a W&B key and a log label,
    so it must never be derived twice."""
    return f"{field}={value}"


NONTOOTH_CLASSES: tuple[tuple[str, str], ...] = tuple(
    (f, v) for f, vocab in NONTOOTH_FIELDS for v in vocab)
NONTOOTH_NAMES: tuple[str, ...] = tuple(nt_name(f, v) for f, v in NONTOOTH_CLASSES)
NONTOOTH_INDEX: dict[str, int] = {n: i for i, n in enumerate(NONTOOTH_NAMES)}
N_NONTOOTH = len(NONTOOTH_NAMES)

# The trainer's per-class recall vocabulary: teeth first, so an index below N_SURFACE means the
# same thing it did before this change and the ep7 W&B keys stay comparable.
CHART_CLASS_NAMES: tuple[str, ...] = SURFACE + NONTOOTH_NAMES

# ⚠ Set against the tooth side's realised maximum (root remnant, 7.38 at power=0.5) so the two
# groups have the same dynamic range. Applied to the inverse-frequency ratio BEFORE the
# normalisation, which only ever shrinks it — so the guarantee "no non-tooth class carries more
# than CAPx the mean non-tooth gradient" holds exactly.
NONTOOTH_WEIGHT_CAP = 8.0

# Surface -> key, per vocabulary. Built once; `enum_surface` is the renderer's own spelling
# function, imported rather than re-implemented.
_FOV_VALUE = {enum_surface(v): v for v in FOV_STATUS}
_CANAL_VALUE = {enum_surface(v): v for v in CANAL_COURSE}
_SINUS_VALUE = {enum_surface(v): v for v in SINUS_STATUS}
_BONE_VALUE = {enum_surface(v): v for v in BONE_FLAGS + (BONE_NONE,)}
_FOV_FIELD = {FOV_SURFACE[r]: f"FOV.{r}" for r in FOV_REGIONS}
_IAN_FIELD = {"left": "IAN.left", "right": "IAN.right"}

# ── decode/offset machinery, copied verbatim from finetune_qwen_v1/chart_loss.py ────
# Copied rather than imported: `finetune_qwen_v2/` must stand alone. A cross-experiment import
# is a live dependency on a directory whose whole purpose is to stay frozen for reproducing
# `fiveviews_chartqa_v1`, and it broke the moment this file moved.


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
    """Concatenated text of a token run plus each token's start offset in it.

    Qwen2.5's tokenizer is byte-level BPE, so for an ASCII target this reconstruction is exact
    and the cumulative offsets give a token -> character map. The pre-flight asserts the
    reconstruction rather than assuming it."""
    starts, n = [], 0
    for t in texts:
        starts.append(n)
        n += len(t)
    return "".join(texts), starts


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy with a per-position weight, matching HF's shift and ignore_index.

    Only supervised positions are gathered before the CE, so the float32 log-softmax is over a
    few hundred rows rather than the full sequence — the whole reason this fits the same VRAM
    budget as the unweighted path."""
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


def supervised_span(texts: list[str]) -> tuple[int, int]:
    """The token range the chart occupies.

    v2 targets are chart-only, so this is the whole supervised run. The v1 delimiter is still
    honoured if present, which keeps the module usable for a chart+prose ablation without a
    second code path."""
    from toothfairy.chart_fmt_v2 import DELIMITER

    text, starts = segment_offsets(texts)
    at = text.find(DELIMITER)
    if at == -1:
        return 0, len(texts)
    end = 0
    while end < len(starts) and starts[end] < at:
        end += 1
    return 0, end


def class_occurrences(texts: list[str], span: tuple[int, int]
                      ) -> list[tuple[int, int, int]]:
    """``[(first_token, last_token_exclusive, class_index), ...]`` for every class name spelled
    inside the chart.

    ``first_token`` is where the decision happens: with the FDI code already decoded, the next
    token is the model choosing between `healthy` / `n/a` / a finding, and after a `+` it is
    choosing the next finding. Teacher-forced accuracy at those positions is per-class recall of
    the chart. The rest of the range exists for the weighting, which wants the whole word.

    Longest-first matching matters here in a way it did not in v1: `root remnant` contains no
    other class name, but scanning short names first would let `crown` inside a longer future
    class name steal the match.

    ⚠ A returned span is a **superset** of its class name, not exactly it. Token boundaries do
    not align with word boundaries — Qwen merges the multilabel `+` into the following word, so
    `absent+crown` yields a token `+crown` and the `crown` span carries that `+`. Weighting it
    is correct (that token *is* the decision to add another label), but any check on these spans
    must test containment, not equality. The pre-flight asserts exactly that.

    ⚠ **The scan is restricted to the `Q` lines**. It used to run over the
    whole chart, which was safe only because no tooth class name happens to hide inside the
    non-tooth vocabulary — a property asserted in the self-test but not enforced by the code,
    and one that a single new enum value would have quietly broken. The self-test asserts the
    restriction is a no-op on the shipped format, so `chartv2_pano_v1` stays reproducible."""
    start, end = span
    text, starts = segment_offsets(texts)
    chart_end_char = starts[end] if end < len(starts) else len(text)

    claimed: list[tuple[int, int]] = []          # character ranges already matched
    out: list[tuple[int, int, int]] = []
    for lo_line, hi_line in _quadrant_ranges(text[:chart_end_char]):
        for cls in sorted(SURFACE, key=len, reverse=True):
            ci = SURFACE_INDEX[cls]
            at = text.find(cls, lo_line, hi_line)
            while at != -1:
                hi = at + len(cls)
                if not any(a < hi and at < b for a, b in claimed):
                    claimed.append((at, hi))
                    toks = _token_range(texts, starts, at, hi, start, end)
                    if toks:
                        out.append((toks[0], toks[-1] + 1, ci))
                at = text.find(cls, hi, hi_line)
    return sorted(out)


# ── the non-tooth scan ───────────────────────────────────────────────────

def _token_range(texts: list[str], starts: list[int], lo_char: int, hi_char: int,
                 start: int, end: int) -> list[int]:
    """Token indices in ``[start, end)`` overlapping the character range ``[lo_char, hi_char)``."""
    return [j for j in range(start, end)
            if starts[j] < hi_char and starts[j] + len(texts[j]) > lo_char]


def _lines(text: str):
    """``(line, start_char)`` for every line of the chart, offsets into ``text``."""
    pos = 0
    for raw in text.split("\n"):
        yield raw, pos
        pos += len(raw) + 1


def _quadrant_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of the four ``Qn:`` lines — the only place a tooth label may live."""
    return [(at, at + len(line)) for line, at in _lines(text)
            if len(line) > 2 and line[0] == "Q" and line[1] in "1234" and line[2] == ":"]


def _items_in_list(line: str, line_at: int, head: int, fields: dict[str, str],
                   values: dict[str, str]) -> list[tuple[str, str, int, int]]:
    """One comma-separated, prefix-labelled line -> ``[(field, value, lo_char, hi_char)]``.

    The value's span is computed from the item's own layout — prefix, one space, value — never
    by searching the chart for the value's text. That is what makes cross-field theft
    impossible: `excluded` on the `Sinuses` line and `excluded` on the `FOV` line are different
    classes and are found by different items, not disambiguated after the fact."""
    out: list[tuple[str, str, int, int]] = []
    off = head
    for chunk in line[head:].split(","):
        c_at, off = off, off + len(chunk) + 1
        lead = len(chunk) - len(chunk.lstrip())
        item = chunk.strip()
        for surf in sorted(fields, key=len, reverse=True):
            if not item.startswith(surf + " "):
                continue
            v_surf = item[len(surf) + 1:].strip()
            v = values.get(v_surf)
            if v is not None:
                lo = line_at + c_at + lead + len(surf) + 1
                out.append((fields[surf], v, lo, lo + len(v_surf)))
            break
    return out


def nontooth_items(text: str) -> list[tuple[str, str, int, int]]:
    """Every non-tooth statement in a rendered chart -> ``(field, value, lo_char, hi_char)``.

    Nine entries for a chart whose `Bone:` line carries one token, ten when it carries two.
    Anything unrecognised is skipped rather than guessed at — this only ever runs on renderer
    output, and the self-test asserts full coverage over all 622 cases x both truncations, so a
    silent skip here is a test failure and not a training-time surprise."""
    out: list[tuple[str, str, int, int]] = []
    for line, at in _lines(text):
        if line.startswith("FOV:"):
            out += _items_in_list(line, at, 4, _FOV_FIELD, _FOV_VALUE)
        elif line.startswith("IAN:"):
            out += _items_in_list(line, at, 4, _IAN_FIELD, _CANAL_VALUE)
        elif line.startswith("Sinuses:"):
            body = line[8:]
            lead = len(body) - len(body.lstrip())
            v = _SINUS_VALUE.get(body.strip())
            if v is not None:
                lo = at + 8 + lead
                out.append(("Sinuses", v, lo, lo + len(body.strip())))
        elif line.startswith("Bone:"):
            off = 5
            for chunk in line[5:].split(","):
                c_at, off = off, off + len(chunk) + 1
                lead = len(chunk) - len(chunk.lstrip())
                v = _BONE_VALUE.get(chunk.strip())
                if v is not None:
                    lo = at + c_at + lead
                    out.append(("Bone", v, lo, lo + len(chunk.strip())))
    return out


def nontooth_occurrences(texts: list[str], span: tuple[int, int]
                         ) -> list[tuple[int, int, int]]:
    """``[(first_token, last_token_exclusive, nontooth_class_index), ...]``.

    Same shape as ``class_occurrences`` but indexing ``NONTOOTH_CLASSES``; the two lists are
    kept apart so a caller cannot confuse the two vocabularies. ``first_token`` is again the
    decision position — with `mandible ` already decoded, the next token is the model choosing
    the region's status."""
    start, end = span
    text, starts = segment_offsets(texts)
    chart_end_char = starts[end] if end < len(starts) else len(text)
    out: list[tuple[int, int, int]] = []
    for field, value, lo, hi in nontooth_items(text[:chart_end_char]):
        toks = _token_range(texts, starts, lo, hi, start, end)
        if toks:
            out.append((toks[0], toks[-1] + 1, NONTOOTH_INDEX[nt_name(field, value)]))
    return sorted(out)


def class_weights(counts: dict[str, int], power: float = 1.0) -> dict[str, float]:
    """Inverse-frequency weight per surface class, normalised so the **mean weight over class
    occurrences is 1.0** — the chart's total gradient is unchanged, only its distribution moves.

    ``power`` softens the ratio: 1.0 is plain inverse frequency (a 124x spread here), 0.5 is
    inverse square root. Sweep it if the run over-corrects onto the rare classes; it is the one
    knob, and its value belongs in the launcher output."""
    present = {c: max(counts.get(c, 0), 1) for c in SURFACE}
    total = sum(present.values())
    raw = {c: (total / n) ** power for c, n in present.items()}
    mean = sum(raw[c] * counts.get(c, 0) for c in SURFACE) / max(sum(counts.values()), 1)
    return {c: raw[c] / mean for c in SURFACE}


def chart_class_counts(case_ids, chart_text_fn) -> dict[str, int]:
    """Surface-class frequencies over the training split's rendered charts.

    Counted off the **rendered target text**, not off the label structure, so the numbers are
    exactly what the loss will see — including `healthy` and `n/a`, which have no entry in the
    label vocabulary at all. The augmentation mirror augmentation is quadrant-symmetric and does not move
    class frequencies, so the un-augmented reading is the right one."""
    counts = {c: 0 for c in SURFACE}
    for cid in case_ids:
        text = chart_text_fn(cid)
        for line in text.splitlines():
            if not line.startswith("Q") or ":" not in line:
                continue                          # findings live on the quadrant lines only
            body = line.split(":", 1)[1]
            for cell in body.split(","):
                cell = cell.strip()
                if " " not in cell:
                    continue
                val = cell.split(" ", 1)[1].strip()
                if val == MASKED:
                    counts[val] += 1
                else:
                    for part in val.split("+"):
                        part = part.strip()
                        if part in counts:
                            counts[part] += 1
    return counts


def nontooth_class_counts(case_ids, chart_text_fn) -> dict[str, int]:
    """``{nt_name: occurrences}`` over the split's rendered charts.

    Counted with the same scanner the loss uses, off the same rendered text, so the printed
    weight vector cannot describe a different corpus from the one being weighted. Values that
    never occur are present with 0 — a zero here is a fact about the split (`FOV.mandible
    not_assessable` is one), not a missing key."""
    counts = {n: 0 for n in NONTOOTH_NAMES}
    for cid in case_ids:
        for field, value, _lo, _hi in nontooth_items(chart_text_fn(cid)):
            counts[nt_name(field, value)] += 1
    return counts


def nontooth_class_weights(counts: dict[str, int], power: float = 0.5,
                           cap: float = NONTOOTH_WEIGHT_CAP) -> dict[str, float]:
    """Inverse-frequency weight per **(field, value)** class.

    Two properties, both asserted in the self-test:

      * **normalised per field** — the mean weight over a field's own occurrences is 1.0, so
        every field contributes the same total gradient whatever its entropy, which is what the
        pre-registered 10-field mean macro-F1 rewards. It also implies the group law: the mean
        over *all* non-tooth occurrences is 1.0 too, so this lever moves gradient inside the
        non-tooth block and never between the block and the teeth.
      * **capped** — see ``NONTOOTH_WEIGHT_CAP``. Bounding the ratio before normalisation makes
        the bound hold after it, because the normaliser is never below 1."""
    w: dict[str, float] = {}
    for field, vocab in NONTOOTH_FIELDS:
        names = [nt_name(field, v) for v in vocab]
        n = {k: counts.get(k, 0) for k in names}
        total = sum(n.values())
        if total == 0:                            # a field absent from this split: stay neutral
            w.update({k: 1.0 for k in names})
            continue
        raw = {k: min((total / max(n[k], 1)) ** power, cap) for k in names}
        mean = sum(raw[k] * n[k] for k in names) / total
        w.update({k: raw[k] / mean for k in names})
    return w


def token_weights(texts: list[str], span: tuple[int, int],
                  occ: list[tuple[int, int, int]],
                  cls_w: dict[str, float] | None,
                  nt_occ: list[tuple[int, int, int]] = (),
                  nt_w: dict[str, float] | None = None) -> torch.Tensor:
    """Per-token loss weight over one supervised run.

    Scaffolding — the FDI codes, separators, headers, the region names on the `FOV:` line —
    stays at 1.0. Only the tokens that actually spell a class *value* are re-weighted, so the
    up-weighting lands on the decision and not on the frame around it.

    The two vocabularies are applied from two dicts and never merged: ``occ``/``cls_w`` index
    ``SURFACE``, ``nt_occ``/``nt_w`` index ``NONTOOTH_CLASSES``. Either may be armed alone,
    which is what makes the tooth-only run (`chartv2_pano_v1`) still reproducible."""
    w = torch.ones(len(texts), dtype=torch.float32)
    if cls_w is not None:
        for lo, hi, ci in occ:
            w[lo:hi] = cls_w[SURFACE[ci]]
    if nt_w is not None:
        for lo, hi, ci in nt_occ:
            w[lo:hi] = nt_w[NONTOOTH_NAMES[ci]]
    return w


def class_id_tensor(n_tokens: int, occ: list[tuple[int, int, int]],
                    nt_occ: list[tuple[int, int, int]] = ()) -> torch.Tensor:
    """int64[n_tokens]: the class index at each occurrence's FIRST token, -1 elsewhere. Rides in
    the batch so the trainer logs per-class chart recall without re-tokenizing.

    Non-tooth ids are offset by ``N_SURFACE``, i.e. they index ``CHART_CLASS_NAMES``. Teeth keep
    indices 0..9, so every `chart/recall_*` key logged before 2026-08-14 still means what it
    meant."""
    ids = torch.full((n_tokens,), -1, dtype=torch.long)
    for lo, _hi, ci in occ:
        ids[lo] = ci
    for lo, _hi, ci in nt_occ:
        ids[lo] = N_SURFACE + ci
    return ids


def _selftest() -> None:
    """Runs without a GPU or a tokenizer: the character machinery is what can silently rot."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "data_canonical"))
    from chart_v2 import all_case_ids, chart_text

    # ── no class name is a substring of another, which the longest-first scan assumes
    for a in SURFACE:
        for b in SURFACE:
            if a != b:
                assert a not in b, f"{a!r} is a substring of {b!r}"

    # ── no class name collides with the non-tooth vocabulary, so a whole-chart scan is safe
    from toothfairy.chart_fmt_v2 import CANAL_COURSE, FOV_STATUS, SINUS_STATUS, BONE_FLAGS
    other = {v.replace("_", " ") for v in FOV_STATUS + CANAL_COURSE + SINUS_STATUS + BONE_FLAGS}
    for c in SURFACE:
        for o in other:
            assert c not in o, f"class {c!r} hides inside non-tooth value {o!r}"

    ids = all_case_ids()
    counts = chart_class_counts(ids, chart_text)

    # ── The counts the loss sees must be exactly the labels the merge produced. This is the
    # check that matters: the weighting is computed off the RENDERED TEXT, so a format change
    # that silently dropped or duplicated a label would move the loss without moving the merge.
    # (A multilabel slot contributes several occurrences, so the total exceeds the slot count.)
    from chart_v2 import LABELS as MERGE_LABELS, chart_v2 as merge_of
    from collections import Counter
    ref = Counter()
    n_slots = n_empty = n_masked = n_labelled = 0
    for cid in ids:
        m = merge_of(cid)
        for fdi, s in m["labels"].items():
            n_slots += 1
            sup = m["mask_status"][fdi] or m["mask_attr"][fdi]
            if not sup:
                n_masked += 1
            elif not s:
                n_empty += 1
            if s:
                n_labelled += 1
            for lab in s:
                ref[lab.replace("_", " ")] += 1
    for c in (x.replace("_", " ") for x in MERGE_LABELS):
        assert counts[c] == ref[c], f"{c}: loss sees {counts[c]}, merge produced {ref[c]}"
    # Since the healthy/n-a collapse (2026-08-13) `n/a` must absorb BOTH populations exactly.
    # An off-by-anything here means the format and the merge have drifted, which is the one
    # failure this whole check exists to catch.
    assert counts[MASKED] == n_empty + n_masked, \
        f"n/a {counts[MASKED]} != supervised-empty {n_empty} + masked {n_masked}"
    assert counts[MASKED] + n_labelled == n_slots, \
        "every slot must be n/a or carry >=1 label"

    w = class_weights(counts)
    # the normalisation law: mean weight over occurrences is 1.0
    mean = sum(w[c] * counts[c] for c in SURFACE) / sum(counts.values())
    assert abs(mean - 1.0) < 1e-9, f"weights not normalised: mean {mean}"
    # rarer class => larger weight, strictly
    ordered = sorted(SURFACE, key=lambda c: counts[c])
    for a, b in zip(ordered, ordered[1:]):
        if counts[a] != counts[b]:
            assert w[a] > w[b], f"{a} ({counts[a]}) should outweigh {b} ({counts[b]})"

    # ── character-level occurrence scan, on a real target split into single-char "tokens"
    text = chart_text(ids[0])
    toks = list(text)
    span = supervised_span(toks)
    occ = class_occurrences(toks, span)
    for lo, hi, ci in occ:
        assert "".join(toks[lo:hi]) == SURFACE[ci], \
            f"span {lo}:{hi} is {''.join(toks[lo:hi])!r}, expected {SURFACE[ci]!r}"
    assert len(occ) >= 32, f"expected >=32 class occurrences in a chart, got {len(occ)}"

    tw = token_weights(toks, span, occ, w)
    assert tw.shape[0] == len(toks)
    assert abs(float(tw[occ[0][0]]) - w[SURFACE[occ[0][2]]]) < 1e-6
    ids_t = class_id_tensor(len(toks), occ)
    assert int((ids_t >= 0).sum()) == len(occ)

    print(f"chart_loss_v2 self-test: OK ({len(ids)} cases, {n_slots} slots, "
          f"{sum(counts.values())} class occurrences)")

    # ══ the non-tooth half ═══════════════════════════════════════════════════════
    from chart_v2 import load_case_records, merge_nontooth

    # ── the tooth scan is now Q-line-only. Assert that is a NO-OP on the shipped format, so
    # `chartv2_pano_v1` remains reproducible: no tooth surface may occur outside a Q line.
    n_outside = 0
    for cid in ids:
        t = chart_text(cid)
        qr = _quadrant_ranges(t)
        for cls in SURFACE:
            at = t.find(cls)
            while at != -1:
                n_outside += not any(a <= at and at + len(cls) <= b for a, b in qr)
                at = t.find(cls, at + len(cls))
    assert n_outside == 0, (f"{n_outside} tooth-class matches outside the Q lines — the "
                            "non-tooth weighting whole-chart scan was mis-weighting the non-tooth lines")
    print("tooth scan restricted to the Q lines: OK (no-op on all 622 charts)")

    # ── every field is stated exactly once per chart, and the value is what the merge produced.
    # Same law as the tooth side: the loss reads RENDERED TEXT, so a format change that dropped
    # or renamed a value would move the weighting without moving the merge. This is the check.
    n_items = 0
    for cid in ids:
        for trunc in (False, True):
            t = chart_text(cid, superior_truncated=trunc)
            items = nontooth_items(t)
            n_items += len(items)
            seen = {}
            for field, value, lo, hi in items:
                assert t[lo:hi] == enum_surface(value), \
                    f"{cid}: span {lo}:{hi} is {t[lo:hi]!r}, expected {value!r}"
                assert value in NONTOOTH_FIELD_VOCAB[field], f"{cid}: {field}={value}"
                seen.setdefault(field, []).append(value)
            nt = merge_nontooth(load_case_records(cid))
            want = {f"FOV.{r}": [nt["fov"][r]] for r in FOV_REGIONS}
            want["IAN.left"] = [nt["ian"]["left"]]
            want["IAN.right"] = [nt["ian"]["right"]]
            want["Sinuses"] = [nt["sinuses"]]
            want["Bone"] = list(nt["bone"]) or [BONE_NONE]
            assert seen == want, f"{cid}: scan {seen} != merge {want}"
    print(f"non-tooth scan == merge: OK ({n_items} items over {len(ids)} cases x 2 truncations)")

    # ── counts, weights, and the two normalisation laws
    nt_counts = nontooth_class_counts(ids, chart_text)
    assert sum(nt_counts.values()) == sum(
        len(nontooth_items(chart_text(c))) for c in ids)
    ntw = nontooth_class_weights(nt_counts)
    tot_all = 0.0
    for field, vocab in NONTOOTH_FIELDS:
        names = [nt_name(field, v) for v in vocab]
        n_f = sum(nt_counts[k] for k in names)
        mean_f = sum(ntw[k] * nt_counts[k] for k in names) / n_f
        assert abs(mean_f - 1.0) < 1e-9, f"{field}: per-field mean weight {mean_f}"
        tot_all += sum(ntw[k] * nt_counts[k] for k in names)
        # rarer value => heavier, within the field, until the cap flattens the tail
        ordered = sorted(names, key=lambda k: nt_counts[k])
        for a, b in zip(ordered, ordered[1:]):
            if nt_counts[a] < nt_counts[b]:
                assert ntw[a] >= ntw[b], f"{a} ({nt_counts[a]}) lighter than {b}"
    assert abs(tot_all / sum(nt_counts.values()) - 1.0) < 1e-9, "group mean weight != 1.0"
    assert max(ntw.values()) <= NONTOOTH_WEIGHT_CAP + 1e-9, "the cap does not hold"
    print(f"non-tooth weights: OK (per-field mean 1.0, group mean 1.0, max "
          f"{max(ntw.values()):.2f}x <= cap {NONTOOTH_WEIGHT_CAP})")

    # ── character-level: every non-tooth span decodes back to exactly its own value, and no
    # class is claimed twice. Equality, not containment — these spans are computed from the
    # item's layout rather than searched for, so they are exact by construction.
    nt_occ = nontooth_occurrences(toks, span)
    for lo, hi, ci in nt_occ:
        field, value = NONTOOTH_CLASSES[ci]
        assert "".join(toks[lo:hi]) == enum_surface(value), \
            f"{field}: {''.join(toks[lo:hi])!r} != {value!r}"
    assert len(nt_occ) >= 9, f"expected >=9 non-tooth occurrences, got {len(nt_occ)}"
    # the two vocabularies never touch the same token
    assert not ({t for lo, hi, _ in occ for t in range(lo, hi)}
                & {t for lo, hi, _ in nt_occ for t in range(lo, hi)}), \
        "a token carries both a tooth and a non-tooth weight"
    both = class_id_tensor(len(toks), occ, nt_occ)
    assert int((both >= 0).sum()) == len(occ) + len(nt_occ)
    assert int(both.max()) < len(CHART_CLASS_NAMES)
    tw2 = token_weights(toks, span, occ, w, nt_occ, ntw)
    assert abs(float(tw2[nt_occ[0][0]]) - ntw[NONTOOTH_NAMES[nt_occ[0][2]]]) < 1e-6
    # arming one lever must not move the other
    assert torch.equal(token_weights(toks, span, occ, w, nt_occ, None), tw)
    print(f"non-tooth occurrence scan: OK ({len(nt_occ)} items, disjoint from the "
          f"{len(occ)} tooth spans)")

    # ── the table the launcher prints, and the majority floor beside it
    ntw10 = nontooth_class_weights(nt_counts, power=1.0)
    print(f"\n{'field':<28}{'value':<20}{'occ':>7}{'share':>8}{'w(p=1)':>8}{'w(p=.5)':>9}")
    floors = []
    for field, vocab in NONTOOTH_FIELDS:
        names = [nt_name(field, v) for v in vocab]
        n_f = sum(nt_counts[k] for k in names)
        floors.append(max(nt_counts[k] for k in names) / n_f)
        for v in sorted(vocab, key=lambda v: -nt_counts[nt_name(field, v)]):
            k = nt_name(field, v)
            if not nt_counts[k]:
                continue
            print(f"   {field:<25}{v:<20}{nt_counts[k]:>7}{nt_counts[k] / n_f:>8.1%}"
                  f"{ntw10[k]:>8.2f}{ntw[k]:>9.2f}")
    print("   (equal values in the w(p=1) column are the CAP binding, not a tie — plain "
          "inverse\n    frequency asks for 622x on a value attested by one case)")
    print(f"\nmajority-value share, mean over {len(NONTOOTH_FIELDS)} fields: "
          f"{sum(floors) / len(floors):.1%}  (per field: "
          f"{', '.join(f'{f:.0%}' for f in floors)})")
    print("   ^ the floor a collapsed model scores. ⚠ Not the same number as the\n"
          "     train-split 68.0%: that is over 10 fields on TRAIN, this is over 9 on all 622 cases,\n"
          "     because `Bone` is ONE field at token level (atrophy / periodontitis / none).\n"
          "     Check it against `eval/eval_chart_v2.py`'s non-tooth block,\n"
          "     not this file.")
    print(f"weight spread: p=1.0 {max(ntw10.values()) / min(ntw10.values()):.0f}x, "
          f"p=0.5 {max(ntw.values()) / min(ntw.values()):.0f}x "
          f"(capped at {NONTOOTH_WEIGHT_CAP})")
    print("\n⚠ IAN is not a harmonised enum — annotators agree 37%/39% and `regular` is a\n"
          "   different AXIS from `lingual`/`buccal`. See the module docstring, and\n"
          "   `python data_canonical/chart_v2.py --nontooth` for the matrices. Weighting it\n"
          "   harder makes the model chase that disagreement; read IAN against a 37% ceiling.")
    w50 = class_weights(counts, power=0.5)
    print(f"\n{'class':<14}{'occurrences':>12}{'w(p=1.0)':>10}{'w(p=0.5)':>10}")
    for c in sorted(SURFACE, key=lambda c: -counts[c]):
        print(f"   {c:<11}{counts[c]:>12}{w[c]:>10.2f}{w50[c]:>10.2f}")
    spread = max(w.values()) / min(w.values())
    spread50 = max(w50.values()) / min(w50.values())
    print(f"\nclass imbalance {max(counts.values()) / max(min(counts.values()), 1):.0f}x")
    print(f"   power=1.0 (plain inverse frequency): weight spread {spread:.0f}x")
    print(f"   power=0.5 (inverse sqrt)           : weight spread {spread50:.0f}x")
    print("\n⚠ `root remnant` has 76 occurrences corpus-wide. At power=1.0 its tokens carry "
          f"{w['root remnant']:.0f}x\n   the mean gradient, which is a lot of the chart's "
          "signal spent on 0.4% of its slots.\n   Launch at power=0.5 unless the smoke shows "
          "the rare classes collapsing; the value\n   belongs in the run config and the "
          "launcher output either way.")


if __name__ == "__main__":
    _selftest()
