"""Chart-v3 non-tooth loss — the VOCABULARY round's half of `chart_loss_v2`.

New module, and `chart_loss_v2.py` is **untouched**: three scored runs (`chartv2_pano_v1`,
`chartv2_pano_nt_v1`, `chartv2_fiveviews_nt_v1`) depend on it byte-for-byte, and the same rule
that kept `chart_loss.py` frozen when v2 landed applies here. Everything shared — the tokenizer
plumbing, the tooth vocabulary, `weighted_ce`, the span scanner's primitives — is IMPORTED from
v2 rather than restated, so the two cannot drift.

Two changes, both consequences of `data_canonical/chart_v3.py`'s measurement (v2's mixed IAN
enum agrees 38%; split into `course` + `position` and restricted to pairs where both annotators
claimed the axis it agrees 97.8% / 87.2%):

1. **`IAN` is two fields per side**, so `IAN.course.left` and `IAN.position.left` are weighted
   and scored independently. v2 had `IAN.left` alone and could not represent a report that
   asserted both facts.

2. **Silence is MASKED, not trained.** Every non-tooth field gains the value `n/a`, and its
   weight is forced to **0.0** after normalisation. `weighted_ce` divides by `ww.sum()`, so a
   zero weight removes the position from both the numerator and the denominator — an exact mask
   with no new mechanism. This is the half that matters most: `Sinuses` is 76% `not_mentioned`
   and `FOV.coronoid_processes` disagrees `excluded|not_mentioned` on 93 annotator pairs against
   **3** `excluded|excluded`, so under v2 the model was being taught to predict *whether an
   annotator would bother to write something down*, which is not a property of the image.

⚠ `irregular` is absent from `CANAL_COURSE` by design — no annotator ever asserts a non-regular
course, and a class with zero training examples is worse than an absent one. The course axis is
therefore `regular | not_assessable | n/a`, which is honest about what the corpus carries.

⚠ The mask classes still appear in `NONTOOTH_NAMES` and in the per-field recall log. That is
deliberate: a masked class whose *recall* is invisible would hide the failure mode where the
model emits `n/a` everywhere. It is excluded from the gradient, not from the readout.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toothfairy.chart_fmt_v3 import (  # noqa: E402
    BONE_FLAGS, CANAL_COURSE, CANAL_POSITION, FOV_REGIONS, FOV_STATUS, FOV_SURFACE,
    IAN_COURSE_PREFIX, IAN_POSITION_PREFIX, MASKED, SINUS_STATUS, SIDES, enum_surface,
)
# shared machinery — imported, never restated
from chart_loss_v2 import (  # noqa: E402
    BONE_NONE, SURFACE, TokenTexts, _items_in_list, _lines, _token_range, class_weights,
    nt_name, segment_offsets, supervised_span, weighted_ce,     # noqa: F401
)

MASK_VALUE = MASKED                      # "n/a" — spelled by the renderer, not re-derived

NONTOOTH_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    tuple((f"FOV.{r}", FOV_STATUS + (MASK_VALUE,)) for r in FOV_REGIONS)
    + tuple((f"IAN.course.{s}", CANAL_COURSE + (MASK_VALUE,)) for s in SIDES)
    + tuple((f"IAN.position.{s}", CANAL_POSITION + (MASK_VALUE,)) for s in SIDES)
    + (("Sinuses", SINUS_STATUS + (MASK_VALUE,)),
       ("Bone", BONE_FLAGS + (BONE_NONE,)))          # Bone has no silence surface (as in v2)
)
NONTOOTH_FIELD_VOCAB: dict[str, tuple[str, ...]] = dict(NONTOOTH_FIELDS)

# Four-character log labels. Written out, not truncated: every mechanical abbreviation of these
# names collides somewhere, and a collided label silently averages two fields into one number.
NONTOOTH_FIELD_ABBR: dict[str, str] = {
    "FOV.mandible": "mand", "FOV.maxilla": "maxl", "FOV.maxillary_sinuses": "msin",
    "FOV.mandibular_condyles": "cond", "FOV.coronoid_processes": "coro",
    "IAN.course.left": "icoL", "IAN.course.right": "icoR",
    "IAN.position.left": "iposL", "IAN.position.right": "iposR",
    "Sinuses": "sinu", "Bone": "bone",
}

NONTOOTH_CLASSES: tuple[tuple[str, str], ...] = tuple(
    (f, v) for f, vocab in NONTOOTH_FIELDS for v in vocab)
NONTOOTH_NAMES: tuple[str, ...] = tuple(nt_name(f, v) for f, v in NONTOOTH_CLASSES)
NONTOOTH_INDEX: dict[str, int] = {n: i for i, n in enumerate(NONTOOTH_NAMES)}
N_NONTOOTH = len(NONTOOTH_NAMES)
MASK_INDICES: frozenset[int] = frozenset(
    i for i, (_, v) in enumerate(NONTOOTH_CLASSES) if v == MASK_VALUE)

CHART_CLASS_NAMES: tuple[str, ...] = SURFACE + NONTOOTH_NAMES
NONTOOTH_WEIGHT_CAP = 8.0

# surface -> key, per vocabulary. `n/a` maps to itself so the scanner can FIND the masked spans;
# they are located precisely so they can be zeroed, which is the opposite of ignoring them.
_MASK_MAP = {MASK_VALUE: MASK_VALUE}
_FOV_VALUE = {enum_surface(v): v for v in FOV_STATUS} | _MASK_MAP
_COURSE_VALUE = {enum_surface(v): v for v in CANAL_COURSE} | _MASK_MAP
_POSITION_VALUE = {enum_surface(v): v for v in CANAL_POSITION} | _MASK_MAP
_SINUS_VALUE = {enum_surface(v): v for v in SINUS_STATUS} | _MASK_MAP
_BONE_VALUE = {enum_surface(v): v for v in BONE_FLAGS + (BONE_NONE,)}
_FOV_FIELD = {FOV_SURFACE[r]: f"FOV.{r}" for r in FOV_REGIONS}
_COURSE_FIELD = {s: f"IAN.course.{s}" for s in SIDES}
_POSITION_FIELD = {s: f"IAN.position.{s}" for s in SIDES}


def nontooth_items(text: str) -> list[tuple[str, str, int, int]]:
    """Every non-tooth statement in a rendered v3 chart -> ``(field, value, lo_char, hi_char)``.

    Eleven entries for a chart whose `Bone:` line carries one token, twelve when it carries two.
    ⚠ `IAN position:` must be tested BEFORE `IAN course:` cannot match it — both are checked by
    full prefix, so ordering is safe either way, but the prefixes are imported from the renderer
    so a format change cannot leave this scanner pointing at the wrong characters.
    """
    out: list[tuple[str, str, int, int]] = []
    for line, at in _lines(text):
        if line.startswith("FOV:"):
            out += _items_in_list(line, at, 4, _FOV_FIELD, _FOV_VALUE)
        elif line.startswith(IAN_COURSE_PREFIX):
            out += _items_in_list(line, at, len(IAN_COURSE_PREFIX), _COURSE_FIELD, _COURSE_VALUE)
        elif line.startswith(IAN_POSITION_PREFIX):
            out += _items_in_list(line, at, len(IAN_POSITION_PREFIX), _POSITION_FIELD,
                                  _POSITION_VALUE)
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
    """``[(first_token, last_token_exclusive, nontooth_class_index), ...]`` over v3's vocabulary."""
    start, end = span
    text, starts = segment_offsets(texts)
    chart_end_char = starts[end] if end < len(starts) else len(text)
    out: list[tuple[int, int, int]] = []
    for field, value, lo, hi in nontooth_items(text[:chart_end_char]):
        toks = _token_range(texts, starts, lo, hi, start, end)
        if toks:
            out.append((toks[0], toks[-1] + 1, NONTOOTH_INDEX[nt_name(field, value)]))
    return sorted(out)


def nontooth_class_counts(case_ids, chart_text_fn) -> dict[str, int]:
    """Occurrences of each non-tooth class over the training charts."""
    counts = {n: 0 for n in NONTOOTH_NAMES}
    for cid in case_ids:
        for field, value, _, _ in nontooth_items(chart_text_fn(cid)):
            n = nt_name(field, value)
            if n in counts:
                counts[n] += 1
    return counts


def nontooth_class_weights(counts: dict[str, int], power: float = 0.5,
                           cap: float = NONTOOTH_WEIGHT_CAP) -> dict[str, float]:
    """Inverse-frequency weights, normalised **per field**, with every mask class at 0.0.

    Order of operations is load-bearing and matches non-tooth weighting: cap the inverse-frequency RATIO
    before normalising (normalisation only ever shrinks it, so "no class carries more than
    `cap`x the mean" holds exactly), then zero the masks, then renormalise over what is left so
    the total non-tooth gradient is unchanged by the masking.
    """
    w: dict[str, float] = {}
    for field, vocab in NONTOOTH_FIELDS:
        live = [v for v in vocab if v != MASK_VALUE]
        sub = {v: counts.get(nt_name(field, v), 0) for v in live}
        present = {v: c for v, c in sub.items() if c > 0}
        if not present:
            for v in vocab:
                w[nt_name(field, v)] = 0.0
            continue
        raw = {v: (1.0 / c) ** power for v, c in present.items()}
        lo = min(raw.values())
        raw = {v: min(x / lo, cap) for v, x in raw.items()}          # cap the RATIO
        mean = sum(raw.values()) / len(raw)
        for v in vocab:
            n = nt_name(field, v)
            w[n] = 0.0 if (v == MASK_VALUE or v not in raw) else raw[v] / mean
    return w


def token_weights(texts: list[str], span: tuple[int, int],
                  occ: list[tuple[int, int, int]],
                  cls_w: dict[str, float] | None,
                  nt_occ: list[tuple[int, int, int]] = (),
                  nt_w: dict[str, float] | None = None):
    """v2's `token_weights` with v3's non-tooth NAME table.

    ⚠ This function exists only because v2's resolves a non-tooth index through its own
    module-global `NONTOOTH_NAMES`. Feeding it v3 occurrences (whose indices address v3's
    `NONTOOTH_CLASSES`) silently addressed the wrong vocabulary and raised
    `KeyError: 'FOV.maxillary_sinuses=not_mentioned'` inside a DataLoader worker — caught by the
    first smoke.

    The tooth half is v2's `SURFACE` unchanged, so tooth weighting is bit-identical.
    """
    import torch
    w = torch.ones(len(texts), dtype=torch.float32)
    if cls_w is not None:
        for lo, hi, ci in occ:
            w[lo:hi] = cls_w[SURFACE[ci]]
    if nt_w is not None:
        for lo, hi, ci in nt_occ:
            w[lo:hi] = nt_w[NONTOOTH_NAMES[ci]]      # v3's table — the whole point
    return w


def _selftest() -> None:
    """Gates. Run by `preflight_chartv3.py`; kept here so the module is self-checking."""
    from toothfairy.chart_fmt_v3 import FDI_ORDER, chart_block

    slots = {f: [] for f in FDI_ORDER}
    txt = chart_block(slots,
                      fov={"mandible": "included", "maxilla": "partial",
                           "maxillary_sinuses": None, "mandibular_condyles": "excluded",
                           "coronoid_processes": None},
                      ian_course={"left": "regular", "right": None},
                      ian_position={"left": "lingual", "right": "buccal"},
                      sinuses=None, bone=["atrophy"], collapse_healthy=False)

    items = nontooth_items(txt)
    got = {(f, v) for f, v, _, _ in items}
    assert ("IAN.course.left", "regular") in got
    assert ("IAN.position.left", "lingual") in got
    assert ("IAN.course.right", MASK_VALUE) in got, "a masked span must still be FOUND"
    assert ("Sinuses", MASK_VALUE) in got
    assert len(items) == 11, f"expected 11 non-tooth items, got {len(items)}: {got}"

    # every span must point at the characters it claims — the failure this catches is weighting
    # the wrong tokens, which is invisible in the loss curve
    for f, v, lo, hi in items:
        assert txt[lo:hi] == enum_surface(v) or txt[lo:hi] == v, (f, v, repr(txt[lo:hi]))

    # masks weigh exactly zero; live classes are normalised to mean 1.0 per field
    counts = {n: 10 for n in NONTOOTH_NAMES}
    counts[nt_name("Sinuses", "opacified")] = 1
    w = nontooth_class_weights(counts)
    assert all(w[NONTOOTH_NAMES[i]] == 0.0 for i in MASK_INDICES), "mask must be 0.0"
    for field, vocab in NONTOOTH_FIELDS:
        live = [w[nt_name(field, v)] for v in vocab if v != MASK_VALUE
                and counts.get(nt_name(field, v), 0) > 0]
        assert abs(sum(live) / len(live) - 1.0) < 1e-9, (field, live)
    assert w[nt_name("Sinuses", "opacified")] > w[nt_name("Sinuses", "clear")], "rarer -> heavier"

    # no value is a substring of another within a field (span scanning would mislabel)
    for field, vocab in NONTOOTH_FIELDS:
        for a in vocab:
            for b in vocab:
                assert a == b or enum_surface(a) not in enum_surface(b), (field, a, b)
    print(f"chart_loss_v3 selftest OK — {N_NONTOOTH} classes, {len(MASK_INDICES)} masked")


if __name__ == "__main__":
    _selftest()
