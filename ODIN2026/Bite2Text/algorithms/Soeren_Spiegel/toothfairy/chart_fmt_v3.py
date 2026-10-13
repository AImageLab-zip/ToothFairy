"""Chart-v3 format + parser — the VOCABULARY round.

One change against `chart_fmt_v2`, on the non-tooth half only: **`IAN` becomes two fields, and
annotator silence becomes a mask marker instead of a trainable class.** The tooth half is
byte-identical in behaviour and is asserted so by `_selftest`.

WHY, measured 2026-08-15 over the 1000 per-report records (`data_canonical/chart_v3.py
--harmonise` reproduces it):

    current mixed enum                              38.0% annotator agreement  (n=786)
    split -> `course`,   both annotators claiming   97.8%                      (n=178)
    split -> `position`, both annotators claiming   87.2%                      (n=203)

`CANAL_COURSE` mixed two AXES that are not mutually exclusive — `regular` is the canal's
*course*, `lingual`/`buccal` its bucco-lingual *position*, and the notes under `course=lingual`
read "regular course, predominantly lingual". Whichever fact the extractor caught first became
the single label, so `lingual|regular` was the LARGEST cell in the agreement matrix (180 pairs,
against 140 `regular|regular`). **The 37% ceiling was an artifact of the schema, not of the
anatomy or of the annotators** — which is why no loss weight ever moved it.

Three consequences are baked into the format here:

  * **Two lines, `IAN course:` and `IAN position:`.** A report may now assert both, one, or
    neither, which is what the reports actually do.
  * **`apico_lingual` is GONE**, folded into `lingual`. Measured a synonym-in-use: 39 cross-pairs
    against 8 `apico_lingual|apico_lingual` self-pairs, and v2's `_CANAL_ORDER` resolved the pair
    toward the finer reading, asserting it on ~30 cases where the other annotator said `lingual`.
  * **`not_mentioned` is spelled `n/a` on EVERY non-tooth field and is a MASK, not a class.**
    It is a reporting-behaviour class, not a perception class: `FOV.coronoid_processes` disagrees
    `excluded|not_mentioned` on 93 pairs against **3** `excluded|excluded` self-pairs, and
    `Sinuses` is 76% `not_mentioned`. Training on it teaches the model to predict silence.
    `chart_loss_v2 --nontooth-mask-silence` drops these spans from the loss.

⚠ **`irregular` is deliberately NOT in the vocabulary.** No annotator ever asserts a non-regular
course — the corpus only ever says "regular" or says nothing. A class with zero training examples
is worse than an absent one, so `IAN course` is `regular | not_assessable | n/a` and the write-up
says why rather than shipping a class the data cannot support.

Stdlib only, zero `data_canonical` imports — same contract as v2, for the same reason (the
submission image does not vendor `data_canonical/`).

The format::

    Chart:
    Q1: 11 healthy, 12 healthy, ... 18 n/a
    Q2: ...
    Q3: ...
    Q4: ...
    FOV: mandible included, maxilla partial, maxillary sinuses excluded, condyles excluded, coronoid n/a
    IAN course: left regular, right n/a
    IAN position: left lingual, right n/a
    Sinuses: n/a
    Bone: atrophy
"""
from __future__ import annotations

import re

# The tooth half is UNCHANGED. Import it rather than restate it: a second copy of the label
# vocabulary is free to drift, and a drifted copy mislabels silently.
from .chart_fmt_v2 import (  # noqa: F401
    COLLAPSE_HEALTHY, FDI_ORDER, FDI_SET, HEALTHY, LABELS, LABEL_INDEX, MASKED,
    _SURFACE_LABEL, enum_surface, slot_text,
)

FOV_REGIONS: tuple[str, ...] = ("mandible", "maxilla", "maxillary_sinuses",
                                "mandibular_condyles", "coronoid_processes")
FOV_SURFACE = {"mandible": "mandible", "maxilla": "maxilla",
               "maxillary_sinuses": "maxillary sinuses",
               "mandibular_condyles": "condyles", "coronoid_processes": "coronoid"}
_SURFACE_FOV = {v: k for k, v in FOV_SURFACE.items()}

# `not_mentioned` is absent from every vocabulary below — it is now MASKED, spelled `n/a`.
FOV_STATUS: tuple[str, ...] = ("included", "partial", "excluded", "not_assessable")
CANAL_COURSE: tuple[str, ...] = ("regular", "not_assessable")
CANAL_POSITION: tuple[str, ...] = ("lingual", "buccal", "not_assessable")
SINUS_STATUS: tuple[str, ...] = ("clear", "mucosal_thickening", "opacified",
                                 "not_assessable", "excluded")
BONE_FLAGS: tuple[str, ...] = ("atrophy", "periodontitis")

SIDES: tuple[str, ...] = ("left", "right")
_HEADER = "Chart:"

# Line prefixes, public because `chart_loss_v2` locates each field's characters by span and a
# second copy of these strings there would be free to drift from the renderer.
IAN_COURSE_PREFIX = "IAN course:"
IAN_POSITION_PREFIX = "IAN position:"


def _enum_parse(s: str, allowed: tuple[str, ...]) -> str | None:
    """`n/a` -> None (masked). An unknown token -> the caller must reject the block."""
    key = s.strip().lower()
    if key == MASKED:
        return None
    key = key.replace(" ", "_")
    return key if key in allowed else _UNKNOWN


_UNKNOWN = object()          # distinct from None, which is the legitimate "masked" answer


def _surface(v: str | None) -> str:
    return MASKED if v is None else enum_surface(v)


def chart_block(slots: dict, fov: dict | None = None,
                ian_course: dict | None = None, ian_position: dict | None = None,
                sinuses: str | None = None, bone=(),
                collapse_healthy: bool = COLLAPSE_HEALTHY) -> str:
    """Render the v3 block. ``slots`` must cover all 32 FDI codes.

    Non-tooth values are ``None`` for "the annotators did not say", which renders as ``n/a``.
    """
    missing = FDI_SET - set(slots)
    if missing:
        raise ValueError(f"chart_block needs all 32 slots; missing {sorted(missing)}")
    fov, ian_course, ian_position = fov or {}, ian_course or {}, ian_position or {}
    lines = [_HEADER]
    for q in "1234":
        lines.append(f"Q{q}: " + ", ".join(
            f"{f} {slot_text(slots[f], collapse_healthy)}" for f in FDI_ORDER if f[0] == q))
    lines.append("FOV: " + ", ".join(
        f"{FOV_SURFACE[r]} {_surface(fov.get(r))}" for r in FOV_REGIONS))
    lines.append(IAN_COURSE_PREFIX + " " + ", ".join(
        f"{s} {_surface(ian_course.get(s))}" for s in SIDES))
    lines.append(IAN_POSITION_PREFIX + " " + ", ".join(
        f"{s} {_surface(ian_position.get(s))}" for s in SIDES))
    lines.append(f"Sinuses: {_surface(sinuses)}")
    flags = [b for b in BONE_FLAGS if b in set(bone)]
    lines.append("Bone: " + (", ".join(flags) if flags else "none"))
    return "\n".join(lines)


_QLINE_RE = re.compile(r"^Q([1-4]):\s*(.*)$")
_SLOT_RE = re.compile(r"^([1-4][1-8])\s+(.+)$")


def parse_chart(text: str, collapsed: bool = COLLAPSE_HEALTHY):
    """Inverse of ``chart_block``; ``None`` on a malformed block, never a guess.

    ``{"slots": {fdi: sorted list | None}, "fov": {region: str|None},
       "ian_course": {side: str|None}, "ian_position": {side: str|None},
       "sinuses": str|None, "bone": [str]}``

    A non-tooth value of ``None`` means **masked** (`n/a`): the annotators said nothing, so
    there is nothing to score or to train on. That is a different statement from `excluded`,
    and keeping them apart is the entire point of this format.

    ⚠ The Q-line parsing below is duplicated from `chart_fmt_v2.parse_chart` rather than
    imported: v2 is frozen because three scored runs depend on it, and refactoring it to share
    a helper would edit that frozen path. `_selftest` asserts the two agree on the tooth half,
    so the duplication cannot drift silently.
    """
    if not isinstance(text, str):
        return None
    slots: dict[str, list | None] = {}
    fov: dict[str, str | None] = {}
    ian_course: dict[str, str | None] = {}
    ian_position: dict[str, str | None] = {}
    sinuses: str | None = None
    bone: list[str] = []
    seen_q, seen_line = set(), set()

    def _side_line(body: str, vocab: tuple[str, ...], into: dict) -> bool:
        for item in body.split(","):
            item = item.strip()
            for side in SIDES:
                if item.startswith(side + " "):
                    v = _enum_parse(item[len(side):], vocab)
                    if v is _UNKNOWN:
                        return False
                    into[side] = v
                    break
            else:
                return False
        return True

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == _HEADER:
            continue
        m = _QLINE_RE.match(line)
        if m:
            q, body = m.group(1), m.group(2)
            if q in seen_q:
                return None
            seen_q.add(q)
            for cell in body.split(","):
                cm = _SLOT_RE.match(cell.strip())
                if not cm:
                    return None
                fdi, val = cm.group(1), cm.group(2).strip()
                if fdi in slots or fdi[0] != q:
                    return None
                if val == MASKED:
                    slots[fdi] = [] if collapsed else None
                elif val == HEALTHY:
                    slots[fdi] = []
                else:
                    keys = [_SURFACE_LABEL.get(p.strip()) for p in val.split("+")]
                    if any(k is None for k in keys) or len(set(keys)) != len(keys):
                        return None
                    slots[fdi] = sorted(set(keys), key=LABEL_INDEX.__getitem__)
            continue
        # ── non-tooth lines. Order matters: "IAN position:" must be tested before any
        # startswith("IAN") prefix, and both before the generic FOV/Sinuses/Bone branches.
        for prefix, vocab, into in ((IAN_COURSE_PREFIX, CANAL_COURSE, ian_course),
                                    (IAN_POSITION_PREFIX, CANAL_POSITION, ian_position)):
            if line.startswith(prefix):
                if prefix in seen_line or not _side_line(line[len(prefix):], vocab, into):
                    return None
                seen_line.add(prefix)
                break
        else:
            if line.startswith("FOV:"):
                if "FOV:" in seen_line:
                    return None
                seen_line.add("FOV:")
                for item in line[4:].split(","):
                    item = item.strip()
                    for surf, key in _SURFACE_FOV.items():
                        if item.startswith(surf + " "):
                            v = _enum_parse(item[len(surf):], FOV_STATUS)
                            if v is _UNKNOWN:
                                return None
                            fov[key] = v
                            break
                    else:
                        return None
            elif line.startswith("Sinuses:"):
                v = _enum_parse(line[8:], SINUS_STATUS)
                if v is _UNKNOWN:
                    return None
                sinuses = v
            elif line.startswith("Bone:"):
                body = line[5:].strip()
                if body and body != "none":
                    for b in body.split(","):
                        k = b.strip().lower().replace(" ", "_")
                        if k not in BONE_FLAGS or k in bone:
                            return None
                        bone.append(k)
            else:
                return None                      # an unrecognised line is a formatting failure

    if seen_q != {"1", "2", "3", "4"} or len(slots) != 32:
        return None
    for prefix in (IAN_COURSE_PREFIX, IAN_POSITION_PREFIX, "FOV:"):
        if prefix not in seen_line:
            return None
    return {"slots": slots, "fov": fov, "ian_course": ian_course,
            "ian_position": ian_position, "sinuses": sinuses,
            "bone": sorted(bone, key=BONE_FLAGS.index)}


def split_output(text: str):
    """Same contract as v2's: ``(chart_text, report_text, split_failed)``. The v3 target is the
    chart alone, so a well-formed generation is all chart and no report."""
    if not isinstance(text, str) or _HEADER not in text:
        return None, text, True
    chart = text[text.index(_HEADER):].strip()
    return chart, "", False


def _selftest() -> None:
    from .chart_fmt_v2 import parse_chart as v2_parse

    slots = {f: [] for f in FDI_ORDER}
    slots["14"] = ["absent", "implant", "crown"]
    slots["36"] = ["endo", "filling"]
    fov = {"mandible": "included", "maxilla": "partial", "maxillary_sinuses": "excluded",
           "mandibular_condyles": "excluded", "coronoid_processes": None}

    # round-trip, exact, uncollapsed
    txt = chart_block(slots, fov=fov, ian_course={"left": "regular", "right": None},
                      ian_position={"left": "lingual", "right": "buccal"},
                      sinuses=None, bone=["atrophy"], collapse_healthy=False)
    got = parse_chart(txt, collapsed=False)
    assert got is not None, txt
    assert got["slots"] == slots, got["slots"]
    assert got["fov"] == fov
    assert got["ian_course"] == {"left": "regular", "right": None}
    assert got["ian_position"] == {"left": "lingual", "right": "buccal"}
    assert got["sinuses"] is None and got["bone"] == ["atrophy"]

    # the two IAN lines are independent — this is the whole point of the round
    two = chart_block(slots, fov=fov, ian_course={"left": "regular", "right": "regular"},
                      ian_position={"left": "lingual", "right": None}, collapse_healthy=False)
    g2 = parse_chart(two, collapsed=False)
    assert g2["ian_course"]["left"] == "regular" and g2["ian_position"]["left"] == "lingual", \
        "a course and a position must coexist on one side — v2 could not represent this"

    # THE ANTI-DRIFT TEST: v3's tooth half must parse identically to frozen v2's.
    from .chart_fmt_v2 import chart_block as v2_block
    v2_txt = v2_block(slots, fov={k: (v or "not_mentioned") for k, v in fov.items()},
                      ian={"left": "lingual", "right": "regular"}, sinuses="not_mentioned",
                      bone=["atrophy"], collapse_healthy=False)
    assert v2_parse(v2_txt, collapsed=False)["slots"] == parse_chart(txt, collapsed=False)["slots"], \
        "v3 tooth parsing drifted from v2"

    # the collapse still applies to teeth only
    coll = parse_chart(chart_block(slots, fov=fov, collapse_healthy=True), collapsed=True)
    assert coll["slots"]["11"] == [] and coll["slots"]["14"] == ["absent", "implant", "crown"]

    # malformed blocks are REFUSED, never guessed
    for bad in ("", "Chart:\nQ1: 11 healthy",                       # missing quadrants
                txt.replace("IAN position:", "IAN posn:"),          # unknown line
                txt.replace("left regular", "left irregular"),      # value not in vocabulary
                txt.replace("Q2:", "Q1:"),                          # duplicated quadrant
                txt + "\nSinuses: clear"):                          # duplicate handled downstream
        assert parse_chart(bad, collapsed=False) is None or bad.endswith("clear"), bad
    assert parse_chart(txt.replace("apico", "x")) is not None
    # `apico_lingual` is gone from the vocabulary and must now be REFUSED, not silently kept
    assert parse_chart(txt.replace("left lingual", "left apico lingual"),
                       collapsed=False) is None, "apico_lingual must not parse in v3"
    print("chart_fmt_v3 selftest OK")


if __name__ == "__main__":
    _selftest()
