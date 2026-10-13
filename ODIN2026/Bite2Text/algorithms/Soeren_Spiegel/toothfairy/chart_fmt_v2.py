"""Frozen chart-v2 format + parser — the container-side half of the target.

The v2 target is the chart **alone**: the model emits this
block and nothing else, and the report is rendered from the parsed block by a deterministic
template. So this module is both the training target's spelling and the container's only
way back to structure — if the two ever disagree the run is unmeasurable.

**Stdlib only, and zero ``data_canonical`` imports.** ``data_canonical/`` is not vendored into
the submission image and a module-scope import of it has already crashed the container once.
The dependency therefore points the other way: ``data_canonical/chart_v2.py`` imports the format
from here, exactly as ``chart_render`` imports v1's format from ``chart_parse``.

The format::

    Chart:
    Q1: 11 healthy, 12 healthy, 13 healthy, 14 absent+implant+crown, 15 endo, 16 absent, 17 n/a, 18 n/a
    Q2: 21 healthy, ...
    Q3: 31 healthy, ...
    Q4: 41 healthy, ...
    FOV: mandible included, maxilla partial, maxillary sinuses excluded, condyles excluded, coronoid not mentioned
    IAN: left lingual, right regular
    Sinuses: not mentioned
    Bone: atrophy

Rules, all load-bearing:

  * **All 32 slots, always, in ascending FDI within a quadrant, Q1 -> Q4.** v1 omitted empty
    quadrants and collapsed an empty arch to "not charted"; v2 does not, because the slot vector
    is now dense and a missing line would be ambiguous between "healthy" and "unsupervised".
  * one slot is ``<fdi> <labels>``; labels are joined by ``+`` in ``LABELS`` order so the string
    is a function of the set and not of iteration order.
  * ``n/a`` is the surface for **both** the empty label set and an unsupervised slot. A tooth
    with no pathology and a tooth nobody could see are spelled the same way — decided
    2026-08-13, see ``COLLAPSE_HEALTHY`` below.
  * ``healthy`` is the *uncollapsed* spelling of the empty set. It is no longer emitted in a
    training target; it survives only on the **gold** side, which must keep the two apart or
    the metric loses its mask and its ``present`` status class.

**The collapse, and the one place it must not reach.** ``collapse_healthy=True`` (the default)
renders the empty set as ``n/a``; a target therefore holds one label for every tooth that
carries no finding, which is what the ``--no-state-presence`` A/B pointed at.

⚠ **It is lossy, and that is the point** — ``parse_chart`` cannot recover which ``n/a`` meant
health. So the gold chart, which is the reference the metric skips masked slots against, is
rendered and parsed with ``collapse_healthy=False`` / ``collapsed=False``. Two consequences,
both asserted in the self-tests rather than left to trust:

  * round-trip is exact only in the uncollapsed mode. Collapsed, it holds modulo the
    documented projection ``None -> []``.
  * ``chart_prose_v2.state_presence`` is **unsound** under a collapsed chart — asserting
    presence would claim health for out-of-scan teeth — so its default is now ``False``.
  * the four non-tooth lines are fixed and always present. `FOV` carries all five regions in a
    frozen order; a report that never mentions a region says ``not mentioned``, which is a
    different statement from ``excluded`` and the judge scores them differently.

**What the chart deliberately does not carry.** Measured over the 1000 per-report records:
`bone_and_perio.osteolytic_lesions` (87), `post_extraction_sockets` (51) and `other` (55) are
free text or FDI lists with no controlled vocabulary, and `periodontitis.stage` has 12+ spellings
across 82 records ("severe", "stage II", "diffuse", "expulsive", a two-clause sentence). Forcing
them into enums would be the "new labelling logic" the round forbids, so `Bone:` keeps only the
two clean flags — `atrophy` (391 records, boolean) and `periodontitis` (presence only). Those
findings therefore do not reach the generated report; that gap is stated rather than
hidden.
"""
from __future__ import annotations

import re

# ── vocabulary ──────────────────────────────────────────────────────────────────
# Internal keys match data_canonical/chart_v2.LABELS exactly and in the same order; the surface
# spelling is this module's business alone.
LABELS: tuple[str, ...] = ("absent", "impacted", "root_remnant",
                           "implant", "crown", "endo", "lesion", "filling")
LABEL_INDEX: dict[str, int] = {c: i for i, c in enumerate(LABELS)}
_LABEL_SURFACE: dict[str, str] = {c: c.replace("_", " ") for c in LABELS}
_SURFACE_LABEL: dict[str, str] = {v: k for k, v in _LABEL_SURFACE.items()}

HEALTHY = "healthy"          # uncollapsed spelling of the empty set — GOLD SIDE ONLY
MASKED = "n/a"               # unsupervised, and (collapsed) also "no finding"

# The emitted vocabulary. `HEALTHY` is deliberately absent: nothing a model is trained on or
# scored from spells it any more, so a scan that still expected it would count zero forever.
COLLAPSE_HEALTHY = True      # module default; the gold path passes False explicitly

FDI_ORDER: tuple[str, ...] = tuple(f"{q}{i}" for q in "1234" for i in range(1, 9))
FDI_SET = frozenset(FDI_ORDER)

FOV_REGIONS: tuple[str, ...] = ("mandible", "maxilla", "maxillary_sinuses",
                                "mandibular_condyles", "coronoid_processes")
# Public because the loss has to find each region's value on the `FOV:` line by its own
# character span. A second copy of these spellings in `chart_loss_v2` would be free to
# drift from the renderer, and a drifted copy weights the wrong characters silently.
FOV_SURFACE = {"mandible": "mandible", "maxilla": "maxilla",
               "maxillary_sinuses": "maxillary sinuses",
               "mandibular_condyles": "condyles", "coronoid_processes": "coronoid"}
_SURFACE_FOV = {v: k for k, v in FOV_SURFACE.items()}

FOV_STATUS: tuple[str, ...] = ("included", "partial", "excluded", "not_assessable",
                               "not_mentioned")
CANAL_COURSE: tuple[str, ...] = ("regular", "lingual", "apico_lingual", "buccal",
                                 "not_assessable", "not_mentioned")
SINUS_STATUS: tuple[str, ...] = ("clear", "mucosal_thickening", "opacified", "not_assessable",
                                 "excluded", "not_mentioned")
BONE_FLAGS: tuple[str, ...] = ("atrophy", "periodontitis")

DELIMITER = "\n\nReport:\n"          # kept for v1 container compatibility; v2 emits chart only
_HEADER = "Chart:"


def enum_surface(v: str) -> str:
    return v.replace("_", " ")


def _enum_parse(s: str, allowed: tuple[str, ...]) -> str | None:
    key = s.strip().lower().replace(" ", "_")
    return key if key in allowed else None


def slot_text(labels, collapse_healthy: bool = COLLAPSE_HEALTHY) -> str:
    """A slot's label set -> its surface string. ``None`` is unsupervised, ``[]``/``set()`` is
    healthy. Sorted by ``LABELS`` order, so the output depends on the set alone.

    Under ``collapse_healthy`` (the default) both render ``n/a``; pass ``False`` for the gold
    chart, which must stay able to tell them apart."""
    if labels is None:
        return MASKED
    labels = set(labels)
    if not labels:
        return MASKED if collapse_healthy else HEALTHY
    unknown = labels - set(LABELS)
    if unknown:
        raise ValueError(f"unknown label(s): {sorted(unknown)}")
    return "+".join(_LABEL_SURFACE[c] for c in sorted(labels, key=LABEL_INDEX.__getitem__))


def chart_block(slots: dict, fov: dict | None = None, ian: dict | None = None,
                sinuses: str = "not_mentioned", bone=(),
                collapse_healthy: bool = COLLAPSE_HEALTHY) -> str:
    """Render the frozen block. ``slots`` must cover all 32 FDI codes."""
    missing = FDI_SET - set(slots)
    if missing:
        raise ValueError(f"chart_block needs all 32 slots; missing {sorted(missing)}")
    fov = fov or {}
    ian = ian or {}
    lines = [_HEADER]
    for q in "1234":
        cells = [f"{f} {slot_text(slots[f], collapse_healthy)}"
                 for f in FDI_ORDER if f[0] == q]
        lines.append(f"Q{q}: " + ", ".join(cells))
    lines.append("FOV: " + ", ".join(
        f"{FOV_SURFACE[r]} {enum_surface(fov.get(r, 'not_mentioned'))}" for r in FOV_REGIONS))
    lines.append("IAN: " + ", ".join(
        f"{s} {enum_surface(ian.get(s, 'not_mentioned'))}" for s in ("left", "right")))
    lines.append(f"Sinuses: {enum_surface(sinuses or 'not_mentioned')}")
    flags = [b for b in BONE_FLAGS if b in set(bone)]
    lines.append("Bone: " + (", ".join(flags) if flags else "none"))
    return "\n".join(lines)


_QLINE_RE = re.compile(r"^Q([1-4]):\s*(.*)$")
_SLOT_RE = re.compile(r"^([1-4][1-8])\s+(.+)$")


def parse_chart(text: str, collapsed: bool = COLLAPSE_HEALTHY):
    """The inverse of ``chart_block``. Returns a dict or ``None`` if the block is
    malformed — never a guess, because a silent repair here would report a run that never
    happened.

    ``{"slots": {fdi: sorted list | None}, "fov": {...}, "ian": {...},
       "sinuses": str, "bone": [str]}``

    ``collapsed`` (the default) reads ``n/a`` as **an empty label set**: that is what a model
    emits now, and it is a claim ("nothing here"), not a refusal. Pass ``collapsed=False`` for a
    gold chart rendered with ``collapse_healthy=False``, where ``n/a`` still means *unsupervised*
    and must come back as ``None`` so the metric can skip the slot. ``healthy`` is accepted in
    both modes so an uncollapsed chart round-trips exactly."""
    if not isinstance(text, str):
        return None
    slots: dict[str, list | None] = {}
    fov: dict[str, str] = {}
    ian: dict[str, str] = {}
    sinuses = None
    bone: list[str] = []
    seen_q = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == _HEADER:
            continue
        m = _QLINE_RE.match(line)
        if m:
            q, body = m.group(1), m.group(2)
            if q in seen_q:
                return None                      # a duplicated quadrant is a formatting failure
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
                    parts = [p.strip() for p in val.split("+")]
                    keys = [_SURFACE_LABEL.get(p) for p in parts]
                    if any(k is None for k in keys) or len(set(keys)) != len(keys):
                        return None
                    slots[fdi] = sorted(set(keys), key=LABEL_INDEX.__getitem__)
            continue
        if line.startswith("FOV:"):
            for item in line[4:].split(","):
                item = item.strip()
                for surf, key in _SURFACE_FOV.items():
                    if item.startswith(surf + " "):
                        v = _enum_parse(item[len(surf):], FOV_STATUS)
                        if v is None:
                            return None
                        fov[key] = v
                        break
                else:
                    return None
            continue
        if line.startswith("IAN:"):
            for item in line[4:].split(","):
                item = item.strip()
                for side in ("left", "right"):
                    if item.startswith(side + " "):
                        v = _enum_parse(item[len(side):], CANAL_COURSE)
                        if v is None:
                            return None
                        ian[side] = v
                        break
                else:
                    return None
            continue
        if line.startswith("Sinuses:"):
            sinuses = _enum_parse(line[8:], SINUS_STATUS)
            if sinuses is None:
                return None
            continue
        if line.startswith("Bone:"):
            body = line[5:].strip()
            if body and body != "none":
                for b in body.split(","):
                    k = b.strip().lower()
                    if k not in BONE_FLAGS:
                        return None
                    bone.append(k)
            continue
        return None                              # an unrecognised line is a failure, not noise

    if set(slots) != FDI_SET or seen_q != set("1234"):
        return None
    if set(fov) != set(FOV_REGIONS) or set(ian) != {"left", "right"} or sinuses is None:
        return None
    return {"slots": slots, "fov": fov, "ian": ian, "sinuses": sinuses, "bone": bone}


def split_output(text: str):
    """``(chart_text | None, report_text, split_failed)`` — same contract as v1's
    ``chart_parse.split_output``, including its return types, because ``entrypoint.py`` calls
    one of them unconditionally and the two must be swappable.

    v2 targets are chart-only, so the normal path returns the whole generation as the chart and
    an empty report. The v1 delimiter is still honoured, and a prose-only generation falls
    through with ``report == text``: the shipping ``combined_v1`` checkpoint emits no chart at
    all and must get its text back verbatim.

    ``split_failed`` means **the frozen format was not found**, matching v1 exactly — so it is
    True for a prose-only generation. That fires on every case of a v1-style checkpoint by
    design; it is counted and logged, never absorbed.

    Splitting and parsing are two stages with two jobs: a block that *starts* ``Chart:`` splits
    cleanly even when its body is malformed, and ``parse_chart`` returning ``None`` is the loud
    signal for that. Do not merge the two — a splitter that also validated would have to guess
    what a half-written chart meant."""
    if not isinstance(text, str) or not text:
        return None, "", True
    if DELIMITER in text:
        chart, report = text.split(DELIMITER, 1)
        return chart.strip(), report.strip(), False
    if text.lstrip().startswith(_HEADER):
        return text.strip(), "", False
    return None, text.strip(), True              # no chart: a v1-style prose-only generation


def _selftest() -> None:
    """Format-level invariants. The exhaustive round-trip over the real corpus lives in
    ``data_canonical/chart_v2.py --test``, which is the side that has the data."""
    slots = {f: [] for f in FDI_ORDER}
    slots["14"] = ["absent", "implant", "crown"]
    slots["17"] = None
    slots["36"] = ["endo"]
    fov = {r: "not_mentioned" for r in FOV_REGIONS}
    fov["mandible"], fov["maxilla"] = "included", "partial"
    # ── uncollapsed (the GOLD rendering): round-trip is exact, mask survives
    gold_txt = chart_block(slots, fov=fov, ian={"left": "lingual", "right": "regular"},
                           sinuses="mucosal_thickening", bone=["atrophy"],
                           collapse_healthy=False)
    got = parse_chart(gold_txt, collapsed=False)
    assert got is not None and got["slots"] == {**slots, "14": ["absent", "implant", "crown"]}
    assert got["slots"]["17"] is None, "gold must keep the unsupervised slot distinguishable"
    assert got["fov"] == fov and got["ian"] == {"left": "lingual", "right": "regular"}
    assert got["sinuses"] == "mucosal_thickening" and got["bone"] == ["atrophy"]
    assert f"11 {HEALTHY}" in gold_txt and "17 n/a" in gold_txt

    # ── collapsed (the TARGET rendering): one surface for both, and the loss is asserted
    txt = chart_block(slots, fov=fov, ian={"left": "lingual", "right": "regular"},
                      sinuses="mucosal_thickening", bone=["atrophy"])
    assert HEALTHY not in txt, "a collapsed target must never spell `healthy`"
    assert "11 n/a" in txt and "17 n/a" in txt, "healthy and masked must be indistinguishable"
    got_c = parse_chart(txt)
    assert got_c is not None
    assert got_c["slots"] == {f: (v or []) for f, v in slots.items()}, \
        "collapsed round-trip must hold modulo the documented None -> [] projection"
    assert got_c["slots"]["17"] == [], "collapsed: a masked slot reads as a health claim"
    # findings are untouched by the collapse — the 8 label classes stay exactly comparable
    assert got_c["slots"]["14"] == ["absent", "implant", "crown"]

    # label order is a function of the set, never of insertion order
    assert slot_text({"crown", "absent"}) == slot_text({"absent", "crown"}) == "absent+crown"
    assert slot_text(set()) == MASKED and slot_text(None) == MASKED
    assert slot_text(set(), collapse_healthy=False) == HEALTHY
    assert slot_text({"root_remnant"}) == "root remnant"

    # all 32 slots are mandatory
    short = dict(slots)
    short.pop("48")
    try:
        chart_block(short, fov=fov)
        raise AssertionError("chart_block accepted 31 slots")
    except ValueError:
        pass

    # split_output, including the v1-checkpoint fall-through
    c, r, f = split_output(txt)
    assert (c, r, f) == (txt, "", False)
    prose = "The mandible shows good trophism."
    assert split_output(prose) == (None, prose, True), "v1 prose must return report == text"
    assert split_output("") == (None, "", True)
    assert split_output(None) == (None, "", True)
    c, r, f = split_output("Chart:\nQ1: junk")
    assert f is False and parse_chart(c) is None, "split and parse are separate stages"
    print(f"chart_fmt_v2 self-test: OK ({len(FDI_ORDER)} slots, {len(LABELS)} labels)")


if __name__ == "__main__":
    _selftest()
