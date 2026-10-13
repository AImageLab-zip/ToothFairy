"""Parsed chart -> report prose. Deterministic, stdlib-only, container-safe.

The v2 target is the chart alone, so this module is the
only thing standing between the model's output and the leaderboard: the container generates a
chart, parses it, and renders the report from the parsed structure.

**Phrasing is copied from `data_canonical/render_prose.py`, deliberately.** That renderer's
output is what scored test 0.2362-0.2482 under the fixed judge, so its section layout
(Mandible / Maxilla / Mandibular condyles) and its sentence forms are known quantities. Inventing
wording here would spend captioning score to no purpose and make any FinalScore delta
unattributable. Where this module has less information than `render_prose` (no free-text notes,
no periodontitis stage) it says less — it never fills the gap with invention.

**The one deliberate departure: presence is stated.** `render_prose._tooth_clause` returns
`None` for a bare present tooth, because in v1 an unmentioned tooth carried no label and
asserting its health would have been a guess. That is no longer true — the v2 chart supervises
health explicitly, and 265/1000 reference reports contain a presence assertion ("Teeth from 18
to 27 are present in the arch"). Dropping it would throw away the round's own product before it
reaches the judge. It is one sentence per arch, and `state_presence=False` turns it off so the
A/B is a flag rather than a rewrite.

**Never empty.** `render_report` always returns a non-empty string for a parseable chart; a
chart that does not parse returns `None` so the caller can use its own fallback rather than
ship an empty report.
"""
from __future__ import annotations

from .chart_fmt_v2 import FDI_ORDER, parse_chart

# ── phrasing, lifted from render_prose so the register is the proven one ─────────
FOV_TXT = {
    "included": "included in the scan volume",
    "partial": "partially included in the acquisition",
    "excluded": "not included in the scan volume",
    "not_assessable": "partially included in the acquisition and not assessable",
    "not_mentioned": None,
}
COURSE_TXT = {
    "regular": "a regular course",
    "lingual": "a predominantly lingual course",
    "apico_lingual": "a regular course in an apico-lingual position",
    "buccal": "a predominantly buccal course",
    "not_assessable": "a course that cannot be fully assessed",
    "not_mentioned": None,
}
# Attribute -> noun phrase. `endo` keeps render_prose's unqualified form: the chart records that
# a root filling is present, never whether it was judged adequate, and "adequate" would be a
# clinical claim the model never made.
ATTR_TXT = {
    "implant": "an endosseous implant",
    "crown": "a prosthetic crown",
    "endo": "endodontic treatment",
    "lesion": "a periapical lesion",
    "filling": "a conservative restoration",
}
_ATTR_ORDER = ("endo", "implant", "crown", "filling", "lesion")

_ARCH_SLOTS = {"mandible": tuple(f for f in FDI_ORDER if f[0] in "34"),
               "maxilla": tuple(f for f in FDI_ORDER if f[0] in "12")}
# A presence list this long reads as an inventory; the corpus writes the collective form
# instead ("presence of all dental elements in the arch").
_ALL_PRESENT_MIN = 12


def _tooth_sentence(fdi: str, labels: list[str]) -> str | None:
    """One slot -> one sentence, or None when the slot says nothing worth a sentence.

    `absent` is handled by the caller (grouped), so an `absent` slot reaching here always
    carries attributes — "tooth 35 is absent, with an endosseous implant" — which is exactly the
    combination the v1 chart could not represent and `render_prose` dropped."""
    s = set(labels)
    extras = [ATTR_TXT[a] for a in _ATTR_ORDER if a in s]
    # `clause` heads are full sentences ("tooth 35 is absent") and take a comma before the
    # attributes; noun-phrase heads ("tooth 36") run straight on. Without the split you get
    # "Tooth 35 is absent with an endosseous implant", which reads as the implant qualifying
    # the absence rather than occupying the site.
    clause = True
    if "absent" in s:
        head = f"tooth {fdi} is absent"
    elif "root_remnant" in s:
        head, clause = f"a root remnant of tooth {fdi}", False
    elif "impacted" in s:
        head = f"tooth {fdi} is impacted"
    elif extras:
        head, clause = f"tooth {fdi}", False
    else:
        return None                              # healthy: the presence sentence covers it
    if extras:
        head += (", with " if clause else " with ") + ", ".join(extras)
    return head[0].upper() + head[1:] + "."


def _arch_block(arch: str, slots: dict, state_presence: bool) -> list[str]:
    out: list[str] = []
    supervised = [f for f in _ARCH_SLOTS[arch] if slots.get(f) is not None]
    # Bare absences group into one sentence, as render_prose does; an absence carrying an
    # implant or a crown keeps its own sentence because it has more to say.
    bare_absent = [f for f in supervised if set(slots[f]) == {"absent"}]
    if bare_absent:
        out.append("Absence of teeth " + ", ".join(bare_absent) + ".")
    for f in supervised:
        if f in bare_absent:
            continue
        s = _tooth_sentence(f, slots[f])
        if s:
            out.append(s)
    if state_presence:
        healthy = [f for f in supervised if not slots[f]]
        if healthy and len(healthy) >= _ALL_PRESENT_MIN and len(healthy) == len(supervised):
            out.append("All dental elements are present in the arch.")
        elif healthy:
            verb = "is" if len(healthy) == 1 else "are"
            noun = "Tooth" if len(healthy) == 1 else "Teeth"
            out.append(f"{noun} {', '.join(healthy)} {verb} present in the arch.")
    return out


def _canal_sentence(ian: dict) -> str | None:
    lc = COURSE_TXT.get(ian.get("left", "not_mentioned"))
    rc = COURSE_TXT.get(ian.get("right", "not_mentioned"))
    if not lc and not rc:
        return None
    if lc == rc:
        return f"The mandibular canal shows {lc} bilaterally."
    segs = []
    if rc:
        segs.append(f"the right mandibular canal shows {rc}")
    if lc:
        segs.append(f"the left mandibular canal shows {lc}")
    s = "; ".join(segs)
    return s[0].upper() + s[1:] + "."


def _fov_for_arch(arch: str, fov: dict, slots: dict) -> str:
    """The FOV status to *report* for an arch, softened when the chart contradicts it.

    Reporting "not included in the scan volume" next to a list of that arch's own teeth is a
    self-contradiction the judge would score as a false claim.

    ⚠ **This fires 0 times on corpus charts** — `merge_case` masks an out-of-scan arch at the
    source, so a training target can no longer contain the contradiction (it did, in 36 arch
    blocks, before that gate went in). The guard stays because at inference the chart is written
    by the *model*, which is free to emit `maxilla excluded` and then chart maxillary teeth
    anyway. Softening the wording is the right response there: the teeth are the specific claim,
    the FOV line is the general one."""
    status = fov.get(arch, "not_mentioned")
    # ⚠ Tests for a slot that carries a FINDING, not merely a parsed slot. Under the collapsed
    # chart every slot parses to a list, so `is not None` became universally true and would have
    # softened every `excluded` arch to `partial` — silently deleting the model's own FOV claim.
    if status == "excluded" and any(slots.get(f) for f in _ARCH_SLOTS[arch]):
        return "partial"
    return status


def render_report(chart: dict, state_presence: bool = False) -> str:
    """Parsed chart (the dict `chart_fmt_v2.parse_chart` returns) -> report text.

    ⚠ ``state_presence`` defaults to **False** since 2026-08-13, for two independent reasons.
    It is now *unsound*: the collapsed chart spells a healthy tooth and an out-of-scan tooth
    identically, so a presence sentence would assert health for teeth nobody could see. And it
    was already the losing arm — chartv2_v1, LogicalF1 +0.0179 and precision +0.0563 with
    it off. Passing ``True`` on a collapsed chart is a measurement error, not an option."""
    slots, fov, bone = chart["slots"], chart["fov"], set(chart.get("bone") or ())
    blocks: list[str] = []

    # ── Mandible
    mand = ["Mandible:"]
    m_fov = _fov_for_arch("mandible", fov, slots)
    if m_fov == "partial":
        mand.append("The mandible is partially included in the acquisition.")
    elif m_fov == "excluded":
        mand.append("The mandible is not included in the scan volume.")
    mand += _arch_block("mandible", slots, state_presence)
    if "atrophy" in bone:
        mand.append("Atrophy is noted.")
    if "periodontitis" in bone:
        mand.append("Periodontal disease is noted.")
    cs = _canal_sentence(chart.get("ian") or {})
    if cs:
        mand.append(cs)
    blocks.append("\n".join(mand))

    # ── Maxilla
    mx = ["Maxilla:"]
    x_fov = _fov_for_arch("maxilla", fov, slots)
    if FOV_TXT.get(x_fov):
        mx.append("The maxilla is " + FOV_TXT[x_fov] + ".")
    mx += _arch_block("maxilla", slots, state_presence)
    sin = chart.get("sinuses", "not_mentioned")
    if sin == "clear":
        mx.append("The maxillary sinuses are normally pneumatized.")
    elif sin == "mucosal_thickening":
        mx.append("Mucosal thickening of the maxillary sinuses.")
    elif sin == "opacified":
        mx.append("Opacification of the maxillary sinuses.")
    elif sin in ("not_assessable", "excluded"):
        mx.append("The maxillary sinuses are not included in the acquisition and not assessable.")
    if len(mx) > 1:
        blocks.append("\n".join(mx))

    # ── Condyles: their own section, exactly as the corpus and render_prose lay it out
    cond = fov.get("mandibular_condyles", "not_mentioned")
    txt = {"excluded": "not included in the scan volume",
           "partial": "only partially included in the scan volume",
           "not_assessable": "partially included in the acquisition and not assessable",
           "included": "included in the scan volume"}.get(cond)
    if txt:
        blocks.append(f"Mandibular condyles: {txt}.")

    return "\n\n".join(blocks)


def render_from_text(text: str, state_presence: bool = False) -> str | None:
    """Generation -> report, or ``None`` when the chart does not parse.

    ``None`` is deliberate: the container must fall back to its own never-empty template rather
    than ship whatever a malformed chart happened to render to."""
    chart = parse_chart(text)
    if chart is None:
        return None
    return render_report(chart, state_presence=state_presence)
