"""Claim extraction — rule-based `(target, axis, value)` claims from generated or reference text.

    claim = (target, axis, value)
      target = "11".."48" (FDI tooth) or "maxilla"/"mandible"/"nerve"
      axis/value = the axes and categories in src/toothfairy/schema/claims.py (canonical source)

Why rules are workable here: per-structure decode emits one region at a time, so the sentences
are short (tooth regions average 8.3 words in the reference) and formulaic, and the region id
already identifies the tooth (a "prosthetic crown" inside the fdi:36 slot is a statement about
tooth 36 even without a number).

This module never imports torch (it runs on the CPU reward path). Schema agreement is verified
lazily against the schema module.
"""

from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Canonical axes and categories (schema/claims.py). The values are duplicated here to avoid
# importing torch. The canonical source stays `schema/claims.py`.
#
# NOTE: what is listed here is the ground-truth vocabulary (claims.Axis.raw_categories), not the
# training labels. Training collapses endo/fracture/atrophy to none/present and merges
# restoration's denture_abutment into prosthetic_crown, but this module extracts claims from text
# in order to compare them against the original ground-truth files, so it keeps the fine-grained
# vocabulary. Collapsing both sides would make the reward and the claim scoring disagree with the
# ground truth.
# ---------------------------------------------------------------------------

TOOTH_AXES: dict[str, tuple[str, ...]] = {
    "state": ("normal", "partial_impacted", "full_impacted", "supraeruption",
              "root_rest", "implant", "missing", "not_available"),
    "orientation": ("normoverted", "mesioverted", "distoverted"),
    "endo": ("none", "complete", "overfill", "underfill", "incomplete"),
    "periapical": ("normal", "apical_osteorarefaction", "nonapical_osteorarefaction",
                   "osteosclerosis", "cementoma"),
    "restoration": ("none", "filling", "prosthetic_crown", "pontic", "denture_abutment"),
    "fracture": ("none", "crown_fracture", "root_fracture",
                 "fractured_endo_material", "fractured_post_material"),
    "atrophy": ("none", "mild", "moderate", "severe"),
    "caries": ("none", "caries"),
    "post_core": ("none", "post_and_core"),
    "periodontal": ("normal", "marginal_bone_loss"),
}

NONTEETH_AXES: dict[str, dict[str, tuple]] = {
    "maxilla": {
        "perio_stage": ("normal", "stage_I", "stage_II", "stage_III"),
        "sinus_inclusion": ("not_included", "pneumatization_increased", "pneumatization_normal"),
        "sinus_mucosa": ("normal", "right_thickening", "left_thickening", "both_thickening",
                         "right_retention_cyst", "left_retention_cyst", "both_retention_cyst"),
        "fracture": ("none", "anterior", "right_premolar", "right_posterior",
                     "left_premolar", "left_posterior"),
        "cyst": ("none", "right_wisdom", "left_wisdom", "both_wisdom",
                 "right_posterior", "left_posterior", "both_posterior", "anterior"),
    },
    "mandible": {
        "perio_stage": ("normal", "stage_I", "stage_II", "stage_III"),
        "condyle_included": (False, True),
        "coronoid_included": (False, True),
        "fracture": ("none", "right_angle", "right_body", "right_parasymphysis",
                     "left_parasymphysis", "left_body", "left_angle",
                     "right_ramus", "left_ramus", "right_condyle", "left_condyle"),
        "cyst": ("none", "right_wisdom", "left_wisdom", "both_wisdom", "right_premolar",
                 "left_premolar", "right_posterior", "left_posterior",
                 "both_posterior", "anterior"),
    },
    "nerve": {
        "position": ("lingual", "buccal", "central"),
        "superficialized": (False, True),
        "canal_near_7": (False, True),
        "canal_near_8": (False, True),
        "not_evaluable": (False, True),
    },
}

FDI_UP = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LO = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
FDI_ALL = FDI_UP + FDI_LO
FDI_SET = set(FDI_ALL)

# Anatomical arch order (used when expanding a range such as "from 18 to 28").
ARCH_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
ARCH_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]




# ---------------------------------------------------------------------------
# 1. Tooth number parsing
# ---------------------------------------------------------------------------

_FDI_RE = re.compile(r"(?<![0-9.,])([1-4][1-8])(?![0-9])")
_DASH = r"[-–—]"
# "from 18 to 28" / "18 to 28" / "tooth 33 to tooth 43"
_RANGE_TO_RE = re.compile(
    r"(?<![0-9])([1-4][1-8])\s*(?:\)|,)?\s*(?:to|through|until|a)\s+(?:tooth\s+|teeth\s+|the\s+)?"
    r"([1-4][1-8])(?![0-9])", re.I)
# "35-36-37" / "34–37" / "12-22"
_DASH_SEQ_RE = re.compile(r"(?<![0-9])([1-4][1-8](?:\s*" + _DASH + r"\s*[1-4][1-8])+)(?![0-9])")
_EXCEPT_RE = re.compile(r"\b(?:except|excluding|apart from|other than|save for)\b", re.I)

_WISDOM_RE = re.compile(r"\b(?:third molars?|wisdom (?:tooth|teeth)|eighths?)\b", re.I)
_UPPER_RE = re.compile(r"\b(?:maxillar\w*|upper|superior)\b", re.I)
_LOWER_RE = re.compile(r"\b(?:mandibular|lower|inferior)\b", re.I)


def _arch_range(a: int, b: int) -> list[int]:
    """Teeth between a and b in arch order. If they are not in the same arch, just [a, b]."""
    for arch in (ARCH_UPPER, ARCH_LOWER):
        if a in arch and b in arch:
            i, j = arch.index(a), arch.index(b)
            lo, hi = min(i, j), max(i, j)
            return arch[lo:hi + 1]
    return [a, b]


# If only these appear between two numbers, the numbers belong to the same enumeration group
# ("teeth 44, 45 and 47" is one group).
_JOINER_RE = re.compile(
    r"^(?:[\s,;/&+.()\-–—]|and|to|or|e\.?d\.?|teeth|tooth|dental|elements?|the|of|in|at|n\.?"
    r"|positions?|sites?|regions?|quadrants?|from|between|through)*$", re.I)


def number_groups(sent: str) -> list[tuple[int, int, set[int]]]:
    """Split the tooth numbers in a sentence into enumeration groups -> [(start, end, teeth)].

    "Cantilever bridge between 23 and 25 replacing missing tooth 24"
      -> [(.., {23,25}), (.., {24})]  — 'missing' attaches only to the nearest group, {24}.
    """
    spans: list[tuple[int, int, set[int]]] = []
    used: list[tuple[int, int]] = []
    for r in _RANGE_TO_RE.finditer(sent):
        spans.append((r.start(), r.end(), set(_arch_range(int(r.group(1)), int(r.group(2))))))
        used.append(r.span())
    for r in _DASH_SEQ_RE.finditer(sent):
        if any(s <= r.start() < e or s < r.end() <= e for s, e in used):
            continue
        parts = [int(x) for x in re.split(_DASH, r.group(1))]
        vals = (set(_arch_range(parts[0], parts[1]))
                if len(parts) == 2 and parts[0] // 10 == parts[1] // 10 else set(parts))
        spans.append((r.start(), r.end(), vals))
        used.append(r.span())
    for r in _FDI_RE.finditer(sent):
        if any(s <= r.start() < e for s, e in used):
            continue
        spans.append((r.start(), r.end(), {int(r.group(1))}))
    spans.sort()
    merged: list[tuple[int, int, set[int]]] = []
    for s, e, v in spans:
        if merged and _JOINER_RE.match(sent[merged[-1][1]:s]):
            ps, pe, pv = merged[-1]
            merged[-1] = (ps, e, pv | v)
        else:
            merged.append((s, e, set(v)))
    return [(s, e, {t for t in v if t in FDI_SET}) for s, e, v in merged
            if any(t in FDI_SET for t in v)]


# If only these appear between a finding word and a number, the two count as adjacent.
_FILLER_RE = re.compile(
    r"^[\s,:;()]*(?:of|on|in|at|the|a|an|to|with|for|involving|affecting|regarding|from"
    r"|tooth|teeth|dental|element|elements|e\.?d\.?|site|sites|position|positions"
    r"|region|regions|level|correspondence|number|n\.?|is|are|was|were|and)?"
    r"(?:[\s,:;()]+(?:of|on|in|at|the|a|an|to|with|tooth|teeth|dental|element|elements"
    r"|e\.?d\.?|site|sites|position|positions|region|regions|level|number|n\.?))*[\s,:;()]*$",
    re.I)


def nearest_group(sent: str, span: tuple[int, int],
                  groups: list[tuple[int, int, set[int]]]) -> set[int]:
    """Number group nearest to the trigger. A gap made only of function words counts as
    distance 0."""
    if not groups:
        return set()
    if len(groups) == 1:
        return set(groups[0][2])
    s, e = span

    def dist(g):
        gs, ge, _ = g
        if gs <= s <= ge:
            return 0
        gap = sent[e:gs] if gs >= e else sent[ge:s]
        if _FILLER_RE.match(gap):
            return 0
        return min(abs(gs - s), abs(ge - s))

    return set(min(groups, key=dist)[2])


def parse_teeth(sent: str, expand_groups: bool = True) -> tuple[set[int], set[int]]:
    """Extract the set of FDI tooth numbers mentioned in a sentence.

    Returns (targets, excluded).
      - "from 18 to 28" / "17 to 27"  -> expanded along the arch order
      - "35-36-37-38" (three or more) -> a plain list
      - "34-37" (two, same quadrant)  -> a range; across quadrants only the two endpoints
      - "... except 31, 32"           -> numbers after 'except' are returned as excluded
      - no numbers and expand_groups  -> group expressions such as "third molars" are expanded
    """
    head, tail = sent, ""
    m = _EXCEPT_RE.search(sent)
    if m:
        head, tail = sent[:m.start()], sent[m.end():]

    def collect(seg: str) -> set[int]:
        out: set[int] = set()
        used: list[tuple[int, int]] = []
        for r in _RANGE_TO_RE.finditer(seg):
            out.update(_arch_range(int(r.group(1)), int(r.group(2))))
            used.append(r.span())
        for r in _DASH_SEQ_RE.finditer(seg):
            if any(s <= r.start() < e or s < r.end() <= e for s, e in used):
                continue
            parts = [int(x) for x in re.split(_DASH, r.group(1))]
            if len(parts) == 2 and parts[0] // 10 == parts[1] // 10:
                out.update(_arch_range(parts[0], parts[1]))
            else:
                out.update(parts)
            used.append(r.span())
        for r in _FDI_RE.finditer(seg):
            if any(s <= r.start() < e for s, e in used):
                continue
            out.add(int(r.group(1)))
        return {t for t in out if t in FDI_SET}

    targets, excluded = collect(head), collect(tail)
    if not targets and expand_groups and _WISDOM_RE.search(head):
        up, lo = bool(_UPPER_RE.search(head)), bool(_LOWER_RE.search(head))
        if up and not lo:
            targets = {18, 28}
        elif lo and not up:
            targets = {38, 48}
        else:
            targets = {18, 28, 38, 48}
    return targets - excluded, excluded

# ---------------------------------------------------------------------------
# 2. Finding vocabulary (a regex per axis/value). Within one axis the earliest rule wins.
# ---------------------------------------------------------------------------

def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


# (axis, value, pattern) — the order within an axis is its priority (earlier is stronger)
TOOTH_RULES: list[tuple[str, str, re.Pattern]] = [

    # --- state -------------------------------------------------------------
    ("state", "implant", _rx(r"\bimplant(?:s|ology)?\b|\bfixture\b|\bosseointegrat")),
    ("state", "root_rest", _rx(r"\broot (?:remnant|rest|fragment|stump|residu)"
                               r"|\bresidual roots?\b|\bretained roots?\b|\bradicular remnant")),
    ("state", "partial_impacted", _rx(r"\bpartial(?:ly)?[\s-]*(?:bony\s*)?(?:impact|includ|erupt)"
                                      r"|\bsemi[\s-]*(?:impact|includ|erupt)"
                                      r"|\bincomplete(?:ly)? erupt"
                                      r"|\bpartial (?:bony )?(?:impaction|inclusion)"
                                      r"|\bnot fully erupt")),
    ("state", "full_impacted", _rx(r"\bimpact(?:ed|ion)\b|\bunerupted\b|\bnon[\s-]?erupted\b"
                                   r"|\bdisodontias|\bincluded in (?:the )?bone|\bbony inclusion"
                                   r"|\bretained (?:tooth|teeth|germ)|\btooth germ|\bgerms?\b"
                                   r"|\bintraosseous inclusion|\binclusion of (?:tooth|teeth)")),
    ("state", "supraeruption", _rx(r"\bextrud|\bextrusion\b|\bsupra[\s-]?erupt|\bover[\s-]?erupt"
                                   r"|\begression\b")),
    ("state", "not_available", _rx(r"\bnot (?:visualiz|visible|includ|assessab|evaluab|apprecia)"
                                   r"|\bcannot be (?:assess|evaluat|visualiz)"
                                   r"|\bnot (?:clearly )?(?:depict|represent)"
                                   r"|\boutside (?:the )?(?:field of view|scan|acquisition|volume)"
                                   r"|\bexcluded from (?:the )?(?:scan|acquisition|volume)"
                                   r"|\bnon[\s-]?evaluable\b|\bnot included\b")),
    ("state", "missing", _rx(r"\babsen(?:ce|t)\b|\bedentul|\bmissing\b|\bagenes"
                             r"|\bnot present\b|\black of (?:tooth|teeth|element)"
                             r"|\bpost[\s-]?extract|\bavuls|\bloss of (?:tooth|teeth|dental element)"
                             r"|\bpreviously extracted")),
    # --- orientation --------------------------------------------------------
    ("orientation", "mesioverted", _rx(r"\bmesio[\s-]?(?:vert|inclin|angul|version|lingual|buccal)"
                                       r"|\bmesialis|\bmesializ|\bmesially (?:inclin|tilt|angul)"
                                       r"|\bmesial (?:inclination|tipping|version)")),
    ("orientation", "distoverted", _rx(r"\bdisto[\s-]?(?:vert|inclin|angul|version)"
                                       r"|\bdistally (?:inclin|tilt|angul)"
                                       r"|\bdistal (?:inclination|tipping|version)")),
    # --- endo (base trigger is ENDO_TRIGGER below; the value is picked by ENDO_SUBTYPE) ---
    # --- periapical ---------------------------------------------------------
    ("periapical", "cementoma", _rx(r"\bcementoma|\bdenticle\b|\bcemento[\s-]?osseous")),
    ("periapical", "osteosclerosis", _rx(r"\bosteosclero|\bosteo[\s-]?condens|\bcondensing osteitis"
                                         r"|\bosteodense|\bsclerotic\b|\bosteo[\s-]?dense"
                                         r"|\bincreased bone density|\bhyperdense (?:area|zone|signal)"
                                         r"|\bradiopaque area|\bosteosclerotic")),
    ("periapical", "nonapical_osteorarefaction",
     _rx(r"\b(?:interdental|inter[\s-]?radicular|pericoronal)[^.]{0,25}"
         r"(?:osteo[\s-]?rarefact|rarefact|radiolucen)"
         r"|\bosteo[\s-]?rarefact\w*[^.]{0,20}(?:interdental|between)")),
    ("periapical", "apical_osteorarefaction",
     _rx(r"\bperi[\s-]?apical\b|\bapico[\s-]?radicular|\bperi[\s-]?radicular"
         r"|\bapical (?:osteo[\s-]?)?rarefact|\bgranuloma|\bradicular cyst"
         r"|\bapical (?:lesion|radiolucen|osteolysis|osteorarefaction)"
         r"|\bosteorarefaction at (?:the )?apex|\brarefaction at (?:the )?ap"
         r"|\blesion .{0,20}apex|\bapex.{0,25}(?:rarefact|radiolucen|lesion)"
         r"|\bosteo[\s-]?rarefact|\bbone rarefaction|\bosteolytic\b")),
    # --- restoration --------------------------------------------------------
    ("restoration", "pontic", _rx(r"\bpontic|\bintermediate element|\bprosthetic replacement element"
                                  r"|\bdummy\b|\bprosthetic element replacing")),
    ("restoration", "denture_abutment", _rx(r"\bclasp\b|\bremovable (?:partial )?(?:denture|prosthe)"
                                            r"|\bskeletal removable|\bbar[\s-]?type"
                                            r"|\bprosthetic attachment|\battachments? for"
                                            r"|\bpartial denture")),
    ("restoration", "prosthetic_crown",
     _rx(r"\bprosthetic crown|\bcrowns? (?:on|of|anchored|covering|over|in)\b"
         r"|\bcovered by (?:a |the )?(?:prosthetic )?crown|\bcapsul"
         r"|\bbridge\b|\babutment|\bfixed prosthe|\bprosthetic (?:restoration|rehabilitation|device)"
         r"|\bprosthetic(?:ally)? (?:crown|rehabilitat|restor)|\bsplint"
         r"|\bcircular prosthe|\bfull[\s-]arch (?:bridge|prosthe)|\bcoronal[\s-]type prosthetic"
         r"|\bimplant[\s-]?prosthetic|\bimplant[\s-]?supported")),
    ("restoration", "filling",
     _rx(r"\bconservative[^.]{0,20}(?:treatment|restor|therap|fill)|\bblack class"
         r"|\bcoronal restorat|\brestorative (?:treatment|material|fill)"
         r"|\bcomposite\b|\bamalgam\b|\brestorations? (?:on|of|involving|in)\b"
         r"|\bmultiple restor|\brestorations?\b|\bfillings?\b")),
    # --- fracture -----------------------------------------------------------
    ("fracture", "fractured_endo_material",
     _rx(r"\bfractur\w*\s+(?:endodontic|instrument|file|canal)"
         r"|\b(?:endodontic|instrument|file)\s+fractur")),
    ("fracture", "root_fracture", _rx(r"\broot fractur|\bradicular fractur"
                                      r"|\bfractur\w*[^.]{0,30}\broot\b")),
    ("fracture", "crown_fracture", _rx(r"\bcoronal fractur|\bcrown fractur"
                                       r"|\bfractur\w*[^.]{0,25}\bcrown\b|\bfractur")),
    # --- atrophy ------------------------------------------------------------
    ("atrophy", "severe", _rx(r"\b(?:severe|marked|advanced|significant)[^.]{0,25}atroph"
                              r"|\batroph\w*[^.]{0,20}(?:severe|marked|advanced)"
                              r"|\bcawood[^.]{0,25}\b(?:v|vi)\b")),
    ("atrophy", "mild", _rx(r"\b(?:mild|slight|initial|minimal)[^.]{0,25}atroph"
                            r"|\batroph\w*[^.]{0,20}(?:mild|slight|initial)"
                            r"|\bcawood[^.]{0,25}\biii\b")),
    ("atrophy", "moderate", _rx(r"\batroph|\bcawood|\bbone resorption of the (?:alveolar )?(?:ridge|crest)"
                                r"|\bridge resorption|\bcrestal resorption")),
    # --- caries -------------------------------------------------------------
    ("caries", "caries", _rx(r"\bcaries\b|\bcarious\b|\bdecay\b|\bcavitat")),
    # --- post_core ----------------------------------------------------------
    ("post_core", "post_and_core", _rx(r"\bpost[\s-]?(?:and|&)[\s-]?core|\bpost and core"
                                       r"|\bpost[\s-]?core|\bendocanal(?:ar)? (?:post|pin)"
                                       r"|\b(?:metal|fiber|fibre|screw) posts?\b"
                                       r"|\brebuilt with (?:a )?post|\bintracanal post"
                                       r"|\bposts? (?:in|on|of) (?:the )?(?:tooth|teeth|canal|root)")),
    # --- periodontal --------------------------------------------------------
    ("periodontal", "marginal_bone_loss",
     _rx(r"\bperiodontitis|\bperiodontal (?:pocket|disease|bone)|\bperiodontopath"
         r"|\bmarginal bone|\bhorizontal (?:bone )?(?:resorption|loss)"
         r"|\bvertical (?:bone )?(?:resorption|loss)|\bbone loss\b"
         r"|\broot exposure|\bfurcation|\bpocket\b"
         r"|\bbone resorption\b")),
]

# Per-rule veto cues — if one is present in the sentence, that (axis, value) is not emitted.
RULE_VETO: dict[tuple[str, str], re.Pattern] = {
    ("periapical", "nonapical_osteorarefaction"): _rx(r"\bperi[\s-]?implant|\bperiodontal pocket"),
    ("periapical", "apical_osteorarefaction"): _rx(r"\bperi[\s-]?implant|\bperiodontal pocket"),
    ("restoration", "prosthetic_crown"): _rx(r"\bconservative|\brestorations? (?:on|of) the crowns?"),
    ("restoration", "filling"): _rx(r"\b(?:canal|root) filling|\bfilling material"
                                    r"|\bendodontic\b.{0,40}\bfilling"),
    ("periodontal", "marginal_bone_loss"): _rx(r"\bperi[\s-]?implant"),
}

ENDO_TRIGGER = _rx(r"\bendodontic|\bendodontically|\broot canal|\bcanal (?:filling|obturation)"
                   r"|\bobturation\b|\bdevitaliz|\bpulpectom|\bcanal treatment"
                   r"|\bfilling material .{0,20}(?:canal|apex|root)")
ENDO_SUBTYPE: list[tuple[str, re.Pattern]] = [
    ("overfill", _rx(r"\bbeyond the ap|\bpast the ap|\boverfill|\bextrud\w* (?:root |endodontic )?"
                     r"(?:filling )?material|\bcaudal to the (?:root )?ap|\bperiradicular endodontic"
                     r"|\bextending beyond|\bover[\s-]?extend|\boutside the ap")),
    ("underfill", _rx(r"\bnot (?:reach|extend|arriv)|\bdoes not reach|\bshort of the ap"
                      r"|\bmm (?:from|short of) the ap|\bfrom the ap(?:ex|ices)\b"
                      r"|\bunderfill|\bnot perfectly reach|\bbut not (?:the )?mesial"
                      r"|\bwithout reaching|\bfail\w* to reach|\bup to (?:mid|the mid|the middle)"
                      r"|\bmid[\s-]root|\bnot up to the ap")),
    ("incomplete", _rx(r"\bdiscontinu|\bincongru|\bnon[\s-]?congru|\binadequa|\bnot adequa"
                       r"|\bdoubtful|\bincomplete|\bpartial\b|\bnot correctly|\bimproper"
                       r"|\bnot clearly identifiable|\bonly at the ap|\bnot congru")),
]

# Cues that negate a finding (suppress the rule when found in the preceding window).
_NEG_CUE = _rx(r"\b(?:no|not|without|nor|neither|free of|negative for|absence of (?:evident )?"
               r"(?:signs?|lesions?|caries|carious))\b")
# state=missing / not_available are exempt: the negation word is itself their trigger.
_NEG_EXEMPT = {("state", "missing"), ("state", "not_available")}
_NEG_WINDOW = 45

_PRESENCE_RE = _rx(r"\bpresen(?:t|ce)\b|\bin the arch\b|\bcomplete dentition|\bdentition\b")


def _negated(sent: str, start: int) -> bool:
    win = sent[max(0, start - _NEG_WINDOW):start]
    return bool(_NEG_CUE.search(win))


def _sentences(text: str) -> list[str]:
    """Split generated or reference text into statements (on line breaks, bullets, semicolons)."""
    out: list[str] = []
    for line in re.split(r"[\n;]+", text or ""):
        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if line:
            out.append(line)
    return out

# ---------------------------------------------------------------------------
# 3. Claim extraction
# ---------------------------------------------------------------------------

Claim = tuple[str, str, object]


def _tooth_findings(sent: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Candidate (axis, value, trigger span) in a sentence. One per axis (the earliest rule)."""
    found: dict[str, tuple[str, tuple[int, int]]] = {}
    for axis, value, pat in TOOTH_RULES:
        if axis in found:
            continue
        m = pat.search(sent)
        if not m:
            continue
        if (axis, value) not in _NEG_EXEMPT and _negated(sent, m.start()):
            continue
        veto = RULE_VETO.get((axis, value))
        if veto is not None and veto.search(sent):
            continue
        found[axis] = (value, m.span())
    m = ENDO_TRIGGER.search(sent)
    if m and not _negated(sent, m.start()):
        val = "complete"
        for v, pat in ENDO_SUBTYPE:
            if pat.search(sent):
                val = v
                break
        found["endo"] = (val, m.span())
    # When state=missing and a presence statement share one sentence (e.g. "presence of teeth
    # ... except 48"), missing applies only to the except list -> handled in _extract_tooth.
    return [(ax, v, p) for ax, (v, p) in found.items()]


_BRIDGE_SPAN_RE = re.compile(
    r"(?:bridge|prosthe\w+|splint\w*|rehabilitation)[^.]{0,60}?(?:from|between|extending)[^.]{0,20}?"
    r"([1-4][1-8])[^.]{0,15}?(?:to|and|[-–])[^.]{0,15}?([1-4][1-8])", re.I)


def _bridge_spans(text: str) -> set[int]:
    """Teeth covered by a prosthetic span such as 'bridge from 33 to 43' — a missing tooth
    inside such a span is a pontic."""
    out: set[int] = set()
    for sent in _sentences(text):
        for m in _BRIDGE_SPAN_RE.finditer(sent):
            out.update(_arch_range(int(m.group(1)), int(m.group(2))))
    return out


def _extract_tooth(text: str, region_fdi: int | None, expand_groups: bool) -> set[Claim]:
    claims: set[Claim] = set()
    for sent in _sentences(text):
        targets, excluded = parse_teeth(sent, expand_groups=expand_groups)
        m = _EXCEPT_RE.search(sent)
        findings = _tooth_findings(sent[:m.start()] if m else sent)
        if not findings:
            # "presence of all teeth except 48" -> 48 is missing
            if excluded and _PRESENCE_RE.search(sent):
                for t in excluded:
                    claims.add((str(t), "state", "missing"))
            continue
        head = sent[:m.start()] if m else sent
        groups = [g for g in number_groups(head) if g[2] & targets]
        for axis, value, span in findings:
            # Attach to the number group nearest the trigger — needed when one sentence carries
            # statements about several teeth (e.g. "bridge between 23 and 25 replacing missing
            # tooth 24").
            tset = (nearest_group(head, span, groups) & targets) if groups else set()
            if not tset:
                tset = targets
            if not tset:
                # With no number the region id supplies the target (the key advantage of
                # per-structure decode).
                if region_fdi is None or region_fdi in excluded:
                    continue
                tset = {region_fdi}
            for t in tset:
                claims.add((str(t), axis, value))
        if excluded and _PRESENCE_RE.search(sent) and not any(
                a == "state" for a, _, _ in findings):
            for t in excluded:
                claims.add((str(t), "state", "missing"))
    return _postprocess(_resolve_softmax(claims), _bridge_spans(text))


# A finding stated on a missing tooth is clinically a different claim — this follows the
# ground-truth curation convention.
def _postprocess(claims: set[Claim], spans: set[int]) -> set[Claim]:
    state = {t: v for t, a, v in claims if a == "state"}
    out: set[Claim] = set()
    for t, axis, value in claims:
        st = state.get(t)
        if axis == "restoration" and value == "prosthetic_crown" and st == "missing":
            value = "pontic"                       # a crown on a missing site is a pontic
        if axis == "periodontal" and st in ("missing", "not_available", "implant"):
            continue                               # no tooth -> not a marginal bone loss claim
        if axis in ("endo", "caries", "post_core") and st in ("missing", "not_available"):
            continue                               # describes residual material, not the tooth
        out.add((t, axis, value))
    for ti in spans:                               # missing tooth inside a prosthetic span
        t = str(ti)
        if state.get(t) == "missing" and not any(
                c[0] == t and c[1] == "restoration" for c in out):
            out.add((t, "restoration", "pontic"))
    return out


# If several values match for the same (target, axis), keep only the highest-priority one
# (these are softmax axes).
_AXIS_PRIORITY: dict[str, list[str]] = {}
for _ax, _val, _ in TOOTH_RULES:
    _AXIS_PRIORITY.setdefault(_ax, []).append(_val)
_AXIS_PRIORITY["endo"] = ["overfill", "underfill", "incomplete", "complete"]


def _resolve_softmax(claims: Iterable[Claim]) -> set[Claim]:
    best: dict[tuple[str, str], str] = {}
    for t, axis, value in claims:
        order = _AXIS_PRIORITY.get(axis, [])
        cur = best.get((t, axis))
        if cur is None or (value in order and (cur not in order
                                               or order.index(value) < order.index(cur))):
            best[(t, axis)] = value
    return {(t, a, v) for (t, a), v in best.items()}


# --- Non-tooth regions ------------------------------------------------------

_MAX_SINUS_INCL = _rx(r"\bsinus")
_MAX_NOT_INCL = _rx(r"\bnot included|\bnot assessab|\bnot evaluab|\bnot visualiz|\bexclud"
                    r"|\boutside|\bnot represent|\bnot in the (?:scan|volume)")
_MAX_PNEUM_UP = _rx(r"\b(?:increased|marked|accentuated|hyper)[^.]{0,20}(?:pneumatiz|aerat)"
                    r"|\b(?:pneumatiz|aerat)\w*[^.]{0,15}(?:increased|marked|accentuated)")
_MAX_PNEUM_OK = _rx(r"\bpneumatiz|\baerat|\bincluded\b|\bpartially included|\bvisible portion")
_THICK = _rx(r"\bthicken|\bmucosal thicken|\bmucositis|\bsinusitis|\bmucosal reaction")
_RETENTION = _rx(r"\bretention cyst|\bmucocele|\bpolyp")
_RIGHT = _rx(r"\bright\b|\bdext")
_LEFT = _rx(r"\bleft\b|\bsinist")
_BILAT = _rx(r"\bbilateral|\bboth\b")
_CONDYLE = _rx(r"\bcondyl")
_CORONOID = _rx(r"\bcoronoid")
_CYST = _rx(r"\bcyst\b|\bcystic\b|\bkeratocyst|\bfollicular (?:lesion|cyst)")
_FRACTURE = _rx(r"\bfractur")
_PERIO_STAGE = [
    ("stage_III", _rx(r"\bstage\s*(?:iii|3)\b|\bsevere periodont|\badvanced periodont")),
    ("stage_II", _rx(r"\bstage\s*(?:ii|2)\b|\bmoderate periodont")),
    ("stage_I", _rx(r"\bstage\s*(?:i|1)\b|\bmild periodont|\binitial periodont")),
]
_PERIO_ANY = _rx(r"\bperiodontitis|\bperiodontal disease|\bperiodontopath")

_NERVE_LING = _rx(r"\blingual\b")
_NERVE_BUCC = _rx(r"\bbuccal\b|\bvestibular\b")
_NERVE_SUP = _rx(r"\bsuperficializ|\bsuperficial course|\bsuperficial\b")
_NERVE_NE = _rx(r"\bnot (?:included|evaluab|assessab|visualiz|visible)|\bcannot be (?:evaluat|assess)"
                r"|\bnon[\s-]?evaluable|\bexcluded")
_NEAR = _rx(r"\bclose (?:relationship|proximity|contact|contiguity)|\bin contact\b|\bcontigu"
            r"|\bintimate (?:relationship|contact)|\badjacent to|\bproximity")


def _extract_maxilla(text: str) -> set[Claim]:
    claims: set[Claim] = set()
    incl, mucosa = None, None
    for sent in _sentences(text):
        if _MAX_SINUS_INCL.search(sent):
            if _MAX_NOT_INCL.search(sent):
                incl = incl or "not_included"
            elif _MAX_PNEUM_UP.search(sent):
                incl = "pneumatization_increased"
            elif _MAX_PNEUM_OK.search(sent):
                incl = incl if incl == "pneumatization_increased" else "pneumatization_normal"
            side = ("both" if _BILAT.search(sent) else
                    "right" if _RIGHT.search(sent) else
                    "left" if _LEFT.search(sent) else "both")
            if _RETENTION.search(sent):
                mucosa = f"{side}_retention_cyst"
            elif _THICK.search(sent) and not _negated(sent, _THICK.search(sent).start()):
                if mucosa is None or "thickening" in mucosa:
                    mucosa = f"{side}_thickening"
        if _CYST.search(sent) and not _RETENTION.search(sent) and not _MAX_SINUS_INCL.search(sent):
            claims.add(("maxilla", "cyst", "anterior"))
        if _FRACTURE.search(sent) and not _negated(sent, _FRACTURE.search(sent).start()):
            claims.add(("maxilla", "fracture", "anterior"))
    # Out-of-field statements are emitted as claims too. `not_included` is index 0 of its axis
    # (the negative default) and used to be dropped, but "the sinus is outside the scanned volume"
    # is a statement about the acquisition rather than a normal finding, and radiologists do write
    # it. Without it those regions end up with zero reference claims and fall out of the scoring
    # entirely (35.2% of the non-tooth slots on the validation set). This breaks the "index 0 is
    # not a claim" convention for these three axes only; to revert it, change this spot and
    # _extract_mandible.
    if incl:
        claims.add(("maxilla", "sinus_inclusion", incl))
    if mucosa:
        claims.add(("maxilla", "sinus_mucosa", mucosa))
    claims |= _perio_claims(text, "maxilla")
    return claims


def _perio_claims(text: str, region: str) -> set[Claim]:
    for sent in _sentences(text):
        if not _PERIO_ANY.search(sent):
            continue
        for val, pat in _PERIO_STAGE:
            if pat.search(sent):
                return {(region, "perio_stage", val)}
        return {(region, "perio_stage", "stage_II")}
    return set()


def _extract_mandible(text: str) -> set[Claim]:
    claims: set[Claim] = set()
    for sent in _sentences(text):
        # Both inclusion and exclusion are emitted as claims (see the note in _extract_maxilla).
        # The decision is made per sentence — "condyles included, coronoid excluded" in a single
        # sentence is read as both being excluded (this coarseness is kept deliberately).
        neg = bool(_MAX_NOT_INCL.search(sent))
        if _CONDYLE.search(sent):
            claims.add(("mandible", "condyle_included", not neg))
        if _CORONOID.search(sent):
            claims.add(("mandible", "coronoid_included", not neg))
        if _CYST.search(sent) and not _negated(sent, _CYST.search(sent).start()):
            side = ("both" if _BILAT.search(sent) else "right" if _RIGHT.search(sent)
                    else "left" if _LEFT.search(sent) else "anterior")
            teeth, _ = parse_teeth(sent)
            if teeth:
                q = {t % 10 for t in teeth}
                quad = {t // 10 for t in teeth}
                loc = "wisdom" if 8 in q else "premolar" if q & {4, 5} else "posterior"
                if side == "anterior":
                    side = "right" if quad & {4} else "left" if quad & {3} else "anterior"
                if side != "anterior":
                    claims.add(("mandible", "cyst", f"{side}_{loc}"))
                else:
                    claims.add(("mandible", "cyst", "anterior"))
            else:
                claims.add(("mandible", "cyst", side if side == "anterior" else f"{side}_posterior"))
        if _FRACTURE.search(sent) and not _negated(sent, _FRACTURE.search(sent).start()):
            side = "right" if _RIGHT.search(sent) else "left"
            claims.add(("mandible", "fracture", f"{side}_body"))
    claims |= _perio_claims(text, "mandible")
    return claims


def _extract_nerve(text: str) -> set[Claim]:
    claims: set[Claim] = set()
    pos = None
    for sent in _sentences(text):
        if _NERVE_NE.search(sent):
            claims.add(("nerve", "not_evaluable", True))
        if _NERVE_LING.search(sent):
            pos = "lingual"
        elif _NERVE_BUCC.search(sent):
            pos = pos or "buccal"
        if _NERVE_SUP.search(sent):
            claims.add(("nerve", "superficialized", True))
        if _NEAR.search(sent) or re.search(r"\bimpacted\b", sent, re.I):
            teeth, _ = parse_teeth(sent, expand_groups=True)
            digits = {t % 10 for t in teeth}
            if 8 in digits:
                claims.add(("nerve", "canal_near_8", True))
            if 7 in digits:
                claims.add(("nerve", "canal_near_7", True))
    if pos:
        claims.add(("nerve", "position", pos))
    return claims


REGION_NONTEETH = {"maxilla": _extract_maxilla, "mandible": _extract_mandible,
                   "nerve_right": _extract_nerve, "nerve_left": _extract_nerve}


def extract_claims(text: str, region_id: str | None = None,
                   expand_groups: bool = True) -> set[Claim]:
    """Text -> set of claims.

    region_id
      "fdi:36"  tooth region — numberless statements attach to 36, and only 36's claims are kept.
      "maxilla"/"mandible"/"nerve_right"/"nerve_left" — only that region's axes.
      "cls_upper"/"cls_lower" — arch-level narrative. Only tooth claims that carry a number.
      None      whole report — every numbered tooth claim plus all non-tooth claims.
    """
    if not (text or "").strip():
        return set()
    if region_id and region_id.startswith("fdi:"):
        t = int(region_id[4:])
        out = _extract_tooth(text, t, expand_groups)
        return {c for c in out if c[0] == str(t)}
    if region_id in REGION_NONTEETH:
        return REGION_NONTEETH[region_id](text)
    if region_id in ("cls_upper", "cls_lower"):
        return _extract_tooth(text, None, expand_groups)
    # whole report
    out = _extract_tooth(text, None, expand_groups)
    out |= _extract_maxilla(text) | _extract_mandible(text) | _extract_nerve(text)
    return out

# ---------------------------------------------------------------------------
# 4. Structured ground truth -> claims
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# 5. Reward
# ---------------------------------------------------------------------------

def set_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    ov = len(pred & gold)
    if ov == 0:
        return 0.0
    p, r = ov / len(pred), ov / len(gold)
    return 2 * p * r / (p + r)


def _region_of(ctx: dict | None) -> str | None:
    if not ctx:
        return None
    return ctx.get("region_id")


def claim_f1(gen: str, gt: str, ctx: dict | None = None,
             excluded_axes: Iterable[str] | None = None) -> float:
    """Set F1 over claims extracted from the generated text and the reference by the *same*
    extractor (exact match, tooth number included).

    If `excluded_axes` is given, those axes are stripped from *both* sides before comparing
    (see FOV_AXES).
    """
    rid = _region_of(ctx)
    g, t = extract_claims(gen, rid), extract_claims(gt, rid)
    if excluded_axes:
        g, t = filter_claims(g, excluded_axes), filter_claims(t, excluded_axes)
    return set_f1(g, t)







# --------------------------------------------------------------------------- #
# Restricting which axes are scored
# --------------------------------------------------------------------------- #
# Spending reward on axes the model already gets right makes RFT selection pick samples that
# merely got those right once more, diluting the pressure on the hard axes. Measured greedy
# utterance rates: condyle_included:False 100.0% · coronoid_included:False 97.4% versus
# implant 0.0% · caries 0.0% · periapical 0.0% · filling 3.1%.
# `state:missing` (75.1%) stays in the scoring — 75% is still low.
# Excluding an axis from the scoring is not the same as excluding it from the training data:
#    excluded axes must still be fed as maintenance samples, otherwise the current behaviour is
#    lost (dropping normal regions from training once collapsed the silence rate from 0.8879 to
#    0.6547).
#
# An entry may be an axis name ("condyle_included") or axis:value ("state:not_available").
FOV_AXES: tuple[str, ...] = ("condyle_included", "coronoid_included", "sinus_inclusion",
                             "not_evaluable", "state:not_available")


def filter_claims(claims: Iterable[Claim], excluded: Iterable[str] | None) -> set[Claim]:
    """Drop claims whose axis (or axis:value) lies outside the scored set."""
    ex = set(excluded or ())
    if not ex:
        return set(claims)
    return {c for c in claims if c[1] not in ex and f"{c[1]}:{c[2]}" not in ex}
