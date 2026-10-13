"""Clinical terms — deterministic extraction and comparison of FDI numbers, finding vocabulary
and polarity.

Three independent terms
-----------------------
1. **FDI number agreement** (`fdi_report`) — absorbs notation variants (`36`, `3.6`, ranges),
   rejects unit false positives (`2 mm`), reports the invalid-number rate, and is region aware
   (`ctx["region_id"] == "fdi:XX"`). `reward.region_reward` turns that report into its FDI gate.
2. **Finding vocabulary agreement** (`term_f1`) — F1 over the **set of finding types**, using a
   surface-form lexicon for the schema axis values.
3. **Negation / presence polarity** (`polarity_f1`) — F1 over (concept, sign) pairs. A polarity
   flip is wrong on both sides, so the score drops sharply.

Everything is regex based: tens of microseconds on CPU, with no model or network call.
"""

from __future__ import annotations

import re

from .extract import ARCH_LOWER, ARCH_UPPER, set_f1 as _set_f1

# ============================================================================
# 1. FDI numbers
# ============================================================================

# FDI order along the arch (used to expand `from A to B` ranges) — same source of truth as extract.
ARCH_POS: dict[int, tuple[int, int]] = {}
for _a, _seq in ((0, ARCH_UPPER), (1, ARCH_LOWER)):
    for _i, _f in enumerate(_seq):
        ARCH_POS[_f] = (_a, _i)
FDI_VALID = frozenset(ARCH_POS)                      # = schema FDI_ALL (verified by a self-check)


# A number followed by a unit is not a tooth number (`2 mm`, `90°`, `0.9 cm`, `20%`).
_UNIT = r"(?!\s*(?:mm|cm|mms|%|°|degree))"
# `36` form — two digits with no adjacent digit or decimal point.
_RE_PLAIN = re.compile(r"(?<![\d.])(\d{2})(?![\d])" + _UNIT)
# `3.6` form (common in the Italian reports). It occurs in both ground truth and generations.
_RE_DOTTED = re.compile(r"(?<![\d.])([1-4])\.([1-9])(?![\d])" + _UNIT)
# Ranges: `from 13 to 23`, `teeth 45-48`, `33 – 43`
_RE_RANGE_WORD = re.compile(
    r"\b(?:from|between)\s+(?:tooth\s+|teeth\s+|element[s]?\s+)?(\d{2})\s+(?:to|and|-|–)\s+"
    r"(?:tooth\s+|teeth\s+|element[s]?\s+)?(\d{2})\b", re.I)
_RE_RANGE_DASH = re.compile(r"(?<![\d.])(\d{2})\s*[-–—]\s*(\d{2})(?![\d])")

# Contextual cues that a number refers to a tooth — used to decide whether it is invalid.
_RE_TOOTH_CTX = re.compile(
    r"\b(?:tooth|teeth|element|elements|e\.d\.|fdi|dente|denti)\b", re.I)


def _expand_range(a: int, b: int) -> list[int]:
    """Expand A..B in arch order if both are in the same arch; otherwise keep the endpoints."""
    if a not in ARCH_POS or b not in ARCH_POS:
        return [x for x in (a, b) if x in ARCH_POS]
    (aa, ia), (ab, ib) = ARCH_POS[a], ARCH_POS[b]
    if aa != ab:
        return [a, b]
    seq = ARCH_UPPER if aa == 0 else ARCH_LOWER
    lo, hi = (ia, ib) if ia <= ib else (ib, ia)
    return seq[lo:hi + 1]


def extract_fdi(text: str, expand_ranges: bool = True) -> tuple[set[int], list[str]]:
    """Text -> (set of valid FDI numbers, list of invalid tooth numbers).

    Invalid = a two-digit number in a tooth context that is not an FDI number (`39`, `49`, `19`,
    `56`), or a quadrant notation such as `3.9` whose tooth position is 9. Deciduous teeth (51-85)
    fall outside the 32-tooth schema, so they are ignored rather than counted as invalid (they are
    outside the adult schema, not an error).
    """
    valid: set[int] = set()
    invalid: list[str] = []

    for m in _RE_DOTTED.finditer(text):
        n = int(m.group(1)) * 10 + int(m.group(2))
        (valid.add(n) if n in FDI_VALID else invalid.append(m.group(0)))

    for m in _RE_PLAIN.finditer(text):
        n = int(m.group(1))
        if n in FDI_VALID:
            valid.add(n)
            continue
        if 51 <= n <= 85:                    # deciduous notation — ignored
            continue
        # Only counted as invalid inside a tooth context (40 characters before, 20 after).
        lo, hi = max(0, m.start() - 40), m.end() + 20
        if _RE_TOOTH_CTX.search(text[lo:hi]) or (10 <= n <= 49):
            invalid.append(m.group(0))

    if expand_ranges:
        for rx in (_RE_RANGE_WORD, _RE_RANGE_DASH):
            for m in rx.finditer(text):
                a, b = int(m.group(1)), int(m.group(2))
                if a in FDI_VALID and b in FDI_VALID:
                    valid.update(_expand_range(a, b))
    return valid, invalid


def _own_fdi(ctx: dict | None) -> int | None:
    if not ctx:
        return None
    rid = ctx.get("region_id")
    if isinstance(rid, str) and rid.startswith("fdi:"):
        try:
            return int(rid.split(":", 1)[1])
        except ValueError:
            return None
    return None


def fdi_report(gen: str, gt: str, ctx: dict | None = None,
               expand_ranges: bool = True) -> dict:
    """All sub-quantities of the FDI term at once (for debugging and breakdown reporting)."""
    P, inv = extract_fdi(gen, expand_ranges)
    G, _ = extract_fdi(gt, expand_ranges)
    own = _own_fdi(ctx)
    ov = len(P & G)
    out = {
        "n_gen": len(P), "n_gt": len(G), "overlap": ov,
        "precision": ov / len(P) if P else (1.0 if not G else 0.0),
        "recall": ov / len(G) if G else (1.0 if not P else 0.0),
        "set_f1": _set_f1(P, G),
        "n_invalid": len(inv),
        "invalid_rate": len(inv) / (len(inv) + len(P)) if (inv or P) else 0.0,
        "own_fdi": own,
        "own_ok": None,
        "off_region_rate": None,
    }
    if own is not None:
        # For a per-tooth slot the correct tooth is fixed: it is `own`.
        if own in G:
            out["own_ok"] = own in P             # did the generation mention the tooth the GT names
        elif not G:
            out["own_ok"] = not P                # normal slot: no number should be mentioned at all
        else:
            out["own_ok"] = own in P or bool(P & G)   # GT phrased as a range or a quadrant
        out["off_region_rate"] = len(P - G) / len(P) if P else 0.0
    return out












# ============================================================================
# 2. Finding vocabulary (schema axis value -> surface-form lexicon)
# ============================================================================
# Code granularity = **as fine as the vocabulary can distinguish**. For instance partial and full
# impaction share surface forms, so they are merged into a single code (`state:impacted`). The
# surface forms below were derived by comparing against the per-axis labels of the ground-truth
# corpus (1,003 reports).

_TERM_PATTERNS: list[tuple[str, str]] = [
    # ---- per-tooth: state ------------------------------------------------
    ("state:missing", r"absen\w*|missing|edentul\w*|agenesi\w*|lack of (?:tooth|teeth)"
                      r"|post[- ]?extract\w*\s+(?:site|saddle)|extraction site"),
    ("state:impacted", r"impact\w*|disodontias\w*|inclus[oa]\w*\s+(?:dental|tooth)"
                       r"|semi[- ]?impacted|retained (?:and )?impacted|pericoronit\w*"),
    ("state:implant", r"implant\w*|endosseous|osseointegrat\w*|fixture"),
    ("state:root_rest", r"root remnant\w*|root fragment\w*|residual root\w*|retained root"),
    ("state:supraeruption", r"extrud\w*|supra[- ]?erupt\w*|over[- ]?erupt\w*"),
    ("state:not_available", r"not visualizable|not included|excluded from|outside the (?:field|scan)"
                            r"|not (?:fully )?(?:visible|evaluable|assessable)|partially included"),
    # ---- per-tooth: orientation -----------------------------------------
    ("orientation:mesioverted", r"mesiovert\w*|mesio[- ]?angulat\w*|mesially (?:inclined|tilted)"),
    ("orientation:distoverted", r"distovert\w*|disto[- ]?angulat\w*|distally (?:inclined|tilted)"),
    # ---- per-tooth: endo -------------------------------------------------
    ("endo:treated", r"endodontic\w*|endodontically treated|root canal (?:treatment|filling|therapy)"),
    ("endo:overfill", r"beyond the ap(?:ex|ices)|overfill\w*|extending beyond|over[- ]?extend\w*"),
    ("endo:underfill", r"not reaching|short of the ap(?:ex|ices)|underfill\w*"
                       r"|\d\s*mm from the ap(?:ex|ices)|does not reach"),
    ("endo:inadequate", r"inadequate|incongru\w*|insufficient|incomplete\w*|non[- ]?homogeneous"),
    # ---- per-tooth: periapical ------------------------------------------
    ("periapical:rarefaction", r"periapical|apical (?:radiolucen\w*|lesion|osteo\w*)|granulom\w*"
                               r"|osteorarefaction|rarefaction|periradicular"
                               r"|radiolucen\w* (?:area|lesion|image)\w* (?:at|of|around) the ap"),
    ("periapical:sclerosis", r"condensing osteitis|osteocondensation|osteosclerosis|sclerot\w*"),
    ("periapical:cementoma", r"cementom\w*|cemento[- ]?osseous"),
    # ---- per-tooth: restoration -----------------------------------------
    ("restoration:filling", r"conservative (?:restoration|dental|filling)\w*|fillings?"
                            r"|composite|coronal restoration\w*|restorative|black class"),
    # Context is required so that an anatomical crown (`crown of tooth 36 with caries`) is not
    # confused with a prosthetic crown.
    ("restoration:crown", r"prosthetic crown\w*|crowns?\s+(?:on|of teeth|from|between|in )"
                          r"|covered by (?:a |the )?(?:prosthetic )?crown|crowned|abutment\w*"
                          r"|capsul\w*|onlay|overlay|\d\s+prosthetic crown"),
    ("restoration:bridge", r"bridge\w*|pontic\w*|fixed prosthe\w*|cantilever"
                           r"|prosthetic replacement element"),
    ("restoration:denture", r"removable|denture\w*|clasp\w*|attachment\w*|toronto|bar[- ]?type"),
    # ---- per-tooth: remaining axes --------------------------------------
    ("fracture:crown", r"coronal fracture|crown fracture|fracture of the crown"),
    ("fracture:root", r"root fracture|fracture of the root|radicular fracture"),
    ("caries", r"cari(?:es|ous)|decay|destructive process"),
    ("post_core", r"post[- ]and[- ]core|post and core|pin[- ]?retained"
                  r"|posts?\b(?![-\s]*(?:extract|surg|operat|traumat))"),
    ("periodontal", r"periodont\w*|marginal bone loss|bone resorption|periodontal pocket"
                    r"|furcation|bone loss"),
    ("atrophy", r"atroph\w*|cawood|bone (?:crest )?(?:resorption of the )?ridge"),
    # ---- non-tooth: maxilla ----------------------------------------------
    ("sinus:pneumatization", r"pneumatiz\w*"),
    ("sinus:mucosa", r"mucosal thickening|mucosa\w* (?:thickening|hypertroph\w*)|sinusit\w*"
                     r"|retention cyst|mucous cyst|inflammatory (?:thickening|mucosal)"),
    ("sinus:included", r"maxillary sinus\w*|caudal portion"),
    # ---- non-tooth: mandible ---------------------------------------------
    ("mandible:condyle", r"condyl\w*"),
    ("mandible:coronoid", r"coronoid"),
    ("bone:cyst", r"\bcyst\w*|follicular|dentigerous|radicular cyst|lesion"),
    ("bone:fracture", r"fracture of the (?:mandible|body|angle|ramus)|mandibular fracture"),
    # ---- non-tooth: mandibular canal -------------------------------------
    ("nerve:canal", r"mandibular canal|alveolar (?:nerve|canal)|inferior alveolar"),
    ("nerve:lingual", r"lingual"),
    ("nerve:buccal", r"buccal|vestibular"),
    ("nerve:emergence", r"emergen\w*|mental foramen|foramen"),
    ("nerve:proximity", r"(?:close|proximity|contact|adjacen\w*|relationship) (?:to|with)"
                        r"|in contact|near the roots|intimate"),
]

TERM_LEXICON: list[tuple[str, re.Pattern]] = [
    (code, re.compile(pat, re.I)) for code, pat in _TERM_PATTERNS]
TERM_CODES = tuple(code for code, _ in _TERM_PATTERNS)


def extract_terms(text: str) -> set[str]:
    """Text -> set of finding-type codes that occur in it."""
    if not text.strip():
        return set()
    return {code for code, rx in TERM_LEXICON if rx.search(text)}


def term_f1(gen: str, gt: str, ctx: dict | None = None) -> float:
    """(2) F1 over the set of finding-vocabulary (type) codes."""
    return _set_f1(extract_terms(gen), extract_terms(gt))


# ============================================================================
# 3. Negation / presence polarity
# ============================================================================
# (concept, positive pattern, negative pattern). If the same concept appears with opposite signs
# on the two sides, the flip is penalised twice over.
_POLARITY_SPEC: list[tuple[str, str, str]] = [
    ("tooth_presence",
     r"present in the arch|presence of (?:tooth|teeth|dental element)|in the arch|dentate"
     r"|complete (?:maxillary |mandibular )?dentition|erupted",
     r"absen\w*|missing|edentul\w*|agenesi\w*|lack of (?:tooth|teeth)"),
    ("fov_inclusion",
     r"\bincluded\b|\bvisible\b|visualizable|within the (?:acquisition|scan|field)|evaluable",
     r"not included|excluded|not visualizable|not visible|outside the (?:field|scan|acquisition)"
     r"|not (?:fully )?evaluable|beyond the (?:field|scan)"),
    ("canal_course", r"regular course|normal course|regularly", r"irregular course|irregular"),
    ("evidence", r"evidence of|signs of|suggestive of|compatible with",
     r"no evidence|no signs|absence of signs|without (?:evidence|signs)"),
    ("pneumatization", r"normally pneumatized|normal pneumatization|well pneumatized",
     r"not pneumatized|reduced pneumatization|hypo[- ]?pneumatiz\w*"),
    ("osseointegration", r"(?:correctly|well|adequately) osseointegrated|adequate osseointegration",
     r"not osseointegrated|peri[- ]?implant (?:radiolucen|bone loss)|failed"),
    ("pathology_free", r"\bnormal\w*|\bregular\b|adequate|physiolog\w*|without (?:lesion|pathol)",
     r"patholog\w*|abnormal|altered|lesion"),
]
POLARITY_RULES = [(c, re.compile(p, re.I), re.compile(n, re.I)) for c, p, n in _POLARITY_SPEC]
POLARITY_CONCEPTS = tuple(c for c, _, _ in _POLARITY_SPEC)


def extract_polarity(text: str) -> set[tuple[str, str]]:
    """Text -> {(concept, '+'|'-')}. If a concept appears with both signs, both are kept (a
    sentence that mixes them)."""
    out: set[tuple[str, str]] = set()
    if not text.strip():
        return out
    for concept, pos, neg in POLARITY_RULES:
        # The negative form is often a superstring of the positive one (`not included` contains
        # `included`), so the spans matched by the negative pattern are blanked out before the
        # positive pattern is searched.
        masked, nhit = neg.subn(" ", text)
        if nhit:
            out.add((concept, "-"))
        if pos.search(masked):
            out.add((concept, "+"))
    return out


def polarity_f1(gen: str, gt: str, ctx: dict | None = None) -> float:
    """(3) F1 over the set of (concept, sign) pairs. A flip costs both precision and recall."""
    return _set_f1(extract_polarity(gen), extract_polarity(gt))


def polarity_contradictions(gen: str, gt: str) -> int:
    """Number of concepts the generation and the ground truth state with **opposite signs**."""
    P, G = extract_polarity(gen), extract_polarity(gt)
    n = 0
    for c in POLARITY_CONCEPTS:
        if ((c, "+") in P and (c, "-") in G and (c, "+") not in G) or \
           ((c, "-") in P and (c, "+") in G and (c, "-") not in G):
            n += 1
    return n




# ============================================================================
# 4. Combinations
# ============================================================================

W_FDI, W_TERM, W_POL = 0.40, 0.40, 0.20

























# ============================================================================
# Lexicon validation — checked against the categorical ground truth (structured labels)
# ============================================================================
# Axis category -> code above. Values the vocabulary cannot separate map to the same code.
_AXIS_TO_CODE: dict[tuple[str, str], str | None] = {
    ("state", "missing"): "state:missing",
    ("state", "partial_impacted"): "state:impacted",
    ("state", "full_impacted"): "state:impacted",
    ("state", "implant"): "state:implant",
    ("state", "root_rest"): "state:root_rest",
    ("state", "supraeruption"): "state:supraeruption",
    ("state", "not_available"): "state:not_available",
    ("orientation", "mesioverted"): "orientation:mesioverted",
    ("orientation", "distoverted"): "orientation:distoverted",
    ("endo", "complete"): "endo:treated",
    ("endo", "overfill"): "endo:overfill",
    ("endo", "underfill"): "endo:underfill",
    ("endo", "incomplete"): "endo:inadequate",
    ("periapical", "apical_osteorarefaction"): "periapical:rarefaction",
    ("periapical", "nonapical_osteorarefaction"): "periapical:rarefaction",
    ("periapical", "osteosclerosis"): "periapical:sclerosis",
    ("periapical", "cementoma"): "periapical:cementoma",
    ("restoration", "filling"): "restoration:filling",
    ("restoration", "prosthetic_crown"): "restoration:crown",
    ("restoration", "pontic"): "restoration:bridge",
    ("restoration", "denture_abutment"): "restoration:denture",
    ("fracture", "crown_fracture"): "fracture:crown",
    ("fracture", "root_fracture"): "fracture:root",
    ("fracture", "fractured_endo_material"): None,      # surface form too rare (n<=2)
    ("fracture", "fractured_post_material"): None,
    ("atrophy", "mild"): "atrophy",
    ("atrophy", "moderate"): "atrophy",
    ("atrophy", "severe"): "atrophy",
    ("caries", "caries"): "caries",
    ("post_core", "post_and_core"): "post_core",
    ("periodontal", "marginal_bone_loss"): "periodontal",
}
_NEGATIVE_VALUES = {"normal", "none", "normoverted", False}
# Only the codes the lexicon covers are scored (codes absent from the schema are out of scope).
_VALIDATED_CODES = {c for c in _AXIS_TO_CODE.values() if c}
