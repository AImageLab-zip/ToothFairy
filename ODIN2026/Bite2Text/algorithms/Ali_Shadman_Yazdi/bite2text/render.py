"""Slot dicts -> report sentences.

Each sentence group first looks for an exact match in a bank of sentences mined
from the training reports (real clinical phrasing, keyed by the slot values it
expresses) and otherwise falls back to programmatic synthesis. A "close but not
exact" mined sentence is never reused, because it would state a wrong finding.

Two banks:
  data/templates_occlusal.json   transverse / vertical / sagittal / midlines / curves / crowding
  data/templates_photo.json      restorations, sealants, caries, staining, recession, hygiene
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .config import DATA_DIR

_OCCLUSAL = json.loads((DATA_DIR / "templates_occlusal.json").read_text(encoding="utf-8"))
_PHOTO = json.loads((DATA_DIR / "templates_photo.json").read_text(encoding="utf-8"))

# ===========================================================================
# Occlusal findings (from the scans)
# ===========================================================================

CONSTR_PHRASE = {
    "none": None, "slight": "a slight", "mild": "a mild", "moderate": "a moderate",
    "marked": "a marked", "unspecified": "a transverse",
}
OVERBITE_PHRASE = {
    "normal": "the vertical relationship is within normal limits",
    "increased": "the overbite is increased",
    "markedly_increased": "the overbite is markedly increased",
    "reduced": "the overbite is reduced",
    "markedly_reduced": "the overbite is markedly reduced",
    "reverse": "the overbite is reversed",
}
OVERJET_PHRASE = {
    "normal": "an overjet within normal limits",
    "increased": "increased overjet",
    "markedly_increased": "markedly increased overjet",
    "reduced": "reduced overjet",
    "markedly_reduced": "markedly reduced overjet",
    "reverse": "a negative overjet",
}
CURVE_PHRASE = {
    "normal": "within normal limits", "increased": "increased",
    "markedly_increased": "accentuated", "reduced": "reduced",
    "markedly_reduced": "markedly reduced", "reverse": "reversed",
}
CROWDING_PHRASE = {
    "none": "no", "slight": "slight", "mild": "mild", "moderate": "moderate",
    "severe": "severe", "mild_moderate": "mild-to-moderate",
    "moderate_severe": "moderate-to-severe",
}


def _occlusal_lookup(group: str, key: tuple) -> Optional[str]:
    return _OCCLUSAL.get(group, {}).get(json.dumps(list(key)))


def _transverse_vertical(s: dict) -> str:
    hit = _occlusal_lookup("transverse_vertical",
                           (s["constr"], s["xbite_present"], s["deep"] is True, s.get("overbite")))
    if hit:
        return hit

    constr, xbite = s["constr"], s["xbite_present"]
    if constr in (None, "none") and not xbite:
        s1 = "The patient presents correct transverse relationships."
    elif xbite:
        art = CONSTR_PHRASE.get(constr)
        s1 = (f"The patient presents {art} constriction of the maxilla with a crossbite."
              if art else "The patient presents a crossbite.")
    else:
        art = CONSTR_PHRASE.get(constr) or "a"
        s1 = f"The patient presents {art} constriction of the maxilla in the absence of crossbite."

    if s["deep"] is True:
        s2 = "From a vertical standpoint, there is a deep bite."
    else:
        s2 = f"From a vertical standpoint, {OVERBITE_PHRASE.get(s.get('overbite'), OVERBITE_PHRASE['normal'])}."
    return f"{s1} {s2}"


def _sagittal(s: dict) -> str:
    mr, cr, ml, cl, oj = s["mr_class"], s["cr_class"], s["ml_class"], s["cl_class"], s.get("overjet")
    hit = _occlusal_lookup("sagittal", (mr, cr, ml, cl, oj))
    if hit:
        return hit
    right = (f"a Class {mr} molar relationship and a Class {cr} canine relationship on the right"
             if mr == cr else f"a Class {mr} molar and a Class {cr} canine relationship on the right")
    left = (f"a Class {ml} molar and canine relationship on the left"
            if ml == cl else f"a Class {ml} molar and a Class {cl} canine relationship on the left")
    return f"Sagittally, there is {right}, and {left}, with {OVERJET_PHRASE.get(oj, OVERJET_PHRASE['normal'])}."


def _midlines(s: dict) -> str:
    hit = _occlusal_lookup("midlines", (s["mid"], s["mid_mag"]))
    if hit:
        return hit
    if s["mid"] == "coincident":
        return "The midlines are coincident."
    if s["mid_mag"] == "slight":
        return "The midlines are slightly deviated."
    return "The midlines are deviated."


def _curves(s: dict) -> str:
    sp, wl = s["spee"], s["wilson"]
    hit = _occlusal_lookup("curves", (sp, wl))
    if hit:
        return hit
    if sp == wl:
        return f"The Curve of Spee and the Curve of Wilson are {CURVE_PHRASE.get(sp, 'within normal limits')}."
    return (f"The Curve of Spee is {CURVE_PHRASE.get(sp, 'within normal limits')} "
            f"and the Curve of Wilson is {CURVE_PHRASE.get(wl, 'within normal limits')}.")


def _crowding(s: dict) -> str:
    cru, crl = s["cru"], s["crl"]
    hit = _occlusal_lookup("crowding", (cru, crl))
    if hit:
        return hit
    if cru == "none" and crl == "none":
        return "No crowding is present in the upper and lower arches."
    if cru == crl:
        return f"{CROWDING_PHRASE.get(cru, cru).capitalize()} crowding is present in the upper and lower arches."
    return (f"There is {CROWDING_PHRASE.get(cru, cru)} crowding in the upper arch "
            f"and {CROWDING_PHRASE.get(crl, crl)} crowding in the lower arch.")


def render_occlusal(slots: dict) -> str:
    parts = [_transverse_vertical(slots), _sagittal(slots), _midlines(slots),
             _curves(slots), _crowding(slots)]
    return " ".join(p for p in parts if p)


# ===========================================================================
# Photo findings
# ===========================================================================

# Narrative order the corpus uses: existing treatment, pathology, soft tissue, hygiene.
EMIT_ORDER = ["restorations", "sealants", "caries", "discoloration",
              "enamel_defect", "recession", "hygiene_gingiva"]

# Mined sentences that localise a finding ("on the first molars", "fifth
# sextant", tooth numbers). The photo heads predict presence/type only, so such
# a sentence would assert something that was not predicted.
_SPECIFIC_RE = re.compile(
    r"\b[1-8][1-8][a-z]{0,3}\b|"
    r"first molar|second molar|third molar|premolar|incisor|canine|molar|"
    r"sextant|quadrant|upper arch|lower arch|maxillar|mandibul|deciduous|"
    r"palatal|lingual|mesial|distal|cervical|posterior|anterior|"
    r"\bupper\b|\blower\b|\bright\b|\bleft\b",
    re.I)


def _photo_key(topic: str, ps: dict) -> tuple:
    if topic == "hygiene_gingiva":
        return (ps.get("hygiene"), ps.get("plaque"), ps.get("calculus"), ps.get("gingivitis"))
    if topic == "caries":
        return ((ps.get("caries") or {}).get("status"),)
    if topic in ("restorations", "discoloration", "enamel_defect"):
        return (tuple(sorted((ps.get(topic) or {}).get("type", []))),)
    if topic == "recession":
        return ((ps.get("recession") or {}).get("severity"),)
    if topic == "sealants":
        return ("present",)
    return ()


def _is_active(topic: str, ps: dict) -> bool:
    if topic == "hygiene_gingiva":
        return any(ps.get(k) is not None for k in ("hygiene", "plaque", "calculus", "gingivitis"))
    if topic == "sealants":
        return bool(ps.get("sealants"))
    return ps.get(topic) is not None


def _photo_fallback(topic: str, ps: dict) -> Optional[str]:
    if topic == "restorations":
        types = (ps.get("restorations") or {}).get("type", [])
        if "crown" in types and "restoration" in types:
            return "Restorations and prosthetic crowns are present."
        if "crown" in types:
            return "Prosthetic crowns are present."
        return "Restorations are present."
    if topic == "sealants":
        return "Fissure sealants are present."
    if topic == "caries":
        if (ps.get("caries") or {}).get("status") == "present":
            return "Carious lesions are present."
        return "The possible presence of caries should be evaluated."
    if topic == "discoloration":
        types = (ps.get("discoloration") or {}).get("type", [])
        if "white_spot" in types:
            return "Diffuse demineralizations are present."
        if "stain" in types:
            return "Diffuse extrinsic staining is present."
        return "Diffuse extrinsic pigmentation is present."
    if topic == "enamel_defect":
        types = (ps.get("enamel_defect") or {}).get("type", [])
        if "fracture" in types:
            return "An enamel fracture is present."
        if "wear" in types or "abrasion" in types:
            return "Wear of the incisal edges is noted."
        return "Enamel hypomineralization is present."
    if topic == "recession":
        severity = (ps.get("recession") or {}).get("severity")
        if severity in ("mild", "initial", "slight"):
            return "Mild buccal gingival recessions are present."
        if severity in ("marked", "significant"):
            return "Marked buccal gingival recessions are present."
        return "Buccal gingival recessions are present."
    if topic == "hygiene_gingiva":
        deposits = [name for key, name in (("plaque", "visible plaque"), ("calculus", "calculus"))
                    if ps.get(key)]
        parts = []
        if ps.get("hygiene") == "poor":
            parts.append("Poor oral hygiene" + (f" with {' and '.join(deposits)}" if deposits else "") + ".")
        elif deposits:
            joined = " and ".join(deposits).capitalize()
            parts.append(f"{joined} are present." if len(deposits) > 1 else f"{joined} is present.")
        gingivitis = ps.get("gingivitis")
        if gingivitis is True:
            parts.append("The gingivae are inflamed.")
        elif gingivitis == "uncertain":
            parts.append("Gingival inflammation may be present; however, this cannot be stated with certainty.")
        elif gingivitis is False:
            parts.append("No clear signs of gingival inflammation are observed.")
        return " ".join(parts) if parts else None
    return None


def render_photo_findings(photo_slots: dict, generic_only: bool = True) -> str:
    """Trailing photo-finding sentences (possibly empty).

    ``generic_only`` drops mined sentences that localise the finding and uses
    the generic synthesis instead.
    """
    out = []
    for topic in EMIT_ORDER:
        if not _is_active(topic, photo_slots):
            continue
        sentence = _PHOTO.get(topic, {}).get(json.dumps(_photo_key(topic, photo_slots), default=list))
        if sentence and generic_only and _SPECIFIC_RE.search(sentence):
            sentence = None
        sentence = sentence or _photo_fallback(topic, photo_slots)
        if sentence:
            out.append(sentence)
    return " ".join(out)
