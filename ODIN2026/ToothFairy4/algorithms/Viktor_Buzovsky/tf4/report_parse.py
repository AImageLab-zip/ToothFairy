"""Reference reports -> Findings, by regex.

This replaces the LLM `report_to_phrases` pass from the original plan. It is what
gives us (a) ground-truth slot values to calibrate the mask-derived findings
against, and (b) the reference side of the fast proxy metric.

Only high-confidence patterns are used: a slot left as None costs recall in the
proxy but never introduces a wrong calibration target.
"""

from __future__ import annotations

import re

from .schema import ALL_FDI, LOWER, THIRD_MOLARS, UPPER, Findings

FDI_RE = r"(?:1[1-8]|2[1-8]|3[1-8]|4[1-8])"

# "teeth 44, 45, 46, 48 and 38"  /  "e.d. 36-46-48"  /  "18-24-25-26"
_TOOTH_LIST = rf"((?:{FDI_RE})(?:\s*(?:,|-|and|e)\s*(?:{FDI_RE}))*)"

ABSENCE_RE = re.compile(
    rf"absen\w*\s+(?:of\s+)?(?:the\s+)?(?:teeth|tooth|e\.?d\.?|dental elements?|elements?)?\s*{_TOOTH_LIST}",
    re.I,
)
PRESENCE_RE = re.compile(
    rf"(?:presence of|present)\s+(?:teeth|tooth|e\.?d\.?|dental elements?)?\s*{_TOOTH_LIST}", re.I
)
FULL_LOWER_RE = re.compile(
    r"(teeth present from 38 to 48|dental elements from 38 to 48|present\w*.{0,40}from 38 to 48"
    r"|complete dentition|dentition is complete|all (?:the )?(?:lower |mandibular )?teeth are present)",
    re.I,
)
IMPACTED_RE = re.compile(rf"(?:tooth|e\.?d\.?|element)?\s*({FDI_RE})\s+(?:is\s+)?(?:impacted|included)\b", re.I)
IMPACTED_RE2 = re.compile(rf"(?:impacted|included)\s+(?:tooth|e\.?d\.?|element)?\s*({FDI_RE})", re.I)
IMPLANT_RE = re.compile(rf"implant\w*\s+(?:in|at|on)?\s*(?:position|site|region)?\s*({FDI_RE})", re.I)
CROWN_RE = re.compile(rf"crown\w*\s+(?:on|of|involving)\s+(?:tooth|e\.?d\.?|element)?\s*({FDI_RE})", re.I)
CANAL_CONTACT_RE = re.compile(
    rf"(right|left)\s+mandibular canal[^.]{{0,70}}?(?:close relation|in relation|relationship|contact|proximity)"
    rf"[^.]{{0,40}}?({FDI_RE})",
    re.I,
)


def _teeth(match_group: str) -> list[int]:
    return [int(t) for t in re.findall(FDI_RE, match_group)]


def parse_report(text: str, case_id: str = "") -> Findings:
    f = Findings(case_id=case_id)
    t = text

    # --- coverage ---------------------------------------------------------
    if re.search(r"condyl", t, re.I):
        # Order matters: "not included in the scan" contains "included in the scan",
        # so the negative form has to be tested first and short-circuit.
        neg = re.search(
            r"condyl[^.]{0,90}?(not included|not incuded|excluded|not visible|not present|not depicted"
            r"|not represented|not assessable|not visualiz|no scans)"
            r"|(?:no scans?|not included|excluded)[^.]{0,45}condyl",
            t, re.I,
        )
        if neg:
            f.condyles_included = False
        elif re.search(r"condyl[^.]{0,60}(included in the scan|are included|within the acquisition)", t, re.I):
            f.condyles_included = True

    if re.search(r"maxill", t, re.I):
        if re.search(r"maxill[^.]{0,90}partially included|partially included[^.]{0,40}maxill", t, re.I):
            f.maxilla_coverage = "partial"
        elif re.search(
            r"maxill[^.]{0,90}(not included|not present|excluded|no diagnostic|not depicted|not represented)"
            r"|(?:not included|excluded)[^.]{0,40}maxill",
            t, re.I,
        ):
            f.maxilla_coverage = "none"

    if re.search(r"mandib\w*\s+(?:bone|body|bodies)?[^.]{0,40}included in the (scan|acquisition)", t, re.I):
        f.mandible_included = True

    # --- canal ------------------------------------------------------------
    if re.search(r"canal[^.]{0,80}lingual|lingual[^.]{0,40}canal", t, re.I):
        f.canal_course = "lingual"
    elif re.search(r"canal[^.]{0,60}regular course|regular course[^.]{0,40}canal", t, re.I):
        f.canal_course = "regular"

    for m in CANAL_CONTACT_RE.finditer(t):
        f.canal_contact[m.group(1).lower()] = int(m.group(2))

    # --- dentition --------------------------------------------------------
    absent: set[int] = set()
    for m in ABSENCE_RE.finditer(t):
        absent.update(_teeth(m.group(1)))
    for fdi in absent:
        f.teeth_present[fdi] = False

    if FULL_LOWER_RE.search(t):
        for fdi in LOWER:
            f.teeth_present.setdefault(fdi, True)

    for m in PRESENCE_RE.finditer(t):
        for fdi in _teeth(m.group(1)):
            f.teeth_present.setdefault(fdi, True)

    for rx in (IMPACTED_RE, IMPACTED_RE2):
        for m in rx.finditer(t):
            fdi = int(m.group(1))
            # "included" is overloaded: it also means "inside the field of view".
            if rx is IMPACTED_RE2 and not re.search(r"impacted", m.group(0), re.I):
                continue
            f.impacted.add(fdi)

    if re.search(r"complete edentulism|total edentulism|edentulous.{0,20}both arch", t, re.I):
        f.edentulism = "complete"
    elif re.search(r"partial edentulism|semi-total edentulism|edentul", t, re.I):
        f.edentulism = "partial"

    # --- restorations -----------------------------------------------------
    for m in IMPLANT_RE.finditer(t):
        f.implants.add(int(m.group(1)))
    for m in CROWN_RE.finditer(t):
        f.crowns.add(int(m.group(1)))
    if re.search(r"\bbridge\b|prosthetic bridge|fixed bridge", t, re.I):
        f.has_bridge = True
    # Site-less variants, so they are comparable with the mask derivation (an
    # implant sits in a missing tooth site, so there is no tooth label to name).
    if re.search(r"implant", t, re.I):
        f.has_implant = True
    if re.search(r"crown|conservative restoration|coronal restoration", t, re.I):
        f.has_crown = True

    # --- bone / pathology -------------------------------------------------
    if re.search(r"severe.{0,25}atroph|atroph.{0,25}severe", t, re.I):
        f.atrophy = "severe"
    elif re.search(r"moderate.{0,25}atroph|atroph.{0,25}moderate", t, re.I):
        f.atrophy = "moderate"
    elif re.search(r"mild.{0,25}atroph|atroph.{0,25}mild", t, re.I):
        f.atrophy = "mild"
    elif re.search(r"(good|preserved|normal).{0,30}(trophism|bone troph)", t, re.I):
        f.atrophy = "none"

    neg_lesion = re.search(
        r"(no|absence of|absent|without)[^.]{0,45}(osteolytic|osteocondensing|osteosclerotic|radiolucen|radiopa|lesion)",
        t, re.I,
    )
    pos_lesion = re.search(r"(osteolytic|osteosclerotic) (lesion|area)|granuloma|radicular cyst", t, re.I)
    if pos_lesion:
        f.osteolytic_lesion = True
    elif neg_lesion:
        f.osteolytic_lesion = False

    return f
