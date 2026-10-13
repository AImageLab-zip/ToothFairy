"""Closed-world verifier for the frozen report-generation runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FDI_PATTERN = re.compile(r"(?<!\d)([1-8][1-8])(?!\d)")
SPACE_PATTERN = re.compile(r"\s+")

RELATIONSHIP_PATTERNS = {
    "Class I": re.compile(r"\bclass\s+i\b", re.I),
    "Class II head-to-head": re.compile(
        r"(?:\b(?:end|edge|head)[ -]to[ -](?:end|edge|head)\b.{0,30}\bclass\s+ii\b|"
        r"\bclass\s+ii\b.{0,30}\b(?:end|edge|head)[ -]to[ -](?:end|edge|head)\b)", re.I,
    ),
    "Class II full": re.compile(r"(?:\bfull(?:[ -](?:unit|cusp))?\s+class\s+ii\b|\bclass\s+ii\s+full\b)", re.I),
    "Class II unspecified": re.compile(r"\bclass\s+ii\b", re.I),
    "Class III": re.compile(r"\bclass\s+iii\b", re.I),
    "Not_Assessable": re.compile(r"\b(?:not assessable|could not be assessed)\b", re.I),
}

OUT_OF_SCHEMA_PATTERNS = {
    "plaque": re.compile(r"\bplaque\b", re.I),
    "calculus": re.compile(r"\b(?:calculus|tartar)\b", re.I),
    "gingival finding": re.compile(r"\bgingiv\w*\b|\bperiodont\w*\b", re.I),
    "inflammation": re.compile(r"\binflamm\w*\b", re.I),
    "demineralization": re.compile(r"\bdeminerali[sz]\w*\b", re.I),
    "sealant": re.compile(r"\bseal\w*\s+fissure\w*\b|\bsealant\w*\b", re.I),
    "eruption status": re.compile(r"\berupt\w*\b", re.I),
    "dentition stage": re.compile(r"\b(?:mixed|deciduous|permanent)\s+dentition\b", re.I),
    "unsupported bite location": re.compile(r"\b(?:anterior|posterior|lateral)\s+open\s+bite\b", re.I),
    "scissor bite": re.compile(r"\bscissor\s+bite\b", re.I),
}


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reasons: tuple[str, ...]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _teeth(value: Any) -> set[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("Tooth collection must be an array")
    return {str(item) for item in value}


def _near(text: str, anchor_pattern: str, value_pattern: str, window: int = 120) -> bool:
    anchor = re.compile(anchor_pattern, re.I)
    value = re.compile(value_pattern, re.I)
    for match in anchor.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        if value.search(text[start:end]):
            return True
    return False


def _side_segment(text: str, side: str) -> str:
    side_match = re.search(rf"\b(?:on\s+the\s+)?{side}\b", text, re.I)
    if not side_match:
        return ""
    other = "left" if side == "right" else "right"
    end_match = re.search(rf"(?:;|\b(?:on\s+the\s+)?{other}\b|\.(?:\s|$))", text[side_match.end():], re.I)
    end = side_match.end() + (end_match.start() if end_match else 260)
    return text[side_match.start():min(len(text), end)]


def _relationship_present(text: str, value: str) -> bool:
    pattern = RELATIONSHIP_PATTERNS[value]
    if value != "Class II unspecified":
        return bool(pattern.search(text))
    for match in pattern.finditer(text):
        local = text[max(0, match.start() - 30):match.end() + 30]
        if not RELATIONSHIP_PATTERNS["Class II head-to-head"].search(local) and not RELATIONSHIP_PATTERNS["Class II full"].search(local):
            return True
    return False


def _verify_sagittal(section: Mapping[str, Any], text: str, reasons: list[str]) -> None:
    fields = {
        ("right", "molar"): section["right_molar"], ("right", "canine"): section["right_canine"],
        ("left", "molar"): section["left_molar"], ("left", "canine"): section["left_canine"],
    }
    if not {key: value for key, value in fields.items() if value != "Not_Mentioned"}:
        return
    relationship_sentences = [sentence for sentence in re.split(r"(?<=\.)\s+", text) if re.search(r"\b(?:molar|canine)\b", sentence, re.I)]
    sagittal_text = " ".join(relationship_sentences) or text
    for tooth_kind in ("molar", "canine"):
        right, left = fields[("right", tooth_kind)], fields[("left", tooth_kind)]
        if right == left and right != "Not_Mentioned":
            bilateral_context = sagittal_text if re.search(r"\bbilateral\b", sagittal_text, re.I) else ""
            if tooth_kind in bilateral_context.lower() and _relationship_present(bilateral_context, right):
                continue
        for side, value in (("right", right), ("left", left)):
            if value == "Not_Mentioned":
                continue
            segment = _side_segment(sagittal_text, side)
            if not segment or tooth_kind not in segment.lower() or not _relationship_present(segment, value):
                reasons.append(f"{side} {tooth_kind} relationship not preserved")


def _verify_transverse(section: Mapping[str, Any], text: str, reasons: list[str]) -> None:
    severity, transverse_type = section["severity"], section["type"]
    present, crossbite_type, location = section["crossbite_present"], section["crossbite_type"], section["crossbite_location"]
    if severity == "Normal" and not _near(text, r"\btransverse\w*\b", r"\b(?:normal|correct)\b"):
        reasons.append("normal transverse relationship not preserved")
    elif severity in {"Mild", "Moderate", "Severe"} and not _near(text, r"\btransverse\w*\b", rf"\b{severity}\b"):
        reasons.append(f"{severity.lower()} transverse severity not preserved")
    elif severity == "Not_Assessable" and not _near(text, r"\b(?:transverse\w*|severity)\b", r"\b(?:not assessable|could not be assessed|undetermined|indeterminate|unknown)\b"):
        reasons.append("transverse Not_Assessable not preserved")
    if transverse_type == "Maxillary_Constriction" and not re.search(r"\b(?:maxillary constriction|constriction of the maxilla)\b", text, re.I):
        reasons.append("maxillary constriction not preserved")
    elif transverse_type == "Contraction" and not re.search(r"\b(?:arch\s+)?contraction\b", text, re.I):
        reasons.append("transverse contraction not preserved")
    if present is False:
        if not re.search(r"\b(?:no|without)\s+(?:a\s+)?crossbite\b|\babsence\s+of\s+(?:a\s+)?crossbite\b", text, re.I):
            reasons.append("explicit absence of crossbite not preserved")
    elif present is True:
        if not re.search(r"\bcrossbite\b", text, re.I):
            reasons.append("crossbite presence not preserved")
        if re.search(r"\b(?:no|without)\s+(?:a\s+)?crossbite\b|\babsence\s+of\s+(?:a\s+)?crossbite\b", text, re.I):
            reasons.append("present crossbite was negated")
        if crossbite_type == "Posterior_Crossbite" and not re.search(r"\bposterior\s+crossbite\b", text, re.I):
            reasons.append("posterior crossbite type not preserved")
        location_patterns = {"Bilateral": r"\bbilateral(?:ly)?\b", "Right_Side": r"\bright(?:\s+side)?\b", "Left_Side": r"\bleft(?:\s+side)?\b", "Not_Assessable": r"\b(?:location\s+)?(?:not assessable|could not be assessed)\b"}
        if location in location_patterns and not _near(text, r"\bcrossbite\b", location_patterns[location], 160):
            reasons.append("crossbite location not preserved")
    elif present is None and re.search(r"\bcrossbite\b", text, re.I):
        reasons.append("crossbite was added despite null presence")


def _verify_vertical(section: Mapping[str, Any], text: str, reasons: list[str]) -> None:
    patterns = {"Normal": r"\b(?:normal|within normal limits)\b", "Increased": r"\bincreased\b", "Decreased": r"\b(?:decreased|reduced)\b", "Negative": r"\bnegative\b", "Not_Assessable": r"\b(?:not assessable|could not be assessed)\b"}
    overjet = section["overjet"]
    if overjet != "Not_Mentioned" and not _near(text, r"\boverjet\b", patterns[overjet]):
        reasons.append("overjet value not preserved")
    overbite = section["overbite"]
    if overbite == "Open_Bite":
        if not re.search(r"\bopen\s+bite\b", text, re.I):
            reasons.append("open bite not preserved")
    elif overbite != "Not_Mentioned" and not _near(text, r"\boverbite\b", patterns[overbite]):
        reasons.append("overbite value not preserved")


def _verify_alignment(section: Mapping[str, Any], text: str, reasons: list[str]) -> None:
    midline = section["midline"]
    midline_patterns = {"Aligned": r"\b(?:aligned|coincident|centered)\b", "Deviated": r"\b(?:deviated|noncoincident|not coincident)\b", "Not_Assessable": r"\b(?:not assessable|could not be assessed)\b"}
    if midline != "Not_Mentioned" and not _near(text, r"\bmidlines?\b", midline_patterns[midline]):
        reasons.append("midline value not preserved")
    crowding_patterns = {"None": r"\b(?:no|without)\s+crowding\b", "Spacing": r"\b(?:spacing|spaces|diastemas?)\b", "Mild": r"\bmild\b", "Mild/Moderate": r"\bmild[ -]to[ -]moderate\b", "Mild-to-Moderate": r"\bmild[ -]to[ -]moderate\b", "Mild to Moderate": r"\bmild[ -]to[ -]moderate\b", "Moderate": r"\bmoderate\b", "Moderate/Severe": r"\bmoderate[ -]to[ -]severe\b", "Moderate-to-Severe": r"\bmoderate[ -]to[ -]severe\b", "Severe": r"\bsevere\b", "Not_Assessable": r"\b(?:not assessable|could not be assessed)\b"}
    upper_crowding, lower_crowding = section["upper_crowding"], section["lower_crowding"]
    combined_crowding = False
    if upper_crowding == lower_crowding and upper_crowding != "Not_Mentioned":
        anchor = re.search(r"\b(?:both\s+the\s+upper\s+and\s+lower\s+arch(?:es)?|upper\s+(?:and|or)\s+lower\s+arch(?:es)?)\b", text, re.I)
        if anchor:
            combined_crowding = bool(re.search(crowding_patterns[upper_crowding], text[max(0, anchor.start()-100):anchor.end()+100], re.I))
    for arch, value in (("upper", upper_crowding), ("lower", lower_crowding)):
        if value != "Not_Mentioned" and not combined_crowding and not _near(text, rf"\b{arch}\s+arch(?:es)?\b", crowding_patterns[value], 100):
            reasons.append(f"{arch} crowding value not preserved")
    curve_patterns = {"Normal": r"\b(?:normal|within normal limits)\b", "Increased": r"\bincreased\b", "Decreased": r"\b(?:decreased|flattened|reduced)\b", "Reversed": r"\breversed\b", "Not_Assessable": r"\b(?:not assessable|could not be assessed)\b"}
    spee_value, wilson_value = section["curve_spee"], section["curve_wilson"]
    combined_curves = False
    if spee_value == wilson_value and spee_value != "Not_Mentioned":
        anchor = re.search(r"\bcurves?\s+of\s+Spee\s+and\s+(?:(?:the\s+)?curve\s+of\s+)?Wilson\b", text, re.I)
        if anchor:
            combined_curves = bool(re.search(curve_patterns[spee_value], text[max(0, anchor.start()-80):anchor.end()+100], re.I))
    for name, value in (("Spee", spee_value), ("Wilson", wilson_value)):
        if value != "Not_Mentioned" and not combined_curves and not _near(text, rf"\bcurve\s+of\s+{name}\b", curve_patterns[value], 100):
            reasons.append(f"Curve of {name} value not preserved")
    location = section["curve_wilson_location"]
    location_patterns = {"Upper_Arch": r"\bupper\s+arch\b", "Lower_Arch": r"\blower\s+arch\b", "Right_Side": r"\bright(?:\s+side)?\b", "Left_Side": r"\bleft(?:\s+side)?\b", "Not_Assessable": r"\b(?:location\s+)?(?:not assessable|could not be assessed)\b"}
    if location != "Not_Mentioned" and not _near(text, r"\bcurve\s+of\s+Wilson\b", location_patterns[location], 140):
        reasons.append("Curve of Wilson location not preserved")


def _verify_dental(section: Mapping[str, Any], text: str, reasons: list[str]) -> set[str]:
    expected: set[str] = set()
    category_patterns = {"missing_teeth": re.compile(r"\b(?:missing|absent)\b", re.I), "restorations": re.compile(r"\b(?:restoration|restored|filled|crown)\w*\b", re.I), "caries": re.compile(r"\b(?:caries|carious)\b", re.I)}
    for field, pattern in category_patterns.items():
        teeth = _teeth(section[field])
        expected.update(teeth)
        if teeth and not pattern.search(text):
            reasons.append(f"{field} category not preserved")
        if not teeth and pattern.search(text):
            reasons.append(f"unsupported {field} claim added")
    return expected


def _verify_hygiene(section: Mapping[str, Any], text: str, reasons: list[str]) -> None:
    hygiene = section["hygiene_status"]
    if hygiene == "Not_Mentioned":
        if re.search(r"\boral\s+hygiene\b", text, re.I):
            reasons.append("Not_Mentioned hygiene was verbalized")
    else:
        expected = {"Good": r"\bgood\b", "Fair": r"\bfair\b", "Poor": r"\bpoor\b", "Not_Assessable": r"\b(?:not assessable|could not be assessed)\b"}[hygiene]
        if not _near(text, r"\boral\s+hygiene\b", expected, 80):
            reasons.append("hygiene status not preserved")
    pigmentation = section["pigmentation"]
    pigment_pattern = re.compile(r"\b(?:extrinsic\s+)?(?:pigmentation|staining)\b", re.I)
    if pigmentation is None and pigment_pattern.search(text):
        reasons.append("null pigmentation was verbalized")
    elif pigmentation is True:
        if not pigment_pattern.search(text) or re.search(r"\b(?:no|without)\s+(?:extrinsic\s+)?(?:pigmentation|staining)\b", text, re.I):
            reasons.append("present pigmentation not preserved")
    elif pigmentation is False and not re.search(r"\b(?:no|without)\s+(?:extrinsic\s+)?(?:pigmentation|staining)\b", text, re.I):
        reasons.append("absent pigmentation not preserved")


def verify_polished_report(report: Mapping[str, Any], candidate: str) -> VerificationResult:
    reasons: list[str] = []
    text = SPACE_PATTERN.sub(" ", candidate).strip()
    if not text:
        return VerificationResult(False, ("empty candidate",))
    if "\n" in candidate.strip() or any(marker in text for marker in ("```", "<think>", "</think>", "{", "}")):
        reasons.append("candidate is not one plain-text paragraph")
    word_count = len(re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+)*\b", text))
    if word_count < 15 or word_count > 240:
        reasons.append(f"candidate length outside 15-240 words: {word_count}")
    for label, pattern in OUT_OF_SCHEMA_PATTERNS.items():
        if pattern.search(text):
            reasons.append(f"out-of-schema detail added: {label}")
    sagittal = _mapping(report["1_sagittal_occlusion"], "1_sagittal_occlusion")
    transverse = _mapping(report["2_transverse_occlusion"], "2_transverse_occlusion")
    vertical = _mapping(report["3_vertical_occlusion"], "3_vertical_occlusion")
    alignment = _mapping(report["4_alignment_and_arch"], "4_alignment_and_arch")
    dental = _mapping(report["5_dental_status"], "5_dental_status")
    hygiene = _mapping(report["6_oral_hygiene"], "6_oral_hygiene")
    _verify_transverse(transverse, text, reasons)
    _verify_sagittal(sagittal, text, reasons)
    _verify_vertical(vertical, text, reasons)
    _verify_alignment(alignment, text, reasons)
    expected_teeth = _verify_dental(dental, text, reasons)
    expected_teeth.update(_teeth(transverse["crossbite_teeth"]))
    observed_teeth = set(FDI_PATTERN.findall(text))
    if observed_teeth != expected_teeth:
        reasons.append(f"FDI tooth set changed: expected={sorted(expected_teeth, key=int)}, observed={sorted(observed_teeth, key=int)}")
    _verify_hygiene(hygiene, text, reasons)
    return VerificationResult(not reasons, tuple(dict.fromkeys(reasons)))
