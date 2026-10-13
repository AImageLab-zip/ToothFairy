#!/usr/bin/env python3
"""Deterministic, fact-preserving English realization for photo_report_v3."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

GENERATOR_VERSION = "schema-report-generator-v1"
REPORT_GENERATOR_VERSION = GENERATOR_VERSION
FDI_RE = re.compile(r"^[1-8][1-8]$")


class ReportGenerationError(ValueError):
    """Raised when a structured report cannot be realized safely."""


SAGITTAL_PHRASES = {
    "Class I": "Class I",
    "Class II head-to-head": "end-to-end Class II",
    "Class II full": "full Class II",
    "Class II unspecified": "Class II",
    "Class III": "Class III",
}

CROWDING_NORMALIZATION = {
    "None": "none", "Spacing": "spacing", "Mild": "mild",
    "Mild/Moderate": "mild-to-moderate", "Mild-to-Moderate": "mild-to-moderate",
    "Mild to Moderate": "mild-to-moderate", "Moderate": "moderate",
    "Moderate/Severe": "moderate-to-severe", "Moderate-to-Severe": "moderate-to-severe",
    "Severe": "severe",
}
CURVE_PHRASES = {"Normal": "within normal limits", "Increased": "increased", "Decreased": "decreased", "Reversed": "reversed"}
LOCATION_PHRASES = {"Upper_Arch": "in the upper arch", "Lower_Arch": "in the lower arch", "Right_Side": "on the right side", "Left_Side": "on the left side"}
_REQUIRED_SECTIONS = (
    "1_sagittal_occlusion", "2_transverse_occlusion", "3_vertical_occlusion",
    "4_alignment_and_arch", "5_dental_status", "6_oral_hygiene",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportGenerationError(f"{path} must be an object")
    return value


def _required(parent: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in parent:
        raise ReportGenerationError(f"Missing required field: {path}.{key}")
    return parent[key]


def _enum(value: Any, allowed: set[Any], path: str) -> Any:
    if value not in allowed:
        raise ReportGenerationError(f"Unsupported value at {path}: {value!r}")
    return value


def _teeth(value: Any, path: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportGenerationError(f"{path} must be an array of FDI strings")
    result: set[str] = set()
    for tooth in value:
        if not isinstance(tooth, str) or not FDI_RE.fullmatch(tooth):
            raise ReportGenerationError(f"Invalid FDI tooth at {path}: {tooth!r}")
        result.add(tooth)
    return sorted(result, key=int)


def _english_list(items: Sequence[str]) -> str:
    if not items:
        raise ReportGenerationError("Cannot format an empty list")
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _capitalize(clause: str) -> str:
    return clause[:1].upper() + clause[1:]


def _transverse_sentences(section: Mapping[str, Any]) -> list[str]:
    severity = _enum(_required(section, "severity", "2_transverse_occlusion"), {"Normal", "Mild", "Moderate", "Severe", "Not_Assessable", "Not_Mentioned"}, "2_transverse_occlusion.severity")
    transverse_type = _enum(_required(section, "type", "2_transverse_occlusion"), {"Contraction", "Maxillary_Constriction", "Not_Assessable", "Not_Mentioned"}, "2_transverse_occlusion.type")
    present = _enum(_required(section, "crossbite_present", "2_transverse_occlusion"), {True, False, None}, "2_transverse_occlusion.crossbite_present")
    crossbite_type = _enum(_required(section, "crossbite_type", "2_transverse_occlusion"), {"None", "Crossbite", "Posterior_Crossbite", "Not_Assessable", "Not_Mentioned"}, "2_transverse_occlusion.crossbite_type")
    location = _enum(_required(section, "crossbite_location", "2_transverse_occlusion"), {"None", "Bilateral", "Right_Side", "Left_Side", "Not_Assessable", "Not_Mentioned"}, "2_transverse_occlusion.crossbite_location")
    teeth = _teeth(_required(section, "crossbite_teeth", "2_transverse_occlusion"), "2_transverse_occlusion.crossbite_teeth")
    if present is False and (crossbite_type != "None" or location != "None" or teeth):
        raise ReportGenerationError("crossbite_present=false conflicts with crossbite details")
    if present is True and crossbite_type == "None":
        raise ReportGenerationError("crossbite_present=true conflicts with crossbite_type=None")
    base: str | None = None
    severity_note: str | None = None
    if transverse_type == "Maxillary_Constriction":
        degree = f"{severity.lower()} " if severity in {"Mild", "Moderate", "Severe"} else ""
        base = f"The patient presents a {degree}transverse constriction of the maxilla"
        if severity == "Not_Assessable": severity_note = "Its severity could not be assessed."
        elif severity == "Normal": severity_note = "The transverse severity is recorded as normal."
    elif transverse_type == "Contraction":
        degree = f"{severity.lower()} " if severity in {"Mild", "Moderate", "Severe"} else ""
        base = f"The patient presents a {degree}transverse arch contraction"
        if severity == "Not_Assessable": severity_note = "Its severity could not be assessed."
        elif severity == "Normal": severity_note = "The transverse severity is recorded as normal."
    elif severity == "Normal": base = "The patient presents normal transverse relationships"
    elif severity == "Not_Assessable": base = "The transverse relationship could not be assessed"
    elif severity in {"Mild", "Moderate", "Severe"}: base = f"The patient presents a {severity.lower()} transverse discrepancy"
    elif transverse_type == "Not_Assessable": base = "The transverse relationship type could not be assessed"
    sentences: list[str] = []
    if present is False:
        sentences.append(base + " in the absence of crossbite." if base else "No crossbite is present.")
    elif base:
        sentences.append(base + ".")
    if severity_note: sentences.append(severity_note)
    if present is True:
        kind = "posterior crossbite" if crossbite_type == "Posterior_Crossbite" else "crossbite"
        clause = f"A {kind} is present"
        if location == "Bilateral": clause += " bilaterally"
        elif location == "Right_Side": clause += " on the right side"
        elif location == "Left_Side": clause += " on the left side"
        if teeth: clause += f", involving teeth {_english_list(teeth)}"
        clause += "."
        if crossbite_type == "Not_Assessable": clause += " Its type could not be assessed."
        if location == "Not_Assessable": clause += " Its location could not be assessed."
        sentences.append(clause)
    return sentences


def _sagittal_value(value: Any, path: str) -> str | None:
    _enum(value, set(SAGITTAL_PHRASES) | {"Not_Assessable", "Not_Mentioned"}, path)
    if value == "Not_Mentioned": return None
    if value == "Not_Assessable": return "not assessable"
    return SAGITTAL_PHRASES[value]


def _side_clause(side: str, molar: str | None, canine: str | None) -> str | None:
    parts: list[str] = []
    if molar == "not assessable": parts.append("the molar relationship could not be assessed")
    elif molar: parts.append(f"the molar relationship is {molar}")
    if canine == "not assessable": parts.append("the canine relationship could not be assessed")
    elif canine: parts.append(f"the canine relationship is {canine}")
    return f"on the {side}, " + " and ".join(parts) if parts else None


def _sagittal_sentences(section: Mapping[str, Any]) -> list[str]:
    values = {key: _sagittal_value(_required(section, key, "1_sagittal_occlusion"), f"1_sagittal_occlusion.{key}") for key in ("right_molar", "right_canine", "left_molar", "left_canine")}
    concrete = [values[key] for key in values]
    if all(value is not None and value != "not assessable" for value in concrete):
        if len(set(concrete)) == 1:
            return [f"Sagittally, bilateral {concrete[0]} molar and canine relationships are present."]
        if values["right_molar"] == values["left_molar"] and values["right_canine"] == values["left_canine"]:
            return ["Sagittally, bilateral " f"{values['right_molar']} molar relationships and bilateral " f"{values['right_canine']} canine relationships are present."]
    clauses = [clause for clause in (_side_clause("right", values["right_molar"], values["right_canine"]), _side_clause("left", values["left_molar"], values["left_canine"])) if clause]
    return ["Sagittally, " + "; ".join(clauses) + "."] if clauses else []


def _vertical_sentences(section: Mapping[str, Any]) -> list[str]:
    overjet = _enum(_required(section, "overjet", "3_vertical_occlusion"), {"Normal", "Increased", "Decreased", "Negative", "Not_Assessable", "Not_Mentioned"}, "3_vertical_occlusion.overjet")
    overbite = _enum(_required(section, "overbite", "3_vertical_occlusion"), {"Normal", "Increased", "Decreased", "Open_Bite", "Not_Assessable", "Not_Mentioned"}, "3_vertical_occlusion.overbite")
    if overjet == "Normal" and overbite == "Normal": return ["Overjet and overbite are within normal limits."]
    overjet_map = {"Normal": "overjet is within normal limits", "Increased": "overjet is increased", "Decreased": "overjet is decreased", "Negative": "overjet is negative", "Not_Assessable": "overjet could not be assessed"}
    overbite_map = {"Normal": "overbite is within normal limits", "Increased": "overbite is increased", "Decreased": "overbite is decreased", "Open_Bite": "an open bite is present", "Not_Assessable": "overbite could not be assessed"}
    clauses = []
    if overjet != "Not_Mentioned": clauses.append(overjet_map[overjet])
    if overbite != "Not_Mentioned": clauses.append(overbite_map[overbite])
    return [_capitalize("; ".join(clauses)) + "."] if clauses else []


def _midline_sentence(value: Any) -> str | None:
    _enum(value, {"Aligned", "Deviated", "Not_Assessable", "Not_Mentioned"}, "4_alignment_and_arch.midline")
    return {"Aligned": "The dental midlines are coincident.", "Deviated": "The dental midlines are deviated relative to each other.", "Not_Assessable": "The dental midlines could not be assessed.", "Not_Mentioned": None}[value]


def _crowding_clause(value: Any, arch: str, path: str) -> str | None:
    _enum(value, set(CROWDING_NORMALIZATION) | {"Not_Assessable", "Not_Mentioned"}, path)
    if value == "Not_Mentioned": return None
    if value == "Not_Assessable": return f"crowding in the {arch} arch could not be assessed"
    normalized = CROWDING_NORMALIZATION[value]
    if normalized == "none": return f"no crowding is present in the {arch} arch"
    if normalized == "spacing": return f"spacing is present in the {arch} arch"
    return f"{normalized} crowding is present in the {arch} arch"


def _curve_clause(name: str, value: Any, location: Any | None, path: str) -> str | None:
    _enum(value, set(CURVE_PHRASES) | {"Not_Assessable", "Not_Mentioned"}, path)
    if value == "Not_Mentioned": return None
    subject = f"the curve of {name}"
    location_suffix = ""
    if location is not None:
        _enum(location, set(LOCATION_PHRASES) | {"Not_Assessable", "Not_Mentioned"}, "4_alignment_and_arch.curve_wilson_location")
        if location in LOCATION_PHRASES: location_suffix = " " + LOCATION_PHRASES[location]
    if value == "Not_Assessable": return f"{subject}{location_suffix} could not be assessed"
    clause = f"{subject}{location_suffix} is {CURVE_PHRASES[value]}"
    if location == "Not_Assessable": clause += ", although its location could not be assessed"
    return clause


def _alignment_sentences(section: Mapping[str, Any]) -> list[str]:
    sentences: list[str] = []
    midline = _midline_sentence(_required(section, "midline", "4_alignment_and_arch"))
    if midline: sentences.append(midline)
    upper = _crowding_clause(_required(section, "upper_crowding", "4_alignment_and_arch"), "upper", "4_alignment_and_arch.upper_crowding")
    lower = _crowding_clause(_required(section, "lower_crowding", "4_alignment_and_arch"), "lower", "4_alignment_and_arch.lower_crowding")
    crowding = [clause for clause in (upper, lower) if clause]
    if crowding: sentences.append(_capitalize(", while ".join(crowding)) + ".")
    spee = _curve_clause("Spee", _required(section, "curve_spee", "4_alignment_and_arch"), None, "4_alignment_and_arch.curve_spee")
    location = _required(section, "curve_wilson_location", "4_alignment_and_arch")
    wilson = _curve_clause("Wilson", _required(section, "curve_wilson", "4_alignment_and_arch"), location, "4_alignment_and_arch.curve_wilson")
    if spee and wilson: sentences.append(_capitalize(spee + ", while " + wilson) + ".")
    elif spee or wilson: sentences.append(_capitalize(spee or wilson or "") + ".")
    return sentences


def _dental_sentences(section: Mapping[str, Any]) -> list[str]:
    missing = _teeth(_required(section, "missing_teeth", "5_dental_status"), "5_dental_status.missing_teeth")
    restorations = _teeth(_required(section, "restorations", "5_dental_status"), "5_dental_status.restorations")
    caries = _teeth(_required(section, "caries", "5_dental_status"), "5_dental_status.caries")
    sentences: list[str] = []
    if missing:
        sentences.append(f"{'Tooth' if len(missing)==1 else 'Teeth'} {_english_list(missing)} {'is' if len(missing)==1 else 'are'} missing.")
    if restorations: sentences.append(f"Restorations are present on teeth {_english_list(restorations)}.")
    if caries: sentences.append(f"Caries lesions are present on teeth {_english_list(caries)}.")
    return sentences


def _hygiene_sentences(section: Mapping[str, Any]) -> list[str]:
    hygiene = _enum(_required(section, "hygiene_status", "6_oral_hygiene"), {"Good", "Fair", "Poor", "Not_Assessable", "Not_Mentioned"}, "6_oral_hygiene.hygiene_status")
    pigmentation = _enum(_required(section, "pigmentation", "6_oral_hygiene"), {True, False, None}, "6_oral_hygiene.pigmentation")
    sentences: list[str] = []
    if hygiene in {"Good", "Fair", "Poor"}: sentences.append(f"Oral hygiene is {hygiene.lower()}.")
    elif hygiene == "Not_Assessable": sentences.append("Oral hygiene could not be assessed.")
    if pigmentation is True: sentences.append("Extrinsic pigmentation is present.")
    elif pigmentation is False: sentences.append("No extrinsic pigmentation is observed.")
    return sentences


def generate_report(report: Mapping[str, Any]) -> str:
    root = _mapping(report, "report")
    sections = {name: _mapping(_required(root, name, "report"), f"report.{name}") for name in _REQUIRED_SECTIONS}
    sentences: list[str] = []
    sentences.extend(_transverse_sentences(sections["2_transverse_occlusion"]))
    sentences.extend(_sagittal_sentences(sections["1_sagittal_occlusion"]))
    sentences.extend(_vertical_sentences(sections["3_vertical_occlusion"]))
    sentences.extend(_alignment_sentences(sections["4_alignment_and_arch"]))
    sentences.extend(_dental_sentences(sections["5_dental_status"]))
    sentences.extend(_hygiene_sentences(sections["6_oral_hygiene"]))
    if not sentences: raise ReportGenerationError("The structured report contains no realizable findings")
    return " ".join(sentences)
