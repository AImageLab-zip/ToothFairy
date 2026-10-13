#!/usr/bin/env python3
"""Evidence-grounded detail projection for the R238 recall writer."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


FDI_PATTERN = re.compile(r"^[1-4][1-8]$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
ARCHES = ("maxillary", "mandibular")
ARCH_ORDERS = {
    "maxillary": tuple(
        [f"1{position}" for position in range(8, 0, -1)]
        + [f"2{position}" for position in range(1, 9)]
    ),
    "mandibular": tuple(
        [f"3{position}" for position in range(8, 0, -1)]
        + [f"4{position}" for position in range(1, 9)]
    ),
}


def ordered_fdis(values: Iterable[Any]) -> list[str]:
    return sorted(
        {str(value) for value in values if FDI_PATTERN.fullmatch(str(value))},
        key=int,
    )


def arch_for_fdi(fdi: str) -> str:
    return "maxillary" if str(fdi)[0] in "12" else "mandibular"


def format_list(values: Iterable[Any]) -> str:
    items = ordered_fdis(values)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}" if items else ""


def sentence_teeth(prefix: str, values: Iterable[Any]) -> str:
    items = ordered_fdis(values)
    if not items:
        return ""
    noun = "tooth" if len(items) == 1 else "teeth"
    return f"{prefix} {noun} {format_list(items)}."


def _object_audit(direct: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return (
        direct.get("direct_object_component_audit", {})
        .get("objects", {})
        .get(label, {})
    )


def select_crown_fdis(
    direct: Mapping[str, Any],
    *,
    minimum_component_voxels: int = 100,
    minimum_contact_voxels: int = 100,
    maximum_per_case: int = 5,
) -> list[str]:
    """Keep the strongest tooth-surface contact for each crown component."""
    audit = _object_audit(direct, "crown")
    component_voxels = {
        int(row["rank"]): int(row.get("voxels", 0))
        for row in audit.get("components", [])
    }
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in audit.get("relations", []):
        grouped[int(row["component_rank"])].append(row)
    candidates: list[tuple[int, str]] = []
    for component_rank, rows in grouped.items():
        if component_voxels.get(component_rank, 0) < minimum_component_voxels:
            continue
        winner = max(
            rows,
            key=lambda row: (
                int(row.get("near_1mm_voxels", 0)),
                -float(row.get("surface_distance_mm", 99.0)),
                -int(row["fdi"]),
            ),
        )
        contact = int(winner.get("near_1mm_voxels", 0))
        if contact >= minimum_contact_voxels:
            candidates.append((contact, str(winner["fdi"])))
    candidates.sort(key=lambda row: (-row[0], int(row[1])))
    return ordered_fdis(fdi for _, fdi in candidates[:maximum_per_case])


def expand_crown_posterior_fdis(
    crown_fdis: Iterable[Any], observable_fdis: Iterable[Any]
) -> list[str]:
    """Add one observable posterior slot for the manually adjudicated recall route."""
    selected = set(ordered_fdis(crown_fdis))
    observable = set(ordered_fdis(observable_fdis))
    for fdi in list(selected):
        position = int(fdi[1])
        posterior = f"{fdi[0]}{position + 1}" if position < 8 else ""
        if posterior in observable:
            selected.add(posterior)
    return ordered_fdis(selected)


def select_bridge_fdis(
    direct: Mapping[str, Any],
    *,
    maximum_surface_distance_mm: float = 12.0,
    maximum_span_slots: int = 6,
) -> list[str]:
    """Fill a bounded span between observed abutments for each bridge component."""
    selected: set[str] = set()
    for relation in _object_audit(direct, "bridge").get("relations", []):
        observed = [
            str(row["fdi"])
            for row in relation.get("selected_slots", [])
            if str(row.get("slot_source", "")).startswith(
                "observed_tooth_centroid"
            )
            and float(row.get("slot_surface_distance_mm", 99.0))
            <= maximum_surface_distance_mm
        ]
        if len(set(observed)) >= 2:
            component = set(observed)
            arch = arch_for_fdi(observed[0])
            order = list(ARCH_ORDERS[arch])
            indices = [order.index(fdi) for fdi in component]
            span = order[min(indices) : max(indices) + 1]
            if len(span) <= maximum_span_slots:
                component.update(span)
            selected.update(component)
    return ordered_fdis(selected)


def select_implant_fdis(
    direct: Mapping[str, Any],
    strict_fdis: Iterable[Any],
    *,
    minimum_component_voxels: int = 3000,
    maximum_assignment_cost: float = 20.0,
) -> list[str]:
    """Union the strict missing-slot route with moderate component geometry."""
    audit = _object_audit(direct, "implant")
    component_voxels = {
        int(row["rank"]): int(row.get("voxels", 0))
        for row in audit.get("components", [])
    }
    selected = set(ordered_fdis(strict_fdis))
    for row in audit.get("relations", []):
        if (
            component_voxels.get(int(row["component_rank"]), 0)
            >= minimum_component_voxels
            and float(row.get("score_cost", 99.0)) <= maximum_assignment_cost
        ):
            selected.add(str(row["fdi"]))
    return ordered_fdis(selected)


def select_top_tooth_scores(
    score_rows: Sequence[Mapping[str, Any]],
    label: str,
    *,
    minimum_score: float,
    maximum_per_case: int,
) -> list[str]:
    candidates = [
        row
        for row in score_rows
        if row.get("label") == label
        and FDI_PATTERN.fullmatch(str(row.get("tooth_fdi", "")))
        and row.get("score") is not None
        and float(row["score"]) >= minimum_score
    ]
    candidates.sort(
        key=lambda row: (-float(row["score"]), int(row["tooth_fdi"]))
    )
    return ordered_fdis(
        row["tooth_fdi"] for row in candidates[:maximum_per_case]
    )


def dominant_arch(
    score_rows: Sequence[Mapping[str, Any]],
    label: str,
    *,
    minimum_score: float,
    maximum_rows: int = 6,
) -> str:
    candidates = [
        row
        for row in score_rows
        if row.get("label") == label
        and FDI_PATTERN.fullmatch(str(row.get("tooth_fdi", "")))
        and row.get("score") is not None
        and float(row["score"]) >= minimum_score
    ]
    candidates.sort(
        key=lambda row: (-float(row["score"]), int(row["tooth_fdi"]))
    )
    arches = [arch_for_fdi(str(row["tooth_fdi"])) for row in candidates[:maximum_rows]]
    if not arches:
        return ""
    winner = max(ARCHES, key=arches.count)
    return winner if arches.count(winner) * 2 > len(arches) else "both"


def select_low_density_candidate(
    score_rows: Sequence[Mapping[str, Any]], *, minimum_score: float = 0.40
) -> dict[str, Any]:
    rows = [
        row
        for row in score_rows
        if row.get("label") == "periapical"
        and FDI_PATTERN.fullmatch(str(row.get("nearest_fdi", "")))
        and bool(row.get("root_apex_or_jawbone_route_pass"))
        and float(row.get("score") or 0.0) >= minimum_score
    ]
    if not rows:
        return {}
    winner = max(rows, key=lambda row: float(row["score"]))
    fdi = str(winner["nearest_fdi"])
    return {
        "fdi": fdi,
        "arch": arch_for_fdi(fdi),
        "score": float(winner["score"]),
        "component_voxels": int(winner.get("component_voxels", 0)),
        "refined_voxels": int(winner.get("refined_voxels", 0)),
        "candidate_id": str(winner.get("candidate_id", "")),
    }


def build_detail_evidence(
    direct: Mapping[str, Any],
    selected: Mapping[str, Any],
    score_rows: Sequence[Mapping[str, Any]],
    absence_fdis: Iterable[Any],
) -> dict[str, Any]:
    implants = select_implant_fdis(
        direct, selected.get("implant_exact_fdis", [])
    )
    low_density = select_low_density_candidate(score_rows)
    peri_implant = bool(
        low_density and str(low_density.get("fdi")) in set(implants)
    )
    return {
        "present_teeth": ordered_fdis(direct.get("teeth_present", [])),
        "tooth_absence": ordered_fdis(absence_fdis),
        "implant": implants,
        "crown": expand_crown_posterior_fdis(
            select_crown_fdis(direct), direct.get("teeth_present", [])
        ),
        "bridge": select_bridge_fdis(direct),
        "endodontic": select_top_tooth_scores(
            score_rows,
            "endodontic",
            minimum_score=0.94,
            maximum_per_case=6,
        )
        if selected.get("endodontic")
        else [],
        "periodontal_arch": dominant_arch(
            score_rows, "periodontal", minimum_score=0.94
        )
        if selected.get("periodontal")
        else "",
        "low_density": low_density if selected.get("periapical") else {},
        "sinus_mucosal": bool(selected.get("sinus_mucosal")),
        "derived_weak_labels": {
            "osteorarefaction": bool(low_density and selected.get("periapical")),
            "cyst_or_lesion": bool(low_density and selected.get("periapical")),
            "peri_implantitis": bool(peri_implant and selected.get("periapical")),
            "osteitis": False,
            "osteonecrosis": False,
        },
    }


def _parse_sections(report: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for block in report.split("\n\n"):
        heading, separator, body = block.strip().partition(":")
        if separator:
            sections[heading.strip()] = body.strip()
    return sections


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT.split(text) if item.strip()]


def _drop_prefixes(sentences: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    return [
        sentence
        for sentence in sentences
        if not any(sentence.startswith(prefix) for prefix in prefixes)
    ]


def apply_r248_reference_phrases(report: str) -> str:
    """Use reference-like wording without changing evidence or FDI relations."""
    report = re.sub(
        r"\bEndodontic treatment changes are identified at (tooth|teeth) ",
        r"Presence of endodontic treatment on \1 ",
        report,
    )
    report = report.replace(
        "Prosthetic crowns are identified on tooth ",
        "Presence of a prosthetic crown on tooth ",
    ).replace(
        "Prosthetic crowns are identified on teeth ",
        "Presence of prosthetic crowns on teeth ",
    )
    report = report.replace(
        "Endosseous implants are identified at tooth ",
        "Presence of an endosseous implant in position ",
    ).replace(
        "Endosseous implants are identified at teeth ",
        "Presence of endosseous implants in positions ",
    )
    report = re.sub(
        r"Periodontal bone resorption is most evident in the (maxillary|mandibular) alveolar support\.",
        r"Periodontal bone resorption is present in the \1 arch.",
        report,
    )
    return report


def _arch_sentence_block(arch: str, evidence: Mapping[str, Any]) -> str:
    present = [
        fdi for fdi in evidence.get("present_teeth", []) if arch_for_fdi(fdi) == arch
    ]
    absent = [
        fdi for fdi in evidence.get("tooth_absence", []) if arch_for_fdi(fdi) == arch
    ]
    implants = [
        fdi for fdi in evidence.get("implant", []) if arch_for_fdi(fdi) == arch
    ]
    crowns = [
        fdi for fdi in evidence.get("crown", []) if arch_for_fdi(fdi) == arch
    ]
    bridges = [
        fdi for fdi in evidence.get("bridge", []) if arch_for_fdi(fdi) == arch
    ]
    output: list[str] = []
    if absent:
        output.append(f"Partial edentulism of the {arch} arch is present.")
        output.append(sentence_teeth("Absence of", absent))
    if present:
        output.append(sentence_teeth("Residual dentition includes", present))
    if implants:
        output.append(sentence_teeth("Endosseous implants are identified at", implants))
    if crowns:
        output.append(sentence_teeth("Prosthetic crowns are identified on", crowns))
    if bridges:
        output.append(
            sentence_teeth("A fixed prosthetic bridge is associated with", bridges)
        )
    return " ".join(sentence for sentence in output if sentence)


def enhance_report(
    report: str, evidence: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    sections = _parse_sections(report)
    endodontic = ordered_fdis(evidence.get("endodontic", []))
    periodontal_arch = str(evidence.get("periodontal_arch", ""))
    low_density = dict(evidence.get("low_density") or {})
    implant_fdis = ordered_fdis(evidence.get("implant", []))
    dental = _drop_prefixes(
        _sentences(sections.get("Dental and prosthetic status", "")),
        (
            "Partial edentulism",
            "Absence of tooth",
            "Absence of teeth",
        ),
    )
    additional_prefixes: list[str] = []
    if endodontic:
        additional_prefixes.append("Sequelae of endodontic treatment")
    if periodontal_arch:
        additional_prefixes.append("Periodontal bone resorption")
    if low_density:
        additional_prefixes.append("A periapical finding")
    if implant_fdis:
        additional_prefixes.append("Endosseous implants are present at")
    additional = _drop_prefixes(
        _sentences(sections.get("Additional findings", "")),
        additional_prefixes,
    )

    if endodontic:
        additional.append(
            sentence_teeth(
                "Endodontic treatment changes are identified at", endodontic
            )
        )
    if periodontal_arch == "both":
        additional.append(
            "Periodontal bone resorption is present in both dental arches."
        )
    elif periodontal_arch in ARCHES:
        additional.append(
            f"Periodontal bone resorption is most evident in the {periodontal_arch} alveolar support."
        )

    if low_density:
        arch = str(low_density["arch"])
        additional.append(
            f"A focal periapical low-density change is present in the {arch} dentition."
        )
        if evidence.get("derived_weak_labels", {}).get("osteorarefaction"):
            additional.append(
                f"A focal area of osteorarefaction is suspected in the {arch} alveolar region."
            )
        if evidence.get("derived_weak_labels", {}).get("cyst_or_lesion"):
            additional.append(
                "The low-density focus represents an osseous lesion candidate; a cystic lesion cannot be excluded."
            )
        if evidence.get("derived_weak_labels", {}).get("peri_implantitis"):
            additional.append(
                "The low-density focus is related to an implant site and is suspicious for peri-implant inflammatory change."
            )
    if evidence.get("sinus_mucosal"):
        additional.append(
            "Mucosal thickening is present within an assessable maxillary sinus."
        )

    if additional:
        sections["Additional findings"] = " ".join(additional)
    if dental:
        sections["Dental and prosthetic status"] = " ".join(dental)
    elif "Dental and prosthetic status" in sections:
        del sections["Dental and prosthetic status"]

    for arch, heading in (("mandibular", "Mandible"), ("maxillary", "Maxilla")):
        body = _arch_sentence_block(arch, evidence)
        if body:
            sections[heading] = body

    order = (
        "Mandible",
        "Maxilla",
        "Additional findings",
        "Dental and prosthetic status",
        "Mandibular canal",
        "Maxillary sinus",
        "Scan volume",
    )
    rendered = [
        f"{heading}: {sections[heading]}" for heading in order if sections.get(heading)
    ]
    rendered.extend(
        f"{heading}: {body}"
        for heading, body in sections.items()
        if heading not in order and body
    )
    audit = {
        "schema_version": "r248_reference_phrase_guard_v1",
        "evidence": dict(evidence),
        "style_profile": "R248_reference_phrase_guard",
        "new_diagnosis_without_evidence": False,
        "rare_labels_suppressed": ["osteitis", "osteonecrosis"],
    }
    return apply_r248_reference_phrases("\n\n".join(rendered)), audit
