"""Corpus-aligned deterministic rendering for structured Bite2Text labels."""

from __future__ import annotations

from typing import Mapping


RELATION_TEXT = {
    "class_i": "Class I",
    "class_ii_edge_to_edge": "edge-to-edge Class II",
    "class_ii_full": "full Class II",
    "class_ii_unspecified": "Class II",
    "class_iii": "Class III",
    "not_assessable": "not assessable",
}

OVERJET_TEXT = {
    "normal": "within normal limits",
    "increased": "increased",
    "reduced": "slightly reduced",
    "negative": "negative",
    "edge_to_edge": "edge-to-edge",
}

VERTICAL_SENTENCES = {
    "normal": "From a vertical standpoint, the overbite is within normal limits.",
    "increased": "From a vertical standpoint, the overbite is increased.",
    "reduced": "From a vertical standpoint, the overbite is reduced.",
    "deep_bite": "From a vertical standpoint, there is a deep bite.",
    "open_bite": "From a vertical standpoint, there is an open bite.",
}

MIDLINE_SENTENCES = {
    "coincident": "The dental midlines are coincident.",
    "slightly_deviated": "The dental midlines are slightly deviated.",
    "deviated": "The dental midlines are deviated relative to each other.",
}

CROSSBITE_SENTENCES = {
    "none": "Transversely, there is no crossbite.",
    "anterior": "Transversely, an anterior crossbite is present.",
    "posterior": "Transversely, a posterior crossbite is present.",
    "present_unspecified": "Transversely, a crossbite is present.",
}

CROWDING_TEXT = {
    "none": "no",
    "mild": "mild",
    "mild-to-moderate": "mild-to-moderate",
    "moderate": "moderate",
    "moderate-to-severe": "moderate-to-severe",
    "severe": "severe",
}


def side_relation(molar: str, canine: str, side: str) -> str:
    molar_text = RELATION_TEXT[molar]
    canine_text = RELATION_TEXT[canine]
    if molar == "not_assessable" and canine == "not_assessable":
        return f"the molar and canine relationships are not assessable on the {side}"
    if molar == "not_assessable":
        return (
            f"the molar relationship is not assessable and there is a "
            f"{canine_text} canine relationship on the {side}"
        )
    if canine == "not_assessable":
        return (
            f"a {molar_text} molar relationship and a canine relationship that is "
            f"not assessable on the {side}"
        )
    return (
        f"a {molar_text} molar relationship and a {canine_text} canine relationship "
        f"on the {side}"
    )


def sagittal_sentence(labels: Mapping[str, str]) -> str | None:
    keys = (
        "right_molar_relation",
        "right_canine_relation",
        "left_molar_relation",
        "left_canine_relation",
    )
    if not all(labels.get(key) in RELATION_TEXT for key in keys):
        return None
    right_molar, right_canine, left_molar, left_canine = (labels[key] for key in keys)
    overjet = labels.get("overjet")
    overjet_suffix = (
        f", with overjet {OVERJET_TEXT[overjet]}" if overjet in OVERJET_TEXT else ""
    )
    if right_molar == left_molar and right_canine == left_canine:
        if right_molar == right_canine and right_molar != "not_assessable":
            relation = RELATION_TEXT[right_molar]
            return (
                f"Sagittally, there is a bilateral {relation} molar and canine relationship"
                f"{overjet_suffix}."
            )
        return (
            "Sagittally, there is "
            f"{side_relation(right_molar, right_canine, 'right')}, and "
            f"{side_relation(left_molar, left_canine, 'left')}{overjet_suffix}."
        )
    return (
        "Sagittally, there is "
        f"{side_relation(right_molar, right_canine, 'right')}, and "
        f"{side_relation(left_molar, left_canine, 'left')}{overjet_suffix}."
    )


def crowding_sentence(labels: Mapping[str, str]) -> str | None:
    upper = labels.get("upper_crowding")
    lower = labels.get("lower_crowding")
    if upper not in CROWDING_TEXT or lower not in CROWDING_TEXT:
        return None
    if upper == lower:
        if upper == "none":
            return "No crowding is present in the upper or lower arch."
        return f"There is {CROWDING_TEXT[upper]} crowding in both arches."
    return (
        f"There is {CROWDING_TEXT[upper]} crowding in the upper arch and "
        f"{CROWDING_TEXT[lower]} crowding in the lower arch."
    )


def curves_sentence(labels: Mapping[str, str]) -> str | None:
    spee = labels.get("curve_spee")
    wilson = labels.get("curve_wilson")
    if spee not in {"normal", "increased"} or wilson not in {"normal", "increased"}:
        return None
    if spee == wilson:
        state = "within normal limits" if spee == "normal" else "increased"
        return f"The curves of Spee and Wilson are {state}."
    spee_text = "within normal limits" if spee == "normal" else "increased"
    wilson_text = "within normal limits" if wilson == "normal" else "increased"
    return f"The curve of Spee is {spee_text}, while the curve of Wilson is {wilson_text}."


def render_report(labels: Mapping[str, str]) -> str:
    sentences: list[str] = []
    crossbite = labels.get("crossbite")
    if crossbite in CROSSBITE_SENTENCES:
        sentences.append(CROSSBITE_SENTENCES[crossbite])
    vertical = labels.get("vertical_relation")
    if vertical in VERTICAL_SENTENCES:
        sentences.append(VERTICAL_SENTENCES[vertical])
    sagittal = sagittal_sentence(labels)
    if sagittal:
        sentences.append(sagittal)
    elif labels.get("overjet") in OVERJET_TEXT:
        sentences.append(f"The overjet is {OVERJET_TEXT[labels['overjet']] }.")
    midline = labels.get("midline_relation")
    if midline in MIDLINE_SENTENCES:
        sentences.append(MIDLINE_SENTENCES[midline])
    curves = curves_sentence(labels)
    if curves:
        sentences.append(curves)
    crowding = crowding_sentence(labels)
    if crowding:
        sentences.append(crowding)
    if not sentences:
        raise ValueError("No supported Bite2Text labels were provided")
    return " ".join(sentences)

