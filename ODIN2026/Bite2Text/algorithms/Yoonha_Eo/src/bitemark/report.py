"""Deterministic report assembly without private corpus-derived phrasing."""

from __future__ import annotations

from collections.abc import Iterable

from .schema import Finding


def _index(findings: Iterable[Finding]) -> dict[tuple[str, str | None], Finding]:
    return {(finding.item, finding.side): finding for finding in findings}


def generate_report(findings: Iterable[Finding]) -> str:
    items = list(findings)
    indexed = _index(items)

    sentences: list[str] = []
    missing = indexed[("missing_teeth", None)].value
    if missing != "None":
        sentences.append(f"Missing permanent teeth are identified at FDI positions {missing}.")

    upper = indexed[("crowding", "upper")].value
    lower = indexed[("crowding", "lower")].value
    midline = indexed[("midline", None)].value.lower()
    sentences.append(f"Crowding is {upper} in the upper arch and {lower} in the lower arch.")
    sentences.append(f"The dental midline is {midline}.")

    for side in ("right", "left"):
        molar = indexed[("molar", side)].value
        canine = indexed[("canine", side)].value
        sentences.append(
            f"On the {side}, the molar relationship is {molar} and the canine relationship is {canine}."
        )

    anterior = indexed[("anterior_bite", None)].value.lower()
    transverse = indexed[("transverse_bite", None)].value.lower()
    overjet = indexed[("overjet", None)].value.lower()
    spee = indexed[("curve_of_spee", None)].value.lower()
    wilson = indexed[("curve_of_wilson", None)].value.lower()
    sentences.extend(
        [
            f"The anterior bite is {anterior}.",
            f"The transverse relationship is {transverse}.",
            f"Overjet is {overjet}.",
            f"The Curve of Spee is {spee}.",
            f"The Curve of Wilson is {wilson}.",
        ]
    )
    return " ".join(sentences)
