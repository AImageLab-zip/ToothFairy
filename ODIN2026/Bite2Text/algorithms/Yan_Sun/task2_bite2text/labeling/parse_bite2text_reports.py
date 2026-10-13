#!/usr/bin/env python3
"""Build patient-disjoint, weakly structured labels from Bite2Text English reports.

The reports are clinical free text, so this is deliberately a deterministic,
auditable rule-based extractor rather than a claim of ground truth.  Every
output row retains the original relative path, source SHA-256, parser version,
and warning flags so that labels can be revised without losing provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PARSER_VERSION = "bite2text-rules-v3"
SEED = 20260807

RELATION_VALUES = {
    "class_i",
    "class_ii_edge_to_edge",
    "class_ii_full",
    "class_ii_unspecified",
    "class_iii",
    "not_assessable",
}

CORE_FIELDS = (
    "overjet",
    "vertical_relation",
    "midline_relation",
    "crossbite",
    "maxillary_constriction",
    "right_molar_relation",
    "right_canine_relation",
    "left_molar_relation",
    "left_canine_relation",
    "upper_crowding",
    "lower_crowding",
    "upper_spacing",
    "lower_spacing",
    "curve_spee",
    "curve_wilson",
)

ROMAN_TO_CLASS = {"i": "class_i", "ii": "class_ii", "iii": "class_iii", "second": "class_ii"}


@dataclass(frozen=True)
class RelationMention:
    start: int
    end: int
    targets: tuple[str, ...]
    label: str
    bilateral: bool


def normalize(text: str) -> str:
    """Normalize punctuation and whitespace while keeping word order intact."""
    text = text.lower().replace("–", "-").replace("—", "-")
    text = text.replace("end to end", "end-to-end").replace("edge to edge", "edge-to-edge")
    text = text.replace("end-on", "end-to-end").replace("end on", "end-to-end")
    text = re.sub(r"\((?:second|class\s+ii)\)", "class ii", text)
    text = re.sub(
        r"(class\s+(?:i{1,3}|second))\s*,\s*(full|end-to-end|edge-to-edge)",
        r"\1 \2",
        text,
    )
    text = text.replace("mild/moderate", "mild-to-moderate")
    text = text.replace("moderate/severe", "moderate-to-severe")
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def resolve(values: Iterable[str]) -> tuple[str | None, bool]:
    """Return the deterministic majority label and whether values conflict."""
    items = [value for value in values if value]
    if not items:
        return None, False
    counts = Counter(items)
    top_count = max(counts.values())
    # Keep first occurrence as the deterministic tie breaker.
    label = next(value for value in items if counts[value] == top_count)
    return label, len(counts) > 1


def resolve_relation(values: Iterable[str]) -> tuple[str | None, bool]:
    """Resolve relation mentions while treating generic Class II as weaker.

    Overlapping grammar patterns can read ``full Class II`` both as the
    specific relation and as generic ``Class II``.  The latter is not a true
    clinical conflict and must not mask an otherwise usable patient target.
    """
    items = [value for value in values if value]
    specific_class_ii = {
        value for value in items if value in {"class_ii_edge_to_edge", "class_ii_full"}
    }
    if specific_class_ii:
        items = [value for value in items if value != "class_ii_unspecified"]
    return resolve(items)


def classify_overjet(text: str) -> tuple[str | None, bool]:
    values: list[str] = []
    for sentence in sentences(text):
        if "overjet" not in sentence:
            continue
        if re.search(r"negative\s+overjet|overjet.{0,45}\bnegative\b", sentence):
            values.append("negative")
        elif re.search(r"reduced\s+overjet|overjet.{0,45}\b(reduced|decreased)\b", sentence):
            values.append("reduced")
        elif re.search(r"edge-to-edge\s+overjet|overjet.{0,45}\bedge-to-edge\b", sentence):
            values.append("edge_to_edge")
        elif re.search(r"increased\s+overjet|overjet.{0,45}\b(increased|excessive)\b", sentence):
            values.append("increased")
        elif re.search(r"overjet.{0,55}\b(within normal limits|normal)\b", sentence):
            values.append("normal")
    return resolve(values)


def classify_vertical(text: str) -> tuple[str | None, bool]:
    values: list[str] = []
    for sentence in sentences(text):
        if not any(term in sentence for term in ("overbite", "deep bite", "open bite", "vertical")):
            continue
        if "open bite" in sentence:
            values.append("open_bite")
        elif "deep bite" in sentence:
            values.append("deep_bite")
        elif re.search(r"reduced\s+overbite|overbite.{0,45}\b(reduced|decreased)\b", sentence):
            values.append("reduced")
        elif re.search(r"increased\s+overbite|overbite.{0,45}\bincreased\b", sentence):
            values.append("increased")
        elif re.search(r"overbite.{0,55}\b(within normal limits|normal)\b", sentence):
            values.append("normal")
        elif re.search(r"vertical (?:relationship|relationships|standpoint).{0,55}\b(within normal limits|correct|normal)\b", sentence):
            values.append("normal")
    return resolve(values)


def classify_midline(text: str) -> tuple[str | None, bool]:
    values: list[str] = []
    for sentence in sentences(text):
        if "midline" not in sentence:
            continue
        if "slightly deviated" in sentence:
            values.append("slightly_deviated")
        elif re.search(r"\b(deviated|not coincident|not centered)\b", sentence):
            values.append("deviated")
        elif re.search(r"\b(coincident|centered|centred)\b", sentence):
            values.append("coincident")
    return resolve(values)


def classify_crossbite(text: str) -> tuple[str | None, bool]:
    values: list[str] = []
    for sentence in sentences(text):
        if "crossbite" not in sentence and "cross bite" not in sentence:
            continue
        if re.search(r"(absence of|without|no) (?:\w+ )*(?:crossbite|crossbites|cross bite)", sentence):
            values.append("none")
        elif "anterior crossbite" in sentence:
            values.append("anterior")
        elif re.search(r"posterior crossbite|lateral crossbite|crossbite involving teeth", sentence):
            values.append("posterior")
        else:
            values.append("present_unspecified")
    return resolve(values)


def classify_constriction(text: str) -> tuple[str | None, bool]:
    values: list[str] = []
    for sentence in sentences(text):
        if "constriction" not in sentence:
            continue
        if not re.search(r"maxill\w*\s+constriction|constriction\s+of\s+the\s+maxill", sentence):
            continue
        if re.search(r"\b(slight|mild)\b", sentence):
            values.append("mild")
        elif re.search(r"\b(severe|marked)\b", sentence):
            values.append("severe")
        else:
            values.append("present")
    return resolve(values)


def severity_from_match(match: re.Match[str]) -> str:
    value = match.group("severity").replace(" ", "_")
    return {"no": "none"}.get(value, value)


def extract_arch_crowding(text: str) -> tuple[dict[str, str | None], bool]:
    """Extract upper/lower crowding; preserve ambiguity rather than guessing."""
    labels: dict[str, list[str]] = {"upper": [], "lower": []}
    severity = r"(?P<severity>no|mild|moderate|severe|mild-to-moderate|moderate-to-severe)"
    for sentence in sentences(text):
        if "crowding" not in sentence:
            continue
        both = re.search(r"\b(both arches|both the upper and lower arches|upper and lower arches|upper and lower arch)\b", sentence)
        if both:
            match = re.search(severity + r"\s+crowding", sentence)
            if match:
                value = severity_from_match(match)
                labels["upper"].append(value)
                labels["lower"].append(value)
        if re.search(r"crowding\s+is\s+absent\s+in\s+(?:the\s+)?upper\s+and\s+lower\s+arches", sentence):
            labels["upper"].append("none")
            labels["lower"].append("none")
        elif re.search(r"\bno\s+crowding\b", sentence) and not re.search(r"\b(?:upper|lower)\s+arch\b", sentence):
            labels["upper"].append("none")
            labels["lower"].append("none")

        for match in re.finditer(
            severity + r"\s+crowding(?:\s+is\s+present)?\s+(?:in\s+)?(?:the\s+)?(?P<arch>upper|lower)\s+arch",
            sentence,
        ):
            labels[match.group("arch")].append(severity_from_match(match))
        for match in re.finditer(severity + r"\s+(?P<arch>upper|lower)\s+crowding", sentence):
            labels[match.group("arch")].append(severity_from_match(match))
        for match in re.finditer(
            r"(?P<arch>upper|lower)\s+arch(?:\s+has|\s+shows|\s+with|\s+is)?\s+" + severity + r"\s+crowding",
            sentence,
        ):
            labels[match.group("arch")].append(severity_from_match(match))
        for match in re.finditer(
            r"(?:in\s+(?:the\s+)?)?(?P<arch>upper|lower)\s+arch\s+(?:there\s+is|there\s+are)\s+" + severity + r"\s+crowding",
            sentence,
        ):
            labels[match.group("arch")].append(severity_from_match(match))
        for match in re.finditer(
            severity + r"(?:\s+crowding)?\s+in\s+(?:the\s+)?(?P<arch>upper|lower)\s+arch",
            sentence,
        ):
            labels[match.group("arch")].append(severity_from_match(match))
        for match in re.finditer(r"\babsent\s+in\s+(?:the\s+)?(?P<arch>upper|lower)\s+arch", sentence):
            labels[match.group("arch")].append("none")

    upper, upper_conflict = resolve(labels["upper"])
    lower, lower_conflict = resolve(labels["lower"])
    return {"upper": upper, "lower": lower}, upper_conflict or lower_conflict


def extract_arch_spacing(text: str) -> tuple[dict[str, str | None], bool]:
    labels: dict[str, list[str]] = {"upper": [], "lower": []}
    for sentence in sentences(text):
        if not re.search(r"\b(spaces?|diastemas?)\b", sentence):
            continue
        if re.search(r"\b(both arches|both the upper and lower arches|upper and lower arches|upper and lower arch)\b", sentence):
            labels["upper"].append("present")
            labels["lower"].append("present")
        for arch in ("upper", "lower"):
            if re.search(rf"\b{arch}\s+arch\b|\b{arch}\s+(?:spaces?|diastemas?)\b", sentence):
                labels[arch].append("present")
    upper, upper_conflict = resolve(labels["upper"])
    lower, lower_conflict = resolve(labels["lower"])
    return {"upper": upper, "lower": lower}, upper_conflict or lower_conflict


def classify_curves(text: str) -> tuple[dict[str, str | None], bool]:
    values: dict[str, list[str]] = {"spee": [], "wilson": []}
    for sentence in sentences(text):
        if "spee" not in sentence and "wilson" not in sentence:
            continue

        # Shared predicates such as ``the curves of Spee and Wilson are
        # increased`` apply to both curves.  Require no competing predicate
        # between the two names so that ``Spee is normal and Wilson is
        # increased`` remains two distinct labels.
        shared = re.search(
            r"spee\s+(?:and|,)\s+(?:(?:the\s+)?(?:curve\s+of\s+)?)?wilson\s+"
            r"(?:is|are|appears?|would\s+appear)?\s*"
            r"(?P<label>increased|accentuated|within\s+normal\s+limits|normal|flat)",
            sentence,
        )
        if shared:
            label = (
                "increased"
                if shared.group("label") in {"increased", "accentuated"}
                else "normal"
            )
            values["spee"].append(label)
            values["wilson"].append(label)
            continue

        occurrences = sorted(
            [(match.start(), match.end(), name) for name in ("spee", "wilson") for match in re.finditer(name, sentence)]
        )
        for index, (_start, end, name) in enumerate(occurrences):
            next_start = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(sentence)
            segment = sentence[end:next_start]
            # Do not borrow a predicate from the following curve clause.
            segment = re.split(r"\b(?:and|while|whereas|but)\b|[;,]", segment, maxsplit=1)[0]
            match = re.search(
                r"\b(increased|accentuated|within\s+normal\s+limits|normal|flat)\b",
                segment,
            )
            if not match:
                continue
            values[name].append(
                "increased" if match.group(1) in {"increased", "accentuated"} else "normal"
            )
    spee, spee_conflict = resolve(values["spee"])
    wilson, wilson_conflict = resolve(values["wilson"])
    return {"spee": spee, "wilson": wilson}, spee_conflict or wilson_conflict


RELATION_RE = re.compile(
    r"(?P<prefix>(?:(?:full|end-to-end|edge-to-edge|super|bilateral)\s+){0,2})"
    r"class\s+(?P<class>i{1,3}|second)"
    r"(?P<infix>\s+(?:with\s+)?(?:full|end-to-end|edge-to-edge))?\s+"
    r"(?P<targets>molar(?:s)?(?:\s+and\s+canines?)?|canine(?:s)?(?:\s+and\s+molars?)?)\s+relationships?",
    flags=re.IGNORECASE,
)

SHARED_CLASS_RELATION_RE = re.compile(
    r"(?P<prefix>bilateral\s+)?class\s+(?P<class>i{1,3}|second)\s+relationship\s+with\s+"
    r"(?P<modifier>full|end-to-end|edge-to-edge)\s+"
    r"(?P<targets>molar(?:s)?(?:\s+and\s+canines?)?|canine(?:s)?(?:\s+and\s+molars?)?)\s+relationships?",
    flags=re.IGNORECASE,
)

TARGETS_PATTERN = r"molar(?:s)?(?:\s+and\s+canines?)?|canine(?:s)?(?:\s+and\s+molars?)?"
MODIFIER_PATTERN = r"full|end-to-end|edge-to-edge|weak|mild|super"

CLASS_FIRST_SIMPLE_RE = re.compile(
    rf"(?P<prefix>(?:(?:{MODIFIER_PATTERN}|bilateral)\s+){{0,3}})"
    r"class\s+(?P<class>i{1,3}|second)\s+"
    rf"(?P<infix>(?:(?:with\s+)?(?:{MODIFIER_PATTERN})\s+){{0,2}})"
    rf"(?P<targets>{TARGETS_PATTERN})(?:\s+relationships?)?",
    flags=re.IGNORECASE,
)

TARGET_FIRST_RELATION_RE = re.compile(
    rf"(?P<target_prefix>(?:{MODIFIER_PATTERN})\s+)?"
    rf"(?:the\s+)?(?P<targets>{TARGETS_PATTERN})(?:\s+relationships?)?\s+"
    rf"(?P<modifier_before>(?:{MODIFIER_PATTERN})\s+)?"
    r"(?:is\s+|are\s+|in\s+)?(?:class\s+)?(?P<class>i{1,3}|second)\b"
    rf"(?:\s+(?P<modifier_after>{MODIFIER_PATTERN}))?",
    flags=re.IGNORECASE,
)

GENERIC_BILATERAL_RELATION_RE = re.compile(
    rf"(?P<prefix>bilateral(?:ly)?\s+)(?P<modifier>{MODIFIER_PATTERN})\s+"
    r"class\s+(?P<class>i{1,3}|second)\s+relationships?\b",
    flags=re.IGNORECASE,
)

NOT_ASSESSABLE_RE = re.compile(
    rf"(?P<targets>{TARGETS_PATTERN})(?:\s+class|\s+relationships?)?"
    r"(?:(?!\bmolars?\b|\bcanines?\b).){0,80}?"
    r"(?:not\s+assessable|cannot\s+be\s+(?:assessed|determined|defined)|"
    r"cannot\s+be\s+evaluated|cannot\s+be\s+established|"
    r"cannot\s+be\s+identified|cannot\s+be\s+classified|"
    r"not\s+possible\s+to\s+(?:assess|determine|define)|"
    r"(?:is\s+)?non-definable|difficult\s+to\s+(?:assess|define))",
    flags=re.IGNORECASE,
)

NOT_ASSESSABLE_PREFIX_RE = re.compile(
    r"(?:not\s+possible\s+to\s+(?:assess|determine|define|establish)|"
    r"cannot\s+be\s+(?:assessed|determined|defined|evaluated|established|identified|classified)|"
    r"non-definable)"
    rf"(?:(?!\bmolars?\b|\bcanines?\b).){{0,40}}?"
    rf"(?P<targets>{TARGETS_PATTERN})(?:\s+class|\s+relationships?)?",
    flags=re.IGNORECASE,
)


def relation_label(roman: str, modifier: str | None) -> str:
    base = ROMAN_TO_CLASS[roman.lower()]
    if base != "class_ii":
        return base
    modifier = (modifier or "").strip().lower()
    if modifier in {"end-to-end", "edge-to-edge"}:
        return "class_ii_edge_to_edge"
    if modifier == "full":
        return "class_ii_full"
    return "class_ii_unspecified"


def relation_mentions(sentence: str) -> list[RelationMention]:
    mentions: list[RelationMention] = []
    def targets_from_text(target_text: str) -> tuple[str, ...]:
        target_text = target_text.lower()
        return ("molar", "canine") if "and" in target_text else (("molar" if target_text.startswith("molar") else "canine"),)

    def append_match(match: re.Match[str], modifier: str | None, bilateral: bool) -> None:
        mentions.append(
            RelationMention(
                start=match.start(),
                end=match.end(),
                targets=targets_from_text(match.group("targets")),
                label=relation_label(match.group("class"), modifier),
                bilateral=bilateral,
            )
        )

    for match in RELATION_RE.finditer(sentence):
        modifier_search = re.search(r"full|end-to-end|edge-to-edge", (match.group("prefix") or "") + (match.group("infix") or ""))
        append_match(match, modifier_search.group(0) if modifier_search else None, "bilateral" in (match.group("prefix") or ""))
    for match in SHARED_CLASS_RELATION_RE.finditer(sentence):
        append_match(match, match.group("modifier"), bool(match.group("prefix")))
    for match in CLASS_FIRST_SIMPLE_RE.finditer(sentence):
        modifier_search = re.search(
            r"full|end-to-end|edge-to-edge",
            (match.group("prefix") or "") + (match.group("infix") or ""),
        )
        append_match(
            match,
            modifier_search.group(0) if modifier_search else None,
            "bilateral" in (match.group("prefix") or ""),
        )
    for match in TARGET_FIRST_RELATION_RE.finditer(sentence):
        append_match(
            match,
            match.group("target_prefix")
            or match.group("modifier_before")
            or match.group("modifier_after"),
            False,
        )
    for match in GENERIC_BILATERAL_RELATION_RE.finditer(sentence):
        mentions.append(
            RelationMention(
                start=match.start(),
                end=match.end(),
                targets=("molar", "canine"),
                label=relation_label(match.group("class"), match.group("modifier")),
                bilateral=True,
            )
        )
    for match in NOT_ASSESSABLE_RE.finditer(sentence):
        mentions.append(
            RelationMention(
                start=match.start(),
                end=match.end(),
                targets=targets_from_text(match.group("targets")),
                label="not_assessable",
                bilateral=False,
            )
        )
    for match in NOT_ASSESSABLE_PREFIX_RE.finditer(sentence):
        mentions.append(
            RelationMention(
                start=match.start(),
                end=match.end(),
                targets=targets_from_text(match.group("targets")),
                label="not_assessable",
                bilateral=False,
            )
        )

    # Several permissive patterns intentionally overlap.  Exact duplicate
    # mentions do not add evidence and can otherwise distort majority ties.
    unique: list[RelationMention] = []
    seen: set[tuple[int, int, tuple[str, ...], str]] = set()
    for mention in sorted(mentions, key=lambda item: (item.start, item.end, item.label)):
        key = (mention.start, mention.end, mention.targets, mention.label)
        if key not in seen:
            unique.append(mention)
            seen.add(key)
    return unique


SIDE_MARKER_RE = re.compile(
    r"\bon\s+(?:the\s+)?(?P<postfix>right|left)\b|"
    r"\b(?P<prefix>right|left)(?=\s+(?:molar|canine)\b)",
    flags=re.IGNORECASE,
)


def marker_side(marker: re.Match[str]) -> str:
    return str(marker.group("postfix") or marker.group("prefix")).lower()


def closest_side(sentence: str, mention: RelationMention, mentions: list[RelationMention]) -> str | None:
    """Associate a relation with a nearby left/right clause.

    Reports use both ``Class II ... on the right`` and ``on the right ...
    Class II`` constructions.  Prefer a preceding marker when it begins a
    clause; otherwise use the following marker, which covers the postfix form.
    """
    markers = list(SIDE_MARKER_RE.finditer(sentence))
    if not markers:
        return None
    prior_index = next((index for index in range(len(markers) - 1, -1, -1) if markers[index].end() <= mention.start), None)
    next_index = next((index for index, marker in enumerate(markers) if marker.start() >= mention.end), None)

    if prior_index is not None:
        prior = markers[prior_index]
        prior_boundary = markers[prior_index - 1].end() if prior_index else 0
        prior_is_postfix = any(
            other is not mention and prior_boundary <= other.start and other.end <= prior.start()
            for other in mentions
        )
        if not prior_is_postfix and mention.start - prior.end() <= 180:
            return marker_side(prior)

    if next_index is not None:
        following = markers[next_index]
        if following.start() - mention.end <= 180:
            return marker_side(following)

    if prior_index is not None and mention.start - markers[prior_index].end() <= 180:
        return marker_side(markers[prior_index])
    return None


def extract_sagittal(text: str) -> tuple[dict[str, str | None], dict[str, bool]]:
    values: dict[str, list[str]] = {
        "right_molar": [],
        "right_canine": [],
        "left_molar": [],
        "left_canine": [],
    }
    for sentence in sentences(text):
        mentions = relation_mentions(sentence)
        for mention in mentions:
            bilateral = mention.bilateral or bool(
                re.search(
                    r"\b(?:on\s+)?(?:both\s+)?(?:the\s+)?right\s+and\s+(?:the\s+)?left(?:\s+sides?)?\b|"
                    r"\bon\s+both\s+sides\b|\bbilateral(?:ly)?\b",
                    sentence,
                )
            )
            side = closest_side(sentence, mention, mentions)
            # In orthodontic reports an unqualified sagittal relation denotes
            # the bilateral relation.  Side-qualified clauses are handled by
            # closest_side above; only truly unqualified mentions reach here.
            sides = ("right", "left") if bilateral or side is None else (side,)
            for side in sides:
                if side is None:
                    continue
                for target in mention.targets:
                    values[f"{side}_{target}"].append(mention.label)
    output: dict[str, str | None] = {}
    conflicts: dict[str, bool] = {}
    for key, labels in values.items():
        output[key], has_conflict = resolve_relation(labels)
        conflicts[key] = has_conflict
    return output, conflicts


def parse_report(text: str) -> dict[str, Any]:
    normalized = normalize(text)
    overjet, overjet_conflict = classify_overjet(normalized)
    vertical, vertical_conflict = classify_vertical(normalized)
    midline, midline_conflict = classify_midline(normalized)
    crossbite, crossbite_conflict = classify_crossbite(normalized)
    constriction, constriction_conflict = classify_constriction(normalized)
    crowding, crowding_conflict = extract_arch_crowding(normalized)
    spacing, spacing_conflict = extract_arch_spacing(normalized)
    curves, curves_conflict = classify_curves(normalized)
    sagittal, sagittal_conflicts = extract_sagittal(normalized)

    labels = {
        "overjet": overjet,
        "vertical_relation": vertical,
        "midline_relation": midline,
        "crossbite": crossbite,
        "maxillary_constriction": constriction,
        "right_molar_relation": sagittal["right_molar"],
        "right_canine_relation": sagittal["right_canine"],
        "left_molar_relation": sagittal["left_molar"],
        "left_canine_relation": sagittal["left_canine"],
        "upper_crowding": crowding["upper"],
        "lower_crowding": crowding["lower"],
        "upper_spacing": spacing["upper"],
        "lower_spacing": spacing["lower"],
        "curve_spee": curves["spee"],
        "curve_wilson": curves["wilson"],
    }
    conflicts = {
        "overjet": overjet_conflict,
        "vertical_relation": vertical_conflict,
        "midline_relation": midline_conflict,
        "crossbite": crossbite_conflict,
        "maxillary_constriction": constriction_conflict,
        "crowding": crowding_conflict,
        "spacing": spacing_conflict,
        "curves": curves_conflict,
        **{f"{key}_relation": present for key, present in sagittal_conflicts.items()},
    }
    warnings = [f"conflict_{name}" for name, present in conflicts.items() if present]
    if not any(labels[field] is not None for field in CORE_FIELDS):
        warnings.append("no_core_label")
    if not any(labels[field] is not None for field in ("right_molar_relation", "left_molar_relation")):
        warnings.append("missing_molar_labels")
    return {"normalized_text": normalized, "labels": labels, "warnings": warnings}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_incomplete_cases(metadata_dir: Path) -> set[str]:
    path = metadata_dir / "incomplete_cases.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle) if row.get("patient_id")}


def create_patient_splits(case_ids: list[str], incomplete: set[str]) -> dict[str, str]:
    complete = [case_id for case_id in sorted(case_ids) if case_id not in incomplete]
    shuffled = complete.copy()
    random.Random(SEED).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.80)
    val_end = train_end + int(len(shuffled) * 0.10)
    mapping = {case_id: "train" for case_id in shuffled[:train_end]}
    mapping.update({case_id: "val" for case_id in shuffled[train_end:val_end]})
    mapping.update({case_id: "test" for case_id in shuffled[val_end:]})
    mapping.update({case_id: "excluded_incomplete" for case_id in incomplete})
    return mapping


def iter_report_paths(data_root: Path) -> Iterable[tuple[str, str, Path]]:
    source_map = {
        "reports_ios_en": "ios",
        "reports_intraoral-photo_en": "intraoral_photo",
    }
    for patient_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for folder, source in source_map.items():
            report_dir = patient_dir / folder
            if report_dir.is_dir():
                for report in sorted(report_dir.glob("*.txt")):
                    yield patient_dir.name, source, report


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    metadata_dir = args.metadata_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Missing data root: {data_root}")
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    case_ids = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    incomplete = load_incomplete_cases(metadata_dir)
    splits = create_patient_splits(case_ids, incomplete)
    records: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    case_source_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for patient_id, source, report_path in iter_report_paths(data_root):
        raw_text = report_path.read_text(encoding="utf-8", errors="replace").strip()
        parsed = parse_report(raw_text)
        source_counts[source] += 1
        case_source_counts[patient_id][source] += 1
        record = {
            "schema_version": PARSER_VERSION,
            "patient_id": patient_id,
            "split": splits.get(patient_id, "excluded_unknown"),
            "report_source": source,
            "report_filename": report_path.name,
            "report_path": str(report_path.relative_to(data_root)),
            "report_sha256": sha256_text(raw_text),
            "report_text": raw_text,
            "normalized_report_text": parsed["normalized_text"],
            "labels": parsed["labels"],
            "parse_warnings": parsed["warnings"],
        }
        records.append(record)
        flattened.append(
            {
                "patient_id": patient_id,
                "split": record["split"],
                "report_source": source,
                "report_filename": report_path.name,
                "report_path": record["report_path"],
                "report_sha256": record["report_sha256"],
                **parsed["labels"],
                "parse_warnings": ";".join(parsed["warnings"]),
            }
        )

    label_coverage = {
        field: sum(record["labels"][field] is not None for record in records) for field in CORE_FIELDS
    }
    label_distribution = {
        field: dict(sorted(Counter(record["labels"][field] for record in records if record["labels"][field] is not None).items()))
        for field in CORE_FIELDS
    }
    warning_counts = Counter(warning for record in records for warning in record["parse_warnings"])
    split_case_counts = Counter(splits.values())
    split_report_counts = Counter(record["split"] for record in records)

    split_rows = [
        {"patient_id": case_id, "split": splits[case_id], "seed": SEED, "parser_version": PARSER_VERSION}
        for case_id in sorted(splits)
    ]
    case_rows = [
        {
            "patient_id": case_id,
            "split": splits[case_id],
            "ios_report_count": case_source_counts[case_id]["ios"],
            "intraoral_photo_report_count": case_source_counts[case_id]["intraoral_photo"],
            "total_report_count": sum(case_source_counts[case_id].values()),
            "eligible_for_training": int(splits[case_id] in {"train", "val", "test"}),
        }
        for case_id in case_ids
    ]

    write_jsonl(output_dir / "report_records.jsonl", records)
    csv_fields = [
        "patient_id", "split", "report_source", "report_filename", "report_path", "report_sha256", *CORE_FIELDS, "parse_warnings"
    ]
    write_csv(output_dir / "report_labels.csv", flattened, csv_fields)
    write_csv(output_dir / "patient_splits.csv", split_rows, ["patient_id", "split", "seed", "parser_version"])
    write_csv(
        output_dir / "case_report_index.csv",
        case_rows,
        ["patient_id", "split", "ios_report_count", "intraoral_photo_report_count", "total_report_count", "eligible_for_training"],
    )

    audit = {
        "parser_version": PARSER_VERSION,
        "seed": SEED,
        "data_root": str(data_root),
        "records": len(records),
        "cases": len(case_ids),
        "incomplete_cases_excluded": sorted(incomplete),
        "source_report_counts": dict(sorted(source_counts.items())),
        "split_case_counts": dict(sorted(split_case_counts.items())),
        "split_report_counts": dict(sorted(split_report_counts.items())),
        "label_coverage": label_coverage,
        "label_coverage_fraction": {field: round(count / len(records), 6) for field, count in label_coverage.items()},
        "label_distribution": label_distribution,
        "warning_counts": dict(sorted(warning_counts.items())),
    }
    (output_dir / "parse_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "splits": dict(split_case_counts), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
