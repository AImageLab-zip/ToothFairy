#!/usr/bin/env python3
"""Deterministic precision-first filtering for retrieved Bite2Text reports."""

from __future__ import annotations

import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FDI_RE = re.compile(r"\b(?:[1-4][1-8]|[5-8][1-5])\b")
_TOOTH_WORD_RE = re.compile(r"\b(?:tooth|teeth)\b", re.IGNORECASE)

# The deployed 12-head geometry/photo models predict occlusal and arch-level
# slots. They do not localize restorative/pathology findings to an FDI tooth.
_UNSUPPORTED_FINDING_RE = re.compile(
    r"\b(?:"
    r"restoration(?:s)?|restored|filling(?:s)?|prosthetic|crown(?:s)?|"
    r"caries|carious|secondary\s+caries|sealant(?:s)?|"
    r"extraction|edentulous|"
    r"impacted|unerupted|erupting|ectopic|tipped|"
    r"attachment(?:s)?|button(?:s)?|rhinestone(?:s)?|"
    r"white\s+spot(?:s)?|deminerali[sz]ation(?:s)?|discolou?r(?:ed|ation)?|"
    r"darker\s+area|pigmented\s+fissure(?:s)?|fissures?\s+(?:are\s+)?pigmented"
    r")\b",
    re.IGNORECASE,
)
_LOCALIZED_UNSUPPORTED_RE = re.compile(
    r"\b(?:crossbite|white\s+spot(?:s)?|deminerali[sz]ation(?:s)?)\b",
    re.IGNORECASE,
)
_TOOTH_STATUS_RE = re.compile(
    r"\b(?:tooth|teeth)\b.*\b(?:present|retained|erupting|ectopic|missing|absent)\b",
    re.IGNORECASE,
)
_DENTITION_STATUS_RE = re.compile(
    r"(?:"
    r"\b(?:tooth|teeth|molar(?:s)?|premolar(?:s)?|incisor(?:s)?|canine(?:s)?)\b"
    r"[^.!?]{0,60}\b(?:missing|absent)\b|"
    r"\b(?:missing|absent|absence)\b"
    r"[^.!?]{0,60}\b(?:tooth|teeth|molar(?:s)?|premolar(?:s)?|incisor(?:s)?|canine(?:s)?)\b"
    r")",
    re.IGNORECASE,
)
_STRUCTURAL_FACT_RE = re.compile(
    r"\b(?:"
    r"(?:molar|canine)\s+(?:class|relationship)|class\s+(?:i|ii|iii).*\b(?:molar|canine)\b|"
    r"overjet|overbite|deep\s+bite|open\s+bite|crossbite|"
    r"transverse|vertical|midline|crowding|spacing|diastema|"
    r"curve\s+of\s+(?:spee|wilson)|maxillary\s+constriction"
    r")\b",
    re.IGNORECASE,
)
_EXPLICIT_NEGATIVE_RE = re.compile(
    r"(?:\b(?:no|without)\b|\bdo\s+not\b|\bdoes\s+not\b)[^.!?]{0,80}\b(?:"
    r"restoration(?:s)?|caries|carious\s+process(?:es)?|missing\s+teeth|"
    r"crown(?:s)?|sealant(?:s)?|attachment(?:s)?"
    r")\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def split_sentences(report: str) -> list[str]:
    """Split reports without changing the retained sentence text."""
    return [value.strip() for value in _SENTENCE_SPLIT_RE.split(report.strip()) if value.strip()]


def _normalized_sentence(sentence: str) -> str:
    return _NORMALIZE_RE.sub(" ", sentence.lower()).strip()


def rejection_reason(sentence: str) -> str | None:
    """Return a stable reason when a retrieved fact lacks model evidence."""
    value = _SPACE_RE.sub(" ", sentence.strip())
    if not value:
        return "empty"

    # Retain explicit negative findings to minimize unnecessary lexical drift;
    # RadFact's normal-finding filter excludes these from abnormal-fact scoring.
    if _EXPLICIT_NEGATIVE_RE.search(value):
        return None

    # Do not discard a complete occlusal sentence just because it also contains
    # a risky subordinate clause. v8a is a precision-only deletion pass; those
    # compound slots require the later structured renderer to edit safely.
    if _STRUCTURAL_FACT_RE.search(value):
        return None

    has_fdi = _FDI_RE.search(value) is not None
    has_tooth_reference = has_fdi or _TOOTH_WORD_RE.search(value) is not None

    if _UNSUPPORTED_FINDING_RE.search(value):
        return "unsupported_tooth_or_restorative_finding"
    if has_tooth_reference and _LOCALIZED_UNSUPPORTED_RE.search(value):
        return "unsupported_tooth_localization"
    if _TOOTH_STATUS_RE.search(value):
        return "unsupported_tooth_status"
    if _DENTITION_STATUS_RE.search(value):
        return "unsupported_tooth_status"
    return None


def sanitize_report(
    report: str,
    *,
    min_unsupported_sentences: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Filter unsupported retrieved facts and remove exact duplicates."""
    if min_unsupported_sentences < 1:
        raise ValueError("min_unsupported_sentences must be positive")

    retained: list[str] = []
    removed: list[dict[str, str]] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    source_sentences = split_sentences(report)
    decisions = [(sentence, rejection_reason(sentence)) for sentence in source_sentences]
    unsupported_count = sum(reason is not None for _, reason in decisions)
    activated = unsupported_count >= min_unsupported_sentences
    for sentence, detected_reason in decisions:
        reason = detected_reason if activated else None
        if reason is not None:
            removed.append({"sentence": sentence, "reason": reason})
            continue
        normalized = _normalized_sentence(sentence)
        if normalized in seen:
            duplicates.append(sentence)
            continue
        seen.add(normalized)
        retained.append(sentence)

    # The official contract requires a non-empty report. This should never be
    # needed for normal orthodontic reports, but provides a deterministic guard.
    fallback_used = not retained
    if fallback_used:
        retained = [
            "The available records are insufficient for a reliable detailed orthodontic assessment."
        ]

    if not removed and not duplicates:
        sanitized_report = report.strip()
    else:
        sanitized_report = " ".join(retained)

    return sanitized_report, {
        "version": "v8a-precision-sanitizer-2",
        "source_sentence_count": len(source_sentences),
        "detected_unsupported_sentence_count": unsupported_count,
        "activation_threshold": min_unsupported_sentences,
        "activated": activated,
        "retained_sentence_count": len(retained),
        "removed_sentence_count": len(removed),
        "duplicate_sentence_count": len(duplicates),
        "removed": removed,
        "duplicates": duplicates,
        "fallback_used": fallback_used,
    }
