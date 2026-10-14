#!/usr/bin/env python3
"""Create an evidence-rich review sheet for Spee, Wilson, and dental midlines."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DESCRIPTORS = (
    ("Reversed", re.compile(r"\b(?:reverse|reversed)\b", re.I)),
    ("Decreased", re.compile(r"\b(?:decreased|flattened|flat)\b", re.I)),
    ("Increased", re.compile(r"\b(?:increased|accentuated|accented|markedly increased)\b", re.I)),
    ("Normal", re.compile(r"\b(?:within normal limits?|normal|correct)\b", re.I)),
)
CURVE_PRIORITY = ("Reversed", "Increased", "Decreased", "Normal")
MIDLINE_PRIORITY = ("Deviated", "Centered")
LOCATION_PRIORITY = ("Right side", "Left side", "Lower arch", "Upper arch")


def split_sentences(text):
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def descriptor_matches(sentence):
    matches = []
    for label, pattern in DESCRIPTORS:
        for match in pattern.finditer(sentence):
            matches.append((match.start(), match.end(), label, match.group(0)))
    return matches


def curve_label(sentence, term):
    lower = sentence.lower()
    term_match = re.search(term, lower)
    if not term_match:
        return None

    clause_pattern = re.compile(
        r"[,;]|\b(?:while|whereas|but|and)\b|"
        r"\band\s+(?=(?:the\s+)?(?:lower\s+|mandibular\s+)?curve of (?:spee|wilson))",
        re.I,
    )
    clauses = clause_pattern.split(sentence)
    local_clause = next(
        (clause for clause in clauses if re.search(term, clause, re.I)),
        sentence,
    )
    local_descriptors = descriptor_matches(local_clause)
    if local_descriptors:
        local_term = re.search(term, local_clause, re.I)
        term_center = (local_term.start() + local_term.end()) / 2
        nearest = min(
            local_descriptors,
            key=lambda item: abs(((item[0] + item[1]) / 2) - term_center),
        )
        return nearest[2]

    other = "wilson" if term == "spee" else "spee"
    if re.search(other, lower):
        other_label = None
        for clause in clauses:
            if re.search(other, clause, re.I):
                descriptors = descriptor_matches(clause)
                if descriptors:
                    other_match = re.search(other, clause, re.I)
                    other_center = (other_match.start() + other_match.end()) / 2
                    other_label = min(
                        descriptors,
                        key=lambda item: abs(((item[0] + item[1]) / 2) - other_center),
                    )[2]
                    break
        between_start = min(lower.index(term), lower.index(other))
        between_end = max(lower.index(term), lower.index(other))
        between = lower[between_start:between_end]
        contrasting = re.search(r"\b(?:while|whereas|but)\b|;", between)
        shared = re.search(r"\b(?:and|both|as is|as are)\b", lower)
        if other_label and shared and not contrasting:
            return other_label

    if re.search(
        r"cannot be assessed|cannot be determined|not assessable|"
        r"difficult to assess|cannot be defined",
        lower,
    ):
        return "unknown"
    return "unknown"

def midline_label(sentence):
    lower = sentence.lower()
    if not re.search(r"\b(?:midlines?|median lines?|centerlines?)\b", lower):
        return None
    if re.search(r"centered or (?:deviated|only|slightly|minimally)", lower):
        return "Centered"
    if re.search(r"centered \(.*deviated.*\)", lower):
        return "Centered"
    if re.search(
        r"\b(?:deviated|not coincident|do not coincide|does not coincide|"
        r"not centered|shifted|discrepant)\b",
        lower,
    ):
        return "Deviated"
    if re.search(r"\b(?:centered|centred|coincident|coincide|coincides)\b", lower):
        return "Centered"
    if re.search(
        r"cannot be (?:assessed|determined|defined)|not assessable|"
        r"difficult to (?:define|determine|assess)|not easily definable",
        lower,
    ):
        return "unknown"
    return "unknown"


def wilson_location(sentence):
    lower = sentence.lower()
    match = re.search(r"\bwilson\b", lower)
    if not match:
        return None

    prefix = lower[:match.start()]
    tail = lower[match.start():]
    local = lower[max(0, match.start() - 35):]

    if re.search(r"\b(?:right side|on the right|right quadrant|fourth quadrant|4[4-8])\b", tail):
        return "Right side"
    if re.search(r"\b(?:left side|on the left|left quadrant|third quadrant|3[4-8])\b", tail):
        return "Left side"
    if re.search(r"\b(?:right|left)(?: side)?(?:[- ](?:sided|side))?\s+(?:curve of\s+)?wilson\b", local):
        return "Right side" if "right" in local else "Left side"

    if re.search(r"\b(?:lower|mandibular)\s+(?:curve of\s+)?wilson\b", local):
        return "Lower arch"
    if re.search(r"\b(?:upper|maxillary)\s+(?:curve of\s+)?wilson\b", local):
        return "Upper arch"
    if re.search(r"\b(?:lower|mandibular)\s+(?:(?:spee|curves? of spee)\s+and\s+(?:wilson|(?:the )?curves? of wilson))", lower):
        return "Lower arch"
    if re.search(r"\b(?:upper|maxillary)\s+(?:(?:spee|curves? of spee)\s+and\s+(?:wilson|(?:the )?curves? of wilson))", lower):
        return "Upper arch"
    return "unknown"


def aggregate(votes, priority):
    known = [vote for vote in votes if vote and vote != "unknown"]
    if not known:
        return "unknown", False
    counts = Counter(known)
    maximum = max(counts.values())
    winners = [label for label, count in counts.items() if count == maximum]
    selected = next(label for label in priority if label in winners)
    return selected, len(counts) > 1


def collapse_source_votes(items, priority):
    """Give each report file one vote, regardless of repeated sentences."""
    grouped = {}
    for source, _, label in items:
        grouped.setdefault(source, []).append(label)
    return [
        (source, "", aggregate(labels, priority)[0])
        for source, labels in grouped.items()
    ]


def source_reports(case_dir):
    result = []
    for report_dir in sorted(case_dir.glob("reports*_en")):
        if not report_dir.is_dir():
            continue
        for source in sorted(report_dir.glob("*.txt")):
            text = source.read_text(encoding="utf-8-sig", errors="replace").strip()
            result.append((f"{report_dir.name}/{source.name}", text))
    return result


def evidence_text(items):
    return " || ".join(f"{source}: {sentence}" for source, sentence, _ in items)


def votes_text(items):
    return " | ".join(f"{source}={label}" for source, _, label in items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--old-annotation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = {}
    if args.old_annotation and args.old_annotation.is_file():
        with args.old_annotation.open(newline="", encoding="utf-8-sig") as stream:
            old = {row["Patient"]: row for row in csv.DictReader(stream)}

    rows = []
    for case_dir in sorted(args.dataset_root.glob("F*")):
        if not case_dir.is_dir():
            continue
        reports = source_reports(case_dir)
        spee_items = []
        wilson_items = []
        location_items = []
        midline_items = []
        for source, text in reports:
            for sentence in split_sentences(text):
                if re.search(r"\bspee\b", sentence, re.I):
                    spee_items.append((source, sentence, curve_label(sentence, "spee")))
                if re.search(r"\bwilson\b", sentence, re.I):
                    wilson_items.append((source, sentence, curve_label(sentence, "wilson")))
                    location_items.append((source, sentence, wilson_location(sentence)))
                if re.search(r"\b(?:midlines?|median lines?|centerlines?)\b", sentence, re.I):
                    midline_items.append((source, sentence, midline_label(sentence)))

        spee_source_items = collapse_source_votes(spee_items, CURVE_PRIORITY)
        wilson_source_items = collapse_source_votes(wilson_items, CURVE_PRIORITY)
        location_source_items = collapse_source_votes(location_items, LOCATION_PRIORITY)
        midline_source_items = collapse_source_votes(midline_items, MIDLINE_PRIORITY)
        spee, spee_conflict = aggregate(
            [item[2] for item in spee_source_items], CURVE_PRIORITY)
        wilson, wilson_conflict = aggregate(
            [item[2] for item in wilson_source_items], CURVE_PRIORITY)
        location, location_conflict = aggregate(
            [item[2] for item in location_source_items], LOCATION_PRIORITY)
        midline, midline_conflict = aggregate(
            [item[2] for item in midline_source_items], MIDLINE_PRIORITY)
        prior = old.get(case_dir.name, {})
        row = {
            "case_id": case_dir.name,
            "report_count": len(reports),
            "curve_of_spee": spee,
            "curve_of_wilson_degree": wilson,
            "curve_of_wilson_location": location,
            "midlines": midline,
            "spee_conflict": int(spee_conflict),
            "wilson_degree_conflict": int(wilson_conflict),
            "wilson_location_conflict": int(location_conflict),
            "midline_conflict": int(midline_conflict),
            "spee_votes": votes_text(spee_source_items),
            "wilson_degree_votes": votes_text(wilson_source_items),
            "wilson_location_votes": votes_text(location_source_items),
            "midline_votes": votes_text(midline_source_items),
            "spee_evidence": evidence_text(spee_items),
            "wilson_evidence": evidence_text(wilson_items),
            "midline_evidence": evidence_text(midline_items),
            "old_curve_of_spee": prior.get("Curve of Spee", ""),
            "old_curve_of_wilson_degree": prior.get("Curve of Wilson Degree", ""),
            "old_curve_of_wilson_location": prior.get("Curve of Wilson Location", ""),
            "old_midlines": prior.get("Midlines", ""),
        }
        row["old_mismatch"] = int(any((
            row["old_curve_of_spee"] and row["old_curve_of_spee"] != spee,
            row["old_curve_of_wilson_degree"] and row["old_curve_of_wilson_degree"] != wilson,
            row["old_curve_of_wilson_location"] and row["old_curve_of_wilson_location"] != location,
            row["old_midlines"] and row["old_midlines"] != midline,
        )))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("cases", len(rows))
    for field in (
        "curve_of_spee",
        "curve_of_wilson_degree",
        "curve_of_wilson_location",
        "midlines",
    ):
        print(field, Counter(row[field] for row in rows))
    print("conflict_cases", sum(any(row[key] for key in (
        "spee_conflict", "wilson_degree_conflict",
        "wilson_location_conflict", "midline_conflict"
    )) for row in rows))
    print("old_mismatch_cases", sum(row["old_mismatch"] for row in rows))


if __name__ == "__main__":
    main()
