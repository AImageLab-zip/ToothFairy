#!/usr/bin/env python3
"""Create a patient-level IOS mesh manifest from structured Bite2Text labels.

One patient can have multiple IOS reports.  A target is emitted only when all
non-conflicted reports that mention that target agree.  Missing or ambiguous
targets are represented by -100 and are masked by the training loss.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "bite2text-mesh-manifest-v3"
IGNORE_INDEX = -100

HEAD_VOCABS: dict[str, list[str]] = {
    "right_molar_relation": ["class_i", "class_ii_edge_to_edge", "class_ii_full", "class_ii_unspecified", "class_iii", "not_assessable"],
    "right_canine_relation": ["class_i", "class_ii_edge_to_edge", "class_ii_full", "class_ii_unspecified", "class_iii", "not_assessable"],
    "left_molar_relation": ["class_i", "class_ii_edge_to_edge", "class_ii_full", "class_ii_unspecified", "class_iii", "not_assessable"],
    "left_canine_relation": ["class_i", "class_ii_edge_to_edge", "class_ii_full", "class_ii_unspecified", "class_iii", "not_assessable"],
    "overjet": ["normal", "increased", "reduced", "negative", "edge_to_edge"],
    "vertical_relation": ["normal", "increased", "reduced", "deep_bite", "open_bite"],
    "midline_relation": ["coincident", "slightly_deviated", "deviated"],
    # Report-extension heads.  They have meaningful negative classes and
    # sufficient patient-level consensus labels; see FOURTEENTH report.
    "crossbite": ["none", "anterior", "posterior", "present_unspecified"],
    "upper_crowding": ["none", "mild", "mild-to-moderate", "moderate", "moderate-to-severe", "severe"],
    "lower_crowding": ["none", "mild", "mild-to-moderate", "moderate", "moderate-to-severe", "severe"],
    "curve_spee": ["normal", "increased"],
    "curve_wilson": ["normal", "increased"],
}

DEFAULT_HEADS = [
    "right_molar_relation",
    "right_canine_relation",
    "left_molar_relation",
    "left_canine_relation",
    "overjet",
    "vertical_relation",
    "midline_relation",
]

FIELD_CONFLICT_WARNING = {
    "right_molar_relation": "conflict_right_molar_relation",
    "right_canine_relation": "conflict_right_canine_relation",
    "left_molar_relation": "conflict_left_molar_relation",
    "left_canine_relation": "conflict_left_canine_relation",
    "overjet": "conflict_overjet",
    "vertical_relation": "conflict_vertical_relation",
    "midline_relation": "conflict_midline_relation",
    "crossbite": "conflict_crossbite",
    "upper_crowding": "conflict_crowding",
    "lower_crowding": "conflict_crowding",
    "curve_spee": "conflict_curves",
    "curve_wilson": "conflict_curves",
}


def valid_value(row: dict[str, str], field: str, head_vocabs: dict[str, list[str]]) -> str | None:
    warnings = set(filter(None, (row.get("parse_warnings") or "").split(";")))
    if FIELD_CONFLICT_WARNING.get(field) in warnings:
        return None
    value = (row.get(field) or "").strip()
    return value if value in head_vocabs[field] else None


def strict_consensus(values: list[str]) -> tuple[str | None, str]:
    if not values:
        return None, "missing"
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0], "consensus"
    return None, "cross_report_conflict"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-policy",
        choices=("ios_consensus", "official_photo_first"),
        default="ios_consensus",
        help=(
            "ios_consensus preserves the historical geometry-only target; "
            "official_photo_first matches the evaluator ground truth by using "
            "the first intraoral-photo English report ordered by filename."
        ),
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=",".join(DEFAULT_HEADS),
        help="Comma-separated label heads. Defaults to the original seven; use an explicit subset for auxiliary report heads.",
    )
    args = parser.parse_args()

    labels_csv = args.labels_csv.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if not labels_csv.is_file():
        raise SystemExit(f"Missing labels CSV: {labels_csv}")
    if not data_root.is_dir():
        raise SystemExit(f"Missing data root: {data_root}")
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output_dir}")
    selected_heads = [name.strip() for name in args.heads.split(",") if name.strip()]
    if not selected_heads:
        raise SystemExit("--heads must select at least one head")
    unknown_heads = sorted(set(selected_heads).difference(HEAD_VOCABS))
    if unknown_heads:
        raise SystemExit(f"Unknown heads: {unknown_heads}")
    if len(selected_heads) != len(set(selected_heads)):
        raise SystemExit("--heads must not contain duplicates")
    head_vocabs = {head: HEAD_VOCABS[head] for head in selected_heads}
    output_dir.mkdir(parents=True)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with labels_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["patient_id"]].append(row)

    records: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    head_counts: dict[str, Counter[str]] = {head: Counter() for head in head_vocabs}
    state_counts: dict[str, Counter[str]] = {head: Counter() for head in head_vocabs}

    for patient_id, rows in sorted(grouped.items()):
        ios_rows = sorted(
            (row for row in rows if row.get("report_source") == "ios"),
            key=lambda row: (row.get("report_filename", ""), row.get("report_path", "")),
        )
        photo_rows = sorted(
            (row for row in rows if row.get("report_source") == "intraoral_photo"),
            key=lambda row: (row.get("report_filename", ""), row.get("report_path", "")),
        )
        if args.target_policy == "official_photo_first":
            target_rows = photo_rows[:1]
        else:
            target_rows = ios_rows
        split = rows[0].get("split", "")
        if split not in {"train", "val", "test"}:
            dropped.append({"patient_id": patient_id, "reason": f"split={split or 'missing'}"})
            continue
        upper = data_root / patient_id / "ios" / "ios_upper.stl"
        lower = data_root / patient_id / "ios" / "ios_lower.stl"
        if not upper.is_file() or not lower.is_file():
            dropped.append({"patient_id": patient_id, "reason": "missing_ios_mesh"})
            continue

        target_values: dict[str, str | None] = {}
        target_states: dict[str, str] = {}
        targets: dict[str, int] = {}
        for head, vocab in head_vocabs.items():
            values = [
                value
                for row in target_rows
                if (value := valid_value(row, head, head_vocabs)) is not None
            ]
            if args.target_policy == "official_photo_first":
                if not photo_rows:
                    value, state = None, "missing_official_photo_report"
                elif values:
                    value, state = values[0], "official_photo_first"
                else:
                    value, state = None, "missing_or_conflicted_official_photo_label"
            else:
                value, state = strict_consensus(values)
            target_values[head] = value
            target_states[head] = state
            targets[head] = vocab.index(value) if value is not None else IGNORE_INDEX
            state_counts[head][state] += 1
            if value is not None:
                head_counts[head][value] += 1

        records.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "patient_id": patient_id,
                "split": split,
                "upper_stl": str(upper.relative_to(data_root)),
                "lower_stl": str(lower.relative_to(data_root)),
                "ios_report_count": len(ios_rows),
                "photo_report_count": len(photo_rows),
                "target_policy": args.target_policy,
                "target_report_path": target_rows[0].get("report_path") if target_rows else None,
                "target_report_sha256": target_rows[0].get("report_sha256") if target_rows else None,
                "targets": targets,
                "target_values": target_values,
                "target_states": target_states,
            }
        )

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    (output_dir / "head_vocabs.json").write_text(json.dumps(head_vocabs, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "dropped_patients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "reason"])
        writer.writeheader()
        writer.writerows(dropped)

    split_counts = Counter(record["split"] for record in records)
    audit = {
        "manifest_version": MANIFEST_VERSION,
        "labels_csv": str(labels_csv),
        "data_root": str(data_root),
        "target_policy": args.target_policy,
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "dropped_patients": dropped,
        "head_vocabs": head_vocabs,
        "head_label_counts": {head: dict(sorted(counts.items())) for head, counts in head_counts.items()},
        "head_target_states": {head: dict(sorted(counts.items())) for head, counts in state_counts.items()},
    }
    (output_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "split_counts": dict(split_counts), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
