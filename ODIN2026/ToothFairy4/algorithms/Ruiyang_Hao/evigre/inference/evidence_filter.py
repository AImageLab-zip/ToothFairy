#!/usr/bin/env python3
"""Versioned filtering of unreliable positive Stage-2 evidence predictions.

The filter removes only active, rendered evidence entries. Raw branch
probabilities under ``predictions`` remain untouched for auditability.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


FILTER_VERSION = "trusted-global-relation-v1"
SUPPORTED_FILTER_VERSIONS = {FILTER_VERSION}

DROPPED_GLOBAL_LABELS = (
    "atrophy_maxilla",
    "periodontitis_present",
    "poor_quality_or_artifact",
)
DROPPED_RELATION_LABELS = (
    "any_tooth_close_to_IAC",
    "lesion_close_to_IAC",
)
RETAINED_SPECIFIC_RELATION_LABELS = (
    "38_close_to_left_IAC",
    "48_close_to_right_IAC",
)

_FACT_SIGNATURES = {
    (
        "stage2_global",
        "periodontitis",
        "periodontal_bone",
        "periodontitis present",
    ): "periodontitis_present",
    (
        "stage2_global",
        "atrophy",
        "maxilla",
        "maxillary alveolar atrophy present",
    ): "atrophy_maxilla",
    (
        "stage2_relation",
        "canal_relationship",
        "mandibular_canal",
        "tooth close to inferior alveolar canal",
    ): "any_tooth_close_to_IAC",
    (
        "stage2_relation",
        "canal_relationship",
        "mandibular_canal",
        "lesion close to inferior alveolar canal",
    ): "lesion_close_to_IAC",
}
_WARNING_SIGNATURES = {
    ("stage2_global", "poor quality or artifact"): "poor_quality_or_artifact",
}


def _prediction_label(section: str, item: Mapping[str, Any]) -> str | None:
    if section == "facts":
        signature = (
            str(item.get("source", "")),
            str(item.get("type", "")),
            str(item.get("target", "")),
            str(item.get("finding", "")),
        )
        return _FACT_SIGNATURES.get(signature)
    if section == "warnings":
        signature = (
            str(item.get("source", "")),
            str(item.get("message", "")),
        )
        return _WARNING_SIGNATURES.get(signature)
    return None


def filter_case(
    case: Dict[str, Any],
    filter_version: str = FILTER_VERSION,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter one raw merged-evidence case before consistency projection."""

    if filter_version not in SUPPORTED_FILTER_VERSIONS:
        raise ValueError(
            f"Unsupported evidence filter {filter_version!r}; "
            f"expected one of {sorted(SUPPORTED_FILTER_VERSIONS)}"
        )

    filtered = copy.deepcopy(case)
    input_fact_count = len(filtered.get("facts") or [])
    input_warning_count = len(filtered.get("warnings") or [])
    removed: List[Dict[str, Any]] = []

    if filter_version == FILTER_VERSION:
        for section in ("facts", "warnings"):
            retained = []
            for item in filtered.get(section) or []:
                label = _prediction_label(section, item)
                if label is None:
                    retained.append(item)
                    continue
                removed.append(
                    {
                        "section": section,
                        "label": label,
                        "item": copy.deepcopy(item),
                    }
                )
            if section in filtered:
                filtered[section] = retained

    removed_by_label = Counter(entry["label"] for entry in removed)
    filter_metadata = {
        "filter_version": filter_version,
        "operation_order": "raw_predicted_evidence -> trusted_evidence_filter -> dental_consistency_projection",
        "dropped_global_labels": list(DROPPED_GLOBAL_LABELS),
        "dropped_relation_labels": list(DROPPED_RELATION_LABELS),
        "retained_specific_relation_labels": list(RETAINED_SPECIFIC_RELATION_LABELS),
        "removed_prediction_count": len(removed),
        "removed_by_label": dict(sorted(removed_by_label.items())),
        "raw_predictions_modified": False,
    }
    filtered["trusted_evidence_filter"] = filter_metadata
    audit = {
        **filter_metadata,
        "input_fact_count": input_fact_count,
        "output_fact_count": len(filtered.get("facts") or []),
        "input_warning_count": input_warning_count,
        "output_warning_count": len(filtered.get("warnings") or []),
        "removed_predictions": removed,
    }
    return filtered, audit


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    audit_dir: Path,
    filter_version: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for source in sorted(input_dir.glob("*.json")):
        case = json.loads(source.read_text(encoding="utf-8"))
        filtered, audit = filter_case(case, filter_version=filter_version)
        (output_dir / source.name).write_text(
            json.dumps(filtered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (audit_dir / source.name).write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cases.append(
            {
                "case_id": filtered.get("case_id", source.stem),
                "file": source.name,
                "removed_prediction_count": audit["removed_prediction_count"],
                "removed_by_label": audit["removed_by_label"],
            }
        )
    return {
        "filter_version": filter_version,
        "num_cases": len(cases),
        "num_modified_cases": sum(item["removed_prediction_count"] > 0 for item in cases),
        "num_removed_predictions": sum(item["removed_prediction_count"] for item in cases),
        "cases": cases,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter unreliable Stage-2 evidence before dental consistency projection"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--filter-version",
        choices=sorted(SUPPORTED_FILTER_VERSIONS),
        default=FILTER_VERSION,
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    summary = filter_directory(
        args.input_dir,
        args.output_dir,
        args.audit_dir,
        args.filter_version,
    )
    (args.audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
