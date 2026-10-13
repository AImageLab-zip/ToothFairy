#!/usr/bin/env python3
"""Conservative dental-consistency projection for predicted Stage-2 evidence.

The projection only removes mutually incompatible positive tuples. It never
creates a finding and leaves the raw branch probabilities under ``predictions``
untouched for auditability.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RULESET_VERSION = "dental-consistency-v1"
THIRD_MOLAR_IAC_TEETH = {"38", "48"}
RETAINED_STRUCTURE_TYPES = {"impaction", "root_remnant"}


def _probability(fact: Dict[str, Any]) -> float:
    value = fact.get("probability")
    return 1.0 if value is None else float(value)


def _active(fact: Dict[str, Any], min_probability: float) -> bool:
    return _probability(fact) >= min_probability


def _fact_teeth(fact: Dict[str, Any]) -> List[str]:
    teeth = [str(value) for value in fact.get("teeth") or []]
    target = str(fact.get("target", ""))
    if not teeth and target.startswith("tooth:"):
        teeth.append(target.split(":", 1)[1])
    return teeth


def _is_specific_iac_relation(fact: Dict[str, Any]) -> bool:
    if fact.get("type") != "canal_relationship":
        return False
    teeth = _fact_teeth(fact)
    return len(teeth) == 1 and teeth[0] in THIRD_MOLAR_IAC_TEETH


def _canal_orientation(fact: Dict[str, Any]) -> Tuple[str, str] | None:
    if fact.get("type") != "canal_relationship":
        return None
    finding = str(fact.get("finding", "")).lower()
    side = str(fact.get("side", "")).lower()
    if side not in {"left", "right"}:
        if "left" in finding:
            side = "left"
        elif "right" in finding:
            side = "right"
    if side not in {"left", "right"}:
        return None
    if "predominantly lingual" in finding:
        return side, "lingual"
    if "predominantly buccal" in finding:
        return side, "buccal"
    return None


def project_case(
    case: Dict[str, Any],
    min_probability: float = 0.6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the final dental consistency constraints to one merged case.

    A tooth-specific 38/48--IAC relation is suppressed when that tooth is
    predicted missing, unless active impaction or root-remnant evidence indicates
    image-verifiable tooth structure. Simultaneous lingual and buccal canal-course
    predictions are resolved by retaining the higher-probability orientation.
    """

    projected = copy.deepcopy(case)
    facts = list(projected.get("facts") or [])
    missing_teeth = {str(value) for value in projected.get("merged_missing_teeth") or []}
    retained_structure_teeth = {
        tooth
        for fact in facts
        if fact.get("type") in RETAINED_STRUCTURE_TYPES and _active(fact, min_probability)
        for tooth in _fact_teeth(fact)
    }

    removed_missing_iac: List[Dict[str, Any]] = []
    retained_missing_iac: List[Dict[str, Any]] = []
    after_missing_rule: List[Dict[str, Any]] = []
    for fact in facts:
        if not _is_specific_iac_relation(fact) or not _active(fact, min_probability):
            after_missing_rule.append(fact)
            continue
        tooth = _fact_teeth(fact)[0]
        if tooth not in missing_teeth:
            after_missing_rule.append(fact)
            continue
        if tooth in retained_structure_teeth:
            after_missing_rule.append(fact)
            retained_missing_iac.append(
                {
                    "tooth": tooth,
                    "reason": "active_impaction_or_root_remnant",
                    "fact": copy.deepcopy(fact),
                }
            )
            continue
        removed_missing_iac.append(
            {
                "tooth": tooth,
                "reason": "missing_tooth_without_active_retained_structure",
                "fact": copy.deepcopy(fact),
            }
        )

    active_orientations: Dict[str, Dict[str, List[Tuple[int, Dict[str, Any]]]]] = {
        "left": {"lingual": [], "buccal": []},
        "right": {"lingual": [], "buccal": []},
    }
    for index, fact in enumerate(after_missing_rule):
        orientation = _canal_orientation(fact)
        if orientation is not None and _active(fact, min_probability):
            side, direction = orientation
            active_orientations[side][direction].append((index, fact))

    remove_indices = set()
    canal_resolutions: List[Dict[str, Any]] = []
    for side, directions in active_orientations.items():
        if not directions["lingual"] or not directions["buccal"]:
            continue
        candidates = directions["lingual"] + directions["buccal"]
        retained_index, retained_fact = max(candidates, key=lambda item: _probability(item[1]))
        retained_orientation = _canal_orientation(retained_fact)[1]
        removed = []
        for index, fact in candidates:
            orientation = _canal_orientation(fact)[1]
            if orientation == retained_orientation:
                continue
            remove_indices.add(index)
            removed.append(
                {
                    "orientation": orientation,
                    "probability": _probability(fact),
                    "fact": copy.deepcopy(fact),
                }
            )
        canal_resolutions.append(
            {
                "side": side,
                "retained_orientation": retained_orientation,
                "retained_probability": _probability(retained_fact),
                "retained_fact_index": retained_index,
                "removed": removed,
            }
        )

    projected["facts"] = [
        fact for index, fact in enumerate(after_missing_rule) if index not in remove_indices
    ]
    audit = {
        "ruleset_version": RULESET_VERSION,
        "min_probability": float(min_probability),
        "input_fact_count": len(facts),
        "output_fact_count": len(projected["facts"]),
        "removed_fact_count": len(facts) - len(projected["facts"]),
        "removed_missing_tooth_iac_relations": removed_missing_iac,
        "retained_missing_tooth_iac_relations": retained_missing_iac,
        "canal_course_resolutions": canal_resolutions,
        "raw_predictions_modified": False,
    }
    return projected, audit


def project_directory(
    input_dir: Path,
    output_dir: Path,
    audit_dir: Path,
    min_probability: float,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for source in sorted(input_dir.glob("*.json")):
        case = json.loads(source.read_text())
        projected, audit = project_case(case, min_probability=min_probability)
        (output_dir / source.name).write_text(json.dumps(projected, indent=2) + "\n")
        (audit_dir / source.name).write_text(json.dumps(audit, indent=2) + "\n")
        cases.append(
            {
                "case_id": projected.get("case_id", source.stem),
                "file": source.name,
                "removed_fact_count": audit["removed_fact_count"],
            }
        )
    return {
        "ruleset_version": RULESET_VERSION,
        "min_probability": float(min_probability),
        "num_cases": len(cases),
        "num_modified_cases": sum(item["removed_fact_count"] > 0 for item in cases),
        "num_removed_facts": sum(item["removed_fact_count"] for item in cases),
        "cases": cases,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project merged evidence onto dental consistency constraints")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--min-probability", type=float, default=0.6)
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    summary = project_directory(
        args.input_dir,
        args.output_dir,
        args.audit_dir,
        args.min_probability,
    )
    (args.audit_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
