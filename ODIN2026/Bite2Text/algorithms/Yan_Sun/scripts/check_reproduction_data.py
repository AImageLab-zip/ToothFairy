#!/usr/bin/env python3
"""Fail closed if prepared v9 cohorts, labels or patient folds disagree."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ids(root: Path, split: str) -> set[str]:
    return {p.stem.removeprefix("dental_") for p in (root / split).glob("dental_*.npz")}


def labels(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [f"label_{i}" for i in range(12)]
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"Expected 12 label columns: {path}")
        rows = list(reader)
    result = {row["patient_id"]: tuple(int(row[key]) for key in columns) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate label IDs: {path}")
    return result


def check(data: Path, cv: Path, full: Path, points: int = 991, supervised: int = 867) -> dict:
    raw_sets = [ids(data, split) for split in ("train", "val", "test")]
    raw = set.union(*raw_sets)
    if len(raw) != points or sum(map(len, raw_sets)) != points:
        raise ValueError("Point cohort count mismatch or overlapping development splits")
    rows = labels(data / "labels.csv")
    eligible = {pid for pid in raw if pid in rows and any(v >= 0 for v in rows[pid])}
    if len(eligible) != supervised or ids(full, "train") != eligible:
        raise ValueError("Full training set is not exactly the eligible supervised cohort")
    if ids(full, "val") or ids(full, "test"):
        raise ValueError("Full training set must have empty val/test directories")
    if labels(full / "labels.csv") != {pid: rows[pid] for pid in eligible}:
        raise ValueError("Full training labels differ from the source")
    vocab = json.loads((data / "head_vocabs.json").read_text())
    if len(vocab) != 12 or json.loads((full / "head_vocabs.json").read_text()) != vocab:
        raise ValueError("Vocabulary mismatch")
    assignments = {}
    with (cv / "fold_assignments.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["patient_id"] in assignments:
                raise ValueError("Duplicate fold assignments")
            assignments[row["patient_id"]] = int(row["fold"])
    if set(assignments) != eligible or set(assignments.values()) != set(range(1, 6)):
        raise ValueError("Fold assignment cohort mismatch")
    sizes = []
    for fold in range(1, 6):
        root = cv / f"fold{fold}"
        valid = {pid for pid, f in assignments.items() if f == fold}
        if ids(root, "val") != valid or ids(root, "test") != valid or ids(root, "train") != eligible - valid:
            raise ValueError(f"Leaking or mismatched fold {fold}")
        if labels(root / "labels.csv") != labels(full / "labels.csv"):
            raise ValueError(f"Fold {fold} label mismatch")
        if json.loads((root / "head_vocabs.json").read_text()) != vocab:
            raise ValueError(f"Fold {fold} vocabulary mismatch")
        sizes.append(len(valid))
    return {"point_cases": len(raw), "supervised_cases": len(eligible), "validation_fold_sizes": sizes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cv", "full"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.data, args.cv, args.full)))


if __name__ == "__main__":
    main()
