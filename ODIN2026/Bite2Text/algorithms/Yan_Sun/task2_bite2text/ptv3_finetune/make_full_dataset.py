#!/usr/bin/env python3
"""Create a train-only Bite2Text dataset containing every supervised case."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from make_cv_folds import class_weights, read_labels, source_samples


def link_sample(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")

    label_columns, labels = read_labels(data_root / "labels.csv")
    vocabs: dict[str, list[str]] = json.loads(
        (data_root / "head_vocabs.json").read_text(encoding="utf-8")
    )
    samples = source_samples(data_root)
    eligible = sorted(
        patient_id
        for patient_id in samples
        if patient_id in labels and any(value >= 0 for value in labels[patient_id])
    )
    excluded = sorted(set(samples).difference(eligible))
    if len(label_columns) != len(vocabs):
        raise RuntimeError("labels.csv/head_vocabs.json head count mismatch")

    for split in ("train", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)
    for patient_id in eligible:
        link_sample(samples[patient_id], output_root / "train" / samples[patient_id].name)

    with (output_root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", *label_columns])
        for patient_id in eligible:
            writer.writerow([patient_id, *labels[patient_id]])
    (output_root / "head_vocabs.json").write_text(
        json.dumps(vocabs, indent=2) + "\n", encoding="utf-8"
    )
    weights = class_weights(eligible, labels, vocabs)
    (output_root / "class_weights.json").write_text(
        json.dumps(weights, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "source_data_root": str(data_root),
        "train_cases": len(eligible),
        "eligible_ids": eligible,
        "excluded_no_supervision": excluded,
        "class_weights": weights,
        "val_cases": 0,
        "test_cases": 0,
        "evaluation_during_training": False,
    }
    (output_root / "full_dataset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"train_cases": len(eligible), "excluded_no_supervision": len(excluded)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

