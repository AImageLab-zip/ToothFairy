#!/usr/bin/env python3
"""Create patient-level multi-label CV folds from a prepared PTv3 dataset.

Point samples are hard-linked, not copied.  Fold assignment uses a deterministic
rarity-aware greedy objective so uncommon head/value labels are spread across
validation folds more evenly than with plain random splitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_INDEX = -100


def read_labels(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    label_columns = sorted(
        (column for column in rows[0] if column.startswith("label_")),
        key=lambda column: int(column.split("_")[1]),
    )
    labels = {
        row["patient_id"]: [int(row[column]) for column in label_columns]
        for row in rows
    }
    return label_columns, labels


def source_samples(data_root: Path) -> dict[str, Path]:
    samples: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for path in sorted((data_root / split).glob("dental_*.npz")):
            patient_id = path.stem.removeprefix("dental_")
            if patient_id in samples:
                raise RuntimeError(f"Duplicate patient sample: {patient_id}")
            samples[patient_id] = path
    return samples


def label_tokens(values: list[int]) -> tuple[tuple[int, int], ...]:
    return tuple((index, value) for index, value in enumerate(values) if value >= 0)


def assign_folds(
    labels: dict[str, list[int]],
    patient_ids: list[str],
    num_folds: int,
    seed: int,
) -> dict[str, int]:
    global_counts: Counter[tuple[int, int]] = Counter()
    for patient_id in patient_ids:
        global_counts.update(label_tokens(labels[patient_id]))

    rng = random.Random(seed)
    tie_breakers = {patient_id: rng.random() for patient_id in patient_ids}
    ordered = sorted(
        patient_ids,
        key=lambda patient_id: (
            -sum(1.0 / global_counts[token] for token in label_tokens(labels[patient_id])),
            tie_breakers[patient_id],
            patient_id,
        ),
    )
    fold_counts = [Counter() for _ in range(num_folds)]
    fold_sizes = [0] * num_folds
    assignments: dict[str, int] = {}
    target_size = len(patient_ids) / num_folds

    for patient_id in ordered:
        tokens = label_tokens(labels[patient_id])
        scores: list[tuple[float, int, int]] = []
        for fold in range(num_folds):
            rarity_cost = sum(
                (fold_counts[fold][token] + 1) / global_counts[token]
                for token in tokens
            )
            size_cost = ((fold_sizes[fold] + 1) / target_size) ** 2
            scores.append((rarity_cost + 0.25 * size_cost, fold_sizes[fold], fold))
        chosen = min(scores)[2]
        assignments[patient_id] = chosen
        fold_counts[chosen].update(tokens)
        fold_sizes[chosen] += 1
    return assignments


def class_weights(
    patient_ids: list[str], labels: dict[str, list[int]], vocabs: dict[str, list[str]]
) -> list[list[float]]:
    output: list[list[float]] = []
    for head_index, vocab in enumerate(vocabs.values()):
        counts = np.zeros(len(vocab), dtype=np.float64)
        for patient_id in patient_ids:
            target = labels[patient_id][head_index]
            if target >= 0:
                counts[target] += 1
        weights = np.ones(len(vocab), dtype=np.float64)
        observed = counts > 0
        weights[observed] = 1.0 / np.sqrt(counts[observed])
        if observed.any():
            weights[observed] /= weights[observed].mean()
        output.append([round(float(value), 8) for value in weights])
    return output


def link_sample(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    label_columns, labels = read_labels(data_root / "labels.csv")
    vocabs: dict[str, list[str]] = json.loads(
        (data_root / "head_vocabs.json").read_text(encoding="utf-8")
    )
    if len(label_columns) != len(vocabs):
        raise RuntimeError("labels.csv/head_vocabs.json head count mismatch")
    samples = source_samples(data_root)
    eligible = sorted(
        patient_id
        for patient_id in samples
        if patient_id in labels and any(value >= 0 for value in labels[patient_id])
    )
    excluded = sorted(set(samples).difference(eligible))
    assignments = assign_folds(labels, eligible, args.folds, args.seed)

    output_root.mkdir(parents=True)
    assignment_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for fold in range(args.folds):
        fold_root = output_root / f"fold{fold + 1}"
        for split in ("train", "val", "test"):
            (fold_root / split).mkdir(parents=True)
        val_ids = sorted(patient_id for patient_id in eligible if assignments[patient_id] == fold)
        train_ids = sorted(set(eligible).difference(val_ids))
        for patient_id in train_ids:
            link_sample(samples[patient_id], fold_root / "train" / samples[patient_id].name)
        for patient_id in val_ids:
            link_sample(samples[patient_id], fold_root / "val" / samples[patient_id].name)
            link_sample(samples[patient_id], fold_root / "test" / samples[patient_id].name)

        with (fold_root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", *label_columns])
            for patient_id in eligible:
                writer.writerow([patient_id, *labels[patient_id]])
        (fold_root / "head_vocabs.json").write_text(
            json.dumps(vocabs, indent=2) + "\n", encoding="utf-8"
        )
        weights = class_weights(train_ids, labels, vocabs)
        (fold_root / "class_weights.json").write_text(
            json.dumps(weights, indent=2) + "\n", encoding="utf-8"
        )
        audit = {
            "fold": fold + 1,
            "seed": args.seed,
            "train_cases": len(train_ids),
            "val_cases": len(val_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "class_weights": weights,
        }
        (fold_root / "fold_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        audits.append(audit)
        assignment_rows.extend(
            {"patient_id": patient_id, "fold": fold + 1, "role": "validation"}
            for patient_id in val_ids
        )

    with (output_root / "fold_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "fold", "role"])
        writer.writeheader()
        writer.writerows(sorted(assignment_rows, key=lambda row: row["patient_id"]))
    summary = {
        "source_data_root": str(data_root),
        "folds": args.folds,
        "seed": args.seed,
        "eligible_cases": len(eligible),
        "excluded_no_supervision": excluded,
        "fold_sizes": [audit["val_cases"] for audit in audits],
    }
    (output_root / "cv_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
