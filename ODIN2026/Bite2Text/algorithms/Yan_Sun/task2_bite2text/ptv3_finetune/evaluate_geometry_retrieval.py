#!/usr/bin/env python3
"""Evaluate a compact, generalizable IOS-to-report nearest-neighbour baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def sample_paths(data_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for path in (data_root / split).glob("dental_*.npz"):
            patient_id = path.stem.removeprefix("dental_")
            output[patient_id] = path
    return output


def descriptor_from_coord(
    coord: np.ndarray, bins_1d: int = 32, bins_2d: int = 24
) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64).copy()
    if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) < 4 or not np.isfinite(coord).all():
        raise ValueError("Expected finite paired-jaw coordinates with shape (N, 3), N >= 4")
    if len(coord) % 2:
        raise ValueError("Expected equal paired-jaw point counts")
    coord -= coord.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(coord, axis=1).max()
    if radius <= 0:
        raise ValueError("Degenerate point cloud")
    coord /= radius
    jaws = np.split(coord, 2)
    features: list[np.ndarray] = []
    quantiles = np.asarray([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    for jaw in jaws:
        features.append(np.quantile(jaw, quantiles, axis=0).ravel())
        covariance = np.cov(jaw, rowvar=False)
        features.append(covariance.ravel())
        for axis in range(3):
            hist, _ = np.histogram(jaw[:, axis], bins=bins_1d, range=(-1.0, 1.0))
            features.append(hist.astype(np.float64) / len(jaw))
        for axes in ((0, 1), (0, 2), (1, 2)):
            hist, _, _ = np.histogram2d(
                jaw[:, axes[0]],
                jaw[:, axes[1]],
                bins=bins_2d,
                range=((-1.0, 1.0), (-1.0, 1.0)),
            )
            features.append((hist / len(jaw)).ravel())
    return np.concatenate(features).astype(np.float32)


def descriptor(path: Path, bins_1d: int = 32, bins_2d: int = 24) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        coord = np.asarray(payload["coord"], dtype=np.float64)
    return descriptor_from_coord(coord, bins_1d=bins_1d, bins_2d=bins_2d)


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+|[^\w\s]", text.lower()) if token.strip()]


def sentence_bleu4(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    precisions: list[float] = []
    for order in range(1, 5):
        predicted_ngrams = Counter(
            tuple(predicted[index : index + order])
            for index in range(max(len(predicted) - order + 1, 0))
        )
        expected_ngrams = Counter(
            tuple(expected[index : index + order])
            for index in range(max(len(expected) - order + 1, 0))
        )
        total = sum(predicted_ngrams.values())
        clipped = sum(
            min(count, expected_ngrams[ngram]) for ngram, count in predicted_ngrams.items()
        )
        # Add-one smoothing is used only for the offline comparison; exact
        # Grand Challenge metrics are rerun in the official evaluator image.
        precisions.append((clipped + 1.0) / (total + 1.0))
    brevity = 1.0 if len(predicted) >= len(expected) else math.exp(1.0 - len(expected) / len(predicted))
    return float(brevity * math.exp(sum(math.log(value) for value in precisions) / 4.0))


def meteor_lite(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    used: set[int] = set()
    indices: list[int] = []
    for token in predicted:
        index = next((i for i, value in enumerate(expected) if value == token and i not in used), None)
        if index is not None:
            used.add(index)
            indices.append(index)
    matches = len(indices)
    if not matches:
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(expected)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)
    chunks = 1 + sum(current != previous + 1 for previous, current in zip(indices, indices[1:]))
    return float((1.0 - 0.5 * (chunks / matches) ** 3) * f_mean)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-split", choices=("val", "test", "val_test", "all"), default="val_test")
    parser.add_argument("--database-split", choices=("train", "all"), default="train")
    parser.add_argument("--leave-one-out", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    samples = sample_paths(args.data_root.resolve())
    records: dict[str, dict[str, Any]] = {}
    for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("target_report_path") and record["patient_id"] in samples:
            records[record["patient_id"]] = record

    reports = {
        patient_id: (args.raw_data_root.resolve() / record["target_report_path"]).read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        for patient_id, record in records.items()
    }
    patient_ids = sorted(records)
    matrix = np.stack([descriptor(samples[patient_id]) for patient_id in patient_ids])
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    matrix = (matrix - mean) / scale
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}

    if args.database_split == "all":
        database_ids = patient_ids
    else:
        database_ids = [patient_id for patient_id in patient_ids if records[patient_id]["split"] == "train"]
    if args.query_split == "all":
        query_ids = patient_ids
    elif args.query_split == "val_test":
        query_ids = [patient_id for patient_id in patient_ids if records[patient_id]["split"] in {"val", "test"}]
    else:
        query_ids = [patient_id for patient_id in patient_ids if records[patient_id]["split"] == args.query_split]

    database_indices = np.asarray([index_by_id[patient_id] for patient_id in database_ids])
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        query_index = index_by_id[query_id]
        similarities = matrix[database_indices] @ matrix[query_index]
        order = np.argsort(-similarities)
        retrieved_id = next(
            database_ids[index]
            for index in order
            if not (args.leave_one_out and database_ids[index] == query_id)
        )
        prediction = reports[retrieved_id]
        reference = reports[query_id]
        rows.append(
            {
                "patient_id": query_id,
                "retrieved_patient_id": retrieved_id,
                "cosine_similarity": float(matrix[index_by_id[retrieved_id]] @ matrix[query_index]),
                "bleu_4_lite": sentence_bleu4(prediction, reference),
                "meteor_lite": meteor_lite(prediction, reference),
                "prediction": prediction,
                "reference": reference,
            }
        )

    metrics = {
        "database_cases": len(database_ids),
        "query_cases": len(query_ids),
        "query_split": args.query_split,
        "database_split": args.database_split,
        "leave_one_out": args.leave_one_out,
        "bleu_4_lite_mean": float(np.mean([row["bleu_4_lite"] for row in rows])),
        "bleu_4_lite_std": float(np.std([row["bleu_4_lite"] for row in rows])),
        "meteor_lite_mean": float(np.mean([row["meteor_lite"] for row in rows])),
        "meteor_lite_std": float(np.std([row["meteor_lite"] for row in rows])),
    }
    (output_dir / "retrieval_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "retrieval_index.npz",
        patient_ids=np.asarray(patient_ids),
        descriptors=matrix.astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
