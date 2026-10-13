#!/usr/bin/env python3
"""Convert normalized Bite2Text IOS pairs into a PTv3-ready point dataset.

The input manifest is the patient-level, strict-consensus manifest produced by
``mesh_baseline/prepare_manifest.py``.  Each jaw is sampled uniformly over STL
triangle area and the two jaws are concatenated without independent centering,
so their occlusal relationship is preserved.  PTv3 performs the final joint
coordinate normalization in its data transform.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_INDEX = -1
DATASET_VERSION = "bite2text-ptv3-surface-v1"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stl_triangles(path: Path) -> np.ndarray:
    """Read binary or ASCII STL triangles with no third-party mesh package."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + 50 * triangle_count == len(payload):
            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vectors", "<f4", (3, 3)), ("attribute", "<u2")]
            )
            mesh = np.frombuffer(payload, dtype=dtype, offset=84, count=triangle_count)
            return np.asarray(mesh["vectors"], dtype=np.float32).copy()
    text = payload.decode("utf-8", errors="ignore")
    values = re.findall(r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if not values or len(values) % 3:
        raise ValueError(f"Unsupported or empty STL: {path}")
    return np.asarray(values, dtype=np.float32).reshape(-1, 3, 3)


def sample_surface(triangles: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    valid = np.isfinite(areas) & (areas > np.finfo(np.float32).eps)
    if not valid.any():
        raise ValueError("Mesh has no finite non-degenerate triangle")
    triangles = triangles[valid]
    probabilities = areas[valid].astype(np.float64)
    probabilities /= probabilities.sum()
    selected = triangles[rng.choice(len(triangles), size=count, replace=True, p=probabilities)]
    first = rng.random(count, dtype=np.float32)
    second = rng.random(count, dtype=np.float32)
    reflected = first + second > 1.0
    first[reflected] = 1.0 - first[reflected]
    second[reflected] = 1.0 - second[reflected]
    points = selected[:, 0] + first[:, None] * (selected[:, 1] - selected[:, 0])
    points += second[:, None] * (selected[:, 2] - selected[:, 0])
    return np.asarray(points, dtype=np.float32)


def patient_seed(base_seed: int, patient_id: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(patient_id.encode("utf-8")).digest()[:8], "little")
    return (base_seed + suffix) % (2**32)


def process_case(job: dict[str, Any]) -> dict[str, Any]:
    patient_id = job["patient_id"]
    rng = np.random.default_rng(patient_seed(job["seed"], patient_id))
    upper = sample_surface(load_stl_triangles(Path(job["upper"])), job["points_per_jaw"], rng)
    lower = sample_surface(load_stl_triangles(Path(job["lower"])), job["points_per_jaw"], rng)
    coord = np.concatenate([upper, lower], axis=0)
    if coord.shape != (2 * job["points_per_jaw"], 3) or not np.isfinite(coord).all():
        raise ValueError(f"Invalid sampled coordinates: shape={coord.shape}")
    extent = coord.max(axis=0) - coord.min(axis=0)
    if float(np.linalg.norm(extent)) < 1e-3:
        raise ValueError("Degenerate paired scan extent")
    output = Path(job["output"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, coord=coord)
    os.replace(temporary, output)
    return {
        "patient_id": patient_id,
        "split": job["split"],
        "output": str(output),
        "points": int(coord.shape[0]),
        "extent": [round(float(value), 5) for value in extent],
    }


def class_weights(records: list[dict[str, Any]], vocabs: dict[str, list[str]]) -> list[list[float]]:
    train = [record for record in records if record["split"] == "train"]
    output: list[list[float]] = []
    for head, vocab in vocabs.items():
        counts = np.zeros(len(vocab), dtype=np.float64)
        for record in train:
            target = int(record["targets"].get(head, IGNORE_INDEX))
            if target >= 0:
                counts[target] += 1
        weights = np.ones(len(vocab), dtype=np.float64)
        observed = counts > 0
        weights[observed] = 1.0 / np.sqrt(counts[observed])
        if observed.any():
            weights[observed] /= weights[observed].mean()
        output.append([round(float(value), 8) for value in weights])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points-per-jaw", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-patient", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0, help="Debug-only cap; zero uses all cases.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    vocab_path = args.head_vocabs.resolve()
    normalized_root = args.normalized_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")
    if args.points_per_jaw < 1024:
        raise SystemExit("--points-per-jaw must be at least 1024")
    records = read_manifest(manifest_path)
    vocabs: dict[str, list[str]] = json.loads(vocab_path.read_text(encoding="utf-8"))
    excluded = set(args.exclude_patient)
    records = [record for record in records if record["patient_id"] not in excluded]
    if args.max_cases:
        records = records[: args.max_cases]

    output_root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir()

    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        patient_id = record["patient_id"]
        case_root = normalized_root / patient_id
        upper = case_root / "ios_upper.stl"
        lower = case_root / "ios_lower.stl"
        if not upper.is_file() or not lower.is_file():
            skipped.append({"patient_id": patient_id, "reason": "missing_normalized_ios"})
            continue
        jobs.append(
            {
                "patient_id": patient_id,
                "split": record["split"],
                "upper": str(upper),
                "lower": str(lower),
                "output": str(output_root / record["split"] / f"dental_{patient_id}.npz"),
                "points_per_jaw": args.points_per_jaw,
                "seed": args.seed,
            }
        )

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # retain every failed patient in the audit
                failed.append({"patient_id": job["patient_id"], "reason": f"{type(exc).__name__}: {exc}"})
            if index == 1 or index % 25 == 0 or index == len(jobs):
                print(json.dumps({"processed": index, "total": len(jobs), "failed": len(failed)}), flush=True)

    successful_ids = {row["patient_id"] for row in completed}
    successful_records = [record for record in records if record["patient_id"] in successful_ids]
    head_names = list(vocabs)
    with (output_root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", *[f"label_{i}" for i in range(len(head_names))]])
        writer.writeheader()
        for record in sorted(successful_records, key=lambda row: row["patient_id"]):
            writer.writerow(
                {
                    "patient_id": record["patient_id"],
                    **{
                        f"label_{i}": int(record["targets"].get(head, IGNORE_INDEX))
                        if int(record["targets"].get(head, IGNORE_INDEX)) >= 0
                        else IGNORE_INDEX
                        for i, head in enumerate(head_names)
                    },
                }
            )
    (output_root / "head_vocabs.json").write_text(json.dumps(vocabs, indent=2) + "\n", encoding="utf-8")
    weights = class_weights(successful_records, vocabs)
    (output_root / "class_weights.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")

    split_counts = Counter(record["split"] for record in successful_records)
    label_counts: dict[str, dict[str, int]] = {}
    for head, vocab in vocabs.items():
        counts: Counter[str] = Counter()
        for record in successful_records:
            target = int(record["targets"].get(head, IGNORE_INDEX))
            if target >= 0:
                counts[vocab[target]] += 1
        label_counts[head] = dict(sorted(counts.items()))
    audit = {
        "dataset_version": DATASET_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "normalized_root": str(normalized_root),
        "points_per_jaw": args.points_per_jaw,
        "points_per_case": 2 * args.points_per_jaw,
        "seed": args.seed,
        "excluded_patients": sorted(excluded),
        "records": len(successful_records),
        "split_counts": dict(sorted(split_counts.items())),
        "head_names": head_names,
        "head_vocabs": vocabs,
        "head_label_counts": label_counts,
        "class_weights": weights,
        "skipped": skipped,
        "failed": failed,
        "outputs": sorted(completed, key=lambda row: row["patient_id"]),
    }
    (output_root / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(successful_records), "split_counts": dict(split_counts), "failed": len(failed)}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
