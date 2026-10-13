#!/usr/bin/env python3
"""Geometry-retrieval Bite2Text inference for the ODIN 2026 interface.

The existing v3 image supplies the validated Grand Challenge input loader,
paired IOS-Normalizer, mesh orientation correction, and surface sampler.  This
entrypoint replaces the seven-head classifier with a standardized geometric
descriptor and nearest-neighbour report retrieval.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from inference import (  # provided by the validated v3 base image
    env_path,
    input_files,
    postcorrect_triangles,
    run_ios_normalizer,
    sampled_coordinates,
)


def descriptor_from_coord(
    coord: np.ndarray, bins_1d: int = 32, bins_2d: int = 24
) -> np.ndarray:
    """Build the same 3,720-dimensional descriptor used for the index."""
    points = np.asarray(coord, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) % 2:
        raise ValueError(f"Expected an even Nx3 paired-jaw cloud, got {points.shape}")
    points -= points.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(points, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Degenerate point cloud")
    points /= radius

    features: list[np.ndarray] = []
    quantiles = np.asarray([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    for jaw in np.split(points, 2):
        features.append(np.quantile(jaw, quantiles, axis=0).ravel())
        features.append(np.cov(jaw, rowvar=False).ravel())
        for axis in range(3):
            histogram, _ = np.histogram(jaw[:, axis], bins=bins_1d, range=(-1.0, 1.0))
            features.append(histogram.astype(np.float64) / len(jaw))
        for first, second in ((0, 1), (0, 2), (1, 2)):
            histogram, _, _ = np.histogram2d(
                jaw[:, first],
                jaw[:, second],
                bins=bins_2d,
                range=((-1.0, 1.0), (-1.0, 1.0)),
            )
            features.append((histogram / len(jaw)).ravel())
    return np.concatenate(features).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_assets(
    index_path: Path, reports_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not index_path.is_file() or not reports_path.is_file():
        raise RuntimeError(f"Missing retrieval assets: {index_path}, {reports_path}")
    metadata = json.loads(reports_path.read_text(encoding="utf-8"))
    with np.load(index_path, allow_pickle=False) as payload:
        patient_ids = np.asarray(payload["patient_ids"]).astype(str)
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
        mean = np.asarray(payload["mean"], dtype=np.float32).reshape(-1)
        scale = np.asarray(payload["scale"], dtype=np.float32).reshape(-1)

    reports = metadata.get("reports")
    metadata_ids = metadata.get("patient_ids")
    if metadata.get("index_sha256") != sha256_file(index_path):
        raise RuntimeError("Retrieval index checksum does not match metadata")
    if metadata_ids != patient_ids.tolist():
        raise RuntimeError("Retrieval report order does not match index patient order")
    if not isinstance(reports, list) or len(reports) != len(patient_ids):
        raise RuntimeError("Retrieval report count does not match index")
    if descriptors.shape != (len(patient_ids), len(mean)) or scale.shape != mean.shape:
        raise RuntimeError("Invalid retrieval descriptor statistics")
    if not np.isfinite(descriptors).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("Non-finite retrieval assets")
    return patient_ids, descriptors, mean, scale, [str(value).strip() for value in reports]


def retrieve(
    descriptor: np.ndarray,
    patient_ids: np.ndarray,
    database: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    reports: list[str],
) -> tuple[str, str, float, float]:
    if descriptor.shape != mean.shape:
        raise RuntimeError(f"Descriptor shape mismatch: {descriptor.shape} != {mean.shape}")
    query = (descriptor - mean) / scale
    norm = float(np.linalg.norm(query))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Degenerate standardized query descriptor")
    query /= norm
    similarities = database @ query
    if len(similarities) < 2 or not np.isfinite(similarities).all():
        raise RuntimeError("Invalid retrieval similarity vector")
    order = np.argsort(-similarities)
    best, second = int(order[0]), int(order[1])
    report = reports[best]
    if not report:
        raise RuntimeError(f"Retrieved empty report for {patient_ids[best]}")
    return (
        str(patient_ids[best]),
        report,
        float(similarities[best]),
        float(similarities[best] - similarities[second]),
    )


def run() -> int:
    input_path = env_path("BITE2TEXT_INPUT_PATH", "/input")
    output_path = env_path("BITE2TEXT_OUTPUT_PATH", "/output")
    index_path = env_path("BITE2TEXT_RETRIEVAL_INDEX", "/opt/ml/model/retrieval_index.npz")
    reports_path = env_path(
        "BITE2TEXT_RETRIEVAL_REPORTS", "/opt/ml/model/retrieval_reports.json"
    )
    patient_ids, database, mean, scale, reports = load_assets(index_path, reports_path)
    upper_path, lower_path = input_files(input_path)

    with tempfile.TemporaryDirectory(prefix="bite2text_retrieval_") as temporary:
        normalized_upper, normalized_lower, normalizer_summary = run_ios_normalizer(
            upper_path, lower_path, Path(temporary)
        )
        upper_triangles, lower_triangles, orientation = postcorrect_triangles(
            normalized_upper, normalized_lower
        )
        coord = sampled_coordinates(upper_triangles, lower_triangles)
        descriptor = descriptor_from_coord(coord)
        retrieved_id, report, similarity, margin = retrieve(
            descriptor, patient_ids, database, mean, scale, reports
        )

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "diagnostic-imaging-report.json"
    output_file.write_text(
        json.dumps({"report": report}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": {"upper": str(upper_path), "lower": str(lower_path)},
                "normalizer": normalizer_summary,
                "orientation": orientation,
                "sampled_points": int(coord.shape[0]),
                "descriptor_dimensions": int(descriptor.shape[0]),
                "retrieved_patient_id": retrieved_id,
                "cosine_similarity": round(similarity, 6),
                "top1_margin": round(margin, 6),
                "output": str(output_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

