#!/usr/bin/env python3
"""Shared geometry descriptor and retrieval-asset validation for Bite2Text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def descriptor_from_coord(
    coord: np.ndarray, bins_1d: int = 32, bins_2d: int = 24
) -> np.ndarray:
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
    if metadata.get("index_sha256") != sha256_file(index_path):
        raise RuntimeError("Retrieval index checksum does not match metadata")
    if metadata.get("patient_ids") != patient_ids.tolist():
        raise RuntimeError("Retrieval report order does not match index")
    if not isinstance(reports, list) or len(reports) != len(patient_ids):
        raise RuntimeError("Retrieval report count does not match index")
    if descriptors.shape != (len(patient_ids), len(mean)) or scale.shape != mean.shape:
        raise RuntimeError("Invalid retrieval descriptor statistics")
    if not np.isfinite(descriptors).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("Non-finite retrieval assets")
    return patient_ids, descriptors, mean, scale, [str(value).strip() for value in reports]
