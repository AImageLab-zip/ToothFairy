"""STL surface sampling, pair normalization, and shared manifest utilities."""

from __future__ import annotations

import json
import random
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_stl_triangles(path: Path) -> np.ndarray:
    """Load STL triangles as float32 [triangle, vertex, xyz], binary or ASCII."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + 50 * triangle_count == len(payload):
            dtype = np.dtype([("normal", "<f4", (3,)), ("vectors", "<f4", (3, 3)), ("attribute", "<u2")])
            mesh = np.frombuffer(payload, dtype=dtype, offset=84, count=triangle_count)
            return np.asarray(mesh["vectors"], dtype=np.float32).copy()
    text = payload.decode("utf-8", errors="ignore")
    vertices = re.findall(r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Unsupported or malformed STL: {path}")
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3, 3)


def sample_surface(triangles: np.ndarray, num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted triangle sampling with barycentric coordinates and normals."""
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    doubled_area = np.linalg.norm(normals, axis=1)
    valid = doubled_area > 1e-12
    if not np.any(valid):
        raise ValueError("Mesh has no non-degenerate triangles")
    triangles = triangles[valid]
    normals = normals[valid]
    doubled_area = doubled_area[valid]
    normals = normals / doubled_area[:, None]
    probabilities = doubled_area / doubled_area.sum()
    indices = rng.choice(len(triangles), size=num_points, replace=True, p=probabilities)
    chosen = triangles[indices]
    u = rng.random(num_points, dtype=np.float32)
    v = rng.random(num_points, dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    return points.astype(np.float32), normals[indices].astype(np.float32)


def normalize_pair(upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([upper, lower], axis=0)
    center = combined.mean(axis=0, keepdims=True)
    scale = max(float(np.linalg.norm(combined - center, axis=1).max()), 1e-6)
    return ((upper - center) / scale).astype(np.float32), ((lower - center) / scale).astype(np.float32)


class SurfacePairDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], data_root: Path, num_points: int, seed: int, augment: bool) -> None:
        self.records = records
        self.data_root = data_root
        self.num_points = num_points
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = np.random.default_rng(self.seed + index)
        upper_xyz, upper_normals = sample_surface(load_stl_triangles(self.data_root / record["upper_stl"]), self.num_points, rng)
        lower_xyz, lower_normals = sample_surface(load_stl_triangles(self.data_root / record["lower_stl"]), self.num_points, rng)
        upper_xyz, lower_xyz = normalize_pair(upper_xyz, lower_xyz)
        if self.augment:
            theta = rng.uniform(-np.pi, np.pi)
            rotation = np.asarray(
                [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            upper_xyz = upper_xyz @ rotation.T
            lower_xyz = lower_xyz @ rotation.T
            upper_normals = upper_normals @ rotation.T
            lower_normals = lower_normals @ rotation.T
            upper_xyz += rng.normal(0.0, 0.003, size=upper_xyz.shape).astype(np.float32)
            lower_xyz += rng.normal(0.0, 0.003, size=lower_xyz.shape).astype(np.float32)
        return {
            "upper_xyz": torch.from_numpy(upper_xyz),
            "upper_normals": torch.from_numpy(upper_normals),
            "lower_xyz": torch.from_numpy(lower_xyz),
            "lower_normals": torch.from_numpy(lower_normals),
            "targets": {head: torch.tensor(value, dtype=torch.long) for head, value in record["targets"].items()},
            "patient_id": record["patient_id"],
        }
