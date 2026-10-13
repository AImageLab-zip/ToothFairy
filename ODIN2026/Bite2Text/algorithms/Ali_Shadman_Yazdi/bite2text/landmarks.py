"""Per-tooth crown landmarks with CLIK's PointMLP detectors.

For every segmented tooth (third molars excepted):

  1. the whole jaw is PCA-canonicalised once (a single tooth has no reliable
     front/back or occlusal cue),
  2. each tooth is centred and scaled to the unit ball,
  3. 2048 points are drawn by farthest-point sampling,
  4. the PointMLP head for the tooth's type (incisor / cuspid / premolar /
     molar) scores every sampled point per landmark; the arg-max point wins,
  5. predictions are mapped back to the scan's own millimetre frame.

All teeth of one type share a forward pass, so a jaw costs four passes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import trimesh
from sklearn.decomposition import PCA

from . import config
from .nets.point_mlp import PointMLP

__all__ = ["get_landmarks", "get_models", "clear_model_cache"]

Point = Tuple[float, float, float]

_MODELS: dict = {}


def get_models(device=None) -> Dict[str, PointMLP]:
    """Load (and cache) the four CLIK crown-landmark networks."""
    dev = config.resolve_device(device)
    if str(dev) in _MODELS:
        return _MODELS[str(dev)]

    models = {}
    for kind, landmark_ids in config.CLIK_LANDMARK_IDS.items():
        path = config.clik_checkpoint(kind)
        if not path.is_file():
            raise FileNotFoundError(
                f"CLIK checkpoint not found: {path}\n"
                "Download it with `python scripts/download_weights.py`.")
        net = PointMLP(num_classes=len(landmark_ids), points=config.FPS_SAMPLES, embed_dim=128)
        net.load_state_dict(torch.load(path, map_location="cpu")["model"])
        models[kind] = net.to(dev).eval()

    _MODELS[str(dev)] = models
    return models


def clear_model_cache() -> None:
    _MODELS.clear()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def canonical_transform(mesh: trimesh.Trimesh, pca_max_verts: int = 20_000,
                        seed: int = 0) -> np.ndarray:
    """4x4 transform taking an IOS jaw into the frame the CLIK models expect.

    Centre, orient by PCA, then resolve the two 180-degree ambiguities: the
    occlusal side faces +Z (mean face normal) and the arch is wider at the back
    than at the front. ``mesh`` is modified in place.
    """
    total = np.eye(4)

    to_origin = trimesh.transformations.translation_matrix(-mesh.vertices.mean(axis=0))
    mesh.apply_transform(to_origin)
    total = to_origin @ total

    verts = mesh.vertices
    if len(verts) > pca_max_verts:
        rng = np.random.default_rng(seed)
        verts = verts[rng.choice(len(verts), pca_max_verts, replace=False)]
    v0, v1, v2 = PCA(n_components=3).fit(verts).components_

    rotation = np.vstack([v1, v0, v2])
    if np.linalg.det(rotation) < 0:
        rotation[2] = -rotation[2]
    transform_pca = np.eye(4)
    transform_pca[:3, :3] = rotation
    mesh.apply_transform(transform_pca)
    total = transform_pca @ total

    if np.mean(mesh.face_normals[:, 2]) < 0:
        flip_z = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        mesh.apply_transform(flip_z)
        total = flip_z @ total

    x_mid = (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2
    front = mesh.vertices[mesh.vertices[:, 0] > x_mid]
    back = mesh.vertices[mesh.vertices[:, 0] < x_mid]
    width_front = np.ptp(front[:, 1]) if len(front) else 0
    width_back = np.ptp(back[:, 1]) if len(back) else 0
    if width_front > width_back:
        flip_x = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        mesh.apply_transform(flip_x)
        total = flip_x @ total

    final = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh.apply_transform(final)
    return final @ total


def farthest_point_sample_batched(xyz_list, npoint: int, device=None,
                                  seed: Optional[int] = None) -> np.ndarray:
    """Farthest-point sampling for many clouds in one padded call -> ``(B, npoint)``.

    Padded slots start at distance 0 and can never win the arg-max, so every
    returned index is valid for its own cloud.
    """
    dev = config.resolve_device(device)
    batch = len(xyz_list)
    lengths = [len(x) for x in xyz_list]
    max_n = max(lengths)

    padded = np.zeros((batch, max_n, 3), dtype=np.float32)
    for i, x in enumerate(xyz_list):
        padded[i, :len(x)] = x
    xyz = torch.from_numpy(padded).to(dev)

    distance = torch.zeros(batch, max_n, device=dev)
    for i, n in enumerate(lengths):
        distance[i, :n] = 1e10

    rng = np.random.default_rng(seed)
    farthest = torch.tensor([rng.integers(0, n) for n in lengths], dtype=torch.long, device=dev)
    batch_idx = torch.arange(batch, dtype=torch.long, device=dev)
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=dev)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.argmax(dim=-1)

    return centroids.cpu().numpy()


def _to_scan_frame(point, inverse: np.ndarray) -> Point:
    p = (inverse @ np.append(np.asarray(point, dtype=np.float64), 1.0))[:3]
    return (float(p[0]), float(p[1]), float(p[2]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_landmarks(teeth: Mapping[int, trimesh.Trimesh], device=None,
                  seed: Optional[int] = 0) -> Dict[int, Dict[str, Point]]:
    """``{fdi: {landmark name: (x, y, z)}}`` in the scan's own coordinate frame.

    Every returned tooth has a ``Centroid``; third molars are skipped (no model).
    """
    if not teeth:
        return {}
    dev = config.resolve_device(device)
    models = get_models(dev)

    combined = trimesh.util.concatenate([m.copy() for m in teeth.values()])
    to_canonical = canonical_transform(combined, seed=seed if seed is not None else 0)
    inverse = np.linalg.inv(to_canonical)

    prepared = []
    for fdi, mesh in teeth.items():
        universal = config.FDI_TO_UNIVERSAL.get(int(fdi))
        kind = config.UNIVERSAL_TO_CLIK.get(universal)
        if kind is None:                                   # third molar
            continue
        mesh = mesh.copy().apply_transform(to_canonical)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) < 4:
            continue

        center = vertices.mean(axis=0)
        scale = float(np.max(np.sqrt(((vertices - center) ** 2).sum(axis=1))))
        prepared.append({
            "fdi": int(fdi), "universal": universal, "kind": kind,
            "points": ((vertices - center) / scale).astype(np.float32),
            "normals": np.asarray(mesh.vertex_normals, dtype=np.float32),
            "center": center, "scale": scale,
        })
    if not prepared:
        return {}

    sampled = farthest_point_sample_batched(
        [t["points"] for t in prepared], config.FPS_SAMPLES, device=dev, seed=seed)
    for tooth, idx in zip(prepared, sampled):
        tooth["sample_points"] = tooth["points"][idx]
        tooth["sample_normals"] = tooth["normals"][idx]

    by_kind = defaultdict(list)
    for tooth in prepared:
        by_kind[tooth["kind"]].append(tooth)

    landmarks: Dict[int, Dict[str, Point]] = {}
    for kind, group in by_kind.items():
        pts = torch.from_numpy(np.stack([t["sample_points"] for t in group])).float().to(dev).permute(0, 2, 1)
        nrm = torch.from_numpy(np.stack([t["sample_normals"] for t in group])).float().to(dev).permute(0, 2, 1)
        tid = torch.tensor([t["universal"] for t in group], dtype=torch.long, device=dev)

        with torch.no_grad():
            pred = models[kind](pts, nrm, tid)             # (B, S, n_landmarks)
        pred_idx = pred.argmax(dim=1).cpu().numpy()        # (B, n_landmarks)

        for tooth, idx in zip(group, pred_idx):
            entry = {"Centroid": _to_scan_frame(tooth["center"], inverse)}
            for i, landmark_id in enumerate(config.CLIK_LANDMARK_IDS[kind]):
                coord = tooth["sample_points"][idx[i]] * tooth["scale"] + tooth["center"]
                entry[config.LANDMARK_NAMES[landmark_id]] = _to_scan_frame(coord, inverse)
            landmarks[tooth["fdi"]] = entry

    return dict(sorted(landmarks.items()))
