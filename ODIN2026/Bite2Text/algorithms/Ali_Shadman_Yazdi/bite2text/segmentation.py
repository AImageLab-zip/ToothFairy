"""TSGCNet tooth segmentation of a single-jaw intraoral scan.

    full-resolution scan
      -> quadric decimation to ~16k faces        (the resolution TSGCNet was trained at)
      -> TSGCNet: one of 17 classes per face     (0 = gingiva, 1..16 = teeth)
      -> absorb label islands smaller than 50 mm^2
      -> distance-weighted kNN back onto the full-resolution faces
      -> one mesh per tooth, keyed by FDI number
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import config
from .nets.tsgcnet import TSGCNet

__all__ = ["segment_faces", "split_teeth", "get_model", "clear_model_cache"]

_MODELS: dict = {}


def get_model(jaw: str, device=None) -> TSGCNet:
    """Load (and cache) the TSGCNet checkpoint for one jaw."""
    jaw = config.jaw_key(jaw)
    dev = config.resolve_device(device)
    key = (jaw, str(dev))
    if key in _MODELS:
        return _MODELS[key]

    ckpt = config.seg_checkpoint(jaw)
    if not ckpt.is_file():
        raise FileNotFoundError(f"TSGCNet checkpoint not found: {ckpt}")

    model = TSGCNet(in_channels=12, output_channels=config.NUM_CLASSES, k=32, device=dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(dev)

    # TSGCNet was trained at batch size 1, so its BatchNorm running statistics
    # are meaningless: in eval() mode it labels every face gingiva. It has to
    # normalise with batch statistics, i.e. stay in train mode. Dropout is not a
    # concern -- TSGCNet.forward discards the dropout outputs. Zero momentum
    # keeps the running buffers from drifting, so the cached model never changes.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0.0

    _MODELS[key] = model
    return model


def clear_model_cache() -> None:
    _MODELS.clear()


# ---------------------------------------------------------------------------
# Decimation + network input
# ---------------------------------------------------------------------------

def decimate(mesh: trimesh.Trimesh, target_faces: int = config.TARGET_FACES) -> trimesh.Trimesh:
    """Quadric decimation to ``target_faces`` triangles.

    TSGCNet has no hardcoded face count, so landing a few faces off target is harmless.
    """
    import fast_simplification

    vertices, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_count=int(target_faces),
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_face_features(mesh: trimesh.Trimesh) -> np.ndarray:
    """TSGCNet's 24-channel per-face input.

    Layout ``[v0 v1 v2 | centroid | n0 n1 n2 | face_normal]`` (9 + 3 + 9 + 3),
    followed by the normalisation used at training time. Two parts of it look
    odd and are kept deliberately, because the weights were trained with them:
    ``scale`` is the max over the whole vertex block *including* normals, and
    the per-axis statistics come from vertices that were centred but never
    divided by ``scale``.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    v = vertices[faces]                                  # (F, 3, 3)
    n = normals[faces]                                   # (F, 3, 3)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)

    feats = np.concatenate(
        [v.reshape(len(faces), 9), v.mean(axis=1), n.reshape(len(faces), 9), face_normals],
        axis=1,
    ).astype(np.float32)                                 # (F, 24)

    points = np.concatenate([vertices, normals], axis=1)  # (V, 6)

    centre = feats[:, 9:12].mean(axis=0)
    for s in (slice(0, 3), slice(3, 6), slice(6, 9)):
        feats[:, s] -= centre
    feats[:, 9:12] = (feats[:, 0:3] + feats[:, 3:6] + feats[:, 6:9]) / 3
    points = points.copy()
    points[:, :3] -= centre

    feats[:, :12] /= points.max()

    maxs = points[:, :3].max(axis=0)
    mins = points[:, :3].min(axis=0)
    means = points[:, :3].mean(axis=0)
    stds = points[:, :3].std(axis=0)
    nmeans = points[:, 3:].mean(axis=0)
    nstds = points[:, 3:].std(axis=0)
    nmeans_f = feats[:, 21:].mean(axis=0)
    nstds_f = feats[:, 21:].std(axis=0)

    for i in range(3):
        for off in (0, 3, 6):                            # the three vertices
            feats[:, i + off] = (feats[:, i + off] - means[i]) / stds[i]
        feats[:, i + 9] = (feats[:, i + 9] - mins[i]) / (maxs[i] - mins[i])
        for off in (12, 15, 18):                         # the three vertex normals
            feats[:, i + off] = (feats[:, i + off] - nmeans[i]) / nstds[i]
        feats[:, i + 21] = (feats[:, i + 21] - nmeans_f[i]) / nstds_f[i]

    return feats


def predict_face_labels(coarse: trimesh.Trimesh, jaw: str, device=None) -> np.ndarray:
    """One TSGCNet forward pass; ``(F,)`` class labels aligned with ``coarse.faces``."""
    model = get_model(jaw, device)
    dev = config.resolve_device(device)

    x = torch.from_numpy(build_face_features(coarse)).float()
    x = x.unsqueeze(0).to(dev).transpose(2, 1)           # (1, 24, F)
    with torch.no_grad():
        pred = model(x)
    labels = pred.contiguous().view(-1, config.NUM_CLASSES).argmax(dim=1)
    return labels.cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Cleanup + upsampling
# ---------------------------------------------------------------------------

def clean_islands(mesh: trimesh.Trimesh, labels: np.ndarray,
                  min_area: float = config.MIN_COMPONENT_AREA) -> np.ndarray:
    """Absorb connected same-label patches below ``min_area`` into their surroundings.

    A label whose *only* component is small is left alone, so a genuinely small
    tooth cannot be erased.
    """
    labels = np.asarray(labels).copy()
    n_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return labels

    a, b = adjacency[:, 0], adjacency[:, 1]
    same = labels[a] == labels[b]
    graph = coo_matrix((np.ones(same.sum(), dtype=np.int8), (a[same], b[same])),
                       shape=(n_faces, n_faces))
    n_comp, comp = connected_components(graph, directed=False)

    comp_area = np.bincount(comp, weights=mesh.area_faces, minlength=n_comp)
    comp_label = np.zeros(n_comp, dtype=labels.dtype)
    comp_label[comp] = labels
    comps_per_label = np.bincount(comp_label, minlength=config.NUM_CLASSES)

    small = np.flatnonzero(comp_area < min_area)
    if len(small) == 0:
        return labels

    # cross-label adjacency = votes for the replacement label
    cross = ~same
    ca, cb = a[cross], b[cross]
    vote_comp = np.concatenate([comp[ca], comp[cb]])
    vote_label = np.concatenate([labels[cb], labels[ca]])

    small_set = set(small.tolist())
    votes: dict = {}
    for c, lab in zip(vote_comp.tolist(), vote_label.tolist()):
        if c in small_set:
            votes.setdefault(c, []).append(lab)

    for c in small:
        if comps_per_label[comp_label[c]] <= 1:
            continue
        neighbours = votes.get(int(c))
        if not neighbours:
            continue
        counts = np.bincount(neighbours, minlength=config.NUM_CLASSES)
        labels[comp == c] = int(counts.argmax())

    return labels


def upsample_labels(coarse: trimesh.Trimesh, coarse_labels: np.ndarray,
                    fine: trimesh.Trimesh,
                    n_neighbors: int = config.UPSAMPLE_NEIGHBORS) -> np.ndarray:
    """Distance-weighted kNN vote over face barycentres (sklearn's ``weights="distance"`` rule)."""
    k = min(n_neighbors, len(coarse.faces))
    coarse_labels = np.ravel(np.asarray(coarse_labels)).astype(np.int64)

    tree = cKDTree(coarse.triangles_center)
    dist, idx = tree.query(fine.triangles_center, k=k, workers=-1)
    if k == 1:
        return coarse_labels[idx].astype(np.int64)

    with np.errstate(divide="ignore"):
        weights = 1.0 / dist
    # an exact hit gives infinite weight; such rows are decided by the coincident neighbours
    exact = np.isinf(weights)
    hit_rows = exact.any(axis=1)
    weights[hit_rows] = exact[hit_rows]

    n = len(idx)
    offsets = np.arange(n)[:, None] * config.NUM_CLASSES
    votes = np.bincount((offsets + coarse_labels[idx]).ravel(),
                        weights=weights.ravel(),
                        minlength=n * config.NUM_CLASSES)
    return votes.reshape(n, config.NUM_CLASSES).argmax(axis=1).astype(np.int64)


def segment_faces(mesh: trimesh.Trimesh, jaw: str, device=None) -> np.ndarray:
    """``(F,)`` class label for every face of the full-resolution scan."""
    coarse = decimate(mesh)
    coarse_labels = predict_face_labels(coarse, jaw, device=device)
    coarse_labels = clean_islands(coarse, coarse_labels)
    return upsample_labels(coarse, coarse_labels, mesh)


# ---------------------------------------------------------------------------
# Per-tooth meshes
# ---------------------------------------------------------------------------

def split_teeth(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: str, *,
                min_faces: int = 20) -> Dict[int, trimesh.Trimesh]:
    """``{fdi: Trimesh}`` for every detected tooth (gingiva excluded).

    Only the largest connected piece of each tooth is kept: landmark detection
    normalises a tooth by its own centroid and radius, so a stray patch on the
    far side of the arch would shift every predicted landmark.
    """
    jaw = config.jaw_key(jaw)
    labels = np.asarray(labels, dtype=np.int64)
    fdi_table = config.LABEL_TO_FDI[jaw]

    teeth: Dict[int, trimesh.Trimesh] = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        face_idx = np.flatnonzero(labels == label)
        if len(face_idx) < min_faces:
            continue

        tooth = mesh.submesh([face_idx], append=True, repair=False)
        parts = tooth.split(only_watertight=False)
        if len(parts) > 1:
            tooth = max(parts, key=lambda p: len(p.faces))
        if len(tooth.faces) < min_faces:
            continue
        teeth[int(fdi_table[label])] = tooth

    return dict(sorted(teeth.items()))
