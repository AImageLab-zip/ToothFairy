#!/usr/bin/env python3
"""Prepare raw pre-treatment STL pairs for five independent classifiers."""

from __future__ import annotations

import argparse
import json
import re
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


TASKS = {
    "dentition": ("Deciduous", "Mixed", "Permanent"),
    "crowding": ("Dental_spacing", "Crowding_below_4", "Crowding_above_4"),
    "malocclusion": ("ClassI", "ClassII", "ClassIII", "Special"),
    "overbite": (
        "Normal_overbite",
        "Deep_overbite",
        "Negative_overbite",
        "Anterior_open_bite",
        "Edge-to-edge_bite",
    ),
    "overjet": ("Normal_overjet", "Deep_overjet", "Negative_overjet", "Edge-to-edge"),
}
LABEL_FILES = {
    "dentition": "Dentition.txt",
    "crowding": "Crowding.txt",
    "malocclusion": "Malocclusion.txt",
    "overbite": "Overbite.txt",
    "overjet": "Overjet.txt",
}
STL_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def parse_id_sets(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    parsed = {}
    for label, body in re.findall(r'"([^"]+)"\s*:\s*\{([^}]*)\}', text):
        parsed[label] = re.findall(r"\b\d{4}\b", body)
    if not parsed:
        raise RuntimeError(f"No labels found in {path}")
    return parsed


def invert_labels(path: Path, class_names: tuple[str, ...]) -> dict[str, int]:
    groups = parse_id_sets(path)
    missing_classes = set(class_names) - set(groups)
    if missing_classes:
        raise RuntimeError(f"Missing classes in {path}: {sorted(missing_classes)}")
    inverse = {}
    for class_index, class_name in enumerate(class_names):
        for case_id in groups[class_name]:
            previous = inverse.get(case_id)
            if previous is not None and previous != class_index:
                raise RuntimeError(f"Conflicting labels for {case_id} in {path}")
            inverse[case_id] = class_index
    return inverse


def read_binary_stl(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    expected_size = 84 + triangle_count * STL_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Not a valid binary STL: {path}")
    records = np.fromfile(path, dtype=STL_DTYPE, count=triangle_count, offset=84)
    if len(records) != triangle_count:
        raise RuntimeError(f"Truncated STL: {path}")
    return records["vertices"]


def sample_stl_surface(path: Path, point_count: int, rng: np.random.Generator):
    triangles = read_binary_stl(path).astype(np.float64, copy=False)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_1, edge_2)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    triangles = triangles[valid]
    cross = cross[valid]
    double_area = double_area[valid]
    if not len(triangles):
        raise RuntimeError(f"No valid triangles in {path}")
    face_index = rng.choice(len(triangles), size=point_count, replace=True, p=double_area / double_area.sum())
    selected = triangles[face_index]
    root_u = np.sqrt(rng.random(point_count))
    v = rng.random(point_count)
    points = (
        (1.0 - root_u)[:, None] * selected[:, 0]
        + (root_u * (1.0 - v))[:, None] * selected[:, 1]
        + (root_u * v)[:, None] * selected[:, 2]
    )
    normals = cross[face_index]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def normalize_vector(vector: np.ndarray, name: str):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-6:
        raise RuntimeError(f"Cannot determine RAS {name} axis")
    return vector / length


def load_tooth_centroids(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    centroids = {}
    for fdi_text, tooth in payload["segmentation"].items():
        vertices = np.asarray(tooth["vertices"], dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices):
            centroids[int(fdi_text)] = vertices.mean(axis=0)
    return centroids


def compute_joint_ras(case_root: Path):
    """Compute one RAS frame for the already occlusally registered arch pair."""
    upper = load_tooth_centroids(case_root / "ori" / "U_Ori.json")
    lower = load_tooth_centroids(case_root / "ori" / "L_Ori.json")
    all_centroids = {**upper, **lower}
    right = [value for fdi, value in all_centroids.items() if fdi // 10 in (1, 4)]
    left = [value for fdi, value in all_centroids.items() if fdi // 10 in (2, 3)]
    anterior = [value for fdi, value in all_centroids.items() if fdi % 10 in (1, 2, 3)]
    posterior = [value for fdi, value in all_centroids.items() if fdi % 10 in (5, 6, 7, 8)]
    if min(len(upper), len(lower), len(right), len(left), len(anterior), len(posterior)) < 2:
        raise RuntimeError(f"Insufficient FDI instances to determine RAS for {case_root.name}")

    right_hint = np.mean(right, axis=0) - np.mean(left, axis=0)
    anterior_hint = np.mean(anterior, axis=0) - np.mean(posterior, axis=0)
    superior_hint = np.mean(list(upper.values()), axis=0) - np.mean(list(lower.values()), axis=0)

    superior_axis = normalize_vector(superior_hint, "superior")
    anterior_axis = anterior_hint - np.dot(anterior_hint, superior_axis) * superior_axis
    anterior_axis = normalize_vector(anterior_axis, "anterior")
    right_axis = normalize_vector(np.cross(anterior_axis, superior_axis), "right")
    if np.dot(right_axis, right_hint) < 0:
        raise RuntimeError(f"FDI-derived anatomical axes are not right-handed for {case_root.name}")
    anterior_axis = normalize_vector(np.cross(superior_axis, right_axis), "anterior")
    basis = np.stack([right_axis, anterior_axis, superior_axis], axis=1)
    if np.linalg.det(basis) < 0.999:
        raise RuntimeError(f"Invalid RAS basis for {case_root.name}: det={np.linalg.det(basis)}")
    origin = np.mean(list(all_centroids.values()), axis=0)
    quality = {
        "right_alignment": float(np.dot(right_axis, normalize_vector(right_hint, "right hint"))),
        "anterior_alignment": float(np.dot(anterior_axis, normalize_vector(anterior_hint, "anterior hint"))),
        "superior_alignment": float(np.dot(superior_axis, normalize_vector(superior_hint, "superior hint"))),
    }
    return origin.astype(np.float32), basis.astype(np.float32), quality


def prepare_case(job):
    case_id, data_root, point_root, points_per_arch, seed = job
    destination = point_root / f"{case_id}.npz"
    if destination.is_file():
        with np.load(destination) as payload:
            if payload["coord"].shape == (points_per_arch * 2, 3) and "ras_basis" in payload:
                return case_id, "cached"
    case_root = data_root / case_id
    ras_origin, ras_basis, ras_quality = compute_joint_ras(case_root)
    arrays = []
    rng = np.random.default_rng(seed + int(case_id))
    for arch_index, filename in enumerate(("U_Ori.stl", "L_Ori.stl")):
        coord, normal = sample_stl_surface(case_root / "ori" / filename, points_per_arch, rng)
        coord = (coord - ras_origin) @ ras_basis
        normal = normal @ ras_basis
        arch = np.zeros((points_per_arch, 2), dtype=np.float32)
        arch[:, arch_index] = 1.0
        arrays.append((coord, normal, arch))
    coord = np.concatenate([item[0] for item in arrays], axis=0)
    normal = np.concatenate([item[1] for item in arrays], axis=0)
    arch = np.concatenate([item[2] for item in arrays], axis=0)
    order = rng.permutation(len(coord))
    np.savez(
        destination,
        coord=coord[order],
        normal=normal[order],
        arch=arch[order],
        ras_origin=ras_origin,
        ras_basis=ras_basis,
        ras_quality=np.asarray(
            [ras_quality["right_alignment"], ras_quality["anterior_alignment"], ras_quality["superior_alignment"]],
            dtype=np.float32,
        ),
    )
    return case_id, "written"


def stratification_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for task_name, class_names in TASKS.items():
        labels = frame[f"{task_name}_index"].to_numpy(dtype=np.int64)
        for class_index in range(len(class_names)):
            columns.append((labels == class_index).astype(np.int8))
        columns.append((labels == -1).astype(np.int8))
    return np.stack(columns, axis=1)


def split_official_train(frame: pd.DataFrame, val_fraction: float, seed: int):
    official_train = frame[frame["official_split"] == "train"].reset_index(drop=True)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_index, val_index = next(splitter.split(np.zeros(len(official_train)), stratification_matrix(official_train)))
    train_ids = set(official_train.iloc[train_index]["case_id"])
    val_ids = set(official_train.iloc[val_index]["case_id"])
    if train_ids & val_ids or len(train_ids | val_ids) != len(official_train):
        raise RuntimeError("Invalid train/validation partition")
    return train_ids, val_ids


def make_summary(frame: pd.DataFrame):
    summary = {"split_counts": frame["split"].value_counts().sort_index().to_dict(), "tasks": {}}
    for task_name, class_names in TASKS.items():
        task_summary = {}
        for split in ("train", "val", "test"):
            subset = frame[frame["split"] == split]
            counts = subset[f"{task_name}_index"].value_counts().to_dict()
            task_summary[split] = {
                class_name: int(counts.get(class_index, 0))
                for class_index, class_name in enumerate(class_names)
            }
            task_summary[split]["unknown"] = int(counts.get(-1, 0))
        summary["tasks"][task_name] = task_summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points-per-arch", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    point_root = args.output_root / "points"
    point_root.mkdir(exist_ok=True)
    split_groups = parse_id_sets(args.label_root / "train-test-split.txt")
    official_train = set(split_groups["train"])
    official_test = set(split_groups["test"])
    if official_train & official_test:
        raise RuntimeError("Official train/test IDs overlap")
    case_ids = sorted(path.name for path in args.dataset_root.iterdir() if path.is_dir())
    actual_cases = set(case_ids)
    split_ids_without_stl = sorted((official_train | official_test) - actual_cases)
    stl_ids_missing_from_split = sorted(actual_cases - (official_train | official_test))
    if actual_cases != official_train | official_test:
        # The released split lists train IDs 0687/1101, but those STL folders do
        # not exist; the archive instead contains 0736/0818. Keep the published
        # 250-case test set untouched and place unmatched real STL cases in train.
        if set(split_ids_without_stl) & official_test:
            raise RuntimeError("The official test split references missing STL data")
        official_train = (official_train & actual_cases) | set(stl_ids_missing_from_split)
        official_test &= actual_cases
    if official_train | official_test != actual_cases or official_train & official_test:
        raise RuntimeError("Could not reconcile extracted cases with the official split")
    if len(official_train) != 810 or len(official_test) != 250:
        raise RuntimeError(f"Unexpected reconciled split sizes: {len(official_train)}/{len(official_test)}")

    inverse = {
        task_name: invert_labels(args.label_root / LABEL_FILES[task_name], class_names)
        for task_name, class_names in TASKS.items()
    }
    rows = []
    for case_id in case_ids:
        row = {
            "case_id": case_id,
            "official_split": "train" if case_id in official_train else "test",
        }
        for task_name, class_names in TASKS.items():
            class_index = inverse[task_name].get(case_id, -1)
            row[f"{task_name}_index"] = class_index
            row[f"{task_name}_label"] = class_names[class_index] if class_index >= 0 else "unknown"
        rows.append(row)
    frame = pd.DataFrame(rows)
    train_ids, val_ids = split_official_train(frame, args.val_fraction, args.seed)
    frame["split"] = np.where(
        frame["official_split"].eq("test"),
        "test",
        np.where(frame["case_id"].isin(val_ids), "val", "train"),
    )
    assert len(train_ids) == 648 and len(val_ids) == 162 and len(official_test) == 250

    jobs = [
        (case_id, args.dataset_root, point_root, args.points_per_arch, args.seed)
        for case_id in case_ids
    ]
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for case_id, status in executor.map(prepare_case, jobs, chunksize=2):
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Prepared {completed}/{len(jobs)} raw STL pairs ({case_id}: {status})", flush=True)

    frame.to_csv(args.output_root / "manifest.csv", index=False)
    summary = make_summary(frame)
    summary.update(
        {
            "seed": args.seed,
            "val_fraction_within_official_train": args.val_fraction,
            "points_per_arch": args.points_per_arch,
            "input": "jointly RAS-aligned U_Ori.stl + L_Ori.stl surface points, normals, and upper/lower flag",
            "ras_alignment": "one shared rigid transform per occlusally registered upper/lower pair; FDI tooth centroids determine Right/Anterior/Superior axes",
            "model_features_excluded": ["landmarks", "FDI", "tooth instance labels"],
            "released_split_reconciliation": {
                "split_ids_without_stl": split_ids_without_stl,
                "stl_ids_missing_from_split": stl_ids_missing_from_split,
                "policy": "official test unchanged; unmatched real STL cases assigned to train because missing-STL split IDs were train IDs",
            },
            "class_mapping": {name: {label: i for i, label in enumerate(labels)} for name, labels in TASKS.items()},
        }
    )
    (args.output_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
