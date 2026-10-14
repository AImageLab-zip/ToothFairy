from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image


SOURCE_IMAGES = Path("/datasets/aihub/views")
SOURCE_LABELS = Path("/datasets/aihub/mask_annotations")
PROJECT_ROOT = Path("/outputs/yolo")
DATASET_ROOT = PROJECT_ROOT / "dataset_fdi24_seed42"
FDI_CLASSES = (
    11, 12, 13, 14, 15, 16,
    21, 22, 23, 24, 25, 26,
    31, 32, 33, 34, 35, 36,
    41, 42, 43, 44, 45, 46,
)
FDI_TO_CLASS = {fdi: index for index, fdi in enumerate(FDI_CLASSES)}


def simplify_polygon(points: list[list[float]], max_points: int = 256) -> np.ndarray:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) < 3:
        return np.empty((0, 2), dtype=np.float32)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, perimeter * 0.001)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(simplified) > max_points:
        epsilon *= 1.25
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return simplified


def make_split(stems_by_view: dict[str, list[str]], seed: int, val_fraction: float):
    assignment = {}
    for offset, view in enumerate(sorted(stems_by_view)):
        stems = sorted(stems_by_view[view])
        random.Random(seed + offset * 1009).shuffle(stems)
        val_count = max(1, round(len(stems) * val_fraction))
        val_stems = set(stems[:val_count])
        for stem in stems:
            assignment[stem] = "val" if stem in val_stems else "train"
    return assignment


def write_data_yaml() -> None:
    lines = [
        f"path: {DATASET_ROOT}",
        "train: images/train",
        "val: images/val",
        "nc: 24",
        "names:",
    ]
    lines.extend(f"  {index}: '{fdi}'" for index, fdi in enumerate(FDI_CLASSES))
    (DATASET_ROOT / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global SOURCE_IMAGES, SOURCE_LABELS, PROJECT_ROOT, DATASET_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    SOURCE_IMAGES = args.images.resolve()
    SOURCE_LABELS = args.labels.resolve()
    PROJECT_ROOT = args.project_root.resolve()
    DATASET_ROOT = PROJECT_ROOT / "dataset_fdi24_seed42"

    if not SOURCE_IMAGES.is_dir() or not SOURCE_LABELS.is_dir():
        raise FileNotFoundError("Source image or annotation directory is missing")
    if DATASET_ROOT.exists():
        raise FileExistsError(f"Use a new project root; refusing to delete {DATASET_ROOT}")
    for split in ("train", "val"):
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_by_stem = {
        path.stem: path
        for path in SOURCE_IMAGES.rglob("*.png")
        if "checkpoint" not in path.stem.lower()
    }
    annotations = {}
    stems_by_view: dict[str, list[str]] = defaultdict(list)
    archive_counts = Counter()
    for archive_path in sorted(SOURCE_LABELS.glob("*.zip")):
        view = archive_path.stem.split(".", 1)[-1]
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() != ".json":
                    continue
                record = json.loads(archive.read(info).decode("utf-8-sig"))
                stem = Path(info.filename).stem
                if stem in annotations:
                    raise RuntimeError(f"Duplicate annotation stem: {stem}")
                if stem not in source_by_stem:
                    raise RuntimeError(f"Annotation has no source image: {stem}")
                if Path(record["image_filepath"]).stem != stem:
                    raise RuntimeError(f"Annotation path mismatch: {info.filename}")
                annotations[stem] = record
                stems_by_view[view].append(stem)
                archive_counts[view] += 1
    if set(annotations) != set(source_by_stem):
        missing = sorted(set(source_by_stem) - set(annotations))
        extra = sorted(set(annotations) - set(source_by_stem))
        raise RuntimeError(f"Source/annotation mismatch; missing={missing[:10]}, extra={extra[:10]}")

    assignment = make_split(stems_by_view, args.seed, args.val_fraction)
    split_images = Counter()
    split_instances = Counter()
    class_instances = {"train": Counter(), "val": Counter()}
    view_split = Counter()
    original_points = 0
    simplified_points = 0
    point_histogram = Counter()
    skipped = []

    for stem in sorted(annotations):
        split = assignment[stem]
        source = source_by_stem[stem]
        destination = DATASET_ROOT / "images" / split / source.name
        os.symlink(source, destination)
        with Image.open(source) as image:
            width, height = image.size

        lines = []
        seen_fdi = set()
        for tooth in annotations[stem]["tooth"]:
            fdi = int(tooth["teeth_num"])
            if fdi not in FDI_TO_CLASS:
                skipped.append({"stem": stem, "reason": "unknown_fdi", "fdi": fdi})
                continue
            if fdi in seen_fdi:
                raise RuntimeError(f"Duplicate FDI {fdi} in {stem}")
            seen_fdi.add(fdi)
            original_points += len(tooth["segmentation"])
            polygon = simplify_polygon(tooth["segmentation"])
            if len(polygon) < 3 or abs(float(cv2.contourArea(polygon))) < 4.0:
                skipped.append({"stem": stem, "reason": "degenerate_polygon", "fdi": fdi})
                continue
            polygon[:, 0] = np.clip(polygon[:, 0] / width, 0.0, 1.0)
            polygon[:, 1] = np.clip(polygon[:, 1] / height, 0.0, 1.0)
            coordinates = " ".join(f"{value:.6f}" for value in polygon.reshape(-1))
            lines.append(f"{FDI_TO_CLASS[fdi]} {coordinates}")
            simplified_points += len(polygon)
            point_histogram[len(polygon)] += 1
            split_instances[split] += 1
            class_instances[split][str(fdi)] += 1
        if not lines:
            raise RuntimeError(f"No valid tooth polygons for {stem}")
        (DATASET_ROOT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )
        split_images[split] += 1
        view = source.parent.name
        view_split[f"{view}/{split}"] += 1

    for split in ("train", "val"):
        missing_classes = [
            str(fdi) for fdi in FDI_CLASSES if class_instances[split][str(fdi)] == 0
        ]
        if missing_classes:
            raise RuntimeError(f"Classes missing from {split}: {missing_classes}")
    write_data_yaml()

    manifest = {
        "format": "Ultralytics YOLO instance segmentation polygons",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "classes": {str(index): fdi for index, fdi in enumerate(FDI_CLASSES)},
        "source_images": str(SOURCE_IMAGES),
        "source_labels": str(SOURCE_LABELS),
        "archive_counts": dict(sorted(archive_counts.items())),
        "split_images": dict(sorted(split_images.items())),
        "split_instances": dict(sorted(split_instances.items())),
        "view_split": dict(sorted(view_split.items())),
        "class_instances": {
            split: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for split, counts in class_instances.items()
        },
        "polygon_points": {
            "original": original_points,
            "simplified": simplified_points,
            "minimum": min(point_histogram),
            "maximum": max(point_histogram),
            "mean": simplified_points / sum(point_histogram.values()),
        },
        "skipped": skipped,
    }
    (DATASET_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
