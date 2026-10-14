#!/usr/bin/env python3
"""Build Bite2Text landmark-only point sets in an FDI-derived RAS frame."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION = "2.0"
POINT_CLASSES = ("Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint")
MOLAR_FDI = {16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48}
CLASS_MINIMUMS = {"Mesial": 1, "Distal": 1, "FacialPoint": 1, "OuterPoint": 1, "InnerPoint": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-arch-axis-angle", type=float, default=55.0)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"degenerate {name} direction")
    return vector / norm


def robust_direction(vectors: list[np.ndarray], name: str) -> np.ndarray:
    if not vectors:
        raise ValueError(f"no vectors support the {name} direction")
    normalized = np.stack([unit(vector, name) for vector in vectors])
    direction = unit(np.median(normalized, axis=0), name)
    aligned = normalized[normalized @ direction > 0.25]
    if len(aligned):
        direction = unit(np.mean(aligned, axis=0), name)
    return direction


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(first, "angle"), unit(second, "angle")), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def fdi_quadrant(fdi: int) -> int:
    return int(fdi) // 10


def tooth_number(fdi: int) -> int:
    return int(fdi) % 10


def fdi_by_instance(segmentation_path: Path) -> dict[int, int]:
    data = json.loads(segmentation_path.read_text(encoding="utf-8"))
    instances = data.get("instances")
    labels = data.get("labels")
    if not isinstance(instances, list) or not isinstance(labels, list) or len(instances) != len(labels):
        raise ValueError(f"malformed segmentation: {segmentation_path}")
    grouped: dict[int, Counter[int]] = defaultdict(Counter)
    for instance, label in zip(instances, labels):
        instance_id = int(instance)
        fdi = int(label)
        if instance_id >= 0 and 11 <= fdi <= 48 and 1 <= tooth_number(fdi) <= 8:
            grouped[instance_id][fdi] += 1
    return {instance: counts.most_common(1)[0][0] for instance, counts in grouped.items()}


def select_scan(
    kpt_path: Path,
    segmentation_path: Path,
    arch: str,
    score_threshold: float,
) -> list[dict[str, object]]:
    landmark_data = json.loads(kpt_path.read_text(encoding="utf-8"))
    landmarks = landmark_data.get("objects")
    if not isinstance(landmarks, list):
        raise ValueError(f"malformed landmarks: {kpt_path}")
    instance_to_fdi = fdi_by_instance(segmentation_path)
    valid_quadrants = {1, 2} if arch == "upper" else {3, 4}
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for landmark in landmarks:
        class_name = landmark.get("class")
        instance_id = int(landmark.get("instance_id", -1))
        coord = landmark.get("coord", [])
        if class_name not in POINT_CLASSES or instance_id not in instance_to_fdi or len(coord) != 3:
            continue
        if not np.all(np.isfinite(np.asarray(coord, dtype=np.float64))):
            continue
        if fdi_quadrant(instance_to_fdi[instance_id]) not in valid_quadrants:
            continue
        grouped[instance_id].append(landmark)

    selected: list[dict[str, object]] = []
    for instance_id, instance_landmarks in grouped.items():
        fdi = instance_to_fdi[instance_id]
        by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
        for landmark in instance_landmarks:
            by_class[str(landmark["class"])].append(landmark)
        minimums = dict(CLASS_MINIMUMS)
        minimums["Cusp"] = 2 if fdi in MOLAR_FDI else 0
        for class_name in POINT_CLASSES:
            candidates = sorted(
                by_class[class_name], key=lambda item: float(item.get("score", 0.0)), reverse=True
            )
            chosen = [item for item in candidates if float(item.get("score", 0.0)) >= score_threshold]
            required = minimums.get(class_name, 0)
            if len(chosen) < required:
                chosen.extend(candidates[len(chosen) : required])
            for item in chosen:
                selected.append(
                    {
                        "class": class_name,
                        "coord": [float(value) for value in item["coord"]],
                        "confidence": float(item.get("score", 0.0)),
                        "fdi": int(fdi),
                        "arch": arch,
                        "modality": 0.0,
                    }
                )
    if not selected:
        raise ValueError(f"no usable landmarks selected from {kpt_path}")
    return selected


def tooth_centers(objects: list[dict[str, object]]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for item in objects:
        grouped[int(item["fdi"])].append(np.asarray(item["coord"], dtype=np.float64))
    return {fdi: np.median(np.stack(points), axis=0) for fdi, points in grouped.items() if points}


def center_of(centers: dict[int, np.ndarray], predicate) -> np.ndarray:
    values = [coord for fdi, coord in centers.items() if predicate(fdi)]
    if not values:
        raise ValueError("empty anatomical tooth group")
    return np.median(np.stack(values), axis=0)


def right_vectors(centers: dict[int, np.ndarray], arches: tuple[str, ...] = ("upper", "lower")) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    if "upper" in arches:
        vectors.extend(centers[10 + digit] - centers[20 + digit] for digit in range(1, 9) if 10 + digit in centers and 20 + digit in centers)
    if "lower" in arches:
        vectors.extend(centers[40 + digit] - centers[30 + digit] for digit in range(1, 9) if 40 + digit in centers and 30 + digit in centers)
    if not vectors:
        right = [coord for fdi, coord in centers.items() if fdi_quadrant(fdi) in {1, 4}]
        left = [coord for fdi, coord in centers.items() if fdi_quadrant(fdi) in {2, 3}]
        if right and left:
            vectors.append(np.median(np.stack(right), axis=0) - np.median(np.stack(left), axis=0))
    return vectors


def anterior_vectors(centers: dict[int, np.ndarray], arches: tuple[str, ...] = ("upper", "lower")) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    arch_quadrants = {"upper": {1, 2}, "lower": {3, 4}}
    for arch in arches:
        quadrants = arch_quadrants[arch]
        for side_quadrants in ({quadrant for quadrant in quadrants}, *({quadrant} for quadrant in quadrants)):
            anterior = [
                coord for fdi, coord in centers.items()
                if fdi_quadrant(fdi) in side_quadrants and tooth_number(fdi) <= 3
            ]
            posterior = [
                coord for fdi, coord in centers.items()
                if fdi_quadrant(fdi) in side_quadrants and tooth_number(fdi) >= 5
            ]
            if anterior and posterior:
                vectors.append(np.median(np.stack(anterior), axis=0) - np.median(np.stack(posterior), axis=0))
    return vectors


def closest_right_handed_basis(right: np.ndarray, anterior: np.ndarray, superior: np.ndarray) -> np.ndarray:
    anatomical = np.stack([unit(right, "right"), unit(anterior, "anterior"), unit(superior, "superior")])
    left, _, right_t = np.linalg.svd(anatomical)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    basis = left @ correction @ right_t
    if np.linalg.det(basis) < 0.999:
        raise ValueError("failed to construct a right-handed RAS basis")
    return basis


def canonicalize(
    patient: str,
    objects: list[dict[str, object]],
    max_arch_axis_angle: float,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    centers = tooth_centers(objects)
    upper_fdi = sorted(fdi for fdi in centers if fdi_quadrant(fdi) in {1, 2})
    lower_fdi = sorted(fdi for fdi in centers if fdi_quadrant(fdi) in {3, 4})
    if len(upper_fdi) < 4 or len(lower_fdi) < 4:
        raise ValueError(f"insufficient FDI teeth: upper={len(upper_fdi)}, lower={len(lower_fdi)}")

    right_support = right_vectors(centers)
    anterior_support = anterior_vectors(centers)
    right = robust_direction(right_support, "right")
    anterior = robust_direction(anterior_support, "anterior")
    upper_center = np.median(np.stack([centers[fdi] for fdi in upper_fdi]), axis=0)
    lower_center = np.median(np.stack([centers[fdi] for fdi in lower_fdi]), axis=0)
    superior = unit(upper_center - lower_center, "superior")

    arch_angles: dict[str, dict[str, float]] = {}
    for arch in ("upper", "lower"):
        arch_right = right_vectors(centers, (arch,))
        arch_anterior = anterior_vectors(centers, (arch,))
        if arch_right and arch_anterior:
            right_angle = angle_degrees(robust_direction(arch_right, f"{arch} right"), right)
            anterior_angle = angle_degrees(robust_direction(arch_anterior, f"{arch} anterior"), anterior)
            arch_angles[arch] = {"right_degrees": right_angle, "anterior_degrees": anterior_angle}
            if max(right_angle, anterior_angle) > max_arch_axis_angle:
                raise ValueError(
                    f"{arch} frame disagrees with pair frame: right={right_angle:.1f}, anterior={anterior_angle:.1f} deg"
                )

    basis = closest_right_handed_basis(right, anterior, superior)
    origin = np.median(np.stack(list(centers.values())), axis=0)
    transformed_centers = {fdi: basis @ (coord - origin) for fdi, coord in centers.items()}
    right_x = center_of(transformed_centers, lambda fdi: fdi_quadrant(fdi) in {1, 4})[0]
    left_x = center_of(transformed_centers, lambda fdi: fdi_quadrant(fdi) in {2, 3})[0]
    anterior_y = center_of(transformed_centers, lambda fdi: tooth_number(fdi) <= 3)[1]
    posterior_y = center_of(transformed_centers, lambda fdi: tooth_number(fdi) >= 5)[1]
    upper_z = center_of(transformed_centers, lambda fdi: fdi_quadrant(fdi) in {1, 2})[2]
    lower_z = center_of(transformed_centers, lambda fdi: fdi_quadrant(fdi) in {3, 4})[2]
    margins = {
        "right_minus_left_mm": float(right_x - left_x),
        "anterior_minus_posterior_mm": float(anterior_y - posterior_y),
        "upper_minus_lower_mm": float(upper_z - lower_z),
    }
    if any(value <= 0.5 for value in margins.values()):
        raise ValueError(f"failed RAS direction validation: {margins}")

    transformed: list[dict[str, object]] = []
    for item in objects:
        result = dict(item)
        result["coord"] = [float(value) for value in basis @ (np.asarray(item["coord"]) - origin)]
        transformed.append(result)

    homogeneous = np.eye(4)
    homogeneous[:3, :3] = basis
    homogeneous[:3, 3] = -basis @ origin
    transform = {
        "method": "FDI-derived pairwise RAS orthogonal Procrustes",
        "origin_original": origin.tolist(),
        "rotation_original_to_ras": basis.tolist(),
        "homogeneous_original_to_ras": homogeneous.tolist(),
        "determinant": float(np.linalg.det(basis)),
    }
    quality = {
        "patient": patient,
        "upper_fdi": upper_fdi,
        "lower_fdi": lower_fdi,
        "tooth_count": len(centers),
        "landmark_count": len(objects),
        "right_pair_support": len(right_support),
        "anterior_group_support": len(anterior_support),
        "arch_axis_angles": arch_angles,
        "direction_margins": margins,
        "orthonormal_error": float(np.linalg.norm(basis @ basis.T - np.eye(3))),
    }
    return transformed, transform, quality


def reusable_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("version") == VERSION and payload.get("coordinate_system") == "RAS" and bool(payload.get("objects"))


def build_case(args: argparse.Namespace, patient: str, destination: Path) -> tuple[int, dict[str, object]]:
    objects: list[dict[str, object]] = []
    for arch in ("lower", "upper"):
        stem = f"{patient}_{arch}"
        segmentation_path = args.prediction_dir / f"{stem}.json"
        kpt_path = args.prediction_dir / f"{stem}__kpt.json"
        if not segmentation_path.is_file() or not kpt_path.is_file():
            raise FileNotFoundError(f"missing {arch} 3DTeethLand output")
        objects.extend(select_scan(kpt_path, segmentation_path, arch, args.score_threshold))
    transformed, transform, quality = canonicalize(patient, objects, args.max_arch_axis_angle)
    payload = {
        "version": VERSION,
        "description": "Bite2Text FDI-canonical RAS landmark-only point set",
        "patient": patient,
        "coordinate_system": "RAS",
        "units": "mm",
        "objects": transformed,
        "transform": transform,
        "quality": quality,
    }
    atomic_write_json(destination, payload)
    return len(transformed), quality


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_cases = args.output_root / "all_cases"
    all_cases.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / ".build.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        print(f"Waiting for point-set build lock: {lock_path}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("Acquired point-set build lock", flush=True)
        patients = sorted(path.name for path in args.data_root.glob("F*") if path.is_dir())
        skipped: list[dict[str, str]] = []
        counts: dict[str, int] = {}
        rebuilt = 0
        reused = 0
        for index, patient in enumerate(patients, start=1):
            destination = all_cases / f"{patient}.json"
            if reusable_output(destination):
                payload = json.loads(destination.read_text(encoding="utf-8"))
                counts[patient] = len(payload["objects"])
                reused += 1
                continue
            try:
                count, _ = build_case(args, patient, destination)
                counts[patient] = count
                rebuilt += 1
            except Exception as error:
                skipped.append({"patient": patient, "reason": f"{type(error).__name__}: {error}"})
            if index % 50 == 0 or index == len(patients):
                print(
                    f"Point-set progress {index}/{len(patients)}: valid={len(counts)}, skipped={len(skipped)}",
                    flush=True,
                )

        completed_shards = [
            index for index in range(args.shard_count)
            if (args.prediction_dir / f"completion_manifest_shard_{index}_of_{args.shard_count}.json").is_file()
        ]
        extraction_complete = len(completed_shards) == args.shard_count
        if not extraction_complete:
            status = "partial"
        elif skipped:
            status = "complete_with_skips"
        else:
            status = "complete"
        count_values = list(counts.values())
        manifest = {
            "version": VERSION,
            "status": status,
            "created_unix_time": time.time(),
            "patient_directory_count": len(patients),
            "valid_case_count": len(counts),
            "skipped_case_count": len(skipped),
            "rebuilt_case_count_this_pass": rebuilt,
            "reused_case_count_this_pass": reused,
            "score_threshold": args.score_threshold,
            "coordinate_system": "RAS",
            "canonicalization": "One shared FDI-derived orthogonal transform is applied to both registered arches.",
            "upper_lower_concatenation": "Lower and upper landmarks are concatenated after the shared RAS transform.",
            "point_schema": ["class", "coord", "confidence", "fdi", "arch", "modality"],
            "completed_shards": completed_shards,
            "expected_shards": args.shard_count,
            "landmarks_per_case": {
                "minimum": min(count_values) if count_values else None,
                "maximum": max(count_values) if count_values else None,
                "mean": float(np.mean(count_values)) if count_values else None,
            },
        }
        atomic_write_json(args.output_root / "manifest.json", manifest)
        atomic_write_json(args.output_root / "skipped_cases.json", skipped)
        for marker in (args.output_root / "_SUCCESS", args.output_root / "_COMPLETE_WITH_SKIPS"):
            marker.unlink(missing_ok=True)
        if extraction_complete:
            marker_name = "_SUCCESS" if not skipped else "_COMPLETE_WITH_SKIPS"
            (args.output_root / marker_name).write_text(
                f"status={status}\nvalid_cases={len(counts)}\nskipped_cases={len(skipped)}\n",
                encoding="utf-8",
            )
        print(json.dumps(manifest, indent=2), flush=True)
        if skipped:
            print(f"Skipped cases are recorded in {args.output_root / 'skipped_cases.json'}", flush=True)


if __name__ == "__main__":
    main()
