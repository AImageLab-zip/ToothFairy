#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate tooth-segmentation quality for the outputs of predict1.py.

This script intentionally separates three concepts:

1. Core teeth required by the current crowding geometry:
   upper: 26..21, 11..16
   lower: 36..31, 41..46

2. Second molars:
   upper: 27, 17
   lower: 37, 47
   Their absence is recorded but is not treated as a segmentation failure.

3. Third molars / wisdom teeth:
   upper: 28, 18
   lower: 38, 48
   Their absence is informational only and does not lower the QC score.

Expected predict1.py outputs
----------------------------
work/output/
    case_ios_upper.json
    case_ios_upper.npy
    case_ios_upper_two_pass_data.npz
    case_ios_upper_pass2_full_raw.ply

The final JSON is used for FDI tooth semantics.
The NPZ is used for confidence, ROI, alignment and contamination metadata.
The PLY is used for connected components, surface triangles and geometry checks.

Outputs
-------
work/segmentation_qc/
    segmentation_qc_summary.csv
    segmentation_qc_tooth_metrics.csv
    retry_cases.txt
    retry_manifest.csv
    cases/<case>_qc.json

Important
---------
A missing core tooth triggers "retry_once", because a second segmentation run may
recover a randomly missed label. However, a tooth that remains missing across several
independent runs may be truly absent. This script does not claim that every missing
core tooth is a segmentation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError as exc:
    raise ImportError(
        "This script requires scipy. Install it with: pip install scipy"
    ) from exc


CASE_PATTERN = re.compile(
    r"^(?P<case>(?P<patient>[A-Za-z0-9_-]+)_ios_(?P<arch>upper|lower))\.json$",
    re.IGNORECASE,
)

ARCH_CONFIG = {
    "upper": {
        "full_order": [28, 27, 26, 25, 24, 23, 22, 21,
                       11, 12, 13, 14, 15, 16, 17, 18],
        "core_order": [26, 25, 24, 23, 22, 21,
                       11, 12, 13, 14, 15, 16],
        "second_molars": [27, 17],
        "wisdom_teeth": [28, 18],
    },
    "lower": {
        "full_order": [38, 37, 36, 35, 34, 33, 32, 31,
                       41, 42, 43, 44, 45, 46, 47, 48],
        "core_order": [36, 35, 34, 33, 32, 31,
                       41, 42, 43, 44, 45, 46],
        "second_molars": [37, 47],
        "wisdom_teeth": [38, 48],
    },
}


@dataclass
class ToothMetric:
    case: str
    patient_id: str
    arch: str
    tooth: int
    tooth_group: str
    present: bool
    vertex_count: int
    vertex_ratio: float
    complete_face_count: int
    component_count: int
    largest_component_vertices: int
    largest_component_fraction: float
    bbox_diag_mm: float
    centroid_x: float
    centroid_y: float
    centroid_z: float
    mean_confidence: float
    median_confidence: float
    p10_confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDI-aware segmentation QC for predict1.py outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/output"),
        help="Directory containing predict1.py output files",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("work/input"),
        help="Directory containing the original input STL files",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("work/segmentation_qc"),
        help="QC report output directory",
    )
    parser.add_argument(
        "--min-core-vertices",
        type=int,
        default=20,
        help="Core tooth below this vertex count is unusable",
    )
    parser.add_argument(
        "--review-core-vertices",
        type=int,
        default=80,
        help="Core tooth below this count is reviewed",
    )
    parser.add_argument(
        "--min-largest-component-fraction",
        type=float,
        default=0.65,
        help="Core tooth below this largest-component fraction is fragmented",
    )
    parser.add_argument(
        "--severe-component-fraction",
        type=float,
        default=0.50,
        help="Core tooth below this fraction triggers retry_once",
    )
    parser.add_argument(
        "--max-core-bbox-diag-mm",
        type=float,
        default=30.0,
        help="Core tooth above this bounding-box diagonal is suspicious",
    )
    parser.add_argument(
        "--severe-core-bbox-diag-mm",
        type=float,
        default=38.0,
        help="Core tooth above this diagonal triggers retry_once",
    )
    parser.add_argument(
        "--min-tooth-confidence",
        type=float,
        default=0.35,
        help="Mean pass-2 confidence below this value is reviewed",
    )
    parser.add_argument(
        "--min-detected-classes",
        type=int,
        default=6,
        help="predict1 detected class count below this is suspicious",
    )
    parser.add_argument(
        "--acceptable-score",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--retry-score",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--copy-retry-input",
        action="store_true",
        help="Copy flagged input STL files into qc-dir/retry_input",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-case QC JSON reports",
    )
    return parser.parse_args()


def safe_scalar(npz: Any, key: str, default: Any) -> Any:
    if npz is None or key not in npz.files:
        return default

    value = np.asarray(npz[key])

    if value.size == 0:
        return default

    item = value.reshape(-1)[0]

    if isinstance(default, bool):
        return bool(item)
    if isinstance(default, int):
        return int(item)
    if isinstance(default, float):
        return float(item)

    return item.item() if hasattr(item, "item") else item


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(
        path,
        force="mesh",
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError("PLY scene contains no geometry")
        loaded = trimesh.util.concatenate(geometries)

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(loaded)}")

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("Mesh is empty")

    return loaded


def build_vertex_graph(
    vertex_count: int,
    faces: np.ndarray,
) -> csr_matrix:
    faces = np.asarray(faces, dtype=np.int64)

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.uint8)

    return csr_matrix(
        (data, (row, col)),
        shape=(vertex_count, vertex_count),
    )


def load_full_confidence(
    npz_path: Path,
    vertex_count: int,
) -> tuple[np.ndarray, bool]:
    """
    Map pass2 ROI maximum probability back to the full mesh.

    Vertices outside the final ROI stay NaN because pass2 did not classify them.
    """
    confidence = np.full(
        vertex_count,
        np.nan,
        dtype=np.float32,
    )

    if not npz_path.is_file():
        return confidence, False

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"pass2_roi_probs", "roi_full_indices"}

        if not required.issubset(set(data.files)):
            return confidence, False

        probabilities = np.asarray(
            data["pass2_roi_probs"],
            dtype=np.float32,
        )
        indices = np.asarray(
            data["roi_full_indices"],
            dtype=np.int64,
        ).reshape(-1)

        if (
            probabilities.ndim != 2
            or len(probabilities) != len(indices)
        ):
            return confidence, False

        valid = (
            (indices >= 0)
            & (indices < vertex_count)
        )

        confidence[indices[valid]] = np.max(
            probabilities[valid],
            axis=1,
        )

    return confidence, True


def tooth_group(arch: str, tooth: int) -> str:
    config = ARCH_CONFIG[arch]

    if tooth in config["core_order"]:
        return "core"
    if tooth in config["second_molars"]:
        return "second_molar"
    if tooth in config["wisdom_teeth"]:
        return "wisdom"
    return "unexpected"


def tooth_metric(
    case: str,
    patient_id: str,
    arch: str,
    tooth: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    graph: csr_matrix,
    full_confidence: np.ndarray,
) -> ToothMetric:
    mask = labels == tooth
    indices = np.flatnonzero(mask)
    vertex_count = int(len(indices))

    if vertex_count == 0:
        return ToothMetric(
            case=case,
            patient_id=patient_id,
            arch=arch,
            tooth=tooth,
            tooth_group=tooth_group(arch, tooth),
            present=False,
            vertex_count=0,
            vertex_ratio=0.0,
            complete_face_count=0,
            component_count=0,
            largest_component_vertices=0,
            largest_component_fraction=0.0,
            bbox_diag_mm=float("nan"),
            centroid_x=float("nan"),
            centroid_y=float("nan"),
            centroid_z=float("nan"),
            mean_confidence=float("nan"),
            median_confidence=float("nan"),
            p10_confidence=float("nan"),
        )

    subgraph = graph[indices][:, indices]
    component_count, component_labels = connected_components(
        subgraph,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(
        component_labels,
        minlength=component_count,
    )
    largest = int(sizes.max())
    largest_fraction = largest / max(vertex_count, 1)

    points = vertices[indices]
    bbox_diag = float(
        np.linalg.norm(
            np.ptp(points, axis=0)
        )
    )
    centroid = np.median(points, axis=0)

    complete_face_count = int(
        np.all(mask[faces], axis=1).sum()
    )

    tooth_confidence = full_confidence[indices]
    tooth_confidence = tooth_confidence[
        np.isfinite(tooth_confidence)
    ]

    if len(tooth_confidence) > 0:
        mean_confidence = float(
            np.mean(tooth_confidence)
        )
        median_confidence = float(
            np.median(tooth_confidence)
        )
        p10_confidence = float(
            np.quantile(tooth_confidence, 0.10)
        )
    else:
        mean_confidence = float("nan")
        median_confidence = float("nan")
        p10_confidence = float("nan")

    return ToothMetric(
        case=case,
        patient_id=patient_id,
        arch=arch,
        tooth=tooth,
        tooth_group=tooth_group(arch, tooth),
        present=True,
        vertex_count=vertex_count,
        vertex_ratio=vertex_count / len(labels),
        complete_face_count=complete_face_count,
        component_count=int(component_count),
        largest_component_vertices=largest,
        largest_component_fraction=float(largest_fraction),
        bbox_diag_mm=bbox_diag,
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        centroid_z=float(centroid[2]),
        mean_confidence=mean_confidence,
        median_confidence=median_confidence,
        p10_confidence=p10_confidence,
    )


def find_input_stl(
    input_dir: Path,
    case: str,
) -> Optional[Path]:
    direct = input_dir / f"{case}.stl"

    if direct.is_file():
        return direct

    candidates = sorted(
        input_dir.glob(f"{case}*.stl")
    )

    return candidates[0] if candidates else None


def inspect_case(
    json_path: Path,
    output_dir: Path,
    input_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[ToothMetric]]:
    match = CASE_PATTERN.fullmatch(json_path.name)

    if match is None:
        raise ValueError(
            f"Unexpected final JSON filename: {json_path.name}"
        )

    case = match.group("case")
    patient_id = match.group("patient")
    arch = match.group("arch").lower()
    config = ARCH_CONFIG[arch]

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        labels = np.asarray(
            json.load(handle),
            dtype=np.int32,
        ).reshape(-1)

    raw_ply = output_dir / f"{case}_pass2_full_raw.ply"
    aligned_ply = output_dir / f"{case}_pass2_full_aligned.ply"

    if raw_ply.is_file():
        mesh_path = raw_ply
        mesh_coordinate_type = "raw"
    elif aligned_ply.is_file():
        mesh_path = aligned_ply
        mesh_coordinate_type = "aligned"
    else:
        raise FileNotFoundError(
            f"Missing full PLY for {case}"
        )

    mesh = load_mesh(mesh_path)
    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )
    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    if len(labels) != len(vertices):
        raise ValueError(
            f"JSON label length {len(labels)} != PLY vertices {len(vertices)}"
        )

    npz_path = output_dir / f"{case}_two_pass_data.npz"
    full_confidence, confidence_available = load_full_confidence(
        npz_path,
        len(vertices),
    )

    graph = build_vertex_graph(
        len(vertices),
        faces,
    )

    expected_set = set(config["full_order"])
    present_labels = sorted(
        int(value)
        for value in np.unique(labels)
        if int(value) != 0
    )
    unexpected_labels = sorted(
        value
        for value in present_labels
        if value not in expected_set
    )

    metrics = [
        tooth_metric(
            case=case,
            patient_id=patient_id,
            arch=arch,
            tooth=tooth,
            vertices=vertices,
            faces=faces,
            labels=labels,
            graph=graph,
            full_confidence=full_confidence,
        )
        for tooth in config["full_order"]
    ]

    by_tooth = {
        metric.tooth: metric
        for metric in metrics
    }

    missing_core = [
        tooth
        for tooth in config["core_order"]
        if not by_tooth[tooth].present
    ]
    tiny_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and by_tooth[tooth].vertex_count
            < args.min_core_vertices
        )
    ]
    low_vertex_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].vertex_count
            >= args.min_core_vertices
            and by_tooth[tooth].vertex_count
            < args.review_core_vertices
        )
    ]
    fragmented_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and by_tooth[tooth].largest_component_fraction
            < args.min_largest_component_fraction
        )
    ]
    severely_fragmented_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and by_tooth[tooth].largest_component_fraction
            < args.severe_component_fraction
        )
    ]
    oversized_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and np.isfinite(by_tooth[tooth].bbox_diag_mm)
            and by_tooth[tooth].bbox_diag_mm
            > args.max_core_bbox_diag_mm
        )
    ]
    severely_oversized_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and np.isfinite(by_tooth[tooth].bbox_diag_mm)
            and by_tooth[tooth].bbox_diag_mm
            > args.severe_core_bbox_diag_mm
        )
    ]
    low_confidence_core = [
        tooth
        for tooth in config["core_order"]
        if (
            by_tooth[tooth].present
            and np.isfinite(by_tooth[tooth].mean_confidence)
            and by_tooth[tooth].mean_confidence
            < args.min_tooth_confidence
        )
    ]

    missing_second_molars = [
        tooth
        for tooth in config["second_molars"]
        if not by_tooth[tooth].present
    ]
    missing_wisdom_teeth = [
        tooth
        for tooth in config["wisdom_teeth"]
        if not by_tooth[tooth].present
    ]

    contamination_detected = False
    contamination_ratio = float("nan")
    crop_fallback = False
    roi_ratio = float("nan")
    detected_tooth_classes = -1
    pass1_retry_used = False
    chosen_align_score = float("nan")
    chosen_align_trial = -1

    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as npz:
            contamination_detected = safe_scalar(
                npz, "contamination_detected", False
            )
            contamination_ratio = safe_scalar(
                npz, "contamination_total_tooth_ratio", float("nan")
            )
            crop_fallback = safe_scalar(
                npz, "crop_fallback", False
            )
            roi_ratio = safe_scalar(
                npz, "roi_ratio", float("nan")
            )
            detected_tooth_classes = safe_scalar(
                npz, "detected_tooth_classes", -1
            )
            pass1_retry_used = safe_scalar(
                npz, "pass1_retry_used", False
            )
            chosen_align_score = safe_scalar(
                npz, "chosen_align_score", float("nan")
            )
            chosen_align_trial = safe_scalar(
                npz, "chosen_align_trial", -1
            )

    score = 100.0
    reasons: list[str] = []
    critical_reasons: list[str] = []
    informational: list[str] = []

    if missing_core:
        penalty = min(30.0, 10.0 * len(missing_core))
        score -= penalty
        reason = "core_missing:" + ",".join(map(str, missing_core))
        reasons.append(reason)
        critical_reasons.append(reason)

    if tiny_core:
        penalty = min(25.0, 10.0 * len(tiny_core))
        score -= penalty
        reason = "core_too_few_vertices:" + ",".join(map(str, tiny_core))
        reasons.append(reason)
        critical_reasons.append(reason)

    if low_vertex_core:
        score -= min(12.0, 2.0 * len(low_vertex_core))
        reasons.append(
            "core_low_vertex_count:" + ",".join(map(str, low_vertex_core))
        )

    if fragmented_core:
        score -= min(20.0, 6.0 * len(fragmented_core))
        reasons.append(
            "core_fragmented:" + ",".join(map(str, fragmented_core))
        )

    if severely_fragmented_core:
        reason = (
            "core_severely_fragmented:"
            + ",".join(map(str, severely_fragmented_core))
        )
        critical_reasons.append(reason)

    if oversized_core:
        score -= min(15.0, 5.0 * len(oversized_core))
        reasons.append(
            "core_oversized:" + ",".join(map(str, oversized_core))
        )

    if severely_oversized_core:
        reason = (
            "core_severely_oversized:"
            + ",".join(map(str, severely_oversized_core))
        )
        critical_reasons.append(reason)

    if low_confidence_core:
        score -= min(12.0, 3.0 * len(low_confidence_core))
        reasons.append(
            "core_low_confidence:"
            + ",".join(map(str, low_confidence_core))
        )

    if unexpected_labels:
        score -= 20.0
        reason = (
            "unexpected_fdi_labels:"
            + ",".join(map(str, unexpected_labels))
        )
        reasons.append(reason)
        critical_reasons.append(reason)

    if contamination_detected:
        score -= 6.0
        reasons.append("pass1_contamination_detected")

    if crop_fallback:
        score -= 8.0
        reasons.append("crop_fallback_used")

    if np.isfinite(roi_ratio) and (
        roi_ratio < 0.15
        or roi_ratio > 0.95
    ):
        score -= 8.0
        reasons.append(f"extreme_roi_ratio:{roi_ratio:.4f}")

    if (
        detected_tooth_classes >= 0
        and detected_tooth_classes < args.min_detected_classes
    ):
        score -= 10.0
        reasons.append(
            f"too_few_detected_classes:{detected_tooth_classes}"
        )

    if missing_second_molars:
        informational.append(
            "missing_second_molars:"
            + ",".join(map(str, missing_second_molars))
        )

    if missing_wisdom_teeth:
        informational.append(
            "missing_wisdom_teeth:"
            + ",".join(map(str, missing_wisdom_teeth))
        )

    score = float(np.clip(score, 0.0, 100.0))

    crowding_geometry_available = (
        len(missing_core) == 0
        and len(tiny_core) == 0
        and all(
            by_tooth[tooth].complete_face_count > 0
            for tooth in config["core_order"]
        )
    )

    if critical_reasons or score < args.retry_score:
        status = "retry_once"
        retry_recommended = True
    elif score < args.acceptable_score or reasons:
        status = "review"
        retry_recommended = False
    else:
        status = "acceptable"
        retry_recommended = False

    input_stl = find_input_stl(
        input_dir,
        case,
    )

    summary = {
        "case": case,
        "patient_id": patient_id,
        "arch": arch,
        "status": status,
        "qc_score": score,
        "retry_recommended": retry_recommended,
        "crowding_geometry_available": crowding_geometry_available,
        "present_fdi_labels": present_labels,
        "present_tooth_count": len(present_labels),
        "missing_core_teeth": missing_core,
        "tiny_core_teeth": tiny_core,
        "low_vertex_core_teeth": low_vertex_core,
        "fragmented_core_teeth": fragmented_core,
        "severely_fragmented_core_teeth": severely_fragmented_core,
        "oversized_core_teeth": oversized_core,
        "severely_oversized_core_teeth": severely_oversized_core,
        "low_confidence_core_teeth": low_confidence_core,
        "missing_second_molars": missing_second_molars,
        "missing_wisdom_teeth": missing_wisdom_teeth,
        "unexpected_fdi_labels": unexpected_labels,
        "reasons": reasons,
        "critical_reasons": critical_reasons,
        "informational": informational,
        "confidence_available": confidence_available,
        "contamination_detected": contamination_detected,
        "contamination_total_tooth_ratio": contamination_ratio,
        "crop_fallback": crop_fallback,
        "roi_ratio": roi_ratio,
        "detected_tooth_classes": detected_tooth_classes,
        "pass1_retry_used": pass1_retry_used,
        "chosen_align_score": chosen_align_score,
        "chosen_align_trial": chosen_align_trial,
        "mesh_coordinate_type": mesh_coordinate_type,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "json_path": str(json_path),
        "mesh_path": str(mesh_path),
        "npz_path": str(npz_path),
        "input_stl": "" if input_stl is None else str(input_stl),
    }

    return summary, metrics


def json_safe(value: Any) -> Any:
    """
    Recursively convert NumPy values and non-finite floats into valid JSON values.

    json.dumps(default=...) cannot intercept built-in float('nan'), so NaN/Inf
    must be removed before serialization.
    """
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

    if isinstance(value, float):
        return (
            float(value)
            if math.isfinite(value)
            else None
        )

    return value


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    input_dir = args.input_dir.resolve()
    qc_dir = args.qc_dir.resolve()

    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cases_dir = qc_dir / "cases"
    cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_jsons = [
        path
        for path in sorted(output_dir.glob("*.json"))
        if CASE_PATTERN.fullmatch(path.name)
    ]

    if not final_jsons:
        raise RuntimeError(
            f"No final case JSON files found in {output_dir}"
        )

    summaries = []
    tooth_rows: list[dict[str, Any]] = []
    failures = []

    print("=" * 78)
    print("FDI-aware segmentation QC")
    print("=" * 78)
    print(f"predict1 output: {output_dir}")
    print(f"original input:  {input_dir}")
    print(f"QC output:       {qc_dir}")
    print(f"cases found:     {len(final_jsons)}")
    print()

    for index, json_path in enumerate(final_jsons, start=1):
        print(
            f"[{index}/{len(final_jsons)}] {json_path.stem}"
        )

        try:
            summary, metrics = inspect_case(
                json_path=json_path,
                output_dir=output_dir,
                input_dir=input_dir,
                args=args,
            )

            summaries.append(summary)
            tooth_rows.extend(
                asdict(metric)
                for metric in metrics
            )

            case_report_path = (
                cases_dir
                / f"{summary['case']}_qc.json"
            )

            if args.overwrite or not case_report_path.exists():
                report_payload = json_safe(
                    {
                        "summary": summary,
                        "teeth": [
                            asdict(metric)
                            for metric in metrics
                        ],
                    }
                )

                case_report_path.write_text(
                    json.dumps(
                        report_payload,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            print(
                f"  {summary['status']}: "
                f"score={summary['qc_score']:.1f}, "
                f"geometry={summary['crowding_geometry_available']}, "
                f"retry={summary['retry_recommended']}"
            )

        except Exception as exc:
            failures.append(
                {
                    "case": json_path.stem,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"  FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

    summary_csv = qc_dir / "segmentation_qc_summary.csv"
    tooth_csv = qc_dir / "segmentation_qc_tooth_metrics.csv"
    retry_txt = qc_dir / "retry_cases.txt"
    retry_manifest = qc_dir / "retry_manifest.csv"
    failure_csv = qc_dir / "qc_failures.csv"

    summary_columns = [
        "case",
        "patient_id",
        "arch",
        "status",
        "qc_score",
        "retry_recommended",
        "crowding_geometry_available",
        "present_tooth_count",
        "present_fdi_labels",
        "missing_core_teeth",
        "tiny_core_teeth",
        "low_vertex_core_teeth",
        "fragmented_core_teeth",
        "severely_fragmented_core_teeth",
        "oversized_core_teeth",
        "severely_oversized_core_teeth",
        "low_confidence_core_teeth",
        "missing_second_molars",
        "missing_wisdom_teeth",
        "unexpected_fdi_labels",
        "reasons",
        "critical_reasons",
        "informational",
        "confidence_available",
        "contamination_detected",
        "contamination_total_tooth_ratio",
        "crop_fallback",
        "roi_ratio",
        "detected_tooth_classes",
        "pass1_retry_used",
        "chosen_align_score",
        "chosen_align_trial",
        "mesh_coordinate_type",
        "vertex_count",
        "face_count",
        "json_path",
        "mesh_path",
        "npz_path",
        "input_stl",
    ]

    with summary_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_columns,
        )
        writer.writeheader()

        for row in summaries:
            writer.writerow(
                {
                    column: csv_value(
                        row.get(column, "")
                    )
                    for column in summary_columns
                }
            )

    if tooth_rows:
        with tooth_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            columns = list(tooth_rows[0].keys())
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
            )
            writer.writeheader()

            for row in tooth_rows:
                writer.writerow(
                    {
                        key: csv_value(value)
                        for key, value in row.items()
                    }
                )

    retry_rows = [
        row
        for row in summaries
        if row["retry_recommended"]
    ]

    retry_txt.write_text(
        "\n".join(
            row["case"]
            for row in retry_rows
        )
        + ("\n" if retry_rows else ""),
        encoding="utf-8",
    )

    retry_columns = [
        "case",
        "arch",
        "qc_score",
        "critical_reasons",
        "input_stl",
    ]

    with retry_manifest.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=retry_columns,
        )
        writer.writeheader()

        for row in retry_rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "arch": row["arch"],
                    "qc_score": row["qc_score"],
                    "critical_reasons": csv_value(
                        row["critical_reasons"]
                    ),
                    "input_stl": row["input_stl"],
                }
            )

    if args.copy_retry_input:
        retry_input_dir = qc_dir / "retry_input"
        retry_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for row in retry_rows:
            source_text = row["input_stl"]

            if not source_text:
                continue

            source = Path(source_text)

            if source.is_file():
                shutil.copy2(
                    source,
                    retry_input_dir / source.name,
                )

    if failures:
        with failure_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            columns = [
                "case",
                "error_type",
                "error",
            ]
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
            )
            writer.writeheader()
            writer.writerows(failures)

    counts = {
        status: sum(
            row["status"] == status
            for row in summaries
        )
        for status in (
            "acceptable",
            "review",
            "retry_once",
        )
    }
    geometry_available = sum(
        row["crowding_geometry_available"]
        for row in summaries
    )

    print()
    print("=" * 78)
    print("QC complete")
    print("=" * 78)
    print(f"successful cases: {len(summaries)}")
    print(f"failed cases:     {len(failures)}")
    print(f"acceptable:       {counts['acceptable']}")
    print(f"review:           {counts['review']}")
    print(f"retry_once:       {counts['retry_once']}")
    print(
        f"geometry usable:  {geometry_available}/"
        f"{len(summaries)}"
    )
    print(f"summary:          {summary_csv}")
    print(f"tooth metrics:    {tooth_csv}")
    print(f"retry list:       {retry_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())