#!/usr/bin/env python3
"""Validate IOS-Normalizer pilot outputs against geometry and invariance checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat-dir", type=Path)
    parser.add_argument("--audit-script", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--max-triangles", type=int, default=3000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip full topology, hash, and transform-residual checks.",
    )
    parser.add_argument("--render-limit", type=int, default=42)
    parser.add_argument("--report-name", default="pilot_validation")
    return parser.parse_args()


def load_audit_module(path):
    spec = importlib.util.spec_from_file_location("bite2text_ras_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import audit script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_arrays(path):
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise RuntimeError(f"Invalid mesh: {path}")
    return vertices, faces


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def main():
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    audit = load_audit_module(args.audit_script)

    manifest_path = args.output_dir / "normalization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    repeat_manifest = None
    if args.repeat_dir:
        repeat_manifest = json.loads(
            (args.repeat_dir / "normalization_manifest.json").read_text()
        )
        repeat_by_patient = {
            Path(record["patient_dir"]).name: record
            for record in repeat_manifest["records"]
        }
    else:
        repeat_by_patient = {}

    results = []
    render_cases = []
    for record in manifest["records"]:
        patient = Path(record["patient_dir"]).name
        result = {
            "patient": patient,
            "normalization_ok": bool(record["ok"]),
            "rotation_index": record.get("rotation_index"),
            "confidence": record.get("confidence"),
            "probability_margin": record.get("probability_margin"),
            "error": record.get("error", ""),
        }
        if not record["ok"]:
            results.append(result)
            continue

        matrix = np.asarray(record["matrix"], dtype=np.float64)
        center = np.asarray(record["center"], dtype=np.float64)
        result["matrix_determinant"] = float(np.linalg.det(matrix))
        result["orthogonality_max_error"] = float(
            np.max(np.abs(matrix.T @ matrix - np.eye(3)))
        )

        maximum_residual = float("nan") if args.quick else 0.0
        topology_preserved = None if args.quick else True
        arch_details = {}
        if not args.quick:
            for arch in ("lower", "upper"):
                input_path = args.input_dir / patient / f"ios_{arch}.stl"
                output_path = args.output_dir / patient / f"ios_{arch}.stl"
                input_vertices, input_faces = mesh_arrays(input_path)
                output_vertices, output_faces = mesh_arrays(output_path)
                same_counts = (
                    input_vertices.shape == output_vertices.shape
                    and input_faces.shape == output_faces.shape
                )
                topology_preserved = topology_preserved and same_counts
                residual = float("inf")
                if same_counts:
                    expected = (input_vertices - center) @ matrix
                    residual = float(np.max(np.abs(expected - output_vertices)))
                    maximum_residual = max(maximum_residual, residual)
                arch_details[arch] = {
                    "input_vertices": int(len(input_vertices)),
                    "output_vertices": int(len(output_vertices)),
                    "input_faces": int(len(input_faces)),
                    "output_faces": int(len(output_faces)),
                    "max_transform_residual": residual,
                    "input_sha256": sha256(input_path),
                    "output_sha256": sha256(output_path),
                }

        lower_path = args.output_dir / patient / "ios_lower.stl"
        upper_path = args.output_dir / patient / "ios_upper.stl"
        lower_points, lower_triangles = audit.sample_binary_stl(
            lower_path.read_bytes(), args.max_triangles
        )
        upper_points, upper_triangles = audit.sample_binary_stl(
            upper_path.read_bytes(), args.max_triangles
        )
        upper_stats = audit.mesh_stats(upper_points, upper_triangles)
        lower_stats = audit.mesh_stats(lower_points, lower_triangles)
        orientation = audit.pair_stats(patient, upper_stats, lower_stats)
        upper_front_angle = math.degrees(
            math.atan2(upper_stats["front_xy_x"], upper_stats["front_xy_y"])
        )
        lower_front_angle = math.degrees(
            math.atan2(lower_stats["front_xy_x"], lower_stats["front_xy_y"])
        )
        front_direction_source = "lower"
        front_direction_pass = abs(lower_front_angle) <= 30.0
        if (
            not front_direction_pass
            and float(orientation["front_upper_lower_agreement"]) < 0.50
            and upper_stats["front_xy_confidence"]
            > lower_stats["front_xy_confidence"]
            and abs(upper_front_angle) <= 30.0
        ):
            front_direction_pass = True
            front_direction_source = "upper_fallback"
        elif not front_direction_pass:
            front_direction_source = "unresolved"
        orientation_pass = (
            orientation["axis_layout"] == "XY_arch_Z_vertical"
            and float(orientation["upper_minus_lower_vertical"]) > 0
            and front_direction_pass
        )
        result.update(
            {
                "topology_preserved": topology_preserved,
                "max_transform_residual": maximum_residual,
                "axis_layout": orientation["axis_layout"],
                "upper_minus_lower_vertical": orientation[
                    "upper_minus_lower_vertical"
                ],
                "front_angle_degrees": orientation["front_angle_degrees"],
                "front_upper_lower_agreement": orientation[
                    "front_upper_lower_agreement"
                ],
                "upper_front_angle_degrees": upper_front_angle,
                "upper_front_shape_confidence": upper_stats[
                    "front_xy_confidence"
                ],
                "lower_front_angle_degrees": lower_front_angle,
                "lower_front_shape_confidence": lower_stats[
                    "front_xy_confidence"
                ],
                "front_direction_source": front_direction_source,
                "orientation_signature": orientation["orientation_signature"],
                "orientation_pass": orientation_pass,
                "arch_details": arch_details,
            }
        )

        repeat = repeat_by_patient.get(patient)
        if repeat is not None and repeat.get("ok"):
            repeat_matrix = np.asarray(repeat["matrix"], dtype=np.float64)
            result["repeat_rotation_index_equal"] = (
                record["rotation_index"] == repeat["rotation_index"]
            )
            result["repeat_matrix_max_difference"] = float(
                np.max(np.abs(matrix - repeat_matrix))
            )
            result["repeat_outputs_byte_identical"] = all(
                sha256(args.output_dir / patient / f"ios_{arch}.stl")
                == sha256(args.repeat_dir / patient / f"ios_{arch}.stl")
                for arch in ("lower", "upper")
            )
        elif repeat_manifest is not None:
            result["repeat_rotation_index_equal"] = False
            result["repeat_matrix_max_difference"] = float("inf")
            result["repeat_outputs_byte_identical"] = False

        results.append(result)
        should_render = (
            not orientation_pass
            or float(record.get("confidence", 1.0)) < 0.9
            or len(results) <= 6
        )
        if should_render and len(render_cases) < max(args.render_limit, 0):
            render_cases.append((patient, upper_points, lower_points))

    for page_start in range(0, len(render_cases), 6):
        page_number = page_start // 6 + 1
        audit.render_projection_page(
            args.report_dir / f"normalized_projection_page_{page_number:02d}.png",
            render_cases[page_start : page_start + 6],
        )

    successful = [item for item in results if item["normalization_ok"]]
    summary = {
        "cases": len(results),
        "normalization_successes": len(successful),
        "normalization_failures": len(results) - len(successful),
        "orientation_passes": sum(item.get("orientation_pass", False) for item in results),
        "axis_layout_counts": dict(
            Counter(item.get("axis_layout", "failed") for item in results)
        ),
        "orientation_signature_counts": dict(
            Counter(item.get("orientation_signature", "failed") for item in results)
        ),
        "topology_preserved": None
        if args.quick
        else sum(item.get("topology_preserved", False) for item in results),
        "matrix_det_min": min(
            (item["matrix_determinant"] for item in successful), default=float("nan")
        ),
        "matrix_det_max": max(
            (item["matrix_determinant"] for item in successful), default=float("nan")
        ),
        "orthogonality_max_error": max(
            (item["orthogonality_max_error"] for item in successful), default=float("nan")
        ),
        "transform_residual_max": float("nan")
        if args.quick
        else max(
            (item["max_transform_residual"] for item in successful),
            default=float("nan"),
        ),
        "confidence_min": min(
            (item["confidence"] for item in successful), default=float("nan")
        ),
        "confidence_median": statistics.median(
            [item["confidence"] for item in successful]
        )
        if successful
        else float("nan"),
        "repeat_rotation_matches": sum(
            item.get("repeat_rotation_index_equal", False) for item in results
        ),
        "repeat_matrix_matches": sum(
            item.get("repeat_matrix_max_difference", float("inf")) <= 1e-7
            for item in results
        ),
        "repeat_byte_identical_outputs": sum(
            item.get("repeat_outputs_byte_identical", False) for item in results
        ),
    }
    summary["acceptance_pass"] = (
        summary["normalization_failures"] == 0
        and summary["orientation_passes"] == summary["cases"]
        and (args.quick or summary["topology_preserved"] == summary["cases"])
        and summary["matrix_det_min"] > 0.999
        and summary["matrix_det_max"] < 1.001
        and summary["orthogonality_max_error"] < 1e-5
        and (args.quick or summary["transform_residual_max"] < 1e-3)
        and (
            repeat_manifest is None
            or (
                summary["repeat_rotation_matches"] == summary["cases"]
                and summary["repeat_matrix_matches"] == summary["cases"]
                and summary["repeat_byte_identical_outputs"] == summary["cases"]
            )
        )
    )

    fieldnames = [
        "patient",
        "normalization_ok",
        "rotation_index",
        "confidence",
        "probability_margin",
        "matrix_determinant",
        "orthogonality_max_error",
        "topology_preserved",
        "max_transform_residual",
        "axis_layout",
        "upper_minus_lower_vertical",
        "front_angle_degrees",
        "front_upper_lower_agreement",
        "upper_front_angle_degrees",
        "upper_front_shape_confidence",
        "lower_front_angle_degrees",
        "lower_front_shape_confidence",
        "front_direction_source",
        "orientation_signature",
        "orientation_pass",
        "repeat_rotation_index_equal",
        "repeat_matrix_max_difference",
        "repeat_outputs_byte_identical",
        "error",
    ]
    with (args.report_dir / f"{args.report_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    payload = {"summary": summary, "cases": results}
    (args.report_dir / f"{args.report_name}.json").write_text(
        json.dumps(clean_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(clean_json(summary), indent=2))
    return 0 if summary["acceptance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
