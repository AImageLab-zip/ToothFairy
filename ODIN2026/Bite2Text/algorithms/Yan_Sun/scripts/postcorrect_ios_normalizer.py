#!/usr/bin/env python3
"""Apply conservative paired-jaw 180-degree corrections after IOS-Normalizer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import trimesh


ROTATE_Y_180 = np.diag([-1.0, 1.0, -1.0])
ROTATE_Z_180 = np.diag([-1.0, -1.0, 1.0])
ROTATE_XZ_ARCH_TO_XY = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def load_mesh(path):
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise RuntimeError(f"Invalid mesh: {path}")
    return mesh, vertices


def lower_front_direction(vertices):
    xy = vertices[:, :2]
    centered = xy - xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    width_vector = eigenvectors[:, -1]
    depth_vector = np.array([-width_vector[1], width_vector[0]])
    width = centered @ width_vector
    depth = centered @ depth_vector
    low_cut, high_cut = np.quantile(depth, [0.15, 0.85])

    def robust_width(values):
        low, high = np.quantile(values, [0.05, 0.95])
        return float(high - low)

    low_width = robust_width(width[depth <= low_cut])
    high_width = robust_width(width[depth >= high_cut])
    if low_width <= 1e-8 or high_width <= 1e-8:
        raise RuntimeError("Could not infer lower-arch anterior direction")
    log_ratio = math.log(low_width / high_width)
    front = depth_vector if log_ratio > 0 else -depth_vector
    angle = math.degrees(math.atan2(front[0], front[1]))
    return angle, abs(log_ratio)


def infer_axis_layout(lower_vertices, upper_vertices):
    spans = []
    for vertices in (lower_vertices, upper_vertices):
        q05, q95 = np.quantile(vertices, [0.05, 0.95], axis=0)
        spans.append(q95 - q05)
    mean_span = np.mean(spans, axis=0)
    if mean_span[1] >= mean_span[2]:
        return "XY_arch_Z_vertical"
    return "XZ_arch_Y_vertical"


def save_affine(matrix, center, output_scan_path):
    affine = np.eye(4, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    affine[:3, :3] = matrix.T
    affine[:3, 3] = -(matrix.T @ center)
    np.save(output_scan_path.with_suffix(".npy"), affine)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.input_dir / "normalization_manifest.json").read_text()
    )

    post_records = []
    corrected_manifest = dict(manifest)
    corrected_manifest["output_dir"] = str(args.output_dir.resolve())
    corrected_manifest["post_correction"] = {
        "xz_arch_plane": "rotate 180 degrees around the Y=Z diagonal",
        "upper_not_superior": "rotate 180 degrees around Y",
        "lower_anterior_clearly_negative_y": "rotate 180 degrees around Z",
        "anterior_flip_threshold_degrees": 150.0,
    }
    corrected_records = []

    total_records = len(manifest["records"])
    for record_index, record in enumerate(manifest["records"], start=1):
        corrected_record = dict(record)
        corrected_records.append(corrected_record)
        if not record["ok"]:
            post_records.append(
                {
                    "patient": Path(record["patient_dir"]).name,
                    "ok": False,
                    "error": "Skipped because IOS-Normalizer failed",
                }
            )
            continue

        patient = Path(record["patient_dir"]).name
        input_lower = args.input_dir / patient / "ios_lower.stl"
        input_upper = args.input_dir / patient / "ios_upper.stl"
        lower_mesh, lower_vertices = load_mesh(input_lower)
        upper_mesh, upper_vertices = load_mesh(input_upper)
        correction = np.eye(3)
        actions = []

        source_axis_layout = infer_axis_layout(lower_vertices, upper_vertices)
        if source_axis_layout == "XZ_arch_Y_vertical":
            correction = correction @ ROTATE_XZ_ARCH_TO_XY
            lower_vertices = lower_vertices @ ROTATE_XZ_ARCH_TO_XY
            upper_vertices = upper_vertices @ ROTATE_XZ_ARCH_TO_XY
            actions.append("rotate_xz_arch_to_xy")

        vertical_gap_before = float(
            np.median(upper_vertices[:, 2]) - np.median(lower_vertices[:, 2])
        )
        if vertical_gap_before <= 0:
            correction = correction @ ROTATE_Y_180
            lower_vertices = lower_vertices @ ROTATE_Y_180
            upper_vertices = upper_vertices @ ROTATE_Y_180
            actions.append("rotate_y_180_for_upper_positive_z")

        lower_front_angle_before, lower_front_confidence = lower_front_direction(
            lower_vertices
        )
        if abs(lower_front_angle_before) >= 150.0:
            correction = correction @ ROTATE_Z_180
            lower_vertices = lower_vertices @ ROTATE_Z_180
            upper_vertices = upper_vertices @ ROTATE_Z_180
            actions.append("rotate_z_180_for_anterior_positive_y")

        lower_front_angle_after, _ = lower_front_direction(lower_vertices)
        vertical_gap_after = float(
            np.median(upper_vertices[:, 2]) - np.median(lower_vertices[:, 2])
        )
        output_patient = args.output_dir / patient
        output_patient.mkdir(parents=True, exist_ok=True)
        lower_output = output_patient / "ios_lower.stl"
        upper_output = output_patient / "ios_upper.stl"
        if actions:
            lower_mesh.vertices = lower_vertices
            upper_mesh.vertices = upper_vertices
            lower_mesh.export(lower_output)
            upper_mesh.export(upper_output)
        else:
            shutil.copy2(input_lower, lower_output)
            shutil.copy2(input_upper, upper_output)

        original_matrix = np.asarray(record["matrix"], dtype=np.float64)
        final_matrix = original_matrix @ correction
        save_affine(final_matrix, record["center"], lower_output)
        save_affine(final_matrix, record["center"], upper_output)
        corrected_record["matrix"] = final_matrix.tolist()
        corrected_record["output_paths"] = [str(lower_output), str(upper_output)]
        corrected_record["post_correction_matrix"] = correction.tolist()
        corrected_record["post_correction_actions"] = actions
        post_records.append(
            {
                "patient": patient,
                "ok": True,
                "actions": actions,
                "source_axis_layout": source_axis_layout,
                "vertical_gap_before": vertical_gap_before,
                "vertical_gap_after": vertical_gap_after,
                "lower_front_angle_before": lower_front_angle_before,
                "lower_front_angle_after": lower_front_angle_after,
                "lower_front_shape_confidence": lower_front_confidence,
                "correction_matrix": correction.tolist(),
                "final_matrix": final_matrix.tolist(),
            }
        )
        if (
            actions
            or record_index == 1
            or record_index == total_records
            or record_index % max(args.log_interval, 1) == 0
        ):
            print(
                f"[{record_index}/{total_records}] {patient}: "
                f"actions={actions or ['none']} "
                f"z={vertical_gap_before:.2f}->{vertical_gap_after:.2f} "
                f"front={lower_front_angle_before:.1f}->{lower_front_angle_after:.1f}",
                flush=True,
            )

    corrected_manifest["records"] = corrected_records
    (args.output_dir / "normalization_manifest.json").write_text(
        json.dumps(corrected_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "post_correction_manifest.json").write_text(
        json.dumps({"records": post_records}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(post_records),
                "corrected": sum(bool(item.get("actions")) for item in post_records),
                "failures": sum(not item["ok"] for item in post_records),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
