#!/usr/bin/env python3
"""Audit Bite2Text IOS meshes for consistency with a shared RAS frame.

The script reads STL files directly from the dataset ZIP and never modifies the
archive. It intentionally uses only the Python standard library so it can run
on a minimal server installation.

What can be checked from unlabelled paired arch geometry:
  * Whether the arch lies in XY with Z vertical (RAS axis roles), or in XZ
    with Y vertical (a Y/Z-swapped layout seen in some exports).
  * Superior ordering along the detected vertical axis.
  * Anterior sign proxy: the positive depth-axis end of a U-shaped arch should
    usually be the narrower/anterior end.
  * Upper/lower relative registration and overlap.

What cannot be proven from nearly symmetric unlabelled geometry alone:
  * Whether +X is truly patient-right rather than patient-left. That requires
    FDI/tooth landmarks or another asymmetric anatomical reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import struct
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-triangles", type=int, default=4000)
    parser.add_argument("--render-cases", type=int, default=24)
    parser.add_argument("--cases-per-page", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    alpha = position - low
    return sorted_values[low] * (1.0 - alpha) + sorted_values[high] * alpha


def robust_summary(values: list[float]) -> dict[str, float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return {"count": 0, "median": float("nan"), "mad": float("nan"), "p05": float("nan"), "p95": float("nan")}
    median = statistics.median(clean)
    mad = statistics.median(abs(v - median) for v in clean)
    return {
        "count": len(clean),
        "median": median,
        "mad": mad,
        "p05": percentile(clean, 0.05),
        "p95": percentile(clean, 0.95),
    }


def robust_z(value: float, summary: dict[str, float]) -> float:
    mad = summary.get("mad", float("nan"))
    median = summary.get("median", float("nan"))
    if not math.isfinite(value) or not math.isfinite(mad) or mad <= 1e-9:
        return 0.0
    return 0.67448975 * (value - median) / mad


def sample_binary_stl(data: bytes, max_triangles: int) -> tuple[list[tuple[float, float, float]], int]:
    if len(data) < 84:
        raise ValueError("STL shorter than 84-byte binary header")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangle_count * 50
    if triangle_count <= 0 or expected > len(data):
        raise ValueError(
            f"Not a valid binary STL: triangles={triangle_count}, expected={expected}, bytes={len(data)}"
        )

    sample_count = min(triangle_count, max_triangles)
    points: list[tuple[float, float, float]] = []
    last_index = -1
    for sample_index in range(sample_count):
        triangle_index = min(
            triangle_count - 1,
            int((sample_index + 0.5) * triangle_count / sample_count),
        )
        if triangle_index == last_index:
            continue
        last_index = triangle_index
        offset = 84 + triangle_index * 50 + 12
        coords = struct.unpack_from("<9f", data, offset)
        for vertex_index in range(0, 9, 3):
            point = coords[vertex_index : vertex_index + 3]
            if all(math.isfinite(value) and abs(value) < 1e6 for value in point):
                points.append((point[0], point[1], point[2]))
    if len(points) < 100:
        raise ValueError(f"Too few finite sampled points: {len(points)}")
    return points, triangle_count


def mesh_stats(points: list[tuple[float, float, float]], triangle_count: int) -> dict[str, float]:
    axes = [sorted(point[axis] for point in points) for axis in range(3)]
    q05 = [percentile(axis, 0.05) for axis in axes]
    q15 = [percentile(axis, 0.15) for axis in axes]
    q50 = [percentile(axis, 0.50) for axis in axes]
    q85 = [percentile(axis, 0.85) for axis in axes]
    q95 = [percentile(axis, 0.95) for axis in axes]
    spans = [q95[axis] - q05[axis] for axis in range(3)]

    def robust_width(values: list[float]) -> float:
        if len(values) < 20:
            return float("nan")
        return percentile(values, 0.95) - percentile(values, 0.05)

    def endpoint_widths(depth_axis: int) -> tuple[float, float, float]:
        low_x = sorted(point[0] for point in points if point[depth_axis] <= q15[depth_axis])
        high_x = sorted(point[0] for point in points if point[depth_axis] >= q85[depth_axis])
        width_low = robust_width(low_x)
        width_high = robust_width(high_x)
        if width_low > 1e-6 and width_high > 1e-6:
            score = math.log(width_low / width_high)
        else:
            score = float("nan")
        return width_low, width_high, score

    width_low_y, width_high_y, anterior_y_log_score = endpoint_widths(1)
    width_low_z, width_high_z, anterior_z_log_score = endpoint_widths(2)

    def pca_front(depth_axis: int) -> tuple[float, float, float]:
        """Return a unit anterior vector in (X, depth-axis) coordinates.

        The larger PCA eigenvector is treated as arch width. The perpendicular
        direction is arch depth, and the end with the narrower robust width is
        treated as anterior.
        """
        mean_x = statistics.mean(point[0] for point in points)
        mean_d = statistics.mean(point[depth_axis] for point in points)
        centered = [(point[0] - mean_x, point[depth_axis] - mean_d) for point in points]
        cov_xx = statistics.mean(x * x for x, _ in centered)
        cov_dd = statistics.mean(d * d for _, d in centered)
        cov_xd = statistics.mean(x * d for x, d in centered)
        theta = 0.5 * math.atan2(2.0 * cov_xd, cov_xx - cov_dd)
        width_vector = (math.cos(theta), math.sin(theta))
        depth_vector = (-math.sin(theta), math.cos(theta))
        projected = [
            (
                x * width_vector[0] + d * width_vector[1],
                x * depth_vector[0] + d * depth_vector[1],
            )
            for x, d in centered
        ]
        depth_values = sorted(depth for _, depth in projected)
        low_cut = percentile(depth_values, 0.15)
        high_cut = percentile(depth_values, 0.85)
        low_width_values = sorted(width for width, depth in projected if depth <= low_cut)
        high_width_values = sorted(width for width, depth in projected if depth >= high_cut)
        low_width = robust_width(low_width_values)
        high_width = robust_width(high_width_values)
        if low_width <= 1e-6 or high_width <= 1e-6:
            return float("nan"), float("nan"), float("nan")
        log_score = math.log(low_width / high_width)
        sign = 1.0 if log_score > 0 else -1.0
        return sign * depth_vector[0], sign * depth_vector[1], abs(log_score)

    front_xy_x, front_xy_y, front_xy_confidence = pca_front(1)
    front_xz_x, front_xz_z, front_xz_confidence = pca_front(2)

    return {
        "triangle_count": triangle_count,
        "sampled_points": len(points),
        "q05_x": q05[0],
        "q05_y": q05[1],
        "q05_z": q05[2],
        "q50_x": q50[0],
        "q50_y": q50[1],
        "q50_z": q50[2],
        "q95_x": q95[0],
        "q95_y": q95[1],
        "q95_z": q95[2],
        "span_x": spans[0],
        "span_y": spans[1],
        "span_z": spans[2],
        "width_low_y": width_low_y,
        "width_high_y": width_high_y,
        "anterior_y_log_score": anterior_y_log_score,
        "width_low_z": width_low_z,
        "width_high_z": width_high_z,
        "anterior_z_log_score": anterior_z_log_score,
        "front_xy_x": front_xy_x,
        "front_xy_y": front_xy_y,
        "front_xy_confidence": front_xy_confidence,
        "front_xz_x": front_xz_x,
        "front_xz_z": front_xz_z,
        "front_xz_confidence": front_xz_confidence,
    }


def interval_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    denominator = max(1e-6, min(a1 - a0, b1 - b0))
    return intersection / denominator


def pair_stats(patient: str, upper: dict[str, float], lower: dict[str, float]) -> dict[str, object]:
    combined_x0 = min(upper["q05_x"], lower["q05_x"])
    combined_x1 = max(upper["q95_x"], lower["q95_x"])
    combined_y0 = min(upper["q05_y"], lower["q05_y"])
    combined_y1 = max(upper["q95_y"], lower["q95_y"])
    combined_z0 = min(upper["q05_z"], lower["q05_z"])
    combined_z1 = max(upper["q95_z"], lower["q95_z"])

    def mean_finite(values: list[float]) -> float:
        values = [value for value in values if math.isfinite(value)]
        return statistics.mean(values) if values else float("nan")

    anterior_y_score = mean_finite(
        [upper["anterior_y_log_score"], lower["anterior_y_log_score"]]
    )
    anterior_z_score = mean_finite(
        [upper["anterior_z_log_score"], lower["anterior_z_log_score"]]
    )

    mean_y_span = statistics.mean([upper["span_y"], lower["span_y"]])
    mean_z_span = statistics.mean([upper["span_z"], lower["span_z"]])
    axis_layout = "XY_arch_Z_vertical" if mean_y_span >= mean_z_span else "XZ_arch_Y_vertical"
    layout_ratio = max(mean_y_span, mean_z_span) / max(min(mean_y_span, mean_z_span), 1e-6)
    upper_minus_lower_y = upper["q50_y"] - lower["q50_y"]
    upper_minus_lower_z = upper["q50_z"] - lower["q50_z"]
    if axis_layout == "XY_arch_Z_vertical":
        vertical_axis = "Z"
        depth_axis = "Y"
        vertical_gap = upper_minus_lower_z
        anterior_axis_score = anterior_y_score
        upper_front = (upper["front_xy_x"], upper["front_xy_y"])
        lower_front = (lower["front_xy_x"], lower["front_xy_y"])
        front_confidence = statistics.mean(
            [upper["front_xy_confidence"], lower["front_xy_confidence"]]
        )
    else:
        vertical_axis = "Y"
        depth_axis = "Z"
        vertical_gap = upper_minus_lower_y
        anterior_axis_score = anterior_z_score
        upper_front = (upper["front_xz_x"], upper["front_xz_z"])
        lower_front = (lower["front_xz_x"], lower["front_xz_z"])
        front_confidence = statistics.mean(
            [upper["front_xz_confidence"], lower["front_xz_confidence"]]
        )

    front_agreement = upper_front[0] * lower_front[0] + upper_front[1] * lower_front[1]
    mean_front_x = upper_front[0] + lower_front[0]
    mean_front_depth = upper_front[1] + lower_front[1]
    mean_front_norm = math.hypot(mean_front_x, mean_front_depth)
    if mean_front_norm > 1e-6:
        mean_front_x /= mean_front_norm
        mean_front_depth /= mean_front_norm
        front_angle_degrees = math.degrees(math.atan2(mean_front_x, mean_front_depth))
    else:
        mean_front_x = float("nan")
        mean_front_depth = float("nan")
        front_angle_degrees = float("nan")

    absolute_front_angle = abs(front_angle_degrees)
    if absolute_front_angle <= 45.0:
        front_direction_class = f"{depth_axis}+front"
    elif absolute_front_angle >= 135.0:
        front_direction_class = f"{depth_axis}-front"
    else:
        front_direction_class = f"{depth_axis}_oblique_front"

    combined_x_span = combined_x1 - combined_x0
    combined_y_span = combined_y1 - combined_y0
    combined_z_span = combined_z1 - combined_z0
    row: dict[str, object] = {
        "patient": patient,
        "upper_triangles": int(upper["triangle_count"]),
        "lower_triangles": int(lower["triangle_count"]),
        "upper_span_x": upper["span_x"],
        "upper_span_y": upper["span_y"],
        "upper_span_z": upper["span_z"],
        "lower_span_x": lower["span_x"],
        "lower_span_y": lower["span_y"],
        "lower_span_z": lower["span_z"],
        "combined_span_x": combined_x_span,
        "combined_span_y": combined_y_span,
        "combined_span_z": combined_z_span,
        "x_to_y_ratio": combined_x_span / max(combined_y_span, 1e-6),
        "z_to_xy_ratio": combined_z_span / max(combined_x_span, combined_y_span, 1e-6),
        "axis_layout": axis_layout,
        "layout_ratio": layout_ratio,
        "vertical_axis": vertical_axis,
        "depth_axis": depth_axis,
        "upper_minus_lower_x": upper["q50_x"] - lower["q50_x"],
        "upper_minus_lower_y": upper_minus_lower_y,
        "upper_minus_lower_z": upper_minus_lower_z,
        "upper_minus_lower_vertical": vertical_gap,
        "upper_lower_x_center_gap": abs(upper["q50_x"] - lower["q50_x"]),
        "upper_lower_y_center_gap": abs(upper["q50_y"] - lower["q50_y"]),
        "upper_lower_z_center_gap": abs(upper["q50_z"] - lower["q50_z"]),
        "x_overlap_ratio": interval_overlap_ratio(
            upper["q05_x"], upper["q95_x"], lower["q05_x"], lower["q95_x"]
        ),
        "y_overlap_ratio": interval_overlap_ratio(
            upper["q05_y"], upper["q95_y"], lower["q05_y"], lower["q95_y"]
        ),
        "z_overlap_ratio": interval_overlap_ratio(
            upper["q05_z"], upper["q95_z"], lower["q05_z"], lower["q95_z"]
        ),
        "depth_overlap_ratio": (
            interval_overlap_ratio(upper["q05_y"], upper["q95_y"], lower["q05_y"], lower["q95_y"])
            if depth_axis == "Y"
            else interval_overlap_ratio(upper["q05_z"], upper["q95_z"], lower["q05_z"], lower["q95_z"])
        ),
        "upper_anterior_y_log_score": upper["anterior_y_log_score"],
        "lower_anterior_y_log_score": lower["anterior_y_log_score"],
        "anterior_y_log_score": anterior_y_score,
        "upper_anterior_z_log_score": upper["anterior_z_log_score"],
        "lower_anterior_z_log_score": lower["anterior_z_log_score"],
        "anterior_z_log_score": anterior_z_score,
        "anterior_axis_log_score": anterior_axis_score,
        "front_vector_x": mean_front_x,
        "front_vector_depth": mean_front_depth,
        "front_angle_degrees": front_angle_degrees,
        "front_direction_class": front_direction_class,
        "front_upper_lower_agreement": front_agreement,
        "front_shape_confidence": front_confidence,
        "orientation_signature": "",
        "flags": "",
        "severity": "ok",
    }
    row["orientation_signature"] = "|".join(
        [
            "XY/Z" if axis_layout == "XY_arch_Z_vertical" else "XZ/Y",
            f"{vertical_axis}+upper" if vertical_gap > 0 else f"{vertical_axis}-upper",
            front_direction_class,
        ]
    )
    return row


def build_archive_index(archive: zipfile.ZipFile) -> tuple[dict[str, dict[str, str]], list[str]]:
    patients: dict[str, dict[str, str]] = defaultdict(dict)
    non_binary_candidates: list[str] = []
    for info in archive.infolist():
        if info.is_dir() or "/" not in info.filename:
            continue
        patient = info.filename.split("/", 1)[0]
        name = Path(info.filename).name.lower()
        if info.filename.lower().endswith(".stl") and "/ios/" in info.filename.lower():
            if "upper" in name:
                patients[patient]["upper"] = info.filename
            elif "lower" in name:
                patients[patient]["lower"] = info.filename
            else:
                non_binary_candidates.append(info.filename)
    return patients, non_binary_candidates


def audit_rows(
    archive: zipfile.ZipFile,
    patients: dict[str, dict[str, str]],
    max_triangles: int,
    limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    complete = sorted(patient for patient, files in patients.items() if {"upper", "lower"} <= files.keys())
    if limit > 0:
        complete = complete[:limit]
    start = time.monotonic()

    for index, patient in enumerate(complete, start=1):
        files = patients[patient]
        try:
            upper_data = archive.read(files["upper"])
            lower_data = archive.read(files["lower"])
            upper_points, upper_triangles = sample_binary_stl(upper_data, max_triangles)
            lower_points, lower_triangles = sample_binary_stl(lower_data, max_triangles)
            row = pair_stats(
                patient,
                mesh_stats(upper_points, upper_triangles),
                mesh_stats(lower_points, lower_triangles),
            )
            rows.append(row)
        except Exception as error:
            errors.append({"patient": patient, "error": repr(error)})

        if index == 1 or index % 25 == 0 or index == len(complete):
            elapsed = time.monotonic() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (len(complete) - index) / rate if rate > 0 else float("nan")
            print(
                f"[{index}/{len(complete)}] rows={len(rows)} errors={len(errors)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return rows, errors


def apply_flags(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metrics = [
        "combined_span_x",
        "combined_span_y",
        "combined_span_z",
        "x_to_y_ratio",
        "layout_ratio",
        "upper_minus_lower_vertical",
        "upper_lower_x_center_gap",
        "upper_lower_y_center_gap",
        "upper_lower_z_center_gap",
        "x_overlap_ratio",
        "y_overlap_ratio",
        "z_overlap_ratio",
        "depth_overlap_ratio",
        "anterior_axis_log_score",
        "front_angle_degrees",
        "front_upper_lower_agreement",
        "front_shape_confidence",
    ]
    summaries = {
        metric: robust_summary([float(row[metric]) for row in rows]) for metric in metrics
    }

    for row in rows:
        flags: list[str] = []
        severe: list[str] = []

        if row["axis_layout"] != "XY_arch_Z_vertical":
            severe.append("axis_layout_not_ras")
        if float(row["upper_minus_lower_vertical"]) <= 0:
            severe.append("upper_not_superior")
        if float(row["x_overlap_ratio"]) < 0.55:
            severe.append("poor_upper_lower_x_overlap")
        if float(row["depth_overlap_ratio"]) < 0.55:
            severe.append("poor_upper_lower_depth_overlap")
        if row["axis_layout"] == "XY_arch_Z_vertical":
            angle = abs(float(row["front_angle_degrees"]))
            if angle >= 150.0:
                flags.append("anterior_near_negative_y")
            elif angle > 30.0:
                flags.append("anterior_not_aligned_positive_y")
        if float(row["front_upper_lower_agreement"]) < 0.50:
            flags.append("upper_lower_front_direction_disagreement")
        if float(row["layout_ratio"]) < 1.15:
            flags.append("ambiguous_yz_axis_roles")

        for metric in (
            "combined_span_x",
            "combined_span_y",
            "combined_span_z",
            "upper_lower_x_center_gap",
            "upper_lower_y_center_gap",
            "upper_lower_z_center_gap",
        ):
            score = abs(robust_z(float(row[metric]), summaries[metric]))
            if score > 8.0:
                severe.append(f"extreme_{metric}")
            elif score > 5.0:
                flags.append(f"outlier_{metric}")

        all_flags = severe + flags
        row["flags"] = ";".join(all_flags)
        row["severity"] = "severe" if severe else ("review" if flags else "ok")
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_for_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    return value


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def save_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    payload = PNG_SIGNATURE
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        set_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def render_projection_page(
    path: Path,
    cases: list[tuple[str, list[tuple[float, float, float]], list[tuple[float, float, float]]]],
) -> None:
    panel_width = 320
    row_height = 230
    margin = 18
    width = panel_width * 3
    height = row_height * len(cases)
    pixels = bytearray([255] * (width * height * 3))
    projections = [(0, 1), (0, 2), (1, 2)]
    upper_color = (215, 55, 55)
    lower_color = (45, 80, 220)
    border = (205, 205, 205)

    for row_index, (_, upper_points, lower_points) in enumerate(cases):
        all_points = upper_points + lower_points
        for panel_index, (axis_x, axis_y) in enumerate(projections):
            x0 = panel_index * panel_width
            y0 = row_index * row_height
            draw_line(pixels, width, height, x0, y0, x0 + panel_width - 1, y0, border)
            draw_line(pixels, width, height, x0, y0, x0, y0 + row_height - 1, border)
            values_x = sorted(point[axis_x] for point in all_points)
            values_y = sorted(point[axis_y] for point in all_points)
            minimum_x = percentile(values_x, 0.01)
            maximum_x = percentile(values_x, 0.99)
            minimum_y = percentile(values_y, 0.01)
            maximum_y = percentile(values_y, 0.99)
            span_x = max(maximum_x - minimum_x, 1e-6)
            span_y = max(maximum_y - minimum_y, 1e-6)

            def draw_points(points, color):
                stride = max(1, len(points) // 2500)
                for point in points[::stride]:
                    px = x0 + margin + int((point[axis_x] - minimum_x) / span_x * (panel_width - 2 * margin))
                    py = y0 + row_height - margin - int((point[axis_y] - minimum_y) / span_y * (row_height - 2 * margin))
                    set_pixel(pixels, width, height, px, py, color)

            draw_points(lower_points, lower_color)
            draw_points(upper_points, upper_color)

    save_png(path, width, height, pixels)


def render_cases(
    archive: zipfile.ZipFile,
    patients: dict[str, dict[str, str]],
    selected: list[str],
    out_dir: Path,
    max_triangles: int,
    cases_per_page: int,
) -> list[dict[str, object]]:
    rendered: list[tuple[str, list[tuple[float, float, float]], list[tuple[float, float, float]]]] = []
    for patient in selected:
        files = patients[patient]
        upper_points, _ = sample_binary_stl(archive.read(files["upper"]), max_triangles)
        lower_points, _ = sample_binary_stl(archive.read(files["lower"]), max_triangles)
        rendered.append((patient, upper_points, lower_points))

    manifest: list[dict[str, object]] = []
    for page_index in range(0, len(rendered), cases_per_page):
        page_cases = rendered[page_index : page_index + cases_per_page]
        page_number = page_index // cases_per_page + 1
        filename = f"ras_projection_page_{page_number:02d}.png"
        render_projection_page(out_dir / filename, page_cases)
        manifest.append(
            {
                "file": filename,
                "rows_top_to_bottom": [case[0] for case in page_cases],
                "columns_left_to_right": ["XY (+Y up)", "XZ (+Z up)", "YZ (+Z up)"],
                "colors": {"upper": "red", "lower": "blue"},
            }
        )
    return manifest


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Opening {args.zip_path}", flush=True)

    with zipfile.ZipFile(args.zip_path) as archive:
        patients, unclassified = build_archive_index(archive)
        incomplete = {
            patient: sorted(files.keys())
            for patient, files in sorted(patients.items())
            if not {"upper", "lower"} <= files.keys()
        }
        complete_count = sum({"upper", "lower"} <= files.keys() for files in patients.values())
        print(
            f"Indexed {len(patients)} patients with IOS entries; complete pairs={complete_count}; "
            f"incomplete={len(incomplete)}",
            flush=True,
        )

        rows, errors = audit_rows(archive, patients, args.max_triangles, args.limit)
        summaries = apply_flags(rows)
        rows.sort(key=lambda row: (0 if row["severity"] == "severe" else 1 if row["severity"] == "review" else 2, row["patient"]))
        write_csv(args.out_dir / "ras_audit.csv", rows)

        severity_counts = Counter(str(row["severity"]) for row in rows)
        signature_counts = Counter(str(row["orientation_signature"]) for row in rows)
        flag_counts = Counter(
            flag
            for row in rows
            for flag in str(row["flags"]).split(";")
            if flag
        )

        flagged = [str(row["patient"]) for row in rows if row["severity"] != "ok"]
        good = [str(row["patient"]) for row in rows if row["severity"] == "ok"]
        rng = random.Random(args.seed)
        rng.shuffle(good)
        selected = flagged[: args.render_cases]
        if len(selected) < args.render_cases:
            selected.extend(good[: args.render_cases - len(selected)])
        render_manifest = render_cases(
            archive,
            patients,
            selected,
            args.out_dir,
            min(args.max_triangles, 3000),
            args.cases_per_page,
        )

    summary = {
        "archive": str(args.zip_path),
        "elapsed_seconds": time.monotonic() - started,
        "patients_with_ios_entries": len(patients),
        "complete_pairs_indexed": complete_count,
        "rows_analyzed": len(rows),
        "incomplete_pairs": incomplete,
        "unclassified_stl_candidates": unclassified,
        "parse_errors": errors,
        "severity_counts": dict(severity_counts),
        "orientation_signature_counts": dict(signature_counts),
        "flag_counts": dict(flag_counts),
        "metric_summaries": summaries,
        "render_manifest": render_manifest,
        "interpretation_limits": [
            "Unlabelled near-symmetric geometry cannot prove whether +X is patient-right rather than patient-left.",
            "The +Y anterior check is a geometric U-shape proxy and requires visual confirmation for flagged cases.",
            "Numeric consistency does not prove that every clinical landmark is anatomically correct.",
        ],
    }
    (args.out_dir / "ras_summary.json").write_text(
        json.dumps(clean_for_json(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "flagged_cases.txt").write_text("\n".join(flagged) + "\n", encoding="utf-8")
    print(json.dumps(clean_for_json(summary), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
