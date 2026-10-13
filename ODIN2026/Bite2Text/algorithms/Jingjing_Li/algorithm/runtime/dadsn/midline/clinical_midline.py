"""按临床正面观察方式计算上下颌牙齿中线。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

import crowding
import midline


@dataclass(frozen=True)
class FrontalRaster:
    """从患者正面观察得到的正交投影标签图和深度图。"""

    labels: np.ndarray
    depth: np.ndarray
    x_min_mm: float
    y_min_mm: float
    resolution_mm: float
    tooth_ids: tuple[int, int]

    def world_to_pixel(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        column = int(math.floor((x_mm - self.x_min_mm) / self.resolution_mm))
        row = int(math.floor((y_mm - self.y_min_mm) / self.resolution_mm))
        return row, column


@dataclass(frozen=True)
class VisibleMidline:
    """正面可见中切牙接触区拟合结果。"""

    arch: str
    incisal_point_2d: np.ndarray
    line_start_2d: np.ndarray
    line_end_2d: np.ndarray
    seam_points_2d: np.ndarray
    slope_dx_dy: float
    intercept_mm: float
    angle_deg: float
    row_count: int
    rmse_mm: float
    median_visible_gap_mm: float
    p95_visible_gap_mm: float


def rasterize_visible_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    tooth_ids: tuple[int, int],
    resolution_mm: float = 0.06,
    padding_mm: float = 0.15,
) -> FrontalRaster:
    """将三维网格正交投影；每个像素只保留最靠患者前方的牙面。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_labels = np.asarray(vertex_labels)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    if len(vertex_labels) != len(vertices):
        raise ValueError("vertex_labels length must match vertices")
    if resolution_mm <= 0 or padding_mm < 0:
        raise ValueError("resolution_mm must be positive and padding_mm non-negative")

    tooth_mask = np.isin(vertex_labels, tooth_ids)
    if not np.any(tooth_mask):
        raise ValueError(f"No vertices found for teeth {tooth_ids}")
    projected = vertices[tooth_mask, :2]
    minimum = projected.min(axis=0) - padding_mm
    maximum = projected.max(axis=0) + padding_mm
    width, height = np.maximum(1, np.ceil((maximum - minimum) / resolution_mm).astype(int))
    depth = np.full((height, width), -np.inf, dtype=np.float64)
    image = np.zeros((height, width), dtype=np.int16)

    face_labels = vertex_labels[faces]
    valid_faces = (
        np.all(face_labels == face_labels[:, :1], axis=1)
        & np.isin(face_labels[:, 0], tooth_ids)
    )
    for face, label in zip(faces[valid_faces], face_labels[valid_faces, 0]):
        triangle = vertices[face]
        projected_triangle = (triangle[:, :2] - minimum) / resolution_mm
        x0 = max(0, int(np.floor(projected_triangle[:, 0].min())))
        x1 = min(width - 1, int(np.ceil(projected_triangle[:, 0].max())))
        y0 = max(0, int(np.floor(projected_triangle[:, 1].min())))
        y1 = min(height - 1, int(np.ceil(projected_triangle[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue

        pixel_x, pixel_y = np.meshgrid(
            np.arange(x0, x1 + 1, dtype=np.float64) + 0.5,
            np.arange(y0, y1 + 1, dtype=np.float64) + 0.5,
        )
        denominator = (
            (projected_triangle[1, 1] - projected_triangle[2, 1])
            * (projected_triangle[0, 0] - projected_triangle[2, 0])
            + (projected_triangle[2, 0] - projected_triangle[1, 0])
            * (projected_triangle[0, 1] - projected_triangle[2, 1])
        )
        if abs(denominator) < 1e-12:
            continue
        weight_a = (
            (projected_triangle[1, 1] - projected_triangle[2, 1])
            * (pixel_x - projected_triangle[2, 0])
            + (projected_triangle[2, 0] - projected_triangle[1, 0])
            * (pixel_y - projected_triangle[2, 1])
        ) / denominator
        weight_b = (
            (projected_triangle[2, 1] - projected_triangle[0, 1])
            * (pixel_x - projected_triangle[2, 0])
            + (projected_triangle[0, 0] - projected_triangle[2, 0])
            * (pixel_y - projected_triangle[2, 1])
        ) / denominator
        weight_c = 1.0 - weight_a - weight_b
        inside = (
            (weight_a >= -1e-9)
            & (weight_b >= -1e-9)
            & (weight_c >= -1e-9)
        )
        triangle_depth = (
            weight_a * triangle[0, 2]
            + weight_b * triangle[1, 2]
            + weight_c * triangle[2, 2]
        )
        depth_view = depth[y0:y1 + 1, x0:x1 + 1]
        label_view = image[y0:y1 + 1, x0:x1 + 1]
        update = inside & (triangle_depth > depth_view)
        depth_view[update] = triangle_depth[update]
        label_view[update] = int(label)

    return FrontalRaster(
        labels=image,
        depth=depth,
        x_min_mm=float(minimum[0]),
        y_min_mm=float(minimum[1]),
        resolution_mm=float(resolution_mm),
        tooth_ids=tooth_ids,
    )


def extract_visible_contact_midline(
    raster: FrontalRaster,
    arch: str,
    left_tooth: int,
    right_tooth: int,
    max_visible_gap_mm: float = 0.5,
    incisal_fraction: float = 0.1,
) -> VisibleMidline:
    """从正面标签图逐行寻找两颗中切牙的内侧可见边缘。"""
    if arch not in {"upper", "lower"}:
        raise ValueError("arch must be upper or lower")
    if max_visible_gap_mm < 0:
        raise ValueError("max_visible_gap_mm must be non-negative")
    if not 0 < incisal_fraction <= 0.5:
        raise ValueError("incisal_fraction must be in (0, 0.5]")

    resolution = raster.resolution_mm
    seam_rows = []
    for row_index, row_labels in enumerate(raster.labels):
        right_columns = np.flatnonzero(row_labels == right_tooth)
        left_columns = np.flatnonzero(row_labels == left_tooth)
        if len(right_columns) == 0 or len(left_columns) == 0:
            continue
        right_inner = int(right_columns.max())
        left_inner = int(left_columns.min())
        if right_inner >= left_inner:
            continue
        visible_gap = max(0.0, (left_inner - right_inner - 1) * resolution)
        if visible_gap > max_visible_gap_mm + resolution * 0.1:
            continue
        x_right = raster.x_min_mm + (right_inner + 0.5) * resolution
        x_left = raster.x_min_mm + (left_inner + 0.5) * resolution
        y = raster.y_min_mm + (row_index + 0.5) * resolution
        seam_rows.append(((x_left + x_right) / 2.0, y, visible_gap))

    if len(seam_rows) < 12:
        raise ValueError(f"Only {len(seam_rows)} visible contact rows were found")
    seam_data = np.asarray(seam_rows, dtype=np.float64)
    keep = np.ones(len(seam_data), dtype=bool)
    for _ in range(5):
        slope, intercept = np.polyfit(seam_data[keep, 1], seam_data[keep, 0], 1)
        residuals = seam_data[:, 0] - (slope * seam_data[:, 1] + intercept)
        residual_center = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - residual_center)))
        threshold = max(3.0 * resolution, 3.0 * 1.4826 * mad)
        next_keep = np.abs(residuals - residual_center) <= threshold
        if np.count_nonzero(next_keep) < 12 or np.array_equal(next_keep, keep):
            break
        keep = next_keep

    points = seam_data[keep, :2]
    gaps = seam_data[keep, 2]
    slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
    residuals = points[:, 0] - (slope * points[:, 1] + intercept)
    low_y, high_y = np.quantile(points[:, 1], (0.02, 0.98))
    line_start = np.array([slope * low_y + intercept, low_y])
    line_end = np.array([slope * high_y + intercept, high_y])

    incisal_quantile = incisal_fraction if arch == "upper" else 1.0 - incisal_fraction
    incisal_threshold = float(np.quantile(points[:, 1], incisal_quantile))
    incisal_mask = (
        points[:, 1] <= incisal_threshold
        if arch == "upper"
        else points[:, 1] >= incisal_threshold
    )
    incisal_y = float(np.median(points[incisal_mask, 1]))
    incisal_point = np.array([slope * incisal_y + intercept, incisal_y])

    return VisibleMidline(
        arch=arch,
        incisal_point_2d=incisal_point,
        line_start_2d=line_start,
        line_end_2d=line_end,
        seam_points_2d=points,
        slope_dx_dy=float(slope),
        intercept_mm=float(intercept),
        angle_deg=float(math.degrees(math.atan(slope))),
        row_count=len(points),
        rmse_mm=float(np.sqrt(np.mean(residuals ** 2))),
        median_visible_gap_mm=float(np.median(gaps)),
        p95_visible_gap_mm=float(np.quantile(gaps, 0.95)),
    )


def axial_angle_difference_deg(first_deg: float, second_deg: float) -> float:
    """比较无方向直线的最小夹角，结果范围为 0 至 90 度。"""
    difference = abs((float(second_deg) - float(first_deg)) % 180.0)
    return min(difference, 180.0 - difference)


def visible_midline_to_result(value: VisibleMidline) -> dict[str, object]:
    return {
        "incisal_midline_point_frontal_mm": value.incisal_point_2d.tolist(),
        "visible_midline_start_frontal_mm": value.line_start_2d.tolist(),
        "visible_midline_end_frontal_mm": value.line_end_2d.tolist(),
        "visible_midline_angle_deg": value.angle_deg,
        "visible_contact_row_count": value.row_count,
        "line_rmse_mm": value.rmse_mm,
        "visible_gap_median_mm": value.median_visible_gap_mm,
        "visible_gap_p95_mm": value.p95_visible_gap_mm,
    }


def calculate_clinical_midlines(
    upper_stl: Path,
    upper_labels: Path,
    lower_stl: Path,
    lower_labels: Path,
    resolution_mm: float = 0.06,
    max_visible_gap_mm: float = 0.5,
) -> tuple[dict[str, object], dict[str, FrontalRaster]]:
    """以同一正面正交视图计算上下中切牙可见接触区的关系。"""
    upper_mesh = crowding.load_binary_stl_stable(upper_stl)
    lower_mesh = crowding.load_binary_stl_stable(lower_stl)
    upper_label_values = crowding.load_labels(upper_labels)
    lower_label_values = crowding.load_labels(lower_labels)
    for arch, mesh, labels in (
        ("upper", upper_mesh, upper_label_values),
        ("lower", lower_mesh, lower_label_values),
    ):
        if len(mesh.vertices) != len(labels):
            raise ValueError(
                f"{arch} STL stable vertices and JSON labels differ: "
                f"{len(mesh.vertices)} != {len(labels)}"
            )

    upper_centers = midline.tooth_centers(upper_mesh.vertices, upper_label_values)
    lower_centers = midline.tooth_centers(lower_mesh.vertices, lower_label_values)
    frame = midline.build_shared_frame(upper_centers, lower_centers)
    upper_vertices = midline.anatomical_coordinates(upper_mesh.vertices, frame)
    lower_vertices = midline.anatomical_coordinates(lower_mesh.vertices, frame)
    upper_raster = rasterize_visible_labels(
        upper_vertices,
        upper_mesh.faces,
        upper_label_values,
        tooth_ids=(21, 11),
        resolution_mm=resolution_mm,
    )
    lower_raster = rasterize_visible_labels(
        lower_vertices,
        lower_mesh.faces,
        lower_label_values,
        tooth_ids=(31, 41),
        resolution_mm=resolution_mm,
    )
    upper_midline = extract_visible_contact_midline(
        upper_raster,
        arch="upper",
        left_tooth=21,
        right_tooth=11,
        max_visible_gap_mm=max_visible_gap_mm,
    )
    lower_midline = extract_visible_contact_midline(
        lower_raster,
        arch="lower",
        left_tooth=31,
        right_tooth=41,
        max_visible_gap_mm=max_visible_gap_mm,
    )

    lateral_offset = float(
        lower_midline.incisal_point_2d[0] - upper_midline.incisal_point_2d[0]
    )
    angle_difference = axial_angle_difference_deg(
        upper_midline.angle_deg, lower_midline.angle_deg
    )
    quality_issues = []
    for value in (upper_midline, lower_midline):
        if value.row_count < 40:
            quality_issues.append(
                f"{value.arch} visible contact has fewer than 40 raster rows"
            )
        if value.rmse_mm > 0.25:
            quality_issues.append(
                f"{value.arch} visible midline RMSE exceeds 0.25 mm"
            )

    result = {
        "case": "case",
        "method": {
            "name": "clinical_frontal_visible_contact_approximation",
            "definition": (
                "Registered central incisors are orthographically projected from "
                "the patient-front direction. A depth buffer keeps only visible "
                "labial surfaces; the maxillary and mandibular midlines are fitted "
                "from the visible contact regions of 21/11 and 31/41."
            ),
            "scope": "upper-to-lower dental midline; not a facial midline",
            "resolution_mm_per_pixel": resolution_mm,
            "maximum_visible_contact_gap_mm": max_visible_gap_mm,
            "incisal_fraction": 0.1,
            "source_basis": [
                "https://ecommons.aku.edu/pakistan_fhs_mc_surg_dent_oral_maxillofac/100",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC11470451/",
            ],
        },
        "input": {
            "upper_stl": str(upper_stl),
            "upper_labels": str(upper_labels),
            "lower_stl": str(lower_stl),
            "lower_labels": str(lower_labels),
            "units": "millimeters_assumed",
            "registered_common_coordinates_assumed": True,
        },
        "shared_frontal_frame": {
            "x": "patient_left",
            "y": "superior",
            "view_direction": "from_anterior_toward_posterior",
            "projection": "orthographic",
            "origin_3d": frame.origin.tolist(),
            "patient_left_axis_3d": frame.left_axis.tolist(),
            "anterior_axis_3d": frame.anterior_axis.tolist(),
            "superior_axis_3d": frame.superior_axis.tolist(),
        },
        "upper": visible_midline_to_result(upper_midline),
        "lower": visible_midline_to_result(lower_midline),
        "comparison": {
            "lower_relative_to_upper_lateral_mm": lateral_offset,
            "absolute_lateral_offset_mm": abs(lateral_offset),
            "direction": midline.offset_direction(lateral_offset),
            "visible_midline_angle_difference_deg": angle_difference,
            "sign_convention": (
                "positive means the lower visible dental midline is toward "
                "patient left relative to the upper visible dental midline"
            ),
        },
        "raster_quality": {
            "upper_shape": list(upper_raster.labels.shape),
            "lower_shape": list(lower_raster.labels.shape),
            "upper_visible_pixels": int(np.count_nonzero(upper_raster.labels)),
            "lower_visible_pixels": int(np.count_nonzero(lower_raster.labels)),
        },
        "quality": {
            "status": "review" if quality_issues else "acceptable",
            "issues": quality_issues,
            "clinical_validation_required": True,
        },
    }
    return result, {"upper": upper_raster, "lower": lower_raster}


TOOTH_COLORS = {
    21: "#d55e00",
    11: "#e6a700",
    31: "#0072b2",
    41: "#009e73",
}


def _raster_world_bounds(raster: FrontalRaster) -> tuple[float, float, float, float]:
    rows, columns = np.nonzero(raster.labels)
    if len(rows) == 0:
        raise ValueError("Visible-label raster is empty")
    resolution = raster.resolution_mm
    x0 = raster.x_min_mm + float(columns.min()) * resolution
    x1 = raster.x_min_mm + float(columns.max() + 1) * resolution
    y0 = raster.y_min_mm + float(rows.min()) * resolution
    y1 = raster.y_min_mm + float(rows.max() + 1) * resolution
    return x0, y0, x1, y1


def _svg_viewbox(
    rasters: dict[str, FrontalRaster], arches: tuple[str, ...], margin_mm: float = 1.0
) -> str:
    bounds = [_raster_world_bounds(rasters[arch]) for arch in arches]
    x0 = min(value[0] for value in bounds) - margin_mm
    y0 = min(value[1] for value in bounds) - margin_mm
    x1 = max(value[2] for value in bounds) + margin_mm
    y1 = max(value[3] for value in bounds) + margin_mm
    return f"{x0:.4f} {-y1:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"


def _tooth_path(raster: FrontalRaster, tooth: int) -> str:
    """按行程编码标签像素，避免为每个像素生成一个 SVG 元素。"""
    commands = []
    resolution = raster.resolution_mm
    for row_index, row_labels in enumerate(raster.labels):
        columns = np.flatnonzero(row_labels == tooth)
        if len(columns) == 0:
            continue
        run_starts = np.r_[0, np.flatnonzero(np.diff(columns) > 1) + 1]
        run_ends = np.r_[run_starts[1:] - 1, len(columns) - 1]
        y = -(raster.y_min_mm + (row_index + 1) * resolution)
        for start_index, end_index in zip(run_starts, run_ends):
            first = int(columns[start_index])
            last = int(columns[end_index])
            x = raster.x_min_mm + first * resolution
            width = (last - first + 1) * resolution
            commands.append(
                f"M{x:.4f},{y:.4f}h{width:.4f}v{resolution:.4f}"
                f"h{-width:.4f}z"
            )
    return "".join(commands)


def render_clinical_svg(
    result: dict[str, object], rasters: dict[str, FrontalRaster]
) -> str:
    """绘制与计算共用同一深度缓冲标签图的正面可视化。"""
    combined_viewbox = _svg_viewbox(rasters, ("upper", "lower"), 1.2)
    upper_viewbox = _svg_viewbox(rasters, ("upper",), 0.8)
    lower_viewbox = _svg_viewbox(rasters, ("lower",), 0.8)
    tooth_groups = []
    annotation_groups = []
    # 下颌牙面先画、上颌牙面后画；两颌标注统一置于所有牙面之上。
    for arch in ("lower", "upper"):
        raster = rasters[arch]
        tooth_paths = []
        for tooth in raster.tooth_ids:
            tooth_paths.append(
                f'<path data-tooth="{tooth}" d="{_tooth_path(raster, tooth)}" '
                f'fill="{TOOTH_COLORS[tooth]}"/>'
            )
        tooth_groups.append(
            f'<g data-arch="{arch}">{"".join(tooth_paths)}</g>'
        )
        arch_result = result[arch]
        line_start = arch_result["visible_midline_start_frontal_mm"]
        line_end = arch_result["visible_midline_end_frontal_mm"]
        incisal = arch_result["incisal_midline_point_frontal_mm"]
        annotation_groups.append(
            f'''<g data-arch="{arch}">
<line class="clinical-midline" x1="{line_start[0]:.4f}" y1="{-line_start[1]:.4f}" x2="{line_end[0]:.4f}" y2="{-line_end[1]:.4f}"/>
<circle class="incisal-point" cx="{incisal[0]:.4f}" cy="{-incisal[1]:.4f}" r="0.22"/>
</g>'''
        )
    return f'''<svg id="clinical-view" xmlns="http://www.w3.org/2000/svg"
 viewBox="{combined_viewbox}" preserveAspectRatio="xMidYMid meet"
 data-combined-viewbox="{combined_viewbox}"
 data-upper-viewbox="{upper_viewbox}"
 data-lower-viewbox="{lower_viewbox}"
 role="img" aria-labelledby="clinical-title clinical-description">
<title id="clinical-title">case upper and lower dental midlines from the clinical frontal view</title>
<desc id="clinical-description">Front-visible central incisors with red inter-incisor midlines and yellow incisal midline points.</desc>
<style>
.clinical-midline{{stroke:#e53935;stroke-width:.08;stroke-linecap:round}}
.incisal-point{{fill:#f5c542;stroke:#1f2933;stroke-width:.04}}
</style>
<rect x="-1000" y="-1000" width="2000" height="2000" fill="#e8edf0"/>
{"".join(tooth_groups)}
{"".join(annotation_groups)}
</svg>'''


def render_clinical_html(
    result: dict[str, object], rasters: dict[str, FrontalRaster]
) -> str:
    comparison = result["comparison"]
    svg = render_clinical_svg(result, rasters)
    direction = str(comparison["direction"]).replace("_", " ")
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>case clinical frontal dental midline</title>
<link rel="icon" href="data:,">
<style>
:root{{--panel:300px;--ink:#182026;--muted:#5f6b73;--line:#c8d0d6;--surface:#fff}}
*{{box-sizing:border-box}}
html,body{{width:100%;height:100%;margin:0;overflow:hidden}}
body{{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;background:#e8edf0;color:var(--ink)}}
#stage{{position:fixed;inset:0 0 0 var(--panel);display:grid;place-items:center;padding:64px 24px 24px}}
#stage svg{{width:100%;height:100%;min-width:0;min-height:0}}
.panel{{position:fixed;inset:0 auto 0 0;width:var(--panel);padding:18px;background:var(--surface);border-right:1px solid var(--line);z-index:3;overflow:auto}}
h1{{font-size:20px;line-height:1.2;margin:0 0 4px}}
.subtitle{{margin:0 0 20px;color:var(--muted)}}
.metric{{padding:12px 0;border-top:1px solid #e2e6e9}}
.metric strong{{display:block;font-size:24px;font-variant-numeric:tabular-nums}}
.metric span{{color:var(--muted)}}
.legend{{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}}
.legend div{{display:flex;align-items:center;gap:8px;margin:8px 0}}
.swatch{{width:22px;height:4px;background:var(--color)}}
.toolbar{{position:fixed;top:16px;left:calc(var(--panel) + 16px);display:flex;gap:6px;z-index:4}}
button{{min-height:36px;border:1px solid #8f9aa3;border-radius:4px;background:#fff;color:var(--ink);padding:7px 11px;cursor:pointer}}
button[aria-pressed="true"]{{background:#1f2933;color:#fff;border-color:#1f2933}}
button:focus-visible{{outline:3px solid #0072b2;outline-offset:2px}}
@media(max-width:720px){{:root{{--panel:0px}}.panel{{inset:auto 0 0 0;width:auto;height:184px;border:0;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:0 16px;padding:12px 16px}}.panel h1,.panel .subtitle,.panel .legend{{display:none}}#stage{{inset:0 0 184px 0;padding:56px 10px 10px}}.toolbar{{left:10px;right:10px;top:10px}}.toolbar button{{min-height:32px;padding:5px 8px;font-size:12px}}.metric{{padding:8px 0;min-width:0}}.metric strong{{font-size:19px}}.metric span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}}}
</style>
</head>
<body>
<aside class="panel" aria-labelledby="title">
<h1 id="title">case dental midline</h1>
<p class="subtitle">Clinical frontal visible-contact method</p>
<div class="metric"><strong>{comparison['absolute_lateral_offset_mm']:.2f} mm</strong><span>lower relative to upper</span></div>
<div class="metric"><strong>{direction}</strong><span>lateral direction</span></div>
<div class="metric"><strong>{comparison['visible_midline_angle_difference_deg']:.2f} deg</strong><span>visible midline angle difference</span></div>
<div class="metric"><strong>{result['quality']['status']}</strong><span>geometric quality status</span></div>
<div class="legend" aria-label="Annotation legend">
<div><i class="swatch" style="--color:#e53935"></i>Visible dental midline</div>
<div><i class="swatch" style="--color:#f5c542"></i>Incisal midline point</div>
</div>
</aside>
<main id="stage">{svg}</main>
<nav class="toolbar" aria-label="Frontal views">
<button type="button" data-view="combined" aria-pressed="true">Combined</button>
<button type="button" data-view="upper" aria-pressed="false">Upper</button>
<button type="button" data-view="lower" aria-pressed="false">Lower</button>
</nav>
<script>
const svg=document.getElementById('clinical-view');
function applyView(name){{
  svg.querySelectorAll('[data-arch="upper"]').forEach(group=>group.style.display=name==='lower'?'none':'');
  svg.querySelectorAll('[data-arch="lower"]').forEach(group=>group.style.display=name==='upper'?'none':'');
  svg.setAttribute('viewBox',svg.dataset[`${{name}}Viewbox`]);
  document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===name)));
}}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>applyView(button.dataset.view)));
window.__CLINICAL_MIDLINE_VIEWER__={{ready:true,applyView}};
</script>
</body>
</html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-stl", type=Path, default=Path("2/ios_upper.stl"))
    parser.add_argument(
        "--upper-labels", type=Path, default=Path("2/case_ios_upper.json")
    )
    parser.add_argument("--lower-stl", type=Path, default=Path("2/ios_lower.stl"))
    parser.add_argument(
        "--lower-labels", type=Path, default=Path("2/case_ios_lower.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("2/case_clinical_midline.json")
    )
    parser.add_argument(
        "--svg", type=Path, default=Path("2/case_clinical_midline.svg")
    )
    parser.add_argument(
        "--viewer",
        type=Path,
        default=Path("2/case_clinical_midline_viewer.html"),
    )
    args = parser.parse_args()
    result, rasters = calculate_clinical_midlines(
        args.upper_stl,
        args.upper_labels,
        args.lower_stl,
        args.lower_labels,
    )
    for path in (args.output, args.svg, args.viewer):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.svg.write_text(render_clinical_svg(result, rasters), encoding="utf-8")
    args.viewer.write_text(render_clinical_html(result, rasters), encoding="utf-8")
    comparison = result["comparison"]
    print("case clinical frontal dental midline")
    print(
        "Lower relative to upper: "
        f"{comparison['absolute_lateral_offset_mm']:.2f} mm toward "
        f"{str(comparison['direction']).replace('_', ' ')}"
    )
    print(
        "Visible midline angle difference: "
        f"{comparison['visible_midline_angle_difference_deg']:.2f} deg"
    )
    print(f"Quality: {result['quality']['status']}")
    print(f"JSON: {args.output}")
    print(f"SVG: {args.svg}")
    print(f"Viewer: {args.viewer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
