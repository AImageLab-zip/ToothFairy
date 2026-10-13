#!/usr/bin/env python3
"""Calculate upper/lower dental midlines from registered STL tooth crowns."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

import crowding


THREE_VERSION = "0.180.0"
CENTRAL_TEETH = {
    "upper": (21, 11),
    "lower": (31, 41),
}
MOLAR_TEETH = {
    "upper": (26, 16),
    "lower": (36, 46),
}


@dataclass(frozen=True)
class AnatomicalFrame:
    """Shared patient-left, anterior, and superior coordinate frame."""

    origin: np.ndarray
    left_axis: np.ndarray
    anterior_axis: np.ndarray
    superior_axis: np.ndarray


@dataclass(frozen=True)
class InterfaceLandmark:
    """Geometry extracted from the interface between two central incisors."""

    incisal_point: np.ndarray
    interface_line_start: np.ndarray
    interface_line_end: np.ndarray
    candidate_midpoints: np.ndarray
    candidate_count: int
    incisal_sample_count: int
    nearest_gap_median_mm: float
    nearest_gap_p95_mm: float
    line_rmse_mm: float


@dataclass(frozen=True)
class ArchMidline:
    """One arch's inter-incisor landmark and sagittal dental midline."""

    arch: str
    left_central: int
    right_central: int
    contact: InterfaceLandmark
    posterior_point: np.ndarray
    direction: np.ndarray
    line_start: np.ndarray
    line_end: np.ndarray
    yaw_deg: float


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def tooth_centers(vertices: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(tooth): np.median(vertices[labels == tooth], axis=0)
        for tooth in np.unique(labels)
        if tooth > 0
    }


def build_shared_frame(
    upper_centers: Mapping[int, np.ndarray],
    lower_centers: Mapping[int, np.ndarray],
) -> AnatomicalFrame:
    """Build a common anatomical frame from registered upper/lower teeth."""
    required_upper = (26, 16, 21, 11)
    required_lower = (36, 46, 31, 41)
    missing = [
        tooth
        for tooth in (*required_upper, *required_lower)
        if tooth not in (upper_centers if tooth < 30 else lower_centers)
    ]
    if missing:
        raise ValueError(f"Missing teeth required for the shared frame: {missing}")

    patient_left = np.mean(
        (upper_centers[26], lower_centers[36]), axis=0
    )
    patient_right = np.mean(
        (upper_centers[16], lower_centers[46]), axis=0
    )
    left_axis = normalize(patient_left - patient_right)

    posterior = np.mean(
        (
            upper_centers[26], upper_centers[16],
            lower_centers[36], lower_centers[46],
        ),
        axis=0,
    )
    anterior = np.mean(
        (
            upper_centers[21], upper_centers[11],
            lower_centers[31], lower_centers[41],
        ),
        axis=0,
    )
    anterior_vector = anterior - posterior
    anterior_vector -= left_axis * float(np.dot(anterior_vector, left_axis))
    anterior_axis = normalize(anterior_vector)
    superior_axis = normalize(np.cross(left_axis, anterior_axis))

    upper_front = np.mean((upper_centers[21], upper_centers[11]), axis=0)
    lower_front = np.mean((lower_centers[31], lower_centers[41]), axis=0)
    if float(np.dot(superior_axis, upper_front - lower_front)) < 0:
        superior_axis = -superior_axis

    return AnatomicalFrame(
        origin=posterior,
        left_axis=left_axis,
        anterior_axis=anterior_axis,
        superior_axis=superior_axis,
    )


def extract_interface_landmark(
    left_points: np.ndarray,
    right_points: np.ndarray,
    superior_axis: np.ndarray,
    arch: str,
    contact_fraction: float = 0.05,
    incisal_fraction: float = 0.12,
    minimum_candidates: int = 80,
) -> InterfaceLandmark:
    """Extract an incisal landmark from mutual near-surface midpoints."""
    if arch not in CENTRAL_TEETH:
        raise ValueError(f"Unsupported arch: {arch}")
    if not 0 < contact_fraction <= 1:
        raise ValueError("contact_fraction must be in (0, 1]")
    if not 0 < incisal_fraction <= 0.5:
        raise ValueError("incisal_fraction must be in (0, 0.5]")

    left_points = np.asarray(left_points, dtype=np.float64)
    right_points = np.asarray(right_points, dtype=np.float64)
    superior_axis = normalize(superior_axis)
    if len(left_points) < 20 or len(right_points) < 20:
        raise ValueError(
            "中切牙预测点数不足："
            f"left={len(left_points)}, right={len(right_points)}"
        )
    if not np.all(np.isfinite(left_points)) or not np.all(np.isfinite(right_points)):
        raise ValueError("中切牙点包含 NaN/Inf")
    if left_points.ndim != 2 or right_points.ndim != 2:
        raise ValueError("Tooth points must be two-dimensional arrays")
    if left_points.shape[1] != 3 or right_points.shape[1] != 3:
        raise ValueError("Tooth points must have three coordinates")

    right_tree = cKDTree(right_points)
    left_distances, right_indices = right_tree.query(left_points, k=1)
    left_midpoints = (left_points + right_points[right_indices]) / 2.0

    left_tree = cKDTree(left_points)
    right_distances, left_indices = left_tree.query(right_points, k=1)
    right_midpoints = (right_points + left_points[left_indices]) / 2.0

    distances = np.concatenate((left_distances, right_distances))
    midpoints = np.vstack((left_midpoints, right_midpoints))
    candidate_count = min(
        len(midpoints),
        max(minimum_candidates, int(math.ceil(contact_fraction * len(midpoints)))),
    )
    candidate_indices = np.argpartition(
        distances, candidate_count - 1
    )[:candidate_count]
    candidates = midpoints[candidate_indices]
    candidate_distances = distances[candidate_indices]
    if len(candidates) < 3:
        raise ValueError(
            f"中切牙接触候选点只有 {len(candidates)} 个，无法拟合界面"
        )

    center = np.median(candidates, axis=0)
    covariance = np.cov(candidates - center, rowvar=False)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("中切牙界面协方差无效")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    line_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(line_axis, superior_axis)) < 0:
        line_axis = -line_axis
    line_projections = (candidates - center) @ line_axis
    line_low, line_high = np.quantile(line_projections, (0.02, 0.98))
    line_start = center + line_axis * float(line_low)
    line_end = center + line_axis * float(line_high)
    line_residuals = np.linalg.norm(
        (candidates - center)
        - np.outer(line_projections, line_axis),
        axis=1,
    )

    heights = candidates @ superior_axis
    quantile = incisal_fraction if arch == "upper" else 1.0 - incisal_fraction
    threshold = float(np.quantile(heights, quantile))
    incisal_mask = heights <= threshold if arch == "upper" else heights >= threshold
    incisal_candidates = candidates[incisal_mask]
    incisal_point = np.median(incisal_candidates, axis=0)

    return InterfaceLandmark(
        incisal_point=incisal_point,
        interface_line_start=line_start,
        interface_line_end=line_end,
        candidate_midpoints=candidates,
        candidate_count=len(candidates),
        incisal_sample_count=len(incisal_candidates),
        nearest_gap_median_mm=float(np.median(candidate_distances)),
        nearest_gap_p95_mm=float(np.quantile(candidate_distances, 0.95)),
        line_rmse_mm=float(np.sqrt(np.mean(line_residuals ** 2))),
    )


def build_arch_midline(
    arch: str,
    mesh: crowding.MeshData,
    labels: np.ndarray,
    centers: Mapping[int, np.ndarray],
    frame: AnatomicalFrame,
) -> ArchMidline:
    left_central, right_central = CENTRAL_TEETH[arch]
    left_molar, right_molar = MOLAR_TEETH[arch]
    contact = extract_interface_landmark(
        mesh.vertices[labels == left_central],
        mesh.vertices[labels == right_central],
        frame.superior_axis,
        arch,
    )
    posterior = np.mean((centers[left_molar], centers[right_molar]), axis=0)
    direction = contact.incisal_point - posterior
    direction -= frame.superior_axis * float(np.dot(direction, frame.superior_axis))
    direction = normalize(direction)
    if float(np.dot(direction, frame.anterior_axis)) < 0:
        direction = -direction

    posterior_projection = float(np.dot(contact.incisal_point - posterior, direction))
    line_start = contact.incisal_point - direction * max(posterior_projection, 25.0)
    line_end = contact.incisal_point + direction * 6.0
    yaw = math.degrees(math.atan2(
        float(np.dot(direction, frame.left_axis)),
        float(np.dot(direction, frame.anterior_axis)),
    ))
    return ArchMidline(
        arch=arch,
        left_central=left_central,
        right_central=right_central,
        contact=contact,
        posterior_point=posterior,
        direction=direction,
        line_start=line_start,
        line_end=line_end,
        yaw_deg=yaw,
    )


def signed_lateral_offset(
    upper_point: np.ndarray,
    lower_point: np.ndarray,
    left_axis: np.ndarray,
) -> float:
    """Positive means the lower dental midline is toward patient left."""
    return float(np.dot(
        np.asarray(lower_point) - np.asarray(upper_point),
        normalize(left_axis),
    ))


def offset_direction(offset_mm: float, tolerance_mm: float = 0.05) -> str:
    if abs(offset_mm) <= tolerance_mm:
        return "coincident"
    return "patient_left" if offset_mm > 0 else "patient_right"


def anatomical_coordinates(points: np.ndarray, frame: AnatomicalFrame) -> np.ndarray:
    """Map registered STL coordinates to x=left, y=superior, z=anterior."""
    centered = np.asarray(points, dtype=np.float64) - frame.origin
    return np.column_stack(
        (
            centered @ frame.left_axis,
            centered @ frame.superior_axis,
            centered @ frame.anterior_axis,
        )
    )


def point_in_anatomical_frame(point: np.ndarray, frame: AnatomicalFrame) -> list[float]:
    return anatomical_coordinates(np.asarray(point)[None, :], frame)[0].tolist()


def compact_tooth_mesh(
    mesh: crowding.MeshData,
    labels: np.ndarray,
    tooth: int,
    frame: AnatomicalFrame,
) -> dict[str, object]:
    """Extract triangles wholly belonging to one tooth and compact the indices."""
    tooth_mask = labels == tooth
    selected_faces = mesh.faces[np.all(tooth_mask[mesh.faces], axis=1)]
    if len(selected_faces) == 0:
        raise ValueError(f"Tooth {tooth} has no complete triangles")
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    positions = anatomical_coordinates(mesh.vertices[used_vertices], frame)
    return {
        "label": tooth,
        "positions": np.round(positions, 5).reshape(-1).tolist(),
        "indices": inverse.reshape(-1).tolist(),
        "vertex_count": len(used_vertices),
        "triangle_count": len(selected_faces),
    }


def interface_to_result(contact: InterfaceLandmark) -> dict[str, object]:
    return {
        "incisal_midline_point_3d": contact.incisal_point.tolist(),
        "interface_line_start_3d": contact.interface_line_start.tolist(),
        "interface_line_end_3d": contact.interface_line_end.tolist(),
        "candidate_count": contact.candidate_count,
        "incisal_sample_count": contact.incisal_sample_count,
        "nearest_gap_median_mm": contact.nearest_gap_median_mm,
        "nearest_gap_p95_mm": contact.nearest_gap_p95_mm,
        "line_rmse_mm": contact.line_rmse_mm,
    }


def arch_to_result(midline: ArchMidline) -> dict[str, object]:
    return {
        "central_teeth": [midline.left_central, midline.right_central],
        "interface": interface_to_result(midline.contact),
        "posterior_arch_center_3d": midline.posterior_point.tolist(),
        "dental_midline": {
            "line_start_3d": midline.line_start.tolist(),
            "line_end_3d": midline.line_end.tolist(),
            "anterior_direction_3d": midline.direction.tolist(),
            "yaw_deg_from_shared_anterior_axis": midline.yaw_deg,
        },
    }


def build_scene_arch(
    arch: str,
    mesh: crowding.MeshData,
    labels: np.ndarray,
    midline: ArchMidline,
    frame: AnatomicalFrame,
) -> dict[str, object]:
    colors = {
        21: "#d55e00",
        11: "#e6a700",
        31: "#0072b2",
        41: "#009e73",
    }
    meshes = []
    for tooth in CENTRAL_TEETH[arch]:
        tooth_mesh = compact_tooth_mesh(mesh, labels, tooth, frame)
        tooth_mesh["color"] = colors[tooth]
        meshes.append(tooth_mesh)
    return {
        "meshes": meshes,
        "candidate_points": np.round(
            anatomical_coordinates(midline.contact.candidate_midpoints, frame), 5
        ).reshape(-1).tolist(),
        "interface_line": [
            point_in_anatomical_frame(midline.contact.interface_line_start, frame),
            point_in_anatomical_frame(midline.contact.interface_line_end, frame),
        ],
        "contact_point": point_in_anatomical_frame(
            midline.contact.incisal_point, frame
        ),
        "posterior_point": point_in_anatomical_frame(
            midline.posterior_point, frame
        ),
        "dental_midline": [
            point_in_anatomical_frame(
                midline.contact.incisal_point - midline.direction * 10.0, frame
            ),
            point_in_anatomical_frame(
                midline.contact.incisal_point + midline.direction * 6.0, frame
            ),
        ],
    }


def calculate_midlines(
    upper_stl: Path,
    upper_labels: Path,
    lower_stl: Path,
    lower_labels: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Calculate case upper/lower dental midlines in one registered frame."""
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

    upper_centers = tooth_centers(upper_mesh.vertices, upper_label_values)
    lower_centers = tooth_centers(lower_mesh.vertices, lower_label_values)
    frame = build_shared_frame(upper_centers, lower_centers)
    upper_midline = build_arch_midline(
        "upper", upper_mesh, upper_label_values, upper_centers, frame
    )
    lower_midline = build_arch_midline(
        "lower", lower_mesh, lower_label_values, lower_centers, frame
    )
    lateral_offset = signed_lateral_offset(
        upper_midline.contact.incisal_point,
        lower_midline.contact.incisal_point,
        frame.left_axis,
    )
    angle_difference = abs(
        (lower_midline.yaw_deg - upper_midline.yaw_deg + 180.0) % 360.0
        - 180.0
    )

    quality_issues = []
    for midline in (upper_midline, lower_midline):
        if midline.contact.nearest_gap_p95_mm > 0.75:
            quality_issues.append(
                f"{midline.arch} central-incisor interface gap p95 exceeds 0.75 mm"
            )
        if midline.contact.line_rmse_mm > 1.25:
            quality_issues.append(
                f"{midline.arch} interface line RMSE exceeds 1.25 mm"
            )

    result = {
        "case": "case",
        "method": {
            "name": "central_incisor_interface_and_arch_axis",
            "definition": (
                "The incisal midline point is the robust incisal endpoint of "
                "near-surface midpoints between the two central incisors. The "
                "dental midline runs from the bilateral first-molar midpoint "
                "through that incisal point."
            ),
            "scope": "upper-to-lower dental midline; not a facial midline",
            "contact_fraction": 0.05,
            "incisal_fraction": 0.12,
        },
        "input": {
            "upper_stl": str(upper_stl),
            "upper_labels": str(upper_labels),
            "lower_stl": str(lower_stl),
            "lower_labels": str(lower_labels),
            "units": "millimeters_assumed",
            "registered_common_coordinates_assumed": True,
        },
        "mapping": {
            "upper_stable_vertices": len(upper_mesh.vertices),
            "upper_json_labels": len(upper_label_values),
            "lower_stable_vertices": len(lower_mesh.vertices),
            "lower_json_labels": len(lower_label_values),
            "stable_order_mapping_valid": True,
        },
        "shared_frame": {
            "origin_3d": frame.origin.tolist(),
            "patient_left_axis_3d": frame.left_axis.tolist(),
            "anterior_axis_3d": frame.anterior_axis.tolist(),
            "superior_axis_3d": frame.superior_axis.tolist(),
        },
        "upper": arch_to_result(upper_midline),
        "lower": arch_to_result(lower_midline),
        "comparison": {
            "lower_relative_to_upper_lateral_mm": lateral_offset,
            "absolute_lateral_offset_mm": abs(lateral_offset),
            "direction": offset_direction(lateral_offset),
            "midline_angle_difference_deg": angle_difference,
            "sign_convention": (
                "positive means the lower dental midline is toward patient left "
                "relative to the upper dental midline"
            ),
        },
        "quality": {
            "status": "review" if quality_issues else "acceptable",
            "issues": quality_issues,
            "clinical_validation_required": True,
        },
    }
    scene_data = {
        "frame": {
            "x": "patient_left",
            "y": "superior",
            "z": "anterior",
        },
        "arches": {
            "upper": build_scene_arch(
                "upper", upper_mesh, upper_label_values, upper_midline, frame
            ),
            "lower": build_scene_arch(
                "lower", lower_mesh, lower_label_values, lower_midline, frame
            ),
        },
    }
    return result, scene_data


def render_viewer_html(
    result: dict[str, object],
    scene_data: dict[str, object],
    three_version: str = THREE_VERSION,
) -> str:
    """Render an interactive Three.js review page with embedded model data."""
    result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    scene_json = json.dumps(scene_data, ensure_ascii=False, separators=(",", ":"))
    for unsafe, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e")):
        result_json = result_json.replace(unsafe, escaped)
        scene_json = scene_json.replace(unsafe, escaped)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>case dental midline review</title>
<link rel="icon" href="data:,">
<script type="importmap">{{"imports":{{"three":"./vendor/three.module.min.js","three/addons/":"./vendor/"}}}}</script>
<style>
:root{{--panel:300px;--ink:#182026;--muted:#5f6b73;--line:#c8d0d6;--surface:#fff}}
*{{box-sizing:border-box}}
html,body{{width:100%;height:100%;margin:0;overflow:hidden}}
body{{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;background:#e8edf0;color:var(--ink)}}
#viewer{{position:fixed;inset:0 0 0 var(--panel)}}
.panel{{position:fixed;inset:0 auto 0 0;width:var(--panel);padding:18px;background:var(--surface);border-right:1px solid var(--line);z-index:3;overflow:auto}}
h1{{font-size:20px;line-height:1.2;margin:0 0 4px}}
.subtitle{{margin:0 0 20px;color:var(--muted)}}
.metric{{padding:12px 0;border-top:1px solid #e2e6e9}}
.metric strong{{display:block;font-size:24px;font-variant-numeric:tabular-nums}}
.metric span{{color:var(--muted)}}
.legend{{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}}
.legend div{{display:flex;align-items:center;gap:8px;margin:8px 0}}
.swatch{{width:22px;height:4px;background:var(--color)}}
.toolbar{{position:fixed;top:16px;left:calc(var(--panel) + 16px);display:flex;flex-wrap:wrap;gap:6px;z-index:4}}
button{{min-height:36px;border:1px solid #8f9aa3;border-radius:4px;background:#fff;color:var(--ink);padding:7px 11px;cursor:pointer}}
button[aria-pressed="true"]{{background:#1f2933;color:#fff;border-color:#1f2933}}
button:focus-visible{{outline:3px solid #0072b2;outline-offset:2px}}
.status{{position:fixed;right:16px;bottom:16px;background:#fff;border:1px solid var(--line);padding:8px 10px;z-index:4}}
@media(max-width:720px){{:root{{--panel:0px}}.panel{{inset:auto 0 0 0;width:auto;height:184px;border:0;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:0 16px;padding:12px 16px}}.panel h1,.panel .subtitle,.panel .legend{{display:none}}#viewer{{inset:0 0 184px 0}}.toolbar{{left:10px;right:10px;top:10px}}.toolbar button{{min-height:32px;padding:5px 8px;font-size:12px}}.status{{display:none}}.metric{{padding:8px 0;min-width:0}}.metric strong{{font-size:19px}}.metric span{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}}}
</style>
</head>
<body>
<aside class="panel" aria-labelledby="title">
<h1 id="title">case dental midline</h1>
<p class="subtitle">Preserved 3D nearest-surface method</p>
<div class="metric"><strong id="offset-value">--</strong><span>lower relative to upper</span></div>
<div class="metric"><strong id="direction-value">--</strong><span>lateral direction</span></div>
<div class="metric"><strong id="angle-value">--</strong><span>preserved arch-axis difference</span></div>
<div class="metric"><strong id="quality-value">--</strong><span>geometric quality status</span></div>
<div class="legend" aria-label="Annotation legend">
<div><i class="swatch" style="--color:#e53935"></i>3D inter-incisor midline</div>
<div><i class="swatch" style="--color:#f5c542"></i>Incisal midline point</div>
<div><i class="swatch" style="--color:#d55e00"></i>Upper tooth 21</div>
<div><i class="swatch" style="--color:#e6a700"></i>Upper tooth 11</div>
<div><i class="swatch" style="--color:#0072b2"></i>Lower tooth 31</div>
<div><i class="swatch" style="--color:#009e73"></i>Lower tooth 41</div>
</div>
</aside>
<div id="viewer" role="img" aria-label="Interactive three-dimensional central-incisor models with dental midlines"></div>
<nav class="toolbar" aria-label="Model views">
<button type="button" data-view="upper" aria-pressed="false">Upper</button>
<button type="button" data-view="lower" aria-pressed="false">Lower</button>
<button type="button" data-view="occlusion" aria-pressed="true">Occlusion</button>
<button type="button" id="wireframe" aria-pressed="false">Wireframe</button>
<button type="button" id="annotations" aria-pressed="true">Annotations</button>
</nav>
<div class="status" id="status" role="status">Loading model...</div>
<script id="result-data" type="application/json">{result_json}</script>
<script id="scene-data" type="application/json">{scene_json}</script>
<script type="module">
// Import-map and addon pattern follows the official Three.js prerequisites manual:
// https://threejs.org/manual/#en/fundamentals and https://threejs.org/docs/#examples/en/controls/OrbitControls
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const result=JSON.parse(document.getElementById('result-data').textContent);
const data=JSON.parse(document.getElementById('scene-data').textContent);
const container=document.getElementById('viewer');
const status=document.getElementById('status');
const scene=new THREE.Scene();
scene.background=new THREE.Color(0xe8edf0);
const camera=new THREE.PerspectiveCamera(38,1,0.1,300);
const renderer=new THREE.WebGLRenderer({{antialias:true,powerPreference:'high-performance'}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;
controls.dampingFactor=0.08;
controls.minDistance=10;
controls.maxDistance=120;
scene.add(new THREE.HemisphereLight(0xffffff,0x69747c,2.2));
const keyLight=new THREE.DirectionalLight(0xffffff,2.6);
keyLight.position.set(20,35,35);
scene.add(keyLight);
const fillLight=new THREE.DirectionalLight(0xffffff,1.2);
fillLight.position.set(-25,-5,15);
scene.add(fillLight);
const grid=new THREE.GridHelper(50,10,0x9ca8b0,0xcbd2d7);
grid.material.transparent=true;
grid.material.opacity=0.32;
scene.add(grid);

const archGroups={{}};
const materials=[];
const annotationObjects=[];
function vector(values){{return new THREE.Vector3(...values)}}
function addCylinder(group,start,end,radius,color){{
  const a=vector(start),b=vector(end),delta=b.clone().sub(a);
  const geometry=new THREE.CylinderGeometry(radius,radius,delta.length(),16);
  const material=new THREE.MeshBasicMaterial({{color,depthTest:false}});
  const tube=new THREE.Mesh(geometry,material);
  tube.position.copy(a).add(b).multiplyScalar(0.5);
  tube.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());
  tube.renderOrder=10;group.add(tube);annotationObjects.push(tube);return tube;
}}
function addSphere(group,point,radius,color){{
  const sphere=new THREE.Mesh(new THREE.SphereGeometry(radius,24,16),new THREE.MeshBasicMaterial({{color,depthTest:false}}));
  sphere.position.copy(vector(point));sphere.renderOrder=11;group.add(sphere);annotationObjects.push(sphere);return sphere;
}}
function addArch(name,arch){{
  const group=new THREE.Group();group.name=name;scene.add(group);archGroups[name]=group;
  for(const item of arch.meshes){{
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(item.positions,3));
    geometry.setIndex(item.indices);geometry.computeVertexNormals();
    const material=new THREE.MeshStandardMaterial({{color:item.color,roughness:.72,metalness:0,side:THREE.DoubleSide}});
    materials.push(material);const mesh=new THREE.Mesh(geometry,material);mesh.userData.tooth=item.label;group.add(mesh);
  }}
  addCylinder(group,...arch.interface_line,.17,0xe53935);
  addSphere(group,arch.contact_point,.40,0xf5c542);
}}
addArch('upper',data.arches.upper);addArch('lower',data.arches.lower);

let currentView='occlusion';
const presets={{
  upper:{{direction:[0,-1,0],up:[0,0,1]}},
  lower:{{direction:[0,1,0],up:[0,0,1]}},
  occlusion:{{direction:[1,1,.82],up:[0,1,0]}}
}};
function visibleBounds(name){{
  const bounds=new THREE.Box3();
  if(name!=='lower')bounds.expandByObject(archGroups.upper,true);
  if(name!=='upper')bounds.expandByObject(archGroups.lower,true);
  return bounds;
}}
function frameVisibleModel(name,directionOverride=null){{
  const bounds=visibleBounds(name);
  if(bounds.isEmpty())return null;
  const sphere=bounds.getBoundingSphere(new THREE.Sphere());
  const preset=presets[name];
  const direction=directionOverride
    ? vector(directionOverride)
    : camera.position.clone().sub(controls.target);
  if(direction.lengthSq()<1e-6)direction.fromArray(preset.direction);
  camera.up.fromArray(preset.up);
  const verticalFov=THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov=2*Math.atan(Math.tan(verticalFov/2)*camera.aspect);
  const limitingFov=Math.min(verticalFov,horizontalFov);
  const distance=Math.max(sphere.radius/Math.sin(limitingFov/2)*1.18,sphere.radius*2);
  camera.position.copy(sphere.center).addScaledVector(direction.normalize(),distance);
  camera.near=Math.max(.01,distance-sphere.radius*2);
  camera.far=distance+sphere.radius*10;
  camera.updateProjectionMatrix();
  controls.target.copy(sphere.center);
  controls.minDistance=Math.max(.5,sphere.radius*.4);
  controls.maxDistance=Math.max(20,sphere.radius*10);
  controls.update();
  return {{center:sphere.center.toArray(),radius:sphere.radius,distance}};
}}
function applyView(name){{
  currentView=name;archGroups.upper.visible=name!=='lower';archGroups.lower.visible=name!=='upper';
  frameVisibleModel(name,presets[name].direction);
  document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===name)));
}}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>applyView(button.dataset.view)));
document.getElementById('wireframe').addEventListener('click',event=>{{
  const active=event.currentTarget.getAttribute('aria-pressed')!=='true';event.currentTarget.setAttribute('aria-pressed',String(active));
  materials.forEach(material=>material.wireframe=active);
}});
document.getElementById('annotations').addEventListener('click',event=>{{
  const active=event.currentTarget.getAttribute('aria-pressed')!=='true';event.currentTarget.setAttribute('aria-pressed',String(active));
  annotationObjects.forEach(object=>object.visible=active);
}});
function resize(refit=true){{
  const width=container.clientWidth,height=container.clientHeight;renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();
  if(refit)frameVisibleModel(currentView);
}}
const requestedView=new URLSearchParams(window.location.search).get('view');
const initialView=Object.hasOwn(presets,requestedView)?requestedView:'occlusion';
window.addEventListener('resize',()=>resize(true));resize(false);applyView(initialView);
const comparison=result.comparison;
document.getElementById('offset-value').textContent=`${{comparison.absolute_lateral_offset_mm.toFixed(2)}} mm`;
document.getElementById('direction-value').textContent=comparison.direction.replace('_',' ');
document.getElementById('angle-value').textContent=`${{comparison.midline_angle_difference_deg.toFixed(2)}} deg`;
document.getElementById('quality-value').textContent=result.quality.status;
status.textContent=`Three.js r${{ {json.dumps(three_version)}.split('.')[1] }} · ${{data.arches.upper.meshes.length+data.arches.lower.meshes.length}} teeth`;
window.__MIDLINE_VIEWER__={{ready:true,result,data,scene,camera,renderer,controls,applyView,frameVisibleModel,visibleBounds}};
renderer.setAnimationLoop(()=>{{controls.update();renderer.render(scene,camera)}});
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-stl", type=Path, default=Path("2/ios_upper.stl"))
    parser.add_argument("--upper-labels", type=Path, default=Path("2/case_ios_upper.json"))
    parser.add_argument("--lower-stl", type=Path, default=Path("2/ios_lower.stl"))
    parser.add_argument("--lower-labels", type=Path, default=Path("2/case_ios_lower.json"))
    parser.add_argument("--output", type=Path, default=Path("2/case_midline.json"))
    parser.add_argument("--viewer", type=Path, default=Path("2/case_midline_viewer.html"))
    args = parser.parse_args()

    result, scene_data = calculate_midlines(
        args.upper_stl,
        args.upper_labels,
        args.lower_stl,
        args.lower_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.viewer.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.viewer.write_text(
        render_viewer_html(result, scene_data),
        encoding="utf-8",
    )
    comparison = result["comparison"]
    print("case dental midline")
    print(
        "Lower relative to upper: "
        f"{comparison['absolute_lateral_offset_mm']:.2f} mm "
        f"toward {comparison['direction'].replace('_', ' ')}"
    )
    print(
        "Midline angle difference: "
        f"{comparison['midline_angle_difference_deg']:.2f} deg"
    )
    print(f"Quality: {result['quality']['status']}")
    print(f"JSON: {args.output}")
    print(f"Viewer: {args.viewer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
