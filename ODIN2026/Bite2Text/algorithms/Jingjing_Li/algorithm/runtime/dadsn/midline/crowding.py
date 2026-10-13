#!/usr/bin/env python3
"""基于二进制 STL 原始坐标和逐顶点 JSON 标签计算上、下颌拥挤度。"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# 导师指定的牙位顺序：均从患者左侧第一磨牙走到右侧第一磨牙。
UPPER_TEETH_ORDER = [26, 25, 24, 23, 22, 21, 11, 12, 13, 14, 15, 16]
LOWER_TEETH_ORDER = [36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46]

# 10 牙口径只用于交叉检查第一磨牙端点，不替代导师指定的 12 牙主结果。
STANDARD_UPPER_TEN_TEETH = [25, 24, 23, 22, 21, 11, 12, 13, 14, 15]
STANDARD_LOWER_TEN_TEETH = [35, 34, 33, 32, 31, 41, 42, 43, 44, 45]

# 第二磨牙不加入牙量，但可帮助确定第一磨牙附近的局部牙弓方向。
ARCH_CONFIGS = {
    "upper": {
        "teeth_order": UPPER_TEETH_ORDER,
        "standard_ten_teeth": STANDARD_UPPER_TEN_TEETH,
        "context_order": [27, *UPPER_TEETH_ORDER, 17],
    },
    "lower": {
        "teeth_order": LOWER_TEETH_ORDER,
        "standard_ten_teeth": STANDARD_LOWER_TEN_TEETH,
        "context_order": [37, *LOWER_TEETH_ORDER, 47],
    },
}

# 二进制 STL 单个三角形：3 个法向量 float32、9 个顶点 float32、1 个 uint16。
STL_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)

# 牙冠 PCA 轴的工程选择：0 为较小方差轴，1 为较大方差轴。
# 该规则按牙位形态区分上下颌，避免使用会随牙齿旋转而变化的牙弓切线。
INTRINSIC_MD_PCA_AXIS = {
    "upper": {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
    "lower": {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
}

# 宽松的工程质控范围只负责提示，不会截断或替换测量值。
WIDTH_RANGES_MM = {
    "upper": {
        1: (7.0, 10.0),
        2: (5.0, 8.5),
        3: (6.0, 9.5),
        4: (5.5, 8.5),
        5: (5.5, 8.5),
        6: (8.5, 13.0),
        7: (8.5, 13.5),
    },
    "lower": {
        1: (4.0, 6.5),
        2: (4.5, 7.0),
        3: (5.5, 8.0),
        4: (5.5, 8.5),
        5: (5.5, 8.5),
        6: (9.0, 13.0),
        7: (9.0, 13.5),
    },
}


@dataclass(frozen=True)
class MeshData:
    """按首次出现顺序稳定去重后的 STL 网格。"""

    vertices: np.ndarray
    faces: np.ndarray
    triangle_count: int
    raw_vertex_count: int


@dataclass(frozen=True)
class ToothMeasurement:
    """单颗牙在咬合平面中的几何测量结果。"""

    label: int
    point_count: int
    center_2d: np.ndarray
    tangent_2d: np.ndarray
    intrinsic_axis_2d: np.ndarray
    pca_eigenvalues: np.ndarray
    width_mm: float
    tangent_width_mm: float
    rotation_from_arch_deg: float
    mesial_2d: np.ndarray
    distal_2d: np.ndarray


@dataclass(frozen=True)
class ArchCurveFit:
    """两段三次曲线组成的个体化平滑牙弓。"""

    length_mm: float
    sampled_points_2d: np.ndarray
    left_control_points_2d: np.ndarray
    right_control_points_2d: np.ndarray
    fit_rmse_mm: float
    max_residual_mm: float
    tooth_residuals_mm: dict[int, float]
    tooth_weights: dict[int, float]

def load_binary_stl_stable(path: Path) -> MeshData:
    """读取二进制 STL，并按首次出现顺序稳定去重。

    该实现与旧版 Python 字典逐点循环结果一致，但使用 NumPy 批量去重，
    对数十万至数百万原始顶点明显更快。
    """
    file_size = path.stat().st_size
    if file_size < 84:
        raise ValueError(f"STL 文件过小，不是有效的二进制 STL：{path}")

    with path.open("rb") as handle:
        handle.seek(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]

    expected_size = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    if file_size != expected_size:
        raise ValueError(
            "当前程序只支持二进制 STL，或三角形数量字段无效："
            f"实际大小={file_size}，期望大小={expected_size}"
        )

    triangles = np.fromfile(
        path,
        dtype=STL_TRIANGLE_DTYPE,
        count=triangle_count,
        offset=84,
    )
    raw_vertices = np.ascontiguousarray(
        triangles["vertices"].reshape(-1, 3)
    )

    # np.unique 会按坐标排序；利用首次出现索引把唯一点重新排回稳定顺序。
    unique_sorted, first_indices, inverse_sorted = np.unique(
        raw_vertices,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    stable_order = np.argsort(first_indices)
    unique_vertices = unique_sorted[stable_order]

    sorted_to_stable = np.empty_like(stable_order)
    sorted_to_stable[stable_order] = np.arange(
        len(stable_order),
        dtype=stable_order.dtype,
    )
    faces = sorted_to_stable[inverse_sorted].reshape(
        triangle_count, 3
    ).astype(np.int32, copy=False)

    return MeshData(
        vertices=unique_vertices.astype(np.float64, copy=False),
        faces=faces,
        triangle_count=triangle_count,
        raw_vertex_count=len(raw_vertices),
    )


def load_labels(path: Path) -> np.ndarray:
    """读取一维逐顶点牙位标签 JSON。"""
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    labels = np.asarray(values, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError("JSON 标签必须是一维数组")
    return labels


def detect_arch(labels: np.ndarray, requested_arch: str) -> str:
    """根据标签自动识别上颌或下颌，也允许命令行显式指定。"""
    if requested_arch != "auto":
        return requested_arch

    present = set(int(value) for value in np.unique(labels))
    has_upper = any(tooth in present for tooth in UPPER_TEETH_ORDER)
    has_lower = any(tooth in present for tooth in LOWER_TEETH_ORDER)
    if has_upper and has_lower:
        raise ValueError("标签同时包含上、下颌牙位，请显式指定 --arch")
    if has_upper:
        return "upper"
    if has_lower:
        return "lower"
    raise ValueError("JSON 中没有识别到上颌或下颌目标牙位")


def available_context_order(labels: np.ndarray, arch: str) -> list[int]:
    """检查指定牙弓的 12 颗目标牙是否完整。"""
    config = ARCH_CONFIGS[arch]
    required_teeth = config["teeth_order"]
    present = set(int(value) for value in np.unique(labels))
    missing = [tooth for tooth in required_teeth if tooth not in present]
    if missing:
        raise ValueError(f"{arch} 颌缺少导师要求的牙位标签：{missing}")
    return [tooth for tooth in config["context_order"] if tooth in present]


def estimate_occlusal_plane(
    vertices: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用全部牙齿点的 PCA 估计当前牙弓的咬合平面。"""
    tooth_points = vertices[labels > 0]
    if len(tooth_points) < 3:
        raise ValueError(
            f"全部牙齿点只有 {len(tooth_points)} 个，无法估计咬合平面"
        )
    if not np.all(np.isfinite(tooth_points)):
        raise ValueError("牙齿点包含 NaN/Inf，无法估计咬合平面")
    origin = tooth_points.mean(axis=0)
    covariance = np.cov(tooth_points - origin, rowvar=False)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("咬合平面协方差无效")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    plane_basis = eigenvectors[:, 1:3]
    return origin, plane_basis, eigenvalues


def robust_tail_landmark(
    points_2d: np.ndarray,
    projections: np.ndarray,
    tail_fraction: float,
    upper: bool,
) -> np.ndarray:
    """用投影一侧约 2% 点的中位数估计近中或远中标志点。"""
    threshold = np.quantile(
        projections,
        1.0 - tail_fraction if upper else tail_fraction,
    )
    selected = (
        points_2d[projections >= threshold]
        if upper
        else points_2d[projections <= threshold]
    )
    return np.median(selected, axis=0)


def intrinsic_width_from_pca(
    points_2d: np.ndarray,
    axis_index: int,
    quantiles: tuple[float, float] = (0.005, 0.995),
) -> tuple[float, np.ndarray, np.ndarray]:
    """沿牙冠自身 PCA 轴测宽；牙齿整体旋转不会改变该宽度。"""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if len(points_2d) < 3:
        raise ValueError(
            f"牙冠点数只有 {len(points_2d)}，无法进行 PCA 测宽"
        )
    if not np.all(np.isfinite(points_2d)):
        raise ValueError("牙冠点包含 NaN/Inf，无法进行 PCA 测宽")
    centered = points_2d - np.median(points_2d, axis=0)
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("牙冠二维 PCA 协方差无效")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, axis_index]
    projections = centered @ axis
    low, high = np.quantile(projections, quantiles)
    return float(high - low), axis, eigenvalues


def measure_teeth(
    projected_vertices: np.ndarray,
    labels: np.ndarray,
    context_order: list[int],
    arch: str,
    legacy_width_quantiles: tuple[float, float] = (0.01, 0.99),
    landmark_tail_fraction: float = 0.02,
    minimum_tooth_points: int = 20,
 ) -> dict[int, ToothMeasurement]:
    """计算旋转不敏感牙宽，并保留旧切线投影宽度作为对照。"""
    point_counts = {
        tooth: int(np.count_nonzero(labels == tooth))
        for tooth in context_order
    }
    insufficient = {
        tooth: count
        for tooth, count in point_counts.items()
        if count < minimum_tooth_points
    }
    if insufficient:
        details = ", ".join(
            f"{tooth}:{count}" for tooth, count in insufficient.items()
        )
        raise ValueError(
            f"以下牙位预测点数不足 {minimum_tooth_points}，无法可靠计算拥挤度：{details}"
        )

    centers = {
        tooth: np.median(projected_vertices[labels == tooth], axis=0)
        for tooth in context_order
    }
    measurements: dict[int, ToothMeasurement] = {}

    for index, tooth in enumerate(context_order):
        previous_index = max(index - 1, 0)
        next_index = min(index + 1, len(context_order) - 1)
        tangent = (
            centers[context_order[next_index]]
            - centers[context_order[previous_index]]
        )
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            raise ValueError(f"牙位 {tooth} 的局部牙弓方向无法确定")
        tangent = tangent / tangent_norm

        points = projected_vertices[labels == tooth]
        centered = points - centers[tooth]
        tangent_projections = centered @ tangent
        legacy_low, legacy_high = np.quantile(
            tangent_projections, legacy_width_quantiles
        )
        tangent_width_mm = float(legacy_high - legacy_low)

        axis_index = INTRINSIC_MD_PCA_AXIS[arch][tooth % 10]
        width_mm, intrinsic_axis, pca_eigenvalues = intrinsic_width_from_pca(
            points, axis_index
        )
        if np.dot(intrinsic_axis, tangent) < 0:
            intrinsic_axis = -intrinsic_axis
        cosine = float(np.clip(abs(np.dot(intrinsic_axis, tangent)), 0.0, 1.0))
        rotation_from_arch_deg = float(np.degrees(np.arccos(cosine)))

        backward = robust_tail_landmark(
            points, tangent_projections, landmark_tail_fraction, upper=False
        )
        forward = robust_tail_landmark(
            points, tangent_projections, landmark_tail_fraction, upper=True
        )

        # 第 2、3 象限沿牙位顺序向前是近中；第 1、4 象限向前是远中。
        if tooth // 10 in (2, 3):
            mesial, distal = forward, backward
        else:
            mesial, distal = backward, forward

        measurements[tooth] = ToothMeasurement(
            label=tooth,
            point_count=len(points),
            center_2d=centers[tooth],
            tangent_2d=tangent,
            intrinsic_axis_2d=intrinsic_axis,
            pca_eigenvalues=pca_eigenvalues,
            width_mm=width_mm,
            tangent_width_mm=tangent_width_mm,
            rotation_from_arch_deg=rotation_from_arch_deg,
            mesial_2d=mesial,
            distal_2d=distal,
        )

    return measurements


def polyline_length(points: Iterable[np.ndarray]) -> tuple[float, list[float]]:
    """计算六个有序牙弓标志点形成的五段总长度。"""
    point_array = np.asarray(list(points), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(point_array, axis=0), axis=1)
    return float(segment_lengths.sum()), [float(value) for value in segment_lengths]


def _bezier_points(control_points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """计算三次 Bezier 曲线点。"""
    t = parameters[:, None]
    return (
        (1.0 - t) ** 3 * control_points[0]
        + 3.0 * (1.0 - t) ** 2 * t * control_points[1]
        + 3.0 * (1.0 - t) * t**2 * control_points[2]
        + t**3 * control_points[3]
    )


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    """按参考点累积弦长分配曲线参数，降低点密度不均的影响。"""
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] == 0:
        raise ValueError("牙弓参考点重合，无法拟合曲线")
    return cumulative / cumulative[-1]


def _fit_cubic_half(
    points: np.ndarray,
    midline_tangent: np.ndarray,
    side: str,
    huber_floor_mm: float = 0.75,
    iterations: int = 30,
 ) -> tuple[np.ndarray, np.ndarray]:
    """拟合半侧三次曲线，并约束中线处两侧具有相同切线方向。"""
    parameters = _chord_parameters(points)
    t = parameters
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * t**2
    b3 = t**3
    weights = np.ones(len(points), dtype=np.float64)

    if side == "left":
        fixed = b0[:, None] * points[0] + (b2 + b3)[:, None] * points[-1]
        design = np.zeros((len(points), 2, 3), dtype=np.float64)
        design[:, 0, 0] = b1
        design[:, 1, 1] = b1
        design[:, :, 2] = -b2[:, None] * midline_tangent
    elif side == "right":
        fixed = (b0 + b1)[:, None] * points[0] + b3[:, None] * points[-1]
        design = np.zeros((len(points), 2, 3), dtype=np.float64)
        design[:, :, 0] = b1[:, None] * midline_tangent
        design[:, 0, 1] = b2
        design[:, 1, 2] = b2
    else:
        raise ValueError(f"未知牙弓半侧：{side}")

    target = points - fixed
    solution = np.zeros(3, dtype=np.float64)
    for _ in range(iterations):
        row_weights = np.repeat(np.sqrt(weights), 2)
        matrix = design.reshape(-1, 3) * row_weights[:, None]
        values = target.reshape(-1) * row_weights
        solution = np.linalg.lstsq(matrix, values, rcond=None)[0]

        if side == "left":
            control_points = np.vstack(
                (points[0], solution[:2], points[-1] - solution[2] * midline_tangent, points[-1])
            )
        else:
            control_points = np.vstack(
                (points[0], points[0] + solution[0] * midline_tangent, solution[1:], points[-1])
            )
        predicted = _bezier_points(control_points, parameters)
        residuals = np.linalg.norm(points - predicted, axis=1)
        interior = residuals[1:-1]
        median = float(np.median(interior))
        mad = float(np.median(np.abs(interior - median)))
        huber_delta = max(huber_floor_mm, 1.345 * 1.4826 * mad)
        new_weights = np.minimum(
            1.0, huber_delta / np.maximum(residuals, 1e-12)
        )
        new_weights[[0, -1]] = 1.0
        if np.max(np.abs(new_weights - weights)) < 1e-6:
            weights = new_weights
            break
        weights = new_weights

    # 防止噪声使中线控制柄反向，从而生成局部回环。
    if side == "left" and solution[2] <= 0:
        control_points[2] = points[-1] - 0.1 * midline_tangent
    if side == "right" and solution[0] <= 0:
        control_points[1] = points[0] + 0.1 * midline_tangent
    return control_points, weights


def fit_smooth_arch_curve(
    tooth_labels: list[int],
    measurements: dict[int, ToothMeasurement],
    left_endpoint: np.ndarray,
    right_endpoint: np.ndarray,
 ) -> ArchCurveFit:
    """以牙冠稳健中心为观测点，拟合端点固定且中线光顺的个体牙弓。"""
    if len(tooth_labels) < 4 or len(tooth_labels) % 2:
        raise ValueError("平滑牙弓拟合需要左右数量相等的牙位")
    half = len(tooth_labels) // 2
    left_teeth = tooth_labels[:half]
    right_teeth = tooth_labels[half:]
    left_central = measurements[left_teeth[-1]].center_2d
    right_central = measurements[right_teeth[0]].center_2d
    midline = (left_central + right_central) / 2.0
    midline_tangent = right_central - left_central
    tangent_norm = np.linalg.norm(midline_tangent)
    if tangent_norm == 0:
        raise ValueError("左右中切牙中心重合，无法确定牙弓中线切线")
    midline_tangent = midline_tangent / tangent_norm

    left_points = np.vstack(
        (left_endpoint, *[measurements[tooth].center_2d for tooth in left_teeth], midline)
    )
    right_points = np.vstack(
        (midline, *[measurements[tooth].center_2d for tooth in right_teeth], right_endpoint)
    )
    left_controls, left_weights = _fit_cubic_half(
        left_points, midline_tangent, "left"
    )
    right_controls, right_weights = _fit_cubic_half(
        right_points, midline_tangent, "right"
    )

    dense_parameters = np.linspace(0.0, 1.0, 2001)
    dense_left = _bezier_points(left_controls, dense_parameters)
    dense_right = _bezier_points(right_controls, dense_parameters)
    dense_curve = np.vstack((dense_left, dense_right[1:]))
    length_mm = float(np.linalg.norm(np.diff(dense_curve, axis=0), axis=1).sum())

    output_parameters = np.linspace(0.0, 1.0, 101)
    output_curve = np.vstack(
        (
            _bezier_points(left_controls, output_parameters),
            _bezier_points(right_controls, output_parameters)[1:],
        )
    )
    tooth_residuals: dict[int, float] = {}
    for tooth in tooth_labels:
        distances = np.linalg.norm(
            dense_curve - measurements[tooth].center_2d, axis=1
        )
        tooth_residuals[tooth] = float(np.min(distances))
    residual_values = np.asarray(list(tooth_residuals.values()))
    tooth_weights = {
        **{tooth: float(weight) for tooth, weight in zip(left_teeth, left_weights[1:-1])},
        **{tooth: float(weight) for tooth, weight in zip(right_teeth, right_weights[1:-1])},
    }

    return ArchCurveFit(
        length_mm=length_mm,
        sampled_points_2d=output_curve,
        left_control_points_2d=left_controls,
        right_control_points_2d=right_controls,
        fit_rmse_mm=float(np.sqrt(np.mean(residual_values**2))),
        max_residual_mm=float(np.max(residual_values)),
        tooth_residuals_mm=tooth_residuals,
        tooth_weights=tooth_weights,
    )

def crowding_grade(crowding_mm: float) -> str:
    """根据当前初始毫米阈值输出拥挤等级。"""
    if crowding_mm < -1.0:
        return "spacing"
    if crowding_mm <= 1.0:
        return "none_or_minimal"
    if crowding_mm < 4.0:
        return "mild"
    if crowding_mm <= 8.0:
        return "moderate"
    return "severe"


def plane_point_to_3d(
    point_2d: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
) -> list[float]:
    """将咬合平面二维标志点还原到当前 STL 三维坐标系。"""
    return [float(value) for value in origin + basis @ point_2d]


def build_protocol_result(
    name: str,
    tooth_labels: list[int],
    endpoint_landmarks: tuple[tuple[int, str], tuple[int, str]],
    legacy_landmarks: list[tuple[int, str]],
    measurements: dict[int, ToothMeasurement],
    origin: np.ndarray,
    basis: np.ndarray,
 ) -> dict[str, object]:
    """汇总新平滑曲线主结果，并附带旧算法的完整对照。"""
    tooth_material = float(
        sum(measurements[tooth].width_mm for tooth in tooth_labels)
    )
    tangent_tooth_material = float(
        sum(measurements[tooth].tangent_width_mm for tooth in tooth_labels)
    )
    endpoint_points = [
        getattr(measurements[tooth], f"{surface}_2d")
        for tooth, surface in endpoint_landmarks
    ]
    curve = fit_smooth_arch_curve(
        tooth_labels, measurements, endpoint_points[0], endpoint_points[1]
    )
    crowding = tooth_material - curve.length_mm

    legacy_points = [
        getattr(measurements[tooth], f"{surface}_2d")
        for tooth, surface in legacy_landmarks
    ]
    legacy_length, legacy_segments = polyline_length(legacy_points)
    legacy_crowding = tangent_tooth_material - legacy_length

    endpoint_output = []
    for (tooth, surface), point_2d in zip(endpoint_landmarks, endpoint_points):
        endpoint_output.append(
            {
                "tooth": tooth,
                "surface": surface,
                "point_2d": point_2d.tolist(),
                "point_3d": plane_point_to_3d(point_2d, origin, basis),
            }
        )

    return {
        "name": name,
        "tooth_labels": tooth_labels,
        "tooth_material_mm": tooth_material,
        "arch_length_mm": curve.length_mm,
        "crowding_mm": crowding,
        "grade": crowding_grade(crowding),
        "arch_endpoints": endpoint_output,
        "arch_curve": {
            "method": "robust_two_piece_cubic_bezier",
            "fit_rmse_mm": curve.fit_rmse_mm,
            "max_residual_mm": curve.max_residual_mm,
            "tooth_residuals_mm": {
                str(tooth): value for tooth, value in curve.tooth_residuals_mm.items()
            },
            "tooth_robust_weights": {
                str(tooth): value for tooth, value in curve.tooth_weights.items()
            },
            "left_control_points_2d": curve.left_control_points_2d.tolist(),
            "right_control_points_2d": curve.right_control_points_2d.tolist(),
            "sampled_points_2d": curve.sampled_points_2d.tolist(),
            "sampled_points_3d": [
                plane_point_to_3d(point, origin, basis)
                for point in curve.sampled_points_2d
            ],
        },
        "same_width_polyline_comparison": {
            "arch_length_mm": legacy_length,
            "crowding_mm": tooth_material - legacy_length,
        },
        "legacy_method_comparison": {
            "tooth_material_mm": tangent_tooth_material,
            "arch_length_mm": legacy_length,
            "crowding_mm": legacy_crowding,
            "grade": crowding_grade(legacy_crowding),
            "segment_lengths_mm": legacy_segments,
        },
    }

def calculate_crowding(
    stl_path: Path,
    labels_path: Path,
    requested_arch: str = "auto",
    minimum_tooth_points: int = 20,
) -> dict[str, object]:
    """执行上颌或下颌的完整拥挤度计算。"""
    mesh = load_binary_stl_stable(stl_path)
    labels = load_labels(labels_path)
    if len(mesh.vertices) != len(labels):
        raise ValueError(
            "STL 稳定去重顶点数与 JSON 标签数不一致："
            f"{len(mesh.vertices)} != {len(labels)}"
        )

    arch = detect_arch(labels, requested_arch)
    config = ARCH_CONFIGS[arch]
    context_order = available_context_order(labels, arch)
    origin, basis, eigenvalues = estimate_occlusal_plane(mesh.vertices, labels)
    projected_vertices = (mesh.vertices - origin) @ basis
    measurements = measure_teeth(
        projected_vertices,
        labels,
        context_order,
        arch,
        minimum_tooth_points=minimum_tooth_points,
    )

    teeth_order = config["teeth_order"]
    standard_ten_teeth = config["standard_ten_teeth"]
    left_molar = teeth_order[0]
    left_canine = teeth_order[3]
    left_central = teeth_order[5]
    right_central = teeth_order[6]
    right_canine = teeth_order[8]
    right_molar = teeth_order[-1]

    # 主口径：12 颗牙，牙弓从两侧第一磨牙远中到远中。
    twelve_tooth_landmarks = [
        (left_molar, "distal"),
        (left_canine, "distal"),
        (left_central, "distal"),
        (right_central, "distal"),
        (right_canine, "distal"),
        (right_molar, "distal"),
    ]
    # 交叉检查口径：10 颗牙，牙弓从两侧第一磨牙近中到近中。
    ten_tooth_landmarks = [
        (left_molar, "mesial"),
        (left_canine, "distal"),
        (left_central, "distal"),
        (right_central, "distal"),
        (right_canine, "distal"),
        (right_molar, "mesial"),
    ]

    primary_result = build_protocol_result(
        f"{arch}_mentor_twelve_tooth_distal_to_distal",
        teeth_order,
        ((left_molar, "distal"), (right_molar, "distal")),
        twelve_tooth_landmarks,
        measurements,
        origin,
        basis,
    )
    cross_check_result = build_protocol_result(
        f"{arch}_standard_ten_tooth_mesial_to_mesial",
        standard_ten_teeth,
        ((left_molar, "mesial"), (right_molar, "mesial")),
        ten_tooth_landmarks,
        measurements,
        origin,
        basis,
    )

    quality_issues: list[str] = []
    for tooth in teeth_order:
        width = measurements[tooth].width_mm
        lower, upper = WIDTH_RANGES_MM[arch][tooth % 10]
        if not lower <= width <= upper:
            quality_issues.append(
                f"牙位 {tooth} 宽度 {width:.2f} mm 超出宽松质控范围 "
                f"{lower:.1f}-{upper:.1f} mm"
            )

    protocol_delta = abs(
        float(primary_result["crowding_mm"])
        - float(cross_check_result["crowding_mm"])
    )
    if protocol_delta > 1.5:
        quality_issues.append(
            f"12 牙与 10 牙口径相差 {protocol_delta:.2f} mm，"
            "需要复查两侧第一磨牙端点"
        )

    tooth_output = []
    for tooth in teeth_order:
        item = measurements[tooth]
        tooth_output.append(
            {
                "label": tooth,
                "point_count": item.point_count,
                "mesiodistal_width_mm": item.width_mm,
                "legacy_tangent_width_mm": item.tangent_width_mm,
                "width_delta_mm": item.width_mm - item.tangent_width_mm,
                "center_2d": item.center_2d.tolist(),
                "tangent_2d": item.tangent_2d.tolist(),
                "intrinsic_axis_2d": item.intrinsic_axis_2d.tolist(),
                "rotation_from_arch_deg": item.rotation_from_arch_deg,
                "pca_eigenvalues": item.pca_eigenvalues.tolist(),
                "mesial_landmark_3d": plane_point_to_3d(
                    item.mesial_2d, origin, basis
                ),
                "distal_landmark_3d": plane_point_to_3d(
                    item.distal_2d, origin, basis
                ),
            }
        )

    return {
        "input": {
            "arch": arch,
            "stl": str(stl_path),
            "labels": str(labels_path),
            "units": "millimeters_assumed",
        },
        "mapping": {
            "stl_triangles": mesh.triangle_count,
            "stl_raw_vertices": mesh.raw_vertex_count,
            "stable_unique_vertices": len(mesh.vertices),
            "json_labels": len(labels),
            "stable_order_mapping_valid": True,
        },
        "occlusal_plane": {
            "origin_3d": origin.tolist(),
            "basis_3x2": basis.tolist(),
            "pca_eigenvalues": eigenvalues.tolist(),
        },
        "teeth": tooth_output,
        "primary_result": primary_result,
        "cross_check_result": cross_check_result,
        "quality": {
            "status": "review" if quality_issues else "acceptable",
            "protocol_crowding_delta_mm": protocol_delta,
            "issues": quality_issues,
        },
        "thresholds_mm": {
            "spacing": "< -1",
            "none_or_minimal": "-1 to 1",
            "mild": "> 1 and < 4",
            "moderate": "4 to 8",
            "severe": "> 8",
        },
        "method": {
            "tooth_width": (
                "咬合平面中沿牙冠自身 PCA 近远中轴的第 0.5 至第 99.5 "
                "百分位投影宽度；旧局部牙弓切线宽度保留为对照"
            ),
            "arch_length": (
                "第一磨牙端点固定、以逐牙稳健中心拟合的左右两段鲁棒三次曲线；"
                "中切牙之间保持共同切线"
            ),
            "primary_protocol": "两侧第一磨牙远中到远中的 12 牙口径",
            "cross_check_protocol": "两侧第一磨牙近中到近中的 10 牙口径",
            "legacy_comparison": "六个标志点五段折线加局部牙弓切线牙宽",
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=Path, help="包含原始坐标的二进制 STL 文件")
    parser.add_argument("labels", type=Path, help="逐稳定顶点牙位标签 JSON 文件")
    parser.add_argument(
        "--arch",
        choices=("auto", "upper", "lower"),
        default="auto",
        help="处理上颌、下颌或根据标签自动识别",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="完整结果 JSON 输出路径；不指定时输出到终端",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = calculate_crowding(args.stl, args.labels, args.arch)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    primary = result["primary_result"]
    print(
        f"{result['input']['arch']} 颌拥挤度：{primary['crowding_mm']:.2f} mm"
        f"（{primary['grade']}），牙量：{primary['tooth_material_mm']:.2f} mm，"
        f"牙弓长度：{primary['arch_length_mm']:.2f} mm",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
