#!/usr/bin/env python3
"""垂直咬合试验所需的网格读取、共同对齐和连续量测。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree


UPPER_VALID = set(range(11, 19)) | set(range(21, 29))
LOWER_VALID = set(range(31, 39)) | set(range(41, 49))

# 牙位顺序均为“患者右侧牙、患者左侧牙”。
UPPER_PAIRS = [(11, 21), (12, 22), (13, 23), (14, 24), (15, 25), (16, 26)]
LOWER_PAIRS = [(41, 31), (42, 32), (43, 33), (44, 34), (45, 35), (46, 36)]
UPPER_ANTERIOR = [13, 12, 11, 21, 22, 23]
LOWER_ANTERIOR = [43, 42, 41, 31, 32, 33]
UPPER_POSTERIOR = [16, 15, 14, 24, 25, 26]
LOWER_POSTERIOR = [46, 45, 44, 34, 35, 36]
UPPER_CORE = [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26]
LOWER_CORE = [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36]
CENTRAL_TEETH = (11, 21, 31, 41)
SIDE_PAIRS = {"right": (11, 41), "left": (21, 31)}

PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class SharedFrame:
    """上下颌共同使用的患者坐标系。"""

    origin: np.ndarray
    basis: np.ndarray
    pair_count: int
    pair_dispersion_deg: float
    front_back_distance_mm: float
    vertical_separation_mm: float
    roll_correction_deg: float

    def transform(self, points: np.ndarray) -> np.ndarray:
        """把原始注册坐标转换为 X左、Y前、Z上的患者坐标。"""
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.basis

    def to_result(self) -> dict[str, Any]:
        return {
            "coordinate_order": ["patient_left", "anterior", "superior"],
            "origin_3d": self.origin.tolist(),
            "basis_columns_3d": self.basis.tolist(),
            "basis_determinant": float(np.linalg.det(self.basis)),
            "pair_count": self.pair_count,
            "pair_dispersion_deg": self.pair_dispersion_deg,
            "front_back_distance_mm": self.front_back_distance_mm,
            "vertical_separation_mm": self.vertical_separation_mm,
            "roll_correction_deg": self.roll_correction_deg,
            "same_transform_for_upper_and_lower": True,
            "scale_changed": False,
        }


@dataclass(frozen=True)
class IncisalLandmark:
    """单颗中切牙的稳健切缘标志点及其质控信息。"""

    tooth: int
    arch: str
    point: np.ndarray
    crown_height_mm: float
    source_point_count: int
    sampled_point_count: int
    incisal_point_count: int
    labial_point_count: int
    incisal_si_iqr_mm: float
    labial_ap_iqr_mm: float
    sampled_points: np.ndarray
    labial_edge_points: np.ndarray

    def to_result(self) -> dict[str, Any]:
        return {
            "tooth": self.tooth,
            "arch": self.arch,
            "point_local_xyz_mm": self.point.tolist(),
            "projected_crown_height_mm": self.crown_height_mm,
            "source_point_count": self.source_point_count,
            "sampled_point_count": self.sampled_point_count,
            "incisal_point_count": self.incisal_point_count,
            "labial_point_count": self.labial_point_count,
            "incisal_si_iqr_mm": self.incisal_si_iqr_mm,
            "labial_ap_iqr_mm": self.labial_ap_iqr_mm,
        }


@dataclass
class CaseArtifacts:
    """仅供质控图使用，不写入病例 JSON 的点集。"""

    tooth_points: dict[int, np.ndarray]
    landmarks: dict[int, IncisalLandmark]


def normalize(vector: np.ndarray) -> np.ndarray:
    """归一化向量，并拒绝退化或非有限输入。"""
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("方向向量退化")
    return value / length


def _parse_ply_header(stream) -> tuple[str, int, list[tuple[str, str]]]:
    """读取 PLY 头，仅收集顶点标量属性。"""
    first = stream.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("文件不是 PLY 格式")

    file_format: Optional[str] = None
    vertex_count: Optional[int] = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: Optional[str] = None

    while True:
        raw_line = stream.readline()
        if not raw_line:
            raise ValueError("PLY 文件头缺少 end_header")
        line = raw_line.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue

        parts = line.split()
        if parts[0] == "format" and len(parts) >= 2:
            file_format = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("暂不支持带列表属性的 PLY 顶点")
            vertex_properties.append((parts[2], parts[1]))

    if file_format is None or vertex_count is None:
        raise ValueError("PLY 文件头缺少 format 或 vertex 数量")
    if not {"x", "y", "z"}.issubset(name for name, _ in vertex_properties):
        raise ValueError("PLY 顶点缺少 x/y/z 坐标")
    return file_format, vertex_count, vertex_properties


def load_ply_points(path: Path) -> np.ndarray:
    """读取 selected 目录中的 PLY 顶点，支持常见 ASCII 和二进制格式。"""
    with path.open("rb") as stream:
        file_format, vertex_count, properties = _parse_ply_header(stream)
        property_names = [name for name, _ in properties]

        if file_format == "ascii":
            rows = []
            for _ in range(vertex_count):
                line = stream.readline()
                if not line:
                    raise ValueError(f"{path.name} 的 ASCII 顶点数量不足")
                values = line.decode("ascii", errors="strict").split()
                if len(values) < len(properties):
                    raise ValueError(f"{path.name} 的 ASCII 顶点属性数量不足")
                rows.append([float(value) for value in values[: len(properties)]])
            table = np.asarray(rows, dtype=np.float64)
            points = table[:, [property_names.index(axis) for axis in ("x", "y", "z")]]
        elif file_format in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if file_format == "binary_little_endian" else ">"
            dtype_fields = []
            for name, scalar_type in properties:
                code = PLY_SCALAR_TYPES.get(scalar_type)
                if code is None:
                    raise ValueError(f"不支持的 PLY 属性类型：{scalar_type}")
                dtype_fields.append((name, endian + code))
            values = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=vertex_count)
            if len(values) != vertex_count:
                raise ValueError(f"{path.name} 的二进制顶点数量不足")
            points = np.column_stack([values[axis] for axis in ("x", "y", "z")])
        else:
            raise ValueError(f"不支持的 PLY 格式：{file_format}")

    points = np.asarray(points, dtype=np.float64)
    if points.shape != (vertex_count, 3):
        raise ValueError(f"PLY 坐标形状异常：{points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{path.name} 含 NaN 或 Inf 坐标")
    return points


def _extract_label_payload(payload: Any) -> np.ndarray:
    """兼容一维列表和少量常见字典包装格式。"""
    if isinstance(payload, list):
        return np.asarray(payload)
    if isinstance(payload, dict):
        for key in ("labels", "label", "pred_labels", "predictions", "segmentation"):
            value = payload.get(key)
            if isinstance(value, list):
                return np.asarray(value)
    raise ValueError("JSON 中没有找到一维顶点标签数组")


def load_labels(path: Path) -> np.ndarray:
    """读取与 pass2_full_raw PLY 顶点一一对应的 FDI 标签。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = _extract_label_payload(payload).reshape(-1)
    if labels.dtype.kind not in "iu":
        numeric = np.asarray(labels, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.allclose(numeric, np.rint(numeric)):
            raise ValueError(f"{path.name} 含非整数标签")
        labels = np.rint(numeric)
    return labels.astype(np.int32, copy=False)


def robust_centroid(points: np.ndarray) -> np.ndarray:
    """用距离中位点最近的 90% 点计算稳健牙齿中心。"""
    values = np.asarray(points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) == 0:
        raise ValueError("无法计算空点集中心")
    median = np.median(values, axis=0)
    if len(values) < 30:
        return median
    distances = np.linalg.norm(values - median, axis=1)
    threshold = float(np.quantile(distances, 0.90))
    trimmed = values[distances <= threshold]
    return trimmed.mean(axis=0) if len(trimmed) >= 10 else median


def tooth_centroids(
    points: np.ndarray,
    labels: np.ndarray,
    minimum_points: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """计算全部有效 FDI 牙位的稳健中心和点数。"""
    centers: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for tooth_value in np.unique(labels):
        tooth = int(tooth_value)
        if tooth <= 0:
            continue
        mask = labels == tooth
        count = int(mask.sum())
        counts[tooth] = count
        if count >= minimum_points:
            centers[tooth] = robust_centroid(points[mask])
    return centers, counts


def average_centroids(
    centers: dict[int, np.ndarray],
    teeth: Iterable[int],
) -> Optional[np.ndarray]:
    """对实际存在的指定牙位中心取平均。"""
    values = [centers[tooth] for tooth in teeth if tooth in centers]
    return np.mean(np.vstack(values), axis=0) if values else None


def pair_vectors(
    centers: dict[int, np.ndarray],
    pairs: Iterable[tuple[int, int]],
) -> list[np.ndarray]:
    """构造从患者右侧同名牙指向患者左侧同名牙的向量。"""
    return [
        centers[left_tooth] - centers[right_tooth]
        for right_tooth, left_tooth in pairs
        if right_tooth in centers and left_tooth in centers
    ]


def robust_mean_direction(vectors: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """通过 SVD 求多条同向单位向量的稳健主方向。"""
    if not vectors:
        raise ValueError("没有可用的左右同名牙对")
    normalized = [normalize(vector) for vector in vectors]
    reference = normalized[0]
    aligned = [vector if np.dot(vector, reference) >= 0 else -vector for vector in normalized]
    _, _, vh = np.linalg.svd(np.vstack(aligned), full_matrices=False)
    direction = normalize(vh[0])
    if np.dot(direction, reference) < 0:
        direction = -direction
    angles = [
        math.degrees(math.acos(float(np.clip(np.dot(vector, direction), -1.0, 1.0))))
        for vector in aligned
    ]
    return direction, float(np.median(angles))


def _rotation_about_y(angle_rad: float) -> np.ndarray:
    """构造患者坐标内绕 AP 轴旋转的矩阵。"""
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def build_shared_frame(
    upper_points: np.ndarray,
    upper_labels: np.ndarray,
    lower_points: np.ndarray,
    lower_labels: np.ndarray,
    minimum_frame_tooth_points: int = 80,
    maximum_roll_correction_deg: float = 12.0,
) -> SharedFrame:
    """从上下颌牙位中心建立唯一共同坐标系，不独立移动任何牙弓。"""
    upper_centers, _ = tooth_centroids(
        upper_points, upper_labels, minimum_frame_tooth_points
    )
    lower_centers, _ = tooth_centroids(
        lower_points, lower_labels, minimum_frame_tooth_points
    )
    all_centers = {**upper_centers, **lower_centers}

    lr_vectors = pair_vectors(upper_centers, UPPER_PAIRS) + pair_vectors(
        lower_centers, LOWER_PAIRS
    )
    left_axis, pair_dispersion_deg = robust_mean_direction(lr_vectors)

    anterior = average_centroids(
        all_centers, UPPER_ANTERIOR + LOWER_ANTERIOR
    )
    posterior = average_centroids(
        all_centers, UPPER_POSTERIOR + LOWER_POSTERIOR
    )
    if anterior is None or posterior is None:
        raise ValueError("前牙区或后牙区标签不足，无法确定前后轴")

    anterior_raw = anterior - posterior
    anterior_raw -= np.dot(anterior_raw, left_axis) * left_axis
    anterior_axis = normalize(anterior_raw)

    upper_center = average_centroids(upper_centers, UPPER_CORE)
    lower_center = average_centroids(lower_centers, LOWER_CORE)
    if upper_center is None or lower_center is None:
        raise ValueError("上下颌核心牙标签不足，无法确定上下轴")

    superior_axis = normalize(np.cross(left_axis, anterior_axis))
    vertical_reference = upper_center - lower_center
    if np.dot(superior_axis, vertical_reference) < 0:
        superior_axis = -superior_axis

    central_values = [
        all_centers[tooth] for tooth in CENTRAL_TEETH if tooth in all_centers
    ]
    origin = (
        np.mean(np.vstack(central_values), axis=0)
        if central_values
        else 0.5 * (upper_center + lower_center)
    )
    basis = np.column_stack([left_axis, anterior_axis, superior_axis])

    # 复用现有中线流程的正视 roll 修正，但对两颌使用同一个旋转。
    upper_local = (upper_points - origin) @ basis
    lower_local = (lower_points - origin) @ basis
    local_centers: dict[int, np.ndarray] = {}
    for points, labels in (
        (upper_local, upper_labels),
        (lower_local, lower_labels),
    ):
        centers, _ = tooth_centroids(points, labels, minimum_frame_tooth_points)
        local_centers.update(centers)

    roll_angles = []
    for right_tooth, left_tooth in UPPER_PAIRS[:3] + LOWER_PAIRS[:3]:
        if right_tooth not in local_centers or left_tooth not in local_centers:
            continue
        vector = local_centers[left_tooth] - local_centers[right_tooth]
        angle = math.degrees(math.atan2(float(vector[2]), float(vector[0])))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        roll_angles.append(angle)

    roll_correction_deg = 0.0
    if len(roll_angles) >= 2:
        roll_correction_deg = float(
            np.clip(
                np.median(roll_angles),
                -maximum_roll_correction_deg,
                maximum_roll_correction_deg,
            )
        )
        rotation = _rotation_about_y(math.radians(-roll_correction_deg))
        basis = basis @ rotation.T

    # X左、Y前、Z上这组三个语义轴允许行列式为 -1；它仍是保距正交变换。
    signed_vertical = float(vertical_reference @ basis[:, 2])
    if signed_vertical < 0:
        basis[:, 2] *= -1.0
        signed_vertical = -signed_vertical

    return SharedFrame(
        origin=origin,
        basis=basis,
        pair_count=len(lr_vectors),
        pair_dispersion_deg=pair_dispersion_deg,
        front_back_distance_mm=float(np.linalg.norm(anterior_raw)),
        vertical_separation_mm=signed_vertical,
        roll_correction_deg=roll_correction_deg,
    )


def voxel_downsample(points: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """按体素平均点坐标，减小不同网格密度对分位数的影响。"""
    values = np.asarray(points, dtype=np.float64)
    if voxel_size_mm <= 0 or len(values) == 0:
        return values.copy()
    keys = np.floor(values / voxel_size_mm).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    output = np.column_stack(
        [np.bincount(inverse, weights=values[:, axis]) / counts for axis in range(3)]
    )
    return output


def prepare_tooth_points(
    points: np.ndarray,
    labels: np.ndarray,
    tooth: int,
    minimum_points: int,
    voxel_size_mm: float,
) -> tuple[np.ndarray, int]:
    """提取单牙点，去除最远的少量标签小岛并进行均匀化。"""
    source = np.asarray(points[labels == tooth], dtype=np.float64)
    source = source[np.all(np.isfinite(source), axis=1)]
    source_count = len(source)
    if source_count < minimum_points:
        raise ValueError(
            f"牙位 {tooth} 点数不足：{source_count} < {minimum_points}"
        )

    median = np.median(source, axis=0)
    distances = np.linalg.norm(source - median, axis=1)
    if len(source) >= 200:
        source = source[distances <= np.quantile(distances, 0.995)]

    sampled = voxel_downsample(source, voxel_size_mm)
    if len(sampled) < max(40, minimum_points // 4):
        raise ValueError(f"牙位 {tooth} 均匀采样后点数不足：{len(sampled)}")
    return sampled, source_count


def extract_incisal_landmark(
    sampled_points: np.ndarray,
    source_point_count: int,
    tooth: int,
    arch: str,
) -> IncisalLandmark:
    """从单牙稳健分位带中提取切缘及切缘唇侧位置。"""
    points = np.asarray(sampled_points, dtype=np.float64)
    x_low, x_high = np.quantile(points[:, 0], [0.10, 0.90])
    crown_height = float(
        np.quantile(points[:, 2], 0.95) - np.quantile(points[:, 2], 0.05)
    )

    if arch == "upper":
        si_low, si_high = np.quantile(points[:, 2], [0.02, 0.08])
    elif arch == "lower":
        si_low, si_high = np.quantile(points[:, 2], [0.92, 0.98])
    else:
        raise ValueError(f"未知牙弓：{arch}")

    incisal_mask = (
        (points[:, 0] >= x_low)
        & (points[:, 0] <= x_high)
        & (points[:, 2] >= si_low)
        & (points[:, 2] <= si_high)
    )
    incisal = points[incisal_mask]
    if len(incisal) < 16:
        incisal = points[(points[:, 2] >= si_low) & (points[:, 2] <= si_high)]
    if len(incisal) < 12:
        raise ValueError(f"牙位 {tooth} 切缘候选点不足：{len(incisal)}")

    # AP 正方向为前方；切缘候选中最前约 35% 作为唇侧切缘带。
    labial_threshold = float(np.quantile(incisal[:, 1], 0.65))
    labial = incisal[incisal[:, 1] >= labial_threshold]
    if len(labial) < 5:
        raise ValueError(f"牙位 {tooth} 唇侧切缘候选点不足：{len(labial)}")

    point = np.asarray(
        [
            np.median(labial[:, 0]),
            np.median(labial[:, 1]),
            np.median(incisal[:, 2]),
        ],
        dtype=np.float64,
    )
    return IncisalLandmark(
        tooth=tooth,
        arch=arch,
        point=point,
        crown_height_mm=crown_height,
        source_point_count=source_point_count,
        sampled_point_count=len(points),
        incisal_point_count=len(incisal),
        labial_point_count=len(labial),
        incisal_si_iqr_mm=float(
            np.quantile(incisal[:, 2], 0.75) - np.quantile(incisal[:, 2], 0.25)
        ),
        labial_ap_iqr_mm=float(
            np.quantile(labial[:, 1], 0.75) - np.quantile(labial[:, 1], 0.25)
        ),
        sampled_points=points,
        labial_edge_points=labial,
    )


def matched_surface_overjet(
    upper: IncisalLandmark,
    lower: IncisalLandmark,
    radius_mm: float,
) -> tuple[Optional[float], int, Optional[float]]:
    """在相近 LR/SI 位置寻找下切牙唇面，估计临床式水平覆盖。"""
    lower_points = lower.sampled_points
    tree = cKDTree(lower_points[:, [0, 2]])
    candidate_values = []

    upper_edge = upper.labial_edge_points
    if len(upper_edge) > 200:
        upper_edge = upper_edge[np.linspace(0, len(upper_edge) - 1, 200).astype(int)]

    for point in upper_edge:
        neighbor_indices = tree.query_ball_point(point[[0, 2]], r=radius_mm)
        if len(neighbor_indices) < 4:
            continue
        # 同一垂直切片可能同时穿过舌面和唇面，较高 AP 分位数代表唇面。
        lower_labial_ap = float(
            np.quantile(lower_points[np.asarray(neighbor_indices), 1], 0.85)
        )
        candidate_values.append(float(point[1] - lower_labial_ap))

    if len(candidate_values) < 8:
        return None, len(candidate_values), None

    values = np.asarray(candidate_values, dtype=np.float64)
    if len(values) >= 20:
        low, high = np.quantile(values, [0.05, 0.95])
        values = values[(values >= low) & (values <= high)]
    return (
        float(np.median(values)),
        len(candidate_values),
        float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    )


def _identity_qc(
    upper_points: np.ndarray,
    upper_labels: np.ndarray,
    lower_points: np.ndarray,
    lower_labels: np.ndarray,
) -> list[str]:
    """检查中央牙左右顺序和是否位于牙弓前方。"""
    centers: dict[int, np.ndarray] = {}
    for points, labels in ((upper_points, upper_labels), (lower_points, lower_labels)):
        local, _ = tooth_centroids(points, labels, minimum_points=80)
        centers.update(local)

    issues = []
    for right_tooth, left_tooth in ((11, 21), (41, 31)):
        if right_tooth in centers and left_tooth in centers:
            if centers[right_tooth][0] >= centers[left_tooth][0]:
                issues.append(f"left_right_order_invalid_{right_tooth}_{left_tooth}")

    posterior_teeth = [
        tooth
        for tooth in (14, 15, 16, 24, 25, 26, 44, 45, 46, 34, 35, 36)
        if tooth in centers
    ]
    if posterior_teeth:
        posterior_y = float(np.median([centers[tooth][1] for tooth in posterior_teeth]))
        for tooth in CENTRAL_TEETH:
            if tooth in centers and centers[tooth][1] <= posterior_y:
                issues.append(f"central_incisor_not_anterior_{tooth}")
    return issues


def _side_measurement(
    side: str,
    upper: IncisalLandmark,
    lower: IncisalLandmark,
    matched_radius_mm: float,
) -> dict[str, Any]:
    """计算单侧 tip overjet、surface overjet 和 overbite。"""
    overjet_tip = float(upper.point[1] - lower.point[1])
    overbite = float(lower.point[2] - upper.point[2])
    ratio = (
        float(overbite / lower.crown_height_mm)
        if lower.crown_height_mm > 1e-6
        else None
    )
    surface, matched_count, surface_iqr = matched_surface_overjet(
        upper, lower, matched_radius_mm
    )
    return {
        "side": side,
        "teeth": [upper.tooth, lower.tooth],
        "overjet_tip_mm": overjet_tip,
        "overjet_surface_mm": surface,
        "overjet_surface_matched_count": matched_count,
        "overjet_surface_iqr_mm": surface_iqr,
        "overbite_mm": overbite,
        "lower_projected_crown_height_mm": lower.crown_height_mm,
        "overbite_projected_crown_ratio": ratio,
    }


def _summary_from_sides(sides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """保留均值、极值和左右差，不预先决定最终临床汇总规则。"""
    summary: dict[str, Any] = {}
    metric_names = (
        "overjet_tip_mm",
        "overjet_surface_mm",
        "overbite_mm",
        "overbite_projected_crown_ratio",
    )
    for metric in metric_names:
        values = [
            float(side[metric])
            for side in sides.values()
            if side.get(metric) is not None and np.isfinite(side[metric])
        ]
        prefix = metric.removesuffix("_mm").removesuffix("_ratio")
        if values:
            suffix = "_ratio" if metric.endswith("_ratio") else "_mm"
            summary[f"{prefix}_mean{suffix}"] = float(np.mean(values))
            summary[f"{prefix}_min{suffix}"] = float(np.min(values))
            summary[f"{prefix}_max{suffix}"] = float(np.max(values))
            if len(values) == 2:
                summary[f"{prefix}_asymmetry{suffix}"] = float(abs(values[0] - values[1]))
    return summary


def measure_arrays(
    case_id: str,
    upper_points: np.ndarray,
    upper_labels: np.ndarray,
    lower_points: np.ndarray,
    lower_labels: np.ndarray,
    minimum_tooth_points: int = 200,
    voxel_size_mm: float = 0.15,
    matched_radius_mm: float = 0.80,
    maximum_roll_correction_deg: float = 12.0,
    selected_segmentation_status: Optional[str] = None,
    source: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], CaseArtifacts]:
    """从同一咬合坐标中的点和FDI标签计算连续overjet/overbite。"""
    upper_points = np.asarray(upper_points, dtype=np.float64)
    lower_points = np.asarray(lower_points, dtype=np.float64)
    upper_labels = np.asarray(upper_labels, dtype=np.int32).reshape(-1)
    lower_labels = np.asarray(lower_labels, dtype=np.int32).reshape(-1)
    if upper_points.ndim != 2 or upper_points.shape[1] != 3:
        raise ValueError(f"上颌坐标形状异常：{upper_points.shape}")
    if lower_points.ndim != 2 or lower_points.shape[1] != 3:
        raise ValueError(f"下颌坐标形状异常：{lower_points.shape}")
    if not np.isfinite(upper_points).all() or not np.isfinite(lower_points).all():
        raise ValueError("上下颌坐标含NaN或Inf")
    if len(upper_points) != len(upper_labels):
        raise ValueError(
            f"上颌点/标签数量不一致：{len(upper_points)} != {len(upper_labels)}"
        )
    if len(lower_points) != len(lower_labels):
        raise ValueError(
            f"下颌点/标签数量不一致：{len(lower_points)} != {len(lower_labels)}"
        )

    frame = build_shared_frame(
        upper_points,
        upper_labels,
        lower_points,
        lower_labels,
        maximum_roll_correction_deg=maximum_roll_correction_deg,
    )
    upper_local = frame.transform(upper_points)
    lower_local = frame.transform(lower_points)

    review_reasons: list[str] = []
    warnings: list[str] = []
    if frame.pair_count < 4:
        review_reasons.append("insufficient_symmetric_pairs")
    if frame.pair_dispersion_deg > 15.0:
        review_reasons.append("pair_dispersion_too_large")
    if not 5.0 <= frame.front_back_distance_mm <= 35.0:
        review_reasons.append("front_back_distance_implausible")
    if abs(frame.roll_correction_deg) >= maximum_roll_correction_deg - 1e-6:
        warnings.append("roll_correction_clipped")
    review_reasons.extend(
        _identity_qc(upper_local, upper_labels, lower_local, lower_labels)
    )
    if selected_segmentation_status == "hard_fail":
        review_reasons.append("selected_segmentation_hard_fail")
    elif selected_segmentation_status == "retry":
        warnings.append("selected_segmentation_needs_review")

    landmarks: dict[int, IncisalLandmark] = {}
    tooth_points: dict[int, np.ndarray] = {}
    landmark_errors: dict[int, str] = {}
    for tooth in CENTRAL_TEETH:
        arch = "upper" if tooth < 30 else "lower"
        points = upper_local if arch == "upper" else lower_local
        labels = upper_labels if arch == "upper" else lower_labels
        try:
            sampled, source_count = prepare_tooth_points(
                points,
                labels,
                tooth,
                minimum_tooth_points,
                voxel_size_mm,
            )
            landmark = extract_incisal_landmark(sampled, source_count, tooth, arch)
            landmarks[tooth] = landmark
            tooth_points[tooth] = sampled
            if not 4.0 <= landmark.crown_height_mm <= 18.0:
                review_reasons.append(f"tooth_{tooth}_crown_height_implausible")
        except Exception as exc:
            landmark_errors[tooth] = str(exc)
            review_reasons.append(f"landmark_unavailable_{tooth}")

    sides: dict[str, dict[str, Any]] = {}
    for side, (upper_tooth, lower_tooth) in SIDE_PAIRS.items():
        if upper_tooth not in landmarks or lower_tooth not in landmarks:
            continue
        sides[side] = _side_measurement(
            side,
            landmarks[upper_tooth],
            landmarks[lower_tooth],
            matched_radius_mm,
        )
        if sides[side]["overjet_surface_mm"] is None:
            warnings.append(f"surface_overjet_unavailable_{side}")

    summary = _summary_from_sides(sides)
    for side, values in sides.items():
        overjet = values["overjet_tip_mm"]
        overbite = values["overbite_mm"]
        if not -10.0 <= overjet <= 15.0:
            review_reasons.append(f"overjet_tip_implausible_{side}")
        if not -10.0 <= overbite <= 15.0:
            review_reasons.append(f"overbite_implausible_{side}")
    if summary.get("overjet_tip_asymmetry_mm", 0.0) > 5.0:
        review_reasons.append("overjet_side_asymmetry_large")
    if summary.get("overbite_asymmetry_mm", 0.0) > 5.0:
        review_reasons.append("overbite_side_asymmetry_large")

    if len(sides) < 2:
        status = "not_assessable"
    elif review_reasons:
        status = "review"
    else:
        status = "ok"
    review_reasons = list(dict.fromkeys(review_reasons))
    warnings = list(dict.fromkeys(warnings))

    source_payload = dict(source or {})
    source_payload.setdefault("units", "millimeters_assumed")
    source_payload.setdefault(
        "selected_segmentation_status",
        selected_segmentation_status,
    )
    result = {
        "case_id": case_id,
        "status": status,
        "method": {
            "name": "registered_fdi_central_incisor_geometry_v1",
            "description": (
                "在上下颌共同患者坐标系中，以11/41和21/31的稳健切缘带"
                "计算连续overjet与overbite。"
            ),
            "overjet_tip_sign": "positive means upper incisor is anterior",
            "overbite_sign": "positive overlap; negative suggests anterior open bite",
            "voxel_size_mm": voxel_size_mm,
            "matched_surface_radius_mm": matched_radius_mm,
        },
        "source": source_payload,
        "mapping": {
            "upper_vertices": len(upper_points),
            "upper_labels": len(upper_labels),
            "lower_vertices": len(lower_points),
            "lower_labels": len(lower_labels),
            "vertex_label_mapping_valid": True,
        },
        "frame": frame.to_result(),
        "landmarks": {
            str(tooth): landmarks[tooth].to_result() for tooth in sorted(landmarks)
        },
        "landmark_errors": {
            str(tooth): message for tooth, message in sorted(landmark_errors.items())
        },
        "sides": sides,
        "summary": summary,
        "quality": {
            "status": status,
            "review_reasons": review_reasons,
            "warnings": warnings,
            "clinical_validation_required": True,
        },
    }
    return result, CaseArtifacts(tooth_points=tooth_points, landmarks=landmarks)


def measure_case(
    case_id: str,
    upper_mesh_path: Path,
    upper_label_path: Path,
    lower_mesh_path: Path,
    lower_label_path: Path,
    minimum_tooth_points: int = 200,
    voxel_size_mm: float = 0.15,
    matched_radius_mm: float = 0.80,
    maximum_roll_correction_deg: float = 12.0,
    selected_segmentation_status: Optional[str] = None,
) -> tuple[dict[str, Any], CaseArtifacts]:
    """读取配对PLY/JSON并计算连续overjet与overbite。"""
    return measure_arrays(
        case_id=case_id,
        upper_points=load_ply_points(upper_mesh_path),
        upper_labels=load_labels(upper_label_path),
        lower_points=load_ply_points(lower_mesh_path),
        lower_labels=load_labels(lower_label_path),
        minimum_tooth_points=minimum_tooth_points,
        voxel_size_mm=voxel_size_mm,
        matched_radius_mm=matched_radius_mm,
        maximum_roll_correction_deg=maximum_roll_correction_deg,
        selected_segmentation_status=selected_segmentation_status,
        source={
            "upper_mesh": str(upper_mesh_path),
            "upper_labels": str(upper_label_path),
            "lower_mesh": str(lower_mesh_path),
            "lower_labels": str(lower_label_path),
            "units": "millimeters_assumed",
            "selected_segmentation_status": selected_segmentation_status,
        },
    )
