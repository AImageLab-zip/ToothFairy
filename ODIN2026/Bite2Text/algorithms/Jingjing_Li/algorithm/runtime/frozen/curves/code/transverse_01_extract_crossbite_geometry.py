#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_extract_crossbite_geometry.py

Crossbite v3 几何特征提取：
- raw registered upper/lower STL
- crowding 使用的逐 stable-vertex FDI JSON
- upper/lower 共同建立咬合平面（绝不分别 PCA）
- label==0 自动排除牙龈/底座
- 左右后牙分别建立局部 tangent / buccal outward
- 沿后牙段切片，计算 upper-lower buccal envelope margin
- 同时输出 full-crown 与 occlusal-band 两套整侧特征
- 保留 v2 的逐上颌后牙局部窗口特征（shared side axis）
- 新增 v3 tooth-specific local frame：每颗上颌后牙使用自己的局部 tangent / buccal outward
- 对 17/27 第二磨牙加入 endpoint protection，减少末端伪负 margin

本脚本不读取 GT、不搜阈值、不直接给 Crossbite True/False。
后续脚本再用 photo_en GT 做 5-fold OOF 阈值搜索。

margin = upper_buccal - lower_buccal
margin < 0 => crossbite tendency

v2 shared-axis 局部牙特征的定义：
- 上颌目标牙：14/15/16/17 或 24/25/26/27
- 下颌参考：同侧犬牙+后牙整体，不做 14↔44 这种硬配对
- 只在该上颌牙真实前后位置的局部切片中比较上下颌 buccal envelope
- 因此允许 Class II / III 的近远中错位，同时保留真实 registered 空间关系

v3 tooth-specific frame：
- 14/15/16 或 24/25/26：优先用相邻上颌牙中心的 secant 估计该牙局部 tangent
- 17/27：用末端最近两段的加权方向估计 tangent，并关闭/缩小窗口外扩
- 每颗牙的 outward 都由该牙中心相对全牙弓中心自动定符号
- upper/lower 始终使用同一个 registered 空间，只改变局部测量坐标轴，不分别对齐
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


RAW_ROOT = Path("data/raw_data")
LABEL_ROOT = Path("work/output")
WORK_ROOT = Path(__file__).resolve().parent
RESULT_DIR = WORK_ROOT / "results"
PROFILE_DIR = WORK_ROOT / "debug" / "profiles"

UPPER_FDI = {11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28}
LOWER_FDI = {31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48}

POSTERIOR = {
    "left":  {"upper": [24,25,26,27], "lower": [34,35,36,37]},
    "right": {"upper": [14,15,16,17], "lower": [44,45,46,47]},
}
CONTEXT = {
    "left":  {"upper": [23,24,25,26,27], "lower": [33,34,35,36,37]},
    "right": {"upper": [13,14,15,16,17], "lower": [43,44,45,46,47]},
}

# 从前向后排列，用于 v3 单牙局部 tangent。
UPPER_SIDE_SEQUENCE = {
    "left": [23,24,25,26,27],
    "right": [13,14,15,16,17],
}
ENDPOINT_SECOND_MOLARS = {17, 27}

STL_TRIANGLE_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


@dataclass(frozen=True)
class Arch:
    vertices: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class CommonFrame:
    origin: np.ndarray
    basis: np.ndarray
    vertical: np.ndarray
    eigenvalues: np.ndarray
    vertical_gap_mm: float


@dataclass(frozen=True)
class SideFrame:
    origin: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray
    eigenvalues: np.ndarray
    axis_ratio: float
    centers: dict[str, list[float]]


@dataclass(frozen=True)
class ToothFrame:
    tooth: int
    origin: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray
    source: str
    angle_to_side_deg: float
    local_curvature_deg: float | None
    centers: dict[str, list[float]]


def load_binary_stl_stable(path: Path) -> np.ndarray:
    """与 crowding 相同：二进制 STL 按顶点首次出现顺序稳定去重。"""
    size = path.stat().st_size
    if size < 84:
        raise ValueError(f"STL 文件过小: {path}")

    with path.open("rb") as f:
        f.seek(80)
        triangle_count = struct.unpack("<I", f.read(4))[0]

    expected = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    if size != expected:
        raise ValueError(
            f"只支持二进制 STL: {path}; actual={size}, expected={expected}"
        )

    triangles = np.fromfile(
        path,
        dtype=STL_TRIANGLE_DTYPE,
        count=triangle_count,
        offset=84,
    )
    raw = np.ascontiguousarray(triangles["vertices"].reshape(-1, 3))

    unique_sorted, first_indices = np.unique(
        raw,
        axis=0,
        return_index=True,
    )
    stable_order = np.argsort(first_indices)
    return unique_sorted[stable_order].astype(np.float64, copy=False)


def load_labels(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        labels = np.asarray(json.load(f), dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError(f"标签 JSON 必须是一维数组: {path}")
    return labels


def load_arch(
    patient_id: str,
    arch: str,
    raw_root: Path,
    label_root: Path,
) -> Arch:
    stl = raw_root / patient_id / "ios" / f"ios_{arch}.stl"
    lab = label_root / f"{patient_id}_ios_{arch}.json"

    if not stl.exists():
        raise FileNotFoundError(stl)
    if not lab.exists():
        raise FileNotFoundError(lab)

    vertices = load_binary_stl_stable(stl)
    labels = load_labels(lab)

    if len(vertices) != len(labels):
        raise ValueError(
            f"{patient_id} {arch}: stable vertices != labels: "
            f"{len(vertices)} != {len(labels)}"
        )

    allowed = UPPER_FDI if arch == "upper" else LOWER_FDI
    invalid = sorted(
        int(x) for x in np.unique(labels)
        if int(x) != 0 and int(x) not in allowed
    )
    if invalid:
        raise ValueError(f"{patient_id} {arch}: 非法 FDI labels={invalid}")

    return Arch(vertices=vertices, labels=labels)


def deterministic_sample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=np.int64)
    return points[idx]


def joint_occlusal_frame(
    upper: Arch,
    lower: Arch,
    min_tooth_points: int,
    max_points_per_tooth: int,
) -> CommonFrame:
    """
    每颗牙均衡采样后 upper+lower 一次 PCA。
    最小方差轴作为上下方向，其余两轴为共同咬合平面。
    """
    chunks = []

    for arch, allowed in ((upper, UPPER_FDI), (lower, LOWER_FDI)):
        for tooth in sorted(allowed):
            pts = arch.vertices[arch.labels == tooth]
            if len(pts) >= min_tooth_points:
                chunks.append(
                    deterministic_sample(pts, max_points_per_tooth)
                )

    if len(chunks) < 8:
        raise ValueError(f"共同 PCA 可用牙齿过少: {len(chunks)}")

    cloud = np.vstack(chunks)
    origin = np.median(cloud, axis=0)
    cov = np.cov(cloud - origin, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)

    vertical = vecs[:, 0].copy()
    basis = vecs[:, 1:3].copy()

    up_center = np.median(
        deterministic_sample(upper.vertices[upper.labels > 0], 30000),
        axis=0,
    )
    low_center = np.median(
        deterministic_sample(lower.vertices[lower.labels > 0], 30000),
        axis=0,
    )

    # 约定 vertical 正方向：lower -> upper
    if np.dot(up_center - low_center, vertical) < 0:
        vertical = -vertical

    gap = float(np.dot(up_center - low_center, vertical))

    return CommonFrame(
        origin=origin,
        basis=basis,
        vertical=vertical,
        eigenvalues=vals,
        vertical_gap_mm=gap,
    )


def project(
    arch: Arch,
    frame: CommonFrame,
) -> tuple[np.ndarray, np.ndarray]:
    centered = arch.vertices - frame.origin
    return centered @ frame.basis, centered @ frame.vertical


def tooth_centers(
    points_2d: np.ndarray,
    labels: np.ndarray,
    teeth: Iterable[int],
    min_points: int,
) -> dict[int, np.ndarray]:
    out = {}
    for tooth in teeth:
        mask = labels == tooth
        if np.count_nonzero(mask) >= min_points:
            out[tooth] = np.median(points_2d[mask], axis=0)
    return out


def global_arch_center(
    up2d: np.ndarray,
    uplab: np.ndarray,
    lo2d: np.ndarray,
    lolab: np.ndarray,
    min_points: int,
) -> np.ndarray:
    centers = []
    centers.extend(
        tooth_centers(up2d, uplab, sorted(UPPER_FDI), min_points).values()
    )
    centers.extend(
        tooth_centers(lo2d, lolab, sorted(LOWER_FDI), min_points).values()
    )
    if len(centers) < 8:
        raise ValueError("全牙弓有效牙齿中心过少")
    return np.median(np.vstack(centers), axis=0)


def build_side_frame(
    side: str,
    up2d: np.ndarray,
    uplab: np.ndarray,
    lo2d: np.ndarray,
    lolab: np.ndarray,
    arch_center: np.ndarray,
    min_points: int,
) -> SideFrame:
    """
    该侧犬牙+后牙 centers 的 PCA 最大方差轴作为前后 tangent；
    tangent 的垂线作为 transverse outward，并自动选朝牙弓外侧的符号。
    """
    records: list[tuple[int, np.ndarray]] = []
    center_json: dict[str, list[float]] = {}

    uc = tooth_centers(
        up2d, uplab, CONTEXT[side]["upper"], min_points
    )
    lc = tooth_centers(
        lo2d, lolab, CONTEXT[side]["lower"], min_points
    )

    for tooth, center in {**uc, **lc}.items():
        records.append((tooth, center))
        center_json[str(tooth)] = center.tolist()

    if len(records) < 4:
        raise ValueError(f"{side}: side-frame 中心不足: {len(records)}")

    centers = np.vstack([c for _, c in records])
    origin = np.median(centers, axis=0)
    cov = np.cov(centers - origin, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    tangent = vecs[:, 1].copy()

    anterior = [
        c for tooth, c in records
        if tooth % 10 in (3, 4, 5)
    ]
    posterior = [
        c for tooth, c in records
        if tooth % 10 in (6, 7)
    ]
    if anterior and posterior:
        direction = np.mean(posterior, axis=0) - np.mean(anterior, axis=0)
        if np.dot(tangent, direction) < 0:
            tangent = -tangent

    tangent /= np.linalg.norm(tangent)
    outward = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    if np.dot(outward, origin - arch_center) < 0:
        outward = -outward

    ratio = float(vals[1] / max(vals[0], 1e-9))

    return SideFrame(
        origin=origin,
        tangent=tangent,
        outward=outward,
        eigenvalues=vals,
        axis_ratio=ratio,
        centers=center_json,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length direction vector")
    return v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def build_tooth_frame(
    side: str,
    target_tooth: int,
    up2d: np.ndarray,
    uplab: np.ndarray,
    arch_center: np.ndarray,
    side_frame: SideFrame,
    min_points: int,
) -> ToothFrame:
    """
    v3：为每颗上颌后牙单独估计局部 tangent / outward。

    内部牙（14/15/16, 24/25/26）优先用前后相邻牙中心的 secant：
        tangent ~ center(next) - center(prev)

    17/27 是牙弓末端，不能用跨越整个 side 的固定 PCA 方向。
    若最近两段都存在，则使用：
        0.75 * unit(last_segment) + 0.25 * unit(previous_segment)
    既保持末端真实方向，又稍微抑制单个中心噪声。

    所有中心都来自 upper FDI labels；lower 不参与方向估计，
    因此不会因为对颌错位把测量轴本身带偏。
    """
    seq = UPPER_SIDE_SEQUENCE[side]
    if target_tooth not in seq:
        raise ValueError(f"{side}: unsupported target tooth {target_tooth}")

    centers = tooth_centers(
        up2d, uplab, seq, min_points
    )
    if target_tooth not in centers:
        raise ValueError(f"upper tooth {target_tooth} center unavailable")

    idx = seq.index(target_tooth)
    target = centers[target_tooth]

    prev1 = seq[idx - 1] if idx - 1 >= 0 else None
    prev2 = seq[idx - 2] if idx - 2 >= 0 else None
    next1 = seq[idx + 1] if idx + 1 < len(seq) else None
    next2 = seq[idx + 2] if idx + 2 < len(seq) else None

    tangent = None
    source = ""
    curvature = None

    # 首选：目标牙前后相邻中心的 secant，最适合弯曲牙弓。
    if prev1 in centers and next1 in centers:
        tangent = centers[next1] - centers[prev1]
        source = f"secant_{prev1}_{next1}"
        try:
            seg_in = _unit(target - centers[prev1])
            seg_out = _unit(centers[next1] - target)
            curvature = _angle_deg(seg_in, seg_out)
        except Exception:
            curvature = None

    # 末端第二磨牙：最近一段为主，前一段只做平滑。
    elif prev1 in centers and prev2 in centers:
        last_seg = _unit(target - centers[prev1])
        prev_seg = _unit(centers[prev1] - centers[prev2])
        tangent = 0.75 * last_seg + 0.25 * prev_seg
        source = f"endpoint_blend_{prev2}_{prev1}_{target_tooth}"
        curvature = _angle_deg(prev_seg, last_seg)

    # 理论上当前目标牙不会走到这个分支，但保留对缺牙情况的稳健回退。
    elif next1 in centers and next2 in centers:
        first_seg = _unit(centers[next1] - target)
        next_seg = _unit(centers[next2] - centers[next1])
        tangent = 0.75 * first_seg + 0.25 * next_seg
        source = f"anterior_blend_{target_tooth}_{next1}_{next2}"
        curvature = _angle_deg(first_seg, next_seg)
    elif prev1 in centers:
        tangent = target - centers[prev1]
        source = f"single_prev_{prev1}_{target_tooth}"
    elif next1 in centers:
        tangent = centers[next1] - target
        source = f"single_next_{target_tooth}_{next1}"
    else:
        tangent = side_frame.tangent.copy()
        source = "fallback_side_frame"

    tangent = _unit(tangent)

    # 与 side-frame 保持统一的 anterior -> posterior 符号。
    if float(np.dot(tangent, side_frame.tangent)) < 0:
        tangent = -tangent

    outward = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    outward = _unit(outward)

    # outward 必须指向牙弓外侧。
    radial = target - arch_center
    if float(np.dot(outward, radial)) < 0:
        outward = -outward

    angle = _angle_deg(tangent, side_frame.tangent)

    return ToothFrame(
        tooth=int(target_tooth),
        origin=target.copy(),
        tangent=tangent,
        outward=outward,
        source=source,
        angle_to_side_deg=angle,
        local_curvature_deg=curvature,
        centers={str(k): v.tolist() for k, v in centers.items()},
    )


def labels_mask(labels: np.ndarray, teeth: Iterable[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(list(teeth), dtype=np.int32))


def occlusal_band_mask(
    vertical: np.ndarray,
    labels: np.ndarray,
    teeth: Iterable[int],
    arch: str,
    fraction: float,
    min_points: int,
) -> np.ndarray:
    """
    vertical 正方向 lower -> upper。
    upper 咬合面朝 lower：每颗牙取 vertical 低端 fraction。
    lower 咬合面朝 upper：每颗牙取 vertical 高端 fraction。
    """
    out = np.zeros(len(labels), dtype=bool)

    for tooth in teeth:
        tm = labels == tooth
        values = vertical[tm]
        if len(values) < min_points:
            continue

        if arch == "upper":
            threshold = np.quantile(values, fraction)
            out |= tm & (vertical <= threshold)
        else:
            threshold = np.quantile(values, 1.0 - fraction)
            out |= tm & (vertical >= threshold)

    return out


def longest_run(
    centers: np.ndarray,
    margins: np.ndarray,
    threshold: float,
    step: float,
) -> float:
    best = 0.0
    current = 0.0
    previous = None

    for s, margin in zip(centers, margins):
        if margin >= threshold:
            current = 0.0
            previous = None
            continue

        if previous is not None and s - previous <= step * 1.51:
            current += float(s - previous)
        else:
            current = step

        best = max(best, current)
        previous = s

    return float(best)


def summarize_profile(
    centers: np.ndarray,
    margins: np.ndarray,
    step: float,
) -> dict:
    if len(margins) == 0:
        return {
            "n_slices": 0,
            "coverage_mm": 0.0,
            "mean_margin_mm": None,
            "median_margin_mm": None,
            "q10_margin_mm": None,
            "q20_margin_mm": None,
            "min_margin_mm": None,
            "std_margin_mm": None,
            "negative_ratio": None,
            "negative_05_ratio": None,
            "negative_10_ratio": None,
            "longest_negative_run_mm": 0.0,
            "longest_negative_05_run_mm": 0.0,
            "negative_area_mm2": 0.0,
        }

    centers = np.asarray(centers, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)

    coverage = (
        step
        if len(centers) == 1
        else float(centers[-1] - centers[0] + step)
    )

    return {
        "n_slices": int(len(margins)),
        "coverage_mm": coverage,
        "mean_margin_mm": float(np.mean(margins)),
        "median_margin_mm": float(np.median(margins)),
        "q10_margin_mm": float(np.quantile(margins, 0.10)),
        "q20_margin_mm": float(np.quantile(margins, 0.20)),
        "min_margin_mm": float(np.min(margins)),
        "std_margin_mm": float(np.std(margins)),
        "negative_ratio": float(np.mean(margins < 0.0)),
        "negative_05_ratio": float(np.mean(margins < -0.5)),
        "negative_10_ratio": float(np.mean(margins < -1.0)),
        "longest_negative_run_mm": longest_run(
            centers, margins, 0.0, step
        ),
        "longest_negative_05_run_mm": longest_run(
            centers, margins, -0.5, step
        ),
        "negative_area_mm2": float(
            np.sum(np.maximum(0.0, -margins)) * step
        ),
    }


def slice_profile(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    frame: SideFrame,
    step: float,
    half_width: float,
    outer_quantile: float,
    min_slice_points: int,
    range_quantile: float,
) -> tuple[dict, dict]:
    up = upper_points - frame.origin
    lo = lower_points - frame.origin

    up_s = up @ frame.tangent
    lo_s = lo @ frame.tangent
    up_u = up @ frame.outward
    lo_u = lo @ frame.outward

    up_lo, up_hi = np.quantile(
        up_s, [range_quantile, 1.0 - range_quantile]
    )
    lo_lo, lo_hi = np.quantile(
        lo_s, [range_quantile, 1.0 - range_quantile]
    )

    start = float(max(up_lo, lo_lo))
    end = float(min(up_hi, lo_hi))
    if end <= start:
        raise ValueError("upper/lower 后牙 s 范围没有重叠")

    requested = np.arange(start, end + step * 0.5, step)

    valid_s = []
    margins = []
    up_outer_list = []
    lo_outer_list = []
    up_counts = []
    lo_counts = []

    for s in requested:
        um = np.abs(up_s - s) <= half_width
        lm = np.abs(lo_s - s) <= half_width

        uc = int(np.count_nonzero(um))
        lc = int(np.count_nonzero(lm))

        if uc < min_slice_points or lc < min_slice_points:
            continue

        up_outer = float(np.quantile(up_u[um], outer_quantile))
        lo_outer = float(np.quantile(lo_u[lm], outer_quantile))

        valid_s.append(float(s))
        margins.append(up_outer - lo_outer)
        up_outer_list.append(up_outer)
        lo_outer_list.append(lo_outer)
        up_counts.append(uc)
        lo_counts.append(lc)

    valid_s_arr = np.asarray(valid_s, dtype=np.float64)
    margins_arr = np.asarray(margins, dtype=np.float64)

    features = summarize_profile(valid_s_arr, margins_arr, step)
    features.update({
        "valid_slice_ratio": (
            float(len(valid_s) / len(requested)) if len(requested) else 0.0
        ),
        "upper_points": int(len(upper_points)),
        "lower_points": int(len(lower_points)),
        "requested_length_mm": float(end - start),
    })

    profile = {
        "s_mm": valid_s,
        "margin_mm": [float(x) for x in margins],
        "upper_outer_u_mm": up_outer_list,
        "lower_outer_u_mm": lo_outer_list,
        "upper_slice_points": up_counts,
        "lower_slice_points": lo_counts,
    }

    return features, profile



def local_tooth_profile(
    upper_points: np.ndarray,
    lower_side_points: np.ndarray,
    frame: SideFrame | ToothFrame,
    step: float,
    half_width: float,
    outer_quantile: float,
    min_slice_points: int,
    range_quantile: float,
    local_padding_mm: float,
) -> tuple[dict, dict]:
    """
    计算单颗上颌后牙的局部 transverse profile。

    关键点：
    1. upper_points 只来自一颗目标上颌牙；
    2. lower_side_points 来自同侧犬牙+后牙整体，而不是硬配一颗下牙；
    3. 局部 s 范围由目标上颌牙自身范围确定，并可向近远中各扩张少量 padding；
    4. 在 registered 共同坐标系中比较同一真实前后位置的 buccal envelope。
    """
    if len(upper_points) < min_slice_points:
        raise ValueError(
            f"目标上颌牙点数不足: {len(upper_points)} < {min_slice_points}"
        )
    if len(lower_side_points) < min_slice_points:
        raise ValueError(
            f"下颌同侧参考点不足: {len(lower_side_points)} < {min_slice_points}"
        )

    up = upper_points - frame.origin
    lo = lower_side_points - frame.origin

    up_s = up @ frame.tangent
    lo_s = lo @ frame.tangent
    up_u = up @ frame.outward
    lo_u = lo @ frame.outward

    # 局部前后窗口主要由这颗上颌牙本身决定。
    # 2%-98% 去掉极端离群点，再向两端扩张少量毫米。
    up_lo, up_hi = np.quantile(
        up_s, [range_quantile, 1.0 - range_quantile]
    )
    target_start = float(up_lo - local_padding_mm)
    target_end = float(up_hi + local_padding_mm)

    # 下颌必须在真实 registered 空间里与该窗口有重叠。
    lo_lo, lo_hi = np.quantile(
        lo_s, [range_quantile, 1.0 - range_quantile]
    )
    start = float(max(target_start, lo_lo))
    end = float(min(target_end, lo_hi))
    if end <= start:
        raise ValueError("目标上颌牙与下颌同侧参考区域没有 s 重叠")

    requested = np.arange(start, end + step * 0.5, step)

    valid_s = []
    margins = []
    up_outer_list = []
    lo_outer_list = []
    up_counts = []
    lo_counts = []

    for s in requested:
        um = np.abs(up_s - s) <= half_width
        lm = np.abs(lo_s - s) <= half_width

        uc = int(np.count_nonzero(um))
        lc = int(np.count_nonzero(lm))
        if uc < min_slice_points or lc < min_slice_points:
            continue

        up_outer = float(np.quantile(up_u[um], outer_quantile))
        lo_outer = float(np.quantile(lo_u[lm], outer_quantile))

        valid_s.append(float(s))
        margins.append(up_outer - lo_outer)
        up_outer_list.append(up_outer)
        lo_outer_list.append(lo_outer)
        up_counts.append(uc)
        lo_counts.append(lc)

    valid_s_arr = np.asarray(valid_s, dtype=np.float64)
    margins_arr = np.asarray(margins, dtype=np.float64)

    features = summarize_profile(valid_s_arr, margins_arr, step)
    features.update({
        "valid_slice_ratio": (
            float(len(valid_s) / len(requested)) if len(requested) else 0.0
        ),
        "upper_points": int(len(upper_points)),
        "lower_points": int(len(lower_side_points)),
        "requested_length_mm": float(end - start),
        "target_start_s_mm": target_start,
        "target_end_s_mm": target_end,
        "actual_start_s_mm": start,
        "actual_end_s_mm": end,
    })

    profile = {
        "s_mm": valid_s,
        "margin_mm": [float(x) for x in margins],
        "upper_outer_u_mm": up_outer_list,
        "lower_outer_u_mm": lo_outer_list,
        "upper_slice_points": up_counts,
        "lower_slice_points": lo_counts,
        "target_start_s_mm": target_start,
        "target_end_s_mm": target_end,
        "actual_start_s_mm": start,
        "actual_end_s_mm": end,
    }
    return features, profile


def aggregate_local_teeth(
    tooth_results: dict[int, dict],
) -> dict:
    """
    将逐牙局部结果聚合成 side-level 局部特征。

    不在这里决定 Crossbite True/False；只输出便于后续 OOF 学阈值的特征。
    """
    valid = {
        int(tooth): feat
        for tooth, feat in tooth_results.items()
        if feat.get("valid")
        and feat.get("q20_margin_mm") is not None
        and feat.get("median_margin_mm") is not None
    }

    if not valid:
        return {
            "valid_teeth": 0,
            "worst_q20_margin_mm": None,
            "second_worst_q20_margin_mm": None,
            "worst_two_q20_mean_mm": None,
            "worst_q20_tooth": None,
            "worst_median_margin_mm": None,
            "second_worst_median_margin_mm": None,
            "worst_two_median_mean_mm": None,
            "worst_median_tooth": None,
            "max_negative_ratio": None,
            "max_negative_05_ratio": None,
            "max_negative_10_ratio": None,
            "max_longest_negative_run_mm": None,
            "max_longest_negative_05_run_mm": None,
            "max_negative_area_mm2": None,
            "q20_below_0_count": 0,
            "q20_below_05_count": 0,
            "median_below_0_count": 0,
        }

    q20_sorted = sorted(
        ((float(f["q20_margin_mm"]), t) for t, f in valid.items()),
        key=lambda x: x[0],
    )
    med_sorted = sorted(
        ((float(f["median_margin_mm"]), t) for t, f in valid.items()),
        key=lambda x: x[0],
    )

    q20_values = [x[0] for x in q20_sorted]
    med_values = [x[0] for x in med_sorted]

    def second_or_none(values):
        return float(values[1]) if len(values) >= 2 else None

    def first_two_mean(values):
        take = values[:2]
        return float(np.mean(take)) if take else None

    return {
        "valid_teeth": int(len(valid)),
        "worst_q20_margin_mm": float(q20_values[0]),
        "second_worst_q20_margin_mm": second_or_none(q20_values),
        "worst_two_q20_mean_mm": first_two_mean(q20_values),
        "worst_q20_tooth": int(q20_sorted[0][1]),
        "worst_median_margin_mm": float(med_values[0]),
        "second_worst_median_margin_mm": second_or_none(med_values),
        "worst_two_median_mean_mm": first_two_mean(med_values),
        "worst_median_tooth": int(med_sorted[0][1]),
        "max_negative_ratio": float(max(f["negative_ratio"] for f in valid.values())),
        "max_negative_05_ratio": float(max(f["negative_05_ratio"] for f in valid.values())),
        "max_negative_10_ratio": float(max(f["negative_10_ratio"] for f in valid.values())),
        "max_longest_negative_run_mm": float(
            max(f["longest_negative_run_mm"] for f in valid.values())
        ),
        "max_longest_negative_05_run_mm": float(
            max(f["longest_negative_05_run_mm"] for f in valid.values())
        ),
        "max_negative_area_mm2": float(
            max(f["negative_area_mm2"] for f in valid.values())
        ),
        "q20_below_0_count": int(
            sum(float(f["q20_margin_mm"]) < 0.0 for f in valid.values())
        ),
        "q20_below_05_count": int(
            sum(float(f["q20_margin_mm"]) < -0.5 for f in valid.values())
        ),
        "median_below_0_count": int(
            sum(float(f["median_margin_mm"]) < 0.0 for f in valid.values())
        ),
    }


def flatten(row: dict, prefix: str, values: dict) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def analyze_patient(patient_id: str, args) -> tuple[dict, dict]:
    upper = load_arch(patient_id, "upper", args.raw_root, args.label_root)
    lower = load_arch(patient_id, "lower", args.raw_root, args.label_root)

    frame = joint_occlusal_frame(
        upper,
        lower,
        args.minimum_tooth_points,
        args.pca_points_per_tooth,
    )
    up2d, up_vertical = project(upper, frame)
    lo2d, lo_vertical = project(lower, frame)

    arch_center = global_arch_center(
        up2d, upper.labels, lo2d, lower.labels,
        args.minimum_tooth_points,
    )

    row = {
        "patient_id": patient_id,
        "geometry_ok": True,
        "error": "",
        "upper_stable_vertices": len(upper.vertices),
        "lower_stable_vertices": len(lower.vertices),
        "upper_tooth_vertices": int(np.count_nonzero(upper.labels > 0)),
        "lower_tooth_vertices": int(np.count_nonzero(lower.labels > 0)),
        "common_eigenvalue_0": float(frame.eigenvalues[0]),
        "common_eigenvalue_1": float(frame.eigenvalues[1]),
        "common_eigenvalue_2": float(frame.eigenvalues[2]),
        "vertical_gap_mm": frame.vertical_gap_mm,
    }

    debug = {
        "patient_id": patient_id,
        "margin_definition": (
            "upper_buccal_envelope - lower_buccal_envelope; "
            "negative means crossbite tendency; v3 keeps v2 shared-axis local windows and adds tooth-specific local frames"
        ),
        "common_frame": {
            "origin": frame.origin.tolist(),
            "basis": frame.basis.tolist(),
            "vertical": frame.vertical.tolist(),
            "eigenvalues": frame.eigenvalues.tolist(),
            "vertical_gap_mm": frame.vertical_gap_mm,
        },
        "sides": {},
    }

    for side in ("left", "right"):
        sf = build_side_frame(
            side,
            up2d,
            upper.labels,
            lo2d,
            lower.labels,
            arch_center,
            args.minimum_tooth_points,
        )

        flatten(row, f"{side}_frame", {
            "axis_ratio": sf.axis_ratio,
            "eig0": float(sf.eigenvalues[0]),
            "eig1": float(sf.eigenvalues[1]),
        })

        up_full_mask = labels_mask(
            upper.labels, POSTERIOR[side]["upper"]
        )
        lo_full_mask = labels_mask(
            lower.labels, POSTERIOR[side]["lower"]
        )

        up_occ_mask = (
            up_full_mask
            & occlusal_band_mask(
                up_vertical,
                upper.labels,
                POSTERIOR[side]["upper"],
                "upper",
                args.occlusal_fraction,
                args.minimum_tooth_points,
            )
        )
        lo_occ_mask = (
            lo_full_mask
            & occlusal_band_mask(
                lo_vertical,
                lower.labels,
                POSTERIOR[side]["lower"],
                "lower",
                args.occlusal_fraction,
                args.minimum_tooth_points,
            )
        )

        variants = {
            "full": (up_full_mask, lo_full_mask),
            "occlusal": (up_occ_mask, lo_occ_mask),
        }

        side_debug = {
            "frame": {
                "origin": sf.origin.tolist(),
                "tangent": sf.tangent.tolist(),
                "outward": sf.outward.tolist(),
                "axis_ratio": sf.axis_ratio,
                "context_centers": sf.centers,
            },
            "variants": {},
        }

        for name, (um, lm) in variants.items():
            try:
                features, profile = slice_profile(
                    up2d[um],
                    lo2d[lm],
                    sf,
                    args.slice_step_mm,
                    args.slice_half_width_mm,
                    args.outer_quantile,
                    args.minimum_slice_points,
                    args.range_quantile,
                )
                features["valid"] = True
                features["error"] = ""
            except Exception as e:
                features = {
                    "valid": False,
                    "error": repr(e),
                    "n_slices": 0,
                    "coverage_mm": 0.0,
                }
                profile = {"error": repr(e)}

            flatten(row, f"{side}_{name}", features)
            side_debug["variants"][name] = {
                "features": features,
                "profile": profile,
            }

        # ------------------------------------------------------------
        # v2: 逐上颌后牙局部特征
        # ------------------------------------------------------------
        lower_context_teeth = CONTEXT[side]["lower"]
        lower_context_full_mask = labels_mask(
            lower.labels, lower_context_teeth
        )
        lower_context_occ_mask = (
            lower_context_full_mask
            & occlusal_band_mask(
                lo_vertical,
                lower.labels,
                lower_context_teeth,
                "lower",
                args.occlusal_fraction,
                args.minimum_tooth_points,
            )
        )

        local_debug = {"full": {}, "occlusal": {}}
        local_results = {"full": {}, "occlusal": {}}

        for tooth in POSTERIOR[side]["upper"]:
            upper_tooth_full = upper.labels == tooth
            tooth_count = int(np.count_nonzero(upper_tooth_full))

            # 明确记录缺失/点数不足，而不是把它伪装成正常。
            if tooth_count < args.minimum_tooth_points:
                for variant_name in ("full", "occlusal"):
                    feat = {
                        "valid": False,
                        "error": (
                            f"upper tooth {tooth} points insufficient: "
                            f"{tooth_count} < {args.minimum_tooth_points}"
                        ),
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": tooth_count,
                    }
                    local_results[variant_name][tooth] = feat
                    local_debug[variant_name][str(tooth)] = {
                        "features": feat,
                        "profile": {"error": feat["error"]},
                    }
                    flatten(row, f"{side}_tooth{tooth}_{variant_name}", feat)
                continue

            upper_tooth_occ = (
                upper_tooth_full
                & occlusal_band_mask(
                    up_vertical,
                    upper.labels,
                    [tooth],
                    "upper",
                    args.occlusal_fraction,
                    args.minimum_tooth_points,
                )
            )

            local_variants = {
                "full": (
                    up2d[upper_tooth_full],
                    lo2d[lower_context_full_mask],
                ),
                "occlusal": (
                    up2d[upper_tooth_occ],
                    lo2d[lower_context_occ_mask],
                ),
            }

            for variant_name, (upper_local_points, lower_local_points) in local_variants.items():
                try:
                    feat, profile = local_tooth_profile(
                        upper_local_points,
                        lower_local_points,
                        sf,
                        args.local_slice_step_mm,
                        args.local_slice_half_width_mm,
                        args.outer_quantile,
                        args.local_minimum_slice_points,
                        args.local_range_quantile,
                        args.local_padding_mm,
                    )
                    feat["valid"] = True
                    feat["error"] = ""
                except Exception as e:
                    feat = {
                        "valid": False,
                        "error": repr(e),
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": int(len(upper_local_points)),
                        "lower_points": int(len(lower_local_points)),
                    }
                    profile = {"error": repr(e)}

                local_results[variant_name][tooth] = feat
                local_debug[variant_name][str(tooth)] = {
                    "features": feat,
                    "profile": profile,
                }
                flatten(row, f"{side}_tooth{tooth}_{variant_name}", feat)

        for variant_name in ("full", "occlusal"):
            aggregate = aggregate_local_teeth(local_results[variant_name])
            flatten(row, f"{side}_local_{variant_name}", aggregate)
            local_debug[variant_name]["aggregate"] = aggregate

        side_debug["local_teeth"] = local_debug

        # ------------------------------------------------------------
        # v3: 每颗上颌后牙自己的 tooth-specific local frame
        # ------------------------------------------------------------
        tf_debug = {"full": {}, "occlusal": {}, "frames": {}}
        tf_results = {"full": {}, "occlusal": {}}

        for tooth in POSTERIOR[side]["upper"]:
            upper_tooth_full = upper.labels == tooth
            tooth_count = int(np.count_nonzero(upper_tooth_full))

            if tooth_count < args.minimum_tooth_points:
                for variant_name in ("full", "occlusal"):
                    feat = {
                        "valid": False,
                        "error": (
                            f"upper tooth {tooth} points insufficient: "
                            f"{tooth_count} < {args.minimum_tooth_points}"
                        ),
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": tooth_count,
                    }
                    tf_results[variant_name][tooth] = feat
                    tf_debug[variant_name][str(tooth)] = {
                        "features": feat,
                        "profile": {"error": feat["error"]},
                    }
                    flatten(row, f"{side}_tf_tooth{tooth}_{variant_name}", feat)
                continue

            try:
                tf = build_tooth_frame(
                    side,
                    tooth,
                    up2d,
                    upper.labels,
                    arch_center,
                    sf,
                    args.minimum_tooth_points,
                )
                tf_frame_features = {
                    "tangent_x": float(tf.tangent[0]),
                    "tangent_y": float(tf.tangent[1]),
                    "outward_x": float(tf.outward[0]),
                    "outward_y": float(tf.outward[1]),
                    "angle_to_side_deg": float(tf.angle_to_side_deg),
                    "local_curvature_deg": (
                        None if tf.local_curvature_deg is None
                        else float(tf.local_curvature_deg)
                    ),
                    "source": tf.source,
                    "endpoint_second_molar": bool(tooth in ENDPOINT_SECOND_MOLARS),
                }
                flatten(row, f"{side}_tf_frame_tooth{tooth}", tf_frame_features)
                tf_debug["frames"][str(tooth)] = {
                    **tf_frame_features,
                    "origin": tf.origin.tolist(),
                    "centers": tf.centers,
                }
            except Exception as e:
                err = repr(e)
                for variant_name in ("full", "occlusal"):
                    feat = {
                        "valid": False,
                        "error": err,
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": tooth_count,
                    }
                    tf_results[variant_name][tooth] = feat
                    tf_debug[variant_name][str(tooth)] = {
                        "features": feat,
                        "profile": {"error": err},
                    }
                    flatten(row, f"{side}_tf_tooth{tooth}_{variant_name}", feat)
                tf_debug["frames"][str(tooth)] = {"error": err}
                continue

            upper_tooth_occ = (
                upper_tooth_full
                & occlusal_band_mask(
                    up_vertical,
                    upper.labels,
                    [tooth],
                    "upper",
                    args.occlusal_fraction,
                    args.minimum_tooth_points,
                )
            )

            endpoint = tooth in ENDPOINT_SECOND_MOLARS
            padding = (
                args.tooth_frame_endpoint_padding_mm
                if endpoint
                else args.tooth_frame_padding_mm
            )
            tf_range_quantile = (
                args.tooth_frame_endpoint_range_quantile
                if endpoint
                else args.tooth_frame_range_quantile
            )
            tf_min_slice_points = (
                args.tooth_frame_endpoint_minimum_slice_points
                if endpoint
                else args.tooth_frame_minimum_slice_points
            )

            tf_local_variants = {
                "full": (
                    up2d[upper_tooth_full],
                    lo2d[lower_context_full_mask],
                ),
                "occlusal": (
                    up2d[upper_tooth_occ],
                    lo2d[lower_context_occ_mask],
                ),
            }

            for variant_name, (upper_local_points, lower_local_points) in tf_local_variants.items():
                try:
                    feat, profile = local_tooth_profile(
                        upper_local_points,
                        lower_local_points,
                        tf,
                        args.tooth_frame_slice_step_mm,
                        args.tooth_frame_slice_half_width_mm,
                        args.outer_quantile,
                        tf_min_slice_points,
                        tf_range_quantile,
                        padding,
                    )
                    feat["valid"] = True
                    feat["error"] = ""
                    feat["frame_angle_to_side_deg"] = float(tf.angle_to_side_deg)
                    feat["frame_local_curvature_deg"] = (
                        None if tf.local_curvature_deg is None
                        else float(tf.local_curvature_deg)
                    )
                    feat["endpoint_second_molar"] = bool(endpoint)
                    feat["padding_mm_used"] = float(padding)
                    feat["range_quantile_used"] = float(tf_range_quantile)
                except Exception as e:
                    feat = {
                        "valid": False,
                        "error": repr(e),
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": int(len(upper_local_points)),
                        "lower_points": int(len(lower_local_points)),
                        "frame_angle_to_side_deg": float(tf.angle_to_side_deg),
                        "frame_local_curvature_deg": (
                            None if tf.local_curvature_deg is None
                            else float(tf.local_curvature_deg)
                        ),
                        "endpoint_second_molar": bool(endpoint),
                        "padding_mm_used": float(padding),
                        "range_quantile_used": float(tf_range_quantile),
                    }
                    profile = {"error": repr(e)}

                tf_results[variant_name][tooth] = feat
                tf_debug[variant_name][str(tooth)] = {
                    "features": feat,
                    "profile": profile,
                }
                flatten(row, f"{side}_tf_tooth{tooth}_{variant_name}", feat)

        for variant_name in ("full", "occlusal"):
            aggregate = aggregate_local_teeth(tf_results[variant_name])
            flatten(row, f"{side}_tf_local_{variant_name}", aggregate)
            tf_debug[variant_name]["aggregate"] = aggregate

        side_debug["tooth_specific_local"] = tf_debug
        debug["sides"][side] = side_debug

    row["local_geometry_valid"] = bool(
        (row.get("left_local_occlusal_valid_teeth") or 0) >= 2
        and (row.get("right_local_occlusal_valid_teeth") or 0) >= 2
    )

    row["tooth_frame_geometry_valid"] = bool(
        (row.get("left_tf_local_occlusal_valid_teeth") or 0) >= 2
        and (row.get("right_tf_local_occlusal_valid_teeth") or 0) >= 2
    )

    row["primary_geometry_valid"] = bool(
        row.get("left_occlusal_valid")
        and row.get("right_occlusal_valid")
        and row.get("tooth_frame_geometry_valid")
    )

    return row, debug


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compact_key_row(row: dict) -> dict:
    """
    生成便于人工比较 v2 shared-axis 与 v3 tooth-frame 的小 CSV。
    完整特征仍保存在 features_v3.csv。
    """
    out = {
        "patient_id": row.get("patient_id"),
        "geometry_ok": row.get("geometry_ok"),
        "primary_geometry_valid": row.get("primary_geometry_valid"),
        "vertical_gap_mm": row.get("vertical_gap_mm"),
    }

    for side in ("left", "right"):
        out[f"{side}_whole_occ_q20"] = row.get(
            f"{side}_occlusal_q20_margin_mm"
        )
        out[f"{side}_whole_occ_median"] = row.get(
            f"{side}_occlusal_median_margin_mm"
        )
        out[f"{side}_whole_occ_neg_ratio"] = row.get(
            f"{side}_occlusal_negative_ratio"
        )

        out[f"{side}_v2_worst_q20"] = row.get(
            f"{side}_local_occlusal_worst_q20_margin_mm"
        )
        out[f"{side}_v2_worst_tooth"] = row.get(
            f"{side}_local_occlusal_worst_q20_tooth"
        )
        out[f"{side}_v3_worst_q20"] = row.get(
            f"{side}_tf_local_occlusal_worst_q20_margin_mm"
        )
        out[f"{side}_v3_second_worst_q20"] = row.get(
            f"{side}_tf_local_occlusal_second_worst_q20_margin_mm"
        )
        out[f"{side}_v3_worst_tooth"] = row.get(
            f"{side}_tf_local_occlusal_worst_q20_tooth"
        )
        out[f"{side}_v3_q20_below0_count"] = row.get(
            f"{side}_tf_local_occlusal_q20_below_0_count"
        )

        for tooth in POSTERIOR[side]["upper"]:
            # v2 shared-axis
            out[f"{side}_{tooth}_v2_q20"] = row.get(
                f"{side}_tooth{tooth}_occlusal_q20_margin_mm"
            )
            out[f"{side}_{tooth}_v2_median"] = row.get(
                f"{side}_tooth{tooth}_occlusal_median_margin_mm"
            )
            out[f"{side}_{tooth}_v2_neg"] = row.get(
                f"{side}_tooth{tooth}_occlusal_negative_ratio"
            )

            # v3 tooth-specific frame
            out[f"{side}_{tooth}_v3_q20"] = row.get(
                f"{side}_tf_tooth{tooth}_occlusal_q20_margin_mm"
            )
            out[f"{side}_{tooth}_v3_median"] = row.get(
                f"{side}_tf_tooth{tooth}_occlusal_median_margin_mm"
            )
            out[f"{side}_{tooth}_v3_neg"] = row.get(
                f"{side}_tf_tooth{tooth}_occlusal_negative_ratio"
            )
            out[f"{side}_{tooth}_angle_deg"] = row.get(
                f"{side}_tf_frame_tooth{tooth}_angle_to_side_deg"
            )
            out[f"{side}_{tooth}_curvature_deg"] = row.get(
                f"{side}_tf_frame_tooth{tooth}_local_curvature_deg"
            )
            out[f"{side}_{tooth}_frame_source"] = row.get(
                f"{side}_tf_frame_tooth{tooth}_source"
            )

    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    p.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    p.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    p.add_argument("--patient-id", action="append", default=[])

    p.add_argument("--minimum-tooth-points", type=int, default=20)
    p.add_argument("--pca-points-per-tooth", type=int, default=2000)

    p.add_argument(
        "--occlusal-fraction",
        type=float,
        default=0.40,
        help="每颗牙朝对颌方向保留 40% 作为 occlusal-band",
    )
    p.add_argument("--slice-step-mm", type=float, default=1.0)
    p.add_argument("--slice-half-width-mm", type=float, default=0.8)
    p.add_argument("--outer-quantile", type=float, default=0.95)
    p.add_argument("--minimum-slice-points", type=int, default=20)
    p.add_argument("--range-quantile", type=float, default=0.02)

    # v2 局部单牙窗口参数。默认比整侧切片更细。
    p.add_argument(
        "--local-slice-step-mm",
        type=float,
        default=0.5,
        help="单牙局部 profile 的切片步长，默认 0.5 mm",
    )
    p.add_argument(
        "--local-slice-half-width-mm",
        type=float,
        default=0.6,
        help="单牙局部 slice 半宽，默认 0.6 mm",
    )
    p.add_argument(
        "--local-padding-mm",
        type=float,
        default=0.75,
        help="目标上颌牙前后窗口两端额外扩张，默认 0.75 mm",
    )
    p.add_argument(
        "--local-minimum-slice-points",
        type=int,
        default=12,
        help="局部 slice 上下颌各至少点数，默认 12",
    )
    p.add_argument(
        "--local-range-quantile",
        type=float,
        default=0.02,
        help="v2 shared-axis 单牙自身前后范围分位数，默认 0.02",
    )

    # v3 tooth-specific local frame 参数。
    p.add_argument(
        "--tooth-frame-slice-step-mm",
        type=float,
        default=0.5,
        help="v3 单牙 tooth-frame profile 步长，默认 0.5 mm",
    )
    p.add_argument(
        "--tooth-frame-slice-half-width-mm",
        type=float,
        default=0.6,
        help="v3 单牙 tooth-frame slice 半宽，默认 0.6 mm",
    )
    p.add_argument(
        "--tooth-frame-padding-mm",
        type=float,
        default=0.50,
        help="14-16/24-26 单牙窗口两端 padding，默认 0.50 mm",
    )
    p.add_argument(
        "--tooth-frame-endpoint-padding-mm",
        type=float,
        default=0.0,
        help="17/27 末端第二磨牙 padding，默认 0，抑制 endpoint artifact",
    )
    p.add_argument(
        "--tooth-frame-minimum-slice-points",
        type=int,
        default=12,
        help="v3 普通单牙 slice 上下颌各至少点数，默认 12",
    )
    p.add_argument(
        "--tooth-frame-endpoint-minimum-slice-points",
        type=int,
        default=16,
        help="v3 17/27 slice 上下颌各至少点数，默认 16",
    )
    p.add_argument(
        "--tooth-frame-range-quantile",
        type=float,
        default=0.03,
        help="v3 普通单牙范围 trimming quantile，默认 0.03",
    )
    p.add_argument(
        "--tooth-frame-endpoint-range-quantile",
        type=float,
        default=0.05,
        help="v3 17/27 范围 trimming quantile，默认 0.05",
    )

    p.add_argument(
        "--save-profiles",
        action="store_true",
        help="保存逐切片 debug JSON；单病例测试建议开启",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.label_root = args.label_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.profile_dir = args.profile_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_profiles:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    if args.patient_id:
        patient_ids = list(dict.fromkeys(args.patient_id))
    else:
        patient_ids = sorted(
            p.name for p in args.raw_root.glob("F*") if p.is_dir()
        )

    print("=" * 88)
    print("Crossbite geometry extraction v3-tooth-specific-frame")
    print("=" * 88)
    print("patients   :", len(patient_ids))
    print("raw root   :", args.raw_root)
    print("label root :", args.label_root)
    print(
        "primary    : occlusal-band; "
        "full-crown 同时保留作对照"
    )
    print("=" * 88)

    rows = []
    failures = []

    for i, patient_id in enumerate(patient_ids, 1):
        try:
            row, debug = analyze_patient(patient_id, args)
            status = "OK" if row["primary_geometry_valid"] else "REVIEW"

            if args.save_profiles:
                path = args.profile_dir / f"{patient_id}_crossbite_profile_v3.json"
                path.write_text(
                    json.dumps(debug, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            row = {
                "patient_id": patient_id,
                "geometry_ok": False,
                "primary_geometry_valid": False,
                "error": repr(e),
            }
            failures.append({
                "patient_id": patient_id,
                "error": repr(e),
            })
            status = "FAIL"

        rows.append(row)

        if (
            len(patient_ids) <= 20
            or i % 10 == 0
            or i == len(patient_ids)
            or status != "OK"
        ):
            print(f"[{i:4d}/{len(patient_ids):4d}] {patient_id}: {status}")

    features_path = args.output_dir / "crossbite_geometry_features_v3.csv"
    key_path = args.output_dir / "crossbite_geometry_key_v3.csv"
    summary_path = args.output_dir / "crossbite_geometry_summary_v3.json"
    failures_path = args.output_dir / "crossbite_geometry_failures_v3.json"

    write_csv(features_path, rows)
    write_csv(key_path, [compact_key_row(r) for r in rows])

    full_valid = sum(
        bool(r.get("left_full_valid"))
        and bool(r.get("right_full_valid"))
        for r in rows
    )
    occ_valid = sum(
        bool(r.get("left_occlusal_valid"))
        and bool(r.get("right_occlusal_valid"))
        for r in rows
    )

    local_valid = sum(
        bool(r.get("local_geometry_valid")) for r in rows
    )
    tooth_frame_valid = sum(
        bool(r.get("tooth_frame_geometry_valid")) for r in rows
    )

    summary = {
        "version": "crossbite-geometry-v3-tooth-specific-frame",
        "patients_requested": len(patient_ids),
        "base_geometry_ok": sum(bool(r.get("geometry_ok")) for r in rows),
        "full_bilateral_valid": full_valid,
        "occlusal_bilateral_valid": occ_valid,
        "local_geometry_valid": local_valid,
        "tooth_frame_geometry_valid": tooth_frame_valid,
        "primary_geometry_valid": sum(
            bool(r.get("primary_geometry_valid")) for r in rows
        ),
        "parameters": {
            "minimum_tooth_points": args.minimum_tooth_points,
            "pca_points_per_tooth": args.pca_points_per_tooth,
            "occlusal_fraction": args.occlusal_fraction,
            "slice_step_mm": args.slice_step_mm,
            "slice_half_width_mm": args.slice_half_width_mm,
            "outer_quantile": args.outer_quantile,
            "minimum_slice_points": args.minimum_slice_points,
            "range_quantile": args.range_quantile,
            "local_slice_step_mm": args.local_slice_step_mm,
            "local_slice_half_width_mm": args.local_slice_half_width_mm,
            "local_padding_mm": args.local_padding_mm,
            "local_minimum_slice_points": args.local_minimum_slice_points,
            "local_range_quantile": args.local_range_quantile,
            "tooth_frame_slice_step_mm": args.tooth_frame_slice_step_mm,
            "tooth_frame_slice_half_width_mm": args.tooth_frame_slice_half_width_mm,
            "tooth_frame_padding_mm": args.tooth_frame_padding_mm,
            "tooth_frame_endpoint_padding_mm": args.tooth_frame_endpoint_padding_mm,
            "tooth_frame_minimum_slice_points": args.tooth_frame_minimum_slice_points,
            "tooth_frame_endpoint_minimum_slice_points": args.tooth_frame_endpoint_minimum_slice_points,
            "tooth_frame_range_quantile": args.tooth_frame_range_quantile,
            "tooth_frame_endpoint_range_quantile": args.tooth_frame_endpoint_range_quantile,
        },
        "failures": failures,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"patients requested      : {len(patient_ids)}")
    print(
        f"base geometry OK        : "
        f"{summary['base_geometry_ok']}/{len(patient_ids)}"
    )
    print(
        f"full bilateral valid    : "
        f"{full_valid}/{len(patient_ids)}"
    )
    print(
        f"occlusal bilateral valid: "
        f"{occ_valid}/{len(patient_ids)}"
    )
    print(
        f"v2 local geometry valid  : "
        f"{local_valid}/{len(patient_ids)}"
    )
    print(
        f"v3 tooth-frame valid     : "
        f"{tooth_frame_valid}/{len(patient_ids)}"
    )
    print()
    print("features :", features_path)
    print("key CSV  :", key_path)
    print("summary  :", summary_path)
    print("failures :", failures_path)
    if args.save_profiles:
        print("profiles :", args.profile_dir)
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
