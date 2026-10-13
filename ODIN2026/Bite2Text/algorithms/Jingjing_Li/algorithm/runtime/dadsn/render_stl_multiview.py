#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从原始 STL 生成不依赖牙齿分割的多视角几何张量。

每个视角包含5个通道：
    0: depth，前景归一化到0~1
    1: normal_x，范围-1~1
    2: normal_y，范围-1~1
    3: normal_z，范围-1~1
    4: silhouette mask，0或1

默认6个视角：
    z_pos, z_neg,
    x_pos_oblique, x_neg_oblique,
    y_pos_oblique, y_neg_oblique

输出：
    <case>.npz
    visual_render_manifest.csv
    visual_render_failures.csv

重要：
- 输入来自原始 work/input，不读取分割标签。
- 默认先调用 from iops import align_scan。
- 使用表面点采样和正交z-buffer，不依赖OpenGL窗口，适合服务器运行。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh


CASE_PATTERN = re.compile(
    r"^(?P<case>(?P<patient>[A-Za-z0-9][A-Za-z0-9_.-]*)_ios_(?P<arch>upper|lower))"
    r"\.(?:stl|ply|obj)$",
    re.IGNORECASE,
)

VIEW_SPECS = [
    ("z_pos", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("z_neg", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("x_pos_oblique", (0.60, 0.0, 0.80), (0.0, 1.0, 0.0)),
    ("x_neg_oblique", (-0.60, 0.0, 0.80), (0.0, 1.0, 0.0)),
    ("y_pos_oblique", (0.0, 0.60, 0.80), (1.0, 0.0, 0.0)),
    ("y_neg_oblique", (0.0, -0.60, 0.80), (1.0, 0.0, 0.0)),
]


def parse_case_list(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    cases = {
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return cases


def stable_seed(case: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{case}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(
        path,
        force="mesh",
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError("mesh scene为空")
        loaded = trimesh.util.concatenate(geometries)

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"不支持的mesh类型：{type(loaded)}")

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("mesh为空")

    return loaded


def align_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        from iops import align_scan
    except ImportError as exc:
        raise ImportError(
            "无法导入 from iops import align_scan。"
            "请在DADSN项目环境中运行，或加 --no-align。"
        ) from exc

    aligned = align_scan(mesh)

    if isinstance(aligned, trimesh.Trimesh):
        return aligned

    if aligned is None and isinstance(mesh, trimesh.Trimesh):
        return mesh

    raise TypeError(
        f"align_scan返回了不支持的类型：{type(aligned)}"
    )


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(seed)

    try:
        points, face_indices = trimesh.sample.sample_surface(
            mesh,
            count,
        )
    finally:
        np.random.set_state(state)

    normals = np.asarray(
        mesh.face_normals,
        dtype=np.float32,
    )[np.asarray(face_indices, dtype=np.int64)]

    points = np.asarray(points, dtype=np.float32)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
    )
    points = points[valid]
    normals = normals[valid]

    if len(points) < max(1000, count // 20):
        raise ValueError(
            f"有效表面采样点过少：{len(points)}"
        )

    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-8)
    return points, normals


def camera_basis(
    direction: Iterable[float],
    up_hint: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = np.asarray(direction, dtype=np.float64)
    camera /= np.linalg.norm(camera)

    up_hint_array = np.asarray(up_hint, dtype=np.float64)
    up_hint_array /= np.linalg.norm(up_hint_array)

    right = np.cross(up_hint_array, camera)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        up_hint_array = np.asarray([1.0, 0.0, 0.0])
        right = np.cross(up_hint_array, camera)
        right_norm = np.linalg.norm(right)

    right /= right_norm
    up = np.cross(camera, right)
    up /= np.linalg.norm(up)
    return right, up, camera


def select_frontmost(
    pixel_ids: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    order = np.lexsort((depth, pixel_ids))
    sorted_pixels = pixel_ids[order]
    keep = np.empty(len(order), dtype=bool)
    keep[:-1] = sorted_pixels[:-1] != sorted_pixels[1:]
    keep[-1] = True
    return order[keep]


def render_view(
    points: np.ndarray,
    normals: np.ndarray,
    direction: tuple[float, float, float],
    up_hint: tuple[float, float, float],
    image_size: int,
    half_extent: float,
    splat_radius: int,
) -> np.ndarray:
    right, up, camera = camera_basis(direction, up_hint)

    x = points @ right
    y = points @ up
    depth = points @ camera

    scale = (image_size - 1) / (2.0 * half_extent)
    u = np.rint((x + half_extent) * scale).astype(np.int64)
    v = np.rint((half_extent - y) * scale).astype(np.int64)

    normal_camera = np.column_stack(
        [
            normals @ right,
            normals @ up,
            normals @ camera,
        ]
    ).astype(np.float32)

    all_u = []
    all_v = []
    all_depth = []
    all_normals = []

    for dv in range(-splat_radius, splat_radius + 1):
        for du in range(-splat_radius, splat_radius + 1):
            shifted_u = u + du
            shifted_v = v + dv
            valid = (
                (shifted_u >= 0)
                & (shifted_u < image_size)
                & (shifted_v >= 0)
                & (shifted_v < image_size)
            )

            if not np.any(valid):
                continue

            all_u.append(shifted_u[valid])
            all_v.append(shifted_v[valid])
            all_depth.append(depth[valid])
            all_normals.append(normal_camera[valid])

    if not all_u:
        raise ValueError("当前视角没有任何可见点")

    u_all = np.concatenate(all_u)
    v_all = np.concatenate(all_v)
    depth_all = np.concatenate(all_depth).astype(np.float32)
    normals_all = np.concatenate(all_normals).astype(np.float32)

    pixel_ids = v_all * image_size + u_all
    selected = select_frontmost(pixel_ids, depth_all)

    selected_pixels = pixel_ids[selected]
    selected_depth = depth_all[selected]
    selected_normals = normals_all[selected]

    tensor = np.zeros(
        (5, image_size, image_size),
        dtype=np.float32,
    )
    rows = selected_pixels // image_size
    columns = selected_pixels % image_size

    d_min = float(np.quantile(selected_depth, 0.01))
    d_max = float(np.quantile(selected_depth, 0.99))
    denominator = max(d_max - d_min, 1e-6)
    normalized_depth = np.clip(
        (selected_depth - d_min) / denominator,
        0.0,
        1.0,
    )

    tensor[0, rows, columns] = normalized_depth
    tensor[1:4, rows, columns] = selected_normals.T
    tensor[4, rows, columns] = 1.0
    return tensor


def render_case(
    mesh_path: Path,
    image_size: int,
    sample_points: int,
    splat_radius: int,
    margin: float,
    seed: int,
    use_align: bool,
) -> tuple[np.ndarray, dict]:
    mesh = load_mesh(mesh_path)

    if use_align:
        mesh = align_mesh(mesh)

    points, normals = sample_surface(
        mesh,
        sample_points,
        seed,
    )

    center = np.median(points, axis=0)
    points = points - center

    radius = np.linalg.norm(points[:, :2], axis=1)
    spatial_radius = np.linalg.norm(points, axis=1)
    half_extent = max(
        float(np.quantile(radius, 0.995)),
        float(np.quantile(spatial_radius, 0.995)) * 0.75,
        1.0,
    )
    half_extent *= margin

    views = []
    for _, direction, up_hint in VIEW_SPECS:
        views.append(
            render_view(
                points=points,
                normals=normals,
                direction=direction,
                up_hint=up_hint,
                image_size=image_size,
                half_extent=half_extent,
                splat_radius=splat_radius,
            )
        )

    tensor = np.stack(views, axis=0).astype(np.float16)
    metadata = {
        "image_size": int(image_size),
        "view_names": [spec[0] for spec in VIEW_SPECS],
        "sample_points": int(len(points)),
        "half_extent_mm": float(half_extent),
        "aligned": bool(use_align),
        "source_mesh": str(mesh_path),
    }
    return tensor, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("work/input"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/visual_multiview"),
    )
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--sample-points", type=int, default=120000)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--margin", type=float, default=1.08)
    parser.add_argument("--base-seed", type=int, default=20260728)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_cases = parse_case_list(args.case_list)

    mesh_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and CASE_PATTERN.fullmatch(path.name)
    ]

    if requested_cases is not None:
        mesh_paths = [
            path
            for path in mesh_paths
            if CASE_PATTERN.fullmatch(path.name).group("case")
            in requested_cases
        ]
    if not mesh_paths:
        raise FileNotFoundError(
            f"No meshes matching '<case>_ios_<upper|lower>.<stl|ply|obj>' in {input_dir}"
        )

    successes = []
    failures = []

    for index, mesh_path in enumerate(mesh_paths, start=1):
        match = CASE_PATTERN.fullmatch(mesh_path.name)
        assert match is not None

        case = match.group("case")
        patient = match.group("patient")
        arch = match.group("arch").lower()
        destination = output_dir / f"{case}.npz"

        print(f"[{index}/{len(mesh_paths)}] {case}")

        if destination.is_file() and not args.overwrite:
            successes.append(
                {
                    "case": case,
                    "patient_id": patient,
                    "arch": arch,
                    "output": str(destination),
                    "status": "existing",
                }
            )
            continue

        try:
            tensor, metadata = render_case(
                mesh_path=mesh_path,
                image_size=args.image_size,
                sample_points=args.sample_points,
                splat_radius=args.splat_radius,
                margin=args.margin,
                seed=stable_seed(case, args.base_seed),
                use_align=not args.no_align,
            )

            np.savez_compressed(
                destination,
                tensor=tensor,
                view_names=np.asarray(
                    metadata["view_names"],
                    dtype="U32",
                ),
                image_size=np.int32(metadata["image_size"]),
                sample_points=np.int32(metadata["sample_points"]),
                half_extent_mm=np.float32(
                    metadata["half_extent_mm"]
                ),
                aligned=np.int8(metadata["aligned"]),
                source_mesh=np.asarray(
                    metadata["source_mesh"],
                    dtype="U512",
                ),
            )

            successes.append(
                {
                    "case": case,
                    "patient_id": patient,
                    "arch": arch,
                    "output": str(destination),
                    "status": "rendered",
                    "tensor_shape": repr(tuple(tensor.shape)),
                    "half_extent_mm": metadata["half_extent_mm"],
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "case": case,
                    "patient_id": patient,
                    "arch": arch,
                    "mesh": str(mesh_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"  FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

    success_path = output_dir / "visual_render_manifest.csv"
    failure_path = output_dir / "visual_render_failures.csv"

    success_columns = sorted(
        {
            key
            for row in successes
            for key in row
        }
    )
    with success_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=success_columns,
        )
        writer.writeheader()
        writer.writerows(successes)

    failure_columns = [
        "case",
        "patient_id",
        "arch",
        "mesh",
        "error_type",
        "error",
    ]
    with failure_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=failure_columns,
        )
        writer.writeheader()
        writer.writerows(failures)

    print("=" * 72)
    print("Visual multiview rendering complete")
    print("=" * 72)
    print(f"Success: {len(successes)}")
    print(f"Failure: {len(failures)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
