#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_relation.py — V2 拉近版

默认只处理前 10 个患者，输出横向矩形图：
  1. pure side
  2. canine oblique 45°（更偏正视）
  3. canine oblique 55°（进一步偏正视）

尖牙斜侧位使用更靠前的焦点和更小的正交投影范围，
允许画面不包含全部牙齿，以放大尖牙/前磨牙区域。

输入：
  work/sagittal/stl/
  Fxxxx_ios_upper.stl
  Fxxxx_ios_lower.stl

输出：
  work/sagittal/render_test10_v2/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv
from PIL import Image, ImageChops, ImageDraw, ImageOps
from tqdm import tqdm

pv.OFF_SCREEN = True

DEFAULT_STL_DIR = Path("work/sagittal/stl")
DEFAULT_OUTPUT_DIR = Path(
    "work/sagittal/render_test10_v2"
)
UPPER_RE = re.compile(r"^([A-Za-z0-9_-]+)_ios_upper\.stl$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="矩形纯侧位 + 拉近版尖牙斜侧位渲染"
    )
    parser.add_argument("--stl-dir", type=Path, default=DEFAULT_STL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="默认处理前 10 个患者；0 表示全部",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="仅处理指定患者，例如 --patients case case002",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument(
        "--oblique-angles",
        type=float,
        nargs="+",
        default=[45.0, 55.0],
        help="从纯侧位向正视方向旋转的角度，默认 45° 55°",
    )
    parser.add_argument(
        "--canine-focus-ratio",
        type=float,
        default=0.70,
        help="后牙到前牙方向的尖牙焦点比例，默认 0.76",
    )
    parser.add_argument(
        "--pure-zoom",
        type=float,
        default=1.08,
        help="纯侧位放大倍率，默认 1.08",
    )
    parser.add_argument(
        "--oblique-zoom",
        type=float,
        default=1.40,
        help="尖牙斜侧位放大倍率，默认 1.90；允许裁掉部分后牙",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=260.0,
    ) 
    parser.add_argument(
        "--window-width",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1200,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError("方向向量长度接近 0")
    return vector / norm


def robust_width(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 20:
        return float("inf")
    return float(np.quantile(values, 0.90) - np.quantile(values, 0.10))


def load_mesh(path: Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points < 100 or mesh.n_cells < 1:
        raise ValueError(f"网格无效：{path}")
    return mesh.copy(deep=True)


def joint_align(
    upper_mesh: pv.PolyData,
    lower_mesh: pv.PolyData,
    max_pca_points: int = 200000,
) -> tuple[pv.PolyData, pv.PolyData, dict[str, float]]:
    """
    upper/lower 使用完全相同的刚性变换：
      X = 左右
      Y = 上下（上颌为正）
      Z = 前后（前牙为正）
    因此不会破坏原始上下颌拟合关系。
    """
    upper = np.asarray(upper_mesh.points, dtype=np.float64).copy()
    lower = np.asarray(lower_mesh.points, dtype=np.float64).copy()
    combined = np.vstack([upper, lower])

    center = combined.mean(axis=0)
    upper -= center
    lower -= center
    combined -= center

    if len(combined) > max_pca_points:
        rng = np.random.default_rng(2026)
        indices = rng.choice(
            len(combined),
            size=max_pca_points,
            replace=False,
        )
        sample = combined[indices]
    else:
        sample = combined

    covariance = np.cov(sample, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    # upper/lower 分离最明显的 PCA 轴作为上下轴。
    vertical_scores: list[float] = []
    for index in range(3):
        axis = axes[:, index]
        separation = abs(
            float((upper @ axis).mean() - (lower @ axis).mean())
        )
        spread = float(np.std(combined @ axis)) + 1e-8
        vertical_scores.append(separation / spread)

    vertical_index = int(np.argmax(vertical_scores))
    y_axis = axes[:, vertical_index]

    if float((upper @ y_axis).mean()) < float((lower @ y_axis).mean()):
        y_axis = -y_axis

    remaining = [index for index in range(3) if index != vertical_index]
    axis_a = axes[:, remaining[0]]
    axis_b = axes[:, remaining[1]]

    if float(np.ptp(combined @ axis_a)) >= float(np.ptp(combined @ axis_b)):
        x_candidate = axis_a
        z_axis = axis_b
    else:
        x_candidate = axis_b
        z_axis = axis_a

    # 较窄的一端通常是前牙端，将其设为 +Z。
    z_projection = combined @ z_axis
    x_projection = combined @ x_candidate
    low = float(np.quantile(z_projection, 0.18))
    high = float(np.quantile(z_projection, 0.82))
    low_width = robust_width(x_projection[z_projection <= low])
    high_width = robust_width(x_projection[z_projection >= high])

    if low_width < high_width:
        z_axis = -z_axis

    y_axis = normalize(y_axis)
    z_axis = normalize(z_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    upper = upper @ rotation
    lower = lower @ rotation

    joined = np.vstack([upper, lower])
    bbox_center = (joined.min(axis=0) + joined.max(axis=0)) / 2.0
    upper -= bbox_center
    lower -= bbox_center

    joined = np.vstack([upper, lower])
    max_dimension = float(np.ptp(joined, axis=0).max())
    if max_dimension <= 1e-8:
        raise ValueError("联合网格尺寸无效")

    scale = 100.0 / max_dimension
    upper *= scale
    lower *= scale

    upper_mesh.points = upper.astype(np.float32)
    lower_mesh.points = lower.astype(np.float32)

    joined = np.vstack([upper, lower])
    minimum = joined.min(axis=0)
    maximum = joined.max(axis=0)
    extent = maximum - minimum

    info = {
        "scale": scale,
        "x_min": float(minimum[0]),
        "x_max": float(maximum[0]),
        "y_min": float(minimum[1]),
        "y_max": float(maximum[1]),
        "z_min": float(minimum[2]),
        "z_max": float(maximum[2]),
        "x_extent": float(extent[0]),
        "y_extent": float(extent[1]),
        "z_extent": float(extent[2]),
        "vertical_score": float(vertical_scores[vertical_index]),
        "end_width_low": float(low_width),
        "end_width_high": float(high_width),
    }
    return upper_mesh, lower_mesh, info


def foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    difference = ImageChops.difference(rgb, background).convert("L")
    mask = difference.point(lambda value: 255 if value > 5 else 0)
    return mask.getbbox()


def crop_to_rectangle(
    image: Image.Image,
    output_width: int,
    output_height: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> Image.Image:
    """裁去白边并输出横向矩形，不拉伸几何。"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    bbox = foreground_bbox(image)

    if bbox is None:
        return Image.new(
            "RGB",
            (output_width, output_height),
            (255, 255, 255),
        )

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    pad_x = max(4, int(round(width * pad_x_ratio)))
    pad_y = max(4, int(round(height * pad_y_ratio)))

    left = max(0, left - pad_x)
    right = min(image.width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(image.height, bottom + pad_y)

    cropped = image.crop((left, top, right, bottom))

    scale = min(
        output_width / max(cropped.width, 1),
        output_height / max(cropped.height, 1),
    )
    new_width = max(1, int(round(cropped.width * scale)))
    new_height = max(1, int(round(cropped.height * scale)))

    resized = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (output_width, output_height),
        (255, 255, 255),
    )
    canvas.paste(
        resized,
        (
            (output_width - new_width) // 2,
            (output_height - new_height) // 2,
        ),
    )
    return canvas


def calculate_pure_parallel_scale(
    info: dict[str, float],
    width: int,
    height: int,
    zoom: float,
) -> float:
    """
    正交投影中 parallel_scale 表示画面半高度。
    同时考虑上下高度和横向前后长度，保证 pure side 基本完整。
    """
    aspect = width / height
    required_half_height = max(
        info["y_extent"] / 2.0,
        info["z_extent"] / (2.0 * aspect),
    )
    return required_half_height * 1.08 / max(zoom, 1e-6)


def render_view(
    upper_mesh: pv.PolyData,
    lower_mesh: pv.PolyData,
    camera_position: list[float],
    focal_point: list[float],
    parallel_scale: float,
    output_width: int,
    output_height: int,
    window_width: int,
    window_height: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> Image.Image:
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(window_width, window_height),
    )
    plotter.set_background("white")

    plotter.add_mesh(
        upper_mesh,
        color=(0.72, 0.72, 0.72),
        smooth_shading=True,
        show_edges=False,
        ambient=0.22,
        diffuse=0.78,
        specular=0.08,
        specular_power=12,
    )
    plotter.add_mesh(
        lower_mesh,
        color=(0.84, 0.84, 0.84),
        smooth_shading=True,
        show_edges=False,
        ambient=0.22,
        diffuse=0.78,
        specular=0.08,
        specular_power=12,
    )

    plotter.camera.position = camera_position
    plotter.camera.focal_point = focal_point
    plotter.camera.up = [0.0, 1.0, 0.0]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = float(parallel_scale)

    array = plotter.screenshot(return_img=True)
    plotter.close()

    return crop_to_rectangle(
        Image.fromarray(array),
        output_width=output_width,
        output_height=output_height,
        pad_x_ratio=pad_x_ratio,
        pad_y_ratio=pad_y_ratio,
    )


def build_camera(
    side: str,
    angle_degrees: float,
    distance: float,
    focal_point: list[float],
) -> list[float]:
    """
    angle=0° 为纯侧位；角度增大时向正视方向（+Z）旋转。
    """
    angle = math.radians(angle_degrees)
    lateral = distance * math.cos(angle)
    frontal = distance * math.sin(angle)

    x = lateral if side == "left" else -lateral
    return [
        float(focal_point[0] + x),
        float(focal_point[1]),
        float(focal_point[2] + frontal),
    ]


def make_compare(
    images: list[tuple[str, Image.Image]],
) -> Image.Image:
    header_height = 30
    total_width = sum(image.width for _title, image in images)
    content_height = max(image.height for _title, image in images)

    canvas = Image.new(
        "RGB",
        (total_width, content_height + header_height),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    offset_x = 0
    for title, image in images:
        draw.text(
            (offset_x + 8, 8),
            title,
            fill=(0, 0, 0),
        )
        canvas.paste(
            image,
            (offset_x, header_height),
        )
        offset_x += image.width

        if offset_x < total_width:
            draw.line(
                (
                    offset_x,
                    0,
                    offset_x,
                    content_height + header_height,
                ),
                fill=(190, 190, 190),
                width=2,
            )

    return canvas


def render_patient(
    patient_id: str,
    upper_path: Path,
    lower_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    upper = load_mesh(upper_path)
    lower = load_mesh(lower_path)
    upper, lower, info = joint_align(upper, lower)

    y_center = (info["y_min"] + info["y_max"]) / 2.0
    z_center = (info["z_min"] + info["z_max"]) / 2.0
    canine_z = info["z_min"] + args.canine_focus_ratio * info["z_extent"]

    pure_focus = [0.0, y_center, z_center]
    canine_focus = [0.0, y_center, canine_z]

    pure_scale = calculate_pure_parallel_scale(
        info,
        width=args.width,
        height=args.height,
        zoom=args.pure_zoom,
    )
    oblique_scale = pure_scale / max(args.oblique_zoom, 1e-6)

    result: dict[str, Any] = {
        "patient_id": patient_id,
        "status": "success",
        "message": "",
        **info,
    }

    for side in ("left", "right"):
        compare_images: list[tuple[str, Image.Image]] = []

        pure_path = output_dir / f"{patient_id}_{side}_pure_side.png"
        pure_camera = build_camera(
            side=side,
            angle_degrees=0.0,
            distance=args.camera_distance,
            focal_point=pure_focus,
        )
        pure_image = render_view(
            upper,
            lower,
            camera_position=pure_camera,
            focal_point=pure_focus,
            parallel_scale=pure_scale,
            output_width=args.width,
            output_height=args.height,
            window_width=args.window_width,
            window_height=args.window_height,
            pad_x_ratio=0.025,
            pad_y_ratio=0.035,
        )
        if args.overwrite or not pure_path.exists():
            pure_image.save(pure_path)
        compare_images.append(("pure side", pure_image))
        result[f"{side}_pure_side_path"] = str(pure_path)

        for angle in args.oblique_angles:
            rounded_angle = int(round(angle))
            output_path = output_dir / (
                f"{patient_id}_{side}_canine_oblique_{rounded_angle}.png"
            )

            camera = build_camera(
                side=side,
                angle_degrees=float(angle),
                distance=args.camera_distance,
                focal_point=canine_focus,
            )
            image = render_view(
                upper,
                lower,
                camera_position=camera,
                focal_point=canine_focus,
                parallel_scale=oblique_scale,
                output_width=args.width,
                output_height=args.height,
                window_width=args.window_width,
                window_height=args.window_height,
                pad_x_ratio=0.015,
                pad_y_ratio=0.025,
            )
            if args.overwrite or not output_path.exists():
                image.save(output_path)

            compare_images.append((f"oblique {rounded_angle}°", image))
            result[f"{side}_oblique_{rounded_angle}_path"] = str(output_path)

        compare_path = output_dir / f"{patient_id}_{side}_compare.png"
        compare_image = make_compare(compare_images)
        if args.overwrite or not compare_path.exists():
            compare_image.save(compare_path)
        result[f"{side}_compare_path"] = str(compare_path)

    return result


def discover_patients(
    stl_dir: Path,
    requested: list[str] | None,
    limit: int,
) -> list[tuple[str, Path, Path]]:
    if requested:
        patient_ids = sorted(
            set(requested),
            key=lambda value: (0, int(value[1:])) if re.fullmatch(r"F\d+", value, re.IGNORECASE) else (1, value),
        )
    else:
        patient_ids = []
        for path in stl_dir.glob("*_ios_upper.stl"):
            match = UPPER_RE.fullmatch(path.name)
            if match:
                patient_ids.append(match.group(1))
        patient_ids = sorted(
            set(patient_ids),
            key=lambda value: (0, int(value[1:])) if re.fullmatch(r"F\d+", value, re.IGNORECASE) else (1, value),
        )

    pairs: list[tuple[str, Path, Path]] = []
    for patient_id in patient_ids:
        upper = stl_dir / f"{patient_id}_ios_upper.stl"
        lower = stl_dir / f"{patient_id}_ios_lower.stl"
        if upper.exists() and lower.exists():
            pairs.append((patient_id, upper, lower))

    if limit > 0:
        pairs = pairs[:limit]
    return pairs


def save_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "render_test10_v2_summary.csv"
    fieldnames = sorted(
        {key for row in rows for key in row.keys()}
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def main() -> int:
    args = parse_args()
    args.stl_dir = args.stl_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.stl_dir.exists():
        raise FileNotFoundError(
            f"STL 目录不存在：{args.stl_dir}"
        )

    if not 0.0 <= args.canine_focus_ratio <= 1.0:
        raise ValueError(
            "canine-focus-ratio 必须位于 [0, 1]"
        )

    for angle in args.oblique_angles:
        if not 0.0 < angle < 90.0:
            raise ValueError(
                "oblique angle 必须位于 (0, 90)"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patients = discover_patients(
        args.stl_dir,
        args.patients,
        args.limit,
    )

    print("=" * 72)
    print("render_relation.py — V2 拉近版")
    print("=" * 72)
    print(f"输入：{args.stl_dir}")
    print(f"输出：{args.output_dir}")
    print(f"患者数：{len(patients)}")
    print(f"矩形尺寸：{args.width} x {args.height}")
    print(f"斜侧角度：{args.oblique_angles}")
    print(f"尖牙焦点比例：{args.canine_focus_ratio}")
    print(f"纯侧位 zoom：{args.pure_zoom}")
    print(f"尖牙斜侧位 zoom：{args.oblique_zoom}")
    print("说明：尖牙斜侧位允许裁掉部分后牙")
    print("=" * 72)

    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0

    for patient_id, upper_path, lower_path in tqdm(
        patients,
        desc="Rendering",
    ):
        try:
            rows.append(
                render_patient(
                    patient_id,
                    upper_path,
                    lower_path,
                    args.output_dir,
                    args,
                )
            )
            success += 1
        except Exception as error:
            failed += 1
            rows.append(
                {
                    "patient_id": patient_id,
                    "status": "failed",
                    "message": str(error),
                }
            )
            print(f"\n❌ {patient_id}: {error}")
            traceback.print_exc()

    summary_path = save_summary(
        rows,
        args.output_dir,
    )

    print("\n" + "=" * 72)
    print(f"成功：{success}")
    print(f"失败：{failed}")
    print(f"输出：{args.output_dir}")
    print(f"汇总：{summary_path}")
    print("=" * 72)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())