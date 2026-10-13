#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量生成咬合面 label mask 图。

输入目录示例：
    work/output/
        case_ios_upper.npy
        case_ios_upper_pass2_full_raw.ply
        case_ios_lower.npy
        case_ios_lower_pass2_full_raw.ply

输出目录示例：
    work/rendered_ios_occlusal_mask/
        case_ios_upper_label.png
        case_ios_lower_label.png
        render_mask_summary.csv

输出 PNG 为单通道 uint8：
    0  = 背景、牙龈、腭部、底座
    1~16 = 牙齿类别

脚本使用与之前 RGB 俯视图相同的规则：
    1. 牙齿标签点 PCA 对齐；
    2. 牙号重心确定左右、前后；
    3. label 0 区域辅助判断 Z 方向；
    4. upper 从 -Z 拍摄，lower 从 +Z 拍摄；
    5. 正交投影；
    6. 按整个 mesh 轮廓裁剪并缩放到固定尺寸。

为了避免光照、阴影、抗锯齿造成的色差：
    - 使用无光照类别渲染；
    - 每个三角面使用固定类别颜色；
    - 关闭 multisampling；
    - 输出前映射回整数 0~16；
    - 缩放使用最近邻插值。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import traceback
from pathlib import Path

# 必须在 import pyvista 前设置。
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv
from PIL import Image
from tqdm import tqdm


FINAL_NPY_PATTERN = re.compile(
    r"^(?P<case>(?P<patient>[A-Za-z0-9_-]+)_ios_(?P<arch>upper|lower))\.npy$",
    re.IGNORECASE,
)


# 17 个彼此区分明显的固定颜色。
# label 0 与屏幕背景均为黑色，最终都会映射为像素值 0。
MASK_PALETTE = np.asarray(
    [
        [0, 0, 0],          # 0
        [255, 32, 32],      # 1
        [32, 255, 32],      # 2
        [32, 32, 255],      # 3
        [255, 255, 32],     # 4
        [255, 32, 255],     # 5
        [32, 255, 255],     # 6
        [255, 128, 32],     # 7
        [128, 32, 255],     # 8
        [32, 128, 255],     # 9
        [128, 255, 32],     # 10
        [255, 32, 128],     # 11
        [32, 255, 128],     # 12
        [255, 128, 192],    # 13
        [128, 255, 192],    # 14
        [192, 128, 255],    # 15
        [255, 192, 128],    # 16
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量生成 0~16 单通道咬合面 label mask"
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("work/output"),
        help="预测结果目录，默认：work/output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/rendered_ios_occlusal_mask"),
        help="mask 输出目录",
    )
    parser.add_argument(
        "--ply-suffix",
        type=str,
        default="pass2_full_raw",
        help=(
            "输入 PLY 后缀，默认读取 "
            "<case>_pass2_full_raw.ply"
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="最终 mask 尺寸，默认 1024",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=1400,
        help="内部渲染尺寸，默认 1400",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="按 mesh 轮廓裁剪时保留的边距",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 mask",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 个牙弓；0 表示全部",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="额外保存彩色预览图，便于人工检查",
    )
    return parser.parse_args()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))

    if length < 1e-10:
        raise ValueError("方向向量长度接近 0")

    return vector / length


def align_mesh_pca_label(
    mesh: pv.PolyData,
    labels: np.ndarray,
    arch: str,
) -> pv.PolyData:
    """
    与先前 RGB 脚本保持一致的 PCA + 标签定向。
    """
    points = np.asarray(
        mesh.points,
        dtype=np.float64,
    ).copy()

    labels = np.asarray(
        labels,
        dtype=np.int32,
    ).reshape(-1)

    if len(points) != len(labels):
        raise ValueError(
            f"PLY 顶点数与 NPY 标签数不一致："
            f"{len(points)} vs {len(labels)}"
        )

    tooth_mask = labels > 0

    if int(tooth_mask.sum()) < 100:
        raise ValueError(
            f"非零牙齿标签点过少：{int(tooth_mask.sum())}"
        )

    tooth_points = points[tooth_mask]
    center = tooth_points.mean(axis=0)
    centered = tooth_points - center

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    if (
        covariance.shape != (3, 3)
        or np.any(~np.isfinite(covariance))
    ):
        raise ValueError("PCA 协方差矩阵无效")

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    x_axis = normalize_vector(
        eigenvectors[:, 0]
    )
    y_axis = normalize_vector(
        eigenvectors[:, 1]
    )

    tooth_labels = labels[tooth_mask]

    # 左右定向。
    right_labels = [1, 2, 9, 10]
    left_labels = [6, 7, 14, 15]

    right_mask = np.isin(
        tooth_labels,
        right_labels,
    )
    left_mask = np.isin(
        tooth_labels,
        left_labels,
    )

    if right_mask.any() and left_mask.any():
        right_center = tooth_points[
            right_mask
        ].mean(axis=0)
        left_center = tooth_points[
            left_mask
        ].mean(axis=0)

        if np.dot(
            right_center - left_center,
            x_axis,
        ) < 0:
            x_axis = -x_axis

    # 前后定向。
    front_labels = [3, 4, 11, 12]
    back_labels = [6, 7, 14, 15]

    front_mask = np.isin(
        tooth_labels,
        front_labels,
    )
    back_mask = np.isin(
        tooth_labels,
        back_labels,
    )

    if front_mask.any() and back_mask.any():
        front_center = tooth_points[
            front_mask
        ].mean(axis=0)
        back_center = tooth_points[
            back_mask
        ].mean(axis=0)

        if np.dot(
            front_center - back_center,
            y_axis,
        ) > 0:
            y_axis = -y_axis

    # 保证正交和右手系。
    z_axis = normalize_vector(
        np.cross(x_axis, y_axis)
    )
    y_axis = normalize_vector(
        np.cross(z_axis, x_axis)
    )

    rotation_matrix = np.column_stack(
        [x_axis, y_axis, z_axis]
    )

    rotated = (
        rotation_matrix.T
        @ (points - center).T
    ).T

    # 使用 label 0 区域辅助判断 Z。
    background_mask = labels == 0

    if int(background_mask.sum()) > 100:
        background_z = float(
            np.median(
                rotated[background_mask, 2]
            )
        )

        if arch == "upper" and background_z < 0:
            rotated[:, 2] *= -1.0
        elif arch == "lower" and background_z > 0:
            rotated[:, 2] *= -1.0
    else:
        if arch == "upper":
            rotated[:, 2] *= -1.0

    aligned = mesh.copy(deep=True)
    aligned.points = rotated.astype(
        np.float32
    )

    return aligned


def vertex_labels_to_cell_labels(
    mesh: pv.PolyData,
    vertex_labels: np.ndarray,
) -> tuple[pv.PolyData, np.ndarray]:
    """
    将逐顶点标签转换为逐三角面标签。

    这样每个三角面只使用一个离散类别，
    避免在三角形内部对 0~16 标签做连续插值。
    """
    triangle_mesh = mesh.triangulate()

    faces = np.asarray(
        triangle_mesh.faces,
        dtype=np.int64,
    ).reshape(-1, 4)

    if not np.all(faces[:, 0] == 3):
        raise ValueError("triangulate 后仍存在非三角形单元")

    triangles = faces[:, 1:4]
    vertex_labels = np.asarray(
        vertex_labels,
        dtype=np.uint8,
    )

    tri_labels = vertex_labels[
        triangles
    ]

    a = tri_labels[:, 0]
    b = tri_labels[:, 1]
    c = tri_labels[:, 2]

    # 三个顶点多数投票。
    cell_labels = np.where(
        a == b,
        a,
        np.where(
            a == c,
            a,
            np.where(
                b == c,
                b,
                a,
            ),
        ),
    ).astype(np.uint8)

    return triangle_mesh, cell_labels


def configure_camera(
    plotter: pv.Plotter,
    arch: str,
) -> None:
    if arch == "upper":
        camera_position = [
            0.0,
            0.0,
            -250.0,
        ]
    else:
        camera_position = [
            0.0,
            0.0,
            250.0,
        ]

    plotter.enable_parallel_projection()
    plotter.camera.position = camera_position
    plotter.camera.focal_point = [
        0.0,
        0.0,
        0.0,
    ]
    plotter.camera.up = [
        0.0,
        1.0,
        0.0,
    ]
    plotter.camera.parallel_scale = 60.0


def render_label_rgb(
    mesh: pv.PolyData,
    cell_labels: np.ndarray,
    arch: str,
    render_size: int,
) -> np.ndarray:
    """
    用固定类别颜色渲染。
    背景和 label 0 都为黑色。
    """
    cell_rgb = MASK_PALETTE[
        cell_labels
    ]

    render_mesh = mesh.copy(deep=True)
    render_mesh.cell_data[
        "mask_rgb"
    ] = cell_rgb

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(
            render_size,
            render_size,
        ),
    )

    # 兼容不同 PyVista 版本：
    # 新版可关闭抗锯齿，旧版没有该方法时直接跳过。
    try:
        if hasattr(plotter, "disable_anti_aliasing"):
            plotter.disable_anti_aliasing()
    except Exception:
        pass

    try:
        plotter.set_background(
            "black"
        )
        plotter.add_mesh(
            render_mesh,
            scalars="mask_rgb",
            preference="cell",
            rgb=True,
            lighting=False,
            show_edges=False,
            smooth_shading=False,
            show_scalar_bar=False,
        )
        configure_camera(
            plotter,
            arch,
        )

        image = plotter.screenshot(
            return_img=True,
            transparent_background=False,
        )
    finally:
        plotter.close()

    return np.asarray(
        image,
        dtype=np.uint8,
    )[:, :, :3]


def render_silhouette(
    mesh: pv.PolyData,
    arch: str,
    render_size: int,
) -> np.ndarray:
    """
    渲染整个 mesh 的白色轮廓，用于计算与 RGB 图一致的裁剪框。
    """
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(
            render_size,
            render_size,
        ),
    )

    # 兼容不同 PyVista 版本：
    # 新版可关闭抗锯齿，旧版没有该方法时直接跳过。
    try:
        if hasattr(plotter, "disable_anti_aliasing"):
            plotter.disable_anti_aliasing()
    except Exception:
        pass

    try:
        plotter.set_background(
            "black"
        )
        plotter.add_mesh(
            mesh,
            color="white",
            lighting=False,
            show_edges=False,
            smooth_shading=False,
        )
        configure_camera(
            plotter,
            arch,
        )

        image = plotter.screenshot(
            return_img=True,
            transparent_background=False,
        )
    finally:
        plotter.close()

    rgb = np.asarray(
        image,
        dtype=np.uint8,
    )[:, :, :3]

    return (
        np.max(rgb, axis=2) > 127
    )


def rgb_render_to_label_mask(
    rendered_rgb: np.ndarray,
) -> np.ndarray:
    """
    将渲染颜色映射回整数 0~16。

    使用最近颜色匹配，可处理边缘少量抗锯齿色差。
    """
    image = np.asarray(
        rendered_rgb,
        dtype=np.int16,
    )

    palette = MASK_PALETTE.astype(
        np.int16
    )

    # [H, W, 1, 3] - [1, 1, 17, 3]
    differences = (
        image[:, :, None, :]
        - palette[None, None, :, :]
    )

    distances = np.sum(
        differences * differences,
        axis=3,
    )

    labels = np.argmin(
        distances,
        axis=2,
    ).astype(np.uint8)

    return labels


def silhouette_bbox(
    silhouette: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(
        silhouette
    )

    if len(xs) == 0:
        raise ValueError("mesh 轮廓渲染为空")

    left = max(
        0,
        int(xs.min()) - padding,
    )
    right = min(
        silhouette.shape[1],
        int(xs.max()) + 1 + padding,
    )
    top = max(
        0,
        int(ys.min()) - padding,
    )
    bottom = min(
        silhouette.shape[0],
        int(ys.max()) + 1 + padding,
    )

    return left, top, right, bottom


def crop_and_resize_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: int,
) -> np.ndarray:
    """
    按整个 mesh 轮廓裁剪。
    使用最近邻缩放，保证像素仍然是 0~16。
    """
    left, top, right, bottom = bbox

    cropped = mask[
        top:bottom,
        left:right,
    ]

    if cropped.size == 0:
        raise ValueError("mask 裁剪结果为空")

    height, width = cropped.shape
    scale = target_size / max(
        width,
        height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )
    new_height = max(
        1,
        int(round(height * scale)),
    )

    cropped_image = Image.fromarray(
        cropped,
        mode="L",
    )

    resized = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST,
    )

    final = Image.new(
        "L",
        (target_size, target_size),
        color=0,
    )

    offset_x = (
        target_size - new_width
    ) // 2
    offset_y = (
        target_size - new_height
    ) // 2

    final.paste(
        resized,
        (
            offset_x,
            offset_y,
        ),
    )

    output = np.asarray(
        final,
        dtype=np.uint8,
    )

    # 最终安全检查。
    invalid = (
        (output < 0)
        | (output > 16)
    )

    if invalid.any():
        raise ValueError(
            "输出 mask 出现 0~16 之外的值"
        )

    return output


def save_color_preview(
    label_mask: np.ndarray,
    path: Path,
) -> None:
    preview = MASK_PALETTE[
        label_mask
    ]
    Image.fromarray(
        preview,
        mode="RGB",
    ).save(
        path,
        format="PNG",
        optimize=True,
    )


def render_case(
    ply_path: Path,
    npy_path: Path,
    output_path: Path,
    arch: str,
    target_size: int,
    render_size: int,
    padding: int,
    preview_path: Path | None,
) -> dict[str, object]:
    mesh = pv.read(
        str(ply_path)
    )

    labels = np.load(
        npy_path,
        allow_pickle=False,
    ).astype(
        np.uint8
    ).reshape(-1)

    if mesh.n_points != len(labels):
        raise ValueError(
            f"{ply_path.name}: PLY 顶点数 "
            f"{mesh.n_points} != NPY 标签数 "
            f"{len(labels)}"
        )

    if labels.min() < 0 or labels.max() > 16:
        raise ValueError(
            f"NPY 标签范围异常："
            f"{labels.min()}~{labels.max()}"
        )

    aligned_mesh = align_mesh_pca_label(
        mesh,
        labels,
        arch,
    )

    # 与 RGB 脚本相同，统一最大尺寸到 100。
    bounds = aligned_mesh.bounds
    dimensions = np.asarray(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ],
        dtype=np.float64,
    )

    max_dimension = float(
        dimensions.max()
    )

    if (
        not np.isfinite(max_dimension)
        or max_dimension <= 0
    ):
        raise ValueError("网格尺寸无效")

    aligned_mesh.points *= (
        100.0 / max_dimension
    )

    triangle_mesh, cell_labels = (
        vertex_labels_to_cell_labels(
            aligned_mesh,
            labels,
        )
    )

    rendered_rgb = render_label_rgb(
        mesh=triangle_mesh,
        cell_labels=cell_labels,
        arch=arch,
        render_size=render_size,
    )

    silhouette = render_silhouette(
        mesh=triangle_mesh,
        arch=arch,
        render_size=render_size,
    )

    label_mask = rgb_render_to_label_mask(
        rendered_rgb
    )

    bbox = silhouette_bbox(
        silhouette,
        padding=padding,
    )

    final_mask = crop_and_resize_mask(
        label_mask,
        bbox=bbox,
        target_size=target_size,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        final_mask,
        mode="L",
    ).save(
        output_path,
        format="PNG",
        optimize=True,
    )

    if preview_path is not None:
        preview_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        save_color_preview(
            final_mask,
            preview_path,
        )

    unique_labels = sorted(
        int(value)
        for value in np.unique(
            final_mask
        )
    )

    return {
        "points": int(
            mesh.n_points
        ),
        "faces": int(
            triangle_mesh.n_cells
        ),
        "tooth_vertex_count": int(
            (labels > 0).sum()
        ),
        "unique_output_labels": unique_labels,
        "nonzero_pixels": int(
            (final_mask > 0).sum()
        ),
    }


def discover_cases(
    segmentation_dir: Path,
) -> list[dict[str, object]]:
    cases = []

    for npy_path in sorted(
        segmentation_dir.glob("*.npy")
    ):
        match = FINAL_NPY_PATTERN.fullmatch(
            npy_path.name
        )

        if match is None:
            continue

        cases.append(
            {
                "case": match.group("case"),
                "patient": match.group("patient"),
                "arch": match.group("arch").lower(),
                "npy_path": npy_path,
            }
        )

    return cases


def main() -> int:
    args = parse_args()

    segmentation_dir = (
        args.segmentation_dir.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    if not segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"预测目录不存在："
            f"{segmentation_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = discover_cases(
        segmentation_dir
    )

    if args.limit > 0:
        cases = cases[
            :args.limit
        ]

    if not cases:
        raise RuntimeError(
            "没有找到形如 "
            "case_ios_upper.npy 的最终标签"
        )

    print("=" * 72)
    print("批量生成咬合面整数 label mask")
    print(f"PyVista version: {pv.__version__}")
    print("=" * 72)
    print(
        f"预测目录: {segmentation_dir}"
    )
    print(
        f"输出目录: {output_dir}"
    )
    print(
        f"PLY 来源: <case>_"
        f"{args.ply_suffix}.ply"
    )
    print(
        f"输出尺寸: "
        f"{args.target_size}×"
        f"{args.target_size}"
    )
    print(
        f"发现牙弓: {len(cases)}"
    )
    print()

    rows = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for item in tqdm(
        cases,
        desc="Rendering label masks",
    ):
        case_name = str(
            item["case"]
        )
        arch = str(
            item["arch"]
        )
        npy_path = Path(
            item["npy_path"]
        )

        ply_path = (
            segmentation_dir
            / (
                f"{case_name}_"
                f"{args.ply_suffix}.ply"
            )
        )

        output_path = (
            output_dir
            / f"{case_name}_label.png"
        )

        preview_path = None

        if args.save_preview:
            preview_path = (
                output_dir
                / "preview"
                / (
                    f"{case_name}_"
                    "label_preview.png"
                )
            )

        row = {
            "case": case_name,
            "arch": arch,
            "status": "",
            "npy": str(npy_path),
            "ply": str(ply_path),
            "mask_png": str(output_path),
            "points": "",
            "faces": "",
            "tooth_vertex_count": "",
            "nonzero_pixels": "",
            "unique_output_labels": "",
            "error": "",
        }

        if (
            output_path.exists()
            and not args.overwrite
        ):
            skipped_count += 1
            row["status"] = (
                "skipped_existing"
            )
            rows.append(row)
            continue

        if not ply_path.is_file():
            failed_count += 1
            row["status"] = "missing_ply"
            row["error"] = (
                f"找不到 {ply_path.name}"
            )
            rows.append(row)

            print(
                f"\n✗ {case_name}: "
                f"找不到 {ply_path.name}"
            )
            continue

        try:
            result = render_case(
                ply_path=ply_path,
                npy_path=npy_path,
                output_path=output_path,
                arch=arch,
                target_size=args.target_size,
                render_size=args.render_size,
                padding=args.padding,
                preview_path=preview_path,
            )

            success_count += 1
            row["status"] = "success"
            row["points"] = result[
                "points"
            ]
            row["faces"] = result[
                "faces"
            ]
            row["tooth_vertex_count"] = (
                result[
                    "tooth_vertex_count"
                ]
            )
            row["nonzero_pixels"] = (
                result[
                    "nonzero_pixels"
                ]
            )
            row["unique_output_labels"] = (
                ",".join(
                    str(value)
                    for value
                    in result[
                        "unique_output_labels"
                    ]
                )
            )

        except Exception as exc:
            failed_count += 1
            row["status"] = "failed"
            row["error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"\n✗ {case_name}: "
                f"{row['error']}"
            )
            traceback.print_exc()

        rows.append(row)

    summary_path = (
        output_dir
        / "render_mask_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        fieldnames = [
            "case",
            "arch",
            "status",
            "points",
            "faces",
            "tooth_vertex_count",
            "nonzero_pixels",
            "unique_output_labels",
            "npy",
            "ply",
            "mask_png",
            "error",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("mask 生成完成")
    print("=" * 72)
    print(
        f"成功: {success_count}"
    )
    print(
        f"已存在跳过: {skipped_count}"
    )
    print(
        f"失败: {failed_count}"
    )
    print(
        f"mask 目录: {output_dir}"
    )
    print(
        f"明细表: {summary_path}"
    )
    print(
        "输出像素值范围严格为 0~16。"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
