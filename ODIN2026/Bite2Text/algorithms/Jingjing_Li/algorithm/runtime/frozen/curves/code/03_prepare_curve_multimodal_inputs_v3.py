#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_prepare_curve_multimodal_inputs.py

Prepare the 2D inputs for the rebuilt Curve of Spee / Curve of Wilson module.

This script does NOT train a model.

It prepares two things for each patient:

A) Existing intraoral photographs
   - keeps the original split intraoral_*.jpg/png paths in a manifest
   - DOES NOT duplicate all photos by default
   - records candidate semantic aliases:
         frontal candidate = intraoral_1
         side candidate A  = intraoral_2
         side candidate B  = intraoral_3
     These aliases are intentionally marked as CANDIDATES, not frozen truth.
     The later multimodal evaluator can use the two side views as an unordered
     pair, so left/right naming is not required yet.

B) Standardized STL-derived curve renders
   Spee:
       spee_left.png
       spee_right.png
       spee_pair.png

   Wilson:
       wilson_frontal.png
       wilson_oblique.png
       wilson_pair.png

Important geometry policy
-------------------------
- Reuses 01_extract_curve_geometry.py
- Reuses the same Transverse stable STL/FDI loader
- Builds one joint upper/lower occlusal frame
- NEVER independently aligns upper and lower arches
- AP/LR signs come from FDI anatomy, not raw STL X/Y/Z

Render content
--------------
Spee renders:
    lower teeth only
    left  = 31..37
    right = 41..47
    x-axis = posterior -> anterior
    y-axis = lower -> upper

Wilson renders:
    lower posterior teeth only
    34..37 + 44..47
    frontal:
        x = patient right -> patient left
        y = lower -> upper
    oblique:
        same x
        vertical axis tilted toward an occlusal view

The renders use a robust target-tooth bbox with fixed output aspect ratio and
safety margin. Absolute physical scale is saved as metadata (FOV/mm-per-pixel)
for later fusion with the 3D branch.

Development input
-----------------
Current training data already contains split images under:
    raw_data/Fxxxx/intraoral-photo/

Final challenge input will be one five-image TIFF. The final unified pipeline
will replace this development photo discovery step with the shared TIFF
splitter described in Bite2Text_FINAL_PIPELINE_INTEGRATION_README.md.
The curve render code itself is already compatible with raw STL + FDI labels.

Outputs
-------
multimodal_v2/
    manifest_v1.csv
    summary_v1.json
    failures_v1.json
    photo_view_candidate_manifest_v1.json
    renders/
        Fxxxx/
            spee_left.png
            spee_right.png
            spee_pair.png
            wilson_frontal.png
            wilson_oblique.png
            wilson_pair.png

Usage
-----
Quick test:
    python 03_prepare_curve_multimodal_inputs.py \
        --patient-id generic_case_1 \
        --patient-id generic_case_2

Full run:
    python 03_prepare_curve_multimodal_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = Path("data/raw_data")
DEFAULT_LABEL_ROOT = Path("work/output")
DEFAULT_CURVE_SCRIPT = ROOT / "01_extract_curve_geometry.py"
DEFAULT_GEOMETRY_HELPER = ROOT / "transverse_01_extract_crossbite_geometry.py"
DEFAULT_OUTPUT = ROOT / "multimodal_v3"

LEFT_LOWER_SIDE = (31, 32, 33, 34, 35, 36, 37)
RIGHT_LOWER_SIDE = (41, 42, 43, 44, 45, 46, 47)
LOWER_WILSON = (34, 35, 36, 37, 44, 45, 46, 47)

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    p.add_argument("--curve-script", type=Path, default=DEFAULT_CURVE_SCRIPT)
    p.add_argument(
        "--geometry-helper",
        type=Path,
        default=DEFAULT_GEOMETRY_HELPER,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--patient-id", action="append", default=[])

    p.add_argument("--minimum-tooth-points", type=int, default=20)
    p.add_argument("--pca-points-per-tooth", type=int, default=2000)

    # Render dimensions / physical FOV.
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--fov-width-mm", type=float, default=80.0)
    p.add_argument("--fov-height-mm", type=float, default=48.0)
    p.add_argument(
        "--render-margin",
        type=float,
        default=0.12,
        help="Safety margin on EACH side of the robust target ROI.",
    )
    p.add_argument(
        "--framing-quantile",
        type=float,
        default=0.002,
        help="Ignore only extreme projected outliers when estimating the render ROI.",
    )
    p.add_argument(
        "--minimum-tight-span-mm",
        type=float,
        default=8.0,
        help=(
            "Safety floor for either physical render dimension. "
            "Normally the target-tooth bbox is much larger; this only guards "
            "against pathological partial segmentations."
        ),
    )
    p.add_argument(
        "--wilson-oblique-deg",
        type=float,
        default=28.0,
        help="Tilt frontal Wilson camera toward an occlusal view.",
    )

    p.add_argument(
        "--max-render-points",
        type=int,
        default=180000,
        help="Deterministic maximum points per single render.",
    )
    p.add_argument(
        "--point-expand",
        type=int,
        default=1,
        help="Dark-pixel expansion iterations after rasterization.",
    )

    # Development photo aliases. These are candidates only.
    p.add_argument("--frontal-index", type=int, default=1)
    p.add_argument("--side-a-index", type=int, default=2)
    p.add_argument("--side-b-index", type=int, default=3)

    return p.parse_args()


def load_python_module(path: Path, module_name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_curve_and_base(args):
    curve = load_python_module(
        args.curve_script,
        "_curve_geometry_v1_for_render",
    )

    required_curve = (
        "load_helper",
        "build_canonical_frame",
        "canonical_coords",
    )
    missing = [x for x in required_curve if not hasattr(curve, x)]
    if missing:
        raise ImportError(
            f"01_extract_curve_geometry.py missing: {missing}"
        )

    base = curve.load_helper(
        args.geometry_helper
    )

    return curve, base


def list_patient_ids(raw_root: Path):
    return sorted(
        p.name
        for p in raw_root.glob("F*")
        if p.is_dir()
    )


def photo_sort_key(path: Path):
    """
    Sort intraoral_1, intraoral_2, ... numerically when possible.
    """
    m = re.search(
        r"(\d+)(?!.*\d)",
        path.stem,
    )
    if m:
        return (0, int(m.group(1)), path.name.lower())
    return (1, 10**9, path.name.lower())


def discover_intraoral_photos(patient_dir: Path):
    d = patient_dir / "intraoral-photo"
    if not d.exists():
        return []

    files = [
        p
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTS
    ]
    return sorted(files, key=photo_sort_key)


def find_photo_by_index(files, index: int):
    """
    Prefer an actual filename ending in the requested numeric index.
    Otherwise use 1-based sorted position as compatibility fallback.
    """
    for p in files:
        m = re.search(
            r"(\d+)(?!.*\d)",
            p.stem,
        )
        if m and int(m.group(1)) == int(index):
            return p

    j = int(index) - 1
    if 0 <= j < len(files):
        return files[j]

    return None


def deterministic_sample_indices(n: int, max_points: int):
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(
        0,
        n - 1,
        max_points,
        dtype=np.int64,
    )


def tight_bbox_pixels(
    x,
    y,
    width,
    height,
    render_margin,
    framing_quantile,
    minimum_tight_span_mm,
):
    """
    Compact, robust, aspect-ratio-preserving target framing.

    Goal
    ----
    Make the target teeth occupy much more of the CNN image WITHOUT bringing
    back left/right clipping.

    Procedure
    ---------
    1. Use robust quantiles of the TARGET-TOOTH coordinates to estimate the
       anatomical bbox, so isolated scan points do not control the camera.
    2. Center on the midpoint of bbox bounds. This is symmetric and does not
       depend on left/right vertex density.
    3. Add the requested safety margin on every side.
    4. Expand the shorter physical dimension so that the physical FOV has the
       same aspect ratio as the output image (e.g. 416/256 = 1.625).
       Therefore there is NO geometric stretch when rasterizing.
    5. Do NOT enforce the old 80x48-mm minimum FOV. This is what removes the
       unnecessary white space.
    6. Record the original bbox and the final physical FOV so absolute scale can
       be passed as scalar metadata to the later multimodal model.

    Note
    ----
    framing_quantile=0.002 means only the outermost 0.2% at each edge is
    ignored for framing. The safety margin normally brings nearly all of those
    points back inside the final FOV.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 50:
        raise ValueError("Too few finite render coordinates")

    xg = x[good]
    yg = y[good]

    q = float(np.clip(framing_quantile, 0.0, 0.03))
    margin = float(np.clip(render_margin, 0.02, 0.25))
    min_span = float(max(minimum_tight_span_mm, 1.0))

    xlo = float(np.quantile(xg, q))
    xhi = float(np.quantile(xg, 1.0 - q))
    ylo = float(np.quantile(yg, q))
    yhi = float(np.quantile(yg, 1.0 - q))

    bbox_w = max(xhi - xlo, 1e-6)
    bbox_h = max(yhi - ylo, 1e-6)

    # Margin m on EACH side means anatomical bbox uses (1-2m) of that axis.
    usable = max(1.0 - 2.0 * margin, 0.50)
    need_w = max(bbox_w / usable, min_span)
    need_h = max(bbox_h / usable, min_span)

    # Physical aspect ratio == pixel aspect ratio => no anisotropic stretching.
    aspect = float(width) / float(height)

    if need_w / need_h >= aspect:
        use_w = need_w
        use_h = use_w / aspect
    else:
        use_h = need_h
        use_w = use_h * aspect

    cx = 0.5 * (xlo + xhi)
    cy = 0.5 * (ylo + yhi)

    xmin = cx - use_w / 2.0
    xmax = cx + use_w / 2.0
    ymin = cy - use_h / 2.0
    ymax = cy + use_h / 2.0

    keep = (
        good
        & (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
    )

    n_good = int(np.count_nonzero(good))
    n_keep = int(np.count_nonzero(keep))

    if n_keep < 50:
        raise ValueError("Too few points inside tight render FOV")

    # Quantify any excluded target points. These should only be tiny outliers.
    outside_fraction = float((n_good - n_keep) / max(n_good, 1))

    px = np.full(len(x), -1, dtype=np.int32)
    py = np.full(len(y), -1, dtype=np.int32)

    px[keep] = np.clip(
        np.round(
            (x[keep] - xmin) / (xmax - xmin) * (width - 1)
        ),
        0,
        width - 1,
    ).astype(np.int32)

    # Higher anatomical y appears higher in image.
    py[keep] = np.clip(
        np.round(
            (ymax - y[keep]) / (ymax - ymin) * (height - 1)
        ),
        0,
        height - 1,
    ).astype(np.int32)

    mm_per_pixel_x = float(use_w / float(width))
    mm_per_pixel_y = float(use_h / float(height))

    meta = {
        "center_x_mm": float(cx),
        "center_y_mm": float(cy),

        # Original robust target bbox, BEFORE padding/aspect-ratio completion.
        "actual_bbox_width_mm": float(bbox_w),
        "actual_bbox_height_mm": float(bbox_h),
        "actual_bbox_aspect": float(bbox_w / bbox_h),

        # Final camera FOV.
        "used_fov_width_mm": float(use_w),
        "used_fov_height_mm": float(use_h),
        "used_fov_aspect": float(use_w / use_h),

        # Absolute physical scale retained as metadata.
        "mm_per_pixel_x": mm_per_pixel_x,
        "mm_per_pixel_y": mm_per_pixel_y,
        "mm_per_pixel_mean": float(
            0.5 * (mm_per_pixel_x + mm_per_pixel_y)
        ),

        "framing_quantile": q,
        "render_margin_each_side": margin,
        "points_inside_fov": n_keep,
        "points_total_for_framing": n_good,
        "outside_fov_fraction": outside_fraction,
    }

    return px, py, keep, meta

def rasterize_depth_points(
    x,
    y,
    depth,
    *,
    width,
    height,
    fov_width_mm,
    fov_height_mm,
    render_margin,
    framing_quantile,
    minimum_tight_span_mm,
    max_points,
    point_expand,
):
    """
    Simple deterministic z-buffer-like point renderer.

    Background:
        white

    Surface:
        dark gray, with nearer points darker

    This intentionally avoids OpenGL/PyRender dependencies so the same
    preprocessing can later run inside the submission container.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
    idx = np.where(good)[0]

    if len(idx) < 100:
        raise ValueError(
            f"Too few render points: {len(idx)}"
        )

    if len(idx) > max_points:
        take = deterministic_sample_indices(
            len(idx),
            max_points,
        )
        idx = idx[take]

    x = x[idx]
    y = y[idx]
    depth = depth[idx]

    px, py, keep, framing_meta = tight_bbox_pixels(
        x,
        y,
        width,
        height,
        render_margin,
        framing_quantile,
        minimum_tight_span_mm,
    )

    px = px[keep]
    py = py[keep]
    depth = depth[keep]

    if len(depth) < 100:
        raise ValueError(
            f"Too few points after FOV crop: {len(depth)}"
        )

    dlo = float(np.quantile(depth, 0.01))
    dhi = float(np.quantile(depth, 0.99))

    if dhi - dlo < 1e-6:
        norm = np.full(
            len(depth),
            0.5,
            dtype=np.float64,
        )
    else:
        norm = np.clip(
            (depth - dlo) / (dhi - dlo),
            0.0,
            1.0,
        )

    # Nearer = darker. 45..155 keeps geometry dark on white background.
    gray = np.round(
        155.0 - 110.0 * norm
    ).astype(np.uint8)

    canvas = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    flat = canvas.ravel()
    flat_idx = py.astype(np.int64) * width + px.astype(np.int64)

    # If several points land on one pixel, keep the darkest / nearest.
    np.minimum.at(
        flat,
        flat_idx,
        gray,
    )
    canvas = flat.reshape(height, width)

    if point_expand > 0:
        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )
        canvas = cv2.erode(
            canvas,
            kernel,
            iterations=int(point_expand),
        )

    # Very light close fills isolated single-pixel holes without smoothing
    # away the tooth contours.
    close_kernel = np.ones(
        (2, 2),
        dtype=np.uint8,
    )
    canvas = cv2.morphologyEx(
        canvas,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR,
    )

    meta = {
        "render_points": int(len(depth)),
        **framing_meta,
        "depth_q01_mm": dlo,
        "depth_q99_mm": dhi,
    }

    return rgb, meta


def concatenate_pair(left_img, right_img, separator=8):
    if left_img.shape[0] != right_img.shape[0]:
        raise ValueError("Pair image heights differ")

    sep = np.full(
        (
            left_img.shape[0],
            separator,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    return np.concatenate(
        [left_img, sep, right_img],
        axis=1,
    )


def save_png(path: Path, image):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ok = cv2.imwrite(
        str(path),
        image,
    )
    if not ok:
        raise IOError(
            f"cv2.imwrite failed: {path}"
        )


def build_patient_geometry(
    pid,
    args,
    curve,
    base,
):
    upper = base.load_arch(
        pid,
        "upper",
        args.raw_root,
        args.label_root,
    )
    lower = base.load_arch(
        pid,
        "lower",
        args.raw_root,
        args.label_root,
    )

    common = base.joint_occlusal_frame(
        upper,
        lower,
        args.minimum_tooth_points,
        args.pca_points_per_tooth,
    )
    up2d, upv = base.project(
        upper,
        common,
    )
    lo2d, lov = base.project(
        lower,
        common,
    )

    canonical = curve.build_canonical_frame(
        base,
        lo2d,
        lower.labels,
        args.minimum_tooth_points,
    )

    # Convert every lower stable tooth vertex once.
    lower_ap, lower_lr = curve.canonical_coords(
        lo2d,
        canonical,
    )

    return {
        "upper": upper,
        "lower": lower,
        "up2d": up2d,
        "lo2d": lo2d,
        "upv": upv,
        "lov": lov,
        "lower_ap": lower_ap,
        "lower_lr": lower_lr,
        "canonical": canonical,
    }


def render_spee_side(
    geometry,
    teeth,
    side,
    args,
):
    lower = geometry["lower"]
    mask = np.isin(
        lower.labels,
        np.asarray(teeth, dtype=np.int32),
    )

    if np.count_nonzero(mask) < 500:
        raise ValueError(
            f"{side} Spee render has too few tooth points"
        )

    ap = geometry["lower_ap"][mask]
    lr = geometry["lower_lr"][mask]
    v = geometry["lov"][mask]

    # Standardized side camera:
    #   horizontal = AP (posterior -> anterior)
    #   vertical   = anatomical vertical
    #   depth      = near side toward the camera
    if side == "left":
        depth = lr
    elif side == "right":
        depth = -lr
    else:
        raise ValueError(side)

    return rasterize_depth_points(
        ap,
        v,
        depth,
        width=args.width,
        height=args.height,
        fov_width_mm=args.fov_width_mm,
        fov_height_mm=args.fov_height_mm,
        render_margin=args.render_margin,
        framing_quantile=args.framing_quantile,
        minimum_tight_span_mm=args.minimum_tight_span_mm,
        max_points=args.max_render_points,
        point_expand=args.point_expand,
    )


def render_wilson_frontal(
    geometry,
    args,
):
    lower = geometry["lower"]
    mask = np.isin(
        lower.labels,
        np.asarray(
            LOWER_WILSON,
            dtype=np.int32,
        ),
    )

    if np.count_nonzero(mask) < 1000:
        raise ValueError(
            "Wilson frontal render has too few posterior points"
        )

    lr = geometry["lower_lr"][mask]
    ap = geometry["lower_ap"][mask]
    v = geometry["lov"][mask]

    # Frontal camera from anterior:
    #   x     = patient right -> patient left
    #   y     = vertical
    #   depth = AP, anterior is nearer
    return rasterize_depth_points(
        lr,
        v,
        ap,
        width=args.width,
        height=args.height,
        fov_width_mm=args.fov_width_mm,
        fov_height_mm=args.fov_height_mm,
        render_margin=args.render_margin,
        framing_quantile=args.framing_quantile,
        minimum_tight_span_mm=args.minimum_tight_span_mm,
        max_points=args.max_render_points,
        point_expand=args.point_expand,
    )


def render_wilson_oblique(
    geometry,
    args,
):
    lower = geometry["lower"]
    mask = np.isin(
        lower.labels,
        np.asarray(
            LOWER_WILSON,
            dtype=np.int32,
        ),
    )

    if np.count_nonzero(mask) < 1000:
        raise ValueError(
            "Wilson oblique render has too few posterior points"
        )

    lr = geometry["lower_lr"][mask]
    ap = geometry["lower_ap"][mask]
    v = geometry["lov"][mask]

    theta = np.deg2rad(
        float(args.wilson_oblique_deg)
    )

    # Rotate the frontal camera toward a superior/occlusal viewpoint
    # around the LR axis.
    image_y = (
        np.cos(theta) * v
        - np.sin(theta) * ap
    )
    depth = (
        np.cos(theta) * ap
        + np.sin(theta) * v
    )

    return rasterize_depth_points(
        lr,
        image_y,
        depth,
        width=args.width,
        height=args.height,
        fov_width_mm=args.fov_width_mm,
        fov_height_mm=args.fov_height_mm,
        render_margin=args.render_margin,
        framing_quantile=args.framing_quantile,
        minimum_tight_span_mm=args.minimum_tight_span_mm,
        max_points=args.max_render_points,
        point_expand=args.point_expand,
    )


def patient_row_base(
    pid,
    photo_files,
    args,
):
    row = {
        "patient_id": pid,
        "photo_count": int(len(photo_files)),
        "has_at_least_5_photos": bool(
            len(photo_files) >= 5
        ),
    }

    # Preserve sorted original views.
    for i in range(1, 6):
        p = (
            photo_files[i - 1]
            if i - 1 < len(photo_files)
            else None
        )
        row[f"photo_view_{i}_path"] = (
            "" if p is None else str(p)
        )

    frontal = find_photo_by_index(
        photo_files,
        args.frontal_index,
    )
    side_a = find_photo_by_index(
        photo_files,
        args.side_a_index,
    )
    side_b = find_photo_by_index(
        photo_files,
        args.side_b_index,
    )

    row["photo_frontal_candidate_path"] = (
        "" if frontal is None else str(frontal)
    )
    row["photo_side_candidate_a_path"] = (
        "" if side_a is None else str(side_a)
    )
    row["photo_side_candidate_b_path"] = (
        "" if side_b is None else str(side_b)
    )

    row["photo_frontal_candidate_ready"] = bool(
        frontal is not None
    )
    row["photo_side_pair_candidate_ready"] = bool(
        side_a is not None
        and side_b is not None
    )

    return row


def write_csv(path: Path, rows):
    rows = list(rows)

    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()

    args.raw_root = (
        args.raw_root
        .expanduser()
        .resolve()
    )
    args.label_root = (
        args.label_root
        .expanduser()
        .resolve()
    )
    args.curve_script = (
        args.curve_script
        .expanduser()
        .resolve()
    )
    args.geometry_helper = (
        args.geometry_helper
        .expanduser()
        .resolve()
    )
    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    render_root = (
        args.output_dir
        / "renders"
    )
    render_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve, base = load_curve_and_base(
        args
    )

    patient_ids = (
        list(
            dict.fromkeys(
                args.patient_id
            )
        )
        if args.patient_id
        else list_patient_ids(
            args.raw_root
        )
    )

    print("=" * 104)
    print("Prepare Curve multimodal inputs v3 (tight bbox framing)")
    print("=" * 104)
    print(f"patients           : {len(patient_ids)}")
    print(f"render size        : {args.width} x {args.height}")
    print(
        f"physical FOV       : "
        f"{args.fov_width_mm:.1f} x "
        f"{args.fov_height_mm:.1f} mm"
    )
    print(
        f"Wilson oblique     : "
        f"{args.wilson_oblique_deg:.1f} deg"
    )
    print(
        "photo aliases      : "
        f"frontal={args.frontal_index}, "
        f"sideA={args.side_a_index}, "
        f"sideB={args.side_b_index} "
        "(CANDIDATE ONLY)"
    )
    print("=" * 104)

    rows = []
    failures = []

    for i, pid in enumerate(
        patient_ids,
        1,
    ):
        patient_dir = (
            args.raw_root / pid
        )
        photo_files = discover_intraoral_photos(
            patient_dir
        )
        row = patient_row_base(
            pid,
            photo_files,
            args,
        )

        patient_render_dir = (
            render_root / pid
        )

        try:
            geometry = build_patient_geometry(
                pid,
                args,
                curve,
                base,
            )
            row["geometry_ok"] = True
            row["geometry_error"] = ""

            # ----------------------- Spee -----------------------
            try:
                spee_left, meta_left = (
                    render_spee_side(
                        geometry,
                        LEFT_LOWER_SIDE,
                        "left",
                        args,
                    )
                )
                spee_right, meta_right = (
                    render_spee_side(
                        geometry,
                        RIGHT_LOWER_SIDE,
                        "right",
                        args,
                    )
                )

                spee_pair = concatenate_pair(
                    spee_left,
                    spee_right,
                )

                spee_left_path = (
                    patient_render_dir
                    / "spee_left.png"
                )
                spee_right_path = (
                    patient_render_dir
                    / "spee_right.png"
                )
                spee_pair_path = (
                    patient_render_dir
                    / "spee_pair.png"
                )

                save_png(
                    spee_left_path,
                    spee_left,
                )
                save_png(
                    spee_right_path,
                    spee_right,
                )
                save_png(
                    spee_pair_path,
                    spee_pair,
                )

                row["spee_render_ready"] = True
                row["spee_render_error"] = ""
                row["spee_left_render_path"] = str(
                    spee_left_path
                )
                row["spee_right_render_path"] = str(
                    spee_right_path
                )
                row["spee_pair_render_path"] = str(
                    spee_pair_path
                )
                row[
                    "spee_left_render_points"
                ] = meta_left["render_points"]
                row[
                    "spee_right_render_points"
                ] = meta_right["render_points"]

                for view_name, meta in (
                    ("left", meta_left),
                    ("right", meta_right),
                ):
                    for key in (
                        "actual_bbox_width_mm",
                        "actual_bbox_height_mm",
                        "used_fov_width_mm",
                        "used_fov_height_mm",
                        "mm_per_pixel_mean",
                        "outside_fov_fraction",
                    ):
                        row[
                            f"spee_{view_name}_{key}"
                        ] = meta[key]

            except Exception as e:
                row["spee_render_ready"] = False
                row["spee_render_error"] = repr(e)
                row["spee_left_render_path"] = ""
                row["spee_right_render_path"] = ""
                row["spee_pair_render_path"] = ""

            # ---------------------- Wilson ----------------------
            try:
                wilson_frontal, meta_front = (
                    render_wilson_frontal(
                        geometry,
                        args,
                    )
                )
                wilson_oblique, meta_oblique = (
                    render_wilson_oblique(
                        geometry,
                        args,
                    )
                )

                wilson_pair = concatenate_pair(
                    wilson_frontal,
                    wilson_oblique,
                )

                wilson_frontal_path = (
                    patient_render_dir
                    / "wilson_frontal.png"
                )
                wilson_oblique_path = (
                    patient_render_dir
                    / "wilson_oblique.png"
                )
                wilson_pair_path = (
                    patient_render_dir
                    / "wilson_pair.png"
                )

                save_png(
                    wilson_frontal_path,
                    wilson_frontal,
                )
                save_png(
                    wilson_oblique_path,
                    wilson_oblique,
                )
                save_png(
                    wilson_pair_path,
                    wilson_pair,
                )

                row["wilson_render_ready"] = True
                row["wilson_render_error"] = ""
                row[
                    "wilson_frontal_render_path"
                ] = str(wilson_frontal_path)
                row[
                    "wilson_oblique_render_path"
                ] = str(wilson_oblique_path)
                row[
                    "wilson_pair_render_path"
                ] = str(wilson_pair_path)
                row[
                    "wilson_frontal_render_points"
                ] = meta_front["render_points"]
                row[
                    "wilson_oblique_render_points"
                ] = meta_oblique["render_points"]

                for view_name, meta in (
                    ("frontal", meta_front),
                    ("oblique", meta_oblique),
                ):
                    for key in (
                        "actual_bbox_width_mm",
                        "actual_bbox_height_mm",
                        "used_fov_width_mm",
                        "used_fov_height_mm",
                        "mm_per_pixel_mean",
                        "outside_fov_fraction",
                    ):
                        row[
                            f"wilson_{view_name}_{key}"
                        ] = meta[key]

            except Exception as e:
                row["wilson_render_ready"] = False
                row["wilson_render_error"] = repr(e)
                row[
                    "wilson_frontal_render_path"
                ] = ""
                row[
                    "wilson_oblique_render_path"
                ] = ""
                row[
                    "wilson_pair_render_path"
                ] = ""

            status = (
                "OK"
                if (
                    row.get("spee_render_ready")
                    and row.get(
                        "wilson_render_ready"
                    )
                )
                else "PARTIAL"
            )

        except Exception as e:
            row["geometry_ok"] = False
            row["geometry_error"] = repr(e)
            row["spee_render_ready"] = False
            row["spee_render_error"] = (
                "base_geometry_failed"
            )
            row["wilson_render_ready"] = False
            row["wilson_render_error"] = (
                "base_geometry_failed"
            )

            failures.append(
                {
                    "patient_id": pid,
                    "error": repr(e),
                }
            )
            status = "FAIL"

        rows.append(row)

        if (
            len(patient_ids) <= 30
            or i % 10 == 0
            or i == len(patient_ids)
            or status != "OK"
        ):
            print(
                f"[{i:4d}/{len(patient_ids):4d}] "
                f"{pid}: {status} | "
                f"photos={len(photo_files)} | "
                f"Spee={row.get('spee_render_ready', False)} | "
                f"Wilson={row.get('wilson_render_ready', False)}"
            )

    manifest_path = (
        args.output_dir
        / "manifest_v1.csv"
    )
    summary_path = (
        args.output_dir
        / "summary_v1.json"
    )
    failures_path = (
        args.output_dir
        / "failures_v1.json"
    )
    photo_manifest_path = (
        args.output_dir
        / "photo_view_candidate_manifest_v1.json"
    )

    write_csv(
        manifest_path,
        rows,
    )

    summary = {
        "version": "curve-multimodal-inputs-v3-tight-bbox",
        "patients_requested": int(
            len(patient_ids)
        ),
        "geometry_ok": int(
            sum(
                bool(r.get("geometry_ok"))
                for r in rows
            )
        ),
        "photo_at_least_5": int(
            sum(
                bool(
                    r.get(
                        "has_at_least_5_photos"
                    )
                )
                for r in rows
            )
        ),
        "photo_frontal_candidate_ready": int(
            sum(
                bool(
                    r.get(
                        "photo_frontal_candidate_ready"
                    )
                )
                for r in rows
            )
        ),
        "photo_side_pair_candidate_ready": int(
            sum(
                bool(
                    r.get(
                        "photo_side_pair_candidate_ready"
                    )
                )
                for r in rows
            )
        ),
        "spee_render_ready": int(
            sum(
                bool(
                    r.get(
                        "spee_render_ready"
                    )
                )
                for r in rows
            )
        ),
        "wilson_render_ready": int(
            sum(
                bool(
                    r.get(
                        "wilson_render_ready"
                    )
                )
                for r in rows
            )
        ),
        "both_curve_renders_ready": int(
            sum(
                bool(
                    r.get(
                        "spee_render_ready"
                    )
                )
                and bool(
                    r.get(
                        "wilson_render_ready"
                    )
                )
                for r in rows
            )
        ),
        "render_config": {
            "width": int(args.width),
            "height": int(args.height),
            "fov_width_mm": float(
                args.fov_width_mm
            ),
            "fov_height_mm": float(
                args.fov_height_mm
            ),
            "render_margin": float(
                args.render_margin
            ),
            "framing_quantile": float(
                args.framing_quantile
            ),
            "minimum_tight_span_mm": float(
                args.minimum_tight_span_mm
            ),
            "framing_policy": (
                "robust tight bbox + symmetric margin + fixed output aspect ratio"
            ),
            "wilson_oblique_deg": float(
                args.wilson_oblique_deg
            ),
            "max_render_points": int(
                args.max_render_points
            ),
            "point_expand": int(
                args.point_expand
            ),
        },
        "photo_candidate_mapping": {
            "status": "candidate_unconfirmed",
            "frontal_index": int(
                args.frontal_index
            ),
            "side_candidate_a_index": int(
                args.side_a_index
            ),
            "side_candidate_b_index": int(
                args.side_b_index
            ),
            "note": (
                "The later evaluator may treat side A/B as an "
                "unordered shared-encoder pair. Final TIFF tile "
                "semantics must be frozen separately before submission."
            ),
        },
        "failures": failures,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    failures_path.write_text(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    photo_manifest = {
        "version": "photo-view-candidate-manifest-v1",
        "status": "candidate_unconfirmed",
        "development_source": (
            "raw_data/<patient>/intraoral-photo/intraoral_*"
        ),
        "final_challenge_source": (
            "single five-image TIFF; use shared final TIFF splitter"
        ),
        "candidate_aliases": {
            "frontal": int(
                args.frontal_index
            ),
            "side_A": int(
                args.side_a_index
            ),
            "side_B": int(
                args.side_b_index
            ),
        },
        "important": (
            "Do not treat these candidate aliases as final TIFF semantics "
            "until the five-view order is explicitly verified and frozen."
        ),
    }

    photo_manifest_path.write_text(
        json.dumps(
            photo_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(
        f"geometry OK            : "
        f"{summary['geometry_ok']}/{len(patient_ids)}"
    )
    print(
        f">=5 photos             : "
        f"{summary['photo_at_least_5']}/{len(patient_ids)}"
    )
    print(
        f"side photo pair ready  : "
        f"{summary['photo_side_pair_candidate_ready']}/{len(patient_ids)}"
    )
    print(
        f"Spee renders ready     : "
        f"{summary['spee_render_ready']}/{len(patient_ids)}"
    )
    print(
        f"Wilson renders ready   : "
        f"{summary['wilson_render_ready']}/{len(patient_ids)}"
    )
    print(
        f"both renders ready     : "
        f"{summary['both_curve_renders_ready']}/{len(patient_ids)}"
    )
    print(f"manifest      : {manifest_path}")
    print(f"summary       : {summary_path}")
    print(f"failures      : {failures_path}")
    print(f"photo mapping : {photo_manifest_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
