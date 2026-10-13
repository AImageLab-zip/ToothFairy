#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_extract_curve_geometry.py

重新实现 Curve of Spee + Curve of Wilson 的 3D 几何特征提取。

设计原则
--------
1) 不再沿用旧的 "Spee = 1 个 depth 数值"。
2) Spee / Wilson 共用同一次 stable STL + FDI segmentation + joint occlusal frame。
3) 直接复用已经验证过的 Transverse/01 helper:
       load_arch()
       joint_occlusal_frame()
       project()
       tooth_centers()
   因而：
       - STL stable vertex 顺序与 DADSN JSON 保持一致
       - upper/lower 不会独立 PCA
       - 两个牙弓仍在同一 registered bite coordinate system
4) 所有方向都由 FDI 解剖语义重新定向：
       AP: posterior -> anterior 为正
       LR: patient right -> patient left 为正
       V : lower -> upper 为正
   不依赖原始 STL X/Y/Z 的符号。
5) 本脚本不读取 GT，不调阈值，只提取几何。

Spee
----
下颌左右两侧分别建立曲线：
    left : 31,33,34,35,36,37
    right: 41,43,44,45,46,47

每颗牙提取 robust occlusal/buccal landmark。
对 AP-V 拟合二次曲线，并输出：
    - chord depth
    - reverse elevation
    - signed dominant depth
    - quadratic coefficient
    - fit RMSE / R2
    - observed depth
    - anterior/posterior slope
    - bilateral asymmetry
    - per-tooth landmark heights

Wilson
------
主分支：lower posterior
    left : 34,35,36,37
    right: 44,45,46,47

辅助分支：upper posterior
    left : 24,25,26,27
    right: 14,15,16,17

每颗后牙：
    - robust buccal cusp-region landmark
    - robust lingual cusp-region landmark
    - buccal-lingual vertical difference
    - transverse inclination angle

每个左右同名 tooth pair：
    - 4 cusp points 的 transverse quadratic curve
    - chord depression/elevation
    - pair width
    - left/right inclination mean/asymmetry

输出
----
results/
    curve_geometry_features_v1.csv
    curve_geometry_key_v1.csv
    curve_geometry_summary_v1.json
    curve_geometry_failures_v1.json

可选：
debug/geometry_v1/
    <patient>_curve_geometry_v1.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = Path("data/raw_data")
DEFAULT_LABEL_ROOT = Path("work/output")
DEFAULT_HELPER = ROOT / "transverse_01_extract_crossbite_geometry.py"
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_DEBUG = ROOT / "debug" / "geometry_v1"

LOWER_SPEE = {
    "left": (31, 33, 34, 35, 36, 37),
    "right": (41, 43, 44, 45, 46, 47),
}

LOWER_WILSON_PAIRS = (
    (34, 44),
    (35, 45),
    (36, 46),
    (37, 47),
)

UPPER_WILSON_PAIRS = (
    (14, 24),
    (15, 25),
    (16, 26),
    (17, 27),
)

LOWER_POSTERIOR = (34,35,36,37,44,45,46,47)
UPPER_POSTERIOR = (14,15,16,17,24,25,26,27)

LOWER_ALL = (
    31,32,33,34,35,36,37,38,
    41,42,43,44,45,46,47,48,
)
UPPER_ALL = (
    11,12,13,14,15,16,17,18,
    21,22,23,24,25,26,27,28,
)


@dataclass(frozen=True)
class CanonicalFrame2D:
    origin: np.ndarray
    ap: np.ndarray
    lr: np.ndarray
    anterior_center: np.ndarray
    posterior_center: np.ndarray
    left_center: np.ndarray
    right_center: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    p.add_argument("--geometry-helper", type=Path, default=DEFAULT_HELPER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG)
    p.add_argument("--patient-id", action="append", default=[])

    p.add_argument("--minimum-tooth-points", type=int, default=20)
    p.add_argument("--pca-points-per-tooth", type=int, default=2000)

    # robust landmark parameters
    p.add_argument(
        "--surface-fraction",
        type=float,
        default=0.12,
        help="Fraction of most occlusal points retained within each candidate region.",
    )
    p.add_argument(
        "--buccal-half-quantile",
        type=float,
        default=0.50,
        help="Quantile separating buccal/outward half from central/lingual half.",
    )
    p.add_argument(
        "--lingual-half-quantile",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--minimum-landmark-points",
        type=int,
        default=8,
    )

    p.add_argument("--save-debug", action="store_true")
    return p.parse_args()


def load_helper(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    name = "_curve_shared_geometry_helper"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import geometry helper: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    required = (
        "load_arch",
        "joint_occlusal_frame",
        "project",
        "tooth_centers",
        "UPPER_FDI",
        "LOWER_FDI",
    )
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise ImportError(
            f"Transverse helper missing required objects: {missing}"
        )
    return mod


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return v / n


def median_centers(centers, teeth):
    pts = [centers[t] for t in teeth if t in centers]
    if not pts:
        raise ValueError(f"No available centers among {list(teeth)}")
    return np.median(np.vstack(pts), axis=0)


def build_canonical_frame(base, lo2d, lower_labels, min_points):
    """
    在 joint occlusal 2D plane 中，利用 FDI 解剖位置确定 AP/LR 符号。

    AP positive : posterior -> anterior
    LR positive : patient right -> patient left
    """
    centers = base.tooth_centers(
        lo2d,
        lower_labels,
        list(LOWER_ALL),
        min_points,
    )

    if len(centers) < 8:
        raise ValueError(
            f"Too few lower tooth centers for canonical frame: {len(centers)}"
        )

    # Anterior: central/lateral incisors; posterior: molars with premolar fallback.
    anterior = median_centers(
        centers,
        (31, 41, 32, 42),
    )
    left_post = median_centers(
        centers,
        (37, 36, 35, 34),
    )
    right_post = median_centers(
        centers,
        (47, 46, 45, 44),
    )
    posterior = 0.5 * (left_post + right_post)

    ap = unit(anterior - posterior)

    left_side = median_centers(
        centers,
        (33,34,35,36,37),
    )
    right_side = median_centers(
        centers,
        (43,44,45,46,47),
    )

    # Gram-Schmidt: left-right axis orthogonal to AP.
    lr_raw = left_side - right_side
    lr = lr_raw - float(np.dot(lr_raw, ap)) * ap
    lr = unit(lr)

    # Recompute AP to make exact orthonormal basis while preserving anterior sign.
    ap2 = np.asarray([lr[1], -lr[0]], dtype=np.float64)
    if np.dot(ap2, ap) < 0:
        ap2 = -ap2
    ap = unit(ap2)

    origin = np.median(np.vstack(list(centers.values())), axis=0)

    return CanonicalFrame2D(
        origin=origin,
        ap=ap,
        lr=lr,
        anterior_center=anterior,
        posterior_center=posterior,
        left_center=left_side,
        right_center=right_side,
    )


def canonical_coords(points2d, frame: CanonicalFrame2D):
    p = np.asarray(points2d, dtype=np.float64) - frame.origin
    ap = p @ frame.ap
    lr = p @ frame.lr
    return ap, lr


def robust_arch_center(base, points2d, labels, teeth, min_points):
    centers = base.tooth_centers(
        points2d,
        labels,
        list(teeth),
        min_points,
    )
    if len(centers) < 6:
        raise ValueError("Too few tooth centers for robust arch center")
    return np.median(np.vstack(list(centers.values())), axis=0)


def surface_mask(vertical, arch: str, fraction: float):
    if len(vertical) == 0:
        return np.zeros(0, dtype=bool)

    fraction = float(np.clip(fraction, 0.02, 0.40))

    if arch == "lower":
        q = np.quantile(vertical, 1.0 - fraction)
        return vertical >= q

    if arch == "upper":
        q = np.quantile(vertical, fraction)
        return vertical <= q

    raise ValueError(arch)


def robust_landmark(
    points2d,
    vertical,
    labels,
    tooth,
    arch,
    arch_center,
    min_tooth_points,
    surface_fraction,
    min_landmark_points,
    region="all",
    buccal_half_quantile=0.50,
    lingual_half_quantile=0.50,
):
    """
    Robust occlusal/incisal/cusp-region landmark.

    region:
        all      : incisal / generic occlusal landmark
        buccal   : outward half of tooth
        lingual  : inward half of tooth
    """
    mask = labels == tooth
    n_tooth = int(np.count_nonzero(mask))

    if n_tooth < min_tooth_points:
        raise ValueError(
            f"tooth {tooth}: too few points ({n_tooth})"
        )

    p = points2d[mask]
    v = vertical[mask]
    center = np.median(p, axis=0)

    outward = unit(center - arch_center)
    radial = (p - center) @ outward

    region_mask = np.ones(len(p), dtype=bool)

    if region == "buccal":
        q = np.quantile(radial, buccal_half_quantile)
        region_mask = radial >= q
    elif region == "lingual":
        q = np.quantile(radial, lingual_half_quantile)
        region_mask = radial <= q
    elif region != "all":
        raise ValueError(region)

    if np.count_nonzero(region_mask) < min_landmark_points:
        raise ValueError(
            f"tooth {tooth} {region}: too few region points"
        )

    pp = p[region_mask]
    vv = v[region_mask]

    sm = surface_mask(vv, arch, surface_fraction)
    if np.count_nonzero(sm) < min_landmark_points:
        # Relax to enough most-occlusal points deterministically.
        order = np.argsort(vv)
        if arch == "lower":
            order = order[::-1]
        k = min(
            len(vv),
            max(min_landmark_points, int(round(len(vv) * surface_fraction))),
        )
        keep = order[:k]
        sel_p = pp[keep]
        sel_v = vv[keep]
    else:
        sel_p = pp[sm]
        sel_v = vv[sm]

    # Median is intentionally used instead of max/min to reject isolated scan noise.
    point2d = np.median(sel_p, axis=0)
    z = float(np.median(sel_v))

    return {
        "tooth": int(tooth),
        "region": region,
        "point2d": point2d,
        "vertical": z,
        "tooth_center2d": center,
        "outward2d": outward,
        "tooth_points": n_tooth,
        "region_points": int(np.count_nonzero(region_mask)),
        "landmark_points": int(len(sel_v)),
        "vertical_std": float(np.std(sel_v)),
    }


def fit_curve_features(ap, z):
    """
    Fit z(u)=c2*u^2+c1*u+c0 where u in [0,1] from posterior->anterior.

    Chord delta = chord_z - fitted_z
        positive -> fitted curve lies below endpoint chord
        negative -> fitted curve lies above endpoint chord
    """
    ap = np.asarray(ap, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    good = np.isfinite(ap) & np.isfinite(z)
    ap = ap[good]
    z = z[good]

    if len(ap) < 4:
        raise ValueError("Need >=4 landmarks to fit Spee curve")

    order = np.argsort(ap)
    ap = ap[order]
    z = z[order]

    span = float(ap[-1] - ap[0])
    if span < 5.0:
        raise ValueError(f"AP span too small: {span:.3f} mm")

    u = (ap - ap[0]) / span

    coef = np.polyfit(u, z, 2)
    fitted = np.polyval(coef, u)

    residual = z - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 1e-9
        else 1.0
    )

    grid = np.linspace(0.0, 1.0, 201)
    curve = np.polyval(coef, grid)
    chord = z[0] + (z[-1] - z[0]) * grid
    delta = chord - curve

    max_depth = float(np.max(delta))
    max_reverse = float(np.max(-delta))
    signed = (
        max_depth
        if max_depth >= max_reverse
        else -max_reverse
    )

    observed_chord = z[0] + (z[-1] - z[0]) * u
    observed_delta = observed_chord - z

    # derivative dz/dx in real mm coordinate
    # z = c2*u^2 + c1*u + c0, u=(x-x0)/span
    slope_post = float(coef[1] / span)
    slope_ant = float((2.0 * coef[0] + coef[1]) / span)

    return {
        "n_landmarks": int(len(ap)),
        "ap_span_mm": span,
        "coef2_normalized": float(coef[0]),
        "coef1_normalized": float(coef[1]),
        "fit_rmse_mm": rmse,
        "fit_r2": r2,
        "max_chord_depth_mm": max_depth,
        "max_reverse_elevation_mm": max_reverse,
        "signed_dominant_depth_mm": signed,
        "mean_chord_depth_mm": float(np.mean(delta)),
        "positive_depth_area_mm": float(
            np.trapz(np.maximum(delta, 0.0), grid) * span
        ),
        "reverse_area_mm": float(
            np.trapz(np.maximum(-delta, 0.0), grid) * span
        ),
        "observed_max_depth_mm": float(np.max(observed_delta)),
        "observed_max_reverse_mm": float(np.max(-observed_delta)),
        "observed_mean_depth_mm": float(np.mean(observed_delta)),
        "posterior_slope_dz_dx": slope_post,
        "anterior_slope_dz_dx": slope_ant,
    }


def analyze_spee_side(
    side,
    base,
    lo2d,
    lov,
    lower_labels,
    lower_center,
    canonical,
    args,
):
    teeth = LOWER_SPEE[side]
    landmarks = {}

    for tooth in teeth:
        # central incisor: incisal landmark from whole tooth
        # canine/premolar/molar: outward/buccal occlusal landmark
        region = "all" if tooth in (31, 41) else "buccal"

        try:
            lm = robust_landmark(
                lo2d,
                lov,
                lower_labels,
                tooth,
                "lower",
                lower_center,
                args.minimum_tooth_points,
                args.surface_fraction,
                args.minimum_landmark_points,
                region=region,
                buccal_half_quantile=args.buccal_half_quantile,
                lingual_half_quantile=args.lingual_half_quantile,
            )
            apc, lrc = canonical_coords(
                lm["point2d"][None, :],
                canonical,
            )
            lm["ap_mm"] = float(apc[0])
            lm["lr_mm"] = float(lrc[0])
            landmarks[tooth] = lm
        except Exception as e:
            landmarks[tooth] = {
                "tooth": int(tooth),
                "valid": False,
                "error": repr(e),
            }

    valid = [
        lm for lm in landmarks.values()
        if "ap_mm" in lm
    ]

    anterior_tooth = 31 if side == "left" else 41
    posterior_core = (36, 37) if side == "left" else (46, 47)

    has_anterior = anterior_tooth in landmarks and "ap_mm" in landmarks[anterior_tooth]
    has_posterior = any(
        t in landmarks and "ap_mm" in landmarks[t]
        for t in posterior_core
    )

    if len(valid) < 4 or not has_anterior or not has_posterior:
        return {
            "valid": False,
            "error": (
                f"insufficient landmarks: n={len(valid)}, "
                f"anterior={has_anterior}, posterior={has_posterior}"
            ),
            "landmarks": landmarks,
        }

    # Sort only for fit; fit_curve_features handles sorting itself.
    ap = [lm["ap_mm"] for lm in valid]
    z = [lm["vertical"] for lm in valid]
    fit = fit_curve_features(ap, z)

    out = {
        "valid": True,
        "error": "",
        **fit,
        "landmarks": landmarks,
    }
    return out


def posterior_cusp_features(
    tooth,
    arch,
    points2d,
    vertical,
    labels,
    arch_center,
    canonical,
    args,
):
    buccal = robust_landmark(
        points2d,
        vertical,
        labels,
        tooth,
        arch,
        arch_center,
        args.minimum_tooth_points,
        args.surface_fraction,
        args.minimum_landmark_points,
        region="buccal",
        buccal_half_quantile=args.buccal_half_quantile,
        lingual_half_quantile=args.lingual_half_quantile,
    )
    lingual = robust_landmark(
        points2d,
        vertical,
        labels,
        tooth,
        arch,
        arch_center,
        args.minimum_tooth_points,
        args.surface_fraction,
        args.minimum_landmark_points,
        region="lingual",
        buccal_half_quantile=args.buccal_half_quantile,
        lingual_half_quantile=args.lingual_half_quantile,
    )

    b_ap, b_lr = canonical_coords(
        buccal["point2d"][None, :],
        canonical,
    )
    l_ap, l_lr = canonical_coords(
        lingual["point2d"][None, :],
        canonical,
    )

    # local transverse distance measured in 2D joint occlusal plane
    vec = buccal["point2d"] - lingual["point2d"]
    span = float(np.linalg.norm(vec))
    if span < 0.5:
        raise ValueError(
            f"tooth {tooth}: buccal-lingual span too small ({span:.3f})"
        )

    dz = float(buccal["vertical"] - lingual["vertical"])
    slope = dz / span
    angle = float(np.degrees(np.arctan(slope)))

    return {
        "valid": True,
        "tooth": int(tooth),
        "arch": arch,
        "buccal_ap_mm": float(b_ap[0]),
        "buccal_lr_mm": float(b_lr[0]),
        "buccal_z_mm": float(buccal["vertical"]),
        "lingual_ap_mm": float(l_ap[0]),
        "lingual_lr_mm": float(l_lr[0]),
        "lingual_z_mm": float(lingual["vertical"]),
        "buccal_lingual_span_mm": span,
        "buccal_minus_lingual_z_mm": dz,
        "transverse_slope_dz_ds": float(slope),
        "transverse_inclination_deg": angle,
        "buccal_landmark_points": int(buccal["landmark_points"]),
        "lingual_landmark_points": int(lingual["landmark_points"]),
        "buccal_vertical_std": float(buccal["vertical_std"]),
        "lingual_vertical_std": float(lingual["vertical_std"]),
    }


def pair_curve_features(left_feat, right_feat):
    """
    Use 4 cusp-region points across one bilateral tooth-number pair.

    LR coordinates:
        right buccal -> right lingual -> left lingual -> left buccal
    in typical anatomy, but we sort numerically and do not hard-code order.

    We fit z(lr) and compare against the chord joining the two outermost points.
    """
    points = [
        (right_feat["buccal_lr_mm"], right_feat["buccal_z_mm"], "right_buccal"),
        (right_feat["lingual_lr_mm"], right_feat["lingual_z_mm"], "right_lingual"),
        (left_feat["lingual_lr_mm"], left_feat["lingual_z_mm"], "left_lingual"),
        (left_feat["buccal_lr_mm"], left_feat["buccal_z_mm"], "left_buccal"),
    ]

    points = sorted(points, key=lambda x: x[0])
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    z = np.asarray([p[1] for p in points], dtype=np.float64)

    span = float(x[-1] - x[0])
    if span < 8.0:
        raise ValueError(f"pair transverse span too small: {span:.3f}")

    u = (x - x[0]) / span
    coef = np.polyfit(u, z, 2)
    fit = np.polyval(coef, u)

    grid = np.linspace(0, 1, 201)
    curve = np.polyval(coef, grid)
    chord = z[0] + (z[-1] - z[0]) * grid
    delta = chord - curve

    residual = z - fit
    rmse = float(np.sqrt(np.mean(residual**2)))

    return {
        "valid": True,
        "pair_width_mm": span,
        "quadratic_coef2_normalized": float(coef[0]),
        "quadratic_coef1_normalized": float(coef[1]),
        "fit_rmse_mm": rmse,
        "max_chord_depression_mm": float(np.max(delta)),
        "max_chord_elevation_mm": float(np.max(-delta)),
        "mean_chord_depression_mm": float(np.mean(delta)),
        "left_inclination_deg": float(left_feat["transverse_inclination_deg"]),
        "right_inclination_deg": float(right_feat["transverse_inclination_deg"]),
        "mean_inclination_deg": float(
            0.5 * (
                left_feat["transverse_inclination_deg"]
                + right_feat["transverse_inclination_deg"]
            )
        ),
        "inclination_asymmetry_deg": float(
            abs(
                left_feat["transverse_inclination_deg"]
                - right_feat["transverse_inclination_deg"]
            )
        ),
        "mean_abs_inclination_deg": float(
            0.5 * (
                abs(left_feat["transverse_inclination_deg"])
                + abs(right_feat["transverse_inclination_deg"])
            )
        ),
        "cusp_order": [p[2] for p in points],
    }


def summarize_values(values):
    arr = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=np.float64,
    )
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def flatten_scalar_dict(row, prefix, d):
    for k, v in d.items():
        if isinstance(v, (str, bool, int, float, np.integer, np.floating)) or v is None:
            row[f"{prefix}_{k}"] = v


def analyze_wilson_arch(
    arch_name,
    pair_defs,
    points2d,
    vertical,
    labels,
    arch_center,
    canonical,
    args,
):
    tooth_results = {}
    pair_results = {}

    teeth = sorted(set(t for pair in pair_defs for t in pair))

    for tooth in teeth:
        try:
            tooth_results[tooth] = posterior_cusp_features(
                tooth,
                arch_name,
                points2d,
                vertical,
                labels,
                arch_center,
                canonical,
                args,
            )
        except Exception as e:
            tooth_results[tooth] = {
                "valid": False,
                "tooth": int(tooth),
                "arch": arch_name,
                "error": repr(e),
            }

    for left_tooth, right_tooth in pair_defs:
        key = f"{left_tooth}_{right_tooth}"
        lf = tooth_results[left_tooth]
        rf = tooth_results[right_tooth]

        if not lf.get("valid") or not rf.get("valid"):
            pair_results[key] = {
                "valid": False,
                "left_tooth": int(left_tooth),
                "right_tooth": int(right_tooth),
                "error": "one_or_both_tooth_features_invalid",
            }
            continue

        try:
            pair_results[key] = {
                "left_tooth": int(left_tooth),
                "right_tooth": int(right_tooth),
                **pair_curve_features(lf, rf),
            }
        except Exception as e:
            pair_results[key] = {
                "valid": False,
                "left_tooth": int(left_tooth),
                "right_tooth": int(right_tooth),
                "error": repr(e),
            }

    valid_teeth = [
        d for d in tooth_results.values()
        if d.get("valid")
    ]
    valid_pairs = [
        d for d in pair_results.values()
        if d.get("valid")
    ]

    aggregates = {
        "valid_teeth": int(len(valid_teeth)),
        "valid_pairs": int(len(valid_pairs)),
        "geometry_valid": bool(len(valid_pairs) >= 3),
        "geometry_any_valid": bool(len(valid_pairs) >= 2),
    }

    for source_key, out_name in (
        ("transverse_inclination_deg", "tooth_inclination"),
        ("buccal_minus_lingual_z_mm", "buccal_minus_lingual_z"),
    ):
        s = summarize_values(
            [d.get(source_key) for d in valid_teeth]
        )
        for stat, value in s.items():
            aggregates[f"{out_name}_{stat}"] = value

    for source_key, out_name in (
        ("max_chord_depression_mm", "pair_chord_depression"),
        ("max_chord_elevation_mm", "pair_chord_elevation"),
        ("mean_abs_inclination_deg", "pair_abs_inclination"),
        ("inclination_asymmetry_deg", "pair_inclination_asymmetry"),
        ("quadratic_coef2_normalized", "pair_quadratic_coef2"),
    ):
        s = summarize_values(
            [d.get(source_key) for d in valid_pairs]
        )
        for stat, value in s.items():
            aggregates[f"{out_name}_{stat}"] = value

    # posterior progression: pair metric vs tooth index rank (premolar -> molar)
    if len(valid_pairs) >= 2:
        pair_order = []
        for rank, pair in enumerate(pair_defs):
            key = f"{pair[0]}_{pair[1]}"
            d = pair_results[key]
            if d.get("valid"):
                pair_order.append(
                    (
                        rank,
                        d["mean_abs_inclination_deg"],
                        d["max_chord_depression_mm"],
                    )
                )

        if len(pair_order) >= 2:
            x = np.asarray([x[0] for x in pair_order], dtype=float)
            incl = np.asarray([x[1] for x in pair_order], dtype=float)
            dep = np.asarray([x[2] for x in pair_order], dtype=float)

            aggregates["posterior_progression_inclination_per_tooth"] = float(
                np.polyfit(x, incl, 1)[0]
            )
            aggregates["posterior_progression_depression_per_tooth"] = float(
                np.polyfit(x, dep, 1)[0]
            )

    return {
        "teeth": tooth_results,
        "pairs": pair_results,
        "aggregates": aggregates,
    }


def analyze_patient(pid, args, base):
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
    up2d, upv = base.project(upper, common)
    lo2d, lov = base.project(lower, common)

    canonical = build_canonical_frame(
        base,
        lo2d,
        lower.labels,
        args.minimum_tooth_points,
    )

    lower_center = robust_arch_center(
        base,
        lo2d,
        lower.labels,
        LOWER_ALL,
        args.minimum_tooth_points,
    )
    upper_center = robust_arch_center(
        base,
        up2d,
        upper.labels,
        UPPER_ALL,
        args.minimum_tooth_points,
    )

    row = {
        "patient_id": pid,
        "geometry_ok": True,
        "error": "",
        "upper_stable_vertices": int(len(upper.vertices)),
        "lower_stable_vertices": int(len(lower.vertices)),
        "joint_vertical_gap_mm": float(common.vertical_gap_mm),
        "canonical_ap_x": float(canonical.ap[0]),
        "canonical_ap_y": float(canonical.ap[1]),
        "canonical_lr_x": float(canonical.lr[0]),
        "canonical_lr_y": float(canonical.lr[1]),
    }

    debug = {
        "version": "curve-geometry-v1",
        "patient_id": pid,
        "coordinate_policy": {
            "upper_lower_aligned_separately": False,
            "joint_frame": True,
            "ap_positive": "posterior_to_anterior",
            "lr_positive": "patient_right_to_patient_left",
            "vertical_positive": "lower_to_upper",
        },
        "canonical_frame": {
            "origin": canonical.origin.tolist(),
            "ap": canonical.ap.tolist(),
            "lr": canonical.lr.tolist(),
            "anterior_center": canonical.anterior_center.tolist(),
            "posterior_center": canonical.posterior_center.tolist(),
            "left_center": canonical.left_center.tolist(),
            "right_center": canonical.right_center.tolist(),
        },
        "spee": {},
        "wilson": {},
    }

    # ----------------------------- Spee -----------------------------
    spee_results = {}

    for side in ("left", "right"):
        try:
            result = analyze_spee_side(
                side,
                base,
                lo2d,
                lov,
                lower.labels,
                lower_center,
                canonical,
                args,
            )
        except Exception as e:
            result = {
                "valid": False,
                "error": repr(e),
                "landmarks": {},
            }

        spee_results[side] = result
        debug["spee"][side] = result

        row[f"spee_{side}_valid"] = bool(result.get("valid"))
        row[f"spee_{side}_error"] = result.get("error", "")

        if result.get("valid"):
            scalar = {
                k: v
                for k, v in result.items()
                if k != "landmarks"
            }
            flatten_scalar_dict(
                row,
                f"spee_{side}",
                scalar,
            )

        for tooth, lm in result.get("landmarks", {}).items():
            p = f"spee_{side}_tooth{tooth}"
            if "ap_mm" in lm:
                row[f"{p}_ap_mm"] = lm["ap_mm"]
                row[f"{p}_lr_mm"] = lm["lr_mm"]
                row[f"{p}_z_mm"] = lm["vertical"]
                row[f"{p}_vertical_std"] = lm["vertical_std"]
                row[f"{p}_landmark_points"] = lm["landmark_points"]
                row[f"{p}_valid"] = True
            else:
                row[f"{p}_valid"] = False

    left_valid = bool(spee_results["left"].get("valid"))
    right_valid = bool(spee_results["right"].get("valid"))
    row["spee_geometry_valid"] = bool(left_valid and right_valid)
    row["spee_geometry_any_valid"] = bool(left_valid or right_valid)

    # bilateral aggregates
    if left_valid or right_valid:
        valid_sides = [
            spee_results[s]
            for s in ("left", "right")
            if spee_results[s].get("valid")
        ]
        for metric in (
            "max_chord_depth_mm",
            "max_reverse_elevation_mm",
            "signed_dominant_depth_mm",
            "observed_max_depth_mm",
            "coef2_normalized",
            "fit_rmse_mm",
        ):
            vals = [x.get(metric) for x in valid_sides]
            s = summarize_values(vals)
            row[f"spee_bilateral_{metric}_mean"] = s["mean"]
            row[f"spee_bilateral_{metric}_max"] = s["max"]
            row[f"spee_bilateral_{metric}_min"] = s["min"]

        if left_valid and right_valid:
            for metric in (
                "max_chord_depth_mm",
                "signed_dominant_depth_mm",
                "coef2_normalized",
            ):
                row[f"spee_bilateral_{metric}_asymmetry"] = float(
                    abs(
                        spee_results["left"][metric]
                        - spee_results["right"][metric]
                    )
                )

    # ----------------------------- Wilson ---------------------------
    lower_w = analyze_wilson_arch(
        "lower",
        LOWER_WILSON_PAIRS,
        lo2d,
        lov,
        lower.labels,
        lower_center,
        canonical,
        args,
    )
    upper_w = analyze_wilson_arch(
        "upper",
        UPPER_WILSON_PAIRS,
        up2d,
        upv,
        upper.labels,
        upper_center,
        canonical,
        args,
    )

    debug["wilson"]["lower"] = lower_w
    debug["wilson"]["upper"] = upper_w

    for arch_name, result in (
        ("lower", lower_w),
        ("upper", upper_w),
    ):
        flatten_scalar_dict(
            row,
            f"wilson_{arch_name}",
            result["aggregates"],
        )

        for tooth, d in result["teeth"].items():
            p = f"wilson_{arch_name}_tooth{tooth}"
            flatten_scalar_dict(row, p, d)

        for pair_key, d in result["pairs"].items():
            p = f"wilson_{arch_name}_pair{pair_key}"
            flatten_scalar_dict(row, p, d)

    row["wilson_geometry_valid"] = bool(
        lower_w["aggregates"]["geometry_valid"]
    )
    row["wilson_geometry_any_valid"] = bool(
        lower_w["aggregates"]["geometry_any_valid"]
    )

    return row, debug


def compact_key_row(row):
    keep = {
        "patient_id",
        "geometry_ok",
        "spee_geometry_valid",
        "spee_geometry_any_valid",
        "spee_left_valid",
        "spee_right_valid",
        "wilson_geometry_valid",
        "wilson_geometry_any_valid",
        "wilson_lower_valid_teeth",
        "wilson_lower_valid_pairs",
        "wilson_upper_valid_teeth",
        "wilson_upper_valid_pairs",
    }

    prefixes = (
        "spee_bilateral_",
        "spee_left_max_chord_depth_mm",
        "spee_right_max_chord_depth_mm",
        "spee_left_signed_dominant_depth_mm",
        "spee_right_signed_dominant_depth_mm",
        "wilson_lower_tooth_inclination_",
        "wilson_lower_pair_chord_depression_",
        "wilson_lower_pair_abs_inclination_",
        "wilson_lower_pair_inclination_asymmetry_",
        "wilson_lower_posterior_progression_",
        "wilson_upper_tooth_inclination_",
        "wilson_upper_pair_chord_depression_",
        "wilson_upper_pair_abs_inclination_",
    )

    out = {}
    for k, v in row.items():
        if k in keep or any(k.startswith(p) for p in prefixes):
            out[k] = v
    return out


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def list_patient_ids(raw_root):
    return sorted(
        p.name
        for p in raw_root.glob("F*")
        if p.is_dir()
    )


def main():
    args = parse_args()

    args.raw_root = args.raw_root.expanduser().resolve()
    args.label_root = args.label_root.expanduser().resolve()
    args.geometry_helper = args.geometry_helper.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.debug_dir = args.debug_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    base = load_helper(args.geometry_helper)

    patient_ids = (
        list(dict.fromkeys(args.patient_id))
        if args.patient_id
        else list_patient_ids(args.raw_root)
    )

    print("=" * 104)
    print("Curve geometry extraction v1")
    print("=" * 104)
    print(f"patients         : {len(patient_ids)}")
    print(f"raw root         : {args.raw_root}")
    print(f"label root       : {args.label_root}")
    print(f"shared helper    : {args.geometry_helper}")
    print("Spee             : lower bilateral robust AP-V curve")
    print("Wilson primary   : lower posterior cusp/inclination geometry")
    print("Wilson secondary : upper posterior geometry")
    print("=" * 104)

    rows = []
    failures = []

    for i, pid in enumerate(patient_ids, 1):
        try:
            row, debug = analyze_patient(
                pid,
                args,
                base,
            )

            if (
                row.get("spee_geometry_valid")
                and row.get("wilson_geometry_valid")
            ):
                status = "OK"
            elif (
                row.get("spee_geometry_any_valid")
                or row.get("wilson_geometry_any_valid")
            ):
                status = "PARTIAL"
            else:
                status = "REVIEW"

            if args.save_debug:
                path = (
                    args.debug_dir
                    / f"{pid}_curve_geometry_v1.json"
                )
                path.write_text(
                    json.dumps(
                        debug,
                        ensure_ascii=False,
                        indent=2,
                        default=lambda x: (
                            x.tolist()
                            if isinstance(x, np.ndarray)
                            else float(x)
                            if isinstance(x, np.floating)
                            else int(x)
                            if isinstance(x, np.integer)
                            else str(x)
                        ),
                    )
                    + "\n",
                    encoding="utf-8",
                )

        except Exception as e:
            row = {
                "patient_id": pid,
                "geometry_ok": False,
                "spee_geometry_valid": False,
                "spee_geometry_any_valid": False,
                "wilson_geometry_valid": False,
                "wilson_geometry_any_valid": False,
                "error": repr(e),
            }
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
                f"{pid}: {status}"
            )

    features_path = args.output_dir / "curve_geometry_features_v1.csv"
    key_path = args.output_dir / "curve_geometry_key_v1.csv"
    summary_path = args.output_dir / "curve_geometry_summary_v1.json"
    failures_path = args.output_dir / "curve_geometry_failures_v1.json"

    write_csv(features_path, rows)
    write_csv(
        key_path,
        [compact_key_row(r) for r in rows],
    )

    summary = {
        "version": "curve-geometry-v1",
        "patients_requested": int(len(patient_ids)),
        "geometry_ok": int(
            sum(bool(r.get("geometry_ok")) for r in rows)
        ),
        "spee_geometry_valid": int(
            sum(bool(r.get("spee_geometry_valid")) for r in rows)
        ),
        "spee_geometry_any_valid": int(
            sum(bool(r.get("spee_geometry_any_valid")) for r in rows)
        ),
        "wilson_geometry_valid": int(
            sum(bool(r.get("wilson_geometry_valid")) for r in rows)
        ),
        "wilson_geometry_any_valid": int(
            sum(bool(r.get("wilson_geometry_any_valid")) for r in rows)
        ),
        "failures": failures,
        "parameters": {
            "minimum_tooth_points": int(args.minimum_tooth_points),
            "pca_points_per_tooth": int(args.pca_points_per_tooth),
            "surface_fraction": float(args.surface_fraction),
            "buccal_half_quantile": float(args.buccal_half_quantile),
            "lingual_half_quantile": float(args.lingual_half_quantile),
            "minimum_landmark_points": int(args.minimum_landmark_points),
        },
        "coordinate_policy": {
            "upper_lower_aligned_separately": False,
            "joint_occlusal_frame": True,
            "AP_positive": "posterior_to_anterior",
            "LR_positive": "patient_right_to_patient_left",
            "vertical_positive": "lower_to_upper",
        },
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

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(
        f"geometry OK          : "
        f"{summary['geometry_ok']}/{len(patient_ids)}"
    )
    print(
        f"Spee valid           : "
        f"{summary['spee_geometry_valid']}/{len(patient_ids)}"
    )
    print(
        f"Spee any valid       : "
        f"{summary['spee_geometry_any_valid']}/{len(patient_ids)}"
    )
    print(
        f"Wilson valid         : "
        f"{summary['wilson_geometry_valid']}/{len(patient_ids)}"
    )
    print(
        f"Wilson any valid     : "
        f"{summary['wilson_geometry_any_valid']}/{len(patient_ids)}"
    )
    print(f"features : {features_path}")
    print(f"key      : {key_path}")
    print(f"summary  : {summary_path}")
    print(f"failures : {failures_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
