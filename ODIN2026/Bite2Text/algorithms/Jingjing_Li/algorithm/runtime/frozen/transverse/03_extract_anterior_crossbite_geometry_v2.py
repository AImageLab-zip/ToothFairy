#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_extract_anterior_crossbite_geometry.py

Anterior crossbite geometry extractor v2.

This replaces the previous anterior v1 implementation but keeps the user's
existing script naming convention:

    00_check_inputs.py
    01_extract_crossbite_geometry.py
    02_evaluate_crossbite_v3.py
    03_extract_anterior_crossbite_geometry.py

Main change from anterior v1
----------------------------
v1 used a neighboring-tooth secant as the PRIMARY local frame. That can become
unstable in crowded/rotated anterior teeth (e.g. a near-180-degree local turn),
and can produce implausible margins such as -20 mm.

v2 therefore computes TWO frames for every upper anterior tooth:

1) RADIAL frame -- PRIMARY
   outward = target upper tooth center - robust upper arch center
   tangent = perpendicular to outward

   This is deliberately robust to local anterior crowding / tooth-center order.

2) SECANT frame -- QC / comparison only
   tangent = nearest available upper-neighbor secant
   outward = perpendicular, sign oriented away from upper arch center

Both frames use the SAME registered upper/lower coordinates. Upper and lower
arches are NEVER aligned independently.

Target upper permanent anterior teeth
-------------------------------------
    13, 12, 11, 21, 22, 23

Lower reference neighborhood
----------------------------
    44, 43, 42, 41, 31, 32, 33, 34

No hard upper-lower tooth pairing is used.

Margin definition
-----------------
    margin = upper_labial_envelope - lower_labial_envelope

    margin > 0 : normal transverse/labiolingual tendency
    margin < 0 : anterior-crossbite tendency

Variants
--------
    full
    incisal

QC
--
Per tooth:
    frame_disagreement_deg
    secant_curvature_deg
    secant_unstable_flag
    radial_implausible_margin_flag
    secant_implausible_margin_flag

The implausible-margin threshold is QC only. It is NOT a classification
threshold and is NOT used to force the prediction.

Outputs
-------
results/
    anterior_crossbite_geometry_features_v2.csv
    anterior_crossbite_geometry_key_v2.csv
    anterior_crossbite_geometry_summary_v2.json
    anterior_crossbite_geometry_failures_v2.json

debug/anterior_profiles_v2/
    <patient_id>_anterior_crossbite_profile_v2.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_POSTERIOR_SCRIPT = ROOT / "01_extract_crossbite_geometry.py"
DEFAULT_RAW_ROOT = Path("data/raw_data")
DEFAULT_LABEL_ROOT = Path("work/output")
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_PROFILE_DIR = ROOT / "debug" / "anterior_profiles_v2"

ANTERIOR_TARGETS = (13, 12, 11, 21, 22, 23)

# First premolars provide context for canine secant frames.
UPPER_CONTEXT_SEQUENCE = (14, 13, 12, 11, 21, 22, 23, 24)

# Lower reference is spatially selected from this neighborhood, not hard-paired.
LOWER_CONTEXT = (44, 43, 42, 41, 31, 32, 33, 34)

RIGHT_ANTERIOR = (13, 12, 11)
LEFT_ANTERIOR = (21, 22, 23)


@dataclass(frozen=True)
class AnteriorToothFrame:
    tooth: int
    origin: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray
    source: str
    local_curvature_deg: float | None
    radial_alignment: float
    centers: dict[str, list[float]]


def resolve_posterior_script(path: Path | None) -> Path:
    """
    Resolve the user's posterior geometry helper.

    Exact current project filename is preferred:
        01_extract_crossbite_geometry.py
    """
    if path is not None:
        p = path.expanduser().resolve()
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"--posterior-script does not exist: {p}")

    here = Path(__file__).resolve().parent
    candidates = [
        here / "01_extract_crossbite_geometry.py",
        ROOT / "01_extract_crossbite_geometry.py",
        here / "01_extract_crossbite_geometry_v3.py",
        ROOT / "01_extract_crossbite_geometry_v3.py",
    ]

    for d in (here, ROOT):
        if not d.exists():
            continue
        for pattern in (
            "01_extract_crossbite_geometry*.py",
            "01*crossbite*geometry*.py",
        ):
            for p in sorted(d.glob(pattern)):
                if p.resolve() != Path(__file__).resolve():
                    candidates.append(p)

    seen = set()
    for p in candidates:
        rp = p.expanduser().resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists() and rp.is_file():
            return rp

    raise FileNotFoundError(
        "Could not find 01_extract_crossbite_geometry.py.\n"
        "Expected it in:\n"
        f"  {ROOT}/\n\n"
        "You can also pass:\n"
        "  --posterior-script /actual/path/01_extract_crossbite_geometry.py"
    )


def load_posterior_helpers(path: Path):
    """
    Reuse the posterior v3 implementation's stable STL/FDI/common-plane/profile
    helpers so anterior and posterior geometry remain coordinate-consistent.
    """
    module_name = "_crossbite_posterior_geometry_helpers"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    required = (
        "load_arch",
        "joint_occlusal_frame",
        "project",
        "tooth_centers",
        "occlusal_band_mask",
        "local_tooth_profile",
        "aggregate_local_teeth",
        "flatten",
        "UPPER_FDI",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"01 helper is missing required functions/objects: {missing}"
        )
    return module


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


def labels_mask(labels: np.ndarray, teeth: Iterable[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(tuple(teeth), dtype=np.int32))


def robust_upper_arch_center(
    base,
    up2d: np.ndarray,
    upper_labels: np.ndarray,
    min_points: int,
) -> np.ndarray:
    """
    Robust upper-arch center used ONLY to define the sign/direction of the
    anterior radial frame.

    We use permanent upper tooth centers when available, then coordinate-wise
    median. This makes the center resistant to a few displaced/crowded teeth.
    """
    centers = base.tooth_centers(
        up2d,
        upper_labels,
        sorted(base.UPPER_FDI),
        min_points,
    )
    if len(centers) < 6:
        raise ValueError(
            f"Too few upper tooth centers for robust arch center: {len(centers)}"
        )

    arr = np.vstack(list(centers.values()))
    return np.median(arr, axis=0)


def get_context_centers(
    base,
    up2d: np.ndarray,
    upper_labels: np.ndarray,
    min_points: int,
) -> dict[int, np.ndarray]:
    return base.tooth_centers(
        up2d,
        upper_labels,
        list(UPPER_CONTEXT_SEQUENCE),
        min_points,
    )


def build_radial_frame(
    target_tooth: int,
    centers: dict[int, np.ndarray],
    arch_center: np.ndarray,
) -> AnteriorToothFrame:
    """
    PRIMARY anterior frame.

    outward points directly from robust upper arch center to the target upper
    tooth center. tangent is perpendicular to outward.

    This does not rely on local anterior tooth ordering.
    """
    if target_tooth not in centers:
        raise ValueError(f"upper tooth {target_tooth} center unavailable")

    origin = np.asarray(centers[target_tooth], dtype=np.float64)
    outward = _unit(origin - arch_center)

    # Tangent sign is irrelevant for the envelope margin; it only reverses
    # slice order. Use a deterministic 90-degree rotation.
    tangent = _unit(
        np.asarray([outward[1], -outward[0]], dtype=np.float64)
    )

    return AnteriorToothFrame(
        tooth=int(target_tooth),
        origin=origin.copy(),
        tangent=tangent,
        outward=outward,
        source="radial_arch_center",
        local_curvature_deg=None,
        radial_alignment=1.0,
        centers={str(k): np.asarray(v).tolist() for k, v in centers.items()},
    )


def build_secant_frame(
    target_tooth: int,
    centers: dict[int, np.ndarray],
    arch_center: np.ndarray,
) -> AnteriorToothFrame:
    """
    SECONDARY/QC frame reproducing the v1 secant idea.

    It is retained so we can quantify disagreement with the radial frame and
    identify crowding/ordering artifacts.
    """
    seq = list(UPPER_CONTEXT_SEQUENCE)
    if target_tooth not in seq:
        raise ValueError(f"unsupported target tooth: {target_tooth}")
    if target_tooth not in centers:
        raise ValueError(f"upper tooth {target_tooth} center unavailable")

    idx = seq.index(target_tooth)
    target = np.asarray(centers[target_tooth], dtype=np.float64)

    prev_tooth = None
    next_tooth = None

    for j in range(idx - 1, -1, -1):
        t = seq[j]
        if t in centers:
            prev_tooth = t
            break

    for j in range(idx + 1, len(seq)):
        t = seq[j]
        if t in centers:
            next_tooth = t
            break

    curvature = None

    if prev_tooth is not None and next_tooth is not None:
        prev_c = np.asarray(centers[prev_tooth], dtype=np.float64)
        next_c = np.asarray(centers[next_tooth], dtype=np.float64)
        tangent = next_c - prev_c
        source = f"secant_{prev_tooth}_{next_tooth}"

        try:
            seg_in = _unit(target - prev_c)
            seg_out = _unit(next_c - target)
            curvature = _angle_deg(seg_in, seg_out)
        except Exception:
            curvature = None

    elif prev_tooth is not None:
        tangent = target - np.asarray(centers[prev_tooth], dtype=np.float64)
        source = f"single_prev_{prev_tooth}_{target_tooth}"

    elif next_tooth is not None:
        tangent = np.asarray(centers[next_tooth], dtype=np.float64) - target
        source = f"single_next_{target_tooth}_{next_tooth}"

    else:
        raise ValueError(
            f"upper tooth {target_tooth}: no neighboring center for secant"
        )

    tangent = _unit(tangent)
    outward = _unit(
        np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    )

    radial = _unit(target - arch_center)
    if float(np.dot(outward, radial)) < 0:
        outward = -outward

    radial_alignment = float(np.dot(outward, radial))

    return AnteriorToothFrame(
        tooth=int(target_tooth),
        origin=target.copy(),
        tangent=tangent,
        outward=outward,
        source=source,
        local_curvature_deg=curvature,
        radial_alignment=radial_alignment,
        centers={str(k): np.asarray(v).tolist() for k, v in centers.items()},
    )


def frame_features(
    radial: AnteriorToothFrame,
    secant: AnteriorToothFrame,
    curvature_warning_deg: float,
    disagreement_warning_deg: float,
) -> dict:
    disagreement = _angle_deg(radial.outward, secant.outward)

    curvature = secant.local_curvature_deg
    curvature_bad = bool(
        curvature is not None
        and np.isfinite(curvature)
        and curvature >= curvature_warning_deg
    )
    disagreement_bad = bool(
        np.isfinite(disagreement)
        and disagreement >= disagreement_warning_deg
    )

    return {
        "radial_tangent_x": float(radial.tangent[0]),
        "radial_tangent_y": float(radial.tangent[1]),
        "radial_outward_x": float(radial.outward[0]),
        "radial_outward_y": float(radial.outward[1]),
        "radial_source": radial.source,
        "secant_tangent_x": float(secant.tangent[0]),
        "secant_tangent_y": float(secant.tangent[1]),
        "secant_outward_x": float(secant.outward[0]),
        "secant_outward_y": float(secant.outward[1]),
        "secant_source": secant.source,
        "secant_curvature_deg": (
            None if curvature is None else float(curvature)
        ),
        "secant_radial_alignment": float(secant.radial_alignment),
        "frame_disagreement_deg": float(disagreement),
        "secant_curvature_warning": curvature_bad,
        "frame_disagreement_warning": disagreement_bad,
        "secant_unstable_flag": bool(curvature_bad or disagreement_bad),
    }


def run_local_profile(
    base,
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    frame: AnteriorToothFrame,
    args,
) -> tuple[dict, dict]:
    feat, profile = base.local_tooth_profile(
        upper_points,
        lower_points,
        frame,
        args.slice_step_mm,
        args.slice_half_width_mm,
        args.outer_quantile,
        args.minimum_slice_points,
        args.range_quantile,
        args.padding_mm,
    )
    feat["valid"] = True
    feat["error"] = ""
    return feat, profile


def implausible_margin_flag(
    feat: dict,
    threshold_mm: float,
) -> bool:
    """
    QC only. Do not use this to alter the margin or classification.

    Flag a profile if one of its robust/central margin summaries is physically
    implausibly large in magnitude.
    """
    vals = []
    for key in (
        "q10_margin_mm",
        "q20_margin_mm",
        "median_margin_mm",
        "mean_margin_mm",
        "min_margin_mm",
    ):
        value = feat.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(abs(value))

    return bool(vals and max(vals) > threshold_mm)


def aggregate_subset(
    base,
    results: dict[int, dict],
    teeth: Iterable[int],
) -> dict:
    subset = {
        int(t): results[int(t)]
        for t in teeth
        if int(t) in results
    }
    return base.aggregate_local_teeth(subset)


def flatten_aggregate_family(
    base,
    row: dict,
    frame_name: str,
    variant: str,
    results: dict[int, dict],
    debug: dict,
) -> None:
    all_agg = base.aggregate_local_teeth(results)
    right_agg = aggregate_subset(base, results, RIGHT_ANTERIOR)
    left_agg = aggregate_subset(base, results, LEFT_ANTERIOR)

    base.flatten(
        row,
        f"anterior_{frame_name}_all_{variant}",
        all_agg,
    )
    base.flatten(
        row,
        f"anterior_{frame_name}_right_{variant}",
        right_agg,
    )
    base.flatten(
        row,
        f"anterior_{frame_name}_left_{variant}",
        left_agg,
    )

    debug["aggregates"][frame_name][variant] = {
        "all": all_agg,
        "right": right_agg,
        "left": left_agg,
    }


def copy_radial_primary_aliases(row: dict) -> None:
    """
    Explicitly expose radial as the PRIMARY anterior branch while retaining
    fully named radial/secant columns.

    This makes future 04 fusion code simpler:
        anterior_primary_* == radial features
    """
    additions = {}
    for key, value in row.items():
        prefix = "anterior_radial_"
        if key.startswith(prefix):
            additions[
                "anterior_primary_" + key[len(prefix):]
            ] = value
    row.update(additions)


def compact_key_row(row: dict) -> dict:
    out = {
        "patient_id": row.get("patient_id"),
        "geometry_ok": row.get("geometry_ok"),
        "anterior_geometry_valid": row.get("anterior_geometry_valid"),
        "anterior_any_geometry_valid": row.get(
            "anterior_any_geometry_valid"
        ),
        "anterior_radial_valid": row.get("anterior_radial_valid"),
        "anterior_secant_valid": row.get("anterior_secant_valid"),
        "anterior_qc_warning_count": row.get(
            "anterior_qc_warning_count"
        ),
        "vertical_gap_mm": row.get("vertical_gap_mm"),
    }

    # Aggregate comparison.
    for frame_name in ("radial", "secant"):
        for variant in ("full", "incisal"):
            p = f"anterior_{frame_name}_all_{variant}"
            for suffix in (
                "valid_teeth",
                "worst_q20_margin_mm",
                "second_worst_q20_margin_mm",
                "worst_two_q20_mean_mm",
                "worst_q20_tooth",
                "worst_median_margin_mm",
                "second_worst_median_margin_mm",
                "max_negative_ratio",
                "q20_below_0_count",
                "q20_below_05_count",
                "median_below_0_count",
            ):
                out[f"{p}_{suffix}"] = row.get(f"{p}_{suffix}")

    # Tooth-level radial vs secant.
    for tooth in ANTERIOR_TARGETS:
        fp = f"anterior_frame_tooth{tooth}"
        for suffix in (
            "secant_curvature_deg",
            "frame_disagreement_deg",
            "secant_radial_alignment",
            "secant_curvature_warning",
            "frame_disagreement_warning",
            "secant_unstable_flag",
        ):
            out[f"{fp}_{suffix}"] = row.get(f"{fp}_{suffix}")

        for frame_name in ("radial", "secant"):
            for variant in ("full", "incisal"):
                p = (
                    f"anterior_{frame_name}_tooth"
                    f"{tooth}_{variant}"
                )
                for suffix in (
                    "q20_margin_mm",
                    "median_margin_mm",
                    "negative_ratio",
                    "negative_05_ratio",
                    "longest_negative_run_mm",
                    "valid",
                    "implausible_margin_flag",
                ):
                    out[f"{p}_{suffix}"] = row.get(f"{p}_{suffix}")

    return out


def analyze_patient(
    patient_id: str,
    args,
    base,
) -> tuple[dict, dict]:
    upper = base.load_arch(
        patient_id,
        "upper",
        args.raw_root,
        args.label_root,
    )
    lower = base.load_arch(
        patient_id,
        "lower",
        args.raw_root,
        args.label_root,
    )

    # ONE shared coordinate system for both registered arches.
    common = base.joint_occlusal_frame(
        upper,
        lower,
        args.minimum_tooth_points,
        args.pca_points_per_tooth,
    )
    up2d, up_vertical = base.project(upper, common)
    lo2d, lo_vertical = base.project(lower, common)

    arch_center = robust_upper_arch_center(
        base,
        up2d,
        upper.labels,
        args.minimum_tooth_points,
    )
    centers = get_context_centers(
        base,
        up2d,
        upper.labels,
        args.minimum_tooth_points,
    )

    row = {
        "patient_id": patient_id,
        "geometry_ok": True,
        "error": "",
        "upper_stable_vertices": int(len(upper.vertices)),
        "lower_stable_vertices": int(len(lower.vertices)),
        "upper_tooth_vertices": int(np.count_nonzero(upper.labels > 0)),
        "lower_tooth_vertices": int(np.count_nonzero(lower.labels > 0)),
        "common_eigenvalue_0": float(common.eigenvalues[0]),
        "common_eigenvalue_1": float(common.eigenvalues[1]),
        "common_eigenvalue_2": float(common.eigenvalues[2]),
        "vertical_gap_mm": float(common.vertical_gap_mm),
        "upper_arch_center_x": float(arch_center[0]),
        "upper_arch_center_y": float(arch_center[1]),
        "anterior_primary_frame": "radial",
    }

    debug = {
        "version": "anterior-crossbite-geometry-v2-radial-primary",
        "patient_id": patient_id,
        "primary_frame": "radial",
        "margin_definition": (
            "upper_labial_envelope - lower_labial_envelope; "
            "negative means anterior-crossbite tendency"
        ),
        "upper_lower_aligned_separately": False,
        "hard_tooth_pairing": False,
        "upper_targets": list(ANTERIOR_TARGETS),
        "upper_context_sequence": list(UPPER_CONTEXT_SEQUENCE),
        "lower_reference_context": list(LOWER_CONTEXT),
        "common_frame": {
            "origin": common.origin.tolist(),
            "basis": common.basis.tolist(),
            "vertical": common.vertical.tolist(),
            "eigenvalues": common.eigenvalues.tolist(),
            "vertical_gap_mm": float(common.vertical_gap_mm),
        },
        "upper_arch_center": arch_center.tolist(),
        "teeth": {},
        "aggregates": {
            "radial": {},
            "secant": {},
        },
    }

    lower_full_mask = labels_mask(lower.labels, LOWER_CONTEXT)
    lower_inc_mask = (
        lower_full_mask
        & base.occlusal_band_mask(
            lo_vertical,
            lower.labels,
            LOWER_CONTEXT,
            "lower",
            args.incisal_fraction,
            args.minimum_tooth_points,
        )
    )

    results = {
        "radial": {
            "full": {},
            "incisal": {},
        },
        "secant": {
            "full": {},
            "incisal": {},
        },
    }

    qc_warning_count = 0

    for tooth in ANTERIOR_TARGETS:
        full_mask = upper.labels == tooth
        tooth_count = int(np.count_nonzero(full_mask))

        if tooth_count < args.minimum_tooth_points:
            error = (
                f"upper tooth {tooth} points insufficient: "
                f"{tooth_count} < {args.minimum_tooth_points}"
            )
            for frame_name in ("radial", "secant"):
                for variant in ("full", "incisal"):
                    feat = {
                        "valid": False,
                        "error": error,
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": tooth_count,
                        "implausible_margin_flag": False,
                    }
                    results[frame_name][variant][tooth] = feat
                    base.flatten(
                        row,
                        f"anterior_{frame_name}_tooth{tooth}_{variant}",
                        feat,
                    )

            debug["teeth"][str(tooth)] = {"error": error}
            continue

        try:
            radial = build_radial_frame(
                tooth,
                centers,
                arch_center,
            )
            secant = build_secant_frame(
                tooth,
                centers,
                arch_center,
            )
            ff = frame_features(
                radial,
                secant,
                args.secant_curvature_warning_deg,
                args.frame_disagreement_warning_deg,
            )
            base.flatten(
                row,
                f"anterior_frame_tooth{tooth}",
                ff,
            )

            if ff["secant_unstable_flag"]:
                qc_warning_count += 1

        except Exception as e:
            error = repr(e)
            for frame_name in ("radial", "secant"):
                for variant in ("full", "incisal"):
                    feat = {
                        "valid": False,
                        "error": error,
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": tooth_count,
                        "implausible_margin_flag": False,
                    }
                    results[frame_name][variant][tooth] = feat
                    base.flatten(
                        row,
                        f"anterior_{frame_name}_tooth{tooth}_{variant}",
                        feat,
                    )

            debug["teeth"][str(tooth)] = {"error": error}
            continue

        upper_inc_mask = (
            full_mask
            & base.occlusal_band_mask(
                up_vertical,
                upper.labels,
                [tooth],
                "upper",
                args.incisal_fraction,
                args.minimum_tooth_points,
            )
        )

        variants = {
            "full": (
                up2d[full_mask],
                lo2d[lower_full_mask],
            ),
            "incisal": (
                up2d[upper_inc_mask],
                lo2d[lower_inc_mask],
            ),
        }

        tooth_debug = {
            "frames": {
                "radial": {
                    "origin": radial.origin.tolist(),
                    "tangent": radial.tangent.tolist(),
                    "outward": radial.outward.tolist(),
                    "source": radial.source,
                },
                "secant": {
                    "origin": secant.origin.tolist(),
                    "tangent": secant.tangent.tolist(),
                    "outward": secant.outward.tolist(),
                    "source": secant.source,
                    "local_curvature_deg": secant.local_curvature_deg,
                    "radial_alignment": secant.radial_alignment,
                },
                "comparison": ff,
            },
            "variants": {
                "radial": {},
                "secant": {},
            },
        }

        for frame_name, frame in (
            ("radial", radial),
            ("secant", secant),
        ):
            for variant_name, (upper_points, lower_points) in variants.items():
                try:
                    feat, profile = run_local_profile(
                        base,
                        upper_points,
                        lower_points,
                        frame,
                        args,
                    )
                    bad_margin = implausible_margin_flag(
                        feat,
                        args.implausible_margin_mm,
                    )
                    feat["implausible_margin_flag"] = bad_margin
                    if bad_margin:
                        qc_warning_count += 1

                except Exception as e:
                    feat = {
                        "valid": False,
                        "error": repr(e),
                        "n_slices": 0,
                        "coverage_mm": 0.0,
                        "upper_points": int(len(upper_points)),
                        "lower_points": int(len(lower_points)),
                        "implausible_margin_flag": False,
                    }
                    profile = {"error": repr(e)}

                results[frame_name][variant_name][tooth] = feat
                base.flatten(
                    row,
                    f"anterior_{frame_name}_tooth{tooth}_{variant_name}",
                    feat,
                )

                tooth_debug["variants"][frame_name][variant_name] = {
                    "features": feat,
                    "profile": profile,
                }

        debug["teeth"][str(tooth)] = tooth_debug

    # Aggregate radial and secant independently.
    for frame_name in ("radial", "secant"):
        for variant in ("full", "incisal"):
            flatten_aggregate_family(
                base,
                row,
                frame_name,
                variant,
                results[frame_name][variant],
                debug,
            )

    # Radial is the primary anterior geometry.
    copy_radial_primary_aliases(row)

    radial_inc_valid = int(
        row.get("anterior_radial_all_incisal_valid_teeth") or 0
    )
    radial_right_valid = int(
        row.get("anterior_radial_right_incisal_valid_teeth") or 0
    )
    radial_left_valid = int(
        row.get("anterior_radial_left_incisal_valid_teeth") or 0
    )

    secant_inc_valid = int(
        row.get("anterior_secant_all_incisal_valid_teeth") or 0
    )
    secant_right_valid = int(
        row.get("anterior_secant_right_incisal_valid_teeth") or 0
    )
    secant_left_valid = int(
        row.get("anterior_secant_left_incisal_valid_teeth") or 0
    )

    row["anterior_radial_valid"] = bool(
        radial_inc_valid >= 4
        and radial_right_valid >= 1
        and radial_left_valid >= 1
    )
    row["anterior_secant_valid"] = bool(
        secant_inc_valid >= 4
        and secant_right_valid >= 1
        and secant_left_valid >= 1
    )

    # Main validity follows PRIMARY radial branch only.
    row["anterior_geometry_valid"] = row["anterior_radial_valid"]
    row["anterior_any_geometry_valid"] = bool(radial_inc_valid >= 2)
    row["anterior_qc_warning_count"] = int(qc_warning_count)
    row["anterior_qc_warning"] = bool(qc_warning_count > 0)

    return row, debug


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--posterior-script",
        type=Path,
        default=None,
        help=(
            "Shared geometry helper. Default auto-resolves "
            f"{ROOT}/"
            "01_extract_crossbite_geometry.py"
        ),
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )
    p.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    p.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
    )
    p.add_argument(
        "--patient-id",
        action="append",
        default=[],
    )

    p.add_argument(
        "--minimum-tooth-points",
        type=int,
        default=20,
    )
    p.add_argument(
        "--pca-points-per-tooth",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--incisal-fraction",
        type=float,
        default=0.40,
    )
    p.add_argument(
        "--slice-step-mm",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--slice-half-width-mm",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--padding-mm",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--outer-quantile",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--minimum-slice-points",
        type=int,
        default=10,
    )
    p.add_argument(
        "--range-quantile",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--secant-curvature-warning-deg",
        type=float,
        default=120.0,
        help=(
            "QC only. Secant local turn >= this angle is flagged unstable."
        ),
    )
    p.add_argument(
        "--frame-disagreement-warning-deg",
        type=float,
        default=35.0,
        help=(
            "QC only. Radial/secant outward disagreement >= this angle "
            "is flagged."
        ),
    )
    p.add_argument(
        "--implausible-margin-mm",
        type=float,
        default=10.0,
        help=(
            "QC only. Robust/profile margin magnitude above this value "
            "is flagged; values are not clipped or changed."
        ),
    )

    p.add_argument(
        "--save-profiles",
        action="store_true",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    args.posterior_script = resolve_posterior_script(args.posterior_script)
    args.raw_root = args.raw_root.expanduser().resolve()
    args.label_root = args.label_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.profile_dir = args.profile_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_profiles:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    base = load_posterior_helpers(args.posterior_script)

    if args.patient_id:
        patient_ids = list(dict.fromkeys(args.patient_id))
    else:
        patient_ids = sorted(
            p.name
            for p in args.raw_root.glob("F*")
            if p.is_dir()
        )

    print("=" * 96)
    print("Anterior crossbite geometry extraction v2")
    print("PRIMARY = RADIAL frame | SECANT = QC/comparison")
    print("=" * 96)
    print(f"patients          : {len(patient_ids)}")
    print(f"raw root          : {args.raw_root}")
    print(f"label root        : {args.label_root}")
    print(f"01 helper         : {args.posterior_script}")
    print(f"upper targets     : {list(ANTERIOR_TARGETS)}")
    print(f"lower context     : {list(LOWER_CONTEXT)}")
    print(
        "margin            : upper labial - lower labial; "
        "negative = anterior-crossbite tendency"
    )
    print(
        "QC                : curvature>="
        f"{args.secant_curvature_warning_deg:g}°, disagreement>="
        f"{args.frame_disagreement_warning_deg:g}°, |margin|>"
        f"{args.implausible_margin_mm:g} mm"
    )
    print("=" * 96)

    rows = []
    failures = []

    for i, patient_id in enumerate(patient_ids, 1):
        try:
            row, debug = analyze_patient(
                patient_id,
                args,
                base,
            )

            if row.get("anterior_geometry_valid"):
                status = "OK"
            elif row.get("anterior_any_geometry_valid"):
                status = "PARTIAL"
            else:
                status = "REVIEW"

            if args.save_profiles:
                profile_path = (
                    args.profile_dir
                    / f"{patient_id}_anterior_crossbite_profile_v2.json"
                )
                profile_path.write_text(
                    json.dumps(
                        debug,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        except Exception as e:
            row = {
                "patient_id": patient_id,
                "geometry_ok": False,
                "anterior_geometry_valid": False,
                "anterior_any_geometry_valid": False,
                "anterior_radial_valid": False,
                "anterior_secant_valid": False,
                "anterior_qc_warning": True,
                "error": repr(e),
            }
            failures.append(
                {
                    "patient_id": patient_id,
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
            qc = row.get("anterior_qc_warning_count", "")
            print(
                f"[{i:4d}/{len(patient_ids):4d}] "
                f"{patient_id}: {status}"
                + (f"  qc={qc}" if qc != "" else "")
            )

    features_path = (
        args.output_dir
        / "anterior_crossbite_geometry_features_v2.csv"
    )
    key_path = (
        args.output_dir
        / "anterior_crossbite_geometry_key_v2.csv"
    )
    summary_path = (
        args.output_dir
        / "anterior_crossbite_geometry_summary_v2.json"
    )
    failures_path = (
        args.output_dir
        / "anterior_crossbite_geometry_failures_v2.json"
    )

    write_csv(features_path, rows)
    write_csv(
        key_path,
        [compact_key_row(row) for row in rows],
    )

    summary = {
        "version": "anterior-crossbite-geometry-v2-radial-primary",
        "primary_frame": "radial",
        "secondary_frame": "secant_qc",
        "patients_requested": int(len(patient_ids)),
        "base_geometry_ok": int(
            sum(bool(r.get("geometry_ok")) for r in rows)
        ),
        "anterior_geometry_valid": int(
            sum(bool(r.get("anterior_geometry_valid")) for r in rows)
        ),
        "anterior_any_geometry_valid": int(
            sum(bool(r.get("anterior_any_geometry_valid")) for r in rows)
        ),
        "anterior_radial_valid": int(
            sum(bool(r.get("anterior_radial_valid")) for r in rows)
        ),
        "anterior_secant_valid": int(
            sum(bool(r.get("anterior_secant_valid")) for r in rows)
        ),
        "patients_with_qc_warning": int(
            sum(bool(r.get("anterior_qc_warning")) for r in rows)
        ),
        "parameters": {
            "upper_targets": list(ANTERIOR_TARGETS),
            "upper_context_sequence": list(UPPER_CONTEXT_SEQUENCE),
            "lower_context": list(LOWER_CONTEXT),
            "minimum_tooth_points": int(args.minimum_tooth_points),
            "pca_points_per_tooth": int(args.pca_points_per_tooth),
            "incisal_fraction": float(args.incisal_fraction),
            "slice_step_mm": float(args.slice_step_mm),
            "slice_half_width_mm": float(args.slice_half_width_mm),
            "padding_mm": float(args.padding_mm),
            "outer_quantile": float(args.outer_quantile),
            "minimum_slice_points": int(args.minimum_slice_points),
            "range_quantile": float(args.range_quantile),
            "secant_curvature_warning_deg": float(
                args.secant_curvature_warning_deg
            ),
            "frame_disagreement_warning_deg": float(
                args.frame_disagreement_warning_deg
            ),
            "implausible_margin_mm": float(args.implausible_margin_mm),
        },
        "margin_definition": (
            "upper_labial_envelope - lower_labial_envelope; "
            "negative means anterior-crossbite tendency"
        ),
        "hard_tooth_pairing": False,
        "upper_lower_aligned_separately": False,
        "failures": failures,
        "output_files": {
            "features": str(features_path),
            "key": str(key_path),
            "summary": str(summary_path),
            "failures": str(failures_path),
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
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"patients requested          : {len(patient_ids)}")
    print(
        "base geometry OK            : "
        f"{summary['base_geometry_ok']}/{len(patient_ids)}"
    )
    print(
        "radial primary valid        : "
        f"{summary['anterior_radial_valid']}/{len(patient_ids)}"
    )
    print(
        "secant comparison valid     : "
        f"{summary['anterior_secant_valid']}/{len(patient_ids)}"
    )
    print(
        "patients with QC warning    : "
        f"{summary['patients_with_qc_warning']}/{len(patient_ids)}"
    )
    print(f"features                    : {features_path}")
    print(f"key                         : {key_path}")
    print(f"summary                     : {summary_path}")
    print(f"failures                    : {failures_path}")
    print("=" * 96)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
