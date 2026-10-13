#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练集几何分布建模与测试时路由。

命令：
1. fit-profile
   从训练集可靠 crowding JSON 拟合 upper/lower 稳健分布范围。
2. route
   根据分割 QC、几何计算结果、固定保护阈值和训练分布 OOD，
   将每个牙弓路由到：
   - full_catboost
   - segmentation_fallback
   - visual_fallback

该脚本不会修改 crowding 数值，也不会用 0 填补失败特征。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


KEY_FEATURES = (
    "primary_crowding_mm",
    "primary_tooth_material_mm",
    "primary_arch_length_mm",
    "primary_fit_rmse_mm",
    "primary_max_residual_mm",
    "cross_crowding_mm",
    "cross_tooth_material_mm",
    "cross_arch_length_mm",
    "cross_fit_rmse_mm",
    "cross_max_residual_mm",
    "protocol_delta_mm",
    "legacy_crowding_mm",
    "same_width_polyline_crowding_mm",
    "tooth_width_mean_mm",
    "tooth_width_min_mm",
    "tooth_width_max_mm",
    "tooth_width_delta_abs_mean_mm",
    "tooth_rotation_mean_deg",
    "tooth_rotation_max_deg",
)

REQUIRED_FINITE = (
    "primary_crowding_mm",
    "primary_tooth_material_mm",
    "primary_arch_length_mm",
    "primary_fit_rmse_mm",
    "primary_max_residual_mm",
    "cross_crowding_mm",
    "cross_tooth_material_mm",
    "cross_arch_length_mm",
    "protocol_delta_mm",
)


def nested(data: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def extract_geometry_features(data: dict[str, Any]) -> dict[str, float]:
    primary = data.get("primary_result", {})
    cross = data.get("cross_check_result", {})
    quality = data.get("quality", {})

    teeth = data.get("teeth", [])
    widths = []
    width_deltas = []
    rotations = []

    if isinstance(teeth, list):
        for item in teeth:
            if not isinstance(item, dict):
                continue
            width = finite(item.get("mesiodistal_width_mm"))
            delta = finite(item.get("width_delta_mm"))
            rotation = finite(item.get("rotation_from_arch_deg"))
            if np.isfinite(width):
                widths.append(width)
            if np.isfinite(delta):
                width_deltas.append(abs(delta))
            if np.isfinite(rotation):
                rotations.append(rotation)

    def aggregate(values: list[float], method: str) -> float:
        if not values:
            return float("nan")
        array = np.asarray(values, dtype=np.float64)
        if method == "mean":
            return float(array.mean())
        if method == "min":
            return float(array.min())
        if method == "max":
            return float(array.max())
        raise ValueError(method)

    return {
        "primary_crowding_mm": finite(primary.get("crowding_mm")),
        "primary_tooth_material_mm": finite(
            primary.get("tooth_material_mm")
        ),
        "primary_arch_length_mm": finite(primary.get("arch_length_mm")),
        "primary_fit_rmse_mm": finite(
            nested(primary, "arch_curve", "fit_rmse_mm")
        ),
        "primary_max_residual_mm": finite(
            nested(primary, "arch_curve", "max_residual_mm")
        ),
        "cross_crowding_mm": finite(cross.get("crowding_mm")),
        "cross_tooth_material_mm": finite(
            cross.get("tooth_material_mm")
        ),
        "cross_arch_length_mm": finite(cross.get("arch_length_mm")),
        "cross_fit_rmse_mm": finite(
            nested(cross, "arch_curve", "fit_rmse_mm")
        ),
        "cross_max_residual_mm": finite(
            nested(cross, "arch_curve", "max_residual_mm")
        ),
        "protocol_delta_mm": finite(
            quality.get("protocol_crowding_delta_mm")
        ),
        "legacy_crowding_mm": finite(
            nested(
                primary,
                "legacy_method_comparison",
                "crowding_mm",
            )
        ),
        "same_width_polyline_crowding_mm": finite(
            nested(
                primary,
                "same_width_polyline_comparison",
                "crowding_mm",
            )
        ),
        "tooth_width_mean_mm": aggregate(widths, "mean"),
        "tooth_width_min_mm": aggregate(widths, "min"),
        "tooth_width_max_mm": aggregate(widths, "max"),
        "tooth_width_delta_abs_mean_mm": aggregate(
            width_deltas,
            "mean",
        ),
        "tooth_rotation_mean_deg": aggregate(rotations, "mean"),
        "tooth_rotation_max_deg": aggregate(rotations, "max"),
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("crowding JSON 顶层必须是对象")
    return value


def find_case_json(
    case_results_dir: Optional[Path],
    patient_id: str,
    arch: str,
    success_output: str = "",
) -> Optional[Path]:
    if success_output:
        candidate = Path(success_output)
        if candidate.is_file():
            return candidate

    if case_results_dir is None:
        return None

    candidate = (
        case_results_dir
        / patient_id
        / f"{arch}_crowding.json"
    )
    return candidate if candidate.is_file() else None


def robust_feature_profile(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    median = float(np.median(values))
    q005, q01, q99, q995 = np.quantile(
        values,
        [0.005, 0.01, 0.99, 0.995],
    )
    mad = float(np.median(np.abs(values - median)))
    robust_scale = max(1.4826 * mad, 1e-6)
    broad_span = max(float(q995 - q005), robust_scale)

    return {
        "count": int(len(values)),
        "median": median,
        "mad_scale": robust_scale,
        "q005": float(q005),
        "q01": float(q01),
        "q99": float(q99),
        "q995": float(q995),
        "lower_bound": float(q005 - 0.25 * broad_span),
        "upper_bound": float(q995 + 0.25 * broad_span),
    }


def load_success_map(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    dataframe = pd.read_csv(path, dtype={"case": str})
    return {
        str(row["case"]): row.to_dict()
        for _, row in dataframe.iterrows()
    }


def load_failure_map(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    dataframe = pd.read_csv(path, dtype={"case": str})
    return {
        str(row["case"]): row.to_dict()
        for _, row in dataframe.iterrows()
    }


def fit_profile(args: argparse.Namespace) -> int:
    qc = pd.read_csv(
        args.selected_qc.resolve(),
        dtype={"case": str},
    )
    success_map = load_success_map(
        args.crowding_success.resolve()
        if args.crowding_success
        else None
    )
    case_results = (
        args.case_results_dir.resolve()
        if args.case_results_dir
        else None
    )

    per_arch_rows: dict[str, list[dict[str, float]]] = {
        "upper": [],
        "lower": [],
    }
    skipped = {}

    for _, row in qc.iterrows():
        case = str(row["case"])
        arch = str(row["arch"]).lower()

        if arch not in per_arch_rows:
            continue

        retry = int(finite(row.get("retry_recommended_v2"), 1.0))
        geometry_available = int(
            finite(row.get("geometry_available_v2"), 0.0)
        )

        if retry != 0 or geometry_available != 1:
            skipped[case] = "segmentation_qc_not_reliable"
            continue

        success = success_map.get(case, {})
        output_text = str(success.get("output", ""))
        geometry_path = find_case_json(
            case_results,
            str(row["patient_id"]),
            arch,
            output_text,
        )

        if geometry_path is None:
            skipped[case] = "geometry_json_missing"
            continue

        try:
            features = extract_geometry_features(read_json(geometry_path))
        except Exception as exc:
            skipped[case] = f"geometry_read_error:{type(exc).__name__}"
            continue

        crowding = features["primary_crowding_mm"]
        if not np.isfinite(crowding) or abs(crowding) > args.max_abs_crowding:
            skipped[case] = "primary_crowding_hard_outlier"
            continue

        if not all(np.isfinite(features[key]) for key in REQUIRED_FINITE):
            skipped[case] = "required_feature_nonfinite"
            continue

        if (
            features["primary_tooth_material_mm"] <= 0
            or features["primary_arch_length_mm"] <= 0
            or features["cross_tooth_material_mm"] <= 0
            or features["cross_arch_length_mm"] <= 0
        ):
            skipped[case] = "nonpositive_length"
            continue

        per_arch_rows[arch].append(features)

    profile: dict[str, Any] = {
        "version": 1,
        "max_abs_crowding_mm": float(args.max_abs_crowding),
        "max_ood_features": int(args.max_ood_features),
        "robust_z_limit": float(args.robust_z_limit),
        "arches": {},
        "fit_counts": {},
        "skipped_count": int(len(skipped)),
    }

    for arch, rows in per_arch_rows.items():
        frame = pd.DataFrame(rows)
        feature_profiles = {}

        for feature in KEY_FEATURES:
            if feature not in frame.columns:
                continue
            values = pd.to_numeric(
                frame[feature],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]

            if len(values) < args.min_feature_count:
                continue

            feature_profiles[feature] = robust_feature_profile(values)

        profile["arches"][arch] = feature_profiles
        profile["fit_counts"][arch] = int(len(frame))

    output_path = args.output_profile.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Geometry profile fitted")
    print("=" * 72)
    print(f"Upper reliable cases: {profile['fit_counts'].get('upper', 0)}")
    print(f"Lower reliable cases: {profile['fit_counts'].get('lower', 0)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Profile: {output_path}")
    return 0


def geometry_ood_reasons(
    features: dict[str, float],
    feature_profile: dict[str, dict[str, float]],
    robust_z_limit: float,
) -> tuple[list[str], int]:
    reasons = []
    ood_count = 0

    for feature, rule in feature_profile.items():
        value = features.get(feature, float("nan"))
        if not np.isfinite(value):
            continue

        lower = float(rule["lower_bound"])
        upper = float(rule["upper_bound"])
        median = float(rule["median"])
        scale = max(float(rule["mad_scale"]), 1e-6)
        robust_z = abs(value - median) / scale

        if value < lower or value > upper or robust_z > robust_z_limit:
            ood_count += 1
            reasons.append(
                f"ood:{feature}={value:.4f}"
            )

    return reasons, ood_count


def severe_segmentation(row: pd.Series) -> tuple[bool, list[str]]:
    reasons = []
    critical_text = str(row.get("critical_reasons_v2", ""))
    score = finite(row.get("selection_score_v2"), 0.0)
    hard_count = int(finite(row.get("hard_failure_count"), 99.0))
    critical_count = int(
        finite(row.get("critical_reason_count"), 99.0)
    )

    if "unexpected_fdi_labels" in critical_text:
        reasons.append("unexpected_fdi_labels")
    if "core_bbox_gt_38_and_component_lt_050" in critical_text:
        reasons.append("severe_label_expansion")
    if "multiple_core_component_fraction_lt_040" in critical_text:
        reasons.append("multiple_severe_fragmentation")
    if hard_count >= 2:
        reasons.append("multiple_core_hard_failures")
    if critical_count >= 3:
        reasons.append("many_critical_segmentation_flags")
    if score < 45.0:
        reasons.append("very_low_segmentation_score")

    return len(reasons) > 0, reasons


def route_cases(args: argparse.Namespace) -> int:
    profile = json.loads(
        args.profile.resolve().read_text(encoding="utf-8")
    )
    qc = pd.read_csv(
        args.selected_qc.resolve(),
        dtype={"case": str},
    )
    success_map = load_success_map(
        args.crowding_success.resolve()
        if args.crowding_success
        else None
    )
    failure_map = load_failure_map(
        args.crowding_failures.resolve()
        if args.crowding_failures
        else None
    )
    case_results = (
        args.case_results_dir.resolve()
        if args.case_results_dir
        else None
    )

    rows = []

    for _, qc_row in qc.iterrows():
        case = str(qc_row["case"])
        patient_id = str(qc_row["patient_id"])
        arch = str(qc_row["arch"]).lower()
        success = success_map.get(case, {})
        failure = failure_map.get(case, {})

        geometry_path = find_case_json(
            case_results,
            patient_id,
            arch,
            str(success.get("output", "")),
        )

        geometry_reasons = []
        features = {key: float("nan") for key in KEY_FEATURES}
        ood_reasons = []
        ood_count = 0
        geometry_json_valid = False

        if geometry_path is None:
            geometry_reasons.append("geometry_json_missing")
            if failure:
                error = str(failure.get("error", ""))
                if error:
                    geometry_reasons.append("crowding_failure:" + error)
        else:
            try:
                data = read_json(geometry_path)
                features = extract_geometry_features(data)
                geometry_json_valid = True
            except Exception as exc:
                geometry_reasons.append(
                    f"geometry_read_error:{type(exc).__name__}"
                )

        if geometry_json_valid:
            nonfinite = [
                key
                for key in REQUIRED_FINITE
                if not np.isfinite(features.get(key, np.nan))
            ]
            if nonfinite:
                geometry_reasons.append(
                    "required_nonfinite:" + ",".join(nonfinite)
                )

            crowding = features["primary_crowding_mm"]
            if np.isfinite(crowding) and abs(crowding) > float(
                profile["max_abs_crowding_mm"]
            ):
                geometry_reasons.append(
                    f"abs_crowding_gt_{profile['max_abs_crowding_mm']}:"
                    f"{crowding:.4f}"
                )

            for key in (
                "primary_tooth_material_mm",
                "primary_arch_length_mm",
                "cross_tooth_material_mm",
                "cross_arch_length_mm",
            ):
                value = features[key]
                if np.isfinite(value) and value <= 0:
                    geometry_reasons.append(f"nonpositive:{key}")

            feature_profile = profile.get("arches", {}).get(arch, {})
            ood_reasons, ood_count = geometry_ood_reasons(
                features,
                feature_profile,
                float(profile["robust_z_limit"]),
            )

            if ood_count > int(profile["max_ood_features"]):
                geometry_reasons.append(
                    f"too_many_ood_features:{ood_count}"
                )

        qc_retry = int(
            finite(qc_row.get("retry_recommended_v2"), 1.0)
        )
        qc_geometry = int(
            finite(qc_row.get("geometry_available_v2"), 0.0)
        )
        severe, severe_reasons = severe_segmentation(qc_row)

        full_eligible = (
            qc_retry == 0
            and qc_geometry == 1
            and geometry_json_valid
            and len(geometry_reasons) == 0
        )

        if full_eligible:
            route = "full_catboost"
            route_reasons = ["segmentation_and_geometry_pass"]
        elif not severe:
            route = "segmentation_fallback"
            route_reasons = (
                ["full_geometry_rejected"]
                + geometry_reasons
                + ood_reasons
            )
        else:
            route = "visual_fallback"
            route_reasons = (
                ["segmentation_severely_unreliable"]
                + severe_reasons
                + geometry_reasons
                + ood_reasons
            )

        output = {
            "case": case,
            "patient_id": patient_id,
            "arch": arch,
            "route": route,
            "full_model_eligible": int(full_eligible),
            "segmentation_fallback_eligible": int(
                route == "segmentation_fallback"
            ),
            "visual_fallback_required": int(
                route == "visual_fallback"
            ),
            "geometry_json_valid": int(geometry_json_valid),
            "geometry_ood_count": int(ood_count),
            "qc_status": str(qc_row.get("status_v2", "")),
            "qc_selection_score": finite(
                qc_row.get("selection_score_v2")
            ),
            "qc_retry_recommended": qc_retry,
            "qc_geometry_available": qc_geometry,
            "route_reasons": "|".join(route_reasons),
            "geometry_path": (
                str(geometry_path)
                if geometry_path is not None
                else ""
            ),
        }
        output.update(features)
        rows.append(output)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    routing = pd.DataFrame(rows)
    routing_path = output_dir / "routing_table.csv"
    routing.to_csv(
        routing_path,
        index=False,
        encoding="utf-8-sig",
    )

    for route in (
        "full_catboost",
        "segmentation_fallback",
        "visual_fallback",
    ):
        cases = routing.loc[routing["route"] == route, "case"].astype(str)
        (output_dir / f"{route}_cases.txt").write_text(
            "\n".join(cases) + ("\n" if len(cases) else ""),
            encoding="utf-8",
        )

    counts = routing["route"].value_counts().to_dict()
    summary = {
        "total": int(len(routing)),
        "route_counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "full_model_rate": float(
            (routing["route"] == "full_catboost").mean()
        ),
    }
    (output_dir / "routing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Geometry routing complete")
    print("=" * 72)
    print(f"Cases: {len(routing)}")
    print(f"Routes: {counts}")
    print(f"Routing table: {routing_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit-profile")
    fit.add_argument("--selected-qc", type=Path, required=True)
    fit.add_argument("--case-results-dir", type=Path)
    fit.add_argument("--crowding-success", type=Path)
    fit.add_argument("--output-profile", type=Path, required=True)
    fit.add_argument("--max-abs-crowding", type=float, default=30.0)
    fit.add_argument("--max-ood-features", type=int, default=3)
    fit.add_argument("--robust-z-limit", type=float, default=8.0)
    fit.add_argument("--min-feature-count", type=int, default=100)
    fit.set_defaults(handler=fit_profile)

    route = subparsers.add_parser("route")
    route.add_argument("--profile", type=Path, required=True)
    route.add_argument("--selected-qc", type=Path, required=True)
    route.add_argument("--case-results-dir", type=Path)
    route.add_argument("--crowding-success", type=Path)
    route.add_argument("--crowding-failures", type=Path)
    route.add_argument("--output-dir", type=Path, required=True)
    route.set_defaults(handler=route_cases)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
