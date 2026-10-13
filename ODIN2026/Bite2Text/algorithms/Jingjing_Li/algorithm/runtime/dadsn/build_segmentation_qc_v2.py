#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build revised FDI-aware segmentation QC from existing detailed QC CSVs.

Inputs:
- segmentation_qc_summary.csv
- segmentation_qc_tooth_metrics.csv

Changes from v1:
- Second-molar and wisdom-tooth absence is informational only.
- pass1 contamination and extreme ROI are informational only.
- Geometry availability depends only on the 12 crowding core teeth.
- Retry uses arch-specific combination rules.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARCH_CONFIG = {
    "upper": {
        "core": [26, 25, 24, 23, 22, 21, 11, 12, 13, 14, 15, 16],
        "second": [27, 17],
        "wisdom": [28, 18],
    },
    "lower": {
        "core": [36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46],
        "second": [37, 47],
        "wisdom": [38, 48],
    },
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def join_values(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "work/segmentation_qc/segmentation_qc_summary.csv"
        ),
    )
    parser.add_argument(
        "--tooth-metrics",
        type=Path,
        default=Path(
            "work/segmentation_qc/segmentation_qc_tooth_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/segmentation_qc_v2"),
    )
    parser.add_argument("--min-core-vertices", type=int, default=20)
    parser.add_argument("--review-core-vertices", type=int, default=80)
    parser.add_argument(
        "--review-component-fraction",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--severe-component-fraction",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--combined-component-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--review-bbox-mm", type=float, default=30.0)
    parser.add_argument("--combined-bbox-mm", type=float, default=38.0)
    parser.add_argument(
        "--upper-multiple-bbox-mm",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.35,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary.resolve(), dtype={"case": str})
    teeth = pd.read_csv(
        args.tooth_metrics.resolve(),
        dtype={"case": str, "arch": str, "tooth": int},
    )

    required_summary = {"case", "patient_id", "arch"}
    required_teeth = {
        "case",
        "arch",
        "tooth",
        "present",
        "vertex_count",
        "complete_face_count",
        "largest_component_fraction",
        "bbox_diag_mm",
        "mean_confidence",
    }

    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"Summary missing columns: {sorted(missing)}")

    missing = required_teeth - set(teeth.columns)
    if missing:
        raise ValueError(f"Tooth metrics missing columns: {sorted(missing)}")

    output_rows = []

    for _, case_row in summary.iterrows():
        case = str(case_row["case"])
        arch = str(case_row["arch"]).lower()

        if arch not in ARCH_CONFIG:
            continue

        config = ARCH_CONFIG[arch]
        case_teeth = teeth[teeth["case"] == case].copy()

        case_teeth["present_bool"] = case_teeth["present"].map(as_bool)
        for source, target in [
            ("vertex_count", "vertices"),
            ("complete_face_count", "faces"),
            ("largest_component_fraction", "fraction"),
            ("bbox_diag_mm", "bbox"),
            ("mean_confidence", "confidence"),
        ]:
            case_teeth[target] = pd.to_numeric(
                case_teeth[source],
                errors="coerce",
            )

        indexed = {
            int(row["tooth"]): row
            for _, row in case_teeth.iterrows()
        }

        missing_core = []
        low_vertex_core = []
        no_face_core = []
        review_low_vertex = []
        fragmented_review = []
        fragmented_severe = []
        oversized_review = []
        oversized_upper_multiple = []
        combined_severe = []
        low_confidence = []
        core_confidences = []

        for tooth in config["core"]:
            row = indexed.get(tooth)

            if row is None or not bool(row["present_bool"]):
                missing_core.append(tooth)
                continue

            vertices = int(finite_float(row["vertices"], 0.0))
            faces = int(finite_float(row["faces"], 0.0))
            fraction = finite_float(row["fraction"])
            bbox = finite_float(row["bbox"])
            confidence = finite_float(row["confidence"])

            if vertices < args.min_core_vertices:
                low_vertex_core.append(tooth)
            elif vertices < args.review_core_vertices:
                review_low_vertex.append(tooth)

            if faces <= 0:
                no_face_core.append(tooth)

            if np.isfinite(fraction) and fraction < args.review_component_fraction:
                fragmented_review.append(tooth)

            if np.isfinite(fraction) and fraction < args.severe_component_fraction:
                fragmented_severe.append(tooth)

            if np.isfinite(bbox) and bbox > args.review_bbox_mm:
                oversized_review.append(tooth)

            if (
                arch == "upper"
                and np.isfinite(bbox)
                and bbox > args.upper_multiple_bbox_mm
            ):
                oversized_upper_multiple.append(tooth)

            if (
                np.isfinite(bbox)
                and bbox > args.combined_bbox_mm
                and np.isfinite(fraction)
                and fraction < args.combined_component_fraction
            ):
                combined_severe.append(tooth)

            if (
                np.isfinite(confidence)
                and confidence < args.minimum_confidence
            ):
                low_confidence.append(tooth)

            if np.isfinite(confidence):
                core_confidences.append(confidence)

        missing_second = [
            tooth
            for tooth in config["second"]
            if tooth not in indexed or not bool(indexed[tooth]["present_bool"])
        ]
        missing_wisdom = [
            tooth
            for tooth in config["wisdom"]
            if tooth not in indexed or not bool(indexed[tooth]["present_bool"])
        ]

        hard_teeth = sorted(
            set(missing_core + low_vertex_core + no_face_core)
        )
        geometry_available = len(hard_teeth) == 0

        if arch == "upper":
            severe_arch_rule = (
                len(oversized_upper_multiple) >= 2
                or len(fragmented_severe) >= 2
                or len(combined_severe) >= 1
            )
        else:
            severe_arch_rule = (
                len(fragmented_severe) >= 2
                or len(combined_severe) >= 1
            )

        unexpected_text = str(
            case_row.get("unexpected_fdi_labels", "")
        ).strip()
        unexpected = unexpected_text not in {"", "nan", "None"}

        retry = (not geometry_available) or severe_arch_rule or unexpected

        critical = []
        soft = []
        info = []

        if missing_core:
            critical.append("core_missing:" + join_values(missing_core))
        if low_vertex_core:
            critical.append(
                "core_too_few_vertices:" + join_values(low_vertex_core)
            )
        if no_face_core:
            critical.append(
                "core_no_complete_faces:" + join_values(no_face_core)
            )
        if len(oversized_upper_multiple) >= 2:
            critical.append(
                "multiple_upper_core_bbox_gt_45:"
                + join_values(oversized_upper_multiple)
            )
        if len(fragmented_severe) >= 2:
            critical.append(
                "multiple_core_component_fraction_lt_040:"
                + join_values(fragmented_severe)
            )
        if combined_severe:
            critical.append(
                "core_bbox_gt_38_and_component_lt_050:"
                + join_values(combined_severe)
            )
        if unexpected:
            critical.append("unexpected_fdi_labels:" + unexpected_text)

        if review_low_vertex:
            soft.append(
                "core_low_vertex_count:" + join_values(review_low_vertex)
            )
        if fragmented_review:
            soft.append(
                "core_fragmented:" + join_values(fragmented_review)
            )
        if oversized_review:
            soft.append(
                "core_oversized:" + join_values(oversized_review)
            )
        if low_confidence:
            soft.append(
                "core_low_confidence:" + join_values(low_confidence)
            )

        if missing_second:
            info.append(
                "missing_second_molars:" + join_values(missing_second)
            )
        if missing_wisdom:
            info.append(
                "missing_wisdom_teeth:" + join_values(missing_wisdom)
            )
        if as_bool(case_row.get("contamination_detected", False)):
            info.append("pass1_contamination_detected")

        roi_ratio = finite_float(case_row.get("roi_ratio"))
        if np.isfinite(roi_ratio) and (roi_ratio < 0.15 or roi_ratio > 0.95):
            info.append(f"extreme_roi_ratio:{roi_ratio:.4f}")

        status = "retry_once" if retry else ("review" if soft else "acceptable")
        mean_confidence = (
            float(np.mean(core_confidences))
            if core_confidences
            else float("nan")
        )

        score = 100.0
        score -= min(60.0, 20.0 * len(hard_teeth))
        score -= min(30.0, 12.0 * len(combined_severe))
        if len(oversized_upper_multiple) >= 2:
            score -= 12.0
        if len(fragmented_severe) >= 2:
            score -= 12.0
        score -= min(10.0, 1.5 * len(soft))
        if np.isfinite(mean_confidence):
            score += (mean_confidence - 0.5) * 10.0
        score = float(np.clip(score, 0.0, 100.0))

        output_rows.append(
            {
                "case": case,
                "patient_id": str(case_row["patient_id"]),
                "arch": arch,
                "status_v2": status,
                "selection_score_v2": score,
                "retry_recommended_v2": int(retry),
                "geometry_available_v2": int(geometry_available),
                "hard_failure_count": len(hard_teeth),
                "critical_reason_count": len(critical),
                "soft_reason_count": len(soft),
                "mean_core_confidence": mean_confidence,
                "missing_core_teeth": join_values(missing_core),
                "low_vertex_core_teeth": join_values(low_vertex_core),
                "no_face_core_teeth": join_values(no_face_core),
                "review_low_vertex_teeth": join_values(review_low_vertex),
                "fragmented_review_teeth": join_values(fragmented_review),
                "fragmented_severe_teeth": join_values(fragmented_severe),
                "oversized_review_teeth": join_values(oversized_review),
                "oversized_upper_multiple_teeth": join_values(
                    oversized_upper_multiple
                ),
                "combined_severe_teeth": join_values(combined_severe),
                "low_confidence_core_teeth": join_values(low_confidence),
                "missing_second_molars": join_values(missing_second),
                "missing_wisdom_teeth": join_values(missing_wisdom),
                "critical_reasons_v2": "|".join(critical),
                "soft_reasons_v2": "|".join(soft),
                "informational_flags_v2": "|".join(info),
                "original_qc_status": str(case_row.get("status", "")),
                "original_qc_score": finite_float(case_row.get("qc_score")),
                "input_stl": str(case_row.get("input_stl", "")),
                "json_path": str(case_row.get("json_path", "")),
                "mesh_path": str(case_row.get("mesh_path", "")),
                "npz_path": str(case_row.get("npz_path", "")),
            }
        )

    output = pd.DataFrame(output_rows)
    summary_path = output_dir / "segmentation_qc_v2_summary.csv"
    output.to_csv(summary_path, index=False, encoding="utf-8-sig")

    retry_df = output[output["retry_recommended_v2"] == 1].copy()
    retry_path = output_dir / "retry_cases_v2.txt"
    retry_path.write_text(
        "\n".join(retry_df["case"].astype(str))
        + ("\n" if len(retry_df) else ""),
        encoding="utf-8",
    )

    retry_df[
        [
            "case",
            "patient_id",
            "arch",
            "selection_score_v2",
            "geometry_available_v2",
            "critical_reasons_v2",
            "input_stl",
        ]
    ].to_csv(
        output_dir / "retry_manifest_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts = Counter(output["status_v2"])
    by_arch = (
        output.groupby(["arch", "status_v2"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    report = {
        "total_cases": int(len(output)),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "by_arch": {
            str(arch): {
                str(key): int(value)
                for key, value in values.items()
            }
            for arch, values in by_arch.items()
        },
        "geometry_available": int(output["geometry_available_v2"].sum()),
        "retry_cases": int(len(retry_df)),
        "informational_only": [
            "missing second molar",
            "missing wisdom tooth",
            "pass1 contamination",
            "extreme ROI ratio",
        ],
    }

    (output_dir / "segmentation_qc_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Segmentation QC v2")
    print("=" * 72)
    print(f"Cases: {len(output)}")
    print(f"Geometry available: {report['geometry_available']}")
    print(f"Retry: {len(retry_df)}")
    print(f"Status: {dict(counts)}")
    print(f"Summary: {summary_path}")
    print(f"Retry list: {retry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
