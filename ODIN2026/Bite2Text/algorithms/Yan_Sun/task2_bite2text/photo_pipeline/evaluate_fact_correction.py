#!/usr/bin/env python3
"""Cross-fit conservative sentence correction on retrieved orthodontic reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptv3_finetune"))
from evaluate_geometry_retrieval import meteor_lite, sentence_bleu4


CATEGORY_HEADS = {
    "transverse": ("crossbite",),
    "vertical": ("vertical_relation",),
    "sagittal": (
        "right_molar_relation",
        "right_canine_relation",
        "left_molar_relation",
        "left_canine_relation",
        "overjet",
    ),
    "midline": ("midline_relation",),
    "curves": ("curve_spee", "curve_wilson"),
    "crowding": ("upper_crowding", "lower_crowding"),
}


def sentences(report: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?])\s+", report.strip()) if value.strip()]


def sentence_categories(sentence: str) -> set[str]:
    value = sentence.lower()
    output: set[str] = set()
    if "crossbite" in value or "transvers" in value:
        output.add("transverse")
    if any(term in value for term in ("vertical", "overbite", "deep bite", "open bite")):
        output.add("vertical")
    if any(term in value for term in ("sagitt", "overjet", "molar", "canine")):
        output.add("sagittal")
    if "midline" in value:
        output.add("midline")
    if "curve of spee" in value or "curve of wilson" in value or "curves of" in value:
        output.add("curves")
    if any(term in value for term in ("crowding", "spacing", "spaces", "diastema")):
        output.add("crowding")
    return output


def replacement_sentences(structured_report: str) -> dict[str, list[str]]:
    output = {category: [] for category in CATEGORY_HEADS}
    for sentence in sentences(structured_report):
        for category in sentence_categories(sentence):
            output[category].append(sentence)
    return output


def category_confidence(query: dict[str, Any], category: str) -> float:
    values = [float(query["confidence"][head]) for head in CATEGORY_HEADS[category]]
    return float(np.mean(values))


def correct_report(
    report: str,
    query: dict[str, Any],
    thresholds: dict[str, float | None],
) -> str:
    replacements = replacement_sentences(query["structured_prediction"])
    enabled = {
        category
        for category, threshold in thresholds.items()
        if threshold is not None
        and replacements.get(category)
        and category_confidence(query, category) >= threshold
    }
    if not enabled:
        return report

    output: list[str] = []
    emitted: set[str] = set()
    for sentence in sentences(report):
        categories = sentence_categories(sentence)
        replaced = [category for category in CATEGORY_HEADS if category in categories & enabled]
        if not replaced:
            output.append(sentence)
            continue
        for category in replaced:
            if category not in emitted:
                output.extend(replacements[category])
                emitted.add(category)
    for category in CATEGORY_HEADS:
        if category in enabled and category not in emitted:
            output.extend(replacements[category])
    return " ".join(output)


def metric_arrays(predictions: list[str], references: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [sentence_bleu4(prediction, reference) for prediction, reference in zip(predictions, references)]
        ),
        np.asarray(
            [meteor_lite(prediction, reference) for prediction, reference in zip(predictions, references)]
        ),
    )


def aggregate(
    bleu: np.ndarray, meteor: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    if mask is None:
        mask = np.ones(len(bleu), dtype=bool)
    return {
        "cases": int(mask.sum()),
        "bleu_4_lite": float(bleu[mask].mean()),
        "meteor_lite": float(meteor[mask].mean()),
        "combined_lite": float((bleu[mask].mean() + meteor[mask].mean()) / 2.0),
    }


def choose_threshold(
    cached_metrics: dict[float | None, tuple[np.ndarray, np.ndarray]],
    train_mask: np.ndarray,
) -> tuple[float | None, list[dict[str, float | None]]]:
    best: tuple[float, int, float, float | None] | None = None
    trace: list[dict[str, float | None]] = []
    for threshold, (bleu, meteor) in cached_metrics.items():
        combined = float((bleu[train_mask].mean() + meteor[train_mask].mean()) / 2.0)
        trace.append({"threshold": threshold, "combined_lite": combined})
        conservative = 1 if threshold is None else 0
        threshold_value = threshold if threshold is not None else 1.0
        candidate = (combined, conservative, threshold_value, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("No threshold candidates")
    return best[3], trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptv3-oof", type=Path, required=True)
    parser.add_argument("--retrieved-oof", type=Path, required=True)
    parser.add_argument("--cross-cohort", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="none,0.35,0.45,0.55,0.65,0.75,0.85")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    queries = {
        row["patient_id"]: row
        for row in (
            json.loads(line)
            for line in args.ptv3_oof.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    retrieved = [
        json.loads(line)
        for line in args.retrieved_oof.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(queries) != 867 or len(retrieved) != 867:
        raise RuntimeError("Expected 867 OOF queries and retrieved reports")
    query_rows = [queries[row["patient_id"]] for row in retrieved]
    base_reports = [row["prediction"] for row in retrieved]
    references = [row["reference"] for row in retrieved]
    folds = np.asarray([int(query["fold"]) for query in query_rows], dtype=np.int64)
    threshold_grid: list[float | None] = [
        None if value.strip().lower() == "none" else float(value)
        for value in args.thresholds.split(",")
    ]

    base_bleu, base_meteor = metric_arrays(base_reports, references)
    cache: dict[str, dict[float | None, tuple[list[str], np.ndarray, np.ndarray]]] = {}
    category_grid: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORY_HEADS:
        cache[category] = {}
        category_grid[category] = []
        for threshold in threshold_grid:
            thresholds = {value: None for value in CATEGORY_HEADS}
            thresholds[category] = threshold
            predictions = [
                correct_report(report, query, thresholds)
                for report, query in zip(base_reports, query_rows, strict=True)
            ]
            bleu, meteor = metric_arrays(predictions, references)
            cache[category][threshold] = (predictions, bleu, meteor)
            category_grid[category].append(
                {"threshold": threshold, **aggregate(bleu, meteor)}
            )

    outer_selection: dict[str, Any] = {}
    crossfit_predictions = list(base_reports)
    for outer_fold in range(1, 6):
        train_mask = folds != outer_fold
        validation_indices = np.flatnonzero(folds == outer_fold).tolist()
        selected: dict[str, float | None] = {}
        traces: dict[str, Any] = {}
        for category in CATEGORY_HEADS:
            compact_cache = {
                threshold: (values[1], values[2])
                for threshold, values in cache[category].items()
            }
            selected[category], traces[category] = choose_threshold(compact_cache, train_mask)
        for index in validation_indices:
            crossfit_predictions[index] = correct_report(
                base_reports[index], query_rows[index], selected
            )
        outer_selection[str(outer_fold)] = {"thresholds": selected, "traces": traces}

    crossfit_bleu, crossfit_meteor = metric_arrays(crossfit_predictions, references)
    full_thresholds: dict[str, float | None] = {}
    full_traces: dict[str, Any] = {}
    all_mask = np.ones(len(retrieved), dtype=bool)
    for category in CATEGORY_HEADS:
        compact_cache = {
            threshold: (values[1], values[2])
            for threshold, values in cache[category].items()
        }
        full_thresholds[category], full_traces[category] = choose_threshold(
            compact_cache, all_mask
        )
    full_predictions = [
        correct_report(report, query, full_thresholds)
        for report, query in zip(base_reports, query_rows, strict=True)
    ]
    full_bleu, full_meteor = metric_arrays(full_predictions, references)

    cross_cohort_metrics: dict[str, Any] | None = None
    if args.cross_cohort is not None:
        cohort_rows = [
            json.loads(line)
            for line in args.cross_cohort.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cohort_predictions = [
            correct_report(row["v6_prediction"], queries[row["patient_id"]], full_thresholds)
            for row in cohort_rows
        ]
        cohort_references = [row["reference"] for row in cohort_rows]
        cohort_bleu, cohort_meteor = metric_arrays(cohort_predictions, cohort_references)
        groups = np.asarray([row["query_cohort"] for row in cohort_rows])
        cross_cohort_metrics = {
            "all": aggregate(cohort_bleu, cohort_meteor),
            "by_query_cohort": {
                group: aggregate(cohort_bleu, cohort_meteor, groups == group)
                for group in sorted(set(groups.tolist()))
            },
        }

    summary = {
        "protocol": {
            "cases": len(retrieved),
            "categories": CATEGORY_HEADS,
            "threshold_grid": threshold_grid,
            "confidence_aggregation": "mean maximum-class probability over category heads",
        },
        "baseline": aggregate(base_bleu, base_meteor),
        "category_grid": category_grid,
        "crossfit": {
            **aggregate(crossfit_bleu, crossfit_meteor),
            "outer_selection": outer_selection,
        },
        "recommended_full_training_thresholds": full_thresholds,
        "full_oof_with_recommended_thresholds": aggregate(full_bleu, full_meteor),
        "cross_cohort_with_recommended_thresholds": cross_cohort_metrics,
        "full_selection_traces": full_traces,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "crossfit_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "patient_id": row["patient_id"],
                    "fold": int(query["fold"]),
                    "prediction": prediction,
                    "reference": reference,
                },
                ensure_ascii=False,
            )
            + "\n"
            for row, query, prediction, reference in zip(
                retrieved, query_rows, crossfit_predictions, references, strict=True
            )
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": summary["baseline"],
                "crossfit": {
                    key: value
                    for key, value in summary["crossfit"].items()
                    if key != "outer_selection"
                },
                "recommended_thresholds": full_thresholds,
                "full_oof": summary["full_oof_with_recommended_thresholds"],
                "cross_cohort": cross_cohort_metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
