#!/usr/bin/env python3
"""Evaluate conservative fact-risk reranking on the strict Bite2Text OOF split.

The production v8a.2 score is kept intact.  A second pass is allowed only when
the v8a.2 winner is risky and an alternative is close in the original score.
This makes the experiment suitable for a last submission: ordinary cases are
bit-for-bit unchanged, while high-risk cases can move to a safer neighbour.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR / "hybrid_submission_v8a_test"))
sys.path.insert(0, str(TASK_DIR / "ptv3_finetune"))

from evaluate_geometry_retrieval import meteor_lite, sentence_bleu4
from report_sanitizer import rejection_reason, sanitize_report, split_sentences


CV_F1 = {
    "right_molar_relation": 0.3829,
    "right_canine_relation": 0.3765,
    "left_molar_relation": 0.3867,
    "left_canine_relation": 0.3955,
    "overjet": 0.6366,
    "vertical_relation": 0.5437,
    "midline_relation": 0.3770,
    "crossbite": 0.4997,
    "upper_crowding": 0.2522,
    "lower_crowding": 0.2543,
    "curve_spee": 0.7466,
    "curve_wilson": 0.6514,
}

PHOTO_F1 = {
    "right_canine_relation": 0.2794493467017999,
    "left_canine_relation": 0.33103239430043335,
    "overjet": 0.3989859426890845,
    "vertical_relation": 0.4345494822975876,
    "midline_relation": 0.4111193194504122,
    "upper_crowding": 0.2804054676068764,
    "lower_crowding": 0.25965581387691944,
    "curve_wilson": 0.6321341380293211,
}

MIDLINE_SENTENCES = {
    "coincident": "The dental midlines are coincident.",
    "slightly_deviated": "The dental midlines are slightly deviated.",
    "deviated": "The dental midlines are deviated relative to each other.",
}


def correct_midline_sentence(
    report: str, predicted_label: str, confidence: float, threshold: float
) -> tuple[str, bool]:
    replacement = MIDLINE_SENTENCES.get(predicted_label)
    if replacement is None or confidence < threshold:
        return report, False
    report_sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", report.strip())
        if value.strip()
    ]
    corrected: list[str] = []
    replaced = False
    for sentence in report_sentences:
        if "midline" in sentence.lower():
            if not replaced:
                corrected.append(replacement)
                replaced = True
            continue
        corrected.append(sentence)
    if not replaced:
        corrected.append(replacement)
    return " ".join(corrected), True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def report_risk(report: str) -> tuple[int, int]:
    sentences = split_sentences(report)
    unsupported = sum(rejection_reason(sentence) is not None for sentence in sentences)
    return unsupported, len(sentences)


def contradiction_risk(
    query: dict[str, Any],
    candidate: dict[str, str | None],
    threshold: float,
) -> float:
    """Reliability-weighted confidence above threshold for explicit conflicts."""
    numerator = 0.0
    denominator = sum(CV_F1.values())
    for head, weight in CV_F1.items():
        candidate_value = candidate.get(head)
        if candidate_value is None or candidate_value == query["predicted_labels"][head]:
            continue
        confidence = float(query["confidence"][head])
        numerator += weight * max(0.0, confidence - threshold)
    return numerator / denominator


def metric_values(
    predictions: list[str], references: list[str], folds: np.ndarray
) -> dict[str, Any]:
    bleu = np.asarray(
        [
            sentence_bleu4(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ],
        dtype=np.float64,
    )
    meteor = np.asarray(
        [
            meteor_lite(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ],
        dtype=np.float64,
    )
    fold_metrics: dict[str, Any] = {}
    for fold in range(1, 6):
        mask = folds == fold
        fold_metrics[str(fold)] = {
            "cases": int(mask.sum()),
            "bleu_4_lite": float(bleu[mask].mean()),
            "meteor_lite": float(meteor[mask].mean()),
            "combined_lite": float((bleu[mask].mean() + meteor[mask].mean()) / 2.0),
        }
    return {
        "cases": len(predictions),
        "bleu_4_lite": float(bleu.mean()),
        "meteor_lite": float(meteor.mean()),
        "combined_lite": float((bleu.mean() + meteor.mean()) / 2.0),
        "folds": fold_metrics,
    }


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptv3-oof", type=Path, required=True)
    parser.add_argument("--photo-oof", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--retrieval-reports", type=Path, required=True)
    parser.add_argument("--retrieval-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--geometry-lambda", type=float, default=0.5)
    parser.add_argument("--photo-lambda", type=float, default=0.2)
    parser.add_argument("--midline-threshold", type=float, default=0.45)
    parser.add_argument("--unsupported-gate", type=int, default=5)
    parser.add_argument("--margins", default="0.005,0.01,0.02,0.03")
    parser.add_argument("--unsupported-penalties", default="0.005,0.01,0.02,0.03")
    parser.add_argument("--contradiction-thresholds", default="0.65,0.75,0.85")
    parser.add_argument("--contradiction-gates", default="0,0.005,0.01")
    parser.add_argument("--contradiction-penalties", default="0,0.5,1,2")
    parser.add_argument("--min-contradiction-improvement", type=float, default=0.0)
    parser.add_argument("--no-new-unsupported", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    queries = load_jsonl(args.ptv3_oof)
    by_id = {str(row["patient_id"]): row for row in queries}
    if len(queries) != 867 or len(by_id) != 867:
        raise RuntimeError("Expected 867 unique PTv3 OOF rows")
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))

    with np.load(args.retrieval_index, allow_pickle=False) as payload:
        patient_ids = np.asarray(payload["patient_ids"]).astype(str).tolist()
        # float64 avoids an Accelerate/BLAS overflow warning observed on macOS
        # for otherwise finite, unit-normalized float32 descriptors.
        descriptors = np.asarray(payload["descriptors"], dtype=np.float64)
    report_payload = json.loads(args.retrieval_reports.read_text(encoding="utf-8"))
    label_payload = json.loads(args.retrieval_labels.read_text(encoding="utf-8"))
    if report_payload["patient_ids"] != patient_ids:
        raise RuntimeError("Retrieval report order does not match the index")
    if label_payload["patient_ids"] != patient_ids:
        raise RuntimeError("Retrieval label order does not match the index")
    reports = [str(value).strip() for value in report_payload["reports"]]
    candidate_labels = label_payload["target_values"]
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    if not set(by_id).issubset(index_by_id):
        raise RuntimeError("OOF patients are missing from the retrieval index")

    with np.load(args.photo_oof) as photo_archive:
        photo_ids = photo_archive["patient_ids"].astype(str).tolist()
        photo_probabilities = {
            head: np.asarray(photo_archive[f"photo_probabilities_{head}"], dtype=np.float64)
            for head in PHOTO_F1
        }
    photo_index = {patient_id: index for index, patient_id in enumerate(photo_ids)}
    if set(photo_ids) != set(by_id):
        raise RuntimeError("Photo and PTv3 OOF patient sets differ")

    risks = [report_risk(report) for report in reports]
    folds = np.asarray([int(query["fold"]) for query in queries], dtype=np.int64)
    references = [str(query["reference"]) for query in queries]
    candidate_indices_by_query: list[np.ndarray] = []
    base_scores_by_query: list[np.ndarray] = []
    unsupported_by_query: list[np.ndarray] = []
    contradiction_cache: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in parse_floats(args.contradiction_thresholds)
    }

    geometry_total = sum(CV_F1.values())
    photo_total = sum(PHOTO_F1.values())
    labeled_patient_ids = [patient_id for patient_id in patient_ids if patient_id in by_id]
    for query in queries:
        patient_id = str(query["patient_id"])
        query_index = index_by_id[patient_id]
        allowed_ids = [
            candidate_id
            for candidate_id in labeled_patient_ids
            if int(by_id[candidate_id]["fold"]) != int(query["fold"])
        ]
        allowed_indices = np.asarray([index_by_id[value] for value in allowed_ids], dtype=np.int64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            similarities = descriptors[allowed_indices] @ descriptors[query_index]
        if not np.isfinite(similarities).all():
            raise RuntimeError(f"Non-finite retrieval similarities for {patient_id}")
        order = np.argsort(-similarities)[: args.top_k]
        selected_indices = allowed_indices[order]
        cosine = similarities[order].astype(np.float64)
        geometry = np.zeros(len(selected_indices), dtype=np.float64)
        photo = np.zeros(len(selected_indices), dtype=np.float64)
        query_photo_index = photo_index[patient_id]
        for position, candidate_index in enumerate(selected_indices.tolist()):
            targets = candidate_labels[candidate_index]
            geometry[position] = sum(
                weight
                for head, weight in CV_F1.items()
                if targets.get(head) is not None
                and targets.get(head) == query["predicted_labels"][head]
            ) / geometry_total
            photo_numerator = 0.0
            for head, weight in PHOTO_F1.items():
                target = targets.get(head)
                if target is None:
                    continue
                try:
                    target_index = head_vocabs[head].index(target)
                except ValueError:
                    continue
                photo_numerator += weight * float(
                    photo_probabilities[head][query_photo_index, target_index]
                )
            photo[position] = photo_numerator / photo_total
        candidate_indices_by_query.append(selected_indices)
        base_scores_by_query.append(
            cosine + args.geometry_lambda * geometry + args.photo_lambda * photo
        )
        unsupported_by_query.append(
            np.asarray([risks[index][0] for index in selected_indices], dtype=np.float64)
        )
        for threshold in contradiction_cache:
            contradiction_cache[threshold].append(
                np.asarray(
                    [
                        contradiction_risk(query, candidate_labels[index], threshold)
                        for index in selected_indices
                    ],
                    dtype=np.float64,
                )
            )

    def materialize(selections: np.ndarray) -> tuple[list[str], list[dict[str, Any]]]:
        predictions: list[str] = []
        rows: list[dict[str, Any]] = []
        for query_number, (query, selected_position) in enumerate(
            zip(queries, selections.tolist())
        ):
            selected_index = int(candidate_indices_by_query[query_number][selected_position])
            report, midline_corrected = correct_midline_sentence(
                reports[selected_index],
                query["predicted_labels"]["midline_relation"],
                float(query["confidence"]["midline_relation"]),
                args.midline_threshold,
            )
            report, sanitizer = sanitize_report(report)
            predictions.append(report)
            rows.append(
                {
                    "patient_id": query["patient_id"],
                    "fold": int(query["fold"]),
                    "retrieved_patient_id": patient_ids[selected_index],
                    "selected_position": int(selected_position),
                    "prediction": report,
                    "reference": query["reference"],
                    "midline_corrected": midline_corrected,
                    "sanitizer": sanitizer,
                }
            )
        return predictions, rows

    baseline_selections = np.asarray(
        [int(np.argmax(values)) for values in base_scores_by_query], dtype=np.int64
    )
    baseline_predictions, baseline_rows = materialize(baseline_selections)
    baseline_metrics = metric_values(baseline_predictions, references, folds)

    grid: list[dict[str, Any]] = []
    materialized: dict[str, tuple[np.ndarray, list[str], list[dict[str, Any]]]] = {}
    combinations = itertools.product(
        parse_floats(args.margins),
        parse_floats(args.unsupported_penalties),
        parse_floats(args.contradiction_thresholds),
        parse_floats(args.contradiction_gates),
        parse_floats(args.contradiction_penalties),
    )
    for margin, unsupported_penalty, contradiction_threshold, contradiction_gate, contradiction_penalty in combinations:
        selections = baseline_selections.copy()
        rerank_reasons: list[str | None] = [None] * len(queries)
        for query_number in range(len(queries)):
            base_position = int(baseline_selections[query_number])
            base_scores = base_scores_by_query[query_number]
            unsupported = unsupported_by_query[query_number]
            contradiction = contradiction_cache[contradiction_threshold][query_number]
            base_unsupported = int(unsupported[base_position])
            base_contradiction = float(contradiction[base_position])
            unsupported_trigger = base_unsupported >= args.unsupported_gate
            contradiction_trigger = (
                contradiction_penalty > 0 and base_contradiction >= contradiction_gate
            )
            if not unsupported_trigger and not contradiction_trigger:
                continue
            eligible = base_scores >= float(base_scores[base_position]) - margin
            penalized = (
                base_scores
                - unsupported_penalty * unsupported
                - contradiction_penalty * contradiction
            )
            penalized[~eligible] = -np.inf
            selected_position = int(np.argmax(penalized))
            if selected_position == base_position:
                continue
            selected_unsupported = int(unsupported[selected_position])
            selected_contradiction = float(contradiction[selected_position])
            improves_unsupported = unsupported_trigger and selected_unsupported < base_unsupported
            improves_contradiction = (
                contradiction_trigger
                and base_contradiction - selected_contradiction
                >= args.min_contradiction_improvement - 1e-12
            )
            if (
                args.no_new_unsupported
                and selected_unsupported > base_unsupported
                and not improves_unsupported
            ):
                continue
            if not improves_unsupported and not improves_contradiction:
                continue
            selections[query_number] = selected_position
            rerank_reasons[query_number] = "+".join(
                value
                for value, enabled in (
                    ("unsupported", improves_unsupported),
                    ("contradiction", improves_contradiction),
                )
                if enabled
            )

        predictions, rows = materialize(selections)
        changed = np.flatnonzero(selections != baseline_selections)
        metrics = metric_values(predictions, references, folds)
        name = (
            f"m{margin:g}_u{unsupported_penalty:g}_ct{contradiction_threshold:g}_"
            f"cg{contradiction_gate:g}_cp{contradiction_penalty:g}"
        ).replace(".", "p")
        fold_deltas = [
            metrics["folds"][str(fold)]["combined_lite"]
            - baseline_metrics["folds"][str(fold)]["combined_lite"]
            for fold in range(1, 6)
        ]
        grid.append(
            {
                "name": name,
                "margin": margin,
                "unsupported_penalty": unsupported_penalty,
                "contradiction_threshold": contradiction_threshold,
                "contradiction_gate": contradiction_gate,
                "contradiction_penalty": contradiction_penalty,
                "changed_cases": int(len(changed)),
                "changed_case_ids": [queries[index]["patient_id"] for index in changed],
                "rerank_reasons": {
                    "unsupported": sum(
                        reason is not None and "unsupported" in reason for reason in rerank_reasons
                    ),
                    "contradiction": sum(
                        reason is not None and "contradiction" in reason for reason in rerank_reasons
                    ),
                },
                **metrics,
                "delta": {
                    key: float(metrics[key]) - float(baseline_metrics[key])
                    for key in ("bleu_4_lite", "meteor_lite", "combined_lite")
                },
                "fold_combined_deltas": fold_deltas,
                "folds_improved": sum(value > 0 for value in fold_deltas),
                "folds_not_worse": sum(value >= -1e-12 for value in fold_deltas),
            }
        )
        materialized[name] = (selections, predictions, rows)

    # Favor text stability first; RadFact is run later on the changed cases of
    # the shortlisted configurations.
    ranked = sorted(
        grid,
        key=lambda row: (
            row["delta"]["combined_lite"] >= -0.0005,
            row["folds_not_worse"],
            row["delta"]["combined_lite"],
            -row["changed_cases"],
        ),
        reverse=True,
    )
    shortlist: list[dict[str, Any]] = []
    seen_case_sets: set[tuple[str, ...]] = set()
    for row in ranked:
        case_set = tuple(row["changed_case_ids"])
        if not case_set or case_set in seen_case_sets:
            continue
        if row["delta"]["combined_lite"] < -0.001:
            continue
        seen_case_sets.add(case_set)
        shortlist.append(row)
        if len(shortlist) == 8:
            break

    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir()
    write_jsonl(predictions_dir / "v8a2_baseline.jsonl", baseline_rows)
    for row in shortlist:
        selections, _, output_rows = materialized[row["name"]]
        threshold = float(row["contradiction_threshold"])
        for query_number, output_row in enumerate(output_rows):
            baseline_position = int(baseline_selections[query_number])
            selected_position = int(selections[query_number])
            contradiction = contradiction_cache[threshold][query_number]
            unsupported = unsupported_by_query[query_number]
            output_row["rerank_audit"] = {
                "changed": selected_position != baseline_position,
                "baseline_score": float(base_scores_by_query[query_number][baseline_position]),
                "selected_score": float(base_scores_by_query[query_number][selected_position]),
                "score_drop": float(
                    base_scores_by_query[query_number][baseline_position]
                    - base_scores_by_query[query_number][selected_position]
                ),
                "baseline_unsupported": int(unsupported[baseline_position]),
                "selected_unsupported": int(unsupported[selected_position]),
                "baseline_contradiction": float(contradiction[baseline_position]),
                "selected_contradiction": float(contradiction[selected_position]),
                "contradiction_improvement": float(
                    contradiction[baseline_position] - contradiction[selected_position]
                ),
            }
        write_jsonl(predictions_dir / f"{row['name']}.jsonl", output_rows)
        changed_ids = set(row["changed_case_ids"])
        write_jsonl(
            predictions_dir / f"{row['name']}_changed.jsonl",
            [output_row for output_row in output_rows if output_row["patient_id"] in changed_ids],
        )
        write_jsonl(
            predictions_dir / f"{row['name']}_baseline_changed.jsonl",
            [output_row for output_row in baseline_rows if output_row["patient_id"] in changed_ids],
        )

    summary = {
        "protocol": {
            "cases": len(queries),
            "candidate_rule": "exclude every labeled candidate in the query fold",
            "top_k": args.top_k,
            "geometry_lambda": args.geometry_lambda,
            "photo_lambda": args.photo_lambda,
            "midline_threshold": args.midline_threshold,
            "unsupported_gate": args.unsupported_gate,
            "min_contradiction_improvement": args.min_contradiction_improvement,
            "no_new_unsupported": args.no_new_unsupported,
            "grid_size": len(grid),
        },
        "baseline": baseline_metrics,
        "shortlist": shortlist,
        "ranked": ranked,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "shortlist": [
                    {
                        "name": row["name"],
                        "changed_cases": row["changed_cases"],
                        "delta": row["delta"],
                        "folds_not_worse": row["folds_not_worse"],
                    }
                    for row in shortlist
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
