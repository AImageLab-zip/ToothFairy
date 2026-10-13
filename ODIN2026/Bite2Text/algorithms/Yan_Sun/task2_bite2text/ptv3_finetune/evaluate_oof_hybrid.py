#!/usr/bin/env python3
"""Evaluate strict OOF retrieval and PTv3-constrained retrieval for Bite2Text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_geometry_retrieval import meteor_lite, sentence_bleu4


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


def parse_numbers(value: str, cast: type) -> list[Any]:
    return [cast(item) for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agreement_score(
    query: dict[str, Any],
    candidate: dict[str, Any],
    score_type: str,
    missing_policy: str,
) -> tuple[float, float]:
    probabilities = query["probabilities"]
    candidate_targets = candidate["target_values"]
    numerator = 0.0
    available_weight = 0.0
    total_weight = 0.0
    for head, probability_by_label in probabilities.items():
        weight = CV_F1[head] if score_type.endswith("reliability") else 1.0
        total_weight += weight
        label = candidate_targets.get(head)
        if label is None or label not in probability_by_label:
            continue
        available_weight += weight
        if score_type.startswith("hard"):
            value = float(query["predicted_labels"][head] == label)
        else:
            value = float(probability_by_label[label])
        numerator += weight * value
    if not available_weight:
        return 0.0, 0.0
    denominator = total_weight if missing_policy == "zero" else available_weight
    return numerator / denominator, available_weight / total_weight


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics_values(
        np.asarray([row["bleu_4_lite"] for row in rows], dtype=np.float64),
        np.asarray([row["meteor_lite"] for row in rows], dtype=np.float64),
        np.asarray([row["fold"] for row in rows], dtype=np.int64),
    )


def metrics_values(
    bleu: np.ndarray, meteor: np.ndarray, fold_values: np.ndarray
) -> dict[str, Any]:
    folds: dict[str, dict[str, float]] = {}
    for fold in range(1, 6):
        selected = fold_values == fold
        folds[str(fold)] = {
            "cases": int(selected.sum()),
            "bleu_4_lite": float(bleu[selected].mean()),
            "meteor_lite": float(meteor[selected].mean()),
        }
    return {
        "cases": int(len(bleu)),
        "bleu_4_lite": float(bleu.mean()),
        "meteor_lite": float(meteor.mean()),
        "combined_lite": float((bleu.mean() + meteor.mean()) / 2.0),
        "folds": folds,
    }


def prediction_row(
    *,
    query: dict[str, Any],
    prediction: str,
    retrieved_id: str | None,
    cosine_similarity: float | None,
    agreement: float | None,
    agreement_coverage: float | None,
) -> dict[str, Any]:
    reference = query["reference"]
    return {
        "patient_id": query["patient_id"],
        "fold": query["fold"],
        "retrieved_patient_id": retrieved_id,
        "cosine_similarity": cosine_similarity,
        "agreement": agreement,
        "agreement_coverage": agreement_coverage,
        "prediction": prediction,
        "reference": reference,
        "bleu_4_lite": sentence_bleu4(prediction, reference),
        "meteor_lite": meteor_lite(prediction, reference),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--retrieval-reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", default="5,10,20,50,100")
    parser.add_argument("--lambdas", default="0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,1.0")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True)

    queries = read_jsonl(args.oof_predictions.resolve())
    by_id = {row["patient_id"]: row for row in queries}
    if len(queries) != 867 or len(by_id) != len(queries):
        raise RuntimeError(f"Expected 867 unique OOF rows, found {len(queries)} / {len(by_id)}")

    with np.load(args.retrieval_index.resolve(), allow_pickle=False) as payload:
        patient_ids = [str(value) for value in payload["patient_ids"]]
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
    report_payload = json.loads(args.retrieval_reports.resolve().read_text(encoding="utf-8"))
    if report_payload["patient_ids"] != patient_ids:
        raise RuntimeError("retrieval_index and retrieval_reports patient order differs")
    reports = {patient_id: report for patient_id, report in zip(patient_ids, report_payload["reports"], strict=True)}
    if set(patient_ids) != set(by_id):
        raise RuntimeError("OOF and retrieval patient sets differ")
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}

    max_top_k = max(parse_numbers(args.top_k, int))
    candidate_cache: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        patient_id = query["patient_id"]
        query_index = index_by_id[patient_id]
        allowed_ids = [
            candidate_id
            for candidate_id in patient_ids
            if by_id[candidate_id]["fold"] != query["fold"]
        ]
        allowed_indices = np.asarray([index_by_id[candidate_id] for candidate_id in allowed_ids])
        similarities = descriptors[allowed_indices] @ descriptors[query_index]
        order = np.argsort(-similarities)[:max_top_k]
        candidate_cache[patient_id] = [
            (allowed_ids[int(position)], float(similarities[int(position)]))
            for position in order
        ]

    # The grid reuses the same query/candidate pairs many times. Model
    # agreements are cheap to precompute. Text metrics are evaluated later,
    # only for candidate indices actually selected by at least one setting.
    agreement_cache: dict[tuple[str, str, str], np.ndarray] = {}
    score_types = (
        "probability_uniform",
        "probability_reliability",
        "hard_uniform",
        "hard_reliability",
    )
    missing_policies = ("available", "zero")
    for query in queries:
        patient_id = query["patient_id"]
        candidates = candidate_cache[patient_id]
        for score_type in score_types:
            for missing_policy in missing_policies:
                agreement_cache[(patient_id, score_type, missing_policy)] = np.asarray(
                    [
                        agreement_score(query, by_id[candidate_id], score_type, missing_policy)[0]
                        for candidate_id, _ in candidates
                    ],
                    dtype=np.float64,
                )

    structured_rows = [
        prediction_row(
            query=query,
            prediction=query["structured_prediction"],
            retrieved_id=None,
            cosine_similarity=None,
            agreement=None,
            agreement_coverage=None,
        )
        for query in queries
    ]
    base_rows = []
    for query in queries:
        candidate_id, cosine = candidate_cache[query["patient_id"]][0]
        base_rows.append(
            prediction_row(
                query=query,
                prediction=reports[candidate_id],
                retrieved_id=candidate_id,
                cosine_similarity=cosine,
                agreement=None,
                agreement_coverage=None,
            )
        )
    write_jsonl(predictions_dir / "structured.jsonl", structured_rows)
    write_jsonl(predictions_dir / "retrieval_top1.jsonl", base_rows)

    grid: list[dict[str, Any]] = [
        {"name": "structured", "strategy": "structured", **metrics(structured_rows)},
        {"name": "retrieval_top1", "strategy": "retrieval", **metrics(base_rows)},
    ]
    top_ks = parse_numbers(args.top_k, int)
    lambdas = parse_numbers(args.lambdas, float)
    fold_values = np.asarray([query["fold"] for query in queries], dtype=np.int64)
    for score_type in score_types:
        for missing_policy in missing_policies:
            for top_k in top_ks:
                for blend_lambda in lambdas:
                    selections: list[int] = []
                    for query in queries:
                        candidates = candidate_cache[query["patient_id"]][:top_k]
                        cosine_values = np.asarray([value for _, value in candidates])
                        agreement_values = agreement_cache[
                            (query["patient_id"], score_type, missing_policy)
                        ][:top_k]
                        selected_index = int(
                            np.argmax(cosine_values + blend_lambda * agreement_values)
                        )
                        selections.append(selected_index)
                    name = (
                        f"hybrid_{score_type}_{missing_policy}_k{top_k}_"
                        f"lambda{str(blend_lambda).replace('.', 'p')}"
                    )
                    result = {
                        "name": name,
                        "strategy": "hybrid",
                        "score_type": score_type,
                        "missing_policy": missing_policy,
                        "top_k": top_k,
                        "lambda": blend_lambda,
                    }
                    grid.append(result)
                    result["_selections"] = selections

    selected_by_query: dict[str, set[int]] = {
        query["patient_id"]: {0} for query in queries
    }
    for result in grid:
        if "_selections" not in result:
            continue
        for query, selected_index in zip(queries, result["_selections"], strict=True):
            selected_by_query[query["patient_id"]].add(selected_index)

    pair_metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for query in queries:
        patient_id = query["patient_id"]
        candidate_count = len(candidate_cache[patient_id])
        bleu = np.full(candidate_count, np.nan, dtype=np.float64)
        meteor = np.full(candidate_count, np.nan, dtype=np.float64)
        for selected_index in selected_by_query[patient_id]:
            candidate_id = candidate_cache[patient_id][selected_index][0]
            bleu[selected_index] = sentence_bleu4(reports[candidate_id], query["reference"])
            meteor[selected_index] = meteor_lite(reports[candidate_id], query["reference"])
        pair_metrics[patient_id] = (bleu, meteor)

    for result in grid:
        if "_selections" not in result:
            continue
        bleu_values = np.asarray(
            [
                pair_metrics[query["patient_id"]][0][selected_index]
                for query, selected_index in zip(queries, result["_selections"], strict=True)
            ]
        )
        meteor_values = np.asarray(
            [
                pair_metrics[query["patient_id"]][1][selected_index]
                for query, selected_index in zip(queries, result["_selections"], strict=True)
            ]
        )
        result.update(metrics_values(bleu_values, meteor_values, fold_values))

    ranked = sorted(grid, key=lambda item: item["combined_lite"], reverse=True)
    materialized: set[str] = {"structured", "retrieval_top1"}
    for result in ranked[:12]:
        if result["name"] in materialized or "_selections" not in result:
            continue
        rows: list[dict[str, Any]] = []
        for query, selected_index in zip(queries, result["_selections"], strict=True):
            candidate_id, cosine = candidate_cache[query["patient_id"]][selected_index]
            agreement, coverage = agreement_score(
                query, by_id[candidate_id], result["score_type"], result["missing_policy"]
            )
            bleu, meteor = pair_metrics[query["patient_id"]]
            row = prediction_row(
                query=query,
                prediction=reports[candidate_id],
                retrieved_id=candidate_id,
                cosine_similarity=cosine,
                agreement=agreement,
                agreement_coverage=coverage,
            )
            # Reuse the precomputed values rather than tokenizing again.
            row["bleu_4_lite"] = float(bleu[selected_index])
            row["meteor_lite"] = float(meteor[selected_index])
            rows.append(row)
        write_jsonl(predictions_dir / f"{result['name']}.jsonl", rows)
        materialized.add(result["name"])
    for result in grid:
        result.pop("_selections", None)

    output = {
        "protocol": {
            "queries": len(queries),
            "candidate_rule": "exclude every candidate in the query patient's CV fold",
            "descriptor_shape": list(descriptors.shape),
            "grid_size": len(grid),
        },
        "ranked": sorted(grid, key=lambda item: item["combined_lite"], reverse=True),
    }
    (output_dir / "metrics_grid.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["ranked"][:12]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
