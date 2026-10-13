#!/usr/bin/env python3
"""Cross-fit photo-only soft evidence on top of the frozen v5 hybrid score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptv3_finetune"))
from evaluate_geometry_retrieval import meteor_lite, sentence_bleu4


PTV3_RELIABILITY = {
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


def metric_values(bleu: np.ndarray, meteor: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
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
        "cases": len(bleu),
        "bleu_4_lite": float(bleu.mean()),
        "meteor_lite": float(meteor.mean()),
        "combined_lite": float((bleu.mean() + meteor.mean()) / 2.0),
        "folds": fold_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptv3-oof", type=Path, required=True)
    parser.add_argument("--photo-oof", type=Path, required=True)
    parser.add_argument("--fusion-summary", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--retrieval-reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--geometry-lambda", type=float, default=0.5)
    parser.add_argument("--photo-lambdas", default="0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--photo-heads", default="all")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    head_names = list(head_vocabs)
    if args.photo_heads == "all":
        photo_heads = head_names
    else:
        photo_heads = [value.strip() for value in args.photo_heads.split(",") if value.strip()]
        unknown = set(photo_heads) - set(head_names)
        if unknown:
            raise RuntimeError(f"Unknown photo heads: {sorted(unknown)}")

    queries = [
        json.loads(line)
        for line in args.ptv3_oof.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["patient_id"]: row for row in queries}
    if len(queries) != 867 or len(by_id) != 867:
        raise RuntimeError("Expected 867 unique PTv3 OOF rows")
    fusion_summary = json.loads(args.fusion_summary.read_text(encoding="utf-8"))
    photo_reliability = {
        head: float(fusion_summary["photo_oof"]["heads"][head]["macro_f1"])
        for head in photo_heads
    }

    photo_payload = np.load(args.photo_oof)
    photo_ids = photo_payload["patient_ids"].astype(str).tolist()
    photo_index = {patient_id: index for index, patient_id in enumerate(photo_ids)}
    if set(photo_ids) != set(by_id):
        raise RuntimeError("Photo and PTv3 OOF patient sets differ")
    photo_probabilities = {
        head: photo_payload[f"photo_probabilities_{head}"] for head in photo_heads
    }

    with np.load(args.retrieval_index, allow_pickle=False) as payload:
        patient_ids = [str(value) for value in payload["patient_ids"]]
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
    report_payload = json.loads(args.retrieval_reports.read_text(encoding="utf-8"))
    if report_payload["patient_ids"] != patient_ids:
        raise RuntimeError("Retrieval assets disagree on patient order")
    reports = dict(zip(patient_ids, report_payload["reports"], strict=True))
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}

    folds = np.asarray([int(query["fold"]) for query in queries], dtype=np.int64)
    candidate_ids_by_query: list[list[str]] = []
    cosine_by_query: list[np.ndarray] = []
    geometry_agreement_by_query: list[np.ndarray] = []
    photo_agreement_by_query: list[np.ndarray] = []
    geometry_total_weight = sum(PTV3_RELIABILITY.values())
    photo_total_weight = sum(photo_reliability.values())
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
        order = np.argsort(-similarities)[: args.top_k]
        candidate_ids = [allowed_ids[int(position)] for position in order]
        candidate_ids_by_query.append(candidate_ids)
        cosine_by_query.append(similarities[order].astype(np.float64))

        geometry_values: list[float] = []
        photo_values: list[float] = []
        query_photo_index = photo_index[patient_id]
        for candidate_id in candidate_ids:
            targets = by_id[candidate_id]["target_values"]
            geometry_numerator = 0.0
            for head, weight in PTV3_RELIABILITY.items():
                label = targets.get(head)
                if label is not None and query["predicted_labels"][head] == label:
                    geometry_numerator += weight
            geometry_values.append(geometry_numerator / geometry_total_weight)

            photo_numerator = 0.0
            for head, weight in photo_reliability.items():
                label = targets.get(head)
                if label is None:
                    continue
                label_index = head_vocabs[head].index(label)
                photo_numerator += weight * float(
                    photo_probabilities[head][query_photo_index, label_index]
                )
            photo_values.append(photo_numerator / photo_total_weight)
        geometry_agreement_by_query.append(np.asarray(geometry_values, dtype=np.float64))
        photo_agreement_by_query.append(np.asarray(photo_values, dtype=np.float64))

    lambdas = [float(value) for value in args.photo_lambdas.split(",") if value.strip()]
    selections_by_lambda: dict[float, np.ndarray] = {}
    selected_candidates: dict[int, set[int]] = {index: set() for index in range(len(queries))}
    for photo_lambda in lambdas:
        selections = np.asarray(
            [
                int(
                    np.argmax(
                        cosine_by_query[index]
                        + args.geometry_lambda * geometry_agreement_by_query[index]
                        + photo_lambda * photo_agreement_by_query[index]
                    )
                )
                for index in range(len(queries))
            ],
            dtype=np.int64,
        )
        selections_by_lambda[photo_lambda] = selections
        for query_index, selection in enumerate(selections.tolist()):
            selected_candidates[query_index].add(selection)

    pair_metrics: dict[tuple[int, int], tuple[float, float]] = {}
    for query_index, query in enumerate(queries):
        for candidate_index in selected_candidates[query_index]:
            candidate_id = candidate_ids_by_query[query_index][candidate_index]
            report = reports[candidate_id]
            pair_metrics[(query_index, candidate_index)] = (
                sentence_bleu4(report, query["reference"]),
                meteor_lite(report, query["reference"]),
            )

    grid: list[dict[str, Any]] = []
    metric_arrays: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for photo_lambda in lambdas:
        selections = selections_by_lambda[photo_lambda]
        bleu = np.asarray(
            [pair_metrics[(index, int(selection))][0] for index, selection in enumerate(selections)]
        )
        meteor = np.asarray(
            [pair_metrics[(index, int(selection))][1] for index, selection in enumerate(selections)]
        )
        metric_arrays[photo_lambda] = (bleu, meteor)
        grid.append({"photo_lambda": photo_lambda, **metric_values(bleu, meteor, folds)})

    crossfit_bleu = np.zeros(len(queries), dtype=np.float64)
    crossfit_meteor = np.zeros(len(queries), dtype=np.float64)
    crossfit_selections = np.zeros(len(queries), dtype=np.int64)
    chosen_lambdas: dict[str, Any] = {}
    for outer_fold in range(1, 6):
        train_mask = folds != outer_fold
        best: tuple[float, float] | None = None
        trace: list[dict[str, float]] = []
        for photo_lambda in lambdas:
            bleu, meteor = metric_arrays[photo_lambda]
            combined = float((bleu[train_mask].mean() + meteor[train_mask].mean()) / 2.0)
            trace.append({"photo_lambda": photo_lambda, "combined_lite": combined})
            candidate = (combined, -photo_lambda)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise RuntimeError("No photo lambda candidates")
        chosen_lambda = -best[1]
        validation_mask = folds == outer_fold
        crossfit_selections[validation_mask] = selections_by_lambda[chosen_lambda][validation_mask]
        crossfit_bleu[validation_mask] = metric_arrays[chosen_lambda][0][validation_mask]
        crossfit_meteor[validation_mask] = metric_arrays[chosen_lambda][1][validation_mask]
        chosen_lambdas[str(outer_fold)] = {
            "photo_lambda": chosen_lambda,
            "trace": trace,
        }

    ranked = sorted(grid, key=lambda row: row["combined_lite"], reverse=True)
    output = {
        "protocol": {
            "cases": len(queries),
            "top_k": args.top_k,
            "geometry_lambda": args.geometry_lambda,
            "photo_heads": photo_heads,
            "photo_reliability": photo_reliability,
            "photo_lambdas": lambdas,
        },
        "grid": grid,
        "ranked": ranked,
        "crossfit": {
            **metric_values(crossfit_bleu, crossfit_meteor, folds),
            "outer_fold_selection": chosen_lambdas,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        selection = int(crossfit_selections[query_index])
        candidate_id = candidate_ids_by_query[query_index][selection]
        prediction_rows.append(
            {
                "patient_id": query["patient_id"],
                "fold": int(query["fold"]),
                "retrieved_patient_id": candidate_id,
                "prediction": reports[candidate_id],
                "reference": query["reference"],
                "bleu_4_lite": float(crossfit_bleu[query_index]),
                "meteor_lite": float(crossfit_meteor[query_index]),
            }
        )
    (args.output_dir / "crossfit_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "photo_heads": photo_heads,
                "best_full_oof": ranked[0],
                "crossfit": output["crossfit"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
