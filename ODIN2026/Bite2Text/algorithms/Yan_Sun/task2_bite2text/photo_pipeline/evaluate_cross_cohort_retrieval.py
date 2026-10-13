#!/usr/bin/env python3
"""Evaluate retrieval when queries and candidates come from different cohorts."""

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

PHOTO_HEADS = (
    "right_canine_relation",
    "left_canine_relation",
    "overjet",
    "vertical_relation",
    "midline_relation",
    "upper_crowding",
    "lower_crowding",
    "curve_wilson",
)


def cohort(patient_id: str) -> str:
    match = re.search(r"\d+", patient_id)
    if match is None:
        raise RuntimeError(f"No numeric patient id: {patient_id}")
    return "external_bits2bites" if int(match.group()) < 3000 else "challenge_release"


def metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    bleu = np.asarray(
        [sentence_bleu4(row[prediction_key], row["reference"]) for row in rows]
    )
    meteor = np.asarray(
        [meteor_lite(row[prediction_key], row["reference"]) for row in rows]
    )

    def aggregate(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "cases": int(mask.sum()),
            "bleu_4_lite": float(bleu[mask].mean()),
            "meteor_lite": float(meteor[mask].mean()),
            "combined_lite": float((bleu[mask].mean() + meteor[mask].mean()) / 2.0),
        }

    groups = np.asarray([row["query_cohort"] for row in rows])
    output = aggregate(np.ones(len(rows), dtype=bool))
    output["by_query_cohort"] = {
        value: aggregate(groups == value) for value in sorted(set(groups.tolist()))
    }
    return output


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
    parser.add_argument("--photo-lambda", type=float, default=0.2)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    queries = [
        json.loads(line)
        for line in args.ptv3_oof.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["patient_id"]: row for row in queries}
    if len(queries) != 867 or len(by_id) != 867:
        raise RuntimeError("Expected 867 unique OOF queries")
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    fusion = json.loads(args.fusion_summary.read_text(encoding="utf-8"))
    photo_reliability = {
        head: float(fusion["photo_oof"]["heads"][head]["macro_f1"])
        for head in PHOTO_HEADS
    }
    photo_total_weight = sum(photo_reliability.values())
    ptv3_total_weight = sum(PTV3_RELIABILITY.values())

    photo = np.load(args.photo_oof)
    photo_ids = photo["patient_ids"].astype(str).tolist()
    photo_index = {patient_id: index for index, patient_id in enumerate(photo_ids)}
    if set(photo_ids) != set(by_id):
        raise RuntimeError("Photo/PTv3 OOF patient sets differ")
    photo_probabilities = {
        head: np.asarray(photo[f"photo_probabilities_{head}"], dtype=np.float32)
        for head in PHOTO_HEADS
    }
    photo.close()

    with np.load(args.retrieval_index, allow_pickle=False) as payload:
        patient_ids = payload["patient_ids"].astype(str).tolist()
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
    reports_payload = json.loads(args.retrieval_reports.read_text(encoding="utf-8"))
    if reports_payload["patient_ids"] != patient_ids:
        raise RuntimeError("Retrieval reports and descriptors disagree")
    reports = dict(zip(patient_ids, reports_payload["reports"], strict=True))
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    if set(patient_ids) != set(by_id):
        raise RuntimeError("Retrieval and OOF patient sets differ")
    similarity_matrix = descriptors @ descriptors.T

    output_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = query["patient_id"]
        query_cohort = cohort(query_id)
        candidate_cohort = (
            "challenge_release"
            if query_cohort == "external_bits2bites"
            else "external_bits2bites"
        )
        candidate_ids = [
            patient_id for patient_id in patient_ids if cohort(patient_id) == candidate_cohort
        ]
        candidate_indices = np.asarray(
            [index_by_id[patient_id] for patient_id in candidate_ids], dtype=np.int64
        )
        similarities = similarity_matrix[index_by_id[query_id], candidate_indices]
        order = np.argsort(-similarities)[: min(args.top_k, len(similarities))]

        query_photo_index = photo_index[query_id]
        candidates: list[dict[str, Any]] = []
        for position in order.tolist():
            candidate_id = candidate_ids[position]
            targets = by_id[candidate_id]["target_values"]
            geometry_numerator = sum(
                weight
                for head, weight in PTV3_RELIABILITY.items()
                if targets.get(head) is not None
                and query["predicted_labels"].get(head) == targets.get(head)
            )
            photo_numerator = 0.0
            for head, weight in photo_reliability.items():
                label = targets.get(head)
                if label is None or label not in head_vocabs[head]:
                    continue
                label_index = head_vocabs[head].index(label)
                photo_numerator += weight * float(
                    photo_probabilities[head][query_photo_index, label_index]
                )
            similarity = float(similarities[position])
            geometry_agreement = geometry_numerator / ptv3_total_weight
            photo_agreement = photo_numerator / photo_total_weight
            candidates.append(
                {
                    "patient_id": candidate_id,
                    "similarity": similarity,
                    "geometry_agreement": geometry_agreement,
                    "photo_agreement": photo_agreement,
                    "v5_score": similarity + args.geometry_lambda * geometry_agreement,
                    "v6_score": similarity
                    + args.geometry_lambda * geometry_agreement
                    + args.photo_lambda * photo_agreement,
                }
            )
        if not candidates:
            raise RuntimeError(f"No cross-cohort candidates for {query_id}")
        pure = max(candidates, key=lambda row: row["similarity"])
        v5 = max(candidates, key=lambda row: row["v5_score"])
        v6 = max(candidates, key=lambda row: row["v6_score"])
        output_rows.append(
            {
                "patient_id": query_id,
                "query_cohort": query_cohort,
                "candidate_cohort": candidate_cohort,
                "reference": query["reference"],
                "structured_prediction": query["structured_prediction"],
                "pure_retrieved_id": pure["patient_id"],
                "pure_prediction": reports[pure["patient_id"]],
                "v5_retrieved_id": v5["patient_id"],
                "v5_prediction": reports[v5["patient_id"]],
                "v6_retrieved_id": v6["patient_id"],
                "v6_prediction": reports[v6["patient_id"]],
            }
        )

    configurations = {
        "structured": "structured_prediction",
        "pure": "pure_prediction",
        "v5": "v5_prediction",
        "v6": "v6_prediction",
    }
    summary = {
        "protocol": {
            "cases": len(output_rows),
            "cohort_rule": "numeric patient id < 3000 is external Bits2Bites cohort",
            "candidate_policy": "opposite cohort only",
            "top_k": args.top_k,
            "geometry_lambda": args.geometry_lambda,
            "photo_lambda": args.photo_lambda,
            "photo_heads": PHOTO_HEADS,
        },
        "metrics": {
            name: metrics(output_rows, key) for name, key in configurations.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
