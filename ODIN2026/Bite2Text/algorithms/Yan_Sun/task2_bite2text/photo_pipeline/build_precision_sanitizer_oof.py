#!/usr/bin/env python3
"""Apply the exact v8a production sanitizer to an OOF prediction file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK_DIR / "hybrid_submission_v8a_test"))
sys.path.insert(0, str(TASK_DIR / "ptv3_finetune"))

from evaluate_geometry_retrieval import meteor_lite, sentence_bleu4
from report_sanitizer import sanitize_report


def metrics(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    bleu = np.asarray(
        [sentence_bleu4(prediction, reference) for prediction, reference in zip(predictions, references, strict=True)]
    )
    meteor = np.asarray(
        [meteor_lite(prediction, reference) for prediction, reference in zip(predictions, references, strict=True)]
    )
    return {
        "cases": len(predictions),
        "bleu_4_lite": float(bleu.mean()),
        "meteor_lite": float(meteor.mean()),
        "combined_lite": float((bleu.mean() + meteor.mean()) / 2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sanitized_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    changed = 0
    removed_sentences = 0
    duplicates = 0

    for row in rows:
        prediction, details = sanitize_report(str(row["prediction"]))
        if prediction != row["prediction"].strip():
            changed += 1
        removed_sentences += int(details["removed_sentence_count"])
        duplicates += int(details["duplicate_sentence_count"])
        reason_counts.update(item["reason"] for item in details["removed"])
        sanitized_rows.append({**row, "prediction": prediction})
        audit_rows.append({"patient_id": row["patient_id"], **details})

    base_predictions = [str(row["prediction"]) for row in rows]
    predictions = [str(row["prediction"]) for row in sanitized_rows]
    references = [str(row["reference"]) for row in rows]
    baseline = metrics(base_predictions, references)
    sanitized = metrics(predictions, references)
    summary = {
        "version": "v8a-precision-sanitizer-2",
        "input": str(args.input.resolve()),
        "cases": len(rows),
        "changed_cases": changed,
        "removed_sentences": removed_sentences,
        "duplicate_sentences": duplicates,
        "removal_reasons": dict(sorted(reason_counts.items())),
        "baseline": baseline,
        "sanitized": sanitized,
        "delta": {
            key: float(sanitized[key]) - float(baseline[key])
            for key in ("bleu_4_lite", "meteor_lite", "combined_lite")
        },
    }

    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sanitized_rows),
        encoding="utf-8",
    )
    changed_patient_ids = {
        str(original["patient_id"])
        for original, sanitized_row in zip(rows, sanitized_rows, strict=True)
        if str(original["prediction"]).strip() != str(sanitized_row["prediction"])
    }
    (args.output_dir / "baseline_changed_cases.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
            if str(row["patient_id"]) in changed_patient_ids
        ),
        encoding="utf-8",
    )
    (args.output_dir / "sanitized_changed_cases.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sanitized_rows
            if str(row["patient_id"]) in changed_patient_ids
        ),
        encoding="utf-8",
    )
    (args.output_dir / "audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
