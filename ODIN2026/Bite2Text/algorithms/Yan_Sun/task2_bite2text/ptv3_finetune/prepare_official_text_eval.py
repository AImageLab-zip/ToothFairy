#!/usr/bin/env python3
"""Convert retrieval JSONL rows into the official Bite2Text evaluator layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite output root: {output_root}")
    input_root = output_root / "input"
    ground_truth_root = output_root / "ground_truth"
    evaluator_output = output_root / "output"
    input_root.mkdir(parents=True)
    ground_truth_root.mkdir()
    evaluator_output.mkdir()

    jobs = []
    for line in args.predictions_jsonl.resolve().read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        patient_id = row["patient_id"]
        job_id = f"job-{patient_id}"
        relative_path = "diagnostic-imaging-report.json"
        report_dir = input_root / job_id / "output"
        report_dir.mkdir(parents=True)
        (report_dir / relative_path).write_text(
            json.dumps({"report": row["prediction"]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (ground_truth_root / f"{patient_id}.txt").write_text(
            row["reference"].strip() + "\n", encoding="utf-8"
        )
        jobs.append(
            {
                "pk": job_id,
                "status": "Succeeded",
                "inputs": [],
                "outputs": [
                    {
                        "socket": {
                            "slug": "diagnostic-imaging-report",
                            "relative_path": relative_path,
                        }
                    }
                ],
            }
        )
    (input_root / "predictions.json").write_text(
        json.dumps(jobs, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(jobs), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
