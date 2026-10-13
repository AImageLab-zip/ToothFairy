#!/usr/bin/env python3
"""Prepare and execute a one-case Bite2Text Docker/evaluation smoke test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mha", ".dzi"}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="F1980")
    parser.add_argument("--algorithm-image", default="bite2text-example-algorithm")
    parser.add_argument("--evaluation-image", default="bite2text-eval-test")
    parser.add_argument("--gpu", default="1", help="Host GPU index, or 'none' for CPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    work_dir = args.work_dir.resolve()
    case_id = args.case_id
    case_source = dataset_root / case_id

    if work_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing work directory: {work_dir}")
    if not case_source.is_dir():
        raise SystemExit(f"Case does not exist: {case_source}")

    lower_source = case_source / "ios" / "ios_lower.stl"
    upper_source = case_source / "ios" / "ios_upper.stl"
    photo_source_dir = case_source / "intraoral-photo"
    photos = sorted(
        path for path in photo_source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES
    )
    if not lower_source.is_file() or not upper_source.is_file():
        raise SystemExit(f"Missing IOS scans for {case_id}")
    if not photos:
        raise SystemExit(f"No supported photos for {case_id}")

    report_candidates = sorted((case_source / "reports_ios_en").glob("*.txt"))
    if not report_candidates:
        report_candidates = sorted((case_source / "reports_intraoral-photo_en").glob("*.txt"))
    if not report_candidates:
        raise SystemExit(f"No English reference report for {case_id}")

    input_dir = work_dir / "algorithm_input"
    output_dir = work_dir / "evaluation_input" / f"job-{case_id.lower()}" / "output"
    evaluation_input = work_dir / "evaluation_input"
    evaluation_output = work_dir / "evaluation_output"
    ground_truth = work_dir / "ground_truth"

    lower_dir = input_dir / "files" / "ios-lower"
    upper_dir = input_dir / "files" / "ios-upper"
    photo_dir = input_dir / "images" / "intraoral-photo"
    for directory in (lower_dir, upper_dir, photo_dir, output_dir, evaluation_output, ground_truth):
        directory.mkdir(parents=True, exist_ok=False if directory == lower_dir else True)
    os.chmod(output_dir, 0o777)
    os.chmod(evaluation_output, 0o777)

    shutil.copy2(lower_source, lower_dir / "ios_lower.stl")
    shutil.copy2(upper_source, upper_dir / "ios_upper.stl")
    for index, photo in enumerate(photos[:5], start=1):
        shutil.copy2(photo, photo_dir / f"intraoral-photo-{index}{photo.suffix.lower()}")

    inputs = [
        {
            "socket": {
                "slug": "3d-lower-teeth-scan",
                "relative_path": "files/ios-lower",
                "is_image_kind": False,
                "is_panimg_kind": False,
                "is_dicom_image_kind": False,
                "is_json_kind": False,
                "is_file_kind": True,
            },
            "file": {"name": "ios_lower.stl"},
            "image": None,
            "value": None,
        },
        {
            "socket": {
                "slug": "3d-upper-teeth-scan",
                "relative_path": "files/ios-upper",
                "is_image_kind": False,
                "is_panimg_kind": False,
                "is_dicom_image_kind": False,
                "is_json_kind": False,
                "is_file_kind": True,
            },
            "file": {"name": "ios_upper.stl"},
            "image": None,
            "value": None,
        },
        {
            "socket": {
                "slug": "2d-intraoral-photographs",
                "relative_path": "images/intraoral-photo",
                "is_image_kind": True,
                "is_panimg_kind": True,
                "is_dicom_image_kind": False,
                "is_json_kind": False,
                "is_file_kind": False,
            },
            "file": None,
            "image": {"name": photos[0].name},
            "value": None,
        },
    ]
    (input_dir / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    docker_command = ["docker", "run", "--rm", "--platform=linux/amd64", "--network", "none"]
    if args.gpu.lower() != "none":
        docker_command.extend(["--gpus", f"device={args.gpu}"])
    docker_command.extend(
        [
            "--volume", f"{input_dir}:/input:ro",
            "--volume", f"{output_dir}:/output",
            args.algorithm_image,
        ]
    )
    run(docker_command)

    output_report = output_dir / "diagnostic-imaging-report.json"
    if not output_report.is_file():
        raise SystemExit(f"Algorithm did not create {output_report}")
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("report"), str) or not payload["report"].strip():
        raise SystemExit(f"Invalid algorithm output: {payload!r}")

    shutil.copy2(report_candidates[0], ground_truth / f"{case_id}.txt")
    predictions = [
        {
            "pk": f"job-{case_id.lower()}",
            "status": "Succeeded",
            "inputs": inputs,
            "outputs": [
                {
                    "image": None,
                    "file": None,
                    "value": None,
                    "socket": {
                        "slug": "diagnostic-imaging-report",
                        "relative_path": "diagnostic-imaging-report.json",
                        "example_value": {"report": "potential long text"},
                        "is_image_kind": False,
                        "is_panimg_kind": False,
                        "is_dicom_image_kind": False,
                        "is_json_kind": True,
                        "is_file_kind": False,
                    },
                }
            ],
        }
    ]
    (evaluation_input / "predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )

    run(
        [
            "docker", "run", "--rm", "--platform=linux/amd64", "--network", "none",
            "--volume", f"{evaluation_input}:/input:ro",
            "--volume", f"{evaluation_output}:/output",
            "--volume", f"{ground_truth}:/opt/ml/input/data/ground_truth:ro",
            args.evaluation_image,
        ]
    )

    metrics_path = evaluation_output / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Evaluation did not create {metrics_path}")
    print("ALGORITHM_OUTPUT")
    print(output_report.read_text(encoding="utf-8"))
    print("METRICS")
    print(metrics_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
