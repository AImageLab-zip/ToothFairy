#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

RAW_DATA_DIR=${RAW_DATA_DIR:-"/mnt/c/Users/Kevin/Downloads/Bite2Text"}
ALGORITHM_DIR="$SCRIPT_DIR/algorithm"
EVALUATION_DIR="$SCRIPT_DIR/evaluation"
SAMPLE_ROOT="$ALGORITHM_DIR/test/input/samples"
WORK_DIR=${WORK_DIR:-"$SCRIPT_DIR/pipeline_work"}

if grep -qi microsoft /proc/version 2>/dev/null && [[ "$WORK_DIR" == *" "* ]]; then
    WORK_DIR="/tmp/bite2text-pipeline-work"
    echo "=+= Using WSL-safe work directory: $WORK_DIR"
fi

EVAL_INPUT_DIR="$WORK_DIR/evaluation_input"
EVAL_OUTPUT_DIR="$WORK_DIR/evaluation_output"
EXTRACTED_GT_DIR="$WORK_DIR/ground_truth"
ALGORITHM_IMAGE="bite2text-example-algorithm"
EVALUATION_IMAGE="bite2text-eval-test"
GROUND_TRUTH_TARBALL="$EVALUATION_DIR/evaluationgroundtruth.tar.gz"
CASES=(F5535 F5520 F5405)

echo "=+= Preparing sample STL/TIFF inputs"
python3 - <<PY
from pathlib import Path
import json
import shutil

raw_data_dir = Path(${RAW_DATA_DIR@Q})
sample_root = Path(${SAMPLE_ROOT@Q})
cases = ["F5535", "F5520", "F5405"]

for case_id in cases:
    case_source = raw_data_dir / case_id
    lower_source = case_source / "ios" / "ios_lower.stl"
    upper_source = case_source / "ios" / "ios_upper.stl"
    photo_source = case_source / "intraoral-photo.tiff"

    case_dir = sample_root / case_id
    lower_dir = case_dir / "files" / "ios-lower"
    upper_dir = case_dir / "files" / "ios-upper"
    photo_dir = case_dir / "images" / "intraoral-photo"
    inputs_json = case_dir / "inputs.json"

    for source in (lower_source, upper_source, photo_source):
        if not source.exists():
            raise FileNotFoundError(source)

    lower_dir.mkdir(parents=True, exist_ok=True)
    upper_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)

    lower_target = lower_dir / "ios_lower.stl"
    upper_target = upper_dir / "ios_upper.stl"
    photo_target = photo_dir / "intraoral-photo.tiff"

    for source, target in (
        (lower_source, lower_target),
        (upper_source, upper_target),
        (photo_source, photo_target),
    ):
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            shutil.copy2(source, target)
            print(f"Copied {source} -> {target}")
        else:
            print(f"Using existing {target}")

    inputs_json.write_text(
        json.dumps(
            [
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
                    "image": {"name": "intraoral-photo.tiff"},
                    "value": None,
                }
            ],
            indent=4,
        ),
        encoding="utf-8",
    )
PY

echo "=+= Building algorithm image"
"$ALGORITHM_DIR/do_build.sh"

echo "=+= Building evaluation image"
"$EVALUATION_DIR/do_build.sh"

echo "=+= Preparing runtime directories"
rm -rf "$WORK_DIR"
mkdir -p "$EVAL_INPUT_DIR" "$EVAL_OUTPUT_DIR" "$EXTRACTED_GT_DIR"
chmod o+rwX "$EVAL_OUTPUT_DIR"

echo "=+= Creating 3-case ground-truth tarball"
tar -czf "$GROUND_TRUTH_TARBALL" -C "$EVALUATION_DIR/ground_truth" "F5535.txt" "F5520.txt" "F5405.txt"
tar -xzf "$GROUND_TRUTH_TARBALL" -C "$EXTRACTED_GT_DIR"

echo "=+= Running algorithm for each case"
for case_id in "${CASES[@]}"; do
  lower_case_id=$(printf '%s' "$case_id" | tr '[:upper:]' '[:lower:]')
  job_pk="job-$lower_case_id"
  job_output_dir="$EVAL_INPUT_DIR/$job_pk/output"
  mkdir -p "$job_output_dir"
  chmod o+rwX "$job_output_dir"

  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$SAMPLE_ROOT/$case_id":/input:ro \
    --volume "$job_output_dir":/output \
    --volume "$ALGORITHM_DIR/model":/opt/ml/model:ro \
    "$ALGORITHM_IMAGE"
done

echo "=+= Writing evaluation predictions.json"
python3 - <<PY
from pathlib import Path
import json

eval_input_dir = Path(${EVAL_INPUT_DIR@Q})
cases = ["F5535", "F5520", "F5405"]
jobs = []

for case_id in cases:
    job_pk = f"job-{case_id.lower()}"
    jobs.append(
        {
            "pk": job_pk,
            "status": "Succeeded",
            "inputs": [
                {
                    "image": None,
                    "file": {"name": "ios_lower.stl"},
                    "value": None,
                    "socket": {
                        "slug": "3d-lower-teeth-scan",
                        "relative_path": "files/ios-lower",
                        "is_image_kind": False,
                        "is_panimg_kind": False,
                        "is_dicom_image_kind": False,
                        "is_json_kind": False,
                        "is_file_kind": True,
                    },
                },
                {
                    "image": None,
                    "file": {"name": "ios_upper.stl"},
                    "value": None,
                    "socket": {
                        "slug": "3d-upper-teeth-scan",
                        "relative_path": "files/ios-upper",
                        "is_image_kind": False,
                        "is_panimg_kind": False,
                        "is_dicom_image_kind": False,
                        "is_json_kind": False,
                        "is_file_kind": True,
                    },
                },
                {
                    "image": {"name": "intraoral-photo.tiff"},
                    "file": None,
                    "value": None,
                    "socket": {
                        "slug": "2d-intraoral-photographs",
                        "relative_path": "images/intraoral-photo",
                        "is_image_kind": True,
                        "is_panimg_kind": True,
                        "is_dicom_image_kind": False,
                        "is_json_kind": False,
                        "is_file_kind": False,
                    },
                }
            ],
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
    )

(eval_input_dir / "predictions.json").write_text(json.dumps(jobs, indent=4), encoding="utf-8")
PY

echo "=+= Running evaluation on algorithm outputs"
docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --volume "$EVAL_INPUT_DIR":/input:ro \
  --volume "$EVAL_OUTPUT_DIR":/output \
  --volume "$EXTRACTED_GT_DIR":/opt/ml/input/data/ground_truth:ro \
  "$EVALUATION_IMAGE"

echo "=+= Pipeline complete"
echo "Metrics: $EVAL_OUTPUT_DIR/metrics.json"
