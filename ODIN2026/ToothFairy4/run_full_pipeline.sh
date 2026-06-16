#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

RAW_DATA_DIR=${RAW_DATA_DIR:-"/home/llumetti/Downloads/toothfairy4-cbct---raw"}
ALGORITHM_DIR="$SCRIPT_DIR/algorithm"
EVALUATION_DIR="$SCRIPT_DIR/evaluation"
SAMPLE_ROOT="$ALGORITHM_DIR/test/input/samples"
WORK_DIR="$SCRIPT_DIR/pipeline_work"
EVAL_INPUT_DIR="$WORK_DIR/evaluation_input"
EVAL_OUTPUT_DIR="$WORK_DIR/evaluation_output"
EXTRACTED_GT_DIR="$WORK_DIR/ground_truth"
ALGORITHM_IMAGE="toothfairy4-example-algorithm"
EVALUATION_IMAGE="toothfairy4-eval-test"
GROUND_TRUTH_TARBALL="$EVALUATION_DIR/evaluationgroundtruth.tar.gz"
CASES=(A003 F001 P001)

echo "=+= Converting sample CBCT volumes to .mha"
python - <<PY
from pathlib import Path
import json
import SimpleITK as sitk

raw_data_dir = Path(${RAW_DATA_DIR@Q})
sample_root = Path(${SAMPLE_ROOT@Q})
cases = ["A003", "F001", "P001"]

for case_id in cases:
    source = raw_data_dir / case_id / "cbct" / "volume.nii.gz"
    case_dir = sample_root / case_id
    image_dir = case_dir / "images" / "cbct"
    output = image_dir / f"{case_id}.mha"
    inputs_json = case_dir / "inputs.json"

    if not source.exists():
        raise FileNotFoundError(source)

    image_dir.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        image = sitk.ReadImage(str(source))
        sitk.WriteImage(image, str(output), True)
        print(f"Converted {source} -> {output}")
    else:
        print(f"Using existing {output}")

    inputs_json.write_text(
        json.dumps(
            [
                {
                    "socket": {
                        "slug": "cbct-image",
                        "relative_path": "images/cbct",
                        "is_image_kind": True,
                        "is_panimg_kind": True,
                        "is_dicom_image_kind": False,
                        "is_json_kind": False,
                        "is_file_kind": False,
                    },
                    "file": None,
                    "image": {"name": f"{case_id}.mha"},
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
tar -czf "$GROUND_TRUTH_TARBALL" -C "$EVALUATION_DIR/ground_truth" "A003.txt" "F001.txt" "P001.txt"
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
python - <<PY
from pathlib import Path
import json

eval_input_dir = Path(${EVAL_INPUT_DIR@Q})
cases = ["A003", "F001", "P001"]
jobs = []

for case_id in cases:
    job_pk = f"job-{case_id.lower()}"
    jobs.append(
        {
            "pk": job_pk,
            "status": "Succeeded",
            "inputs": [
                {
                    "image": {"name": f"{case_id}.mha"},
                    "file": None,
                    "value": None,
                    "socket": {
                        "slug": "cbct-image",
                        "relative_path": "images/cbct",
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
