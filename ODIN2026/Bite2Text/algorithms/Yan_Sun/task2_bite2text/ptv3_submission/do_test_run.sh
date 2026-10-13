#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="odin2026-bite2text-ptv3-debug-v3"
CASE_ID="${BITE2TEXT_TEST_CASE:-F5535}"
GPU_DEVICE="${BITE2TEXT_TEST_GPU:-1}"
INPUT_LAYOUT="${BITE2TEXT_TEST_LAYOUT:-official-obj-wrapped-stl}"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
INPUT_DIR="$SCRIPT_DIR/test/input/$INPUT_LAYOUT/$CASE_ID"
OUTPUT_DIR="$SCRIPT_DIR/test/output/$CASE_ID-$INPUT_LAYOUT-$RUN_ID"

for input_name in 3d-lower-teeth-scan.obj 3d-upper-teeth-scan.obj; do
  if [[ ! -f "$INPUT_DIR/$input_name" ]]; then
    echo "Missing official-layout test input: $INPUT_DIR/$input_name" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
chmod -R -f o+rX "$SCRIPT_DIR/model"
chmod -R -f o+rwX "$OUTPUT_DIR"

echo "=+= Running $CASE_ID layout=$INPUT_LAYOUT on GPU $GPU_DEVICE"
/usr/bin/time -f "ELAPSED=%e MAXRSS_KB=%M" docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus "device=$GPU_DEVICE" \
  --memory 16g \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  --volume "$SCRIPT_DIR/model":/opt/ml/model:ro \
  "$DOCKER_IMAGE_TAG"

python3 "$SCRIPT_DIR/verify_output.py" "$OUTPUT_DIR/diagnostic-imaging-report.json"
echo "OUTPUT_DIR=$OUTPUT_DIR"
