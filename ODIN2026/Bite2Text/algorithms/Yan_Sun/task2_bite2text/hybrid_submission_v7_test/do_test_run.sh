#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="odin2026-bite2text-hybrid-photo-test-v7"
CASE_ID="${BITE2TEXT_TEST_CASE:-F5535}"
GPU_DEVICE="${BITE2TEXT_TEST_GPU:-1}"
INPUT_ROOT="${BITE2TEXT_TEST_INPUT_ROOT:-$SCRIPT_DIR/test/input}"
INPUT_DIR="$INPUT_ROOT/$CASE_ID"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="$SCRIPT_DIR/test/output/$CASE_ID-$RUN_ID"

if [[ ! -f "$INPUT_DIR/3d-lower-teeth-scan.obj" && ! -d "$INPUT_DIR/files/ios-lower" ]]; then
  echo "Missing lower IOS test input below $INPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$INPUT_DIR/3d-upper-teeth-scan.obj" && ! -d "$INPUT_DIR/files/ios-upper" ]]; then
  echo "Missing upper IOS test input below $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR/images/intraoral-photo" ]]; then
  echo "Warning: no official-layout intraoral-photo directory; v5 fallback will be tested" >&2
fi

mkdir -p "$OUTPUT_DIR"
chmod -R -f o+rX "$SCRIPT_DIR/model"
chmod -R -f o+rwX "$OUTPUT_DIR"

echo "=+= Running $CASE_ID on GPU $GPU_DEVICE"
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
