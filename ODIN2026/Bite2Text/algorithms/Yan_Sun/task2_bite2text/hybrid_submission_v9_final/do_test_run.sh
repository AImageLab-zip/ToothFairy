#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="${BITE2TEXT_TEST_IMAGE:-odin2026-bite2text-hybrid-photo-test-v9}"
CASE_ID="${BITE2TEXT_TEST_CASE:-F5535}"
GPU_DEVICE="${BITE2TEXT_TEST_GPU:-0}"
INPUT_ROOT="${BITE2TEXT_TEST_INPUT_ROOT:-$SCRIPT_DIR/test/input}"
INPUT_DIR="$INPUT_ROOT/$CASE_ID"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="${BITE2TEXT_TEST_OUTPUT_ROOT:-$SCRIPT_DIR/test/output}/$CASE_ID-$RUN_ID"
MODEL_DIR="${BITE2TEXT_TEST_MODEL_ROOT:-$SCRIPT_DIR/model}"

if [[ ! "$CASE_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "CASE_ID must be a single path component" >&2
  exit 1
fi
for asset in config.py head_vocabs.json ios_normalizer_best.pt model_final.pth \
  photo_model_final.pt photo_view_classifier.pt retrieval_index.npz \
  retrieval_labels.json retrieval_reports.json; do
  test -r "$MODEL_DIR/$asset" || { echo "Missing model asset: $asset" >&2; exit 1; }
done
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)

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

mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
# Only this newly created output is writable by the container's unprivileged user.
# Input/model permissions are left untouched; use a readable dedicated copy.
chmod o+rwX "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
INPUT_DIR=$(cd "$INPUT_DIR" && pwd)

echo "=+= Running $CASE_ID on GPU $GPU_DEVICE"
/usr/bin/time -f "ELAPSED=%e MAXRSS_KB=%M" docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus "device=$GPU_DEVICE" \
  --memory 16g \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  --volume "$MODEL_DIR":/opt/ml/model:ro \
  "$DOCKER_IMAGE_TAG"

python3 "$SCRIPT_DIR/verify_output.py" "$OUTPUT_DIR/diagnostic-imaging-report.json"
echo "OUTPUT_DIR=$OUTPUT_DIR"
