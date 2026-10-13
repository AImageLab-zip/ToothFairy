#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_NIFTI_OR_DIRECTORY OUTPUT_DIRECTORY [options]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${EVIDENT_IMAGE:-evident-cbct:latest}"
MODEL_DIR="${EVIDENT_MODELS:-$REPO_ROOT/models}"
INPUT_PATH="$(realpath "$1")"
OUTPUT_DIR="$2"
shift 2

[[ -d "$MODEL_DIR" ]] || { echo "Model directory not found: $MODEL_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
MODEL_DIR="$(realpath "$MODEL_DIR")"
if [[ -f "$INPUT_PATH" ]]; then
  INPUT_MOUNT="$(dirname "$INPUT_PATH")"
  CONTAINER_INPUT="/input/$(basename "$INPUT_PATH")"
else
  INPUT_MOUNT="$INPUT_PATH"
  CONTAINER_INPUT="/input"
fi

docker run --rm --gpus all --network none --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  --volume "$INPUT_MOUNT:/input:ro" \
  --volume "$OUTPUT_DIR:/output" \
  --volume "$MODEL_DIR:/opt/ml/model:ro" \
  --entrypoint /usr/local/bin/evident-cbct-cli \
  "$IMAGE_NAME" "$CONTAINER_INPUT" /output "$@"

