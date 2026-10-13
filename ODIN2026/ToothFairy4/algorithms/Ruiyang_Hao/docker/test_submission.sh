#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 OFFICIAL_CASE_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${EVIDENT_IMAGE:-evident-cbct:latest}"
MODEL_DIR="${EVIDENT_MODELS:-$REPO_ROOT/models}"
INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
MODEL_DIR="$(realpath "$MODEL_DIR")"

timeout --signal=TERM --kill-after=30s "${TIME_LIMIT_SECONDS:-900}s" \
  docker run --rm --platform linux/amd64 \
    --gpus "device=${GPU_DEVICE:-0}" --network none \
    --cpus 8 --memory 32g --memory-swap 32g --shm-size 8g \
    --user "$(id -u):$(id -g)" \
    --volume "$INPUT_DIR:/input:ro" \
    --volume "$OUTPUT_DIR:/output" \
    --volume "$MODEL_DIR:/opt/ml/model:ro" \
    "$IMAGE_NAME"

python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()); assert isinstance(d.get("report"),str) and d["report"].strip(); print(p)' \
  "$OUTPUT_DIR/diagnostic-imaging-report.json"

