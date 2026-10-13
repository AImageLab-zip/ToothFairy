#!/usr/bin/env bash
set -euo pipefail

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CONFIG="${EVIDENT_INFERENCE_CONFIG:-/opt/evident/EviDent-CBCT/evigre/configs/inference/evident_cbct.container.json}"
INPUT_PATH="${1:-/input}"
OUTPUT_DIR="${2:-/output}"
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift

[[ -e "$INPUT_PATH" ]] || { echo "Input does not exist: $INPUT_PATH" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

exec python3 -m evigre.inference "$INPUT_PATH" "$OUTPUT_DIR" \
  --config "$CONFIG" --clean-work "$@"

