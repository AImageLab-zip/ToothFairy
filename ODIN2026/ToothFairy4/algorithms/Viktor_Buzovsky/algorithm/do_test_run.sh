#!/usr/bin/env bash
# Local forward pass, mirroring how Grand Challenge invokes the algorithm.
#
# Two deliberate differences from the ODIN template:
#   * `sg docker` wraps the docker calls, because usermod's group change does not
#     apply to already-running login sessions.
#   * GPU is passed through by default (USE_GPU=0 to force the CPU path, which is
#     what you want when checking the no-GPU fallback -- but expect it to be slow).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# USE_GPU=1 needs the cu128 --local image (Blackwell) plus nvidia-container-toolkit.
USE_GPU="${USE_GPU:-1}"
if [ "$USE_GPU" = "1" ]; then
  DOCKER_IMAGE_TAG="${IMAGE_TAG:-toothfairy4-trinity-algorithm-local}"
  GPU_FLAG="--gpus all"
else
  DOCKER_IMAGE_TAG="${IMAGE_TAG:-toothfairy4-trinity-algorithm}"
  GPU_FLAG=""
fi
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="$SCRIPT_DIR/test/input/samples"
OUTPUT_DIR="$SCRIPT_DIR/test/output"
CASES=("${@:-A003 F001 P001}")

d() { sg docker -c "docker $*"; }

chmod -R -f o+rX "$INPUT_DIR" "$SCRIPT_DIR/model" || true
mkdir -p -m o+rwX "$OUTPUT_DIR"
chmod -R -f o+rwX "$OUTPUT_DIR" || true

d "volume create $DOCKER_NOOP_VOLUME" > /dev/null
cleanup() {
  d "volume rm $DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}
trap cleanup EXIT

for case_id in ${CASES[@]}; do
  case_output_dir="$OUTPUT_DIR/$case_id"
  rm -rf "$case_output_dir"
  mkdir -p -m o+rwX "$case_output_dir"

  echo "=+= Forward pass on $case_id"
  start=$(date +%s)
  d "run --rm --platform=linux/amd64 $GPU_FLAG \
      --network none \
      --memory 32g \
      --volume $INPUT_DIR/$case_id:/input:ro \
      --volume $case_output_dir:/output \
      --volume $DOCKER_NOOP_VOLUME:/tmp \
      --volume $SCRIPT_DIR/model:/opt/ml/model:ro \
      $DOCKER_IMAGE_TAG"
  echo "=+= $case_id took $(( $(date +%s) - start ))s (limit is 900s per case)"
done
