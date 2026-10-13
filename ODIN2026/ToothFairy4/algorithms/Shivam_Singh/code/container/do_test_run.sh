#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="toothfairy4-example-algorithm"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="$SCRIPT_DIR/test/input/samples"
OUTPUT_DIR="$SCRIPT_DIR/test/output"
CASES=(A003 F001 P001)

echo "=+= (Re)build the container"
"$SCRIPT_DIR/do_build.sh"

cleanup() {
  echo "=+= Cleaning permissions ..."
  docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "$OUTPUT_DIR":/output \
    --entrypoint /bin/sh \
    "$DOCKER_IMAGE_TAG" \
    -c "chmod -R -f o+rwX /output/* || true"

  docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}

chmod -R -f o+rX "$INPUT_DIR" "$SCRIPT_DIR/model" || true

if [ -d "$OUTPUT_DIR" ]; then
  chmod -R -f o+rwX "$OUTPUT_DIR" || true
  echo "=+= Cleaning up any earlier output"
  docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "$OUTPUT_DIR":/output \
    --entrypoint /bin/sh \
    "$DOCKER_IMAGE_TAG" \
    -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "$OUTPUT_DIR"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
trap cleanup EXIT

for case_id in "${CASES[@]}"; do
  case_output_dir="$OUTPUT_DIR/$case_id"
  mkdir -p -m o+rwX "$case_output_dir"

  echo "=+= Doing a forward pass on $case_id"
  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "$INPUT_DIR/$case_id":/input:ro \
    --volume "$case_output_dir":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "$SCRIPT_DIR/model":/opt/ml/model:ro \
    "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to $case_output_dir"
done
