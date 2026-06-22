#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
IMAGE_TAG="bite2text-eval-test"
DOCKER_NOOP_VOLUME="${IMAGE_TAG}-volume"
ONLINE_METRICS="${ENABLE_ONLINE_METRICS:-0}"

if [ "$ONLINE_METRICS" = "1" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required when ENABLE_ONLINE_METRICS=1" >&2
  exit 1
fi

cleanup() {
  docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "$SCRIPT_DIR/test/output":/output \
    --entrypoint /bin/sh \
    "$IMAGE_TAG" \
    -c "chmod -R -f o+rwX /output/* || true"

  docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}

echo "=+= (Re)build the container"
"$SCRIPT_DIR/do_build.sh"

mkdir -p "$SCRIPT_DIR/test/output"
chmod -R -f o+rX "$SCRIPT_DIR/test/input" "$SCRIPT_DIR/ground_truth" || true
chmod -f o+rwX "$SCRIPT_DIR/test/output" || true

docker run --rm \
  --platform=linux/amd64 \
  --volume "$SCRIPT_DIR/test/output":/output \
  --entrypoint /bin/sh \
  "$IMAGE_TAG" \
  -c "rm -rf /output/* || true"

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
trap cleanup EXIT

if [ "$ONLINE_METRICS" = "1" ]; then
  docker run --rm \
    --platform=linux/amd64 \
    --env ENABLE_ONLINE_METRICS=1 \
    --env RUNNING_ON_GRAND_CHALLENGE=0 \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --env RADFACT_MODEL="${RADFACT_MODEL:-gpt-4o-mini}" \
    --env RADFACT_TIMEOUT="${RADFACT_TIMEOUT:-30}" \
    --env RADFACT_MAX_RETRIES="${RADFACT_MAX_RETRIES:-0}" \
    --volume "$SCRIPT_DIR/test/input":/input:ro \
    --volume "$SCRIPT_DIR/test/output":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "$SCRIPT_DIR/ground_truth":/opt/ml/input/data/ground_truth:ro \
    "$IMAGE_TAG"
else
  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$SCRIPT_DIR/test/input":/input:ro \
    --volume "$SCRIPT_DIR/test/output":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "$SCRIPT_DIR/ground_truth":/opt/ml/input/data/ground_truth:ro \
    "$IMAGE_TAG"
fi

echo "Wrote $SCRIPT_DIR/test/output/metrics.json"
