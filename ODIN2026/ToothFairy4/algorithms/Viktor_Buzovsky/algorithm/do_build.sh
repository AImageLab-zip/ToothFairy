#!/usr/bin/env bash
# Build the algorithm image.
#
#   ./do_build.sh            -> cu126, tag toothfairy4-trinity-algorithm   (SUBMIT THIS)
#   ./do_build.sh --local    -> cu128, tag toothfairy4-trinity-algorithm-local
#
# The --local image exists only because the RTX 5090 is Blackwell (sm_120), which
# cu126 cannot run. Grand Challenge uses T4/A10G, where cu126 is correct -- so
# never submit the --local image.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

DOCKER_IMAGE_TAG="toothfairy4-trinity-algorithm"
BASE_IMAGE="pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime"

if [ "${1:-}" = "--local" ]; then
  DOCKER_IMAGE_TAG="toothfairy4-trinity-algorithm-local"
  BASE_IMAGE="pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
  echo "=+= LOCAL GPU build (cu128) -- do not submit this image"
fi

# vendor/ is a build artefact, not a hand-maintained fork. It silently drifted
# once already, so it is regenerated on every build.
"$SCRIPT_DIR/sync_vendor.sh"

sg docker -c "docker build \
  --platform=linux/amd64 \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --tag $DOCKER_IMAGE_TAG \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  $SCRIPT_DIR"

echo "=+= Built $DOCKER_IMAGE_TAG from $BASE_IMAGE"
