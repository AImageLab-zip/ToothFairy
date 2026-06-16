#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="toothfairy4-eval-test"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR"
