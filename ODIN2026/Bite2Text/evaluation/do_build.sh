#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="bite2text-eval-test"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR"
